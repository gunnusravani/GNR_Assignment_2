from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from config import CFG
from utils.seed import set_seed
from data.dataloaders import (
    build_dataloaders,
    build_subset_trainloaders,
    build_corrupted_eval_datasets,
    build_fixed_val_subset_loader,
)
from models.backbone import TransferSummary, build_model, set_transfer_mode
from experiments.trainer import TrainResult, train as train_one, evaluate as eval_one
from utils.checkpoint import load_checkpoint

from analysis.feature_visualization import (
    extract_features,
    extract_features_at_depths,
    train_linear_classifiers_on_depths,
    plot_val_accuracy_vs_depth,
    visualize_features_pca_tsne,
    get_depth_layer_selection,
    compute_feature_norm_stats,
    plot_feature_norm_stats,
    plot_depthwise_pca_2d,
)
from analysis.metrics import (
    compute_classification_metrics,
    make_classification_report,
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
)


def _device() -> str:
    return CFG.device if (CFG.device == "cpu" or torch.cuda.is_available()) else "cpu"


def _append_row(csv_path: Path, fieldnames: list[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    mode = "a" if (CFG.exp_append_csv and exists) else "w"
    with csv_path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            w.writeheader()
        w.writerow(row)


def _train_and_get_best_ckpt(
    backbone: str,
    *,
    transfer_mode: str,
    train_loader,
    val_loader,
    run_dir: Path,
    ckpt_dir: Path,
    epochs: int | None = None,
) -> tuple[TrainResult, TransferSummary]:
    model, _info = build_model(backbone, num_classes=CFG.num_classes, pretrained=CFG.pretrained)
    summary = set_transfer_mode(model, transfer_mode, backbone=backbone)

    print(
        f"[train_start] model={backbone} mode={transfer_mode} "
        f"unfrozen={summary.trainable_params}/{summary.total_params} "
        f"trainable_frac={summary.trainable_frac:.4f}"
    )

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    res = train_one(
        model,
        train_loader,
        val_loader,
        opt,
        int(epochs if epochs is not None else CFG.epochs),
        _device(),
        run_dir,
        checkpoint_dir=ckpt_dir,
        results_dir=run_dir,
    )
    return res, summary


def _latest_grad_stats(run_dir: Path) -> dict[str, float]:
    files = sorted(run_dir.glob("grad_norms_epoch_*.json"))
    if not files:
        return {}
    with files[-1].open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        return {}

    means = []
    maxs = []
    for v in payload.values():
        if isinstance(v, dict):
            m = v.get("mean")
            mx = v.get("max")
            if isinstance(m, (int, float)):
                means.append(float(m))
            if isinstance(mx, (int, float)):
                maxs.append(float(mx))

    out: dict[str, float] = {}
    if means:
        out["grad_mean_over_params"] = float(sum(means) / len(means))
        out["grad_median_over_params"] = float(sorted(means)[len(means) // 2])
    if maxs:
        out["grad_max_over_params"] = float(max(maxs))
    return out


def _plot_val_acc_vs_unfrozen(rows: list[dict], out_path: Path, *, title: str) -> Path:
    import matplotlib.pyplot as plt

    xs = [float(r.get("trainable_frac", 0.0)) * 100.0 for r in rows]
    ys = [float(r.get("val_acc", 0.0)) for r in rows]
    labels = [str(r.get("transfer_mode", "")) for r in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    plt.xlabel("Unfrozen Parameters (%)")
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


@torch.no_grad()
def _collect_preds(model, loader, *, device: str) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    return y_true, y_pred


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


@torch.no_grad()
def _eval_corruptions(model, *, batch_size: int, num_workers: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ds_map = build_corrupted_eval_datasets(
        CFG.data_root,
        CFG.image_size,
        gaussian_sigmas=CFG.corruption_gaussian_sigmas,
        motion_blur_kernel=CFG.corruption_motion_blur_kernel,
        brightness_deltas=CFG.corruption_brightness_deltas,
    )
    for name, ds in ds_map.items():
        pw = CFG.persistent_workers if (CFG.persistent_workers is not None) else (num_workers > 0)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=CFG.pin_memory,
            persistent_workers=pw,
        )
        out[name] = float(eval_one(model, loader, _device()))
    return out


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _write_csv(path: Path, rows: list[dict], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write header-only if possible
        if fieldnames:
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
        return
    if fieldnames is None:
        # stable ordering: common keys first, then alphabetical extras
        common = ["backbone", "scenario", "transfer_mode", "val_acc"]
        keys = set().union(*(r.keys() for r in rows))
        rest = sorted([k for k in keys if k not in common])
        fieldnames = [k for k in common if k in keys] + rest

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_latex_table(path: Path, rows: list[dict], *, columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            # keep consistent short floats
            if v != v:  # NaN
                return ""
            return f"{v:.4f}"
        return _latex_escape(str(v))

    aligns = "l" * max(1, len(columns))
    with path.open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        f.write(f"\\label{{{_latex_escape(label)}}}\n")
        f.write(f"\\begin{{tabular}}{{{aligns}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join([_latex_escape(c) for c in columns]) + " \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(" & ".join(fmt(r.get(c)) for c in columns) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _best_by_key(rows: list[dict], *, key_cols: tuple[str, ...], score_col: str) -> list[dict]:
    """
    Keep the best row (max score_col) per key. Ties: keep first (stable).
    """
    best: dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r.get(c, "") for c in key_cols)
        s = _to_float(r.get(score_col))
        if s is None:
            continue
        if k not in best or (s > _to_float(best[k].get(score_col))):
            best[k] = r
    return list(best.values())


def _param_efficiency(backbone: str, transfer_mode: str) -> dict[str, float]:
    """
    Rebuild model to get trainable_param stats deterministically.
    """
    model, _ = build_model(backbone, num_classes=CFG.num_classes, pretrained=CFG.pretrained)
    summary = set_transfer_mode(model, transfer_mode, backbone=backbone)
    return {
        "trainable_params": float(summary.trainable_params),
        "total_params": float(summary.total_params),
        "trainable_frac": float(summary.trainable_frac),
    }


def aggregate_results_to_tables() -> None:
    """
    Reads scenario CSVs (if present) from CFG.exp_out_dir and writes:
      - outputs/tables/accuracy_comparison.(csv|tex)
      - outputs/tables/robustness_comparison.(csv|tex)
      - outputs/tables/parameter_efficiency.(csv|tex)
    """
    exp_dir = Path(CFG.exp_out_dir)
    out_dir = Path(CFG.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = {
        "linear_probe": exp_dir / "results_linear_probe.csv",
        "fine_tune": exp_dir / "results_fine_tune.csv",
        "few_shot": exp_dir / "results_few_shot.csv",
        "corruption": exp_dir / "results_corruption.csv",
        "layer_probe": exp_dir / "results_layer_probe.csv",
    }

    # ---- load / normalize ----
    all_rows: list[dict] = []
    for scen, p in scenario_files.items():
        for r in _read_csv_rows(p):
            r = dict(r)
            r["scenario"] = r.get("scenario") or scen
            all_rows.append(r)

    # ---- accuracy comparison (best val_acc per backbone,scenario) ----
    acc_best = _best_by_key(all_rows, key_cols=("backbone", "scenario"), score_col="val_acc")
    acc_best_norm: list[dict] = []
    for r in acc_best:
        acc_best_norm.append(
            {
                "backbone": r.get("backbone", ""),
                "scenario": r.get("scenario", ""),
                "transfer_mode": r.get("transfer_mode", ""),
                "val_acc": _to_float(r.get("val_acc")),
                "seconds": _to_float(r.get("seconds")),
                "few_shot_frac": r.get("few_shot_frac", ""),
            }
        )
    acc_best_norm.sort(key=lambda x: (x["backbone"], x["scenario"]))

    # deltas vs fine_tune (best fine_tune per backbone)
    ft_map: dict[str, float] = {}
    for r in acc_best_norm:
        if r["scenario"] == "fine_tune" and r["val_acc"] is not None:
            ft_map[str(r["backbone"])] = float(r["val_acc"])

    for r in acc_best_norm:
        base = ft_map.get(str(r["backbone"]))
        r["delta_vs_fine_tune"] = (r["val_acc"] - base) if (base is not None and r["val_acc"] is not None) else None

    _write_csv(out_dir / "accuracy_comparison.csv", acc_best_norm)
    _write_latex_table(
        out_dir / "accuracy_comparison.tex",
        acc_best_norm,
        columns=["backbone", "scenario", "transfer_mode", "val_acc", "delta_vs_fine_tune"],
        caption="Accuracy comparison across backbones and scenarios (best run per setting).",
        label="tab:accuracy_comparison",
    )

    # ---- robustness comparison (from corruption scenario) ----
    corr_rows = [r for r in all_rows if (r.get("scenario") == "corruption")]
    corr_best = _best_by_key(corr_rows, key_cols=("backbone", "scenario"), score_col="val_acc")

    rob_tbl: list[dict] = []
    for r in corr_best:
        clean = _to_float(r.get("val_acc"))
        mean_corr = _to_float(r.get("corruption_mean_acc"))
        mean_err = _to_float(r.get("corruption_mean_error"))
        rel_mean = _to_float(r.get("relative_robustness_mean"))

        # compute worst corruption across all "corruption__*" columns
        worst = None
        for k, v in r.items():
            if not str(k).startswith("corruption__"):
                continue
            fv = _to_float(v)
            if fv is None:
                continue
            worst = fv if (worst is None or fv < worst) else worst

        rob_tbl.append(
            {
                "backbone": r.get("backbone", ""),
                "clean_val_acc": clean,
                "corruption_mean_acc": mean_corr,
                "corruption_mean_error": mean_err,
                "relative_robustness_mean": rel_mean,
                "corruption_worst_acc": worst,
                "mean_drop": (clean - mean_corr) if (clean is not None and mean_corr is not None) else None,
                "worst_drop": (clean - worst) if (clean is not None and worst is not None) else None,
            }
        )
    rob_tbl.sort(key=lambda x: x["backbone"])

    _write_csv(out_dir / "robustness_comparison.csv", rob_tbl)
    _write_latex_table(
        out_dir / "robustness_comparison.tex",
        rob_tbl,
        columns=[
            "backbone",
            "clean_val_acc",
            "corruption_mean_acc",
            "corruption_mean_error",
            "relative_robustness_mean",
            "corruption_worst_acc",
            "mean_drop",
            "worst_drop",
        ],
        caption="Robustness comparison: clean vs corrupted validation accuracy (best corruption-run per backbone).",
        label="tab:robustness_comparison",
    )

    # ---- parameter efficiency (accuracy per million trainable params) ----
    pe_tbl: list[dict] = []
    for r in acc_best_norm:
        backbone = str(r["backbone"])
        transfer_mode = str(r.get("transfer_mode") or "")
        val_acc = r.get("val_acc")

        # Only compute if transfer mode maps to our enum
        if transfer_mode not in ("linear_probe", "full_ft", "last_block", "selective_20"):
            continue
        if val_acc is None:
            continue

        pstats = _param_efficiency(backbone, transfer_mode)
        trainable = pstats["trainable_params"]
        acc_per_m = float(val_acc) / (trainable / 1e6) if trainable > 0 else None

        pe_tbl.append(
            {
                "backbone": backbone,
                "scenario": r["scenario"],
                "transfer_mode": transfer_mode,
                "val_acc": float(val_acc),
                "trainable_params": int(trainable),
                "trainable_frac": float(pstats["trainable_frac"]),
                "acc_per_m_trainable_params": acc_per_m,
            }
        )
    pe_tbl.sort(key=lambda x: (x["backbone"], x["scenario"]))

    _write_csv(out_dir / "parameter_efficiency.csv", pe_tbl)
    _write_latex_table(
        out_dir / "parameter_efficiency.tex",
        pe_tbl,
        columns=["backbone", "scenario", "transfer_mode", "val_acc", "trainable_params", "trainable_frac", "acc_per_m_trainable_params"],
        caption="Parameter efficiency: validation accuracy vs number of trainable parameters.",
        label="tab:parameter_efficiency",
    )

    print(f"[tables] wrote: {out_dir}/accuracy_comparison.(csv|tex), robustness_comparison.(csv|tex), parameter_efficiency.(csv|tex)")


def run() -> None:
    set_seed(CFG.seed)
    out_dir = Path(CFG.exp_out_dir)
    ckpt_out_dir = Path(CFG.exp_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    for backbone in CFG.exp_models:
        for scenario in CFG.exp_scenarios:
            t0 = time.time()

            run_dir = out_dir / scenario / backbone / time.strftime("%Y%m%d-%H%M%S")
            ckpt_run_dir = ckpt_out_dir / scenario / backbone / run_dir.name
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_run_dir.mkdir(parents=True, exist_ok=True)

            # scenario-specific loaders / transfer mode
            if scenario == "few_shot":
                train_loaders, val_loader = build_subset_trainloaders(
                    data_root=CFG.data_root,
                    image_size=CFG.image_size,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    seed=CFG.few_shot_seed,
                    fracs=CFG.few_shot_fracs,
                    pin_memory=CFG.pin_memory,
                    persistent_workers=CFG.persistent_workers,
                )
                train_loader = train_loaders[CFG.few_shot_frac]
            else:
                train_loader, val_loader = build_dataloaders(
                    data_root=CFG.data_root,
                    image_size=CFG.image_size,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    pin_memory=CFG.pin_memory,
                    persistent_workers=CFG.persistent_workers,
                )

            transfer_mode = CFG.scenario_transfer_mode.get(scenario, "full_ft")

            row_base = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "backbone": backbone,
                "scenario": scenario,
                "epochs": CFG.epochs,
                "batch_size": CFG.batch_size,
                "image_size": CFG.image_size,
                "seed": CFG.seed,
                "few_shot_frac": (CFG.few_shot_frac if scenario == "few_shot" else ""),
                "transfer_mode": transfer_mode,
            }

            if scenario == "few_shot":
                # Scenario-3: evaluate data efficiency at 100%, 20%, 5% with fixed seed
                few_rows: list[dict] = []
                fracs = tuple(float(f) for f in CFG.few_shot_fracs)

                for frac in fracs:
                    frac_train_loader = train_loaders[frac]
                    frac_dir = run_dir / f"frac_{frac:.2f}"
                    frac_ckpt_dir = ckpt_run_dir / f"frac_{frac:.2f}"
                    frac_dir.mkdir(parents=True, exist_ok=True)
                    frac_ckpt_dir.mkdir(parents=True, exist_ok=True)

                    train_res_frac, summary_frac = _train_and_get_best_ckpt(
                        backbone,
                        transfer_mode=transfer_mode,
                        train_loader=frac_train_loader,
                        val_loader=val_loader,
                        run_dir=frac_dir,
                        ckpt_dir=frac_ckpt_dir,
                        epochs=CFG.few_shot_epochs,
                    )

                    model_frac, _ = build_model(backbone, num_classes=CFG.num_classes, pretrained=False)
                    load_checkpoint(train_res_frac.best_ckpt, model_frac)
                    model_frac.to(_device())
                    val_acc_frac = float(eval_one(model_frac, val_loader, _device()))

                    train_acc_last = float(train_res_frac.history["train_acc"][-1]) if train_res_frac.history["train_acc"] else float("nan")
                    val_acc_last = float(train_res_frac.history["val_acc"][-1]) if train_res_frac.history["val_acc"] else float("nan")
                    train_val_gap = train_acc_last - val_acc_last if (train_acc_last == train_acc_last and val_acc_last == val_acc_last) else float("nan")

                    acc_curve_path = plot_accuracy_curves(
                        train_res_frac.history,
                        out_path=frac_dir / "accuracy_curves.png",
                        title=f"{backbone} few_shot frac={frac:.2f}: train/val accuracy",
                    )
                    loss_curve_path = plot_loss_curves(
                        train_res_frac.history,
                        out_path=frac_dir / "loss_curves.png",
                        title=f"{backbone} few_shot frac={frac:.2f}: train/val loss",
                    )

                    few_rows.append(
                        {
                            **row_base,
                            "epochs": CFG.few_shot_epochs,
                            "few_shot_frac": frac,
                            "ckpt": str(train_res_frac.best_ckpt),
                            "val_acc": val_acc_frac,
                            "train_acc_last": train_acc_last,
                            "val_acc_last": val_acc_last,
                            "train_val_gap": train_val_gap,
                            "trainable_params": int(summary_frac.trainable_params),
                            "total_params": int(summary_frac.total_params),
                            "trainable_frac": float(summary_frac.trainable_frac),
                            "mode_notes": summary_frac.notes,
                            "artifact_accuracy_curves": str(acc_curve_path),
                            "artifact_loss_curves": str(loss_curve_path),
                            "seconds": round(time.time() - t0, 2),
                        }
                    )

                # Relative performance drop per assignment: (Acc100 - Acc5)/Acc100
                acc_map = {float(r["few_shot_frac"]): float(r["val_acc"]) for r in few_rows}
                acc_100 = acc_map.get(1.0)
                acc_5 = acc_map.get(0.05)
                rel_drop = ((acc_100 - acc_5) / acc_100) if (acc_100 is not None and acc_5 is not None and acc_100 > 0) else float("nan")

                for r in few_rows:
                    r["acc_100"] = acc_100
                    r["acc_5"] = acc_5
                    r["relative_performance_drop"] = rel_drop
                    csv_path = out_dir / "results_few_shot.csv"
                    fieldnames = list(r.keys())
                    _append_row(csv_path, fieldnames, r)

                continue

            # --- layer_probe: no training required; probe the (pretrained OR trained?) model.
            # Keep it simple: probe the ImageNet-pretrained backbone with task head (untrained) is not useful,
            # so probe a trained full_ft model for this backbone.
            if scenario == "layer_probe":
                train_res, _summary = _train_and_get_best_ckpt(
                    backbone,
                    transfer_mode=transfer_mode,  # changed: respect scenario mapping
                    train_loader=train_loader,
                    val_loader=val_loader,
                    run_dir=run_dir / f"train_{transfer_mode}_for_probe",
                    ckpt_dir=ckpt_run_dir / f"train_{transfer_mode}_for_probe",
                    epochs=CFG.epochs,
                )
                ckpt = train_res.best_ckpt

                model, _info = build_model(backbone, num_classes=CFG.num_classes, pretrained=False)
                load_checkpoint(ckpt, model)
                model.to(_device())

                feats_tr = extract_features_at_depths(
                    model, train_loader, device=_device(), max_batches=CFG.layer_probe_max_batches
                )
                feats_va = extract_features_at_depths(
                    model, val_loader, device=_device(), max_batches=CFG.layer_probe_max_batches
                )
                accs = train_linear_classifiers_on_depths(feats_tr, feats_va)
                layer_map = get_depth_layer_selection(model)
                norm_stats = compute_feature_norm_stats(feats_va)

                plot_path = plot_val_accuracy_vs_depth(
                    accs, out_path=run_dir / "val_acc_vs_depth.png"
                )

                norm_plot_path = plot_feature_norm_stats(
                    norm_stats,
                    out_path=run_dir / "feature_norm_vs_depth.png",
                    title=f"{backbone}: feature norm statistics vs depth",
                )

                fixed_val_loader = build_fixed_val_subset_loader(
                    data_root=CFG.data_root,
                    image_size=CFG.image_size,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    samples_per_class=CFG.layer_probe_samples_per_class,
                    seed=CFG.seed,
                    pin_memory=CFG.pin_memory,
                    persistent_workers=CFG.persistent_workers,
                )
                feats_fixed = extract_features_at_depths(
                    model, fixed_val_loader, device=_device(), max_batches=None
                )
                pca_depth_path = plot_depthwise_pca_2d(
                    feats_fixed,
                    out_path=run_dir / "pca_2d_across_depths_fixed_subset.png",
                    title_prefix=(
                        f"{backbone}: PCA(2D) across depths "
                        f"(fixed subset: {CFG.layer_probe_samples_per_class}/class)"
                    ),
                )

                row = {
                    **row_base,
                    "val_acc_early": accs.get("early", float("nan")),
                    "val_acc_mid": accs.get("mid", float("nan")),
                    "val_acc_final": accs.get("final", float("nan")),
                    "layer_early": layer_map.get("early", ""),
                    "layer_mid": layer_map.get("mid", ""),
                    "layer_final": layer_map.get("final", ""),
                    "norm_mean_early": norm_stats.get("early", {}).get("mean", float("nan")),
                    "norm_mean_mid": norm_stats.get("mid", {}).get("mean", float("nan")),
                    "norm_mean_final": norm_stats.get("final", {}).get("mean", float("nan")),
                    "norm_std_early": norm_stats.get("early", {}).get("std", float("nan")),
                    "norm_std_mid": norm_stats.get("mid", {}).get("std", float("nan")),
                    "norm_std_final": norm_stats.get("final", {}).get("std", float("nan")),
                    "artifact_val_acc_vs_depth": str(plot_path),
                    "artifact_feature_norm_vs_depth": str(norm_plot_path),
                    "artifact_pca_2d_across_depths_fixed_subset": str(pca_depth_path),
                    "fixed_subset_samples_per_class": int(CFG.layer_probe_samples_per_class),
                    "fixed_subset_total_samples": int(CFG.layer_probe_samples_per_class * CFG.num_classes),
                    "seconds": round(time.time() - t0, 2),
                }

                csv_path = out_dir / "results_layer_probe.csv"
                fieldnames = list(row.keys())
                _append_row(csv_path, fieldnames, row)
                continue

            # --- train (for standard scenarios) ---
            if scenario == "fine_tune":
                mode_rows: list[dict] = []
                mode_results_dir = run_dir / "modes"
                mode_ckpt_dir = ckpt_run_dir / "modes"
                mode_results_dir.mkdir(parents=True, exist_ok=True)
                mode_ckpt_dir.mkdir(parents=True, exist_ok=True)

                for mode in CFG.fine_tune_modes:
                    mode_run_dir = mode_results_dir / mode
                    mode_ckpt_run_dir = mode_ckpt_dir / mode
                    mode_run_dir.mkdir(parents=True, exist_ok=True)
                    mode_ckpt_run_dir.mkdir(parents=True, exist_ok=True)

                    train_res_mode, summary = _train_and_get_best_ckpt(
                        backbone,
                        transfer_mode=mode,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        run_dir=mode_run_dir,
                        ckpt_dir=mode_ckpt_run_dir,
                        epochs=CFG.epochs,
                    )

                    model_mode, _ = build_model(backbone, num_classes=CFG.num_classes, pretrained=False)
                    load_checkpoint(train_res_mode.best_ckpt, model_mode)
                    model_mode.to(_device())
                    val_acc_mode = float(eval_one(model_mode, val_loader, _device()))

                    acc_curve_path = plot_accuracy_curves(
                        train_res_mode.history,
                        out_path=mode_run_dir / "accuracy_curves.png",
                        title=f"{backbone} {mode}: train/val accuracy",
                    )
                    loss_curve_path = plot_loss_curves(
                        train_res_mode.history,
                        out_path=mode_run_dir / "loss_curves.png",
                        title=f"{backbone} {mode}: train/val loss",
                    )

                    grad_stats = _latest_grad_stats(mode_run_dir)

                    mode_row = {
                        **row_base,
                        "scenario": "fine_tune",
                        "transfer_mode": mode,
                        "ckpt": str(train_res_mode.best_ckpt),
                        "val_acc": val_acc_mode,
                        "trainable_params": int(summary.trainable_params),
                        "total_params": int(summary.total_params),
                        "trainable_frac": float(summary.trainable_frac),
                        "unfrozen_percent": float(summary.trainable_frac) * 100.0,
                        "mode_notes": summary.notes,
                        "artifact_accuracy_curves": str(acc_curve_path),
                        "artifact_loss_curves": str(loss_curve_path),
                        **grad_stats,
                        "seconds": round(time.time() - t0, 2),
                    }
                    mode_rows.append(mode_row)

                # Scenario-2 required comparison plot: accuracy vs unfrozen %
                mode_rows_sorted = sorted(mode_rows, key=lambda r: float(r.get("unfrozen_percent", 0.0)))
                cmp_plot = _plot_val_acc_vs_unfrozen(
                    mode_rows_sorted,
                    run_dir / "val_acc_vs_unfrozen_percent.png",
                    title=f"{backbone}: val acc vs unfrozen params",
                )

                for r in mode_rows_sorted:
                    r["artifact_val_acc_vs_unfrozen_percent"] = str(cmp_plot)
                    csv_path = out_dir / "results_fine_tune.csv"
                    fieldnames = list(r.keys())
                    _append_row(csv_path, fieldnames, r)

                # fine_tune handled per mode; continue to next scenario
                continue

            train_res, _summary = _train_and_get_best_ckpt(
                backbone,
                transfer_mode=transfer_mode,
                train_loader=train_loader,
                val_loader=val_loader,
                run_dir=run_dir,
                ckpt_dir=ckpt_run_dir,
                epochs=CFG.epochs,
            )
            ckpt = train_res.best_ckpt

            # --- eval clean + optional corruption ---
            model, _info = build_model(backbone, num_classes=CFG.num_classes, pretrained=False)
            load_checkpoint(ckpt, model)
            model.to(_device())

            clean_acc = float(eval_one(model, val_loader, _device()))

            row = {
                **row_base,
                "ckpt": str(ckpt),
                "val_acc": clean_acc,
                "seconds": round(time.time() - t0, 2),
            }

            if scenario == "linear_probe":
                # 1) Training/validation accuracy curves
                acc_curve_path = plot_accuracy_curves(
                    train_res.history,
                    out_path=run_dir / "accuracy_curves.png",
                    title=f"{backbone} linear_probe: train/val accuracy",
                )

                # 2) Confusion matrix + classification report
                y_true, y_pred = _collect_preds(model, val_loader, device=_device())
                class_names = list(getattr(val_loader.dataset, "classes", []))
                labels = list(range(len(class_names))) if class_names else None
                cls_metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

                cm_path = plot_confusion_matrix(
                    cls_metrics.confusion,
                    out_path=run_dir / "confusion_matrix.png",
                    class_names=class_names if class_names else None,
                    normalize=True,
                    title=f"{backbone} linear_probe: normalized confusion matrix",
                )

                report_txt = make_classification_report(
                    y_true,
                    y_pred,
                    target_names=class_names if class_names else None,
                    labels=labels,
                )
                report_path = run_dir / "classification_report.txt"
                with report_path.open("w") as f:
                    f.write(report_txt)

                _write_json(
                    run_dir / "classification_summary.json",
                    {
                        "accuracy": cls_metrics.accuracy,
                        "precision_macro": cls_metrics.precision_macro,
                        "recall_macro": cls_metrics.recall_macro,
                        "f1_macro": cls_metrics.f1_macro,
                        "precision_weighted": cls_metrics.precision_weighted,
                        "recall_weighted": cls_metrics.recall_weighted,
                        "f1_weighted": cls_metrics.f1_weighted,
                    },
                )

                # 3) Feature embeddings (PCA + t-SNE)
                feat_val = extract_features(
                    model,
                    val_loader,
                    device=_device(),
                    max_batches=CFG.feature_extract_max_batches,
                )
                pca_path, tsne_path = visualize_features_pca_tsne(
                    feat_val,
                    out_dir=run_dir / "feature_viz",
                    seed=CFG.seed,
                )

                row["artifact_accuracy_curves"] = str(acc_curve_path)
                row["artifact_confusion_matrix"] = str(cm_path)
                row["artifact_classification_report"] = str(report_path)
                row["artifact_pca_2d"] = str(pca_path)
                row["artifact_tsne_2d"] = str(tsne_path)
                row["val_precision_macro"] = float(cls_metrics.precision_macro)
                row["val_recall_macro"] = float(cls_metrics.recall_macro)
                row["val_f1_macro"] = float(cls_metrics.f1_macro)

            if scenario == "corruption" and CFG.corruptions_enabled:
                corr = _eval_corruptions(model, batch_size=CFG.batch_size, num_workers=CFG.num_workers)
                # Flatten a few canonical keys; also store mean over all corruptions
                row["corruption_mean_acc"] = float(sum(corr.values()) / max(1, len(corr)))
                row["corruption_mean_error"] = float(1.0 - row["corruption_mean_acc"])
                row["relative_robustness_mean"] = float(
                    row["corruption_mean_acc"] / clean_acc if clean_acc > 0 else float("nan")
                )

                if corr:
                    worst_name, worst_acc = min(corr.items(), key=lambda kv: kv[1])
                    row["corruption_worst_name"] = worst_name
                    row["corruption_worst_acc"] = float(worst_acc)
                    row["corruption_worst_error"] = float(1.0 - worst_acc)
                    row["relative_robustness_worst"] = float(worst_acc / clean_acc if clean_acc > 0 else float("nan"))

                for k, v in sorted(corr.items()):
                    row[f"corruption__{k}"] = v
                    row[f"corruption_error__{k}"] = float(1.0 - v)
                    row[f"relative_robustness__{k}"] = float(v / clean_acc if clean_acc > 0 else float("nan"))

            csv_name = f"results_{scenario}.csv"
            csv_path = out_dir / csv_name
            fieldnames = list(row.keys())
            _append_row(csv_path, fieldnames, row)

    # NEW: aggregate after runs finish
    aggregate_results_to_tables()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
