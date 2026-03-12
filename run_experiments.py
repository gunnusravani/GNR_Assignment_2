from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch

from config import CFG
from utils.seed import set_seed
from data.dataloaders import (
    build_dataloaders,
    build_subset_trainloaders,
    build_corrupted_eval_datasets,
)
from models.backbone import build_model, set_transfer_mode
from experiments.trainer import train as train_one, evaluate as eval_one
from utils.checkpoint import load_checkpoint

from analysis.feature_visualization import (
    extract_features_at_depths,
    train_linear_classifiers_on_depths,
    plot_val_accuracy_vs_depth,
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
) -> Path:
    model, _info = build_model(backbone, num_classes=CFG.num_classes, pretrained=CFG.pretrained)
    _ = set_transfer_mode(model, transfer_mode, backbone=backbone)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    res = train_one(model, train_loader, val_loader, opt, CFG.epochs, _device(), run_dir)
    return res.best_ckpt


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
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=CFG.pin_memory,
            persistent_workers=CFG.persistent_workers,
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
        columns=["backbone", "clean_val_acc", "corruption_mean_acc", "corruption_worst_acc", "mean_drop", "worst_drop"],
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
    out_dir.mkdir(parents=True, exist_ok=True)

    for backbone in CFG.exp_models:
        for scenario in CFG.exp_scenarios:
            t0 = time.time()

            run_dir = out_dir / backbone / scenario / time.strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)

            # scenario-specific loaders / transfer mode
            if scenario == "few_shot":
                train_loaders, val_loader = build_subset_trainloaders(
                    data_root=CFG.data_root,
                    image_size=CFG.image_size,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    seed=CFG.few_shot_seed,
                    fracs=(CFG.few_shot_frac,),
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

            # --- layer_probe: no training required; probe the (pretrained OR trained?) model.
            # Keep it simple: probe the ImageNet-pretrained backbone with task head (untrained) is not useful,
            # so probe a trained full_ft model for this backbone.
            if scenario == "layer_probe":
                ckpt = _train_and_get_best_ckpt(
                    backbone,
                    transfer_mode=transfer_mode,  # changed: respect scenario mapping
                    train_loader=train_loader,
                    val_loader=val_loader,
                    run_dir=run_dir / f"train_{transfer_mode}_for_probe",
                )

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

                plot_path = plot_val_accuracy_vs_depth(
                    accs, out_path=run_dir / "val_acc_vs_depth.png"
                )

                row = {
                    **row_base,
                    "val_acc_early": accs.get("early", float("nan")),
                    "val_acc_mid": accs.get("mid", float("nan")),
                    "val_acc_final": accs.get("final", float("nan")),
                    "plot_path": str(plot_path),
                    "seconds": round(time.time() - t0, 2),
                }

                csv_path = out_dir / "results_layer_probe.csv"
                fieldnames = list(row.keys())
                _append_row(csv_path, fieldnames, row)
                continue

            # --- train (for standard scenarios) ---
            ckpt = _train_and_get_best_ckpt(
                backbone,
                transfer_mode=transfer_mode,
                train_loader=train_loader,
                val_loader=val_loader,
                run_dir=run_dir,
            )

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

            if scenario == "corruption" and CFG.corruptions_enabled:
                corr = _eval_corruptions(model, batch_size=CFG.batch_size, num_workers=CFG.num_workers)
                # Flatten a few canonical keys; also store mean over all corruptions
                row["corruption_mean_acc"] = float(sum(corr.values()) / max(1, len(corr)))
                for k, v in sorted(corr.items()):
                    row[f"corruption__{k}"] = v

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
