from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from config import CFG
from data.dataloaders import build_dataloaders, build_corrupted_eval_datasets
from experiments.trainer import evaluate as eval_one
from models.backbone import build_model
from utils.checkpoint import load_checkpoint


def _to_float(x: object) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return None if math.isnan(v) else v
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _best_by(rows: Iterable[dict], key_cols: tuple[str, ...], score_col: str) -> List[dict]:
    best: Dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r.get(c, "") for c in key_cols)
        s = _to_float(r.get(score_col))
        if s is None:
            continue
        cur = best.get(k)
        if cur is None:
            best[k] = r
            continue
        cur_s = _to_float(cur.get(score_col))
        if cur_s is None or s > cur_s:
            best[k] = r
    return list(best.values())


def _pick_checkpoints(results_dir: Path) -> List[dict]:
    """
    Choose checkpoints to evaluate:
    - Scenario 1: best linear_probe per backbone
    - Scenario 2: best last_block/selective_20/full_ft per backbone
    - Skip Scenario 2 linear_probe if Scenario 1 linear_probe exists for that backbone
    """
    lp_rows = _read_csv_rows(results_dir / "results_linear_probe.csv")
    ft_rows = _read_csv_rows(results_dir / "results_fine_tune.csv")

    # Scenario 1 linear probes (one per backbone)
    lp_best = _best_by(lp_rows, ("backbone", "transfer_mode"), "val_acc")
    picked: List[dict] = []
    lp_backbones = set()
    for r in lp_best:
        if str(r.get("transfer_mode")) != "linear_probe":
            continue
        bb = str(r.get("backbone", ""))
        if not bb:
            continue
        lp_backbones.add(bb)
        picked.append(
            {
                "source": "scenario1",
                "scenario": str(r.get("scenario", "linear_probe")),
                "backbone": bb,
                "transfer_mode": "linear_probe",
                "ckpt": str(r.get("ckpt", "")),
                "base_val_acc": _to_float(r.get("val_acc")),
            }
        )

    # Scenario 2 transfer modes except duplicate linear_probe
    ft_best = _best_by(ft_rows, ("backbone", "transfer_mode"), "val_acc")
    keep_modes = {"last_block", "selective_20", "full_ft", "linear_probe"}
    for r in ft_best:
        mode = str(r.get("transfer_mode", ""))
        bb = str(r.get("backbone", ""))
        if mode not in keep_modes or not bb:
            continue
        if mode == "linear_probe" and bb in lp_backbones:
            # Requested: do not repeat linear probing twice.
            continue
        picked.append(
            {
                "source": "scenario2",
                "scenario": str(r.get("scenario", "fine_tune")),
                "backbone": bb,
                "transfer_mode": mode,
                "ckpt": str(r.get("ckpt", "")),
                "base_val_acc": _to_float(r.get("val_acc")),
            }
        )

    # Stable order: backbone then mode
    mode_order = {"linear_probe": 0, "selective_20": 1, "last_block": 2, "full_ft": 3}
    picked.sort(key=lambda r: (r["backbone"], mode_order.get(r["transfer_mode"], 99), r["source"]))
    return picked


def _make_loaders(image_size: int, batch_size: int, num_workers: int):
    _, clean_val_loader = build_dataloaders(
        data_root=CFG.data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=CFG.persistent_workers,
    )

    corr_ds = build_corrupted_eval_datasets(
        CFG.data_root,
        image_size,
        gaussian_sigmas=CFG.corruption_gaussian_sigmas,
        motion_blur_kernel=CFG.corruption_motion_blur_kernel,
        brightness_deltas=CFG.corruption_brightness_deltas,
    )

    pw = CFG.persistent_workers if (CFG.persistent_workers is not None) else (num_workers > 0)
    corr_loaders = {
        name: torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=CFG.pin_memory,
            persistent_workers=pw,
        )
        for name, ds in corr_ds.items()
    }
    return clean_val_loader, corr_loaders


def _evaluate_one(ckpt_path: Path, backbone: str, clean_loader, corr_loaders: Dict[str, torch.utils.data.DataLoader], device: str) -> dict:
    model, _ = build_model(backbone, num_classes=CFG.num_classes, pretrained=False)
    load_checkpoint(str(ckpt_path), model)
    model.to(device)

    clean_acc = float(eval_one(model, clean_loader, device))
    corr_accs: Dict[str, float] = {}
    for cname, loader in corr_loaders.items():
        corr_accs[cname] = float(eval_one(model, loader, device))

    out = {
        "clean_acc": clean_acc,
        "corruption_mean_acc": float(sum(corr_accs.values()) / max(1, len(corr_accs))),
    }
    out["corruption_mean_error"] = float(1.0 - out["corruption_mean_acc"])
    out["relative_robustness_mean"] = float(out["corruption_mean_acc"] / clean_acc) if clean_acc > 0 else float("nan")

    if corr_accs:
        worst_name, worst_acc = min(corr_accs.items(), key=lambda kv: kv[1])
        out["corruption_worst_name"] = worst_name
        out["corruption_worst_acc"] = float(worst_acc)
        out["corruption_worst_error"] = float(1.0 - worst_acc)
        out["relative_robustness_worst"] = float(worst_acc / clean_acc) if clean_acc > 0 else float("nan")

    for cname, acc in sorted(corr_accs.items()):
        out[f"corruption__{cname}"] = acc
        out[f"corruption_error__{cname}"] = float(1.0 - acc)
        out[f"relative_robustness__{cname}"] = float(acc / clean_acc) if clean_acc > 0 else float("nan")

    return out


def _write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    preferred = [
        "timestamp",
        "source",
        "scenario",
        "backbone",
        "transfer_mode",
        "ckpt",
        "base_val_acc",
        "clean_acc",
        "corruption_mean_acc",
        "corruption_mean_error",
        "relative_robustness_mean",
        "corruption_worst_name",
        "corruption_worst_acc",
        "corruption_worst_error",
        "relative_robustness_worst",
        "seconds",
    ]
    keys = set().union(*(r.keys() for r in rows))
    rest = sorted(k for k in keys if k not in preferred)
    fieldnames = [k for k in preferred if k in keys] + rest

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate selected Scenario 1/2 checkpoints on Scenario 4 corruptions")
    ap.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing results_linear_probe.csv and results_fine_tune.csv")
    ap.add_argument("--out", type=Path, default=Path("results/results_corruption_from_s1_s2.csv"), help="Output CSV path")
    ap.add_argument("--batch-size", type=int, default=CFG.batch_size)
    ap.add_argument("--num-workers", type=int, default=CFG.num_workers)
    ap.add_argument("--image-size", type=int, default=CFG.image_size)
    args = ap.parse_args()

    candidates = _pick_checkpoints(args.results_dir)
    if not candidates:
        print("No checkpoints found from Scenario 1/2 CSVs.")
        return

    device = CFG.device if (CFG.device == "cpu" or torch.cuda.is_available()) else "cpu"
    clean_loader, corr_loaders = _make_loaders(args.image_size, args.batch_size, args.num_workers)

    rows: List[dict] = []
    for idx, c in enumerate(candidates, start=1):
        ckpt = Path(str(c["ckpt"]))
        if not ckpt.exists():
            print(f"[{idx}/{len(candidates)}] skip missing ckpt: {ckpt}")
            continue

        print(f"[{idx}/{len(candidates)}] evaluating backbone={c['backbone']} mode={c['transfer_mode']} source={c['source']}")
        t0 = time.time()
        metrics = _evaluate_one(ckpt, str(c["backbone"]), clean_loader, corr_loaders, device)
        rows.append(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **c,
                **metrics,
                "seconds": round(time.time() - t0, 2),
            }
        )

    _write_rows(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
