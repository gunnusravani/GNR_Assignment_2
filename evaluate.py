import argparse
import torch

from config import CFG
from data.dataloaders import build_dataloaders, build_corrupted_eval_datasets
from models.backbone import build_model
from utils.checkpoint import load_checkpoint
from experiments.trainer import evaluate

from analysis.feature_visualization import (
    extract_features_at_depths,
    train_linear_classifiers_on_depths,
    plot_val_accuracy_vs_depth,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--robust", action="store_true", help="Evaluate on corrupted validation sets")
    ap.add_argument(
        "--linear-probe-depths",
        action="store_true",
        help="Extract early/mid/final features and train separate linear probes; plot val acc vs depth",
    )
    ap.add_argument(
        "--max-feature-batches",
        type=int,
        default=CFG.feature_extract_max_batches,
        help="Optional cap for feature extraction (default from config)",
    )
    args = ap.parse_args()

    train_loader, val_loader = build_dataloaders(
        data_root=CFG.data_root,
        image_size=CFG.image_size,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=CFG.persistent_workers,
    )

    model, _info = build_model(CFG.backbone, num_classes=CFG.num_classes, pretrained=False)
    load_checkpoint(args.ckpt, model)

    device = CFG.device if (CFG.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model.to(device)

    acc = evaluate(model, val_loader, device)
    print(f"ckpt={args.ckpt} val_acc={acc:.4f}")

    if args.linear_probe_depths:
        # Use CPU for sklearn; features are extracted with torch (can still be on GPU for speed)
        feats_tr = extract_features_at_depths(model, train_loader, device=device, max_batches=args.max_feature_batches)
        feats_va = extract_features_at_depths(model, val_loader, device=device, max_batches=args.max_feature_batches)
        accs = train_linear_classifiers_on_depths(feats_tr, feats_va)
        out_path = plot_val_accuracy_vs_depth(
            accs,
            out_path=str(CFG.output_dir / "linear_probe_depths" / "val_acc_vs_depth.png"),
        )
        print(f"linear_probe_depths_accs={accs} plot={out_path}")

    if args.robust:
        corrupted = build_corrupted_eval_datasets(
            CFG.data_root,
            CFG.image_size,
            gaussian_sigmas=CFG.corruption_gaussian_sigmas,
            motion_blur_kernel=CFG.corruption_motion_blur_kernel,
            brightness_deltas=CFG.corruption_brightness_deltas,
        )
        pw = CFG.persistent_workers if (CFG.persistent_workers is not None) else (CFG.num_workers > 0)
        for name, ds in corrupted.items():
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=CFG.pin_memory,
                persistent_workers=pw,
            )
            cacc = evaluate(model, loader, device)
            print(f"corruption={name} val_acc={cacc:.4f}")

if __name__ == "__main__":
    main()
