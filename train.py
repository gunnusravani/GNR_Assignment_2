import csv
import time
from pathlib import Path

import torch

from config import CFG
from utils.seed import set_seed
from data.dataloaders import build_dataloaders, build_corrupted_eval_datasets, build_subset_trainloaders
from models.backbone import build_model, set_transfer_mode
from experiments.trainer import train, evaluate
from utils.model_stats import print_model_stats

def main():
    set_seed(CFG.seed)

    train_loader, val_loader = build_dataloaders(
        data_root=CFG.data_root,
        image_size=CFG.image_size,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        persistent_workers=CFG.persistent_workers,
    )

    model, _info = build_model(CFG.backbone, num_classes=CFG.num_classes, pretrained=CFG.pretrained)
    summary = set_transfer_mode(model, CFG.transfer_mode, backbone=CFG.backbone)
    print(
        f"transfer_mode={summary.mode} "
        f"trainable_params={summary.trainable_params}/{summary.total_params} "
        f"trainable_frac={summary.trainable_frac:.4f} "
        f"notes={summary.notes}"
    )

    device = CFG.device if (CFG.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Print efficiency stats before training
    print_model_stats(model, input_res=(CFG.image_size, CFG.image_size), device="cpu")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    out_dir = CFG.output_dir / CFG.backbone / CFG.transfer_mode
    ckpt_dir = CFG.checkpoint_dir / CFG.backbone / CFG.transfer_mode
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        CFG.epochs,
        device,
        out_dir,
        checkpoint_dir=ckpt_dir,
        results_dir=out_dir,
        log_grad_norms=CFG.log_grad_norms,
        grad_norms_every_n_steps=CFG.grad_norms_every_n_steps,
        log_metrics_jsonl=CFG.log_metrics_jsonl,
    )

if __name__ == "__main__":
    main()
