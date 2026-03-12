from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def _accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += bs

    avg_loss = total_loss / max(1, total_seen)
    avg_acc = total_correct / max(1, total_seen)
    return avg_loss, avg_acc


def train_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int = 10,
    device: Optional[str] = None,
    save_dir: str | Path = "outputs",
    save_name: str = "best.pt",
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    use_amp: bool = False,
) -> Dict[str, list]:
    """
    Returns history dict with keys:
      - train_loss, train_acc, val_loss, val_acc
    Saves best checkpoint (by val_acc) to: save_dir/save_name
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / save_name

    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.to(dev)

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{max_epochs}", leave=False)
        for x, y in pbar:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_seen += bs

            pbar.set_postfix(loss=float(loss.item()))

        train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)

        val_loss, val_acc = _eval_one_epoch(model, val_loader, criterion, dev)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            # Support both StepLR-style and ReduceLROnPlateau-style
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )

    return history
