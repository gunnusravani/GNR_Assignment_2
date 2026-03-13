from dataclasses import dataclass
from pathlib import Path
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import accuracy_top1
from utils.checkpoint import save_checkpoint

@dataclass
class TrainResult:
    best_val_acc: float
    best_ckpt: Path
    history: dict[str, list[float]]

@torch.no_grad()
def _grad_l2_norms_by_param(model: nn.Module) -> dict[str, float]:
    """
    Returns L2 norm of gradients per parameter tensor (keyed by parameter name).
    Only includes params with p.grad is not None.
    """
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # Use float32 accumulation for stability
        g = p.grad.detach()
        out[name] = float(torch.linalg.vector_norm(g.float(), ord=2).item())
    return out


def _update_epoch_grad_stats(
    stats: dict[str, dict[str, float]],
    batch_norms: dict[str, float],
) -> None:
    """
    Online aggregation for each param name:
      - sum (for mean)
      - max
      - count
    """
    for name, v in batch_norms.items():
        s = stats.get(name)
        if s is None:
            stats[name] = {"sum": v, "max": v, "count": 1.0}
        else:
            s["sum"] += v
            s["max"] = max(s["max"], v)
            s["count"] += 1.0


def _finalize_epoch_grad_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Converts {name: {sum,max,count}} -> {name: {mean,max}}.
    Deterministic ordering for easier diffs.
    """
    out: dict[str, dict[str, float]] = {}
    for name in sorted(stats.keys()):
        s = stats[name]
        denom = max(1.0, float(s["count"]))
        out[name] = {"mean": float(s["sum"]) / denom, "max": float(s["max"])}
    return out

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    epochs: int,
    device: str,
    out_dir: Path,
    *,
    checkpoint_dir: Path | None = None,
    results_dir: Path | None = None,
    log_grad_norms: bool = True,
    grad_norms_every_n_steps: int = 1,
    log_metrics_jsonl: bool = True,
) -> TrainResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is None:
        checkpoint_dir = out_dir / "checkpoints"
    if results_dir is None:
        results_dir = out_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_acc = -1.0
    best_path = checkpoint_dir / "best.pt"

    metrics_path = results_dir / "metrics.jsonl"
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_grad_stats: dict[str, dict[str, float]] = {}
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            global_step += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if log_grad_norms and (grad_norms_every_n_steps <= 1 or (global_step % grad_norms_every_n_steps == 0)):
                batch_norms = _grad_l2_norms_by_param(model)
                _update_epoch_grad_stats(epoch_grad_stats, batch_norms)

            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_seen += int(bs)

        train_loss = total_loss / max(1, total_seen)
        train_acc = total_correct / max(1, total_seen)

        if log_grad_norms:
            grad_epoch = _finalize_epoch_grad_stats(epoch_grad_stats)
            with (results_dir / f"grad_norms_epoch_{epoch:03d}.json").open("w") as f:
                json.dump(grad_epoch, f, indent=2, sort_keys=True)

            if grad_epoch:
                top = sorted(grad_epoch.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:5]
                top_str = ", ".join([f"{n}={v['mean']:.3e}" for n, v in top])
                print(f"epoch={epoch} grad_norm_mean_top5: {top_str}")

        val_loss, val_acc = evaluate_with_loss(model, val_loader, criterion, device)
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer=optimizer, epoch=epoch, val_acc=val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, model, optimizer=optimizer, epoch=epoch, val_acc=val_acc)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if log_metrics_jsonl:
            rec = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "best_val_acc": float(best_acc),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"best={best_acc:.4f}"
        )

    return TrainResult(best_val_acc=best_acc, best_ckpt=best_path, history=history)


@torch.no_grad()
def evaluate_with_loss(model: nn.Module, loader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += int(bs)

    avg_loss = total_loss / max(1, total_seen)
    avg_acc = total_correct / max(1, total_seen)
    return float(avg_loss), float(avg_acc)

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    accs = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        accs.append(accuracy_top1(logits, y))
    return float(sum(accs) / max(1, len(accs)))
