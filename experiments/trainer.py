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
    log_grad_norms: bool = True,
    grad_norms_every_n_steps: int = 1,
    log_metrics_jsonl: bool = True,
) -> TrainResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_acc = -1.0
    best_path = out_dir / "best.pt"

    metrics_path = out_dir / "metrics.jsonl"

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_grad_stats: dict[str, dict[str, float]] = {}

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

        if log_grad_norms:
            grad_epoch = _finalize_epoch_grad_stats(epoch_grad_stats)
            with (out_dir / f"grad_norms_epoch_{epoch:03d}.json").open("w") as f:
                json.dump(grad_epoch, f, indent=2, sort_keys=True)

            if grad_epoch:
                top = sorted(grad_epoch.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:5]
                top_str = ", ".join([f"{n}={v['mean']:.3e}" for n, v in top])
                print(f"epoch={epoch} grad_norm_mean_top5: {top_str}")

        val_acc = evaluate(model, val_loader, device)
        save_checkpoint(out_dir / "latest.pt", model, optimizer=optimizer, epoch=epoch, val_acc=val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, model, optimizer=optimizer, epoch=epoch, val_acc=val_acc)

        if log_metrics_jsonl:
            rec = {"epoch": epoch, "val_acc": float(val_acc), "best_val_acc": float(best_acc)}
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

        print(f"epoch={epoch} val_acc={val_acc:.4f} best={best_acc:.4f}")

    return TrainResult(best_val_acc=best_acc, best_ckpt=best_path)

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    accs = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        accs.append(accuracy_top1(logits, y))
    return float(sum(accs) / max(1, len(accs)))
