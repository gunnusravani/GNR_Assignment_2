from pathlib import Path
import torch

def save_checkpoint(path: Path, model, optimizer=None, epoch: int = 0, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)

def load_checkpoint(path: Path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
