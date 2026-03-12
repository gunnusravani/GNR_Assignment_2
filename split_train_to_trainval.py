from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitStats:
    classes: int
    total_images: int
    train_images: int
    val_images: int


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _list_images(class_dir: Path) -> List[Path]:
    # Deterministic ordering before shuffling
    return sorted([p for p in class_dir.iterdir() if _is_image(p)])


def _split_indices(n: int, val_frac: float, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1).")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n >= 2 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _copy_or_move(src: Path, dst: Path, *, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def split_imagefolder_train_only(
    *,
    src_root: Path,
    dst_root: Path,
    val_frac: float,
    seed: int,
    move: bool,
    overwrite: bool,
) -> SplitStats:
    """
    Expects: src_root/train/<class>/*.jpg
    Writes:  dst_root/train/<class>/*.jpg and dst_root/val/<class>/*.jpg
    """
    src_train = src_root / "train"
    if not src_train.exists():
        raise FileNotFoundError(f"Expected folder not found: {src_train}")

    dst_train = dst_root / "train"
    dst_val = dst_root / "val"

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination '{dst_root}' already exists. Use --overwrite or change --dst."
            )
        shutil.rmtree(dst_root)

    class_dirs = sorted([d for d in src_train.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {src_train}")

    total = tr_total = va_total = 0

    for cdir in class_dirs:
        images = _list_images(cdir)
        if len(images) == 0:
            continue

        train_idx, val_idx = _split_indices(len(images), val_frac, seed=seed)

        # If class has 1 image, train_idx will be empty and val_idx empty (n_val=0).
        # Prefer to keep at least 1 in train if possible.
        if len(images) == 1:
            train_idx = np.array([0], dtype=int)
            val_idx = np.array([], dtype=int)

        # Copy/move
        cname = cdir.name
        for i in train_idx:
            src = images[int(i)]
            dst = dst_train / cname / src.name
            _copy_or_move(src, dst, move=move)
        for i in val_idx:
            src = images[int(i)]
            dst = dst_val / cname / src.name
            _copy_or_move(src, dst, move=move)

        total += len(images)
        tr_total += int(len(train_idx))
        va_total += int(len(val_idx))

    return SplitStats(
        classes=len(class_dirs),
        total_images=total,
        train_images=tr_total,
        val_images=va_total,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("AID"), help="Source dataset root containing train/")
    ap.add_argument("--dst", type=Path, default=Path("AID_split"), help="Output dataset root to create")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Per-class validation fraction (e.g., 0.2)")
    ap.add_argument("--seed", type=int, default=1337, help="Deterministic split seed")
    ap.add_argument("--move", action="store_true", help="Move instead of copy (destructive)")
    ap.add_argument("--overwrite", action="store_true", help="Delete dst if it exists")
    args = ap.parse_args()

    stats = split_imagefolder_train_only(
        src_root=args.src,
        dst_root=args.dst,
        val_frac=args.val_frac,
        seed=args.seed,
        move=args.move,
        overwrite=args.overwrite,
    )
    print(
        f"done: classes={stats.classes} total={stats.total_images} "
        f"train={stats.train_images} val={stats.val_images} dst={args.dst}"
    )


if __name__ == "__main__":
    main()
