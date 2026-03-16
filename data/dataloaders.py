from pathlib import Path
from typing import Tuple, Dict, Iterable, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    *,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(image_size)

    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=eval_tf)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader

def _subset_indices(n: int, frac: float, *, seed: int) -> list[int]:
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")
    k = max(1, int(round(n * frac)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[:k]

def build_subset_trainloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    *,
    seed: int = 1337,
    fracs: Tuple[float, float, float] = (1.0, 0.2, 0.05),
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
) -> Tuple[Dict[float, DataLoader], DataLoader]:
    """
    Returns:
      - train_loaders: dict keyed by fraction (e.g., 1.0/0.2/0.05)
      - val_loader: full validation loader

    Subset selection is deterministic given `seed`.
    """
    train_tf, eval_tf = build_transforms(image_size)

    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=eval_tf)

    n = len(train_ds)
    train_loaders: Dict[float, DataLoader] = {}

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    for frac in fracs:
        idx = _subset_indices(n, frac, seed=seed)
        subset = Subset(train_ds, idx)
        train_loaders[frac] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loaders, val_loader


def build_fixed_val_subset_loader(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    *,
    samples_per_class: int = 30,
    seed: int = 1337,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    """
    Build a deterministic validation subset with exactly `samples_per_class`
    images per class (for fixed-sample analyses such as Scenario-5 PCA).
    """
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be > 0")

    _, eval_tf = build_transforms(image_size)
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=eval_tf)

    class_to_indices: Dict[int, list[int]] = {}
    for idx, y in enumerate(val_ds.targets):
        class_to_indices.setdefault(int(y), []).append(idx)

    g = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for cls in sorted(class_to_indices.keys()):
        idxs = class_to_indices[cls]
        if len(idxs) < samples_per_class:
            raise ValueError(
                f"Class {cls} has only {len(idxs)} samples in val; "
                f"needs at least {samples_per_class}."
            )

        perm = torch.randperm(len(idxs), generator=g).tolist()
        chosen = [idxs[i] for i in perm[:samples_per_class]]
        selected.extend(chosen)

    selected = sorted(selected)
    subset = Subset(val_ds, selected)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

class AddGaussianNoise(torch.nn.Module):
    """
    Expects input tensor in [0, 1] before Normalize (recommended placement: after ToTensor, before Normalize).
    """
    def __init__(self, sigma: float, clip: bool = True):
        super().__init__()
        self.sigma = float(sigma)
        self.clip = bool(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        y = x + noise
        if self.clip:
            y = y.clamp(0.0, 1.0)
        return y


class BrightnessShift(torch.nn.Module):
    """
    Simple additive brightness shift: x' = clamp(x + delta).
    Expects input tensor in [0, 1] before Normalize.
    """
    def __init__(self, delta: float, clip: bool = True):
        super().__init__()
        self.delta = float(delta)
        self.clip = bool(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.delta
        if self.clip:
            y = y.clamp(0.0, 1.0)
        return y


class MotionBlur(torch.nn.Module):
    """
    Approximates motion blur via depthwise conv with a horizontal or vertical box kernel.
    Expects input tensor in [0, 1] before Normalize.
    """
    def __init__(self, kernel_size: int = 9, direction: str = "horizontal", clip: bool = True):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size must be odd and >= 3")
        if direction not in ("horizontal", "vertical"):
            raise ValueError("direction must be 'horizontal' or 'vertical'")
        self.kernel_size = int(kernel_size)
        self.direction = direction
        self.clip = bool(clip)

        k = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        if direction == "horizontal":
            k[kernel_size // 2, :] = 1.0
        else:
            k[:, kernel_size // 2] = 1.0
        k = k / k.sum()
        self.register_buffer("kernel2d", k[None, None, :, :])  # [1,1,ks,ks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W] -> [1,C,H,W]
        x4 = x.unsqueeze(0)
        c = x4.shape[1]
        w = self.kernel2d.expand(c, 1, self.kernel_size, self.kernel_size)  # depthwise
        y = torch.nn.functional.conv2d(x4, w, padding=self.kernel_size // 2, groups=c)
        y = y.squeeze(0)
        if self.clip:
            y = y.clamp(0.0, 1.0)
        return y


def build_corrupted_eval_datasets(
    data_root: Path,
    image_size: int,
    *,
    gaussian_sigmas: Iterable[float] = (0.05, 0.10, 0.20),
    motion_blur_kernel: int = 9,
    brightness_deltas: Iterable[float] = (-0.20, -0.10, 0.10, 0.20),
) -> Dict[str, datasets.ImageFolder]:
    """
    Returns dict of corrupted *validation* datasets (ImageFolder), keyed by corruption name.

    Corruptions are applied ONLY for evaluation by injecting them into the eval transform
    between ToTensor() and Normalize().

    The corruption set is deterministic (fixed severities; no RNG).
    """
    # NOTE: do not use the default eval transform; rebuild explicitly to guarantee ordering:
    # Resize -> ToTensor -> Corruption -> Normalize
    def _eval_tf_with(corrupt: torch.nn.Module) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                corrupt,
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    val_dir = data_root / "val"

    out: Dict[str, datasets.ImageFolder] = {}

    # 1) Gaussian noise
    for sigma in gaussian_sigmas:
        key = f"gaussian_noise_sigma_{float(sigma):g}"
        out[key] = datasets.ImageFolder(
            str(val_dir),
            transform=_eval_tf_with(AddGaussianNoise(sigma=float(sigma))),
        )

    # 2) Motion blur (deterministic variants)
    out["motion_blur_h_k9"] = datasets.ImageFolder(
        str(val_dir),
        transform=_eval_tf_with(MotionBlur(kernel_size=int(motion_blur_kernel), direction="horizontal")),
    )
    out["motion_blur_v_k9"] = datasets.ImageFolder(
        str(val_dir),
        transform=_eval_tf_with(MotionBlur(kernel_size=int(motion_blur_kernel), direction="vertical")),
    )

    # 3) Brightness shift (fixed deltas)
    for delta in brightness_deltas:
        key = f"brightness_shift_{float(delta):+.2f}"
        out[key] = datasets.ImageFolder(
            str(val_dir),
            transform=_eval_tf_with(BrightnessShift(delta=float(delta))),
        )

    return out
