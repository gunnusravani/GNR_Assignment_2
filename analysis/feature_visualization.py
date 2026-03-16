from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray  # [N, D]
    labels: np.ndarray    # [N]
    class_names: Optional[List[str]] = None


@dataclass(frozen=True)
class DepthFeatureSet:
    """
    Features at multiple depths:
      features_by_depth[name] -> [N, D]
    """
    features_by_depth: Dict[str, np.ndarray]
    labels: np.ndarray
    class_names: Optional[List[str]] = None


def _infer_classifier_attr(model: nn.Module) -> str:
    for name in ("fc", "classifier", "head"):
        if hasattr(model, name):
            return name
    raise ValueError("Could not infer classifier attribute (expected one of: fc/classifier/head).")


def _infer_feature_module(model: nn.Module) -> nn.Module:
    """
    Tries to pick a reasonable module to hook for penultimate features.
    Strategy:
      1) If timm model exposes forward_features, prefer that (no hook needed).
      2) Else hook the classifier module input by attaching hook on the classifier module itself
         and reading its *input* (penultimate vector).
    """
    _ = _infer_classifier_attr(model)  # validates classifier presence
    return model  # placeholder; actual logic is in extract_features(...)


def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    # Accept [B,C], [B,C,H,W], or others -> [B, D]
    if x.ndim == 4:
        x = x.mean(dim=(2, 3))  # GAP for conv features
    elif x.ndim != 2:
        x = x.view(x.size(0), -1)
    return x


def _pick_hook_modules_with_names(model: nn.Module) -> Dict[str, Tuple[str, nn.Module]]:
    """
    Heuristic mapping: pick early/mid/final modules across common CNN backbones.
    Works for many timm models but is best-effort.

    Returns dict: {"early": module, "mid": module, "final": module}
    """
    b = model

    # ResNet-like
    if hasattr(b, "layer1") and hasattr(b, "layer2") and hasattr(b, "layer4"):
        return {
            "early": ("layer1", b.layer1),
            "mid": ("layer2", b.layer2),
            "final": ("layer4", b.layer4),
        }

    # DenseNet-like
    if hasattr(b, "features"):
        feats = b.features
        # Try to hook dense blocks if present
        if all(hasattr(feats, n) for n in ("denseblock1", "denseblock2", "denseblock4")):
            return {
                "early": ("features.denseblock1", feats.denseblock1),
                "mid": ("features.denseblock2", feats.denseblock2),
                "final": ("features.denseblock4", feats.denseblock4),
            }
        # Fallback: first/middle/last child in features
        children = list(feats.named_children())
        if len(children) >= 3:
            return {
                "early": (f"features.{children[0][0]}", children[0][1]),
                "mid": (f"features.{children[len(children) // 2][0]}", children[len(children) // 2][1]),
                "final": (f"features.{children[-1][0]}", children[-1][1]),
            }

    # EfficientNet-like (timm)
    if hasattr(b, "blocks"):
        blocks = b.blocks
        n = len(blocks) if hasattr(blocks, "__len__") else 0
        if n >= 3:
            return {
                "early": ("blocks.0", blocks[0]),
                "mid": (f"blocks.{n // 2}", blocks[n // 2]),
                "final": (f"blocks.{n - 1}", blocks[-1]),
            }
        return {
            "early": ("blocks", blocks),
            "mid": ("blocks", blocks),
            "final": ("blocks", blocks),
        }

    # Generic fallback: pick first/mid/last top-level child
    kids = list(b.named_children())
    if len(kids) >= 3:
        return {
            "early": (kids[0][0], kids[0][1]),
            "mid": (kids[len(kids) // 2][0], kids[len(kids) // 2][1]),
            "final": (kids[-1][0], kids[-1][1]),
        }
    if len(kids) == 1:
        return {
            "early": (kids[0][0], kids[0][1]),
            "mid": (kids[0][0], kids[0][1]),
            "final": (kids[0][0], kids[0][1]),
        }
    raise ValueError("Could not pick hook modules for this model.")


def get_depth_layer_selection(model: nn.Module) -> Dict[str, str]:
    """Returns the selected layer/module names for early/mid/final depth hooks."""
    picked = _pick_hook_modules_with_names(model)
    return {k: v[0] for k, v in picked.items()}


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: str | torch.device = "cuda",
    max_batches: Optional[int] = None,
) -> FeatureSet:
    """
    Extract penultimate features from a trained classifier.

    - If model.forward_features exists (common in timm), uses it and then global-pools if needed.
    - Otherwise, uses a forward hook on the classifier module and captures its input.

    Returns numpy arrays: features [N, D], labels [N].
    """
    dev = torch.device(device if (str(device) == "cpu" or torch.cuda.is_available()) else "cpu")
    model.eval().to(dev)

    class_names: Optional[List[str]] = None
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "classes"):
        class_names = list(ds.classes)

    feats: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    # Path A: timm-style forward_features
    if hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
        for bidx, (x, y) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            f = model.forward_features(x)
            # Common shapes:
            #  - [B, C] already pooled
            #  - [B, C, H, W] needs global average pool
            if f.ndim == 4:
                f = f.mean(dim=(2, 3))
            elif f.ndim != 2:
                f = f.view(f.size(0), -1)

            feats.append(f.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())

        return FeatureSet(
            features=np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32),
            labels=np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64),
            class_names=class_names,
        )

    # Path B: hook classifier input (penultimate vector)
    classifier_attr = _infer_classifier_attr(model)
    classifier_mod: nn.Module = getattr(model, classifier_attr)

    captured: List[torch.Tensor] = []

    def hook_fn(_module, inputs, _outputs):
        # inputs is a tuple; classifier input should be (B, D)
        x_in = inputs[0]
        if x_in.ndim > 2:
            x_in = x_in.view(x_in.size(0), -1)
        captured.append(x_in.detach())

    handle = classifier_mod.register_forward_hook(hook_fn)
    try:
        for bidx, (x, y) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            captured.clear()

            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            _ = model(x)
            if not captured:
                raise RuntimeError("Feature hook did not capture features; model may not use the inferred classifier.")

            f = captured[-1]
            feats.append(f.cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    finally:
        handle.remove()

    return FeatureSet(
        features=np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32),
        labels=np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64),
        class_names=class_names,
    )


@torch.no_grad()
def extract_features_at_depths(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: str | torch.device = "cuda",
    max_batches: Optional[int] = None,
) -> DepthFeatureSet:
    """
    Extract features at 3 depths: early/mid/final.
    Uses forward hooks on selected modules. Features are pooled/flattened to [N, D].
    """
    dev = torch.device(device if (str(device) == "cpu" or torch.cuda.is_available()) else "cpu")
    model.eval().to(dev)

    class_names: Optional[List[str]] = None
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "classes"):
        class_names = list(ds.classes)

    picked = _pick_hook_modules_with_names(model)
    hooks: Dict[str, nn.Module] = {k: v[1] for k, v in picked.items()}

    # Accumulate features per depth across batches
    feats_by_depth: Dict[str, List[np.ndarray]] = {k: [] for k in hooks.keys()}
    ys: List[np.ndarray] = []

    captured: Dict[str, List[torch.Tensor]] = {k: [] for k in hooks.keys()}

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            captured[name].append(_flatten_if_needed(output.detach()))
        return _hook

    handles = [mod.register_forward_hook(_make_hook(name)) for name, mod in hooks.items()]
    try:
        for bidx, (x, y) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break

            for k in captured.keys():
                captured[k].clear()

            x = x.to(dev, non_blocking=True)
            _ = model(x)

            for k in hooks.keys():
                if not captured[k]:
                    raise RuntimeError(f"Hook '{k}' did not capture features.")
                f = captured[k][-1].cpu().numpy()
                feats_by_depth[k].append(f)

            ys.append(y.numpy())
    finally:
        for h in handles:
            h.remove()

    features_by_depth_np = {
        k: (np.concatenate(v, axis=0) if v else np.zeros((0, 0), dtype=np.float32))
        for k, v in feats_by_depth.items()
    }
    y_np = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.int64)

    return DepthFeatureSet(features_by_depth=features_by_depth_np, labels=y_np, class_names=class_names)


def train_linear_classifiers_on_depths(
    train_feats: DepthFeatureSet,
    val_feats: DepthFeatureSet,
    *,
    max_iter: int = 2000,
    C: float = 1.0,
) -> Dict[str, float]:
    """
    Trains separate linear classifiers (LogisticRegression) on depth features.
    Returns dict: depth_name -> val_accuracy
    """
    accs: Dict[str, float] = {}
    for depth_name, Xtr in train_feats.features_by_depth.items():
        Xva = val_feats.features_by_depth[depth_name]
        ytr = train_feats.labels
        yva = val_feats.labels

        if Xtr.size == 0 or Xva.size == 0:
            accs[depth_name] = float("nan")
            continue

        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                max_iter=max_iter,
                C=C,
                n_jobs=-1,
                multi_class="auto",
            ),
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        accs[depth_name] = float((pred == yva).mean())
    return accs


def plot_val_accuracy_vs_depth(
    accs: Dict[str, float],
    *,
    out_path: str | Path = "outputs/linear_probe/val_acc_vs_depth.png",
    title: str = "Validation Accuracy vs Network Depth (Linear Probes)",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = ["early", "mid", "final"]
    xs = np.arange(len(order))
    ys = [accs.get(k, float("nan")) for k in order]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs, order)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Depth")
    plt.ylabel("Val Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def compute_pca_2d(features: np.ndarray, *, seed: int = 1337) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(features)


def compute_tsne_2d(
    features: np.ndarray,
    *,
    seed: int = 1337,
    perplexity: float = 30.0,
    max_points: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (emb_2d, selected_indices). Subsamples for speed by default.
    """
    if features.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n = features.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        x = features[idx]
    else:
        idx = np.arange(n)
        x = features

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (len(idx) - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    emb = tsne.fit_transform(x)
    return emb, idx


def plot_embedding(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Embedding",
    out_path: str | Path = "outputs/embedding.png",
    alpha: float = 0.7,
    s: float = 8.0,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    unique = np.unique(labels) if labels.size else np.array([], dtype=np.int64)

    # Scatter per-class for stable legend
    for c in unique:
        m = labels == c
        name = class_names[int(c)] if (class_names is not None and int(c) < len(class_names)) else str(int(c))
        plt.scatter(emb_2d[m, 0], emb_2d[m, 1], s=s, alpha=alpha, label=name)

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if len(unique) <= 20:
        plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def visualize_features_pca_tsne(
    feature_set: FeatureSet,
    *,
    out_dir: str | Path = "outputs/feature_viz",
    seed: int = 1337,
) -> Tuple[Path, Path]:
    """
    Convenience wrapper: compute PCA + t-SNE and save plots.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pca2 = compute_pca_2d(feature_set.features, seed=seed)
    pca_path = plot_embedding(
        pca2,
        feature_set.labels,
        class_names=feature_set.class_names,
        title="PCA (2D) of Backbone Features",
        out_path=out_dir / "pca_2d.png",
    )

    tsne2, idx = compute_tsne_2d(feature_set.features, seed=seed)
    tsne_path = plot_embedding(
        tsne2,
        feature_set.labels[idx],
        class_names=feature_set.class_names,
        title="t-SNE (2D) of Backbone Features",
        out_path=out_dir / "tsne_2d.png",
    )

    return pca_path, tsne_path


def compute_feature_norm_stats(depth_features: DepthFeatureSet) -> Dict[str, Dict[str, float]]:
    """
    Computes L2-norm statistics for each depth feature matrix.
    Returns depth -> {mean, std, median}.
    """
    out: Dict[str, Dict[str, float]] = {}
    for depth, X in depth_features.features_by_depth.items():
        if X.size == 0:
            out[depth] = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
            continue
        norms = np.linalg.norm(X, axis=1)
        out[depth] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
        }
    return out


def plot_feature_norm_stats(
    norm_stats: Dict[str, Dict[str, float]],
    *,
    out_path: str | Path,
    title: str = "Feature Norm Statistics vs Depth",
) -> Path:
    """Bar plot of mean feature L2 norm with std error bars across depths."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = ["early", "mid", "final"]
    means = [norm_stats.get(k, {}).get("mean", float("nan")) for k in order]
    stds = [norm_stats.get(k, {}).get("std", 0.0) for k in order]

    x = np.arange(len(order))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, order)
    plt.ylabel("Feature L2 Norm")
    plt.xlabel("Depth")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_depthwise_pca_2d(
    depth_features: DepthFeatureSet,
    *,
    out_path: str | Path,
    title_prefix: str = "PCA (2D) on Fixed Subset",
) -> Path:
    """
    Creates one figure with PCA projections for early/mid/final depths.
    Assumes labels correspond to the same fixed sample set across depths.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = ["early", "mid", "final"]
    y = depth_features.labels

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for i, depth in enumerate(order):
        X = depth_features.features_by_depth.get(depth)
        ax = axs[i]
        if X is None or X.size == 0:
            ax.set_title(f"{depth} (empty)")
            ax.axis("off")
            continue

        p2 = PCA(n_components=2, random_state=1337).fit_transform(X)
        ax.scatter(p2[:, 0], p2[:, 1], c=y, s=10, alpha=0.7, cmap="tab20")
        ax.set_title(f"{depth}")
        ax.set_xlabel("dim-1")
        if i == 0:
            ax.set_ylabel("dim-2")

    fig.suptitle(title_prefix)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
