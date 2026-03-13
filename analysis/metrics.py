from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion: np.ndarray


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Optional[Sequence[int]] = None,
) -> ClassificationMetrics:
    """
    Computes:
      - accuracy
      - precision/recall/F1 (macro + weighted)
      - confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationMetrics(
        accuracy=acc,
        precision_macro=float(p_m),
        recall_macro=float(r_m),
        f1_macro=float(f1_m),
        precision_weighted=float(p_w),
        recall_weighted=float(r_w),
        f1_weighted=float(f1_w),
        confusion=cm,
    )


def make_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    target_names: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[int]] = None,
) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def plot_accuracy_curves(
    history: Dict[str, List[float]],
    *,
    out_path: str | Path,
    title: str = "Accuracy vs Epoch",
) -> Path:
    """
    Expects history keys:
      - train_acc
      - val_acc
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    epochs = np.arange(1, max(len(train_acc), len(val_acc)) + 1)

    plt.figure(figsize=(7, 4))
    if train_acc:
        plt.plot(epochs[: len(train_acc)], train_acc, label="train_acc")
    if val_acc:
        plt.plot(epochs[: len(val_acc)], val_acc, label="val_acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    out_path: str | Path,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mat = cm.astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig_w = 10 if mat.shape[0] > 20 else 8
    fig_h = 8 if mat.shape[0] > 20 else 6
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if class_names is not None and len(class_names) == mat.shape[0]:
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=90, fontsize=7)
        plt.yticks(ticks, class_names, fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path
