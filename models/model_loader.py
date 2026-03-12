from __future__ import annotations

from typing import Literal, Tuple
import timm
import torch
import torch.nn as nn

SupportedBackbone = Literal["resnet50", "densenet121", "efficientnet_b0"]


def _infer_classifier_module(model: nn.Module) -> nn.Module:
    """
    timm models typically expose one of: fc / classifier / head
    """
    for name in ("fc", "classifier", "head"):
        if hasattr(model, name):
            return getattr(model, name)
    raise ValueError("Cannot infer classifier module (expected one of: fc, classifier, head).")


def _count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_pretrained_backbone(
    backbone: SupportedBackbone,
    num_classes: int,
    *,
    linear_probe: bool = False,
    print_stats: bool = True,
) -> nn.Module:
    """
    Loads an ImageNet-pretrained CNN backbone via timm and replaces the classifier head
    to output `num_classes`.

    If linear_probe=True:
      - freezes entire model
      - unfreezes classifier only
      - prints trainable params
    """
    allowed = {"resnet50", "densenet121", "efficientnet_b0"}
    if backbone not in allowed:
        raise ValueError(f"Unsupported backbone='{backbone}'. Allowed: {sorted(allowed)}")

    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)

    if linear_probe:
        for p in model.parameters():
            p.requires_grad = False

        head = _infer_classifier_module(model)
        for p in head.parameters():
            p.requires_grad = True

        if print_stats:
            print(f"[linear_probe] trainable_params={_count_trainable_params(model)}")

    return model


def build_linear_probe_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Optimizer that only updates trainable params (i.e., classifier in linear probe mode).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("No trainable parameters found. Did you enable linear_probe=True?")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
