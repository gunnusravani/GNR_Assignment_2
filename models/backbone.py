from dataclasses import dataclass
from typing import Tuple, Optional, List
import timm
import torch
import torch.nn as nn

@dataclass(frozen=True)
class ModelInfo:
    num_features: int
    classifier_name: str  # e.g. "fc", "classifier", "head"

@dataclass(frozen=True)
class TransferSummary:
    mode: str
    classifier_name: str
    total_params: int
    trainable_params: int
    trainable_frac: float
    notes: str

def _infer_classifier_name(model: nn.Module) -> str:
    for name in ["fc", "classifier", "head"]:
        if hasattr(model, name):
            return name
    raise ValueError("Cannot infer classifier name for this model (expected fc/classifier/head).")

def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, ModelInfo]:
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    classifier_name = _infer_classifier_name(model)
    num_features = getattr(model, classifier_name).in_features
    return model, ModelInfo(num_features=num_features, classifier_name=classifier_name)

def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

def _unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True

def _unfreeze_classifier(model: nn.Module, classifier_name: str) -> None:
    _unfreeze_module(getattr(model, classifier_name))

def _backbone_named_parameters(model: nn.Module, classifier_name: str) -> List[tuple[str, nn.Parameter]]:
    """
    Deterministic list of backbone parameters (excludes classifier head params).
    Order is the stable Python iteration order of model.named_parameters().
    """
    head_prefix = classifier_name + "."
    out: List[tuple[str, nn.Parameter]] = []
    for n, p in model.named_parameters():
        if n == classifier_name or n.startswith(head_prefix):
            continue
        out.append((n, p))
    return out

def _unfreeze_last_block(model: nn.Module, backbone: str) -> str:
    """
    Model-specific last-block unfreezing (documented):
      - resnet50: layer4
      - densenet121: features.denseblock4 + features.norm5
      - efficientnet_b0: blocks (entire blocks stack) + conv_head/bn2 (if present)

    Fallback heuristic:
      - unfreeze the last immediate child module (by model.named_children() order)
    Returns a short note describing what was unfrozen.
    """
    b = backbone.lower()

    if b == "resnet50" and hasattr(model, "layer4"):
        _unfreeze_module(getattr(model, "layer4"))
        return "unfrozen=resnet.layer4"

    if b == "densenet121" and hasattr(model, "features"):
        feats = getattr(model, "features")
        for name in ("denseblock4", "norm5"):
            if hasattr(feats, name):
                _unfreeze_module(getattr(feats, name))
        return "unfrozen=densenet.features.(denseblock4,norm5)"

    if b == "efficientnet_b0":
        unfrozen = []
        for name in ("blocks", "conv_head", "bn2"):
            if hasattr(model, name):
                _unfreeze_module(getattr(model, name))
                unfrozen.append(name)
        if unfrozen:
            return f"unfrozen=efficientnet.{tuple(unfrozen)}"

    # Fallback: last top-level child
    last_name = None
    last_mod = None
    for name, mod in model.named_children():
        last_name, last_mod = name, mod
    if last_mod is not None:
        _unfreeze_module(last_mod)
        return f"unfrozen=fallback.last_child({last_name})"

    return "unfrozen=none(fallback_failed)"

def _unfreeze_selective_percent(
    model: nn.Module,
    *,
    classifier_name: str,
    percent: float = 0.20,
) -> str:
    """
    Reproducible selection: unfreeze the last `percent` of backbone parameters by numel,
    iterating deterministically over model.named_parameters() order (excluding classifier).

    NOTE: This is parameter-count based, not layer-count based.
    """
    if not (0.0 < percent <= 1.0):
        raise ValueError("percent must be in (0, 1].")

    backbone_params = _backbone_named_parameters(model, classifier_name)
    total_backbone = sum(p.numel() for _, p in backbone_params)
    target = int(total_backbone * percent)

    # Unfreeze from the end until reaching target
    unfrozen_names: List[str] = []
    running = 0
    for name, p in reversed(backbone_params):
        p.requires_grad = True
        running += p.numel()
        unfrozen_names.append(name)
        if running >= target:
            break

    # Keep note short but deterministic
    unfrozen_names = list(reversed(unfrozen_names))
    preview = unfrozen_names[:3]
    tail = unfrozen_names[-3:] if len(unfrozen_names) > 3 else []
    return (
        f"unfrozen=selective_last_{int(percent*100)}pct "
        f"backbone_params_total={total_backbone} target={target} "
        f"unfrozen_param_tensors={len(unfrozen_names)} "
        f"first={preview} last={tail or preview}"
    )

def set_transfer_mode(model: nn.Module, mode: str, *, backbone: Optional[str] = None) -> TransferSummary:
    """
    Fine-tuning strategies (reproducible, documented):
      - full_ft: unfreeze all parameters (backbone + head)
      - linear_probe: freeze all, unfreeze classifier head only
      - last_block: freeze all, unfreeze *last block* (model-specific) + classifier head
      - selective_20: freeze all, unfreeze classifier head + last 20% of backbone params (by numel)

    `backbone` is used only for the model-specific last_block mapping.
    """
    classifier_name = _infer_classifier_name(model)

    total = sum(p.numel() for p in model.parameters())

    notes = []
    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        notes.append("unfrozen=all_params")

    elif mode == "linear_probe":
        _freeze_all(model)
        _unfreeze_classifier(model, classifier_name)
        notes.append(f"unfrozen=classifier_only({classifier_name})")

    elif mode == "last_block":
        _freeze_all(model)
        _unfreeze_classifier(model, classifier_name)
        if backbone is None:
            notes.append("warn=backbone_name_missing_for_last_block; used_fallback_mapping")
            notes.append(_unfreeze_last_block(model, backbone=""))
        else:
            notes.append(_unfreeze_last_block(model, backbone=backbone))
        notes.append(f"unfrozen=classifier({classifier_name})")

    elif mode == "selective_20":
        _freeze_all(model)
        _unfreeze_classifier(model, classifier_name)
        notes.append(_unfreeze_selective_percent(model, classifier_name=classifier_name, percent=0.20))
        notes.append(f"unfrozen=classifier({classifier_name})")

    else:
        raise NotImplementedError(f"Transfer mode not implemented: {mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frac = float(trainable) / float(total) if total > 0 else 0.0

    return TransferSummary(
        mode=mode,
        classifier_name=classifier_name,
        total_params=int(total),
        trainable_params=int(trainable),
        trainable_frac=frac,
        notes="; ".join(notes),
    )
