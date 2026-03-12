from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

try:
    from ptflops import get_model_complexity_info
except Exception as e:  # pragma: no cover
    get_model_complexity_info = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelStats:
    total_params: int
    trainable_params: int
    macs: int
    flops: int  # approx: 2 * MACs


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _parse_ptflops_str_to_int(s: str) -> int:
    """
    ptflops returns strings like: '4.12 GMac', '123.4 MMac', '12.3 KMac', ...
    Convert to integer count.
    """
    s = s.strip().replace(",", "")
    parts = s.split()
    if not parts:
        return 0
    value = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else ""

    mult = 1.0
    if unit.startswith("g"):
        mult = 1e9
    elif unit.startswith("m"):
        mult = 1e6
    elif unit.startswith("k"):
        mult = 1e3
    elif unit.startswith("b"):  # sometimes "BMac" or similar
        mult = 1.0

    return int(value * mult)


@torch.no_grad()
def compute_macs_flops_ptflops(
    model: nn.Module,
    input_res: Tuple[int, int],
    *,
    in_channels: int = 3,
    device: str = "cpu",
) -> tuple[int, int]:
    """
    Returns (macs, flops) for a single forward pass.
    FLOPs are approximated as 2 * MACs (common convention).
    """
    if get_model_complexity_info is None:
        raise ImportError("ptflops is not available. Add `ptflops` to requirements and install it.")

    was_training = model.training
    model.eval()

    # ptflops uses forward hooks; keep model on CPU for stability unless you need GPU-only ops
    model = model.to(device)

    macs_str, _params_str = get_model_complexity_info(
        model,
        (in_channels, input_res[0], input_res[1]),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    macs = _parse_ptflops_str_to_int(macs_str)
    flops = int(2 * macs)

    if was_training:
        model.train()

    return macs, flops


def compute_model_stats(
    model: nn.Module,
    *,
    input_res: Tuple[int, int],
    in_channels: int = 3,
    device: str = "cpu",
) -> ModelStats:
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    macs, flops = compute_macs_flops_ptflops(
        model, input_res=input_res, in_channels=in_channels, device=device
    )
    return ModelStats(total_params=total, trainable_params=trainable, macs=macs, flops=flops)


def print_model_stats(
    model: nn.Module,
    *,
    input_res: Tuple[int, int],
    in_channels: int = 3,
    device: str = "cpu",
) -> ModelStats:
    stats = compute_model_stats(model, input_res=input_res, in_channels=in_channels, device=device)
    print(
        "model_stats:"
        f" total_params={stats.total_params}"
        f" trainable_params={stats.trainable_params}"
        f" MACs={stats.macs}"
        f" FLOPs≈{stats.flops}"
        f" input={in_channels}x{input_res[0]}x{input_res[1]}"
    )
    return stats
