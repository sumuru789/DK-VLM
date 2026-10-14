"""Shared answer parsing, numeric scoring, and distributed evaluation utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def fuzzy_matching(pred: str) -> str:
    """Keep the first whitespace-delimited token, which is where the answer lives."""
    if pred is None:
        return ""
    pred = str(pred).strip()
    return pred.split(" ")[0].rstrip(".").strip() if pred else ""


def to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Fraction of thresholds at which the relative error is small enough to count."""
    if target == 0:
        return 0.0
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    abs_dist_norm = abs(pred - target) / target
    return float((abs_dist_norm <= 1 - conf_intervs).mean())


def setup_distributed() -> Tuple[torch.device, int, int]:
    """Initialize the process group when launched under torchrun."""
    if os.environ.get("LOCAL_RANK") is None:
        return (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            0,
            1,
        )
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, dist.get_rank(), dist.get_world_size()


def gather_results(payload: Dict[str, Any], world_size: int) -> List[Dict[str, Any]]:
    if world_size <= 1:
        return [payload]
    import torch.distributed as dist

    bucket: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(bucket, payload)
    return [b for b in bucket if b is not None]
