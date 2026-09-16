from __future__ import annotations

"""Reduction helpers for reward accounting over existing pure factor kernels."""

from typing import Any, Callable
import numpy as np


def scipy_least_squares_cost(residual: Any, loss: str) -> float:
    """Return SciPy ``least_squares``' objective ``0.5 * sum(rho(r**2))``.

    In particular, soft-L1 is ``sum(sqrt(1 + r**2) - 1)``. Keeping the
    optimizer convention here prevents selectively doubling object factors.
    """
    value = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite robust residual")
    if loss not in {"linear", "soft_l1"}:
        raise ValueError(f"unsupported least-squares loss {loss!r}")
    # The shared kernel stores rho(r**2). SciPy reports 0.5 * sum(rho),
    # so the reward reduction wraps that source rather than duplicating it.
    from rewards.terms.common import robust_cost
    return 0.5 * float(robust_cost(value, loss))


def contiguous_segments(valid: np.ndarray, frame_idx: np.ndarray) -> list[np.ndarray]:
    """Return valid, frame-consecutive row segments."""
    mask = np.asarray(valid, dtype=bool)
    frames = np.asarray(frame_idx, dtype=np.int64)
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    cuts = np.flatnonzero((np.diff(ids) != 1) | (np.diff(frames[ids]) != 1)) + 1
    return [segment for segment in np.split(ids, cuts) if len(segment)]


def aggregate_temporal(
    value: Any,
    valid: np.ndarray,
    frame_idx: np.ndarray,
    weight: float,
    kernel: Callable[[Any, float], Any],
    order: int,
) -> tuple[Any, float, float, int]:
    """Aggregate one global mean over all admitted edges or triples.

    The existing kernel remains the numerical source. Segment kernel means are
    recombined by their element counts, rather than summed with equal segment
    weight.
    """
    import torch

    total = value.new_zeros(())
    numerator = 0.0
    denominator = 0
    group_count = 0
    for segment in contiguous_segments(valid, frame_idx):
        required = order + 1
        if len(segment) < required:
            continue
        selected = value[torch.as_tensor(segment, dtype=torch.long, device=value.device)]
        groups = len(segment) - order
        count = groups * int(np.prod(tuple(value.shape[1:])))
        if count <= 0:
            continue
        local = kernel(selected, float(weight))
        total = total + local * count
        numerator += float(local.detach().cpu()) * count
        denominator += count
        group_count += groups
    if denominator:
        total = total / denominator
        numerator /= denominator
    return total, numerator, float(denominator), group_count
