"""Named quadratic/hinge kernels; normalization is explicit at each call site."""
from __future__ import annotations

def mean_square(value, sigma=1.0, weight=1.0):
    return weight * (value / sigma).square().mean()

def magnitude_hinge(value, onset, weight=1.0):
    import torch
    return weight * torch.relu(torch.linalg.vector_norm(value, dim=-1) - onset).square().mean()

def robust_cost(residual, loss):
    import numpy as np
    return float(np.sum(2 * (np.sqrt(1 + residual ** 2) - 1) if loss == 'soft_l1' else residual ** 2))
