from __future__ import annotations

def depth_violation(depths, weights, cfg):
    import numpy as np
    return float(np.sum(weights * np.maximum(cfg.near_depth_m - depths, 0)))
