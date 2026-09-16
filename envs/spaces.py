from __future__ import annotations

"""Gymnasium spaces shared by models and the annotation environment."""

import numpy as np

from actions.space import ActionLayout


def action_space(layout: ActionLayout):
    from gymnasium import spaces
    # Unbounded normalized residual coordinates: soft factor bounds remain soft.
    return spaces.Box(low=-np.inf, high=np.inf, shape=(layout.size,), dtype=np.float32)


def observation_space(size: int):
    from gymnasium import spaces
    return spaces.Box(low=-np.inf, high=np.inf, shape=(int(size),), dtype=np.float32)
