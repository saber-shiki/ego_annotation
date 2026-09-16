from __future__ import annotations

"""Unit-explicit action scaling and semantic hard gating."""

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from actions.space import ActionLayout


@dataclass(frozen=True)
class ActionScales:
    hand_root_rad: float = 0.12
    hand_articulation_rad: float = 0.18
    hand_translation_m: float = 0.018
    object_rotation_rad: float = 0.08
    object_translation_m: float = 0.02
    contact_logit: float = 1.0

    def __post_init__(self) -> None:
        if any((not np.isfinite(value) or value <= 0.0) for value in self.__dict__.values()):
            raise ValueError("all action scales must be positive finite unit conversions")

    def vector(self, layout: ActionLayout) -> np.ndarray:
        values = {
            "hand_root": self.hand_root_rad,
            "hand_articulation": self.hand_articulation_rad,
            "hand_translation": self.hand_translation_m,
            "object_rotation": self.object_rotation_rad,
            "object_translation": self.object_translation_m,
            "contact": self.contact_logit,
        }
        result = np.empty(layout.size, dtype=np.float32)
        for block in layout.blocks:
            result[block.slice] = values[block.kind]
        return result


@dataclass(frozen=True)
class GateTrace:
    raw_action: np.ndarray
    normalized_executed_action: np.ndarray
    executed_delta: np.ndarray
    allowed_mask: np.ndarray
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (self.raw_action, self.normalized_executed_action,
                      self.executed_delta, self.allowed_mask):
            value.setflags(write=False)

    def document(self) -> dict:
        return {
            "raw_action": self.raw_action.tolist(),
            "normalized_executed_action": self.normalized_executed_action.tolist(),
            "executed_delta": self.executed_delta.tolist(),
            "allowed_mask": self.allowed_mask.astype(bool).tolist(),
            "reasons": list(self.reasons),
        }


def scale_action(raw_action: Sequence[float], layout: ActionLayout, scales: ActionScales) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape != (layout.size,) or not np.isfinite(raw).all():
        raise ValueError(f"raw_action must be finite shape {(layout.size,)}")
    # Deliberately no tanh/clipping: legacy magnitude limits are soft factor terms.
    return raw * scales.vector(layout)


def filter_scaled_action(
    raw_action: Sequence[float],
    layout: ActionLayout,
    scales: ActionScales,
    allowed_mask: Sequence[bool],
    *,
    reason_by_dimension: Sequence[str] | None = None,
) -> GateTrace:
    raw = np.asarray(raw_action, dtype=np.float32).copy()
    allowed = np.asarray(allowed_mask, dtype=bool).copy()
    if raw.shape != (layout.size,) or allowed.shape != (layout.size,):
        raise ValueError("raw action/gate mask shape mismatch")
    normalized = np.where(allowed, raw, 0.0).astype(np.float32)
    executed = scale_action(normalized, layout, scales)
    if reason_by_dimension is None:
        reasons = tuple("allowed" if value else "hard_gate_filtered" for value in allowed)
    else:
        if len(reason_by_dimension) != layout.size:
            raise ValueError("gate reason shape mismatch")
        reasons = tuple(str(value) for value in reason_by_dimension)
    return GateTrace(raw, normalized, executed, allowed, reasons)
