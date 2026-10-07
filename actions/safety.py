"""Fail-closed physical action projection for the RL environment.

The policy emits normalized, unconstrained residuals.  This module is the
single place that turns them into executable residuals.  It deliberately does
not alter the policy distribution or its raw action trace; callers must retain
both values and the per-dimension reason for every zero/projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from actions.scaling import ActionScales
from actions.space import ActionLayout


@dataclass(frozen=True)
class HardActionBounds:
    """Per-step residual bounds in physical units.

    Object translation bounds are expressed in the camera frame, matching the
    flow-chart contract.  ``object_translation_frame`` controls how the safe
    camera-frame delta is converted back for the environment decoder.
    """

    hand_root_rad: float = 0.15
    hand_articulation_rad: float = 0.35
    hand_translation_m: float = 0.025
    object_rotation_rad: float = 0.08
    object_translation_xy_m: float = 0.015
    object_translation_z_m: float = 0.035
    hand_log_scale_min: float = float(np.log(0.85))
    hand_log_scale_max: float = float(np.log(1.18))
    object_log_scale_min: float = float(np.log(0.85))
    object_log_scale_max: float = float(np.log(1.18))
    contact_logit_abs: float = 4.0
    enabled: bool = True

    def __post_init__(self) -> None:
        values = (
            self.hand_root_rad,
            self.hand_articulation_rad,
            self.hand_translation_m,
            self.object_rotation_rad,
            self.object_translation_xy_m,
            self.object_translation_z_m,
            abs(self.hand_log_scale_min), abs(self.hand_log_scale_max),
            abs(self.object_log_scale_min), abs(self.object_log_scale_max),
            self.contact_logit_abs,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("hard action bounds must be positive finite values")
        if self.hand_log_scale_min >= self.hand_log_scale_max or self.object_log_scale_min >= self.object_log_scale_max:
            raise ValueError("log-scale hard bounds must be ordered")

    @classmethod
    def from_config(cls, value: object | None) -> "HardActionBounds":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("hard_action_bounds must be a mapping")
        fields = {
            "hand_root_rad", "hand_articulation_rad", "hand_translation_m",
            "object_rotation_rad", "object_translation_xy_m",
            "object_translation_z_m", "hand_log_scale_min", "hand_log_scale_max",
            "object_log_scale_min", "object_log_scale_max", "contact_logit_abs", "enabled",
        }
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown hard_action_bounds fields: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class SafetyProjection:
    normalized_action: np.ndarray
    executed_delta: np.ndarray
    allowed_mask: np.ndarray
    reasons: tuple[str, ...]
    constraint_ids: tuple[str, ...]
    violation_magnitudes: np.ndarray
    projected: np.ndarray

    def __post_init__(self) -> None:
        for value in (
            self.normalized_action,
            self.executed_delta,
            self.allowed_mask,
            self.violation_magnitudes,
            self.projected,
        ):
            value.setflags(write=False)

    def document(self) -> dict[str, object]:
        return {
            "normalized_safe_action": self.normalized_action.tolist(),
            "executed_delta": self.executed_delta.tolist(),
            "allowed_mask": self.allowed_mask.astype(bool).tolist(),
            "reasons": list(self.reasons),
            "constraint_ids": list(self.constraint_ids),
            "violation_magnitudes": self.violation_magnitudes.tolist(),
            "projected": self.projected.astype(bool).tolist(),
            "hard_safety": True,
        }


def _clip_block(
    physical: np.ndarray,
    block_slice: slice,
    bound: float,
    *,
    constraint_id: str,
    reasons: list[str],
    constraint_ids: list[str],
    violation: np.ndarray,
    projected: np.ndarray,
) -> None:
    before = physical[block_slice].copy()
    after = np.clip(before, -float(bound), float(bound))
    physical[block_slice] = after
    amount = np.abs(before - after)
    changed = amount > 0
    violation[block_slice] = np.maximum(violation[block_slice], amount)
    projected[block_slice] |= changed
    for index, changed_here in zip(range(block_slice.start, block_slice.stop), changed):
        if changed_here:
            reasons[index] = f"hard_bound:{constraint_id}"
            constraint_ids[index] = constraint_id


def _clip_interval_block(
    physical: np.ndarray,
    block_slice: slice,
    low: float,
    high: float,
    *,
    constraint_id: str,
    reasons: list[str],
    constraint_ids: list[str],
    violation: np.ndarray,
    projected: np.ndarray,
) -> None:
    before = physical[block_slice].copy()
    after = np.clip(before, float(low), float(high))
    physical[block_slice] = after
    amount = np.abs(before - after)
    changed = amount > 0
    violation[block_slice] = np.maximum(violation[block_slice], amount)
    projected[block_slice] |= changed
    for index, changed_here in zip(range(block_slice.start, block_slice.stop), changed):
        if changed_here:
            reasons[index] = f"hard_bound:{constraint_id}"
            constraint_ids[index] = constraint_id


def project_action(
    raw_action: Sequence[float],
    layout: ActionLayout,
    scales: ActionScales,
    allowed_mask: Sequence[bool],
    *,
    T_world_camera: np.ndarray | None = None,
    object_translation_frame: str = "world",
    hard_bounds: HardActionBounds | None = None,
    reason_by_dimension: Sequence[str] | None = None,
) -> SafetyProjection:
    """Apply semantic masks and hard physical bounds to one policy action.

    The projection is component-wise for hand/action blocks.  Object
    translation is clipped in camera coordinates and transformed back to the
    declared decoder frame.  The function is pure, so it can be tested
    independently of MANO and the dataset.
    """
    raw = np.asarray(raw_action, dtype=np.float32).copy()
    allowed = np.asarray(allowed_mask, dtype=bool).copy()
    if raw.shape != (layout.size,) or allowed.shape != (layout.size,):
        raise ValueError("raw action/gate mask shape mismatch")
    if not np.isfinite(raw).all():
        raise ValueError("raw action must be finite")
    bounds = hard_bounds or HardActionBounds()
    if object_translation_frame not in {"world", "camera"}:
        raise ValueError("object_translation_frame must be 'world' or 'camera'")
    normalized = np.where(allowed, raw, 0.0).astype(np.float32)
    physical = normalized.astype(np.float64) * scales.vector(layout).astype(np.float64)
    reasons = list(reason_by_dimension or ("allowed" if value else "semantic_gate" for value in allowed))
    if len(reasons) != layout.size:
        raise ValueError("gate reason shape mismatch")
    constraint_ids = ["none" for _ in range(layout.size)]
    violation = np.zeros(layout.size, dtype=np.float32)
    projected = np.zeros(layout.size, dtype=bool)

    # A disabled safety layer is an explicit legacy escape hatch for tests and
    # historical comparisons; production configs leave it enabled.
    if bounds.enabled:
        for block in layout.blocks:
            if block.kind == "hand_root":
                _clip_block(physical, block.slice, bounds.hand_root_rad,
                            constraint_id="G8_hand_root_rad", reasons=reasons,
                            constraint_ids=constraint_ids, violation=violation,
                            projected=projected)
            elif block.kind == "hand_articulation":
                _clip_block(physical, block.slice, bounds.hand_articulation_rad,
                            constraint_id="G8_hand_articulation_rad", reasons=reasons,
                            constraint_ids=constraint_ids, violation=violation,
                            projected=projected)
            elif block.kind == "hand_translation":
                _clip_block(physical, block.slice, bounds.hand_translation_m,
                            constraint_id="G8_hand_translation_m", reasons=reasons,
                            constraint_ids=constraint_ids, violation=violation,
                            projected=projected)
            elif block.kind == "hand_scale":
                _clip_interval_block(physical, block.slice, bounds.hand_log_scale_min,
                                     bounds.hand_log_scale_max,
                                     constraint_id="G8_hand_scale",
                                     reasons=reasons, constraint_ids=constraint_ids,
                                     violation=violation, projected=projected)
            elif block.kind == "object_scale":
                _clip_interval_block(physical, block.slice, bounds.object_log_scale_min,
                                     bounds.object_log_scale_max,
                                     constraint_id="G8_object_shared_scale",
                                     reasons=reasons, constraint_ids=constraint_ids,
                                     violation=violation, projected=projected)
            elif block.kind == "object_rotation":
                _clip_block(physical, block.slice, bounds.object_rotation_rad,
                            constraint_id="G8_object_rotation_rad", reasons=reasons,
                            constraint_ids=constraint_ids, violation=violation,
                            projected=projected)
            elif block.kind == "contact":
                _clip_block(physical, block.slice, bounds.contact_logit_abs,
                            constraint_id="G8_contact_logit", reasons=reasons,
                            constraint_ids=constraint_ids, violation=violation,
                            projected=projected)

        for block in layout.blocks:
            if block.kind != "object_translation":
                continue
            world_delta = physical[block.slice].copy()
            if object_translation_frame == "camera":
                camera_delta = world_delta
                rotation = None
            elif T_world_camera is None:
                camera_delta = world_delta
                rotation = None
            else:
                transform = np.asarray(T_world_camera, dtype=np.float64)
                if transform.shape != (4, 4) or not np.isfinite(transform).all():
                    raise ValueError("T_world_camera must be finite shape (4, 4)")
                rotation = transform[:3, :3]
                camera_delta = world_delta @ rotation
            before = camera_delta.copy()
            camera_delta[0:2] = np.clip(camera_delta[0:2],
                                        -bounds.object_translation_xy_m,
                                        bounds.object_translation_xy_m)
            camera_delta[2] = np.clip(camera_delta[2],
                                      -bounds.object_translation_z_m,
                                      bounds.object_translation_z_m)
            amount = np.abs(before - camera_delta)
            if np.any(amount > 0):
                projected[block.slice] |= amount > 0
                violation[block.slice] = np.maximum(violation[block.slice], amount.astype(np.float32))
                ids = ("G8_object_translation_camera_xy",
                       "G8_object_translation_camera_xy",
                       "G8_object_translation_camera_z")
                for local, changed in enumerate(amount > 0):
                    if changed:
                        reasons[block.start + local] = f"hard_bound:{ids[local]}"
                        constraint_ids[block.start + local] = ids[local]
            physical[block.slice] = camera_delta if rotation is None else camera_delta @ rotation.T

    # Semantic gates are hard zeros and must remain visible even when the
    # physical bound projection did not change the action.
    for index, permitted in enumerate(allowed):
        if not permitted:
            normalized[index] = 0.0
            physical[index] = 0.0
            if constraint_ids[index] == "none":
                constraint_ids[index] = "semantic_gate"
    scales_vector = scales.vector(layout).astype(np.float64)
    safe_normalized = (physical / scales_vector).astype(np.float32)
    return SafetyProjection(
        normalized_action=safe_normalized,
        executed_delta=physical.astype(np.float32),
        allowed_mask=allowed,
        reasons=tuple(reasons),
        constraint_ids=tuple(constraint_ids),
        violation_magnitudes=violation,
        projected=projected,
    )
