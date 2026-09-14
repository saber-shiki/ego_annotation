"""Shared typed numeric records and JSON IO; no business dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np
import json

@dataclass
class PoseNode:
    frame_idx: int
    initial_rotation: np.ndarray
    initial_translation: np.ndarray
    observed_world: np.ndarray | None
    observation_source: str
    metric_observation: bool
    uncertainty: float
    source_row: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RGBEdge:
    source_frame_idx: int
    target_frame_idx: int
    rotation_source_to_target_world: np.ndarray
    translation_source_to_target_world_m: np.ndarray
    weight: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    translation_reference_rotation: np.ndarray | None = None
    translation_source_support_world: np.ndarray | None = None


@dataclass(frozen=True)
class RGBReprojectionFactor:
    source_frame_idx: int
    target_frame_idx: int
    source_pos: int
    target_pos: int
    source_world: np.ndarray
    target_uv: np.ndarray
    target_intrinsics: np.ndarray
    target_camera_rotation: np.ndarray
    target_camera_translation: np.ndarray
    weights: np.ndarray
    edge_weight: float
    rotation_eligible: bool = True


@dataclass(frozen=True)
class PointFactor:
    source_pos: int
    target_pos: int
    source_world: np.ndarray
    target_world: np.ndarray
    target_normals: np.ndarray
    weights: np.ndarray
    kind: str = "p09_observed_rebuilt"


@dataclass
class SolverConfig:
    """Only the residual/correspondence options consumed by the extracted kernels."""
    max_points: int = 350
    min_pairs: int = 12
    max_correspondence_m: float = 0.08
    observed_factor_mode: str = "legacy_nn"
    rgb_reprojection_weight_scale: float = 0.0
    sigma_rgb_reprojection_px: float = 2.0
    max_rgb_reprojection_residual_px: float = 40.0
    seed: int = 20260907


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


RKEY = 'rotation_world_from_completed_canonical_matrix'


TKEY = 'translation_world_m'


def indexed(rows):
    result = {}
    for row in rows:
        idx = row['frame_idx']
        if type(idx) is not int or idx in result:
            raise ValueError('duplicate/noninteger frame ID')
        result[idx] = row
    return result


def time_keyframes(ids, times, spacing):
    if not ids or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('positive time spacing required')
    keys = [ids[0]]
    for idx in ids[1:]:
        if times[idx] - times[keys[-1]] >= spacing:
            keys.append(idx)
    if ids[-1] != keys[-1]:
        keys.append(ids[-1])
    return keys


def time_source_pairs(ids, keys, sources, span, times, max_gap):
    if span < 1 or max_gap <= 0:
        raise ValueError('positive pair planning span/gap required')
    planned = set(zip(ids[:-1], ids[1:]))
    for hop in range(1, span + 1):
        planned.update(zip(keys[:-hop], keys[hop:]))
    if len(keys) > 2:
        planned.update([(keys[0], keys[-1]), (keys[0], keys[len(keys)//2]), (keys[len(keys)//2], keys[-1])])
    pairs, rejected = set(), []
    for a, b in planned:
        if a in sources: pairs.add((a, b))
        elif b in sources: pairs.add((b, a))
        else: rejected.append({'source_frame_idx': a, 'target_frame_idx': b, 'reason': 'neither_endpoint_has_retained_depth'})
    for b in ids:
        if b in sources: continue
        before = [a for a in sources if 0 < times[b] - times[a] <= max_gap]
        after = [a for a in sources if 0 < times[a] - times[b] <= max_gap]
        if before: pairs.add((max(before, key=times.get), b))
        if after: pairs.add((min(after, key=times.get), b))
    return sorted(pairs, key=lambda p: (times[p[0]], times[p[1]])), rejected
