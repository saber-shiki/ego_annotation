"""Row-vector metric transforms; rotations map canonical/camera coordinates to world."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("array contains non-finite values")
    return result


def valid_rotation(value: Any) -> bool:
    try:
        matrix = _array(value, (3, 3))
    except (TypeError, ValueError):
        return False
    return bool(np.linalg.det(matrix) > 0.0 and np.linalg.norm(matrix.T @ matrix - np.eye(3)) < 1.0e-3)


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = _array(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    return points @ _array(rotation, (3, 3)).T + _array(translation, (3,))[None, :]


def inverse_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = _array(points)
    return (points - _array(translation, (3,))[None, :]) @ _array(rotation, (3, 3))


def camera_to_world(camera_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """Convert camera-frame row vectors to world exactly once."""
    T = _array(T_world_camera, (4, 4))
    return apply_pose(camera_points, T[:3, :3], T[:3, 3])


def apply_left_correction(
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    delta_rotation: np.ndarray,
    delta_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the V20 additive-translation/left-SO(3) correction contract."""
    return (
        Rotation.from_rotvec(_array(delta_rotation, (3,))).as_matrix()
        @ _array(initial_rotation, (3, 3)),
        _array(initial_translation, (3,)) + _array(delta_translation, (3,)),
    )


def pose_correction(
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    final_rotation: np.ndarray,
    final_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover additive correction from two poses under the left-SO(3) model."""
    delta_rotation = _array(final_rotation, (3, 3)) @ _array(initial_rotation, (3, 3)).T
    return Rotation.from_matrix(delta_rotation).as_rotvec(), _array(final_translation, (3,)) - _array(initial_translation, (3,))
