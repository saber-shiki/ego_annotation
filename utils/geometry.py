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

def camera(points, T):
    return (points - T[:3, 3]) @ T[:3, :3]


def world(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def project(points, K):
    return points[..., :2] / np.maximum(points[..., 2:], 1e-6) * K[:2] + K[2:]


def resized_K(K, source=(1408, 1408), target=(960, 960)):
    s = np.asarray(target, dtype=float) / np.asarray(source, dtype=float)
    K = np.asarray(K, dtype=float)
    if (K.shape != (4,) or not np.isfinite(K).all() or np.any(K[:2] <= 0)
            or s.shape != (2,) or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError("invalid intrinsics or image resize")
    return np.r_[K[:2] * s, (K[2:] + 0.5) * s - 0.5]


def torch_rodrigues(x):
    import torch

    z = torch.zeros_like(x[..., 0])
    a, b, c = x.unbind(-1)
    mat = torch.stack([z, -c, b, c, z, -a, -b, a, z], -1).reshape(*x.shape[:-1], 3, 3)
    return torch.matrix_exp(mat)


def torch_project(p, K):
    return p[..., :2] / p[..., 2:].clamp_min(0.01) * K[..., :2] + K[..., 2:]


def validate_transforms(values):
    T = np.asarray(values, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (4, 4) or not np.isfinite(T).all():
        raise ValueError("invalid camera transforms")
    R = T[:, :3, :3]
    if not (np.allclose(T[:, 3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-4)
            and np.allclose(np.linalg.det(R), 1, atol=1e-4)):
        raise ValueError("camera transforms must be rigid world-from-camera")
    return T

def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise RuntimeError(f"SAM3D rotation must be finite wxyz quaternion, got {q}")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-3:
        raise RuntimeError(f"SAM3D quaternion norm is invalid: {norm}")
    w, x, y, z = (q / norm).tolist()
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def apply_native_pose(
    vertices_local: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    vertices = np.asarray(vertices_local, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise RuntimeError(f"SAM3D translation must be finite xyz, got {translation}")
    if scale.shape == (1,):
        scale = np.repeat(scale, 3)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise RuntimeError(f"SAM3D scale must be positive scalar/xyz, got {scale}")
    rotation = quaternion_wxyz_to_matrix(rotation_wxyz)
    vertices_p3d = (vertices * scale[None, :]) @ rotation + translation[None, :]
    p3d_to_opencv = np.diag([-1.0, -1.0, 1.0])
    vertices_opencv = vertices_p3d @ p3d_to_opencv
    return vertices_p3d, vertices_opencv, {
        "quaternion_order": "wxyz_scalar_first_pytorch3d",
        "row_vector_formula": "p_p3d_camera = (p_raw_local * scale_xyz) @ quaternion_to_matrix(q_wxyz) + translation_xyz",
        "pytorch3d_camera_convention": "x_left_y_up_z_forward",
        "opencv_camera_convention": "x_right_y_down_z_forward",
        "p3d_to_opencv_row_matrix": p3d_to_opencv.tolist(),
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "scale_xyz": scale.tolist(),
        "metric_status": "native_monocular_scene_units_not_sensor_meters",
    }
