"""Rigid/similarity transforms; all points are row vectors in metres."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform_points(T, points):
    T, points = np.asarray(T, float), np.asarray(points, float)
    return points @ T[:3, :3].T + T[:3, 3]


def pose(t, q_wxyz):
    q = np.asarray(q_wxyz, float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = t
    return T


def validate_pose(T):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Camera/object pose must be finite 4x4')
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-4)):
        raise ValueError('Expected SE(3), not a scaled/reflected pose')
    return T


def first_frame_registration(gt_camera, pred_camera):
    return validate_pose(gt_camera) @ np.linalg.inv(validate_pose(pred_camera))


def fit_points(pred, gt, scale=False):
    """Umeyama: return homogeneous map from pred to gt (no reflection).

    Rank-one trajectories admit nonunique rotations; SVD selects one optimum.
    Constant predictions permit SE3 translation but not a scale estimate.
    """
    x, y = np.asarray(pred, float).reshape(-1, 3), np.asarray(gt, float).reshape(-1, 3)
    if x.shape != y.shape or len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Alignment needs >=2 finite paired points')
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    U, singular, Vt = np.linalg.svd(yc.T @ xc / len(x))
    D = np.ones(3)
    D[-1] = np.linalg.det(U @ Vt)
    R = (U * D) @ Vt
    variance = np.mean(np.sum(xc ** 2, axis=1))
    if scale and variance < 1e-20:
        raise ValueError('Sim3 scale undefined for constant prediction')
    s = float(singular @ D / variance) if scale else 1.0
    if scale and s <= 1e-12:
        raise ValueError('Sim3 optimum has zero scale')
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = my - s * R @ mx
    return T


def interpolate_pose(T0, T1, fraction):
    """Translation-linear, SO(3)-geodesic pose interpolation."""
    T = np.eye(4)
    T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))([fraction]).as_matrix()[0]
    T[:3, 3] = (1 - fraction) * T0[:3, 3] + fraction * T1[:3, 3]
    return T
