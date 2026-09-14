from __future__ import annotations

def unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    arr = x.reshape(n, 6)
    return (arr[:, :3], arr[:, 3:6])

def corrected_pose(obs: PoseObservation, rot_delta: np.ndarray, trans_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from utils.data import PoseObservation
    from scipy.spatial.transform import Rotation
    import numpy as np
    r_corr = Rotation.from_rotvec(rot_delta).as_matrix() @ obs.rotation_world_from_canonical
    t_corr = obs.translation_world_m + trans_delta
    return (r_corr, t_corr)

def canonical_gauge(native_cv, scale, T, pose):
    from utils.geometry import validate_transforms
    from observations.object.shape import digest_json
    import numpy as np
    from observations.object.shape import _pose
    T = validate_transforms([T])[0]
    (R, t) = _pose(pose)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid scene similarity scale')
    world = native_cv * scale @ T[:3, :3].T + T[:3, 3]
    canonical = (world - t) @ R
    replay = (canonical @ R.T + t - T[:3, 3]) @ T[:3, :3] / scale
    error = float(np.max(np.abs(replay - native_cv)))
    if error > 1e-05:
        raise ValueError('native/canonical gauge roundtrip failed')
    transform = np.eye(4)
    transform[:3, :3] = scale * R.T @ T[:3, :3]
    transform[:3, 3] = R.T @ (T[:3, 3] - t)
    return (canonical, {'T_original_canonical_from_native_opencv': transform.tolist(), 'T_world_camera_metric': T.tolist(), 'initializer_pose': pose, 'initializer_pose_sha256': digest_json(pose), 'roundtrip_max_abs_native_scene_error': error, 'formula': 'Vcanonical=((s*Vnative)@Rcamera.T+tcamera-tinitializer)@Rinitializer'})
