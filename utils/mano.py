from __future__ import annotations
"""MANO asset loading, fixed topology, and supported parameter interpolation."""
from pathlib import Path
import inspect
import numpy as np

TIP_VERTICES = np.array([744, 320, 443, 554, 671])


JOINT_ORDER = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
)


def mano_models(path, device="cpu"):
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in [
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        if name not in np.__dict__:
            setattr(np, name, value)
    import smplx

    models = {
        side: smplx.MANO(
            str(Path(path) / f"MANO_{side.upper()}.pkl"),
            is_rhand=(side == "right"),
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)
        for side in ["left", "right"]
    }
    models["left"].shapedirs[:, 0, :] *= -1
    return models


def hand_validity(z, side, count, fitted=False):
    suffixes = ("vertices_world",) if fitted else ("vertices_world_m", "joints_world_m", "root_orient_axis_angle", "hand_pose_axis_angle", "betas", "trans_world_m")
    keys = [side + "_" + s for s in suffixes]
    present = [k in z for k in keys]
    raw = np.asarray(z.get(side + "_valid", np.ones(count) if all(present) else np.zeros(count)))
    if raw.shape != (count,) or not np.isin(raw, [0, 1]).all():
        raise ValueError(f"{side}: invalid validity mask")
    valid = raw.astype(bool)
    if not all(present):
        if valid.any():
            raise ValueError(f"{side}: valid hand lacks MANO state")
        return valid
    for k in keys:
        value = np.asarray(z[k])
        if len(value) != count:
            raise ValueError(f"{k}: timeline length mismatch")
        valid &= np.isfinite(value.reshape(count, -1)).all(axis=1)
    return valid


def supported_segments(valid):
    indices = np.flatnonzero(valid)
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def fit_keyframes(count, valid, stride=5):
    keys = [0, count - 1, *range(0, count, stride)]
    for mask in valid.values():
        for segment in supported_segments(mask):
            keys.extend([segment[0], segment[-1]])
    return np.unique(keys)


def interpolate_supported(times, keys, values, valid):
    """Zero correction outside observed segments; never bridge a missing-hand gap."""
    full = np.zeros((len(times), *values.shape[1:]), dtype=values.dtype)
    for segment in supported_segments(valid):
        chosen = np.flatnonzero(np.isin(keys, segment))
        if not len(chosen):
            continue
        flat = values[chosen].reshape(len(chosen), -1)
        full[segment] = np.stack([np.interp(times[segment], times[keys[chosen]], flat[:, j]) for j in range(flat.shape[1])], axis=1).reshape(len(segment), *values.shape[1:])
    return full

def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    import torch
    return torch.linalg.matrix_exp(skew(rotvec))

def skew(rotvec: torch.Tensor) -> torch.Tensor:
    import torch
    (x, y, z) = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack([torch.stack([zero, -z, y], dim=-1), torch.stack([z, zero, -x], dim=-1), torch.stack([-y, x, zero], dim=-1)], dim=-2)

def load_source_arrays(cache: dict[Path, Any], path: Path) -> dict[str, np.ndarray]:
    from typing import Any
    import numpy as np
    from pathlib import Path
    if path not in cache:
        with np.load(path, allow_pickle=True) as z:
            cache[path] = {key: np.asarray(z[key]) for key in z.files}
    return cache[path]

def similarity_from_to(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError('similarity expects matching Nx3 arrays')
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    src = source - source_center
    tgt = target - target_center
    covariance = src.T @ tgt / len(source)
    (u, s, vt) = np.linalg.svd(covariance)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    var = float(np.sum(src * src) / len(source))
    scale = float(np.sum(s) / max(var, 1e-12))
    trans = target_center - scale * (source_center @ rot.T)
    pred = scale * (source @ rot.T) + trans[None, :]
    err = np.linalg.norm(pred - target, axis=1)
    return (scale, rot, trans, err)

def bridge_vertices_and_joints(hand: dict[str, Any], bridge_cache: dict[Path, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    from typing import Any
    import numpy as np
    from pathlib import Path
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    reference = metric.get('vertices_reference') if isinstance(metric.get('vertices_reference'), dict) else {}
    bridge_path_raw = reference.get('bridge_npz')
    vertices_array = reference.get('bridge_vertices_world_array')
    row_index_raw = reference.get('bridge_row_index')
    if not isinstance(bridge_path_raw, str) or not isinstance(vertices_array, str) or row_index_raw is None:
        return None
    bridge_path = Path(bridge_path_raw)
    if not bridge_path.exists():
        return None
    row_index = int(row_index_raw)
    vertices_world = load_bridge_array(bridge_cache, bridge_path, vertices_array, row_index)
    joints_world = load_bridge_array(bridge_cache, bridge_path, 'joints_current_v18_world_from_hawor_projection_relift_m', row_index)
    return (vertices_world, joints_world)

def load_bridge_array(cache: dict[Path, Any], bridge_path: Path, array_name: str, row_index: int) -> np.ndarray:
    from typing import Any
    import numpy as np
    from pathlib import Path
    if bridge_path not in cache:
        with np.load(bridge_path, allow_pickle=True) as z:
            needed = {'vertices_current_v18_world_from_hawor_projection_relift_m', 'joints_current_v18_world_from_hawor_projection_relift_m'}
            cache[bridge_path] = {key: np.asarray(z[key]) for key in z.files if key in needed or key == array_name}
    return np.asarray(cache[bridge_path][array_name][row_index], dtype=float)

def source_npz_for_hand(hand: dict[str, Any]) -> tuple[Path, int] | None:
    from typing import Any
    from pathlib import Path
    metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
    params = metric.get('mano_params') if isinstance(metric.get('mano_params'), dict) else {}
    source = params.get('source_hawor_npz')
    source_frame = params.get('source_frame_index')
    if not isinstance(source, str) or source_frame is None:
        ref = metric.get('vertices_reference') if isinstance(metric.get('vertices_reference'), dict) else {}
        source = ref.get('source_hawor_npz')
        source_frame = ref.get('source_frame_index')
    if not isinstance(source, str) or source_frame is None:
        return None
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.glob('*with_track_support*.npz')) + sorted(path.glob('*.npz'))
        if not candidates:
            return None
        path = candidates[0]
    if not path.exists():
        return None
    return (path, int(source_frame))

def load_models(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from typing import Any
    import torch
    import argparse
    patch_legacy_mano_loader()
    mano_cls = load_wilor_mano_class(args.wilor_root)
    right_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / 'mano_data' / 'MANO_RIGHT.pkl'
    models: dict[str, Any] = {}
    if 'right' in args.sides:
        if not right_path.exists():
            raise FileNotFoundError(f'missing right MANO model: {right_path}')
        models['right'] = mano_cls(model_path=str(right_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    if 'left' in args.sides:
        left_path = args.wilor_mano_left
        if left_path is None or not left_path.exists():
            raise FileNotFoundError(f'missing left MANO model: {left_path}')
        left_model = mano_cls(model_path=str(left_path), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
        if bool(args.hawor_left_shapedirs_x_fix):
            with torch.no_grad():
                left_model.shapedirs[:, 0, :] *= -1
        models['left'] = left_model
    for m in models.values():
        m.eval()
    return models

def load_wilor_mano_class(wilor_root: Path):
    import importlib.util
    from pathlib import Path
    path = wilor_root / 'wilor' / 'models' / 'mano_wrapper.py'
    if not path.exists():
        raise FileNotFoundError(f'missing WiLoR MANO wrapper: {path}')
    spec = importlib.util.spec_from_file_location('wilor_mano_wrapper_v18_articulated', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load WiLoR MANO wrapper spec from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANO

def patch_legacy_mano_loader() -> None:
    import numpy as np
    import inspect
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for (name, value) in [('bool', np.bool_), ('int', int), ('float', float), ('complex', complex), ('object', object), ('unicode', str), ('str', str)]:
        if not hasattr(np, name):
            setattr(np, name, value)
