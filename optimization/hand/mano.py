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
