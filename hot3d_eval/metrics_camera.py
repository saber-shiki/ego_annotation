"""Camera ATE / adjacent-pair RPE. Input is already first-frame registered."""
import numpy as np
from .alignment import fit_points, transform_points
from .metrics_hand import metric


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def evaluate_camera(pred, gt, frame_ids):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    n = len(pred)
    p, g = pred[:, :3, 3], gt[:, :3, 3]
    out = {'ATE_first_mm': metric(rmse(np.linalg.norm(p - g, axis=-1)) * 1000, n, n)}
    for key, scale in [('ATE_SE3_mm', False), ('ATE_Sim3_mm', True)]:
        try:
            mapped = transform_points(fit_points(p, g, scale), p)
            out[key] = metric(rmse(np.linalg.norm(mapped - g, axis=-1)) * 1000, n, n)
        except ValueError as e:
            out[key] = metric(None, 0, n, str(e))
    translation, rotation = [], []
    for i in range(n - 1):
        if frame_ids[i + 1] != frame_ids[i] + 1:
            continue
        dp = np.linalg.inv(pred[i]) @ pred[i + 1]
        dg = np.linalg.inv(gt[i]) @ gt[i + 1]
        error = np.linalg.inv(dg) @ dp
        translation.append(np.linalg.norm(error[:3, 3]) * 1000)
        rotation.append(np.degrees(np.arccos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1))))
    out['RPE_T_mm'] = metric(rmse(translation) if translation else None, len(translation), max(0, n - 1))
    out['RPE_R_deg'] = metric(rmse(rotation) if rotation else None, len(rotation), max(0, n - 1))
    return out
