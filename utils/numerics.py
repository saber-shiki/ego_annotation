from __future__ import annotations

def rigid_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError('rigid fit requires matched arrays with >=3 points')
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    xd = dst - mu_dst
    covariance = xs.T @ xd / len(src)
    (u, _, vt) = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = mu_dst - rotation @ mu_src
    return (rotation.astype(float), translation.astype(float))

def regularize_translation_timeline(measurements: np.ndarray, weights: np.ndarray, *, velocity_weight: float, acceleration_weight: float, zero_prior_weight: float=0.0) -> np.ndarray:
    import numpy as np
    count = len(measurements)
    if count < 3:
        return measurements.copy()
    first_difference = np.zeros((count - 1, count), dtype=float)
    for row in range(count - 1):
        first_difference[row, row] = -1.0
        first_difference[row, row + 1] = 1.0
    second_difference = np.zeros((count - 2, count), dtype=float)
    for row in range(count - 2):
        second_difference[row, row] = 1.0
        second_difference[row, row + 1] = -2.0
        second_difference[row, row + 2] = 1.0
    weight_matrix = np.diag(weights)
    system = weight_matrix + float(zero_prior_weight) * np.eye(count, dtype=float) + float(velocity_weight) * (first_difference.T @ first_difference) + float(acceleration_weight) * (second_difference.T @ second_difference)
    rhs = weight_matrix @ measurements
    return np.column_stack([np.linalg.solve(system, rhs[:, axis]) for axis in range(3)])

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    import numpy as np
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError('Umeyama requires matched arrays with >=3 points')
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    xd = dst - mu_dst
    cov = xd.T @ xs / len(src)
    (u, s, vt) = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vt
    var_src = np.sum(xs * xs) / len(src)
    scale = float(np.sum(s * d) / max(var_src, 1e-12))
    t = mu_dst - scale * (r @ mu_src)
    return (scale, r, t)

def least_norm_halfspace_escape(points: np.ndarray, closest: np.ndarray, signed_depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Least-norm rigid translation satisfying local escape halfspaces.

    For each penetrating vertex p_i with closest surface point q_i, n_i=(q_i-p_i)/||q_i-p_i||
    is the local outward escape direction. A rigid translation t should satisfy
    n_i dot t >= signed_depth_i in the first-order signed-distance model. The
    solution is the projection of zero onto the intersection of those halfspaces.
    """
    from typing import Any
    import numpy as np
    from scipy.optimize import minimize
    from scipy.optimize import LinearConstraint
    disp = np.asarray(closest, dtype=float) - np.asarray(points, dtype=float)
    signed_depth = np.asarray(signed_depth, dtype=float)
    norms = np.linalg.norm(disp, axis=1)
    valid = (norms > 1e-12) & np.isfinite(norms) & np.isfinite(signed_depth) & (signed_depth > 0.0)
    if not np.any(valid):
        return (np.zeros(3, dtype=float), {'solver': 'no_valid_penetrating_halfspaces', 'success': False, 'constraint_count': 0})
    normals = disp[valid] / norms[valid, None]
    depths = signed_depth[valid]
    x0 = np.average(disp[valid], axis=0, weights=np.maximum(depths, 1e-12))
    constraint = LinearConstraint(normals, depths, np.full_like(depths, np.inf))
    res = minimize(lambda x: 0.5 * float(np.dot(x, x)), x0=x0, jac=lambda x: np.asarray(x, dtype=float), constraints=[constraint], method='SLSQP', options={'ftol': 1e-12, 'maxiter': 200, 'disp': False})
    if res.success and np.all(np.isfinite(res.x)):
        slack = normals @ res.x - depths
        return (np.asarray(res.x, dtype=float), {'solver': 'least_norm_local_halfspace_escape', 'success': True, 'constraint_count': int(len(depths)), 'min_constraint_slack_m': float(np.min(slack)) if slack.size else None, 'max_constraint_violation_m': float(max(0.0, -float(np.min(slack)))) if slack.size else 0.0})
    fallback = np.average(disp[valid], axis=0, weights=np.maximum(depths, 1e-12))
    slack = normals @ fallback - depths
    return (np.asarray(fallback, dtype=float), {'solver': 'weighted_mean_escape_fallback', 'success': False, 'constraint_count': int(len(depths)), 'message': str(getattr(res, 'message', 'unknown')), 'min_constraint_slack_m': float(np.min(slack)) if slack.size else None, 'max_constraint_violation_m': float(max(0.0, -float(np.min(slack)))) if slack.size else None})
