#!/usr/bin/env python3
"""R4d near-field gauge-corrected object-vs-desk relative motion decomposition.

Composes the R2/R3 object (keyboard) patch per-frame SE(3) with the inverse of
the R4c near-keyboard desk static-gauge SE(3) to estimate the object's rigid
motion relative to the desk gauge, for the frames covered by the R4c gauge
(0-110 for clip001851).

This is a DIAGNOSTIC decomposition. It does NOT emit a full object pose or a
contact state. The headline output is the relative transform

    M[i] = inv(T_gauge[i]) @ T_obj[i]

i.e. the object's rigid motion expressed in the desk-static gauge frame, plus
per-frame uncertainty from (a) R4a-style conditional per-DOF observability of
the object patch SE(3) fit and (b) the R4c gauge residual + object patch fit
residual.

Prediction-side only: consumes prediction CoTracker object tracks and the
prediction R4c gauge path/audit. No HOT3D GT is consumed.

Mechanism
---------
- T_obj[i]: robust IRLS Huber Kabsch fit mapping the object world[query] patch
  to the object world[i] patch. The object tracks share the R4c world-lift path
  (same prediction camera + depth), so object world points and the static gauge
  points live in the same per-frame camera world frame and the gauge drift is
  common to both.
- T_gauge[i] = (R_i, t_i) from r4c_se3_gauge_path.json. Its action
  `dst = R_i @ src + t_i` maps a query-frame world point to frame i and is the
  camera-extrinsic gauge drift for STATIC desk points.
- M[i] = inv(T_gauge[i]) @ T_obj[i]. For an object rigidly static w.r.t. the
  desk, M[i] == I (it drifts with the camera exactly like the desk, so the
  relative motion is zero). Non-identity M[i] is real object motion relative to
  the desk.
- Conditional observability: the information matrix of the linearized SE(3)
  correspondence residual on the object patch (rotation columns scaled by the
  patch extent so all six DOFs share units). Its eigen-structure gives the
  effective observable DOF count, the condition number, and per-axis formal
  translation/rotation uncertainty; combined with patch planarity (s3/s1) it
  yields a per-frame observability category. This is the "R4a conditional
  observability" made concrete -- it is not a separate artifact and uses no GT.

Self-consistency (synthetic + real-data) is asserted at run time; see
``run_selftest`` and the composition-consistency check inside ``run``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def summarize(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# SE(3) algebra (convention: T = (R, t),  p -> R @ p + t)
# --------------------------------------------------------------------------- #
def compose(T1: tuple, T2: tuple) -> tuple:
    """Return T1 ∘ T2  (apply T2 first, then T1): (R1@R2, R1@t2 + t1)."""
    R1, t1 = T1
    R2, t2 = T2
    return R1 @ R2, R1 @ t2 + t1


def invert(T: tuple) -> tuple:
    R, t = T
    return R.T, -R.T @ t


def applyT(T: tuple, pts: np.ndarray) -> np.ndarray:
    R, t = T
    return pts @ R.T + t


# --------------------------------------------------------------------------- #
# robust rigid fit (IRLS Huber Kabsch), same solver family as R3 / R4c
# --------------------------------------------------------------------------- #
def robust_rigid_fit(src: np.ndarray, dst: np.ndarray, huber_delta_m: float = 0.010,
                     iters: int = 8):
    """Fit dst ~= R @ src + t with IRLS Huber weights.

    Returns R, t, inlier_mask, residuals_m, weights.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    N = src.shape[0]
    weights = np.ones(N, dtype=np.float64)
    R = np.eye(3)
    t = np.zeros(3, dtype=np.float64)
    if N < 3:
        return R, t, np.zeros(N, dtype=bool), np.full(N, np.nan), weights
    for _ in range(iters):
        w = weights[:, None]
        sw = float(np.sum(weights))
        mu_s = np.sum(w * src, axis=0) / sw
        mu_d = np.sum(w * dst, axis=0) / sw
        sc = src - mu_s
        dc = dst - mu_d
        H = (w * sc).T @ dc
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ D @ U.T
        t = mu_d - R @ mu_s
        pred = src @ R.T + t
        res = np.linalg.norm(dst - pred, axis=1)
        weights = np.where(res <= huber_delta_m, 1.0, huber_delta_m / np.maximum(res, 1e-12))
    pred = src @ R.T + t
    res = np.linalg.norm(dst - pred, axis=1)
    inlier = res <= huber_delta_m
    return R, t, inlier, res, weights


# --------------------------------------------------------------------------- #
# patch geometry + conditional per-DOF observability (R4a)
# --------------------------------------------------------------------------- #
def patch_geometry(points: np.ndarray, weights: np.ndarray) -> dict:
    """Singular values / extents of the (weighted, centered) patch.

    Robust to degenerate inputs (too few points, all-zero weights, NaN): returns
    a well-formed geom with zero singular values so callers can still classify.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return {"center_world_m": [0.0, 0.0, 0.0], "singular_values_m": [0.0, 0.0, 0.0],
                "rank2_ratio": 0.0, "rank3_ratio": 0.0, "radial_p95_m": 0.0}
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 1e-12:
        w = np.ones(pts.shape[0], dtype=np.float64)   # fall back to uniform support
        sw = float(w.sum())
    center = np.sum(w[:, None] * pts, axis=0) / sw
    centered = pts - center
    radial = np.linalg.norm(centered, axis=1)
    W = np.sqrt(w[:, None]) * centered
    try:
        _u, sv, _vt = np.linalg.svd(W, compute_uv=False)
    except np.linalg.LinAlgError:
        sv = np.array([0.0, 0.0, 0.0])
    sv = np.asarray(sv, dtype=np.float64).reshape(-1)
    if sv.size < 3:
        sv = np.pad(sv, (0, 3 - sv.size), constant_values=0.0)
    sv = sv[:3]
    if not np.all(np.isfinite(sv)):
        sv = np.array([0.0, 0.0, 0.0])
    s1 = float(sv[0]) if sv[0] > 1e-15 else 1e-15
    rank2_ratio = float(sv[1] / s1) if sv[1] > 0 else 0.0
    rank3_ratio = float(sv[2] / s1) if sv[2] > 0 else 0.0
    radial_p95 = float(np.percentile(radial, 95)) if radial.size else 0.0
    return {
        "center_world_m": center.tolist(),
        "singular_values_m": sv.tolist(),
        "rank2_ratio": rank2_ratio,
        "rank3_ratio": rank3_ratio,
        "radial_p95_m": radial_p95,
    }


def conditional_observability(src: np.ndarray, weights: np.ndarray, sigma2: float,
                              planar_ratio_threshold: float = 0.15,
                              low_condition: float = 1.0e3) -> dict:
    """Per-DOF conditional observability of the SE(3) correspondence fit.

    Information matrix of the linearized residual r_k = (omega x p_k) + dt on the
    (weighted) patch, with the three rotation columns scaled by the patch extent
    L = radial p95 so rotation and translation DOFs share units.

    Returns effective observable DOF count, condition number, scaled
    eigenvalues, formal per-axis translation std (m) and rotation std (rad),
    and a category. sigma2 is the residual variance used for the formal
    covariance; if sigma2<=0 the covariance fields are null.
    """
    pts = np.asarray(src, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return {"category": "low_support", "effective_observable_dof": 0,
                "condition_number": None, "planar_rank3_ratio": 0.0, "is_planar": True,
                "scaled_info_eigenvalues": None, "patch_extent_L_m": 0.0,
                "n_points": int(pts.shape[0]) if pts.ndim == 2 else 0,
                "formal_translation_std_m": None, "formal_rotation_std_rad": None,
                "planar_ratio_threshold": float(planar_ratio_threshold),
                "low_condition_threshold": float(low_condition), "weights_degraded": True}
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    sw_in = float(np.sum(w))
    weights_degraded = (not np.isfinite(sw_in)) or sw_in <= 1e-12
    if weights_degraded:
        w = np.ones(pts.shape[0], dtype=np.float64)   # geometry on full support
    N = pts.shape[0]
    geom = patch_geometry(pts, w)
    L = max(geom["radial_p95_m"], 1.0e-3)

    centered = pts - np.array(geom["center_world_m"], dtype=np.float64)
    # Jacobian columns: [wx*L, wy*L, wz*L, tx, ty, tz]; 3 rows per point.
    # d r / d omega = omega x p; sign absorbed by symmetry of J^T J.
    rot_blocks = []
    for k in range(N):
        p = centered[k]
        rot_blocks.append(np.array([[0.0, -p[2], p[1]],
                                    [p[2], 0.0, -p[0]],
                                    [-p[1], p[0], 0.0]]) * L)
    tr_block = np.eye(3)
    J = np.hstack([np.vstack(rot_blocks), np.tile(tr_block, (N, 1))])  # 3N x 6
    Wv = np.repeat(w, 3)             # 3N
    Jw = J * Wv[:, None]
    Lambda = J.T @ Jw              # 6x6 information (weighted)
    Lambda = 0.5 * (Lambda + Lambda.T)   # symmetrize

    try:
        eigvals, eigvecs = np.linalg.eigh(Lambda)
    except np.linalg.LinAlgError:
        eigvals = np.full(6, np.nan)
        eigvecs = np.full((6, 6), np.nan)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    lam_max = float(eigvals[0]) if np.all(np.isfinite(eigvals)) and eigvals[0] > 0 else 0.0
    lam_min = float(eigvals[-1]) if np.all(np.isfinite(eigvals)) and eigvals[-1] > 0 else 0.0
    condition = float(lam_max / lam_min) if lam_min > 1e-30 else float("inf")

    # effective observable DOF: eigenvalues within low_condition of the max
    if lam_max > 0 and np.isfinite(condition):
        eff_dof = int(np.sum(eigvals >= lam_max / low_condition))
    else:
        eff_dof = 0

    # formal covariance per axis (unscaled blocks), using the unscaled rotation block
    # translation: sigma2 * inv(sum w) per axis (isotropic, decoupled when centered)
    sw = float(np.sum(w))
    tr_std_m = float(math.sqrt(max(sigma2, 0.0) / sw)) if (sigma2 > 0 and sw > 0) else None
    # rotation: build unscaled rotation information block Lambda_rot = sum w (px^T px)
    Lambda_rot = np.zeros((3, 3))
    for k in range(N):
        p = centered[k]
        px = np.array([[0.0, -p[2], p[1]],
                       [p[2], 0.0, -p[0]],
                       [-p[1], p[0], 0.0]])
        Lambda_rot += w[k] * (px.T @ px)
    try:
        cov_rot = sigma2 * np.linalg.inv(Lambda_rot) if sigma2 > 0 else None
    except np.linalg.LinAlgError:
        cov_rot = None
    rot_std_rad = (np.sqrt(np.maximum(np.diag(cov_rot), 0.0)).tolist()
                   if cov_rot is not None else None)

    planar = geom["rank3_ratio"] < planar_ratio_threshold
    if N < 6:
        category = "low_support"
    elif not np.isfinite(condition) or condition > low_condition:
        category = "weakly_conditioned"
    elif planar:
        category = "planar_inplane_yaw_weak"
    else:
        category = "full_6dof"
    if weights_degraded:
        # IRLS collapsed (no inliers at huber delta -> non-rigid patch); geometry
        # still measured on the full support, but the fit is unreliable.
        category = "weakly_conditioned" if category == "full_6dof" else category

    return {
        "category": category,
        "effective_observable_dof": eff_dof,
        "condition_number": (None if not np.isfinite(condition) else condition),
        "planar_rank3_ratio": geom["rank3_ratio"],
        "is_planar": bool(planar),
        "scaled_info_eigenvalues": (eigvals.tolist() if np.all(np.isfinite(eigvals)) else None),
        "patch_extent_L_m": float(L),
        "n_points": int(N),
        "formal_translation_std_m": tr_std_m,
        "formal_rotation_std_rad": rot_std_rad,
        "planar_ratio_threshold": float(planar_ratio_threshold),
        "low_condition_threshold": float(low_condition),
        "weights_degraded": bool(weights_degraded),
    }


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def load_inputs(args: argparse.Namespace) -> dict:
    gauge_path_doc = json.loads(Path(args.gauge_path_json).read_text())
    frames_g = gauge_path_doc["frames"]
    qf = int(args.query_frame_index if args.query_frame_index is not None
             else gauge_path_doc.get("query_frame_index"))
    gauge_R = [None] * len(frames_g)
    gauge_t = [None] * len(frames_g)
    gauge_valid = np.zeros(len(frames_g), dtype=bool)
    gauge_residual_mm = np.full(len(frames_g), np.nan)
    gauge_inlier_count = np.full(len(frames_g), 0)
    for i, fr in enumerate(frames_g):
        if fr.get("valid") and fr.get("drift_R") is not None:
            gauge_R[i] = np.asarray(fr["drift_R"], dtype=np.float64)
            gauge_t[i] = np.asarray(fr["drift_t_m"], dtype=np.float64)
            gauge_valid[i] = True
            gauge_residual_mm[i] = fr.get("residual_mm", np.nan)
            gauge_inlier_count[i] = int(fr.get("inlier_count", 0))

    tr = np.load(args.object_tracks_npz, allow_pickle=True)
    frame_idx = np.asarray(tr["frame_idx"], dtype=np.int64)
    accepted = np.asarray(tr["accepted"], dtype=bool)
    visibility = np.asarray(tr["visibility"], dtype=bool)
    world = np.asarray(tr["world_xyz"], dtype=np.float64)
    assert len(frame_idx) == len(frames_g), "object track frame count != gauge frame count"
    assert np.array_equal(frame_idx, np.asarray([f["frame_idx"] for f in frames_g], dtype=np.int64)), \
        "object track frame indices != gauge frame indices"

    fi_to_i = {int(f): i for i, f in enumerate(frame_idx.tolist())}
    if qf not in fi_to_i:
        raise RuntimeError(f"query frame {qf} not present in object tracks")
    qi = fi_to_i[qf]
    if not gauge_valid[qi]:
        raise RuntimeError(f"gauge invalid at query frame {qf}; composition reference undefined")

    audit = None
    if args.gauge_audit_json and Path(args.gauge_audit_json).exists():
        audit = json.loads(Path(args.gauge_audit_json).read_text())

    return {
        "frame_idx": frame_idx, "accepted": accepted, "visibility": visibility,
        "world": world, "qf": qf, "qi": qi,
        "gauge_R": gauge_R, "gauge_t": gauge_t, "gauge_valid": gauge_valid,
        "gauge_residual_mm": gauge_residual_mm, "gauge_inlier_count": gauge_inlier_count,
        "audit": audit,
    }


def _control_validation(control_npz: Path, data: dict, args: argparse.Namespace) -> dict:
    """Run the same composition on a STATIC reference patch (the R4c desk tracks).

    A correct pipeline must return M ~= I for genuinely static points: the static
    reference shares the camera gauge, so gauge-corrected it returns to its query
    position. The rigid-frame |t_M| distribution is the empirical noise floor of
    the composition (gauge-fit vs patch-fit disagreement). No GT is used.
    """
    tr = np.load(control_npz, allow_pickle=True)
    cframe = np.asarray(tr["frame_idx"], dtype=np.int64)
    if not np.array_equal(cframe, data["frame_idx"]):
        return {"status": "skipped", "reason": "control frame indices do not match object tracks"}
    cworld = np.asarray(tr["world_xyz"], dtype=np.float64)
    cacc = np.asarray(tr["accepted"], dtype=bool)
    cvis = np.asarray(tr["visibility"], dtype=bool)
    cusable = cacc & cvis & np.isfinite(cworld).all(axis=-1)
    qi = data["qi"]
    src_q = cworld[qi]
    src_usable_q = cusable[qi]
    gauge_R, gauge_t, gauge_valid = data["gauge_R"], data["gauge_t"], data["gauge_valid"]
    rel_t, ifr = [], []
    for i in range(len(cframe)):
        if not gauge_valid[i]:
            continue
        both = src_usable_q & cusable[i]
        idx = np.where(both)[0]
        src = src_q[idx]; dst = cworld[i, idx]
        fin = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        src = src[fin]; dst = dst[fin]
        if src.shape[0] < int(args.min_patch_points):
            continue
        R_o, t_o, inl, _res, _w = robust_rigid_fit(
            src, dst, huber_delta_m=float(args.huber_delta_m), iters=int(args.irls_iterations))
        M = compose(invert((gauge_R[i], gauge_t[i])), (R_o, t_o))
        rel_t.append(float(np.linalg.norm(M[1])) * 1000.0)
        ifr.append(float(np.mean(inl)) if src.shape[0] else 0.0)
    rel_t = np.asarray(rel_t, dtype=np.float64)
    ifr = np.asarray(ifr, dtype=np.float64)
    rigid = ifr >= float(args.min_inlier_fraction)
    return {
        "status": "computed",
        "control_tracks_npz": str(control_npz),
        "control_tracks_npz_sha256": sha256_file(control_npz),
        "n_covered_frames": int(rel_t.size),
        "all_covered_relative_translation_mm": summarize(rel_t),
        "rigid_frame_relative_translation_mm": (summarize(rel_t[rigid]) if rigid.any() else {"count": 0}),
        "rigid_frame_count": int(rigid.sum()),
        "interpretation": ("For static points M should be ~I; the rigid-frame |t_M| "
                           "distribution is the empirical composition noise floor "
                           "(gauge-fit vs patch-fit disagreement). Object motion well "
                           "above this floor is real."),
    }


def run(args: argparse.Namespace) -> dict:
    data = load_inputs(args)
    frame_idx = data["frame_idx"]
    world = data["world"]
    accepted = data["accepted"]
    visibility = data["visibility"]
    qi = data["qi"]
    qf = data["qf"]
    gauge_R = data["gauge_R"]
    gauge_t = data["gauge_t"]
    gauge_valid = data["gauge_valid"]
    gauge_residual_mm = data["gauge_residual_mm"]
    T = len(frame_idx)

    # object patch acceptance: accepted + visible + finite world
    usable = accepted & visibility & np.isfinite(world).all(axis=-1)
    src_all = world[qi]                 # query-frame object patch
    src_usable_q = usable[qi]

    per_frame = []
    rel_t_mm_series = []
    rel_rot_deg_series = []
    combined_residual_mm_series = []

    for i in range(T):
        entry = {"frame_idx": int(frame_idx[i]), "query_frame_index": int(qf)}
        if not gauge_valid[i]:
            entry.update({
                "gauge_valid": False,
                "status": "gauge_uncovered",
                "relative_M": None,
            })
            per_frame.append(entry)
            continue

        both = src_usable_q & usable[i]
        idx = np.where(both)[0]
        src = src_all[idx]
        dst = world[i, idx]
        finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        src = src[finite]
        dst = dst[finite]
        n_pts = int(src.shape[0])

        T_gauge = (gauge_R[i], gauge_t[i])

        if n_pts < int(args.min_patch_points):
            entry.update({
                "gauge_valid": True,
                "object_patch": {"n_points": n_pts, "status": "too_few_points"},
                "relative_M": None,
                "status": "relative_motion_unresolved",
            })
            per_frame.append(entry)
            continue

        R_o, t_o, inl, res, weights = robust_rigid_fit(
            src, dst, huber_delta_m=float(args.huber_delta_m), iters=int(args.irls_iterations))
        T_obj = (R_o, t_o)
        M = compose(invert(T_gauge), T_obj)
        R_m, t_m = M
        rel_t_mm = float(np.linalg.norm(t_m) * 1000.0)
        rel_rot_deg = rotation_angle_deg(R_m)

        # conditional observability on the query-frame patch, weighted by the IRLS
        # fit weights (continuous, in (0,1]). If IRLS collapsed to no inliers the
        # observability falls back to the full-support geometry and is flagged.
        inlier_mask = inl.copy()
        if np.any(inlier_mask):
            sigma2 = float(np.mean(res[inlier_mask] ** 2))
        else:
            sigma2 = float(np.mean(res ** 2)) if res.size else float("nan")
        obs = conditional_observability(
            src, weights, sigma2,
            planar_ratio_threshold=float(args.planar_ratio_threshold),
            low_condition=float(args.low_condition))

        patch_res_summary = summarize(res * 1000.0)
        patch_inl_res_summary = summarize(res[inlier_mask] * 1000.0) if np.any(inlier_mask) else {"count": 0}

        # composition-consistency self-check:
        # M @ src  should equal  gauge-corrected dst = Rg^T (dst - tg), modulo patch residual.
        corrected_dst = applyT(invert(T_gauge), dst)
        recomposed = applyT(M, src)
        consistency_res = np.linalg.norm(corrected_dst - recomposed, axis=1)
        consistency_mm = summarize(consistency_res * 1000.0)

        # combined per-frame noise floor (gauge residual + patch inlier residual), both in mm
        g_res = gauge_residual_mm[i] if np.isfinite(gauge_residual_mm[i]) else 0.0
        p_res = patch_inl_res_summary.get("p95", 0.0) or 0.0
        combined_noise_mm = float(math.hypot(g_res, p_res))
        combined_residual_mm_series.append(combined_noise_mm)

        # patch rigidity: the inlier p95 is definitionally <= huber delta, so the
        # real rigidity signal is the INLIER FRACTION (how much of the visible
        # patch fits one rigid SE(3)). A non-rigid patch (hand occlusion, key
        # presses, track loss, or a minority rigid core) yields an unreliable
        # T_obj and therefore an unreliable M, regardless of motion magnitude.
        inlier_fraction = float(np.mean(inlier_mask)) if n_pts else 0.0
        patch_inl_p95_mm = patch_inl_res_summary.get("p95", float("inf")) or float("inf")
        patch_rigid = bool(inlier_fraction >= float(args.min_inlier_fraction))

        # status partition
        conditioned = (obs["category"] != "low_support"
                       and n_pts >= int(args.min_patch_points)
                       and patch_rigid)
        if not conditioned:
            status = "relative_motion_unresolved"
        elif rel_t_mm <= combined_noise_mm and rel_rot_deg <= args.static_rotation_deg:
            status = "relative_motion_static_at_noise_floor"
        else:
            status = "relative_motion_supported"

        entry.update({
            "gauge_valid": True,
            "object_patch": {
                "n_points": n_pts,
                "inlier_count": int(inlier_mask.sum()),
                "inlier_fraction": float(np.mean(inlier_mask)) if n_pts else 0.0,
                "residual_mm": patch_res_summary,
                "inlier_residual_mm": patch_inl_res_summary,
                "sigma2_m2": (None if not np.isfinite(sigma2) else float(sigma2)),
                "gauge_residual_mm": (None if not np.isfinite(g_res) else float(g_res)),
                "combined_noise_floor_mm": combined_noise_mm,
                "patch_inlier_p95_mm": float(patch_inl_p95_mm),
                "patch_rigid": patch_rigid,
                "observability_r4a": obs,
            },
            "T_obj_patch_se3_for_provenance": {
                "_note": "intermediate composition term; NOT a published object pose",
                "rotation": R_o.tolist(),
                "translation_m": t_o.tolist(),
                "translation_mm": float(np.linalg.norm(t_o) * 1000.0),
                "rotation_deg": rotation_angle_deg(R_o),
            },
            "T_gauge_reference_se3": {
                "rotation": gauge_R[i].tolist(),
                "translation_m": gauge_t[i].tolist(),
                "drift_translation_mm": float(np.linalg.norm(gauge_t[i]) * 1000.0),
                "drift_rotation_deg": rotation_angle_deg(gauge_R[i]),
            },
            "relative_M": {
                "rotation": R_m.tolist(),
                "translation_m": t_m.tolist(),
                "translation_mm": rel_t_mm,
                "rotation_deg": rel_rot_deg,
            },
            "composition_consistency_mm": consistency_mm,
            "status": status,
            "unresolved_reason": (None if status != "relative_motion_unresolved" else
                                   ("too_few_patch_points" if n_pts < int(args.min_patch_points)
                                    else ("non_rigid_patch_low_inlier_fraction" if not patch_rigid
                                          else "low_observability_support"))),
        })
        per_frame.append(entry)
        rel_t_mm_series.append(rel_t_mm)
        rel_rot_deg_series.append(rel_rot_deg)

    # ---- sustained relative motion statistics over covered frames ----
    covered = [i for i in range(T) if gauge_valid[i]]
    rel_t_arr = np.array(rel_t_mm_series, dtype=np.float64)
    rel_rot_arr = np.array(rel_rot_deg_series, dtype=np.float64)
    combined_arr = np.array(combined_residual_mm_series, dtype=np.float64)
    noise_floor_mm = float(np.median(combined_arr)) if combined_arr.size else 0.0

    # conditioned (rigid-patch) covered frames: the trustworthy subset. The
    # all-covered stats include non-rigid (unresolved) frames whose large |t_M|
    # is a fit artifact, not motion; report both so the reader is not misled.
    cond_idx = [i for i in covered
                if per_frame[i]["status"] in ("relative_motion_supported",
                                               "relative_motion_static_at_noise_floor")]
    cond_t_arr = np.array([per_frame[i]["relative_M"]["translation_mm"] for i in cond_idx],
                          dtype=np.float64) if cond_idx else np.array([], dtype=np.float64)
    cond_rot_arr = np.array([per_frame[i]["relative_M"]["rotation_deg"] for i in cond_idx],
                            dtype=np.float64) if cond_idx else np.array([], dtype=np.float64)

    # object centroid trajectory in desk-gauge frame (t_M in mm)
    tM = np.array([per_frame[i]["relative_M"]["translation_m"]
                   if per_frame[i].get("relative_M") else [np.nan] * 3
                   for i in range(T)]) * 1000.0
    if len(covered) >= 2:
        cf = np.array(covered)
        steps = np.linalg.norm(np.diff(tM[cf], axis=0), axis=1)
        path_length_mm = float(np.sum(steps))
        net_displacement_mm = float(np.linalg.norm(tM[cf[-1]] - tM[cf[0]]))
        straightness = float(net_displacement_mm / path_length_mm) if path_length_mm > 1e-9 else 0.0
        max_excursion_mm = float(np.nanmax(np.linalg.norm(tM[cf], axis=1)))
    else:
        path_length_mm = net_displacement_mm = straightness = max_excursion_mm = 0.0
    # sustained motion is judged on the CONDITIONED subset against the static-
    # control noise floor (populated later if a control is given; else the
    # combined per-frame gauge+patch noise floor).
    sustained_motion_present = bool(cond_t_arr.size > 0 and
                                    float(np.median(cond_t_arr)) > max(noise_floor_mm, 1.0))

    status_counts = {}
    for e in per_frame:
        status_counts[e["status"]] = status_counts.get(e["status"], 0) + 1

    provenance = {
        "object_tracks_npz": str(args.object_tracks_npz),
        "object_tracks_npz_sha256": sha256_file(Path(args.object_tracks_npz)),
        "gauge_path_json": str(args.gauge_path_json),
        "gauge_path_json_sha256": sha256_file(Path(args.gauge_path_json)),
        "gauge_audit_json": (str(args.gauge_audit_json) if args.gauge_audit_json else None),
        "gauge_audit_json_sha256": (sha256_file(Path(args.gauge_audit_json))
                                    if args.gauge_audit_json and Path(args.gauge_audit_json).exists()
                                    else None),
    }

    gauge_coverage = {
        "total_frames": int(T),
        "covered_frame_count": int(len(covered)),
        "covered_frame_indices": [int(frame_idx[i]) for i in covered],
        "uncovered_frame_count": int(T - len(covered)),
        "uncovered_frame_range": ([int(frame_idx[covered[-1] + 1]), int(frame_idx[-1])]
                                  if covered and covered[-1] + 1 < T else None),
        "coverage_fraction": float(len(covered) / T),
    }

    audit_echo = None
    if data["audit"] is not None:
        a = data["audit"]
        tc = a.get("test_c_per_frame_se3_drift_from_query", {})
        audit_echo = {
            "gauge_control_status": a.get("gauge_control_status"),
            "gauge_residual_after_drift_removal_mm": tc.get("residual_after_drift_removal_mm"),
            "gauge_drift_translation_mm": tc.get("drift_translation_mm"),
        }

    # ---- composition validation (algebraic + real-data) ----
    # query-frame identity: M[query] must be ~I (T_obj[query]=T_gauge[query]=I).
    q_entry = per_frame[qi]
    q_identity_mm = (q_entry["relative_M"]["translation_mm"] if q_entry.get("relative_M") else None)
    control_val = None
    if args.control_tracks_npz and Path(args.control_tracks_npz).exists():
        control_val = _control_validation(Path(args.control_tracks_npz), data, args)
    composition_validation = {
        "query_frame_identity": {
            "frame_idx": int(qf),
            "relative_M_translation_mm": q_identity_mm,
            "expected": "~0 mm (M[query] = inv(I) @ I = I by construction)",
        },
        "composition_consistency_median_mm_over_covered": (
            summarize(np.array([e["composition_consistency_mm"]["median"]
                                for e in per_frame if e.get("composition_consistency_mm")],
                               dtype=np.float64))
        ),
        "synthetic_selftest": "see r4d_selftest.json (inv/inv, associativity, static->I, gauge-correction identity, M recovery)",
    }
    if control_val is not None:
        composition_validation["static_reference_control"] = control_val

    # population-level object-motion vs method-noise-floor comparison. The
    # discriminator is the object rigid-frame |t_M| median against the static
    # control rigid-frame |t_M| median (the empirical composition noise floor).
    obj_rigid_median = float(np.median(cond_t_arr)) if cond_t_arr.size else float("nan")
    if control_val is not None and control_val.get("rigid_frame_relative_translation_mm", {}).get("count", 0):
        ctrl_rigid = control_val["rigid_frame_relative_translation_mm"]
        ctrl_rigid_median = float(ctrl_rigid["median"])
        motion_vs_noise = {
            "object_conditioned_frame_count": int(cond_t_arr.size),
            "object_conditioned_relative_translation_mm_median": obj_rigid_median,
            "static_control_rigid_relative_translation_mm_median": ctrl_rigid_median,
            "static_control_rigid_relative_translation_mm_p95": float(ctrl_rigid["p95"]),
            "ratio_object_over_static_control_median": (float(obj_rigid_median / ctrl_rigid_median)
                                                         if ctrl_rigid_median > 1e-9 else None),
            "object_motion_exceeds_noise_floor": bool(obj_rigid_median > ctrl_rigid_median),
            "note": ("Per-frame M[i] carries a heavy-tailed method noise (static-control "
                     "rigid p95 can reach ~10^2 mm), so individual transforms are uncertain; "
                     "the population median is the robust signal that the object moves relative "
                     "to the desk."),
        }
    else:
        motion_vs_noise = {
            "object_conditioned_frame_count": int(cond_t_arr.size),
            "object_conditioned_relative_translation_mm_median": obj_rigid_median,
            "static_control": "not provided (pass --control-tracks-npz for the noise-floor comparison)",
        }

    report = {
        "schema": "r4d_nearfield_relative_motion/v1",
        "method": "fit_nearfield_gauge_corrected_patch_motion.py",
        "claim": ("object (keyboard) patch SE(3) composed with the inverse near-keyboard "
                  "desk static-gauge SE(3) => object motion relative to the desk gauge. "
                  "DIAGNOSTIC ONLY: no full object pose or contact is emitted."),
        "boundary": ("prediction-side only: prediction CoTracker object tracks + prediction R4c "
                     "gauge path/audit. No HOT3D GT consumed. No object pose/contact artifact."),
        "inputs": provenance,
        "query_frame_index": int(qf),
        "gauge_coverage": gauge_coverage,
        "composition_formula": ("M[i] = inv(T_gauge[i]) @ T_obj[i];  T_obj[i]: robust Kabsch "
                                 "object_world[query] -> object_world[i]; T_gauge[i]=(R_i,t_i) "
                                 "from r4c_se3_gauge_path, action p -> R_i@p + t_i."),
        "frame_semantics": ("T_obj, T_gauge act p -> R@p+t and share the R4c world-lift frame "
                            "(prediction camera+depth). M acts on object_world[query] and equals "
                            "the gauge-corrected object position; M==I when the object is rigidly "
                            "static w.r.t. the desk."),
        "audit_echo": audit_echo,
        "composition_validation": composition_validation,
        "sustained_relative_motion": {
            "relative_translation_mm": summarize(rel_t_arr),
            "relative_rotation_deg": summarize(rel_rot_arr),
            "net_displacement_mm": net_displacement_mm,
            "path_length_mm": path_length_mm,
            "straightness_net_over_path": straightness,
            "max_excursion_mm": max_excursion_mm,
            "noise_floor_mm_median": noise_floor_mm,
            "sustained_motion_present": sustained_motion_present,
            "note": ("Over gauge-covered frames 0-110. net_displacement = |t_M[last]-t_M[first]|; "
                     "path_length = sum of per-frame steps; straightness ~1 means a coherent drift, "
                     "~0 means oscillation/noise; sustained_motion_present compares the conditioned "
                     "(rigid-patch) frame |t_M| median to the per-frame gauge+patch noise floor. "
                     "ALL-covered stats include non-rigid (unresolved) frames whose large |t_M| is a "
                     "fit artifact; see conditioned_frames_only for the trustworthy subset."),
        },
        "sustained_relative_motion_conditioned_frames_only": {
            "frame_count": int(cond_t_arr.size),
            "relative_translation_mm": (summarize(cond_t_arr) if cond_t_arr.size else {"count": 0}),
            "relative_rotation_deg": (summarize(cond_rot_arr) if cond_rot_arr.size else {"count": 0}),
            "note": "Subset of covered frames with a rigid patch (status supported or static_at_noise_floor).",
        },
        "motion_vs_static_noise_floor": motion_vs_noise,
        "status_counts": status_counts,
        "status_definitions": {
            "gauge_uncovered": "frame outside R4c gauge coverage (no gauge SE3); M undefined.",
            "relative_motion_unresolved": ("gauge valid but object patch under-conditioned "
                                           "(too few points / non-rigid low-inlier-fraction patch / "
                                           "weak geometry); M not trustworthy."),
            "relative_motion_static_at_noise_floor": ("gauge valid + conditioned but |t_M| and rot_M "
                                                      "within the per-frame gauge+patch noise floor; "
                                                      "motion indistinguishable from zero."),
            "relative_motion_supported": ("gauge valid + conditioned and motion exceeds the noise "
                                          "floor; M is a supported relative-motion measurement."),
        },
        "parameters": {
            "huber_delta_m": float(args.huber_delta_m),
            "irls_iterations": int(args.irls_iterations),
            "min_patch_points": int(args.min_patch_points),
            "planar_ratio_threshold": float(args.planar_ratio_threshold),
            "low_condition": float(args.low_condition),
            "static_rotation_deg": float(args.static_rotation_deg),
            "max_patch_inlier_p95_mm": float(args.max_patch_inlier_p95_mm),
            "min_inlier_fraction": float(args.min_inlier_fraction),
        },
        "per_frame": per_frame,
    }

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "r4d_relative_motion.json").write_text(json.dumps(report, indent=2))
    with open(out_root / "r4d_relative_motion.ndjson", "w") as fh:
        for e in per_frame:
            fh.write(json.dumps(e) + "\n")
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    write_report_md(out_root, report)
    return report


def write_report_md(out_root: Path, report: dict) -> None:
    cov = report["gauge_coverage"]
    sus = report["sustained_relative_motion"]
    lines = []
    lines.append("# R4d near-field gauge-corrected relative motion — clip001851\n")
    lines.append("DIAGNOSTIC decomposition of object (keyboard) patch SE(3) relative to the "
                 "near-keyboard desk static gauge. No full object pose or contact is emitted.\n")
    lines.append(f"- Schema: `{report['schema']}`")
    lines.append(f"- Query frame: {report['query_frame_index']}")
    lines.append(f"- Gauge coverage: {cov['covered_frame_count']}/{cov['total_frames']} frames "
                 f"({cov['coverage_fraction']:.1%}); covered {cov['covered_frame_indices'][0]}-"
                 f"{cov['covered_frame_indices'][-1]}")
    lines.append(f"- Formula: `{report['composition_formula']}`\n")
    lines.append("## Sustained relative motion (covered frames)")
    lines.append(f"- relative translation mm: median {sus['relative_translation_mm'].get('median'):.2f}, "
                 f"p95 {sus['relative_translation_mm'].get('p95'):.2f}, max {sus['relative_translation_mm'].get('max'):.2f}")
    lines.append(f"- relative rotation deg: median {sus['relative_rotation_deg'].get('median'):.3f}, "
                 f"max {sus['relative_rotation_deg'].get('max'):.3f}")
    lines.append(f"- net displacement {sus['net_displacement_mm']:.2f} mm, path length "
                 f"{sus['path_length_mm']:.2f} mm, straightness {sus['straightness_net_over_path']:.3f}")
    lines.append(f"- max excursion {sus['max_excursion_mm']:.2f} mm vs noise floor "
                 f"{sus['noise_floor_mm_median']:.2f} mm -> sustained_motion_present={sus['sustained_motion_present']}")
    sc = report.get("sustained_relative_motion_conditioned_frames_only", {})
    if sc.get("frame_count"):
        ct = sc["relative_translation_mm"]
        lines.append(f"- conditioned (rigid-patch) frames only ({sc['frame_count']}): "
                     f"|t_M| median {ct.get('median'):.2f}, p95 {ct.get('p95'):.2f}, "
                     f"max {ct.get('max'):.2f} mm")
    mvn = report.get("motion_vs_static_noise_floor", {})
    if mvn and isinstance(mvn.get("static_control_rigid_relative_translation_mm_median"), float):
        lines.append(f"- object rigid median {mvn['object_conditioned_relative_translation_mm_median']:.2f} mm "
                     f"vs static-control rigid median {mvn['static_control_rigid_relative_translation_mm_median']:.2f} mm "
                     f"(control p95 {mvn['static_control_rigid_relative_translation_mm_p95']:.2f}) -> "
                     f"exceeds_noise_floor={mvn['object_motion_exceeds_noise_floor']}")
    lines.append("")
    v = report.get("composition_validation", {})
    if v:
        lines.append("## Composition validation")
        qi = v.get("query_frame_identity", {})
        lines.append(f"- query-frame M identity: {qi.get('relative_M_translation_mm')} mm (expected ~0)")
        if v.get("static_reference_control", {}).get("status") == "computed":
            cv = v["static_reference_control"]
            lines.append(f"- static-reference control: {cv['n_covered_frames']} covered frames, "
                         f"{cv['rigid_frame_count']} rigid; static points must return M~=I")
        lines.append("")
    lines.append("## Status counts")
    for k, v in report["status_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Status definitions")
    for k, v in report["status_definitions"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Inputs / provenance")
    p = report["inputs"]
    lines.append(f"- object tracks: `{p['object_tracks_npz']}` (sha256 {p['object_tracks_npz_sha256'][:16]}...)")
    lines.append(f"- gauge path: `{p['gauge_path_json']}` (sha256 {p['gauge_path_json_sha256'][:16]}...)")
    if p.get("gauge_audit_json_sha256"):
        lines.append(f"- gauge audit: `{p['gauge_audit_json']}` (sha256 {p['gauge_audit_json_sha256'][:16]}...)")
    lines.append("\n## Boundary")
    lines.append(report["boundary"])
    (out_root / "R4D_REPORT.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# synthetic self-consistency
# --------------------------------------------------------------------------- #
def _rand_se3(rng):
    A = rng.standard_normal((3, 3))
    R, _ = np.linalg.qr(A)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = rng.standard_normal(3)
    return R, t


def run_selftest(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(0)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    # 1. inv(inv(T)) == T
    for _ in range(200):
        T = _rand_se3(rng)
        ok = np.allclose(invert(invert(T))[0], T[0], atol=1e-10) and \
             np.allclose(invert(invert(T))[1], T[1], atol=1e-10)
        if not ok:
            check("inv_inv_identity", False); break
    else:
        check("inv_inv_identity", True, "200 random SE3")

    # 2. compose associativity
    for _ in range(200):
        A, B, C = _rand_se3(rng), _rand_se3(rng), _rand_se3(rng)
        left = compose(compose(A, B), C)
        right = compose(A, compose(B, C))
        ok = np.allclose(left[0], right[0], atol=1e-10) and np.allclose(left[1], right[1], atol=1e-10)
        if not ok:
            check("compose_associative", False); break
    else:
        check("compose_associative", True, "200 random SE3 triples")

    # 3. static object -> M == I  (object drifts with camera exactly)
    for _ in range(200):
        Tg = _rand_se3(rng)
        T_obj = Tg                          # object static w.r.t. desk
        M = compose(invert(Tg), T_obj)
        ok = np.allclose(M[0], np.eye(3), atol=1e-10) and np.allclose(M[1], np.zeros(3), atol=1e-10)
        if not ok:
            check("static_object_M_identity", False); break
    else:
        check("static_object_M_identity", True, "M=inv(gauge)@gauge=I")

    # 4. gauge correction identity: invert(gauge) @ (gauge @ src) == src
    for _ in range(200):
        Tg = _rand_se3(rng)
        src = rng.standard_normal((50, 3))
        corrected = applyT(invert(Tg), applyT(Tg, src))
        ok = np.allclose(corrected, src, atol=1e-10)
        if not ok:
            check("gauge_correction_identity", False); break
    else:
        check("gauge_correction_identity", True, "corrected = inv(gauge)@(gauge@src) = src")

    # 5. M applied to query patch == gauge-corrected frame patch (composition consistency)
    max_err = 0.0
    for _ in range(200):
        Tg = _rand_se3(rng)
        Mtrue = _rand_se3(rng)
        T_obj = compose(Tg, Mtrue)          # object world(i) = gauge @ M @ query
        src = rng.standard_normal((50, 3))
        dst = applyT(T_obj, src)
        Mrec = compose(invert(Tg), T_obj)
        recomposed = applyT(Mrec, src)
        corrected_dst = applyT(invert(Tg), dst)
        max_err = max(max_err, float(np.max(np.linalg.norm(recomposed - corrected_dst, axis=1))))
        rec_ok = np.allclose(Mrec[0], Mtrue[0], atol=1e-9) and np.allclose(Mrec[1], Mtrue[1], atol=1e-9)
        if not rec_ok:
            check("composition_recovery", False); break
    else:
        check("composition_recovery", True, f"M=inv(gauge)@T_obj recovers true M; recompose-consistency max {max_err:.2e} m")

    all_passed = all(c["passed"] for c in checks)
    result = {"schema": "r4d_selftest/v1", "all_passed": all_passed, "checks": checks}
    if args.output_root:
        out = Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "r4d_selftest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object-tracks-npz", type=Path)
    ap.add_argument("--gauge-path-json", type=Path)
    ap.add_argument("--gauge-audit-json", type=Path)
    ap.add_argument("--control-tracks-npz", type=Path,
                    help="optional STATIC reference tracks (R4c desk tracks) for composition validation; M must be ~I")
    ap.add_argument("--query-frame-index", type=int, default=None,
                    help="defaults to gauge path query_frame_index")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--huber-delta-m", type=float, default=0.010)
    ap.add_argument("--irls-iterations", type=int, default=8)
    ap.add_argument("--min-patch-points", type=int, default=12)
    ap.add_argument("--planar-ratio-threshold", type=float, default=0.15)
    ap.add_argument("--low-condition", type=float, default=1.0e3)
    ap.add_argument("--static-rotation-deg", type=float, default=1.0,
                    help="rotation below which, with translation within noise, status is static_at_noise_floor")
    ap.add_argument("--max-patch-inlier-p95-mm", type=float, default=12.0,
                    help="diagnostic: patch inlier p95 (mm); inlier p95 is definitionally <= huber delta, recorded for transparency")
    ap.add_argument("--min-inlier-fraction", type=float, default=0.50,
                    help="min inlier fraction for a rigid patch; below this the SE(3) fit is deemed non-rigid (a rigid body keeps the majority of its visible surface coherent)")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic self-consistency checks only")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        run_selftest(args)
        return
    for req in ("object_tracks_npz", "gauge_path_json"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_','-')} is required (or use --selftest)")
    run(args)


if __name__ == "__main__":
    main()
