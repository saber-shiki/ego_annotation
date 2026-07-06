#!/usr/bin/env python3
"""CPU-only diagnostic: visible-surface patch SE(3) per rigid epoch.

This is a *diagnostic only*. It consumes already-corrected R2/R3 outputs
(CoTracker object tracks v5, sparse correspondence edges v5, pairwise rigid
factors v6) and asks a single, narrow question:

    Given the frame pairs that the v6 factor stage already declared rigid-ready,
    how well does a single anchor-relative 6DOF Procrustes (Kabsch) SE(3) explain
    the *held-out* visible-surface patch tracks, and how observable is that fit
    from the active cloud geometry?

It produces NO object pose, NO contact / nonpenetration / occlusion / mesh
output, and NO annotation-ready rows. The fitted transform is an
anchor-relative visible-surface-patch transform, explicitly NOT a full object
pose, explicitly NOT separable from camera/depth frame drift, and implies NO
static/moving/object-motion verdict. See ``measurement_type``, ``claim_tested``,
``gauge_caveat`` in the emitted JSON.

Input contracts
---------------
--cotracker-npz         cotracker_object_tracks_v5.npz (must be the scale-corrected
                        ``_scaledepth`` archive, NOT the raw unscaled lift).
                        Required keys: frame_idx (T,), tracks_xy (T,N,2),
                              accepted (T,N) bool, world_xyz (T,N,3) float,
                              tracks_depth_xy (T,N,2) float
                        ``tracks_depth_xy`` proves the archive is the
                        depth-scale-corrected product; pass
                        ``--allow-unscaled-input`` ONLY for synthetic fixtures
                        that legitimately lack it.
--sparse-edges-json     sparse_correspondence_edges_v5.json
                        supplies the usable track-id pool and per-frame counts.
--pair-factors-json     pairwise_rigid_factors_v6.json
                        supplies the ordered ``frames`` list and per adjacent
                        pair ``rigid_factor_ready`` used to segment epochs.

By default the three inputs must resolve under the SAME directory (they are
sibling products of one consistent run); pass ``--allow-mixed-dirs`` only if you
intentionally assemble inputs from different runs.

All math is numpy-only and CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def rotation_angle_rad(rot: np.ndarray) -> float:
    cos_theta = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input_record(path: Path, role: str) -> dict:
    """Resolved absolute path + size + sha256 for one input file."""
    if not path.is_file():
        raise FileNotFoundError(f"input not found ({role}): {path}")
    rp = path.resolve()
    return {
        "role": role,
        "resolved_path": str(rp),
        "basename": rp.name,
        "size_bytes": int(rp.stat().st_size),
        "sha256": sha256_of(rp),
    }


def parse_z_range(spec: str) -> tuple[float, float]:
    try:
        lo_s, hi_s = spec.split(",")
        lo, hi = float(lo_s.strip()), float(hi_s.strip())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"invalid --depth-z-range '{spec}' (expect lo,hi)") from exc
    if not (lo < hi):
        raise RuntimeError(f"--depth-z-range lo must be < hi (got {lo},{hi})")
    return lo, hi


# --------------------------------------------------------------------------- #
# SE(3) Procrustes (Kabsch) -- equal-weight, CPU
# --------------------------------------------------------------------------- #
def kabsch_se3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) with target ~= source @ R + t. R in SO(3)."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError("invalid Kabsch inputs")
    if len(source) < 3:
        raise RuntimeError("Kabsch needs >=3 points")
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)
    cov = (source - src_c).T @ (target - tgt_c)
    u, _s, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    trans = tgt_c - src_c @ rot
    return rot, trans


def apply_se3(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return points @ rot + trans


# --------------------------------------------------------------------------- #
# active-cloud geometry + conditional-fit observability
# --------------------------------------------------------------------------- #
def active_cloud_svd(points: np.ndarray) -> dict:
    """Singular values of the centered active (training source) cloud.

    These describe the geometric observability of a rigid fit: a near-zero
    third singular value means the patch is (near-)planar so rotation about the
    patch normal is poorly constrained; a near-zero second value means the patch
    is (near-)linear.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise RuntimeError("invalid active cloud")
    center = pts.mean(axis=0)
    centered = pts - center
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size < 3:
        singular = np.pad(singular, (0, 3 - singular.size), constant_values=0.0)
    s0 = singular[0] if singular[0] > 1e-15 else 1e-15
    return {
        "center_world_m": center.astype(float).tolist(),
        "singular_values_m": [float(v) for v in singular[:3]],
        "rank2_ratio": float(singular[1] / s0) if singular.size > 1 else 0.0,
        "rank3_ratio": float(singular[2] / s0) if singular.size > 2 else 0.0,
        "radial_extent_m": summarize(np.linalg.norm(centered, axis=1)),
    }


def procrustes_conditional_fit_observability(
    source_centered: np.ndarray,
    n_train: int,
    noise_variance_m2: float,
    sigma_source: str,
) -> dict:
    """Approximate *conditional-fit observability* for the 6DOF Procrustes fit.

    Assumption: isotropic, independent per-coordinate Gaussian observation noise
    of variance ``noise_variance_m2``. Under that model the 6DOF Procrustes
    information is block-diagonal (translation x rotation) at the optimum:

      * translation: each point contributes equally -> F_t = (N / sigma^2) I_3,
        three equal eigenvalues (isotropic -> translation conditioning = 1).
      * rotation: the information (Hessian of the sum-of-squares cost wrt a
        small axis-angle perturbation) is the inertia tensor of the centered
        source cloud, M_ab = sum_i (||p_i||^2 delta_ab - p_i,a p_i,b), scaled by
        1/sigma^2. Its three eigenvalues measure how well each rotation axis is
        constrained by the cloud geometry (planar patch -> one small eigenvalue).

    Rotation and translation conditioning are reported SEPARATELY; there is no
    mixed 6DOF condition number (mixing rotation/translation eigenvalues is not
    a meaningful conditioning measure because they carry different units/scales).

    This EXCLUDES camera and depth covariance: ``noise_variance_m2`` is the
    patch-fit residual scale only; camera-frame drift and depth-scale error are
    NOT represented here (see ``camera_drift_separated`` / ``gauge_caveat``).
    """
    p = np.asarray(source_centered, dtype=np.float64)
    var = float(noise_variance_m2) if noise_variance_m2 > 1e-15 else 1e-15
    n = int(n_train)
    # translation information: N / sigma^2 along each axis (isotropic).
    trans_eig = np.full(3, n / var, dtype=np.float64)
    # rotation information: inertia tensor of the centered source cloud.
    if len(p) >= 1:
        M = (
            np.sum(p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2) * np.eye(3)
            - (p.T @ p)
        )
        rot_eig = np.linalg.eigvalsh(M) / var
    else:
        rot_eig = np.zeros(3, dtype=np.float64)
    rot_eig = np.sort(rot_eig)[::-1]

    def _cond(eig: np.ndarray) -> float:
        pos = eig[eig > 0.0]
        if pos.size == 0 or pos.min() <= 0.0:
            return float("inf")
        return float(pos.max() / pos.min())

    return {
        "noise_variance_m2": float(noise_variance_m2),
        "sigma_source": sigma_source,
        "translation_eigenvalues": [float(v) for v in trans_eig],
        "translation_condition_number": _cond(trans_eig),
        "rotation_eigenvalues": [float(v) for v in rot_eig],
        "rotation_condition_number": _cond(rot_eig),
        "excludes": "camera_and_depth_covariance",
        "note": (
            "isotropic per-coordinate gaussian-noise approximation at the Kabsch "
            "optimum; translation info = N/sigma^2 per axis, rotation info = "
            "inertia tensor of the centered source cloud / sigma^2. Rotation and "
            "translation conditioning are reported separately by design; there is "
            "no mixed 6DOF condition number. EXCLUDES camera and depth covariance: "
            "sigma is the patch-fit residual scale only."
        ),
    }


# --------------------------------------------------------------------------- #
# approximate projective camera resection (normalized DLT) for reprojection
# --------------------------------------------------------------------------- #
def _normalize_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    d = points - c
    mean_dist = float(np.mean(np.linalg.norm(d, axis=1))) if len(d) else 0.0
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    return (points - c) * s, T


def _normalize_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    d = points - c
    mean_dist = float(np.mean(np.linalg.norm(d, axis=1))) if len(d) else 0.0
    s = np.sqrt(3.0) / mean_dist if mean_dist > 1e-12 else 1.0
    U = np.eye(4)
    U[:3, :3] = s * np.eye(3)
    U[:3, 3] = -s * c
    return d * s, U


def resect_camera_dlt(world: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Normalized DLT for a 3x4 projection matrix P (world X -> pixel p).

    Returns (P, condition_number). P is None if under-determined (<6 points) or
    the design matrix is rank-deficient. This is an *approximate projective*
    camera recovered from the training world<->pixel correspondences -- a
    diagnostic resection, not a calibrated metric camera claim.
    """
    world = np.asarray(world, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if len(world) < 6 or world.shape[1] != 3 or pixels.shape[1] != 2:
        return None, float("inf")
    w_n, U = _normalize_3d(world)
    p_n, T = _normalize_2d(pixels)
    rows = []
    for i in range(len(w_n)):
        X, Y, Z = w_n[i]
        u, v = p_n[i]
        rows.append([X, Y, Z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * X, -u * Y, -u * Z, -u])
        rows.append([0.0, 0.0, 0.0, 0.0, X, Y, Z, 1.0, -v * X, -v * Y, -v * Z, -v])
    A = np.asarray(rows, dtype=np.float64)
    try:
        _, svals, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, float("inf")
    cond = float(svals[0] / svals[-1]) if svals[-1] > 1e-15 else float("inf")
    p_norm = Vt[-1].reshape(3, 4)
    # de-normalize: p = T^-1 * p_norm * U
    P = np.linalg.inv(T) @ p_norm @ U
    return P, cond


def project_points(P: np.ndarray, world: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([world, np.ones((len(world), 1))])
    proj = homogeneous @ P.T
    depth = proj[:, 2:3]
    depth = np.where(np.abs(depth) < 1e-12, np.nan, depth)
    return proj[:, :2] / depth


# --------------------------------------------------------------------------- #
# epoch segmentation from ready adjacent factors
# --------------------------------------------------------------------------- #
def segment_epochs(frames: list[int], ready_pairs: set[tuple[int, int]]) -> list[list[int]]:
    """Maximal contiguous runs of frames joined by ready adjacent factors.

    ``frames`` is the canonical ordered frame list. A ready adjacent factor
    (frames[i], frames[i+1]) welds those two frames into the same epoch. A frame
    with no ready factor on either side becomes a singleton (degenerate) epoch.
    """
    epochs: list[list[int]] = []
    current: list[int] = []
    for i, frame in enumerate(frames):
        if not current:
            current = [frame]
            continue
        prev = current[-1]
        if (prev, frame) in ready_pairs:
            current.append(frame)
        else:
            epochs.append(current)
            current = [frame]
    if current:
        epochs.append(current)
    return epochs


# --------------------------------------------------------------------------- #
# per-epoch anchor-relative fit
# --------------------------------------------------------------------------- #
def frame_cloud(
    world: np.ndarray, accepted: np.ndarray, usable_ids: np.ndarray, frame_i: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (track_local_ids_into_usable, world_points) for usable+accepted+finite."""
    rows = usable_ids[
        accepted[frame_i, usable_ids]
        & np.all(np.isfinite(world[frame_i, usable_ids]), axis=1)
    ]
    return rows, world[frame_i, rows]


def select_anchor(
    frames_in_epoch: list[int],
    frame_index_of: dict[int, int],
    world: np.ndarray,
    accepted: np.ndarray,
    usable_ids: np.ndarray,
) -> tuple[int, str]:
    """Pick the epoch anchor: the frame whose usable+accepted+finite cloud is
    largest (most observed), tie-broken by cloud rank3 ratio (best 3D spread)."""
    best_frame = frames_in_epoch[0]
    best_count = -1
    best_rank3 = -1.0
    for frame in frames_in_epoch:
        i = frame_index_of[frame]
        _, pts = frame_cloud(world, accepted, usable_ids, i)
        rank3 = 0.0
        if len(pts) >= 3:
            sv = active_cloud_svd(pts)
            rank3 = sv["rank3_ratio"]
        if (len(pts) > best_count) or (len(pts) == best_count and rank3 > best_rank3):
            best_count = len(pts)
            best_rank3 = rank3
            best_frame = frame
    rationale = (
        f"selected frame with max usable+accepted+finite tracks ({best_count}); "
        f"tie-break rank3_ratio={best_rank3:.4f}"
    )
    return best_frame, rationale


def fit_frame(
    anchor_frame: int,
    target_frame: int,
    frame_index_of: dict[int, int],
    world: np.ndarray,
    accepted: np.ndarray,
    tracks_xy: np.ndarray,
    usable_ids: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    a_i = frame_index_of[anchor_frame]
    t_i = frame_index_of[target_frame]
    a_rows, _a_pts_all = frame_cloud(world, accepted, usable_ids, a_i)
    t_rows, _t_pts_all = frame_cloud(world, accepted, usable_ids, t_i)
    # intersection: usable tracks observed (accepted+finite) in BOTH frames.
    common = np.intersect1d(a_rows, t_rows)
    row = {
        "frame_idx": int(target_frame),
        "anchor_frame": int(anchor_frame),
        "is_anchor": bool(target_frame == anchor_frame),
        "track_intersection_count": int(len(common)),
        "train_count": 0,
        "heldout_count": 0,
        "status": "ok",
        "rotation": np.eye(3).astype(float).tolist(),
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_angle_rad": 0.0,
        "translation_norm_m": 0.0,
        "train_world_residual_m": {"count": 0},
        "heldout_world_residual_m": {"count": 0},
        "reprojection": {"available": False, "reason": "not_computed"},
        "active_cloud_svd": {"count": 0},
        "conditional_fit_observability": {"note": "not_computed"},
        "_raw_train_world": np.array([], dtype=np.float64),
        "_raw_heldout_world": np.array([], dtype=np.float64),
        "_raw_heldout_reproj": np.array([], dtype=np.float64),
    }
    if target_frame == anchor_frame:
        row["status"] = "anchor_identity"
        return row
    if len(common) < int(args.min_train_tracks):
        row["status"] = "insufficient_intersection"
        return row

    rng = np.random.RandomState(int(args.split_seed))
    perm = rng.permutation(len(common))
    n_train = max(int(round(len(common) * float(args.train_fraction))), 3)
    n_train = min(n_train, len(common))
    train_local = common[perm[:n_train]]
    held_local = common[perm[n_train:]] if n_train < len(common) else np.array([], dtype=common.dtype)

    src_train = world[a_i, train_local]
    tgt_train = world[t_i, train_local]
    rot, trans = kabsch_se3(src_train, tgt_train)
    row["rotation"] = rot.astype(float).tolist()
    row["translation_m"] = trans.astype(float).tolist()
    row["rotation_angle_rad"] = rotation_angle_rad(rot)
    row["translation_norm_m"] = float(np.linalg.norm(trans))
    row["train_count"] = int(n_train)
    row["heldout_count"] = int(len(held_local))

    # world-space residuals (raw arrays stashed for global aggregation, stripped before dump)
    train_pred = apply_se3(src_train, rot, trans)
    train_res = np.linalg.norm(train_pred - tgt_train, axis=1)
    row["train_world_residual_m"] = summarize(train_res)
    row["_raw_train_world"] = train_res.astype(float)

    held_res = np.array([], dtype=np.float64)
    if len(held_local) > 0:
        src_held = world[a_i, held_local]
        tgt_held = world[t_i, held_local]
        held_pred = apply_se3(src_held, rot, trans)
        held_res = np.linalg.norm(held_pred - tgt_held, axis=1)
        row["heldout_world_residual_m"] = summarize(held_res)
    else:
        row["heldout_world_residual_m"] = {"count": 0, "note": "no held-out tracks (intersection fully used for training)"}
    row["_raw_heldout_world"] = held_res.astype(float)

    # active-cloud SVD on the training SOURCE (anchor) cloud
    cloud = active_cloud_svd(src_train)
    row["active_cloud_svd"] = cloud

    # conditional-fit observability for the 6DOF Procrustes fit.
    # sigma scale: prefer HELD-OUT residual scale when available (independent of
    # the fit), else fall back to the training residual. This is a patch-fit
    # residual scale only -- it EXCLUDES camera/depth covariance.
    if held_res.size:
        sigma2 = float(np.mean(held_res ** 2) / 3.0)
        sigma_source = "heldout_residual"
    elif train_res.size:
        sigma2 = float(np.mean(train_res ** 2) / 3.0)
        sigma_source = "train_residual"
    else:
        sigma2 = 1e-12
        sigma_source = "none"
    sigma2 = max(sigma2, 1e-15)
    centered_src = src_train - src_train.mean(axis=0)
    row["conditional_fit_observability"] = procrustes_conditional_fit_observability(
        centered_src, n_train=len(src_train), noise_variance_m2=sigma2,
        sigma_source=sigma_source,
    )

    # reprojection: approximate projective camera resection on the TARGET frame
    # from training world<->pixel correspondences, then reproject the
    # SE(3)-predicted held-out world points and compare to observed pixels.
    rep = {
        "available": False,
        "camera_model": "approx_projective_dlt_resection_from_target_train_world_pixel",
        "note": "projective resection diagnostic; not a calibrated metric camera",
    }
    train_px = tracks_xy[t_i, train_local]
    if n_train >= int(args.min_resection_points):
        P, cond = resect_camera_dlt(tgt_train, train_px)
        if P is not None:
            # resection sanity on training points
            proj_train = project_points(P, tgt_train)
            resection_res = np.linalg.norm(proj_train - train_px, axis=1)
            rep["available"] = True
            rep["design_condition_number"] = float(cond)
            rep["resection_train_residual_px"] = summarize(resection_res)
            if len(held_local) > 0:
                # predicted held-out world point in the target frame
                held_pred_world = apply_se3(world[a_i, held_local], rot, trans)
                proj_held = project_points(P, held_pred_world)
                held_px = tracks_xy[t_i, held_local]
                finite = np.all(np.isfinite(proj_held), axis=1)
                if np.any(finite):
                    rep_res = np.linalg.norm(proj_held[finite] - held_px[finite], axis=1)
                    rep["heldout_predicted_reprojection_px"] = summarize(rep_res)
                    row["_raw_heldout_reproj"] = rep_res.astype(float)
                else:
                    rep["heldout_predicted_reprojection_px"] = {"count": 0, "note": "no finite projections"}
                    row["_raw_heldout_reproj"] = np.array([], dtype=np.float64)
            else:
                rep["heldout_predicted_reprojection_px"] = {"count": 0, "note": "no held-out tracks"}
                row["_raw_heldout_reproj"] = np.array([], dtype=np.float64)
        else:
            rep["reason"] = "dlt_rank_deficient_or_underdetermined"
    else:
        rep["reason"] = f"fewer than {int(args.min_resection_points)} training correspondences"
    row["reprojection"] = rep
    row.setdefault("_raw_heldout_reproj", np.array([], dtype=np.float64))
    return row


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
GAUGE_CAVEAT = (
    "GAUGE / DRIFT CAVEAT. The fitted SE(3) is an ANCHOR-RELATIVE "
    "visible-surface-patch transform: the anchor frame fixes the gauge, so only "
    "inter-frame patch deformation is measured, NOT absolute world-frame object "
    "pose. Camera/depth frame drift between frames is NOT separated or "
    "compensated (camera_drift_separated=false); therefore part of any residual "
    "may be camera-frame or depth-scale drift rather than patch deformation. "
    "This diagnostic CANNOT be read as object motion, static/still "
    "classification, contact, occlusion, nonpenetration, or mesh-reconstruction "
    "evidence. The conditional_fit_observability block EXCLUDES camera and depth "
    "covariance by construction."
)


# --------------------------------------------------------------------------- #
# R4a hard gate: gauge precondition / GT quarantine / visible-surfel provenance
# --------------------------------------------------------------------------- #
# A static gauge (verified by an R4b gauge-control audit on static background
# points) is the precondition for ever reading the anchor-relative SE(3) as
# separable from camera/depth frame drift. Without a verified gauge the fitted
# transform is gauge-confounded and CANNOT be read as object pose. This gate
# never promotes the output to object pose: even when the gauge is verified the
# output remains diagnostic-only (annotation_ready=false, is_full_object_pose=false).
GAUGE_ACCEPTED_STATUS_TOKENS = (
    "static_gauge_established",
    "gauge_accepted",
    "gauge_static",
)


def _gauge_status_is_accepted(status) -> bool:
    if status is None:
        return False
    s = str(status).strip().lower()
    if not s:
        return False
    if s in {"accepted", "static", "background_static_gauge_established"}:
        return True
    return any(tok in s for tok in GAUGE_ACCEPTED_STATUS_TOKENS)


def evaluate_gauge_gate(gauge_audit_path) -> tuple[dict, str, str]:
    """Evaluate the R4a gauge precondition from an optional R4b gauge audit JSON.

    Returns (gauge_audit_record, gate_status, reason).

    gate_status is one of:
      * blocked_unverified_gauge   -- no gauge audit supplied / missing / unreadable / no status
      * blocked_rejected_gauge     -- gauge audit present but status not accepted/static
      * diagnostic_only_gauge_verified -- gauge accepted; output STILL diagnostic-only

    This NEVER promotes the output to object pose: even when the gauge is
    verified the caller must explicitly consume it downstream with its own
    covariance/gauge handling.
    """
    rec = {
        "provided": gauge_audit_path is not None,
        "resolved_path": None,
        "basename": None,
        "sha256": None,
        "schema": None,
        "status": None,
        "accepted": False,
    }
    if gauge_audit_path is None:
        return rec, "blocked_unverified_gauge", (
            "no --gauge-audit-json supplied; the static gauge is unverified, so the "
            "anchor-relative SE(3) is gauge-confounded and cannot be read as object pose")
    path = Path(gauge_audit_path)
    if not path.is_file():
        return rec, "blocked_unverified_gauge", (
            f"gauge audit JSON not found: {path}; gauge is unverified")
    rp = path.resolve()
    rec["resolved_path"] = str(rp)
    rec["basename"] = rp.name
    rec["sha256"] = sha256_of(rp)
    try:
        doc = load_json(rp)
        rec["_text"] = rp.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return rec, "blocked_unverified_gauge", (
            f"gauge audit JSON unreadable: {exc}; gauge is unverified")
    rec["schema"] = doc.get("schema")
    status = doc.get("gauge_control_status")
    if status is None:
        status = doc.get("status")
    rec["status"] = status
    if not status:
        return rec, "blocked_unverified_gauge", (
            "gauge audit JSON carries no gauge_control_status/status field; "
            "gauge is unverified")
    if _gauge_status_is_accepted(status):
        rec["accepted"] = True
        return rec, "diagnostic_only_gauge_verified", (
            "gauge verified as static; output remains diagnostic-only and is NOT "
            "object pose unless a caller explicitly consumes it downstream with "
            "its own covariance/gauge handling")
    return rec, "blocked_rejected_gauge", (
        f"gauge audit status '{status}' is not an accepted/static gauge; the "
        "anchor-relative SE(3) is gauge-confounded and cannot be read as object pose")


# Forbidden GT signals. Any hit in inputs/config is a GT-free contract violation.
# Token names are deliberately listed here only as a negative quarantine note.
_GT_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("hot3d_gt", re.compile(r"hot3d[_/\.\-]?gt", re.IGNORECASE)),
    ("gt_sidecar", re.compile(r"gt[_/\.\-]?sidecar", re.IGNORECASE)),
    ("ground_truth_sidecar", re.compile(r"ground[_ ]?truth[_ ]?sidecar", re.IGNORECASE)),
    ("evaluation_hot3d_gt_dir", re.compile(r"evaluation[/\\]hot3d_gt", re.IGNORECASE)),
    ("gt_frame_index", re.compile(r"gt[_/\.\-]?frame", re.IGNORECASE)),
    ("270.3mm_range", re.compile(r"270\.3")),
    ("ADD-S_metric", re.compile(r"\bADD[-_ ]?S\b")),
    ("ADD_metric", re.compile(r"\bADD\b")),
]
GT_FORBIDDEN_TOKENS_DOCUMENTED = [name for name, _ in _GT_FORBIDDEN_PATTERNS]


def _scan_forbidden(text: str) -> list[dict]:
    hits = []
    for name, pat in _GT_FORBIDDEN_PATTERNS:
        m = pat.search(text or "")
        if m:
            hits.append({"forbidden_token": name, "match": m.group(0)})
    return hits


def gt_quarantine_assertion(args, input_paths, json_texts, gauge_text) -> dict:
    """Hard GT quarantine over config + inputs + gauge audit. Raises on any hit."""
    config_blob = json.dumps({
        "cotracker_npz": str(args.cotracker_npz),
        "sparse_edges_json": str(args.sparse_edges_json),
        "pair_factors_json": str(args.pair_factors_json),
        "output_dir": str(args.output_dir),
        "gauge_audit_json": str(args.gauge_audit_json) if args.gauge_audit_json else None,
        "depth_z_range": list(args.depth_z_range),
        "train_fraction": args.train_fraction,
        "split_seed": args.split_seed,
        "min_train_tracks": args.min_train_tracks,
        "min_resection_points": args.min_resection_points,
        "allow_unscaled_input": bool(args.allow_unscaled_input),
        "allow_mixed_dirs": bool(args.allow_mixed_dirs),
    }, sort_keys=True)
    corpora = [
        ("config", config_blob),
        ("input_paths", "\n".join(str(p) for p in input_paths)),
    ]
    for role, text in json_texts:
        corpora.append((role, text))
    if gauge_text is not None:
        corpora.append(("gauge_audit_json_text", gauge_text))
    all_hits = []
    for role, text in corpora:
        for h in _scan_forbidden(text):
            all_hits.append({"corpus": role, **h})
    result = {
        "passed": not all_hits,
        "scanned_corpora": [role for role, _ in corpora],
        "hits": all_hits,
        "forbidden_tokens": GT_FORBIDDEN_TOKENS_DOCUMENTED,
        "note": (
            "Hard GT quarantine: inputs, config, and gauge audit must contain no "
            "HOT3D GT sidecar path, GT frame index, 270.3mm range, ADD/ADD-S "
            "metric, or hot3d_gt string. This diagnostic is a GT-free prediction-"
            "side measurement; any forbidden GT token in the input/config path is "
            "a contract violation. The forbidden token names below appear ONLY as "
            "a negative quarantine note and are not evidence."
        ),
    }
    if all_hits:
        raise RuntimeError(
            "GT QUARANTINE FAILED: forbidden GT tokens detected in inputs/config: "
            f"{all_hits}. This diagnostic must remain GT-free; refusing to produce "
            "pose-shaped output from GT-contaminated inputs.")
    return result


def assert_visible_surfel_provenance(edges_doc, edges_path) -> dict:
    """Assert the sparse-edges JSON names a visible-SURFEL archive/method and NOT
    a mesh / completed-mesh archive. Raises on failure."""
    surfel_archive = edges_doc.get("visible_surfel_archive")
    method = str(edges_doc.get("method", ""))
    claim = str(edges_doc.get("claim_tested", ""))
    term = str(edges_doc.get("terminology_note", ""))
    surfel_language = "surfel" in (method + " " + claim + " " + term).lower()
    names_surfel_archive = bool(surfel_archive) and str(surfel_archive).strip() != ""
    mesh_signals = []
    for key in ("mesh_archive", "completed_mesh_archive", "object_mesh_archive"):
        if edges_doc.get(key):
            mesh_signals.append(f"{key}={edges_doc.get(key)}")
    if edges_doc.get("is_completed_mesh"):
        mesh_signals.append("is_completed_mesh=true")
    kind = str(edges_doc.get("archive_kind", ""))
    if kind and "mesh" in kind.lower() and "compat" not in kind.lower():
        mesh_signals.append(f"archive_kind={kind}")
    if surfel_archive:
        sa_low = str(surfel_archive).lower()
        if any(tok in sa_low for tok in
               ("completed_mesh", "object_mesh", "canonical_mesh", "completed_object")):
            mesh_signals.append(f"visible_surfel_archive names a mesh: {surfel_archive}")
    result = {
        "passed": False,
        "visible_surfel_archive": surfel_archive,
        "method": method,
        "names_visible_surfel_archive": names_surfel_archive,
        "carries_surfel_language": bool(surfel_language),
        "mesh_archive_signals": mesh_signals,
        "source_json": str(edges_path),
        "note": (
            "Visible-surfel provenance: the sparse-edges JSON must name the "
            "visible-surfel archive (visible_surfel_archive) and visible-surfel "
            "method/claim language, and must NOT name a completed/object mesh "
            "archive. A mesh archive would turn the 'visible-surface patch' into a "
            "full-mesh claim, which this diagnostic explicitly is not."
        ),
    }
    if not names_surfel_archive:
        raise RuntimeError(
            "VISIBLE-SURFEL PROVENANCE FAILED: sparse-edges JSON has no "
            f"visible_surfel_archive field ({edges_path}). Cannot assert "
            "visible-surface-patch provenance.")
    if not surfel_language:
        raise RuntimeError(
            "VISIBLE-SURFEL PROVENANCE FAILED: sparse-edges JSON method/claim "
            f"does not reference visible surfels ({edges_path}).")
    if mesh_signals:
        raise RuntimeError(
            "VISIBLE-SURFEL PROVENANCE FAILED: sparse-edges JSON names a mesh "
            f"archive / completed mesh: {mesh_signals}. A mesh archive is "
            "forbidden for a visible-surface-patch diagnostic.")
    result["passed"] = True
    return result


def epoch_break_semantics_block(epochs: list[list[int]]) -> dict:
    return {
        "semantics": "uncertified_cut",
        "stitching": False,
        "interpolation": False,
        "smoothing": False,
        "epoch_count": int(len(epochs)),
        "break_count": int(max(0, len(epochs) - 1)),
        "note": (
            "Epoch breaks are uncertified cuts: a gap between ready adjacent "
            "factors severs the epoch with NO stitching, NO interpolation, and NO "
            "smoothing across the break. Each epoch is fit independently against "
            "its own anchor; a degenerate (single-frame) epoch produces no "
            "anchor-relative fit. No cross-epoch transform, trajectory, or "
            "continuous pose is constructed."
        ),
    }


def run(args: argparse.Namespace) -> dict:
    # ---- input resolution + integrity records ------------------------------ #
    inputs_dir_set = {
        args.cotracker_npz.resolve().parent,
        args.sparse_edges_json.resolve().parent,
        args.pair_factors_json.resolve().parent,
    }
    if len(inputs_dir_set) != 1 and not args.allow_mixed_dirs:
        raise RuntimeError(
            "inputs must resolve under the SAME directory (sibling products of "
            f"one run); got distinct dirs {sorted(str(d) for d in inputs_dir_set)}. "
            "Pass --allow-mixed-dirs only if you intentionally assemble inputs "
            "from different runs."
        )
    input_records = {
        "cotracker_npz": resolve_input_record(args.cotracker_npz, "cotracker_object_tracks_v5"),
        "sparse_edges_json": resolve_input_record(args.sparse_edges_json, "sparse_correspondence_edges_v5"),
        "pair_factors_json": resolve_input_record(args.pair_factors_json, "pairwise_rigid_factors_v6"),
    }

    # ---- R4a gauge precondition (R4b gauge-control audit) ------------------ #
    # Evaluated up front; it only needs the optional gauge-audit JSON. The SE(3)
    # math below still runs (this is a rigid-fit diagnostic), but the gate
    # status records whether the transform can EVER be read as separable from
    # camera/depth drift. It never promotes the output to object pose.
    gauge_audit_path = args.gauge_audit_json if args.gauge_audit_json else None
    gauge_audit_record, r4_gate_status, r4_gate_reason = evaluate_gauge_gate(gauge_audit_path)
    gauge_audit_text = gauge_audit_record.pop("_text", None)

    # ---- load CoTracker archive + enforce scale-depth contract ------------- #
    tracks = np.load(args.cotracker_npz)
    required_always = {"frame_idx", "tracks_xy", "accepted", "world_xyz"}
    missing = required_always.difference(tracks.files)
    if missing:
        raise RuntimeError(f"CoTracker archive missing keys: {sorted(missing)}")
    has_depth_xy = "tracks_depth_xy" in tracks.files
    if not has_depth_xy and not args.allow_unscaled_input:
        raise RuntimeError(
            "CoTracker archive lacks 'tracks_depth_xy': this is not the "
            "depth-scale-corrected (_scaledepth) product. Refusing ambiguous "
            "pre-fix input. Pass --allow-unscaled-input ONLY for synthetic "
            "fixtures that legitimately lack it."
        )
    input_scale_status = (
        "scale_depth_tracks_depth_xy_present"
        if has_depth_xy
        else "unscaled_allowed_synthetic"
    )
    npz_frames = np.asarray(tracks["frame_idx"], dtype=np.int64).tolist()
    tracks_xy = np.asarray(tracks["tracks_xy"], dtype=np.float64)
    accepted = np.asarray(tracks["accepted"], dtype=bool)
    world = np.asarray(tracks["world_xyz"], dtype=np.float64)
    if world.shape[:2] != accepted.shape:
        raise RuntimeError(f"world/accepted shape mismatch: {world.shape} vs {accepted.shape}")
    if tracks_xy.shape[:2] != accepted.shape:
        raise RuntimeError(f"tracks_xy/accepted shape mismatch: {tracks_xy.shape} vs {accepted.shape}")
    if has_depth_xy:
        depth_xy = np.asarray(tracks["tracks_depth_xy"], dtype=np.float64)
        if depth_xy.shape[:2] != accepted.shape:
            raise RuntimeError(f"tracks_depth_xy/accepted shape mismatch: {depth_xy.shape} vs {accepted.shape}")

    # ---- assert finite median world z in plausible metric range ------------ #
    z_lo, z_hi = args.depth_z_range
    z_all = np.asarray(world[..., 2], dtype=np.float64).ravel()
    z_fin = z_all[np.isfinite(z_all)]
    if z_fin.size == 0:
        raise RuntimeError("world_xyz has no finite z values; cannot assert plausible depth range")
    z_median = float(np.median(z_fin))
    if not (z_lo <= z_median <= z_hi):
        raise RuntimeError(
            f"finite median world_xyz z = {z_median:.6f} m is outside the "
            f"plausible metric range [{z_lo}, {z_hi}] m. This typically means "
            "the input is an UNSCALED depth lift (raw camera/world units), not "
            "the depth-scale-corrected (_scaledepth) product. Adjust "
            "--depth-z-range only if this range is genuinely wrong for the clip."
        )

    frame_index_of = {int(f): i for i, f in enumerate(npz_frames)}

    # ---- load edges + pair factors ----------------------------------------- #
    edges_doc = load_json(args.sparse_edges_json)
    edge_list = edges_doc.get("edges")
    if not isinstance(edge_list, list):
        raise RuntimeError(f"sparse edge report has no edges list: {args.sparse_edges_json}")
    usable_ids = np.array(sorted({int(e["track_id"]) for e in edge_list}), dtype=np.int64)

    factor_doc = load_json(args.pair_factors_json)
    factor_frames = factor_doc.get("frames")
    if not isinstance(factor_frames, list) or not factor_frames:
        raise RuntimeError(f"pair factor report has no frames list: {args.pair_factors_json}")
    frames = [int(f) for f in factor_frames]
    if any(f not in frame_index_of for f in frames):
        raise RuntimeError("pair-factor frames are not a subset of CoTracker frames")
    pair_rows = factor_doc.get("pair_rows")
    if not isinstance(pair_rows, list):
        raise RuntimeError(f"pair factor report has no pair_rows list: {args.pair_factors_json}")
    ready_pairs: set[tuple[int, int]] = set()
    for r in pair_rows:
        if not r.get("rigid_factor_ready"):
            continue
        ready_pairs.add((int(r["source_frame"]), int(r["target_frame"])))

    # ---- visible-surfel provenance + GT quarantine (hard input contracts) -- #
    visible_surfel_provenance = assert_visible_surfel_provenance(
        edges_doc, args.sparse_edges_json.resolve())
    gt_quarantine = gt_quarantine_assertion(
        args,
        input_paths=[
            args.cotracker_npz.resolve(),
            args.sparse_edges_json.resolve(),
            args.pair_factors_json.resolve(),
        ] + ([Path(gauge_audit_path).resolve()] if gauge_audit_path else []),
        json_texts=[
            ("sparse_edges_json_text", args.sparse_edges_json.read_text(encoding="utf-8")),
            ("pair_factors_json_text", args.pair_factors_json.read_text(encoding="utf-8")),
        ],
        gauge_text=gauge_audit_text,
    )

    epochs = segment_epochs(frames, ready_pairs)

    epoch_summaries = []
    frame_rows = []
    for eid, epoch_frames in enumerate(epochs):
        anchor_frame, rationale = select_anchor(epoch_frames, frame_index_of, world, accepted, usable_ids)
        ready_edge_count = sum(1 for i in range(len(epoch_frames) - 1)
                               if (epoch_frames[i], epoch_frames[i + 1]) in ready_pairs)
        degenerate = len(epoch_frames) < 2
        epoch_summaries.append({
            "epoch_id": int(eid),
            "frames": [int(f) for f in epoch_frames],
            "frame_count": int(len(epoch_frames)),
            "anchor_frame": int(anchor_frame),
            "anchor_selection": rationale,
            "ready_edge_count": int(ready_edge_count),
            "degenerate": bool(degenerate),
            "note": ("single-frame epoch: no ready adjacent factor on either side; "
                     "no anchor-relative fit possible") if degenerate else "",
        })
        for frame in epoch_frames:
            fr = fit_frame(anchor_frame, frame, frame_index_of, world, accepted,
                           tracks_xy, usable_ids, args)
            fr["epoch_id"] = int(eid)
            frame_rows.append(fr)

    # Global summaries aggregated from per-frame raw residuals (exact, no recompute).
    train_world_residuals = np.concatenate(
        [np.asarray(r["_raw_train_world"], dtype=np.float64) for r in frame_rows]
    ) if frame_rows else np.array([], dtype=np.float64)
    heldout_world_residuals = np.concatenate(
        [np.asarray(r["_raw_heldout_world"], dtype=np.float64) for r in frame_rows]
    ) if frame_rows else np.array([], dtype=np.float64)
    heldout_reproj_residuals = np.concatenate(
        [np.asarray(r["_raw_heldout_reproj"], dtype=np.float64) for r in frame_rows]
    ) if frame_rows else np.array([], dtype=np.float64)
    # strip private raw keys before serialization
    for r in frame_rows:
        for k in ("_raw_train_world", "_raw_heldout_world", "_raw_heldout_reproj"):
            r.pop(k, None)

    report = {
        "status": "ok",
        "measurement_type": "visible_surface_patch_relative_se3",
        "diagnostic_only": True,
        "annotation_ready": False,
        "is_full_object_pose": False,
        "camera_drift_separated": False,
        # The R4a gate is the hard control on reading this as pose evidence.
        # pose_evidence_admissible is ALWAYS false from this diagnostic alone:
        # even a gauge-verified run stays diagnostic-only unless a caller
        # explicitly consumes it downstream with its own covariance/gauge handling.
        "pose_evidence_admissible": False,
        "r4_pose_gate_status": r4_gate_status,
        "label": "visible_surface_patch_relative_se3",
        "gauge_caveat": GAUGE_CAVEAT,
        "r4_pose_gate": {
            "status": r4_gate_status,
            "gauge_audit": gauge_audit_record,
            "diagnostic_only": True,
            "annotation_ready": False,
            "is_full_object_pose": False,
            "camera_drift_separated": False,
            "pose_evidence_admissible": False,
            "reason": r4_gate_reason,
            "note": (
                "Hard R4a gate. The anchor-relative SE(3) is gauge-confounded "
                "(camera/depth drift not separated) unless a static gauge is "
                "verified by an R4b gauge-control audit. blocked_* => the transform "
                "MUST NOT be read as object pose. diagnostic_only_gauge_verified => "
                "the gauge precondition holds, but the output is STILL diagnostic-"
                "only and is NOT object pose unless a caller explicitly consumes "
                "it downstream with its own covariance/gauge handling. In no case "
                "does this script emit a full object pose, contact, occlusion, "
                "nonpenetration, or mesh verdict."),
        },
        "gt_quarantine": gt_quarantine,
        "visible_surfel_provenance": visible_surfel_provenance,
        "epoch_break_semantics": epoch_break_semantics_block(epochs),
        "method": "fit_visible_surface_patch_se3_epochs",
        "claim_tested": (
            "On frame spans that the v6 pairwise stage already declared "
            "rigid-ready, a single anchor-relative 6DOF Procrustes SE(3) over the "
            "visible-surface patch tracks explains held-out tracks with bounded "
            "world and reprojection residual, and the active cloud geometry makes "
            "the conditional fit observably conditioned. This is a "
            "visible-surface-patch, anchor-relative transform; it is NOT a full "
            "object pose, camera/depth drift is NOT separated, and it implies NO "
            "object-motion/static/contact/occlusion/nonpenetration/mesh verdict."
        ),
        "input_scale_status": input_scale_status,
        "inputs": input_records,
        "input_validation": {
            "tracks_depth_xy_present": bool(has_depth_xy),
            "allow_unscaled_input": bool(args.allow_unscaled_input),
            "same_directory_enforced": (len(inputs_dir_set) == 1),
            "allow_mixed_dirs": bool(args.allow_mixed_dirs),
            "world_z_median_m": z_median,
            "world_z_range_m": [z_lo, z_hi],
            "world_z_in_range": bool(z_lo <= z_median <= z_hi),
        },
        "frames": [int(f) for f in frames],
        "frames_in_npz_order": [int(f) for f in npz_frames],
        "track_count": int(accepted.shape[1]),
        "usable_track_count": int(len(usable_ids)),
        "usable_track_ids": [int(v) for v in usable_ids.tolist()],
        "ready_adjacent_pair_count": int(len(ready_pairs)),
        "epoch_count": int(len(epochs)),
        "degenerate_epoch_count": int(sum(1 for e in epoch_summaries if e["degenerate"])),
        "epochs": epoch_summaries,
        "global_train_world_residual_m": summarize(train_world_residuals),
        "global_heldout_world_residual_m": summarize(heldout_world_residuals),
        "global_heldout_reprojection_residual_px": summarize(heldout_reproj_residuals),
        "frame_rows": frame_rows,
        "parameters": {
            "train_fraction": float(args.train_fraction),
            "split_seed": int(args.split_seed),
            "min_train_tracks": int(args.min_train_tracks),
            "min_resection_points": int(args.min_resection_points),
            "depth_z_range_m": [z_lo, z_hi],
        },
    }
    # ---- self-scan: the report itself must carry no forbidden GT token -- #
    # except inside the gt_quarantine block, which legitimately names the
    # forbidden tokens as a negative note. Scan a copy with that block blanked.
    report_self = dict(report)
    report_self["gt_quarantine"] = "<quarantine_negative_note_excluded>"
    report_self_blob = json.dumps(report_self, sort_keys=True)
    self_hits = _scan_forbidden(report_self_blob)
    if self_hits:
        raise RuntimeError(
            "GT QUARANTINE SELF-SCAN FAILED: the emitted report carries a "
            f"forbidden GT token outside the quarantine note: {self_hits}. "
            "This is an implementation bug; refusing to write GT-tainted output.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "qc_visible_surface_patch_se3_epochs.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    omitted = {"epochs", "frame_rows"}
    print(json.dumps({k: v for k, v in report.items() if k not in omitted}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cotracker-npz", type=Path, required=True)
    parser.add_argument("--sparse-edges-json", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--min-train-tracks", type=int, default=4)
    parser.add_argument("--min-resection-points", type=int, default=8)
    parser.add_argument(
        "--depth-z-range", type=parse_z_range, default=(0.25, 0.9),
        dest="depth_z_range",
        help="comma-separated lo,hi plausible metric range for finite median "
             "world_xyz z (default 0.25,0.9 for clip001851 _scaledepth inputs)",
    )
    parser.add_argument(
        "--allow-unscaled-input", action="store_true",
        help="synthetic ONLY: do not require tracks_depth_xy in the CoTracker NPZ",
    )
    parser.add_argument(
        "--allow-mixed-dirs", action="store_true",
        help="allow the three inputs to resolve under different directories",
    )
    parser.add_argument(
        "--gauge-audit-json", type=Path, default=None,
        help="optional R4b gauge-control audit JSON (e.g. "
             "PROVENANCE_AND_STATICNESS_AUDIT.json with gauge_control_status). "
             "If absent/missing/no-status => r4_pose_gate_status=blocked_unverified_gauge; "
             "if status is not accepted/static => blocked_rejected_gauge; if "
             "accepted/static => diagnostic_only_gauge_verified (still diagnostic "
             "only). The gate never promotes the output to object pose.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
