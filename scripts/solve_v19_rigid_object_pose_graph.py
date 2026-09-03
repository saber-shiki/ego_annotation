#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


POSE_MEASUREMENT_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "fit_to_object_owned_rgb_calibrated_pnp",
}
CORRECTED_POSE_STATUS = "corrected_temporal_rigid_pose_graph"
COMPLETED_POSE_STATUS = "completed_temporal_rigid_pose_uncertain"
STRICT_ROTATION_STEP_DEG = 15.0
CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS = {
    "max_rotation_step_deg": 18.0,
    "max_count": 2,
    "max_fraction": 0.015,
    "max_translation_step_m": 0.020,
    "max_endpoint_observability_score": 0.030,
}


@dataclass(frozen=True)
class PoseObservation:
    frame_idx: int
    source_row: dict[str, Any]
    rotation_world_from_canonical: np.ndarray
    translation_world_m: np.ndarray
    translation_sigma_m: float
    rotation_sigma_rad: float
    visible_sample_count: int
    observed_points_world: np.ndarray
    nonpenetration_target_world_m: np.ndarray | None
    nonpenetration_weight: float
    nonpenetration_source_rows: int


@dataclass(frozen=True)
class ImageFactorObservation:
    frame_idx: int
    T_world_camera: np.ndarray
    K_raster: np.ndarray
    depth_points_canonical: np.ndarray
    depth_observed_z: np.ndarray
    silhouette_points_canonical: np.ndarray
    silhouette_target_uv: np.ndarray
    silhouette_kind: np.ndarray


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"invalid {name}: expected {shape}, got {arr.shape}")
    return arr


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no triangle mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, int(count), seed=rng)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise RuntimeError("sampled mesh points invalid")
    return pts


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    query = np.asarray(query, dtype=float)
    target = np.asarray(target, dtype=float)
    if query.ndim != 2 or target.ndim != 2 or query.shape[1:] != (3,) or target.shape[1:] != (3,) or len(query) == 0 or len(target) == 0:
        return {"count": int(len(query)) if query.ndim == 2 else 0, "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
    d, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "count": int(len(query)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90.0)),
        "p95_m": float(np.percentile(d, 95.0)),
        "mean_m": float(np.mean(d)),
        "max_m": float(np.max(d)),
    }


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation[None, :]


def corrected_pose(obs: PoseObservation, rot_delta: np.ndarray, trans_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_corr = Rotation.from_rotvec(rot_delta).as_matrix() @ obs.rotation_world_from_canonical
    t_corr = obs.translation_world_m + trans_delta
    return r_corr, t_corr


def pose_row_sigma(row: dict[str, Any], object_radius_m: float, args: argparse.Namespace) -> tuple[float, float]:
    final = row.get("observed_to_mesh_final") if isinstance(row.get("observed_to_mesh_final"), dict) else {}
    median = final.get("median_m")
    p95 = final.get("p95_m")
    visible_n = int(row.get("visible_sample_count") or final.get("count") or 0)
    finite = [float(x) for x in (median, p95) if x is not None and math.isfinite(float(x))]
    residual_scale = max(finite) if finite else float(args.default_pose_sigma_m)
    # ICP pose is a measurement of the visible surface. More visible points tighten it, but not below
    # the physical depth/camera floor; high residuals widen it so the graph can expose systematic conflict.
    sample_factor = math.sqrt(max(1.0, min(float(visible_n), 400.0)) / 100.0)
    sigma_t = residual_scale / max(1.0, sample_factor)
    sigma_t = float(np.clip(max(float(args.min_pose_sigma_m), sigma_t), float(args.min_pose_sigma_m), float(args.max_pose_sigma_m)))
    sigma_r = sigma_t / max(float(object_radius_m), 1.0e-3)
    sigma_r = float(np.clip(sigma_r, float(args.min_pose_rotation_sigma_rad), float(args.max_pose_rotation_sigma_rad)))
    return sigma_t, sigma_r


def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if obj.get("object_id") == object_id:
            return obj
    legacy = frame.get("object") if isinstance(frame.get("object"), dict) else None
    if legacy and legacy.get("object_id") in (None, object_id):
        return legacy
    return None


def constraint_targets(path: Path | None, max_target_m: float, accepted_states: set[str]) -> dict[int, tuple[np.ndarray, float, int]]:
    if path is None:
        return {}
    data = load_json(path)
    by_frame: dict[int, list[np.ndarray]] = {}
    for row in data.get("constraint_rows", []) if isinstance(data, dict) else []:
        state = str(row.get("candidate_application_state") or "")
        if state not in accepted_states:
            continue
        cand = np.asarray(row.get("candidate_translation_world_m") or [], dtype=float)
        if cand.shape != (3,) or not np.isfinite(cand).all():
            continue
        norm = float(np.linalg.norm(cand))
        if norm <= 0.0:
            continue
        # The row's vector moves the hand out of the object. The object-pose graph can only use the
        # opposite direction as bounded pressure, never as a hard truth about the object state.
        target = -cand
        tnorm = float(np.linalg.norm(target))
        if tnorm > max_target_m:
            target = target * (max_target_m / max(tnorm, 1.0e-12))
        by_frame.setdefault(int(row["frame_idx"]), []).append(target.astype(float))
    out: dict[int, tuple[np.ndarray, float, int]] = {}
    for idx, targets in by_frame.items():
        stack = np.vstack(targets)
        med = np.median(stack, axis=0)
        weight = min(1.0, math.sqrt(len(targets)))
        out[int(idx)] = (med.astype(float), float(weight), int(len(targets)))
    return out


def load_image_factor_observations(
    path: Path | None,
    observations: list[PoseObservation],
    args: argparse.Namespace,
    expected_completed_mesh: Path,
) -> tuple[dict[int, ImageFactorObservation], dict[str, Any]]:
    if path is None:
        return {}, {"enabled": False}
    data = np.load(path.expanduser().resolve())
    metadata_raw = data["metadata"][0] if "metadata" in data.files else "{}"
    metadata = json.loads(str(metadata_raw))
    resolved = path.expanduser().resolve()
    expected = {
        "annotations": str(args.annotations.expanduser().resolve()),
        "pose_report": str(args.pose_report.expanduser().resolve()),
        "completed_mesh": str(expected_completed_mesh.expanduser().resolve()),
        "object_id": str(args.object_id),
    }
    for key, expected_value in expected.items():
        actual = str(metadata.get(key) or "")
        if actual != expected_value:
            raise RuntimeError(
                f"image factor contract mismatch for {key}: expected {expected_value}, got {actual}; "
                "rebuild first-hit/silhouette factors for this exact P14/annotation/mesh input"
            )
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
    T_world_camera = np.asarray(data["T_world_camera"], dtype=np.float64)
    K_raster = np.asarray(data["K_raster"], dtype=np.float64)
    depth_offsets = np.asarray(data["depth_offsets"], dtype=np.int64)
    silhouette_offsets = np.asarray(data["silhouette_offsets"], dtype=np.int64)
    depth_points = np.asarray(data["depth_points_canonical"], dtype=np.float64)
    depth_z = np.asarray(data["depth_observed_z"], dtype=np.float64)
    silhouette_points = np.asarray(data["silhouette_points_canonical"], dtype=np.float64)
    silhouette_targets = np.asarray(data["silhouette_target_uv"], dtype=np.float64)
    silhouette_kind = np.asarray(data["silhouette_kind"], dtype=np.int8)
    if T_world_camera.shape != (len(frame_idx), 4, 4) or K_raster.shape != (3, 3):
        raise RuntimeError("invalid image factor camera arrays")
    if depth_offsets.shape != (len(frame_idx) + 1,) or silhouette_offsets.shape != (len(frame_idx) + 1,):
        raise RuntimeError("invalid image factor offset arrays")
    by_frame: dict[int, ImageFactorObservation] = {}
    for pos, idx in enumerate(frame_idx.tolist()):
        d0, d1 = int(depth_offsets[pos]), int(depth_offsets[pos + 1])
        s0, s1 = int(silhouette_offsets[pos]), int(silhouette_offsets[pos + 1])
        factor = ImageFactorObservation(
            frame_idx=int(idx),
            T_world_camera=T_world_camera[pos],
            K_raster=K_raster,
            depth_points_canonical=depth_points[d0:d1],
            depth_observed_z=depth_z[d0:d1],
            silhouette_points_canonical=silhouette_points[s0:s1],
            silhouette_target_uv=silhouette_targets[s0:s1],
            silhouette_kind=silhouette_kind[s0:s1],
        )
        if factor.depth_points_canonical.ndim != 2 or factor.depth_points_canonical.shape[1] != 3:
            raise RuntimeError(f"invalid depth factors for frame {idx}")
        if factor.silhouette_points_canonical.ndim != 2 or factor.silhouette_points_canonical.shape[1] != 3:
            raise RuntimeError(f"invalid silhouette factors for frame {idx}")
        if factor.silhouette_target_uv.shape != (len(factor.silhouette_points_canonical), 2):
            raise RuntimeError(f"invalid silhouette targets for frame {idx}")
        by_frame[int(idx)] = factor
    missing = [obs.frame_idx for obs in observations if obs.frame_idx not in by_frame]
    if missing:
        raise RuntimeError(f"image factors missing P15 observation frames: {missing[:12]}")
    metadata_out = {
        **metadata,
        "enabled": True,
        "factor_npz": str(resolved),
        "loaded_frame_count": int(len(by_frame)),
        "depth_factor_count": int(sum(len(f.depth_observed_z) for f in by_frame.values())),
        "silhouette_factor_count": int(sum(len(f.silhouette_kind) for f in by_frame.values())),
    }
    return by_frame, metadata_out


def build_observations(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any], mesh: trimesh.Trimesh) -> tuple[list[PoseObservation], list[dict[str, Any]], dict[int, tuple[np.ndarray, float, int]]]:
    frames_by_idx = {int(frame.get("frame_idx")): frame for frame in annotations.get("frames", []) if isinstance(frame, dict) and frame.get("frame_idx") is not None}
    radius = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)) / 2.0)
    accepted = set(args.nonpenetration_states)
    targets = constraint_targets(args.constraint_report, float(args.max_nonpenetration_target_m), accepted)
    observations: list[PoseObservation] = []
    skipped: list[dict[str, Any]] = []
    for row in pose_report.get("pose_rows", []) if isinstance(pose_report.get("pose_rows"), list) else []:
        if str(row.get("status") or "") not in POSE_MEASUREMENT_STATUSES:
            continue
        idx = int(row["frame_idx"])
        if row.get("rigid_pose_observation_eligible") is False and not args.include_ineligible_rigid_pose_observations:
            skipped.append(
                {
                    "frame_idx": idx,
                    "reason": "explicit upstream rigid_pose_observation_eligible=false",
                    "policy": "hard rejected; the production CLI rejects the historical override flag",
                }
            )
            continue
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        try:
            rot = as_array(row.get("rotation_world_from_completed_canonical_matrix"), (3, 3), f"pose rotation frame {idx}")
            trans = as_array(row.get("translation_world_m"), (3,), f"pose translation frame {idx}")
            frame = frames_by_idx[idx]
            obj = annotation_object(frame, args.object_id)
            if obj is None:
                raise RuntimeError("annotation object missing")
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
            observed_evidence_source = "current_frame_accepted_metric_surfels"
            if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < int(args.min_visible_points) or not np.isfinite(observed).all():
                tracked_evidence = np.asarray(row.get("direct_pose_evidence_world_points_m") or [], dtype=float)
                tracked_kind = str(row.get("direct_pose_evidence_kind") or "")
                direct_source = str(row.get("direct_pose_observation_source") or "")
                generated_consumed = row.get("generated_geometry_pose_evidence_consumed")
                if (
                    direct_source == "object_owned_rgb_optical_flow_calibrated_pnp"
                    and tracked_kind == "tracked_source_metric_surfels_at_target_rgb_frame"
                    and generated_consumed is False
                    and tracked_evidence.ndim == 2
                    and tracked_evidence.shape[1] == 3
                    and len(tracked_evidence) >= int(args.min_visible_points)
                    and np.isfinite(tracked_evidence).all()
                ):
                    observed = tracked_evidence
                    observed_evidence_source = "source_metric_surfels_tracked_to_current_object_owned_rgb_by_calibrated_pnp"
                else:
                    raise RuntimeError(f"insufficient visible surfels: {observed.shape}")
            sigma_t, sigma_r = pose_row_sigma(row, radius, args)
            target_tuple = targets.get(idx)
            if target_tuple is None:
                target, weight, target_rows = None, 0.0, 0
            else:
                target, weight, target_rows = target_tuple
            observations.append(
                PoseObservation(
                    frame_idx=idx,
                    source_row={**row, "solver_observed_evidence_source": observed_evidence_source},
                    rotation_world_from_canonical=rot,
                    translation_world_m=trans,
                    translation_sigma_m=sigma_t,
                    rotation_sigma_rad=sigma_r,
                    visible_sample_count=int(row.get("visible_sample_count") or len(observed)),
                    observed_points_world=observed.astype(float),
                    nonpenetration_target_world_m=target,
                    nonpenetration_weight=weight,
                    nonpenetration_source_rows=target_rows,
                )
            )
        except Exception as exc:
            skipped.append({"frame_idx": idx, "reason": str(exc)})
    if not observations:
        raise RuntimeError(f"no usable pose observations; skipped={skipped[:8]}")
    measurement_row_count = sum(
        str(row.get("status") or "") in POSE_MEASUREMENT_STATUSES
        for row in pose_report.get("pose_rows", [])
        if isinstance(row, dict)
    )
    if len(observations) != measurement_row_count:
        raise RuntimeError(
            "P15 fail-closed observation validation rejected one or more declared direct P14 rows; "
            f"accepted={len(observations)} declared={measurement_row_count} skipped={skipped[:8]}"
        )
    if len(observations) < int(args.min_graph_frames) and not args.complete_full_timeline_rigid_pose:
        raise RuntimeError(
            f"only {len(observations)} usable pose observations below min_graph_frames={args.min_graph_frames}; "
            f"uncertain full-timeline completion is disabled; skipped={skipped[:8]}"
        )
    observations.sort(key=lambda obs: obs.frame_idx)
    return observations, skipped, targets


def unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    arr = x.reshape(n, 6)
    return arr[:, :3], arr[:, 3:6]


def image_factors_active(args: argparse.Namespace, image_factors: dict[int, ImageFactorObservation] | None) -> bool:
    return bool(
        image_factors
        and (
            float(args.image_first_hit_weight) > 0.0
            or float(args.image_silhouette_weight) > 0.0
        )
    )


def image_factor_blocks(
    obs: PoseObservation,
    factor: ImageFactorObservation | None,
    rot_delta_i: np.ndarray,
    trans_delta_i: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    if factor is None:
        return []
    r, t = corrected_pose(obs, rot_delta_i, trans_delta_i)
    blocks: list[np.ndarray] = []
    if len(factor.depth_observed_z) and float(args.image_first_hit_weight) > 0.0:
        world = apply_pose(factor.depth_points_canonical, r, t)
        camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
        depth = np.clip(
            camera[:, 2] - factor.depth_observed_z,
            -float(args.max_image_first_hit_residual_m),
            float(args.max_image_first_hit_residual_m),
        ) / float(args.sigma_image_first_hit_m)
        blocks.append(
            math.sqrt(float(args.image_first_hit_weight))
            * depth
            / math.sqrt(max(1, len(depth)))
        )
    if len(factor.silhouette_kind) and float(args.image_silhouette_weight) > 0.0:
        world = apply_pose(factor.silhouette_points_canonical, r, t)
        camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
        z = np.maximum(camera[:, 2], 1.0e-9)
        uv = np.column_stack((
            factor.K_raster[0, 0] * camera[:, 0] / z + factor.K_raster[0, 2],
            factor.K_raster[1, 1] * camera[:, 1] / z + factor.K_raster[1, 2],
        ))
        diff = uv - factor.silhouette_target_uv
        norms = np.linalg.norm(diff, axis=1)
        clip_scale = np.minimum(
            1.0,
            float(args.max_image_silhouette_residual_px) / np.maximum(norms, 1.0e-9),
        )
        silhouette = (diff * clip_scale[:, None]) / float(args.sigma_image_silhouette_px)
        blocks.append(
            math.sqrt(float(args.image_silhouette_weight))
            * silhouette.reshape(-1)
            / math.sqrt(max(1, len(factor.silhouette_kind)))
        )
    return blocks


def residual_vector(
    x: np.ndarray,
    observations: list[PoseObservation],
    args: argparse.Namespace,
    image_factors: dict[int, ImageFactorObservation] | None = None,
) -> np.ndarray:
    rot_delta, trans_delta = unpack(x, len(observations))
    residuals: list[np.ndarray] = []
    for i, obs in enumerate(observations):
        residuals.append(trans_delta[i] / obs.translation_sigma_m)
        residuals.append(rot_delta[i] / obs.rotation_sigma_rad)
        if obs.nonpenetration_target_world_m is not None and obs.nonpenetration_weight > 0.0:
            residuals.append(
                math.sqrt(obs.nonpenetration_weight)
                * (trans_delta[i] - obs.nonpenetration_target_world_m)
                / float(args.sigma_nonpenetration_target_m)
            )
        if image_factors:
            residuals.extend(image_factor_blocks(obs, image_factors.get(obs.frame_idx), rot_delta[i], trans_delta[i], args))
    # Smooth the correction field, not the physical object trajectory, so real object motion measured by ICP is preserved.
    for i in range(1, len(observations)):
        gap = max(1, observations[i].frame_idx - observations[i - 1].frame_idx)
        scale = math.sqrt(float(gap))
        residuals.append((trans_delta[i] - trans_delta[i - 1]) / (float(args.sigma_translation_delta_step_m) * scale))
        residuals.append((rot_delta[i] - rot_delta[i - 1]) / (float(args.sigma_rotation_delta_step_rad) * scale))
    for i in range(1, len(observations) - 1):
        gap0 = max(1, observations[i].frame_idx - observations[i - 1].frame_idx)
        gap1 = max(1, observations[i + 1].frame_idx - observations[i].frame_idx)
        # Use a gap-scaled second difference of the correction field as a soft acceleration prior.
        prev_v_t = (trans_delta[i] - trans_delta[i - 1]) / float(gap0)
        next_v_t = (trans_delta[i + 1] - trans_delta[i]) / float(gap1)
        prev_v_r = (rot_delta[i] - rot_delta[i - 1]) / float(gap0)
        next_v_r = (rot_delta[i + 1] - rot_delta[i]) / float(gap1)
        scale = math.sqrt(float(max(gap0, gap1)))
        residuals.append((next_v_t - prev_v_t) / (float(args.sigma_translation_delta_accel_m) * scale))
        residuals.append((next_v_r - prev_v_r) / (float(args.sigma_rotation_delta_accel_rad) * scale))
    return np.concatenate([r.reshape(-1) for r in residuals]).astype(float)


def residual_sparsity(
    observations: list[PoseObservation],
    image_factors: dict[int, ImageFactorObservation] | None = None,
    args: argparse.Namespace | None = None,
) -> sparse.csr_matrix:
    n = len(observations)
    cols = n * 6
    entries: list[tuple[int, int]] = []
    row = 0

    def add(rows: range, frame_ids: list[int]) -> None:
        for rr in rows:
            for fi in frame_ids:
                for cc in range(fi * 6, fi * 6 + 6):
                    entries.append((rr, cc))

    for i, obs in enumerate(observations):
        add(range(row, row + 3), [i])
        row += 3
        add(range(row, row + 3), [i])
        row += 3
        if obs.nonpenetration_target_world_m is not None and obs.nonpenetration_weight > 0.0:
            add(range(row, row + 3), [i])
            row += 3
        if image_factors and args is not None:
            factor = image_factors.get(obs.frame_idx)
            if factor is not None:
                depth_rows = len(factor.depth_observed_z) if float(args.image_first_hit_weight) > 0.0 else 0
                silhouette_rows = (
                    2 * len(factor.silhouette_kind)
                    if float(args.image_silhouette_weight) > 0.0
                    else 0
                )
                if depth_rows:
                    add(range(row, row + depth_rows), [i])
                    row += depth_rows
                if silhouette_rows:
                    add(range(row, row + silhouette_rows), [i])
                    row += silhouette_rows
    for i in range(1, n):
        add(range(row, row + 3), [i - 1, i])
        row += 3
        add(range(row, row + 3), [i - 1, i])
        row += 3
    for i in range(1, n - 1):
        add(range(row, row + 3), [i - 1, i, i + 1])
        row += 3
        add(range(row, row + 3), [i - 1, i, i + 1])
        row += 3
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, cols))


def image_factor_metrics(
    x: np.ndarray,
    observations: list[PoseObservation],
    image_factors: dict[int, ImageFactorObservation],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not image_factors_active(args, image_factors):
        return {"enabled": False}
    rot_delta, trans_delta = unpack(x, len(observations))
    depth_errors: list[np.ndarray] = []
    outside_errors: list[np.ndarray] = []
    missing_errors: list[np.ndarray] = []
    normalized_blocks: list[np.ndarray] = []
    per_frame: dict[str, Any] = {}
    depth_count = 0
    silhouette_count = 0
    for i, obs in enumerate(observations):
        factor = image_factors.get(obs.frame_idx)
        if factor is None:
            continue
        r, t = corrected_pose(obs, rot_delta[i], trans_delta[i])
        frame_depth: np.ndarray | None = None
        frame_outside: np.ndarray | None = None
        frame_missing: np.ndarray | None = None
        if len(factor.depth_observed_z):
            world = apply_pose(factor.depth_points_canonical, r, t)
            camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
            frame_depth = camera[:, 2] - factor.depth_observed_z
            depth_errors.append(frame_depth)
            depth_count += len(frame_depth)
        if len(factor.silhouette_kind):
            world = apply_pose(factor.silhouette_points_canonical, r, t)
            camera = (world - factor.T_world_camera[:3, 3]) @ factor.T_world_camera[:3, :3]
            z = np.maximum(camera[:, 2], 1.0e-9)
            uv = np.column_stack((
                factor.K_raster[0, 0] * camera[:, 0] / z + factor.K_raster[0, 2],
                factor.K_raster[1, 1] * camera[:, 1] / z + factor.K_raster[1, 2],
            ))
            norms = np.linalg.norm(uv - factor.silhouette_target_uv, axis=1)
            outside = norms[factor.silhouette_kind == 0]
            missing = norms[factor.silhouette_kind == 1]
            if len(outside):
                outside_errors.append(outside)
                frame_outside = outside
            if len(missing):
                missing_errors.append(missing)
                frame_missing = missing
            silhouette_count += len(norms)
        blocks = image_factor_blocks(obs, factor, rot_delta[i], trans_delta[i], args)
        if blocks:
            normalized_blocks.append(np.concatenate([block.reshape(-1) for block in blocks]))
        per_frame[str(obs.frame_idx)] = {
            "first_hit_depth_error_m": numeric_summary(frame_depth if frame_depth is not None else []),
            "outside_silhouette_error_px": numeric_summary(frame_outside if frame_outside is not None else []),
            "missing_silhouette_error_px": numeric_summary(frame_missing if frame_missing is not None else []),
        }
    normalized = np.concatenate(normalized_blocks) if normalized_blocks else np.empty(0, dtype=float)
    return {
        "enabled": True,
        "frame_count": int(len(per_frame)),
        "depth_factor_count": int(depth_count),
        "silhouette_factor_count": int(silhouette_count),
        "image_first_hit_weight": float(args.image_first_hit_weight),
        "image_silhouette_weight": float(args.image_silhouette_weight),
        "first_hit_depth_error_m": numeric_summary(np.concatenate(depth_errors) if depth_errors else []),
        "outside_silhouette_error_px": numeric_summary(np.concatenate(outside_errors) if outside_errors else []),
        "missing_silhouette_error_px": numeric_summary(np.concatenate(missing_errors) if missing_errors else []),
        "normalized_image_factor_rms": float(np.sqrt(np.mean(normalized * normalized))) if len(normalized) else None,
        "per_frame": per_frame,
    }


def target_residual_summary(x: np.ndarray, observations: list[PoseObservation]) -> dict[str, Any]:
    _, trans_delta = unpack(x, len(observations))
    vals = []
    norms = []
    for i, obs in enumerate(observations):
        if obs.nonpenetration_target_world_m is None or obs.nonpenetration_weight <= 0.0:
            continue
        diff = trans_delta[i] - obs.nonpenetration_target_world_m
        vals.append(float(np.linalg.norm(diff)))
        norms.append(float(np.linalg.norm(obs.nonpenetration_target_world_m)))
    return {"target_residual_norm_m": numeric_summary(vals), "target_norm_m": numeric_summary(norms), "target_frame_count": len(vals)}


def correction_summary(x: np.ndarray, observations: list[PoseObservation]) -> dict[str, Any]:
    rot_delta, trans_delta = unpack(x, len(observations))
    gaps = np.asarray([observations[i].frame_idx - observations[i - 1].frame_idx for i in range(1, len(observations))], dtype=float)
    trans_norm = np.linalg.norm(trans_delta, axis=1)
    rot_norm = np.linalg.norm(rot_delta, axis=1)
    step_t = np.linalg.norm(np.diff(trans_delta, axis=0), axis=1) if len(observations) > 1 else np.asarray([])
    step_r = np.linalg.norm(np.diff(rot_delta, axis=0), axis=1) if len(observations) > 1 else np.asarray([])
    return {
        "translation_delta_norm_m": numeric_summary(trans_norm),
        "rotation_delta_norm_rad": numeric_summary(rot_norm),
        "translation_delta_step_norm_m": numeric_summary(step_t),
        "rotation_delta_step_norm_rad": numeric_summary(step_r),
        "visible_frame_gap": numeric_summary(gaps),
    }


def rotation_observability_summary(
    observations: list[PoseObservation],
    minimum_score: float,
) -> dict[str, Any]:
    rows = []
    for observation in observations:
        centered = observation.observed_points_world - observation.observed_points_world.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        largest = max(float(eigenvalues[0]), 1.0e-12)
        normalized = eigenvalues / largest
        # Distinct adjacent principal spreads are a conservative proxy for
        # whether a one-sided rigid fit can observe all three rotation axes.
        score = float(min(normalized[0] - normalized[1], normalized[1] - normalized[2]))
        rows.append({
            "frame_idx": int(observation.frame_idx),
            "normalized_covariance_eigenvalues": normalized.astype(float).tolist(),
            "rotation_observability_score": score,
            "observable": bool(score >= float(minimum_score)),
        })
    observable_count = int(sum(row["observable"] for row in rows))
    return {
        "minimum_score": float(minimum_score),
        "observable_frame_count": observable_count,
        "observation_frame_count": int(len(rows)),
        "observable_fraction": float(observable_count / max(1, len(rows))),
        "score_summary": numeric_summary([row["rotation_observability_score"] for row in rows]),
        "rows": rows,
        "interpretation": "Conservative covariance-eigenvalue proxy; symmetric, line-like, or near-planar-isotropic visible support cannot establish all rotation axes.",
    }


def sparse_conditional_rotation_tail_decision(
    *,
    step_rows: list[dict[str, Any]],
    accepted_rows_by_idx: dict[int, dict[str, Any]],
    observability: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bound a tiny underobservable 15--18 degree tail without changing poses.

    The strict 15 degree tier remains the default. The conditional tier can
    only admit adjacent direct metric-surface rows, never interpolated/held/RGB
    or generated-geometry rows, and every excess transition must be sparse,
    translation-continuous, and weakly or marginally rotation-observable.
    """
    strict_limit = float(args.max_rotation_step_deg)
    enabled = bool(getattr(args, "allow_sparse_conditional_rotation_tail", False))
    conditional_limit = float(getattr(args, "conditional_max_rotation_step_deg", 18.0))
    maximum_count = int(getattr(args, "conditional_max_rotation_step_count", 2))
    maximum_fraction = float(getattr(args, "conditional_max_rotation_step_fraction", 0.015))
    maximum_translation_m = float(
        getattr(args, "conditional_max_rotation_step_translation_m", 0.020)
    )
    maximum_endpoint_observability = float(
        getattr(args, "conditional_max_endpoint_rotation_observability_score", 0.030)
    )
    numeric_policy = {
        "strict_max_rotation_step_deg": strict_limit,
        "conditional_max_rotation_step_deg": conditional_limit,
        "conditional_max_rotation_step_fraction": maximum_fraction,
        "conditional_max_translation_step_m": maximum_translation_m,
        "conditional_max_endpoint_observability_score": maximum_endpoint_observability,
    }
    if not all(math.isfinite(value) for value in numeric_policy.values()):
        raise RuntimeError(f"non-finite conditional rotation-tail policy: {numeric_policy}")
    if enabled and not math.isclose(strict_limit, STRICT_ROTATION_STEP_DEG, abs_tol=1.0e-12):
        raise RuntimeError(
            f"conditional rotation tail requires the fixed {STRICT_ROTATION_STEP_DEG:g}-degree strict tier"
        )
    if conditional_limit < strict_limit:
        raise RuntimeError("conditional rotation-step limit must not be below the strict limit")
    if (
        maximum_count < 0
        or not (0.0 <= maximum_fraction <= 1.0)
        or maximum_translation_m < 0.0
        or maximum_endpoint_observability < 0.0
    ):
        raise RuntimeError("invalid conditional rotation-tail bounds")
    configured = {
        "max_rotation_step_deg": conditional_limit,
        "max_count": maximum_count,
        "max_fraction": maximum_fraction,
        "max_translation_step_m": maximum_translation_m,
        "max_endpoint_observability_score": maximum_endpoint_observability,
    }
    broadened = {
        key: value
        for key, value in configured.items()
        if value > CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS[key] + 1.0e-12
    }
    if broadened:
        raise RuntimeError(
            "conditional rotation-tail CLI exceeds immutable policy ceilings: "
            f"configured={broadened}, ceilings={CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS}"
        )

    observability_by_idx = {
        int(row["frame_idx"]): row
        for row in observability.get("rows", [])
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    excess = [
        row for row in step_rows
        if float(row.get("rotation_step_deg") or 0.0) > strict_limit
    ]
    evaluated = []
    for step in excess:
        from_idx = int(step["from_frame_idx"])
        to_idx = int(step["to_frame_idx"])
        previous = accepted_rows_by_idx.get(from_idx, {})
        current = accepted_rows_by_idx.get(to_idx, {})
        previous_score = observability_by_idx.get(from_idx, {}).get(
            "rotation_observability_score"
        )
        current_score = observability_by_idx.get(to_idx, {}).get(
            "rotation_observability_score"
        )
        endpoint_scores_available = bool(
            isinstance(previous_score, (int, float))
            and isinstance(current_score, (int, float))
            and np.isfinite([previous_score, current_score]).all()
        )
        direct_metric_rows = bool(
            previous.get("status") == CORRECTED_POSE_STATUS
            and current.get("status") == CORRECTED_POSE_STATUS
            and previous.get("direct_pose_observation_source")
            == "adjacent_observed_metric_surfel_registration"
            and current.get("direct_pose_observation_source")
            == "adjacent_observed_metric_surfel_registration"
            and previous.get("rigid_pose_observation_eligible") is True
            and current.get("rigid_pose_observation_eligible") is True
        )
        generated_geometry_excluded = bool(
            previous.get("generated_geometry_pose_evidence_consumed") is False
            and current.get("generated_geometry_pose_evidence_consumed") is False
        )
        criteria = {
            "adjacent_frame_step": int(step.get("frame_gap", -1)) == 1,
            "within_conditional_rotation_limit": float(step["rotation_step_deg"])
            <= conditional_limit,
            "within_conditional_translation_limit": float(step["translation_step_m"])
            <= maximum_translation_m,
            "direct_adjacent_metric_surface_rows": direct_metric_rows,
            "generated_geometry_pose_evidence_excluded": generated_geometry_excluded,
            "endpoint_observability_available": endpoint_scores_available,
            "endpoints_weak_or_marginally_observable": bool(
                endpoint_scores_available
                and float(previous_score) <= maximum_endpoint_observability
                and float(current_score) <= maximum_endpoint_observability
            ),
        }
        evaluated.append({
            **step,
            "from_rotation_observability_score": previous_score,
            "to_rotation_observability_score": current_score,
            "criteria": criteria,
            "transition_eligible": bool(all(criteria.values())),
        })

    excess_fraction = float(len(excess) / max(1, len(step_rows)))
    population_criteria = {
        "conditional_tier_enabled": enabled,
        "has_strict_exceedance": bool(excess),
        "exceedance_count_within_limit": len(excess) <= maximum_count,
        "exceedance_fraction_within_limit": excess_fraction <= maximum_fraction,
        "all_exceedances_individually_eligible": bool(
            excess and all(row["transition_eligible"] for row in evaluated)
        ),
    }
    accepted = bool(all(population_criteria.values()))
    return {
        "strict_max_rotation_step_deg": strict_limit,
        "strict_gate_passed": not excess,
        "conditional_tier_enabled": enabled,
        "conditional_max_rotation_step_deg": conditional_limit,
        "conditional_max_rotation_step_count": maximum_count,
        "conditional_max_rotation_step_fraction": maximum_fraction,
        "conditional_max_translation_step_m": maximum_translation_m,
        "conditional_max_endpoint_rotation_observability_score": maximum_endpoint_observability,
        "immutable_policy_ceilings": dict(CONDITIONAL_ROTATION_STEP_POLICY_CEILINGS),
        "strict_exceedance_count": int(len(excess)),
        "strict_exceedance_fraction": excess_fraction,
        "conditional_tier_applied": accepted,
        "gate_passed": bool(not excess or accepted),
        "acceptance_mode": (
            "strict_max_rotation_step"
            if not excess
            else "conditional_sparse_underobservable_rotation_tail"
            if accepted
            else "failed_rotation_step_gate"
        ),
        "population_criteria": population_criteria,
        "conditional_transitions": evaluated,
        "trajectory_values_modified_or_clipped": False,
        "generated_geometry_pose_evidence_consumed": False,
        "uncertainty": (
            "A sparse direct observed-metric rotation tail exceeds the strict 15-degree tier "
            "but remains at or below the bounded 18-degree conditional tier. The original "
            "trajectory is preserved without clipping; these transitions are explicitly low confidence."
            if accepted else None
        ),
    }


def temporal_readiness_diagnostics(
    *,
    pose_rows: list[dict[str, Any]],
    annotations: dict[str, Any],
    observations: list[PoseObservation],
    full_timeline_completion: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    timeline = sorted(
        int(frame["frame_idx"])
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict)
        and frame.get("frame_idx") is not None
        and (args.frame_start is None or int(frame["frame_idx"]) >= int(args.frame_start))
        and (args.frame_end is None or int(frame["frame_idx"]) <= int(args.frame_end))
    )
    timeline_set = set(timeline)
    accepted = []
    for row in pose_rows:
        idx = int(row.get("frame_idx", -1))
        if idx not in timeline_set or row.get("status") not in {CORRECTED_POSE_STATUS, COMPLETED_POSE_STATUS}:
            continue
        rotation = np.asarray(row.get("rotation_world_from_completed_canonical_matrix"), dtype=np.float64)
        translation = np.asarray(row.get("translation_world_m"), dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            continue
        accepted.append((idx, rotation, translation, row))
    accepted.sort(key=lambda value: value[0])
    direct_frames = sorted(int(observation.frame_idx) for observation in observations if observation.frame_idx in timeline_set)
    direct_fraction = float(len(direct_frames) / max(1, len(timeline)))
    internal_gaps = [direct_frames[i] - direct_frames[i - 1] for i in range(1, len(direct_frames))]
    max_direct_gap = max(internal_gaps) if internal_gaps else (0 if direct_frames else None)
    leading_gap = direct_frames[0] - timeline[0] if direct_frames and timeline else None
    trailing_gap = timeline[-1] - direct_frames[-1] if direct_frames and timeline else None

    rotation_steps_deg = []
    translation_steps_m = []
    step_rows = []
    for previous, current in zip(accepted[:-1], accepted[1:]):
        prev_idx, prev_rotation, prev_translation, _ = previous
        idx, rotation, translation, _ = current
        relative = rotation @ prev_rotation.T
        rotation_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
        translation_m = float(np.linalg.norm(translation - prev_translation))
        rotation_steps_deg.append(rotation_deg)
        translation_steps_m.append(translation_m)
        step_rows.append({
            "from_frame_idx": int(prev_idx),
            "to_frame_idx": int(idx),
            "frame_gap": int(idx - prev_idx),
            "rotation_step_deg": rotation_deg,
            "translation_step_m": translation_m,
        })

    mode_counts = full_timeline_completion.get("mode_counts") if isinstance(full_timeline_completion.get("mode_counts"), dict) else {}
    completed_count = int(full_timeline_completion.get("completed_row_count") or 0)
    hold_count = int(mode_counts.get("nearest_visible_pose_hold") or 0)
    completed_fraction = float(completed_count / max(1, len(timeline)))
    hold_fraction = float(hold_count / max(1, len(timeline)))
    observability = rotation_observability_summary(observations, float(args.min_rotation_observability_score))
    accepted_rows_by_idx = {idx: row for idx, _rotation, _translation, row in accepted}
    rotation_step_gate = sparse_conditional_rotation_tail_decision(
        step_rows=step_rows,
        accepted_rows_by_idx=accepted_rows_by_idx,
        observability=observability,
        args=args,
    )

    max_rotation_step = max(rotation_steps_deg) if rotation_steps_deg else 0.0
    max_translation_step = max(translation_steps_m) if translation_steps_m else 0.0
    failure_reasons = []
    if len(accepted) != len(timeline):
        failure_reasons.append("incomplete_full_timeline_pose_rows")
    if direct_fraction < float(args.min_direct_pose_fraction):
        failure_reasons.append("insufficient_direct_pose_fraction")
    if max_direct_gap is None or max_direct_gap > int(args.max_direct_pose_gap_frames):
        failure_reasons.append("direct_pose_gap_too_large")
    if leading_gap is None or leading_gap > int(args.max_rigid_pose_extrapolation_gap_frames):
        failure_reasons.append("leading_pose_extrapolation_too_large")
    if trailing_gap is None or trailing_gap > int(args.max_rigid_pose_extrapolation_gap_frames):
        failure_reasons.append("trailing_pose_extrapolation_too_large")
    if completed_fraction > float(args.max_completed_pose_fraction):
        failure_reasons.append("completed_pose_fraction_too_large")
    if hold_fraction > float(args.max_nearest_hold_fraction):
        failure_reasons.append("nearest_hold_fraction_too_large")
    if not rotation_step_gate["gate_passed"]:
        failure_reasons.append("rotation_step_jump")
    if max_translation_step > float(args.max_translation_step_m):
        failure_reasons.append("translation_step_jump")
    if observability["observable_fraction"] < float(args.min_rotation_observable_fraction):
        failure_reasons.append("insufficient_rotation_observability")
    return {
        "ready": not failure_reasons,
        "failure_reasons": failure_reasons,
        "timeline_frame_count": int(len(timeline)),
        "accepted_full_timeline_pose_count": int(len(accepted)),
        "direct_pose_count": int(len(direct_frames)),
        "direct_pose_fraction": direct_fraction,
        "direct_frame_span": [direct_frames[0], direct_frames[-1]] if direct_frames else None,
        "max_direct_frame_gap": max_direct_gap,
        "leading_direct_frame_gap": leading_gap,
        "trailing_direct_frame_gap": trailing_gap,
        "completed_pose_count": completed_count,
        "completed_pose_fraction": completed_fraction,
        "nearest_hold_count": hold_count,
        "nearest_hold_fraction": hold_fraction,
        "completion_mode_counts": mode_counts,
        "rotation_step_deg": numeric_summary(rotation_steps_deg),
        "translation_step_m": numeric_summary(translation_steps_m),
        "max_rotation_step_deg": max_rotation_step,
        "max_translation_step_m": max_translation_step,
        "rotation_observability": observability,
        "rotation_step_gate": rotation_step_gate,
        "conditional_temporal_uncertainty": rotation_step_gate.get("uncertainty"),
        "thresholds": {
            "min_direct_pose_fraction": float(args.min_direct_pose_fraction),
            "max_direct_pose_gap_frames": int(args.max_direct_pose_gap_frames),
            "max_completed_pose_fraction": float(args.max_completed_pose_fraction),
            "max_nearest_hold_fraction": float(args.max_nearest_hold_fraction),
            "max_rotation_step_deg": float(args.max_rotation_step_deg),
            "allow_sparse_conditional_rotation_tail": bool(
                getattr(args, "allow_sparse_conditional_rotation_tail", False)
            ),
            "conditional_max_rotation_step_deg": float(
                getattr(args, "conditional_max_rotation_step_deg", 18.0)
            ),
            "conditional_max_rotation_step_count": int(
                getattr(args, "conditional_max_rotation_step_count", 2)
            ),
            "conditional_max_rotation_step_fraction": float(
                getattr(args, "conditional_max_rotation_step_fraction", 0.015)
            ),
            "conditional_max_rotation_step_translation_m": float(
                getattr(args, "conditional_max_rotation_step_translation_m", 0.020)
            ),
            "conditional_max_endpoint_rotation_observability_score": float(
                getattr(args, "conditional_max_endpoint_rotation_observability_score", 0.030)
            ),
            "max_translation_step_m": float(args.max_translation_step_m),
            "min_rotation_observable_fraction": float(args.min_rotation_observable_fraction),
            "min_rotation_observability_score": float(args.min_rotation_observability_score),
        },
        "step_rows": step_rows,
    }


def surface_metrics(observations: list[PoseObservation], mesh_samples: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    rot_delta, trans_delta = unpack(x, len(observations))
    obs_to_mesh_medians: list[float] = []
    mesh_to_obs_medians: list[float] = []
    per_frame: dict[str, Any] = {}
    for i, obs in enumerate(observations):
        r, t = corrected_pose(obs, rot_delta[i], trans_delta[i])
        mesh_world = apply_pose(mesh_samples, r, t)
        o2m = nearest_summary(obs.observed_points_world, mesh_world)
        m2o = nearest_summary(mesh_world, obs.observed_points_world)
        if o2m["median_m"] is not None:
            obs_to_mesh_medians.append(float(o2m["median_m"]))
        if m2o["median_m"] is not None:
            mesh_to_obs_medians.append(float(m2o["median_m"]))
        per_frame[str(obs.frame_idx)] = {"observed_to_mesh": o2m, "mesh_to_observed": m2o}
    return {
        "observed_to_mesh_median_m": numeric_summary(obs_to_mesh_medians),
        "mesh_to_observed_median_m": numeric_summary(mesh_to_obs_medians),
        "per_frame": per_frame,
    }


def build_pose_rows(
    original_pose_rows: list[dict[str, Any]],
    observations: list[PoseObservation],
    x: np.ndarray,
    args: argparse.Namespace,
    graph_support_sufficient: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs_by_idx = {obs.frame_idx: i for i, obs in enumerate(observations)}
    rot_delta, trans_delta = unpack(x, len(observations))
    corrected_by_idx: dict[int, tuple[np.ndarray, np.ndarray, PoseObservation, np.ndarray, np.ndarray]] = {}
    out: list[dict[str, Any]] = []
    for row in original_pose_rows:
        idx = int(row.get("frame_idx", -1))
        if idx not in obs_by_idx:
            out.append(dict(row))
            continue
        i = obs_by_idx[idx]
        obs = observations[i]
        r, t = corrected_pose(obs, rot_delta[i], trans_delta[i])
        corrected_by_idx[idx] = (r, t, obs, rot_delta[i], trans_delta[i])
        new_row = dict(row)
        new_row["status"] = CORRECTED_POSE_STATUS
        new_row["pose_measurement_status"] = row.get("status")
        new_row["rotation_world_from_completed_canonical_matrix"] = r.astype(float).tolist()
        new_row["translation_world_m"] = t.astype(float).tolist()
        new_row["graph_support_sufficient"] = graph_support_sufficient
        new_row["temporal_pose_graph"] = {
            "pose_source": "direct_visible_pose_observation_corrected",
            "direct_visible_measurement": True,
            "rotation_delta_rotvec_rad": rot_delta[i].astype(float).tolist(),
            "translation_delta_world_m": trans_delta[i].astype(float).tolist(),
            "translation_prior_sigma_m": float(obs.translation_sigma_m),
            "rotation_prior_sigma_rad": float(obs.rotation_sigma_rad),
            "nonpenetration_target_world_m": None if obs.nonpenetration_target_world_m is None else obs.nonpenetration_target_world_m.astype(float).tolist(),
            "nonpenetration_weight": float(obs.nonpenetration_weight),
            "nonpenetration_source_rows": int(obs.nonpenetration_source_rows),
            "graph_support_sufficient": graph_support_sufficient,
            "uncertainty": None if graph_support_sufficient else "direct visible observation exists, but trusted graph-frame count is below the configured support minimum",
        }
        out.append(new_row)

    summary: dict[str, Any] = {
        "enabled": bool(args.complete_full_timeline_rigid_pose),
        "graph_support_sufficient": graph_support_sufficient,
        "completed_row_count": 0,
        "direct_row_count": int(len(corrected_by_idx)),
        "mode_counts": {},
        "max_interpolation_gap_frames": int(args.max_rigid_pose_interpolation_gap_frames),
        "max_extrapolation_gap_frames": int(args.max_rigid_pose_extrapolation_gap_frames),
    }
    if not args.complete_full_timeline_rigid_pose or not corrected_by_idx:
        return out, summary

    key_frames = sorted(corrected_by_idx)
    key_rots = Rotation.from_matrix([corrected_by_idx[idx][0] for idx in key_frames])
    slerp = Slerp(np.asarray(key_frames, dtype=float), key_rots) if len(key_frames) >= 2 else None
    key_trans = np.asarray([corrected_by_idx[idx][1] for idx in key_frames], dtype=float)

    def add_mode(mode: str) -> None:
        counts = summary.setdefault("mode_counts", {})
        counts[mode] = int(counts.get(mode, 0)) + 1

    for row in out:
        idx = int(row.get("frame_idx", -1))
        if idx in corrected_by_idx or idx < 0:
            continue
        before = [f for f in key_frames if f < idx]
        after = [f for f in key_frames if f > idx]
        mode: str | None = None
        r_fill: np.ndarray | None = None
        t_fill: np.ndarray | None = None
        gap_frames: int | None = None
        bracket: list[int] | None = None
        if before and after:
            lo = before[-1]
            hi = after[0]
            gap_frames = int(hi - lo)
            if gap_frames <= int(args.max_rigid_pose_interpolation_gap_frames) and slerp is not None:
                alpha = float(idx - lo) / float(max(1, hi - lo))
                r_fill = slerp([float(idx)]).as_matrix()[0]
                t_lo = key_trans[key_frames.index(lo)]
                t_hi = key_trans[key_frames.index(hi)]
                t_fill = (1.0 - alpha) * t_lo + alpha * t_hi
                mode = "interpolated_between_visible_pose_observations"
                bracket = [int(lo), int(hi)]
        if mode is None:
            nearest = min(key_frames, key=lambda f: abs(f - idx))
            dist = abs(nearest - idx)
            if dist <= int(args.max_rigid_pose_extrapolation_gap_frames):
                r_fill = corrected_by_idx[nearest][0]
                t_fill = corrected_by_idx[nearest][1]
                gap_frames = int(dist)
                mode = "nearest_visible_pose_hold"
                bracket = [int(nearest)]
        if mode is None or r_fill is None or t_fill is None:
            continue
        old_status = row.get("status")
        row["status"] = COMPLETED_POSE_STATUS
        row["pose_measurement_status"] = old_status
        row["rotation_world_from_completed_canonical_matrix"] = r_fill.astype(float).tolist()
        row["translation_world_m"] = t_fill.astype(float).tolist()
        row["graph_support_sufficient"] = graph_support_sufficient
        row["temporal_pose_graph"] = {
            "pose_source": mode,
            "bracket_visible_pose_frames": bracket,
            "gap_frames": gap_frames,
            "direct_visible_measurement": False,
            "graph_support_sufficient": graph_support_sufficient,
            "uncertainty": (
                "rigid-body temporal completion; not a direct object mask/depth observation"
                if graph_support_sufficient
                else "insufficient trusted graph support; sparse-cluster interpolation/nearest hold is an unresolved trajectory hypothesis"
            ),
        }
        summary["completed_row_count"] = int(summary["completed_row_count"]) + 1
        add_mode(mode)
    return out, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    pose_measurement_candidates = [
        row
        for row in (pose_report.get("pose_rows", []) if isinstance(pose_report.get("pose_rows"), list) else [])
        if isinstance(row, dict)
        and str(row.get("status") or "") in POSE_MEASUREMENT_STATUSES
        and (args.frame_start is None or int(row.get("frame_idx", -1)) >= int(args.frame_start))
        and (args.frame_end is None or int(row.get("frame_idx", -1)) <= int(args.frame_end))
    ]
    completion = load_json(args.completion_report) if args.completion_report else {}
    if args.completion_report:
        pose_binding = pose_report.get("selected_anchor_atomic_binding")
        completion_binding = (
            (completion.get("inputs") or {}).get("selected_anchor_atomic_binding")
            if isinstance(completion.get("inputs"), dict)
            else None
        )
        pose_evidence_sha256 = str(pose_report.get("selected_anchor_evidence_report_sha256") or "")
        completion_evidence_sha256 = str(
            (completion.get("inputs") or {}).get("candidate_evidence_report_sha256") or ""
        )
        if (
            not isinstance(pose_binding, dict)
            or pose_binding.get("validated") is not True
            or pose_binding != completion_binding
            or not pose_evidence_sha256
            or pose_evidence_sha256 != completion_evidence_sha256
        ):
            raise RuntimeError("P15 pose/completion reports do not share one byte-bound atomic selected anchor")
    completion_outputs = completion.get("outputs") if isinstance(completion.get("outputs"), dict) else {}
    completion_geometry_readiness = completion.get("geometry_readiness") if isinstance(completion.get("geometry_readiness"), dict) else {}
    completion_collision_mesh = completion_outputs.get("collision_eligible_mesh_labeled")
    completion_pose_mesh = completion_outputs.get("pose_hypothesis_mesh_labeled") or completion_outputs.get("completed_mesh_labeled")
    mesh_path = args.completed_mesh or Path(completion_pose_mesh or "")
    if not mesh_path:
        raise RuntimeError("pose-hypothesis mesh path missing; pass --completed-mesh or --completion-report")
    pose_mesh_semantics = (
        "explicit_cli_mesh"
        if args.completed_mesh is not None
        else "pose_hypothesis_mesh_labeled"
        if completion_outputs.get("pose_hypothesis_mesh_labeled")
        else "legacy_completed_mesh_labeled"
    )
    mesh = load_mesh(Path(mesh_path))
    observations, skipped, targets = build_observations(args, annotations, pose_report, mesh)
    image_factors, image_factor_metadata = load_image_factor_observations(
        args.image_factor_npz,
        observations,
        args,
        Path(mesh_path),
    )
    if args.image_factor_npz is not None and not image_factors_active(args, image_factors):
        raise RuntimeError("--image-factor-npz was provided but both image factor weights are zero")
    graph_support_sufficient = len(observations) >= int(args.min_graph_frames)
    x0 = np.zeros(len(observations) * 6, dtype=float)
    before = residual_vector(x0, observations, args, image_factors)
    jac = residual_sparsity(observations, image_factors, args)
    if jac.shape != (len(before), len(x0)):
        raise RuntimeError(f"sparsity shape {jac.shape} != residual/vector {(len(before), len(x0))}")
    result = least_squares(
        lambda x: residual_vector(x, observations, args, image_factors),
        x0,
        jac_sparsity=jac,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after = residual_vector(result.x, observations, args, image_factors)
    mesh_samples = deterministic_sample_mesh(mesh, int(args.surface_metric_sample_count), int(args.seed) + 73)
    before_surface = surface_metrics(observations, mesh_samples, x0)
    after_surface = surface_metrics(observations, mesh_samples, result.x)
    before_target = target_residual_summary(x0, observations)
    after_target = target_residual_summary(result.x, observations)
    before_image = image_factor_metrics(x0, observations, image_factors, args)
    after_image = image_factor_metrics(result.x, observations, image_factors, args)
    pose_rows, full_timeline_completion = build_pose_rows(
        pose_report.get("pose_rows", []),
        observations,
        result.x,
        args,
        graph_support_sufficient,
    )
    temporal_readiness = temporal_readiness_diagnostics(
        pose_rows=pose_rows,
        annotations=annotations,
        observations=observations,
        full_timeline_completion=full_timeline_completion,
        args=args,
    )
    conditional_transitions = (
        temporal_readiness.get("rotation_step_gate", {}).get("conditional_transitions", [])
        if temporal_readiness.get("rotation_step_gate", {}).get("conditional_tier_applied") is True
        else []
    )
    pose_rows_by_idx = {
        int(row["frame_idx"]): row
        for row in pose_rows
        if isinstance(row, dict) and row.get("frame_idx") is not None
    }
    for transition in conditional_transitions:
        row = pose_rows_by_idx.get(int(transition["to_frame_idx"]))
        if row is not None:
            row["conditional_rotation_step_uncertainty"] = {
                "acceptance_mode": "conditional_sparse_underobservable_rotation_tail",
                "from_frame_idx": int(transition["from_frame_idx"]),
                "rotation_step_deg": float(transition["rotation_step_deg"]),
                "translation_step_m": float(transition["translation_step_m"]),
                "trajectory_values_modified_or_clipped": False,
                "generated_geometry_pose_evidence_consumed": False,
                "interpretation": "Direct observed-metric pose retained unchanged under the bounded sparse rotation-tail tier.",
            }
    correction_is_exact_zero = bool(np.array_equal(result.x, np.zeros_like(result.x)))
    image_terms_active = image_factors_active(args, image_factors)
    no_active_nonpenetration_targets = not any(
        observation.nonpenetration_target_world_m is not None and observation.nonpenetration_weight > 0.0
        for observation in observations
    )
    optimization_effect = {
        "correction_is_exact_zero": correction_is_exact_zero,
        "no_active_nonpenetration_targets": no_active_nonpenetration_targets,
        "image_factors_active": image_terms_active,
        "zero_correction_is_structural_objective_minimum": bool(
            correction_is_exact_zero and no_active_nonpenetration_targets and not image_terms_active
        ),
        "physical_trajectory_smoothed_directly": False,
        "interpretation": (
            "The objective regularizes correction deltas. Optional first-hit/silhouette factors are frozen local image correspondences re-linearized under each per-frame SE(3) correction; without nonzero external targets or image factors, all-zero deltas are the exact minimum. Temporal readiness must therefore be established from direct coverage, observability, and SE(3) jump diagnostics, not optimizer success."
        ),
    }
    surface_before_med = before_surface["observed_to_mesh_median_m"]["median"]
    surface_after_med = after_surface["observed_to_mesh_median_m"]["median"]
    target_before_med = before_target["target_residual_norm_m"]["median"]
    target_after_med = after_target["target_residual_norm_m"]["median"]
    surface_degraded_m = None
    if surface_before_med is not None and surface_after_med is not None:
        surface_degraded_m = float(surface_after_med) - float(surface_before_med)
    target_improved = target_before_med is not None and target_after_med is not None and float(target_after_med) < float(target_before_med)
    before_image_rms = before_image.get("normalized_image_factor_rms") if isinstance(before_image, dict) else None
    after_image_rms = after_image.get("normalized_image_factor_rms") if isinstance(after_image, dict) else None
    image_improved = before_image_rms is not None and after_image_rms is not None and float(after_image_rms) < float(before_image_rms)
    surface_preserved = surface_degraded_m is None or surface_degraded_m <= float(args.max_surface_median_degradation_m)
    if not graph_support_sufficient:
        status = "completed_uncertain_insufficient_trusted_pose_graph_support"
    elif not temporal_readiness["ready"]:
        status = "completed_unready_temporal_coverage_observability_or_se3_jump"
    elif result.success and surface_preserved and image_terms_active:
        status = (
            "corrected_pose_graph_surface_preserved_image_factors_improved"
            if image_improved
            else "corrected_pose_graph_surface_preserved_no_image_factor_gain"
        )
    elif result.success and surface_preserved and target_improved:
        status = "corrected_pose_graph_surface_preserved_nonpenetration_pressure_improved"
    elif result.success and surface_preserved:
        status = "corrected_pose_graph_surface_preserved_no_nonpenetration_gain"
    elif result.success:
        status = "corrected_pose_graph_surface_degraded_untrusted"
    else:
        status = "corrected_pose_graph_optimizer_incomplete"
    annotation_ready = bool(
        result.success
        and surface_preserved
        and graph_support_sufficient
        and temporal_readiness["ready"]
    )
    rotation_acceptance_mode = str(
        temporal_readiness.get("rotation_step_gate", {}).get("acceptance_mode")
        or "unknown_rotation_step_gate"
    )
    annotation_readiness_mode = (
        "conditional_sparse_underobservable_rotation_tail"
        if annotation_ready
        and temporal_readiness.get("rotation_step_gate", {}).get("conditional_tier_applied") is True
        else "strict_temporal_readiness"
        if annotation_ready
        else "not_annotation_ready"
    )
    for row in pose_rows:
        if row.get("status") in {CORRECTED_POSE_STATUS, COMPLETED_POSE_STATUS}:
            row["annotation_ready"] = annotation_ready
            row["annotation_readiness_mode"] = annotation_readiness_mode
    report = {
        "method": "solve_v19_rigid_object_pose_graph",
        "status": status,
        "annotation_ready": annotation_ready,
        "annotation_readiness_mode": annotation_readiness_mode,
        "rotation_step_acceptance_mode": rotation_acceptance_mode,
        "conditional_temporal_uncertainty": temporal_readiness.get("conditional_temporal_uncertainty"),
        "claim_scope": "Temporal rigid-object pose correction over eligible visible-frame SE(3) measurements. Visible-surface ICP rows are pose observations; nonpenetration rows exert only clipped soft pressure. Optional first-hit/silhouette rows are frozen local image correspondences over the observed-only pose-hypothesis mesh, with hand projection treated as unknown support. Annotation readiness additionally requires full timeline coverage, direct-observation density, bounded gaps/holds, conservative rotation observability, and bounded per-frame SE(3) steps. A named sparse conditional rotation tail is explicit uncertainty, not clipping or ground-truth angular velocity. Optimizer success alone is insufficient.",
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completion_report": str(args.completion_report) if args.completion_report else None,
            "completed_mesh": str(mesh_path),
            "pose_mesh_semantics": pose_mesh_semantics,
            "collision_eligible_mesh_not_used_as_pose_body": completion_collision_mesh,
            "completion_geometry_readiness": completion_geometry_readiness,
            "constraint_report": str(args.constraint_report) if args.constraint_report else None,
            "image_factor_npz": str(args.image_factor_npz) if args.image_factor_npz else None,
            "image_factor_metadata": image_factor_metadata,
        },
        "geometry_contract": {
            "pose_hypothesis_mesh": str(mesh_path),
            "pose_mesh_semantics": pose_mesh_semantics,
            "collision_eligible_mesh": completion_collision_mesh,
            "completion_geometry_readiness": completion_geometry_readiness,
            "pose_graph_consumes_collision_surface": False,
            "claim_scope": "P15 estimates temporal SE(3) in the P13 pose-hypothesis canonical frame. Collision/sign readiness is recorded for P16/P15b but does not replace the pose body or turn completion rows into observations.",
        },
        "graph_frames": [obs.frame_idx for obs in observations],
        "graph_frame_count": int(len(observations)),
        "graph_support": {
            "sufficient": graph_support_sufficient,
            "configured_min_graph_frames": int(args.min_graph_frames),
            "actual_graph_frames": int(len(observations)),
            "frame_span": [int(observations[0].frame_idx), int(observations[-1].frame_idx)],
            "continued_only_as_uncertain_full_timeline_completion": bool(not graph_support_sufficient and args.complete_full_timeline_rigid_pose),
        },
        "pose_observation_eligibility_policy": {
            "explicit_false": "included_only_with_override" if args.include_ineligible_rigid_pose_observations else "hard_rejected",
            "missing_field": "allowed_for_legacy_compatibility",
            "include_ineligible_override": bool(args.include_ineligible_rigid_pose_observations),
            "candidate_measurement_row_count": len(pose_measurement_candidates),
            "explicit_eligible_candidate_count": int(
                sum(row.get("rigid_pose_observation_eligible") is True for row in pose_measurement_candidates)
            ),
            "eligibility_unspecified_candidate_count": int(
                sum(not isinstance(row.get("rigid_pose_observation_eligible"), bool) for row in pose_measurement_candidates)
            ),
            "explicit_ineligible_candidate_count": int(
                sum(row.get("rigid_pose_observation_eligible") is False for row in pose_measurement_candidates)
            ),
            "explicit_ineligible_skipped_count": int(
                sum(row.get("reason") == "explicit upstream rigid_pose_observation_eligible=false" for row in skipped)
            ),
            "explicit_ineligible_admitted_count": int(
                sum(obs.source_row.get("rigid_pose_observation_eligible") is False for obs in observations)
            ),
        },
        "skipped_pose_observations": skipped,
        "nonpenetration_target_frame_count": int(sum(1 for obs in observations if obs.nonpenetration_target_world_m is not None)),
        "nonpenetration_target_source_frame_count": int(len(targets)),
        "parameters": {
            "max_nonpenetration_target_m": float(args.max_nonpenetration_target_m),
            "sigma_nonpenetration_target_m": float(args.sigma_nonpenetration_target_m),
            "sigma_translation_delta_step_m": float(args.sigma_translation_delta_step_m),
            "sigma_rotation_delta_step_rad": float(args.sigma_rotation_delta_step_rad),
            "sigma_translation_delta_accel_m": float(args.sigma_translation_delta_accel_m),
            "sigma_rotation_delta_accel_rad": float(args.sigma_rotation_delta_accel_rad),
            "max_surface_median_degradation_m": float(args.max_surface_median_degradation_m),
            "min_graph_frames": int(args.min_graph_frames),
            "nonpenetration_states": list(args.nonpenetration_states),
            "complete_full_timeline_rigid_pose": bool(args.complete_full_timeline_rigid_pose),
            "max_rigid_pose_interpolation_gap_frames": int(args.max_rigid_pose_interpolation_gap_frames),
            "max_rigid_pose_extrapolation_gap_frames": int(args.max_rigid_pose_extrapolation_gap_frames),
            "min_direct_pose_fraction": float(args.min_direct_pose_fraction),
            "max_direct_pose_gap_frames": int(args.max_direct_pose_gap_frames),
            "max_completed_pose_fraction": float(args.max_completed_pose_fraction),
            "max_nearest_hold_fraction": float(args.max_nearest_hold_fraction),
            "max_rotation_step_deg": float(args.max_rotation_step_deg),
            "allow_sparse_conditional_rotation_tail": bool(args.allow_sparse_conditional_rotation_tail),
            "conditional_max_rotation_step_deg": float(args.conditional_max_rotation_step_deg),
            "conditional_max_rotation_step_count": int(args.conditional_max_rotation_step_count),
            "conditional_max_rotation_step_fraction": float(args.conditional_max_rotation_step_fraction),
            "conditional_max_rotation_step_translation_m": float(args.conditional_max_rotation_step_translation_m),
            "conditional_max_endpoint_rotation_observability_score": float(
                args.conditional_max_endpoint_rotation_observability_score
            ),
            "max_translation_step_m": float(args.max_translation_step_m),
            "min_rotation_observable_fraction": float(args.min_rotation_observable_fraction),
            "min_rotation_observability_score": float(args.min_rotation_observability_score),
            "image_first_hit_weight": float(args.image_first_hit_weight),
            "image_silhouette_weight": float(args.image_silhouette_weight),
            "sigma_image_first_hit_m": float(args.sigma_image_first_hit_m),
            "sigma_image_silhouette_px": float(args.sigma_image_silhouette_px),
            "max_image_first_hit_residual_m": float(args.max_image_first_hit_residual_m),
            "max_image_silhouette_residual_px": float(args.max_image_silhouette_residual_px),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "residual_rms_before": float(np.sqrt(np.mean(before * before))),
            "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        },
        "correction_summary": correction_summary(result.x, observations),
        "optimization_effect": optimization_effect,
        "temporal_readiness": temporal_readiness,
        "nonpenetration_target_before": before_target,
        "nonpenetration_target_after": after_target,
        "image_factor_before": before_image,
        "image_factor_after": after_image,
        "image_factor_rms_improvement": (
            float(before_image_rms) - float(after_image_rms)
            if before_image_rms is not None and after_image_rms is not None
            else None
        ),
        "surface_before": before_surface,
        "surface_after": after_surface,
        "surface_observed_to_mesh_median_degradation_m": surface_degraded_m,
        "full_timeline_rigid_pose_completion": full_timeline_completion,
        "pose_rows": pose_rows,
        "outputs": {
            "pose_graph_report": str(args.output_dir / "v19_rigid_object_pose_graph_report.json"),
            "v18_compatible_pose_report": str(args.output_dir / "v18_compact_rigid_object_pose_fit_report.json"),
        },
        "elapsed_s": float(time.time() - started),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "v19_rigid_object_pose_graph_report.json", report)
    # Same payload under the legacy name lets existing V18 render/constraint tools consume corrected rows.
    write_json(args.output_dir / "v18_compact_rigid_object_pose_fit_report.json", report)
    print(json.dumps({k: report[k] for k in ["status", "annotation_ready", "graph_frame_count", "nonpenetration_target_frame_count", "optimizer", "correction_summary", "optimization_effect", "temporal_readiness", "full_timeline_rigid_pose_completion", "nonpenetration_target_before", "nonpenetration_target_after", "image_factor_before", "image_factor_after", "image_factor_rms_improvement", "surface_observed_to_mesh_median_degradation_m"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completion-report", type=Path, default=None)
    p.add_argument("--completed-mesh", type=Path, default=None)
    p.add_argument("--constraint-report", type=Path, default=None)
    p.add_argument("--image-factor-npz", type=Path, default=None)
    p.add_argument("--image-first-hit-weight", type=float, default=1.0)
    p.add_argument("--image-silhouette-weight", type=float, default=1.0)
    p.add_argument("--sigma-image-first-hit-m", type=float, default=0.008)
    p.add_argument("--sigma-image-silhouette-px", type=float, default=4.0)
    p.add_argument("--max-image-first-hit-residual-m", type=float, default=0.030)
    p.add_argument("--max-image-silhouette-residual-px", type=float, default=16.0)
    p.add_argument("--object-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--min-graph-frames", type=int, default=8)
    p.add_argument(
        "--include-ineligible-rigid-pose-observations",
        action="store_true",
        help="Forbidden production flag retained only so historical commands fail closed explicitly",
    )
    p.add_argument("--min-visible-points", type=int, default=20)
    p.add_argument("--min-pose-sigma-m", type=float, default=0.004)
    p.add_argument("--max-pose-sigma-m", type=float, default=0.045)
    p.add_argument("--default-pose-sigma-m", type=float, default=0.018)
    p.add_argument("--min-pose-rotation-sigma-rad", type=float, default=0.035)
    p.add_argument("--max-pose-rotation-sigma-rad", type=float, default=0.35)
    p.add_argument("--max-nonpenetration-target-m", type=float, default=0.015)
    p.add_argument("--sigma-nonpenetration-target-m", type=float, default=0.030)
    p.add_argument("--nonpenetration-states", nargs="+", default=["candidate_coordinate_correction_visible_2d_compatible"])
    p.add_argument("--sigma-translation-delta-step-m", type=float, default=0.010)
    p.add_argument("--sigma-rotation-delta-step-rad", type=float, default=0.080)
    p.add_argument("--sigma-translation-delta-accel-m", type=float, default=0.006)
    p.add_argument("--sigma-rotation-delta-accel-rad", type=float, default=0.050)
    p.add_argument("--max-surface-median-degradation-m", type=float, default=0.003)
    p.add_argument("--complete-full-timeline-rigid-pose", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-rigid-pose-interpolation-gap-frames", type=int, default=10)
    p.add_argument("--max-rigid-pose-extrapolation-gap-frames", type=int, default=10)
    p.add_argument("--min-direct-pose-fraction", type=float, default=0.80)
    p.add_argument("--max-direct-pose-gap-frames", type=int, default=10)
    p.add_argument("--max-completed-pose-fraction", type=float, default=0.20)
    p.add_argument("--max-nearest-hold-fraction", type=float, default=0.05)
    p.add_argument("--max-rotation-step-deg", type=float, default=15.0)
    p.add_argument(
        "--allow-sparse-conditional-rotation-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow only the bounded, explicitly uncertain 15--18 degree direct metric-surface tail.",
    )
    p.add_argument("--conditional-max-rotation-step-deg", type=float, default=18.0)
    p.add_argument("--conditional-max-rotation-step-count", type=int, default=2)
    p.add_argument("--conditional-max-rotation-step-fraction", type=float, default=0.015)
    p.add_argument("--conditional-max-rotation-step-translation-m", type=float, default=0.020)
    p.add_argument(
        "--conditional-max-endpoint-rotation-observability-score", type=float, default=0.030
    )
    p.add_argument("--max-translation-step-m", type=float, default=0.05)
    p.add_argument("--min-rotation-observable-fraction", type=float, default=0.80)
    p.add_argument("--min-rotation-observability-score", type=float, default=0.02)
    p.add_argument("--surface-metric-sample-count", type=int, default=2500)
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--seed", type=int, default=1907)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.include_ineligible_rigid_pose_observations:
        raise RuntimeError(
            "--include-ineligible-rigid-pose-observations is forbidden by the production observed-only pose contract"
        )
    return args


if __name__ == "__main__":
    run(parse_args())
