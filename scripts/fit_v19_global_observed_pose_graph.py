#!/usr/bin/env python3
"""Experimental global observed-only rigid pose graph for HOT3D P14.

This is an explicit replacement/diagnostic for the open-chain P14 fit.  It
starts from the existing P14 direct poses, but jointly optimizes bounded SE(3)
corrections using:

* fixed observed-anchor surface loop closures,
* fixed local observed-surface correspondences,
* relative RGB/PnP edge measurements when available,
* depth-ownership quality and RGB-vs-depth conflict weights, and
* temporal velocity/acceleration priors on the correction field.

No generated SAM3D face is loaded or consumed.  The output keeps the P14 pose
row status so it can be used as an explicitly experimental input to P15 while
the top-level report records that it is not a formal annotation state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

POSE_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "fit_to_object_owned_rgb_calibrated_pnp",
}


@dataclass
class FrameState:
    frame_idx: int
    source_row: dict[str, Any]
    rotation0: np.ndarray
    translation0: np.ndarray
    observed_world: np.ndarray
    observed_canonical0: np.ndarray
    depth_quality: float
    ownership_removed_fraction: float
    anchor_canonical: np.ndarray
    anchor_target_world: np.ndarray
    anchor_weights: np.ndarray
    rgb_quality: float = 1.0


@dataclass
class SurfaceEdge:
    source_pos: int
    target_pos: int
    source_frame_idx: int
    target_frame_idx: int
    source_canonical: np.ndarray
    target_canonical: np.ndarray
    weights: np.ndarray
    metric_rotation: np.ndarray
    metric_translation: np.ndarray
    edge_weight: float
    frame_gap: int
    kind: str


@dataclass
class RGBEdge:
    source_pos: int
    target_pos: int
    source_frame_idx: int
    target_frame_idx: int
    rotation: np.ndarray
    translation: np.ndarray
    weight: float
    reprojection_median_px: float | None
    rotation_conflict_deg: float | None
    translation_conflict_m: float | None


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {"count": int(len(arr)), "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90.0)), "p95": float(np.percentile(arr, 95.0)), "max": float(np.max(arr)), "mean": float(np.mean(arr))}


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T + np.asarray(translation, dtype=np.float64)[None, :]


def inverse_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - np.asarray(translation, dtype=np.float64)[None, :]) @ np.asarray(rotation, dtype=np.float64)


def corrected_pose(state: FrameState, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(delta, dtype=np.float64)
    return Rotation.from_rotvec(d[:3]).as_matrix() @ state.rotation0, state.translation0 + d[3:6]


def deterministic_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return points.copy()
    rng = np.random.default_rng(int(seed))
    # Random sampling with a fixed seed avoids privileging the serialized pixel order.
    return points[rng.choice(len(points), size=int(count), replace=False)]


def load_pose_mesh(path: Path) -> Any:
    import trimesh
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid observed-only pose mesh: {path}")
    if not np.isfinite(np.asarray(mesh.vertices, dtype=np.float64)).all():
        raise RuntimeError(f"observed-only pose mesh contains non-finite vertices: {path}")
    return mesh


def sample_pose_mesh(mesh: Any, count: int, seed: int) -> np.ndarray:
    import trimesh
    rng = np.random.default_rng(int(seed))
    points, _ = trimesh.sample.sample_surface(mesh, int(count), seed=rng)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("sampled observed-only pose mesh points are invalid")
    return points


def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def depth_quality(obj: dict[str, Any], args: argparse.Namespace) -> tuple[float, float]:
    geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    ownership = geom.get("first_surface_depth_ownership") if isinstance(geom.get("first_surface_depth_ownership"), dict) else {}
    if ownership.get("enabled") is not True or ownership.get("fail_closed") is True:
        raise RuntimeError("global P14 refuses a frame without validated first-surface ownership")
    removed = float(ownership.get("removed_fraction") or 0.0)
    quality = 1.0 - removed / max(float(args.removed_fraction_full_weight), 1.0e-6)
    if ownership.get("raw_tail_dominates_extent_but_is_quarantined") is True:
        quality *= float(args.quarantined_tail_quality_multiplier)
    return float(np.clip(quality, float(args.min_depth_quality), 1.0)), removed


def mutual_pair_indices(
    source: np.ndarray,
    target: np.ndarray,
    max_pairs: int,
    trim_fraction: float,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mutually nearest source/target indices in one common coordinate frame."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    target_tree = cKDTree(target)
    d_st, target_indices = target_tree.query(source, k=1, workers=-1)
    source_tree = cKDTree(source)
    _d_ts, source_indices_for_target = source_tree.query(target, k=1, workers=-1)
    source_indices = np.arange(len(source), dtype=np.int64)
    mutual = source_indices_for_target[target_indices] == source_indices
    pool = np.flatnonzero(mutual)
    if len(pool) < max(10, min(max_pairs, 30)):
        pool = np.argsort(d_st)[: min(len(d_st), max(10, max_pairs))]
    if len(pool) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    cap = min(float(max_distance), float(np.percentile(d_st[pool], 100.0 * float(trim_fraction))))
    keep = pool[d_st[pool] <= cap]
    if len(keep) > max_pairs:
        keep = keep[np.argsort(d_st[keep])[:max_pairs]]
    if len(keep) < min(10, max_pairs):
        keep = np.argsort(d_st)[: min(len(d_st), max(10, max_pairs))]
    return keep.astype(np.int64), target_indices[keep].astype(np.int64), d_st[keep].astype(np.float64)


def mutual_pairs(
    source: np.ndarray,
    target: np.ndarray,
    max_pairs: int,
    trim_fraction: float,
    max_distance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ``seed`` is retained in the signature for compatibility with the original
    # implementation; correspondence selection is deterministic and distance ordered.
    del seed
    source_indices, target_indices, distances = mutual_pair_indices(
        source, target, max_pairs, trim_fraction, max_distance
    )
    return np.asarray(source)[source_indices], np.asarray(target)[target_indices], distances

def load_states(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any]) -> tuple[list[FrameState], dict[int, dict[str, Any]]]:
    frames = {int(f["frame_idx"]): f for f in annotations.get("frames", []) if isinstance(f, dict) and f.get("frame_idx") is not None}
    rows = {int(r["frame_idx"]): r for r in pose_report.get("pose_rows", []) if isinstance(r, dict) and str(r.get("status") or "") in POSE_STATUSES}
    selected = sorted(idx for idx in rows if idx in frames and (args.frame_start is None or idx >= args.frame_start) and (args.frame_end is None or idx <= args.frame_end))
    if len(selected) < int(args.min_frames):
        raise RuntimeError(f"only {len(selected)} direct P14 rows, need {args.min_frames}")
    states: list[FrameState] = []
    full_observed_by_idx: dict[int, np.ndarray] = {}
    for idx in selected:
        frame = frames[idx]
        obj = annotation_object(frame, args.object_id)
        if obj is None:
            raise RuntimeError(f"frame {idx} lacks object {args.object_id}")
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        if obj.get("rigid_pose_observation_eligible") is False or geom.get("rigid_pose_observation_eligible") is False:
            raise RuntimeError(f"global P14 refuses explicitly ineligible metric frame {idx}")
        points = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < int(args.min_observed_points) or not np.isfinite(points).all():
            raise RuntimeError(f"frame {idx} lacks accepted observed metric surfels")
        R0 = np.asarray(rows[idx].get("rotation_world_from_completed_canonical_matrix"), dtype=np.float64)
        t0 = np.asarray(rows[idx].get("translation_world_m"), dtype=np.float64)
        if R0.shape != (3, 3) or t0.shape != (3,) or not np.isfinite(R0).all() or not np.isfinite(t0).all() or abs(np.linalg.det(R0) - 1.0) > 1.0e-4:
            raise RuntimeError(f"invalid P14 pose at frame {idx}")
        quality, removed = depth_quality(obj, args)
        full_observed_by_idx[idx] = points.copy()
        states.append(FrameState(idx, rows[idx], R0, t0, deterministic_sample(points, int(args.max_observed_points), int(args.seed) + idx), np.empty((0, 3)), quality, removed, np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))))
    anchor_idx = int(args.anchor_frame) if args.anchor_frame is not None else int(pose_report.get("anchor_frame_idx", states[0].frame_idx))
    anchor_state = next((s for s in states if s.frame_idx == anchor_idx), None)
    if anchor_state is None:
        raise RuntimeError(f"anchor frame {anchor_idx} is not in direct P14 states")
    anchor_centroid = full_observed_by_idx[anchor_idx].mean(axis=0)
    reported_anchor_centroid = np.asarray(pose_report.get("anchor_centroid_world_m") or [], dtype=np.float64)
    if reported_anchor_centroid.shape == (3,) and np.linalg.norm(anchor_centroid - reported_anchor_centroid) > 1.0e-6:
        raise RuntimeError(
            f"global P14 anchor centroid disagrees with P14 binding: {anchor_centroid} vs {reported_anchor_centroid}"
        )
    anchor_points = deterministic_sample(full_observed_by_idx[anchor_idx] - anchor_centroid[None, :], int(args.max_anchor_points), int(args.seed) + 9001)
    for state in states:
        state.observed_canonical0 = inverse_pose(state.observed_world, state.rotation0, state.translation0)
        # Anchor loop closure is built in the initial P14 basin, then re-evaluated jointly.
        predicted = apply_pose(anchor_points, state.rotation0, state.translation0)
        source_indices, target_indices, distances = mutual_pair_indices(
            predicted,
            state.observed_world,
            int(args.max_anchor_pairs),
            float(args.anchor_trim_fraction),
            float(args.max_anchor_correspondence_m),
        )
        if len(source_indices) < int(args.min_anchor_pairs):
            state.anchor_canonical = np.empty((0, 3)); state.anchor_target_world = np.empty((0, 3)); state.anchor_weights = np.empty((0,))
        else:
            # Keep the canonical anchor coordinates, not their initial world transform.
            state.anchor_canonical = anchor_points[source_indices]
            state.anchor_target_world = state.observed_world[target_indices]
            state.anchor_weights = np.clip(state.depth_quality * np.exp(-distances / max(float(args.anchor_weight_scale_m), 1.0e-6)), 0.10, 1.0)
    return states, frames


def build_surface_edges(states: list[FrameState], args: argparse.Namespace, pose_report: dict[str, Any], rgb_pair_quality: dict[tuple[int, int], float] | None = None) -> list[SurfaceEdge]:
    edges: list[SurfaceEdge] = []
    # Existing P14 direct sequence supplies local edges; add long-range loop edges for global closure.
    pair_specs: list[tuple[int, int, str]] = []
    for i in range(len(states) - 1):
        pair_specs.append((i, i + 1, "local"))
    stride = int(args.loop_stride)
    if stride > 1:
        for i in range(len(states) - stride):
            pair_specs.append((i, i + stride, "loop"))
    # Avoid duplicate pairs and cap the optional loop population.
    seen: set[tuple[int, int]] = set()
    edge_rows = {}
    for row in (pose_report.get("pairwise_observed_surface_registration", {}) or {}).get("edges", []):
        if isinstance(row, dict): edge_rows[(int(row.get("source_frame_idx")), int(row.get("target_frame_idx")))] = row
    for si, ti, kind in pair_specs:
        s, t = states[si], states[ti]
        key = (s.frame_idx, t.frame_idx)
        if key in seen: continue
        seen.add(key)
        # Match both clouds in the common initial canonical frame.
        src, dst, distances = mutual_pairs(s.observed_canonical0, t.observed_canonical0, int(args.max_edge_pairs), float(args.edge_trim_fraction), float(args.max_edge_correspondence_m) * math.sqrt(max(1, t.frame_idx - s.frame_idx)), int(args.seed) + 13000 + s.frame_idx * 17 + t.frame_idx)
        if len(src) < int(args.min_edge_pairs):
            continue
        row = edge_rows.get(key, {})
        local_residual = float((row.get("residual_m") or {}).get("median") or np.median(distances))
        rgb_quality = float((rgb_pair_quality or {}).get(key, 1.0))
        edge_weight = float(np.clip(math.exp(-local_residual / max(float(args.edge_residual_scale_m), 1.0e-6)) * min(s.depth_quality, t.depth_quality) * rgb_quality, float(args.min_edge_weight), 1.0))
        Rm = t.rotation0 @ s.rotation0.T
        tm = t.translation0 - Rm @ s.translation0
        weights = np.clip(edge_weight * np.exp(-distances / max(float(args.edge_weight_scale_m), 1.0e-6)), 0.10, 1.0)
        edges.append(SurfaceEdge(si, ti, s.frame_idx, t.frame_idx, src, dst, weights, Rm, tm, edge_weight, max(1, t.frame_idx - s.frame_idx), kind))
    return edges


def load_rgb_edges(args: argparse.Namespace, states: list[FrameState], path: Path | None) -> list[RGBEdge]:
    if path is None:
        return []
    with np.load(path, allow_pickle=False) as data:
        required = {"metadata", "source_frame_idx", "target_frame_idx", "rotation_rgb", "translation_rgb_m", "quality_weight", "accepted"}
        missing = sorted(required.difference(data.files))
        if missing: raise RuntimeError(f"RGB factor NPZ missing keys: {missing}")
        metadata = json.loads(str(data["metadata"][0]))
        if metadata.get("annotations") != str(args.annotations.expanduser().resolve()) or metadata.get("pose_report") != str(args.pose_report.expanduser().resolve()) or metadata.get("object_id") != str(args.object_id):
            raise RuntimeError("RGB factor metadata contract does not match global P14 inputs")
        hashes = metadata.get("input_sha256") if isinstance(metadata.get("input_sha256"), dict) else {}
        if str(hashes.get("annotations") or "") != sha256_file(args.annotations) or str(hashes.get("pose_report") or "") != sha256_file(args.pose_report):
            raise RuntimeError("RGB factor input hash contract mismatch")
        src=np.asarray(data["source_frame_idx"],dtype=np.int64);dst=np.asarray(data["target_frame_idx"],dtype=np.int64);R=np.asarray(data["rotation_rgb"],dtype=np.float64);t=np.asarray(data["translation_rgb_m"],dtype=np.float64);w=np.asarray(data["quality_weight"],dtype=np.float64);ok=np.asarray(data["accepted"],dtype=bool)
    pos={s.frame_idx:i for i,s in enumerate(states)};out=[]
    for i in range(len(src)):
        if not ok[i] or int(src[i]) not in pos or int(dst[i]) not in pos: continue
        out.append(RGBEdge(pos[int(src[i])],pos[int(dst[i])],int(src[i]),int(dst[i]),R[i],t[i],float(np.clip(w[i],0.05,1.0)),None,None,None))
    return out


def apply_rgb_quality_weights(states: list[FrameState], rgb_edges: list[RGBEdge], args: argparse.Namespace) -> dict[tuple[int, int], float]:
    """Turn RGB-vs-depth conflict evidence into conservative metric-edge weights."""
    by_frame: dict[int, list[float]] = {state.frame_idx: [] for state in states}
    pair_quality: dict[tuple[int, int], float] = {}
    for edge in rgb_edges:
        q = float(np.clip(edge.weight, float(args.min_rgb_edge_quality), 1.0))
        pair_quality[(edge.source_frame_idx, edge.target_frame_idx)] = q
        by_frame.setdefault(edge.source_frame_idx, []).append(q)
        by_frame.setdefault(edge.target_frame_idx, []).append(q)
    for state in states:
        values = by_frame.get(state.frame_idx, [])
        # A lower quartile catches a localized depth jump without allowing one
        # bad edge to erase all otherwise useful frame evidence.
        q = float(np.percentile(values, 25.0)) if values else 1.0
        q = float(np.clip(0.25 + 0.75 * q, float(args.min_rgb_frame_quality), 1.0))
        state.rgb_quality = q
        if len(state.anchor_weights):
            state.anchor_weights = np.clip(state.anchor_weights * q, 0.05, 1.0)
    return pair_quality
def unpack(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(n, 6)


def pose_arrays(states: list[FrameState], x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    d=unpack(x,len(states));Rs=[];ts=[]
    for s,delta in zip(states,d):
        r,t=corrected_pose(s,delta);Rs.append(r);ts.append(t)
    return Rs,ts


def residual_vector(x: np.ndarray, states: list[FrameState], edges: list[SurfaceEdge], rgb_edges: list[RGBEdge], args: argparse.Namespace) -> np.ndarray:
    d=unpack(x,len(states)); Rs,ts=pose_arrays(states,x); parts=[]
    # Per-frame bounded priors keep the experiment a correction around P14.
    for i,delta in enumerate(d):
        parts.append(delta[3:] / float(args.sigma_pose_prior_translation_m))
        parts.append(delta[:3] / float(args.sigma_pose_prior_rotation_rad))
        s=states[i]
        if len(s.anchor_canonical):
            pred=apply_pose(s.anchor_canonical,Rs[i],ts[i]);err=pred-s.anchor_target_world
            clipped=np.clip(err,-float(args.max_anchor_residual_m),float(args.max_anchor_residual_m))
            parts.append(np.sqrt(s.anchor_weights[:,None]) * clipped / float(args.sigma_anchor_point_m))
    # Surface correspondence factors jointly depend on both endpoint poses.
    for e in edges:
        ps=apply_pose(e.source_canonical,Rs[e.source_pos],ts[e.source_pos]);pt=apply_pose(e.target_canonical,Rs[e.target_pos],ts[e.target_pos])
        parts.append(np.sqrt(e.weights[:,None]) * np.clip(ps-pt,-float(args.max_edge_residual_m),float(args.max_edge_residual_m)) / float(args.sigma_edge_point_m))
        Rpred=Rs[e.target_pos] @ Rs[e.source_pos].T
        tpred=ts[e.target_pos] - Rpred @ ts[e.source_pos]
        rerr=Rotation.from_matrix(Rpred @ e.metric_rotation.T).as_rotvec()
        parts.append(np.sqrt(e.edge_weight) * rerr / float(args.sigma_edge_rotation_rad))
        parts.append(np.sqrt(e.edge_weight) * (tpred-e.metric_translation) / float(args.sigma_edge_translation_m))
    for e in rgb_edges:
        Rpred=Rs[e.target_pos] @ Rs[e.source_pos].T
        tpred=ts[e.target_pos] - Rpred @ ts[e.source_pos]
        rerr=Rotation.from_matrix(Rpred @ e.rotation.T).as_rotvec()
        parts.append(np.sqrt(e.weight) * rerr / float(args.sigma_rgb_rotation_rad))
        parts.append(np.sqrt(e.weight) * (tpred-e.translation) / float(args.sigma_rgb_translation_m))
    # Correction-field temporal priors; the physical P14 trajectory is not smoothed directly.
    for i in range(1,len(states)):
        gap=max(1,states[i].frame_idx-states[i-1].frame_idx)
        parts.append((d[i,3:]-d[i-1,3:])/(float(args.sigma_correction_translation_step_m)*math.sqrt(gap)))
        parts.append((d[i,:3]-d[i-1,:3])/(float(args.sigma_correction_rotation_step_rad)*math.sqrt(gap)))
    for i in range(1,len(states)-1):
        g0=max(1,states[i].frame_idx-states[i-1].frame_idx);g1=max(1,states[i+1].frame_idx-states[i].frame_idx)
        vt=(d[i,3:]-d[i-1,3:])/g0-(d[i+1,3:]-d[i,3:])/g1
        vr=(d[i,:3]-d[i-1,:3])/g0-(d[i+1,:3]-d[i,:3])/g1
        parts.append(vt/(float(args.sigma_correction_translation_accel_m)*math.sqrt(max(g0,g1))))
        parts.append(vr/(float(args.sigma_correction_rotation_accel_rad)*math.sqrt(max(g0,g1))))
    # Anchor gauge.  It is nearly fixed by tight bounds as well, but retaining
    # the residual makes the gauge explicit in the report/objective.
    a=int(args.anchor_position)
    parts.append(d[a,3:]/float(args.sigma_anchor_gauge_translation_m))
    parts.append(d[a,:3]/float(args.sigma_anchor_gauge_rotation_rad))
    return np.concatenate([p.reshape(-1) for p in parts]).astype(np.float64)


def residual_sparsity(states: list[FrameState], edges: list[SurfaceEdge], rgb_edges: list[RGBEdge], args: argparse.Namespace) -> sparse.csr_matrix:
    n=len(states);cols=n*6;entries=[];row=0
    def add(count:int, positions:list[int]):
        nonlocal row
        for rr in range(row,row+count):
            for pos in positions:
                entries.extend((rr,c) for c in range(pos*6,pos*6+6))
        row+=count
    for i,s in enumerate(states):
        add(6,[i]);add(3*len(s.anchor_canonical),[i]) if len(s.anchor_canonical) else None
    for e in edges:
        add(3*len(e.source_canonical),[e.source_pos,e.target_pos]);add(3,[e.source_pos,e.target_pos]);add(3,[e.source_pos,e.target_pos])
    for e in rgb_edges:
        add(3,[e.source_pos,e.target_pos]);add(3,[e.source_pos,e.target_pos])
    for i in range(1,n): add(3,[i-1,i]);add(3,[i-1,i])
    for i in range(1,n-1): add(3,[i-1,i,i+1]);add(3,[i-1,i,i+1])
    add(3,[int(args.anchor_position)]);add(3,[int(args.anchor_position)])
    if row==0: raise RuntimeError("empty global pose graph residual")
    rr,cc=np.asarray(entries,dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(rr),dtype=bool),(rr,cc)),shape=(row,cols))


def surface_metrics(states: list[FrameState], x: np.ndarray, mesh_points_canonical: np.ndarray) -> dict[str, Any]:
    Rs, ts = pose_arrays(states, x)
    observed_to_mesh: list[float] = []
    mesh_to_observed: list[float] = []
    self_consistency: list[float] = []
    per_frame: dict[str, Any] = {}
    for i, state in enumerate(states):
        mesh_world = apply_pose(mesh_points_canonical, Rs[i], ts[i])
        observed_dist = cKDTree(mesh_world).query(state.observed_world, k=1, workers=-1)[0]
        mesh_dist = cKDTree(state.observed_world).query(mesh_world, k=1, workers=-1)[0]
        self_pred = apply_pose(state.observed_canonical0, Rs[i], ts[i])
        self_dist = cKDTree(state.observed_world).query(self_pred, k=1, workers=-1)[0]
        observed_to_mesh.append(float(np.median(observed_dist)))
        mesh_to_observed.append(float(np.median(mesh_dist)))
        self_consistency.append(float(np.median(self_dist)))
        per_frame[str(state.frame_idx)] = {
            "observed_to_mesh_median_m": float(np.median(observed_dist)),
            "observed_to_mesh_p95_m": float(np.percentile(observed_dist, 95.0)),
            "mesh_to_observed_median_m": float(np.median(mesh_dist)),
            "mesh_to_observed_p95_m": float(np.percentile(mesh_dist, 95.0)),
            "self_consistency_median_m": float(np.median(self_dist)),
        }
    return {
        "observed_to_mesh_m": numeric_summary(observed_to_mesh),
        "mesh_to_observed_m": numeric_summary(mesh_to_observed),
        "trajectory_self_consistency_m": numeric_summary(self_consistency),
        "per_frame": per_frame,
    }

def build_output_report(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any], states: list[FrameState], edges: list[SurfaceEdge], rgb_edges: list[RGBEdge], result: Any, before: np.ndarray, after: np.ndarray, before_surface: dict[str,Any], after_surface: dict[str,Any]) -> dict[str,Any]:
    d=unpack(result.x,len(states));Rs,ts=pose_arrays(states,result.x);state_by_idx={s.frame_idx:s for s in states}
    rows=[]
    for original in pose_report.get("pose_rows",[]):
        if not isinstance(original,dict): rows.append(original);continue
        idx=int(original.get("frame_idx",-1));row=dict(original)
        if idx in state_by_idx:
            i=next(i for i,s in enumerate(states) if s.frame_idx==idx);s=states[i]
            row["rotation_world_from_completed_canonical_matrix"]=Rs[i].astype(float).tolist();row["translation_world_m"]=ts[i].astype(float).tolist();row["pose_source"]="experimental_global_observed_only_pose_graph";row["direct_pose_observation_source"]="global_observed_surface_pose_graph";row["generated_geometry_pose_evidence_consumed"]=False
            row["experimental_global_pose_graph"]={"rotation_delta_rotvec_rad":d[i,:3].astype(float).tolist(),"translation_delta_world_m":d[i,3:].astype(float).tolist(),"depth_quality":s.depth_quality,"rgb_consistency_quality":s.rgb_quality,"effective_quality":s.depth_quality*s.rgb_quality,"ownership_removed_fraction":s.ownership_removed_fraction,"anchor_loop_pair_count":int(len(s.anchor_canonical)),"correction_is_not_formal_annotation":True}
        rows.append(row)
    anchor_idx=int(args.anchor_frame) if args.anchor_frame is not None else int(pose_report.get("anchor_frame_idx",states[0].frame_idx))
    return {
      "schema":"v19_experimental_global_observed_pose_graph_v1","status":"experimental_global_pose_graph_complete" if result.success else "experimental_global_pose_graph_incomplete","annotation_ready":False,"diagnostic_only":True,
      "claim_scope":"Prediction-side observed-only global SE(3) correction experiment. Anchor/local correspondences and RGB/PnP edges are soft prediction evidence; generated SAM3D faces are not loaded or consumed.","object_id":args.object_id,"pose_hypothesis_mesh_labeled":pose_report.get("pose_hypothesis_mesh_labeled"),"completion_report":pose_report.get("completion_report"),
      "inputs":{"annotations":str(args.annotations.expanduser().resolve()),"initial_pose_report":str(args.pose_report.expanduser().resolve()),"pose_mesh":str(args.pose_mesh.expanduser().resolve()),"pose_mesh_semantics":"P13 observed-only collision surface; generated SAM3D faces are not consumed","rgb_edge_npz":str(args.rgb_edge_npz.expanduser().resolve()) if args.rgb_edge_npz else None,"generated_geometry_consumed":False,"input_sha256":{"annotations":sha256_file(args.annotations),"initial_pose_report":sha256_file(args.pose_report),"pose_mesh":sha256_file(args.pose_mesh),**({"rgb_edge_npz":sha256_file(args.rgb_edge_npz)} if args.rgb_edge_npz else {})}},
      "anchor_frame_idx":anchor_idx,"anchor_position":int(args.anchor_position),"selected_anchor_atomic_binding":pose_report.get("selected_anchor_atomic_binding"),"selected_anchor_evidence_report":pose_report.get("selected_anchor_evidence_report"),"selected_anchor_evidence_report_sha256":pose_report.get("selected_anchor_evidence_report_sha256"),"anchor_centroid_world_m":pose_report.get("anchor_centroid_world_m"),"anchor_observed_extent_m":pose_report.get("anchor_observed_extent_m"),"direct_frame_count":len(states),"direct_frames":[s.frame_idx for s in states],
      "objective":{"terms":["observed-anchor loop-closure point factors","local observed-surface correspondence factors","relative observed metric edge priors","prediction-side RGB/PnP relative edge factors","depth ownership quality weights","correction velocity/acceleration priors","fixed anchor gauge"],"generated_faces_consumed":False,"collision_surface_consumed":False},
      "depth_quality":{"removed_fraction":numeric_summary([s.ownership_removed_fraction for s in states]),"quality":numeric_summary([s.depth_quality for s in states]),"rgb_frame_quality":numeric_summary([s.rgb_quality for s in states]),"frame_rows":[{"frame_idx":s.frame_idx,"removed_fraction":s.ownership_removed_fraction,"ownership_quality":s.depth_quality,"rgb_consistency_quality":s.rgb_quality,"effective_quality":s.depth_quality*s.rgb_quality} for s in states]},
      "surface_edges":{"count":len(edges),"local_count":sum(e.kind=="local" for e in edges),"loop_count":sum(e.kind=="loop" for e in edges),"pair_count_summary":numeric_summary([len(e.source_canonical) for e in edges]),"weight_summary":numeric_summary([e.edge_weight for e in edges]),"rgb_conflict_weighted_local_edges":sum(1 for e in edges if (e.source_frame_idx,e.target_frame_idx) in {(r.source_frame_idx,r.target_frame_idx) for r in rgb_edges})},
      "rgb_edges":{"count":len(rgb_edges),"weight_summary":numeric_summary([e.weight for e in rgb_edges]),"generated_geometry_consumed":False},
      "optimizer":{"success":bool(result.success),"message":str(result.message),"nfev":int(result.nfev),"njev":int(getattr(result,"njev",-1) or -1),"cost":float(result.cost),"residual_rms_before":float(np.sqrt(np.mean(before*before))),"residual_rms_after":float(np.sqrt(np.mean(after*after)))},
      "correction_summary":{"translation_norm_m":numeric_summary(np.linalg.norm(d[:,3:],axis=1)),"rotation_norm_rad":numeric_summary(np.linalg.norm(d[:,:3],axis=1)),"max_translation_m":float(np.max(np.linalg.norm(d[:,3:],axis=1))),"max_rotation_deg":float(np.degrees(np.max(np.linalg.norm(d[:,:3],axis=1))))},
      "surface_before":before_surface,"surface_after":after_surface,"surface_median_degradation_m":(after_surface["observed_to_mesh_m"]["median"]-before_surface["observed_to_mesh_m"]["median"] if before_surface["observed_to_mesh_m"]["median"] is not None and after_surface["observed_to_mesh_m"]["median"] is not None else None),
      "parameters":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items() if k not in {"annotations","pose_report","rgb_edge_npz","pose_mesh","output_dir"}},"pose_rows":rows,"outputs":{"pose_report":str(args.output_dir/"v19_global_observed_pose_graph_report.json"),"v18_compatible_pose_report":str(args.output_dir/"v18_compact_rigid_object_pose_fit_report.json")}
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations",type=Path,required=True);p.add_argument("--pose-report",type=Path,required=True);p.add_argument("--pose-mesh",type=Path,required=True);p.add_argument("--rgb-edge-npz",type=Path,default=None);p.add_argument("--object-id",required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--frame-start",type=int,default=None);p.add_argument("--frame-end",type=int,default=None);p.add_argument("--anchor-frame",type=int,default=None);p.add_argument("--min-frames",type=int,default=8)
    p.add_argument("--max-observed-points",type=int,default=600);p.add_argument("--max-mesh-points",type=int,default=2500);p.add_argument("--max-anchor-points",type=int,default=800);p.add_argument("--max-anchor-pairs",type=int,default=160);p.add_argument("--max-edge-pairs",type=int,default=100);p.add_argument("--min-observed-points",type=int,default=50);p.add_argument("--min-anchor-pairs",type=int,default=20);p.add_argument("--min-edge-pairs",type=int,default=20);p.add_argument("--seed",type=int,default=20260904)
    p.add_argument("--anchor-trim-fraction",type=float,default=.75);p.add_argument("--edge-trim-fraction",type=float,default=.70);p.add_argument("--max-anchor-correspondence-m",type=float,default=.045);p.add_argument("--max-edge-correspondence-m",type=float,default=.035);p.add_argument("--anchor-weight-scale-m",type=float,default=.012);p.add_argument("--edge-weight-scale-m",type=float,default=.010);p.add_argument("--edge-residual-scale-m",type=float,default=.004);p.add_argument("--min-edge-weight",type=float,default=.10);p.add_argument("--loop-stride",type=int,default=10)
    p.add_argument("--removed-fraction-full-weight",type=float,default=.10);p.add_argument("--min-depth-quality",type=float,default=.25);p.add_argument("--quarantined-tail-quality-multiplier",type=float,default=.75);p.add_argument("--min-rgb-edge-quality",type=float,default=.05);p.add_argument("--min-rgb-frame-quality",type=float,default=.25)
    p.add_argument("--sigma-pose-prior-translation-m",type=float,default=.035);p.add_argument("--sigma-pose-prior-rotation-rad",type=float,default=.18);p.add_argument("--sigma-anchor-point-m",type=float,default=.012);p.add_argument("--sigma-edge-point-m",type=float,default=.010);p.add_argument("--sigma-edge-rotation-rad",type=float,default=.10);p.add_argument("--sigma-edge-translation-m",type=float,default=.018);p.add_argument("--sigma-rgb-rotation-rad",type=float,default=.08);p.add_argument("--sigma-rgb-translation-m",type=float,default=.015);p.add_argument("--sigma-correction-translation-step-m",type=float,default=.012);p.add_argument("--sigma-correction-rotation-step-rad",type=float,default=.10);p.add_argument("--sigma-correction-translation-accel-m",type=float,default=.008);p.add_argument("--sigma-correction-rotation-accel-rad",type=float,default=.06);p.add_argument("--sigma-anchor-gauge-translation-m",type=float,default=1e-6);p.add_argument("--sigma-anchor-gauge-rotation-rad",type=float,default=1e-6);p.add_argument("--max-anchor-residual-m",type=float,default=.06);p.add_argument("--max-edge-residual-m",type=float,default=.06);p.add_argument("--max-nfev",type=int,default=60);p.add_argument("--max-correction-translation-m",type=float,default=.06);p.add_argument("--max-correction-rotation-rad",type=float,default=.35);p.add_argument("--verbose",action="store_true")
    args=p.parse_args()
    if args.loop_stride<1: raise RuntimeError("loop_stride must be >=1")
    if not (0.0 < args.min_rgb_edge_quality <= 1.0 and 0.0 < args.min_rgb_frame_quality <= 1.0): raise RuntimeError("RGB quality floors must be in (0,1]")
    annotations=load_json(args.annotations);pose_report=load_json(args.pose_report)
    if not args.pose_mesh.is_file(): raise RuntimeError(f"pose mesh does not exist: {args.pose_mesh}")
    pose_mesh=load_pose_mesh(args.pose_mesh);pose_mesh_points=sample_pose_mesh(pose_mesh,int(args.max_mesh_points),int(args.seed)+7001)
    states,_=load_states(args,annotations,pose_report)
    args.anchor_position=next((i for i,s in enumerate(states) if s.frame_idx==int(args.anchor_frame if args.anchor_frame is not None else pose_report.get("anchor_frame_idx",states[0].frame_idx))),0)
    rgb_edges=load_rgb_edges(args,states,args.rgb_edge_npz)
    rgb_pair_quality=apply_rgb_quality_weights(states,rgb_edges,args)
    edges=build_surface_edges(states,args,pose_report,rgb_pair_quality)
    x0=np.zeros(len(states)*6,dtype=np.float64);before=residual_vector(x0,states,edges,rgb_edges,args);pattern=residual_sparsity(states,edges,rgb_edges,args)
    if pattern.shape!=(len(before),len(x0)): raise RuntimeError(f"sparsity shape {pattern.shape} != {(len(before),len(x0))}")
    lower=np.full_like(x0,-np.inf);upper=np.full_like(x0,np.inf)
    lower[::6]= -float(args.max_correction_rotation_rad);lower[1::6]= -float(args.max_correction_rotation_rad);lower[2::6]= -float(args.max_correction_rotation_rad);upper[::6]=float(args.max_correction_rotation_rad);upper[1::6]=float(args.max_correction_rotation_rad);upper[2::6]=float(args.max_correction_rotation_rad)
    lower[3::6]=-float(args.max_correction_translation_m);lower[4::6]=-float(args.max_correction_translation_m);lower[5::6]=-float(args.max_correction_translation_m);upper[3::6]=float(args.max_correction_translation_m);upper[4::6]=float(args.max_correction_translation_m);upper[5::6]=float(args.max_correction_translation_m)
    a=6*int(args.anchor_position);lower[a:a+6]=-1e-10;upper[a:a+6]=1e-10
    result=least_squares(lambda x:residual_vector(x,states,edges,rgb_edges,args),x0,jac_sparsity=pattern,bounds=(lower,upper),max_nfev=int(args.max_nfev),loss="soft_l1",f_scale=1.0,x_scale="jac",verbose=2 if args.verbose else 0)
    after=residual_vector(result.x,states,edges,rgb_edges,args);before_surface=surface_metrics(states,x0,pose_mesh_points);after_surface=surface_metrics(states,result.x,pose_mesh_points)
    report=build_output_report(args,annotations,pose_report,states,edges,rgb_edges,result,before,after,before_surface,after_surface)
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/"v19_global_observed_pose_graph_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");(args.output_dir/"v18_compact_rigid_object_pose_fit_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["status","optimizer","correction_summary","surface_median_degradation_m","surface_edges","rgb_edges"]},indent=2))


if __name__=="__main__": main()
