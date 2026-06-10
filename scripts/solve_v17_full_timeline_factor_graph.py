#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

from run_v16_full_pipeline import load_mesh_archive, save_mesh_archive

ACCEPTED_STATUS = "accepted_sparse_full_timeline_evidence_graph"
PARTIAL_STATUS = "partial_sparse_full_timeline_evidence_graph"
REJECTED_STATUS = "rejected_sparse_full_timeline_evidence_graph"
SOLVER_COMPLETENESS = "sparse_evidence_consistency_only"


@dataclass(frozen=True)
class GraphFrame:
    frame_idx: int
    time_s: float
    camera_forward_axis_world: np.ndarray
    object_center: np.ndarray | None
    mesh_vertices: np.ndarray | None
    mesh_faces: np.ndarray | None
    hand_points: dict[str, np.ndarray]
    contact_sides: tuple[str, ...]
    contact_source: str | None


@dataclass(frozen=True)
class GraphData:
    frames: list[GraphFrame]
    active_indices: list[int]
    object_var_by_frame: dict[int, int]
    hand_var_by_frame_side: dict[tuple[int, str], int]
    contact_pairs: list[tuple[int, str]]
    contact_sources: dict[tuple[int, str], str]
    skipped_contacts: list[dict[str, Any]]


@dataclass(frozen=True)
class LinearSystem:
    matrix: sparse.csr_matrix
    target: np.ndarray
    contact_correspondence_count: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def array3(value: object) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def shift_vector3_value(value: object, shift: np.ndarray) -> object:
    arr = array3(value)
    if arr is None:
        return value
    return (arr + shift).astype(float).tolist()


def shift_points3_value(value: object, shift: np.ndarray) -> object:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return value
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        return value
    return (arr + shift[None, :]).astype(float).tolist()


def camera_forward_axis_world(frame: dict[str, Any], source: Path, row_i: int) -> np.ndarray:
    camera = frame.get("camera")
    matrix = camera.get("T_world_camera_metric") if isinstance(camera, dict) else None
    try:
        arr = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{source} frame row {row_i} has no valid T_world_camera_metric") from exc
    if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{source} frame row {row_i} has no valid T_world_camera_metric")
    axis = arr[:3, 2]
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError(f"{source} frame row {row_i} has invalid camera forward axis")
    return axis / norm


def side_from_measurement_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parts = re.split(r"[:/]", value)
    for part in parts:
        if part in ("left", "right"):
            return part
    return None


def side_key(hand: dict[str, Any], fallback: int) -> str:
    side = hand.get("side")
    if side in ("left", "right"):
        return str(side)
    entity_id = hand.get("entity_id")
    if entity_id == "hand:left":
        return "left"
    if entity_id == "hand:right":
        return "right"
    return f"hand_{fallback}"


def hand_residual_ok(hand: dict[str, Any], max_median_px: float, max_p95_px: float) -> bool:
    residual = hand.get("projection_residual_to_measurement_px")
    if not isinstance(residual, dict):
        return False
    median = finite_float(residual.get("median"))
    p95 = finite_float(residual.get("p95"))
    if median is None or p95 is None:
        return False
    return median <= max_median_px and p95 <= max_p95_px


def hand_world_points(hand: dict[str, Any], max_points: int, seed: int) -> np.ndarray | None:
    for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
        if key not in hand:
            continue
        points = np.asarray(hand.get(key), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            continue
        if len(points) <= max_points:
            return points
        rng = np.random.default_rng(seed)
        return points[rng.choice(len(points), size=max_points, replace=False)]
    return None


def contact_sides_from_measurements(path: Path) -> dict[int, set[str]]:
    if not path.exists():
        return {}
    rows = load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain a JSON list")
    out: dict[int, set[str]] = {}
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} row {row_i} is not a JSON object")
        if row.get("contact_state_measurement") != "candidate_contact_image_and_metric":
            continue
        if row.get("hand_measurement_valid_for_contact") is False:
            continue
        idx = row.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} row {row_i} has invalid frame_idx {idx!r}")
        side = row.get("hand_side") or side_from_measurement_id(row.get("measurement_id"))
        if side in ("left", "right"):
            out.setdefault(idx, set()).add(str(side))
    return out


def measurement_contact_sides(measurement_store_root: Path, case: str) -> dict[int, set[str]]:
    measurements = measurement_store_root / case / "measurements_v17"
    merged: dict[int, set[str]] = {}
    for name in (
        "contact_measurements.json",
        "hand_repair_contact_measurements.json",
        "local_contact_patch_contact_measurements.json",
    ):
        for idx, sides in contact_sides_from_measurements(measurements / name).items():
            merged.setdefault(idx, set()).update(sides)
    return merged


def contact_mode_graph_sides(contact_mode_graph_root: Path, case: str) -> dict[int, set[str]]:
    path = contact_mode_graph_root / case / "v17_contact_mode_graph_report.json"
    report = load_json(path)
    if not isinstance(report, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    if report.get("status") != "accepted_v17_contact_mode_graph":
        raise RuntimeError(f"{path} is not an accepted contact-mode graph report")
    if report.get("solver_completeness") != "contact_mode_latent_only":
        raise RuntimeError(f"{path} has unexpected solver_completeness {report.get('solver_completeness')!r}")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} must contain rows")
    out: dict[int, set[str]] = {}
    for row_i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} row {row_i} is not a JSON object")
        if row.get("contact_factor_ready") is not True:
            continue
        idx = row.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{path} row {row_i} has invalid frame_idx {idx!r}")
        side = row.get("side")
        if side not in ("left", "right"):
            raise RuntimeError(f"{path} row {row_i} has invalid side {side!r}")
        out.setdefault(idx, set()).add(str(side))
    return out


def contact_mode_factor_ready_count(contact_mode_graph_root: Path, case: str) -> int:
    path = contact_mode_graph_root / case / "v17_contact_mode_graph_report.json"
    report = load_json(path)
    if not isinstance(report, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    count = report.get("contact_factor_ready_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeError(f"{path} has invalid contact_factor_ready_count {count!r}")
    return count


def selected_graph_contact_sides(frame: dict[str, Any]) -> tuple[str, ...]:
    contact = frame.get("v17_contact_state")
    if not isinstance(contact, dict) or contact.get("status") != "accepted_contact":
        return ()
    side = side_from_measurement_id(contact.get("selected_measurement_id")) or side_from_measurement_id(
        contact.get("local_patch_state_id")
    )
    return (side,) if side in ("left", "right") else ()


def local_patch_contact_sides(frame: dict[str, Any]) -> tuple[str, ...]:
    obj = frame.get("object")
    patch = obj.get("v17_local_contact_patch_state") if isinstance(obj, dict) else None
    if not isinstance(patch, dict):
        return ()
    status = patch.get("status")
    if status not in ("accepted_local_contact_patch", "accepted_contact_patch"):
        return ()
    side = patch.get("hand_side") or side_from_measurement_id(patch.get("state_id"))
    return (side,) if side in ("left", "right") else ()


def load_graph_frames(
    args: argparse.Namespace,
    annotations_path: Path,
    mesh_archive_path: Path,
    measurement_sides: dict[int, set[str]],
    contact_mode_sides: dict[int, set[str]] | None,
) -> list[GraphFrame]:
    payload = load_json(annotations_path)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise RuntimeError(f"{annotations_path} must contain a frames list")
    mesh_by_frame = load_mesh_archive(mesh_archive_path)
    out: list[GraphFrame] = []
    for row_i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"{annotations_path} frame row {row_i} is not a JSON object")
        idx = frame.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{annotations_path} frame row {row_i} has invalid frame_idx {idx!r}")
        obj = frame.get("object")
        object_center = array3(obj.get("center_world_m")) if isinstance(obj, dict) else None
        if isinstance(obj, dict) and isinstance(obj.get("v17_shape_state"), dict):
            shape_center = array3(obj["v17_shape_state"].get("object_center_world_m"))
            if shape_center is not None:
                object_center = shape_center
        mesh_vertices: np.ndarray | None = None
        mesh_faces: np.ndarray | None = None
        if idx in mesh_by_frame:
            mesh_vertices, mesh_faces = mesh_by_frame[idx]
            if object_center is None:
                object_center = np.median(mesh_vertices, axis=0)
        hand_points: dict[str, np.ndarray] = {}
        hands = frame.get("hands")
        if isinstance(hands, list):
            for hand_i, hand in enumerate(hands):
                if not isinstance(hand, dict):
                    continue
                if not hand_residual_ok(hand, float(args.max_hand_median_px), float(args.max_hand_p95_px)):
                    continue
                points = hand_world_points(hand, int(args.max_hand_points), int(args.seed) + idx * 17 + hand_i)
                if points is None:
                    continue
                hand_points[side_key(hand, hand_i)] = points
        if contact_mode_sides is not None:
            sides = tuple(sorted(contact_mode_sides.get(idx, set())))
            contact_source = "contact_mode_factor_ready" if sides else None
        else:
            graph_sides = selected_graph_contact_sides(frame)
            if graph_sides:
                sides = graph_sides
                contact_source = "selected_contact_state_graph"
            elif bool(args.allow_measurement_candidate_contacts):
                sides = tuple(sorted(measurement_sides.get(idx, set())))
                contact_source = "measurement_store_candidate_contact" if sides else None
            else:
                sides = local_patch_contact_sides(frame)
                contact_source = "local_contact_patch_state" if sides else None
        out.append(
            GraphFrame(
                frame_idx=idx,
                time_s=float(frame.get("time_s") or idx / 30.0),
                camera_forward_axis_world=camera_forward_axis_world(frame, annotations_path, row_i),
                object_center=object_center,
                mesh_vertices=mesh_vertices,
                mesh_faces=mesh_faces,
                hand_points=hand_points,
                contact_sides=sides,
                contact_source=contact_source,
            )
        )
    return out


def build_graph_data(frames: list[GraphFrame]) -> GraphData:
    active_indices: list[int] = []
    object_var_by_frame: dict[int, int] = {}
    hand_var_by_frame_side: dict[tuple[int, str], int] = {}
    contact_pairs: list[tuple[int, str]] = []
    contact_sources: dict[tuple[int, str], str] = {}
    skipped_contacts: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        if frame.object_center is None or frame.mesh_vertices is None:
            continue
        object_var_by_frame[frame.frame_idx] = len(active_indices)
        active_indices.append(i)
        for side, points in frame.hand_points.items():
            if len(points):
                hand_var_by_frame_side[(frame.frame_idx, side)] = len(hand_var_by_frame_side)
    for i in active_indices:
        frame = frames[i]
        for side in frame.contact_sides:
            if (frame.frame_idx, side) not in hand_var_by_frame_side:
                skipped_contacts.append(
                    {
                        "frame_idx": frame.frame_idx,
                        "side": side,
                        "source": frame.contact_source or "unknown_contact_source",
                        "reason": "no_valid_hand_points_for_contact",
                    }
                )
                continue
            contact_pairs.append((frame.frame_idx, side))
            contact_sources[(frame.frame_idx, side)] = frame.contact_source or "unknown_contact_source"
    if not active_indices:
        raise RuntimeError("no active object frames with mesh data")
    return GraphData(
        frames=frames,
        active_indices=active_indices,
        object_var_by_frame=object_var_by_frame,
        hand_var_by_frame_side=hand_var_by_frame_side,
        contact_pairs=contact_pairs,
        contact_sources=contact_sources,
        skipped_contacts=skipped_contacts,
    )


def variable_counts(graph: GraphData) -> tuple[int, int, int]:
    object_vars = len(graph.object_var_by_frame) * 3
    hand_vars = len(graph.hand_var_by_frame_side)
    return object_vars, hand_vars, object_vars + hand_vars


def unpack(params: np.ndarray, graph: GraphData) -> tuple[np.ndarray, np.ndarray]:
    object_width, hand_width, _ = variable_counts(graph)
    return params[:object_width].reshape(len(graph.object_var_by_frame), 3), params[object_width : object_width + hand_width]


def object_col(graph: GraphData, frame_idx: int, axis: int) -> int:
    return graph.object_var_by_frame[frame_idx] * 3 + axis


def hand_col(graph: GraphData, frame_idx: int, side: str) -> int:
    object_width, _hand_width, _total = variable_counts(graph)
    return object_width + graph.hand_var_by_frame_side[(frame_idx, side)]


def optical_axis(frame: GraphFrame) -> np.ndarray:
    return frame.camera_forward_axis_world


def add_row(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    targets: list[float],
    row_i: int,
    coeffs: list[tuple[int, float]],
    target: float,
    sigma: float,
) -> int:
    inv_sigma = 1.0 / float(sigma)
    for col, val in coeffs:
        rows.append(row_i)
        cols.append(col)
        vals.append(float(val) * inv_sigma)
    targets.append(float(target) * inv_sigma)
    return row_i + 1


def build_linear_system(graph: GraphData, args: argparse.Namespace) -> LinearSystem:
    _object_width, _hand_width, total_width = variable_counts(graph)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    targets: list[float] = []
    row_i = 0
    for frame_idx in sorted(graph.object_var_by_frame):
        for axis in range(3):
            row_i = add_row(
                rows,
                cols,
                vals,
                targets,
                row_i,
                [(object_col(graph, frame_idx, axis), 1.0)],
                0.0,
                float(args.sigma_object_prior_m),
            )
    for frame_idx, side in sorted(graph.hand_var_by_frame_side):
        row_i = add_row(
            rows,
            cols,
            vals,
            targets,
            row_i,
            [(hand_col(graph, frame_idx, side), 1.0)],
            0.0,
            float(args.sigma_hand_ray_prior_m),
        )
    active_frames = [graph.frames[i].frame_idx for i in graph.active_indices]
    for left, right in zip(active_frames, active_frames[1:]):
        sigma = float(args.sigma_object_step_m) * math.sqrt(float(max(1, right - left)))
        for axis in range(3):
            row_i = add_row(
                rows,
                cols,
                vals,
                targets,
                row_i,
                [(object_col(graph, right, axis), 1.0), (object_col(graph, left, axis), -1.0)],
                0.0,
                sigma,
            )
    for left, mid, right in zip(active_frames, active_frames[1:], active_frames[2:]):
        sigma = float(args.sigma_object_accel_m) * float(max(1, right - left))
        for axis in range(3):
            row_i = add_row(
                rows,
                cols,
                vals,
                targets,
                row_i,
                [
                    (object_col(graph, right, axis), 1.0),
                    (object_col(graph, mid, axis), -2.0),
                    (object_col(graph, left, axis), 1.0),
                ],
                0.0,
                sigma,
            )
    hand_series: dict[str, list[int]] = {}
    for frame_idx, side in graph.hand_var_by_frame_side:
        hand_series.setdefault(side, []).append(frame_idx)
    for side, series in hand_series.items():
        series.sort()
        for left, right in zip(series, series[1:]):
            sigma = float(args.sigma_hand_ray_step_m) * math.sqrt(float(max(1, right - left)))
            row_i = add_row(
                rows,
                cols,
                vals,
                targets,
                row_i,
                [(hand_col(graph, right, side), 1.0), (hand_col(graph, left, side), -1.0)],
                0.0,
                sigma,
            )
    contact_count = 0
    frame_by_idx = {graph.frames[i].frame_idx: graph.frames[i] for i in graph.active_indices}
    for frame_idx, side in graph.contact_pairs:
        frame = frame_by_idx[frame_idx]
        if frame.mesh_vertices is None:
            continue
        points = frame.hand_points[side]
        distances, nearest = cKDTree(frame.mesh_vertices).query(points, k=1)
        order = np.argsort(np.asarray(distances, dtype=float))[: min(len(points), int(args.max_contact_points))]
        ray = optical_axis(frame)
        for point_i in order:
            obj_point = frame.mesh_vertices[int(nearest[int(point_i)])]
            hand_point = points[int(point_i)]
            target = hand_point - obj_point
            for axis in range(3):
                row_i = add_row(
                    rows,
                    cols,
                    vals,
                    targets,
                    row_i,
                    [(object_col(graph, frame_idx, axis), 1.0), (hand_col(graph, frame_idx, side), -float(ray[axis]))],
                    float(target[axis]),
                    float(args.sigma_contact_m),
                )
            contact_count += 1
    if row_i == 0:
        raise RuntimeError("linear graph produced no rows")
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(row_i, total_width))
    return LinearSystem(matrix=matrix, target=np.asarray(targets, dtype=float), contact_correspondence_count=contact_count)


def solve_linear_system(system: LinearSystem, graph: GraphData, args: argparse.Namespace) -> Any:
    _object_width, _hand_width, total_width = variable_counts(graph)
    lower = np.full(total_width, -np.inf, dtype=float)
    upper = np.full(total_width, np.inf, dtype=float)
    object_width = len(graph.object_var_by_frame) * 3
    lower[:object_width] = -float(args.max_object_shift_m)
    upper[:object_width] = float(args.max_object_shift_m)
    lower[object_width:] = -float(args.max_hand_ray_shift_m)
    upper[object_width:] = float(args.max_hand_ray_shift_m)
    return lsq_linear(
        system.matrix,
        system.target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=int(args.max_iter),
        verbose=0,
    )


def contact_distances(frame: GraphFrame, side: str, object_shift: np.ndarray, hand_shift_m: float, max_points: int) -> np.ndarray:
    if frame.mesh_vertices is None:
        raise RuntimeError("contact frame has no mesh")
    points = frame.hand_points.get(side)
    if points is None or len(points) == 0:
        return np.zeros((0,), dtype=float)
    mesh = frame.mesh_vertices + object_shift[None, :]
    shifted_hand = points + float(hand_shift_m) * optical_axis(frame)[None, :]
    distances, _ = cKDTree(mesh).query(shifted_hand, k=1)
    distances = np.asarray(distances, dtype=float)
    if len(distances) > max_points:
        distances = np.sort(distances)[:max_points]
    return distances


def frame_contact_metrics(params: np.ndarray, graph: GraphData, args: argparse.Namespace) -> dict[str, Any]:
    object_shift, hand_shift = unpack(params, graph)
    frame_by_idx = {graph.frames[i].frame_idx: graph.frames[i] for i in graph.active_indices}
    rows: list[dict[str, Any]] = []
    for frame_idx, side in graph.contact_pairs:
        frame = frame_by_idx[frame_idx]
        distances = contact_distances(
            frame,
            side,
            object_shift[graph.object_var_by_frame[frame_idx]],
            float(hand_shift[graph.hand_var_by_frame_side[(frame_idx, side)]]),
            int(args.max_contact_points),
        )
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "side": side,
                "count": int(distances.size),
                "median_m": float(np.median(distances)) if distances.size else None,
                "p95_m": float(np.percentile(distances, 95.0)) if distances.size else None,
                "min_m": float(np.min(distances)) if distances.size else None,
            }
        )
    medians = np.asarray([row["median_m"] for row in rows if row["median_m"] is not None], dtype=float)
    p95s = np.asarray([row["p95_m"] for row in rows if row["p95_m"] is not None], dtype=float)
    return {
        "rows": rows,
        "contact_factor_count": int(len(rows)),
        "median_m_median": float(np.median(medians)) if medians.size else None,
        "median_m_p95": float(np.percentile(medians, 95.0)) if medians.size else None,
        "p95_m_median": float(np.median(p95s)) if p95s.size else None,
        "p95_m_p95": float(np.percentile(p95s, 95.0)) if p95s.size else None,
    }


def summarize_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {"count": int(arr.size), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95.0)), "max": float(np.max(arr))}


def summarize_shifts(params: np.ndarray, graph: GraphData) -> dict[str, Any]:
    object_shift, hand_shift = unpack(params, graph)
    norms = np.linalg.norm(object_shift, axis=1) if len(object_shift) else np.zeros((0,), dtype=float)
    return {"object_shift_norm_m": summarize_array(norms), "hand_ray_shift_abs_m": summarize_array(np.abs(hand_shift))}


def save_corrected_mesh_archive(path: Path, graph: GraphData, params: np.ndarray) -> dict[str, Any]:
    object_shift, _hand_shift = unpack(params, graph)
    frames: list[int] = []
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for frame_i in graph.active_indices:
        frame = graph.frames[frame_i]
        if frame.mesh_vertices is None or frame.mesh_faces is None:
            continue
        shift = object_shift[graph.object_var_by_frame[frame.frame_idx]]
        frames.append(frame.frame_idx)
        vertices.append(frame.mesh_vertices + shift[None, :])
        faces.append(frame.mesh_faces)
    save_mesh_archive(path, frames, vertices, faces)
    return {"path": str(path), "mesh_frames": int(len(frames)), "first_frame": int(frames[0]) if frames else None, "last_frame": int(frames[-1]) if frames else None}


def apply_object_shift(obj: dict[str, Any], shift: np.ndarray, report: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    for key in ("center_world_m", "position_world_m"):
        if key in out:
            out[key] = shift_vector3_value(out[key], shift)
    for nested_key in ("v17_shape_state", "v17_local_contact_patch_state"):
        nested = out.get(nested_key)
        if not isinstance(nested, dict):
            continue
        nested_out = dict(nested)
        for key in ("object_center_world_m", "center_world_m"):
            if key in nested_out:
                nested_out[key] = shift_vector3_value(nested_out[key], shift)
        for key in ("surface_vertices", "mesh_vertices"):
            if key in nested_out:
                nested_out[key] = shift_points3_value(nested_out[key], shift)
        out[nested_key] = nested_out
    out["v17_full_timeline_factor_graph"] = {
        "object_translation_correction_m": shift.astype(float).tolist(),
        "correction_archive": report["corrected_mesh_archive"]["path"],
    }
    return out


def count_contact_sources(graph: GraphData) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in graph.contact_pairs:
        source = graph.contact_sources.get(pair, "unknown_contact_source")
        out[source] = out.get(source, 0) + 1
    return dict(sorted(out.items()))


def apply_hand_ray_shift(hand: dict[str, Any], shift_vec: np.ndarray, shift_m: float) -> None:
    for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m", "joints_world_m", "mano_vertices_world_m", "mano_joints_world_m"):
        if key in hand:
            hand[key] = shift_points3_value(hand[key], shift_vec)
    for key in ("root_world_m", "wrist_world_m", "palm_center_world_m"):
        if key in hand:
            hand[key] = shift_vector3_value(hand[key], shift_vec)
    hand["v17_full_timeline_factor_graph"] = {
        "hand_ray_shift_m": float(shift_m),
        "hand_world_geometry_shift_m": shift_vec.astype(float).tolist(),
    }


def write_corrected_annotations(path: Path, source_annotations: Path, graph: GraphData, params: np.ndarray, report: dict[str, Any]) -> None:
    payload = load_json(source_annotations)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise RuntimeError(f"{source_annotations} must contain frames")
    object_shift, hand_shift = unpack(params, graph)
    object_shift_by_frame = {idx: object_shift[var_i].astype(float) for idx, var_i in graph.object_var_by_frame.items()}
    hand_shift_by_frame_side = {f"{idx}:{side}": float(hand_shift[var_i]) for (idx, side), var_i in graph.hand_var_by_frame_side.items()}
    frame_by_idx = {graph.frames[i].frame_idx: graph.frames[i] for i in graph.active_indices}
    out_frames: list[dict[str, Any]] = []
    for frame in frames:
        copied = copy.deepcopy(frame)
        idx = copied.get("frame_idx")
        if isinstance(idx, int):
            shift = object_shift_by_frame.get(idx)
            if shift is not None:
                copied["object"] = apply_object_shift(dict(copied.get("object") or {}), shift, report)
            graph_frame = frame_by_idx.get(idx)
            for hand_i, hand in enumerate(copied.get("hands") or []):
                if not isinstance(hand, dict):
                    continue
                hshift = hand_shift_by_frame_side.get(f"{idx}:{side_key(hand, hand_i)}")
                if hshift is not None and graph_frame is not None:
                    apply_hand_ray_shift(hand, graph_frame.camera_forward_axis_world * float(hshift), float(hshift))
        out_frames.append(copied)
    payload["frames"] = out_frames
    payload["v17_full_timeline_factor_graph"] = {
        "status": report["status"],
        "annotation_ready": report["annotation_ready"],
        "solver_completeness": report["solver_completeness"],
        "v3_solver_complete": False,
        "report": report["report_path"],
    }
    write_json(path, payload)


def solve_case(args: argparse.Namespace, case_manifest: Path, output_root: Path) -> dict[str, Any]:
    state = load_json(case_manifest)
    case = str(state["case"])
    annotations = Path(state["annotations"])
    mesh_archive = Path(state["object_mesh_archive"])
    contact_mode_sides = (
        contact_mode_graph_sides(Path(args.contact_mode_graph_root), case)
        if args.contact_mode_graph_root is not None
        else None
    )
    measurement_sides = {} if contact_mode_sides is not None else measurement_contact_sides(Path(args.measurement_store_root), case)
    frames = load_graph_frames(args, annotations, mesh_archive, measurement_sides, contact_mode_sides)
    graph = build_graph_data(frames)
    system = build_linear_system(graph, args)
    x0 = np.zeros(variable_counts(graph)[2], dtype=float)
    before_residual = system.matrix @ x0 - system.target
    result = solve_linear_system(system, graph, args)
    solution = np.asarray(result.x, dtype=float)
    after_residual = system.matrix @ solution - system.target
    before_contact = frame_contact_metrics(x0, graph, args)
    after_contact = frame_contact_metrics(solution, graph, args)
    shifts = summarize_shifts(solution, graph)
    contact_p95 = after_contact.get("p95_m_p95")
    object_shift_max = shifts["object_shift_norm_m"]["max"]
    hand_shift_max = shifts["hand_ray_shift_abs_m"]["max"]
    rejection_reasons: list[str] = []
    if len(graph.contact_pairs) == 0:
        rejection_reasons.append("no_selected_contact_constraints")
    if not bool(result.success):
        rejection_reasons.append("least_squares_failed")
    if contact_p95 is None:
        rejection_reasons.append("contact_p95_unavailable")
    elif float(contact_p95) > float(args.accept_contact_p95_m):
        rejection_reasons.append("contact_p95_exceeds_threshold")
    if object_shift_max is None:
        rejection_reasons.append("object_shift_unavailable")
    elif float(object_shift_max) > float(args.accept_object_shift_max_m):
        rejection_reasons.append("object_shift_exceeds_threshold")
    if hand_shift_max is None:
        rejection_reasons.append("hand_ray_shift_unavailable")
    elif float(hand_shift_max) > float(args.accept_hand_ray_shift_max_m):
        rejection_reasons.append("hand_ray_shift_exceeds_threshold")
    accepted = (
        bool(result.success)
        and contact_p95 is not None
        and len(graph.contact_pairs) > 0
        and float(contact_p95) <= float(args.accept_contact_p95_m)
        and object_shift_max is not None
        and float(object_shift_max) <= float(args.accept_object_shift_max_m)
        and hand_shift_max is not None
        and float(hand_shift_max) <= float(args.accept_hand_ray_shift_max_m)
    )
    case_dir = output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    corrected_archive = case_dir / "object_meshes_v17_full_timeline_graph.npz"
    archive_report = save_corrected_mesh_archive(corrected_archive, graph, solution)
    report_path = case_dir / "v17_full_timeline_factor_graph_report.json"
    corrected_annotations = case_dir / "annotations_v17_full_timeline_graph.json"
    graph_manifest = case_dir / "v17_full_timeline_factor_graph_manifest.json"
    system_shape = system.matrix.shape
    if system_shape is None:
        raise RuntimeError("linear system matrix shape is unavailable")
    contact_factor_source = "contact_mode_factor_ready" if contact_mode_sides is not None else "selected_v17_contact_state"
    contact_mode_factor_ready_total = (
        contact_mode_factor_ready_count(Path(args.contact_mode_graph_root), case)
        if args.contact_mode_graph_root is not None
        else None
    )
    skipped_contact_factor_count = len(graph.skipped_contacts)
    structural_consistency_pass = bool(accepted)
    accuracy_target_met = bool(
        after_contact.get("p95_m_p95") is not None
        and float(after_contact["p95_m_p95"]) <= float(args.accuracy_target_m)
        and object_shift_max is not None
        and float(object_shift_max) <= float(args.accuracy_target_m)
        and hand_shift_max is not None
        and float(hand_shift_max) <= float(args.accuracy_target_m)
    )
    contact_factor_complete = skipped_contact_factor_count == 0
    deliverable_ready = bool(structural_consistency_pass and accuracy_target_met and contact_factor_complete)
    status = ACCEPTED_STATUS if structural_consistency_pass and contact_factor_complete else PARTIAL_STATUS if structural_consistency_pass else REJECTED_STATUS
    contact_constraint_rule = (
        "When --contact-mode-graph-root is provided, only contact_factor_ready rows from the accepted contact-mode graph become contact factors. "
        "Otherwise only selected V17 contact states and accepted local contact patch states become contact factors by default; candidate contact measurements remain evidence until a contact graph selects them."
    )
    report: dict[str, Any] = {
        "case": case,
        "status": status,
        "structural_consistency_pass": bool(structural_consistency_pass),
        "accuracy_target_met": bool(accuracy_target_met),
        "annotation_ready": bool(deliverable_ready),
        "deliverable_ready": bool(deliverable_ready),
        "solver_completeness": SOLVER_COMPLETENESS,
        "v3_solver_complete": False,
        "method": "solve_v17_full_timeline_factor_graph",
        "semantics": {
            "optimized_variables": ["per-active-frame object translation correction", "per-valid-hand camera-ray depth correction"],
            "fixed_variables": ["camera trajectory", "MANO articulation and shape", "object mesh topology", "contact mode labels from current V17 evidence"],
            "contact_constraint_rule": contact_constraint_rule,
            "claim_limit": "This sparse graph tests full-timeline consistency of accepted evidence under bounded translation/depth corrections. The complete V3 joint camera-MANO-object-depth-contact solver remains open.",
        },
        "contact_factor_source": contact_factor_source,
        "contact_mode_graph_root": str(args.contact_mode_graph_root) if args.contact_mode_graph_root is not None else None,
        "source_manifest": str(case_manifest),
        "source_annotations": str(annotations),
        "source_mesh_archive": str(mesh_archive),
        "report_path": str(report_path),
        "corrected_annotations": str(corrected_annotations),
        "corrected_mesh_archive": archive_report,
        "frame_count": int(len(frames)),
        "active_object_frame_count": int(len(graph.active_indices)),
        "inactive_or_missing_object_frame_count": int(len(frames) - len(graph.active_indices)),
        "object_variable_frames": int(len(graph.object_var_by_frame)),
        "hand_variable_count": int(len(graph.hand_var_by_frame_side)),
        "contact_factor_count": int(len(graph.contact_pairs)),
        "contact_mode_factor_ready_input_count": contact_mode_factor_ready_total,
        "skipped_contact_factor_count": int(skipped_contact_factor_count),
        "contact_factor_complete": bool(contact_factor_complete),
        "contact_factor_source_counts": count_contact_sources(graph),
        "linearized_contact_correspondences": int(system.contact_correspondence_count),
        "skipped_contacts": graph.skipped_contacts,
        "variable_count": int(variable_counts(graph)[2]),
        "linear_system_shape": [int(system_shape[0]), int(system_shape[1])],
        "least_squares_success": bool(result.success),
        "least_squares_status": int(result.status),
        "least_squares_message": str(result.message),
        "nit": int(result.nit),
        "cost": float(result.cost),
        "weighted_residual_rms_before": float(np.sqrt(np.mean(before_residual * before_residual))),
        "weighted_residual_rms_after": float(np.sqrt(np.mean(after_residual * after_residual))),
        "contact_before": before_contact,
        "contact_after": after_contact,
        "correction_summary": shifts,
        "acceptance": {
            "accept_contact_p95_m": float(args.accept_contact_p95_m),
            "accept_object_shift_max_m": float(args.accept_object_shift_max_m),
            "accept_hand_ray_shift_max_m": float(args.accept_hand_ray_shift_max_m),
            "accuracy_target_m": float(args.accuracy_target_m),
            "structural_consistency_passed": bool(structural_consistency_pass),
            "accuracy_target_met": bool(accuracy_target_met),
            "contact_factor_complete": bool(contact_factor_complete),
            "deliverable_ready": bool(deliverable_ready),
            "rejection_reasons": rejection_reasons,
        },
    }
    write_json(report_path, report)
    write_corrected_annotations(corrected_annotations, annotations, graph, solution, report)
    write_json(
        graph_manifest,
        {
            "case": case,
            "status": report["status"],
            "raw_frame_count": int(len(frames)),
            "v16_manifest": state["v16_manifest"],
            "annotations": str(corrected_annotations),
            "object_mesh_archive": str(corrected_archive),
            "solver_status": report["status"],
            "solver_completeness": report["solver_completeness"],
            "v3_solver_complete": False,
            "solver_report": str(report_path),
        },
    )
    return report


def solve(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [solve_case(args, manifest, args.output_root) for manifest in args.case_manifests]
    summary = {
        "status": "deliverable_ready" if all(case["deliverable_ready"] for case in cases) else "partial",
        "structural_consistency_status": "pass" if all(case["structural_consistency_pass"] for case in cases) else "fail",
        "accuracy_target_status": "pass" if all(case["accuracy_target_met"] for case in cases) else "fail",
        "deliverable_ready": bool(all(case["deliverable_ready"] for case in cases)),
        "method": "solve_v17_full_timeline_factor_graph",
        "cases": cases,
    }
    write_json(args.output_root / "v17_full_timeline_factor_graph_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "cases": [
                    {
                        "case": c["case"],
                        "status": c["status"],
                        "annotation_ready": c["annotation_ready"],
                        "deliverable_ready": c["deliverable_ready"],
                        "structural_consistency_pass": c["structural_consistency_pass"],
                        "accuracy_target_met": c["accuracy_target_met"],
                        "contact_after_p95_m_p95": c["contact_after"]["p95_m_p95"],
                        "weighted_residual_rms_after": c["weighted_residual_rms_after"],
                    }
                    for c in cases
                ],
            },
            indent=2,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_full_timeline_factor_graph"))
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--contact-mode-graph-root", type=Path, default=None)
    parser.add_argument(
        "--case-manifests",
        type=Path,
        nargs="+",
        default=[
            Path("/data2/ego_annotation_outputs/v17_full_state/trash_1050/v17_full_state_manifest.json"),
            Path("/data2/ego_annotation_outputs/v17_full_state/task5_tomato_960/v17_full_state_manifest.json"),
        ],
    )
    parser.add_argument("--max-hand-points", type=int, default=778)
    parser.add_argument("--max-contact-points", type=int, default=80)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--sigma-object-prior-m", type=float, default=0.015)
    parser.add_argument("--sigma-hand-ray-prior-m", type=float, default=0.015)
    parser.add_argument("--sigma-object-step-m", type=float, default=0.006)
    parser.add_argument("--sigma-object-accel-m", type=float, default=0.012)
    parser.add_argument("--sigma-hand-ray-step-m", type=float, default=0.01)
    parser.add_argument("--sigma-contact-m", type=float, default=0.006)
    parser.add_argument("--max-object-shift-m", type=float, default=0.06)
    parser.add_argument("--max-hand-ray-shift-m", type=float, default=0.06)
    parser.add_argument("--accept-contact-p95-m", type=float, default=0.035)
    parser.add_argument("--accept-object-shift-max-m", type=float, default=0.04)
    parser.add_argument("--accept-hand-ray-shift-max-m", type=float, default=0.04)
    parser.add_argument("--accuracy-target-m", type=float, default=0.005)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-measurement-candidate-contacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    solve(parse_args())


if __name__ == "__main__":
    main()
