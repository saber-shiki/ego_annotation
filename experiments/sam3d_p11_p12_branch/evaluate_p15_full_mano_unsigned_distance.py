#!/usr/bin/env python3
"""Measure full-MANO unsigned proximity to P15 observed/generated surfaces.

Distances to the shared observed-only surface are the only physically eligible
proximity diagnostic.  Distances to generated layers are reported only to explain
rendered hand/object layout; they cannot be interpreted as signed penetration,
contact, collision, or nonpenetration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
import render_v19_rigid_state_artifact as canonical  # noqa: E402
import render_p14_p15_layered_state as layered  # noqa: E402

SCHEMA = "v19_experimental_p15_full_mano_unsigned_surface_distance_v1"
EXPECTED_ORDER = [
    "sam3d_owned_dual_mesh",
    "sam3d_owned_legacy_cut",
    "trellis_frozen_legacy_cut",
]
THRESHOLDS_M = (0.005, 0.010, 0.020, 0.030, 0.050)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def summarize(values: np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def distance_summary(values: np.ndarray) -> dict[str, Any]:
    report = summarize(values)
    report["fraction_within_threshold"] = {
        f"{int(round(threshold * 1000))}mm": float(np.mean(values <= threshold)) if len(values) else None
        for threshold in THRESHOLDS_M
    }
    return report


def per_hand_proximity_summary(values: np.ndarray, row_specs: list[dict[str, Any]]) -> dict[str, Any]:
    minima = np.asarray(
        [np.min(values[int(spec["start"]):int(spec["stop"])]) for spec in row_specs],
        dtype=np.float64,
    )
    p05 = np.asarray(
        [np.percentile(values[int(spec["start"]):int(spec["stop"])], 5) for spec in row_specs],
        dtype=np.float64,
    )
    medians = np.asarray(
        [np.median(values[int(spec["start"]):int(spec["stop"])]) for spec in row_specs],
        dtype=np.float64,
    )
    return {
        "minimum_vertex_distance_m_across_frame_side_rows": summarize(minima),
        "p05_vertex_distance_m_across_frame_side_rows": summarize(p05),
        "median_vertex_distance_m_across_frame_side_rows": summarize(medians),
        "frame_side_row_fraction_with_any_vertex_within_threshold": {
            f"{int(round(threshold * 1000))}mm": float(np.mean(minima <= threshold))
            for threshold in THRESHOLDS_M
        },
    }


def build_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"invalid scene vertices: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid scene faces: {faces.shape}")
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(vertices, dtype=np.float32), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(faces, dtype=np.int32), dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def compute_distances(scene: o3d.t.geometry.RaycastingScene, points: np.ndarray, batch: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(points), int(batch)):
        tensor = o3d.core.Tensor(np.asarray(points[start:start + int(batch)], dtype=np.float32))
        chunks.append(scene.compute_distance(tensor).numpy().astype(np.float64))
    output = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float64)
    if output.shape != (len(points),) or not np.isfinite(output).all() or np.any(output < 0):
        raise RuntimeError(f"invalid unsigned distance output: {output.shape}")
    return output


def generated_layer(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    layers, summaries = layered.load_object_layers(state, [], 0.0)
    rows = [layer for layer in layers if layer.role == "generated_complete_prior_underlay"]
    if len(rows) != 1:
        raise RuntimeError("render state does not have exactly one generated layer")
    layer = rows[0]
    summary_row = [row for row in summaries if row["role"] == layer.role]
    return layer.vertices, layer.faces, summary_row[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = require_file(args.adapter_report, "P14/P15 adapter report")
    state_paths = [require_file(path, "layered render state") for path in args.render_states]
    if len(state_paths) != 3:
        raise RuntimeError("exactly three render states are required")
    adapter = load_json(adapter_path)
    states = [load_json(path) for path in state_paths]
    branch_ids = [str((state.get("experimental_p14_p15_adapter") or {}).get("branch_id")) for state in states]
    if branch_ids != EXPECTED_ORDER:
        raise RuntimeError(f"render state order must be {EXPECTED_ORDER}, got {branch_ids}")
    shared_blocks = adapter.get("shared_state_value_sha256") if isinstance(adapter.get("shared_state_value_sha256"), dict) else {}
    if not shared_blocks:
        raise RuntimeError("adapter report lacks shared-state hashes")
    for block, expected_hash in shared_blocks.items():
        hashes = [value_sha256(state.get(block)) for state in states]
        if any(value != expected_hash for value in hashes):
            raise RuntimeError(f"state block {block} no longer matches adapter report")

    annotation_path = require_file(
        Path(str((states[0].get("annotation_backbone") or {}).get("path", ""))),
        "annotation backbone",
    )
    annotations = load_json(annotation_path)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotation backbone has no frames")
    poses = canonical.pose_map(states[0])
    if len(poses) != len(frames):
        raise RuntimeError(f"pose/annotation frame mismatch: {len(poses)} vs {len(frames)}")

    physical_path = require_file(
        Path(str(((states[0].get("object_geometry") or {}).get("physical_surface") or {}).get("mesh", ""))),
        "shared observed-only surface",
    )
    observed_vertices, observed_faces, observed_mesh_summary = canonical.load_mesh(physical_path)

    mano_cache = layered.ManoArchiveCache([])
    all_points: list[np.ndarray] = []
    row_specs: list[dict[str, Any]] = []
    try:
        for position, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise RuntimeError(f"annotation frame {position} is not an object")
            frame_idx = canonical.frame_id(frame, position)
            if frame_idx not in poses:
                raise RuntimeError(f"frame {frame_idx} has no object pose")
            rotation, translation, pose_status = poses[frame_idx]
            for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
                if not isinstance(hand, dict):
                    continue
                vertices_world, faces, provenance = mano_cache.hand_mesh(frame_idx, hand)
                del faces
                # Forward pose convention is canonical @ R.T + t, therefore the
                # row-vector inverse is (world - t) @ R.
                vertices_canonical = (vertices_world - translation[None, :]) @ rotation
                start = sum(len(chunk) for chunk in all_points)
                all_points.append(vertices_canonical)
                row_specs.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_side": provenance["side"],
                        "pose_status": pose_status,
                        "start": start,
                        "stop": start + len(vertices_canonical),
                        "vertex_count": int(len(vertices_canonical)),
                    }
                )
    finally:
        mano_summary = mano_cache.summary()
        mano_cache.close()
    query_points = np.vstack(all_points)
    if len(row_specs) != 2 * len(frames) or query_points.shape != (len(row_specs) * 778, 3):
        raise RuntimeError(
            f"incomplete full MANO query set: rows={len(row_specs)} points={query_points.shape} frames={len(frames)}"
        )

    surfaces: list[dict[str, Any]] = [
        {
            "surface_id": "shared_observed_metric_surface",
            "semantics": "only_collision_eligible_unsigned_surface",
            "vertices": observed_vertices,
            "faces": observed_faces,
            "mesh_summary": observed_mesh_summary,
            "branch_id": None,
        }
    ]
    for branch_id, state in zip(branch_ids, states):
        vertices, faces, layer_summary = generated_layer(state)
        surfaces.append(
            {
                "surface_id": f"{branch_id}_generated_render_prior",
                "semantics": "render_only_unsigned_diagnostic_not_collision_or_contact",
                "vertices": vertices,
                "faces": faces,
                "mesh_summary": layer_summary,
                "branch_id": branch_id,
            }
        )

    distances_by_surface: dict[str, np.ndarray] = {}
    surface_reports: list[dict[str, Any]] = []
    for surface in surfaces:
        scene = build_scene(surface["vertices"], surface["faces"])
        distances = compute_distances(scene, query_points, int(args.query_batch_size))
        distances_by_surface[str(surface["surface_id"])] = distances
        surface_reports.append(
            {
                "surface_id": surface["surface_id"],
                "branch_id": surface["branch_id"],
                "semantics": surface["semantics"],
                "mesh_summary": surface["mesh_summary"],
                "all_full_mano_vertices_unsigned_distance_m": distance_summary(distances),
                "per_frame_side_hand_proximity": per_hand_proximity_summary(distances, row_specs),
            }
        )
        print(
            f"queried {surface['surface_id']}: {len(distances)} points, median={np.median(distances)*1000:.2f}mm",
            flush=True,
        )

    per_hand_rows: list[dict[str, Any]] = []
    for spec in row_specs:
        start, stop = int(spec["start"]), int(spec["stop"])
        row = {key: value for key, value in spec.items() if key not in ("start", "stop")}
        row["surfaces"] = {
            surface_id: distance_summary(values[start:stop])
            for surface_id, values in distances_by_surface.items()
        }
        per_hand_rows.append(row)

    aggregate_by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        side_specs = [spec for spec in row_specs if spec["hand_side"] == side]
        side_indices = np.concatenate(
            [np.arange(int(spec["start"]), int(spec["stop"]), dtype=np.int64) for spec in side_specs]
        )
        aggregate_by_side[side] = {
            surface_id: distance_summary(values[side_indices])
            for surface_id, values in distances_by_surface.items()
        }

    observed_id = "shared_observed_metric_surface"
    comparisons: dict[str, Any] = {}
    dual_id = "sam3d_owned_dual_mesh_generated_render_prior"
    cut_id = "sam3d_owned_legacy_cut_generated_render_prior"
    if dual_id in distances_by_surface and cut_id in distances_by_surface:
        dual = distances_by_surface[dual_id]
        cut = distances_by_surface[cut_id]
        delta = dual - cut
        comparisons["sam3d_dual_minus_same_prior_legacy_cut_unsigned_distance_m"] = {
            "summary": summarize(delta),
            "dual_is_closer_fraction": float(np.mean(delta < -1.0e-6)),
            "legacy_cut_is_closer_fraction": float(np.mean(delta > 1.0e-6)),
            "equal_within_1um_fraction": float(np.mean(np.abs(delta) <= 1.0e-6)),
            "interpretation": (
                "Negative values mean intact topology provides a nearer render-prior surface. "
                "This is not evidence of true contact or penetration."
            ),
        }
    comparisons["observed_surface_minimum_distance_m"] = {
        "summary": summarize(distances_by_surface[observed_id]),
        "interpretation": (
            "Unsigned distance to prediction-side observed object geometry. Near-zero values are proximity only; "
            "the surface is partial/non-watertight and cannot determine inside/outside."
        ),
    }

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "evaluate_experimental_p15_full_mano_unsigned_surface_distance",
        "claim_scope": (
            "Full 778-vertex source metric MANO unsigned proximity in the frozen observed-only object canonical frame. "
            "Only the shared observed surface is collision-eligible, and it remains partial/non-watertight. Generated "
            "surface distances explain rendering only and cannot establish signed contact, penetration, collision, or nonpenetration."
        ),
        "inputs": {
            "adapter_report": str(adapter_path),
            "adapter_report_sha256": sha256_file(adapter_path),
            "render_states": [
                {"branch_id": branch_id, "path": str(path), "sha256": sha256_file(path)}
                for branch_id, path in zip(branch_ids, state_paths)
            ],
            "annotations": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "shared_observed_surface": str(physical_path),
            "shared_observed_surface_sha256": sha256_file(physical_path),
        },
        "frame_count": len(frames),
        "hand_row_count": len(row_specs),
        "mano_full_surface": mano_summary,
        "distance_contract": {
            "coordinate_frame": "completed_object_canonical_via_inverse_frozen_observed_only_SE3_pose",
            "distance": "Open3D exact unsigned point-to-triangle surface distance",
            "query_vertex_count": int(len(query_points)),
            "query_batch_size": int(args.query_batch_size),
            "signed_distance_enabled": False,
            "generated_faces_collision_eligible": False,
            "generated_faces_contact_eligible": False,
        },
        "surfaces": surface_reports,
        "aggregate_by_hand_side": aggregate_by_side,
        "comparisons": comparisons,
        "per_frame_hand_rows": per_hand_rows,
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace:
            raise RuntimeError(f"refusing to overwrite non-empty output: {output_dir}")
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "p15_full_mano_unsigned_surface_distance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "frame_count": report["frame_count"],
        "hand_row_count": report["hand_row_count"],
        "surfaces": [
            {
                "surface_id": row["surface_id"],
                "semantics": row["semantics"],
                "distance": row["all_full_mano_vertices_unsigned_distance_m"],
            }
            for row in surface_reports
        ],
        "comparisons": comparisons,
        "report": str(report_path),
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--render-states", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=100000)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
