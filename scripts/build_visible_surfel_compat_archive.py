#!/usr/bin/env python3
"""Build a visible-surfel compatibility archive for CoTracker sparse-edge diagnostics.

The sparse-edge diagnostic reuses
``diagnose_object_mesh_temporal_consistency_v3.load_mesh_archive``. That loader requires
NPZ keys named ``frame_idx, vertex_offsets, face_offsets, vertices, faces`` and rejects
frames with empty face slices. The diagnostic that follows uses only the vertices in a
KDTree; it ignores faces.

This script therefore packs prediction-side
``objects[*].visible_geometry_candidate.world_vertices_sample_m`` into that loader schema
without converting the samples into an object mesh. Vertices are the evidence. Faces are a
single local placeholder triangle per frame so the legacy loader accepts the archive; they
are non-evidential padding and must not be used for normals, areas, rendering,
intersection, contact, nonpenetration, occlusion, pose fitting, or surface continuity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ARCHIVE_KIND = "visible_surfel_compatibility_archive"
FACE_POLICY = "non_evidential_loader_placeholder_triangle"
PROVENANCE = (
    "Prediction-side visible metric surfel samples only. Source field: "
    "objects[*].visible_geometry_candidate.world_vertices_sample_m. The vertices are "
    "depth+segmentation back-projections of the observed visible object surface in the "
    "metric world frame. No HOT3D GT, hidden geometry, completed mesh, canonical mesh, "
    "object pose, contact, occlusion, or nonpenetration evidence is represented. The NPZ "
    "faces array is non-evidential loader padding for consumers that only use vertices; "
    "it must not be used for geometry, normals, areas, rendering, intersection, contact, "
    "nonpenetration, occlusion, pose fitting, or surface continuity."
)
FORBIDDEN_CLAIMS = [
    "repaired object mesh",
    "complete object geometry",
    "hidden geometry",
    "object pose reconstructed from mesh",
    "rigidity proven",
    "contact/nonpenetration/occlusion grounded by this archive",
    "surface distance to object mesh",
]
ALLOWED_CLAIMS = [
    "prediction-side visible metric surfel samples",
    "nearest visible-surface sample filtering",
    "candidate sparse correspondence edges on observed visible samples",
    "diagnostic rigidity measurement input, not final pose evidence",
]


@dataclass(frozen=True)
class FrameSurfelBlock:
    frame_idx: int
    vertices: np.ndarray
    faces: np.ndarray
    selected_object_id: str | None


def summarize(values: np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return data


def parse_object_selector(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def select_object(frame: dict[str, Any], selector: str | int | None) -> dict[str, Any] | None:
    objects = frame.get("objects")
    if not isinstance(objects, list) or not objects:
        return None
    if selector is None:
        obj = objects[0]
        return obj if isinstance(obj, dict) else None
    if isinstance(selector, int):
        if 0 <= selector < len(objects) and isinstance(objects[selector], dict):
            return objects[selector]
        return None
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        candidates = [obj.get("object_id"), obj.get("track_id"), obj.get("id"), obj.get("label")]
        if any(str(candidate) == selector for candidate in candidates if candidate is not None):
            return obj
    return None


def visible_surfel_vertices(obj: dict[str, Any]) -> np.ndarray | None:
    geom = obj.get("visible_geometry_candidate")
    if not isinstance(geom, dict):
        return None
    raw = geom.get("world_vertices_sample_m")
    if not isinstance(raw, list) or not raw:
        return None
    pts = np.asarray(raw, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return None
    return pts


def placeholder_face(vertex_count: int) -> np.ndarray:
    """One valid local face slice for loader compatibility; it is non-evidential."""
    if vertex_count <= 0:
        raise RuntimeError("placeholder face requested for empty vertex block")
    if vertex_count == 1:
        return np.asarray([[0, 0, 0]], dtype=np.int32)
    if vertex_count == 2:
        return np.asarray([[0, 1, 1]], dtype=np.int32)
    return np.asarray([[0, 1, 2]], dtype=np.int32)


def nearest_neighbor_spacing(vertices: np.ndarray) -> dict[str, Any]:
    if len(vertices) < 2:
        return {"count": 0}
    tree = cKDTree(vertices)
    distances, _ids = tree.query(vertices, k=2)
    # k=2 returns self at column 0, nearest other sample at column 1.
    return summarize(distances[:, 1])


def build_blocks(frames: list[Any], selector: str | int | None) -> list[FrameSurfelBlock]:
    blocks: list[FrameSurfelBlock] = []
    for frame in frames:
        if not isinstance(frame, dict) or "frame_idx" not in frame:
            continue
        obj = select_object(frame, selector)
        if obj is None:
            continue
        pts = visible_surfel_vertices(obj)
        if pts is None:
            continue
        blocks.append(
            FrameSurfelBlock(
                frame_idx=int(frame["frame_idx"]),
                vertices=pts.astype(np.float64),
                faces=placeholder_face(len(pts)),
                selected_object_id=str(obj.get("object_id") or obj.get("track_id") or obj.get("id") or "") or None,
            )
        )
    by_frame = {block.frame_idx: block for block in blocks}
    return [by_frame[idx] for idx in sorted(by_frame)]


def required_frames_from_npz(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    blob = np.load(path)
    if "frame_idx" not in blob.files:
        raise RuntimeError(f"{path}: missing frame_idx")
    return [int(v) for v in np.asarray(blob["frame_idx"], dtype=np.int64).tolist()]


def validate_required_frames(blocks: list[FrameSurfelBlock], required: list[int] | None) -> dict[str, Any]:
    present = {block.frame_idx for block in blocks}
    if required is None:
        return {"checked": False}
    missing = [idx for idx in required if idx not in present]
    extra = [idx for idx in sorted(present) if idx not in set(required)]
    return {
        "checked": True,
        "required_frame_count": int(len(required)),
        "missing_required_frames": missing,
        "extra_archive_frames": extra,
        "matched": not missing,
    }


def write_npz(blocks: list[FrameSurfelBlock], path: Path) -> None:
    if not blocks:
        raise RuntimeError("no visible surfel sample frames found")
    frame_idx = np.asarray([block.frame_idx for block in blocks], dtype=np.int64)
    vertex_offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    face_offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    for i, block in enumerate(blocks):
        vertex_offsets[i + 1] = vertex_offsets[i] + int(block.vertices.shape[0])
        face_offsets[i + 1] = face_offsets[i] + int(block.faces.shape[0])
    vertices = np.concatenate([block.vertices for block in blocks], axis=0).astype(np.float64)
    faces = np.concatenate([block.faces for block in blocks], axis=0).astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame_idx=frame_idx,
        vertex_offsets=vertex_offsets,
        face_offsets=face_offsets,
        vertices=vertices,
        faces=faces,
        archive_kind=np.asarray(ARCHIVE_KIND),
        face_policy=np.asarray(FACE_POLICY),
        faces_non_evidential=np.asarray(True),
    )


def write_report(
    *,
    blocks: list[FrameSurfelBlock],
    annotations: Path,
    output_npz: Path,
    output_report: Path,
    object_selector: str | int | None,
    required_frame_validation: dict[str, Any],
    distance_threshold_m: float,
) -> dict[str, Any]:
    vertex_counts = np.asarray([len(block.vertices) for block in blocks], dtype=np.int64)
    spacing_rows = [nearest_neighbor_spacing(block.vertices) for block in blocks]
    spacing_medians = np.asarray([row.get("median", np.nan) for row in spacing_rows], dtype=np.float64)
    threshold = float(distance_threshold_m)
    report = {
        "status": "ok",
        "method": "build_visible_surfel_compat_archive",
        "archive_kind": ARCHIVE_KIND,
        "claim_scope": "visible metric surfel sample compatibility archive; diagnostic input only; not object mesh or pose evidence",
        "provenance": PROVENANCE,
        "allowed_claims": ALLOWED_CLAIMS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "annotations": str(annotations),
        "object_selector": object_selector,
        "output_npz": str(output_npz),
        "frame_count": int(len(blocks)),
        "first_frame": int(blocks[0].frame_idx),
        "last_frame": int(blocks[-1].frame_idx),
        "face_policy": FACE_POLICY,
        "faces_non_evidential": True,
        "face_count_total": int(sum(len(block.faces) for block in blocks)),
        "vertex_count": {
            "total": int(vertex_counts.sum()),
            "per_frame_min": int(vertex_counts.min()),
            "per_frame_median": float(np.median(vertex_counts)),
            "per_frame_max": int(vertex_counts.max()),
        },
        "nearest_neighbor_sample_spacing_m": {
            "per_frame_median_spacing_summary": summarize(spacing_medians),
            "distance_threshold_m": threshold,
            "threshold_interpretation": (
                "If spacing is comparable to or larger than the threshold, downstream filtering is a sample-proximity gate, "
                "not a true continuous surface-membership test."
            ),
            "frames_with_median_spacing_gt_threshold": int(np.count_nonzero(spacing_medians > threshold)),
            "frames_with_median_spacing_gt_half_threshold": int(np.count_nonzero(spacing_medians > 0.5 * threshold)),
        },
        "required_frame_validation": required_frame_validation,
        "frames": [
            {
                "frame_idx": int(block.frame_idx),
                "object_id": block.selected_object_id,
                "vertices": int(len(block.vertices)),
                "faces": int(len(block.faces)),
                "face_policy": FACE_POLICY,
                "nearest_neighbor_spacing_m": spacing_rows[i],
            }
            for i, block in enumerate(blocks)
        ],
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True, help="Prediction annotations JSON containing frames[].objects[].visible_geometry_candidate.world_vertices_sample_m")
    parser.add_argument("--object-id", default=None, help="Object index or object_id/track_id/label; defaults to objects[0].")
    parser.add_argument("--output-npz", type=Path, required=True, help="Output loader-compatible NPZ containing visible surfel samples as vertices.")
    parser.add_argument("--output-report", type=Path, required=True, help="Output provenance and validation report JSON.")
    parser.add_argument("--required-frame-npz", type=Path, help="Optional CoTracker NPZ whose frame_idx values must be present in this archive.")
    parser.add_argument("--distance-threshold-m", type=float, default=0.004, help="Downstream nearest visible-sample threshold to contextualize sample spacing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector = parse_object_selector(args.object_id)
    data = load_json(args.annotations)
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{args.annotations}: missing frames list")
    blocks = build_blocks(frames, selector)
    required = required_frames_from_npz(args.required_frame_npz)
    required_validation = validate_required_frames(blocks, required)
    if required_validation.get("checked") and not required_validation.get("matched"):
        raise RuntimeError(f"archive frame mismatch: {required_validation}")
    write_npz(blocks, args.output_npz)
    report = write_report(
        blocks=blocks,
        annotations=args.annotations,
        output_npz=args.output_npz,
        output_report=args.output_report,
        object_selector=selector,
        required_frame_validation=required_validation,
        distance_threshold_m=float(args.distance_threshold_m),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
