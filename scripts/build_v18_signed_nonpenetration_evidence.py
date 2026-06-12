#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_signed_nonpenetration_evidence"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def load_v16(case: str, args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    ann_path = args.v16_root / case / "annotations_v16_full.json"
    ann = load_json(ann_path)
    frames = {int(frame["frame_idx"]): frame for frame in ann.get("frames", []) if isinstance(frame, dict) and isinstance(frame.get("frame_idx"), int)}
    mesh_archive = None
    for frame in frames.values():
        obj_raw = frame.get("object")
        obj: dict[str, Any] = obj_raw if isinstance(obj_raw, dict) else {}
        if obj.get("mesh_archive"):
            mesh_archive = Path(str(obj["mesh_archive"]))
            break
    if mesh_archive is None or not mesh_archive.exists():
        return frames, {"ann_path": str(ann_path), "mesh_archive": None, "data": None, "frame_rows": {}}
    data = np.load(mesh_archive, allow_pickle=True)
    return frames, {"ann_path": str(ann_path), "mesh_archive": str(mesh_archive), "data": data, "frame_rows": {int(f): int(i) for i, f in enumerate(data["frame_idx"])}}


def frame_mesh(mesh_index: dict[str, Any], frame_idx: int) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    data = mesh_index.get("data")
    rows = mesh_index.get("frame_rows")
    if data is None or not isinstance(rows, dict):
        return None, None, "missing_mesh_archive"
    row_idx = rows.get(frame_idx)
    if row_idx is None:
        return None, None, "missing_mesh_frame"
    v0, v1 = int(data["vertex_offsets"][row_idx]), int(data["vertex_offsets"][row_idx + 1])
    f0, f1 = int(data["face_offsets"][row_idx]), int(data["face_offsets"][row_idx + 1])
    vertices = np.asarray(data["vertices"][v0:v1], dtype=np.float64)
    faces = np.asarray(data["faces"][f0:f1], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or len(vertices) == 0 or len(faces) == 0:
        return None, None, "invalid_mesh_frame"
    if faces.max(initial=-1) >= len(vertices):
        # Archives may store global face indices. Normalize to the frame-local vertex slice if needed.
        faces = faces - v0
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid]
    if len(faces) == 0:
        return None, None, "invalid_mesh_face_indices"
    return vertices, faces, None


def hand_points(v16_frame: dict[str, Any], hand_side: str, max_points: int) -> tuple[np.ndarray | None, str | None]:
    for hand in v16_frame.get("hands", []):
        if not isinstance(hand, dict):
            continue
        if str(hand.get("side", hand.get("hand_side"))) != hand_side:
            continue
        pts_raw = hand.get("vertices_world_m") or hand.get("joints3d_world_m")
        if not isinstance(pts_raw, list) or not pts_raw:
            return None, "missing_hand_points"
        pts = np.asarray(pts_raw, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            return None, "invalid_hand_points"
        if len(pts) > max_points:
            step = max(1, int(math.ceil(len(pts) / max_points)))
            pts = pts[::step]
        return pts, None
    return None, "missing_hand_side"


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-9
    normals[valid] /= norms[valid, None]
    centroids = centroids[valid]
    normals = normals[valid]
    mesh_center = vertices.mean(axis=0)
    outward = np.sum(normals * (centroids - mesh_center), axis=1)
    flip = outward < 0.0
    normals[flip] *= -1.0
    return centroids, normals


def signed_stats(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, max_query_points: int) -> dict[str, Any]:
    if len(points) > max_query_points:
        step = max(1, int(math.ceil(len(points) / max_query_points)))
        points = points[::step]
    centroids, normals = face_geometry(vertices, faces)
    if len(centroids) == 0:
        return {"blocker": "invalid_face_normals"}
    tree = cKDTree(centroids)
    _, face_idx = tree.query(points, k=1)
    nearest_centroids = centroids[np.asarray(face_idx, dtype=np.int64)]
    nearest_normals = normals[np.asarray(face_idx, dtype=np.int64)]
    signed = np.sum((points - nearest_centroids) * nearest_normals, axis=1)
    abs_signed = np.abs(signed)
    negative = signed < 0.0
    return {
        "sampled_hand_points": int(len(points)),
        "mesh_face_count": int(len(faces)),
        "mesh_vertex_count": int(len(vertices)),
        "min_local_signed_distance_m": float(np.min(signed)),
        "median_local_signed_distance_m": float(np.median(signed)),
        "min_abs_local_signed_distance_m": float(np.min(abs_signed)),
        "negative_signed_distance_count": int(np.sum(negative)),
        "negative_signed_distance_fraction": float(np.mean(negative)),
        "local_signed_distance_semantics": "nearest_face_centroid_normal_projection_not_watertight_sdf",
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    contact_path = args.contact_ownership_root / case / "v18_contact_ownership_graph_report.json"
    contact = load_json(contact_path)
    v16_frames, mesh_index = load_v16(case, args)
    mesh_cache: dict[int, tuple[np.ndarray | None, np.ndarray | None, str | None]] = {}
    hand_cache: dict[tuple[int, str], tuple[np.ndarray | None, str | None]] = {}
    rows: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    for raw in contact.get("rows", []):
        if not isinstance(raw, dict) or raw.get("accepted_contact_owner") is not True:
            continue
        frame_idx = raw.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        hand_side = str(raw.get("hand_side"))
        mesh = mesh_cache.get(frame_idx)
        if mesh is None:
            mesh = frame_mesh(mesh_index, frame_idx)
            mesh_cache[frame_idx] = mesh
        vertices, faces, mesh_blocker = mesh
        hand_key = (frame_idx, hand_side)
        hand = hand_cache.get(hand_key)
        if hand is None:
            hand = hand_points(v16_frames.get(frame_idx, {}), hand_side, args.max_hand_points)
            hand_cache[hand_key] = hand
        points, hand_blocker = hand
        row = {
            "frame_idx": frame_idx,
            "hand_side": hand_side,
            "object_id": raw.get("object_id"),
            "source_contact_owner_claim": raw.get("contact_owner_claim"),
            "source_min_unsigned_distance_m": raw.get("min_hand_surface_to_v16_object_mesh_m"),
            "v16_mesh_match": raw.get("v16_mesh_match"),
            "signed_nonpenetration_claim": "not_evaluated",
            "signed_nonpenetration_complete": False,
        }
        if mesh_blocker or hand_blocker or vertices is None or faces is None or points is None:
            blocker = mesh_blocker or hand_blocker or "missing_geometry"
            blockers[str(blocker)] = blockers.get(str(blocker), 0) + 1
            row.update({"blocker": blocker, "signed_nonpenetration_claim": "blocked"})
            rows.append(row)
            continue
        stats = signed_stats(points, vertices, faces, args.max_query_hand_points)
        if stats.get("blocker"):
            blocker = str(stats["blocker"])
            blockers[blocker] = blockers.get(blocker, 0) + 1
            row.update({"blocker": blocker, "signed_nonpenetration_claim": "blocked"})
            rows.append(row)
            continue
        min_signed = finite_float(stats.get("min_local_signed_distance_m"), 0.0)
        penetration = min_signed < -args.penetration_tolerance_m
        row.update(
            {
                **stats,
                "penetration_tolerance_m": args.penetration_tolerance_m,
                "local_penetration_detected": penetration,
                "signed_nonpenetration_claim": "local_normal_penetration_evidence" if penetration else "local_normal_no_penetration_beyond_tolerance_evidence",
                "signed_nonpenetration_complete": False,
            }
        )
        rows.append(row)
    penetration_rows = sum(1 for row in rows if row.get("local_penetration_detected") is True)
    evaluated_rows = sum(1 for row in rows if row.get("signed_nonpenetration_claim") in {"local_normal_penetration_evidence", "local_normal_no_penetration_beyond_tolerance_evidence"})
    out = {
        "method": "build_v18_signed_nonpenetration_evidence",
        "status": STATUS,
        "claim": "Computes local signed hand-object distance evidence for accepted contact-owner rows using V16 object mesh face normals. This is nearest-face normal projection evidence, not a watertight SDF or complete nonpenetration solve.",
        "case": case,
        "sources": {"contact_ownership_graph": str(contact_path), "v16_annotations": str(mesh_index.get("ann_path")), "v16_object_mesh_archive": mesh_index.get("mesh_archive")},
        "accepted_contact_rows": int(contact.get("contact_ownership_accepted_rows", 0)),
        "signed_rows": len(rows),
        "evaluated_signed_rows": evaluated_rows,
        "local_penetration_detected_rows": penetration_rows,
        "blocker_counts": dict(sorted(blockers.items())),
        "parameters": {"max_hand_points": args.max_hand_points, "max_query_hand_points": args.max_query_hand_points, "penetration_tolerance_m": args.penetration_tolerance_m},
        "rows": rows,
        "signed_nonpenetration_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_signed_nonpenetration_evidence_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_signed_nonpenetration_evidence",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {"case": r["case"], "signed_rows": r["signed_rows"], "evaluated_signed_rows": r["evaluated_signed_rows"], "local_penetration_detected_rows": r["local_penetration_detected_rows"], "signed_nonpenetration_complete": r["signed_nonpenetration_complete"]}
            for r in reports
        ],
        "claim_scope": "local_signed_normal_evidence_not_complete_nonpenetration",
    }
    write_json(args.output_root / "v18_signed_nonpenetration_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-ownership-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-hand-points", type=int, default=256)
    parser.add_argument("--max-query-hand-points", type=int, default=128)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
