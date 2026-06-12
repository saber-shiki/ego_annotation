#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_triangle_nonpenetration_evidence"


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
    return frames, {"ann_path": str(ann_path), "mesh_archive": str(mesh_archive), "data": data, "frame_rows": {int(f): int(i) for i, f in enumerate(data["frame_idx"])} }


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
        faces = faces - v0
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid]
    if len(faces) == 0:
        return None, None, "invalid_mesh_face_indices"
    return vertices, faces, None


def hand_points(v16_frame: dict[str, Any], hand_side: str, max_points: int) -> tuple[np.ndarray | None, str | None, str | None]:
    for hand in v16_frame.get("hands", []):
        if not isinstance(hand, dict):
            continue
        if str(hand.get("side", hand.get("hand_side"))) != hand_side:
            continue
        pts_raw = hand.get("vertices_world_m") or hand.get("joints3d_world_m")
        source = "vertices_world_m" if hand.get("vertices_world_m") else "joints3d_world_m"
        if not isinstance(pts_raw, list) or not pts_raw:
            return None, "missing_hand_points", None
        pts = np.asarray(pts_raw, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            return None, "invalid_hand_points", None
        if len(pts) > max_points:
            step = max(1, int(math.ceil(len(pts) / max_points)))
            pts = pts[::step]
        return pts, None, source
    return None, "missing_hand_side", None


def mesh_edge_diagnostics(faces: np.ndarray) -> dict[str, Any]:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    return {
        "unique_edge_count": int(len(unique)),
        "boundary_edge_count": boundary,
        "nonmanifold_edge_count": nonmanifold,
        "mesh_watertight_by_edges": bool(boundary == 0 and nonmanifold == 0),
    }


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-10
    tri = tri[valid]
    centroids = centroids[valid]
    normals = normals[valid] / norms[valid, None]
    mesh_center = vertices.mean(axis=0)
    outward = np.sum(normals * (centroids - mesh_center), axis=1)
    normals[outward < 0.0] *= -1.0
    face_ids = np.nonzero(valid)[0]
    return tri, centroids, normals, face_ids


def closest_points_on_triangles(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ab = b - a
    ac = c - a
    ap = point[None, :] - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    v = vb / denom_safe
    w = vc / denom_safe
    out = a + ab * v[:, None] + ac * w[:, None]
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    out[mask_a] = a[mask_a]
    mask_b = (d3 >= 0.0) & (d4 <= d3)
    out[mask_b] = b[mask_b]
    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    vab = d1 / np.where(np.abs(d1 - d3) > 1e-12, d1 - d3, 1.0)
    out[mask_ab] = a[mask_ab] + ab[mask_ab] * vab[mask_ab, None]
    mask_c = (d6 >= 0.0) & (d5 <= d6)
    out[mask_c] = c[mask_c]
    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    wac = d2 / np.where(np.abs(d2 - d6) > 1e-12, d2 - d6, 1.0)
    out[mask_ac] = a[mask_ac] + ac[mask_ac] * wac[mask_ac, None]
    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    wbc = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-12, (d4 - d3) + (d5 - d6), 1.0)
    out[mask_bc] = b[mask_bc] + (c[mask_bc] - b[mask_bc]) * wbc[mask_bc, None]
    return out


def triangle_signed_stats(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if len(points) > args.max_query_hand_points:
        step = max(1, int(math.ceil(len(points) / args.max_query_hand_points)))
        points = points[::step]
    tri, centroids, normals, face_ids = face_geometry(vertices, faces)
    if len(tri) == 0:
        return {"blocker": "invalid_triangle_normals"}
    diagnostics = mesh_edge_diagnostics(faces)
    k = min(args.nearest_triangle_candidates, len(tri))
    tree = cKDTree(centroids)
    _, raw_indices = tree.query(points, k=k)
    indices = np.atleast_2d(raw_indices)
    if indices.shape[0] != len(points):
        indices = indices.T
    signed: list[float] = []
    unsigned: list[float] = []
    nearest_face_ids: list[int] = []
    for i, point in enumerate(points):
        cand = np.asarray(indices[i], dtype=np.int64)
        cand_tri = tri[cand]
        closest = closest_points_on_triangles(point, cand_tri)
        vec = point[None, :] - closest
        d2 = np.einsum("ij,ij->i", vec, vec)
        best_local = int(np.argmin(d2))
        best_idx = int(cand[best_local])
        normal = normals[best_idx]
        s = float(np.dot(vec[best_local], normal))
        signed.append(s)
        unsigned.append(float(math.sqrt(max(0.0, float(d2[best_local])))))
        nearest_face_ids.append(int(face_ids[best_idx]))
    signed_arr = np.asarray(signed, dtype=np.float64)
    unsigned_arr = np.asarray(unsigned, dtype=np.float64)
    negative = signed_arr < 0.0
    penetration = signed_arr < -args.penetration_tolerance_m
    return {
        **diagnostics,
        "sampled_hand_points": int(len(points)),
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "nearest_triangle_candidate_count": int(k),
        "min_triangle_unsigned_distance_m": float(np.min(unsigned_arr)),
        "median_triangle_unsigned_distance_m": float(np.median(unsigned_arr)),
        "min_local_triangle_signed_distance_m": float(np.min(signed_arr)),
        "median_local_triangle_signed_distance_m": float(np.median(signed_arr)),
        "negative_triangle_signed_distance_count": int(np.sum(negative)),
        "negative_triangle_signed_distance_fraction": float(np.mean(negative)),
        "local_triangle_penetration_detected": bool(np.any(penetration)),
        "local_triangle_signed_distance_semantics": "closest_point_on_nearest_centroid_triangles_with_centroid_oriented_normals_not_watertight_sdf",
        "nearest_face_ids_sample": nearest_face_ids[: min(8, len(nearest_face_ids))],
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    contact_path = args.contact_ownership_root / case / "v18_contact_ownership_graph_report.json"
    contact = load_json(contact_path)
    v16_frames, mesh_index = load_v16(case, args)
    mesh_cache: dict[int, tuple[np.ndarray | None, np.ndarray | None, str | None]] = {}
    hand_cache: dict[tuple[int, str], tuple[np.ndarray | None, str | None, str | None]] = {}
    rows: list[dict[str, Any]] = []
    blockers: Counter[str] = Counter()
    for raw in contact.get("rows", []):
        if not isinstance(raw, dict) or raw.get("accepted_contact_owner") is not True:
            continue
        frame_idx = raw.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        hand_side = str(raw.get("hand_side"))
        if frame_idx not in mesh_cache:
            mesh_cache[frame_idx] = frame_mesh(mesh_index, frame_idx)
        vertices, faces, mesh_blocker = mesh_cache[frame_idx]
        hand_key = (frame_idx, hand_side)
        if hand_key not in hand_cache:
            hand_cache[hand_key] = hand_points(v16_frames.get(frame_idx, {}), hand_side, args.max_hand_points)
        points, hand_blocker, hand_source = hand_cache[hand_key]
        row = {
            "frame_idx": frame_idx,
            "hand_side": hand_side,
            "object_id": raw.get("object_id"),
            "source_contact_owner_claim": raw.get("contact_owner_claim"),
            "source_min_unsigned_distance_m": raw.get("min_hand_surface_to_v16_object_mesh_m"),
            "v16_mesh_match": raw.get("v16_mesh_match"),
            "triangle_nonpenetration_claim": "not_evaluated",
            "triangle_nonpenetration_complete": False,
        }
        if mesh_blocker or hand_blocker or vertices is None or faces is None or points is None:
            blocker = mesh_blocker or hand_blocker or "missing_geometry"
            blockers[str(blocker)] += 1
            row.update({"blocker": blocker, "triangle_nonpenetration_claim": "blocked"})
            rows.append(row)
            continue
        stats = triangle_signed_stats(points, vertices, faces, args)
        if stats.get("blocker"):
            blocker = str(stats["blocker"])
            blockers[blocker] += 1
            row.update({"blocker": blocker, "triangle_nonpenetration_claim": "blocked"})
            rows.append(row)
            continue
        penetration = bool(stats.get("local_triangle_penetration_detected") is True)
        row.update(
            {
                **stats,
                "hand_geometry_source": hand_source,
                "penetration_tolerance_m": args.penetration_tolerance_m,
                "triangle_nonpenetration_claim": "local_triangle_penetration_evidence" if penetration else "local_triangle_no_penetration_beyond_tolerance_evidence",
                "triangle_nonpenetration_complete": False,
            }
        )
        rows.append(row)
    evaluated = sum(1 for r in rows if r.get("triangle_nonpenetration_claim") in {"local_triangle_penetration_evidence", "local_triangle_no_penetration_beyond_tolerance_evidence"})
    penetration_rows = sum(1 for r in rows if r.get("local_triangle_penetration_detected") is True)
    watertight_rows = sum(1 for r in rows if r.get("mesh_watertight_by_edges") is True)
    out = {
        "method": "build_v18_triangle_nonpenetration_evidence",
        "status": STATUS,
        "claim": "Computes closest-point-to-triangle unsigned distance and local oriented-normal signed evidence for graph-accepted contact-owner rows. Mesh boundary diagnostics are reported; because representative meshes are open, this is not a watertight SDF or complete nonpenetration proof.",
        "case": case,
        "sources": {"contact_ownership_graph": str(contact_path), "v16_annotations": str(mesh_index.get("ann_path")), "v16_object_mesh_archive": mesh_index.get("mesh_archive")},
        "accepted_contact_rows": int(contact.get("contact_ownership_accepted_rows", 0)),
        "triangle_rows": len(rows),
        "evaluated_triangle_rows": evaluated,
        "local_triangle_penetration_detected_rows": penetration_rows,
        "mesh_watertight_rows": watertight_rows,
        "blocker_counts": dict(sorted(blockers.items())),
        "parameters": {"max_hand_points": args.max_hand_points, "max_query_hand_points": args.max_query_hand_points, "nearest_triangle_candidates": args.nearest_triangle_candidates, "penetration_tolerance_m": args.penetration_tolerance_m},
        "rows": rows,
        "triangle_nonpenetration_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_triangle_nonpenetration_evidence_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_triangle_nonpenetration_evidence",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {"case": r["case"], "triangle_rows": r["triangle_rows"], "evaluated_triangle_rows": r["evaluated_triangle_rows"], "local_triangle_penetration_detected_rows": r["local_triangle_penetration_detected_rows"], "mesh_watertight_rows": r["mesh_watertight_rows"], "triangle_nonpenetration_complete": r["triangle_nonpenetration_complete"]}
            for r in reports
        ],
        "claim_scope": "nearest_triangle_local_signed_evidence_not_complete_sdf",
    }
    write_json(args.output_root / "v18_triangle_nonpenetration_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-ownership-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-hand-points", type=int, default=256)
    parser.add_argument("--max-query-hand-points", type=int, default=128)
    parser.add_argument("--nearest-triangle-candidates", type=int, default=32)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
