#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from align_mesh_prior_v3 import choose_alignment, sample_mesh_surface
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def select_frames(meshes: dict[int, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> list[int]:
    available = sorted(meshes)
    if args.frames:
        requested = [int(part) for raw in args.frames for part in raw.split(",") if part.strip()]
    else:
        lo = min(available) if args.frame_start is None else int(args.frame_start)
        hi = max(available) if args.frame_end is None else int(args.frame_end)
        requested = [idx for idx in available if lo <= idx <= hi]
    missing = [idx for idx in requested if idx not in meshes]
    if missing:
        raise RuntimeError(f"observed mesh archive lacks requested frames: {missing}")
    if not requested:
        raise RuntimeError("no frames selected")
    return requested


def write_mesh_archive(path: Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    if not rows:
        raise RuntimeError("no aligned meshes to write")
    frame_idx = []
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame, vertices, faces in rows:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise RuntimeError(f"invalid vertices for frame {frame}: {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise RuntimeError(f"invalid faces for frame {frame}: {faces.shape}")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise RuntimeError(f"face index out of range for frame {frame}")
        frame_idx.append(int(frame))
        vertices_all.append(vertices.astype(np.float32))
        faces_all.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.concatenate(vertices_all, axis=0),
        faces=np.concatenate(faces_all, axis=0),
    )


def transform_row(prior_mesh: trimesh.Trimesh, observed_vertices: np.ndarray, observed_faces: np.ndarray, frame_idx: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    observed_mesh = trimesh.Trimesh(vertices=observed_vertices, faces=observed_faces, process=False)
    prior_points = sample_mesh_surface(prior_mesh, int(args.samples), int(args.seed) + int(frame_idx))
    observed_count = min(int(args.samples), max(int(args.samples) // 2, len(observed_faces) * 4))
    observed_points = sample_mesh_surface(observed_mesh, observed_count, int(args.seed) + int(frame_idx) + 100_000)
    sim, alignment = choose_alignment(prior_points, observed_points)
    transformed = sim.apply(np.asarray(prior_mesh.vertices, dtype=np.float64))
    selected = alignment.get("selected", {})
    p95 = [
        float(selected.get("p95_prior_to_observed_m", np.nan)),
        float(selected.get("p95_observed_to_prior_m", np.nan)),
    ]
    row = {
        "frame_idx": int(frame_idx),
        "observed_vertices": int(len(observed_vertices)),
        "observed_faces": int(len(observed_faces)),
        "prior_vertices": int(len(prior_mesh.vertices)),
        "prior_faces": int(len(prior_mesh.faces)),
        "selected_alignment": selected,
        "candidate_count": int(len(alignment.get("candidates", []))),
        "bidirectional_p95_m": float(np.nanmax(p95)),
        "sim3": {
            "scale": float(sim.scale),
            "rotation": np.asarray(sim.rotation, dtype=np.float64).tolist(),
            "translation_m": np.asarray(sim.translation, dtype=np.float64).tolist(),
        },
    }
    return transformed, row


def run(args: argparse.Namespace) -> dict:
    prior_mesh = trimesh.load(args.mesh_prior, force="mesh", process=False)
    if not isinstance(prior_mesh, trimesh.Trimesh):
        raise RuntimeError(f"prior path did not load as a triangle mesh: {args.mesh_prior}")
    if len(prior_mesh.vertices) == 0 or len(prior_mesh.faces) == 0:
        raise RuntimeError(f"prior mesh has no triangle surface: {args.mesh_prior}")
    observed_meshes = load_mesh_archive(args.observed_mesh_archive)
    frames = select_frames(observed_meshes, args)

    rows = []
    archive_rows = []
    faces = np.asarray(prior_mesh.faces, dtype=np.int32)
    for frame_idx in frames:
        observed_vertices, observed_faces = observed_meshes[frame_idx]
        aligned_vertices, row = transform_row(prior_mesh, observed_vertices, observed_faces, frame_idx, args)
        rows.append(row)
        archive_rows.append((frame_idx, aligned_vertices, faces))

    write_mesh_archive(args.output_mesh_archive, archive_rows)
    p95 = np.asarray([row["bidirectional_p95_m"] for row in rows], dtype=np.float64)
    scales = np.asarray([row["sim3"]["scale"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "archive_aligned_mesh_prior_v7",
        "claim_tested": "a generated object mesh prior can be aligned to measured per-frame object surfaces and then replayed through the standard mask/depth/contact checks",
        "mesh_prior": str(args.mesh_prior),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "output_mesh_archive": str(args.output_mesh_archive),
        "frame_count": int(len(frames)),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "alignment_bidirectional_p95_m": summarize(p95),
        "alignment_scale": summarize(scales),
        "acceptance_policy": {
            "alignment_pass": bool(np.all(p95 <= float(args.max_bidirectional_p95_m))),
            "max_bidirectional_p95_m": float(args.max_bidirectional_p95_m),
            "delivery_requires_downstream_replay": [
                "z-buffer silhouette and depth",
                "mesh-surface contact",
                "selected-contact SDF",
                "full-hand SDF",
                "visual review",
            ],
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frames", action="append", default=[])
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7807)
    parser.add_argument("--max-bidirectional-p95-m", type=float, default=0.010)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
