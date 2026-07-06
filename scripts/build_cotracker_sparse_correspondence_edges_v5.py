#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive


def summarize(values: np.ndarray) -> dict:
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
    }


def track_max_visible_sample_distance(distances: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    out = np.full((accepted.shape[1],), np.nan, dtype=np.float64)
    for track_id in range(accepted.shape[1]):
        vals = distances[accepted[:, track_id], track_id]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[track_id] = float(np.max(vals))
    return out


def run(args: argparse.Namespace) -> dict:
    tracks = np.load(args.cotracker_npz)
    required = {"frame_idx", "accepted", "world_xyz"}
    missing = required.difference(tracks.files)
    if missing:
        raise RuntimeError(f"CoTracker archive missing keys: {sorted(missing)}")
    frame_idx = np.asarray(tracks["frame_idx"], dtype=np.int64)
    accepted = np.asarray(tracks["accepted"], dtype=bool)
    world = np.asarray(tracks["world_xyz"], dtype=np.float64)
    if world.shape[:2] != accepted.shape:
        raise RuntimeError(f"world/accepted shape mismatch: {world.shape} vs {accepted.shape}")

    # The loader name is historical. For R2 visible-surfel compatibility archives, faces
    # are non-evidential padding and this script uses only vertices for nearest-neighbor
    # filtering against prediction-side visible metric surfel samples.
    visible_samples = load_mesh_archive(args.mesh_archive)
    if any(int(frame) not in visible_samples for frame in frame_idx.tolist()):
        raise RuntimeError("visible-sample archive missing frames in CoTracker archive")

    nearest_vertex = np.full(accepted.shape, -1, dtype=np.int64)
    nearest_visible_sample_distance = np.full(accepted.shape, np.nan, dtype=np.float64)
    for i, frame in enumerate(frame_idx.tolist()):
        vertices, _non_evidential_faces = visible_samples[int(frame)]
        tree = cKDTree(vertices)
        valid = accepted[i] & np.all(np.isfinite(world[i]), axis=1)
        if not np.any(valid):
            continue
        distances, ids = tree.query(world[i, valid], k=1)
        take = np.where(valid)[0]
        nearest_vertex[i, take] = ids.astype(np.int64)
        nearest_visible_sample_distance[i, take] = distances.astype(np.float64)

    support = accepted.sum(axis=0)
    max_sample_distance = track_max_visible_sample_distance(nearest_visible_sample_distance, accepted)
    sample_proximity_ok = np.isfinite(max_sample_distance) & (max_sample_distance <= float(args.max_visible_sample_distance_m))
    support_ok = support >= int(args.min_track_frames)
    usable_track = support_ok & sample_proximity_ok
    all_frame_track = usable_track & (support == len(frame_idx))

    edge_rows = []
    edge_steps = []
    edge_visible_sample_distances = []
    for track_id in np.where(usable_track)[0].tolist():
        valid_frames = np.where(accepted[:, track_id])[0]
        for a, b in zip(valid_frames[:-1], valid_frames[1:]):
            if int(frame_idx[b] - frame_idx[a]) > int(args.max_frame_gap):
                continue
            step = float(np.linalg.norm(world[b, track_id] - world[a, track_id]))
            if step > float(args.max_world_step_m):
                continue
            source_distance = float(nearest_visible_sample_distance[a, track_id])
            target_distance = float(nearest_visible_sample_distance[b, track_id])
            edge_rows.append(
                {
                    "track_id": int(track_id),
                    "source_frame": int(frame_idx[a]),
                    "target_frame": int(frame_idx[b]),
                    "source_visible_sample_vertex": int(nearest_vertex[a, track_id]),
                    "target_visible_sample_vertex": int(nearest_vertex[b, track_id]),
                    "world_step_m": step,
                    "source_nearest_visible_sample_distance_m": source_distance,
                    "target_nearest_visible_sample_distance_m": target_distance,
                }
            )
            edge_steps.append(step)
            edge_visible_sample_distances.extend([source_distance, target_distance])

    per_frame_rows = []
    for i, frame in enumerate(frame_idx.tolist()):
        per_frame_rows.append(
            {
                "frame_idx": int(frame),
                "accepted_tracks": int(np.count_nonzero(accepted[i])),
                "usable_tracks_visible_sample_proximity": int(np.count_nonzero(usable_track & accepted[i])),
                "all_frame_tracks_visible_sample_proximity": int(np.count_nonzero(all_frame_track & accepted[i])),
                "nearest_visible_sample_distance_m": summarize(nearest_visible_sample_distance[i, accepted[i]]),
            }
        )

    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "build_cotracker_sparse_correspondence_edges_v5",
        "claim_tested": (
            "CoTracker world-space point tracks can provide candidate sparse correspondence edges "
            "whose tracked points remain close to prediction-side visible metric surfel samples. "
            "This is a diagnostic sample-proximity test, not an object mesh, pose, contact, "
            "occlusion, or nonpenetration claim."
        ),
        "terminology_note": (
            "Distances are nearest visible-surface sample distances. If visible-sample spacing is "
            "comparable to the threshold, the filter is a sample-proximity gate rather than a "
            "continuous surface-membership test. Faces in a visible-surfel compatibility archive "
            "are ignored by this script."
        ),
        "cotracker_npz": str(args.cotracker_npz),
        "visible_surfel_archive": str(args.mesh_archive),
        "frames": [int(frame) for frame in frame_idx.tolist()],
        "track_count": int(accepted.shape[1]),
        "usable_track_count": int(np.count_nonzero(usable_track)),
        "all_frame_usable_track_count": int(np.count_nonzero(all_frame_track)),
        "edge_count": int(len(edge_rows)),
        "valid_frames_per_track": summarize(support),
        "usable_valid_frames_per_track": summarize(support[usable_track]),
        "nearest_visible_sample_distance_m": summarize(nearest_visible_sample_distance[accepted]),
        "usable_nearest_visible_sample_distance_m": (
            summarize(nearest_visible_sample_distance[:, usable_track][accepted[:, usable_track]]) if np.any(usable_track) else {"count": 0}
        ),
        "edge_world_step_m": summarize(np.asarray(edge_steps, dtype=np.float64)),
        "edge_nearest_visible_sample_distance_m": summarize(np.asarray(edge_visible_sample_distances, dtype=np.float64)),
        "per_frame_rows": per_frame_rows,
        "parameters": {
            "min_track_frames": int(args.min_track_frames),
            "max_visible_sample_distance_m": float(args.max_visible_sample_distance_m),
            "max_world_step_m": float(args.max_world_step_m),
            "max_frame_gap": int(args.max_frame_gap),
        },
        "edges": edge_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "edges"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotracker-npz", type=Path, required=True)
    parser.add_argument(
        "--mesh-archive",
        "--visible-surfel-archive",
        dest="mesh_archive",
        type=Path,
        required=True,
        help="Loader-compatible archive whose vertices are prediction-side visible metric surfel samples; faces are ignored.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-track-frames", type=int, default=4)
    parser.add_argument(
        "--max-visible-sample-distance-m",
        "--max-surface-distance-m",
        dest="max_visible_sample_distance_m",
        type=float,
        default=0.004,
        help="Maximum nearest visible-sample distance. The --max-surface-distance-m spelling is a deprecated alias.",
    )
    parser.add_argument("--max-world-step-m", type=float, default=0.040)
    parser.add_argument("--max-frame-gap", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
