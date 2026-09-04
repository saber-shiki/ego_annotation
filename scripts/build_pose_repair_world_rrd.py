#!/usr/bin/env python3
"""Build an isolated world-coordinate Rerun view for repaired object pose + P09 points."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--label", default="repaired_global_image")
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    pose_rows = {
        int(row["frame_idx"]): row
        for row in pose_report.get("pose_rows", [])
        if isinstance(row, dict)
        and row.get("rotation_world_from_completed_canonical_matrix") is not None
        and row.get("translation_world_m") is not None
    }
    mesh = trimesh.load(args.observed_mesh.expanduser().resolve(), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid observed mesh: {args.observed_mesh}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    timeline = list(range(min(frames), max(frames) + 1)) if frames else []
    if not timeline or any(idx not in pose_rows for idx in timeline):
        missing = [idx for idx in timeline if idx not in pose_rows]
        raise RuntimeError(f"pose report does not cover complete timeline; missing={missing[:12]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"milk_pose_repair_world_{args.label}", spawn=False)
    rr.save(str(args.output.expanduser().resolve()))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "/world/object/observed",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
            albedo_factor=[60, 200, 90, 190],
        ),
        static=True,
    )

    metric_frames: list[int] = []
    empty_frames: list[int] = []
    point_counts: list[int] = []
    for frame_idx in timeline:
        rr.set_time("frame", sequence=frame_idx)
        row = pose_rows[frame_idx]
        rotation = np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
        translation = np.asarray(row["translation_world_m"], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise RuntimeError(f"invalid pose row at frame {frame_idx}")
        rr.log("/world/object", rr.Transform3D(translation=translation.tolist(), mat3x3=rotation.tolist()))

        frame = frames[frame_idx]
        obj = next((candidate for candidate in frame.get("objects", []) if candidate.get("object_id") == args.object_id), None)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        points = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 3 and len(points) > 0 and np.isfinite(points).all():
            metric_frames.append(frame_idx)
            point_counts.append(len(points))
            rr.log(
                "/world/visible_surface",
                rr.Points3D(
                    points,
                    radii=0.0018,
                    colors=[255, 220, 0, 255],
                    point_shading=rr.components.PointShading.Flat,
                ),
            )
        else:
            empty_frames.append(frame_idx)
            rr.log("/world/visible_surface", rr.Clear(recursive=False))
            rr.log("/world/visible_surface/status", rr.TextLog("no accepted metric surface at this frame"))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="World object + visible surface (points only)",
                contents=["+ /world/visible_surface", "+ /world/visible_surface/**", "- /world/object/**"],
            ),
            rrb.Spatial3DView(
                origin="/world",
                name="World object mesh + visible surface overlay",
                contents=["+ /world/object/**", "+ /world/visible_surface", "+ /world/visible_surface/**"],
            ),
            column_shares=[1, 1],
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    report = {
        "schema": "v19_pose_repair_world_rrd_v1",
        "status": "ok",
        "diagnostic_only": True,
        "label": args.label,
        "rrd": str(args.output.expanduser().resolve()),
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "pose_report": str(args.pose_report.expanduser().resolve()),
            "observed_mesh": str(args.observed_mesh.expanduser().resolve()),
            "object_id": args.object_id,
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "pose_report": sha256_file(args.pose_report),
                "observed_mesh": sha256_file(args.observed_mesh),
            },
        },
        "timeline": {"frame_count": len(timeline), "first_frame": timeline[0], "last_frame": timeline[-1]},
        "layers": {
            "world_object_observed_mesh": {
                "entity": "/world/object/observed",
                "canonical_mesh": True,
                "parent_transform": "/world/object",
                "color": "green",
                "authority": "observed-only P13/P15 pose body; not generated hidden geometry",
            },
            "world_visible_surface_points": {
                "entity": "/world/visible_surface",
                "source": "corrected P09 world_vertices_sample_m",
                "color": "yellow",
                "points_per_metric_frame": 2500,
                "metric_frame_count": len(metric_frames),
                "metric_frames": metric_frames,
                "no_metric_frames": empty_frames,
            },
        },
        "coordinate_contract": {
            "point_formula": "P_world = P_camera @ R_world_camera.T + t_world_camera",
            "object_formula": "V_world = V_canonical @ R_world_from_completed_canonical.T + t_world",
            "generated_geometry_consumed": False,
        },
        "views": {
            "points_only": "Use this view to inspect every visible-surface point without mesh depth occlusion.",
            "overlay": "Supplemental mesh/point alignment view; normal 3D depth testing may apply.",
        },
        "claim_scope": "Visualization only. Visible points are prediction-side corrected P09 first-hit measurements, not GT; the observed mesh is an open canonical surface and is not a complete object.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rrd": str(args.output), "report": str(report_path), "metric_frames": len(metric_frames), "empty_frames": empty_frames, "point_count_summary": {"min": min(point_counts), "max": max(point_counts)}}, indent=2))


if __name__ == "__main__":
    main()
