#!/usr/bin/env python3
"""Build a diagnostic world-coordinate Rerun recording for V20 keyframe poses.

This deliberately produces an experiment artifact, not a formal pipeline
artifact.  It overlays the formal P14 pose with V20 K=5/K=10 poses, the
prediction-side observed mesh, P09 visible surface points, MANO bridge points,
and the camera trajectory.  Missing metric frames are explicitly cleared.
Generated SAM3D completion faces are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path.expanduser().resolve(), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return geom


def pose_rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in report.get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        if row.get("rotation_world_from_completed_canonical_matrix") is None:
            continue
        if row.get("translation_world_m") is None:
            continue
        out[int(row["frame_idx"])] = row
    return out


def log_pose(path: str, row: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    translation = np.asarray(row["translation_world_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise RuntimeError(f"invalid pose row for {path}")
    rr.log(path, rr.Transform3D(translation=translation.tolist(), mat3x3=rotation.tolist()))
    return translation


def find_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--formal-pose-report", type=Path, required=True)
    parser.add_argument("--v20-k5-pose-report", type=Path, required=True)
    parser.add_argument("--v20-k10-pose-report", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--mano-bridge", type=Path, default=None)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--frames", default="all", help="all or comma-separated frame indices")
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="v20_k5_k10_formal")
    args = parser.parse_args()
    if args.point_stride < 1:
        raise RuntimeError("--point-stride must be positive")

    annotations = load_json(args.annotations)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    if not frames:
        raise RuntimeError("annotations contain no frames")
    formal = pose_rows(load_json(args.formal_pose_report))
    k5 = pose_rows(load_json(args.v20_k5_pose_report))
    k10 = pose_rows(load_json(args.v20_k10_pose_report))
    if args.frames == "all":
        selected = sorted(frames)
    else:
        selected = sorted({int(value) for value in args.frames.split(",") if value.strip()})
        unknown = [idx for idx in selected if idx not in frames]
        if unknown:
            raise RuntimeError(f"requested frames absent from annotations: {unknown[:10]}")

    mesh = load_mesh(args.observed_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    hands: dict[int, dict[str, np.ndarray]] = {}
    if args.mano_bridge is not None:
        with np.load(args.mano_bridge.expanduser().resolve(), allow_pickle=False) as bridge:
            required = {
                "frame_idx",
                "hand_side",
                "vertices_current_v18_world_from_hawor_projection_relift_m",
            }
            missing = sorted(required.difference(bridge.files))
            if missing:
                raise RuntimeError(f"MANO bridge missing keys: {missing}")
            bridge_frames = np.asarray(bridge["frame_idx"], dtype=np.int64)
            bridge_sides = np.asarray(bridge["hand_side"])
            bridge_vertices = np.asarray(
                bridge["vertices_current_v18_world_from_hawor_projection_relift_m"],
                dtype=np.float32,
            )
        for idx, side, verts in zip(bridge_frames.tolist(), bridge_sides.tolist(), bridge_vertices):
            side_name = side.decode() if isinstance(side, bytes) else str(side)
            hands.setdefault(int(idx), {})[side_name] = verts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"milk_{args.label}", spawn=False)
    rr.save(str(args.output.expanduser().resolve()))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # One static canonical observed surface, instanced below under each pose.
    rr.log(
        "/world/formal/object_mesh",
        rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, albedo_factor=[220, 60, 60, 150]),
        static=True,
    )
    rr.log(
        "/world/v20_k5/object_mesh",
        rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, albedo_factor=[40, 220, 100, 190]),
        static=True,
    )
    rr.log(
        "/world/v20_k10/object_mesh",
        rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces, albedo_factor=[50, 120, 255, 170]),
        static=True,
    )

    formal_traj: list[np.ndarray] = []
    k5_traj: list[np.ndarray] = []
    k10_traj: list[np.ndarray] = []
    metric_frames: list[int] = []
    no_metric_frames: list[int] = []
    pose_presence = {"formal": [], "v20_k5": [], "v20_k10": []}

    for frame_idx in selected:
        rr.set_time("frame", sequence=frame_idx)
        frame = frames[frame_idx]
        camera = frame.get("camera") or {}
        transform = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"invalid camera transform at frame {frame_idx}")
        rr.log(
            "/world/camera",
            rr.Transform3D(translation=transform[:3, 3].tolist(), mat3x3=transform[:3, :3].tolist()),
        )
        intrinsics = camera.get("intrinsics_fx_fy_cx_cy") or [974.3447265625, 974.3447265625, 702.6607055664062, 706.1102294921875]
        fx, fy, cx, cy = [float(value) for value in intrinsics]
        rr.log(
            "/world/camera/pinhole",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                resolution=[1408, 1408],
                image_plane_distance=0.1,
            ),
        )

        for name, rows, color in (
            ("formal", formal, [220, 60, 60, 255]),
            ("v20_k5", k5, [40, 220, 100, 255]),
            ("v20_k10", k10, [50, 120, 255, 255]),
        ):
            row = rows.get(frame_idx)
            if row is None:
                pose_presence[name].append(False)
                rr.log(f"/world/{name}/object", rr.Clear(recursive=False))
                rr.log(f"/world/{name}/status", rr.TextLog("no pose at this frame"))
                continue
            pose_presence[name].append(True)
            position = log_pose(f"/world/{name}/object", row)
            rr.log(f"/world/{name}/pose_origin", rr.Points3D([position], radii=0.006, colors=color))
            if name == "formal":
                formal_traj.append(position)
            elif name == "v20_k5":
                k5_traj.append(position)
            else:
                k10_traj.append(position)

        obj = find_object(frame, args.object_id)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        points = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 3 and len(points) > 0 and np.isfinite(points).all():
            metric_frames.append(frame_idx)
            rr.log(
                "/world/visible_surface",
                rr.Points3D(
                    points[:: args.point_stride],
                    radii=0.0022,
                    colors=[255, 220, 0, 255],
                    point_shading=rr.components.PointShading.Flat,
                ),
            )
        else:
            no_metric_frames.append(frame_idx)
            rr.log("/world/visible_surface", rr.Clear(recursive=False))
            rr.log("/world/visible_surface/status", rr.TextLog("no accepted P09 metric surface"))

        for side, color in (("left", [255, 150, 40, 255]), ("right", [80, 150, 255, 255])):
            verts = hands.get(frame_idx, {}).get(side)
            if verts is None:
                rr.log(f"/world/hand/{side}", rr.Clear(recursive=False))
                continue
            rr.log(f"/world/hand/{side}", rr.Points3D(verts, radii=0.0025, colors=color))

    # Timeless trajectories make camera/object drift easy to inspect in the 3D view.
    for path, values, color in (
        ("/world/trajectory/formal", formal_traj, [220, 60, 60, 180]),
        ("/world/trajectory/v20_k5", k5_traj, [40, 220, 100, 220]),
        ("/world/trajectory/v20_k10", k10_traj, [50, 120, 255, 220]),
    ):
        if values:
            rr.log(path, rr.Points3D(np.asarray(values, dtype=np.float32), radii=0.0035, colors=color), static=True)
    camera_positions = [
        np.asarray(frames[idx]["camera"]["T_world_camera_metric"], dtype=np.float32)[:3, 3]
        for idx in selected
    ]
    if camera_positions:
        rr.log("/world/trajectory/camera", rr.Points3D(np.asarray(camera_positions), radii=0.0025, colors=[180, 180, 180, 220]), static=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="All poses + P09 surface",
                contents=[
                    "+ /world/formal/**",
                    "+ /world/v20_k5/**",
                    "+ /world/v20_k10/**",
                    "+ /world/visible_surface",
                    "+ /world/hand/**",
                    "+ /world/camera/**",
                    "+ /world/trajectory/**",
                ],
            ),
            rrb.Spatial3DView(
                origin="/world",
                name="Observed mesh comparison",
                contents=[
                    "+ /world/formal/object_mesh",
                    "+ /world/v20_k5/object_mesh",
                    "+ /world/v20_k10/object_mesh",
                    "+ /world/formal/object",
                    "+ /world/v20_k5/object",
                    "+ /world/v20_k10/object",
                    "+ /world/visible_surface",
                ],
            ),
            column_shares=[1, 1],
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    report = {
        "schema": "v20_diagnostic_world_rrd_v1",
        "status": "ok",
        "diagnostic_only": True,
        "formal_state_modified": False,
        "label": args.label,
        "rrd": str(args.output.expanduser().resolve()),
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "formal_pose_report": str(args.formal_pose_report.expanduser().resolve()),
            "v20_k5_pose_report": str(args.v20_k5_pose_report.expanduser().resolve()),
            "v20_k10_pose_report": str(args.v20_k10_pose_report.expanduser().resolve()),
            "observed_mesh": str(args.observed_mesh.expanduser().resolve()),
            "mano_bridge": str(args.mano_bridge.expanduser().resolve()) if args.mano_bridge else None,
            "object_id": args.object_id,
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "formal_pose_report": sha256_file(args.formal_pose_report),
                "v20_k5_pose_report": sha256_file(args.v20_k5_pose_report),
                "v20_k10_pose_report": sha256_file(args.v20_k10_pose_report),
                "observed_mesh": sha256_file(args.observed_mesh),
                "mano_bridge": sha256_file(args.mano_bridge) if args.mano_bridge else None,
            },
        },
        "timeline": {
            "selected_frame_count": len(selected),
            "first_frame": selected[0],
            "last_frame": selected[-1],
            "selected_frames": selected,
            "metric_surface_frames": metric_frames,
            "no_metric_surface_frames": no_metric_frames,
            "pose_presence": pose_presence,
        },
        "layers": {
            "formal": {"entity": "/world/formal", "color": "red", "authority": "formal P14 diagnostic reference"},
            "v20_k5": {"entity": "/world/v20_k5", "color": "green", "authority": "V20 diagnostic-only keyframe image-factor pose"},
            "v20_k10": {"entity": "/world/v20_k10", "color": "blue", "authority": "V20 diagnostic-only keyframe image-factor pose"},
            "observed_mesh": {"entity": "/world/*/object_mesh", "source": "prediction-side observed P13 surface", "generated_geometry_consumed": False},
            "visible_surface": {"entity": "/world/visible_surface", "source": "P09 prediction-side first-hit world points", "point_stride": args.point_stride},
            "mano": {"entity": "/world/hand", "source": "prediction-side HaWoR/MANO bridge", "authority": "visual reference only"},
        },
        "coordinate_contract": {
            "camera": "p_world = (p_camera @ T_world_camera[:3,:3].T) + T_world_camera[:3,3]",
            "object": "V_world = V_canonical @ R_world_from_canonical.T + t_world",
            "generated_geometry_consumed": False,
        },
        "claim_scope": "Visualization only. V20 poses and visible points are prediction-side diagnostics; the observed mesh is an open observed surface, not a complete object. This RRD must not be used to promote formal P15/D18/D19.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rrd": str(args.output), "report": str(report_path), "selected_frames": len(selected), "metric_surface_frames": len(metric_frames), "no_metric_surface_frames": no_metric_frames}, indent=2))


if __name__ == "__main__":
    main()
