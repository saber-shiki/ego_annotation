#!/usr/bin/env python3
"""Build a full world-coordinate hand/object interaction RRD for V20.

The SAM3D completion mesh is deliberately included only as a render-only
completion hypothesis.  It is transformed by the selected V20 pose so that a
viewer can inspect the apparent hand/object interaction, but it is never used
as pose, collision, contact, signed-distance, or nonpenetration authority.
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
from scipy.spatial.transform import Rotation


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> trimesh.Trimesh:
    value = trimesh.load(path.expanduser().resolve(), process=False)
    if isinstance(value, trimesh.Scene):
        parts = [g for g in value.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"no mesh in {path}")
        value = trimesh.util.concatenate(parts)
    if not isinstance(value, trimesh.Trimesh) or len(value.vertices) == 0 or len(value.faces) == 0:
        raise RuntimeError(f"invalid mesh in {path}")
    return value


def load_pose_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in load_json(path).get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        if row.get("rotation_world_from_completed_canonical_matrix") is None:
            continue
        if row.get("translation_world_m") is None:
            continue
        rows[int(row["frame_idx"])] = row
    return rows


def get_pose(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    translation = np.asarray(row["translation_world_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise RuntimeError("invalid pose row")
    return rotation, translation


def find_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def log_pose(path: str, row: dict[str, Any], color: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = get_pose(row)
    rr.log(path, rr.Transform3D(translation=translation.tolist(), mat3x3=rotation.tolist()))
    rr.log(path + "/origin", rr.Points3D([translation], radii=0.006, colors=color))
    axes = np.stack(
        [
            np.stack([translation, translation + rotation[:, 0] * 0.045]),
            np.stack([translation, translation + rotation[:, 1] * 0.045]),
            np.stack([translation, translation + rotation[:, 2] * 0.045]),
        ],
        axis=0,
    ).astype(np.float32)
    rr.log(path + "/axes", rr.LineStrips3D(axes, radii=0.0025, colors=[color, color, color]))
    return rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--formal-pose-report", type=Path, required=True)
    parser.add_argument("--v20-k5-pose-report", type=Path, required=True)
    parser.add_argument("--v20-k10-pose-report", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--generated-mesh", type=Path, required=True)
    parser.add_argument("--mano-bridge", type=Path, default=None)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--frames", default="all")
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--delta-vertex-stride", type=int, default=120)
    parser.add_argument("--delta-amplification", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="v20_hand_object_interaction")
    args = parser.parse_args()
    if args.point_stride < 1 or args.delta_vertex_stride < 1 or args.delta_amplification < 1.0:
        raise RuntimeError("strides must be positive and delta amplification must be >= 1")

    annotations = load_json(args.annotations)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    if not frames:
        raise RuntimeError("annotations contain no frames")
    formal = load_pose_rows(args.formal_pose_report)
    k5 = load_pose_rows(args.v20_k5_pose_report)
    k10 = load_pose_rows(args.v20_k10_pose_report)
    if args.frames == "all":
        selected = sorted(frames)
    else:
        selected = sorted({int(x) for x in args.frames.split(",") if x.strip()})
    if not selected:
        raise RuntimeError("no frames selected")
    missing = [idx for idx in selected if idx not in frames]
    if missing:
        raise RuntimeError(f"selected frames absent from annotations: {missing[:10]}")

    observed_mesh = load_mesh(args.observed_mesh)
    generated_mesh = load_mesh(args.generated_mesh)
    observed_vertices = np.asarray(observed_mesh.vertices, dtype=np.float32)
    observed_faces = np.asarray(observed_mesh.faces, dtype=np.int32)
    generated_vertices = np.asarray(generated_mesh.vertices, dtype=np.float32)
    generated_faces = np.asarray(generated_mesh.faces, dtype=np.int32)
    delta_vertices = np.asarray(generated_mesh.vertices[:: args.delta_vertex_stride], dtype=np.float64)

    hands: dict[int, dict[str, np.ndarray]] = {}
    if args.mano_bridge is not None:
        with np.load(args.mano_bridge.expanduser().resolve(), allow_pickle=False) as bridge:
            required = {"frame_idx", "hand_side", "vertices_current_v18_world_from_hawor_projection_relift_m"}
            missing_keys = sorted(required.difference(bridge.files))
            if missing_keys:
                raise RuntimeError(f"MANO bridge missing keys: {missing_keys}")
            bridge_frames = np.asarray(bridge["frame_idx"], dtype=np.int64)
            bridge_sides = np.asarray(bridge["hand_side"])
            bridge_vertices = np.asarray(bridge["vertices_current_v18_world_from_hawor_projection_relift_m"], dtype=np.float32)
            bridge_joints = np.asarray(bridge["joints_current_v18_world_from_hawor_projection_relift_m"], dtype=np.float32) if "joints_current_v18_world_from_hawor_projection_relift_m" in bridge.files else None
        for i, (idx, side, vertices) in enumerate(zip(bridge_frames.tolist(), bridge_sides.tolist(), bridge_vertices)):
            side_name = side.decode() if isinstance(side, bytes) else str(side)
            hands.setdefault(int(idx), {})[side_name] = vertices
            if bridge_joints is not None:
                hands.setdefault(int(idx), {})[side_name + "_joints"] = bridge_joints[i]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"milk_{args.label}", spawn=False)
    rr.save(str(args.output.expanduser().resolve()))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)

    # Static geometry.  The generated mesh is explicitly a render-only layer.
    for name, color in (("formal", [220, 50, 50, 125]), ("v20_k5", [220, 40, 210, 145]), ("v20_k10", [50, 110, 255, 135])):
        rr.log(
            f"/world/{name}/object/sam3d_completion_render_only",
            rr.Mesh3D(vertex_positions=generated_vertices, triangle_indices=generated_faces, albedo_factor=color),
            static=True,
        )
    rr.log(
        "/world/v20_k5/object/observed_surface_mesh",
        rr.Mesh3D(vertex_positions=observed_vertices, triangle_indices=observed_faces, albedo_factor=[40, 220, 100, 210]),
        static=True,
    )

    pose_presence = {"formal": [], "v20_k5": [], "v20_k10": []}
    metric_frames: list[int] = []
    no_metric_frames: list[int] = []
    delta_values = {"k5_translation_m": [], "k5_rotation_deg": [], "k10_translation_m": [], "k10_rotation_deg": []}
    trajectories: dict[str, list[np.ndarray]] = {"formal": [], "v20_k5": [], "v20_k10": [], "camera": []}

    for frame_idx in selected:
        rr.set_time("frame", sequence=frame_idx)
        frame = frames[frame_idx]
        camera = frame.get("camera") or {}
        T = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"invalid camera transform at frame {frame_idx}")
        trajectories["camera"].append(T[:3, 3].copy())
        rr.log("/world/camera", rr.Transform3D(translation=T[:3, 3].tolist(), mat3x3=T[:3, :3].tolist()))
        intrinsics = camera.get("intrinsics_fx_fy_cx_cy") or [974.3447265625, 974.3447265625, 702.6607055664062, 706.1102294921875]
        fx, fy, cx, cy = [float(x) for x in intrinsics]
        rr.log("/world/camera/pinhole", rr.Pinhole(focal_length=[fx, fy], principal_point=[cx, cy], resolution=[1408, 1408], image_plane_distance=0.1))

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, rows, color in (("formal", formal, [220, 50, 50, 255]), ("v20_k5", k5, [40, 220, 100, 255]), ("v20_k10", k10, [50, 120, 255, 255])):
            row = rows.get(frame_idx)
            pose_presence[name].append(row is not None)
            if row is None:
                rr.log(f"/world/{name}/object", rr.Clear(recursive=False))
                rr.log(f"/world/{name}/status", rr.TextLog("no pose at this frame"))
                continue
            rotation, translation = log_pose(f"/world/{name}/object", row, color)
            poses[name] = (rotation, translation)
            trajectories[name].append(translation)

        obj = find_object(frame, args.object_id)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float32)
        if observed.ndim == 2 and observed.shape[1] == 3 and len(observed) and np.isfinite(observed).all():
            metric_frames.append(frame_idx)
            rr.log("/world/p09_visible_surface", rr.Points3D(observed[:: args.point_stride], radii=0.0022, colors=[255, 220, 0, 255]))
        else:
            no_metric_frames.append(frame_idx)
            rr.log("/world/p09_visible_surface", rr.Clear(recursive=False))
            rr.log("/world/p09_visible_surface/status", rr.TextLog("no accepted P09 metric surface"))

        for side, color in (("left", [255, 150, 40, 255]), ("right", [80, 150, 255, 255])):
            vertices = hands.get(frame_idx, {}).get(side)
            if vertices is None:
                rr.log(f"/world/hands/{side}", rr.Clear(recursive=False))
                continue
            rr.log(f"/world/hands/{side}", rr.Points3D(vertices, radii=0.0025, colors=color))
            joints = hands.get(frame_idx, {}).get(side + "_joints")
            if joints is not None:
                rr.log(f"/world/hands/{side}/joints", rr.Points3D(joints, radii=0.005, colors=color))

        formal_pose = poses.get("formal")
        for name, key in (("v20_k5", "k5"), ("v20_k10", "k10")):
            candidate = poses.get(name)
            if formal_pose is None or candidate is None:
                rr.log(f"/world/pose_delta/{key}", rr.Clear(recursive=False))
                continue
            formal_R, formal_t = formal_pose
            candidate_R, candidate_t = candidate
            delta_t = candidate_t - formal_t
            delta_r_deg = float(np.degrees(Rotation.from_matrix(candidate_R @ formal_R.T).magnitude()))
            delta_t_norm = float(np.linalg.norm(delta_t))
            delta_values[f"{key}_translation_m"].append(delta_t_norm)
            delta_values[f"{key}_rotation_deg"].append(delta_r_deg)
            rr.log(f"/world/pose_delta/{key}/true", rr.Arrows3D(vectors=[delta_t.astype(np.float32)], origins=[formal_t.astype(np.float32)], radii=0.0018, colors=[[240, 0, 240, 255]]))
            amplified = delta_t * float(args.delta_amplification)
            rr.log(f"/world/pose_delta/{key}/amplified_{int(args.delta_amplification)}x", rr.Arrows3D(vectors=[amplified.astype(np.float32)], origins=[formal_t.astype(np.float32)], radii=0.0022, colors=[[255, 150, 0, 255]], labels=[f"{key} {args.delta_amplification:g}x"], show_labels=True))
            formal_world = delta_vertices @ formal_R.T + formal_t[None, :]
            candidate_world = delta_vertices @ candidate_R.T + candidate_t[None, :]
            rr.log(f"/world/pose_delta/{key}/true_vertex_segments", rr.LineStrips3D(np.stack([formal_world, candidate_world], axis=1).astype(np.float32), radii=0.001, colors=[[240, 0, 240, 190]]))
            amplified_world = formal_world + float(args.delta_amplification) * (candidate_world - formal_world)
            rr.log(f"/world/pose_delta/{key}/amplified_vertex_segments", rr.LineStrips3D(np.stack([formal_world, amplified_world], axis=1).astype(np.float32), radii=0.0015, colors=[[255, 150, 0, 210]]))
            rr.log(f"/world/pose_delta/{key}/metrics", rr.TextLog(f"frame={frame_idx}  Δt={delta_t_norm * 1000.0:.3f} mm  ΔR={delta_r_deg:.3f} deg"))

    for name, values, color in (("formal", trajectories["formal"], [220, 50, 50, 190]), ("v20_k5", trajectories["v20_k5"], [40, 220, 100, 220]), ("v20_k10", trajectories["v20_k10"], [50, 120, 255, 220]), ("camera", trajectories["camera"], [170, 170, 170, 220])):
        if values:
            rr.log(f"/world/trajectory/{name}", rr.Points3D(np.asarray(values, dtype=np.float32), radii=0.0028, colors=color), static=True)

    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(
                origin="/world",
                name="K5 full hand-object interaction",
                contents=[
                    "+ /world/v20_k5/**",
                    "+ /world/p09_visible_surface",
                    "+ /world/hands/**",
                    "+ /world/camera/**",
                    "+ /world/trajectory/**",
                ],
            ),
            rrb.Spatial3DView(
                origin="/world",
                name="Generated mesh formal / K5 / K10 comparison",
                contents=[
                    "+ /world/formal/object/**",
                    "+ /world/v20_k5/object/**",
                    "+ /world/v20_k10/object/**",
                    "+ /world/p09_visible_surface",
                ],
            ),
            rrb.Spatial3DView(
                origin="/world",
                name="Pose delta (10x orange is visual-only)",
                contents=[
                    "+ /world/pose_delta/**",
                    "+ /world/formal/object/origin",
                    "+ /world/v20_k5/object/origin",
                    "+ /world/v20_k10/object/origin",
                ],
            ),
            rrb.TextLogView(origin="/world/pose_delta", name="Pose delta metrics"),
            active_tab=0,
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    report = {
        "schema": "v20_full_hand_object_interaction_world_rrd_v1",
        "status": "ok",
        "diagnostic_only": True,
        "formal_state_modified": False,
        "generated_mesh_role": "render_only_completion_hypothesis",
        "generated_geometry_pose_authority": False,
        "generated_geometry_collision_authority": False,
        "generated_geometry_contact_authority": False,
        "rrd": str(args.output.expanduser().resolve()),
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "formal_pose_report": str(args.formal_pose_report.expanduser().resolve()),
            "v20_k5_pose_report": str(args.v20_k5_pose_report.expanduser().resolve()),
            "v20_k10_pose_report": str(args.v20_k10_pose_report.expanduser().resolve()),
            "observed_mesh": str(args.observed_mesh.expanduser().resolve()),
            "generated_mesh_render_only": str(args.generated_mesh.expanduser().resolve()),
            "mano_bridge": str(args.mano_bridge.expanduser().resolve()) if args.mano_bridge else None,
            "object_id": args.object_id,
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "formal_pose_report": sha256_file(args.formal_pose_report),
                "v20_k5_pose_report": sha256_file(args.v20_k5_pose_report),
                "v20_k10_pose_report": sha256_file(args.v20_k10_pose_report),
                "observed_mesh": sha256_file(args.observed_mesh),
                "generated_mesh_render_only": sha256_file(args.generated_mesh),
                "mano_bridge": sha256_file(args.mano_bridge) if args.mano_bridge else None,
            },
        },
        "timeline": {
            "selected_frame_count": len(selected),
            "first_frame": selected[0],
            "last_frame": selected[-1],
            "metric_surface_frames": metric_frames,
            "no_metric_surface_frames": no_metric_frames,
            "pose_presence": pose_presence,
        },
        "delta_summary": {
            key: {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
                "p90": float(np.percentile(values, 90)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
            for key, values in delta_values.items()
        },
        "layers": {
            "sam3d_completion_render_only": "The generated mesh is transformed by each pose layer for visual inspection only.",
            "p09_visible_surface": "Prediction-side first-hit visible points; missing frames are cleared.",
            "mano": "Prediction-side HaWoR/MANO bridge points; visual reference only.",
            "pose_delta": "Magenta true delta and orange labelled visual amplification.",
        },
        "coordinate_contract": {
            "object_formula": "V_world = V_canonical @ R_world_from_canonical.T + t_world",
            "camera_formula": "p_world = p_camera @ T_world_camera[:3,:3].T + T_world_camera[:3,3]",
            "delta_rotation": "Log(R_v20 @ R_formal.T)",
            "generated_geometry_consumed_as_authority": False,
        },
        "claim_scope": "Visualization only. This recording shows a render-only SAM3D completion hypothesis moving with formal/V20 pose transforms alongside prediction-side P09 surface and MANO points. It is not a physical collision/contact/SDF result and must not replace formal D19.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rrd": str(args.output), "report": str(report_path), "selected_frames": len(selected), "metric_surface_frames": len(metric_frames), "no_metric_surface_frames": no_metric_frames, "generated_mesh_vertices": len(generated_vertices), "generated_mesh_faces": len(generated_faces)}, indent=2))


if __name__ == "__main__":
    main()
