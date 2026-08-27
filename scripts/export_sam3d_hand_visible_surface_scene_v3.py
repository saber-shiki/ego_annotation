#!/usr/bin/env python3
"""Export corrected SAM3D object + MANO hands + visible surfels as one scene.

Coordinates are the requested frame's OpenCV camera frame in metres. Outputs:

* GLB scene with named, independently selectable colored triangle meshes;
* binary/ASCII colored PLY with all components concatenated into one mesh;
* JSON provenance/geometry report.

Observed surfels are represented as small icosahedra so the PLY remains a
triangle mesh rather than a point cloud. Generated object hidden faces remain
render-only and gain no collision/sign/contact authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

COLORS = {
    "object": np.asarray([80, 190, 230, 255], dtype=np.uint8),
    "left_hand": np.asarray([60, 220, 90, 255], dtype=np.uint8),
    "right_hand": np.asarray([40, 150, 70, 255], dtype=np.uint8),
    "visible_surface": np.asarray([255, 215, 0, 255], dtype=np.uint8),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def colored_mesh(vertices: np.ndarray, faces: np.ndarray, color: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def surfel_spheres(points: np.ndarray, radius_m: float) -> trimesh.Trimesh:
    """Vectorized subdivision-0 icospheres: 12 vertices/20 faces per surfel."""
    points = np.asarray(points, dtype=np.float64)
    unit = trimesh.creation.icosphere(subdivisions=0, radius=float(radius_m))
    unit_vertices = np.asarray(unit.vertices, dtype=np.float64)
    unit_faces = np.asarray(unit.faces, dtype=np.int64)
    vertices = (points[:, None, :] + unit_vertices[None, :, :]).reshape(-1, 3)
    offsets = np.arange(len(points), dtype=np.int64)[:, None, None] * len(unit_vertices)
    faces = (unit_faces[None, :, :] + offsets).reshape(-1, 3)
    return colored_mesh(vertices, faces, COLORS["visible_surface"])


def concatenate_colored(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
        if vertex_colors.shape != (len(mesh.vertices), 4):
            raise RuntimeError("component lacks RGBA vertex colors")
        colors.append(vertex_colors)
        offset += len(mesh.vertices)
    result = trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )
    result.visual.vertex_colors = np.concatenate(colors, axis=0)
    return result


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]


def world_to_camera(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    pose_graph = load_json(args.pose_graph)
    frame_by_idx = {int(row["frame_idx"]): row for row in annotations["frames"]}
    pose_by_idx = {int(row["frame_idx"]): row for row in pose_graph["pose_rows"]}
    if args.frame not in frame_by_idx or args.frame not in pose_by_idx:
        raise RuntimeError(f"frame {args.frame} missing from annotations or pose graph")
    if args.anchor_frame not in frame_by_idx or args.anchor_frame not in pose_by_idx:
        raise RuntimeError(f"anchor frame {args.anchor_frame} missing")

    source_mesh = trimesh.load(args.object_mesh_anchor_camera, force="mesh", process=False)
    if not isinstance(source_mesh, trimesh.Trimesh):
        raise RuntimeError(f"invalid object mesh: {args.object_mesh_anchor_camera}")
    object_anchor = np.asarray(source_mesh.vertices, dtype=np.float64)
    object_faces = np.asarray(source_mesh.faces, dtype=np.int64)

    anchor_frame = frame_by_idx[args.anchor_frame]
    anchor_pose = pose_by_idx[args.anchor_frame]
    T_anchor = np.asarray(anchor_frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    R_anchor = np.asarray(anchor_pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    t_anchor = np.asarray(anchor_pose["translation_world_m"], dtype=np.float64)
    object_world_anchor = camera_to_world(object_anchor, T_anchor)
    object_canonical = (object_world_anchor - t_anchor[None, :]) @ R_anchor

    frame = frame_by_idx[args.frame]
    pose = pose_by_idx[args.frame]
    T_frame = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
    R_frame = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    t_frame = np.asarray(pose["translation_world_m"], dtype=np.float64)
    object_world = object_canonical @ R_frame.T + t_frame[None, :]
    object_camera = world_to_camera(object_world, T_frame)
    object_mesh = colored_mesh(object_camera, object_faces, COLORS["object"])

    hands = np.load(args.hand_npz)
    hand_indices = hands["frame_idx"].astype(int).tolist()
    if args.frame not in hand_indices:
        raise RuntimeError(f"frame {args.frame} missing from hand archive")
    hand_position = hand_indices.index(args.frame)
    components: list[tuple[str, trimesh.Trimesh]] = [("sam3d_object", object_mesh)]
    hand_rows: dict[str, Any] = {}
    for side in ("left", "right"):
        valid = int(np.asarray(hands[f"{side}_valid"])[hand_position]) == 1
        if not valid:
            hand_rows[side] = {"valid": False}
            continue
        vertices_world = np.asarray(hands[f"{side}_vertices_world_m"][hand_position], dtype=np.float64)
        vertices_camera = world_to_camera(vertices_world, T_frame)
        faces = np.asarray(hands[f"{side}_faces"], dtype=np.int64)
        mesh = colored_mesh(vertices_camera, faces, COLORS[f"{side}_hand"])
        components.append((f"mano_{side}_hand", mesh))
        hand_rows[side] = {
            "valid": True,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bbox_m": np.ptp(vertices_camera, axis=0).astype(float).tolist(),
        }

    visible = frame["objects"][0].get("visible_geometry_candidate")
    if not isinstance(visible, dict):
        raise RuntimeError(f"frame {args.frame} lacks visible geometry")
    ownership = visible.get("first_surface_depth_ownership")
    if not isinstance(ownership, dict) or ownership.get("enabled") is not True or ownership.get("fail_closed") is True:
        raise RuntimeError("visible surface lacks valid first-surface ownership")
    surfels = np.asarray(visible["camera_vertices_sample_m"], dtype=np.float64)
    surfels = surfels[np.isfinite(surfels).all(axis=1) & (surfels[:, 2] > 1.0e-6)]
    surfel_mesh = surfel_spheres(surfels, args.surfel_radius_m)
    components.append(("observed_visible_surface_surfels", surfel_mesh))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{args.frame:06d}_object_hands_visible_surface"
    scene = trimesh.Scene()
    for name, mesh in components:
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    glb_path = args.output_dir / f"{stem}.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    combined = concatenate_colored([mesh for _, mesh in components])
    binary_ply = args.output_dir / f"{stem}.binary.ply"
    ascii_ply = args.output_dir / f"{stem}.ascii.ply"
    combined.export(binary_ply, file_type="ply")
    combined.export(ascii_ply, file_type="ply", encoding="ascii")

    report = {
        "schema": "sam3d_hand_visible_surface_scene_v1",
        "frame_idx": args.frame,
        "anchor_frame": args.anchor_frame,
        "coordinate_frame": "requested_frame_opencv_camera",
        "unit": "metre",
        "object_mesh": str(args.object_mesh_anchor_camera),
        "hands": hand_rows,
        "visible_surface": {
            "source": "camera_vertices_sample_m",
            "first_surface_ownership_state": ownership.get("state"),
            "surfel_count": int(len(surfels)),
            "surfel_radius_m": float(args.surfel_radius_m),
            "sphere_topology": "subdivision_0_icosphere_12_vertices_20_faces_per_surfel",
        },
        "legend_rgba": {key: value.astype(int).tolist() for key, value in COLORS.items()},
        "combined_triangle_mesh": {
            "vertices": int(len(combined.vertices)),
            "faces": int(len(combined.faces)),
            "bbox_m": np.ptp(np.asarray(combined.vertices), axis=0).astype(float).tolist(),
        },
        "outputs": {
            "recommended_scene_glb": str(glb_path),
            "combined_binary_ply": str(binary_ply),
            "combined_ascii_ply": str(ascii_ply),
        },
        "claim_scope": (
            "Visualization scene combining corrected SAM3D render prior, metric HaWoR MANO meshes, "
            "and trusted observed first-surface surfels. Generated hidden faces are not collision/sign/contact authority."
        ),
    }
    report_path = args.output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "HOW_TO_VIEW.txt").write_text(
        "Open the .glb in Blender or MeshLab for named/selectable components.\n"
        "Open the .binary.ply or .ascii.ply in MeshLab/CloudCompare for one colored triangle mesh.\n"
        "Legend: cyan=corrected SAM3D object; bright green=left MANO; dark green=right MANO; yellow=trusted visible-surface surfels.\n"
        "Coordinates: requested frame OpenCV camera (+x right, +y down, +z forward), metres.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-mesh-anchor-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-graph", type=Path, required=True)
    parser.add_argument("--hand-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=92)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--surfel-radius-m", type=float, default=0.0012)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
