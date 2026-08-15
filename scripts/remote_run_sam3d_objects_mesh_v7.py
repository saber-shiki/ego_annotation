#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.asarray(mask, dtype=np.uint8) > 0


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mesh_arrays(mesh_obj) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertices = as_numpy(mesh_obj.vertices).astype(np.float64)
    faces = as_numpy(mesh_obj.faces).astype(np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("SAM3D mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("SAM3D mesh has no valid triangular faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError("SAM3D mesh vertices contain non-finite values")
    colors = None
    attrs = getattr(mesh_obj, "vertex_attrs", None)
    if attrs is not None:
        attrs_np = as_numpy(attrs)
        if attrs_np.ndim == 2 and attrs_np.shape[0] == vertices.shape[0] and attrs_np.shape[1] >= 3:
            colors = np.clip(attrs_np[:, :3], 0.0, 1.0)
    return vertices, faces, colors


def export_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
    colors: np.ndarray | None = None,
) -> dict:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        mesh.visual.vertex_colors = np.concatenate(
            [(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8), np.full((len(colors), 1), 255, dtype=np.uint8)],
            axis=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent": [float(x) for x in extent],
        "center": [float(x) for x in vertices.mean(axis=0)],
        "depth_positive_fraction": float(np.count_nonzero(vertices[:, 2] > 0.0) / len(vertices)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def export_raw_mesh(mesh_obj, path: Path) -> dict:
    vertices, faces, colors = mesh_arrays(mesh_obj)
    stats = export_mesh_arrays(vertices, faces, path, colors)
    return {
        "vertices": stats["vertices"],
        "faces": stats["faces"],
        "extent_model_units": stats["extent"],
        "center_model_units": stats["center"],
        "watertight": stats["watertight"],
        "winding_consistent": stats["winding_consistent"],
    }


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise RuntimeError(f"SAM3D rotation must be finite wxyz quaternion, got {q}")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-3:
        raise RuntimeError(f"SAM3D quaternion norm is invalid: {norm}")
    w, x, y, z = (q / norm).tolist()
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def apply_native_pose(
    vertices_local: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    vertices = np.asarray(vertices_local, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise RuntimeError(f"SAM3D translation must be finite xyz, got {translation}")
    if scale.shape == (1,):
        scale = np.repeat(scale, 3)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise RuntimeError(f"SAM3D scale must be positive scalar/xyz, got {scale}")
    rotation = quaternion_wxyz_to_matrix(rotation_wxyz)
    vertices_p3d = (vertices * scale[None, :]) @ rotation + translation[None, :]
    p3d_to_opencv = np.diag([-1.0, -1.0, 1.0])
    vertices_opencv = vertices_p3d @ p3d_to_opencv
    return vertices_p3d, vertices_opencv, {
        "quaternion_order": "wxyz_scalar_first_pytorch3d",
        "row_vector_formula": "p_p3d_camera = (p_raw_local * scale_xyz) @ quaternion_to_matrix(q_wxyz) + translation_xyz",
        "pytorch3d_camera_convention": "x_left_y_up_z_forward",
        "opencv_camera_convention": "x_right_y_down_z_forward",
        "p3d_to_opencv_row_matrix": p3d_to_opencv.tolist(),
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
        "scale_xyz": scale.tolist(),
        "metric_status": "native_monocular_scene_units_not_sensor_meters",
    }


def tensor_list(value) -> list[float] | None:
    if value is None:
        return None
    arr = as_numpy(value).astype(float).reshape(-1)
    return [float(x) for x in arr]


def parse_case(raw: str) -> tuple[str, Path, Path, int]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise RuntimeError("--case must have format name|image_path|mask_path|seed")
    name, image, mask, seed = parts
    if not name.strip():
        raise RuntimeError("case name is empty")
    return name.strip(), Path(image), Path(mask), int(seed)


def run_one_case(inference, output_root: Path, name: str, image_path: Path, mask_path: Path, seed: int, args: argparse.Namespace) -> dict:
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    mask = load_mask(mask_path)
    if image.shape[:2] != mask.shape:
        raise RuntimeError(f"{name}: image shape {image.shape[:2]} differs from mask shape {mask.shape}")
    if int(mask.sum()) < int(args.min_mask_pixels):
        raise RuntimeError(f"{name}: mask has too few foreground pixels: {int(mask.sum())}")

    output = inference(image, mask, seed=int(seed))
    mesh_list = output.get("mesh")
    if not mesh_list:
        raise RuntimeError(f"{name}: SAM3D returned no mesh output")
    mesh_path = case_dir / args.mesh_name
    mesh_obj = mesh_list[0]
    mesh_stats = export_raw_mesh(mesh_obj, mesh_path)
    vertices_local, faces, colors = mesh_arrays(mesh_obj)

    rotation = tensor_list(output.get("rotation"))
    translation = tensor_list(output.get("translation"))
    scale = tensor_list(output.get("scale"))
    if rotation is None or translation is None or scale is None:
        raise RuntimeError(f"{name}: SAM3D returned mesh without complete native rotation/translation/scale")
    vertices_p3d, vertices_opencv, pose_contract = apply_native_pose(
        vertices_local,
        np.asarray(rotation, dtype=np.float64),
        np.asarray(translation, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )
    native_p3d_path = case_dir / args.native_p3d_mesh_name
    native_opencv_path = case_dir / args.native_opencv_mesh_name
    native_p3d_stats = export_mesh_arrays(vertices_p3d, faces, native_p3d_path, colors)
    native_opencv_stats = export_mesh_arrays(vertices_opencv, faces, native_opencv_path, colors)
    if native_p3d_stats["depth_positive_fraction"] < float(args.min_native_positive_depth_fraction):
        raise RuntimeError(
            f"{name}: SAM3D native pose leaves only {native_p3d_stats['depth_positive_fraction']:.4f} vertices at positive depth"
        )

    glb_path = case_dir / args.glb_name
    glb = output.get("glb")
    if glb is None:
        raise RuntimeError(f"{name}: SAM3D returned no GLB output")
    glb.export(str(glb_path))

    gs_path = None
    if output.get("gs") is not None:
        gs_path = case_dir / args.gaussian_name
        output["gs"].save_ply(str(gs_path))

    report = {
        "status": "ok",
        "name": name,
        "image": str(image_path),
        "mask": str(mask_path),
        "seed": int(seed),
        "mask_pixels": int(mask.sum()),
        "image_shape_h_w": [int(image.shape[0]), int(image.shape[1])],
        "mesh": str(mesh_path),
        "native_pose_mesh_pytorch3d_camera": str(native_p3d_path),
        "native_pose_mesh_opencv_camera": str(native_opencv_path),
        "glb": str(glb_path),
        "gaussian": str(gs_path) if gs_path else None,
        "mesh_stats": mesh_stats,
        "pose": {
            "rotation": rotation,
            "translation": translation,
            "scale": scale,
        },
        "native_pose_contract": {
            **pose_contract,
            "raw_mesh_frame": "native local generator frame shared with Gaussian local coordinates",
            "glb_frame": "standalone presentation frame; official to_glb applies a separate z-up-to-y-up rotation and must not be fed back into native pose",
            "native_pose_mesh_pytorch3d_camera": str(native_p3d_path),
            "native_pose_mesh_opencv_camera": str(native_opencv_path),
            "native_pose_mesh_pytorch3d_camera_stats": native_p3d_stats,
            "native_pose_mesh_opencv_camera_stats": native_opencv_stats,
            "metric_bridge_requirement": "scale the complete camera-origin scene similarity (native translation and object scale together) using prediction-side sensor depth; never treat native units as meters",
        },
    }
    (case_dir / "qc_sam3d_objects_mesh_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "notebook"))

    from inference import Inference

    cases = [parse_case(raw) for raw in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference = Inference(str(args.config), compile=bool(args.compile))
    case_reports = [run_one_case(inference, args.output_dir, name, image, mask, seed, args) for name, image, mask, seed in cases]

    report = {
        "status": "ok",
        "method": "remote_run_sam3d_objects_mesh_v7",
        "repo": str(args.repo),
        "config": str(args.config),
        "cases": case_reports,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }
    (args.output_dir / "qc_sam3d_objects_mesh_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-name", default="sam3d_mesh.ply")
    parser.add_argument("--native-p3d-mesh-name", default="sam3d_mesh_native_pose_pytorch3d_camera.ply")
    parser.add_argument("--native-opencv-mesh-name", default="sam3d_mesh_native_pose_opencv_camera.ply")
    parser.add_argument("--glb-name", default="sam3d_mesh.glb")
    parser.add_argument("--gaussian-name", default="sam3d_gaussian.ply")
    parser.add_argument("--min-mask-pixels", type=int, default=100)
    parser.add_argument("--min-native-positive-depth-fraction", type=float, default=0.99)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
