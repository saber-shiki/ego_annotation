#!/usr/bin/env python3
"""GHOST-lite multi-frame object alignment: one global Sim3 for the SAM3D mesh.

The complete SAM3D generated mesh is given in the anchor camera frame with the
native local-to-camera pose and the camera-origin metric scale already applied
(see audit_sam3d_native_pose_contract_v3.py).  This optimizer refines a single
global similarity transform (uniform scale + rotation + translation) so that the
generated mesh satisfies, across many frames at once:

  * one-way observed-surfel-to-generated-surface proximity (only observed ->
    generated, so hidden/back surfaces are never pulled toward the visible front),
  * first-hit depth at object-owned pixels: behind-observed penalty (mesh must not
    float behind the measured front surface) and front-poke penalty (mesh must not
    poke in front of the measured front surface),
  * silhouette outside distance on the object-owned mask plane,
  * small priors on scale/rotation/translation around the P15 observed pose.

The output is a diagnostic render/geometry alignment only; generated faces are
never promoted to collision or annotation authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {label}: {path}")
    return path


def finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise RuntimeError(f"{label} must be finite length {size}, got {result}")
    return result


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"not a triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"invalid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid faces: {path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"non-finite vertices: {path}")
    return loaded


def sample_rows(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("cannot sample an empty point set")
    if len(points) <= int(count):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(count), replace=False)]


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    tri = vertices[faces]
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        raise RuntimeError("mesh face areas are invalid")
    face_ids = rng.choice(len(faces), size=int(count), replace=True, p=areas / areas.sum())
    chosen = tri[face_ids]
    u = rng.random(int(count))
    v = rng.random(int(count))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])


def mask_plane_transform(visible: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    fx, fy, cx, cy = finite_vector(visible.get("intrinsics_fx_fy_cx_cy"), 4, "depth intrinsics")
    K_depth = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = visible.get("mask_depth_transform_contract")
    if not isinstance(transform, dict) or transform.get("camera_contract_consistent") is not True:
        raise RuntimeError("visible evidence lacks a camera-consistent mask/depth transform")
    A_depth_from_mask = np.asarray(transform.get("A_depth_from_mask_coordinate_model"), dtype=np.float64)
    if A_depth_from_mask.shape != (3, 3) or not np.isfinite(A_depth_from_mask).all():
        raise RuntimeError("invalid A_depth_from_mask_coordinate_model")
    return K_depth, {
        "A_depth_from_mask_coordinate_model": A_depth_from_mask.tolist(),
        "A_mask_from_source_coordinate_model": np.linalg.inv(A_depth_from_mask).tolist(),
        "depth_size_wh": [int(x) for x in transform.get("depth_size_wh") or []],
        "mask_size_wh": [int(x) for x in transform.get("mask_size_wh") or []],
        "pixel_center_convention": transform.get("pixel_center_convention"),
    }


@dataclass
class FrameData:
    frame_idx: int
    T_world_camera: np.ndarray  # camera -> world
    K_depth: np.ndarray  # source-plane intrinsics
    A_mask_from_source: np.ndarray  # 3x3 source -> mask plane
    mask_source: np.ndarray  # object-owned mask on source plane (bool)
    depth_m: np.ndarray  # metric depth on source plane
    depth_valid: np.ndarray  # finite depth > 0
    observed_anchor: np.ndarray  # trusted first-surface surfels in anchor camera frame


class Rasterizer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._rasterizer = None
        self._size = None
        self._cameras = None
        self._lazy = None

    def _ensure(self, vertices: torch.Tensor, faces: torch.Tensor, size: int, K: torch.Tensor) -> None:
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

        if self._rasterizer is not None and self._size == size:
            return
        self._meshes_cls = Meshes
        self._rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(
                image_size=size, blur_radius=0, faces_per_pixel=1, cull_backfaces=False
            ),
        )
        self._size = size

    def zbuffer_batch(self, vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, size: int, batch: int = 10) -> np.ndarray:
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

        v = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        f = torch.tensor(faces, dtype=torch.int64, device=self.device)
        K = torch.tensor(K, dtype=torch.float32, device=self.device)[None]
        n = v.shape[0]
        all_z = []
        with torch.no_grad():
            for start in range(0, n, batch):
                stop = min(start + batch, n)
                m = stop - start
                R = torch.eye(3, device=self.device)[None].expand(m, -1, -1)
                T = torch.zeros(m, 3, device=self.device)
                Kk = K.expand(m, -1, -1)
                size_t = torch.tensor([[size, size]], device=self.device).expand(m, -1)
                cameras = cameras_from_opencv_projection(R, T, Kk, size_t)
                rasterizer = MeshRasterizer(
                    cameras=cameras,
                    raster_settings=RasterizationSettings(
                        image_size=size, blur_radius=0, faces_per_pixel=1, cull_backfaces=False
                    ),
                )
                zb = rasterizer(Meshes(verts=[v[i] for i in range(start, stop)], faces=[f] * m)).zbuf
                all_z.append(zb[..., 0].detach().cpu().numpy())
        return np.concatenate(all_z, axis=0)


def unpack(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    log_scale = float(params[0])
    rotvec = params[1:4]
    translation = params[4:7]
    return float(np.exp(log_scale)), Rotation.from_rotvec(rotvec).as_matrix(), translation


def apply_sim3(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (np.asarray(points, dtype=np.float64) @ R.T) + t[None, :]


def world_to_camera(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    return (np.asarray(points_world, dtype=float) - t[None, :]) @ R


def camera_to_world(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    return np.asarray(points_camera, dtype=float) @ R.T + t[None, :]


def load_frames(args: argparse.Namespace, anchor_idx: int, anchor_T: np.ndarray) -> list[FrameData]:
    annotations = load_json(args.annotations)
    frames = []
    used = []
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        if idx < int(args.frame_start) or idx > int(args.frame_end):
            continue
        if idx != int(args.anchor_frame) and (idx - int(args.frame_start)) % max(1, int(args.frame_step)) != 0:
            continue
        obj = frame.get("objects", [{}])[0]
        visible = obj.get("visible_geometry_candidate") if isinstance(obj, dict) else None
        if not isinstance(visible, dict):
            continue
        ownership = visible.get("first_surface_depth_ownership")
        if not isinstance(ownership, dict) or ownership.get("enabled") is not True:
            continue
        if ownership.get("fail_closed") is True:
            continue
        if float(ownership.get("retained_fraction") or 0.0) < float(args.min_ownership_fraction):
            continue
        camera = frame.get("camera")
        if not isinstance(camera, dict):
            continue
        T_world_camera = np.asarray(camera.get("T_world_camera_metric") or [], dtype=np.float64)
        if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
            continue
        mask_path = Path(str(visible.get("mask_path") or obj.get("mask_path") or ""))
        if not mask_path.is_file():
            continue
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_image is None or np.count_nonzero(mask_image) == 0:
            continue
        K_depth, contract = mask_plane_transform(visible)
        depth_size_wh = contract["depth_size_wh"]
        mask_size_wh = contract["mask_size_wh"]
        if not depth_size_wh or not mask_size_wh:
            continue
        depth_w, depth_h = int(depth_size_wh[0]), int(depth_size_wh[1])
        mask_w, mask_h = int(mask_size_wh[0]), int(mask_size_wh[1])
        if mask_image.shape[:2] != (mask_h, mask_w):
            continue
        depths = np.load(args.depth_npz)
        depth_map = depths["depth"].astype(np.float64)
        depth_intr = depths["intrinsics_fx_fy_cx_cy"].astype(np.float64)
        depth_frame_idx = depths["frame_idx"].astype(int).tolist()
        if idx not in depth_frame_idx:
            continue
        depth_row = depth_frame_idx.index(idx)
        depth = depth_map[depth_row]
        if depth.shape != (depth_h, depth_w):
            continue
        if not np.allclose(depth_intr[depth_row], [K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]], atol=1.0):
            continue
        A_mask_from_source = np.asarray(contract["A_mask_from_source_coordinate_model"], dtype=np.float64)
        mask_source = (
            cv2.warpAffine(
                (mask_image > 0).astype(np.uint8),
                A_mask_from_source[:2, :],
                (depth_w, depth_h),
                flags=cv2.INTER_NEAREST,
            )
            > 0
        )
        observed_camera = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
        observed_camera = observed_camera[np.isfinite(observed_camera).all(axis=1) & (observed_camera[:, 2] > 0.0)]
        if len(observed_camera) < int(args.min_observed_points):
            continue
        observed_world = camera_to_world(observed_camera, T_world_camera)
        observed_anchor = world_to_camera(observed_world, anchor_T)
        frames.append(
            FrameData(
                frame_idx=idx,
                T_world_camera=T_world_camera,
                K_depth=K_depth,
                A_mask_from_source=A_mask_from_source,
                mask_source=mask_source,
                depth_m=depth,
                depth_valid=np.isfinite(depth) & (depth > float(args.min_depth_m)),
                observed_anchor=observed_anchor,
            )
        )
        used.append(idx)
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames selected (need {args.min_frames})")
    return frames


def initial_params(frames: list[FrameData], anchor_idx: int) -> np.ndarray:
    anchor_obs = [f.observed_anchor for f in frames if f.frame_idx == anchor_idx]
    if anchor_obs:
        target = np.median(np.concatenate(anchor_obs, axis=0), axis=0)
    else:
        target = np.median(np.concatenate([f.observed_anchor for f in frames], axis=0), axis=0)
    return np.r_[0.0, np.zeros(3), target]


def normalized_block(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise RuntimeError("cannot normalize an empty residual block")
    return values / np.sqrt(float(len(values)))


def residual_vector(
    params: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_surface: np.ndarray,
    mesh_projection: np.ndarray,
    frames: list[FrameData],
    rasterizer: Rasterizer,
    args: argparse.Namespace,
) -> np.ndarray:
    scale, R, t = unpack(params)
    residuals = []
    surface = apply_sim3(mesh_surface, scale, R, t)
    projection = apply_sim3(mesh_projection, scale, R, t)
    surface_tree = cKDTree(surface)
    for frame in frames:
        d_obs, _ = surface_tree.query(frame.observed_anchor, k=1)
        residuals.append(
            normalized_block(np.clip(d_obs, 0.0, float(args.max_observed_residual_m)) / float(args.sigma_observed_m))
        )
        front, poke, count = depth_terms(frame, projection, rasterizer, args)
        residuals.append(normalized_block(front))
        residuals.append(normalized_block(poke))
        sil = silhouette_terms(frame, projection, args)
        residuals.append(normalized_block(sil))
    residuals.append(np.asarray([params[0] / float(args.sigma_log_scale)], dtype=np.float64))
    residuals.append(params[1:4] / float(args.sigma_rotation_rad))
    residuals.append(params[4:7] / float(args.sigma_translation_m))
    return np.concatenate([part.reshape(-1) for part in residuals])


def depth_terms(
    frame: FrameData,
    projection: np.ndarray,
    rasterizer: Rasterizer,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    # Project mesh samples, sample observed depth at those pixels on the source plane.
    fx, fy, cx, cy = [frame.K_depth[0, 0], frame.K_depth[1, 1], frame.K_depth[0, 2], frame.K_depth[1, 2]]
    z = projection[:, 2]
    positive = z > 1.0e-6
    uv = np.full((len(projection), 2), np.nan, dtype=np.float64)
    uv[positive, 0] = fx * projection[positive, 0] / z[positive] + cx
    uv[positive, 1] = fy * projection[positive, 1] / z[positive] + cy
    in_bounds = (
        positive
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < frame.depth_m.shape[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < frame.depth_m.shape[0])
    )
    front = np.zeros(len(projection), dtype=np.float64)
    poke = np.zeros(len(projection), dtype=np.float64)
    if np.any(in_bounds):
        x = np.clip(np.rint(uv[in_bounds, 0]).astype(np.int64), 0, frame.depth_m.shape[1] - 1)
        y = np.clip(np.rint(uv[in_bounds, 1]).astype(np.int64), 0, frame.depth_m.shape[0] - 1)
        owned = frame.mask_source[y, x]
        valid = owned & frame.depth_valid[y, x]
        measured = frame.depth_m[y, x]
        mesh_z = projection[in_bounds, 2]
        behind = np.zeros(len(valid), dtype=np.float64)
        poke_res = np.zeros(len(valid), dtype=np.float64)
        behind[valid] = np.maximum(0.0, mesh_z[valid] - measured[valid] - float(args.depth_behind_tolerance_m))
        poke_res[valid] = np.maximum(0.0, measured[valid] - mesh_z[valid] - float(args.depth_front_tolerance_m))
        front[in_bounds] = behind
        poke[in_bounds] = poke_res
    front = np.clip(front, 0.0, float(args.max_depth_residual_m)) / float(args.sigma_depth_m)
    poke = np.clip(poke, 0.0, float(args.max_depth_residual_m)) / float(args.sigma_depth_m)
    return front, poke, int(np.count_nonzero(in_bounds & frame.mask_source[
        np.clip(np.rint(np.nan_to_num(uv[in_bounds, 0], nan=0.0)).astype(np.int64), 0, frame.depth_m.shape[1] - 1),
        np.clip(np.rint(np.nan_to_num(uv[in_bounds, 1], nan=0.0)).astype(np.int64), 0, frame.depth_m.shape[0] - 1),
    ]))


def silhouette_terms(frame: FrameData, projection: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    # Project into the mask plane and measure distance to the mask silhouette.
    A_mask_from_source = frame.A_mask_from_source
    fx, fy, cx, cy = [frame.K_depth[0, 0], frame.K_depth[1, 1], frame.K_depth[0, 2], frame.K_depth[1, 2]]
    z = projection[:, 2]
    positive = z > 1.0e-6
    uv = np.full((len(projection), 2), np.nan, dtype=np.float64)
    uv[positive, 0] = fx * projection[positive, 0] / z[positive] + cx
    uv[positive, 1] = fy * projection[positive, 1] / z[positive] + cy
    uv = np.c_[uv, np.ones(len(uv))]
    uv_mask = (A_mask_from_source @ uv.T).T[:, :2]
    mask_h, mask_w = frame.mask_source.shape[0], frame.mask_source.shape[1]
    in_bounds = (
        positive
        & np.isfinite(uv_mask).all(axis=1)
        & (uv_mask[:, 0] >= 0)
        & (uv_mask[:, 0] < mask_w)
        & (uv_mask[:, 1] >= 0)
        & (uv_mask[:, 1] < mask_h)
    )
    outside = np.zeros(len(projection), dtype=np.float64)
    if np.any(in_bounds):
        x = np.clip(np.rint(uv_mask[in_bounds, 0]).astype(np.int64), 0, mask_w - 1)
        y = np.clip(np.rint(uv_mask[in_bounds, 1]).astype(np.int64), 0, mask_h - 1)
        if args._mask_distance is None:
            mask = cv2.resize(
                frame.mask_source.astype(np.uint8),
                (mask_w, mask_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            args._mask_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float64)
        outside[in_bounds] = args._mask_distance[y, x]
    outside = np.clip(outside, 0.0, float(args.max_silhouette_px)) / float(args.sigma_silhouette_px)
    return outside


def run(args: argparse.Namespace) -> dict[str, Any]:
    mesh = load_mesh(args.mesh_prior_anchor_camera)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    pivot = np.median(mesh_vertices, axis=0)
    mesh_centered = mesh_vertices - pivot[None, :]
    annotations = load_json(args.annotations)
    anchor_T = None
    for frame in annotations["frames"]:
        if int(frame["frame_idx"]) == int(args.anchor_frame):
            anchor_T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
            break
    if anchor_T is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} absent from annotations")
    frames = load_frames(args, int(args.anchor_frame), anchor_T)
    mesh_surface = sample_mesh_surface(mesh, int(args.max_prior_surface_points), int(args.seed) + 500)
    mesh_projection = sample_mesh_surface(mesh, int(args.max_projection_points), int(args.seed) + 900)
    x0 = initial_params(frames, int(args.anchor_frame))
    args._mask_distance = None
    rasterizer = Rasterizer(args.device)
    before = residual_vector(x0, mesh_centered, mesh_surface - pivot[None, :], mesh_projection - pivot[None, :], frames, rasterizer, args)
    result = least_squares(
        lambda x: residual_vector(x, mesh_centered, mesh_surface - pivot[None, :], mesh_projection - pivot[None, :], frames, rasterizer, args),
        x0,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=int(args.max_nfev),
        verbose=2 if args.verbose else 0,
    )
    after = residual_vector(result.x, mesh_centered, mesh_surface - pivot[None, :], mesh_projection - pivot[None, :], frames, rasterizer, args)
    scale, R, t = unpack(result.x)
    aligned_vertices = apply_sim3(mesh_centered, scale, R, t)
    out_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=mesh_faces, process=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "sam3d_ghost_lite_sim3_aligned_anchor_camera.ply"
    out_mesh.export(mesh_path)

    # Per-frame evaluation: observed-to-mesh + first-hit depth + hand gaps.
    surface_tree = cKDTree(aligned_vertices)
    rows = {}
    hands = np.load(args.hand_npz) if args.hand_npz else None
    hand_rows = {}
    T_by_frame = {int(f.frame_idx): f.T_world_camera for f in frames}
    if hands is not None:
        hand_frame_idx = hands["frame_idx"].astype(int).tolist()
        for pos, idx in enumerate(hand_frame_idx):
            if idx not in T_by_frame:
                continue
            for side in ("left", "right"):
                if int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                    continue
                vw = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
                T_f = T_by_frame[idx]
                vc = world_to_camera(vw, T_f)
                v_anchor = world_to_camera(vw, anchor_T)
                d, _ = surface_tree.query(v_anchor, k=1)
                hand_rows[(idx, side)] = {
                    "min_distance_to_aligned_mesh_m": float(np.min(d)),
                    "median_distance_to_aligned_mesh_m": float(np.median(d)),
                    "hand_camera_z_percentiles_m": np.percentile(vc[:, 2], [5, 50, 95]).astype(float).tolist(),
                }
    for frame in frames:
        d_obs, _ = surface_tree.query(frame.observed_anchor, k=1)
        obj_depth = np.median(frame.observed_anchor[:, 2])
        row = {
            "observed_to_aligned_mesh_median_m": float(np.median(d_obs)),
            "observed_to_aligned_mesh_p95_m": float(np.percentile(d_obs, 95.0)),
            "observed_anchor_median_depth_m": float(obj_depth),
        }
        for (idx, side), hr in hand_rows.items():
            if idx == frame.frame_idx:
                row[f"hand_{side}_min_gap_m"] = hr["min_distance_to_aligned_mesh_m"]
        rows[str(frame.frame_idx)] = row

    report = {
        "schema": "sam3d_ghost_lite_global_sim3_alignment_v1",
        "status": "ok" if bool(result.success) else "optimizer_incomplete",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "single_global_sim3_multi_frame_ghost_lite",
        "claim_scope": (
            "One global similarity transform fit to multi-frame observed metric front-surface surfels, "
            "object-owned first-hit depth, and mask silhouette. Generated SAM3D faces remain render-only; "
            "no collision, sign, contact, or nonpenetration authority."
        ),
        "mesh_prior_anchor_camera": str(args.mesh_prior_anchor_camera),
        "anchor_frame": int(args.anchor_frame),
        "used_frames": [int(f.frame_idx) for f in frames],
        "pivot_anchor_camera_m": pivot.tolist(),
        "sim3": {
            "scale": float(scale),
            "rotation": R.tolist(),
            "translation_anchor_camera_m": t.tolist(),
            "translation_world_m": (t @ anchor_T[:3, :3] + anchor_T[:3, 3]).tolist(),
        },
        "parameters": {
            "sigma_observed_m": float(args.sigma_observed_m),
            "sigma_depth_m": float(args.sigma_depth_m),
            "depth_behind_tolerance_m": float(args.depth_behind_tolerance_m),
            "depth_front_tolerance_m": float(args.depth_front_tolerance_m),
            "sigma_silhouette_px": float(args.sigma_silhouette_px),
        },
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "mesh_out": str(mesh_path),
        "frame_metrics": rows,
    }
    report_path = args.output_dir / "qc_sam3d_ghost_lite_sim3_alignment.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frame_metrics"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-prior-anchor-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--hand-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--min-observed-points", type=int, default=100)
    parser.add_argument("--min-ownership-fraction", type=float, default=0.90)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-observed-points", type=int, default=400)
    parser.add_argument("--max-prior-surface-points", type=int, default=4000)
    parser.add_argument("--max-projection-points", type=int, default=600)
    parser.add_argument("--sigma-observed-m", type=float, default=0.012)
    parser.add_argument("--sigma-depth-m", type=float, default=0.020)
    parser.add_argument("--depth-behind-tolerance-m", type=float, default=0.006)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.006)
    parser.add_argument("--max-depth-residual-m", type=float, default=0.120)
    parser.add_argument("--max-observed-residual-m", type=float, default=0.080)
    parser.add_argument("--sigma-silhouette-px", type=float, default=4.0)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--sigma-log-scale", type=float, default=0.15)
    parser.add_argument("--sigma-rotation-rad", type=float, default=0.25)
    parser.add_argument("--sigma-translation-m", type=float, default=0.060)
    parser.add_argument("--max-depth-terms-per-frame", type=int, default=2500)
    parser.add_argument("--max-nfev", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
