#!/usr/bin/env python3
"""P15-aware multi-frame first-hit Sim(3) alignment for a SAM3D mesh.

This is the corrected GHOST-lite object-prior alignment.  A single small
canonical Sim(3) correction is shared across frames, while every frame uses
its P15 object SE(3) followed by the metric camera transform.  The objective
uses only:

* true rasterized first-hit depth at trusted observed-surface surfel pixels,
* bidirectional silhouette support (observed->rendered and rendered->mask),
* zero-centred priors on the *incremental* canonical correction.

Nearest-neighbour/Chamfer is diagnostic only and never drives alignment.  The
optimization mesh is a QEM simplification; the solved transform is applied to
the original complete topology.  Generated faces remain render-only and are
not collision, sign, contact, or nonpenetration authority.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class FrameData:
    frame_idx: int
    T_world_camera: np.ndarray
    R_world_object: np.ndarray
    t_world_object: np.ndarray
    observed_camera: np.ndarray
    observed_uv: np.ndarray
    target_mask: np.ndarray
    hand_unknown_mask: np.ndarray
    distance_to_target_or_unknown: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid triangle mesh: {path}")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError(f"non-finite mesh vertices: {path}")
    return mesh


def resize_mask(mask: np.ndarray, raster_size: int) -> np.ndarray:
    interpolation = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
    return cv2.resize(mask.astype(np.uint8), (raster_size, raster_size), interpolation=interpolation) > 0


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    simplified = source.simplify_quadric_decimation(int(target_faces), boundary_weight=10.0)
    result = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices, dtype=np.float64),
        faces=np.asarray(simplified.triangles, dtype=np.int64),
        process=False,
    )
    if len(result.faces) == 0 or not np.isfinite(result.vertices).all():
        raise RuntimeError("QEM simplification produced an invalid mesh")
    return result


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]


def world_to_camera(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T_world_camera[:3, 3]) @ T_world_camera[:3, :3]


def anchor_camera_to_canonical(
    vertices: np.ndarray,
    T_world_camera_anchor: np.ndarray,
    R_world_object_anchor: np.ndarray,
    t_world_object_anchor: np.ndarray,
) -> np.ndarray:
    world = camera_to_world(vertices, T_world_camera_anchor)
    return (world - t_world_object_anchor[None, :]) @ R_world_object_anchor


def canonical_to_camera(
    canonical: np.ndarray,
    R_world_object: np.ndarray,
    t_world_object: np.ndarray,
    T_world_camera: np.ndarray,
) -> np.ndarray:
    world = canonical @ R_world_object.T + t_world_object[None, :]
    return world_to_camera(world, T_world_camera)


def unpack(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return (
        float(np.exp(params[0])),
        Rotation.from_rotvec(np.asarray(params[1:4])).as_matrix(),
        np.asarray(params[4:7], dtype=np.float64),
    )


def correct_canonical(canonical: np.ndarray, pivot: np.ndarray, params: np.ndarray) -> np.ndarray:
    scale, rotation, translation = unpack(params)
    return scale * ((canonical - pivot[None, :]) @ rotation.T) + pivot[None, :] + translation[None, :]


def summarize(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median_m": float(np.median(arr)),
        "q25_m": float(np.percentile(arr, 25.0)),
        "q75_m": float(np.percentile(arr, 75.0)),
        "p05_m": float(np.percentile(arr, 5.0)),
        "p95_m": float(np.percentile(arr, 95.0)),
    }


class FirstHitRasterizer:
    def __init__(self, faces: np.ndarray, raster_size: int, K_source: np.ndarray, source_size: int, device: str):
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

        self.device = device
        self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        self.size = int(raster_size)
        ratio = float(raster_size) / float(source_size)
        K = np.asarray(K_source, dtype=np.float64).copy()
        # OpenCV half-pixel resize contract: x' = r * (x + 0.5) - 0.5.
        K[0, 0] *= ratio
        K[1, 1] *= ratio
        K[0, 2] = ratio * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = ratio * (K[1, 2] + 0.5) - 0.5
        self.K = K
        cameras = cameras_from_opencv_projection(
            torch.eye(3, dtype=torch.float32, device=device)[None],
            torch.zeros(1, 3, dtype=torch.float32, device=device),
            torch.tensor(K, dtype=torch.float32, device=device)[None],
            torch.tensor([[self.size, self.size]], dtype=torch.float32, device=device),
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=self.size,
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=False,
            ),
        )

    def render(self, vertices_camera: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        from pytorch3d.structures import Meshes

        verts = [torch.tensor(v, dtype=torch.float32, device=self.device) for v in vertices_camera]
        faces = [self.faces] * len(verts)
        with torch.no_grad():
            fragments = self.rasterizer(Meshes(verts=verts, faces=faces))
        zbuf = fragments.zbuf[..., 0].detach().cpu().numpy().astype(np.float64)
        pix = fragments.pix_to_face[..., 0].detach().cpu().numpy()
        valid = np.isfinite(zbuf) & (zbuf > 1.0e-6) & (pix >= 0)
        return np.where(valid, zbuf, np.nan), valid


def load_frames(args: argparse.Namespace) -> tuple[list[FrameData], np.ndarray, dict[int, dict[str, Any]]]:
    annotations = load_json(args.annotations)
    pose = load_json(args.pose_graph)
    pose_by_idx = {int(r["frame_idx"]): r for r in pose["pose_rows"]}
    hands = np.load(args.hand_npz)
    hand_position = {int(idx): pos for pos, idx in enumerate(hands["frame_idx"].astype(int).tolist())}
    selected: list[FrameData] = []
    K_source: np.ndarray | None = None
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        if idx < args.frame_start or idx > args.frame_end:
            continue
        if idx != args.anchor_frame and (idx - args.frame_start) % args.frame_step != 0:
            continue
        if idx not in pose_by_idx:
            continue
        obj = frame.get("objects", [{}])[0]
        visible = obj.get("visible_geometry_candidate") if isinstance(obj, dict) else None
        if not isinstance(visible, dict):
            continue
        ownership = visible.get("first_surface_depth_ownership")
        if not isinstance(ownership, dict) or ownership.get("enabled") is not True or ownership.get("fail_closed") is True:
            continue
        if float(ownership.get("retained_fraction") or 0.0) < args.min_ownership_fraction:
            continue
        obs = np.asarray(visible.get("camera_vertices_sample_m"), dtype=np.float64)
        obs = obs[np.isfinite(obs).all(axis=1) & (obs[:, 2] > 1.0e-6)]
        if len(obs) < args.min_observed_points:
            continue
        if len(obs) > args.max_observed_points:
            rng = np.random.default_rng(args.seed + idx)
            obs = obs[rng.choice(len(obs), size=args.max_observed_points, replace=False)]
        fx, fy, cx, cy = np.asarray(visible["intrinsics_fx_fy_cx_cy"], dtype=np.float64).tolist()
        K_frame = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        if K_source is None:
            K_source = K_frame
        elif not np.allclose(K_frame, K_source, atol=1.0e-3):
            raise RuntimeError("per-frame intrinsics differ; batch rasterizer contract is invalid")
        ratio = float(args.raster_size) / float(args.source_size)
        u_source = fx * obs[:, 0] / obs[:, 2] + cx
        v_source = fy * obs[:, 1] / obs[:, 2] + cy
        uv = np.column_stack((
            ratio * (u_source + 0.5) - 0.5,
            ratio * (v_source + 0.5) - 0.5,
        ))
        uv = np.rint(uv).astype(np.int64)
        keep = (
            (uv[:, 0] >= 0) & (uv[:, 0] < args.raster_size)
            & (uv[:, 1] >= 0) & (uv[:, 1] < args.raster_size)
        )
        obs = obs[keep]
        uv = uv[keep]
        mask_path = Path(str(visible.get("mask_path") or obj.get("mask_path") or ""))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None or np.count_nonzero(mask_raw) == 0:
            continue
        mask = resize_mask(mask_raw > 0, args.raster_size)
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        # Hand projection is explicit unknown support: rendered object pixels in
        # this region are neither foreground evidence nor known background.
        hand_unknown = np.zeros_like(mask, dtype=np.uint8)
        if idx in hand_position:
            pos = hand_position[idx]
            for side in ("left", "right"):
                if int(np.asarray(hands[f"{side}_valid"])[pos]) != 1:
                    continue
                hand_world = np.asarray(hands[f"{side}_vertices_world_m"][pos], dtype=np.float64)
                hand_camera = world_to_camera(hand_world, T_world_camera)
                valid_hand = np.isfinite(hand_camera).all(axis=1) & (hand_camera[:, 2] > 1.0e-6)
                if np.count_nonzero(valid_hand) < 3:
                    continue
                hand_camera = hand_camera[valid_hand]
                hu_source = fx * hand_camera[:, 0] / hand_camera[:, 2] + cx
                hv_source = fy * hand_camera[:, 1] / hand_camera[:, 2] + cy
                huv = np.column_stack((
                    ratio * (hu_source + 0.5) - 0.5,
                    ratio * (hv_source + 0.5) - 0.5,
                ))
                huv = huv[np.isfinite(huv).all(axis=1)]
                huv = huv[
                    (huv[:, 0] >= 0) & (huv[:, 0] < args.raster_size)
                    & (huv[:, 1] >= 0) & (huv[:, 1] < args.raster_size)
                ]
                if len(huv) >= 3:
                    cv2.fillConvexPoly(hand_unknown, cv2.convexHull(np.rint(huv).astype(np.int32)), 1)
        if args.hand_unknown_dilation_px > 0:
            radius = int(args.hand_unknown_dilation_px)
            hand_unknown = cv2.dilate(
                hand_unknown,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
            )
        hand_unknown = hand_unknown > 0
        target_or_unknown = mask | hand_unknown
        distance_to_target_or_unknown = cv2.distanceTransform(
            (~target_or_unknown).astype(np.uint8), cv2.DIST_L2, 3
        ).astype(np.float64)
        row = pose_by_idx[idx]
        selected.append(FrameData(
            frame_idx=idx,
            T_world_camera=T_world_camera,
            R_world_object=np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64),
            t_world_object=np.asarray(row["translation_world_m"], dtype=np.float64),
            observed_camera=obs,
            observed_uv=uv,
            target_mask=mask,
            hand_unknown_mask=hand_unknown,
            distance_to_target_or_unknown=distance_to_target_or_unknown,
        ))
    if len(selected) < args.min_frames:
        raise RuntimeError(f"only {len(selected)} eligible frames, need {args.min_frames}")
    assert K_source is not None
    return selected, K_source, pose_by_idx


def frame_residuals(frame: FrameData, zbuf: np.ndarray, rendered: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    u = frame.observed_uv[:, 0]
    v = frame.observed_uv[:, 1]
    hit_z = zbuf[v, u]
    hit = np.isfinite(hit_z)
    depth = np.zeros(len(hit_z), dtype=np.float64)
    depth[hit] = np.clip(
        hit_z[hit] - frame.observed_camera[hit, 2],
        -args.max_depth_residual_m,
        args.max_depth_residual_m,
    ) / args.sigma_depth_m
    # A missing first hit is a coverage defect, not a fake metric depth value.
    distance_to_render = cv2.distanceTransform((~rendered).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float64)
    missing = np.clip(distance_to_render[v, u], 0.0, args.max_silhouette_px) / args.sigma_silhouette_px
    # Rendered -> target: penalize any generated silhouette in known background.
    outside_values = frame.distance_to_target_or_unknown[rendered]
    outside_rms = float(np.sqrt(np.mean(np.square(np.clip(outside_values, 0.0, args.max_silhouette_px))))) if len(outside_values) else args.max_silhouette_px
    outside = np.asarray([outside_rms / args.sigma_silhouette_px], dtype=np.float64)
    norm = np.sqrt(max(1, len(depth)))
    return [depth / norm, missing / norm, outside]


def evaluate(
    params: np.ndarray,
    canonical: np.ndarray,
    pivot: np.ndarray,
    frames: list[FrameData],
    rasterizer: FirstHitRasterizer,
    args: argparse.Namespace,
    include_priors: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    corrected = correct_canonical(canonical, pivot, params)
    vertices_camera = [
        canonical_to_camera(corrected, f.R_world_object, f.t_world_object, f.T_world_camera)
        for f in frames
    ]
    zbuf, rendered = rasterizer.render(vertices_camera)
    blocks: list[np.ndarray] = []
    metrics: dict[str, Any] = {}
    for i, frame in enumerate(frames):
        blocks.extend(frame_residuals(frame, zbuf[i], rendered[i], args))
        u, v = frame.observed_uv[:, 0], frame.observed_uv[:, 1]
        hit = np.isfinite(zbuf[i, v, u])
        signed = zbuf[i, v[hit], u[hit]] - frame.observed_camera[hit, 2]
        known = ~frame.hand_unknown_mask
        intersection = np.count_nonzero(rendered[i] & frame.target_mask & known)
        union = np.count_nonzero((rendered[i] | frame.target_mask) & known)
        metrics[str(frame.frame_idx)] = {
            "first_hit_minus_observed_depth_m": summarize(signed),
            "surfel_first_hit_coverage_fraction": float(np.mean(hit)),
            "silhouette_iou": float(intersection / max(1, union)),
            "rendered_pixels": int(np.count_nonzero(rendered[i])),
            "target_pixels": int(np.count_nonzero(frame.target_mask)),
            "hand_unknown_pixels": int(np.count_nonzero(frame.hand_unknown_mask)),
        }
    if include_priors:
        blocks.extend([
            np.asarray([params[0] / args.sigma_log_scale]),
            params[1:4] / args.sigma_rotation_rad,
            params[4:7] / args.sigma_translation_m,
        ])
    return np.concatenate([b.reshape(-1) for b in blocks]), metrics


def numerical_jacobian(fun: Any, x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    base = fun(x)
    jac = np.empty((len(base), len(x)), dtype=np.float64)
    for j, step in enumerate(steps):
        xp = x.copy(); xm = x.copy()
        xp[j] += step; xm[j] -= step
        jac[:, j] = (fun(xp) - fun(xm)) / (2.0 * step)
    return jac


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_mesh = load_mesh(args.mesh_anchor_camera)
    optimization_mesh = simplify_mesh(full_mesh, args.optimization_faces)
    frames, K_source, pose_by_idx = load_frames(args)
    anchor = next((f for f in frames if f.frame_idx == args.anchor_frame), None)
    if anchor is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} is not selected")
    canonical_full = anchor_camera_to_canonical(
        np.asarray(full_mesh.vertices, dtype=np.float64),
        anchor.T_world_camera,
        anchor.R_world_object,
        anchor.t_world_object,
    )
    canonical_opt = anchor_camera_to_canonical(
        np.asarray(optimization_mesh.vertices, dtype=np.float64),
        anchor.T_world_camera,
        anchor.R_world_object,
        anchor.t_world_object,
    )
    pivot = np.median(canonical_full, axis=0)
    rasterizer = FirstHitRasterizer(
        np.asarray(optimization_mesh.faces, dtype=np.int64),
        args.raster_size,
        K_source,
        args.source_size,
        args.device,
    )
    x0 = np.zeros(7, dtype=np.float64)
    fun = lambda x: evaluate(x, canonical_opt, pivot, frames, rasterizer, args, True)[0]
    before, before_metrics = evaluate(x0, canonical_opt, pivot, frames, rasterizer, args, True)
    steps = np.asarray([
        args.jac_log_scale_step,
        args.jac_rotation_step_rad, args.jac_rotation_step_rad, args.jac_rotation_step_rad,
        args.jac_translation_step_m, args.jac_translation_step_m, args.jac_translation_step_m,
    ], dtype=np.float64)
    lower = np.asarray([
        -args.max_abs_log_scale,
        -args.max_abs_rotation_rad, -args.max_abs_rotation_rad, -args.max_abs_rotation_rad,
        -args.max_abs_translation_m, -args.max_abs_translation_m, -args.max_abs_translation_m,
    ])
    upper = -lower
    result = least_squares(
        fun,
        x0,
        jac=lambda x: numerical_jacobian(fun, x, steps),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale=np.asarray([0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02]),
        max_nfev=args.max_nfev,
        verbose=2 if args.verbose else 0,
    )
    after, after_metrics = evaluate(result.x, canonical_opt, pivot, frames, rasterizer, args, True)
    corrected_full = correct_canonical(canonical_full, pivot, result.x)
    aligned_anchor_camera = canonical_to_camera(
        corrected_full,
        anchor.R_world_object,
        anchor.t_world_object,
        anchor.T_world_camera,
    )
    out_mesh = trimesh.Trimesh(
        vertices=aligned_anchor_camera,
        faces=np.asarray(full_mesh.faces, dtype=np.int64),
        process=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.output_dir / "sam3d_p15_first_hit_aligned_anchor_camera.ply"
    canonical_path = args.output_dir / "sam3d_p15_first_hit_aligned_canonical.ply"
    out_mesh.export(mesh_path)
    trimesh.Trimesh(vertices=corrected_full, faces=np.asarray(full_mesh.faces), process=False).export(canonical_path)
    scale, rotation, translation = unpack(result.x)
    report = {
        "schema": "sam3d_p15_first_hit_global_sim3_alignment_v1",
        "status": "ok" if result.success else "optimizer_incomplete",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "p15_per_frame_se3_plus_shared_canonical_sim3_true_first_hit",
        "claim_scope": (
            "True first-hit visible-surface alignment under P15 per-frame object SE(3). "
            "Generated faces remain render-only and provide no signed/collision/contact authority."
        ),
        "mesh_input_anchor_camera": str(args.mesh_anchor_camera),
        "mesh_output_anchor_camera": str(mesh_path),
        "mesh_output_canonical": str(canonical_path),
        "anchor_frame": args.anchor_frame,
        "used_frames": [f.frame_idx for f in frames],
        "optimization_mesh": {
            "full_vertices": int(len(full_mesh.vertices)),
            "full_faces": int(len(full_mesh.faces)),
            "simplified_vertices": int(len(optimization_mesh.vertices)),
            "simplified_faces": int(len(optimization_mesh.faces)),
        },
        "sim3_increment": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation_canonical_m": translation.tolist(),
            "parameters": result.x.tolist(),
        },
        "objective": {
            "terms": [
                "true_first_hit_depth_at_trusted_surfel_pixels",
                "observed_surfel_to_rendered_silhouette_distance",
                "rendered_silhouette_to_object_owned_mask_distance",
                "zero_centered_incremental_sim3_priors",
            ],
            "nearest_neighbor_drives_optimization": False,
            "raster_size": args.raster_size,
            "sigma_depth_m": args.sigma_depth_m,
            "sigma_silhouette_px": args.sigma_silhouette_px,
            "sigma_log_scale": args.sigma_log_scale,
            "sigma_rotation_rad": args.sigma_rotation_rad,
            "sigma_translation_m": args.sigma_translation_m,
            "increment_bounds": {
                "max_abs_log_scale": args.max_abs_log_scale,
                "max_abs_rotation_component_rad": args.max_abs_rotation_rad,
                "max_abs_translation_component_m": args.max_abs_translation_m,
            },
        },
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": int(result.njev) if result.njev is not None else None,
            "cost": float(result.cost),
        },
        "frame_metrics_before": before_metrics,
        "frame_metrics_after": after_metrics,
    }
    report_path = args.output_dir / "qc_sam3d_p15_first_hit_alignment.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "sim3_increment": report["sim3_increment"],
        "residual_rms_before": report["residual_rms_before"],
        "residual_rms_after": report["residual_rms_after"],
        "mesh": str(mesh_path),
        "report": str(report_path),
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-anchor-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-graph", type=Path, required=True)
    parser.add_argument("--hand-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, default=92)
    parser.add_argument("--frame-start", type=int, default=30)
    parser.add_argument("--frame-end", type=int, default=146)
    parser.add_argument("--frame-step", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--source-size", type=int, default=1408)
    parser.add_argument("--raster-size", type=int, default=704)
    parser.add_argument("--optimization-faces", type=int, default=20000)
    parser.add_argument("--min-observed-points", type=int, default=100)
    parser.add_argument("--max-observed-points", type=int, default=1200)
    parser.add_argument("--min-ownership-fraction", type=float, default=0.90)
    parser.add_argument("--hand-unknown-dilation-px", type=int, default=4)
    parser.add_argument("--sigma-depth-m", type=float, default=0.008)
    parser.add_argument("--max-depth-residual-m", type=float, default=0.050)
    parser.add_argument("--sigma-silhouette-px", type=float, default=3.0)
    parser.add_argument("--max-silhouette-px", type=float, default=40.0)
    parser.add_argument("--sigma-log-scale", type=float, default=0.10)
    parser.add_argument("--sigma-rotation-rad", type=float, default=0.12)
    parser.add_argument("--sigma-translation-m", type=float, default=0.025)
    parser.add_argument("--max-abs-log-scale", type=float, default=0.15)
    parser.add_argument("--max-abs-rotation-rad", type=float, default=0.25)
    parser.add_argument("--max-abs-translation-m", type=float, default=0.05)
    parser.add_argument("--jac-log-scale-step", type=float, default=0.002)
    parser.add_argument("--jac-rotation-step-rad", type=float, default=0.002)
    parser.add_argument("--jac-translation-step-m", type=float, default=0.001)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
