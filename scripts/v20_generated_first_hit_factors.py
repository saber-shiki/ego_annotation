#!/usr/bin/env python3
"""Dynamic generated-mesh first-hit factors for the V20 pose solver.

This module deliberately keeps generated-completion factors separate from the
frozen observed-only P15 factor contract.  It reuses the existing, audited
PyTorch3D raster and mask/hand ownership implementation, but rebuilds the
visible first-hit and bidirectional silhouette correspondences from the
current object poses at every outer iteration.

Generated faces remain a diagnostic visible-pose/render hypothesis.  Loading
this module does not promote them to collision, contact, SDF, or signed-volume
authority.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class GeneratedImageFactor:
    frame_idx: int
    T_world_camera: np.ndarray
    K_raster: np.ndarray
    depth_points_canonical: np.ndarray
    depth_observed_z: np.ndarray
    depth_weight: np.ndarray
    silhouette_points_canonical: np.ndarray
    silhouette_target_uv: np.ndarray
    silhouette_kind: np.ndarray
    evaluation_hit: np.ndarray
    evaluation_signed_depth: np.ndarray


@dataclass
class GeneratedFactorContext:
    builder: Any
    mesh_path: Path
    mesh_sha256: str
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    source_frames: list[Any]
    K_raster: np.ndarray
    rasterizer: Any


def numeric_summary(values: np.ndarray | list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_factor_builder(path: Path) -> Any:
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("v20_generated_factor_builder", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated first-hit factor builder: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _builder_args(args: Any) -> SimpleNamespace:
    """Create the exact argument contract consumed by the existing builder."""
    return SimpleNamespace(
        annotations=args.annotations,
        pose_report=args.pose_report,
        factor_pose_report=args.pose_report,
        hand_npz=args.generated_visible_hand_npz,
        mano_faces_pkl=args.generated_visible_mano_faces_pkl,
        object_id=args.object_id,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        raster_size=int(args.generated_visible_raster_size),
        source_size=int(args.generated_visible_source_size),
        min_frames=int(args.generated_visible_min_frames),
        min_observed_points=int(args.generated_visible_min_observed_points),
        min_ownership_fraction=float(args.generated_visible_min_ownership_fraction),
        hand_unknown_dilation_px=int(args.generated_visible_hand_unknown_dilation_px),
        max_removed_fraction_for_full_weight=float(
            args.generated_visible_max_removed_fraction_for_full_weight
        ),
        min_depth_factor_weight=float(args.generated_visible_min_depth_factor_weight),
        boundary_downweight_radius_px=float(
            args.generated_visible_boundary_downweight_radius_px
        ),
        boundary_weight=float(args.generated_visible_boundary_weight),
        max_observed_factors_per_frame=int(
            args.generated_visible_max_observed_factors_per_frame
        ),
        max_outside_silhouette_factors_per_frame=int(
            args.generated_visible_max_outside_factors_per_frame
        ),
        max_missing_silhouette_factors_per_frame=int(
            args.generated_visible_max_missing_factors_per_frame
        ),
    )


def create_context(args: Any) -> GeneratedFactorContext | None:
    mesh_value = getattr(args, "generated_visible_mesh", None)
    if mesh_value is None:
        return None
    mesh_path = Path(mesh_value).expanduser().resolve()
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise RuntimeError(f"missing generated visible mesh: {mesh_path}")
    builder_path = Path(args.generated_visible_factor_builder_script).expanduser().resolve()
    builder = import_factor_builder(builder_path)
    source_frames, K_raster = builder.load_frames(_builder_args(args))
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [part for part in mesh.geometry.values() if isinstance(part, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"generated visible mesh scene has no triangle mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"generated visible input is not a triangle mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(vertices) == 0
        or len(faces) == 0
        or not np.isfinite(vertices).all()
    ):
        raise RuntimeError(f"invalid generated visible mesh arrays: {mesh_path}")
    rasterizer = builder.BatchFirstHitRasterizer(
        faces,
        int(args.generated_visible_raster_size),
        np.asarray(K_raster, dtype=np.float64),
        str(args.generated_visible_device),
    )
    return GeneratedFactorContext(
        builder=builder,
        mesh_path=mesh_path,
        mesh_sha256=sha256_file(mesh_path),
        mesh_vertices=vertices,
        mesh_faces=faces,
        source_frames=source_frames,
        K_raster=np.asarray(K_raster, dtype=np.float64),
        rasterizer=rasterizer,
    )


def rebuild_factors(
    context: GeneratedFactorContext,
    pose_by_frame: dict[int, tuple[np.ndarray, np.ndarray]],
    args: Any,
) -> tuple[dict[int, GeneratedImageFactor], dict[str, Any]]:
    """Rerender the generated mesh and rebuild factors at current poses."""
    missing = sorted(
        int(frame.frame_idx)
        for frame in context.source_frames
        if int(frame.frame_idx) not in pose_by_frame
    )
    if missing:
        raise RuntimeError(
            "generated first-hit frames lack current object poses: "
            f"{missing[:20]}"
        )
    current_frames = [
        replace(
            frame,
            rotation_world_object=np.asarray(
                pose_by_frame[int(frame.frame_idx)][0], dtype=np.float64
            ),
            translation_world_object=np.asarray(
                pose_by_frame[int(frame.frame_idx)][1], dtype=np.float64
            ),
        )
        for frame in context.source_frames
    ]
    builder_args = _builder_args(args)
    factors: dict[int, GeneratedImageFactor] = {}
    metrics: list[dict[str, Any]] = []
    all_signed_depth: list[np.ndarray] = []
    batch_size = int(args.generated_visible_render_batch_size)
    if batch_size < 1:
        raise RuntimeError("generated visible render batch size must be positive")
    for start in range(0, len(current_frames), batch_size):
        chunk = current_frames[start : start + batch_size]
        camera_vertices = [
            context.builder.canonical_to_camera(
                context.mesh_vertices,
                frame.rotation_world_object,
                frame.translation_world_object,
                frame.T_world_camera,
            )
            for frame in chunk
        ]
        zbuf, rendered = context.rasterizer.render(camera_vertices)
        for local_pos, frame in enumerate(chunk):
            observed_px = np.rint(frame.observed_uv).astype(np.int64)
            observed_px[:, 0] = np.clip(
                observed_px[:, 0], 0, int(args.generated_visible_raster_size) - 1
            )
            observed_px[:, 1] = np.clip(
                observed_px[:, 1], 0, int(args.generated_visible_raster_size) - 1
            )
            observed_hit = rendered[
                local_pos, observed_px[:, 1], observed_px[:, 0]
            ]
            signed_depth_full = np.full(
                len(observed_hit), np.nan, dtype=np.float64
            )
            signed_depth_full[observed_hit] = (
                zbuf[local_pos, observed_px[observed_hit, 1], observed_px[observed_hit, 0]]
                - frame.observed_camera[observed_hit, 2]
            )
            signed_depth = signed_depth_full[np.isfinite(signed_depth_full)]
            all_signed_depth.append(signed_depth)
            raw, frame_metrics = context.builder.build_frame_factors(
                frame,
                zbuf[local_pos],
                rendered[local_pos],
                context.K_raster,
                builder_args,
            )
            frame_metrics = {
                **frame_metrics,
                "true_first_hit_coverage_fraction": float(np.mean(observed_hit))
                if len(observed_hit)
                else 0.0,
                "true_first_hit_signed_depth_m": numeric_summary(signed_depth),
                "true_first_hit_abs_depth_m": numeric_summary(np.abs(signed_depth)),
            }
            idx = int(frame.frame_idx)
            if idx in factors:
                raise RuntimeError(f"duplicate generated first-hit frame: {idx}")
            factors[idx] = GeneratedImageFactor(
                frame_idx=idx,
                T_world_camera=np.asarray(frame.T_world_camera, dtype=np.float64),
                K_raster=context.K_raster.copy(),
                depth_points_canonical=np.asarray(raw["depth_points"], dtype=np.float64),
                depth_observed_z=np.asarray(raw["depth_z"], dtype=np.float64),
                depth_weight=np.asarray(raw["depth_weight"], dtype=np.float64),
                silhouette_points_canonical=np.asarray(
                    raw["silhouette_points"], dtype=np.float64
                ),
                silhouette_target_uv=np.asarray(
                    raw["silhouette_target_uv"], dtype=np.float64
                ),
                silhouette_kind=np.asarray(raw["silhouette_kind"], dtype=np.int8),
                evaluation_hit=np.asarray(observed_hit, dtype=bool),
                evaluation_signed_depth=signed_depth_full,
            )
            metrics.append(frame_metrics)
    depth_count = int(sum(len(f.depth_observed_z) for f in factors.values()))
    silhouette_count = int(sum(len(f.silhouette_kind) for f in factors.values()))
    signed_depth_all = (
        np.concatenate(all_signed_depth) if all_signed_depth else np.empty(0, dtype=np.float64)
    )
    return factors, {
        "frame_count": int(len(factors)),
        "frame_ids": sorted(factors),
        "depth_factor_count": depth_count,
        "silhouette_factor_count": silhouette_count,
        "outside_silhouette_factor_count": int(
            sum(np.count_nonzero(f.silhouette_kind == 0) for f in factors.values())
        ),
        "missing_silhouette_factor_count": int(
            sum(np.count_nonzero(f.silhouette_kind == 1) for f in factors.values())
        ),
        "initial_silhouette_iou_median": float(
            np.median([row["initial_silhouette_iou"] for row in metrics])
        )
        if metrics
        else None,
        "initial_observed_hit_fraction_median": float(
            np.median([row["initial_observed_hit_fraction"] for row in metrics])
        )
        if metrics
        else None,
        "true_first_hit_coverage_fraction_median": float(
            np.median([row["true_first_hit_coverage_fraction"] for row in metrics])
        )
        if metrics
        else None,
        "true_first_hit_signed_depth_m": numeric_summary(signed_depth_all),
        "true_first_hit_abs_depth_m": numeric_summary(np.abs(signed_depth_all)),
        "mesh_path": str(context.mesh_path),
        "mesh_sha256": context.mesh_sha256,
        "mesh_vertices": int(len(context.mesh_vertices)),
        "mesh_faces": int(len(context.mesh_faces)),
        "raster_size": int(args.generated_visible_raster_size),
        "correspondence_policy": "true_first_hit_rerendered_at_current_outer_pose",
        "per_frame": metrics,
    }
