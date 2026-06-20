#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Solve a continuous MANO interval trajectory with object/visual constraints.

This is the workbench solver path, not a diagnostic classifier.  It jointly
optimizes, over a contiguous interval, current-space MANO root translation,
root/wrist orientation delta, and finger articulation delta.  The zero state is
side-specific exact HaWoR MANO replay mapped to the current V18 bridge surface.

The physical objective is:
  - stay close to the current visible/depth hand observation,
  - stay temporally smooth in root motion, wrist orientation, and articulation,
  - move penetrating MANO surface out of trusted observed object surface, and
  - ignore hidden/free-space-conflicted completion volume as a force.

If this solver fails to make a coherent rendered hand trajectory, that is a
failure of a named scientific assumption, not an acceptance/count result.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_mano_object_constraint_state import frame_intrinsics, project  # noqa: E402
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources  # noqa: E402
from build_v18_observed_surface_mano_constraint_state import (  # noqa: E402
    VERTEX_OBSERVED_SUPPORTED,
    classify_object_vertices_against_depth,
    face_provenance,
)
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    HAND_EDGES,
    bridge_vertices_and_joints,
    load_source_arrays,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
    rotvec_to_matrix,
    similarity_from_to,
    source_npz_for_hand,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    inverse_object,
    load_json,
    load_mesh,
    numeric_summary,
    object_vec_to_world,
    pose_map,
    write_json,
)

DEFAULT_ANNOTATIONS = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/"
    "task5_tomato_960/annotations_v18_full.json"
)
DEFAULT_POSE_REPORT = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json"
)
DEFAULT_MESH = Path(
    "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/"
    "completed_mesh_frame929prior_frame806scale_v1/object_obj_tomato_scale_sane_completed_mesh_labeled.ply"
)
DEFAULT_DEPTH = Path(
    "/data2/ego_annotation_outputs/v18_unidepth_extension/complete_depth_root/task5_tomato_960/"
    "unidepth_metric/unidepth_metric_depth_v3.npz"
)
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_task5_joint_mano_interval_solver_v1")
DEFAULT_LEFT_MANO = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_LEFT.pkl")
DEFAULT_WILOR_ROOT = Path("third_party/WiLoR")


@dataclass(frozen=True)
class FrameHandRow:
    frame_idx: int
    side: str
    frame: dict[str, Any]
    current_vertices_world: np.ndarray
    current_joints_world: np.ndarray
    raw_vertices_world: np.ndarray
    raw_joints_world: np.ndarray
    root_orient_axis_angle: np.ndarray
    hand_pose_axis_angle: np.ndarray
    betas: np.ndarray
    trans_world_m: np.ndarray
    similarity_scale: float
    similarity_rotation_raw_to_current: np.ndarray
    source_hawor_npz: Path
    source_frame_index: int
    constraint_indices: np.ndarray
    constraint_normals_world: np.ndarray
    constraint_depths_m: np.ndarray
    observed_initial_measure: dict[str, Any]
    observed_constraint_count: int
    object_rotation_world_from_object: np.ndarray
    object_translation_world_m: np.ndarray
    face_strict_observed: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="task5_tomato_960")
    p.add_argument("--object-id", default="object:obj_tomato")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT)
    p.add_argument("--completed-mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--depth-npz", type=Path, action="append", default=[DEFAULT_DEPTH])
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    p.add_argument("--wilor-mano-right", type=Path, default=None)
    p.add_argument("--wilor-mano-left", type=Path, default=DEFAULT_LEFT_MANO)
    p.add_argument("--hawor-left-shapedirs-x-fix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--start-frame", type=int, default=453)
    p.add_argument("--end-frame", type=int, default=508)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--support-margin-m", type=float, default=0.015)
    p.add_argument("--free-space-margin-m", type=float, default=0.025)
    p.add_argument("--penetration-epsilon-m", type=float, default=1.0e-5)
    p.add_argument("--max-constraints-per-frame", type=int, default=96)
    p.add_argument("--max-optimizer-iterations", type=int, default=120)
    p.add_argument("--visible-shift-limit-px", type=float, default=12.0)
    p.add_argument("--depth-shift-limit-m", type=float, default=0.035)
    p.add_argument("--max-translation-m", type=float, default=0.045)
    p.add_argument("--max-root-delta-rad", type=float, default=0.30)
    p.add_argument("--max-pose-delta-rad", type=float, default=0.45)
    p.add_argument("--translation-prior-weight", type=float, default=2.0e3)
    p.add_argument("--root-prior-weight", type=float, default=1.5e2)
    p.add_argument("--pose-prior-weight", type=float, default=7.5e1)
    p.add_argument("--smooth-weight", type=float, default=5.0e3)
    p.add_argument("--accel-weight", type=float, default=1.0e4)
    p.add_argument("--observed-penetration-weight", type=float, default=3.0e5)
    p.add_argument("--visible-hinge-weight", type=float, default=8.0e2)
    p.add_argument("--depth-hinge-weight", type=float, default=2.0e4)
    p.add_argument("--bound-hinge-weight", type=float, default=3.0e3)
    p.add_argument("--sample-vertex-count-for-render", type=int, default=160)
    p.add_argument("--active-set-iterations", type=int, default=6, help="Maximum active-set passes. Each pass optimizes, remeasures full observed-surface penetration, and expands constraints. A closed pass adds zero constraints.")
    return p.parse_args()


def project_world(points_world: np.ndarray, frame: dict[str, Any], side: str) -> np.ndarray | None:
    intr = frame_intrinsics(frame, side)
    if intr is None:
        return None
    r_c2w, t_c2w = frame_camera_pose(frame)
    return project(points_world, r_c2w, t_c2w, intr)


def world_to_camera(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    r_c2w, t_c2w = frame_camera_pose(frame)
    return (points_world - t_c2w[None, :]) @ r_c2w


def sample_ids(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)


def load_models(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    patch_legacy_mano_loader()
    mano_cls = load_wilor_mano_class(args.wilor_root)
    right_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    models: dict[str, Any] = {}
    if "right" in args.sides:
        if not right_path.exists():
            raise FileNotFoundError(f"missing right MANO model: {right_path}")
        models["right"] = mano_cls(model_path=str(right_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    if "left" in args.sides:
        left_path = args.wilor_mano_left
        if left_path is None or not left_path.exists():
            raise FileNotFoundError(f"missing left MANO model: {left_path}")
        left_model = mano_cls(model_path=str(left_path), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
        if bool(args.hawor_left_shapedirs_x_fix):
            with torch.no_grad():
                left_model.shapedirs[:, 0, :] *= -1
        models["left"] = left_model
    for m in models.values():
        m.eval()
    return models


def observed_constraints_for_hand(
    *,
    vertices_world: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    scene: Any,
    face_strict_observed: np.ndarray,
    frame_idx: int,
    max_constraints: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    r_obj, t_obj = pose
    vertices_object = inverse_object(vertices_world, r_obj, t_obj)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating = np.where(signed > float(eps))[0]
    if penetrating.size == 0:
        measure = {
            "frame_idx": frame_idx,
            "penetrating_vertex_count": 0,
            "observed_supported_penetrating_vertex_count": 0,
            "observed_supported_penetration_m": numeric_summary(np.asarray([], dtype=float)),
        }
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), measure
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = face_strict_observed[primitive_ids[valid]]
    obs_idx = penetrating[observed]
    obs_depth = signed[obs_idx]
    if len(obs_idx) > int(max_constraints):
        order = np.argsort(obs_depth)[::-1][: int(max_constraints)]
        obs_idx = obs_idx[order]
        obs_depth = obs_depth[order]
    normals_world = np.zeros((0, 3), dtype=float)
    depths = np.zeros((0,), dtype=float)
    if len(obs_idx):
        closest_obs = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[obs_idx], dtype=np.float32)))["points"].numpy().astype(float)
        disp = closest_obs - vertices_object[obs_idx]
        norms = np.linalg.norm(disp, axis=1)
        good = norms > 1.0e-12
        normals_world = object_vec_to_world(disp[good] / norms[good, None], r_obj)
        depths = obs_depth[good]
        obs_idx = obs_idx[good]
    measure = {
        "frame_idx": frame_idx,
        "penetrating_vertex_count": int(penetrating.size),
        "observed_supported_penetrating_vertex_count": int(len(obs_idx)),
        "observed_supported_penetration_m": numeric_summary(depths),
    }
    return obs_idx.astype(np.int64), normals_world.astype(float), depths.astype(float), measure


def build_rows(args: argparse.Namespace, side: str) -> tuple[list[FrameHandRow], dict[str, Any], Any]:
    annotations = load_json(args.annotations)
    frames = [f for f in as_list(annotations.get("frames")) if isinstance(f, dict)]
    frames_by_idx = {int(f["frame_idx"]): f for f in frames}
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    mesh = load_mesh(args.completed_mesh)
    vertices_object = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices_object.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    depth_rows = load_depth_sources(args.depth_npz)
    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    rows: list[FrameHandRow] = []
    object_depth_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frame_idx in range(int(args.start_frame), int(args.end_frame) + 1):
        frame = frames_by_idx.get(frame_idx)
        pose = poses.get(frame_idx)
        if frame is None or pose is None:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_frame_or_object_pose"})
            continue
        vertex_classes, obj_summary = classify_object_vertices_against_depth(
            frame=frame,
            vertices_object=vertices_object,
            pose=pose,
            depth_row=depth_rows.get(frame_idx),
            support_margin_m=float(args.support_margin_m),
            free_space_margin_m=float(args.free_space_margin_m),
        )
        object_depth_summaries.append(obj_summary)
        prov = face_provenance(vertex_classes, faces)
        strict = np.asarray(prov["observed_supported_strict"], dtype=bool)
        hand = None
        for h in as_list(frame.get("hands")):
            if isinstance(h, dict) and str(h.get("hand_side")) == side:
                hand = h
                break
        if hand is None:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_hand"})
            continue
        arrays = bridge_vertices_and_joints(hand, bridge_cache)
        source_info = source_npz_for_hand(hand)
        if arrays is None or source_info is None:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_bridge_or_source"})
            continue
        current_vertices, current_joints = arrays
        source_path, source_frame = source_info
        source = load_source_arrays(source_cache, source_path)
        required = [
            f"{side}_vertices_world_m",
            f"{side}_joints_world_m",
            f"{side}_root_orient_axis_angle",
            f"{side}_hand_pose_axis_angle",
            f"{side}_betas",
            f"{side}_trans_world_m",
        ]
        missing = [key for key in required if key not in source]
        if missing:
            skipped.append({"frame_idx": frame_idx, "side": side, "reason": "source_missing_arrays", "missing": missing})
            continue
        raw_vertices = np.asarray(source[f"{side}_vertices_world_m"][source_frame], dtype=float)
        raw_joints = np.asarray(source[f"{side}_joints_world_m"][source_frame], dtype=float)
        scale, rot, _trans, _err = similarity_from_to(raw_vertices, current_vertices)
        cidx, normals, depths, measure = observed_constraints_for_hand(
            vertices_world=current_vertices,
            pose=pose,
            scene=scene,
            face_strict_observed=strict,
            frame_idx=frame_idx,
            max_constraints=int(args.max_constraints_per_frame),
            eps=float(args.penetration_epsilon_m),
        )
        r_obj, t_obj = pose
        rows.append(
            FrameHandRow(
                frame_idx=frame_idx,
                side=side,
                frame=frame,
                current_vertices_world=current_vertices,
                current_joints_world=current_joints,
                raw_vertices_world=raw_vertices,
                raw_joints_world=raw_joints,
                root_orient_axis_angle=np.asarray(source[f"{side}_root_orient_axis_angle"][source_frame], dtype=float),
                hand_pose_axis_angle=np.asarray(source[f"{side}_hand_pose_axis_angle"][source_frame], dtype=float),
                betas=np.asarray(source[f"{side}_betas"][source_frame], dtype=float),
                trans_world_m=np.asarray(source[f"{side}_trans_world_m"][source_frame], dtype=float),
                similarity_scale=float(scale),
                similarity_rotation_raw_to_current=rot.astype(float),
                source_hawor_npz=source_path,
                source_frame_index=int(source_frame),
                constraint_indices=cidx,
                constraint_normals_world=normals,
                constraint_depths_m=depths,
                observed_initial_measure=measure,
                observed_constraint_count=int(len(depths)),
                object_rotation_world_from_object=np.asarray(r_obj, dtype=float),
                object_translation_world_m=np.asarray(t_obj, dtype=float),
                face_strict_observed=strict.astype(bool),
            )
        )
    meta = {
        "side": side,
        "requested_start_frame": int(args.start_frame),
        "requested_end_frame": int(args.end_frame),
        "row_count": int(len(rows)),
        "skipped": skipped,
        "object_depth_summaries": object_depth_summaries[:5],
    }
    return rows, meta, scene


def active_constraints_from_vertices(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, max_constraints: int, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating = np.where(signed > float(eps))[0]
    if penetrating.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(row.face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = row.face_strict_observed[primitive_ids[valid]]
    idx = penetrating[observed]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    closest_obj = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[idx], dtype=np.float32)))["points"].numpy().astype(float)
    disp = closest_obj - vertices_object[idx]
    norms = np.linalg.norm(disp, axis=1)
    good = norms > 1.0e-12
    idx = idx[good]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    normals_world = object_vec_to_world(disp[good] / norms[good, None], row.object_rotation_world_from_object)
    depths = signed[idx]
    current_to_query = vertices_world[idx] - row.current_vertices_world[idx]
    required = np.sum(normals_world * current_to_query, axis=1) + depths
    order = np.argsort(required)[::-1]
    if len(order) > int(max_constraints):
        order = order[: int(max_constraints)]
    return idx[order].astype(np.int64), normals_world[order].astype(float), required[order].astype(float)


def merge_constraints(base_idx: np.ndarray, base_normals: np.ndarray, base_depths: np.ndarray, new_idx: np.ndarray, new_normals: np.ndarray, new_depths: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best: dict[int, tuple[np.ndarray, float]] = {}
    for idx, normal, depth in zip(base_idx.astype(int), base_normals, base_depths.astype(float)):
        best[int(idx)] = (np.asarray(normal, dtype=float), float(depth))
    for idx, normal, depth in zip(new_idx.astype(int), new_normals, new_depths.astype(float)):
        old = best.get(int(idx))
        if old is None or float(depth) > old[1]:
            best[int(idx)] = (np.asarray(normal, dtype=float), float(depth))
    if not best:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
    items = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)[: int(cap)]
    idx = np.asarray([k for k, _ in items], dtype=np.int64)
    normals = np.stack([v[0] for _, v in items]).astype(float)
    depths = np.asarray([v[1] for _, v in items], dtype=float)
    return idx, normals, depths


def full_observed_surface_measure(vertices_world: np.ndarray, row: FrameHandRow, scene: Any, eps: float) -> dict[str, Any]:
    vertices_object = inverse_object(vertices_world, row.object_rotation_world_from_object, row.object_translation_world_m)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating = np.where(signed > float(eps))[0]
    if penetrating.size == 0:
        return {
            "penetrating_vertex_count": 0,
            "observed_supported_penetrating_vertex_count": 0,
            "observed_supported_penetration_m": numeric_summary(np.asarray([], dtype=float)),
        }
    closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[penetrating], dtype=np.float32)))
    primitive_ids = closest["primitive_ids"].numpy().astype(np.int64)
    valid = (primitive_ids >= 0) & (primitive_ids < len(row.face_strict_observed))
    observed = np.zeros_like(valid, dtype=bool)
    observed[valid] = row.face_strict_observed[primitive_ids[valid]]
    observed_depths = signed[penetrating][observed]
    return {
        "penetrating_vertex_count": int(penetrating.size),
        "observed_supported_penetrating_vertex_count": int(np.count_nonzero(observed)),
        "observed_supported_penetration_m": numeric_summary(observed_depths),
    }


def optimize_rows(rows: list[FrameHandRow], model: Any, args: argparse.Namespace, device: torch.device, scene: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        return {"status": "no_rows"}, []
    b = len(rows)
    root = torch.tensor(np.stack([r.root_orient_axis_angle for r in rows]).reshape(b, 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.stack([r.hand_pose_axis_angle for r in rows]).reshape(b, 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.stack([r.betas for r in rows]), dtype=torch.float32, device=device)
    trans = torch.tensor(np.stack([r.trans_world_m for r in rows]), dtype=torch.float32, device=device)
    base_root_mat = rotvec_to_matrix(root)
    base_pose_mat = rotvec_to_matrix(pose)
    with torch.no_grad():
        base_out = model(global_orient=base_root_mat, hand_pose=base_pose_mat, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_base_vertices = base_out.vertices.detach().cpu().numpy().astype(float)
        raw_base_joints = base_out.joints.detach().cpu().numpy().astype(float)
    replay_vertex_err = [np.linalg.norm(raw_base_vertices[i] - rows[i].raw_vertices_world, axis=1) for i in range(b)]
    replay_joint_err = [np.linalg.norm(raw_base_joints[i] - rows[i].raw_joints_world, axis=1) for i in range(b)]
    replay_ok = max(float(np.median(e)) for e in replay_vertex_err) <= 1.0e-5 and max(float(np.median(e)) for e in replay_joint_err) <= 1.0e-5

    root_delta = torch.zeros((b, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    pose_delta = torch.zeros((b, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    trans_delta = torch.zeros((b, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([root_delta, pose_delta, trans_delta], lr=0.35, max_iter=int(args.max_optimizer_iterations), line_search_fn="strong_wolfe")

    active_constraint_indices = [r.constraint_indices.copy() for r in rows]
    active_constraint_normals = [r.constraint_normals_world.copy() for r in rows]
    active_constraint_depths = [r.constraint_depths_m.copy() for r in rows]
    current_vertices_t = [torch.tensor(r.current_vertices_world, dtype=torch.float32, device=device) for r in rows]
    current_joints_t = [torch.tensor(r.current_joints_world, dtype=torch.float32, device=device) for r in rows]
    raw_base_vertices_t = torch.tensor(raw_base_vertices, dtype=torch.float32, device=device)
    raw_base_joints_t = torch.tensor(raw_base_joints, dtype=torch.float32, device=device)
    sim_scale_t = torch.tensor([r.similarity_scale for r in rows], dtype=torch.float32, device=device).reshape(b, 1, 1)
    sim_rot_t = torch.tensor(np.stack([r.similarity_rotation_raw_to_current for r in rows]), dtype=torch.float32, device=device)
    intr_t: list[torch.Tensor | None] = []
    base_uv: list[torch.Tensor | None] = []
    r_c2w_t: list[torch.Tensor] = []
    t_c2w_t: list[torch.Tensor] = []
    base_depth: list[torch.Tensor] = []
    for row in rows:
        intr = frame_intrinsics(row.frame, row.side)
        intr_t.append(None if intr is None else torch.tensor(intr, dtype=torch.float32, device=device))
        uv0 = project_world(row.current_joints_world, row.frame, row.side)
        base_uv.append(None if uv0 is None else torch.tensor(uv0, dtype=torch.float32, device=device))
        r_c2w, t_c2w = frame_camera_pose(row.frame)
        r_c2w_t.append(torch.tensor(r_c2w, dtype=torch.float32, device=device))
        t_c2w_t.append(torch.tensor(t_c2w, dtype=torch.float32, device=device))
        base_depth.append(torch.tensor(world_to_camera(row.current_joints_world, row.frame)[:, 2], dtype=torch.float32, device=device))

    def hypothesis() -> tuple[torch.Tensor, torch.Tensor]:
        new_root = rotvec_to_matrix(root_delta) @ base_root_mat
        new_pose = rotvec_to_matrix(pose_delta) @ base_pose_mat
        out = model(global_orient=new_root, hand_pose=new_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_delta_vertices = out.vertices - raw_base_vertices_t
        raw_delta_joints = out.joints - raw_base_joints_t
        mapped_vertices = sim_scale_t * torch.matmul(raw_delta_vertices, sim_rot_t.transpose(1, 2))
        mapped_joints = sim_scale_t * torch.matmul(raw_delta_joints, sim_rot_t.transpose(1, 2))
        verts = torch.stack(current_vertices_t, dim=0) + mapped_vertices + trans_delta[:, None, :]
        joints = torch.stack(current_joints_t, dim=0) + mapped_joints + trans_delta[:, None, :]
        return verts, joints

    def project_torch(points_world: torch.Tensor, i: int) -> torch.Tensor | None:
        intr = intr_t[i]
        if intr is None:
            return None
        cam = torch.matmul(points_world - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
        z = cam[:, 2].clamp_min(1.0e-5)
        fx, fy, cx, cy = intr
        return torch.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], dim=-1)

    def temporal_terms(x: torch.Tensor, weight: float) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        vel = x[1:] - x[:-1]
        loss = float(weight) * torch.mean(vel * vel)
        if x.shape[0] > 2:
            acc = x[2:] - 2.0 * x[1:-1] + x[:-2]
            loss = loss + float(args.accel_weight) * torch.mean(acc * acc)
        return loss

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        hyp_vertices, hyp_joints = hypothesis()
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss = loss + float(args.translation_prior_weight) * torch.mean(trans_delta * trans_delta)
        loss = loss + float(args.root_prior_weight) * torch.mean(root_delta * root_delta)
        loss = loss + float(args.pose_prior_weight) * torch.mean(pose_delta * pose_delta)
        loss = loss + temporal_terms(trans_delta, float(args.smooth_weight))
        loss = loss + temporal_terms(root_delta, float(args.smooth_weight))
        loss = loss + temporal_terms(pose_delta, float(args.smooth_weight))
        trans_norm = torch.linalg.norm(trans_delta, dim=1)
        root_norm = torch.linalg.norm(root_delta.reshape(b, 3), dim=1)
        pose_norm = torch.linalg.norm(pose_delta, dim=2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(trans_norm - float(args.max_translation_m)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(root_norm - float(args.max_root_delta_rad)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(pose_norm - float(args.max_pose_delta_rad)) ** 2)
        for i, row in enumerate(rows):
            if len(active_constraint_indices[i]):
                ids = torch.tensor(active_constraint_indices[i], dtype=torch.long, device=device)
                normals = torch.tensor(active_constraint_normals[i], dtype=torch.float32, device=device)
                depths = torch.tensor(active_constraint_depths[i], dtype=torch.float32, device=device)
                moved = hyp_vertices[i, ids] - current_vertices_t[i][ids]
                residual = torch.relu(depths - torch.sum(normals * moved, dim=1))
                loss = loss + float(args.observed_penetration_weight) * torch.mean(residual * residual)
            uv = project_torch(hyp_joints[i], i)
            if uv is not None and base_uv[i] is not None:
                shift = torch.linalg.norm(uv - base_uv[i], dim=1)
                loss = loss + float(args.visible_hinge_weight) * torch.mean(torch.relu(shift - float(args.visible_shift_limit_px)) ** 2)
            cam = torch.matmul(hyp_joints[i] - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
            depth_shift = torch.abs(cam[:, 2] - base_depth[i])
            loss = loss + float(args.depth_hinge_weight) * torch.mean(torch.relu(depth_shift - float(args.depth_shift_limit_m)) ** 2)
        loss.backward()
        return loss

    active_set_added_counts: list[int] = []
    active_set_closed = False
    active_set_pass_count = 0
    if replay_ok:
        for active_iter in range(max(1, int(args.active_set_iterations))):
            active_set_pass_count = active_iter + 1
            if active_iter > 0:
                optimizer = torch.optim.LBFGS([root_delta, pose_delta, trans_delta], lr=0.25, max_iter=int(args.max_optimizer_iterations), line_search_fn="strong_wolfe")
            optimizer.step(closure)
            with torch.no_grad():
                hyp_vertices_t, _hyp_joints_t = hypothesis()
                hyp_vertices_np = hyp_vertices_t.detach().cpu().numpy().astype(float)
            added_total = 0
            for i, row in enumerate(rows):
                new_idx, new_normals, new_depths = active_constraints_from_vertices(
                    hyp_vertices_np[i],
                    row,
                    scene,
                    int(args.max_constraints_per_frame),
                    float(args.penetration_epsilon_m),
                )
                before = len(active_constraint_indices[i])
                merged = merge_constraints(
                    active_constraint_indices[i],
                    active_constraint_normals[i],
                    active_constraint_depths[i],
                    new_idx,
                    new_normals,
                    new_depths,
                    max(int(args.max_constraints_per_frame), before),
                )
                active_constraint_indices[i], active_constraint_normals[i], active_constraint_depths[i] = merged
                added_total += max(0, len(active_constraint_indices[i]) - before)
            active_set_added_counts.append(int(added_total))
            if added_total == 0:
                active_set_closed = True
                break
    with torch.no_grad():
        hyp_vertices_t, hyp_joints_t = hypothesis()
        hyp_vertices = hyp_vertices_t.detach().cpu().numpy().astype(float)
        hyp_joints = hyp_joints_t.detach().cpu().numpy().astype(float)
        trans_np = trans_delta.detach().cpu().numpy().astype(float)
        root_np = root_delta.detach().cpu().numpy().reshape(b, 3).astype(float)
        pose_np = pose_delta.detach().cpu().numpy().astype(float)

    render_ids = sample_ids(778, int(args.sample_vertex_count_for_render))
    states: list[dict[str, Any]] = []
    initial_obs_max: list[float] = []
    final_linear_residual_max: list[float] = []
    final_full_observed_max: list[float] = []
    visible_max: list[float] = []
    depth_max: list[float] = []
    trans_max: list[float] = []
    root_max: list[float] = []
    pose_max: list[float] = []
    corrected_frames = 0
    for i, row in enumerate(rows):
        if len(active_constraint_indices[i]):
            moved = hyp_vertices[i, active_constraint_indices[i]] - row.current_vertices_world[active_constraint_indices[i]]
            residual = np.maximum(0.0, active_constraint_depths[i] - np.sum(active_constraint_normals[i] * moved, axis=1))
        else:
            residual = np.zeros((0,), dtype=float)
        init_max = float((row.observed_initial_measure.get("observed_supported_penetration_m") or {}).get("max") or 0.0)
        final_max = float(np.max(residual)) if residual.size else 0.0
        full_post = full_observed_surface_measure(hyp_vertices[i], row, scene, float(args.penetration_epsilon_m))
        full_post_max = float((full_post.get("observed_supported_penetration_m") or {}).get("max") or 0.0)
        uv0 = project_world(row.current_joints_world, row.frame, row.side)
        uv1 = project_world(hyp_joints[i], row.frame, row.side)
        if uv0 is not None and uv1 is not None:
            shift = np.linalg.norm(uv1 - uv0, axis=1)
            shift_max = float(np.max(shift))
            shift_med = float(np.median(shift))
        else:
            shift = np.zeros((0,), dtype=float)
            shift_max = float("nan")
            shift_med = float("nan")
        cam0 = world_to_camera(row.current_joints_world, row.frame)
        cam1 = world_to_camera(hyp_joints[i], row.frame)
        dshift = np.abs(cam1[:, 2] - cam0[:, 2])
        tnorm = float(np.linalg.norm(trans_np[i]))
        rnorm = float(np.linalg.norm(root_np[i]))
        pnorm = float(np.max(np.linalg.norm(pose_np[i], axis=1)))
        changed = tnorm > 1.0e-4 or rnorm > 1.0e-4 or pnorm > 1.0e-4
        if changed:
            corrected_frames += 1
        initial_obs_max.append(init_max)
        final_linear_residual_max.append(final_max)
        final_full_observed_max.append(full_post_max)
        if np.isfinite(shift_max):
            visible_max.append(shift_max)
        depth_max.append(float(np.max(dshift)))
        trans_max.append(tnorm)
        root_max.append(rnorm)
        pose_max.append(pnorm)
        states.append(
            {
                "frame_idx": int(row.frame_idx),
                "hand_side": row.side,
                "temporal_mano_state": "joint_continuous_mano_trajectory_correction",
                "source_hawor_npz": str(row.source_hawor_npz),
                "source_frame_index": int(row.source_frame_index),
                "optimized_translation_world_m": trans_np[i].astype(float).tolist(),
                "optimized_root_delta_axis_angle_rad": root_np[i].astype(float).tolist(),
                "optimized_hand_pose_delta_axis_angle_rad": pose_np[i].reshape(-1).astype(float).tolist(),
                "optimized_joints_world_m": hyp_joints[i].astype(float).tolist(),
                "optimized_vertices_world_sample_m": hyp_vertices[i, render_ids].astype(float).tolist(),
                "optimized_vertices_sample_ids": render_ids.astype(int).tolist(),
                "initial_observed_surface_penetration_m": row.observed_initial_measure.get("observed_supported_penetration_m"),
                "final_active_constraint_residual_after_solver_m": numeric_summary(residual),
                "full_observed_surface_penetration_after_solver_m": full_post.get("observed_supported_penetration_m"),
                "full_observed_supported_penetrating_vertex_count_after_solver": int(full_post.get("observed_supported_penetrating_vertex_count", 0)),
                "visible_joint_shift_px": {"count": int(len(shift)), "median": shift_med, "max": shift_max},
                "joint_camera_depth_shift_m": {"count": int(len(dshift)), "median": float(np.median(dshift)), "max": float(np.max(dshift))},
                "delta_norms": {"translation_m": tnorm, "root_rad": rnorm, "max_pose_joint_rad": pnorm},
            }
        )
    interval = {
        "hand_side": rows[0].side,
        "start_frame": int(rows[0].frame_idx),
        "end_frame": int(rows[-1].frame_idx),
        "frame_count": int(len(rows)),
        "solver": "joint_root_translation_root_orientation_and_articulation",
        "optimizer_ran": bool(replay_ok),
        "replay_ok": bool(replay_ok),
        "raw_replay_vertex_error_median_m": numeric_summary(np.asarray([float(np.median(e)) for e in replay_vertex_err], dtype=float)),
        "raw_replay_joint_error_median_m": numeric_summary(np.asarray([float(np.median(e)) for e in replay_joint_err], dtype=float)),
        "corrected_frame_count": int(corrected_frames),
        "initial_observed_surface_penetration_max_m": numeric_summary(np.asarray(initial_obs_max, dtype=float)),
        "final_active_constraint_residual_after_solver_max_m": numeric_summary(np.asarray(final_linear_residual_max, dtype=float)),
        "full_observed_surface_penetration_after_solver_max_m": numeric_summary(np.asarray(final_full_observed_max, dtype=float)),
        "visible_joint_shift_max_px": numeric_summary(np.asarray(visible_max, dtype=float)),
        "joint_camera_depth_shift_max_m": numeric_summary(np.asarray(depth_max, dtype=float)),
        "translation_delta_norm_m": numeric_summary(np.asarray(trans_max, dtype=float)),
        "root_delta_norm_rad": numeric_summary(np.asarray(root_max, dtype=float)),
        "pose_delta_max_joint_norm_rad": numeric_summary(np.asarray(pose_max, dtype=float)),
        "active_set_added_constraint_counts": active_set_added_counts,
        "active_set_pass_count": int(active_set_pass_count),
        "active_set_closed": bool(active_set_closed),
        "active_constraint_count_final": numeric_summary(np.asarray([len(x) for x in active_constraint_indices], dtype=float)),
    }
    return interval, states


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    models = load_models(args, device)
    intervals: list[dict[str, Any]] = []
    per_frame_states: list[dict[str, Any]] = []
    build_meta: dict[str, Any] = {}
    for side in args.sides:
        rows, meta, scene = build_rows(args, side)
        build_meta[side] = meta
        if not rows:
            intervals.append({"hand_side": side, "start_frame": int(args.start_frame), "end_frame": int(args.end_frame), "frame_count": 0, "solver": "joint_root_translation_root_orientation_and_articulation", "state": "no_rows"})
            continue
        interval, states = optimize_rows(rows, models[side], args, device, scene)
        interval["interval_id"] = f"{side}_{rows[0].frame_idx:04d}_{rows[-1].frame_idx:04d}_joint_mano"
        for st in states:
            st["interval_id"] = interval["interval_id"]
        intervals.append(interval)
        per_frame_states.extend(states)
    report = {
        "method": "solve_v18_joint_mano_interval_trajectory",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": "Continuous interval MANO trajectory correction candidate: root translation, root orientation, and finger articulation optimized jointly against visible/depth compatibility and trusted observed object surface.",
        "inputs": {"annotations": str(args.annotations), "pose_report": str(args.pose_report), "completed_mesh": str(args.completed_mesh), "depth_npz": [str(p) for p in args.depth_npz]},
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in {"depth_npz"}},
        "build_meta": build_meta,
        "summary": {"interval_count": int(len(intervals)), "per_frame_state_count": int(len(per_frame_states)), "frame_span": [int(args.start_frame), int(args.end_frame)], "sides": list(args.sides)},
        "intervals": intervals,
        "per_frame_states": per_frame_states,
        "scientific_test": "If this sequence still looks incoherent after rendering, the remaining failure is not that the optimizer was missing root/articulation coupling; it is a conflict among hand observation, observed tomato geometry, camera/depth alignment, and interval observability.",
    }
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v18_joint_mano_interval_trajectory_state.json"
    write_json(out, report)
    print(json.dumps({"output": str(out), "summary": report["summary"], "intervals": intervals}, indent=2)[:6000])


if __name__ == "__main__":
    main()
