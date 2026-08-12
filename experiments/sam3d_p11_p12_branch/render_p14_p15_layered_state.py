#!/usr/bin/env python3
"""Render an experimental layered P14/P15 state with full MANO surfaces.

This is an additive renderer adapter.  It reuses the canonical V19 projection and
video encoding helpers, but consumes source-neutral object layers and recovers the
full 778-vertex metric MANO meshes from annotation references.  Object, observed
surface, and hand triangles share one deterministic far-to-near painter order so
hand/object depth order is visible in the diagnostic artifact.  The painter uses
mean triangle depth and is a review renderer, not a metric z-buffer evaluator.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
import render_v19_rigid_state_artifact as canonical  # noqa: E402

SCHEMA = "v19_experimental_p14_p15_full_mano_layered_render_v1"
ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    "generated_complete_prior_underlay": (205, 75, 190),
    "observed_metric_surface_overlay": (65, 205, 75),
    "mano_left_full_surface": (245, 175, 55),
    "mano_right_full_surface": (45, 145, 255),
}
ROLE_ALPHA: dict[str, float] = {
    "generated_complete_prior_underlay": 0.46,
    "observed_metric_surface_overlay": 0.72,
    "mano_left_full_surface": 0.62,
    "mano_right_full_surface": 0.62,
}


@dataclasses.dataclass(frozen=True)
class ObjectLayer:
    name: str
    role: str
    mesh_path: Path
    vertices: np.ndarray
    faces: np.ndarray
    source_face_count: int
    face_selection: dict[str, Any]
    color: tuple[int, int, int]
    alpha: float
    depth_priority_bias_m: float


@dataclasses.dataclass(frozen=True)
class SceneLayer:
    name: str
    role: str
    vertices_world: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int]
    alpha: float
    face_budget: int
    depth_priority_bias_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Optional source frame_idx subset")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--generated-face-budget", type=int, default=12000)
    parser.add_argument("--observed-face-budget", type=int, default=0, help="0 means all observed faces")
    parser.add_argument("--mano-face-budget", type=int, default=0, help="0 means all MANO faces")
    parser.add_argument("--observed-depth-priority-bias-m", type=float, default=0.002)
    parser.add_argument("--local-world-padding-m", type=float, default=0.08)
    parser.add_argument("--camera-frustum-depth-m", type=float, default=0.055)
    parser.add_argument("--export-glb-frame", type=int, default=109)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--encode-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--path-rewrite", action="append", default=[], metavar="OLD=NEW")
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_output(path: Path, replace: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_faces(faces: np.ndarray, selector: dict[str, Any], path: Path) -> np.ndarray:
    mode = str(selector.get("mode", "all"))
    if mode == "all":
        return faces
    if mode == "contiguous_range":
        start = int(selector.get("start", -1))
        stop = int(selector.get("stop", -1))
        if start < 0 or stop <= start or stop > len(faces):
            raise RuntimeError(f"invalid face range [{start}, {stop}) for {len(faces)} faces: {path}")
        return faces[start:stop]
    raise RuntimeError(f"unsupported face-selection mode {mode!r}: {path}")


def load_object_layers(
    state: dict[str, Any], rewrites: list[tuple[str, str]], observed_bias_m: float
) -> tuple[list[ObjectLayer], list[dict[str, Any]]]:
    geometry = state.get("object_geometry") if isinstance(state.get("object_geometry"), dict) else {}
    rows = geometry.get("render_layers_back_to_front")
    if not isinstance(rows, list) or len(rows) < 1:
        raise RuntimeError("experimental render state has no object render layers")
    cache: dict[Path, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    layers: list[ObjectLayer] = []
    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"render layer {index} is not an object")
        role = str(row.get("role", ""))
        if role not in ("generated_complete_prior_underlay", "observed_metric_surface_overlay"):
            raise RuntimeError(f"unsupported object render role: {role!r}")
        path = canonical.rewrite_path(row.get("mesh"), rewrites)
        if path is None:
            raise RuntimeError(f"render layer {index} has no mesh")
        path = require_file(path, f"{role} mesh")
        if path not in cache:
            cache[path] = canonical.load_mesh(path)
        vertices, source_faces, mesh_summary = cache[path]
        selector = row.get("face_selection") if isinstance(row.get("face_selection"), dict) else {"mode": "all"}
        faces = select_faces(source_faces, selector, path)
        bias = float(observed_bias_m) if role == "observed_metric_surface_overlay" else 0.0
        layers.append(
            ObjectLayer(
                name=f"object_layer_{index}_{role}",
                role=role,
                mesh_path=path,
                vertices=vertices,
                faces=faces,
                source_face_count=int(len(source_faces)),
                face_selection=selector,
                color=ROLE_COLORS[role],
                alpha=ROLE_ALPHA[role],
                depth_priority_bias_m=bias,
            )
        )
        summaries.append(
            {
                "role": role,
                "mesh": str(path),
                "mesh_sha256": sha256_file(path),
                "mesh_vertices": int(len(vertices)),
                "mesh_source_faces": int(len(source_faces)),
                "selected_layer_faces": int(len(faces)),
                "face_selection": selector,
                "depth_priority_bias_m": bias,
                "canonical_loader_summary": mesh_summary,
            }
        )
    roles = [layer.role for layer in layers]
    if roles.count("generated_complete_prior_underlay") != 1 or roles.count("observed_metric_surface_overlay") != 1:
        raise RuntimeError(f"expected one generated and one observed layer, got {roles}")
    return layers, summaries


class ManoArchiveCache:
    def __init__(self, rewrites: list[tuple[str, str]]) -> None:
        self.rewrites = rewrites
        self.archives: dict[Path, Any] = {}
        self.rows_read = 0
        self.sample_error_m: list[float] = []
        self.sources: dict[str, dict[str, Any]] = {}

    def archive(self, raw_path: str, description: str) -> tuple[Path, Any]:
        path = canonical.rewrite_path(raw_path, self.rewrites)
        if path is None:
            raise RuntimeError(f"missing {description} path")
        path = require_file(path, description)
        if path not in self.archives:
            self.archives[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return path, self.archives[path]

    def hand_mesh(self, frame_idx: int, hand: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        side = str(hand.get("hand_side", ""))
        if side not in ("left", "right"):
            raise RuntimeError(f"frame {frame_idx}: unsupported hand side {side!r}")
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        reference = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
        bridge_path, bridge = self.archive(str(reference.get("bridge_npz", "")), "MANO bridge archive")
        array_name = str(reference.get("bridge_vertices_world_array", ""))
        row_index = int(reference.get("bridge_row_index", -1))
        if array_name not in bridge.files or row_index < 0 or row_index >= len(bridge[array_name]):
            raise RuntimeError(
                f"frame {frame_idx} {side}: invalid bridge reference array={array_name!r} row={row_index}"
            )
        vertices = np.asarray(bridge[array_name][row_index], dtype=np.float64)
        if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
            raise RuntimeError(f"frame {frame_idx} {side}: invalid full MANO vertices {vertices.shape}")
        if "frame_idx" in bridge.files and int(bridge["frame_idx"][row_index]) != frame_idx:
            raise RuntimeError(f"frame {frame_idx} {side}: bridge row frame mismatch")
        if "hand_side" in bridge.files and str(bridge["hand_side"][row_index]) != side:
            raise RuntimeError(f"frame {frame_idx} {side}: bridge row side mismatch")

        source_path, source = self.archive(str(reference.get("source_hawor_npz", "")), "HaWoR MANO source archive")
        face_key = f"{side}_faces"
        if face_key not in source.files:
            raise RuntimeError(f"frame {frame_idx} {side}: source archive lacks {face_key}")
        faces = np.asarray(source[face_key], dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or int(faces.max()) >= 778:
            raise RuntimeError(f"frame {frame_idx} {side}: invalid MANO face topology {faces.shape}")

        sample_indices = np.asarray(metric.get("vertices_sample_indices") or [], dtype=np.int64)
        sample_world = np.asarray(metric.get("vertices_world_sample_m") or [], dtype=np.float64)
        sample_error = None
        if sample_indices.ndim == 1 and sample_world.shape == (len(sample_indices), 3) and len(sample_indices) > 0:
            if int(sample_indices.min()) < 0 or int(sample_indices.max()) >= len(vertices):
                raise RuntimeError(f"frame {frame_idx} {side}: sample indices outside full MANO surface")
            sample_error = float(np.max(np.linalg.norm(vertices[sample_indices] - sample_world, axis=1)))
            self.sample_error_m.append(sample_error)
            if sample_error > 1.0e-5:
                raise RuntimeError(
                    f"frame {frame_idx} {side}: full MANO bridge disagrees with inline samples by {sample_error:.6g} m"
                )
        self.rows_read += 1
        if str(bridge_path) not in self.sources:
            self.sources[str(bridge_path)] = {
                "kind": "full_metric_mano_vertices",
                "array": array_name,
                "sha256": sha256_file(bridge_path),
            }
        if str(source_path) not in self.sources:
            self.sources[str(source_path)] = {
                "kind": "mano_face_topology",
                "arrays": ["left_faces", "right_faces"],
                "sha256": sha256_file(source_path),
            }
        return vertices, faces, {
            "side": side,
            "bridge": str(bridge_path),
            "bridge_array": array_name,
            "bridge_row_index": row_index,
            "source_faces": str(source_path),
            "face_array": face_key,
            "sample_reproduction_max_error_m": sample_error,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "full_surface_rows_read": int(self.rows_read),
            "vertices_per_hand": 778,
            "sample_reproduction_max_error_m": max(self.sample_error_m) if self.sample_error_m else None,
            "sample_reproduction_median_error_m": float(np.median(self.sample_error_m)) if self.sample_error_m else None,
            "archives": self.sources,
        }

    def close(self) -> None:
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()


def choose_face_ids(face_count: int, budget: int) -> np.ndarray:
    if budget <= 0 or face_count <= budget:
        return np.arange(face_count, dtype=np.int64)
    return np.linspace(0, face_count - 1, budget, dtype=np.int64)


def project_world_axes(
    points_world: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    width: int,
    height: int,
    horizontal_axis: int,
    vertical_axis: int,
    depth_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    extent = np.maximum(max_xyz - min_xyz, 1.0e-6)
    uv = np.empty((len(points_world), 2), dtype=np.float64)
    uv[:, 0] = (points_world[:, horizontal_axis] - min_xyz[horizontal_axis]) / extent[horizontal_axis] * width
    uv[:, 1] = height - (points_world[:, vertical_axis] - min_xyz[vertical_axis]) / extent[vertical_axis] * height
    depth = points_world[:, depth_axis] - min_xyz[depth_axis] + 1.0
    return uv, depth


def rasterize_scene(
    background: np.ndarray,
    layers: list[SceneLayer],
    projected: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    height, width = background.shape[:2]
    if len(layers) != len(projected):
        raise RuntimeError("scene layer/projection count mismatch")
    per_layer: list[dict[str, Any]] = []
    depth_chunks: list[np.ndarray] = []
    layer_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    selected_by_layer: list[np.ndarray] = []
    for layer_index, (layer, (uv, depth)) in enumerate(zip(layers, projected)):
        valid_face = np.all(np.isfinite(uv[layer.faces]), axis=(1, 2)) & np.all(depth[layer.faces] > 0.01, axis=1)
        candidates = np.flatnonzero(valid_face)
        selected = candidates[choose_face_ids(len(candidates), int(layer.face_budget))] if len(candidates) else np.asarray([], dtype=np.int64)
        selected_by_layer.append(selected)
        means = depth[layer.faces[selected]].mean(axis=1) - float(layer.depth_priority_bias_m) if len(selected) else np.asarray([], dtype=np.float64)
        depth_chunks.append(means)
        layer_chunks.append(np.full(len(selected), layer_index, dtype=np.int32))
        face_chunks.append(selected.astype(np.int64))
        per_layer.append(
            {
                "name": layer.name,
                "role": layer.role,
                "input_faces": int(len(layer.faces)),
                "valid_faces": int(len(candidates)),
                "selected_faces": int(len(selected)),
                "face_budget": int(layer.face_budget),
                "visible_pixels": 0,
            }
        )
    if not depth_chunks or sum(len(chunk) for chunk in depth_chunks) == 0:
        return background.copy(), {"layers": per_layer, "visible_pixels_total": 0}, np.zeros((height, width), dtype=np.int16)

    all_depth = np.concatenate(depth_chunks)
    all_layers = np.concatenate(layer_chunks)
    all_faces = np.concatenate(face_chunks)
    order = np.argsort(all_depth, kind="stable")[::-1]
    label_map = np.zeros((height, width), dtype=np.int16)
    color_canvas = np.zeros_like(background)
    observed_owner_map = np.zeros((height, width), dtype=np.uint8)
    observed_color_canvas = np.zeros_like(background)
    observed_layer_indices = {
        index for index, layer in enumerate(layers) if layer.role == "observed_metric_surface_overlay"
    }
    if len(observed_layer_indices) > 1:
        raise RuntimeError("scene contains more than one observed metric surface layer")
    rasterized = np.zeros(len(layers), dtype=np.int64)
    for item in order:
        layer_index = int(all_layers[item])
        face_id = int(all_faces[item])
        layer = layers[layer_index]
        uv, depth = projected[layer_index]
        poly_float = uv[layer.faces[face_id]]
        if np.any(poly_float[:, 0] < -width) or np.any(poly_float[:, 0] > 2 * width):
            continue
        if np.any(poly_float[:, 1] < -height) or np.any(poly_float[:, 1] > 2 * height):
            continue
        poly = np.round(poly_float).astype(np.int32)
        if np.unique(poly, axis=0).shape[0] < 3:
            continue
        mean_depth = float(np.mean(depth[layer.faces[face_id]]))
        shade = float(np.clip(0.80 + 0.18 / max(mean_depth, 0.20), 0.78, 1.10))
        color = tuple(int(np.clip(component * shade, 0, 255)) for component in layer.color)
        cv2.fillConvexPoly(color_canvas, poly, color, cv2.LINE_8)
        cv2.fillConvexPoly(label_map, poly, layer_index + 1, cv2.LINE_8)
        if layer_index in observed_layer_indices:
            cv2.fillConvexPoly(observed_color_canvas, poly, color, cv2.LINE_8)
            cv2.fillConvexPoly(observed_owner_map, poly, 255, cv2.LINE_8)
        rasterized[layer_index] += 1

    # The generated and observed layers describe one object, not independent
    # occluders.  Prediction-side metric observations own every pixel they cover;
    # this prevents an aligned hidden prior from punching speckled holes through
    # measured support.  Full MANO triangles that won the shared depth painter are
    # preserved so hand/object depth order remains visible.
    if observed_layer_indices:
        observed_index = next(iter(observed_layer_indices))
        mano_labels = np.asarray(
            [index + 1 for index, layer in enumerate(layers) if layer.role.startswith("mano_")],
            dtype=np.int16,
        )
        mano_visible = np.isin(label_map, mano_labels) if len(mano_labels) else np.zeros_like(observed_owner_map, dtype=bool)
        ownership_override = (observed_owner_map > 0) & ~mano_visible
        label_map[ownership_override] = observed_index + 1
        color_canvas[ownership_override] = observed_color_canvas[ownership_override]

    output = background.copy()
    for layer_index, layer in enumerate(layers):
        visible = label_map == layer_index + 1
        count = int(np.count_nonzero(visible))
        per_layer[layer_index]["rasterized_faces"] = int(rasterized[layer_index])
        per_layer[layer_index]["visible_pixels"] = count
        if count:
            mixed = cv2.addWeighted(color_canvas, float(layer.alpha), background, 1.0 - float(layer.alpha), 0.0)
            output[visible] = mixed[visible]
            contours, _ = cv2.findContours(visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge = tuple(int(component * 0.48) for component in layer.color)
            cv2.drawContours(output, contours, -1, edge, 1, cv2.LINE_AA)
    return output, {
        "layers": per_layer,
        "visible_pixels_total": int(np.count_nonzero(label_map)),
        "depth_order_method": "shared_far_to_near_mean_triangle_depth_painter_with_observed_object_ownership_override_and_mano_depth_winners_preserved",
    }, label_map


def object_scene_layers(
    object_layers: list[ObjectLayer],
    rotation: np.ndarray,
    translation: np.ndarray,
    generated_budget: int,
    observed_budget: int,
) -> list[SceneLayer]:
    rows: list[SceneLayer] = []
    for layer in object_layers:
        vertices_world = layer.vertices @ rotation.T + translation[None, :]
        budget = generated_budget if layer.role == "generated_complete_prior_underlay" else observed_budget
        rows.append(
            SceneLayer(
                name=layer.name,
                role=layer.role,
                vertices_world=vertices_world,
                faces=layer.faces,
                color=layer.color,
                alpha=layer.alpha,
                face_budget=int(budget),
                depth_priority_bias_m=float(layer.depth_priority_bias_m),
            )
        )
    return rows


def add_mano_layers(
    scene: list[SceneLayer],
    frame_idx: int,
    frame: dict[str, Any],
    mano_cache: ManoArchiveCache,
    mano_budget: int,
) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        vertices, faces, report = mano_cache.hand_mesh(frame_idx, hand)
        side = str(report["side"])
        role = f"mano_{side}_full_surface"
        scene.append(
            SceneLayer(
                name=role,
                role=role,
                vertices_world=vertices,
                faces=faces,
                color=ROLE_COLORS[role],
                alpha=ROLE_ALPHA[role],
                face_budget=int(mano_budget),
                depth_priority_bias_m=0.0,
            )
        )
        provenance.append(report)
    return provenance


def shared_frame_bounds(
    reference_vertices: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scene_layers: list[SceneLayer],
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = [reference_vertices[:: max(1, len(reference_vertices) // 3000)] @ rotation.T + translation[None, :]]
    chunks.extend(layer.vertices_world[:: max(1, len(layer.vertices_world) // 1200)] for layer in scene_layers if layer.role.startswith("mano_"))
    points = np.vstack(chunks)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(high - low, 0.18)
    low = center - 0.5 * extent - float(padding)
    high = center + 0.5 * extent + float(padding)
    return low, high


def points_to_uv(
    points: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    size: tuple[int, int],
    axes: tuple[int, int],
) -> np.ndarray:
    width, height = size
    extent = np.maximum(high - low, 1.0e-6)
    uv = np.empty((len(points), 2), dtype=np.float64)
    uv[:, 0] = (points[:, axes[0]] - low[axes[0]]) / extent[axes[0]] * width
    uv[:, 1] = height - (points[:, axes[1]] - low[axes[1]]) / extent[axes[1]] * height
    return uv


def draw_camera_state(
    image: np.ndarray,
    T_world_camera: np.ndarray,
    camera_path: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    axes: tuple[int, int],
    frustum_depth: float,
) -> None:
    width, height = image.shape[1], image.shape[0]
    if camera_path.ndim == 2 and camera_path.shape[1] == 3 and len(camera_path) > 1:
        uv_path = points_to_uv(camera_path, low, high, (width, height), axes)
        finite = np.isfinite(uv_path).all(axis=1)
        for a, b in zip(np.flatnonzero(finite)[:-1], np.flatnonzero(finite)[1:]):
            pa, pb = uv_path[a], uv_path[b]
            if np.all(pa > -2 * max(width, height)) and np.all(pa < 3 * max(width, height)) and np.all(pb > -2 * max(width, height)) and np.all(pb < 3 * max(width, height)):
                cv2.line(image, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), (130, 130, 130), 1, cv2.LINE_AA)
    depth = float(frustum_depth)
    camera_points = np.asarray(
        [[0, 0, 0], [-0.55 * depth, -0.40 * depth, depth], [0.55 * depth, -0.40 * depth, depth],
         [0.55 * depth, 0.40 * depth, depth], [-0.55 * depth, 0.40 * depth, depth]],
        dtype=np.float64,
    )
    world = camera_points @ T_world_camera[:3, :3].T + T_world_camera[:3, 3][None, :]
    uv = points_to_uv(world, low, high, (width, height), axes)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    for a, b in edge_pairs:
        cv2.line(image, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (235, 235, 235), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(np.round(uv[0]).astype(int)), 5, (255, 255, 255), -1, cv2.LINE_AA)


def title_panel(image: np.ndarray, title: str, subtitle: str, frame_idx: int) -> None:
    canonical.put_text_with_bg(image, title[:115], (14, 28), font_scale=0.52, color=(255, 255, 255), thickness=1, bg_alpha=0.68)
    canonical.put_text_with_bg(image, f"frame {frame_idx:04d} | {subtitle}"[:150], (14, 53), font_scale=0.40, color=(225, 225, 225), thickness=1, bg_alpha=0.64)


def legend_panel(image: np.ndarray) -> None:
    labels = [
        ("magenta: generated complete render prior (no collision)", ROLE_COLORS["generated_complete_prior_underlay"]),
        ("green: observed metric surface / only collision-eligible layer", ROLE_COLORS["observed_metric_surface_overlay"]),
        ("blue + orange: full 778-vertex source metric MANO surfaces", (245, 205, 85)),
        ("triangle depth order shown; signed contact/nonpenetration disabled", (220, 220, 220)),
    ]
    y = image.shape[0] - 82
    for text, color in labels:
        canonical.put_text_with_bg(image, text, (14, y), font_scale=0.35, color=color, thickness=1, bg_alpha=0.58)
        y += 20


def resize_height(image: np.ndarray, height: int) -> np.ndarray:
    width = int(round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def export_world_glb(path: Path, scene_layers: list[SceneLayer]) -> dict[str, Any]:
    scene = trimesh.Scene()
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(scene_layers):
        mesh = trimesh.Trimesh(
            vertices=np.asarray(layer.vertices_world, dtype=np.float64),
            faces=np.asarray(layer.faces, dtype=np.int64),
            process=False,
            validate=False,
        )
        rgba = np.asarray([layer.color[2], layer.color[1], layer.color[0], 210], dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba[None, :], (len(mesh.faces), 1))
        node_name = f"{index:02d}_{layer.role}"
        scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)
        rows.append({
            "node": node_name,
            "role": layer.role,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        })
    scene.export(str(path))
    return {"path": str(path), "sha256": sha256_file(path), "layers": rows}


def render(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rewrites = canonical.parse_rewrites(list(args.path_rewrite or []))
    state_path = canonical.rewrite_path(args.render_state, rewrites)
    output_dir_raw = canonical.rewrite_path(args.output_dir, rewrites)
    if state_path is None or output_dir_raw is None:
        raise RuntimeError("render state/output path resolved to None")
    state_path = require_file(state_path, "experimental layered render state")
    output_dir = prepare_output(output_dir_raw, bool(args.replace))
    state = canonical.load_json(state_path)
    if not isinstance(state, dict) or not str(state.get("status") or "").startswith("ok_experimental"):
        raise RuntimeError(f"not an experimental layered render state: {state_path}")
    adapter = state.get("experimental_p14_p15_adapter") if isinstance(state.get("experimental_p14_p15_adapter"), dict) else {}
    if adapter.get("generated_faces_collision_eligible") is not False or adapter.get("signed_geometry_ready") is not False:
        raise RuntimeError("render state does not preserve generated-geometry physical quarantine")

    annotation_path = canonical.rewrite_path((state.get("annotation_backbone") or {}).get("path"), rewrites)
    if annotation_path is None:
        raise RuntimeError("render state lacks annotation backbone")
    annotation_path = require_file(annotation_path, "annotation backbone")
    annotations = canonical.load_json(annotation_path)
    frames = annotations.get("frames") if isinstance(annotations, dict) else None
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"annotations have no frames: {annotation_path}")
    if args.frames:
        requested = [int(value) for value in args.frames]
        if len(set(requested)) != len(requested):
            raise RuntimeError("--frames contains duplicates")
        by_idx = {canonical.frame_id(frame, pos): frame for pos, frame in enumerate(frames) if isinstance(frame, dict)}
        missing = [idx for idx in requested if idx not in by_idx]
        if missing:
            raise RuntimeError(f"requested frames absent from annotations: {missing}")
        frames = [by_idx[idx] for idx in requested]
    if args.max_frames is not None:
        frames = frames[: int(args.max_frames)]
    if not frames:
        raise RuntimeError("no frames selected")

    poses = canonical.pose_map(state)
    selected_ids = [canonical.frame_id(frame, pos) for pos, frame in enumerate(frames)]
    missing_poses = [idx for idx in selected_ids if idx not in poses]
    if missing_poses:
        raise RuntimeError(f"selected frames lack object poses: {missing_poses[:20]}")
    object_layers, object_summaries = load_object_layers(state, rewrites, float(args.observed_depth_priority_bias_m))
    physical_path = canonical.rewrite_path(
        ((state.get("object_geometry") or {}).get("physical_surface") or {}).get("mesh"), rewrites
    )
    if physical_path is None:
        raise RuntimeError("state lacks shared physical observation mesh")
    physical_path = require_file(physical_path, "shared observed-only physical surface")
    reference_vertices, _reference_faces, reference_summary = canonical.load_mesh(physical_path)

    overlay_dir = output_dir / "overlay_frames"
    world_dir = output_dir / "world_frames"
    side_dir = output_dir / "side_world_frames"
    triptych_dir = output_dir / "side_by_side_frames"
    for directory in (overlay_dir, world_dir, side_dir, triptych_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_annotation_frames = annotations.get("frames", [])
    camera_path = np.asarray(
        [
            ((frame.get("camera") or {}).get("position_world_m") or np.asarray((frame.get("camera") or {}).get("T_world_camera_metric") or np.eye(4))[:3, 3].tolist())
            for frame in all_annotation_frames
            if isinstance(frame, dict)
        ],
        dtype=np.float64,
    )
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else {}
    fps = float(args.fps if args.fps is not None else raw_video.get("fps") or 30.0)
    branch_id = str(adapter.get("branch_id") or state.get("object_label") or "experimental_branch")
    source_model = str(adapter.get("source_model") or "unknown geometry source")
    integration = str(adapter.get("integration") or "unknown integration")

    mano_cache = ManoArchiveCache(rewrites)
    frame_rows: list[dict[str, Any]] = []
    glb_report: dict[str, Any] | None = None
    try:
        for output_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise RuntimeError(f"annotation row {output_index} is not an object")
            frame_idx = canonical.frame_id(frame, output_index)
            rotation, translation, pose_status = poses[frame_idx]
            raw_path = canonical.rewrite_path(frame.get("raw_frame_path"), rewrites)
            if raw_path is None:
                raise RuntimeError(f"frame {frame_idx} lacks raw RGB path")
            rgb = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"failed to read frame {frame_idx}: {raw_path}")
            height, width = rgb.shape[:2]
            intrinsics, intrinsics_report = canonical.scaled_intrinsics_for_frame(frame, width, height, raw_video)
            camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
            T_world_camera = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or np.eye(4), dtype=np.float64)
            if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
                raise RuntimeError(f"frame {frame_idx}: invalid camera transform")

            scene_layers = object_scene_layers(
                object_layers,
                rotation,
                translation,
                int(args.generated_face_budget),
                int(args.observed_face_budget),
            )
            mano_provenance = add_mano_layers(scene_layers, frame_idx, frame, mano_cache, int(args.mano_face_budget))

            camera_projected: list[tuple[np.ndarray, np.ndarray]] = []
            for layer in scene_layers:
                camera_vertices = canonical.world_points_to_camera(layer.vertices_world, T_world_camera)
                u, v, depth, _valid = canonical.project_camera_points(camera_vertices, intrinsics, width, height)
                camera_projected.append((np.c_[u, v], depth))
            overlay, overlay_stats, overlay_labels = rasterize_scene(rgb, scene_layers, camera_projected)
            title_panel(overlay, f"{branch_id} | camera overlay", f"{source_model} | {integration}", frame_idx)
            legend_panel(overlay)

            low, high = shared_frame_bounds(
                reference_vertices,
                rotation,
                translation,
                scene_layers,
                float(args.local_world_padding_m),
            )
            world_background = np.full((720, 1280, 3), 16, dtype=np.uint8)
            world_projected = [
                project_world_axes(layer.vertices_world, low, high, 1280, 720, 0, 2, 1)
                for layer in scene_layers
            ]
            world, world_stats, world_labels = rasterize_scene(world_background, scene_layers, world_projected)
            draw_camera_state(
                world, T_world_camera, camera_path, low, high, (0, 2), float(args.camera_frustum_depth_m)
            )
            title_panel(world, f"{branch_id} | local metric world X-Z", "shared observed/MANO framing + camera frustum", frame_idx)
            legend_panel(world)

            side_background = np.full((720, 1280, 3), 16, dtype=np.uint8)
            side_projected = [
                project_world_axes(layer.vertices_world, low, high, 1280, 720, 1, 2, 0)
                for layer in scene_layers
            ]
            side, side_stats, side_labels = rasterize_scene(side_background, scene_layers, side_projected)
            draw_camera_state(
                side, T_world_camera, camera_path, low, high, (1, 2), float(args.camera_frustum_depth_m)
            )
            title_panel(side, f"{branch_id} | local metric side Y-Z", "same pose/camera/MANO; alternate world axis", frame_idx)
            legend_panel(side)

            triptych = np.hstack([resize_height(overlay, 720), world, side])
            overlay_path = overlay_dir / f"{output_index:06d}.jpg"
            world_path = world_dir / f"{output_index:06d}.jpg"
            side_path = side_dir / f"{output_index:06d}.jpg"
            triptych_path = triptych_dir / f"{output_index:06d}.jpg"
            for path, image in ((overlay_path, overlay), (world_path, world), (side_path, side), (triptych_path, triptych)):
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"failed to write render frame: {path}")
            if len(frames) == 1 or frame_idx == int(args.export_glb_frame):
                review_path = output_dir / f"frame_{frame_idx:06d}_overlay_world_side.png"
                if not cv2.imwrite(str(review_path), triptych):
                    raise RuntimeError(f"failed to write frame review: {review_path}")
            if frame_idx == int(args.export_glb_frame):
                glb_report = export_world_glb(output_dir / f"frame_{frame_idx:06d}_object_full_mano_world.glb", scene_layers)

            role_pixels: dict[str, int] = {}
            for stats in overlay_stats["layers"]:
                role_pixels[str(stats["role"])] = role_pixels.get(str(stats["role"]), 0) + int(stats["visible_pixels"])
            frame_rows.append(
                {
                    "output_frame_index": output_index,
                    "source_frame_idx": frame_idx,
                    "pose_status": pose_status,
                    "intrinsics": intrinsics_report,
                    "mano": mano_provenance,
                    "overlay": overlay_stats,
                    "world": world_stats,
                    "side_world": side_stats,
                    "overlay_visible_pixels_by_role": role_pixels,
                    "overlay_label_nonzero_pixels": int(np.count_nonzero(overlay_labels)),
                    "world_label_nonzero_pixels": int(np.count_nonzero(world_labels)),
                    "side_label_nonzero_pixels": int(np.count_nonzero(side_labels)),
                    "shared_world_bounds_m": [low.tolist(), high.tolist()],
                }
            )
            if output_index % 25 == 0 or output_index + 1 == len(frames):
                print(f"{branch_id}: rendered {output_index + 1}/{len(frames)} source_frame={frame_idx}", flush=True)
    finally:
        mano_summary = mano_cache.summary()
        mano_cache.close()

    outputs: dict[str, Any] = {
        "overlay_frames": str(overlay_dir),
        "world_frames": str(world_dir),
        "side_world_frames": str(side_dir),
        "side_by_side_frames": str(triptych_dir),
        "frame_review": str(output_dir / f"frame_{selected_ids[0]:06d}_overlay_world_side.png") if len(frames) == 1 else (
            str(output_dir / f"frame_{int(args.export_glb_frame):06d}_overlay_world_side.png") if int(args.export_glb_frame) in selected_ids else None
        ),
        "world_scene_glb": glb_report,
    }
    if args.encode_video:
        safe = canonical.re.sub(r"[^A-Za-z0-9_.-]+", "_", branch_id).strip("_") or "branch"
        video_specs = [
            (overlay_dir, output_dir / f"p15_overlay_{safe}.mp4", "overlay_video"),
            (world_dir, output_dir / f"p15_world_{safe}.mp4", "world_video"),
            (side_dir, output_dir / f"p15_side_world_{safe}.mp4", "side_world_video"),
            (triptych_dir, output_dir / f"p15_side_by_side_{safe}.mp4", "side_by_side_video"),
        ]
        for frame_dir, video_path, key in video_specs:
            canonical.encode_video(frame_dir, video_path, fps, frame_count=len(frames))
            outputs[key] = str(video_path)

    pixel_roles: dict[str, list[int]] = {}
    for row in frame_rows:
        for role, count in row["overlay_visible_pixels_by_role"].items():
            pixel_roles.setdefault(role, []).append(int(count))
    role_summary = {
        role: {
            "count": len(values),
            "median": float(np.median(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }
        for role, values in pixel_roles.items()
    }
    manifest = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "render_experimental_p14_p15_layered_state_with_full_mano",
        "claim_scope": (
            "Full-timeline visual review of source-neutral render geometry under one frozen observed-only pose, "
            "camera state, and full source metric MANO trajectory. Mean-triangle-depth painter ordering is visible "
            "evidence only; generated geometry is not a contact, collision, signed-distance, or physical surface."
        ),
        "branch": {
            "branch_id": branch_id,
            "source_model": source_model,
            "integration": integration,
        },
        "inputs": {
            "render_state": str(state_path),
            "render_state_sha256": sha256_file(state_path),
            "annotations": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "shared_observed_physical_surface": str(physical_path),
            "shared_observed_physical_surface_sha256": sha256_file(physical_path),
            "canonical_renderer_helper": str(Path(canonical.__file__).resolve()),
            "canonical_renderer_helper_sha256": sha256_file(Path(canonical.__file__).resolve()),
        },
        "render_contract": {
            "object_layers": object_summaries,
            "generated_face_budget": int(args.generated_face_budget),
            "observed_face_budget": int(args.observed_face_budget),
            "mano_face_budget": int(args.mano_face_budget),
            "observed_depth_priority_bias_m": float(args.observed_depth_priority_bias_m),
            "depth_order_method": (
                "shared_far_to_near_mean_triangle_depth_painter; observed metric pixels own object support; "
                "full MANO depth winners are preserved; not a metric z-buffer"
            ),
            "observed_object_ownership_override": True,
            "world_framing": "branch-invariant observed-only physical surface plus full MANO bounds",
            "world_views": ["X-Z with Y painter depth", "Y-Z side with X painter depth"],
            "source_to_render_intrinsics": "canonical V19 explicit source-size scaling",
        },
        "shared_state_consumption": {
            "object_pose": "render_state.object_pose_trajectory.pose_rows_observed_only",
            "camera": "annotation_backbone.frame.camera.T_world_camera_metric",
            "mano": "annotation metric_mano_state.vertices_reference full bridge 778 vertices + HaWoR face arrays",
            "inherited_mano_constraint_payload_rendered": False,
            "inherited_temporal_contact_hypothesis_rendered": False,
            "generated_faces_contact_eligible": False,
            "generated_faces_collision_eligible": False,
            "signed_geometry_ready": False,
        },
        "reference_surface": reference_summary,
        "mano_full_surface": mano_summary,
        "frame_count": len(frames),
        "source_frame_ids": selected_ids,
        "visible_pixel_summary_by_role": role_summary,
        "outputs": outputs,
        "visual_inspection_required": True,
        "frame_rows": frame_rows,
        "total_elapsed_s": time.time() - started,
    }
    manifest_path = output_dir / "p14_p15_layered_full_mano_render_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "branch": manifest["branch"],
        "frame_count": manifest["frame_count"],
        "mano_full_surface": manifest["mano_full_surface"],
        "visible_pixel_summary_by_role": role_summary,
        "outputs": outputs,
        "manifest": str(manifest_path),
        "elapsed_s": manifest["total_elapsed_s"],
    }, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    render(parse_args())
