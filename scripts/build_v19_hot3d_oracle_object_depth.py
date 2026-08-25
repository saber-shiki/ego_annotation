#!/usr/bin/env python3
"""Render evaluator-only HOT3D target first-surface depth for a VRS slice.

The renderer consumes the evaluation sidecar produced by
``build_v19_hot3d_vrs_slice_adapter.py``.  Released HOT3D camera/object poses are
used only inside this renderer.  The output depth raster contains the currently
visible first surface of one target CAD instance; unknown/background pixels and
modeled occluders are zero/invalid.

All released dynamic-object CADs and GT MANO hands are included as z-buffer
occluders by default.  The target-only amodal pass is diagnostic and never fills
occluded pixels in the oracle archive.  This is a conditional downstream/oracle
input, not an RGB-only prediction and not evidence of contact/nonpenetration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Must be selected before importing pyrender / PyOpenGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
import trimesh


OPENGL_FROM_ARIA_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
TARGET_SEG_COLOR = np.asarray([255, 0, 0], dtype=np.uint8)
OBJECT_OCCLUDER_COLOR = np.asarray([0, 255, 0], dtype=np.uint8)
HAND_OCCLUDER_COLOR = np.asarray([0, 0, 255], dtype=np.uint8)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summarize(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0 or not math.isfinite(n):
        raise RuntimeError(f"invalid quaternion {q}")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_from_dict(payload: dict[str, Any]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_R(payload["quaternion_wxyz"])
    T[:3, 3] = np.asarray(payload["translation_xyz"], dtype=np.float64)
    if not np.isfinite(T).all():
        raise RuntimeError("non-finite transform")
    return T


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(str(path), process=True, merge_primitives=True, file_type="glb")
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_mesh"):
            mesh = loaded.to_mesh()
        else:  # compatibility with older trimesh
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported mesh payload from {path}: {type(loaded)}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"empty CAD mesh: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise RuntimeError(f"non-finite CAD mesh: {path}")
    return mesh


def object_entries(frame: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    objects = (frame.get("json") or {}).get("objects.json") or {}
    for entries in objects.values():
        if isinstance(entries, list):
            out.extend(e for e in entries if isinstance(e, dict))
    return out


def target_entry(frame: dict[str, Any], uid: str) -> dict[str, Any]:
    matches = [e for e in object_entries(frame) if str(e.get("object_uid")) == uid]
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has {len(matches)} target entries for UID {uid}")
    return matches[0]


def frame_camera(frame: dict[str, Any], stream_id: str) -> dict[str, Any]:
    camera = ((frame.get("json") or {}).get("cameras.json") or {}).get(stream_id)
    if not isinstance(camera, dict):
        raise RuntimeError(f"frame {frame.get('frame_idx')} missing camera {stream_id}")
    return camera


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def load_mano_layers(mano_model_dir: Path, batch_sizes: dict[str, int]):
    # smplx's MANO loader still imports legacy chumpy code on this Python 3.11
    # runtime.  Apply the same compatibility shim used by the project's MANO
    # scripts before importing smplx; this changes no model parameters.
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    import smplx

    layers: dict[str, Any] = {}
    for side, is_right in (("left", False), ("right", True)):
        count = max(1, int(batch_sizes.get(side, 0)))
        filename = mano_model_dir / ("MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl")
        if not filename.is_file():
            raise FileNotFoundError(filename)
        layer = smplx.create(
            str(filename),
            "mano",
            use_pca=True,
            is_rhand=is_right,
            num_pca_comps=15,
            batch_size=count,
        )
        layer.to("cpu")
        layer.eval()
        layers[side] = layer
    # Match the HOT3D toolkit's correction for a known mirrored-left shapedirs file.
    if torch.sum(torch.abs(layers["left"].shapedirs[:, 0, :] - layers["right"].shapedirs[:, 0, :])) < 1:
        layers["left"].shapedirs[:, 0, :] *= -1
    return layers


def rotation_vector_from_wxyz(q: Iterable[float]) -> np.ndarray:
    # scipy's implementation is robust near pi, unlike a compact hand-written log map.
    from scipy.spatial.transform import Rotation

    w, x, y, z = [float(v) for v in q]
    return Rotation.from_quat([x, y, z, w]).as_rotvec().astype(np.float32)


def precompute_mano_world_meshes(
    frames: list[dict[str, Any]], mano_model_dir: Path
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    records: dict[str, list[tuple[int, dict[str, Any]]]] = {"left": [], "right": []}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        hands = ((frame.get("json") or {}).get("hands.json") or {})
        for side in ("left", "right"):
            mano = (hands.get(side) or {}).get("mano_pose")
            if isinstance(mano, dict) and len(mano.get("thetas") or []) == 15:
                records[side].append((frame_idx, mano))
    layers = load_mano_layers(mano_model_dir, {side: len(rows) for side, rows in records.items()})
    by_frame: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    faces: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    with torch.no_grad():
        for side in ("left", "right"):
            layer = layers[side]
            faces[side] = np.asarray(layer.faces, dtype=np.int32)
            rows = records[side]
            if not rows:
                summary[side] = {"frame_count": 0, "face_count": int(len(faces[side]))}
                continue
            betas: list[list[float]] = []
            hand_pose: list[list[float]] = []
            global_orient: list[np.ndarray] = []
            transl: list[list[float]] = []
            for _, mano in rows:
                beta = [float(v) for v in mano.get("betas") or []]
                if len(beta) != 10:
                    raise RuntimeError(f"{side} MANO row has {len(beta)} betas, expected 10")
                hand_pose.append([float(v) for v in mano["thetas"]])
                betas.append(beta)
                wrist = mano.get("T_world_from_wrist")
                if not isinstance(wrist, dict):
                    raise RuntimeError(f"{side} MANO row lacks T_world_from_wrist")
                global_orient.append(rotation_vector_from_wxyz(wrist["quaternion_wxyz"]))
                transl.append([float(v) for v in wrist["translation_xyz"]])
            result = layer(
                betas=torch.as_tensor(np.asarray(betas), dtype=torch.float32),
                global_orient=torch.as_tensor(np.asarray(global_orient), dtype=torch.float32),
                hand_pose=torch.as_tensor(np.asarray(hand_pose), dtype=torch.float32),
                transl=torch.as_tensor(np.asarray(transl), dtype=torch.float32),
                return_verts=True,
            )
            vertices = result.vertices.detach().cpu().numpy().astype(np.float32)
            if vertices.shape != (len(rows), 778, 3):
                raise RuntimeError(f"unexpected {side} MANO vertex shape {vertices.shape}")
            for (frame_idx, _), verts in zip(rows, vertices):
                by_frame[frame_idx][side] = verts
            summary[side] = {
                "frame_count": len(rows),
                "vertex_count_per_frame": 778,
                "face_count": int(len(faces[side])),
            }
    return dict(by_frame), faces, summary


def depth_colormap(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    vals = depth[mask]
    if vals.size == 0:
        return out
    lo, hi = np.quantile(vals, [0.02, 0.98])
    if hi <= lo:
        hi = lo + 1e-3
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[mask] = colored[mask]
    return out


def make_review_tile(
    rgb_path: Path,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    depth: np.ndarray,
    label: str,
) -> np.ndarray:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if rgb.shape[:2] != visible_mask.shape:
        raise RuntimeError(f"review RGB/mask shape mismatch: {rgb.shape[:2]} vs {visible_mask.shape}")
    overlay = rgb.copy()
    overlay[visible_mask] = (
        0.35 * overlay[visible_mask].astype(np.float32) + 0.65 * np.asarray([60, 230, 60], dtype=np.float32)
    ).astype(np.uint8)
    occluded = amodal_mask & ~visible_mask
    overlay[occluded] = (
        0.45 * overlay[occluded].astype(np.float32) + 0.55 * np.asarray([220, 40, 220], dtype=np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(amodal_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 180, 255), 3, cv2.LINE_AA)
    depth_vis = depth_colormap(depth, visible_mask)
    panel = np.hstack([overlay, depth_vis])
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 56), (8, 8, 8), -1)
    cv2.putText(panel, label, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
    target_w = 1200
    return cv2.resize(
        panel,
        (target_w, int(round(panel.shape[0] * target_w / panel.shape[1]))),
        interpolation=cv2.INTER_AREA,
    )


def write_review(tiles: list[np.ndarray], output: Path) -> None:
    if not tiles:
        return
    width = max(t.shape[1] for t in tiles)
    norm: list[np.ndarray] = []
    for tile in tiles:
        if tile.shape[1] != width:
            tile = cv2.resize(tile, (width, int(round(tile.shape[0] * width / tile.shape[1]))), interpolation=cv2.INTER_AREA)
        norm.append(tile)
    sheet = np.vstack(norm)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"failed to write review {output}")


def render(args: argparse.Namespace) -> dict[str, Any]:
    if args.egl_device_id is not None:
        os.environ["EGL_DEVICE_ID"] = str(args.egl_device_id)
    # Import only after EGL environment selection.
    import pyrender

    adapter_root = args.adapter_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        raise FileExistsError(f"non-empty output directory; use --replace explicitly: {output_dir}")
    if output_dir.exists() and args.replace:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = adapter_root / "evaluation" / "hot3d_gt" / "hot3d_vrs_gt_sidecar.json"
    manifest_path = adapter_root / "input" / "raw_frame_manifest" / "manifest.json"
    contract_path = adapter_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    sidecar = load_json(sidecar_path)
    manifest = load_json(manifest_path)
    contract = load_json(contract_path)
    all_frames = sidecar.get("frames") or []
    manifest_frames = manifest.get("frames") or []
    if len(all_frames) != len(manifest_frames):
        raise RuntimeError("GT sidecar and raw-frame manifest frame counts differ")
    manifest_by_idx = {int(row["frame_idx"]): row for row in manifest_frames}
    if args.frame_indices:
        selected_indices = [int(v) for v in args.frame_indices]
    else:
        lo = int(args.frame_start)
        hi = int(args.frame_end) if args.frame_end is not None else max(int(f["frame_idx"]) for f in all_frames)
        selected_indices = list(range(lo, hi + 1))
    if selected_indices != sorted(selected_indices) or len(set(selected_indices)) != len(selected_indices):
        raise RuntimeError("frame indices must be unique and ascending")
    frame_by_idx = {int(f["frame_idx"]): f for f in all_frames}
    missing = [i for i in selected_indices if i not in frame_by_idx or i not in manifest_by_idx]
    if missing:
        raise RuntimeError(f"selected frames missing from adapter: {missing[:20]}")
    frames = [frame_by_idx[i] for i in selected_indices]

    stream_id = str(args.stream_id)
    K = np.asarray(contract["K"], dtype=np.float64)
    width = int(contract["image_width"])
    height = int(contract["image_height"])
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError("invalid camera K")
    if not np.allclose(K[0, 1:], [0.0, K[0, 2]]) or K[1, 0] != 0.0:
        # This message is intentionally strict for the current no-skew contract.
        if abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
            raise RuntimeError("renderer requires zero-skew pinhole K")

    target_uid = str(args.target_object_uid)
    first_target = target_entry(frames[0], target_uid)
    target_bop_id = str(first_target.get("object_bop_id"))
    target_name = str(first_target.get("object_name"))

    all_uids = sorted({str(e["object_uid"]) for f in frames for e in object_entries(f)})
    if target_uid not in all_uids:
        raise RuntimeError(f"target UID {target_uid} absent from selected frames")
    assets_root = args.assets_root.resolve()
    trimeshes: dict[str, trimesh.Trimesh] = {}
    render_meshes: dict[str, Any] = {}
    mesh_reports: dict[str, Any] = {}
    for uid in all_uids:
        path = assets_root / f"{uid}.glb"
        mesh = load_trimesh(path)
        trimeshes[uid] = mesh
        render_meshes[uid] = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        mesh_reports[uid] = {
            "path": str(path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "bounds_m": np.asarray(mesh.bounds, dtype=float).tolist(),
        }

    if args.include_hands:
        mano_by_frame, mano_faces, mano_summary = precompute_mano_world_meshes(frames, args.mano_model_dir.resolve())
    else:
        mano_by_frame, mano_faces, mano_summary = {}, {}, {"status": "disabled"}

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    review_tiles: list[np.ndarray] = []
    review_set = set(int(v) for v in args.review_frames)
    all_depth_values: list[np.ndarray] = []
    try:
        for sequence_pos, frame in enumerate(frames):
            frame_idx = int(frame["frame_idx"])
            camera_payload = frame_camera(frame, stream_id)
            T_world_camera = transform_from_dict(camera_payload["T_world_from_camera"])
            scene = pyrender.Scene(bg_color=np.asarray([0, 0, 0, 0], dtype=np.uint8), ambient_light=np.asarray([0.1, 0.1, 0.1]))
            seg_node_map: dict[Any, np.ndarray] = {}
            target_node = None
            object_node_count = 0
            for entry in object_entries(frame):
                uid = str(entry["object_uid"])
                if uid != target_uid and not args.include_other_objects:
                    continue
                node = scene.add(render_meshes[uid], pose=transform_from_dict(entry["T_world_from_object"]))
                object_node_count += 1
                if uid == target_uid:
                    target_node = node
                    seg_node_map[node] = TARGET_SEG_COLOR
                else:
                    seg_node_map[node] = OBJECT_OCCLUDER_COLOR
            if target_node is None:
                raise RuntimeError(f"frame {frame_idx}: target mesh node missing")
            hand_node_count = 0
            if args.include_hands:
                for side, vertices in mano_by_frame.get(frame_idx, {}).items():
                    hand_tm = trimesh.Trimesh(
                        vertices=np.asarray(vertices, dtype=np.float32),
                        faces=mano_faces[side],
                        process=False,
                        validate=False,
                    )
                    hand_mesh = pyrender.Mesh.from_trimesh(hand_tm, smooth=False)
                    hand_node = scene.add(hand_mesh, pose=np.eye(4))
                    seg_node_map[hand_node] = HAND_OCCLUDER_COLOR
                    hand_node_count += 1
            camera = pyrender.IntrinsicsCamera(
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                znear=float(args.znear),
                zfar=float(args.zfar),
            )
            camera_node = scene.add(camera, pose=T_world_camera @ OPENGL_FROM_ARIA_CAMERA)
            seg, full_depth = renderer.render(
                scene,
                flags=pyrender.RenderFlags.SEG,
                seg_node_map=seg_node_map,
            )
            visible_mask = (
                (seg[..., 0] == int(TARGET_SEG_COLOR[0]))
                & (seg[..., 1] == int(TARGET_SEG_COLOR[1]))
                & (seg[..., 2] == int(TARGET_SEG_COLOR[2]))
                & np.isfinite(full_depth)
                & (full_depth > 0.0)
            )
            target_depth = np.zeros((height, width), dtype=np.float32)
            target_depth[visible_mask] = np.asarray(full_depth[visible_mask], dtype=np.float32)

            # Render the target alone for visibility diagnostics only.  Its hidden pixels
            # are never copied into target_depth.
            amodal_scene = pyrender.Scene(bg_color=np.asarray([0, 0, 0, 0], dtype=np.uint8), ambient_light=np.asarray([0.1, 0.1, 0.1]))
            target = target_entry(frame, target_uid)
            amodal_node = amodal_scene.add(render_meshes[target_uid], pose=transform_from_dict(target["T_world_from_object"]))
            amodal_camera = pyrender.IntrinsicsCamera(
                fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), znear=float(args.znear), zfar=float(args.zfar)
            )
            amodal_scene.add(amodal_camera, pose=T_world_camera @ OPENGL_FROM_ARIA_CAMERA)
            amodal_seg, amodal_depth = renderer.render(
                amodal_scene,
                flags=pyrender.RenderFlags.SEG,
                seg_node_map={amodal_node: TARGET_SEG_COLOR},
            )
            amodal_mask = (
                (amodal_seg[..., 0] == int(TARGET_SEG_COLOR[0]))
                & (amodal_seg[..., 1] == int(TARGET_SEG_COLOR[1]))
                & (amodal_seg[..., 2] == int(TARGET_SEG_COLOR[2]))
                & np.isfinite(amodal_depth)
                & (amodal_depth > 0.0)
            )
            # First-surface invariant: every valid oracle pixel is visible in the full
            # z-buffer and belongs to the target in the instance segmentation pass.
            if np.count_nonzero(visible_mask & ~amodal_mask) != 0:
                raise RuntimeError(f"frame {frame_idx}: full-scene target mask is not a subset of target-only raster")
            visible_count = int(np.count_nonzero(visible_mask))
            amodal_count = int(np.count_nonzero(amodal_mask))
            if visible_count <= 0 or amodal_count <= 0:
                raise RuntimeError(f"frame {frame_idx}: empty target raster visible={visible_count} amodal={amodal_count}")
            modeled_visibility = float(visible_count / amodal_count)
            released_visibility = (target.get("visibilities_modeled") or {}).get(stream_id)
            depth_values = target_depth[visible_mask]
            all_depth_values.append(depth_values)
            row = {
                "frame_idx": frame_idx,
                "source_frame_idx": int(frame.get("source_frame_idx", frame_idx)),
                "timestamp_ns": int(frame.get("timestamp_ns") or (frame.get("json") or {}).get("info.json", {}).get("ref_timestamp_ns")),
                "valid_target_surface_pixel_count": visible_count,
                "target_amodal_pixel_count_diagnostic": amodal_count,
                "modeled_first_surface_visibility_ratio": modeled_visibility,
                "released_hot3d_visibility_ratio": float(released_visibility) if released_visibility is not None else None,
                "visibility_ratio_delta": float(modeled_visibility - float(released_visibility)) if released_visibility is not None else None,
                "visible_bbox_xyxy_output_pinhole": bbox_from_mask(visible_mask),
                "amodal_bbox_xyxy_output_pinhole": bbox_from_mask(amodal_mask),
                "depth_m": summarize(depth_values.tolist()),
                "object_node_count": object_node_count,
                "hand_occluder_node_count": hand_node_count,
                "unknown_or_non_target_pixel_count": int(width * height - visible_count),
                "unknown_or_non_target_depth_policy": "zero_invalid",
            }
            rows.append(row)
            depths.append(target_depth)
            masks.append(visible_mask.astype(bool))
            if frame_idx in review_set:
                rgb_path = Path(manifest_by_idx[frame_idx]["raw_frame_path"])
                label = (
                    f"frame {frame_idx} | green=visible target, magenta=modeled occluded | "
                    f"vis={modeled_visibility:.3f}, HOT3D={float(released_visibility):.3f}" if released_visibility is not None else
                    f"frame {frame_idx} | vis={modeled_visibility:.3f}"
                )
                review_tiles.append(make_review_tile(rgb_path, visible_mask, amodal_mask, target_depth, label))
            scene.remove_node(camera_node)
    finally:
        renderer.delete()

    depth_array = np.stack(depths, axis=0).astype(np.float32)
    mask_array = np.stack(masks, axis=0).astype(bool)
    if np.count_nonzero(depth_array[~mask_array]) != 0:
        raise RuntimeError("oracle archive would contain nonzero depth outside target first-surface mask")
    if not np.isfinite(depth_array).all() or float(np.min(depth_array)) < 0.0:
        raise RuntimeError("oracle depth contains invalid values")
    frame_idx_array = np.asarray(selected_indices, dtype=np.int32)
    source_frame_idx_array = np.asarray([int(f.get("source_frame_idx", f["frame_idx"])) for f in frames], dtype=np.int32)
    timestamp_array = np.asarray([int(f.get("timestamp_ns") or (f.get("json") or {}).get("info.json", {}).get("ref_timestamp_ns")) for f in frames], dtype=np.int64)
    intrinsics = np.repeat(
        np.asarray([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]], dtype=np.float32),
        len(frames),
        axis=0,
    )
    npz_path = output_dir / args.npz_name
    np.savez_compressed(
        npz_path,
        frame_idx=frame_idx_array,
        source_frame_idx=source_frame_idx_array,
        timestamp_ns=timestamp_array,
        depth=depth_array,
        source_size=np.asarray([width, height], dtype=np.int32),
        focal_px=np.repeat(np.float32(K[0, 0]), len(frames)),
        intrinsics_fx_fy_cx_cy=intrinsics,
        depth_source=np.asarray("HOT3D_GT_rendered_target_visible_first_surface_v1"),
        valid_target_surface_mask=mask_array,
        object_bop_id=np.asarray(target_bop_id),
        model_uid=np.asarray(target_uid),
        oracle_input=np.asarray(True),
        unknown_background_depth_value=np.asarray(0.0, dtype=np.float32),
    )
    review_path = output_dir / "hot3d_oracle_object_depth_review.jpg"
    write_review(review_tiles, review_path)

    valid_counts = [int(r["valid_target_surface_pixel_count"]) for r in rows]
    modeled_vis = [float(r["modeled_first_surface_visibility_ratio"]) for r in rows]
    vis_deltas = [float(r["visibility_ratio_delta"]) for r in rows if r["visibility_ratio_delta"] is not None]
    concat_depth = np.concatenate(all_depth_values) if all_depth_values else np.asarray([], dtype=np.float32)
    report = {
        "status": "ok",
        "method": "build_v19_hot3d_oracle_object_depth",
        "claim_scope": "evaluator-only rendered target first-surface depth; conditional/oracle downstream validation, not RGB-only prediction",
        "oracle_input": True,
        "inputs": {
            "adapter_root": str(adapter_root),
            "ground_truth_sidecar": str(sidecar_path),
            "raw_frame_manifest": str(manifest_path),
            "official_calibration_contract": str(contract_path),
            "assets_root": str(assets_root),
            "mano_model_dir": str(args.mano_model_dir.resolve()) if args.include_hands else None,
        },
        "output": {
            "depth_npz": str(npz_path),
            "review": str(review_path) if review_tiles else None,
        },
        "target": {
            "object_uid": target_uid,
            "object_bop_id": target_bop_id,
            "object_name": target_name,
            "cad": mesh_reports[target_uid],
        },
        "camera": {
            "stream_id": stream_id,
            "width": width,
            "height": height,
            "K": K.tolist(),
            "image_orientation": contract.get("image_orientation"),
            "world_camera_source": "HOT3D released headset trajectory and official online T_device_camera, evaluator-only",
        },
        "z_buffer_contract": {
            "valid_depth": "target CAD pixels that win the full modeled scene z-buffer",
            "unknown_background": "zero/invalid; never treated as dense background",
            "target_occluded_pixels": "zero/invalid; target-only amodal depth is diagnostic only",
            "other_object_occluders": bool(args.include_other_objects),
            "gt_mano_hand_occluders": bool(args.include_hands),
            "modeled_dynamic_object_count": len(all_uids),
            "znear_m": float(args.znear),
            "zfar_m": float(args.zfar),
        },
        "frame_count": len(frames),
        "frame_idx_range": [int(selected_indices[0]), int(selected_indices[-1])],
        "source_frame_idx_range": [int(source_frame_idx_array[0]), int(source_frame_idx_array[-1])],
        "valid_target_surface_pixel_count": summarize(valid_counts),
        "modeled_first_surface_visibility_ratio": summarize(modeled_vis),
        "visibility_ratio_delta_vs_released_hot3d": summarize(vis_deltas),
        "valid_target_depth_m": summarize(concat_depth.tolist()),
        "mesh_reports": mesh_reports,
        "mano_occluder_summary": mano_summary,
        "render_environment": {
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "EGL_DEVICE_ID": os.environ.get("EGL_DEVICE_ID"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pyrender_version": getattr(pyrender, "__version__", None),
            "trimesh_version": getattr(trimesh, "__version__", None),
        },
        "scientific_boundaries": [
            "GT object pose is used only to render camera-frame depth and is not a P14/P15 predicted state.",
            "The archive is an oracle intervention and cannot establish RGB-only end-to-end performance.",
            "The released CAD may be non-watertight; visible depth does not establish signed nonpenetration or contact.",
            "Only modeled dynamic objects and GT MANO hands are occluders; unmodeled static-scene occlusion remains a limitation.",
        ],
        "rows": rows,
    }
    report_path = output_dir / "hot3d_oracle_object_depth_report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "claim_scope": report["claim_scope"],
                "depth_npz": str(npz_path),
                "target": report["target"],
                "frame_count": len(frames),
                "valid_target_surface_pixel_count": report["valid_target_surface_pixel_count"],
                "modeled_first_surface_visibility_ratio": report["modeled_first_surface_visibility_ratio"],
                "visibility_ratio_delta_vs_released_hot3d": report["visibility_ratio_delta_vs_released_hot3d"],
                "valid_target_depth_m": report["valid_target_depth_m"],
            },
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--assets-root", type=Path, required=True)
    p.add_argument("--mano-model-dir", type=Path, required=True)
    p.add_argument("--target-object-uid", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stream-id", default="214-1")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--frame-indices", type=int, nargs="*", default=None)
    p.add_argument("--review-frames", type=int, nargs="*", default=[0, 37, 75, 112, 149])
    p.add_argument("--npz-name", default="hot3d_oracle_object_depth_v1.npz")
    p.add_argument("--znear", type=float, default=0.02)
    p.add_argument("--zfar", type=float, default=10.0)
    p.add_argument("--egl-device-id", type=int, default=None)
    p.add_argument("--include-other-objects", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-hands", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--replace", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    try:
        render(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
