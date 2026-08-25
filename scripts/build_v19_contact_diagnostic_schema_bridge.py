#!/usr/bin/env python3
"""Bridge V19 frames[].objects[] / metric_mano_state into the legacy contact-QC schema.

This is an evidence adapter only.  It does not alter object geometry, camera state,
MANO trajectories, or canonical V19 annotations.  The legacy contact diagnostic
receives sampled MANO vertices when the V19 annotation contains only a sample;
that limitation is recorded explicitly in the output provenance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    if points.shape != (21, 3) or np.any(~np.isfinite(points)) or np.any(z <= 0.0):
        raise RuntimeError("cannot project invalid 21x3 camera-space MANO joints")
    fx, fy, cx, cy = intrinsics
    return np.column_stack((fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy))


def image_size(path: Path) -> list[int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return [int(image.width), int(image.height)]
    except Exception as exc:  # pragma: no cover - error path is provenance validation
        raise RuntimeError(f"cannot read mask image size for {path}: {exc}") from exc


def bridge_hand(hand: dict[str, Any], frame_intrinsics: np.ndarray, frame_idx: int) -> dict[str, Any]:
    state = hand.get("metric_mano_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"frame {frame_idx} hand lacks metric_mano_state")
    joints_camera = np.asarray(state.get("joints_current_v18_camera_m", []), dtype=float)
    joints_world = np.asarray(state.get("joints_current_v18_world_m", state.get("joints_world_m", [])), dtype=float)
    vertices_camera = np.asarray(state.get("vertices_camera_sample_m", []), dtype=float)
    vertices_world = np.asarray(state.get("vertices_world_sample_m", []), dtype=float)
    if joints_camera.shape != (21, 3) or joints_world.shape != (21, 3):
        raise RuntimeError(f"frame {frame_idx} hand has invalid metric MANO joints")
    if vertices_camera.ndim != 2 or vertices_camera.shape[1:] != (3,) or len(vertices_camera) < 8:
        raise RuntimeError(f"frame {frame_idx} hand has fewer than 8 sampled camera-space vertices")
    if vertices_world.shape != vertices_camera.shape:
        raise RuntimeError(f"frame {frame_idx} hand camera/world vertex samples disagree")
    keypoints = project(joints_camera, frame_intrinsics)
    side = str(hand.get("hand_side", "unknown"))
    confidence = hand.get("confidence")
    score = float(confidence) if isinstance(confidence, (int, float)) else float("nan")
    measured = bool(hand.get("same_frame_detection", hand.get("hawor_same_frame_detection", False)))
    return {
        "side": side,
        "hand_side": side,
        "track_id": f"{side}_v19_metric_mano",
        "track_source": state.get("source", hand.get("hand_geometry_source")),
        "filter_status": state.get("support_state", hand.get("visibility_state")),
        "measurement_available": measured,
        "detector_score": score,
        "joints3d_source_camera_m": joints_camera.tolist(),
        "joints3d_camera": joints_camera.tolist(),
        "joints3d_world_m": joints_world.tolist(),
        "joints2d_raw": keypoints.tolist(),
        "joints2d": keypoints.tolist(),
        "vertices_source_camera_m_sample": vertices_camera.tolist(),
        "vertices_camera_sample": vertices_camera.tolist(),
        "vertices_world_m_sample": vertices_world.tolist(),
        "vertices_sample_indices": state.get("vertices_sample_indices", list(range(len(vertices_camera)))),
        "source_intrinsics": frame_intrinsics.tolist(),
        "schema_bridge_note": "2D joints are deterministic projections of V19 metric camera-space joints; MANO vertices may be sampled rather than the full 778-vertex mesh",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = load_json(args.annotations)
    frames = source.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations contain no frames")
    bridged = copy.deepcopy(source)
    sampled_counts: list[int] = []
    for frame in bridged["frames"]:
        frame_idx = int(frame["frame_idx"])
        camera = frame.get("camera", {})
        intrinsics = np.asarray(camera.get("intrinsics_fx_fy_cx_cy", []), dtype=float)
        if intrinsics.shape != (4,) or np.any(~np.isfinite(intrinsics)):
            raise RuntimeError(f"frame {frame_idx} has invalid V19 intrinsics")
        camera["vggt_source_intrinsics_fx_fy_cx_cy"] = intrinsics.tolist()
        objects = frame.get("objects", [])
        matches = [item for item in objects if str(item.get("object_id")) == str(args.object_id)]
        if len(matches) != 1:
            raise RuntimeError(f"frame {frame_idx} has {len(matches)} objects matching {args.object_id}")
        obj = copy.deepcopy(matches[0])
        mask_path = Path(str(obj.get("mask_path", "")))
        if not mask_path.exists():
            raise RuntimeError(f"frame {frame_idx} mask does not exist: {mask_path}")
        obj["source_image_size"] = [int(frame["source_width"]), int(frame["source_height"])]
        obj["mask_image_size"] = image_size(mask_path)
        obj["schema_bridge_source"] = "frames[].objects[]"
        frame["object"] = obj
        frame["hands"] = [bridge_hand(hand, intrinsics, frame_idx) for hand in frame.get("hands", [])]
        sampled_counts.extend(len(hand["vertices_camera_sample"]) for hand in frame["hands"])

    bridge = {
        "schema": "legacy_mesh_surface_contact_v3_bridge_from_v19",
        "status": "diagnostic_only_not_canonical",
        "source_annotations": str(args.annotations),
        "source_sha256": sha256(args.annotations),
        "object_id": str(args.object_id),
        "frame_count": len(bridged["frames"]),
        "hand_row_count": len(sampled_counts),
        "mano_vertex_count_per_row": {
            "min": int(min(sampled_counts)) if sampled_counts else 0,
            "max": int(max(sampled_counts)) if sampled_counts else 0,
        },
        "limitations": [
            "joints2d_raw is projected from metric MANO camera joints, so reprojection residual is not independent evidence",
            "V19 annotations expose sampled MANO vertices here; contact patches can be missed between samples",
            "signed gaps use local oriented object vertex normals and do not prove watertight inside/outside nonpenetration",
        ],
    }
    bridged["contact_diagnostic_schema_bridge"] = bridge
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bridged, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(bridge, indent=2))


if __name__ == "__main__":
    main()
