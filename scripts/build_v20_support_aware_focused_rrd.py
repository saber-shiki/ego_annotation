#!/usr/bin/env python3
"""Focused V20 visualization with an objective-mesh render contract.

When the pose report binds a generated mesh/hash, this renderer verifies the
input mesh and preserves all display faces so a presentation-only depth gate
cannot hide pose/surface errors.  Legacy pose reports without that contract
retain the previous diagnostic depth-order pruning behavior.  Neither mode
changes collision, contact, SDF, sign, or formal annotation authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from scipy.spatial import cKDTree

LEFT_COLOR_BGR = (255, 150, 40)
RIGHT_COLOR_BGR = (80, 150, 255)
GENERATED_COLOR_BGR = (215, 45, 190)
VISIBLE_POINT_COLOR_BGR = (0, 220, 255)
MASK_CONTOUR_BGR = (60, 255, 80)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


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
    value = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if isinstance(value, trimesh.Scene):
        parts = [g for g in value.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"no mesh in {path}")
        value = trimesh.util.concatenate(parts)
    if not isinstance(value, trimesh.Trimesh) or len(value.vertices) == 0 or len(value.faces) == 0:
        raise RuntimeError(f"invalid mesh in {path}")
    return value


def pose_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in load_json(path).get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        if row.get("rotation_world_from_completed_canonical_matrix") is None or row.get("translation_world_m") is None:
            continue
        rows[int(row["frame_idx"])] = row
    return rows


def validate_render_mesh_contract(
    pose_payload: dict[str, Any], generated_mesh: Path
) -> dict[str, Any]:
    contract = pose_payload.get("render_mesh_contract")
    if not isinstance(contract, dict) or contract.get("renderer_must_match") is not True:
        return {
            "required": False,
            "validated": False,
            "reason": "pose report does not require a render mesh binding",
        }
    expected_hash = str(contract.get("required_mesh_sha256") or "")
    expected_path_value = str(contract.get("required_mesh_path") or "")
    if not expected_hash or not expected_path_value:
        raise RuntimeError("pose report render mesh contract is incomplete")
    actual_path = generated_mesh.expanduser().resolve()
    expected_path = Path(expected_path_value).expanduser().resolve()
    if actual_path != expected_path:
        raise RuntimeError(
            "renderer generated mesh path does not match pose objective: "
            f"{actual_path} != {expected_path}"
        )
    actual_hash = sha256_file(actual_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "renderer generated mesh hash does not match pose objective: "
            f"{actual_hash} != {expected_hash}"
        )
    return {
        "required": True,
        "validated": True,
        "expected_path": str(expected_path),
        "actual_path": str(actual_path),
        "path_matches": True,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "display_face_pruning_allowed": False,
    }


def resize_intrinsics_half_pixel(K: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    sx = float(target_wh[0]) / float(source_wh[0])
    sy = float(target_wh[1]) / float(source_wh[1])
    out = np.asarray(K, dtype=np.float64).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = sx * (out[0, 2] + 0.5) - 0.5
    out[1, 2] = sy * (out[1, 2] + 0.5) - 0.5
    return out


def camera_points_from_world(points_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]


def camera_points_to_world(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """Convert camera-frame visible points to world exactly once."""
    transform = np.asarray(T_world_camera, dtype=np.float64)
    return np.asarray(points_camera, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3][None, :]


def project_points(points_camera: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(z) & (z > 0.01)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv[valid, 0] = K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]
        uv[valid, 1] = K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]
        valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if int(target_faces) <= 0 or len(mesh.faces) <= int(target_faces):
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)
    o3 = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    reduced = o3.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    vertices = np.asarray(reduced.vertices, dtype=np.float64)
    faces = np.asarray(reduced.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("mesh decimation returned empty geometry")
    return vertices, faces


def build_support_rays(
    p09_camera: np.ndarray,
    K: np.ndarray,
    mask_path: str | None,
    *,
    radius_px: float,
    stride: int,
    max_rays: int = 12000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Densify P09 support only inside its observed 2-D mask.

    The depth for each added ray is nearest-neighbor copied from an existing
    P09 first-hit point, and the ray is retained only within ``radius_px``.
    This is a display gate, not a depth completion or pose measurement.
    """
    if not mask_path:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), {"active": False, "reason": "missing_mask_path"}
    mask = cv2.imread(str(Path(mask_path).expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), {"active": False, "reason": "mask_unreadable"}
    p09 = np.asarray(p09_camera, dtype=np.float64)
    if len(p09) < 3:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), {"active": False, "reason": "insufficient_p09_points"}
    p09_uv = np.column_stack((
        K[0, 0] * p09[:, 0] / np.maximum(p09[:, 2], 1.0e-9) + K[0, 2],
        K[1, 1] * p09[:, 1] / np.maximum(p09[:, 2], 1.0e-9) + K[1, 2],
    ))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), {"active": False, "reason": "empty_mask"}
    # P09 mask is 960x960 and calibration is 1408x1408. Use the same
    # half-pixel affine required by the image contract.
    xy = np.column_stack((
        (xs.astype(np.float64) + 0.5) * 1408.0 / float(mask.shape[1]) - 0.5,
        (ys.astype(np.float64) + 0.5) * 1408.0 / float(mask.shape[0]) - 0.5,
    ))
    step = max(1, int(stride))
    xy = xy[::step]
    tree = cKDTree(p09_uv)
    distance, nearest = tree.query(xy, k=1, workers=-1)
    keep = np.isfinite(distance) & (distance <= float(radius_px))
    xy = xy[keep]
    nearest = nearest[keep]
    distance = distance[keep]
    if len(xy) > int(max_rays):
        ids = np.linspace(0, len(xy) - 1, int(max_rays), dtype=np.int64)
        xy, nearest, distance = xy[ids], nearest[ids], distance[ids]
    rays = np.column_stack((
        (xy[:, 0] - K[0, 2]) / K[0, 0],
        (xy[:, 1] - K[1, 2]) / K[1, 1],
        np.ones(len(xy), dtype=np.float64),
    )).astype(np.float32)
    depth = p09[nearest, 2].astype(np.float32)
    return rays, depth, {
        "active": bool(len(rays)),
        "mask_path": str(Path(mask_path).expanduser().resolve()),
        "mask_source_wh": [int(mask.shape[1]), int(mask.shape[0])],
        "mask_stride": int(step),
        "support_radius_px": float(radius_px),
        "candidate_mask_pixels": int(len(xy)),
        "support_ray_count": int(len(rays)),
        "nearest_p09_distance_px": {
            "p50": float(np.percentile(distance, 50.0)) if len(distance) else None,
            "p90": float(np.percentile(distance, 90.0)) if len(distance) else None,
            "max": float(np.max(distance)) if len(distance) else None,
        },
        "depth_source": "nearest existing P09 first-hit point; no synthetic depth",
    }


def depth_order_faces(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    p09_camera: np.ndarray,
    *,
    support_rays: np.ndarray | None = None,
    support_depth: np.ndarray | None = None,
    neighbor_px: float,
    front_tolerance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    camera = camera_points_from_world(vertices_world, T_world_camera)
    uv, valid = project_points(camera, K, 1408, 1408)
    face_valid = np.all(valid[faces], axis=1)
    candidate_ids = np.flatnonzero(face_valid)
    if len(candidate_ids) == 0 or len(p09_camera) == 0:
        return candidate_ids.astype(np.int32), {"input_faces": int(len(faces)), "candidate_faces": int(len(candidate_ids)), "kept_faces": int(len(candidate_ids)), "dropped_front_faces": 0, "gate_active": False}
    if len(candidate_ids) == 0 or len(p09_camera) == 0:
        return candidate_ids.astype(np.int32), {"input_faces": int(len(faces)), "candidate_faces": int(len(candidate_ids)), "kept_faces": int(len(candidate_ids)), "dropped_front_faces": 0, "gate_active": False}
    p09_uv = np.column_stack((
        K[0, 0] * p09_camera[:, 0] / np.maximum(p09_camera[:, 2], 1.0e-9) + K[0, 2],
        K[1, 1] * p09_camera[:, 1] / np.maximum(p09_camera[:, 2], 1.0e-9) + K[1, 2],
    ))
    tree = cKDTree(p09_uv)
    tri = faces[candidate_ids]
    centroid_uv = np.mean(uv[tri], axis=1)
    distances, nearest = tree.query(centroid_uv, k=1, workers=-1)
    observed_z = p09_camera[nearest, 2]
    generated_z = np.mean(camera[tri, 2], axis=1)
    near = distances <= float(neighbor_px)
    front = generated_z < observed_z - float(front_tolerance_m)
    centroid_drop = near & front
    # Direct ray-wise pruning uses both the original P09 samples and a dense
    # mask-supported ray set.  Added rays copy depth from an existing P09
    # point; they are not synthetic geometry or a pose target.
    p09_rays = np.column_stack((
        p09_camera[:, 0] / np.maximum(p09_camera[:, 2], 1.0e-9),
        p09_camera[:, 1] / np.maximum(p09_camera[:, 2], 1.0e-9),
        np.ones(len(p09_camera), dtype=np.float64),
    )).astype(np.float32)
    all_rays = p09_rays
    all_observed_z = p09_camera[:, 2].astype(np.float32)
    support_count = 0
    if support_rays is not None and support_depth is not None:
        sr = np.asarray(support_rays, dtype=np.float32)
        sz = np.asarray(support_depth, dtype=np.float32).reshape(-1)
        if sr.ndim == 2 and sr.shape[1] == 3 and len(sr) == len(sz) and len(sr):
            all_rays = np.vstack((p09_rays, sr))
            all_observed_z = np.concatenate((all_observed_z, sz))
            support_count = int(len(sr))
    current_faces = faces[candidate_ids].copy()
    dropped_ray_faces: set[int] = set()
    for _iteration in range(20):
        if len(current_faces) == 0:
            break
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(camera.astype(np.float32)),
                o3d.core.Tensor(current_faces.astype(np.uint32)),
            )
        )
        hit = scene.cast_rays(o3d.core.Tensor(np.hstack([np.zeros((len(all_rays), 3), np.float32), all_rays])))
        hit_z = hit["t_hit"].numpy()
        primitive = hit["primitive_ids"].numpy().astype(np.int64)
        valid_hit = np.isfinite(hit_z) & (hit_z < 1.0e10) & (primitive >= 0)
        front_hit = valid_hit & (hit_z < all_observed_z - float(front_tolerance_m))
        if not np.any(front_hit):
            break
        remove = np.unique(primitive[front_hit])
        remove = remove[(remove >= 0) & (remove < len(current_faces))]
        if len(remove) == 0:
            break
        dropped_ray_faces.update(int(x) for x in remove.tolist())
        keep_mask = np.ones(len(current_faces), dtype=bool)
        keep_mask[remove] = False
        current_faces = current_faces[keep_mask]
    # Map the final face list back to the original candidate-face indices for
    # provenance and video/RRD reuse.
    kept = []
    face_lookup = {tuple(face.tolist()): int(idx) for idx, face in zip(candidate_ids.tolist(), faces[candidate_ids])}
    for face in current_faces:
        kept.append(face_lookup.get(tuple(face.tolist()), -1))
    kept = np.asarray([x for x in kept if x >= 0], dtype=np.int32)
    dropped_count = int(len(candidate_ids) - len(kept))
    # Verify the final ray-wise contract on the original P09 samples.  This is
    # intentionally separate from the denser support-ray diagnostic.
    post_front_fraction = None
    post_delta = None
    post_support_front_fraction = None
    if len(current_faces):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(camera.astype(np.float32)), o3d.core.Tensor(current_faces.astype(np.uint32))))
        hit_z = scene.cast_rays(o3d.core.Tensor(np.hstack([np.zeros((len(p09_rays), 3), np.float32), p09_rays])))['t_hit'].numpy()
        valid_hit = np.isfinite(hit_z) & (hit_z < 1.0e10)
        if np.any(valid_hit):
            delta = (hit_z[valid_hit] - p09_camera[:, 2][valid_hit]) * 1000.0
            post_front_fraction = float(np.mean(hit_z[valid_hit] < p09_camera[:, 2][valid_hit]))
            post_delta = {
                "p10": float(np.percentile(delta, 10)),
                "median": float(np.percentile(delta, 50)),
                "p90": float(np.percentile(delta, 90)),
            }
        if support_count:
            support_hit = scene.cast_rays(o3d.core.Tensor(np.hstack([np.zeros((support_count, 3), np.float32), all_rays[len(p09_rays):]])))['t_hit'].numpy()
            support_valid = np.isfinite(support_hit) & (support_hit < 1.0e10)
            if np.any(support_valid):
                post_support_front_fraction = float(np.mean(support_hit[support_valid] < all_observed_z[len(p09_rays):][support_valid]))
    return kept, {
        "input_faces": int(len(faces)),
        "candidate_faces": int(len(candidate_ids)),
        "kept_faces": int(len(kept)),
        "dropped_front_faces": dropped_count,
        "dropped_by_centroid_heuristic": int(centroid_drop.sum()),
        "dropped_by_direct_p09_ray_pruning": int(len(dropped_ray_faces)),
        "support_ray_count": int(support_count),
        "gate_active": True,
        "neighbor_px": float(neighbor_px),
        "front_tolerance_m": float(front_tolerance_m),
        "near_fraction": float(np.mean(near)),
        "front_fraction_near": float(np.mean(front[near])) if np.any(near) else 0.0,
        "median_depth_delta_near_mm": float(np.median((generated_z[near] - observed_z[near]) * 1000.0)) if np.any(near) else None,
        "post_prune_front_fraction_on_p09_rays": post_front_fraction,
        "post_prune_front_fraction_on_support_rays": post_support_front_fraction,
        "post_prune_delta_z_mm": post_delta,
    }


def compact_mesh(vertices_world: np.ndarray, faces: np.ndarray, face_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = faces[face_ids]
    if len(selected) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    used, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    return vertices_world[used].astype(np.float32), inverse.reshape(-1, 3).astype(np.int32)


def overlay_mesh(image: np.ndarray, vertices_world: np.ndarray, faces: np.ndarray, T_world_camera: np.ndarray, K: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    camera = camera_points_from_world(vertices_world, T_world_camera)
    uv, valid = project_points(camera, K, image.shape[1], image.shape[0])
    ids = np.flatnonzero(np.all(valid[faces], axis=1))
    if len(ids) == 0:
        return
    order = ids[np.argsort(np.mean(camera[faces[ids], 2], axis=1))[::-1]]
    polygons = []
    for face_id in order:
        poly = np.rint(uv[faces[face_id]]).astype(np.int32)
        if len(np.unique(poly, axis=0)) >= 3 and abs(float(cv2.contourArea(poly.astype(np.float32)))) > 0.25:
            polygons.append(poly)
    if not polygons:
        return
    layer = image.copy()
    cv2.fillPoly(layer, polygons, color)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    keep = mask > 0
    image[keep] = cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0)[keep]
    for poly in polygons[:: max(1, len(polygons) // 700)]:
        cv2.polylines(image, [poly], True, tuple(int(x * .65) for x in color), 1, cv2.LINE_AA)


def overlay_points(image: np.ndarray, points_world: np.ndarray, T_world_camera: np.ndarray, K: np.ndarray, stride: int) -> None:
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) == 0:
        return
    camera = camera_points_from_world(points_world[::stride], T_world_camera)
    uv, valid = project_points(camera, K, image.shape[1], image.shape[0])
    for point in np.rint(uv[valid]).astype(np.int32):
        cv2.circle(image, tuple(point), 1, VISIBLE_POINT_COLOR_BGR, -1, cv2.LINE_AA)


def overlay_hands(image: np.ndarray, hands: dict[str, tuple[np.ndarray, np.ndarray | None]], T_world_camera: np.ndarray, K: np.ndarray) -> None:
    for side, color in (("left", LEFT_COLOR_BGR), ("right", RIGHT_COLOR_BGR)):
        if side not in hands:
            continue
        vertices, joints = hands[side]
        camera = camera_points_from_world(vertices, T_world_camera)
        uv, valid = project_points(camera, K, image.shape[1], image.shape[0])
        if valid.sum() >= 3:
            hull = cv2.convexHull(np.rint(uv[valid]).astype(np.int32))
            layer = image.copy(); cv2.fillConvexPoly(layer, hull, color); mask=np.zeros(image.shape[:2],np.uint8);cv2.fillConvexPoly(mask,hull,255);keep=mask>0;image[keep]=cv2.addWeighted(layer,.16,image,.84,0)[keep];cv2.polylines(image,[hull],True,color,2,cv2.LINE_AA)
        if joints is None: continue
        jc=camera_points_from_world(joints,T_world_camera);ju,jv=project_points(jc,K,image.shape[1],image.shape[0])
        for a,b in HAND_EDGES:
            if jv[a] and jv[b]:cv2.line(image,tuple(np.rint(ju[a]).astype(int)),tuple(np.rint(ju[b]).astype(int)),color,2,cv2.LINE_AA)
        for j in ju[jv]:cv2.circle(image,tuple(np.rint(j).astype(int)),2,color,-1,cv2.LINE_AA)


def draw_mask(image: np.ndarray, path: str | None) -> None:
    if not path:return
    mask=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if mask is None:return
    if mask.shape[:2]!=image.shape[:2]:mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST_EXACT)
    c,_=cv2.findContours((mask>0).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if c:cv2.drawContours(image,c,-1,MASK_CONTOUR_BGR,2,cv2.LINE_AA)


def add_banner(image: np.ndarray, lines: list[str]) -> None:
    layer=image.copy();cv2.rectangle(layer,(0,0),(image.shape[1],64),(8,8,8),-1);image[:64]=cv2.addWeighted(layer[:64],.82,image[:64],.18,0)
    for i,line in enumerate(lines):cv2.putText(image,line,(12,24+i*22),cv2.FONT_HERSHEY_SIMPLEX,.5 if i==0 else .38,(245,245,245),1,cv2.LINE_AA)


def encode_video(frame_dir: Path, output: Path, fps: float, start_number: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-framerate", str(fps), "-start_number", str(int(start_number)), "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True);parser.add_argument('--pose-report',type=Path,required=True);parser.add_argument('--generated-mesh',type=Path,required=True);parser.add_argument('--mano-bridge',type=Path,required=True);parser.add_argument('--mano-topology',type=Path,required=True);parser.add_argument('--source-video',type=Path,required=True);parser.add_argument('--object-id',default='carton_milk');parser.add_argument('--frame-start',type=int,default=0);parser.add_argument('--frame-end',type=int,default=149);parser.add_argument('--output-rrd',type=Path,required=True);parser.add_argument('--output-video',type=Path,required=True);parser.add_argument('--decimated-face-count',type=int,default=30000);parser.add_argument('--depth-neighbor-px',type=float,default=12.0);parser.add_argument('--support-radius-px',type=float,default=8.0);parser.add_argument('--support-mask-stride',type=int,default=4);parser.add_argument('--support-max-rays',type=int,default=12000);parser.add_argument('--front-tolerance-mm',type=float,default=3.0);parser.add_argument('--point-stride',type=int,default=1);parser.add_argument('--overlay-point-stride',type=int,default=8);parser.add_argument('--hide-visible-surface-points',action='store_true',help='Hide P09 visible-surface points from RRD and video while retaining SAM3D, MANO, and camera world pose.');parser.add_argument('--jpeg-quality',type=int,default=84);parser.add_argument('--label',default='v20_support_aware_focused')
    args=parser.parse_args()
    if args.frame_end<args.frame_start:raise RuntimeError('invalid frame range')
    for p in (args.annotations,args.pose_report,args.generated_mesh,args.mano_bridge,args.mano_topology,args.source_video):
        if not p.expanduser().resolve().is_file():raise FileNotFoundError(p)
    pose_payload = load_json(args.pose_report)
    render_mesh_contract = validate_render_mesh_contract(
        pose_payload, args.generated_mesh
    )
    preserve_objective_mesh = bool(render_mesh_contract.get("required"))
    ann = load_json(args.annotations)
    frames = {
        int(f["frame_idx"]): f
        for f in ann.get("frames", [])
        if isinstance(f, dict) and f.get("frame_idx") is not None
    }
    selected = list(range(args.frame_start, args.frame_end + 1))
    poses = pose_rows(args.pose_report)
    missing = [i for i in selected if i not in poses]
    if missing:
        raise RuntimeError(f"pose report missing frames: {missing[:10]}")
    mesh = load_mesh(args.generated_mesh)
    canon_vertices, canon_faces = decimate_mesh(
        mesh, int(args.decimated_face_count)
    )
    with np.load(args.mano_bridge.expanduser().resolve(),allow_pickle=False) as bridge:
        bf=np.asarray(bridge['frame_idx'],np.int64);bs=np.asarray(bridge['hand_side']).astype(str);bv=np.asarray(bridge['vertices_current_v18_world_from_hawor_projection_relift_m'],np.float32);bj=np.asarray(bridge['joints_current_v18_world_from_hawor_projection_relift_m'],np.float32) if 'joints_current_v18_world_from_hawor_projection_relift_m' in bridge.files else None
    with np.load(args.mano_topology.expanduser().resolve(),allow_pickle=False) as top:left_faces=np.asarray(top['left_faces'],np.int32);right_faces=np.asarray(top['right_faces'],np.int32)
    hands={}
    for i,(idx,side,v) in enumerate(zip(bf.tolist(),bs.tolist(),bv)):hands.setdefault(int(idx),{})[side]=(v,bj[i] if bj is not None else None)
    video=cv2.VideoCapture(str(args.source_video.expanduser().resolve()));
    if not video.isOpened():raise RuntimeError('failed to open source video')
    fps=float(video.get(cv2.CAP_PROP_FPS) or 30.);vw=int(video.get(cv2.CAP_PROP_FRAME_WIDTH));vh=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT));video.set(cv2.CAP_PROP_POS_FRAMES,args.frame_start)
    args.output_rrd.parent.mkdir(parents=True,exist_ok=True);rr.init(f'milk_{args.label}',spawn=False);rr.save(str(args.output_rrd.expanduser().resolve()));rr.log('/',rr.ViewCoordinates.RDF,static=True)
    rr.log('/world/sam3d_raw/mesh',rr.Mesh3D(vertex_positions=canon_vertices.astype(np.float32),triangle_indices=canon_faces,albedo_factor=[215,45,190,130]),static=True)
    rr.log(
        "/metadata/description",
        rr.TextDocument(
            "# Objective-mesh V20 visualization\n\n"
            + (
                "The generated mesh source is hash-bound to the pose objective. "
                "No display-only front-face pruning is applied."
                if preserve_objective_mesh
                else "Legacy display-only depth-order pruning is active; this does not modify pose or formal state."
            )
        ),
        static=True,
    )
    tmp=Path(tempfile.mkdtemp(prefix='v20_depth_ordered_overlay_',dir=str(args.output_rrd.parent)));metric=[];empty=[];stats=[];camtraj=[]
    try:
        for idx in selected:
            ok,frame_bgr=video.read()
            if not ok:raise RuntimeError(f'failed decode frame {idx}')
            fr=frames[idx];row=poses[idx];R=np.asarray(row['rotation_world_from_completed_canonical_matrix'],float);t=np.asarray(row['translation_world_m'],float);T=np.asarray(fr['camera']['T_world_camera_metric'],float);kv=np.asarray(fr['camera']['intrinsics_fx_fy_cx_cy'],float);K=np.array([[kv[0],0,kv[2]],[0,kv[1],kv[3]],[0,0,1.]],float);K960=resize_intrinsics_half_pixel(K,(1408,1408),(960,960));
            obj=next((o for o in fr.get('objects',[]) if o.get('object_id')==args.object_id),None);geom=obj.get('visible_geometry_candidate') if isinstance(obj,dict) and isinstance(obj.get('visible_geometry_candidate'),dict) else {};p09=np.asarray(geom.get('camera_vertices_sample_m') or [],float)
            p09_world = camera_points_to_world(p09, T) if p09.ndim == 2 and p09.shape[1] == 3 else np.empty((0, 3), dtype=np.float64)
            rr.set_time('frame',sequence=idx);cp=T[:3,3];camtraj.append(cp);rr.log('/world/camera',rr.Transform3D(translation=cp.tolist(),mat3x3=T[:3,:3].tolist()));rr.log('/world/camera/world_pose',rr.TextLog('T_world_camera_metric = '+np.array2string(T,precision=6,suppress_small=True)));rr.log('/world/camera/trajectory',rr.LineStrips3D([np.asarray(camtraj,np.float32)],radii=.0015,colors=[[180,180,180,220]]));rr.log('/world/camera/image',rr.Pinhole(image_from_camera=K960.tolist(),resolution=[960,960]));rgb960=cv2.resize(frame_bgr,(960,960),interpolation=cv2.INTER_AREA);rr.log('/world/camera/image',rr.Image(cv2.cvtColor(rgb960,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=args.jpeg_quality))
            world=canon_vertices@R.T+t[None,:];
            if p09.ndim == 2 and p09.shape == (2500, 3) and np.isfinite(p09).all():
                metric.append(idx)
                support_rays, support_depth, support_stats = build_support_rays(
                    p09,
                    K,
                    geom.get("mask_path"),
                    radius_px=args.support_radius_px,
                    stride=args.support_mask_stride,
                    max_rays=args.support_max_rays,
                )
                if preserve_objective_mesh:
                    kept = np.arange(len(canon_faces), dtype=np.int32)
                    st = {
                        "frame_idx": idx,
                        "input_faces": int(len(canon_faces)),
                        "candidate_faces": int(len(canon_faces)),
                        "kept_faces": int(len(canon_faces)),
                        "dropped_front_faces": 0,
                        "gate_active": False,
                        "reason": "pose objective render-mesh contract forbids display-only face pruning",
                        "objective_mesh_preserved": True,
                        "support_raster": support_stats,
                    }
                else:
                    kept, st = depth_order_faces(
                        world,
                        canon_faces,
                        T,
                        K,
                        p09,
                        support_rays=support_rays,
                        support_depth=support_depth,
                        neighbor_px=args.depth_neighbor_px,
                        front_tolerance_m=args.front_tolerance_mm / 1000.0,
                    )
                    st["support_raster"] = support_stats
                    st["objective_mesh_preserved"] = False
                stats.append({"frame_idx": idx, **st})
                rv, rf = compact_mesh(world, canon_faces, kept)
                rr.log(
                    "/world/sam3d_depth_ordered",
                    rr.Mesh3D(
                        vertex_positions=rv,
                        triangle_indices=rf,
                        albedo_factor=[215, 45, 190, 150],
                    ),
                )
                if not args.hide_visible_surface_points:
                    rr.log(
                        "/world/visible_surface",
                        rr.Points3D(
                            p09_world[:: args.point_stride],
                            radii=0.0018,
                            colors=[255, 220, 0, 255],
                            point_shading=rr.components.PointShading.Flat,
                        ),
                    )
            else:
                empty.append(idx)
                kept = np.arange(len(canon_faces), dtype=np.int32)
                stats.append(
                    {
                        "frame_idx": idx,
                        "gate_active": False,
                        "input_faces": len(canon_faces),
                        "kept_faces": len(canon_faces),
                        "dropped_front_faces": 0,
                        "objective_mesh_preserved": preserve_objective_mesh,
                    }
                )
                rr.log(
                    "/world/sam3d_depth_ordered",
                    rr.Mesh3D(
                        vertex_positions=world.astype(np.float32),
                        triangle_indices=canon_faces,
                        albedo_factor=[215, 45, 190, 110],
                    ),
                )
                if not args.hide_visible_surface_points:
                    rr.log("/world/visible_surface", rr.Clear(recursive=False))
                    rr.log(
                        "/world/visible_surface/status",
                        rr.TextLog("no accepted P09 metric surface; depth gate inactive"),
                    )
            for side,faces,color in [('left',left_faces,[255,150,40,255]),('right',right_faces,[80,150,255,255])]:
                if side in hands:
                    v,j=hands[side];rr.log(f'/world/hands/{side}',rr.Mesh3D(vertex_positions=v,triangle_indices=faces,albedo_factor=color));rr.log(f'/world/hands/{side}/status',rr.TextLog('MANO world mesh shown; joints '+('shown' if j is not None else 'unavailable')))
                else:rr.log(f'/world/hands/{side}',rr.Clear(recursive=False));rr.log(f'/world/hands/{side}/status',rr.TextLog('MANO side unavailable for this frame'))
            # Overlay corrected mesh to original video: use the same face gate and
            # a sampled world mesh, not the full 476k-face topology.
            overlay=rgb960.copy()
            if p09.ndim==2 and p09.shape==(2500,3) and np.isfinite(p09).all():
                if preserve_objective_mesh:
                    kept = np.arange(len(canon_faces), dtype=np.int32)
                    stats[-1]["video_gate"] = {
                        "gate_active": False,
                        "objective_mesh_preserved": True,
                        "kept_faces": int(len(kept)),
                    }
                else:
                    kept,st2=depth_order_faces(world,canon_faces,T,K,p09,support_rays=support_rays,support_depth=support_depth,neighbor_px=args.depth_neighbor_px,front_tolerance_m=args.front_tolerance_mm/1000.)
                    stats[-1]['video_gate']=st2
            else:
                kept=np.arange(len(canon_faces),dtype=np.int32)
            overlay_mesh(overlay,world,canon_faces[kept],T,K960,GENERATED_COLOR_BGR,.25);overlay_hands(overlay,hands.get(idx,{}),T,K960);
            if not args.hide_visible_surface_points:
                overlay_points(overlay,p09_world,T,K960,args.overlay_point_stride)
            draw_mask(overlay,geom.get('mask_path'));add_banner(overlay,[f'Objective-mesh render | frame={idx:03d} | SAM3D magenta',('mesh/hash bound to pose objective; no display-only face pruning' if preserve_objective_mesh else f'front tolerance={args.front_tolerance_mm:g} mm; display-only depth gate'),('P09 visible-surface point cloud hidden' if args.hide_visible_surface_points else 'P09 visible-surface points shown')])
            side=np.hstack([rgb960,overlay]);cv2.imwrite(str(tmp/f'{idx:06d}.jpg'),side,[cv2.IMWRITE_JPEG_QUALITY,args.jpeg_quality]);rr.log('/comparison/original_video',rr.Image(cv2.cvtColor(rgb960,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=args.jpeg_quality));rr.log('/comparison/depth_ordered_overlay',rr.Image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=args.jpeg_quality));rr.log('/comparison/side_by_side',rr.Image(cv2.cvtColor(side,cv2.COLOR_BGR2RGB)).compress(jpeg_quality=args.jpeg_quality));
        video.release();contents=['+ /world/sam3d_depth_ordered','+ /world/hands/**','+ /world/camera/**'];
        if not args.hide_visible_surface_points:
            contents.append('+ /world/visible_surface')
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(rrb.Spatial3DView(origin='/world',name='SAM3D + MANO + camera world pose',contents=contents),rrb.Vertical(rrb.Spatial2DView(origin='/comparison/original_video',name='Original video'),rrb.Spatial2DView(origin='/comparison/depth_ordered_overlay',name='SAM3D + MANO overlay'),rrb.Spatial2DView(origin='/comparison/side_by_side',name='Original | SAM3D + MANO')),column_shares=[3,2]),collapse_panels=False));rr.disconnect();encode_video(tmp,args.output_video,fps,start_number=args.frame_start)
    finally:shutil.rmtree(tmp,ignore_errors=True)
    report = {
        "schema": "v20_support_aware_focused_visualization_v2",
        "status": "ok",
        "diagnostic_only": True,
        "formal_state_modified": False,
        "display_only_depth_gate": not preserve_objective_mesh,
        "objective_mesh_preserved": preserve_objective_mesh,
        "render_mesh_contract": render_mesh_contract,
        "pose_report": str(args.pose_report.expanduser().resolve()),
        "generated_mesh": str(args.generated_mesh.expanduser().resolve()),
        "generated_mesh_role": "visible_pose_objective_and_render_hypothesis"
        if preserve_objective_mesh
        else "render_only_completion_hypothesis",
        "visible_surface_points_displayed": not bool(args.hide_visible_surface_points),
        "mano_display": {
            "world_frame": "MANO bridge world coordinates",
            "mesh_logged": True,
            "joint_skeleton_overlay": True,
        },
        "camera_world_pose_display": {
            "transform_logged": True,
            "trajectory_logged": True,
            "transform_field": "T_world_camera_metric",
            "pose_frame": "world",
        },
        "visible_surface_coordinate_contract": {
            "source_field": "visible_geometry_candidate.camera_vertices_sample_m",
            "source_frame": "camera",
            "rrd_frame": "world",
            "conversion": "camera_to_world_once",
            "overlay_projection": "world_to_camera_once",
        },
        "display_geometry": {
            "source_vertices": int(len(mesh.vertices)),
            "source_faces": int(len(mesh.faces)),
            "display_vertices": int(len(canon_vertices)),
            "display_faces": int(len(canon_faces)),
            "decimation_requested_face_count": int(args.decimated_face_count),
            "face_pruning_applied": not preserve_objective_mesh,
        },
        "depth_gate": {
            "neighbor_px": args.depth_neighbor_px,
            "support_radius_px": args.support_radius_px,
            "support_mask_stride": args.support_mask_stride,
            "support_max_rays": args.support_max_rays,
            "front_tolerance_mm": args.front_tolerance_mm,
            "policy": (
                "disabled because the pose report binds the objective mesh; all display faces are retained"
                if preserve_objective_mesh
                else "legacy display-only P09 first-hit pruning; no pose/state changes"
            ),
        },
        "timeline": {
            "frame_count": len(selected),
            "metric_frames": metric,
            "no_metric_frames": empty,
        },
        "stats": stats,
        "outputs": {
            "rrd": str(args.output_rrd.expanduser().resolve()),
            "video": str(args.output_video.expanduser().resolve()),
        },
        "inputs_sha256": {
            "annotations": sha256_file(args.annotations),
            "pose_report": sha256_file(args.pose_report),
            "generated_mesh": sha256_file(args.generated_mesh),
            "mano_bridge": sha256_file(args.mano_bridge),
            "source_video": sha256_file(args.source_video),
        },
        "claim_scope": (
            "Visualization of the exact source mesh bound to the pose objective when a render contract is present. "
            "Display decimation is recorded, but no front faces are hidden. Generated geometry remains outside "
            "collision/contact/SDF/sign authority."
        ),
    }
    args.output_rrd.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'ok','rrd':str(args.output_rrd),'video':str(args.output_video),'metric_frames':len(metric),'empty_frames':empty,'stats':stats},indent=2))
if __name__=='__main__':main()
