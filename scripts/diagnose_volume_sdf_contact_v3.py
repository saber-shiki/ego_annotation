#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt

from diagnose_contact_kinematics_v3 import selected_vertex_ids
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, contact_rows, hand_vertices_camera
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def voxel_sdf(mesh: trimesh.Trimesh, pitch: float, pad_voxels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vox = mesh.voxelized(pitch=float(pitch)).fill()
    occ = np.asarray(vox.matrix, dtype=bool)
    if np.count_nonzero(occ) == 0:
        raise RuntimeError("voxelized mesh has no occupied cells")
    pad = int(pad_voxels)
    occ_pad = np.pad(occ, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occ_pad, sampling=[float(pitch)] * 3)
    inside = distance_transform_edt(occ_pad, sampling=[float(pitch)] * 3)
    sdf = outside - inside
    transform = np.asarray(vox.transform, dtype=np.float64).copy()
    transform[:3, 3] -= float(pitch) * pad
    return sdf.astype(np.float32), transform, occ_pad


def sample_sdf(points: np.ndarray, sdf: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pitch = float(transform[0, 0])
    if pitch <= 0.0:
        raise RuntimeError("invalid SDF transform pitch")
    origin = transform[:3, 3]
    coords = (points - origin[None, :]) / pitch
    base = np.floor(coords).astype(np.int64)
    frac = coords - base.astype(np.float64)
    in_bounds = (
        (base[:, 0] >= 0)
        & (base[:, 0] + 1 < sdf.shape[0])
        & (base[:, 1] >= 0)
        & (base[:, 1] + 1 < sdf.shape[1])
        & (base[:, 2] >= 0)
        & (base[:, 2] + 1 < sdf.shape[2])
    )
    values = np.full(len(points), np.nan, dtype=np.float64)
    if np.any(in_bounds):
        b = base[in_bounds]
        f = frac[in_bounds]
        x0, y0, z0 = b[:, 0], b[:, 1], b[:, 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xd, yd, zd = f[:, 0], f[:, 1], f[:, 2]
        c000 = sdf[x0, y0, z0]
        c100 = sdf[x1, y0, z0]
        c010 = sdf[x0, y1, z0]
        c110 = sdf[x1, y1, z0]
        c001 = sdf[x0, y0, z1]
        c101 = sdf[x1, y0, z1]
        c011 = sdf[x0, y1, z1]
        c111 = sdf[x1, y1, z1]
        c00 = c000 * (1.0 - xd) + c100 * xd
        c10 = c010 * (1.0 - xd) + c110 * xd
        c01 = c001 * (1.0 - xd) + c101 * xd
        c11 = c011 * (1.0 - xd) + c111 * xd
        c0 = c00 * (1.0 - yd) + c10 * yd
        c1 = c01 * (1.0 - yd) + c11 * yd
        values[in_bounds] = c0 * (1.0 - zd) + c1 * zd
    return values


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    meshes = load_mesh_archive(args.mesh_archive)
    rows = []
    all_sdf = []
    all_abs_sdf = []
    sdf_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in contact_rows(args.contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        if frame_idx not in annotations or frame_idx not in meshes:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices_camera = hand_vertices_camera(hand)
        patch_ids = selected_vertex_ids(row)
        if int(patch_ids.max()) >= len(vertices_camera):
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} patch id exceeds MANO vertex count")
        patch_camera = vertices_camera[patch_ids]
        if frame_idx not in sdf_by_frame:
            mesh_world, mesh_faces = meshes[frame_idx]
            T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
            mesh_camera = camera_points(mesh_world, T_world_camera)
            mesh = trimesh.Trimesh(vertices=mesh_camera.astype(np.float32), faces=np.asarray(mesh_faces, dtype=np.int32), process=True)
            sdf_by_frame[frame_idx] = voxel_sdf(mesh, float(args.pitch_m), int(args.pad_voxels))
        sdf, transform, occ = sdf_by_frame[frame_idx]
        values = sample_sdf(patch_camera, sdf, transform)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} has no SDF samples in bounds")
        all_sdf.extend(finite.astype(float).tolist())
        all_abs_sdf.extend(np.abs(finite).astype(float).tolist())
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "hand_idx": int(hand_idx),
                "track_id": row.get("track_id"),
                "selected_patch_source": row.get("selected_patch_source"),
                "selected_patch_region": row.get("selected_patch_region"),
                "patch_points": int(len(patch_camera)),
                "sdf_m": summarize(finite),
                "abs_sdf_m": summarize(np.abs(finite)),
                "penetration_fraction": float(np.mean(finite < -float(args.penetration_tolerance_m))),
                "near_surface_fraction": float(np.mean(np.abs(finite) <= float(args.near_surface_m))),
                "voxel_occupied": int(np.count_nonzero(occ)),
                "voxel_shape": [int(v) for v in occ.shape],
            }
        )
    if not rows:
        raise RuntimeError("no contact rows diagnosed")
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "volume_sdf_contact_v3",
        "claim_tested": "MANO contact patches are evaluated against a conservative voxel SDF for open fused object geometry",
        "annotations": str(args.annotations),
        "mesh_archive": str(args.mesh_archive),
        "contact_report": str(args.contact_report),
        "frames": sorted({int(row["frame_idx"]) for row in rows}),
        "summary": {
            "sdf_m": summarize(all_sdf),
            "abs_sdf_m": summarize(all_abs_sdf),
            "penetration_fraction": float(np.mean(np.asarray(all_sdf, dtype=np.float64) < -float(args.penetration_tolerance_m))),
            "near_surface_fraction": float(np.mean(np.asarray(all_abs_sdf, dtype=np.float64) <= float(args.near_surface_m))),
        },
        "rows": rows,
        "parameters": {
            "pitch_m": float(args.pitch_m),
            "pad_voxels": int(args.pad_voxels),
            "penetration_tolerance_m": float(args.penetration_tolerance_m),
            "near_surface_m": float(args.near_surface_m),
        },
    }
    save_json(args.output_json, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--pad-voxels", type=int, default=8)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.002)
    parser.add_argument("--near-surface-m", type=float, default=0.006)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
