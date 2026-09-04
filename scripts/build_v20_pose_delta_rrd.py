#!/usr/bin/env python3
"""Build a Rerun recording that makes V20 pose changes explicit.

The ordinary combined RRD can make small pose changes hard to see because the
red/green/blue meshes overlap.  This recording additionally logs true formal
-> V20 translation arrows, corresponding mesh-vertex displacement segments,
object axes, and a clearly labelled 10x visualization aid.  The amplification
is visual-only and never changes the stored pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from scipy.spatial.transform import Rotation


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


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in load_json(path).get("pose_rows", []):
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        if row.get("rotation_world_from_completed_canonical_matrix") is None:
            continue
        if row.get("translation_world_m") is None:
            continue
        rows[int(row["frame_idx"])] = row
    return rows


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"no mesh in {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh in {path}")
    return mesh


def get_pose(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(row["rotation_world_from_completed_canonical_matrix"], dtype=np.float64)
    translation = np.asarray(row["translation_world_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise RuntimeError("invalid pose row")
    return rotation, translation


def log_axes(path: str, rotation: np.ndarray, translation: np.ndarray, color: list[int], length: float = 0.045) -> None:
    strips = np.stack(
        [
            np.stack([translation, translation + rotation[:, 0] * length]),
            np.stack([translation, translation + rotation[:, 1] * length]),
            np.stack([translation, translation + rotation[:, 2] * length]),
        ],
        axis=0,
    ).astype(np.float32)
    rr.log(path, rr.LineStrips3D(strips, radii=0.0025, colors=[color, color, color]))


def log_pose_layer(path: str, row: dict[str, Any], color: list[int]) -> np.ndarray:
    rotation, translation = get_pose(row)
    rr.log(path, rr.Transform3D(translation=translation.tolist(), mat3x3=rotation.tolist()))
    rr.log(path + "/origin", rr.Points3D([translation], radii=0.006, colors=color))
    log_axes(path + "/axes", rotation, translation, color)
    return translation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--formal-pose-report", type=Path, required=True)
    parser.add_argument("--v20-k5-pose-report", type=Path, required=True)
    parser.add_argument("--v20-k10-pose-report", type=Path, required=True)
    parser.add_argument("--observed-mesh", type=Path, required=True)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--frames", default="all")
    parser.add_argument("--vertex-stride", type=int, default=80)
    parser.add_argument("--amplification", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="v20_pose_delta")
    args = parser.parse_args()
    if args.vertex_stride < 1 or args.amplification < 1.0:
        raise RuntimeError("vertex stride must be positive and amplification must be >= 1")

    annotations = load_json(args.annotations)
    frames = {
        int(frame["frame_idx"]): frame
        for frame in annotations.get("frames", [])
        if isinstance(frame, dict) and frame.get("frame_idx") is not None
    }
    formal = load_rows(args.formal_pose_report)
    k5 = load_rows(args.v20_k5_pose_report)
    k10 = load_rows(args.v20_k10_pose_report)
    if args.frames == "all":
        selected = sorted(frames)
    else:
        selected = sorted({int(x) for x in args.frames.split(",") if x.strip()})
    if not selected:
        raise RuntimeError("no frames selected")
    unknown = [idx for idx in selected if idx not in frames]
    if unknown:
        raise RuntimeError(f"frames absent from annotations: {unknown[:10]}")

    mesh = load_mesh(args.observed_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    sample_vertices = vertices[:: int(args.vertex_stride)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"milk_{args.label}", spawn=False)
    rr.save(str(args.output.expanduser().resolve()))
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    layer_info = (
        ("formal", formal, [220, 50, 50, 170]),
        ("v20_k5", k5, [30, 220, 90, 210]),
        ("v20_k10", k10, [50, 110, 255, 190]),
    )
    for name, _rows, color in layer_info:
        rr.log(
            f"/world/{name}/mesh",
            rr.Mesh3D(vertex_positions=vertices.astype(np.float32), triangle_indices=faces, albedo_factor=color),
            static=True,
        )

    delta_values: dict[str, list[float]] = {"k5_translation_m": [], "k5_rotation_deg": [], "k10_translation_m": [], "k10_rotation_deg": []}
    pose_presence: dict[str, list[bool]] = {name: [] for name, _rows, _color in layer_info}
    metric_frames: list[int] = []
    no_metric_frames: list[int] = []

    for frame_idx in selected:
        rr.set_time("frame", sequence=frame_idx)
        frame = frames[frame_idx]
        camera = frame.get("camera") or {}
        T = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"invalid camera at frame {frame_idx}")
        rr.log("/world/camera", rr.Transform3D(translation=T[:3, 3].tolist(), mat3x3=T[:3, :3].tolist()))

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, rows, color in layer_info:
            row = rows.get(frame_idx)
            pose_presence[name].append(row is not None)
            if row is None:
                rr.log(f"/world/{name}/object", rr.Clear(recursive=False))
                rr.log(f"/world/{name}/status", rr.TextLog("no pose at this frame"))
                continue
            rotation, translation = get_pose(row)
            poses[name] = (rotation, translation)
            log_pose_layer(f"/world/{name}/object", row, color)

        obj = next((x for x in frame.get("objects", []) if isinstance(x, dict) and x.get("object_id") == args.object_id), None)
        geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=np.float32)
        if observed.ndim == 2 and observed.shape[1] == 3 and len(observed) and np.isfinite(observed).all():
            metric_frames.append(frame_idx)
            rr.log("/world/visible_surface", rr.Points3D(observed[::4], radii=0.0022, colors=[255, 220, 0, 255]))
        else:
            no_metric_frames.append(frame_idx)
            rr.log("/world/visible_surface", rr.Clear(recursive=False))

        formal_pose = poses.get("formal")
        for name, key in (("v20_k5", "k5"), ("v20_k10", "k10")):
            candidate = poses.get(name)
            if formal_pose is None or candidate is None:
                rr.log(f"/world/delta/{key}", rr.Clear(recursive=False))
                continue
            formal_R, formal_t = formal_pose
            candidate_R, candidate_t = candidate
            delta_t = candidate_t - formal_t
            delta_r = Rotation.from_matrix(candidate_R @ formal_R.T).magnitude()
            delta_t_norm = float(np.linalg.norm(delta_t))
            delta_r_deg = float(np.degrees(delta_r))
            delta_values[f"{key}_translation_m"].append(delta_t_norm)
            delta_values[f"{key}_rotation_deg"].append(delta_r_deg)

            rr.log(
                f"/world/delta/{key}/true_translation",
                rr.Arrows3D(vectors=[delta_t.astype(np.float32)], origins=[formal_t.astype(np.float32)], radii=0.0018, colors=[[240, 0, 240, 255]]),
            )
            amplified = delta_t * float(args.amplification)
            rr.log(
                f"/world/delta/{key}/amplified_{int(args.amplification)}x",
                rr.Arrows3D(vectors=[amplified.astype(np.float32)], origins=[formal_t.astype(np.float32)], radii=0.0022, colors=[[255, 150, 0, 255]], labels=[f"{key} {args.amplification:g}x"], show_labels=True),
            )
            formal_vertices = sample_vertices @ formal_R.T + formal_t[None, :]
            candidate_vertices = sample_vertices @ candidate_R.T + candidate_t[None, :]
            true_strips = np.stack([formal_vertices, candidate_vertices], axis=1).astype(np.float32)
            amplified_end = formal_vertices + float(args.amplification) * (candidate_vertices - formal_vertices)
            amplified_strips = np.stack([formal_vertices, amplified_end], axis=1).astype(np.float32)
            rr.log(f"/world/delta/{key}/true_vertex_segments", rr.LineStrips3D(true_strips, radii=0.001, colors=[[240, 0, 240, 190]]))
            rr.log(f"/world/delta/{key}/amplified_vertex_segments", rr.LineStrips3D(amplified_strips, radii=0.0015, colors=[[255, 150, 0, 210]]))
            rr.log(
                f"/world/delta/{key}/metrics",
                rr.TextLog(f"frame={frame_idx}  Δt={delta_t_norm * 1000.0:.3f} mm  ΔR={delta_r_deg:.3f} deg"),
            )

    camera_positions = np.asarray(
        [np.asarray(frames[idx]["camera"]["T_world_camera_metric"], dtype=np.float32)[:3, 3] for idx in selected],
        dtype=np.float32,
    )
    rr.log("/world/trajectory/camera", rr.Points3D(camera_positions, radii=0.0025, colors=[170, 170, 170, 220]), static=True)

    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial3DView(
                origin="/world",
                name="Before / after overlay",
                contents=[
                    "+ /world/formal/**",
                    "+ /world/v20_k5/**",
                    "+ /world/v20_k10/**",
                    "+ /world/visible_surface",
                    "+ /world/trajectory/**",
                ],
            ),
            rrb.Spatial3DView(
                origin="/world",
                name="True pose delta",
                contents=[
                    "+ /world/delta/k5/true_translation",
                    "+ /world/delta/k10/true_translation",
                    "+ /world/delta/k5/true_vertex_segments",
                    "+ /world/delta/k10/true_vertex_segments",
                    "+ /world/formal/object/origin",
                    "+ /world/v20_k5/object/origin",
                    "+ /world/v20_k10/object/origin",
                ],
            ),
            rrb.Spatial3DView(
                origin="/world",
                name="10x visual aid (not metric)",
                contents=[
                    "+ /world/delta/k5/amplified_10x",
                    "+ /world/delta/k10/amplified_10x",
                    "+ /world/delta/k5/amplified_vertex_segments",
                    "+ /world/delta/k10/amplified_vertex_segments",
                    "+ /world/formal/object/origin",
                ],
            ),
            active_tab=0,
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    report = {
        "schema": "v20_pose_delta_world_rrd_v1",
        "status": "ok",
        "diagnostic_only": True,
        "formal_state_modified": False,
        "rrd": str(args.output.expanduser().resolve()),
        "amplification_is_visual_only": True,
        "amplification_factor": float(args.amplification),
        "inputs": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "formal_pose_report": str(args.formal_pose_report.expanduser().resolve()),
            "v20_k5_pose_report": str(args.v20_k5_pose_report.expanduser().resolve()),
            "v20_k10_pose_report": str(args.v20_k10_pose_report.expanduser().resolve()),
            "observed_mesh": str(args.observed_mesh.expanduser().resolve()),
            "object_id": args.object_id,
            "sha256": {
                "annotations": sha256_file(args.annotations),
                "formal_pose_report": sha256_file(args.formal_pose_report),
                "v20_k5_pose_report": sha256_file(args.v20_k5_pose_report),
                "v20_k10_pose_report": sha256_file(args.v20_k10_pose_report),
                "observed_mesh": sha256_file(args.observed_mesh),
            },
        },
        "timeline": {
            "selected_frame_count": len(selected),
            "first_frame": selected[0],
            "last_frame": selected[-1],
            "metric_surface_frames": metric_frames,
            "no_metric_surface_frames": no_metric_frames,
            "pose_presence": pose_presence,
        },
        "delta_summary": {
            key: {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
                "p90": float(np.percentile(values, 90)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
            for key, values in delta_values.items()
        },
        "layers": {
            "formal": {"path": "/world/formal", "color": "red"},
            "v20_k5": {"path": "/world/v20_k5", "color": "green"},
            "v20_k10": {"path": "/world/v20_k10", "color": "blue"},
            "true_delta": {"path": "/world/delta/*/true_*", "color": "magenta", "metric": True},
            "amplified_delta": {"path": "/world/delta/*/amplified_*", "color": "orange", "metric": False, "note": "visual aid only"},
        },
        "coordinate_contract": {
            "object_formula": "V_world = V_canonical @ R_world_from_canonical.T + t_world",
            "delta_translation": "t_v20 - t_formal",
            "delta_rotation": "Log(R_v20 @ R_formal.T)",
            "generated_geometry_consumed": False,
        },
        "claim_scope": "Visualization only. The true delta layers show the difference between prediction-side formal and V20 diagnostic poses; amplified layers are explicitly non-metric. Generated SAM3D completion faces are not loaded.",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rrd": str(args.output), "report": str(report_path), "selected_frames": len(selected), "metric_surface_frames": len(metric_frames), "no_metric_surface_frames": no_metric_frames}, indent=2))


if __name__ == "__main__":
    main()
