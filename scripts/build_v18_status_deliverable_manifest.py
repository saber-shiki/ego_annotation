#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_status_deliverable_manifest"
CLAIM = (
    "This manifest closes a V18 status deliverable: full-duration 2D overlay, abstract world/status, "
    "side-by-side status videos, bounded state evidence, and visible-surface geometry evidence. It does not "
    "close final hidden object geometry, object pose, or physical contact requirements."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise RuntimeError(f"{label} must be numeric")
    return float(value)


def read_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    solution_path = args.solution_root / case / "v18_bounded_state_solution.json"
    overlay_qc_path = args.render_root / case / "v18_status_overlay_qc.json"
    world_qc_path = args.render_root / case / "v18_world_status_qc.json"
    side_qc_path = args.render_root / case / "v18_status_side_by_side_qc.json"
    visible_geometry_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    solution = require_dict(load_json(solution_path), f"{case} solution")
    overlay = require_dict(load_json(overlay_qc_path), f"{case} overlay qc")
    world = require_dict(load_json(world_qc_path), f"{case} world qc")
    side = require_dict(load_json(side_qc_path), f"{case} side qc")
    visible_geometry = require_dict(load_json(visible_geometry_path), f"{case} visible geometry")
    frame_count = require_int(annotation.get("frame_count"), "annotation frame_count")
    raw_frame_count = require_int(annotation.get("raw_frame_count"), "annotation raw_frame_count")
    raw_video = require_dict(annotation.get("raw_video"), "annotation raw_video")
    duration_s = require_float(annotation.get("duration_s"), "duration_s")
    qcs = {
        "annotation_state": bool(annotation.get("frame_count_match")),
        "bounded_state_solution": bool(solution.get("frame_count_match")),
        "status_overlay": bool(overlay.get("frame_count_match")),
        "world_status": bool(world.get("frame_count_match")),
        "side_by_side": bool(side.get("frame_count_match")),
    }
    if frame_count != raw_frame_count:
        raise RuntimeError(f"{case}: frame_count != raw_frame_count")
    for label, ok in qcs.items():
        if not ok:
            raise RuntimeError(f"{case}: {label} frame-count QC is false")
    overlay_frames = require_int(overlay.get("video_frame_count"), "overlay video_frame_count")
    world_frames = require_int(world.get("video_frame_count"), "world video_frame_count")
    side_frames = require_int(side.get("side_by_side_frame_count"), "side side_by_side_frame_count")
    if not (overlay_frames == world_frames == side_frames == frame_count):
        raise RuntimeError(f"{case}: output video frame counts do not all equal {frame_count}")
    render_elapsed_s = require_float(overlay.get("elapsed_s"), "overlay elapsed") + require_float(world.get("elapsed_s"), "world elapsed") + require_float(side.get("elapsed_s"), "side elapsed")
    return {
        "case": case,
        "raw_video": raw_video,
        "duration_s": duration_s,
        "frame_count": frame_count,
        "raw_frame_count": raw_frame_count,
        "frame_count_match": True,
        "status_outputs": {
            "annotation_state": str(annotation_path),
            "bounded_state_solution": str(solution_path),
            "status_overlay_video": overlay.get("output_video"),
            "world_status_video": world.get("output_video"),
            "side_by_side_status_video": side.get("output_video"),
            "status_overlay_qc": str(overlay_qc_path),
            "world_status_qc": str(world_qc_path),
            "side_by_side_qc": str(side_qc_path),
            "visible_geometry_archive_report": str(visible_geometry_path),
            "visible_geometry_archive_npz": visible_geometry.get("archive_npz"),
        },
        "frame_count_qc": {
            "annotation_state_frames": frame_count,
            "bounded_solution_frames": require_int(solution.get("frame_count"), "solution frame_count"),
            "overlay_video_frames": overlay_frames,
            "world_status_video_frames": world_frames,
            "side_by_side_video_frames": side_frames,
            "all_match_raw": True,
        },
        "status_runtime_qc": {
            "measured_render_elapsed_s": render_elapsed_s,
            "duration_s": duration_s,
            "measured_render_to_video_ratio": render_elapsed_s / duration_s if duration_s > 0 else None,
            "under_10x_realtime_for_status_render": render_elapsed_s <= 10.0 * duration_s,
        },
        "bounded_state_qc": {
            "hand_solution_state_counts": solution.get("hand_solution_state_counts"),
            "object_solution_state_counts": solution.get("object_solution_state_counts"),
            "contact_solution_state_counts": solution.get("contact_solution_state_counts"),
            "occlusion_solution_counts": solution.get("occlusion_solution_counts"),
            "contact_factor_ready_rows": solution.get("contact_factor_ready_rows"),
            "pose_filled_through_occlusion_rows": solution.get("pose_filled_through_occlusion_rows"),
        },
        "visible_geometry_qc": {
            "visible_geometry_archive_ready": visible_geometry.get("visible_geometry_archive_ready"),
            "surface_frame_rows": visible_geometry.get("surface_frame_rows"),
            "rejected_visible_object_frame_rows": visible_geometry.get("rejected_visible_object_frame_rows"),
            "total_vertices": visible_geometry.get("total_vertices"),
            "total_faces": visible_geometry.get("total_faces"),
            "v18_visible_geometry_status_counts": visible_geometry.get("v18_visible_geometry_status_counts"),
            "hidden_geometry_reconstructed": visible_geometry.get("hidden_geometry_reconstructed"),
            "canonical_mesh_ready": visible_geometry.get("canonical_mesh_ready"),
            "complete_object_pose_ready": visible_geometry.get("complete_object_pose_ready"),
        },
        "status_deliverable_ready": True,
        "final_pose_complete_deliverable_ready": False,
        **FALSE_READY,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [read_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    total_duration = sum(require_float(case.get("duration_s"), "case duration_s") for case in cases)
    total_render_elapsed = sum(require_float(require_dict(case.get("status_runtime_qc"), "runtime qc").get("measured_render_elapsed_s"), "render elapsed") for case in cases)
    visible_surface_rows = sum(
        require_int(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("surface_frame_rows"), "surface rows")
        for case in cases
    )
    visible_geometry_vertices = sum(
        require_int(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("total_vertices"), "vertices") for case in cases
    )
    visible_geometry_faces = sum(
        require_int(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("total_faces"), "faces") for case in cases
    )
    manifest = {
        "method": "build_v18_status_deliverable_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "build_elapsed_s": elapsed,
        "case_count": len(cases),
        "status_deliverable_ready": True,
        "final_pose_complete_deliverable_ready": False,
        "all_frame_counts_match_raw": all(bool(case.get("frame_count_match")) for case in cases),
        "all_status_renders_under_10x_realtime": all(
            bool(require_dict(case.get("status_runtime_qc"), "runtime qc").get("under_10x_realtime_for_status_render")) for case in cases
        ),
        "visible_geometry_archive_ready": all(
            bool(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("visible_geometry_archive_ready")) for case in cases
        ),
        "visible_geometry_surface_frame_rows": visible_surface_rows,
        "visible_geometry_vertices": visible_geometry_vertices,
        "visible_geometry_faces": visible_geometry_faces,
        "total_duration_s": total_duration,
        "total_measured_render_elapsed_s": total_render_elapsed,
        "total_measured_render_to_video_ratio": total_render_elapsed / total_duration if total_duration > 0 else None,
        "default_path_uses_bundlesdf_or_nerf": False,
        "contact_factor_ready_rows": sum(
            require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("contact_factor_ready_rows"), "contact ready") for case in cases
        ),
        "pose_filled_through_occlusion_rows": sum(
            require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("pose_filled_through_occlusion_rows"), "pose filled") for case in cases
        ),
        "cases": cases,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_status_deliverable_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--solution-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--render-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_renders"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_status_deliverable_manifest"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
