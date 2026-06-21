#!/usr/bin/env python3
"""Assemble the current V18 interval-MANO frontier artifact.

This is not a new physical factor.  It is the artifact-consumption step for the
current workbench state: expose the full-video interval MANO renders and backing
solver states that actually drive the present hand annotation, instead of the
older sparse H-prime final roots that have been ruled out for the MANO objective.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_OUTPUT_ROOT = Path("/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v1")
DEFAULT_CASE_RENDER_ROOTS = {
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_task5_joint_mano_surface_support_uncertain_full_video_v1/task5_tomato_960"),
    "trash_1050": Path("/data2/ego_annotation_outputs/v18_trash_joint_mano_latent_coherent_transition_v2_contract_repro_full_video_v1/trash_1050"),
}
DEFAULT_REVIEW_FRAMES = {
    "task5_tomato_960": [481, 499, 525, 690, 720, 780, 902],
    "trash_1050": [958, 970, 972, 982, 988, 999, 1000, 1002],
}
CASE_CLAIMS = {
    "task5_tomato_960": {
        "frontier_mechanism": "support_bounded_interval_mano",
        "claim": "Tomato exact-surface/contact residuals are mostly below independent object-support uncertainty; the artifact should show support-bounded MANO uncertainty rather than confident millimetre correction.",
        "not_claimed": [
            "solved contact state",
            "accepted hidden tomato volume nonpenetration",
            "millimetre-accurate object pose",
            "task5 V18 closure",
        ],
    },
    "trash_1050": {
        "frontier_mechanism": "latent_occlusion_transition_interval_mano",
        "claim": "Late trash left-hand MANO observations become invalid when the hand transitions under the lid; the artifact should show bounded latent occluded-hand hypotheses driven by visible first-surface/observation-validity factors.",
        "not_claimed": [
            "known hidden-hand pose",
            "solved hand-lid contact",
            "accepted compact-lid hidden-volume nonpenetration",
            "trash V18 closure",
        ],
    },
}
EXPECTED_VIDEO_NAMES = {
    "overlay": "v18_overlay_joint_mano_full_video_correction.mp4",
    "world": "v18_world_joint_mano_full_video_correction.mp4",
    "side_by_side": "v18_side_by_side_joint_mano_full_video_correction.mp4",
}
STANDARD_VIDEO_NAMES = {
    "overlay": "v18_overlay.mp4",
    "world": "v18_world.mp4",
    "side_by_side": "v18_side_by_side.mp4",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def parse_case_render_roots(items: list[str]) -> dict[str, Path]:
    if not items:
        return dict(DEFAULT_CASE_RENDER_ROOTS)
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--case-render-root must be CASE=PATH, got {item!r}")
        case, raw = item.split("=", 1)
        case = case.strip()
        if not case:
            raise ValueError(f"empty case in {item!r}")
        out[case] = Path(raw).expanduser()
    return out


def parse_review_frames(items: list[str]) -> dict[str, list[int]]:
    out = {k: list(v) for k, v in DEFAULT_REVIEW_FRAMES.items()}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--review-frames must be CASE=f0,f1,..., got {item!r}")
        case, raw = item.split("=", 1)
        frames = [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]
        out[case.strip()] = frames
    return out


def copy_or_hardlink(src: Path, dst: Path, *, prefer_hardlink: bool = False) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    method = "copy"
    if prefer_hardlink:
        try:
            os.link(src, dst)
            method = "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            method = "copy"
    else:
        shutil.copy2(src, dst)
    return {"source": str(src), "path": str(dst), "method": method, "bytes": int(dst.stat().st_size)}


def ffprobe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = json.loads(proc.stdout)
        stream = (raw.get("streams") or [{}])[0]
        return {
            "width": int(stream.get("width")) if stream.get("width") is not None else None,
            "height": int(stream.get("height")) if stream.get("height") is not None else None,
            "nb_frames": int(stream.get("nb_frames")) if str(stream.get("nb_frames") or "").isdigit() else stream.get("nb_frames"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "duration_s": float(stream.get("duration")) if stream.get("duration") is not None else None,
        }
    except Exception as exc:  # ffprobe is evidence, not the artifact source.
        return {"ffprobe_error": repr(exc)}


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


def norm3(raw: Any) -> float | None:
    try:
        arr = np.asarray(raw, dtype=float)
    except Exception:
        return None
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return float(np.linalg.norm(arr))


def optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def summarize_states(case: str, state_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for path in state_paths:
        state = load_json(path)
        rows = state.get("per_frame_states")
        if not isinstance(rows, list):
            raise ValueError(f"{path} has no per_frame_states list")
        summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
        interval = {
            "state_path": str(path),
            "method": state.get("method"),
            "case": state.get("case"),
            "object_id": state.get("object_id"),
            "row_count": len(rows),
            "summary": summary,
            "claim_scope": state.get("claim_scope"),
            "scientific_test": state.get("scientific_test"),
        }
        intervals.append(interval)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains non-dict per_frame_state")
            out = dict(row)
            out["source_interval_state_path"] = str(path)
            merged.append(out)
    frame_ids = sorted({int(r["frame_idx"]) for r in merged if "frame_idx" in r})
    sides = sorted({str(r.get("hand_side")) for r in merged if r.get("hand_side") is not None})
    translation_norms = [v for r in merged if (v := norm3(r.get("optimized_translation_world_m"))) is not None]
    ray_shift_norms = [v for r in merged if (v := norm3(r.get("hand_ray_shift_prior_translation_world_m"))) is not None]
    visible_surface_initial = [float(r.get("visible_surface_depth_order_selected_initial_in_front_count") or 0.0) for r in merged]
    visible_surface_final = [float(r.get("visible_surface_depth_order_selected_final_in_front_count") or 0.0) for r in merged]
    support_uncertainty = [float(r.get("observed_surface_support_uncertainty_m") or 0.0) for r in merged]
    contact_rows = [r for r in merged if r.get("contact_patch_factor_state") == "active_contact_patch"]
    hand_obs_zeroed = [
        r for r in merged
        if (optional_float(r.get("hand_observation_visibility_weight_multiplier")) is not None)
        and (optional_float(r.get("hand_observation_visibility_weight_multiplier")) <= 1e-9)
    ]
    summary = {
        "case": case,
        "optimized_state_count": int(len(merged)),
        "unique_optimized_frame_count": int(len(frame_ids)),
        "first_optimized_frame": int(frame_ids[0]) if frame_ids else None,
        "last_optimized_frame": int(frame_ids[-1]) if frame_ids else None,
        "hand_sides": sides,
        "interval_count": int(len(intervals)),
        "intervals": intervals,
        "optimized_translation_norm_m": summarize_numeric(translation_norms),
        "hand_ray_shift_prior_norm_m": summarize_numeric(ray_shift_norms),
        "visible_surface_depth_order_initial_in_front_sum": int(sum(visible_surface_initial)),
        "visible_surface_depth_order_final_in_front_sum": int(sum(visible_surface_final)),
        "observed_surface_support_uncertainty_m": summarize_numeric(support_uncertainty),
        "active_contact_patch_state_count": int(len(contact_rows)),
        "zero_weight_hand_observation_state_count": int(len(hand_obs_zeroed)),
    }
    return merged, summary


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_idx} from {video_path}")
    return frame


def cell_with_label(img: np.ndarray, label: str, cell_w: int, cell_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(cell_w / max(1, w), (cell_h - 24) / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
    x0 = (cell_w - nw) // 2
    y0 = 22 + (cell_h - 24 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    cv2.putText(canvas, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_review_sheet(case: str, case_dir: Path, frames: list[int]) -> dict[str, Any]:
    views = ["overlay", "world", "side_by_side"]
    video_paths = {view: case_dir / STANDARD_VIDEO_NAMES[view] for view in views}
    cell_w, cell_h = 300, 210
    rows: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    for frame_idx in frames:
        cells: list[np.ndarray] = []
        for view in views:
            try:
                img = read_frame(video_paths[view], frame_idx)
                cell = cell_with_label(img, f"{case} f{frame_idx} {view}", cell_w, cell_h)
            except Exception as exc:
                failures.append({"frame_idx": int(frame_idx), "view": view, "error": repr(exc)})
                cell = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)
                cv2.putText(cell, f"missing f{frame_idx} {view}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0) if rows else np.zeros((1, 1, 3), dtype=np.uint8)
    out_path = case_dir / "current_frontier_interval_mano_review.jpg"
    ok = cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"failed to write {out_path}")
    return {"path": str(out_path), "frames": [int(x) for x in frames], "views": views, "failures": failures}


def build_case(case: str, render_root: Path, output_root: Path, review_frames: list[int], *, prefer_hardlink: bool) -> dict[str, Any]:
    if not render_root.exists():
        raise FileNotFoundError(render_root)
    render_manifest_path = render_root / "v18_joint_mano_interval_correction_render_manifest.json"
    render_manifest = load_json(render_manifest_path)
    state_paths = [Path(p) for p in render_manifest.get("state_paths") or []]
    if not state_paths:
        raise ValueError(f"{render_manifest_path} has no state_paths")
    for state_path in state_paths:
        if not state_path.exists():
            raise FileNotFoundError(state_path)

    case_dir = output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    linked_videos: dict[str, Any] = {}
    video_probe: dict[str, Any] = {}
    for view, src_name in EXPECTED_VIDEO_NAMES.items():
        src = render_root / src_name
        dst = case_dir / STANDARD_VIDEO_NAMES[view]
        linked_videos[view] = copy_or_hardlink(src, dst, prefer_hardlink=prefer_hardlink)
        video_probe[view] = ffprobe_video(dst)

    linked_manifest = copy_or_hardlink(render_manifest_path, case_dir / "source_render_manifest.json", prefer_hardlink=prefer_hardlink)
    merged_rows, state_summary = summarize_states(case, state_paths)
    frame_count = render_manifest.get("frame_count")
    try:
        frame_count_int = int(frame_count)
    except (TypeError, ValueError):
        frame_count_int = None
    optimized_unique_count = int(state_summary.get("unique_optimized_frame_count", 0) or 0)
    state_summary["full_video_frame_count"] = frame_count_int
    state_summary["context_passthrough_frame_count"] = (
        max(0, frame_count_int - optimized_unique_count) if frame_count_int is not None else None
    )
    state_summary["frame_policy"] = (
        "Frames with interval solver states are rendered from optimized MANO variables; "
        "frames without interval solver states are full-video context/passthrough frames and do not claim a new MANO correction."
    )
    backing_path = case_dir / "frontier_interval_mano_states.json"
    write_json(backing_path, {
        "method": "build_v18_current_frontier_interval_artifact.merge_interval_mano_states",
        "case": case,
        "source_render_manifest": str(render_manifest_path),
        "frontier_claim_scope": CASE_CLAIMS.get(case, {}),
        "state_summary": state_summary,
        "per_frame_states": merged_rows,
    })
    review = make_review_sheet(case, case_dir, review_frames)
    return {
        "case": case,
        "render_root_source": str(render_root),
        "case_output_root": str(case_dir),
        "frontier_claim_scope": CASE_CLAIMS.get(case, {}),
        "videos": linked_videos,
        "video_probe": video_probe,
        "source_render_manifest": linked_manifest,
        "backing_interval_mano_states": str(backing_path),
        "state_summary": state_summary,
        "review_sheet": review,
        "frame_count_from_source_render_manifest": frame_count,
        "full_video_from_source_render_manifest": render_manifest.get("full_video"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--case-render-root", action="append", default=[], help="CASE=render_root. Defaults to current task5/trash frontier roots.")
    ap.add_argument("--review-frames", action="append", default=[], help="CASE=f0,f1,... for review sheet frames.")
    ap.add_argument("--hardlink-existing-files", action="store_true", help="Use hardlinks for existing videos/manifests. Default copies to freeze the artifact against later in-place source overwrites.")
    args = ap.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    case_roots = parse_case_render_roots(args.case_render_root)
    review_frames = parse_review_frames(args.review_frames)

    cases: dict[str, Any] = {}
    for case, root in case_roots.items():
        cases[case] = build_case(case, root, output_root, review_frames.get(case, []), prefer_hardlink=bool(args.hardlink_existing_files))

    artifact_manifest = {
        "method": "build_v18_current_frontier_interval_artifact",
        "purpose": "Expose the current full-video interval-MANO frontier artifacts and backing optimized MANO states; avoid presenting older sparse H-prime roots as the V18 MANO answer.",
        "output_root": str(output_root),
        "claim_scope": {
            "primary_deliverable": "full-video rendered metric MANO trajectory artifact with bounded/latent physical uncertainty",
            "not_claimed": [
                "V18 closure",
                "solved contact",
                "solved object pose",
                "solved nonpenetration",
                "known hidden-hand pose through occlusion",
            ],
            "ruled_out_as_source": "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard",
        },
        "cases": cases,
    }
    write_json(output_root / "v18_current_frontier_interval_mano_artifact_manifest.json", artifact_manifest)
    print(json.dumps({
        "status": "ok",
        "output_root": str(output_root),
        "cases": {case: {
            "optimized_state_count": data["state_summary"]["optimized_state_count"],
            "unique_optimized_frame_count": data["state_summary"]["unique_optimized_frame_count"],
            "review_sheet": data["review_sheet"]["path"],
        } for case, data in cases.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
