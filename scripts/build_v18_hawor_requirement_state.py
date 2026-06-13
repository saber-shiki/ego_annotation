#!/usr/bin/env python3
"""Build the V18 HaWoR hard-requirement state.

This artifact is intentionally HaWoR-only. It does not use WiLoR, HaMeR,
MANO2D, or depth probes as substitutes. It records which required V18 cases have
actual HaWoR MANO outputs and which are blocked by missing HaWoR execution assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

SIDES = ("left", "right")
EXPECTED_VERTICES = 778
EXPECTED_JOINTS = 21
EXPECTED_CASES = ("trash_1050", "task5_tomato_960")
DEFAULT_HAWOR_OUTPUTS = {
    "trash_1050": Path("/data2/ego_annotation_outputs/representative_trash/v3_hawor_world/hawor_world_hands.npz"),
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands.npz"),
}
EXPECTED_SOURCE_CLIP_SHA256 = {
    # Source identity for the task5 clip named in the HaWoR export contract.
    "task5_tomato_960": "66791eaa646aac2e8cb24bb00fe30b2801436302327b1c46fea650446c41c4ac",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path | None, hash_file: bool = False) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        st = path.stat()
        out.update({"bytes": st.st_size, "is_file": path.is_file(), "is_dir": path.is_dir(), "mtime_ns": st.st_mtime_ns})
        if hash_file and path.is_file():
            out["sha256"] = sha256(path)
    return out


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5.0)),
        "p95": float(np.percentile(vals, 95.0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def frame_count_for_case(case: str, v16_root: Path, v18_root: Path) -> int:
    for path in [
        v18_root / case / "annotations_v18_full.json",
        v16_root / case / "raw_frame_manifest" / "manifest.json",
    ]:
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            if isinstance(payload.get("frame_count"), int):
                return int(payload["frame_count"])
            video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
            if isinstance(video.get("frame_count"), int):
                return int(video["frame_count"])
            frames = payload.get("frames")
            if isinstance(frames, list):
                return len(frames)
    raise RuntimeError(f"cannot determine frame count for {case}")


def clip_for_case(case: str, v16_root: Path) -> str | None:
    path = v16_root / case / "raw_frame_manifest" / "manifest.json"
    if not path.exists():
        return None
    payload = load_json(path)
    clip = payload.get("clip") if isinstance(payload, dict) else None
    return str(clip) if isinstance(clip, str) else None


def validate_hawor_npz(path: Path, expected_frame_count: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    z = np.load(path)
    required = [
        "frame_idx",
        "R_c2w",
        "t_c2w",
        "img_focal",
        "video_path",
        "seq_folder",
    ]
    for side in SIDES:
        required += [
            f"{side}_vertices_world_m",
            f"{side}_joints_world_m",
            f"{side}_trans_world_m",
            f"{side}_root_orient_axis_angle",
            f"{side}_hand_pose_axis_angle",
            f"{side}_betas",
            f"{side}_valid",
            f"{side}_faces",
        ]
    missing = sorted([key for key in required if key not in z.files])
    if missing:
        return {"status": "invalid_hawor_npz_missing_arrays", "missing_arrays": missing}, {}
    frame_idx = np.asarray(z["frame_idx"], dtype=np.int32)
    arrays: dict[str, np.ndarray] = {key: np.asarray(z[key]) for key in z.files}
    failures: list[str] = []
    if frame_idx.shape != (expected_frame_count,):
        failures.append(f"frame_idx_shape_{frame_idx.shape}_expected_{expected_frame_count}")
    if len(set(int(x) for x in frame_idx.tolist())) != len(frame_idx):
        failures.append("duplicate_frame_idx")
    if len(frame_idx) and (int(np.min(frame_idx)) != 0 or int(np.max(frame_idx)) != expected_frame_count - 1):
        failures.append("frame_idx_not_zero_to_expected_minus_one")
    if np.asarray(z["R_c2w"]).shape != (expected_frame_count, 3, 3):
        failures.append("R_c2w_shape_mismatch")
    if np.asarray(z["t_c2w"]).shape != (expected_frame_count, 3):
        failures.append("t_c2w_shape_mismatch")
    side_reports: dict[str, Any] = {}
    for side in SIDES:
        vertices = np.asarray(z[f"{side}_vertices_world_m"])
        joints = np.asarray(z[f"{side}_joints_world_m"])
        valid = np.asarray(z[f"{side}_valid"]).astype(bool)
        pose = np.asarray(z[f"{side}_hand_pose_axis_angle"])
        betas = np.asarray(z[f"{side}_betas"])
        faces = np.asarray(z[f"{side}_faces"])
        if vertices.shape != (expected_frame_count, EXPECTED_VERTICES, 3):
            failures.append(f"{side}_vertices_shape_mismatch")
        if joints.shape != (expected_frame_count, EXPECTED_JOINTS, 3):
            failures.append(f"{side}_joints_shape_mismatch")
        if valid.shape != (expected_frame_count,):
            failures.append(f"{side}_valid_shape_mismatch")
        if pose.shape != (expected_frame_count, 45):
            failures.append(f"{side}_hand_pose_shape_mismatch")
        if betas.shape != (expected_frame_count, 10):
            failures.append(f"{side}_betas_shape_mismatch")
        if faces.ndim != 2 or faces.shape[1] != 3:
            failures.append(f"{side}_faces_shape_mismatch")
        finite_valid_vertices = bool(np.isfinite(vertices[valid]).all()) if vertices.shape[:1] == valid.shape else False
        finite_valid_joints = bool(np.isfinite(joints[valid]).all()) if joints.shape[:1] == valid.shape else False
        if not finite_valid_vertices:
            failures.append(f"{side}_valid_vertices_nonfinite")
        if not finite_valid_joints:
            failures.append(f"{side}_valid_joints_nonfinite")
        spans = []
        if joints.shape == (expected_frame_count, EXPECTED_JOINTS, 3) and valid.shape == (expected_frame_count,):
            # Same wrist-to-middle-tip diagnostic used elsewhere: useful scale evidence, not an acceptance gate.
            spans = np.linalg.norm(joints[valid, 12] - joints[valid, 0], axis=1).astype(float).tolist()
        side_reports[side] = {
            "valid_frames": int(np.count_nonzero(valid)) if valid.shape == (expected_frame_count,) else 0,
            "missing_or_invalid_frames": int(expected_frame_count - np.count_nonzero(valid)) if valid.shape == (expected_frame_count,) else expected_frame_count,
            "vertex_shape": list(vertices.shape),
            "joint_shape": list(joints.shape),
            "mano_pose_shape": list(pose.shape),
            "betas_shape": list(betas.shape),
            "faces_shape": list(faces.shape),
            "wrist_to_middle_tip_m": summarize(spans),
        }
    status = "hawor_full_timeline_npz_shape_valid" if not failures else "invalid_hawor_npz_shape_or_content"
    return {
        "status": status,
        "failures": failures,
        "frame_idx_count": int(len(frame_idx)),
        "frame_idx_min": int(np.min(frame_idx)) if len(frame_idx) else None,
        "frame_idx_max": int(np.max(frame_idx)) if len(frame_idx) else None,
        "img_focal": float(np.asarray(z["img_focal"]).reshape(-1)[0]) if np.asarray(z["img_focal"]).size else None,
        "video_path_recorded_in_npz": str(np.asarray(z["video_path"]).reshape(-1)[0]) if np.asarray(z["video_path"]).size else None,
        "seq_folder_recorded_in_npz": str(np.asarray(z["seq_folder"]).reshape(-1)[0]) if np.asarray(z["seq_folder"]).size else None,
        "sides": side_reports,
    }, arrays


def build_case(case: str, args: argparse.Namespace, provisioning: dict[str, Any]) -> dict[str, Any]:
    expected_frames = frame_count_for_case(case, args.v16_root, args.v18_root)
    local_clip = clip_for_case(case, args.v16_root)
    configured_path = DEFAULT_HAWOR_OUTPUTS.get(case)
    output_path = Path(str(configured_path)) if configured_path is not None else None
    bridge_path = args.output_root / "hawor_bridge_state" / case / "v18_hawor_bridge_state_report.json"
    bridge = load_json(bridge_path) if bridge_path.exists() else None
    report: dict[str, Any] = {
        "case": case,
        "expected_frame_count": expected_frames,
        "expected_frame_side_rows": expected_frames * 2,
        "hard_requirement": "HaWoR_full_timeline_metric_MANO_required_for_V18_physical_hand_state",
        "accepted_v18_hawor_requirement_met": False,
        "accepted_metric_hand_state_from_hawor": False,
        "hawor_output": file_info(output_path, hash_file=bool(args.hash_sources)),
        "local_raw_clip": local_clip,
        "current_v18_bridge_candidate": {
            "report_path": str(bridge_path),
            "exists": bridge_path.exists(),
            "status": bridge.get("status") if isinstance(bridge, dict) else None,
            "bridge_candidate_rows": bridge.get("bridge_candidate_rows") if isinstance(bridge, dict) else None,
            "accepted_v18_hawor_foundation": bridge.get("accepted_v18_hawor_foundation") if isinstance(bridge, dict) else None,
            "reference_projection_residual_px_median_per_row": bridge.get("reference_projection_residual_px_median_per_row") if isinstance(bridge, dict) else None,
            "blocking_reasons": bridge.get("blocking_reasons") if isinstance(bridge, dict) else None,
        },
        "claim_scope": "HaWoR_requirement_state_only_no_WiLoR_or_other_backend_substitution",
    }
    if output_path is None or not output_path.exists():
        report.update({
            "status": "blocked_no_case_hawor_output",
            "blocking_reasons": [
                "case_hawor_world_hands_npz_missing",
                "HaWoR_repo_weights_or_MANO_assets_missing_locally" if provisioning.get("status") == "blocked_missing_required_hawor_assets" else "HaWoR_execution_not_validated_for_case",
            ],
            "available_hawor_frame_side_rows": 0,
            "full_timeline_hawor_npz_shape_valid": False,
        })
        return report
    qc_path = output_path.parent / "qc_hawor_world_hands.json"
    qc = load_json(qc_path) if qc_path.exists() else None
    npz_report, _arrays = validate_hawor_npz(output_path, expected_frames)
    side_valid = npz_report.get("sides", {}) if isinstance(npz_report.get("sides"), dict) else {}
    available_rows = sum(int(side_valid.get(side, {}).get("valid_frames", 0)) for side in SIDES)
    full_shape_valid = npz_report.get("status") == "hawor_full_timeline_npz_shape_valid"
    full_valid_rows = available_rows == expected_frames * 2
    blockers: list[str] = []
    if not full_shape_valid:
        blockers.append("hawor_npz_shape_or_content_invalid")
    if not full_valid_rows:
        blockers.append("hawor_valid_rows_do_not_cover_all_frame_sides")
    # Even when a HaWoR NPZ exists, current V18 cannot accept it blindly. A bridge report can reduce
    # uncertainty about the coordinate path, but it is still candidate-only until residual tails are explained
    # and downstream contact/occlusion/nonpenetration are recomputed from the HaWoR state.
    expected_clip_sha256 = EXPECTED_SOURCE_CLIP_SHA256.get(case)
    qc_video_sha256 = qc.get("video_sha256") if isinstance(qc, dict) else None
    qc_video_sha256_matches_expected = bool(expected_clip_sha256 and qc_video_sha256 == expected_clip_sha256)
    if expected_clip_sha256 and not qc_video_sha256_matches_expected:
        blockers.append("hawor_qc_video_sha256_missing_or_mismatch_for_expected_case_clip")
    if isinstance(bridge, dict) and bridge.get("bridge_candidate_rows"):
        blockers.append("HaWoR_current_V18_bridge_candidate_built_not_foundation_accepted")
        bridge_blockers = bridge.get("blocking_reasons") if isinstance(bridge.get("blocking_reasons"), list) else []
        if "projection_residual_tail_too_large_for_foundation_acceptance" in bridge_blockers:
            blockers.append("HaWoR_bridge_projection_residual_tail_blocks_foundation_acceptance")
        if "single_global_HaWoR_to_V18_world_sim3_alignment_too_loose_for_physical_contact" in bridge_blockers:
            blockers.append("single_global_HaWoR_to_V18_world_sim3_alignment_too_loose_for_physical_contact")
    else:
        blockers.append("HaWoR_coordinate_bridge_to_current_V18_world_not_residual_checked_for_full_pipeline")
    blockers.append("contact_occlusion_nonpenetration_not_recomputed_from_HaWoR_full_timeline_state")
    report.update({
        "status": "hawor_output_available_but_not_accepted_v18_foundation" if full_shape_valid else "hawor_output_present_but_invalid",
        "qc_report": file_info(qc_path, hash_file=bool(args.hash_sources)),
        "qc_status": qc.get("status") if isinstance(qc, dict) else None,
        "qc_valid_hand_frames": qc.get("valid_hand_frames") if isinstance(qc, dict) else None,
        "expected_source_clip_sha256": expected_clip_sha256,
        "qc_video_sha256": qc_video_sha256,
        "qc_video_sha256_matches_expected": qc_video_sha256_matches_expected if expected_clip_sha256 else None,
        "npz_validation": npz_report,
        "available_hawor_frame_side_rows": available_rows,
        "full_timeline_hawor_npz_shape_valid": full_shape_valid,
        "full_timeline_hawor_valid_rows": full_valid_rows,
        "recorded_remote_video_path": npz_report.get("video_path_recorded_in_npz"),
        "local_clip_identity_note": "remote_path_recorded_in_old_HaWoR_npz; local V16 clip path used for expected frame count",
        "blocking_reasons": blockers,
    })
    return report


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V18 HaWoR hard-requirement state",
        "",
        "This artifact is HaWoR-only. It does not use WiLoR, HaMeR, MANO2D, depth probes, validators, or rendered audits as substitutes for the HaWoR requirement.",
        "",
        f"Status: `{summary['status']}`",
        f"All cases HaWoR requirement met: `{summary['all_cases_hawor_requirement_met']}`",
        f"V18 physical hand state valid from HaWoR: `{summary['v18_physical_hand_state_valid_from_hawor']}`",
        "",
        "## Provisioning",
        "",
        f"Provisioning status: `{summary['provisioning_status']}`",
        f"Missing required files: `{summary['missing_required']}`",
        "",
    ]
    for case in summary["cases"]:
        lines += [
            f"## {case['case']}",
            "",
            f"Status: `{case['status']}`",
            f"HaWoR output: `{case['hawor_output'].get('path')}` exists=`{case['hawor_output'].get('exists')}`",
            f"Available HaWoR frame-side rows: `{case.get('available_hawor_frame_side_rows')}/{case.get('expected_frame_side_rows')}`",
            f"Full-timeline NPZ shape valid: `{case.get('full_timeline_hawor_npz_shape_valid')}`",
            f"Accepted V18 HaWoR requirement met: `{case.get('accepted_v18_hawor_requirement_met')}`",
            f"Blocking reasons: `{case.get('blocking_reasons')}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    provisioning_path = args.output_root / "hawor_provisioning_audit" / "v18_hawor_provisioning_audit_report.json"
    provisioning = load_json(provisioning_path) if provisioning_path.exists() else {"status": "missing_hawor_provisioning_audit"}
    cases = [build_case(case, args, provisioning) for case in args.cases]
    all_met = all(case.get("accepted_v18_hawor_requirement_met") is True for case in cases)
    summary = {
        "method": "build_v18_hawor_requirement_state",
        "status": "blocked_hawor_hard_requirement_not_met" if not all_met else "hawor_hard_requirement_met_not_downstream_validated",
        "claim_scope": "HaWoR_hard_requirement_state_no_model_substitution_no_full_V18_closure",
        "output_root": str(args.output_root),
        "provisioning_report": str(provisioning_path),
        "provisioning_status": provisioning.get("status"),
        "missing_required": provisioning.get("missing_required"),
        "all_cases_hawor_requirement_met": all_met,
        "v18_physical_hand_state_valid_from_hawor": False,
        "cases": cases,
        "blocking_reasons": sorted({reason for case in cases for reason in case.get("blocking_reasons", []) if isinstance(reason, str)}),
        "elapsed_s": time.perf_counter() - start,
    }
    out_dir = args.output_root / "hawor_requirement_state"
    write_json(out_dir / "v18_hawor_requirement_state.json", summary)
    write_markdown(out_dir / "V18_HAWOR_REQUIREMENT_STATE.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--v18-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=list(EXPECTED_CASES))
    parser.add_argument("--hash-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
