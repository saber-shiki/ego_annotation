#!/usr/bin/env python3
"""Run and validate the exact HOT3D D18 SAM3D/TRELLIS render commands.

This wrapper removes command-name ambiguity from the runtime agent.  It consumes
only the already validated D17 layered states, calls the one bundled renderer
for both branches, verifies 150-frame/30-FPS outputs and visible consumption of
the shared P18b uncertain surface samples, and writes a shared D18 report. It
never changes camera, metric MANO, object pose, or branch geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

SCHEMA = "v19_hot3d_d18_dual_backend_render_runner_v1"
VIDEO_KEYS = ("overlay_video", "world_video", "side_world_video", "side_by_side_video")


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {description}: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    path = require_file(path, "JSON input")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_info(path: Path) -> dict[str, Any]:
    path = require_file(path, "D18 rendered video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open D18 rendered video: {path}")
    try:
        return {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def validate_conditional_warning_rows(
    manifest: dict[str, Any], expected_frames: list[int]
) -> list[int]:
    actual_frames = [
        int(value) for value in manifest.get("conditional_rotation_tail_frames") or []
    ]
    if actual_frames != expected_frames:
        raise RuntimeError(
            "D18 conditional rotation-tail warning frames mismatch: "
            f"{actual_frames} != {expected_frames}"
        )
    rows = manifest.get("frame_rows") if isinstance(manifest.get("frame_rows"), list) else []
    warned_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("conditional_rotation_step_uncertainty"), dict)
    ]
    warned_row_frames = [int(row["source_frame_idx"]) for row in warned_rows]
    if warned_row_frames != expected_frames:
        raise RuntimeError(
            "D18 frame rows do not preserve exact conditional warning payloads: "
            f"{warned_row_frames} != {expected_frames}"
        )
    for row in warned_rows:
        payload = row["conditional_rotation_step_uncertainty"]
        if payload.get("acceptance_mode") != "conditional_sparse_underobservable_rotation_tail":
            raise RuntimeError("D18 frame row has malformed conditional warning mode")
        if payload.get("trajectory_values_modified_or_clipped") is not False:
            raise RuntimeError("D18 conditional warning row reports pose clipping")
        if payload.get("generated_geometry_pose_evidence_consumed") is not False:
            raise RuntimeError("D18 conditional warning row consumed generated pose evidence")
    return actual_frames


def validate_render_manifest(
    path: Path,
    *,
    expected_frame_count: int,
    expected_fps: float,
    expected_source_model: str,
    expected_conditional_rotation_tail_frames: list[int],
) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("status") != "ok":
        raise RuntimeError(f"D18 renderer manifest is not ok: {path}")
    if int(manifest.get("frame_count", -1)) != int(expected_frame_count):
        raise RuntimeError(
            f"D18 manifest frame count mismatch: {manifest.get('frame_count')} != {expected_frame_count}"
        )
    branch = manifest.get("branch") if isinstance(manifest.get("branch"), dict) else {}
    if str(branch.get("source_model")) != expected_source_model:
        raise RuntimeError(
            f"D18 source model mismatch: {branch.get('source_model')} != {expected_source_model}"
        )
    actual_conditional_frames = validate_conditional_warning_rows(
        manifest, expected_conditional_rotation_tail_frames
    )
    shared_consumption = (
        manifest.get("shared_state_consumption")
        if isinstance(manifest.get("shared_state_consumption"), dict)
        else {}
    )
    temporal = (
        manifest.get("shared_p18b_temporal_surface")
        if isinstance(manifest.get("shared_p18b_temporal_surface"), dict)
        else {}
    )
    if int(temporal.get("row_count", 0)) <= 0:
        raise RuntimeError(f"D18 renderer did not consume shared P18b rows: {path}")
    if not str(temporal.get("value_sha256") or ""):
        raise RuntimeError(f"D18 manifest lacks shared P18b value hash: {path}")
    if int(temporal.get("surface_point_count", 0)) > 0:
        if int(temporal.get("rendered_frame_count_with_input_points", 0)) <= 0:
            raise RuntimeError(f"D18 P18b samples were present but no frame consumed them: {path}")
        if int(temporal.get("rendered_input_point_count", 0)) <= 0:
            raise RuntimeError(f"D18 P18b samples were present but renderer input stayed empty: {path}")
        for view_key in (
            "rendered_overlay_point_count",
            "rendered_world_point_count",
            "rendered_side_world_point_count",
        ):
            if int(temporal.get(view_key, 0)) <= 0:
                raise RuntimeError(
                    f"D18 P18b samples did not change the {view_key} view: {path}"
                )
    for key in (
        "generated_faces_collision_eligible",
        "generated_faces_contact_eligible",
    ):
        if shared_consumption.get(key) is not False:
            raise RuntimeError(f"D18 manifest lacks explicit {key}=false: {path}")
    signed_geometry_ready = shared_consumption.get("signed_geometry_ready") is True
    accepted_signed_rows = int(temporal.get("accepted_signed_full_mano_row_count", 0))
    if accepted_signed_rows not in (0, int(expected_frame_count) * 2):
        raise RuntimeError(
            f"D18 partially rendered accepted full MANO on {accepted_signed_rows}/{expected_frame_count * 2} rows"
        )
    if not signed_geometry_ready and accepted_signed_rows:
        raise RuntimeError("D18 unsigned run rendered signed-accepted full MANO rows")
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    videos: dict[str, Any] = {}
    for key in VIDEO_KEYS:
        video = video_info(Path(str(outputs.get(key) or "")))
        if video["frame_count"] != int(expected_frame_count):
            raise RuntimeError(
                f"D18 {key} frame count mismatch: {video['frame_count']} != {expected_frame_count}"
            )
        if abs(float(video["fps"]) - float(expected_fps)) > 1.0e-3:
            raise RuntimeError(
                f"D18 {key} FPS mismatch: {video['fps']} != {expected_fps}"
            )
        videos[key] = video
    review = require_file(Path(str(outputs.get("frame_review") or "")), "D18 anchor review PNG")
    return {
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256_file(path.resolve()),
        "branch": branch,
        "frame_count": int(manifest["frame_count"]),
        "videos": videos,
        "anchor_review": str(review),
        "anchor_review_sha256": sha256_file(review),
        "generated_faces_collision_eligible": shared_consumption[
            "generated_faces_collision_eligible"
        ],
        "generated_faces_contact_eligible": shared_consumption[
            "generated_faces_contact_eligible"
        ],
        "signed_geometry_ready": signed_geometry_ready,
        "accepted_signed_full_mano_row_count": accepted_signed_rows,
        "signed_full_mano_accepted": accepted_signed_rows == int(expected_frame_count) * 2,
        "conditional_rotation_tail_frames": actual_conditional_frames,
        "shared_p18b_temporal_surface": temporal,
    }


def run_renderer(
    *,
    renderer: Path,
    render_state: Path,
    output_dir: Path,
    anchor_frame: int,
    fps: float,
    generated_face_budget: int,
    replace: bool,
) -> None:
    command = [
        sys.executable,
        str(renderer),
        "--render-state",
        str(render_state),
        "--output-dir",
        str(output_dir),
        "--generated-face-budget",
        str(int(generated_face_budget)),
        "--observed-face-budget",
        "0",
        "--mano-face-budget",
        "0",
        "--export-glb-frame",
        str(int(anchor_frame)),
        "--fps",
        str(float(fps)),
    ]
    if replace:
        command.append("--replace")
    print(json.dumps({"event": "d18_renderer_launch", "command": command}), flush=True)
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise RuntimeError(f"missing run root: {run_root}")
    if not (0 <= int(args.anchor_frame) < int(args.expected_frame_count)):
        raise RuntimeError(f"anchor frame outside timeline: {args.anchor_frame}")

    bundle_root = Path(__file__).resolve().parents[1]
    renderer = require_file(
        bundle_root
        / "experiments/sam3d_p11_p12_branch/render_p14_p15_layered_state.py",
        "bundled D18 renderer",
    )
    exp_root = run_root / "experiments/sam3d_trellis_controlled"
    adapter_path = require_file(
        exp_root / "P15_layered_states/p14_p15_layered_render_state_adapter_report.json",
        "D17 state adapter report",
    )
    adapter = load_json(adapter_path)
    if adapter.get("status") != "ok":
        raise RuntimeError("D17 state adapter report is not ok")
    branch_policy = (
        adapter.get("branch_variable_policy")
        if isinstance(adapter.get("branch_variable_policy"), dict)
        else {}
    )
    required_identical = {
        "annotation_backbone",
        "object_pose_trajectory",
        "mano_constraint_state",
        "temporal_mano_state",
        "hidden_volume_validation",
        "projection_contract",
    }
    declared_identical = {
        str(value) for value in branch_policy.get("must_be_identical") or []
    }
    if not required_identical.issubset(declared_identical):
        raise RuntimeError(
            f"D17 adapter lacks shared-state equality policy: {sorted(required_identical - declared_identical)}"
        )
    if branch_policy.get("inherited_geometry_dependent_constraint_payload_rendered") is not False:
        raise RuntimeError("D17 adapter renders a geometry-dependent inherited constraint payload")
    if branch_policy.get("shared_prebranch_temporal_mano_surface_hypothesis_rendered") is not True:
        raise RuntimeError("D17 adapter does not route the shared pre-branch P18b state to D18")
    if branch_policy.get("contact_or_collision_recomputed") is not False:
        raise RuntimeError("D17 adapter recomputed branch-dependent contact/collision")
    if len(adapter.get("shared_state_value_sha256") or {}) != len(required_identical):
        raise RuntimeError("D17 adapter lacks all six byte-bound shared-state hashes")
    source_validation = (
        adapter.get("source_validation")
        if isinstance(adapter.get("source_validation"), dict)
        else {}
    )
    temporal_readiness = (
        source_validation.get("temporal_readiness")
        if isinstance(source_validation.get("temporal_readiness"), dict)
        else {}
    )
    rotation_step_gate = (
        temporal_readiness.get("rotation_step_gate")
        if isinstance(temporal_readiness.get("rotation_step_gate"), dict)
        else {}
    )
    if rotation_step_gate.get("gate_passed") is not True:
        raise RuntimeError("D17 adapter lacks an explicit passing rotation-step gate")
    conditional_applied = rotation_step_gate.get("conditional_tier_applied") is True
    if conditional_applied:
        if rotation_step_gate.get("acceptance_mode") != "conditional_sparse_underobservable_rotation_tail":
            raise RuntimeError("D17 adapter has malformed conditional rotation-tail mode")
        if rotation_step_gate.get("trajectory_values_modified_or_clipped") is not False:
            raise RuntimeError("D17 conditional trajectory was modified or clipped")
        if rotation_step_gate.get("generated_geometry_pose_evidence_consumed") is not False:
            raise RuntimeError("D17 conditional trajectory consumed generated pose evidence")
    expected_conditional_frames = [
        int(row["to_frame_idx"])
        for row in rotation_step_gate.get("conditional_transitions") or []
    ] if conditional_applied else []
    if len(expected_conditional_frames) != len(set(expected_conditional_frames)):
        raise RuntimeError("D17 conditional rotation-tail target frames are duplicated")

    branches = {
        "sam3d": {
            "source_model": "sam3d_objects",
            "state": exp_root
            / "P15_layered_states/sam3d_owned_dual_mesh/experimental_layered_render_state.json",
            "output": exp_root / "renders/sam3d",
        },
        "trellis": {
            "source_model": "trellis",
            "state": exp_root
            / "P15_layered_states/trellis_frozen_legacy_cut/experimental_layered_render_state.json",
            "output": exp_root / "renders/trellis",
        },
    }
    results: dict[str, Any] = {}
    for name, row in branches.items():
        state = require_file(Path(row["state"]), f"D17 {name} layered state")
        output = Path(row["output"])
        run_renderer(
            renderer=renderer,
            render_state=state,
            output_dir=output,
            anchor_frame=int(args.anchor_frame),
            fps=float(args.expected_fps),
            generated_face_budget=int(args.generated_face_budget),
            replace=bool(args.replace),
        )
        manifest_path = output / "p14_p15_layered_full_mano_render_manifest.json"
        results[name] = validate_render_manifest(
            manifest_path,
            expected_frame_count=int(args.expected_frame_count),
            expected_fps=float(args.expected_fps),
            expected_source_model=str(row["source_model"]),
            expected_conditional_rotation_tail_frames=expected_conditional_frames,
        )
        if results[name]["generated_faces_collision_eligible"]:
            raise RuntimeError(f"D18 {name} manifest promoted generated collision geometry")
        if results[name]["generated_faces_contact_eligible"]:
            raise RuntimeError(f"D18 {name} manifest promoted generated contact geometry")

    temporal_hashes = {
        str(row["shared_p18b_temporal_surface"].get("value_sha256"))
        for row in results.values()
    }
    expected_temporal_hash = (adapter.get("shared_state_value_sha256") or {}).get(
        "temporal_mano_state"
    )
    # Adapter hashes the complete temporal_mano_state block while renderers hash
    # its payload. Equality between branches is mandatory; both hash scopes are
    # retained explicitly rather than compared as if they represented one value.
    if len(temporal_hashes) != 1 or "" in temporal_hashes or "None" in temporal_hashes:
        raise RuntimeError(
            f"D18 branches consumed different/empty P18b payload hashes: {sorted(temporal_hashes)}"
        )
    if not expected_temporal_hash:
        raise RuntimeError("D17 adapter lacks temporal_mano_state shared-block hash")

    report = {
        "schema": SCHEMA,
        "status": "ok_dual_backend_full_timeline_renders",
        "claim_scope": (
            "Deterministic execution/validation of the two D18 render branches. Camera, MANO, "
            "observed-only pose, and physical surface come unchanged from D17; generated faces remain render-only."
        ),
        "run_root": str(run_root),
        "renderer": str(renderer),
        "renderer_sha256": sha256_file(renderer),
        "state_adapter_report": str(adapter_path),
        "state_adapter_report_sha256": sha256_file(adapter_path),
        "anchor_frame": int(args.anchor_frame),
        "expected_frame_count": int(args.expected_frame_count),
        "expected_fps": float(args.expected_fps),
        "geometry_is_sole_branch_variable": True,
        "shared_p18b_temporal_state": {
            "branch_payload_value_sha256": next(iter(temporal_hashes)),
            "d17_temporal_block_value_sha256": expected_temporal_hash,
            "both_branches_identical": True,
            "rendered_as_uncertain_surface_only": not all(
                bool(row.get("signed_geometry_ready")) for row in results.values()
            ),
            "signed_geometry_ready": all(
                bool(row.get("signed_geometry_ready")) for row in results.values()
            ),
            "accepted_signed_full_mano_row_count": min(
                int(row.get("accepted_signed_full_mano_row_count", 0))
                for row in results.values()
            ),
            "signed_full_mano_accepted": all(
                bool(row.get("signed_full_mano_accepted")) for row in results.values()
            ),
        },
        "rotation_step_acceptance_mode": rotation_step_gate.get("acceptance_mode"),
        "conditional_rotation_tail_frames": expected_conditional_frames,
        "conditional_temporal_uncertainty": temporal_readiness.get(
            "conditional_temporal_uncertainty"
        ),
        "branches": results,
    }
    report_path = exp_root / "D18_dual_backend_render_report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "sam3d_manifest": results["sam3d"]["manifest"],
                "trellis_manifest": results["trellis"]["manifest"],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--expected-frame-count", type=int, default=150)
    parser.add_argument("--expected-fps", type=float, default=30.0)
    parser.add_argument("--generated-face-budget", type=int, default=12000)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
