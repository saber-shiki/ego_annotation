#!/usr/bin/env python3
"""Run paired P09 branches from one verified frozen-upstream depth pair.

The wrapper resolves every shared input from the freeze contract, never from
branch-local arguments.  It re-verifies all frozen assets and both camera-bound
depth archives before and after P09, invokes the same P09 script/parameters for
UniDepth and DA3, and requires all generated object-owned appearance masks plus
camera/hand frame state to be byte/semantically identical across branches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import verify_v19_depth_source_ab_pair as pair_verifier

SCHEMA = "v19_depth_source_ab_p09_pair_execution_v1"
PAIR_SCHEMA = "v19_depth_source_ab_pair_contract_v1"
BRANCH_ORDER = (
    "unidepth_official_K",
    "da3_nested_official_K_hawor_conditioned",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_file(path: Path, role: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty {role}: {path}")
    return path


def require_python(path: Path) -> Path:
    # Do not Path.resolve() a virtual-environment interpreter. The executable is
    # commonly a symlink to the base Python; invoking the resolved target loses
    # pyvenv.cfg discovery and therefore the venv's site-packages.
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(str(expanded)))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise RuntimeError(f"missing or non-executable P09 Python: {absolute}")
    return absolute


def frozen_path(freeze: dict[str, Any], role: str) -> Path:
    rows = [row for row in freeze.get("fixed_assets") or [] if row.get("role") == role]
    if len(rows) != 1:
        raise RuntimeError(f"freeze contract must bind exactly one {role}, got {len(rows)}")
    return require_file(Path(str(rows[0]["path"])), role)


def provider(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        if "depth_provider" not in archive.files:
            return "unidepth"
        value = np.asarray(archive["depth_provider"])
        if value.size != 1:
            raise RuntimeError(f"depth_provider is not scalar: {path}")
        return str(value.reshape(-1)[0])


def pair_snapshot(
    pair: dict[str, Any], pair_path: Path, output: Path, *, replace: bool = False
) -> dict[str, Any]:
    freeze_path = require_file(Path(str((pair.get("freeze_contract") or {}).get("path"))), "freeze contract")
    branches = pair.get("branches") if isinstance(pair.get("branches"), dict) else {}
    for name in BRANCH_ORDER:
        if name not in branches or not isinstance(branches[name], dict):
            raise RuntimeError(f"pair contract lacks branch {name}")
    result = pair_verifier.verify(
        SimpleNamespace(
            freeze_contract=freeze_path,
            unidepth_depth=Path(str(branches[BRANCH_ORDER[0]]["path"])),
            da3_depth=Path(str(branches[BRANCH_ORDER[1]]["path"])),
            output=output,
            replace=replace,
        )
    )
    if pair.get("schema") != PAIR_SCHEMA or pair.get("status") != "ready_for_frozen_upstream_depth_provider_branches":
        raise RuntimeError(f"input pair contract is not ready: {pair_path}")
    if sha256_file(freeze_path) != str(pair["freeze_contract"].get("sha256")):
        raise RuntimeError("input pair freeze-contract SHA256 changed")
    if result["freeze_contract"]["contract_id"] != pair["freeze_contract"].get("contract_id"):
        raise RuntimeError("input pair freeze contract ID differs from fresh verification")
    for name in BRANCH_ORDER:
        old = pair["branches"][name]
        new = result["branches"][name]
        for key in ("path", "sha256", "provider", "depth_array_sha256", "confidence_array_sha256"):
            if old.get(key) != new.get(key):
                raise RuntimeError(f"pair branch {name} changed at {key}: {old.get(key)!r} != {new.get(key)!r}")
    return result


def p09_command(
    *,
    python: Path,
    p09_script: Path,
    case_id: str,
    object_id: str,
    manifest: Path,
    sam2_track: Path,
    depth: Path,
    output: Path,
    base_annotations: Path,
    camera_contract: Path,
    object_plan: Path,
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    seed: int,
) -> list[str]:
    return [
        str(python),
        "-B",
        str(p09_script),
        "--case",
        case_id,
        "--track-id",
        object_id,
        "--object-id",
        object_id,
        "--raw-frame-manifest",
        str(manifest),
        "--sam2-track-json",
        str(sam2_track),
        "--depth-npz",
        str(depth),
        "--output-dir",
        str(output),
        "--base-annotations",
        str(base_annotations),
        "--calibration-contract",
        str(camera_contract),
        "--depth-image-plane",
        "source_rgb",
        "--mask-image-plane",
        "sam2_mask",
        "--object-plan",
        str(object_plan),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
        "--anchor-frame",
        str(anchor_frame),
        "--require-anchor-frame",
        "--preserve-source-index",
        "--exclude-hand-regions",
        "--hand-bbox-exclusion-pad-px",
        "4",
        "--robust-first-surface-depth-ownership",
        "--first-surface-confidence-seed-percentile",
        "95",
        "--seed",
        str(seed),
    ]


def run_command(command: list[str], log_path: Path) -> None:
    environment = dict(os.environ)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    started = datetime.now(timezone.utc)
    with log_path.open("wb") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if process.returncode != 0:
        raise RuntimeError(
            f"paired P09 command failed with exit {process.returncode}; log={log_path}; started={started.isoformat()}"
        )


def validate_branch_output(
    *,
    branch: str,
    branch_dir: Path,
    expected_provider: str,
    case_id: str,
    object_id: str,
    frame_ids: list[int],
    anchor_frame: int,
    expected_inputs: dict[str, Path],
) -> dict[str, Any]:
    report_path = require_file(branch_dir / "v19_visible_geometry_adapter_report.json", f"{branch} P09 report")
    annotations_path = require_file(branch_dir / "annotations_v19_visible_geometry.json", f"{branch} P09 annotations")
    depth_fused_path = require_file(
        branch_dir / "v19_visible_geometry_depth_fused_report.json", f"{branch} P09 depth-fused report"
    )
    report = load_json(report_path)
    if (
        report.get("status") != "ok"
        or str(report.get("case")) != case_id
        or str(report.get("track_id")) != object_id
        or str(report.get("object_id")) != object_id
        or int(report.get("anchor_frame_idx", -1)) != anchor_frame
        or int(report.get("output_frame_count", -1)) != len(frame_ids)
        or report.get("preserve_source_index") is not True
    ):
        raise RuntimeError(f"{branch} P09 report violates case/object/anchor/timeline invariants")
    inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
    input_keys = {
        "raw_frame_manifest": "manifest",
        "sam2_track": "sam2_track",
        "depth_npz": "depth",
        "base_annotations": "base_annotations",
        "calibration_contract": "camera_contract",
        "object_plan": "object_plan",
    }
    for report_key, expected_key in input_keys.items():
        actual = Path(str(inputs.get(report_key))).expanduser().resolve()
        if actual != expected_inputs[expected_key]:
            raise RuntimeError(f"{branch} P09 input {report_key} differs from the paired contract")
    parameters = report.get("parameters") if isinstance(report.get("parameters"), dict) else {}
    if (
        float(parameters.get("first_surface_confidence_seed_percentile", -1.0)) != 95.0
        or int(parameters.get("seed", -1)) != int(expected_inputs["seed"])
        or parameters.get("require_anchor_frame") is not True
    ):
        raise RuntimeError(f"{branch} P09 parameters differ from the paired rank-only confidence contract")

    annotations = load_json(annotations_path)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or [int(row["frame_idx"]) for row in frames] != frame_ids:
        raise RuntimeError(f"{branch} P09 annotation timeline differs from the freeze contract")
    masks: dict[int, dict[str, Any]] = {}
    shared_frame_state: dict[int, str] = {}
    providers: set[str] = set()
    for frame in frames:
        idx = int(frame["frame_idx"])
        objects = [row for row in frame.get("objects") or [] if str(row.get("object_id")) == object_id]
        if len(objects) != 1:
            raise RuntimeError(f"{branch} frame {idx} does not contain exactly one target object row")
        obj = objects[0]
        mask_path = require_file(Path(str(obj.get("mask_path"))), f"{branch} object-owned mask {idx}")
        masks[idx] = {
            "path": str(mask_path),
            "bytes": int(mask_path.stat().st_size),
            "sha256": sha256_file(mask_path),
        }
        candidate = obj.get("visible_geometry_candidate")
        if isinstance(candidate, dict) and candidate.get("depth_provider"):
            providers.add(str(candidate["depth_provider"]))
        shared_frame_state[idx] = canonical_sha256({
            "frame_idx": frame.get("frame_idx"),
            "time_s": frame.get("time_s"),
            "raw_frame_path": frame.get("raw_frame_path"),
            "source_width": frame.get("source_width"),
            "source_height": frame.get("source_height"),
            "manifest_width": frame.get("manifest_width"),
            "manifest_height": frame.get("manifest_height"),
            "camera": frame.get("camera"),
            "hands": frame.get("hands"),
        })
    if not providers or providers != {expected_provider}:
        raise RuntimeError(f"{branch} P09 rows contain depth providers {sorted(providers)}, expected {expected_provider}")
    return {
        "report": {
            "path": str(report_path), "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path),
        },
        "annotations": {
            "path": str(annotations_path), "bytes": annotations_path.stat().st_size,
            "sha256": sha256_file(annotations_path),
        },
        "depth_fused_report": {
            "path": str(depth_fused_path), "bytes": depth_fused_path.stat().st_size,
            "sha256": sha256_file(depth_fused_path),
        },
        "visible_metric_frame_count": int(report.get("visible_metric_frame_count", 0)),
        "skipped_row_count": int(len(report.get("skipped_rows_preview") or [])),
        "anchor_frame": anchor_frame,
        "depth_provider_rows": sorted(providers),
        "object_owned_masks": masks,
        "shared_frame_state_sha256": shared_frame_state,
    }


def compare_branch_outputs(branches: dict[str, dict[str, Any]], frame_ids: list[int]) -> dict[str, Any]:
    left = branches[BRANCH_ORDER[0]]
    right = branches[BRANCH_ORDER[1]]
    mask_failures = []
    state_failures = []
    for idx in frame_ids:
        left_mask = left["object_owned_masks"].get(idx)
        right_mask = right["object_owned_masks"].get(idx)
        if not left_mask or not right_mask or left_mask["sha256"] != right_mask["sha256"]:
            mask_failures.append({"frame_idx": idx, "left": left_mask, "right": right_mask})
        if left["shared_frame_state_sha256"].get(idx) != right["shared_frame_state_sha256"].get(idx):
            state_failures.append({
                "frame_idx": idx,
                "left": left["shared_frame_state_sha256"].get(idx),
                "right": right["shared_frame_state_sha256"].get(idx),
            })
    if mask_failures:
        raise RuntimeError(f"paired P09 object-owned masks differ: {mask_failures[:10]}")
    if state_failures:
        raise RuntimeError(f"paired P09 camera/hand shared state differs: {state_failures[:10]}")
    return {
        "status": "ok",
        "frame_count": len(frame_ids),
        "object_owned_masks_byte_identical": True,
        "camera_hand_shared_frame_state_identical": True,
        "mask_failures": [],
        "shared_state_failures": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    pair_path = require_file(args.pair_contract, "depth-source pair contract")
    pair = load_json(pair_path)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise RuntimeError(f"paired P09 output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    verification_dir = output_root / "pair_verification"
    verification_dir.mkdir()
    before = pair_snapshot(pair, pair_path, verification_dir / "pair_before_p09.json")
    freeze_path = require_file(Path(str(before["freeze_contract"]["path"])), "freeze contract")
    freeze = load_json(freeze_path)
    frame_ids = [int(value) for value in freeze["frame_ids"]]
    anchor_frame = int(freeze["anchor"]["frame_idx"])
    case_id = str(freeze["case_id"])
    object_id = str(freeze["object_id"])
    python = require_python(args.python)
    p09_script = require_file(args.p09_script, "P09 script")
    canonical_p09_script = Path(__file__).resolve().with_name("build_v19_visible_geometry_from_sam2_depth.py")
    if p09_script != canonical_p09_script:
        raise RuntimeError(
            f"paired P09 must use the hash-bound sibling canonical script: {p09_script} != {canonical_p09_script}"
        )
    shared = {
        "manifest": frozen_path(freeze, "raw_frame_manifest"),
        "sam2_track": frozen_path(freeze, "sam2_track"),
        "base_annotations": frozen_path(freeze, "base_annotations"),
        "camera_contract": frozen_path(freeze, "official_camera_contract"),
        "object_plan": frozen_path(freeze, "object_plan"),
    }
    branch_results: dict[str, dict[str, Any]] = {}
    invocation_rows = []
    for name in BRANCH_ORDER:
        depth = require_file(Path(str(before["branches"][name]["path"])), f"{name} depth")
        branch_dir = output_root / "branches" / name / "P09_visible_geometry"
        branch_dir.parent.mkdir(parents=True)
        command = p09_command(
            python=python,
            p09_script=p09_script,
            case_id=case_id,
            object_id=object_id,
            manifest=shared["manifest"],
            sam2_track=shared["sam2_track"],
            depth=depth,
            output=branch_dir,
            base_annotations=shared["base_annotations"],
            camera_contract=shared["camera_contract"],
            object_plan=shared["object_plan"],
            frame_start=frame_ids[0],
            frame_end=frame_ids[-1],
            anchor_frame=anchor_frame,
            seed=int(args.seed),
        )
        log_path = output_root / "branches" / name / "P09.log"
        run_command(command, log_path)
        expected_inputs = {
            **shared,
            "depth": depth,
            "seed": int(args.seed),
        }
        branch_results[name] = validate_branch_output(
            branch=name,
            branch_dir=branch_dir,
            expected_provider=before["branches"][name]["provider"],
            case_id=case_id,
            object_id=object_id,
            frame_ids=frame_ids,
            anchor_frame=anchor_frame,
            expected_inputs=expected_inputs,
        )
        invocation_rows.append({
            "branch": name,
            "depth_provider": before["branches"][name]["provider"],
            "depth_archive_sha256": before["branches"][name]["sha256"],
            "command": command,
            "command_except_depth_and_output_sha256": canonical_sha256([
                "<PYTHON>", "-B", "<P09_SCRIPT>", "--case", case_id, "--track-id", object_id,
                "--object-id", object_id, "<SHARED_INPUTS>", "--frame-start", str(frame_ids[0]),
                "--frame-end", str(frame_ids[-1]), "--anchor-frame", str(anchor_frame),
                "--require-anchor-frame", "--preserve-source-index", "--exclude-hand-regions",
                "--hand-bbox-exclusion-pad-px", "4", "--robust-first-surface-depth-ownership",
                "--first-surface-confidence-seed-percentile", "95", "--seed", str(args.seed),
            ]),
            "log": str(log_path),
        })
    comparison = compare_branch_outputs(branch_results, frame_ids)
    after = pair_snapshot(pair, pair_path, verification_dir / "pair_after_p09.json")
    if before["freeze_contract"]["contract_id"] != after["freeze_contract"]["contract_id"]:
        raise RuntimeError("frozen upstream changed while paired P09 was running")
    for name in BRANCH_ORDER:
        if before["branches"][name]["sha256"] != after["branches"][name]["sha256"]:
            raise RuntimeError(f"depth archive changed while paired P09 was running: {name}")
    invocation_hashes = {row["command_except_depth_and_output_sha256"] for row in invocation_rows}
    if len(invocation_hashes) != 1:
        raise RuntimeError("paired P09 invocation parameters differ outside depth/output bindings")
    report = {
        "schema": SCHEMA,
        "status": "ready_for_depth_dependent_downstream_pair_after_p09",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "prediction-side paired P09 execution only; no reference labels, scoring, or SAM3D inference",
        "pair_contract": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
        "freeze_contract": {
            "path": str(freeze_path), "sha256": sha256_file(freeze_path),
            "contract_id": freeze["contract_id"], "asset_count": freeze["asset_counts"]["total"],
        },
        "case_id": case_id,
        "object_id": object_id,
        "frame_count": len(frame_ids),
        "anchor_frame": anchor_frame,
        "only_intervention": "external_metric_depth_provider_and_monotonic_provider_specific_error_ranking",
        "p09_script": {"path": str(p09_script), "sha256": sha256_file(p09_script)},
        "python": str(python),
        "seed": int(args.seed),
        "confidence_policy": {
            "use": "within-provider per-component 95th-percentile monotonic rank only",
            "absolute_metric_error_calibration_required_for_this_P09_gate": False,
            "cross_provider_numeric_threshold_comparison": False,
            "absolute_confidence_threshold_use_downstream": "forbidden until separately calibrated",
        },
        "invocations": invocation_rows,
        "branches": branch_results,
        "shared_output_invariants": comparison,
        "pair_verification_before": str(verification_dir / "pair_before_p09.json"),
        "pair_verification_after": str(verification_dir / "pair_after_p09.json"),
        "sam3d_invoked": False,
        "annotation_ready": False,
    }
    report_path = output_root / "v19_depth_source_ab_p09_pair_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"], "report": str(report_path), "case_id": case_id,
        "object_id": object_id, "anchor_frame": anchor_frame,
        "visible_metric_frame_count": {
            name: branch_results[name]["visible_metric_frame_count"] for name in BRANCH_ORDER
        },
        "shared_output_invariants": comparison,
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-contract", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--p09-script",
        type=Path,
        default=Path(__file__).with_name("build_v19_visible_geometry_from_sam2_depth.py"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1901)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
