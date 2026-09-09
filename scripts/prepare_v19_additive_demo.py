#!/usr/bin/env python3
"""Prepare a generic isolated additive visual-v2 demo request.

The request byte-binds the selected object, full SAM2 appearance track/masks,
P14/P15, completion, P17 inputs and source timeline. It never creates or
relabels core prediction outputs.
"""
from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v19_run_lineage import RunLineage
from v19_runtime_assets import verify_bundle
from v19_runtime_contracts import load, sha

AGENT = "ego-annotation.rgb-sam3d-demo"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def selected_track_masks(track_path: Path, root: Path, frame_ids: set[int]) -> dict[str, dict[str, str]]:
    track = load(track_path)
    rows = {}
    for key, row in track.items():
        idx = int(key)
        if str(idx) != str(key) or idx in rows or not isinstance(row, dict):
            raise ValueError("invalid/duplicate SAM2 track row")
        rows[idx] = row
    if set(rows) != frame_ids:
        raise ValueError("SAM2 track does not exactly cover raw timeline")
    result = {}
    for idx, row in sorted(rows.items()):
        value = row.get("mask_path")
        if value is None:
            if row.get("visible") is True:
                raise ValueError(f"visible SAM2 frame {idx} lacks mask")
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"SAM2 mask missing/outside run at frame {idx}")
        result[str(idx)] = {"path": str(path), "sha256": sha(path)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-bundle", "demo-bundle", "output-dir", "sam2-track"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--demo-id", default="visual_v2")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--visual-scale-record", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=20000)
    args = parser.parse_args()
    root, source, bundle, out = (value.resolve() for value in (
        args.run_root, args.source_bundle, args.demo_bundle, args.output_dir))
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("new developer request directory separate from run required")
    if not SAFE_ID.fullmatch(args.demo_id) or args.render_width < 2:
        raise ValueError("unsafe demo id or render width")
    verify_bundle(bundle)
    if not (bundle / ".pi/agents/ego-annotation.rgb-sam3d-demo.md").is_file():
        raise ValueError("curated demo profile missing")
    verify_bundle(source, asset_role="demo")
    lineage = RunLineage(root, source, source / "runtime/runtime_config.json", asset_role="demo")

    selection_path = root / "state/object_selection.json"
    selection = load(selection_path)
    object_id = args.object_id or selection.get("object_id")
    if not isinstance(object_id, str) or not SAFE_ID.fullmatch(object_id):
        raise ValueError("selected object id missing or unsafe")
    if selection.get("object_id") != object_id:
        raise ValueError("--object-id does not match current object selection")
    case_id = selection.get("case") or load(root / "input/runtime_input_contract.json").get("case")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("current case id unavailable")

    raw_manifest = root / "input/raw_frame_manifest/manifest.json"
    raw = load(raw_manifest)
    frame_rows = raw.get("frames")
    if raw.get("full_source_timeline") is not True:
        raise ValueError("additive visual-v2 requires full raw source timeline")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("raw frame manifest has no timeline")
    frame_ids = {int(row["frame_idx"]) for row in frame_rows}
    if len(frame_ids) != len(frame_rows):
        raise ValueError("raw frame manifest has duplicate frame IDs")
    raw_frame_inputs = {}
    for row in frame_rows:
        idx = int(row["frame_idx"])
        value = row.get("raw_frame_path") or row.get("rgb")
        path = Path(value).expanduser().resolve() if value else Path()
        if not value or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"raw RGB missing/outside run at frame {idx}")
        raw_frame_inputs[str(idx)] = {"path": str(path), "sha256": sha(path)}
    sam2_track = args.sam2_track.resolve()
    mask_inputs = selected_track_masks(sam2_track, root, frame_ids)

    paths = {
        "object_selection": selection_path,
        "raw_frame_manifest": raw_manifest,
        "sam2_track": sam2_track,
        "annotations": root / f"measurements/object_geometry/visible_geometry/{object_id}/annotations_v19_visible_geometry.json",
        "p14": root / f"measurements/pose_fits/{object_id}/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json",
        "p15": root / f"measurements/pose_fits/{object_id}/temporal/v19_rigid_object_pose_graph_report.json",
        "completion": root / f"measurements/geometry_completion/sam3d_{object_id}/anchor_bound_completion_report.json",
        "hawor": root / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz",
        "mano_bridge": root / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz",
        "judgment": root / f"state/agent_interaction_judgments/{object_id}.json",
        "p17": root / f"measurements/contact_visibility_factors/{object_id}/{case_id}/v19_visible_contact_ownership_factor_report.json",
        "source_video": Path(load(root / "input/runtime_input_contract.json")["input_video"]),
    }
    if args.visual_scale_record:
        paths["visual_scale_record"] = args.visual_scale_record.resolve()
    completion_payload = load(paths["completion"])
    mesh_contract = completion_payload.get("render_mesh_contract") or {}
    completion_mesh = Path(mesh_contract.get("required_mesh_path", "")).expanduser().resolve()
    if (not completion_mesh.is_relative_to(root) or not completion_mesh.is_file()
            or sha(completion_mesh) != mesh_contract.get("required_mesh_sha256")):
        raise ValueError("completion render mesh identity invalid")
    paths["completion_mesh"] = completion_mesh
    # Selection is an immutable current-run state binding but is not a model
    # producer. All prediction artifacts and every raw mask require lineage.
    with lineage.verification_batch():
        for name, path in paths.items():
            if name != "object_selection":
                lineage.require(path)
        for item in mask_inputs.values():
            lineage.require(item["path"])
        for item in raw_frame_inputs.values():
            lineage.require(item["path"])
    if not selection_path.is_file():
        raise ValueError("object selection missing")

    namespace = root / "state/demo_visual" / object_id / args.demo_id
    contract_dir = namespace / "receipts"
    if namespace.exists():
        raise ValueError("demo visual-v2 namespace already exists")
    contract = {
        "schema": "v19_additive_demo_launch_v2",
        "mode": "additive_demo_visual_v2",
        "authority": "demo_only_visual_hypothesis",
        "run_root": str(root),
        "run_identity": lineage.identity,
        "case_id": case_id,
        "object_id": object_id,
        "demo_id": args.demo_id,
        "render_width": args.render_width,
        "compute_host": "dexrobot-Standard-PC-Q35-ICH9-2009",
        "physical_gpu": args.gpu,
        "minimum_free_mib": args.minimum_free_mib,
        "runtime_config": str(bundle / "runtime/runtime_config.json"),
        "demo_bundle_manifest_sha256": sha(bundle / "runtime/bundle_manifest.json"),
        "source_execution_bundle_manifest": str(source / "runtime/bundle_manifest.json"),
        "source_execution_bundle_manifest_sha256": sha(source / "runtime/bundle_manifest.json"),
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "sam2_mask_inputs": mask_inputs,
        "raw_frame_inputs": raw_frame_inputs,
        "core_outputs_are_read_only": True,
        "core_state_modified": False,
        "sam3d_reconditioning": {
            "mode": "optional_external_demo_only_hook",
            "enabled": False,
            "reason": "no fresh demo SAM3D completion supplied; retain full current topology and optional bounded static alignment",
            "core_completion_fallback": True,
        },
        "native_handoff": str(contract_dir / "native_handoff.md"),
        "command": [
            str(load(bundle / "runtime/runtime_config.json")["main_python"]),
            str(bundle / "scripts/run_v19_additive_demo_branch.py"),
            "--contract", str(contract_dir / "launch_contract.json"),
        ],
        "tmux_target": "ego_annotation_rgb_sam3d_demo_" + uuid.uuid4().hex[:12],
    }
    target = contract_dir / "launch_contract.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(contract, indent=2) + "\n")
    task = json.dumps({
        "mode": "additive_demo_visual_v2",
        "runtime_cwd": str(bundle),
        "contract_path": str(target),
        "run_root": str(root),
        "tmux_target": contract["tmux_target"],
    }, sort_keys=True)
    out.mkdir(parents=True)
    (out / "native.workflow.js").write_text(
        "const result=await runs.run('native-additive-demo',{agent:" + json.dumps(AGENT)
        + ",task:" + json.dumps(task) + ",output:" + json.dumps(contract["native_handoff"])
        + "}); return {result,outputReference:result.outputReference,artifactPaths:result.artifactPaths};\n"
    )
    request = {
        "cwd": str(bundle), "agentScope": "project", "context": "fresh", "async": True,
        "worktree": False, "timeoutMs": 3600000, "toolTimeoutMs": 1800000,
        "workflowScriptPath": str(out / "native.workflow.js"),
    }
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(json.dumps(request, indent=2))


if __name__ == "__main__":
    main()
