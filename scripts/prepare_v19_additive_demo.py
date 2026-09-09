#!/usr/bin/env python3
"""Prepare a generic isolated additive visual-v2 demo request.

The request byte-binds the selected object, full SAM2 appearance track/masks,
P14/P15, completion, P17 visual judgment and source timeline. Core P16–P19
factor/solver outputs are not prerequisites. It never creates or
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


def selected_track_binding(root: Path, object_id: str, selection: dict,
                           runtime_input: dict, annotations_path: Path,
                           supplied_track: Path) -> dict[str, object]:
    """Bind the supplied P07 track to selected P10/object-plan producer identity."""
    case_id = runtime_input.get("case")
    if not isinstance(case_id, str) or not SAFE_ID.fullmatch(case_id):
        raise ValueError("runtime input case is missing or unsafe")
    if selection.get("case") is not None and selection.get("case") != case_id:
        raise ValueError("object selection case does not match runtime input case")
    annotations = load(annotations_path)
    embedded = annotations.get("v19_visible_geometry_adapter")
    if not isinstance(embedded, dict):
        raise ValueError("annotations lack visible-geometry adapter binding")
    adapter_path = annotations_path.parent / "v19_visible_geometry_adapter_report.json"
    adapter = load(adapter_path)
    object_plan_path = root / "measurements/object_candidates/object_plan_agent.json"
    object_plan = load(object_plan_path)
    for label, value in (("embedded adapter", embedded), ("adapter report", adapter)):
        if (value.get("case") != case_id or value.get("object_id") != object_id):
            raise ValueError(f"{label} case/object does not match selection")
    track_id = embedded.get("track_id")
    if (not isinstance(track_id, str) or not track_id
            or adapter.get("track_id") != track_id):
        raise ValueError("visible-geometry track identity is missing or ambiguous")
    expected_tracks = [embedded.get("sam2_track"), (adapter.get("inputs") or {}).get("sam2_track")]
    supplied = supplied_track.resolve()
    if any(not value or Path(value).expanduser().resolve() != supplied for value in expected_tracks):
        raise ValueError("supplied SAM2 track does not match visible-geometry producer")
    plan = object_plan.get("plan") if isinstance(object_plan.get("plan"), dict) else object_plan
    rows = plan.get("objects") if isinstance(plan, dict) else None
    matches = [row for row in rows or [] if isinstance(row, dict) and row.get("track_id") == track_id]
    if len(matches) != 1:
        raise ValueError("selected track is not uniquely bound in object plan")
    plan_object = matches[0].get("object_id")
    if plan_object is not None and plan_object != object_id:
        raise ValueError("object-plan track belongs to a different object")
    observed_track_ids = {
        obj.get("track_id")
        for frame in annotations.get("frames", []) if isinstance(frame, dict)
        for obj in frame.get("objects", []) if isinstance(obj, dict) and obj.get("object_id") == object_id
    }
    if observed_track_ids != {track_id}:
        raise ValueError("annotation object rows do not uniquely bind selected track")
    return {
        "case_id": case_id,
        "track_id": track_id,
        "adapter_path": adapter_path,
        "object_plan_path": object_plan_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-bundle", "demo-bundle", "output-dir", "sam2-track"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--object-id")
    parser.add_argument("--demo-id", default="visual_v2")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--visual-scale-record", type=Path)
    parser.add_argument("--reconstruct-visual-mesh", action="store_true")
    parser.add_argument("--sam3d-seed", type=int, default=42)
    parser.add_argument("--estimate-visual-scale", action="store_true")
    parser.add_argument("--visual-scale-bounds", type=float, nargs=2, default=[0.8, 1.2])
    parser.add_argument("--visual-scale-samples", type=int, default=41)
    parser.add_argument("--visual-scale-max-frames", type=int, default=7)
    parser.add_argument("--visual-scale-raster-width", type=int, default=192)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=20000)
    args = parser.parse_args()
    import math
    lo, hi = args.visual_scale_bounds
    if (not all(math.isfinite(x) for x in (lo, hi)) or not 0 < lo <= 1 <= hi or lo == hi
            or not 3 <= args.visual_scale_samples <= 101 or not 3 <= args.visual_scale_max_frames <= 15
            or not 32 <= args.visual_scale_raster_width <= 512 or not 0 <= args.sam3d_seed < 2**32
            or (args.estimate_visual_scale and args.visual_scale_record)):
        raise ValueError("invalid/conflicting native reconstruction or visual scale configuration")
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

    runtime_input_path = root / "input/runtime_input_contract.json"
    runtime_input = load(runtime_input_path)
    selection_path = root / "state/object_selection.json"
    selection = load(selection_path)
    object_id = args.object_id or selection.get("object_id")
    if not isinstance(object_id, str) or not SAFE_ID.fullmatch(object_id):
        raise ValueError("selected object id missing or unsafe")
    if selection.get("object_id") != object_id:
        raise ValueError("--object-id does not match current object selection")
    binding = selected_track_binding(
        root, object_id, selection, runtime_input,
        root / f"measurements/object_geometry/visible_geometry/{object_id}/annotations_v19_visible_geometry.json",
        args.sam2_track,
    )
    case_id = str(binding["case_id"])

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
        "runtime_input": runtime_input_path,
        "object_selection": selection_path,
        "object_plan": Path(binding["object_plan_path"]),
        "visible_geometry_adapter": Path(binding["adapter_path"]),
        "raw_frame_manifest": raw_manifest,
        "sam2_track": sam2_track,
        "annotations": root / f"measurements/object_geometry/visible_geometry/{object_id}/annotations_v19_visible_geometry.json",
        "p14": root / f"measurements/pose_fits/{object_id}/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json",
        "p15": root / f"measurements/pose_fits/{object_id}/temporal/v19_rigid_object_pose_graph_report.json",
        "completion": root / f"measurements/geometry_completion/sam3d_{object_id}/anchor_bound_completion_report.json",
        "hawor": root / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz",
        "mano_bridge": root / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz",
        "judgment": root / f"state/agent_interaction_judgments/{object_id}.json",
        "source_video": Path(runtime_input["input_video"]),
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
        "track_id": binding["track_id"],
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
            "mode": "native_demo_appearance_sam3d" if args.reconstruct_visual_mesh else "disabled_core_mesh",
            "enabled": args.reconstruct_visual_mesh,
            "seed": args.sam3d_seed,
            "core_completion_fallback": not args.reconstruct_visual_mesh,
        },
        "visual_scale_alignment": {
            "mode": "image_uniform_grid" if args.estimate_visual_scale else "disabled",
            "enabled": args.estimate_visual_scale, "bounds": args.visual_scale_bounds,
            "samples": args.visual_scale_samples, "max_frames": args.visual_scale_max_frames,
            "raster_width": args.visual_scale_raster_width,
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
