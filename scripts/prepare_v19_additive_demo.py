#!/usr/bin/env python3
"""Prepare a native additive-demo request after current-run pose/judgment inputs."""
import argparse, hashlib, json, uuid
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v19_run_lineage import RunLineage
from v19_runtime_assets import verify_bundle
from v19_runtime_contracts import load, sha

AGENT = "ego-annotation.rgb-sam3d-demo"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["run-root", "source-bundle", "demo-bundle", "output-dir"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--minimum-free-mib", type=int, default=20000)
    p.add_argument("--retry-permission-fix", action="store_true")
    a = p.parse_args()
    root, source, bundle, out = map(
        lambda x: x.resolve(),
        [a.run_root, a.source_bundle, a.demo_bundle, a.output_dir],
    )
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("new developer request directory separate from run required")
    manifest = verify_bundle(bundle)
    if not (bundle / ".pi/agents/ego-annotation.rgb-sam3d-demo.md").is_file():
        raise ValueError("curated demo profile missing")
    source_manifest = verify_bundle(source, asset_role="demo")
    lineage = RunLineage(
        root, source, source / "runtime/runtime_config.json", asset_role="demo"
    )
    obj = "milk_carton"
    paths = {
        "annotations": root
        / f"measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json",
        "p15": root
        / f"measurements/pose_fits/{obj}/temporal/v19_rigid_object_pose_graph_report.json",
        "completion": root
        / f"measurements/geometry_completion/sam3d_{obj}/anchor_bound_completion_report.json",
        "hawor": root
        / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz",
        "mano_bridge": root
        / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz",
        "judgment": root / f"state/agent_interaction_judgments/{obj}.json",
        "source_video": root / "input/source.mp4",
    }
    with lineage.verification_batch():
        for path in paths.values():
            lineage.require(path)
    retry = root / "state/additive_demo/retry_exec_v1"
    if a.retry_permission_fix:
        initial = root / "state/additive_demo"
        old_contract = initial / "launch_contract.json"
        old_handoff = initial / "native_handoff.md"
        if (
            retry.exists()
            or not old_contract.is_file()
            or not old_handoff.is_file()
            or "Permission denied" not in old_handoff.read_text()
        ):
            raise ValueError("specific failed direct-execution demo attempt required")
        contract_dir = retry
    else:
        contract_dir = root / "state/additive_demo"
        if contract_dir.exists():
            raise ValueError("initial additive demo already prepared")
    for path in [
        root / "state/demo_inputs/milk_carton_pose_mesh.json",
        root / "measurements/demo_visual/milk_carton",
        root / "renders/milk_carton_demo",
        root / "state/additive_demo/branch_result.json",
    ]:
        if path.exists():
            raise ValueError("additive demo already started: " + str(path))
    contract = {
        "schema": "v19_additive_demo_launch_v1",
        "mode": "additive_demo",
        "authority": "demo_only_visual_hypothesis",
        "run_root": str(root),
        "run_identity": lineage.identity,
        "compute_host": "dexrobot-Standard-PC-Q35-ICH9-2009",
        "physical_gpu": a.gpu,
        "minimum_free_mib": a.minimum_free_mib,
        "runtime_config": str(bundle / "runtime/runtime_config.json"),
        "demo_bundle_manifest_sha256": sha(bundle / "runtime/bundle_manifest.json"),
        "source_execution_bundle_manifest_sha256": sha(
            source / "runtime/bundle_manifest.json"
        ),
        "inputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in paths.items()},
        "core_outputs_are_read_only": True,
        "final_package_waits_for_core_P19": True,
        "native_handoff": str(contract_dir / "native_handoff.md"),
        "supersedes_failed_contract": (
            str(root / "state/additive_demo/launch_contract.json")
            if a.retry_permission_fix
            else None
        ),
        "supersedes_failed_handoff_sha256": (
            sha(root / "state/additive_demo/native_handoff.md")
            if a.retry_permission_fix
            else None
        ),
        "command": [
            str(load(bundle / "runtime/runtime_config.json")["main_python"]),
            str(bundle / "scripts/run_v19_additive_demo_branch.py"),
            "--contract",
            str(contract_dir / "launch_contract.json"),
        ],
        "tmux_target": "ego_annotation_rgb_sam3d_demo_" + uuid.uuid4().hex[:12],
    }
    target = contract_dir / "launch_contract.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(contract, indent=2) + "\n")
    task = json.dumps(
        {
            "mode": "additive_demo",
            "runtime_cwd": str(bundle),
            "contract_path": str(target),
            "run_root": str(root),
            "tmux_target": contract["tmux_target"],
        },
        sort_keys=True,
    )
    script = (
        "const result=await runs.run('native-additive-demo',{agent:"
        + json.dumps(AGENT)
        + ",task:"
        + json.dumps(task)
        + ",output:"
        + json.dumps(contract["native_handoff"])
        + "}); return {result,outputReference:result.outputReference,artifactPaths:result.artifactPaths};\n"
    )
    out.mkdir(parents=True)
    (out / "native.workflow.js").write_text(script)
    request = {
        "cwd": str(bundle),
        "agentScope": "project",
        "context": "fresh",
        "async": True,
        "worktree": False,
        "timeoutMs": 3600000,
        "toolTimeoutMs": 1800000,
        "workflowScriptPath": str(out / "native.workflow.js"),
    }
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(json.dumps(request, indent=2))


if __name__ == "__main__":
    main()
