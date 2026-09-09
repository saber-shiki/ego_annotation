#!/usr/bin/env python3
"""Developer-only P09 recovery preparation, not prediction or cache registration.

Only a verification/launch-contract change is permitted. This does not rewrite
original events, predictions, judgments or the immutable birth input contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from v19_run_lineage import RunLineage
from v19_run_recovery import (
    RECOVERY_DIR,
    RECOVERY_FILE,
    SCHEMA,
    run_identity,
    validate_transition,
    recovery_context,
)
from v19_runtime_assets import verify_bundle
from v19_runtime_contracts import in_run, load, sha


REQUIRED_COMPLETED = {
    "P01",
    "P03b_camera",
    "P03_conditioned_depth",
    "P03c_depth_binding",
    "P04_HaWoR",
    "P06",
    "P07",
    "P08",
    "P09-propose",
}


def prepare_recovery(root, source_bundle, target_bundle):
    root, source_bundle, target_bundle = map(
        lambda p: Path(p).resolve(), (root, source_bundle, target_bundle)
    )
    contract_path = in_run(root / "input/runtime_input_contract.json", root)
    contract = load(contract_path)
    directory = root / RECOVERY_DIR
    if directory.exists() or source_bundle == target_bundle:
        raise ValueError("new explicit P09 recovery transition required")
    if contract.get("run_root") != str(root):
        raise ValueError("recovery run root mismatch")
    old_manifest_file = source_bundle / "runtime/bundle_manifest.json"
    new_manifest_file = target_bundle / "runtime/bundle_manifest.json"
    if sha(old_manifest_file) != contract["bundle_manifest_sha256"]:
        raise ValueError("source bundle differs from immutable birth identity")
    old = verify_bundle(source_bundle, asset_role="P09")
    new = verify_bundle(target_bundle)
    if (
        old.get("preflight_status") != "imports_checked"
        or new.get("preflight_status") != "imports_checked"
    ):
        raise ValueError("both recovery bundle imports must be checked")
    changed = validate_transition(
        old,
        new,
        load(source_bundle / "runtime/runtime_config.json"),
        load(target_bundle / "runtime/runtime_config.json"),
        source_bundle,
        target_bundle,
    )
    # Use corrected verification mechanics against the original manifest and
    # records; there is no old-output re-stamping or new producer execution.
    lineage = RunLineage(
        root,
        source_bundle,
        source_bundle / "runtime/runtime_config.json",
        asset_role="P09",
    )
    plan_path = root / "measurements/object_candidates/object_plan_agent.json"
    lineage.require(plan_path)
    objects = load(plan_path)["objects"]
    if len(objects) != 1:
        raise ValueError("P09 recovery requires the existing single-object plan")
    obj = objects[0]["object_id"]
    if (
        not obj
        or obj in {".", ".."}
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in obj
        )
    ):
        raise ValueError("unsafe recovery object ID")
    anchor = root / "state/anchor_decisions" / (obj + ".json")
    lineage.require(anchor)
    blocker = in_run(root / "state/runtime_blockers/P09.json", root)
    stopped = load(blocker)
    if (
        stopped.get("phase_id", stopped.get("phase")) != "P09"
        or stopped.get("status") != "blocked"
    ):
        raise ValueError("recorded stopped P09 required")
    evidence = stopped.get("evidence", {})
    if "input changed during byte verification" not in str(evidence.get("error")):
        raise ValueError(
            "recovery only supports the reviewed pre-invocation verification failure"
        )
    if (root / f"measurements/object_geometry/visible_geometry/{obj}").exists() or (
        root / "logs/integration/P09.log"
    ).exists():
        raise ValueError(
            "P09 producer already started; this pre-invocation recovery is not applicable"
        )
    events_path = in_run(root / "logs/integration/events.jsonl", root)
    prefix = events_path.read_bytes()
    if not prefix or not prefix.endswith(b"\n"):
        raise ValueError("complete recorded producer prefix required")
    events = [json.loads(line) for line in prefix.splitlines()]
    successful = [e for e in events if e.get("returncode") == 0 and "error" not in e]
    if not REQUIRED_COMPLETED <= {e["phase"] for e in successful}:
        raise ValueError("P09 recovery prerequisites not completed")
    if any(e.get("phase") in {"P09", "P11"} for e in successful):
        raise ValueError(
            "P09 recovery cannot replay already completed canonical geometry"
        )
    protected = {str(p): sha(p) for p in (contract_path, plan_path, anchor, blocker)}
    metadata = {
        "schema": SCHEMA,
        "mode": "same_run_recovery",
        "resume_phase": "P09",
        "run_identity": run_identity(contract),
        "input_contract_sha256": sha(contract_path),
        "object_id": obj,
        "selected_anchor_frame_idx": load(anchor)["selected_anchor_frame_idx"],
        "source_bundle": str(source_bundle),
        "target_bundle": str(target_bundle),
        "target_bundle_manifest_sha256": sha(new_manifest_file),
        "source_events_bytes": len(prefix),
        "source_events_sha256": hashlib.sha256(prefix).hexdigest(),
        "changed_bundle_files": changed,
        "protected_inputs": protected,
        "completed_phases_retained": sorted({e["phase"] for e in successful}),
        "claim_scope": "same-run continuation after verification-only correction; not a new cold run or fabricated producer success",
    }
    directory.mkdir(parents=True)
    shutil.copyfile(old_manifest_file, directory / "source_bundle_manifest.json")
    shutil.copyfile(
        source_bundle / "runtime/runtime_config.json",
        directory / "source_runtime_config.json",
    )
    (directory / "source_events.jsonl").write_bytes(prefix)
    (root / RECOVERY_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    recovery_context(root, target_bundle, contract, new)
    return root / RECOVERY_FILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-bundle", "target-bundle"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(prepare_recovery(args.run_root, args.source_bundle, args.target_bundle))


if __name__ == "__main__":
    main()
