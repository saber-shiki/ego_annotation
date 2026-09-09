#!/usr/bin/env python3
"""Prepare the specific same-run P14 anchor-binding repair after P09 recovery.

No old prediction is rewritten, no model runs, and no successful receipt is
invented. This second bounded transition preserves all previous identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from v19_run_lineage import RunLineage
from v19_run_recovery import (
    RECOVERY_FILE,
    P14_RECOVERY_DIR,
    P14_RECOVERY_FILE,
    P14_SCHEMA,
    code_identity,
    event_code_is_valid,
    recovery_context,
    run_identity,
    validate_transition,
)
from v19_runtime_assets import verify_bundle
from v19_runtime_contracts import in_run, load, sha


def prepare_p14_recovery(root, source_bundle, target_bundle):
    root, source_bundle, target_bundle = map(
        lambda p: Path(p).resolve(), (root, source_bundle, target_bundle)
    )
    directory = root / P14_RECOVERY_DIR
    if directory.exists() or source_bundle == target_bundle:
        raise ValueError("new explicit P14 recovery required")
    birth_path = in_run(root / "input/runtime_input_contract.json", root)
    birth = load(birth_path)
    old = verify_bundle(source_bundle, asset_role="pose")
    new = verify_bundle(target_bundle)
    if (
        old.get("preflight_status") != "imports_checked"
        or new.get("preflight_status") != "imports_checked"
    ):
        raise ValueError("recovery bundles must pass import preflight")
    previous = recovery_context(root, source_bundle, birth, old)
    if previous is None or previous["contract"]["resume_phase"] != "P09":
        raise ValueError("P14 binding repair requires the completed P09 transition")
    changed = validate_transition(
        old,
        new,
        load(source_bundle / "runtime/runtime_config.json"),
        load(target_bundle / "runtime/runtime_config.json"),
        source_bundle,
        target_bundle,
        phase="P14",
    )
    obj = previous["contract"]["object_id"]
    lineage = RunLineage(
        root,
        source_bundle,
        source_bundle / "runtime/runtime_config.json",
        asset_role="pose",
    )
    completion = (
        root / f"measurements/geometry_completion/sam3d_{obj}/completion_report.json"
    )
    ann = (
        root
        / f"measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json"
    )
    with lineage.verification_batch():
        lineage.require(completion)
        lineage.require(ann)
    blocker = in_run(root / "state/runtime_blockers/P14_visible_pose.json", root)
    failure = load(blocker)
    if failure.get("phase") != "P14_visible_pose" or failure.get("returncode") != 1:
        raise ValueError("recorded P14 visible-pose failure required")
    log_path = in_run(root / "logs/integration/P14_visible_pose.log", root)
    if (
        "P14 completion lacks a byte-bound selected-anchor evidence report"
        not in log_path.read_text()
    ):
        raise ValueError("unsupported P14 failure; no automatic recovery")
    old_output = root / f"measurements/pose_fits/{obj}/visible"
    if not old_output.is_dir() or any(old_output.iterdir()):
        raise ValueError("P14 repair is restricted to an empty failed fitting output")
    for path in [
        root / f"measurements/pose_fits/{obj}/visible_anchor_bound",
        root / "logs/integration/P14_anchor_bound_visible_pose.log",
        completion.parent / "anchor_bound_completion_report.json",
        root / f"measurements/pose_fits/{obj}/temporal",
    ]:
        if path.exists():
            raise ValueError("corrected P14 or downstream output already exists")
    events_path = in_run(root / "logs/integration/events.jsonl", root)
    prefix = events_path.read_bytes()
    if not prefix.endswith(b"\n"):
        raise ValueError("complete current-run event prefix required")
    successful = []
    for line in prefix.splitlines():
        event = json.loads(line)
        if event.get("returncode") == 0 and "error" not in event:
            if event.get("run_identity") != run_identity(
                birth
            ) or not event_code_is_valid(event, code_identity(old), previous):
                raise ValueError("P14 recovery prefix has foreign code/run history")
            successful.append(event["phase"])
    if (
        not {"P09", "P11", "P12_P13_SAM3D"} <= set(successful)
        or "P14_visible_pose" in successful
    ):
        raise ValueError("completed current reconstruction and failed P14 required")
    protected_paths = [
        birth_path,
        completion,
        ann,
        blocker,
        log_path,
        root / f"state/anchor_decisions/{obj}.json",
        root / "measurements/object_candidates/object_plan_agent.json",
        root / "state/object_selection.json",
        root / RECOVERY_FILE,
    ]
    metadata = {
        "schema": P14_SCHEMA,
        "mode": "same_run_recovery",
        "resume_phase": "P14",
        "run_identity": run_identity(birth),
        "input_contract_sha256": sha(birth_path),
        "object_id": obj,
        "selected_anchor_frame_idx": previous["contract"]["selected_anchor_frame_idx"],
        "source_bundle": str(source_bundle),
        "target_bundle": str(target_bundle),
        "target_bundle_manifest_sha256": sha(
            target_bundle / "runtime/bundle_manifest.json"
        ),
        "previous_recovery_sha256": sha(root / RECOVERY_FILE),
        "changed_bundle_files": changed,
        "source_events_bytes": len(prefix),
        "source_events_sha256": hashlib.sha256(prefix).hexdigest(),
        "protected_inputs": {str(p): sha(in_run(p, root)) for p in protected_paths},
        "completed_phases_retained": sorted(set(successful)),
        "claim_scope": "same-run P14 report-interface repair; existing SAM3D and physical authority unchanged",
    }
    directory.mkdir(parents=True)
    shutil.copyfile(
        source_bundle / "runtime/bundle_manifest.json",
        directory / "source_bundle_manifest.json",
    )
    shutil.copyfile(
        source_bundle / "runtime/runtime_config.json",
        directory / "source_runtime_config.json",
    )
    (directory / "source_events.jsonl").write_bytes(prefix)
    (root / P14_RECOVERY_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    recovery_context(root, target_bundle, birth, new)
    return root / P14_RECOVERY_FILE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-bundle", "target-bundle"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(prepare_p14_recovery(a.run_root, a.source_bundle, a.target_bundle))


if __name__ == "__main__":
    main()
