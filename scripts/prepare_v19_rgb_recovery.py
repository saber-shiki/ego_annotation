#!/usr/bin/env python3
"""Prepare only the current-run missing-target-pose RGB recovery after P14.

No completed P14 pose, completion, geometry, source event or prior recovery is
rewritten. The new native actor starts pose-rgb, not geometry or P14 fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from v19_run_lineage import RunLineage
from v19_run_recovery import (
    P14_RECOVERY_FILE,
    P14_RECOVERY_DIR,
    RGB_RECOVERY_FILE,
    RGB_RECOVERY_DIR,
    RGB_SCHEMA,
    code_identity,
    event_code_is_valid,
    recovery_context,
    run_identity,
    validate_transition,
)
from v19_runtime_assets import verify_bundle
from v19_runtime_contracts import in_run, load, sha


def prepare_rgb_recovery(root, source_bundle, target_bundle):
    root, source_bundle, target_bundle = map(
        lambda p: Path(p).resolve(), (root, source_bundle, target_bundle)
    )
    directory = root / RGB_RECOVERY_DIR
    if directory.exists() or source_bundle == target_bundle:
        raise ValueError("new explicit RGB recovery required")
    birth_path = in_run(root / "input/runtime_input_contract.json", root)
    birth = load(birth_path)
    old = verify_bundle(source_bundle, asset_role="pose")
    new = verify_bundle(target_bundle)
    if (
        old.get("preflight_status") != "imports_checked"
        or new.get("preflight_status") != "imports_checked"
    ):
        raise ValueError("checked bundle imports required")
    previous = recovery_context(root, source_bundle, birth, old)
    if previous is None or previous["contract"]["resume_phase"] != "P14":
        raise ValueError("RGB recovery requires the verified P14 predecessor")
    changed = validate_transition(
        old,
        new,
        load(source_bundle / "runtime/runtime_config.json"),
        load(target_bundle / "runtime/runtime_config.json"),
        source_bundle,
        target_bundle,
        phase="P14_RGB",
    )
    obj = previous["contract"]["object_id"]
    pose_root = root / f"measurements/pose_fits/{obj}"
    pose = (
        pose_root / "visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json"
    )
    completion = (
        root / f"measurements/geometry_completion/sam3d_{obj}/completion_report.json"
    )
    bound = completion.parent / "anchor_bound_completion_report.json"
    ann = (
        root
        / f"measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json"
    )
    lineage = RunLineage(
        root,
        source_bundle,
        source_bundle / "runtime/runtime_config.json",
        asset_role="pose",
    )
    with lineage.verification_batch():
        for path in [pose, completion, bound, ann]:
            lineage.require(path)
    blocker = in_run(root / P14_RECOVERY_DIR / "failure.json", root)
    failure = load(blocker)
    if failure.get("phase") != "P14_RGB_observations" or failure.get("returncode") != 1:
        raise ValueError("recorded RGB observation failure required")
    log = in_run(root / "logs/integration/P14_RGB_observations.log", root)
    if "initial pose report lacks selected frames:" not in log.read_text():
        raise ValueError("unsupported RGB failure; no automatic retry")
    for path in [
        pose_root / "rgb/rgb_edges.npz",
        pose_root / "rgb/rgb_edges.json",
        pose_root / "rgb_full_timeline",
        pose_root / "rgb_initialized_pose.json",
        pose_root / "temporal",
        root / "logs/integration/P14_RGB_full_timeline_observations.log",
    ]:
        if path.exists():
            raise ValueError("RGB edges or downstream results already exist")
    events_path = in_run(root / "logs/integration/events.jsonl", root)
    prefix = events_path.read_bytes()
    if not prefix.endswith(b"\n"):
        raise ValueError("complete producer prefix required")
    phases = []
    for line in prefix.splitlines():
        event = json.loads(line)
        if event.get("returncode") == 0 and "error" not in event:
            if event.get("run_identity") != run_identity(
                birth
            ) or not event_code_is_valid(event, code_identity(old), previous):
                raise ValueError("RGB recovery prefix has foreign producer identity")
            phases.append(event["phase"])
    if (
        not {"P13_anchor_binding", "P14_anchor_bound_visible_pose"} <= set(phases)
        or "P14_RGB_observations" in phases
    ):
        raise ValueError("successful P14 and failed RGB stage required")
    protected = [
        birth_path,
        pose,
        completion,
        bound,
        ann,
        blocker,
        log,
        root / P14_RECOVERY_FILE,
        root / "state/object_selection.json",
        root / f"state/anchor_decisions/{obj}.json",
        root / "measurements/object_candidates/object_plan_agent.json",
    ]
    metadata = {
        "schema": RGB_SCHEMA,
        "mode": "same_run_recovery",
        "resume_phase": "P14_RGB",
        "run_identity": run_identity(birth),
        "input_contract_sha256": sha(birth_path),
        "object_id": obj,
        "selected_anchor_frame_idx": previous["contract"]["selected_anchor_frame_idx"],
        "source_bundle": str(source_bundle),
        "target_bundle": str(target_bundle),
        "target_bundle_manifest_sha256": sha(
            target_bundle / "runtime/bundle_manifest.json"
        ),
        "previous_recovery_sha256": sha(root / P14_RECOVERY_FILE),
        "changed_bundle_files": changed,
        "source_events_bytes": len(prefix),
        "source_events_sha256": hashlib.sha256(prefix).hexdigest(),
        "protected_inputs": {str(p): sha(in_run(p, root)) for p in protected},
        "completed_phases_retained": sorted(set(phases)),
        "claim_scope": "same-run RGB target-missing-pose repair, not new P14 measurement or geometry prediction",
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
    (root / RGB_RECOVERY_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    recovery_context(root, target_bundle, birth, new)
    return root / RGB_RECOVERY_FILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-bundle", "target-bundle"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(prepare_rgb_recovery(args.run_root, args.source_bundle, args.target_bundle))


if __name__ == "__main__":
    main()
