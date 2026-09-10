"""Bounded developer-side P08 rewind into standalone P09 depth-order recovery.

Preparation is read-only with respect to the run.  Apply is the only operation
that archives invalid suffix files, and its create-only journal makes an
interrupted move visible.  Runtime lineage accepts only the retained old event
prefix plus events written under this recovery namespace.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import uuid

from v19_runtime_contracts import in_run, load, sha
from v19_runtime_assets import verify_bundle

SCHEMA = "rgb_scene_P08_depth_order_recovery_v1"
TRANSITION = "P09_DEPTH_ORDER"
RECOVERY_REL = Path("state/runtime_recovery/P09_DEPTH_ORDER")
RECOVERY_FILE = RECOVERY_REL / "recovery.json"
COMPLETE_FILE = RECOVERY_REL / "apply_complete.json"
JOURNAL_FILE = RECOVERY_REL / "apply_journal.jsonl"
RECOVERY_EVENTS = Path("logs/recovery/P09_DEPTH_ORDER/events.jsonl")
RECOVERY_LOG_DIR = RECOVERY_EVENTS.parent
VALIDATOR_REVISION_SCHEMA = "rgb_scene_P09_depth_order_validator_revision_v1"
SECOND_VALIDATOR_REVISION_SCHEMA = "rgb_scene_P09_anchor_validator_revision_v2"
VALIDATOR_REVISION_DIR = RECOVERY_REL / "validator_revisions"
VALIDATOR_REVISION_RESUME_PHASE = "P09-propose"
SECOND_VALIDATOR_REVISION_RESUME_PHASE = "P09"
# A validator revision may only replace the ancestry validator/stamping seam and
# its runtime instructions/config paths. Producer implementations and assets
# must remain byte-identical to the already-executed recovery bundle.
ALLOWED_VALIDATOR_REVISION_FILES = {
    ".pi/agents/ego-annotation.rgb-scene-runtime.md",
    "runtime/rgb_scene_runtime_spec.md",
    "runtime/runtime_config.json",
    "scripts/rgb_scene_lineage_base.py",
    "scripts/rgb_scene_recovery.py",
}
RETAINED_PHASES = (
    "P01", "P03a_intrinsics_probe", "P03b_camera", "P03_conditioned_depth",
    "P03c_depth_binding", "P04_HaWoR", "P06", "P07", "P08",
)
INVALIDATED_PHASES = {
    "P09-propose", "P09", "P11", "scene_geometry", "scene_anchor_binding",
    "scene_source_pose", "scene_rgb", "scene_refined_pose", "scene_appearance",
    "scene_native_appearance", "scene_refined_shape", "scene_visual_fit", "scene_export",
}
RECOVERY_PHASE_SEQUENCE = (
    "P09_DEPTH_ORDER", "P09-propose", "P09", "P11", "scene_geometry",
    "scene_anchor_binding", "scene_source_pose", "scene_rgb", "scene_refined_pose",
    "scene_appearance", "scene_native_appearance", "scene_refined_shape",
    "scene_visual_fit", "scene_export",
)
RECOVERY_FRONTEND_PHASES = {"P09_DEPTH_ORDER", "P09-propose", "P09", "P11"}
RECOVERY_STAGES = {
    "geometry", "initial-pose", "refined-pose", "appearance", "native-shape",
    "shape", "visual-fit", "export",
}
# This is deliberately a standalone transition allowlist, not a general-purpose
# relaxation of v19 recovery or runtime asset verification.
ALLOWED_CHANGED_BUNDLE_FILES = {
    ".pi/agents/ego-annotation.rgb-scene-runtime.md",
    "runtime/rgb_scene_runtime_spec.md",
    "runtime/runtime_config.json",
    "scripts/build_v19_visible_geometry_from_sam2_depth.py",
    "scripts/export_rgb_scene.py",
    "scripts/rgb_scene_depth_order.py",
    "scripts/rgb_scene_metric_eligibility.py",
    "scripts/rgb_scene_depth_order_appearance.py",
    "scripts/rgb_scene_identity.py",
    "scripts/rgb_scene_lineage_base.py",
    "scripts/rgb_scene_recovery.py",
    "scripts/rgb_scene_runtime.py",
    "scripts/run_rgb_scene_frontend.py",
    "scripts/run_rgb_scene_stage.py",
}
# Every executable/module reachable from the standalone frontend, stage,
# camera, nested reconstruction dispatch, plus the archived appearance leaf.
# Matching remains additionally bound to the exact run root.
OWNED_PROCESS_EXECUTABLES = {
    "run_rgb_scene_stage.py", "run_rgb_scene_frontend.py", "rgb_scene_camera.py",
    "run_rgb_scene_reconstruction.py", "build_v19_raw_frame_manifest.py",
    "run_unidepth_full_frame_v3.py", "resolve_v19_camera_contract.py",
    "adapt_v19_depth_to_camera_contract.py", "remote_run_hawor_export.sh",
    "export_hawor_world.py", "build_v19_owlv2_object_box_prompts.py",
    "run_sam2_vlm_points_multiobject.py", "build_v19_base_annotations.py",
    "rgb_scene_depth_order.py", "build_v19_visible_geometry_from_sam2_depth.py",
    "build_v18_compact_rigid_evidence_bundle.py", "bind_v19_completion_anchor.py",
    "fit_v18_compact_rigid_object_pose.py", "build_rgb_scene_edges.py",
    "rgb_scene_pose_io.py", "rgb_scene_depth_order_appearance.py",
    "remote_run_sam3d_objects_mesh_v7.py", "bind_rgb_scene_shape.py",
    "fit_rgb_scene_visual.py", "export_rgb_scene.py",
    "build_p11_dual_geometry_inputs.py", "run_p12_parallel_geometry_priors.py",
    "build_p13_sam3d_native_metric_bridge.py",
    "build_v18_compact_rigid_trellis_completion.py",
    "build_v19_demo_visual_inputs.py",
}
OWNED_PROCESS_MODULES = {Path(name).stem for name in OWNED_PROCESS_EXECUTABLES if name.endswith(".py")}


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value) if not isinstance(value, bytes) else value)
    return path


def run_identity(contract):
    return {k: contract[k] for k in ("run_id", "run_root", "input_video_sha256", "bundle_manifest_sha256")}


def code_identity(manifest):
    return {k: v for k, v in manifest["files"].items() if k.endswith((".py", ".sh"))}


def event_digest(event):
    return hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalized_config(value, bundle):
    if isinstance(value, dict):
        return {k: normalized_config(v, bundle) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized_config(v, bundle) for v in value]
    prefix = str(Path(bundle).resolve()) + "/"
    if isinstance(value, str) and value.startswith(prefix):
        return "$BUNDLE/" + value[len(prefix):]
    return value


def validate_transition(old, new, old_config, new_config, old_bundle, new_bundle):
    changed = {name for name in set(old["files"]) | set(new["files"])
               if old["files"].get(name) != new["files"].get(name)}
    unexpected = changed - ALLOWED_CHANGED_BUNDLE_FILES
    if unexpected:
        raise ValueError("depth-order recovery changed non-allowlisted bundle files: " + ", ".join(sorted(unexpected)))
    required = {"scripts/rgb_scene_depth_order.py", "scripts/rgb_scene_metric_eligibility.py",
                "scripts/build_v19_visible_geometry_from_sam2_depth.py", "scripts/rgb_scene_recovery.py"}
    if not required <= changed:
        raise ValueError("depth-order recovery bundle is missing the reviewed mechanism/eligibility/transition")
    # This is a file inventory, not an ordered execution plan. The bundle
    # builder discovers the two Rerun package roots through a set, so their
    # enumeration order may differ despite identical per-file identities.
    old_assets = dict(old.get("external_assets", {}))
    new_assets = dict(new.get("external_assets", {}))
    for assets in (old_assets, new_assets):
        if "scene_rrd_files" in assets:
            inventory = assets["scene_rrd_files"]
            if (not isinstance(inventory, list)
                    or not all(isinstance(path, str) for path in inventory)
                    or len(set(inventory)) != len(inventory)):
                raise ValueError("invalid recovery Rerun asset inventory")
            assets["scene_rrd_files"] = sorted(inventory)
    if old_assets != new_assets:
        raise ValueError("depth-order recovery external assets/models differ")
    if normalized_config(old_config, old_bundle) != normalized_config(new_config, new_bundle):
        raise ValueError("depth-order recovery runtime/model config differs")
    return sorted(changed)


def validate_validator_transition(old, new, old_config, new_config, old_bundle, new_bundle,
                                  *, require_lineage_change=True):
    """Permit only the fixed validator deployment revision after P09 depth order."""
    changed = {name for name in set(old["files"]) | set(new["files"])
               if old["files"].get(name) != new["files"].get(name)}
    unexpected = changed - ALLOWED_VALIDATOR_REVISION_FILES
    if unexpected:
        raise ValueError("validator revision changed producer code/assets: " + ", ".join(sorted(unexpected)))
    required = {"scripts/rgb_scene_recovery.py"}
    if require_lineage_change:
        required.add("scripts/rgb_scene_lineage_base.py")
    elif ("scripts/rgb_scene_lineage_base.py" not in old["files"]
            or "scripts/rgb_scene_lineage_base.py" not in new["files"]):
        raise ValueError("second validator revision requires existing lineage stamping")
    if not required <= changed:
        raise ValueError("validator revision is missing validator/lineage changes")
    old_assets = dict(old.get("external_assets", {}))
    new_assets = dict(new.get("external_assets", {}))
    for assets in (old_assets, new_assets):
        if "scene_rrd_files" in assets:
            inventory = assets["scene_rrd_files"]
            if (not isinstance(inventory, list)
                    or not all(isinstance(path, str) for path in inventory)
                    or len(set(inventory)) != len(inventory)):
                raise ValueError("invalid validator revision Rerun asset inventory")
            assets["scene_rrd_files"] = sorted(inventory)
    if old_assets != new_assets:
        raise ValueError("validator revision external assets/models differ")
    if normalized_config(old_config, old_bundle) != normalized_config(new_config, new_bundle):
        raise ValueError("validator revision runtime/model config differs")
    return sorted(changed)


def active_owned_processes(root, process_rows=None):
    root = str(Path(root).resolve())
    if process_rows is None:
        result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True, timeout=30)
        process_rows = result.stdout.splitlines()
    found = []
    for row in process_rows:
        text = str(row).strip()
        if not text:
            continue
        first, _, argv = text.partition(" ")
        try:
            pid = int(first)
        except ValueError:
            pid, argv = -1, text
        if pid == os.getpid():
            continue
        try:
            words = shlex.split(argv)
        except ValueError:
            words = argv.split()
        basenames = {Path(word).name for word in words}
        module_names = {words[i + 1].split(".")[-1] for i, word in enumerate(words[:-1]) if word == "-m"}
        owned_executable = bool(basenames & OWNED_PROCESS_EXECUTABLES or module_names & OWNED_PROCESS_MODULES)
        if root in argv and owned_executable:
            found.append({"pid": pid, "argv": argv})
    return found


def require_no_active_owned_processes(root, process_rows=None):
    active = active_owned_processes(root, process_rows)
    if active:
        raise ValueError("active owned scene/model process forbids recovery: " + json.dumps(active, sort_keys=True))


def validate_stop_evidence(path, root, contract, events):
    value = load(path)
    required = {
        "kind": "parent_owner_stop_and_reconciliation_not_prediction_receipt",
        "run_root": str(root), "run_uuid": contract["run_id"],
        "actor_process_terminal": "observed", "actor_resumable": False,
        "stop_tool_acknowledged": True, "new_runtime_actor_required": True,
        "run_outputs_moved_or_mutated_by_capture": False,
        "birth_input_sha256": sha(root / "input/runtime_input_contract.json"),
        "original_events_sha256": hashlib.sha256(events).hexdigest(),
        "original_events_bytes": len(events), "retained_prefix_through": "P08",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"stopped-run evidence identity mismatch: {key}")
    try:
        uuid.UUID(value["native_actor_id"]); uuid.UUID(value["workflow_id"])
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("stopped-run actor/workflow identity malformed") from exc
    if value.get("active_owned_pipeline_processes_at_check") != []:
        raise ValueError("stop evidence recorded active owned processes")
    count = value.get("retained_event_prefix_bytes")
    digest = value.get("retained_event_prefix_sha256")
    if type(count) is not int or count <= 0 or count > len(events):
        raise ValueError("invalid retained prefix byte count")
    prefix = events[:count]
    if not prefix.endswith(b"\n") or hashlib.sha256(prefix).hexdigest() != digest:
        raise ValueError("retained event prefix differs from stop evidence")
    return value, prefix


def _safe_id(value, field):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not isinstance(value, str) or not value or value in {".", ".."} or any(c not in allowed for c in value):
        raise ValueError(f"unsafe recovery {field}")
    return value


def _suffix_roots(root, case, obj, track):
    return [
        root / f"measurements/object_tracks/depth_order_owned/{track}",
        root / f"measurements/object_geometry/anchor_candidates/{obj}",
        root / f"measurements/object_geometry/visible_geometry/{obj}",
        root / f"measurements/geometry_completion/rigid_evidence/{case}/{obj}",
        root / f"measurements/geometry_completion/sam3d_{obj}",
        root / f"measurements/pose_fits/{obj}",
        root / f"measurements/demo_visual/{obj}",
        root / f"measurements/scene_native/{obj}",
        root / f"measurements/scene_shape/{obj}",
        root / f"measurements/scene_fit/{obj}",
        root / f"state/anchor_decisions/{obj}.json",
        root / f"state/physical_branch_decisions/{obj}.json",
        root / "state/object_selection.json",
        root / f"state/scene/{obj}",
        root / f"state/appearance_decisions/{obj}.json",
        root / f"state/agent_interaction_judgments/{obj}.json",
        root / "final",
    ]


def inventory_invalid_suffix(root, suffix_events, case, obj, track):
    root = Path(root).resolve()
    roots = _suffix_roots(root, case, obj, track)
    files = set()
    remove_dirs = set()
    for candidate in roots:
        if candidate.is_symlink():
            raise ValueError("suffix archive source may not be a symlink")
        if candidate.is_file():
            files.add(candidate)
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if path.is_symlink():
                    raise ValueError("suffix archive tree may not contain symlinks")
                if path.is_file(): files.add(path)
                elif path.is_dir(): remove_dirs.add(path)
            remove_dirs.add(candidate)
    covered = lambda p: any(p == base or p.is_relative_to(base) for base in roots)
    for event in suffix_events:
        for name in event.get("output_sha256", {}):
            path = in_run(name, root)
            if not covered(path):
                raise ValueError("invalidated producer output is outside bounded suffix roots: " + name)
            files.add(path)
    mapping = []
    for path in sorted(files):
        rel = path.relative_to(root)
        mapping.append({"source": str(path), "source_relative": str(rel),
                        "archive_relative": str(Path("invalidated") / rel), "sha256": sha(path),
                        "bytes": path.stat().st_size})
    return mapping, [str(p.relative_to(root)) for p in sorted(remove_dirs, key=lambda p: (-len(p.parts), str(p)))]


def _load_package(package):
    package = Path(package).resolve()
    contract_path = package / "recovery_contract.json"
    archive_path = package / "archive_manifest.json"
    if not contract_path.is_file() or not archive_path.is_file():
        raise ValueError("incomplete recovery preparation package")
    contract, archive = load(contract_path), load(archive_path)
    if contract.get("schema") != SCHEMA or contract.get("transition") != TRANSITION:
        raise ValueError("wrong standalone recovery schema/transition")
    if contract.get("archive_manifest_sha256") != sha(archive_path):
        raise ValueError("recovery archive manifest changed")
    for name, expected in contract.get("package_files_sha256", {}).items():
        path = package / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError("recovery preparation snapshot changed: " + name)
    return package, contract, archive


def prepare_recovery(root, source_bundle, target_bundle, stop_evidence, package, *, process_rows=None):
    root, source_bundle, target_bundle, package = map(lambda p: Path(p).resolve(), (root, source_bundle, target_bundle, package))
    if package.exists() or package.is_relative_to(root) or package.is_relative_to(source_bundle) or package.is_relative_to(target_bundle):
        raise ValueError("new external recovery preparation directory required")
    require_no_active_owned_processes(root, process_rows)
    contract_path = in_run(root / "input/runtime_input_contract.json", root)
    birth = load(contract_path)
    if birth.get("schema") != "rgb_scene_input_v1" or birth.get("run_root") != str(root):
        raise ValueError("immutable birth contract mismatch")
    old_manifest_path = source_bundle / "runtime/bundle_manifest.json"
    new_manifest_path = target_bundle / "runtime/bundle_manifest.json"
    if sha(old_manifest_path) != birth.get("bundle_manifest_sha256"):
        raise ValueError("source bundle differs from immutable birth bundle")
    old = verify_bundle(source_bundle)
    new = verify_bundle(target_bundle)
    if old.get("preflight_status") != "imports_checked" or new.get("preflight_status") != "imports_checked":
        raise ValueError("source and target bundles require checked import preflight")
    changed = validate_transition(old, new, load(source_bundle / "runtime/runtime_config.json"),
                                  load(target_bundle / "runtime/runtime_config.json"), source_bundle, target_bundle)
    events_path = in_run(root / "logs/integration/events.jsonl", root)
    events = events_path.read_bytes()
    stopped, prefix = validate_stop_evidence(stop_evidence, root, birth, events)
    rows = [json.loads(line) for line in prefix.splitlines()]
    if [event.get("phase") for event in rows] != list(RETAINED_PHASES):
        raise ValueError("retained producer prefix is not the exact P01-through-P08 sequence")
    old_code = code_identity(old); retained_outputs = {}
    for event in rows:
        if event.get("returncode") != 0 or "error" in event or event.get("run_identity") != run_identity(birth) or event.get("code_sha256") != old_code:
            raise ValueError("retained producer receipt has foreign/failed identity")
        for name, digest in event.get("output_sha256", {}).items():
            path = in_run(name, root)
            if sha(path) != digest: raise ValueError("retained producer output changed: " + name)
            retained_outputs[name] = digest
    suffix = [json.loads(line) for line in events[len(prefix):].splitlines()]
    if not suffix or any(e.get("phase") not in INVALIDATED_PHASES for e in suffix):
        raise ValueError("event suffix is empty or contains an unbounded phase")
    if any(e.get("returncode") != 0 or "error" in e or e.get("run_identity") != run_identity(birth) or e.get("code_sha256") != old_code for e in suffix):
        raise ValueError("invalidated suffix receipt has foreign/failed identity")
    plan_path = in_run(root / "measurements/object_candidates/object_plan_agent.json", root)
    # SceneLineage checks the current P05 visual judgment and all referenced old-code evidence.
    from rgb_scene_runtime import SceneLineage
    lineage = SceneLineage(root, source_bundle, source_bundle / "runtime/runtime_config.json")
    lineage.require(plan_path)
    objects = load(plan_path).get("objects", [])
    if len(objects) != 1: raise ValueError("bounded recovery requires the existing single object plan")
    obj = _safe_id(objects[0]["object_id"], "object ID")
    track = _safe_id(objects[0].get("track_id", obj), "track ID")
    case = _safe_id(birth["case"], "case ID")
    mapping, remove_dirs = inventory_invalid_suffix(root, suffix, case, obj, track)
    if not mapping: raise ValueError("completed invalid suffix has no files to archive")
    package.mkdir(parents=True)
    snapshots = {
        "source_bundle_manifest.json": old_manifest_path.read_bytes(),
        "source_runtime_config.json": (source_bundle / "runtime/runtime_config.json").read_bytes(),
        "retained_events.jsonl": prefix,
        "stopped_run_evidence.json": Path(stop_evidence).read_bytes(),
    }
    for name, data in snapshots.items(): _write_new(package / name, data)
    archive = {"schema": "rgb_scene_invalid_suffix_archive_v1", "run_root": str(root),
               "transition": TRANSITION, "files": mapping, "remove_dirs": remove_dirs}
    _write_new(package / "archive_manifest.json", archive)
    package_hashes = {name: sha(package / name) for name in snapshots}
    recovery = {
        "schema": SCHEMA, "mode": "same_run_recovery", "transition": TRANSITION,
        "run_identity": run_identity(birth), "input_contract_sha256": sha(contract_path),
        "source_bundle": str(source_bundle), "source_bundle_manifest_sha256": sha(old_manifest_path),
        "target_bundle": str(target_bundle), "target_bundle_manifest_sha256": sha(new_manifest_path),
        "changed_bundle_files": changed, "external_assets_equivalent": True,
        "retained_phases": list(RETAINED_PHASES), "retained_events_bytes": len(prefix),
        "retained_events_sha256": hashlib.sha256(prefix).hexdigest(),
        "original_events_bytes": len(events), "original_events_sha256": hashlib.sha256(events).hexdigest(),
        "retained_outputs_sha256": retained_outputs, "p05_judgment_path": str(plan_path),
        "p05_judgment_sha256": sha(plan_path), "object_id": obj, "track_id": track,
        "archive_manifest_sha256": sha(package / "archive_manifest.json"),
        "native_handoff_path": str(root / RECOVERY_REL / "native_recovery_handoff.md"),
        "package_files_sha256": package_hashes,
        "stop_identity": {k: stopped[k] for k in ("native_actor_id", "workflow_id", "status", "actor_process_terminal", "actor_resumable")},
        "recovery_events_path": str(root / RECOVERY_EVENTS),
        "claim_scope": "P08 retained; all P09-propose-and-later products invalidated and re-produced under the target bundle",
    }
    _write_new(package / "recovery_contract.json", recovery)
    verify_prepared_recovery(package, root=root, target_bundle=target_bundle, process_rows=process_rows)
    return package / "recovery_contract.json"


def verify_prepared_recovery(package, *, root=None, target_bundle=None, process_rows=None):
    package, contract, archive = _load_package(package)
    root = Path(root or contract["run_identity"]["run_root"]).resolve()
    target_bundle = Path(target_bundle or contract["target_bundle"]).resolve()
    require_no_active_owned_processes(root, process_rows)
    birth_path = in_run(root / "input/runtime_input_contract.json", root)
    birth = load(birth_path)
    if run_identity(birth) != contract["run_identity"] or sha(birth_path) != contract["input_contract_sha256"]:
        raise ValueError("birth run/input identity changed since recovery preparation")
    if str(target_bundle) != contract["target_bundle"] or sha(target_bundle / "runtime/bundle_manifest.json") != contract["target_bundle_manifest_sha256"]:
        raise ValueError("target recovery bundle changed")
    old_snapshot = package / "source_bundle_manifest.json"
    source_bundle = Path(contract["source_bundle"]).resolve()
    source_manifest = source_bundle / "runtime/bundle_manifest.json"
    if (sha(old_snapshot) != contract["source_bundle_manifest_sha256"]
            or sha(source_manifest) != contract["source_bundle_manifest_sha256"]):
        raise ValueError("source/birth bundle changed since recovery preparation")
    old = verify_bundle(source_bundle); new = verify_bundle(target_bundle)
    if old != load(old_snapshot): raise ValueError("source bundle snapshot differs")
    changed = validate_transition(old, new, load(package / "source_runtime_config.json"),
                                  load(target_bundle / "runtime/runtime_config.json"), Path(contract["source_bundle"]), target_bundle)
    if changed != contract["changed_bundle_files"]: raise ValueError("recovery bundle change set changed")
    events = in_run(root / "logs/integration/events.jsonl", root).read_bytes()
    _, stopped_prefix = validate_stop_evidence(package / "stopped_run_evidence.json", root, birth, events)
    if stopped_prefix != (package / "retained_events.jsonl").read_bytes():
        raise ValueError("stop-evidence prefix differs from recovery snapshot")
    if len(events) != contract["original_events_bytes"] or hashlib.sha256(events).hexdigest() != contract["original_events_sha256"]:
        raise ValueError("original events changed since recovery preparation")
    prefix = (package / "retained_events.jsonl").read_bytes()
    if events[:len(prefix)] != prefix: raise ValueError("retained old event prefix changed")
    if sha(in_run(contract["p05_judgment_path"], root)) != contract["p05_judgment_sha256"]:
        raise ValueError("retained P05 judgment changed")
    for name, digest in contract["retained_outputs_sha256"].items():
        if sha(in_run(name, root)) != digest: raise ValueError("retained prefix output changed: " + name)
    for row in archive["files"]:
        source = in_run(row["source"], root)
        if str(source.relative_to(root)) != row["source_relative"] or sha(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise ValueError("invalid suffix source changed: " + row["source"])
    return contract


def _journal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def apply_recovery(package, *, process_rows=None):
    package, contract, archive = _load_package(package)
    root = Path(contract["run_identity"]["run_root"]).resolve()
    target = Path(contract["target_bundle"]).resolve()
    recovery_dir = root / RECOVERY_REL
    if (root / COMPLETE_FILE).exists() or (root / RECOVERY_FILE).exists():
        raise ValueError("recovery was already applied")
    require_no_active_owned_processes(root, process_rows)
    # Verify before creating any run-side migration state.
    verify_prepared_recovery(package, root=root, target_bundle=target, process_rows=process_rows)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    journal = root / JOURNAL_FILE
    try:
        for row in archive["files"]:
            source = root / row["source_relative"]
            destination = recovery_dir / row["archive_relative"]
            _journal(journal, {"operation": "move_intent", "source": str(source), "archive": str(destination), "sha256": row["sha256"]})
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if source.exists() or sha(destination) != row["sha256"]:
                    raise ValueError("partial archive has conflicting source/destination: " + str(source))
            else:
                if not source.is_file() or sha(source) != row["sha256"]:
                    raise ValueError("partial archive source missing/changed: " + str(source))
                os.replace(source, destination)
            if source.exists() or sha(destination) != row["sha256"]:
                raise ValueError("archive move verification failed: " + str(source))
            _journal(journal, {"operation": "move_complete", "source": str(source), "archive": str(destination), "sha256": row["sha256"]})
        for relative in archive["remove_dirs"]:
            directory = root / relative
            if directory.exists(): directory.rmdir()
        for name in contract["package_files_sha256"]:
            destination = recovery_dir / name
            if not destination.exists(): shutil.copyfile(package / name, destination)
            if sha(destination) != contract["package_files_sha256"][name]:
                raise ValueError("applied recovery snapshot differs: " + name)
        shutil.copyfile(package / "archive_manifest.json", recovery_dir / "archive_manifest.json")
        if sha(recovery_dir / "archive_manifest.json") != contract["archive_manifest_sha256"]:
            raise ValueError("applied archive manifest differs")
        shutil.copyfile(package / "recovery_contract.json", root / RECOVERY_FILE)
        marker = {"schema": "rgb_scene_recovery_apply_complete_v1", "transition": TRANSITION,
                  "recovery_contract_sha256": sha(root / RECOVERY_FILE),
                  "archive_manifest_sha256": sha(recovery_dir / "archive_manifest.json"),
                  "journal_sha256": sha(journal), "file_count": len(archive["files"])}
        _write_new(root / COMPLETE_FILE, marker)
        verify_applied_recovery(root, target, process_rows=process_rows, check_processes=True)
        return root / RECOVERY_FILE
    except Exception as exc:
        _journal(journal, {"operation": "apply_failed", "error": str(exc)})
        raise


def verify_applied_recovery(root, bundle, *, process_rows=None, check_processes=False):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    if check_processes: require_no_active_owned_processes(root, process_rows)
    birth = load(in_run(root / "input/runtime_input_contract.json", root))
    manifest = verify_bundle(bundle)
    context = recovery_context(root, bundle, birth, manifest)
    if context is None: raise ValueError("applied recovery expected")
    return context


def _validate_recovery_judgment(path, *, root, contract, records, retained_records, obj, track, case):
    """Validate an authored canonical suffix decision against current producers."""
    path = Path(path).resolve()
    payload = load(path)
    provenance = payload.get("visual_judgment_provenance")
    if (not isinstance(provenance, dict)
            or provenance.get("kind") != "current_run_visual_judgment"
            or provenance.get("run_identity") != run_identity(contract)
            or not isinstance(provenance.get("evidence_sha256"), dict)
            or not provenance["evidence_sha256"]):
        raise ValueError("regenerated judgment lacks strict current-run provenance: " + str(path))
    evidence = provenance["evidence_sha256"]

    def producer(name, phase, *, retained=False):
        source = in_run(name, root)
        event = (retained_records if retained else records).get(str(source))
        if (event is None or event.get("phase") != phase
                or str(source) not in event.get("output_sha256", {})
                or evidence.get(str(source)) != sha(source)
                or event["output_sha256"][str(source)] != sha(source)):
            raise ValueError("regenerated judgment evidence has wrong producer/hash: " + str(source))
        return event

    anchor = root / f"state/anchor_decisions/{obj}.json"
    branch = root / f"state/physical_branch_decisions/{obj}.json"
    selection = root / "state/object_selection.json"
    appearance = root / f"state/appearance_decisions/{obj}.json"
    visibility = root / f"state/agent_interaction_judgments/{obj}.json"
    phase = provenance.get("phase")
    if path == anchor:
        if phase != "P09_anchor" or payload.get("object_id") != obj or type(payload.get("selected_anchor_frame_idx")) is not int:
            raise ValueError("regenerated anchor decision identity mismatch")
        proposal = root / f"measurements/object_geometry/anchor_candidates/{obj}/anchor_candidate_proposals.json"
        review = root / f"measurements/object_geometry/anchor_candidates/{obj}/anchor_candidate_review.jpg"
        event = producer(proposal, "P09-propose")
        if producer(review, "P09-propose") is not event:
            raise ValueError("regenerated anchor evidence producer mismatch")
        proposed = load(proposal)
        if (payload["selected_anchor_frame_idx"] not in [r.get("frame_idx") for r in proposed.get("review_candidates", [])]
                or proposed.get("outputs", {}).get("anchor_candidate_review") != str(review)):
            raise ValueError("regenerated anchor selection is absent from current review")
    elif path in {branch, selection}:
        if phase != "P10" or payload.get("object_id") != obj:
            raise ValueError("regenerated P10/selection identity mismatch")
        annotation = root / f"measurements/object_geometry/visible_geometry/{obj}/annotations_v19_visible_geometry.json"
        producer(annotation, "P09")
        current_p09_images = [name for name in evidence if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                              and records.get(name, {}).get("phase") == "P09"]
        if not current_p09_images or not all(evidence[name] == sha(in_run(name, root)) for name in current_p09_images):
            raise ValueError("regenerated P10/selection lacks current P09 image evidence")
        if path == branch and payload.get("physical_branch") != "rigid":
            raise ValueError("bounded rigid recovery requires a new rigid P10 decision")
        if path == selection:
            anchor_value = load(anchor)
            if (payload.get("case") != case or payload.get("track_id") != track
                    or payload.get("anchor_frame") != anchor_value.get("selected_anchor_frame_idx")):
                raise ValueError("regenerated object selection differs from current anchor/plan")
            _validate_recovery_judgment(branch, root=root, contract=contract, records=records,
                                        retained_records=retained_records, obj=obj, track=track, case=case)
    elif path == appearance:
        if phase != "appearance_anchor" or payload.get("object_id") != obj or type(payload.get("selected_frame_idx")) is not int:
            raise ValueError("regenerated appearance decision identity mismatch")
        manifest = root / f"measurements/demo_visual/{obj}/appearance_evidence/appearance_mask_manifest.json"
        event = producer(manifest, "scene_appearance")
        rows = [row for row in load(manifest).get("frames", []) if row.get("frame_idx") == payload["selected_frame_idx"]]
        if len(rows) != 1 or rows[0].get("status") != "available" or producer(rows[0]["appearance_mask_path"], "scene_appearance") is not event:
            raise ValueError("regenerated appearance selection lacks current mask evidence")
        producer(rows[0]["raw_rgb_path"], "P01", retained=True)
    elif path == visibility:
        if phase != "P17_judgment" or payload.get("object_id") != obj:
            raise ValueError("regenerated visibility judgment identity mismatch")
        producer(root / f"state/scene/{obj}/refined_pose.json", "scene_refined_pose")
        raw_images = [name for name in evidence if retained_records.get(name, {}).get("phase") == "P01"
                      and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not raw_images or not all(evidence[name] == sha(in_run(name, root)) for name in raw_images):
            raise ValueError("regenerated visibility judgment lacks retained P01 image evidence")
    else:
        raise ValueError("unrecognized regenerated authored suffix path: " + str(path))
    return True


def _retained_input_seeds(root, retained_rows, contract):
    """Revalidate run-local inputs explicitly recorded by retained producers.

    These are ancestry leaves, not newly produced outputs.  Acceptance is bound
    to the old-code successful receipt, run containment, and current bytes.  A
    HaWoR video copy is additionally required to equal the immutable birth
    video, matching ordinary lineage semantics.
    """
    seeds = {}
    hawor_video_dir = root / "input/hawor_sequence"
    for event in retained_rows:
        inputs = event.get("input_sha256", {})
        if not isinstance(inputs, dict):
            raise ValueError("retained producer input hashes are malformed")
        for name, expected in inputs.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ValueError("retained producer input identity is malformed")
            source = in_run(name, root)
            actual = sha(source)
            if actual != expected:
                raise ValueError("retained producer input changed: " + str(source))
            if source.parent == hawor_video_dir and source.suffix.lower() == ".mp4":
                if expected != contract["input_video_sha256"]:
                    raise ValueError("retained HaWoR raw copy differs from birth video")
            previous = seeds.get(str(source))
            if previous is not None and previous != expected:
                raise ValueError("retained producer input has conflicting hashes: " + str(source))
            seeds[str(source)] = expected
    return seeds


def _validated_recovery_events(*, root, contract, current_code, target_hash, contract_hash,
                               retained_rows, retained_outputs, obj, track, case,
                               accepted_event_digests=(), revision_hash=None,
                               event_bytes=None, validate_live_judgments=True):
    event_path = root / RECOVERY_EVENTS
    retained_records = {}
    for event in retained_rows:
        for name in event.get("output_sha256", {}):
            retained_records[name] = event
    available = dict(retained_outputs)
    available.update(_retained_input_seeds(root, retained_rows, contract))
    available[str(root / "input/runtime_input_contract.json")] = sha(root / "input/runtime_input_contract.json")
    available[contract["input_video"]] = contract["input_video_sha256"]
    p05_path = root / "measurements/object_candidates/object_plan_agent.json"
    available[str(p05_path)] = sha(p05_path)
    records = {}
    authorized = {}
    if event_bytes is None:
        if not event_path.exists():
            return records, retained_records, authorized
        if not event_path.is_file():
            raise ValueError("recovery event log is partial/malformed")
        event_bytes = event_path.read_bytes()
    if not event_bytes.endswith(b"\n"):
        raise ValueError("recovery event log is partial/malformed")
    events = [json.loads(line) for line in event_bytes.splitlines()]
    phases = [event.get("phase") for event in events]
    if tuple(phases) != RECOVERY_PHASE_SEQUENCE[:len(phases)]:
        raise ValueError("recovery producer completion ordering is not the canonical suffix prefix")
    judgment_paths = {
        str(root / f"state/anchor_decisions/{obj}.json"),
        str(root / f"state/physical_branch_decisions/{obj}.json"),
        str(root / "state/object_selection.json"),
        str(root / f"state/appearance_decisions/{obj}.json"),
        str(root / f"state/agent_interaction_judgments/{obj}.json"),
    }
    for event in events:
        if (event.get("returncode") != 0 or "error" in event
                or event.get("run_identity") != run_identity(contract)
                or event.get("input_contract_sha256") != sha(root / "input/runtime_input_contract.json")
                or not event_code_is_valid(event, current_code, {
                    "retained_event_digests": set(), "accepted_revision_event_digests": set(accepted_event_digests),
                    "execution_bundle_sha256": target_hash, "contract_sha256": contract_hash,
                    "validator_revision_sha256": revision_hash,
                })):
            raise ValueError("foreign/unbound/failed recovery producer event")
        for name, expected in event.get("input_sha256", {}).items():
            source = in_run(name, root)
            if str(source) in judgment_paths:
                _validate_recovery_judgment(source, root=root, contract=contract, records=records,
                                            retained_records=retained_records, obj=obj, track=track, case=case)
                available[str(source)] = sha(source)
            if available.get(str(source)) != expected or sha(source) != expected:
                raise ValueError("recovery producer input lacks completed bound ancestor: " + str(source))
        output_hashes = event.get("output_sha256")
        if not isinstance(output_hashes, dict) or not output_hashes:
            raise ValueError("successful recovery producer has no recorded outputs")
        for name, expected in output_hashes.items():
            output = in_run(name, root)
            if sha(output) != expected:
                raise ValueError("recovery producer output changed: " + str(output))
            if str(output) in records:
                raise ValueError("recovery output has multiple producers: " + str(output))
            records[str(output)] = event
            available[str(output)] = expected
            authorized[str(output)] = expected
        if not set(map(str, event.get("outputs", []))) <= set(output_hashes):
            raise ValueError("recovery producer declared output lacks recorded SHA")
    # Historical prefix verification must not inspect decisions created after
    # that prefix. Judgments consumed by a receipt above are still validated at
    # the point where its input_sha256 is checked.
    if validate_live_judgments:
        for judgment in judgment_paths:
            path = Path(judgment)
            if path.exists():
                _validate_recovery_judgment(path, root=root, contract=contract, records=records,
                                            retained_records=retained_records, obj=obj, track=track, case=case)
                authorized[judgment] = sha(path)
    return records, retained_records, authorized


def _validator_revision_paths(root, manifest_hash):
    if (not isinstance(manifest_hash, str) or len(manifest_hash) != 64
            or any(c not in "0123456789abcdef" for c in manifest_hash)):
        raise ValueError("validator revision manifest SHA is malformed")
    directory = root / VALIDATOR_REVISION_DIR
    return directory / f"{manifest_hash}.json", directory / f"{manifest_hash}.terminal_actor.json"


def _validate_validator_terminal_evidence(value, *, root, prior_bundle, prior_manifest, contract):
    required = {
        "state": "failed", "process_terminal": "observed",
        "completed_phases": ["P09_DEPTH_ORDER"],
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError("validator revision terminal actor evidence mismatch: " + key)
    try:
        uuid.UUID(value["actor"]); uuid.UUID(value["workflow"])
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("validator revision actor/workflow identity malformed") from exc
    source = value.get("source_state")
    if not isinstance(source, dict) or source.get("cwd") != str(prior_bundle):
        raise ValueError("validator revision prior actor bundle mismatch")
    if prior_manifest.get("source_commit") is not None and source.get("head") != prior_manifest["source_commit"]:
        raise ValueError("validator revision prior actor source commit mismatch")
    if source.get("git_status") not in (None, ""):
        raise ValueError("validator revision prior actor source was dirty")
    if not isinstance(value.get("pipeline_error"), str) or "P09-propose" not in value["pipeline_error"]:
        raise ValueError("validator revision terminal evidence lacks P09-propose failure")
    if (value.get("alias_sha256") != contract["input_video_sha256"]
            or value.get("birth_video_sha256") != contract["input_video_sha256"]):
        raise ValueError("validator revision terminal evidence birth alias mismatch")
    return {"actor": value["actor"], "workflow": value["workflow"],
            "state": value["state"], "process_terminal": value["process_terminal"]}


def _validate_validator_revision(root, bundle, manifest, contract, recovery_contract,
                                 recovery_contract_path, prior_bundle, prior_manifest,
                                 recovery_events):
    target_hash = sha(bundle / "runtime/bundle_manifest.json")
    revision_path, terminal_path = _validator_revision_paths(root, target_hash)
    if not revision_path.is_file() or not terminal_path.is_file():
        raise ValueError("missing fixed validator revision contract")
    revision = load(revision_path)
    prefix_bytes = revision.get("recovery_events_bytes")
    if type(prefix_bytes) is not int or prefix_bytes <= 0 or prefix_bytes > len(recovery_events):
        raise ValueError("validator revision recovery prefix length malformed")
    recovery_prefix = recovery_events[:prefix_bytes]
    expected = {
        "schema": VALIDATOR_REVISION_SCHEMA,
        "transition": "P09_DEPTH_ORDER_VALIDATOR_REVISION",
        "run_identity": run_identity(contract),
        "input_contract_sha256": sha(root / "input/runtime_input_contract.json"),
        "original_recovery_contract_path": str(recovery_contract_path),
        "original_recovery_contract_sha256": sha(recovery_contract_path),
        "prior_execution_bundle": str(prior_bundle),
        "prior_execution_bundle_manifest_sha256": sha(prior_bundle / "runtime/bundle_manifest.json"),
        "new_execution_bundle": str(bundle),
        "new_execution_bundle_manifest_sha256": target_hash,
        "recovery_events_path": str(root / RECOVERY_EVENTS),
        "recovery_events_bytes": prefix_bytes,
        "recovery_events_sha256": hashlib.sha256(recovery_prefix).hexdigest(),
        "resume_phase": VALIDATOR_REVISION_RESUME_PHASE,
        "terminal_actor_evidence_path": str(terminal_path),
        "terminal_actor_evidence_sha256": sha(terminal_path),
        "birth_input_video": contract["input_video"],
        "birth_input_video_sha256": contract["input_video_sha256"],
    }
    for key, wanted in expected.items():
        if revision.get(key) != wanted:
            raise ValueError("validator revision identity mismatch: " + key)
    if not recovery_prefix.endswith(b"\n"):
        raise ValueError("validator revision recovery prefix is partial")
    rows = [json.loads(line) for line in recovery_prefix.splitlines()]
    if len(rows) != 1 or rows[0].get("phase") != "P09_DEPTH_ORDER":
        raise ValueError("validator revision requires exactly completed P09_DEPTH_ORDER")
    receipt = {"phase": "P09_DEPTH_ORDER", "event_sha256": event_digest(rows[0]),
               "output_sha256": rows[0].get("output_sha256")}
    if revision.get("accepted_prior_receipts") != [receipt]:
        raise ValueError("validator revision accepted prior receipt differs")
    changed = validate_validator_transition(
        prior_manifest, manifest, load(prior_bundle / "runtime/runtime_config.json"),
        load(bundle / "runtime/runtime_config.json"), prior_bundle, bundle)
    if revision.get("changed_bundle_files") != changed:
        raise ValueError("validator revision bundle change set differs")
    terminal = load(terminal_path)
    terminal_identity = _validate_validator_terminal_evidence(
        terminal, root=root, prior_bundle=prior_bundle,
        prior_manifest=prior_manifest, contract=contract)
    if revision.get("terminal_actor_identity") != terminal_identity:
        raise ValueError("validator revision terminal actor identity differs")
    return revision_path, revision, {receipt["event_sha256"]}


def _validate_second_validator_terminal_evidence(value, *, root, prior_bundle, prior_manifest,
                                                 contract, recovery_events):
    """Bind the observed post-proposal stop; this is the only supported second checkpoint."""
    required = {"process_terminal": "observed", "completed_phases": ["P09_DEPTH_ORDER", "P09-propose"],
                "run_root": str(root)}
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError("second validator revision terminal evidence mismatch: " + key)
    actor = value.get("actor")
    workflow = value.get("workflow")
    try:
        uuid.UUID(actor); uuid.UUID(workflow)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("second validator revision actor/workflow identity malformed") from exc
    if value.get("actor_state") != "complete" or value.get("task_result") != "blocked before P09 producer":
        raise ValueError("second validator revision requires the terminal pre-P09 blocker")
    if not isinstance(value.get("error"), str) or not value["error"]:
        raise ValueError("second validator revision terminal evidence lacks recorded failure")
    source = value.get("source_state")
    if (value.get("runtime_cwd") != str(prior_bundle) or not isinstance(source, dict)
            or source.get("status") not in (None, "")):
        raise ValueError("second validator revision prior actor bundle/source mismatch")
    if prior_manifest.get("source_commit") is not None and source.get("head") != prior_manifest["source_commit"]:
        raise ValueError("second validator revision prior actor source commit mismatch")
    if value.get("recovery_event_bytes") != len(recovery_events) or value.get("recovery_event_sha256") != hashlib.sha256(recovery_events).hexdigest():
        raise ValueError("second validator revision terminal recovery log identity mismatch")
    if value.get("P09_producer_executed") is not False or value.get("SAM3D_executed") is not False:
        raise ValueError("second validator revision terminal phase boundary mismatch")
    if value.get("owned_stage_processes") != []:
        raise ValueError("second validator revision terminal evidence recorded active processes")
    return {"actor": actor, "workflow": workflow, "state": value["actor_state"],
            "process_terminal": value["process_terminal"]}


def _validate_second_validator_revision(root, bundle, manifest, contract, recovery_contract_path,
                                         prior_bundle, prior_manifest, recovery_events,
                                         first_revision_path, first_revision):
    target_hash = sha(bundle / "runtime/bundle_manifest.json")
    revision_path, terminal_path = _validator_revision_paths(root, target_hash)
    handoff_path = revision_path.with_suffix(".native_handoff.md")
    if not revision_path.is_file() or not terminal_path.is_file() or not handoff_path.is_file():
        raise ValueError("missing fixed second validator revision contract")
    revision = load(revision_path)
    prefix_bytes = revision.get("recovery_events_bytes")
    if type(prefix_bytes) is not int or prefix_bytes != len(recovery_events) or not recovery_events.endswith(b"\n"):
        raise ValueError("second validator revision recovery prefix length malformed")
    rows = [json.loads(line) for line in recovery_events.splitlines()]
    if [row.get("phase") for row in rows] != ["P09_DEPTH_ORDER", "P09-propose"]:
        raise ValueError("second validator revision requires exactly two successful frontend receipts")
    receipts = [{"phase": row["phase"], "event_sha256": event_digest(row),
                 "output_sha256": row.get("output_sha256")} for row in rows]
    anchor_path = root / f"state/anchor_decisions/{load(root / 'measurements/object_candidates/object_plan_agent.json')['objects'][0]['object_id']}.json"
    expected = {
        "schema": SECOND_VALIDATOR_REVISION_SCHEMA,
        "transition": "P09_ANCHOR_VALIDATOR_REVISION",
        "run_identity": run_identity(contract),
        "input_contract_sha256": sha(root / "input/runtime_input_contract.json"),
        "original_recovery_contract_path": str(recovery_contract_path),
        "original_recovery_contract_sha256": sha(recovery_contract_path),
        "prior_validator_revision_path": str(first_revision_path),
        "prior_validator_revision_sha256": sha(first_revision_path),
        "prior_execution_bundle": str(prior_bundle),
        "prior_execution_bundle_manifest_sha256": sha(prior_bundle / "runtime/bundle_manifest.json"),
        "new_execution_bundle": str(bundle),
        "new_execution_bundle_manifest_sha256": target_hash,
        "recovery_events_path": str(root / RECOVERY_EVENTS),
        "recovery_events_bytes": len(recovery_events),
        "recovery_events_sha256": hashlib.sha256(recovery_events).hexdigest(),
        "accepted_prior_receipts": receipts,
        "anchor_decision_path": str(anchor_path),
        "anchor_decision_sha256": sha(anchor_path),
        "resume_phase": SECOND_VALIDATOR_REVISION_RESUME_PHASE,
        "terminal_actor_evidence_path": str(terminal_path),
        "terminal_actor_evidence_sha256": sha(terminal_path),
        "native_handoff_path": str(handoff_path),
        "native_handoff_sha256": sha(handoff_path),
        "birth_input_video": contract["input_video"],
        "birth_input_video_sha256": contract["input_video_sha256"],
    }
    for key, wanted in expected.items():
        if revision.get(key) != wanted:
            raise ValueError("second validator revision identity mismatch: " + key)
    changed = validate_validator_transition(prior_manifest, manifest,
        load(prior_bundle / "runtime/runtime_config.json"), load(bundle / "runtime/runtime_config.json"),
        prior_bundle, bundle, require_lineage_change=False)
    if revision.get("changed_bundle_files") != changed:
        raise ValueError("second validator revision bundle change set differs")
    terminal_identity = _validate_second_validator_terminal_evidence(
        load(terminal_path), root=root, prior_bundle=prior_bundle, prior_manifest=prior_manifest,
        contract=contract, recovery_events=recovery_events)
    if revision.get("terminal_actor_identity") != terminal_identity:
        raise ValueError("second validator revision terminal actor identity differs")
    return revision_path, revision, {receipt["event_sha256"] for receipt in receipts}


def _prepare_second_validator_revision(root, prior_bundle, new_bundle, terminal_actor_evidence,
                                       contract, prior_manifest, prior_context):
    first_revision = prior_context.get("validator_revision")
    first_revision_path, _ = _validator_revision_paths(
        root, sha(prior_bundle / "runtime/bundle_manifest.json"))
    if (not isinstance(first_revision, dict) or first_revision.get("schema") != VALIDATOR_REVISION_SCHEMA
            or first_revision.get("resume_phase") != VALIDATOR_REVISION_RESUME_PHASE):
        raise ValueError("second validator revision requires the fixed first validator revision")
    manifest = verify_bundle(new_bundle)
    changed = validate_validator_transition(prior_manifest, manifest,
        load(prior_bundle / "runtime/runtime_config.json"), load(new_bundle / "runtime/runtime_config.json"),
        prior_bundle, new_bundle, require_lineage_change=False)
    events_path = in_run(root / RECOVERY_EVENTS, root)
    events = events_path.read_bytes()
    if not events.endswith(b"\n"):
        raise ValueError("second validator revision recovery event log is partial")
    rows = [json.loads(line) for line in events.splitlines()]
    if [row.get("phase") for row in rows] != ["P09_DEPTH_ORDER", "P09-propose"]:
        raise ValueError("second validator revision requires exactly P09_DEPTH_ORDER and P09-propose")
    terminal = load(terminal_actor_evidence)
    terminal_identity = _validate_second_validator_terminal_evidence(
        terminal, root=root, prior_bundle=prior_bundle, prior_manifest=prior_manifest,
        contract=contract, recovery_events=events)
    plan = load(root / "measurements/object_candidates/object_plan_agent.json")
    if len(plan.get("objects", [])) != 1:
        raise ValueError("second validator revision requires the retained single-object plan")
    obj = _safe_id(plan["objects"][0]["object_id"], "object ID")
    anchor_path = root / f"state/anchor_decisions/{obj}.json"
    # Full prior-context verification has already validated this judgment and
    # its proposal/review-image ancestry. Preserve its exact bytes here.
    anchor_hash = sha(anchor_path)
    target_hash = sha(new_bundle / "runtime/bundle_manifest.json")
    revision_path, terminal_path = _validator_revision_paths(root, target_hash)
    handoff_path = revision_path.with_suffix(".native_handoff.md")
    if revision_path.exists() or terminal_path.exists() or handoff_path.exists():
        raise ValueError("second validator revision target already exists")
    _write_new(terminal_path, terminal_actor_evidence.read_bytes())
    command = (f'"/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python" scripts/run_rgb_scene_stage.py '
               f'--stage frontend --phase P09 --run-root "{root}" '
               f'--runtime-config runtime/runtime_config.json --gpu 5 '
               f'--recovery-contract "{root / RECOVERY_FILE}"\n')
    _write_new(handoff_path, ("# Native P09 continuation\n\nStart one new native actor. Do not rerun P09_DEPTH_ORDER or P09-propose.\n\n```bash\n" + command + "```\n").encode())
    receipts = [{"phase": row["phase"], "event_sha256": event_digest(row),
                 "output_sha256": row.get("output_sha256")} for row in rows]
    revision = {
        "schema": SECOND_VALIDATOR_REVISION_SCHEMA,
        "transition": "P09_ANCHOR_VALIDATOR_REVISION",
        "run_identity": run_identity(contract),
        "input_contract_sha256": sha(root / "input/runtime_input_contract.json"),
        "original_recovery_contract_path": str(root / RECOVERY_FILE),
        "original_recovery_contract_sha256": sha(root / RECOVERY_FILE),
        "prior_validator_revision_path": str(first_revision_path),
        "prior_validator_revision_sha256": sha(first_revision_path),
        "prior_execution_bundle": str(prior_bundle),
        "prior_execution_bundle_manifest_sha256": sha(prior_bundle / "runtime/bundle_manifest.json"),
        "new_execution_bundle": str(new_bundle),
        "new_execution_bundle_manifest_sha256": target_hash,
        "changed_bundle_files": changed,
        "recovery_events_path": str(events_path),
        "recovery_events_bytes": len(events),
        "recovery_events_sha256": hashlib.sha256(events).hexdigest(),
        "accepted_prior_receipts": receipts,
        "anchor_decision_path": str(anchor_path),
        "anchor_decision_sha256": anchor_hash,
        "resume_phase": SECOND_VALIDATOR_REVISION_RESUME_PHASE,
        "terminal_actor_evidence_path": str(terminal_path),
        "terminal_actor_evidence_sha256": sha(terminal_path),
        "terminal_actor_identity": terminal_identity,
        "native_handoff_path": str(handoff_path),
        "native_handoff_sha256": sha(handoff_path),
        "birth_input_video": contract["input_video"],
        "birth_input_video_sha256": contract["input_video_sha256"],
        "claim_scope": "second and final validator-only continuation after exact depth-order/proposal receipts and current anchor; resume at P09 without recomputation",
    }
    _write_new(revision_path, revision)
    revised = recovery_context(root, new_bundle, contract, manifest)
    if revised.get("validator_revision_sha256") != sha(revision_path):
        raise ValueError("installed second validator revision did not activate")
    return revision_path


def prepare_validator_revision(root, prior_bundle, new_bundle, terminal_actor_evidence, *, process_rows=None):
    """Install the one bounded validator-only continuation sidecar, create-only."""
    root, prior_bundle, new_bundle = map(lambda p: Path(p).resolve(), (root, prior_bundle, new_bundle))
    terminal_actor_evidence = Path(terminal_actor_evidence).resolve()
    require_no_active_owned_processes(root, process_rows)
    contract = load(in_run(root / "input/runtime_input_contract.json", root))
    prior_manifest = verify_bundle(prior_bundle)
    context = recovery_context(root, prior_bundle, contract, prior_manifest)
    if context is None:
        raise ValueError("validator revision requires an applied recovery bundle")
    if context.get("validator_revision") is not None:
        return _prepare_second_validator_revision(
            root, prior_bundle, new_bundle, terminal_actor_evidence,
            contract, prior_manifest, context)
    recovery_contract_path = root / RECOVERY_FILE
    recovery_contract = load(recovery_contract_path)
    if (recovery_contract.get("target_bundle") != str(prior_bundle)
            or recovery_contract.get("target_bundle_manifest_sha256") != sha(prior_bundle / "runtime/bundle_manifest.json")):
        raise ValueError("prior execution bundle is not the applied recovery bundle")
    manifest = verify_bundle(new_bundle)
    changed = validate_validator_transition(
        prior_manifest, manifest, load(prior_bundle / "runtime/runtime_config.json"),
        load(new_bundle / "runtime/runtime_config.json"), prior_bundle, new_bundle)
    events_path = in_run(root / RECOVERY_EVENTS, root)
    events = events_path.read_bytes()
    rows = [json.loads(line) for line in events.splitlines()]
    if len(rows) != 1 or rows[0].get("phase") != "P09_DEPTH_ORDER":
        raise ValueError("validator revision requires exactly completed P09_DEPTH_ORDER")
    terminal = load(terminal_actor_evidence)
    terminal_identity = _validate_validator_terminal_evidence(
        terminal, root=root, prior_bundle=prior_bundle,
        prior_manifest=prior_manifest, contract=contract)
    target_hash = sha(new_bundle / "runtime/bundle_manifest.json")
    revision_path, terminal_path = _validator_revision_paths(root, target_hash)
    if revision_path.exists() or terminal_path.exists():
        raise ValueError("validator revision target already exists")
    _write_new(terminal_path, terminal_actor_evidence.read_bytes())
    receipt = {"phase": "P09_DEPTH_ORDER", "event_sha256": event_digest(rows[0]),
               "output_sha256": rows[0].get("output_sha256")}
    revision = {
        "schema": VALIDATOR_REVISION_SCHEMA,
        "transition": "P09_DEPTH_ORDER_VALIDATOR_REVISION",
        "run_identity": run_identity(contract),
        "input_contract_sha256": sha(root / "input/runtime_input_contract.json"),
        "original_recovery_contract_path": str(recovery_contract_path),
        "original_recovery_contract_sha256": sha(recovery_contract_path),
        "prior_execution_bundle": str(prior_bundle),
        "prior_execution_bundle_manifest_sha256": sha(prior_bundle / "runtime/bundle_manifest.json"),
        "new_execution_bundle": str(new_bundle),
        "new_execution_bundle_manifest_sha256": target_hash,
        "changed_bundle_files": changed,
        "recovery_events_path": str(events_path),
        "recovery_events_bytes": len(events),
        "recovery_events_sha256": hashlib.sha256(events).hexdigest(),
        "accepted_prior_receipts": [receipt],
        "resume_phase": VALIDATOR_REVISION_RESUME_PHASE,
        "terminal_actor_evidence_path": str(terminal_path),
        "terminal_actor_evidence_sha256": sha(terminal_path),
        "terminal_actor_identity": terminal_identity,
        "birth_input_video": contract["input_video"],
        "birth_input_video_sha256": contract["input_video_sha256"],
        "claim_scope": "validator-only continuation after exact successful P09_DEPTH_ORDER receipt; resume at P09-propose",
    }
    _write_new(revision_path, revision)
    # Revalidate the installed bytes and all old ancestry before returning.
    revised = recovery_context(root, new_bundle, contract, manifest)
    if revised.get("validator_revision_sha256") != sha(revision_path):
        raise ValueError("installed validator revision did not activate")
    return revision_path


def verify_validator_revision(root, new_bundle, *, process_rows=None, check_processes=False):
    root, new_bundle = Path(root).resolve(), Path(new_bundle).resolve()
    if check_processes:
        require_no_active_owned_processes(root, process_rows)
    contract = load(in_run(root / "input/runtime_input_contract.json", root))
    context = recovery_context(root, new_bundle, contract, verify_bundle(new_bundle))
    if context is None or context.get("validator_revision") is None:
        raise ValueError("fixed validator revision expected")
    return context


def recovery_context(root, bundle, contract, manifest):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    target_hash = sha(bundle / "runtime/bundle_manifest.json")
    if target_hash == contract.get("bundle_manifest_sha256"):
        return None
    path = root / RECOVERY_FILE
    complete_path = root / COMPLETE_FILE
    if not path.is_file() or not complete_path.is_file():
        raise ValueError("incomplete or missing explicit P08 recovery apply")
    value = load(path); complete = load(complete_path)
    prior_bundle = Path(value.get("target_bundle", "")).resolve()
    prior_hash = value.get("target_bundle_manifest_sha256")
    expected = {
        "schema": SCHEMA, "mode": "same_run_recovery", "transition": TRANSITION,
        "run_identity": run_identity(contract), "input_contract_sha256": sha(root / "input/runtime_input_contract.json"),
        "target_bundle": str(prior_bundle), "target_bundle_manifest_sha256": prior_hash,
        "recovery_events_path": str(root / RECOVERY_EVENTS),
        "native_handoff_path": str(root / RECOVERY_REL / "native_recovery_handoff.md"),
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted: raise ValueError("depth-order recovery identity mismatch: " + key)
    if sha(prior_bundle / "runtime/bundle_manifest.json") != prior_hash:
        raise ValueError("applied recovery execution bundle changed")
    prior_manifest = verify_bundle(prior_bundle)
    if complete.get("schema") != "rgb_scene_recovery_apply_complete_v1" or complete.get("transition") != TRANSITION or complete.get("recovery_contract_sha256") != sha(path):
        raise ValueError("depth-order recovery completion marker mismatch")
    journal_path = in_run(root / JOURNAL_FILE, root)
    if complete.get("journal_sha256") != sha(journal_path):
        raise ValueError("recovery apply journal changed")
    directory = root / RECOVERY_REL
    archive_path = in_run(directory / "archive_manifest.json", root)
    if sha(archive_path) != value["archive_manifest_sha256"] or complete.get("archive_manifest_sha256") != sha(archive_path):
        raise ValueError("applied invalid-suffix archive manifest changed")
    for name, digest in value["package_files_sha256"].items():
        snapshot = in_run(directory / name, root)
        if sha(snapshot) != digest: raise ValueError("applied recovery snapshot changed: " + name)
    old = load(directory / "source_bundle_manifest.json")
    if sha(directory / "source_bundle_manifest.json") != value["source_bundle_manifest_sha256"] or sha(Path(value["source_bundle"]) / "runtime/bundle_manifest.json") != value["source_bundle_manifest_sha256"]:
        raise ValueError("birth producer bundle/snapshot changed")
    changed = validate_transition(old, prior_manifest, load(directory / "source_runtime_config.json"),
                                  load(prior_bundle / "runtime/runtime_config.json"), Path(value["source_bundle"]), prior_bundle)
    if changed != value["changed_bundle_files"]: raise ValueError("applied recovery change set differs")
    events = in_run(root / "logs/integration/events.jsonl", root).read_bytes()
    if len(events) != value["original_events_bytes"] or hashlib.sha256(events).hexdigest() != value["original_events_sha256"]:
        raise ValueError("original event log changed after recovery")
    prefix = (directory / "retained_events.jsonl").read_bytes()
    if len(prefix) != value["retained_events_bytes"] or hashlib.sha256(prefix).hexdigest() != value["retained_events_sha256"] or events[:len(prefix)] != prefix:
        raise ValueError("retained old event prefix changed")
    old_code = code_identity(old); retained_digests = set()
    rows = [json.loads(line) for line in prefix.splitlines()]
    if [row.get("phase") for row in rows] != list(RETAINED_PHASES): raise ValueError("retained phase sequence changed")
    for event in rows:
        if event.get("returncode") != 0 or "error" in event or event.get("run_identity") != run_identity(contract) or event.get("code_sha256") != old_code:
            raise ValueError("retained old-code receipt invalid")
        retained_digests.add(event_digest(event))
    for name, digest in value["retained_outputs_sha256"].items():
        if sha(in_run(name, root)) != digest: raise ValueError("retained prefix output changed: " + name)
    if sha(in_run(value["p05_judgment_path"], root)) != value["p05_judgment_sha256"]:
        raise ValueError("retained P05 judgment changed")

    # First validate the existing suffix against the immutable recovery bundle.
    # This proves the accepted P09_DEPTH_ORDER receipt and all of its outputs and
    # ancestors before a validator-only revision may preserve that exact event.
    recovery_event_bytes = b""
    recovery_event_path = root / RECOVERY_EVENTS
    if recovery_event_path.exists():
        recovery_event_bytes = recovery_event_path.read_bytes()
    base_event_bytes = None
    preliminary = None
    if target_hash != prior_hash:
        preliminary_path, _ = _validator_revision_paths(root, target_hash)
        if not preliminary_path.is_file():
            raise ValueError("missing fixed validator revision contract")
        preliminary = load(preliminary_path)
        if preliminary.get("schema") == SECOND_VALIDATOR_REVISION_SCHEMA:
            first_bundle = Path(preliminary.get("prior_execution_bundle", "")).resolve()
            first_manifest = verify_bundle(first_bundle)
            first_path, first_revision, first_accepted = _validate_validator_revision(
                root, first_bundle, first_manifest, contract, value, path, prior_bundle,
                prior_manifest, recovery_event_bytes)
            count = first_revision.get("recovery_events_bytes")
        else:
            count = preliminary.get("recovery_events_bytes")
        if type(count) is not int or count <= 0 or count > len(recovery_event_bytes):
            raise ValueError("validator revision recovery prefix length malformed")
        base_event_bytes = recovery_event_bytes[:count]
    base_records, retained_records, authorized_active = _validated_recovery_events(
        root=root, contract=contract, current_code=code_identity(prior_manifest), target_hash=prior_hash,
        contract_hash=sha(path), retained_rows=rows,
        retained_outputs=value["retained_outputs_sha256"], obj=value["object_id"],
        track=value["track_id"], case=contract["case"], event_bytes=base_event_bytes,
        validate_live_judgments=(target_hash == prior_hash),
    )
    revision_path = None; revision = None; accepted_revision_digests = set()
    if target_hash != prior_hash and preliminary.get("schema") == SECOND_VALIDATOR_REVISION_SCHEMA:
        first_prefix_count = preliminary.get("recovery_events_bytes")
        if type(first_prefix_count) is not int or first_prefix_count != len(recovery_event_bytes):
            raise ValueError("second validator revision prefix differs from terminal event log")
        _validated_recovery_events(
            root=root, contract=contract, current_code=code_identity(first_manifest),
            target_hash=sha(first_bundle / "runtime/bundle_manifest.json"), contract_hash=sha(path),
            retained_rows=rows, retained_outputs=value["retained_outputs_sha256"],
            obj=value["object_id"], track=value["track_id"], case=contract["case"],
            accepted_event_digests=first_accepted, revision_hash=sha(first_path),
            event_bytes=recovery_event_bytes, validate_live_judgments=True)
        revision_path, revision, accepted_revision_digests = _validate_second_validator_revision(
            root, bundle, manifest, contract, path, first_bundle, first_manifest,
            recovery_event_bytes, first_path, first_revision)
        recovery_records, retained_records, authorized_active = _validated_recovery_events(
            root=root, contract=contract, current_code=code_identity(manifest), target_hash=target_hash,
            contract_hash=sha(path), retained_rows=rows,
            retained_outputs=value["retained_outputs_sha256"], obj=value["object_id"],
            track=value["track_id"], case=contract["case"],
            accepted_event_digests=accepted_revision_digests, revision_hash=sha(revision_path))
    elif target_hash != prior_hash:
        revision_path, revision, accepted_revision_digests = _validate_validator_revision(
            root, bundle, manifest, contract, value, path, prior_bundle, prior_manifest,
            recovery_event_bytes)
        recovery_records, retained_records, authorized_active = _validated_recovery_events(
            root=root, contract=contract, current_code=code_identity(manifest), target_hash=target_hash,
            contract_hash=sha(path), retained_rows=rows,
            retained_outputs=value["retained_outputs_sha256"], obj=value["object_id"],
            track=value["track_id"], case=contract["case"],
            accepted_event_digests=accepted_revision_digests,
            revision_hash=sha(revision_path))
    else:
        recovery_records = base_records
    archive = load(archive_path)
    if complete.get("file_count") != len(archive["files"]): raise ValueError("archive completion count differs")
    for row in archive["files"]:
        source = root / row["source_relative"]
        archived = directory / row["archive_relative"]
        if source.exists():
            expected = authorized_active.get(str(source))
            if expected is None or sha(source) != expected:
                raise ValueError("stale invalidated suffix restored as active: " + str(source))
        if not archived.is_file() or sha(archived) != row["sha256"] or archived.stat().st_size != row["bytes"]:
            raise ValueError("invalidated archive file missing/changed: " + str(archived))
    failure = root / RECOVERY_REL / "failure.json"
    if failure.exists(): raise ValueError("recovery already failed; relaunch is forbidden")
    return {"contract": value, "contract_sha256": sha(path), "execution_bundle_sha256": target_hash,
            "source_code": old_code, "retained_event_digests": retained_digests,
            "accepted_revision_event_digests": accepted_revision_digests,
            "validator_revision": revision,
            "validator_revision_sha256": sha(revision_path) if revision_path else None,
            "event_sources": [directory / "retained_events.jsonl", root / RECOVERY_EVENTS],
            "events_path": root / RECOVERY_EVENTS, "log_dir": root / RECOVERY_LOG_DIR}


def event_code_is_valid(event, current_code, context):
    if context is None: return event.get("code_sha256") == current_code
    digest = event_digest(event)
    if digest in context["retained_event_digests"]: return True
    if digest in context.get("accepted_revision_event_digests", set()): return True
    valid = (event.get("code_sha256") == current_code
             and event.get("execution_bundle_manifest_sha256") == context["execution_bundle_sha256"]
             and event.get("recovery_contract_sha256") == context["contract_sha256"])
    revision_hash = context.get("validator_revision_sha256")
    return valid and (revision_hash is None or event.get("validator_revision_sha256") == revision_hash)


def require_recovery_command(context, stage, phase=None):
    if context is None: raise ValueError("explicit applied recovery context required")
    if stage == "frontend":
        if phase not in RECOVERY_FRONTEND_PHASES: raise ValueError("recovery forbids prefix frontend phase execution")
    elif stage not in RECOVERY_STAGES:
        raise ValueError("stage is outside bounded recovery suffix allowlist")
    return True


def execution_identity(context):
    if context is None: return None
    c = context["contract"]
    value = {"schema": "rgb_scene_recovery_execution_identity_v1", "transition": TRANSITION,
             "birth_bundle_manifest_sha256": c["run_identity"]["bundle_manifest_sha256"],
             "execution_bundle_manifest_sha256": context["execution_bundle_sha256"],
             "recovery_contract_sha256": context["contract_sha256"],
             "source_events_sha256": c["original_events_sha256"],
             "retained_events_sha256": c["retained_events_sha256"],
             "stop_identity": c["stop_identity"]}
    if context.get("validator_revision_sha256") is not None:
        revision = context["validator_revision"]
        value.update(validator_revision_sha256=context["validator_revision_sha256"],
                     prior_execution_bundle_manifest_sha256=revision["prior_execution_bundle_manifest_sha256"],
                     accepted_prior_receipts=revision["accepted_prior_receipts"],
                     validator_revision_terminal_actor_identity=revision["terminal_actor_identity"])
    return value
