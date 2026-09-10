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


def _validated_recovery_events(*, root, contract, current_code, target_hash, contract_hash,
                               retained_rows, retained_outputs, obj, track, case):
    event_path = root / RECOVERY_EVENTS
    retained_records = {}
    for event in retained_rows:
        for name in event.get("output_sha256", {}):
            retained_records[name] = event
    available = dict(retained_outputs)
    available[str(root / "input/runtime_input_contract.json")] = sha(root / "input/runtime_input_contract.json")
    available[contract["input_video"]] = contract["input_video_sha256"]
    p05_path = root / "measurements/object_candidates/object_plan_agent.json"
    available[str(p05_path)] = sha(p05_path)
    records = {}
    authorized = {}
    if not event_path.exists():
        return records, retained_records, authorized
    if not event_path.is_file() or not event_path.read_bytes().endswith(b"\n"):
        raise ValueError("recovery event log is partial/malformed")
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
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
                    "retained_event_digests": set(), "execution_bundle_sha256": target_hash,
                    "contract_sha256": contract_hash,
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
    for judgment in judgment_paths:
        path = Path(judgment)
        if path.exists():
            _validate_recovery_judgment(path, root=root, contract=contract, records=records,
                                        retained_records=retained_records, obj=obj, track=track, case=case)
            authorized[judgment] = sha(path)
    return records, retained_records, authorized


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
    expected = {
        "schema": SCHEMA, "mode": "same_run_recovery", "transition": TRANSITION,
        "run_identity": run_identity(contract), "input_contract_sha256": sha(root / "input/runtime_input_contract.json"),
        "target_bundle": str(bundle), "target_bundle_manifest_sha256": target_hash,
        "recovery_events_path": str(root / RECOVERY_EVENTS),
        "native_handoff_path": str(root / RECOVERY_REL / "native_recovery_handoff.md"),
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted: raise ValueError("depth-order recovery identity mismatch: " + key)
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
    changed = validate_transition(old, manifest, load(directory / "source_runtime_config.json"),
                                  load(bundle / "runtime/runtime_config.json"), Path(value["source_bundle"]), bundle)
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
    current_code = code_identity(manifest)
    recovery_records, retained_records, authorized_active = _validated_recovery_events(
        root=root, contract=contract, current_code=current_code, target_hash=target_hash,
        contract_hash=sha(path), retained_rows=rows,
        retained_outputs=value["retained_outputs_sha256"], obj=value["object_id"],
        track=value["track_id"], case=contract["case"],
    )
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
    recovery_events = root / RECOVERY_EVENTS
    failure = root / RECOVERY_REL / "failure.json"
    if failure.exists(): raise ValueError("recovery already failed; relaunch is forbidden")
    return {"contract": value, "contract_sha256": sha(path), "execution_bundle_sha256": target_hash,
            "source_code": old_code, "retained_event_digests": retained_digests,
            "event_sources": [directory / "retained_events.jsonl", root / RECOVERY_EVENTS],
            "events_path": root / RECOVERY_EVENTS, "log_dir": root / RECOVERY_LOG_DIR}


def event_code_is_valid(event, current_code, context):
    if context is None: return event.get("code_sha256") == current_code
    if event_digest(event) in context["retained_event_digests"]: return True
    return (event.get("code_sha256") == current_code
            and event.get("execution_bundle_manifest_sha256") == context["execution_bundle_sha256"]
            and event.get("recovery_contract_sha256") == context["contract_sha256"])


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
    return {"schema": "rgb_scene_recovery_execution_identity_v1", "transition": TRANSITION,
            "birth_bundle_manifest_sha256": c["run_identity"]["bundle_manifest_sha256"],
            "execution_bundle_manifest_sha256": context["execution_bundle_sha256"],
            "recovery_contract_sha256": context["contract_sha256"],
            "source_events_sha256": c["original_events_sha256"],
            "retained_events_sha256": c["retained_events_sha256"],
            "stop_identity": c["stop_identity"]}
