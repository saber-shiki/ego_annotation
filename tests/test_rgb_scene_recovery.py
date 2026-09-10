import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rgb_scene_recovery as recovery
from rgb_scene_runtime import SceneLineage
from v19_runtime_contracts import load, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes): path.write_bytes(value)
    else: path.write_text(json.dumps(value) + "\n")
    return path


class RecoveryFixture:
    def __init__(self, base):
        self.base = Path(base); self.root = self.base / "run"
        self.old = self.base / "old_bundle"; self.new = self.base / "new_bundle"
        for bundle in (self.old, self.new):
            write(bundle / "runtime/runtime_config.json", {"model": "same", "copied_root": str(bundle / "asset")})
            write(bundle / "scripts/producer.py", {"producer": "same"})
        write(self.new / "scripts/rgb_scene_depth_order.py", {"mechanism": "front-only"})
        write(self.new / "scripts/rgb_scene_metric_eligibility.py", {"mechanism": "native-depth-eligibility"})
        write(self.new / "scripts/build_v19_visible_geometry_from_sam2_depth.py", {"seam": "default-off"})
        write(self.new / "scripts/rgb_scene_recovery.py", {"transition": "P09_DEPTH_ORDER"})
        self._manifest(self.old); self._manifest(self.new)
        raw = write(self.root / "input/source.mp4", b"synthetic raw bytes")
        self.contract = {
            "schema": "rgb_scene_input_v1", "run_id": str(uuid.uuid4()), "run_root": str(self.root),
            "case": "arbitrary_case", "target_description": None, "input_video": str(raw),
            "input_video_sha256": sha(raw), "bundle_manifest_sha256": sha(self.old / "runtime/bundle_manifest.json"),
            "sensor_calibration_metadata": None, "sensor_calibration_sha256": None,
            "source_timing": {"frame_count": 1, "fps_ratio": "30", "time_base_s": "1/30", "first_pts_s": 0.0,
                              "duration_s": 1/30, "max_cfr_grid_error_s": 0.0,
                              "fps_source_field": "fixture", "validation": "fixture"},
        }
        self.contract_path = write(self.root / "input/runtime_input_contract.json", self.contract)
        identity = recovery.run_identity(self.contract)
        frame = write(self.root / "input/raw_frame_manifest/rgb/000000.jpg", b"jpeg fixture")
        manifest = write(self.root / "input/raw_frame_manifest/manifest.json",
                         {"frames": [{"frame_idx": 0, "time_s": 0.0, "raw_frame_path": str(frame)}]})
        old_code = recovery.code_identity(load(self.old / "runtime/bundle_manifest.json"))
        self.prefix_events = []
        for index, phase in enumerate(recovery.RETAINED_PHASES):
            outputs = []
            if phase == "P01": outputs = [manifest, frame]
            else: outputs = [write(self.root / f"state/retained/{index}_{phase}.json", {"phase": phase})]
            self.prefix_events.append({"phase": phase, "returncode": 0, "run_identity": identity,
                                       "code_sha256": old_code, "input_sha256": {},
                                       "outputs": list(map(str, outputs)),
                                       "output_sha256": {str(p): sha(p) for p in outputs},
                                       "input_contract_sha256": sha(self.contract_path)})
        evidence = {str(manifest): sha(manifest), str(frame): sha(frame)}
        self.plan = write(self.root / "measurements/object_candidates/object_plan_agent.json", {
            "objects": [{"object_id": "arbitrary_object", "track_id": "arbitrary_track"}],
            "visual_judgment_provenance": {"kind": "current_run_visual_judgment", "phase": "P05",
                                           "run_identity": identity, "evidence_sha256": evidence}})
        proposal_path = self.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json"
        suffix_paths = [
            write(proposal_path, {"object_id": "arbitrary_object", "review_candidates": [{"frame_idx": 7}],
                                  "outputs": {"anchor_candidate_review": str(proposal_path.with_name("anchor_candidate_review.jpg"))}}),
            write(self.root / "measurements/object_geometry/visible_geometry/arbitrary_object/annotations_v19_visible_geometry.json", {"old": True}),
            write(self.root / "measurements/geometry_completion/sam3d_arbitrary_object/mesh.ply", b"old mesh"),
            write(self.root / "state/scene/arbitrary_object/refined_pose.json", {"old": True}),
            write(self.root / "measurements/demo_visual/arbitrary_object/appearance_evidence/mask.png", b"old mask"),
        ]
        phases = ["P09-propose", "P09", "scene_geometry", "scene_refined_pose", "scene_appearance"]
        self.suffix_events = [{"phase": phase, "returncode": 0, "run_identity": identity,
                               "code_sha256": old_code, "input_sha256": {}, "outputs": [str(path)],
                               "output_sha256": {str(path): sha(path)}, "input_contract_sha256": sha(self.contract_path)}
                              for phase, path in zip(phases, suffix_paths)]
        self.decisions = [
            write(self.root / "state/anchor_decisions/arbitrary_object.json", {"old": "anchor"}),
            write(self.root / "state/physical_branch_decisions/arbitrary_object.json", {"old": "P10"}),
            write(self.root / "state/object_selection.json", {"old": "selection"}),
        ]
        self.events_path = self.root / "logs/integration/events.jsonl"
        prefix = b"".join((json.dumps(e) + "\n").encode() for e in self.prefix_events)
        events = prefix + b"".join((json.dumps(e) + "\n").encode() for e in self.suffix_events)
        write(self.events_path, events)
        self.stop = write(self.base / "stopped_run_evidence.json", {
            "kind": "parent_owner_stop_and_reconciliation_not_prediction_receipt",
            "run_root": str(self.root), "run_uuid": self.contract["run_id"],
            "native_actor_id": str(uuid.uuid4()), "workflow_id": str(uuid.uuid4()),
            "status": "stopped_by_explicit_user_request", "actor_process_terminal": "observed",
            "actor_resumable": False, "stop_tool_acknowledged": True,
            "active_owned_pipeline_processes_at_check": [], "new_runtime_actor_required": True,
            "run_outputs_moved_or_mutated_by_capture": False, "birth_input_sha256": sha(self.contract_path),
            "original_events_sha256": hashlib.sha256(events).hexdigest(), "original_events_bytes": len(events),
            "retained_prefix_through": "P08", "retained_event_prefix_bytes": len(prefix),
            "retained_event_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        })
        self.package = self.base / "package"
        self.original_hashes = {p: sha(p) for p in [self.contract_path, self.events_path, self.plan, *suffix_paths, *self.decisions]}

    def _manifest(self, bundle):
        files = {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob("*") if p.is_file()}
        write(bundle / "runtime/bundle_manifest.json", {"schema": "rgb_scene_bundle_v1", "files": files,
              "external_assets": {"files": {}, "trees": {}}, "preflight_status": "imports_checked"})

    def prepare(self):
        return recovery.prepare_recovery(self.root, self.old, self.new, self.stop, self.package, process_rows=[])

    def append_recovery_event(self, phase, outputs, *, inputs=None):
        context = {"execution_bundle_sha256": sha(self.new / "runtime/bundle_manifest.json"),
                   "contract_sha256": sha(self.root / recovery.RECOVERY_FILE)}
        outputs = list(map(Path, outputs))
        event = {
            "phase": phase, "returncode": 0, "run_identity": recovery.run_identity(self.contract),
            "input_contract_sha256": sha(self.contract_path),
            "code_sha256": recovery.code_identity(load(self.new / "runtime/bundle_manifest.json")),
            "execution_bundle_manifest_sha256": context["execution_bundle_sha256"],
            "recovery_contract_sha256": context["contract_sha256"],
            "inputs": list(map(str, (inputs or {}).keys())),
            "input_sha256": {str(Path(path)): digest for path, digest in (inputs or {}).items()},
            "outputs": list(map(str, outputs)),
            "output_sha256": {str(path): sha(path) for path in outputs},
        }
        path = self.root / recovery.RECOVERY_EVENTS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle: handle.write(json.dumps(event) + "\n")
        return event


class RGBSceneRecoveryTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        return RecoveryFixture(td.name)

    def test_prepare_is_read_only_and_binds_prefix_stop_bundle_assets_and_archive(self):
        f = self.fixture(); contract_path = f.prepare()
        self.assertEqual({p: sha(p) for p in f.original_hashes}, f.original_hashes)
        value = load(contract_path); archive = load(f.package / "archive_manifest.json")
        self.assertEqual(value["retained_phases"], list(recovery.RETAINED_PHASES))
        self.assertEqual(value["stop_identity"]["actor_resumable"], False)
        self.assertEqual(set(value["changed_bundle_files"]), {"runtime/runtime_config.json", "scripts/rgb_scene_depth_order.py",
            "scripts/rgb_scene_metric_eligibility.py", "scripts/build_v19_visible_geometry_from_sam2_depth.py",
            "scripts/rgb_scene_recovery.py"})
        archived_sources = {row["source"] for row in archive["files"]}
        self.assertTrue({str(p) for p in f.decisions} <= archived_sources)
        self.assertNotIn(str(f.plan), archived_sources)
        recovery.verify_prepared_recovery(f.package, root=f.root, target_bundle=f.new, process_rows=[])

    def test_apply_archives_suffix_preserves_prefix_and_attributes_old_and_new_events(self):
        f = self.fixture(); f.prepare(); old_events = f.events_path.read_bytes(); old_contract = f.contract_path.read_bytes()
        recovery.apply_recovery(f.package, process_rows=[])
        context = recovery.verify_applied_recovery(f.root, f.new)
        self.assertEqual(f.events_path.read_bytes(), old_events); self.assertEqual(f.contract_path.read_bytes(), old_contract)
        archive = load(f.package / "archive_manifest.json")
        for row in archive["files"]:
            self.assertFalse(Path(row["source"]).exists())
            self.assertEqual(sha(f.root / recovery.RECOVERY_REL / row["archive_relative"]), row["sha256"])
        lineage = SceneLineage(f.root, f.new, f.new / "runtime/runtime_config.json")
        retained = Path(next(iter(f.prefix_events[-1]["output_sha256"])))
        self.assertEqual(lineage.require(retained), retained)
        self.assertEqual(lineage.records[str(retained)]["code_sha256"], recovery.code_identity(load(f.old / "runtime/bundle_manifest.json")))
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        event = {"phase": "P09_DEPTH_ORDER", "returncode": 0, "outputs": [str(output)]}
        lineage.success(event, {}, [output], [])
        self.assertEqual(event["execution_bundle_manifest_sha256"], context["execution_bundle_sha256"])
        self.assertEqual(event["recovery_contract_sha256"], context["contract_sha256"])
        write(f.root / recovery.RECOVERY_EVENTS, (json.dumps(event) + "\n").encode())
        self.assertEqual(SceneLineage(f.root, f.new).require(output), output)
        unbound = dict(event); unbound.pop("recovery_contract_sha256")
        write(f.root / recovery.RECOVERY_EVENTS, (json.dumps(unbound) + "\n").encode())
        with self.assertRaisesRegex(ValueError, "foreign/unbound"):
            recovery.verify_applied_recovery(f.root, f.new)

    def test_stale_suffix_restoration_and_changed_archive_are_rejected(self):
        f = self.fixture(); f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        row = load(f.package / "archive_manifest.json")["files"][0]
        source = Path(row["source"]); source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(b"restored")
        with self.assertRaisesRegex(ValueError, "stale invalidated suffix"):
            recovery.verify_applied_recovery(f.root, f.new)
        source.unlink(); archived = f.root / recovery.RECOVERY_REL / row["archive_relative"]
        archived.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "archive file"):
            recovery.verify_applied_recovery(f.root, f.new)

    def test_active_missing_foreign_code_assets_and_input_mutations_are_rejected(self):
        f = self.fixture()
        active = [f"412 {sys.executable} run_rgb_scene_stage.py --run-root {f.root}"]
        with self.assertRaisesRegex(ValueError, "active owned"):
            recovery.prepare_recovery(f.root, f.old, f.new, f.stop, f.package, process_rows=active)
        foreign = json.loads(f.events_path.read_text().splitlines()[0]); foreign["code_sha256"] = {}
        lines = f.events_path.read_text().splitlines(); lines[0] = json.dumps(foreign); f.events_path.write_text("\n".join(lines) + "\n")
        stop = load(f.stop); data = f.events_path.read_bytes(); stop["original_events_sha256"] = hashlib.sha256(data).hexdigest(); stop["original_events_bytes"] = len(data)
        prefix = b"".join(line.encode()+b"\n" for line in f.events_path.read_text().splitlines()[:len(recovery.RETAINED_PHASES)])
        stop["retained_event_prefix_bytes"] = len(prefix); stop["retained_event_prefix_sha256"] = hashlib.sha256(prefix).hexdigest(); write(f.stop, stop)
        with self.assertRaisesRegex(ValueError, "foreign"):
            recovery.prepare_recovery(f.root, f.old, f.new, f.stop, f.package, process_rows=[])
        f = self.fixture(); f.prepare(); Path(next(iter(f.suffix_events[0]["output_sha256"]))).unlink()
        with self.assertRaisesRegex(ValueError, "missing|source changed"):
            recovery.verify_prepared_recovery(f.package, root=f.root, target_bundle=f.new, process_rows=[])
        f = self.fixture(); shutil.copytree(f.new, f.base / "bad_bundle"); bad = f.base / "bad_bundle"
        manifest = load(bad / "runtime/bundle_manifest.json"); manifest["external_assets"] = {"files": {"model": {"sha256": "x"}}, "trees": {}}; write(bad / "runtime/bundle_manifest.json", manifest)
        with self.assertRaises((ValueError, KeyError)):
            recovery.validate_transition(load(f.old / "runtime/bundle_manifest.json"), manifest,
                load(f.old / "runtime/runtime_config.json"), load(bad / "runtime/runtime_config.json"), f.old, bad)

    def test_partial_move_journal_refuses_launch(self):
        f = self.fixture(); f.prepare()
        with patch.object(recovery.os, "replace", side_effect=OSError("synthetic move failure")):
            with self.assertRaisesRegex(OSError, "synthetic move failure"): recovery.apply_recovery(f.package, process_rows=[])
        journal = f.root / recovery.JOURNAL_FILE
        self.assertTrue(journal.is_file()); self.assertIn("apply_failed", journal.read_text())
        self.assertFalse((f.root / recovery.COMPLETE_FILE).exists())
        with self.assertRaisesRegex(ValueError, "incomplete or missing"):
            recovery.recovery_context(f.root, f.new, f.contract, load(f.new / "runtime/bundle_manifest.json"))

    def test_recovery_stage_allowlist_and_fresh_mode(self):
        f = self.fixture()
        self.assertIsNone(recovery.recovery_context(f.root, f.old, f.contract, load(f.old / "runtime/bundle_manifest.json")))
        context = {"contract": {}}
        for phase in recovery.RECOVERY_FRONTEND_PHASES: self.assertTrue(recovery.require_recovery_command(context, "frontend", phase))
        for phase in ("P01", "P06", "P07", "P08"):
            with self.assertRaisesRegex(ValueError, "forbids prefix"): recovery.require_recovery_command(context, "frontend", phase)
        with self.assertRaisesRegex(ValueError, "outside bounded"): recovery.require_recovery_command(context, "camera")
        source = Path(__file__).resolve().parents[1] / "scripts/run_rgb_scene_stage.py"
        text = source.read_text(); self.assertIn("--recovery-contract", text); self.assertIn("require_recovery_command", text)
        preparer = (Path(__file__).resolve().parents[1] / "scripts/prepare_rgb_scene_runtime.py").read_text()
        for command in ("prepare-recovery", "verify-recovery", "apply-recovery", "verify-applied-recovery"):
            self.assertIn(command, preparer)

    def test_actual_cli_recovery_wiring_rejects_prefix_and_forwards_contract(self):
        import run_rgb_scene_stage as stage
        f = self.fixture(); cfg = write(f.base / "runtime_config.json", {"compute_host": "fixture", "gpu": 5})
        applied = f.root / recovery.RECOVERY_FILE
        forwarded = []
        with patch.object(stage, "verify_applied_recovery", return_value={"contract": {}}), \
             patch.object(stage, "execution_target", return_value=None), \
             patch.object(stage.runpy, "run_path", side_effect=lambda *a, **k: forwarded.append(list(sys.argv))):
            with patch.object(sys, "argv", ["stage", "--stage", "frontend", "--phase", "P08", "--run-root", str(f.root),
                                                   "--runtime-config", str(cfg), "--gpu", "5", "--recovery-contract", str(applied)]):
                with self.assertRaisesRegex(ValueError, "forbids prefix"): stage.main()
            with patch.object(sys, "argv", ["stage", "--stage", "frontend", "--phase", "P09_DEPTH_ORDER", "--run-root", str(f.root),
                                                   "--runtime-config", str(cfg), "--gpu", "5", "--recovery-contract", str(applied)]): stage.main()
        self.assertEqual(len(forwarded), 1); self.assertIn("--recovery-contract", forwarded[0]); self.assertIn(str(applied), forwarded[0])
        spec = (Path(__file__).resolve().parents[1] / "runtime/rgb_scene_runtime_spec.md").read_text()
        self.assertIn("--phase P09_DEPTH_ORDER", spec); self.assertIn("--phase P09-propose", spec)
        self.assertIn("No annotation MP4", spec)
        profile = (Path(__file__).resolve().parents[1] / ".pi/agents/ego-annotation.rgb-scene-runtime.md").read_text()
        self.assertIn("p08_depth_order_recovery", profile); self.assertIn("never invoke P01, camera, P06, P07, or P08", profile)
        self.assertIn("native_recovery_handoff.md", profile)

    def test_canonical_regeneration_receipts_judgments_and_nested_ordering(self):
        f = self.fixture(); f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        identity = recovery.run_identity(f.contract)
        def judgment(path, phase, payload, evidence):
            return write(path, {**payload, "visual_judgment_provenance": {
                "kind": "current_run_visual_judgment", "phase": phase,
                "run_identity": identity, "evidence_sha256": {str(p): sha(p) for p in evidence}}})

        owned = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/native_depth_metric_eligibility.json", {"new": True})
        f.append_recovery_event("P09_DEPTH_ORDER", [owned])
        recovery.verify_applied_recovery(f.root, f.new)

        proposal = f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json"
        review = write(proposal.with_name("anchor_candidate_review.jpg"), b"new review pixels")
        # A genuine deterministic current producer may recreate the exact old
        # proposal bytes at its canonical archived source path.
        archived_proposal = f.root / recovery.RECOVERY_REL / "invalidated" / proposal.relative_to(f.root)
        proposal.parent.mkdir(parents=True, exist_ok=True); proposal.write_bytes(archived_proposal.read_bytes())
        f.append_recovery_event("P09-propose", [proposal, review], inputs={owned: sha(owned)})
        recovery.verify_applied_recovery(f.root, f.new)

        anchor = judgment(f.root / "state/anchor_decisions/arbitrary_object.json", "P09_anchor",
                          {"object_id": "arbitrary_object", "selected_anchor_frame_idx": 7}, [proposal, review])
        recovery.verify_applied_recovery(f.root, f.new)
        annotations = write(f.root / "measurements/object_geometry/visible_geometry/arbitrary_object/annotations_v19_visible_geometry.json", {"new": "P09"})
        p09_image = write(annotations.parent / "object_owned_masks/000007.png", b"new P09 owned pixels")
        f.append_recovery_event("P09", [annotations, p09_image], inputs={anchor: sha(anchor), proposal: sha(proposal)})
        recovery.verify_applied_recovery(f.root, f.new)

        branch = judgment(f.root / "state/physical_branch_decisions/arbitrary_object.json", "P10",
                          {"object_id": "arbitrary_object", "physical_branch": "rigid"}, [annotations, p09_image])
        selection = judgment(f.root / "state/object_selection.json", "P10",
                             {"case": "arbitrary_case", "object_id": "arbitrary_object",
                              "track_id": "arbitrary_track", "anchor_frame": 7}, [annotations, p09_image])
        recovery.verify_applied_recovery(f.root, f.new)
        p11 = write(f.root / "measurements/geometry_completion/rigid_evidence/arbitrary_case/arbitrary_object/evidence_bundle/evidence_bundle_report.json", {"new": "P11"})
        f.append_recovery_event("P11", [p11], inputs={branch: sha(branch), selection: sha(selection), annotations: sha(annotations)})
        recovery.verify_applied_recovery(f.root, f.new)
        self.assertEqual(SceneLineage(f.root, f.new).require(p11), p11)

        geometry = write(f.root / "measurements/geometry_completion/sam3d_arbitrary_object/completion_report.json", {"new": "geometry"})
        f.append_recovery_event("scene_geometry", [geometry], inputs={p11: sha(p11)})
        bound = write(geometry.with_name("anchor_bound_completion_report.json"), {"new": "bound"})
        f.append_recovery_event("scene_anchor_binding", [bound], inputs={geometry: sha(geometry), annotations: sha(annotations)})
        pose = write(f.root / "measurements/pose_fits/arbitrary_object/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json", {"new": "pose"})
        f.append_recovery_event("scene_source_pose", [pose], inputs={bound: sha(bound), annotations: sha(annotations)})
        self.assertEqual(SceneLineage(f.root, f.new).require(pose), pose)

        rgb = write(f.root / "measurements/pose_fits/arbitrary_object/scene_rgb/edges.npz", b"rgb edges")
        f.append_recovery_event("scene_rgb", [rgb], inputs={pose: sha(pose), annotations: sha(annotations)})
        posterior = write(f.root / "state/scene/arbitrary_object/refined_pose.json", {"new": "posterior"})
        f.append_recovery_event("scene_refined_pose", [posterior], inputs={rgb: sha(rgb), pose: sha(pose)})
        raw_image = Path(f.prefix_events[0]["outputs"][1])
        appearance_manifest = f.root / "measurements/demo_visual/arbitrary_object/appearance_evidence/appearance_mask_manifest.json"
        appearance_mask = write(appearance_manifest.with_name("mask.png"), b"new appearance mask")
        write(appearance_manifest, {"object_id": "arbitrary_object", "frames": [{"frame_idx": 0,
            "status": "available", "raw_rgb_path": str(raw_image), "appearance_mask_path": str(appearance_mask)}]})
        f.append_recovery_event("scene_appearance", [appearance_manifest, appearance_mask],
                                inputs={posterior: sha(posterior), annotations: sha(annotations)})
        appearance_decision = judgment(f.root / "state/appearance_decisions/arbitrary_object.json", "appearance_anchor",
            {"object_id": "arbitrary_object", "selected_frame_idx": 0}, [appearance_manifest, appearance_mask, raw_image])
        recovery.verify_applied_recovery(f.root, f.new)
        native = write(f.root / "measurements/scene_native/arbitrary_object/native.json", {"new": "native"})
        f.append_recovery_event("scene_native_appearance", [native], inputs={appearance_decision: sha(appearance_decision)})
        binding = write(f.root / "measurements/scene_shape/arbitrary_object/pose_mesh.json", {"new": "shape"})
        f.append_recovery_event("scene_refined_shape", [binding], inputs={native: sha(native), posterior: sha(posterior)})
        visibility = judgment(f.root / "state/agent_interaction_judgments/arbitrary_object.json", "P17_judgment",
            {"object_id": "arbitrary_object", "status": "ok", "interaction_judgments": [], "frames": []},
            [posterior, raw_image])
        recovery.verify_applied_recovery(f.root, f.new)
        fit = write(f.root / "measurements/scene_fit/arbitrary_object/fit_report.json", {"new": "fit"})
        f.append_recovery_event("scene_visual_fit", [fit], inputs={binding: sha(binding), visibility: sha(visibility)})
        final = write(f.root / "final/annotation.json", {"new": "final"})
        f.append_recovery_event("scene_export", [final], inputs={fit: sha(fit), visibility: sha(visibility)})
        self.assertEqual(SceneLineage(f.root, f.new).require(final), final)
        cli = Path(__file__).resolve().parents[1] / "scripts/prepare_rgb_scene_runtime.py"
        done = subprocess.run([sys.executable, str(cli), "verify-applied-recovery",
                               "--run-root", str(f.root), "--target-bundle", str(f.new)],
                              check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(done.stdout)["transition"], "P09_DEPTH_ORDER")

    def test_recovery_rejects_out_of_order_or_modified_canonical_producer(self):
        f = self.fixture(); f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/geometry_completion/sam3d_arbitrary_object/completion_report.json", {"bad": "skip"})
        f.append_recovery_event("scene_geometry", [output])
        with self.assertRaisesRegex(ValueError, "completion ordering"):
            recovery.verify_applied_recovery(f.root, f.new)

    def test_owned_process_inventory_covers_dispatch_leaves_modules_and_run_binding(self):
        source = Path(__file__).resolve().parents[1]
        import re
        dispatched = set()
        for relative in ("scripts/run_rgb_scene_frontend.py", "scripts/run_rgb_scene_stage.py",
                         "scripts/run_rgb_scene_reconstruction.py", "scripts/rgb_scene_camera.py",
                         "scripts/remote_run_hawor_export.sh"):
            dispatched.update(re.findall(r"[A-Za-z0-9_]+(?:\.py|\.sh)", (source / relative).read_text()))
        existing_names = {path.name for path in source.rglob("*") if path.is_file()}
        dispatched &= existing_names
        self.assertTrue(dispatched <= recovery.OWNED_PROCESS_EXECUTABLES,
                        sorted(dispatched - recovery.OWNED_PROCESS_EXECUTABLES))
        self.assertIn("build_v19_demo_visual_inputs.py", recovery.OWNED_PROCESS_EXECUTABLES)
        root = "/runs/current"
        leaves = [
            f"101 python run_rgb_scene_frontend.py --run-root {root}",
            f"102 python build_v18_compact_rigid_evidence_bundle.py --run-root {root}",
            f"103 python fit_rgb_scene_visual.py --run-root {root} --device cuda:0",
            f"104 python remote_run_sam3d_objects_mesh_v7.py --output-dir {root}/native",
            f"105 python -m rgb_scene_pose_io --run-root {root}",
            f"106 python build_v19_demo_visual_inputs.py --run-root {root}",
        ]
        self.assertEqual({row["pid"] for row in recovery.active_owned_processes(root, leaves)},
                         {101, 102, 103, 104, 105, 106})
        self.assertEqual(recovery.active_owned_processes(root, [
            "201 python fit_rgb_scene_visual.py --run-root /runs/foreign",
            f"202 python unrelated.py --run-root {root}"]), [])
        with patch.object(recovery.os, "getpid", return_value=103):
            self.assertNotIn(103, {row["pid"] for row in recovery.active_owned_processes(root, leaves)})

    def test_recovery_identity_retains_birth_and_execution_for_json_and_rrd_document(self):
        from rgb_scene_identity import InputIdentity
        context = {"contract": {"run_identity": {"bundle_manifest_sha256": "a"*64},
                                "original_events_sha256": "d"*64, "retained_events_sha256": "e"*64,
                                "stop_identity": {"actor_process_terminal": "observed"}},
                   "execution_bundle_sha256": "b"*64, "contract_sha256": "c"*64}
        execution = recovery.execution_identity(context)
        identity = InputIdentity(str(uuid.uuid4()), "case", "f"*64, "a"*64, None, "1"*64,
            json.dumps({"frame_count": 1, "fps_ratio": "30", "time_base_s": "1/30", "first_pts_s": 0.0,
                        "duration_s": .03, "max_cfr_grid_error_s": 0.0, "fps_source_field": "fixture", "validation": "fixture"}),
            None, json.dumps(execution))
        doc = identity.document()
        self.assertEqual(doc["bundle_manifest_sha256"], "a"*64)
        self.assertEqual(doc["recovery_execution_identity"]["execution_bundle_manifest_sha256"], "b"*64)
        export_text = (Path(__file__).resolve().parents[1] / "scripts/export_rgb_scene.py").read_text()
        self.assertIn("recovery_context", export_text); self.assertIn("identity.document()", export_text)


if __name__ == "__main__": unittest.main()
