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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rgb_scene_recovery as recovery
from rgb_scene_lineage_base import RunLineage
from test_rgb_scene_recovery import RecoveryFixture, write
from v19_runtime_contracts import load, sha


def refresh_original_events(fixture):
    prefix = b"".join((json.dumps(event) + "\n").encode() for event in fixture.prefix_events)
    events = prefix + b"".join((json.dumps(event) + "\n").encode() for event in fixture.suffix_events)
    write(fixture.events_path, events)
    stop = load(fixture.stop)
    stop.update(original_events_sha256=hashlib.sha256(events).hexdigest(),
                original_events_bytes=len(events), retained_event_prefix_bytes=len(prefix),
                retained_event_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    write(fixture.stop, stop)


def terminal_evidence(fixture, path):
    manifest = load(fixture.new / "runtime/bundle_manifest.json")
    return write(path, {
        "workflow": str(uuid.uuid4()), "actor": str(uuid.uuid4()),
        "state": "failed", "process_terminal": "observed",
        "pipeline_error": "P09-propose pre-producer ancestry validation failure",
        "completed_phases": ["P09_DEPTH_ORDER"],
        "alias_sha256": fixture.contract["input_video_sha256"],
        "birth_video_sha256": fixture.contract["input_video_sha256"],
        "source_state": {"cwd": str(fixture.new),
                         "head": manifest.get("source_commit"), "git_status": ""},
    })


def second_terminal_evidence(fixture, prior_bundle, event_bytes, path):
    manifest = load(prior_bundle / "runtime/bundle_manifest.json")
    return write(path, {
        "workflow": str(uuid.uuid4()), "actor": str(uuid.uuid4()),
        "actor_state": "complete", "process_terminal": "observed",
        "task_result": "blocked before P09 producer",
        "error": "P09 ancestry validation failure",
        "run_root": str(fixture.root), "runtime_cwd": str(prior_bundle),
        "completed_phases": ["P09_DEPTH_ORDER", "P09-propose"],
        "recovery_event_bytes": len(event_bytes),
        "recovery_event_sha256": hashlib.sha256(event_bytes).hexdigest(),
        "P09_producer_executed": False, "SAM3D_executed": False,
        "owned_stage_processes": [],
        "source_state": {"head": manifest.get("source_commit"), "status": ""},
    })


def third_terminal_evidence(fixture, prior_bundle, event_bytes, prior_prefix_bytes, decisions, path):
    manifest = load(prior_bundle / "runtime/bundle_manifest.json")
    return write(path, {
        "kind": "parent_terminal_failure_capture_not_prediction_receipt",
        "workflow": str(uuid.uuid4()), "actor": str(uuid.uuid4()),
        "actor_state": "complete", "process_terminal": "observed",
        "task_result": "blocked before P11 producer",
        "error": "second validator revision prefix differs from terminal event log",
        "run_root": str(fixture.root), "runtime_cwd": str(prior_bundle),
        "completed_phases": ["P09_DEPTH_ORDER", "P09-propose", "P09"],
        "recovery_event_bytes": len(event_bytes),
        "recovery_event_sha256": hashlib.sha256(event_bytes).hexdigest(),
        "frozen_prefix_bytes": prior_prefix_bytes,
        "frozen_prefix_sha_matches": True,
        "decision_sha256": {str(path): sha(path) for path in decisions},
        "P11_producer_executed": False, "SAM3D_executed": False,
        "owned_stage_processes": [],
        "source_state": {"head": manifest.get("source_commit"), "status": ""},
    })


class RetainedInputAliasTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        return RecoveryFixture(td.name)

    def recorded_alias(self, fixture, data=None):
        raw = Path(fixture.contract["input_video"])
        alias = write(fixture.root / "input/hawor_sequence/rgb_scene_fixture.mp4",
                      raw.read_bytes() if data is None else data)
        p04 = fixture.prefix_events[list(recovery.RETAINED_PHASES).index("P04_HaWoR")]
        p04["input_sha256"] = {str(alias): sha(alias)}
        refresh_original_events(fixture)
        return alias

    def test_recorded_birth_equivalent_alias_is_a_valid_recovery_ancestor(self):
        f = self.fixture(); alias = self.recorded_alias(f)
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        f.append_recovery_event("P09_DEPTH_ORDER", [output], inputs={alias: sha(alias)})
        context = recovery.verify_applied_recovery(f.root, f.new)
        self.assertEqual(context["events_path"], f.root / recovery.RECOVERY_EVENTS)

    def test_recorded_alias_mutation_and_non_birth_bytes_are_rejected(self):
        f = self.fixture(); alias = self.recorded_alias(f)
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        alias.write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError, "retained producer input changed"):
            recovery.verify_applied_recovery(f.root, f.new)

        f = self.fixture(); self.recorded_alias(f, b"not the birth video")
        f.prepare()
        with self.assertRaisesRegex(ValueError, "birth alias|birth"):
            recovery.apply_recovery(f.package, process_rows=[])

    def test_unrecorded_same_byte_alias_and_foreign_path_are_rejected(self):
        f = self.fixture()
        alias = write(f.root / "input/hawor_sequence/unrecorded.mp4",
                      Path(f.contract["input_video"]).read_bytes())
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        f.append_recovery_event("P09_DEPTH_ORDER", [output], inputs={alias: sha(alias)})
        with self.assertRaisesRegex(ValueError, "lacks completed bound ancestor"):
            recovery.verify_applied_recovery(f.root, f.new)
        retained = f.prefix_events[list(recovery.RETAINED_PHASES).index("P04_HaWoR")]
        retained["input_sha256"] = {str(f.base / "foreign.mp4"): "0" * 64}
        with self.assertRaisesRegex(ValueError, "escapes run root"):
            recovery._retained_input_seeds(f.root, [retained], f.contract)


class ValidatorRevisionTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        f = RecoveryFixture(td.name)
        f.prepare(); recovery.apply_recovery(f.package, process_rows=[])
        output = write(f.root / "measurements/object_tracks/depth_order_owned/arbitrary_track/new.json", {"new": True})
        event = f.append_recovery_event("P09_DEPTH_ORDER", [output])
        recovery.verify_applied_recovery(f.root, f.new)
        return f, output, event

    def revised_bundle(self, f, *, extra_change=None, asset_change=False):
        revised = f.base / "revised_bundle"
        shutil.copytree(f.new, revised)
        (revised / "runtime/bundle_manifest.json").unlink()
        write(revised / "scripts/rgb_scene_recovery.py", {"validator": "fixed"})
        write(revised / "scripts/rgb_scene_lineage_base.py", {"stamp": "validator revision"})
        config = load(revised / "runtime/runtime_config.json")
        config["copied_root"] = str(revised / "asset")
        write(revised / "runtime/runtime_config.json", config)
        if extra_change:
            write(revised / extra_change, {"unexpected": True})
        f._manifest(revised)
        if asset_change:
            manifest = load(revised / "runtime/bundle_manifest.json")
            manifest["external_assets"] = {"files": {"model": {"sha256": "changed"}}, "trees": {}}
            write(revised / "runtime/bundle_manifest.json", manifest)
        return revised

    def prepare_revision(self, f, revised):
        evidence = terminal_evidence(f, f.base / "terminal_actor.json")
        return recovery.prepare_validator_revision(
            f.root, f.new, revised, evidence, process_rows=[])

    def test_exact_prior_receipt_is_preserved_and_new_receipt_is_revision_stamped(self):
        f, depth_output, old_event = self.fixture(); revised = self.revised_bundle(f)
        revision_path = self.prepare_revision(f, revised)
        context = recovery.verify_validator_revision(f.root, revised)
        self.assertEqual(context["validator_revision"]["resume_phase"], "P09-propose")
        self.assertTrue(recovery.event_code_is_valid(
            old_event, recovery.code_identity(load(revised / "runtime/bundle_manifest.json")), context))
        lineage = RunLineage(f.root, revised, revised / "runtime/runtime_config.json")
        proposal = write(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json",
                         {"object_id": "arbitrary_object", "review_candidates": []})
        event = {"phase": "P09-propose", "returncode": 0, "outputs": [str(proposal)],
                 "input_contract_sha256": sha(f.contract_path)}
        lineage.success(event, {str(depth_output): sha(depth_output)}, [proposal], [])
        self.assertEqual(event["execution_bundle_manifest_sha256"],
                         sha(revised / "runtime/bundle_manifest.json"))
        self.assertEqual(event["recovery_contract_sha256"], sha(f.root / recovery.RECOVERY_FILE))
        self.assertEqual(event["validator_revision_sha256"], sha(revision_path))
        unbound = dict(event); unbound.pop("validator_revision_sha256")
        self.assertFalse(recovery.event_code_is_valid(
            unbound, recovery.code_identity(load(revised / "runtime/bundle_manifest.json")), context))
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        self.assertEqual(RunLineage(f.root, revised).require(proposal), proposal)

    def test_historical_first_event_prefix_ignores_valid_later_anchor_but_full_validation_binds_it(self):
        f, depth_output, _ = self.fixture(); revised = self.revised_bundle(f)
        self.prepare_revision(f, revised)
        lineage = RunLineage(f.root, revised, revised / "runtime/runtime_config.json")
        proposal = write(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json",
                         {"object_id": "arbitrary_object", "review_candidates": [{"frame_idx": 145}],
                          "outputs": {"anchor_candidate_review": str(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_review.jpg")}})
        review = write(proposal.with_name("anchor_candidate_review.jpg"), b"review pixels")
        event = {"phase": "P09-propose", "returncode": 0, "outputs": [str(proposal), str(review)],
                 "input_contract_sha256": sha(f.contract_path)}
        lineage.success(event, {str(depth_output): sha(depth_output)}, [proposal, review], [])
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        anchor = write(f.root / "state/anchor_decisions/arbitrary_object.json", {
            "object_id": "arbitrary_object", "selected_anchor_frame_idx": 145,
            "visual_judgment_provenance": {
                "kind": "current_run_visual_judgment", "phase": "P09_anchor",
                "run_identity": recovery.run_identity(f.contract),
                "evidence_sha256": {str(proposal): sha(proposal), str(review): sha(review)}}})
        context = recovery.verify_validator_revision(f.root, revised)
        self.assertEqual(context["validator_revision"]["resume_phase"], "P09-propose")
        bad = load(anchor)
        bad["visual_judgment_provenance"]["evidence_sha256"][str(proposal)] = "0" * 64
        write(anchor, bad)
        with self.assertRaisesRegex(ValueError, "evidence changed|wrong producer/hash"):
            recovery.verify_validator_revision(f.root, revised)

    def test_second_fixed_revision_preserves_two_receipts_and_anchor_then_resumes_p09(self):
        f, depth_output, _ = self.fixture(); first = self.revised_bundle(f)
        self.prepare_revision(f, first)
        lineage = RunLineage(f.root, first, first / "runtime/runtime_config.json")
        proposal = write(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json",
                         {"object_id": "arbitrary_object", "review_candidates": [{"frame_idx": 145}],
                          "outputs": {"anchor_candidate_review": str(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_review.jpg")}})
        review = write(proposal.with_name("anchor_candidate_review.jpg"), b"review pixels")
        event = {"phase": "P09-propose", "returncode": 0, "outputs": [str(proposal), str(review)],
                 "input_contract_sha256": sha(f.contract_path)}
        lineage.success(event, {str(depth_output): sha(depth_output)}, [proposal, review], [])
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        anchor = write(f.root / "state/anchor_decisions/arbitrary_object.json", {
            "object_id": "arbitrary_object", "selected_anchor_frame_idx": 145,
            "visual_judgment_provenance": {
                "kind": "current_run_visual_judgment", "phase": "P09_anchor",
                "run_identity": recovery.run_identity(f.contract),
                "evidence_sha256": {str(proposal): sha(proposal), str(review): sha(review)}}})
        recovery.verify_validator_revision(f.root, first)

        second = f.base / "second_revised_bundle"
        shutil.copytree(first, second)
        (second / "runtime/bundle_manifest.json").unlink()
        write(second / "scripts/rgb_scene_recovery.py", {"validator": "second fixed revision"})
        # Regression: the actual scope fix does not change existing receipt
        # stamping. Do not require an unrelated lineage edit for a new bundle.
        self.assertEqual(sha(second / "scripts/rgb_scene_lineage_base.py"),
                         sha(first / "scripts/rgb_scene_lineage_base.py"))
        config = load(second / "runtime/runtime_config.json")
        config["copied_root"] = str(second / "asset")
        write(second / "runtime/runtime_config.json", config)
        f._manifest(second)
        event_bytes = (f.root / recovery.RECOVERY_EVENTS).read_bytes()
        terminal = second_terminal_evidence(f, first, event_bytes, f.base / "terminal_second.json")
        revision_path = recovery.prepare_validator_revision(f.root, first, second, terminal, process_rows=[])
        context = recovery.verify_validator_revision(f.root, second)
        self.assertEqual(context["validator_revision"]["resume_phase"], "P09")
        self.assertEqual(len(context["validator_revision"]["accepted_prior_receipts"]), 2)
        self.assertEqual(context["validator_revision"]["anchor_decision_sha256"], sha(anchor))
        self.assertTrue(revision_path.with_suffix(".native_handoff.md").is_file())

    def test_post_p09_checkpoint_survives_each_future_append_and_rejects_bad_identity(self):
        f, depth_output, _ = self.fixture(); first = self.revised_bundle(f)
        self.prepare_revision(f, first)
        first_lineage = RunLineage(f.root, first, first / "runtime/runtime_config.json")
        proposal = write(
            f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_proposals.json",
            {"object_id": "arbitrary_object", "review_candidates": [{"frame_idx": 145}],
             "outputs": {"anchor_candidate_review": str(f.root / "measurements/object_geometry/anchor_candidates/arbitrary_object/anchor_candidate_review.jpg")}})
        review = write(proposal.with_name("anchor_candidate_review.jpg"), b"review pixels")
        proposal_event = {"phase": "P09-propose", "returncode": 0,
                          "outputs": [str(proposal), str(review)],
                          "input_contract_sha256": sha(f.contract_path)}
        first_lineage.success(proposal_event, {str(depth_output): sha(depth_output)},
                              [proposal, review], [])
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(proposal_event) + "\n")
        anchor = write(f.root / "state/anchor_decisions/arbitrary_object.json", {
            "object_id": "arbitrary_object", "selected_anchor_frame_idx": 145,
            "visual_judgment_provenance": {
                "kind": "current_run_visual_judgment", "phase": "P09_anchor",
                "run_identity": recovery.run_identity(f.contract),
                "evidence_sha256": {str(proposal): sha(proposal), str(review): sha(review)}}})

        second = f.base / "second_revised_bundle"
        shutil.copytree(first, second); (second / "runtime/bundle_manifest.json").unlink()
        write(second / "scripts/rgb_scene_recovery.py", {"validator": "second fixed revision"})
        config = load(second / "runtime/runtime_config.json");config["copied_root"] = str(second / "asset")
        write(second / "runtime/runtime_config.json", config);f._manifest(second)
        second_prefix = (f.root / recovery.RECOVERY_EVENTS).read_bytes()
        terminal = second_terminal_evidence(f, first, second_prefix, f.base / "terminal_second.json")
        recovery.prepare_validator_revision(f.root, first, second, terminal, process_rows=[])
        recovery.verify_validator_revision(f.root, second)

        second_lineage = RunLineage(f.root, second, second / "runtime/runtime_config.json")
        annotations = write(
            f.root / "measurements/object_geometry/visible_geometry/arbitrary_object/annotations_v19_visible_geometry.json",
            {"new": "P09"})
        p09_image = write(annotations.parent / "object_owned_masks/000145.png", b"P09 pixels")
        p09_event = {"phase": "P09", "returncode": 0,
                     "outputs": [str(annotations), str(p09_image)],
                     "input_contract_sha256": sha(f.contract_path)}
        second_lineage.success(p09_event, {str(anchor): sha(anchor), str(proposal): sha(proposal)},
                               [annotations, p09_image], [])
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(p09_event) + "\n")
        # This is the original P11 blocker: the two-receipt frozen prefix must
        # remain historical while the full current suffix validates P09.
        recovery.verify_validator_revision(f.root, second)

        p10_evidence = {str(annotations): sha(annotations), str(p09_image): sha(p09_image)}
        branch = write(f.root / "state/physical_branch_decisions/arbitrary_object.json", {
            "object_id": "arbitrary_object", "physical_branch": "rigid",
            "visual_judgment_provenance": {"kind": "current_run_visual_judgment", "phase": "P10",
                "run_identity": recovery.run_identity(f.contract), "evidence_sha256": p10_evidence}})
        selection = write(f.root / "state/object_selection.json", {
            "case": "arbitrary_case", "object_id": "arbitrary_object",
            "track_id": "arbitrary_track", "anchor_frame": 145,
            "visual_judgment_provenance": {"kind": "current_run_visual_judgment", "phase": "P10",
                "run_identity": recovery.run_identity(f.contract), "evidence_sha256": p10_evidence}})
        recovery.verify_validator_revision(f.root, second)

        third = f.base / "third_revised_bundle"
        shutil.copytree(second, third); (third / "runtime/bundle_manifest.json").unlink()
        write(third / "scripts/rgb_scene_recovery.py", {"validator": "post P09 fixed revision"})
        write(third / "scripts/run_rgb_scene_stage.py", {"entry_check": "role scoped"})
        write(third / "scripts/run_rgb_scene_frontend.py", {"entry_check": "role scoped"})
        config = load(third / "runtime/runtime_config.json");config["copied_root"] = str(third / "asset")
        write(third / "runtime/runtime_config.json", config);f._manifest(third)
        three_events = (f.root / recovery.RECOVERY_EVENTS).read_bytes()
        terminal = third_terminal_evidence(
            f, second, three_events, len(second_prefix), [anchor, branch, selection],
            f.base / "terminal_third.json")
        cli = Path(__file__).resolve().parents[1] / "scripts/prepare_rgb_scene_runtime.py"
        done = subprocess.run([
            sys.executable, str(cli), "prepare-validator-revision",
            "--run-root", str(f.root), "--prior-bundle", str(second),
            "--new-bundle", str(third), "--terminal-actor-evidence", str(terminal)],
            check=True, capture_output=True, text=True, timeout=30)
        revision_path = Path(json.loads(done.stdout)["validator_revision_contract"])
        context = recovery.verify_validator_revision(f.root, third)
        self.assertEqual(context["validator_revision"]["resume_phase"], "P11")
        self.assertEqual(len(context["validator_revision"]["accepted_prior_receipts"]), 3)
        self.assertEqual(set(context["validator_revision"]["changed_bundle_files"]), {
            "runtime/runtime_config.json", "scripts/rgb_scene_recovery.py",
            "scripts/run_rgb_scene_stage.py", "scripts/run_rgb_scene_frontend.py"})
        self.assertIn("--phase P11", revision_path.with_suffix(".native_handoff.md").read_text())
        self.assertIn("--stage geometry", revision_path.with_suffix(".native_handoff.md").read_text())

        third_lineage = RunLineage(f.root, third, third / "runtime/runtime_config.json")
        p11 = write(f.root / "measurements/geometry_completion/rigid_evidence/arbitrary_case/arbitrary_object/evidence_bundle/evidence_bundle_report.json", {"new": "P11"})
        p11_event = {"phase": "P11", "returncode": 0, "outputs": [str(p11)],
                     "input_contract_sha256": sha(f.contract_path)}
        third_lineage.success(p11_event, {
            str(branch): sha(branch), str(selection): sha(selection), str(annotations): sha(annotations)},
            [p11], [])
        for mutation in ("unbound", "newcode"):
            bad = dict(p11_event)
            if mutation == "unbound": bad.pop("validator_revision_sha256")
            else: bad["code_sha256"] = {}
            self.assertFalse(recovery.event_code_is_valid(
                bad, recovery.code_identity(load(third / "runtime/bundle_manifest.json")), context))
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(p11_event) + "\n")
        recovery.verify_validator_revision(f.root, third)
        valid_log = (f.root / recovery.RECOVERY_EVENTS).read_bytes()
        failed = dict(p11_event); failed["returncode"] = 1
        (f.root / recovery.RECOVERY_EVENTS).write_bytes(three_events + (json.dumps(failed) + "\n").encode())
        with self.assertRaisesRegex(ValueError, "failed"):
            recovery.verify_validator_revision(f.root, third)
        (f.root / recovery.RECOVERY_EVENTS).write_bytes(valid_log)

        geometry = write(f.root / "measurements/geometry_completion/sam3d_arbitrary_object/completion_report.json", {"new": "geometry"})
        geometry_event = {"phase": "scene_geometry", "returncode": 0,
                          "outputs": [str(geometry)],
                          "input_contract_sha256": sha(f.contract_path)}
        third_lineage.success(geometry_event, {str(p11): sha(p11)}, [geometry], [])
        with (f.root / recovery.RECOVERY_EVENTS).open("a") as handle:
            handle.write(json.dumps(geometry_event) + "\n")
        recovery.verify_validator_revision(f.root, third)

        revision_bytes = revision_path.read_bytes(); revision = load(revision_path)
        revision["recovery_events_sha256"] = "0" * 64; write(revision_path, revision)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            recovery.verify_validator_revision(f.root, third)
        revision_path.write_bytes(revision_bytes)
        revision = load(revision_path); revision["recovery_events_bytes"] = len(second_prefix)
        write(revision_path, revision)
        with self.assertRaisesRegex(ValueError, "three completed|identity mismatch|wrong producer"):
            recovery.verify_validator_revision(f.root, third)
        revision_path.write_bytes(revision_bytes)
        decision_bytes = selection.read_bytes(); selection.write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "decision_sha256|identity mismatch|judgment"):
            recovery.verify_validator_revision(f.root, third)
        selection.write_bytes(decision_bytes)
        recovery.verify_validator_revision(f.root, third)

    def test_runtime_roles_reach_current_and_historical_verifiers_without_full_weight_scan(self):
        f, _, _ = self.fixture(); revised = self.revised_bundle(f)
        self.prepare_revision(f, revised)
        import rgb_scene_lineage_base as lineage_module
        manifests = {str(bundle.resolve()): load(bundle / "runtime/bundle_manifest.json")
                     for bundle in (f.old, f.new, revised)}
        for role in ("P09", "P11"):
            calls = []
            def role_spy(bundle, *, asset_role=None):
                # A default/full call is the mocked stand-in for opening every
                # unrelated model weight in the external-asset inventory.
                if asset_role is None:
                    raise AssertionError("runtime attempted full unrelated-weight verification")
                calls.append((str(Path(bundle).resolve()), asset_role))
                return manifests[str(Path(bundle).resolve())]
            with patch.object(lineage_module, "verify_bundle", side_effect=role_spy), \
                 patch.object(recovery, "verify_bundle", side_effect=role_spy):
                RunLineage(f.root, revised, revised / "runtime/runtime_config.json", asset_role=role)
            self.assertEqual(len(calls), 2)
            self.assertEqual({Path(bundle) for bundle, _ in calls}, {f.new, revised})
            self.assertEqual({asset_role for _, asset_role in calls}, {role})
        stage_source = (Path(__file__).resolve().parents[1] / "scripts/run_rgb_scene_stage.py").read_text()
        nested_source = (Path(__file__).resolve().parents[1] / "scripts/run_rgb_scene_reconstruction.py").read_text()
        self.assertIn("'geometry':'geometry_dispatch'", stage_source)
        self.assertIn("asset_role='geometry'", nested_source)
        self.assertIn("scripts/run_rgb_scene_stage.py", recovery.ALLOWED_VALIDATOR_REVISION_FILES)
        self.assertIn("scripts/run_rgb_scene_frontend.py", recovery.ALLOWED_VALIDATOR_REVISION_FILES)

    def test_changed_prefix_or_accepted_output_is_rejected(self):
        f, output, _ = self.fixture(); revised = self.revised_bundle(f); self.prepare_revision(f, revised)
        event_path = f.root / recovery.RECOVERY_EVENTS
        original = json.loads(event_path.read_text())
        event_path.write_text(json.dumps(original, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "receipt differs|identity mismatch"):
            recovery.verify_validator_revision(f.root, revised)

        f, output, _ = self.fixture(); revised = self.revised_bundle(f); self.prepare_revision(f, revised)
        output.write_bytes(b"changed accepted output")
        with self.assertRaisesRegex(ValueError, "output changed"):
            recovery.verify_validator_revision(f.root, revised)

    def test_producer_source_model_and_malformed_contract_are_rejected(self):
        f, _, _ = self.fixture(); revised = self.revised_bundle(
            f, extra_change="scripts/rgb_scene_depth_order.py")
        with self.assertRaisesRegex(ValueError, "producer code/assets"):
            self.prepare_revision(f, revised)

        f, _, _ = self.fixture(); revised = self.revised_bundle(f, asset_change=True)
        with self.assertRaisesRegex(ValueError, "external assets|asset"):
            self.prepare_revision(f, revised)

        f, _, _ = self.fixture(); revised = self.revised_bundle(f)
        revision_path = self.prepare_revision(f, revised)
        malformed = load(revision_path); malformed["resume_phase"] = "P09_DEPTH_ORDER"
        write(revision_path, malformed)
        with self.assertRaisesRegex(ValueError, "resume_phase"):
            recovery.verify_validator_revision(f.root, revised)


if __name__ == "__main__":
    unittest.main()
