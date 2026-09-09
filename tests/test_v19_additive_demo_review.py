"""CPU-only negative tests for review-bound demo identity and atomic execution."""
import contextlib
import copy
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_v19_additive_demo as prepare
import run_v19_additive_demo_branch as runner
import v19_runtime_assets as assets


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class SelectedTrackIdentity(unittest.TestCase):
    def test_two_tracks_and_selection_case_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = dump(root / "tracks/first/sam2_track.json", {"3": {}})
            b = dump(root / "tracks/second/sam2_track.json", {"3": {}})
            ann = dump(root / "visible/annotations.json", {
                "v19_visible_geometry_adapter": {"case": "clip-A", "object_id": "object-A",
                    "track_id": "first", "sam2_track": str(a)},
                "frames": [{"objects": [{"object_id": "object-A", "track_id": "first"}]}],
            })
            dump(ann.parent / "v19_visible_geometry_adapter_report.json", {
                "case": "clip-A", "object_id": "object-A", "track_id": "first",
                "inputs": {"sam2_track": str(a)},
            })
            dump(root / "measurements/object_candidates/object_plan_agent.json", {
                "objects": [{"object_id": "object-A", "track_id": "first"},
                            {"object_id": "object-B", "track_id": "second"}],
            })
            selection = {"case": "clip-A", "object_id": "object-A"}
            runtime = {"case": "clip-A"}
            bound = prepare.selected_track_binding(root, "object-A", selection, runtime, ann, a)
            self.assertEqual(bound["track_id"], "first")
            with self.assertRaisesRegex(ValueError, "supplied SAM2 track"):
                prepare.selected_track_binding(root, "object-A", selection, runtime, ann, b)
            for case in ("different", "../unsafe"):
                with self.assertRaisesRegex(ValueError, "case"):
                    prepare.selected_track_binding(root, "object-A", {**selection, "case": case}, runtime, ann, a)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                prepare.selected_track_binding(root, "object-A", {}, {"case": "../unsafe"}, ann, a)


class AtomicDemoExecution(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "run"
        self.bundle = self.base / "bundle"
        self.cfg = dump(self.bundle / "runtime/runtime_config.json", {
            "main_python": sys.executable, "mano_models": str(self.bundle / "models"),
        })
        self.helper = self.bundle / "scripts/phase_helper.py"
        self.helper.parent.mkdir(parents=True)
        self.helper.write_text("# bound helper\n")
        self.manifest = dump(self.bundle / "runtime/bundle_manifest.json", {
            "files": {str(p.relative_to(self.bundle)): runner.sha(p) for p in (self.cfg, self.helper)},
            "external_assets": {"files": {}, "trees": {}},
        })
        self.live = dump(self.root / "input/runtime_input_contract.json", {
            "case": "clip-A", "run_id": "run-A", "run_root": str(self.root),
            "input_video_sha256": "video-A", "bundle_manifest_sha256": "source-A",
        })
        self.sources = {}
        for name in ("raw_frame_manifest", "annotations", "sam2_track", "p15", "p14", "completion",
                     "mano_bridge", "hawor", "source_video", "judgment"):
            self.sources[name] = dump(self.root / "input" / (name + ".json"), {"source": name})
        dump(self.sources["judgment"], {
            "case": "clip-A", "target_entity_id": "object:object-A", "status": "ok",
            "interaction_judgments": [{"start_frame": 10, "end_frame": 10,
                "hand_side": "left", "target_entity_id": "object:object-A",
                "contact_state": "possible_contact"}],
        })
        self.sources["runtime_input"] = self.live
        self.mask = dump(self.root / "input/mask.png", {"mask": "byte fixture"})
        self.rgb = dump(self.root / "input/rgb.png", {"rgb": "byte fixture"})
        self.contract = {
            "mode": "additive_demo_visual_v2", "authority": "demo_only_visual_hypothesis",
            "core_outputs_are_read_only": True, "run_root": str(self.root),
            "case_id": "clip-A", "object_id": "object-A", "track_id": "first", "demo_id": "v2",
            "run_identity": {k: runner.load(self.live)[k] for k in runner.RUN_IDENTITY_KEYS},
            "runtime_config": str(self.cfg), "compute_host": socket.gethostname(),
            "physical_gpu": 5, "minimum_free_mib": 100, "render_width": 80,
            "demo_bundle_manifest_sha256": runner.sha(self.manifest),
            "source_execution_bundle_manifest": str(self.manifest),
            "source_execution_bundle_manifest_sha256": runner.sha(self.manifest),
            "inputs": {k: {"path": str(p), "sha256": runner.sha(p)} for k, p in self.sources.items()},
            "sam2_mask_inputs": {"10": {"path": str(self.mask), "sha256": runner.sha(self.mask)}},
            "raw_frame_inputs": {"10": {"path": str(self.rgb), "sha256": runner.sha(self.rgb)}},
            "sam3d_reconditioning": {"enabled": False},
        }
        self.receipts = self.root / "state/demo_visual/object-A/v2/receipts"
        self.contract_path = dump(self.receipts / "launch_contract.json", self.contract)
        self.calls = []

    def invoke(self, launch=None, probe_error=None):
        # Real per-bundle file hashing; external heavy assets are irrelevant to these fixtures.
        with (patch.object(runner, "ROOT", self.bundle),
              patch.object(sys, "argv", ["runner", "--contract", str(self.contract_path)]),
              patch.object(assets, "runtime_asset_subset", side_effect=lambda identity, config, role: identity),
              patch.object(runner, "gpu_probe", side_effect=probe_error, return_value={"test": "cpu"}) as probe,
              patch.object(runner.subprocess, "run", side_effect=launch or self.launch) as process,
              contextlib.redirect_stdout(io.StringIO())):
            try:
                runner.main()
            finally:
                self.probe_count = probe.call_count
                self.process_count = process.call_count

    def launch(self, argv, **kwargs):
        self.calls.append(argv)
        def value(flag):
            return Path(argv[argv.index(flag) + 1])
        script = Path(argv[1]).name
        if script == "build_v19_demo_pose_binding.py":
            dump(value("--output"), {"render_mesh_contract": {"required_mesh_path": str(value("--published-output-mesh"))}})
            value("--output-mesh").write_bytes(b"test mesh")
        else:
            out = value("--output-dir")
            out.mkdir(parents=True)
            names = {
                "build_v19_demo_visual_inputs.py": ["appearance_mask_manifest.json", "anchor_decision.json"],
                "run_v19_demo_visual_sam3d.py": ["visual_completion.json", "original_canonical_mesh.ply"],
                "estimate_v19_demo_visual_scale.py": ["visual_scale_record.json"],
                "fit_v20_visual_contact_demo.py": ["demo_state.npz", "fit_report.json", "contact_hypotheses.json"],
                "render_v20_visual_contact_demo.py": ["interaction_demo.mp4", "contact_detail.mp4", "interaction_demo.rrd", "render_report.json"],
            }[script]
            for name in names:
                (out / name).write_bytes(b"test phase output")
        return SimpleNamespace(returncode=0)

    def enable_native_and_scale(self):
        self.contract["sam3d_reconditioning"] = {
            "enabled": True, "mode": "native_demo_appearance_sam3d", "seed": 42,
            "core_completion_fallback": False,
        }
        self.contract["visual_scale_alignment"] = {
            "enabled": True, "mode": "image_uniform_grid", "bounds": [.8, 1.2],
            "samples": 41, "max_frames": 7, "raster_width": 192,
        }
        dump(self.contract_path, self.contract)

    def test_enabled_native_and_scale_route_published_outputs_to_binding(self):
        self.enable_native_and_scale()
        self.invoke()
        self.assertEqual([Path(call[1]).name for call in self.calls], [
            "build_v19_demo_visual_inputs.py", "run_v19_demo_visual_sam3d.py",
            "estimate_v19_demo_visual_scale.py", "build_v19_demo_pose_binding.py",
            "fit_v20_visual_contact_demo.py", "render_v20_visual_contact_demo.py"])
        bind = self.calls[3]
        self.assertEqual(bind[bind.index("--completion-report") + 1], str(self.sources["completion"]))
        self.assertEqual(bind[bind.index("--visual-completion-report") + 1],
                         str(self.root / "measurements/demo_visual/object-A/v2/visual_sam3d/visual_completion.json"))
        self.assertEqual(bind[bind.index("--visual-scale-record") + 1],
                         str(self.root / "measurements/demo_visual/object-A/v2/visual_scale/visual_scale_record.json"))
        result = runner.load(self.receipts / "branch_result.json")
        for event in result["events"]:
            for path, digest in event["output_sha256"].items():
                self.assertEqual(runner.sha(path), digest)

    def test_enabled_native_failure_stops_before_binding_without_core_fallback(self):
        self.enable_native_and_scale()
        def fail_native(argv, **kwargs):
            result = self.launch(argv, **kwargs)
            if Path(argv[1]).name == "run_v19_demo_visual_sam3d.py":
                return SimpleNamespace(returncode=7)
            return result
        with self.assertRaisesRegex(RuntimeError, "D00b_visual_sam3d failed"):
            self.invoke(fail_native)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.root / "state/demo_visual/object-A/v2/visual_binding").exists())
        self.assertTrue((self.root / "measurements/demo_visual/object-A/v2/.D00b_visual_sam3d.failed").is_dir())

    def test_enabled_invalid_scale_bounds_stop_before_any_phase(self):
        self.enable_native_and_scale()
        self.contract["visual_scale_alignment"]["bounds"] = [-1., 1.2]
        dump(self.contract_path, self.contract)
        with self.assertRaisesRegex(ValueError, "invalid image scale"):
            self.invoke()
        self.assertEqual(self.process_count, 0)

    def test_helper_or_config_mutation_and_config_redirect_fail_before_probe(self):
        for changed in (self.helper, self.cfg):
            original = changed.read_bytes()
            changed.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "bundle file missing/changed"):
                self.invoke()
            self.assertEqual(self.probe_count, 0)
            self.assertEqual(self.process_count, 0)
            self.assertTrue((self.receipts / "failure.json").is_file())
            (self.receipts / "failure.json").unlink()
            changed.write_bytes(original)
        redirected = dump(self.base / "other_config.json", runner.load(self.cfg))
        dump(self.contract_path, {**self.contract, "runtime_config": str(redirected)})
        with self.assertRaisesRegex(ValueError, "runtime config must"):
            self.invoke()
        self.assertEqual(self.probe_count, 0)

    def test_stale_run_identity_rejected_before_probe(self):
        live = runner.load(self.live)
        live["run_id"] = "different-run"
        dump(self.live, live)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.invoke()
        self.assertEqual(self.probe_count, 0)
        self.assertEqual(self.process_count, 0)

    def test_probe_failure_receipt(self):
        with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
            self.invoke(probe_error=RuntimeError("GPU unavailable"))
        self.assertEqual(self.process_count, 0)
        self.assertIn("GPU unavailable", runner.load(self.receipts / "failure.json")["error"])

    def test_post_phase_core_mutation_is_not_rebound_and_stage_not_published(self):
        def mutate(argv, **kwargs):
            result = self.launch(argv, **kwargs)
            self.sources["p15"].write_bytes(b"new core bytes")
            return result
        with self.assertRaisesRegex(ValueError, "contract input after phase changed"):
            self.invoke(mutate)
        self.assertEqual(self.process_count, 1)
        parent = self.root / "measurements/demo_visual/object-A/v2"
        self.assertFalse((parent / "appearance_evidence").exists())
        self.assertTrue((parent / ".D00_appearance_anchor.failed").is_dir())
        self.assertFalse((self.receipts / "branch_result.json").exists())

    def test_generated_input_mutation_by_consumer_is_rejected(self):
        def mutate(argv, **kwargs):
            result = self.launch(argv, **kwargs)
            if len(self.calls) == 2:
                manifest = self.root / "measurements/demo_visual/object-A/v2/appearance_evidence/appearance_mask_manifest.json"
                manifest.write_bytes(b"changed by consumer")
            return result
        with self.assertRaisesRegex(ValueError, "generated phase input after consumer changed"):
            self.invoke(mutate)
        self.assertEqual(self.process_count, 2)
        self.assertFalse((self.root / "state/demo_visual/object-A/v2/visual_binding").exists())

    def test_mutation_between_phases_is_rejected_before_next_consumer(self):
        real_publish = runner.atomic_publish
        def publish(staging, destination):
            real_publish(staging, destination)
            self.mask.write_bytes(b"changed between phases")
        with patch.object(runner, "atomic_publish", side_effect=publish):
            with self.assertRaisesRegex(ValueError, "contract input before phase changed"):
                self.invoke()
        self.assertEqual(self.process_count, 1)

    def test_publication_cannot_rehash_changed_generated_bytes(self):
        real_publish = runner.atomic_publish
        def publish(staging, destination):
            real_publish(staging, destination)
            (destination / "appearance_mask_manifest.json").write_bytes(b"publication mutation")
        with patch.object(runner, "atomic_publish", side_effect=publish):
            with self.assertRaisesRegex(ValueError, "published phase output changed"):
                self.invoke()
        self.assertFalse((self.root / "measurements/demo_visual/object-A/v2/appearance_evidence").exists())
        self.assertEqual(self.process_count, 1)

    def test_interpreter_and_launch_errors_have_receipts(self):
        with patch.object(runner, "python", side_effect=ValueError("interpreter unavailable")):
            with self.assertRaisesRegex(ValueError, "interpreter unavailable"):
                self.invoke()
        self.assertEqual(self.probe_count, 0)
        self.assertIn("interpreter unavailable", runner.load(self.receipts / "failure.json")["error"])
        (self.receipts / "failure.json").unlink()
        with self.assertRaisesRegex(OSError, "launch failed"):
            self.invoke(launch=lambda *args, **kwargs: (_ for _ in ()).throw(OSError("launch failed")))
        failure = runner.load(self.receipts / "failure.json")
        self.assertEqual(failure["events"][0]["phase"], "D00_appearance_anchor")
        self.assertIsNone(failure["events"][0]["returncode"])
        self.assertTrue(failure["events"][0]["argv"])

    def test_prepare_uses_real_judgment_lineage_without_core_factors(self):
        # Synthetic on-disk producers exercise RunLineage itself, not a permissive mock.
        import uuid
        from v19_demo_timeline import interaction_states
        root, obj = self.root, "object-A"
        live = {**runner.load(self.live), "run_id": str(uuid.uuid4()),
                "input_video": str(self.sources["source_video"]),
                "input_video_sha256": runner.sha(self.sources["source_video"]),
                "bundle_manifest_sha256": runner.sha(self.manifest)}
        dump(self.live, live)
        identity = {k: live[k] for k in runner.RUN_IDENTITY_KEYS}
        dump(self.bundle / ".pi/agents/ego-annotation.rgb-sam3d-demo.md", {})
        dump(root / "state/object_selection.json", {"object_id": obj, "case": "clip-A"})
        raw = dump(root / "input/raw_frame_manifest/manifest.json", {
            "full_source_timeline": True,
            "frames": [{"frame_idx": 10, "time_s": 0.0, "raw_frame_path": str(self.rgb)}],
        })
        track = dump(root / "measurements/object_tracks/first/sam2_track.json", {
            "10": {"visible": True, "mask_path": str(self.mask)},
        })
        visible = root / f"measurements/object_geometry/visible_geometry/{obj}"
        adapter = {"case": "clip-A", "object_id": obj, "track_id": "first", "sam2_track": str(track)}
        ann = dump(visible / "annotations_v19_visible_geometry.json", {
            "v19_visible_geometry_adapter": adapter,
            "frames": [{"frame_idx": 10, "time_s": 0.0,
                        "objects": [{"object_id": obj, "track_id": "first"}]}],
        })
        adapter_path = dump(visible / "v19_visible_geometry_adapter_report.json", {
            **adapter, "inputs": {"sam2_track": str(track)},
        })
        def provenance(phase, evidence):
            return {"kind": "current_run_visual_judgment", "phase": phase,
                    "run_identity": identity,
                    "evidence_sha256": {str(p): runner.sha(p) for p in evidence}}
        dump(root / "measurements/object_candidates/object_plan_agent.json", {
            "objects": [{"object_id": obj, "track_id": "first"}],
            "visual_judgment_provenance": provenance("P05", [raw]),
        })
        p14 = dump(root / f"measurements/pose_fits/{obj}/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json", {})
        p15 = dump(root / f"measurements/pose_fits/{obj}/temporal/v19_rigid_object_pose_graph_report.json", {})
        mesh = dump(root / "measurements/geometry_completion/mesh.ply", {})
        completion = dump(root / f"measurements/geometry_completion/sam3d_{obj}/anchor_bound_completion_report.json", {
            "render_mesh_contract": {"required_mesh_path": str(mesh), "required_mesh_sha256": runner.sha(mesh)},
        })
        hawor = dump(root / "measurements/hand_candidates/hawor_world/hawor_world_hands.npz", {})
        mano = dump(root / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz", {})
        judgment = dump(root / f"state/agent_interaction_judgments/{obj}.json", {
            **runner.load(self.sources["judgment"]),
            "visual_judgment_provenance": provenance("P17_judgment", [raw, self.rgb, p15]),
        })
        self.assertEqual(interaction_states(runner.load(judgment), [10], obj).tolist(),
                         [["possible_contact", "unresolved"]])
        code = {k: v for k, v in runner.load(self.manifest)["files"].items() if k.endswith((".py", ".sh"))}
        events = []
        for phase, outputs in (("P01", [raw, self.rgb]), ("P15_temporal_pose", [p15]),
                               ("fixture_upstream", [track, self.mask, ann, adapter_path, p14, mesh, completion, hawor, mano])):
            events.append({"phase": phase, "returncode": 0, "run_identity": identity,
                           "code_sha256": code, "input_sha256": {},
                           "outputs": list(map(str, outputs)),
                           "output_sha256": {str(p): runner.sha(p) for p in outputs}})
        log = root / "logs/integration/events.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text("".join(json.dumps(e) + "\n" for e in events))
        argv = ["prepare", "--run-root", str(root), "--source-bundle", str(self.bundle),
                "--demo-bundle", str(self.bundle), "--output-dir", str(self.base / "request"),
                "--sam2-track", str(track), "--demo-id", "judgment-only", "--gpu", "5",
                "--reconstruct-visual-mesh", "--sam3d-seed", "91", "--estimate-visual-scale",
                "--visual-scale-bounds", "0.85", "1.15", "--visual-scale-max-frames", "5"]
        from v19_run_lineage import RunLineage
        with (patch.object(sys, "argv", argv),
              patch.object(assets, "runtime_asset_subset", side_effect=lambda identity, config, role: identity),
              patch.object(prepare, "RunLineage", wraps=RunLineage) as lineage,
              contextlib.redirect_stdout(io.StringIO())):
            prepare.main()
        self.assertEqual(lineage.call_count, 1)
        contract = runner.load(root / f"state/demo_visual/{obj}/judgment-only/receipts/launch_contract.json")
        self.assertNotIn("p17", contract["inputs"])
        self.assertEqual(contract["sam3d_reconditioning"], {"enabled": True,
            "mode": "native_demo_appearance_sam3d", "seed": 91, "core_completion_fallback": False})
        self.assertEqual(contract["visual_scale_alignment"]["bounds"], [.85, 1.15])
        self.assertEqual(contract["visual_scale_alignment"]["max_frames"], 5)
        self.assertEqual(contract["inputs"]["judgment"], {"path": str(judgment), "sha256": runner.sha(judgment)})
        self.assertFalse((root / "measurements/contact_visibility_factors").exists())
        self.assertEqual({e["phase"] for e in events}, {"P01", "P15_temporal_pose", "fixture_upstream"})
        self.assertTrue((self.base / "request/request.json").is_file())

    def test_judgment_mutation_is_rejected_before_probe(self):
        self.sources["judgment"].write_bytes(b"changed visual judgment")
        with self.assertRaisesRegex(ValueError, "contract input changed"):
            self.invoke()
        self.assertEqual(self.probe_count, 0)
        self.assertEqual(self.process_count, 0)

    def test_success_publishes_only_complete_namespaces_and_exact_hashes(self):
        self.assertNotIn("p17", self.contract["inputs"])
        self.invoke()
        self.assertEqual(self.process_count, 4)
        fit = self.calls[2]
        self.assertEqual(fit[fit.index("--interaction-judgments") + 1], str(self.sources["judgment"]))
        self.assertFalse((self.root / "measurements/contact_visibility_factors").exists())
        result = runner.load(self.receipts / "branch_result.json")
        for event in result["events"]:
            self.assertEqual(event["input_sha256"][str(self.sources["judgment"])],
                             runner.sha(self.sources["judgment"]))
        self.assertFalse(result["annotation_ready"])
        self.assertFalse(list(self.root.rglob("*.staging")))
        for event in result["events"]:
            for path, digest in event["output_sha256"].items():
                self.assertEqual(runner.sha(path), digest)
        self.assertFalse((self.root / "logs/integration").exists())


if __name__ == "__main__":
    unittest.main()
