"""CPU native-subprocess fixtures: no model or GPU is executed."""
import copy
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_v19_demo_visual_inputs as appearance
import build_v19_demo_pose_binding as binding
import run_v19_demo_visual_sam3d as native
import v19_demo_reconditioning as contracts
import estimate_v19_demo_visual_scale as alignment
from remote_run_sam3d_objects_mesh_v7 import apply_native_pose


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class NativeReconditioning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "run"
        self.obj = "item-Z9"
        self.ns = self.root / "measurements/demo_visual" / self.obj / "new_mesh"
        self.output = self.ns / ".D00b_visual_sam3d.staging"
        self.published = self.ns / "visual_sam3d"
        self.frames, self.raw, self.track, self.poses = [], [], {}, []
        self.T = np.eye(4)
        self.T[:3, :3] = Rotation.from_euler("xyz", [12, -9, 23], degrees=True).as_matrix()
        self.T[:3, 3] = [.2, -.1, .4]
        K = [100., 100., 80., 48.]
        surfels = np.array([[x, y, .7] for x in np.linspace(-.1, .1, 20) for y in np.linspace(-.06, .06, 12)])
        for i, idx in enumerate([11, 29, 57]):
            image = self.root / f"input/rgb/{idx}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image), np.random.default_rng(1).integers(0, 255, (96, 160, 3), dtype=np.uint8))
            mask_path = self.root / f"input/masks/{idx}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((96, 160), np.uint8)
            extent = 36 if idx == 29 else 30
            mask[26:70, 80-extent:80+extent] = 255
            cv2.imwrite(str(mask_path), mask)
            owned = self.root / f"input/owned/{idx}.png"
            owned.parent.mkdir(parents=True, exist_ok=True)
            smaller = np.zeros_like(mask); smaller[38:58, 70:90] = 255
            cv2.imwrite(str(owned), smaller)
            world = surfels @ self.T[:3, :3].T + self.T[:3, 3]
            geom = {"frame_idx": idx, "object_id": self.obj,
                    "world_vertices_sample_m": world.tolist(), "camera_vertices_sample_m": surfels.tolist(),
                    "centroid_world_m": world.mean(axis=0).tolist(),
                    "first_surface_depth_ownership": {"enabled": True, "fail_closed": False, "retained_fraction": .96}}
            self.frames.append({"frame_idx": idx, "time_s": i * .2,
                "source_width": 160, "source_height": 96,
                "camera": {"T_world_camera_metric": self.T.tolist(), "intrinsics_fx_fy_cx_cy": K},
                "objects": [{"object_id": self.obj, "mask_path": str(owned), "visible_geometry_candidate": geom}]})
            self.raw.append({"frame_idx": idx, "raw_frame_path": str(image)})
            self.track[str(idx)] = {"mask_path": str(mask_path), "visible": True}
            self.poses.append({"frame_idx": idx, "time_s": i * .2, "object_id": self.obj,
                "status": binding.DIRECT_STATUS, "rigid_pose_observation_eligible": True,
                binding.ROTATION_KEY: Rotation.from_euler("zyx", [35 + i, 8, -12], degrees=True).as_matrix().tolist(),
                binding.TRANSLATION_KEY: (self.T[:3, :3] @ [0, 0, .7] + self.T[:3, 3]).tolist()})
        self.inputs = {
            "annotations": dump(self.root / "input/annotations.json", {"frames": self.frames}),
            "raw_frame_manifest": dump(self.root / "input/raw.json", {"frames": self.raw}),
            "sam2_track": dump(self.root / "input/track.json", self.track),
            "p14": dump(self.root / "input/p14.json", {"object_id": self.obj, "pose_rows": self.poses}),
            "p15": dump(self.root / "input/p15.json", {"object_id": self.obj, "annotation_ready": False, "pose_rows": []}),
            "completion": dump(self.root / "input/core_completion.json", {"original_anchor_frame_idx": 11}),
        }
        core_mesh = self.root / "input/core_mesh.ply"
        trimesh.Trimesh(vertices=np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]),
                        faces=[[0, 1, 2]], process=False).export(core_mesh)
        dump(self.inputs["completion"], {"original_anchor_frame_idx": 11, "render_mesh_contract": {
            "required_mesh_path": str(core_mesh), "required_mesh_sha256": contracts.record(core_mesh)["sha256"]}})
        self.cfg = dump(Path(self.tmp.name) / "bundle/runtime/runtime_config.json", {
            "main_python": sys.executable, "sam3d_python": sys.executable,
            **{k: str(Path(self.tmp.name) / k) for k in ("sam3d_repo", "sam3d_config", "sam3d_moge_checkpoint", "sam3d_dinov2_repo", "sam3d_dinov2_checkpoint")},
            "sam3d_environment": {"HF_HUB_OFFLINE": "0", "CUDA_VISIBLE_DEVICES": "2"}})
        live = {"case": "case-xy", "run_root": str(self.root), "run_id": "fixture",
                "input_video_sha256": "video", "bundle_manifest_sha256": "bundle"}
        dump(self.root / "input/runtime_input_contract.json", live)
        self.contract = {"mode": "additive_demo_visual_v2", "authority": "demo_only_visual_hypothesis",
            "core_outputs_are_read_only": True,
            "run_root": str(self.root), "object_id": self.obj, "case_id": "case-xy", "demo_id": "new_mesh",
            "run_identity": {k: live[k] for k in ("run_root", "run_id", "input_video_sha256", "bundle_manifest_sha256")},
            "runtime_config": str(self.cfg), "compute_host": socket.gethostname(),
            "physical_gpu": 5, "minimum_free_mib": 23456,
            "inputs": {k: contracts.record(v) for k, v in self.inputs.items()},
            "raw_frame_inputs": {str(r["frame_idx"]): contracts.record(Path(r["raw_frame_path"])) for r in self.raw},
            "sam2_mask_inputs": {k: contracts.record(Path(v["mask_path"])) for k, v in self.track.items()},
            "sam3d_reconditioning": {"enabled": True, "mode": "native_demo_appearance_sam3d", "seed": 73, "core_completion_fallback": False}}
        self.launch = dump(self.root / "state/demo_visual" / self.obj / "new_mesh/receipts/launch_contract.json", self.contract)
        evidence_dir = self.ns / "appearance_evidence"
        appearance.build(self.root, self.obj, self.inputs["raw_frame_manifest"], self.inputs["annotations"], self.inputs["sam2_track"], evidence_dir)
        self.manifest = evidence_dir / "appearance_mask_manifest.json"
        self.anchor = evidence_dir / "anchor_decision.json"
        self.assertEqual(json.loads(self.anchor.read_text())["selected_frame_idx"], 29)
        # Full planar triangle mesh, not a predicted case asset. More than 100
        # distinct projected vertices exercise the unchanged front-depth floor.
        self.raw_vertices = np.array([[x, y, 0] for y in np.linspace(-.3, .3, 19) for x in np.linspace(-.5, .5, 31)])
        self.faces = np.array([[y*31+x, y*31+x+1, (y+1)*31+x] for y in range(18) for x in range(30)] +
                              [[y*31+x+1, (y+1)*31+x+1, (y+1)*31+x] for y in range(18) for x in range(30)])
        self.calls = []
        self.fail = False
        self.tamper_topology = False

    def model(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.fail:
            return SimpleNamespace(returncode=7)
        name, image, mask, seed = argv[argv.index("--case") + 1].split("|")
        output = Path(argv[argv.index("--output-dir") + 1]) / name
        output.mkdir(parents=True)
        pose = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 2], "scale": [1]}
        p3d, cv, pc = apply_native_pose(self.raw_vertices, np.array(pose["rotation"]), np.array(pose["translation"]), np.array(pose["scale"]))
        paths = {}
        for key, vertices in (("mesh", self.raw_vertices), ("native_pose_mesh_opencv_camera", cv), ("native_pose_mesh_pytorch3d_camera", p3d)):
            path = output / (key + ".ply")
            faces = self.faces[::-1] if self.tamper_topology and key == "native_pose_mesh_opencv_camera" else self.faces
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
            paths[key] = str(path)
        (output / "mesh.glb").write_bytes(b"display only, never consumed")
        dump(output / "qc_sam3d_objects_mesh_v7.json", {
            "status": "ok", "name": name, "image": image, "mask": mask, "seed": int(seed),
            **paths, "glb": str(output / "mesh.glb"), "pose": pose, "native_pose_contract": pc})
        return SimpleNamespace(returncode=0)

    def invoke(self):
        with (patch.object(native, "verify_execution_bundle", return_value=({}, {})),
              patch.object(native, "verify_model_assets", return_value={}),
              patch.object(native, "gpu_probe", return_value={"fixture": True}) as probe,
              patch.object(native, "runtime_environment", return_value={}),
              patch.object(native.subprocess, "run", side_effect=self.model)):
            result = native.reconstruct(self.launch, self.manifest, self.anchor, self.output, self.published)
        probe.assert_called_once_with(5, 23456)
        return result

    def refresh_source(self, name, payload):
        dump(self.inputs[name], payload)
        self.contract["inputs"][name] = contracts.record(self.inputs[name])
        dump(self.launch, self.contract)
        if name == "annotations":
            manifest = json.loads(self.manifest.read_text())
            manifest["source_annotations_sha256"] = contracts.record(self.inputs[name])["sha256"]
            dump(self.manifest, manifest)

    def test_native_argv_gauge_and_published_full_mesh_binding(self):
        result = self.invoke()
        argv, kwargs = self.calls[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(Path(argv[1]).name, "remote_run_sam3d_objects_mesh_v7.py")
        self.assertTrue(argv[argv.index("--case") + 1].endswith("|73"))
        for flag, key in (("--repo", "sam3d_repo"), ("--config", "sam3d_config"), ("--moge-checkpoint", "sam3d_moge_checkpoint"), ("--dinov2-repo", "sam3d_dinov2_repo"), ("--dinov2-checkpoint", "sam3d_dinov2_checkpoint")):
            self.assertEqual(argv[argv.index(flag) + 1], json.loads(self.cfg.read_text())[key])
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(kwargs["timeout"], 1200)
        self.assertAlmostEqual(result["scene_depth_similarity"]["scene_similarity_scale"], .35)
        self.assertLess(result["original_canonical_gauge"]["roundtrip_max_abs_native_scene_error"], 1e-5)
        self.assertFalse(result["generated_faces_contact_eligible"])
        self.assertTrue(result["demo_visual_proximity_fit_eligible"])
        self.output.rename(self.published)
        mesh, value = contracts.validate_visual_completion(self.published / "visual_completion.json", self.contract, self.manifest)
        v, f = contracts.mesh_arrays(mesh)
        np.testing.assert_array_equal(f, self.faces)
        cv = (self.raw_vertices + [0, 0, 2]) @ np.diag([-1, -1, 1])
        R, t = binding._pose(self.poses[1])
        np.testing.assert_allclose(v @ R.T + t, (.35 * cv) @ self.T[:3, :3].T + self.T[:3, 3], atol=1e-6)
        self.assertEqual(value["inputs"]["completion"], self.contract["inputs"]["completion"])
        self.assertNotEqual(result["conditioning"]["mask"]["path"], self.frames[1]["objects"][0]["mask_path"])
        # An updated output digest cannot hide topology corruption from replay.
        trimesh.Trimesh(vertices=v, faces=f[::-1], process=False).export(mesh)
        value["render_mesh_contract"]["required_mesh_sha256"] = contracts.record(mesh)["sha256"]
        dump(self.published / "visual_completion.json", value)
        with self.assertRaisesRegex(ValueError, "canonical mesh replay"):
            contracts.validate_visual_completion(self.published / "visual_completion.json", self.contract, self.manifest)

    def test_callable_scale_stage_and_binder_keep_core_identity_and_visual_gauge(self):
        self.contract["visual_scale_alignment"] = {"enabled": True, "mode": "image_uniform_grid",
            "bounds": [.8, 1.2], "samples": 11, "max_frames": 3, "raster_width": 96}
        dump(self.launch, self.contract)
        self.invoke()
        self.output.rename(self.published)
        completion = self.published / "visual_completion.json"
        staged_scale = self.ns / ".D00c_visual_scale.staging"
        report = alignment.align(self.launch, self.manifest, staged_scale, completion)
        self.assertEqual([r["frame_idx"] for r in report["evidence_frames"]], [11, 29, 57])
        self.assertEqual(report["inputs"]["p15"], self.contract["inputs"]["p15"])
        self.assertLessEqual(report["result_objective"], report["initial_objective"])
        scale_dir = self.ns / "visual_scale"
        staged_scale.rename(scale_dir)
        out = self.root / "state/demo_visual" / self.obj / "new_mesh/visual_binding"
        argv = ["binding", "--p15-report", str(self.inputs["p15"]), "--p14-report", str(self.inputs["p14"]),
                "--annotations", str(self.inputs["annotations"]), "--completion-report", str(self.inputs["completion"]),
                "--object-id", self.obj, "--appearance-mask-manifest", str(self.manifest),
                "--output", str(out / "pose_binding.json"), "--output-mesh", str(out / "visual_mesh.ply"),
                "--visual-completion-report", str(completion), "--demo-launch-contract", str(self.launch),
                "--visual-scale-record", str(scale_dir / "visual_scale_record.json")]
        before = {k: v.read_bytes() for k, v in self.inputs.items()}
        with patch.object(sys, "argv", argv):
            binding.main()
        result = json.loads((out / "pose_binding.json").read_text())
        self.assertEqual(result["core_source_identity"]["completion_report"], str(self.inputs["completion"]))
        self.assertEqual(result["core_source_identity"]["p15_sha256"], self.contract["inputs"]["p15"]["sha256"])
        self.assertEqual(result["visual_completion_provenance"]["path"], str(completion))
        self.assertTrue(result["visual_mesh_receipt"]["faces_exactly_preserved"])
        self.assertEqual(result["visual_mesh_receipt"]["scale_alignment"]["uniform_scale"], report["uniform_scale"])
        for key, value in before.items():
            self.assertEqual(self.inputs[key].read_bytes(), value)

    def test_missing_direct_pose_rejected_before_model(self):
        poses = copy.deepcopy(self.poses)
        poses[1] = {"frame_idx": 29, "status": binding.MISSING_STATUS}
        self.refresh_source("p14", {"object_id": self.obj, "pose_rows": poses})
        with self.assertRaisesRegex(ValueError, "direct P14"):
            self.invoke()
        self.assertFalse(self.calls)

    def test_metric_wrong_frame_or_empty_rejected_before_model(self):
        for mutation in ("empty", "wrong_frame", "camera"):
            frames = copy.deepcopy(self.frames)
            geom = frames[1]["objects"][0]["visible_geometry_candidate"]
            if mutation == "empty": geom["camera_vertices_sample_m"] = []
            elif mutation == "wrong_frame": geom["frame_idx"] = 11
            else: frames[1]["camera"]["T_world_camera_metric"][0][0] = 2
            self.refresh_source("annotations", {"frames": frames})
            with self.assertRaises(ValueError): self.invoke()
            self.assertFalse(self.calls)

    def test_source_hash_tamper_rejected_before_model(self):
        self.inputs["p14"].write_text("{}")
        with self.assertRaisesRegex(ValueError, "contract input changed"):
            self.invoke()
        self.assertFalse(self.calls)

    def test_appearance_wrong_frame_rejected_before_model(self):
        anchor = json.loads(self.anchor.read_text())
        anchor["appearance_conditioning"]["raw_rgb"] = self.raw[0]["raw_frame_path"]
        dump(self.anchor, anchor)
        with self.assertRaisesRegex(ValueError, "conditioning binding"):
            self.invoke()
        self.assertFalse(self.calls)

    def test_ready_P15_anchor_uses_ready_pose_not_direct_P14(self):
        ready = copy.deepcopy(self.poses)
        ready[1][binding.TRANSLATION_KEY] = [.3, -.2, .8]
        self.refresh_source("p15", {"annotation_ready": True, "object_id": self.obj, "pose_rows": ready})
        result = self.invoke()
        self.assertEqual(result["original_canonical_gauge"]["initializer_pose"], ready[1])
        self.assertEqual(result["anchor_evidence"]["pose_initializer"]["mode"], "ready_P15_pass_through")

    def test_nonpositive_and_weak_retained_metric_evidence(self):
        for mutation in ("negative_depth", "weak_retained"):
            frames = copy.deepcopy(self.frames)
            geom = frames[1]["objects"][0]["visible_geometry_candidate"]
            if mutation == "negative_depth":
                geom["camera_vertices_sample_m"][0][2] = -.1
            else:
                geom["first_surface_depth_ownership"]["retained_fraction"] = .2
            self.refresh_source("annotations", {"frames": frames})
            anchor = json.loads(self.anchor.read_text())
            anchor["metric_alignment"] = appearance.metric_alignment(frames[1]["objects"][0], frames[1])
            dump(self.anchor, anchor)
            with self.assertRaisesRegex(ValueError, "metric"):
                self.invoke()
            self.assertFalse(self.calls)

    def test_failed_native_never_falls_back(self):
        self.fail = True
        with self.assertRaisesRegex(RuntimeError, "no fallback"):
            self.invoke()
        self.assertTrue((self.output / "native_execution.json").is_file())
        self.assertFalse((self.output / "visual_completion.json").exists())
        self.assertFalse(self.published.exists())

    def test_native_full_topology_tamper(self):
        self.tamper_topology = True
        with self.assertRaisesRegex(ValueError, "topology changed"):
            self.invoke()
        self.assertFalse((self.output / "visual_completion.json").exists())

    def test_output_confinement(self):
        self.output = self.root / "measurements/geometry_completion/forbidden"
        with self.assertRaisesRegex(ValueError, "confined"):
            self.invoke()
        self.assertFalse(self.calls)

    def test_model_asset_role_checks_byte_hashes_before_launch(self):
        checkpoint = Path(self.tmp.name) / "checkpoint"
        checkpoint.write_bytes(b"pinned")
        subset = {"files": {}, "trees": {}, "resolved_targets": {str(checkpoint): {
            "resolved_path": str(checkpoint), "sha256": contracts.record(checkpoint)["sha256"]}}}
        with (patch.object(native, "verify_bundle", return_value={"external_assets": {}}) as bundle,
              patch.object(native, "runtime_asset_subset", return_value=subset)):
            native.verify_model_assets({})
            bundle.assert_called_with(native.ROOT, asset_role="geometry")
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "external asset changed"):
                native.verify_model_assets({})


class ScaleAndAnchor(unittest.TestCase):
    def test_runtime_closure_includes_real_optional_producers(self):
        import prepare_v19_sam3d_runtime as preparation
        root = Path(__file__).resolve().parents[1]
        closure = {str(path) for path in preparation.closure(root)}
        self.assertTrue({"scripts/run_v19_demo_visual_sam3d.py", "scripts/v19_demo_reconditioning.py",
            "scripts/estimate_v19_demo_visual_scale.py", "scripts/remote_run_sam3d_objects_mesh_v7.py",
            "experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py"} <= closure)
        profile = (root / ".pi/agents/ego-annotation.rgb-sam3d-demo.md").read_text()
        self.assertIn("D00b", profile)
        self.assertIn("D00c", profile)
        self.assertNotIn("No new SAM3D inference hook is implemented", profile)
        self.assertIn("timeout=1200", profile)

    def test_nonidentity_static_uniform_scale_recovery_and_bounds(self):
        for shape in ((100, 180), (180, 100)):
            vertices = np.array([[-.3, -.2, 0], [.3, -.2, 0], [.3, .2, 0], [-.3, .2, 0]])
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            K = np.array([140., 140., shape[1] / 2, shape[0] / 2])
            observations = []
            for angle in (0, 7, 13):
                R = Rotation.from_euler("z", angle, degrees=True).as_matrix()
                t = np.array([.02, .01, 1.])
                mask = alignment.silhouette((vertices * 1.14) @ R.T + t, faces, K, shape)
                observations.append({"mask": mask, "K": K, "T": np.eye(4), "R": R, "t": t})
            report = alignment.estimate_scale(vertices, faces, observations)
            self.assertAlmostEqual(report["uniform_scale"], 1.14, places=2)
            self.assertLess(report["result_objective"], report["initial_objective"])
            self.assertFalse(report["boundary_hit"])
            limited = alignment.estimate_scale(vertices, faces, observations, bounds=(.8, 1.1))
            self.assertTrue(limited["boundary_hit"])
            for bounds in ((-1, 1.2), (.8, float("nan")), (1.1, 1.2)):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    alignment.estimate_scale(vertices, faces, observations, bounds=bounds)
            with self.assertRaisesRegex(ValueError, "three valid"):
                alignment.estimate_scale(vertices, faces, observations[:1])

    def test_safely_interior_support_beats_center_bonus(self):
        image = np.zeros((200, 320, 3), np.uint8)
        large = np.zeros((200, 320), np.uint8); large[20:180, 40:150] = 255
        small = np.zeros_like(large); small[60:140, 130:190] = 255
        def score(mask):
            return appearance.score_mask(image, mask, min_area_fraction=.002, min_border_fraction=.005, min_sharpness=0)
        a, b = score(large), score(small)
        self.assertTrue(a["eligible"] and b["eligible"])
        self.assertEqual(a["score_terms"]["border_clearance"], b["score_terms"]["border_clearance"])
        self.assertGreater(a["score"], b["score"])
        truncated = large.copy(); truncated[:40, :40] = 255
        self.assertFalse(score(truncated)["eligible"])


if __name__ == "__main__":
    unittest.main()
