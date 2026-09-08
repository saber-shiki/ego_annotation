"""CPU synthetic contracts for the source-named final annotation exporters."""
import argparse
import builtins
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assemble_v19_annotation_layers import assemble
from render_v19_annotation_layers_rrd import render
from v19_annotation_layers import load_package, sha256


class AnnotationLayersTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        n = 3
        ids = np.array([4, 5, 6])
        T = np.repeat(np.eye(4)[None], n, axis=0)
        T[:, 0, 3] = np.arange(n) * .01
        K = np.tile([400., 410., 320., 240.], (n, 1))
        vertices = np.zeros((n, 778, 3), dtype=np.float32)
        vertices[:, :, 2] = 1
        vertices[:, 1, 0] = .04
        vertices[:, 2, 1] = .05
        joints = vertices[:, :21].copy()
        valid = np.array([True, False, True])
        vertices[1] = np.nan
        joints[1] = np.nan
        self.hawor = dict(frame_idx=ids, R_c2w=T[:, :3, :3], t_c2w=T[:, :3, 3],
                          left_vertices_world_m=vertices, left_joints_world_m=joints,
                          left_faces=np.array([[0, 1, 2]], dtype=np.int32), left_valid=valid)
        np.savez(self.run / "hawor.npz", **self.hawor)
        m = trimesh.Trimesh(vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
        m.export(self.run / "mesh.ply")
        # PLY export uses float32, matching the actual completed mesh contract.
        m = trimesh.load(self.run / "mesh.ply", force="mesh", process=False)
        self.annotations = {"frames": [{"frame_idx": int(fid), "time_s": i / 30.,
            "source_width": 640, "source_height": 480,
            "camera": {"T_world_camera_metric": T[i].tolist(), "intrinsics_fx_fy_cx_cy": K[i].tolist()},
            "hands": [{"hand_side": "left", "visibility_state": "visible" if valid[i] else "unresolved", "uncertainty": ["measurement_uncertainty"]}],
            "objects": [{"object_id": "thing", "visibility_state": "partially_visible"}]} for i, fid in enumerate(ids)]}
        self.annotations["v19_inputs"] = {"hawor_npz": str(self.run / "hawor.npz")}
        for i in (0, 2):
            self.annotations["frames"][i]["hands"][0]["metric_mano_state"] = {
                "vertices_reference": {"source_hawor_npz": str(self.run / "hawor.npz"),
                                       "source_frame_index": i,
                                       "source_vertices_world_array": "left_vertices_world_m",
                                       "source_joints_world_array": "left_joints_world_m"}}
        self.dump("annotations.json", self.annotations)
        self.dump("p15.json", {"annotation_ready": False, "pose_rows": []})
        self.dump("p16.json", {"constraint_rows": [{"original": True}]})
        self.dump("p18b.json", {"per_frame_states": [{"source_joints_preserved": True}]})
        self.dump("completion.json", {"outputs": {"completed_mesh": str(self.run / "mesh.ply")}})
        self.judgments = {"case": "synthetic", "interaction_judgments": [{"frame_start": 4, "frame_end": 6,
            "hand_side": "left", "contact_state": "unresolved", "occlusion_relation": "unresolved",
            "contact_support_uncertainty_m": .04}]}
        self.dump("p17.json", self.judgments)
        R = np.repeat(np.eye(3)[None], n, axis=0)
        t = np.tile([0., 0., 1.], (n, 1))
        self.core = {"method": "build_v19_rigid_render_state", "case": "synthetic", "object_id": "thing",
            "annotation_ready": False, "physical_state_quarantined": True,
            "inputs": {"annotations": str(self.run / "annotations.json"), "pose_report": str(self.run / "p15.json"),
                       "constraint_report": str(self.run / "p16.json"), "temporal_mano_state": str(self.run / "p18b.json"),
                       "completion_report": str(self.run / "completion.json"), "completed_mesh": str(self.run / "mesh.ply")},
            "annotation_backbone": {"frame_ids": ids.tolist()},
            "object_geometry": {"mesh_frame": "completed_canonical", "completed_mesh_path": str(self.run / "mesh.ply"),
                                "physical_surface": {"mesh": str(self.run / "mesh.ply"), "signed_geometry_ready": False}},
            "object_pose_trajectory": {"pose_rows": [{"frame_idx": int(ids[i]),
                "rotation_world_from_completed_canonical_matrix": R[i].tolist(), "translation_world_m": t[i].tolist()} for i in [0, 2]]},
            "mano_constraint_state": {"constraint_rows": [{"original": True}]},
            "temporal_mano_state": {"payload": {"per_frame_states": [{"source_joints_preserved": True}]}}}
        self.dump("core.json", self.core)
        self.demo = dict(frame_idx=ids, time_s=np.arange(n)/30., render_size_wh=np.array([320, 240]),
            intrinsics_render=np.c_[K[:, :2] / 2, (K[:, 2:] + .5) / 2 - .5], T_world_camera=T,
            object_vertices_canonical=np.asarray(m.vertices), object_faces=np.asarray(m.faces),
            base_object_rotation=R, base_object_translation_world=t, object_rotation=R,
            object_translation_world=t + [.03, 0, 0], left_valid=valid,
            left_vertices_world=vertices + [.02, 0, 0], left_joints_world=joints + [.02, 0, 0],
            left_base_vertices_world=vertices, left_base_joints_world=joints, left_faces=self.hawor["left_faces"])
        np.savez(self.run / "demo.npz", **self.demo)
        self.dump("demo.json", {"annotation_ready": True, "demo_only": True, "corrections": "test"})
        self.args = argparse.Namespace(render_state=self.run / "core.json", annotations=self.run / "annotations.json",
            hawor_npz=self.run / "hawor.npz", interaction_judgments=self.run / "p17.json",
            demo_state=self.run / "demo.npz", demo_report=self.run / "demo.json", output_dir=self.root / "final",
            run_root=self.run, allow_external_replay=False)

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def test_source_immutability_layer_separation_and_missing_hand(self):
        before = {str(p): sha256(p) for p in self.run.iterdir()}
        path = assemble(self.args)
        package, layers = load_package(path)
        self.assertEqual(package["metric_prediction"]["render_state"], self.core)
        self.assertEqual(package["metric_prediction"]["interaction_judgments"], self.judgments)
        self.assertFalse(package["demo_visual"]["annotation_ready"])
        self.assertFalse(package["metric_prediction"]["render_state"]["annotation_ready"])
        self.assertEqual(before, {str(p): sha256(p) for p in self.run.iterdir()})
        self.assertFalse(layers["prediction"]["right_valid"].any())
        self.assertNotIn("right_vertices_world", layers["prediction"])
        self.assertFalse(layers["prediction"]["object_valid"][1])
        self.assertTrue(layers["demo"]["object_valid"][1])
        self.assertAlmostEqual(layers["demo"]["left_vertices_world"][0, 0, 0] - layers["prediction"]["left_vertices_world"][0, 0, 0], .02)
        self.assertEqual(package["timeline"][1]["object"]["uncertainty"], ["missing_object_pose"])
        # Relocation and removal of every source prove renderer is package-only.
        shutil.rmtree(self.run)
        moved = self.root / "moved"
        shutil.move(self.args.output_dir, moved)
        load_package(moved / path.name)

    def test_actual_mesh_rrd_smoke(self):
        path = assemble(self.args)
        shutil.rmtree(self.run)
        target = render(path, self.root / "world.rrd")
        self.assertGreater(target.stat().st_size, 1000)
        # RRD carries actual mesh geometry components, not only labels.
        raw = target.read_bytes()
        self.assertIn(b"Mesh3D", raw)
        self.assertIn(b"/world/prediction", raw)
        self.assertIn(b"/world/demo", raw)

    def test_missing_components(self):
        del self.hawor["left_vertices_world_m"]
        np.savez(self.args.hawor_npz, **self.hawor)
        with self.assertRaisesRegex(ValueError, "required component"):
            assemble(self.args)

    def test_wrong_transform(self):
        self.annotations["frames"][0]["camera"]["T_world_camera_metric"][0][0] = 2
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "rigid rotation"):
            assemble(self.args)

    def test_timeline_camera_mesh_demo_mismatches(self):
        mutations = {"frame_idx": self.demo["frame_idx"][::-1],
                     "time_s": self.demo["time_s"] + .2,
                     "intrinsics_render": self.demo["intrinsics_render"] + 1,
                     "object_vertices_canonical": self.demo["object_vertices_canonical"] + .1,
                     "T_world_camera": self.demo["T_world_camera"] + .1,
                     "base_object_translation_world": self.demo["base_object_translation_world"] + .2}
        for key, value in mutations.items():
            with self.subTest(key=key):
                np.savez(self.args.demo_state, **{**self.demo, key: value})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    assemble(self.args)

    def test_hash_missing_and_path_escape_rejected(self):
        path = assemble(self.args)
        original = json.loads(path.read_text())
        relative = original["metric_prediction"]["source_assets"]["hawor_npz"]
        asset = path.parent / relative
        raw = asset.read_bytes()
        asset.write_bytes(raw + b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_package(path)
        asset.unlink()
        with self.assertRaises(FileNotFoundError):
            load_package(path)
        asset.write_bytes(raw)
        for bad in ("../escape.npz", "/tmp/escape.npz", "assets/../../escape.npz", "assets\\escape.npz"):
            with self.subTest(path=bad):
                data = json.loads(json.dumps(original))
                data["assets"][bad] = data["assets"].pop(relative)
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "path"):
                    load_package(path)

    def test_fresh_boundary_and_labeled_replay(self):
        external = self.root / "external.npz"
        shutil.copyfile(self.args.hawor_npz, external)
        self.args.hawor_npz = external
        with self.assertRaisesRegex(ValueError, "escapes run-root"):
            assemble(self.args)
        self.args.allow_external_replay = True
        path = assemble(self.args)
        self.assertEqual(load_package(path)[0]["run_mode"], "external_replay_not_fresh")

    def test_package_metadata_tampering(self):
        path = assemble(self.args)
        original = path.read_text()
        for field in ("frame_idx", "camera", "interaction_judgments", "authority"):
            with self.subTest(field=field):
                data = json.loads(original)
                if field == "frame_idx":
                    data["timeline"][0]["frame_idx"] += 1
                elif field == "camera":
                    data["timeline"][0]["camera"]["intrinsics_fx_fy_cx_cy"][0] += 1
                elif field == "interaction_judgments":
                    data["timeline"][0][field] = []
                else:
                    data["demo_visual"]["annotation_ready"] = True
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_package(path)

    def test_render_arrays_cannot_disagree_even_with_updated_hash(self):
        path = assemble(self.args)
        package = json.loads(path.read_text())
        relative = package["metric_prediction"]["render_arrays_asset"]
        asset = path.parent / relative
        with np.load(asset, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["intrinsics_render"][0, 0] += 10
        np.savez_compressed(asset, **arrays)
        package["assets"][relative]["sha256"] = sha256(asset)
        package["assets"][relative]["size_bytes"] = asset.stat().st_size
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "render array mismatch: intrinsics_render"):
            load_package(path)

    def test_demo_report_hash_and_source_validation(self):
        self.dump("demo.json", {"inputs": {"hawor_params": str(self.args.hawor_npz)},
                                "input_hashes": {"hawor_params": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            assemble(self.args)

    def test_explicit_missing_physical_components(self):
        self.core["temporal_mano_state"]["payload"] = None
        self.dump("core.json", self.core)
        with self.assertRaisesRegex(ValueError, "required component: P18"):
            assemble(self.args)

    def test_source_camera_and_validity_mismatch(self):
        self.hawor["R_c2w"][0, 0, 0] = 2
        np.savez(self.args.hawor_npz, **self.hawor)
        with self.assertRaisesRegex(ValueError, "HaWoR world camera rotation"):
            assemble(self.args)
        self.hawor["R_c2w"][0, 0, 0] = 1
        del self.hawor["left_valid"]
        np.savez(self.args.hawor_npz, **self.hawor)
        with self.assertRaisesRegex(ValueError, "required component: left_valid"):
            assemble(self.args)

    def test_declared_hawor_rejects_surface_and_invalid_row_substitution(self):
        for kind in ("vertices", "validity"):
            with self.subTest(kind=kind):
                replacement = {k: v.copy() for k, v in self.hawor.items()}
                if kind == "vertices":
                    replacement["left_vertices_world_m"][0, 300, 0] += .1
                else:
                    replacement["left_valid"][1] = True
                    replacement["left_vertices_world_m"][1] = 0
                    replacement["left_joints_world_m"][1] = 0
                self.args.hawor_npz = self.run / "replacement.npz"
                np.savez(self.args.hawor_npz, **replacement)
                with self.assertRaisesRegex(ValueError, "source archive mismatch"):
                    assemble(self.args)

    def test_per_hand_source_binding_without_global_reference(self):
        del self.annotations["v19_inputs"]
        self.dump("annotations.json", self.annotations)
        self.args.hawor_npz = self.run / "replacement.npz"
        replacement = {k: v.copy() for k, v in self.hawor.items()}
        replacement["left_valid"][1] = True
        replacement["left_vertices_world_m"][1] = 0
        replacement["left_joints_world_m"][1] = 0
        np.savez(self.args.hawor_npz, **replacement)
        with self.assertRaisesRegex(ValueError, "source archive mismatch"):
            assemble(self.args)

    def test_declared_source_array_and_row_binding(self):
        ref = self.annotations["frames"][0]["hands"][0]["metric_mano_state"]["vertices_reference"]
        ref["source_frame_index"] = 2
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "source row mismatch"):
            assemble(self.args)
        ref["source_frame_index"] = 0
        ref["source_vertices_world_array"] = "left_joints_world_m"
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "source vertices mismatch"):
            assemble(self.args)

    def test_hawor_binding_replayed_after_relocation_and_rejects_substitution(self):
        # Content-identical copies are legitimate, but both identities are saved.
        self.args.hawor_npz = self.run / "copy.npz"
        shutil.copyfile(self.run / "hawor.npz", self.args.hawor_npz)
        path = assemble(self.args)
        shutil.rmtree(self.run)
        moved = self.root / "moved"
        shutil.move(path.parent, moved)
        path = moved / path.name
        package, _ = load_package(path)
        relative = package["metric_prediction"]["source_assets"]["hawor_npz"]
        asset = moved / relative
        replacement = {k: v.copy() for k, v in self.hawor.items()}
        replacement["left_vertices_world_m"][0, 300, 0] += .1
        np.savez(asset, **replacement)
        package["assets"][relative].update(sha256=sha256(asset), size_bytes=asset.stat().st_size)
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "source archive mismatch"):
            load_package(path)
        package["metric_prediction"]["hawor_source_bindings"] = {}
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "source bindings mismatch"):
            load_package(path)

    def test_same_snapshot_hawor_identity_survives_manifest_rehash(self):
        path = assemble(self.args)
        package = json.loads(path.read_text())
        relative = package["metric_prediction"]["source_assets"]["hawor_npz"]
        replacement = {k: v.copy() for k, v in self.hawor.items()}
        replacement["left_vertices_world_m"][0, 300, 0] += .1
        asset = path.parent / relative
        np.savez(asset, **replacement)
        package["assets"][relative].update(sha256=sha256(asset), size_bytes=asset.stat().st_size)
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "source archive mismatch"):
            load_package(path)

    def test_prefixed_object_track_metadata_and_ambiguity(self):
        for frame in self.annotations["frames"]:
            frame["objects"][0].update(object_id="object:thing", track_id="producer_track", occluder="hand:left")
        self.judgments["interaction_judgments"][0]["target_entity_id"] = "producer_track"
        self.dump("p17.json", self.judgments)
        self.dump("annotations.json", self.annotations)
        path = assemble(self.args)
        package, _ = load_package(path)
        self.assertEqual(package["timeline"][0]["object"]["source_state"], self.annotations["frames"][0]["objects"][0])
        self.assertEqual(package["timeline"][0]["object"]["occluder"], "hand:left")
        self.assertEqual(package["timeline"][0]["interaction_judgments"], self.judgments["interaction_judgments"])
        self.args.output_dir = self.root / "ambiguous"
        self.annotations["frames"][0]["objects"].append({"object_id": "thing"})
        self.dump("annotations.json", self.annotations)
        with self.assertRaisesRegex(ValueError, "ambiguous object/track"):
            assemble(self.args)

    def test_obj_external_material_reads_denied_in_assembly_and_replay(self):
        self.args.demo_state = self.args.demo_report = None
        external = self.root / "outside.mtl"
        external.write_text("newmtl forbidden\nKd 1 0 0\n")
        mesh_path = self.run / "mesh.obj"
        original_open, original_io_open = builtins.open, io.open
        attempted = []
        def guarded(opener):
            def open_file(file, *args, **kwargs):
                if isinstance(file, (str, Path)) and Path(file).name == "outside.mtl":
                    attempted.append(str(file))
                    raise AssertionError("external material read")
                return opener(file, *args, **kwargs)
            return open_file
        for i, reference in enumerate((str(external), "../outside.mtl")):
            with self.subTest(reference=reference):
                mesh_path.write_text(f"mtllib {reference}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl forbidden\nf 1 2 3\n")
                self.core["object_geometry"]["completed_mesh_path"] = str(mesh_path)
                self.dump("core.json", self.core)
                self.args.output_dir = self.root / f"obj_final_{i}"
                with mock.patch("builtins.open", guarded(original_open)), mock.patch("io.open", guarded(original_io_open)):
                    path = assemble(self.args)
                    # In replay, ../outside.mtl would resolve here rather than
                    # relative to the deleted/original mesh source directory.
                    with original_io_open(path.parent / "outside.mtl", "w") as f:
                        f.write("newmtl forbidden\n")
                    load_package(path)
                self.assertEqual(attempted, [])

    def distinct_demo_mesh(self):
        generated = trimesh.Trimesh(
            vertices=[[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1], [.1, .1, .1]],
            faces=[[0, 1, 2], [0, 2, 3], [1, 3, 4], [1, 2, 4]], process=False)
        path = self.run / "intact_generated.ply"
        generated.export(path)
        generated = trimesh.load(path, force="mesh", process=False)
        completion = {"outputs": {"completed_mesh": str(self.run / "mesh.ply")},
                      "render_mesh_contract": {"required_mesh_path": str(path),
                                               "required_mesh_sha256": sha256(path), "renderer_must_match": True}}
        self.dump("completion.json", completion)
        self.demo["object_vertices_canonical"] = np.asarray(generated.vertices)
        self.demo["object_faces"] = np.asarray(generated.faces)
        np.savez(self.args.demo_state, **self.demo)
        self.dump("demo.json", {"demo_only": True, "mesh": str(path), "mesh_sha256": sha256(path)})
        return path, completion

    def test_distinct_completion_bound_demo_mesh_relocated_rrd(self):
        _, completion = self.distinct_demo_mesh()
        before = {str(p): sha256(p) for p in self.run.iterdir()}
        path = assemble(self.args)
        self.assertEqual(before, {str(p): sha256(p) for p in self.run.iterdir()})
        shutil.rmtree(self.run)
        moved = self.root / "relocated"
        shutil.move(path.parent, moved)
        package, layers = load_package(moved / path.name)
        self.assertEqual(package["metric_prediction"]["render_state"], self.core)
        self.assertEqual(len(layers["prediction"]["object_vertices_canonical"]), 4)
        self.assertEqual(len(layers["demo"]["object_vertices_canonical"]), 5)
        self.assertEqual(len(layers["prediction"]["object_faces"]), 2)
        self.assertEqual(len(layers["demo"]["object_faces"]), 4)
        binding = package["demo_visual"]["canonical_mesh_binding"]
        self.assertEqual(binding["reference"], completion["render_mesh_contract"]["required_mesh_path"])
        target = render(moved / path.name, self.root / "distinct.rrd")
        self.assertGreater(target.stat().st_size, 1000)

    def test_distinct_demo_mesh_without_contract_rejected(self):
        self.distinct_demo_mesh()
        self.dump("completion.json", {"outputs": {"completed_mesh": str(self.run / "mesh.ply")}})
        with self.assertRaisesRegex(ValueError, "demo canonical mesh"):
            assemble(self.args)

    def test_distinct_demo_mesh_bad_hash_or_substitution_rejected(self):
        path, completion = self.distinct_demo_mesh()
        completion["render_mesh_contract"]["required_mesh_sha256"] = "0" * 64
        self.dump("completion.json", completion)
        with self.assertRaisesRegex(ValueError, "bound canonical mesh hash mismatch"):
            assemble(self.args)
        completion["render_mesh_contract"]["required_mesh_sha256"] = sha256(path)
        self.dump("completion.json", completion)
        np.savez(self.args.demo_state, **{**self.demo, "object_vertices_canonical": self.demo["object_vertices_canonical"] + .1})
        with self.assertRaisesRegex(ValueError, "demo canonical mesh"):
            assemble(self.args)
        np.savez(self.args.demo_state, **self.demo)
        self.dump("demo.json", {"mesh": str(self.run / "mesh.ply"), "mesh_sha256": sha256(self.run / "mesh.ply")})
        with self.assertRaisesRegex(ValueError, "demo report canonical mesh mismatch"):
            assemble(self.args)

    def test_distinct_demo_retains_base_pose_camera_mano_checks(self):
        self.distinct_demo_mesh()
        for key in ("base_object_translation_world", "T_world_camera", "left_base_vertices_world"):
            with self.subTest(key=key):
                np.savez(self.args.demo_state, **{**self.demo, key: self.demo[key] + .1})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    assemble(self.args)

    def test_distinct_demo_replay_requires_original_contract_binding(self):
        self.distinct_demo_mesh()
        path = assemble(self.args)
        shutil.rmtree(self.run)
        original = path.read_text()
        package = json.loads(original)
        package["demo_visual"]["canonical_mesh_binding"]["asset"] = package["metric_prediction"]["canonical_mesh_asset"]
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "bound canonical mesh hash mismatch"):
            load_package(path)
        package = json.loads(original)
        package["demo_visual"]["canonical_mesh_binding"]["reference"] = "unbound_mesh.ply"
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "canonical mesh binding mismatch"):
            load_package(path)
        package = json.loads(original)
        relative = package["metric_prediction"]["source_assets"]["completion_report"]
        report_path = path.parent / relative
        report = json.loads(report_path.read_text())
        report["render_mesh_contract"]["required_mesh_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
        package["assets"][relative].update(sha256=sha256(report_path), size_bytes=report_path.stat().st_size)
        path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "completion report binding hash mismatch"):
            load_package(path)

    def test_prediction_only(self):
        self.args.demo_state = self.args.demo_report = None
        path = assemble(self.args)
        self.assertEqual(list(load_package(path)[1]), ["prediction"])
        with self.assertRaisesRegex(ValueError, "demo layer"):
            render(path, self.root / "no.rrd", "demo")


if __name__ == "__main__":
    unittest.main()
