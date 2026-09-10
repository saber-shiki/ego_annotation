import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rgb_scene_depth_order as ownership
import build_v19_visible_geometry_from_sam2_depth as visible_geometry
from rgb_scene_metric_eligibility import CLASS_CODES, MetricEligibility, classify_native_depth
from rgb_scene_depth_order_appearance import build as build_appearance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangle_at_depth(z):
    uv = np.array([[1.0, 1.0], [6.0, 1.0], [1.0, 6.0]])
    return np.c_[uv * z, np.full(3, z)]


class DepthOrderNumericsTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((4, 5), bool)

    def test_front_behind_near_nohit_and_unsupported_preservation(self):
        hand = np.full(self.mask.shape, np.inf)
        depth = np.ones(self.mask.shape)
        hand[0, 0] = .8
        hand[0, 1] = 1.2
        hand[0, 2] = 1.003
        hand[0, 3] = .8
        supported = np.ones(self.mask.shape, bool)
        supported[0, 3] = False
        owned, counts, classes = ownership.front_only(
            self.mask, hand, depth, tolerance_m=.005, supported=supported)
        self.assertFalse(owned[0, 0])
        self.assertTrue(owned[0, 1])
        self.assertTrue(owned[0, 2])
        self.assertTrue(owned[0, 3])
        self.assertTrue(owned[1, 1])
        self.assertEqual(counts["removed_front_pixels"], 1)
        self.assertTrue(classes["behind"][0, 1])
        self.assertTrue(classes["near"][0, 2])
        self.assertTrue(classes["unsupported_depth"][0, 3])
        self.assertTrue(classes["no_hit"][1, 1])

    def test_slanted_triangle_uses_perspective_correct_depth(self):
        uv = np.array([[1., 1.], [5., 1.], [1., 5.]])
        z = np.array([1., 2., 4.])
        vertices = np.c_[uv * z[:, None], z]
        raster, _ = ownership.first_hit_z(vertices, np.array([[0, 1, 2]]), np.eye(3), (7, 7))
        expected = 1.0 / (.5 / 1.0 + .25 / 2.0 + .25 / 4.0)
        self.assertAlmostEqual(raster[2, 2], expected, places=10)
        self.assertNotAlmostEqual(raster[2, 2], .5 * 1 + .25 * 2 + .25 * 4, places=3)

    def test_multiple_hands_take_nearest_first_hit(self):
        K = np.eye(3)
        nearest, reports = ownership.nearest_hand_first_hits([
            (triangle_at_depth(2.), np.array([[0, 1, 2]], np.int32)),
            (triangle_at_depth(.7), np.array([[0, 1, 2]], np.int32)),
        ], K, (8, 8))
        self.assertAlmostEqual(nearest[2, 2], .7)
        self.assertEqual(len(reports), 2)

    def test_near_plane_nan_and_malformed_contracts(self):
        vertices = np.array([[0., 0., .001], [1., 0., 1.], [0., 1., 1.]])
        _, report = ownership.first_hit_z(vertices, np.array([[0, 1, 2]], np.int32),
                                          np.eye(3), (5, 5), near_m=.01)
        self.assertEqual(report["clipped_faces"], 1)
        vertices[0, 0] = np.nan
        raster, report = ownership.first_hit_z(vertices, np.array([[0, 1, 2]], np.int32),
                                               np.eye(3), (5, 5))
        self.assertEqual(report["nonfinite_faces"], 1)
        self.assertFalse(np.isfinite(raster).any())
        for faces in (np.array([[0, 1, 9]], np.int32), np.array([[0., 1., 2.]])):
            with self.assertRaises(ValueError):
                ownership.first_hit_z(np.ones((3, 3)), faces, np.eye(3), (5, 5))
        with self.assertRaises(ValueError):
            ownership.first_hit_z(np.ones((3, 3)), np.array([[0, 1, 2]], np.int32),
                                  np.zeros((3, 3)), (5, 5))

    def test_half_pixel_non_square_depth_sampling(self):
        depth = np.arange(8 * 6).reshape(6, 8)
        # 4x3 -> 8x6 OpenCV half-pixel mapping.
        A = np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]])
        sampled, inside = ownership.sample_depth_at_mask_pixels(depth, A, (3, 4))
        self.assertTrue(inside.all())
        np.testing.assert_array_equal(sampled, depth[np.ix_([0, 2, 4], [0, 2, 4, 6])])


class DepthOrderAdapterTest(unittest.TestCase):
    def test_native_metric_classification_rejects_ambiguity_and_keeps_clear_body_candidates(self):
        mask = np.ones((3, 4), bool)
        depth = np.ones((6, 8), float)
        hand = np.full(depth.shape, np.inf)
        hand[3, 3] = 1.003
        A = np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]])
        codes, eligible, counts = classify_native_depth(
            owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A,
            complete_hand_geometry=True, validated_hand_absence=False,
            tolerance_m=.005, uncertainty_padding_mask_px=0)
        self.assertFalse(eligible[3, 3])
        self.assertEqual(codes[3, 3], CLASS_CODES["near_depth_ambiguous"])
        self.assertGreater(counts["clear_no_hand_overlap_candidate"], 0)
        self.assertTrue(eligible[-2:, -2:].any())
        self.assertEqual(codes[0, 0], CLASS_CODES["invalid_mask_coordinate"])
        self.assertFalse(eligible[0, 0])
        hand[:] = np.inf; hand[3, 3] = 1.02
        codes, eligible, _ = classify_native_depth(
            owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A,
            complete_hand_geometry=True, validated_hand_absence=False,
            tolerance_m=.005, uncertainty_padding_mask_px=1)
        self.assertEqual(codes[3, 4], CLASS_CODES["no_hit_uncertainty_fringe"])
        self.assertFalse(eligible[3, 4])
        self.assertTrue(eligible[4, 6])
        hand[:] = 1.02
        codes, eligible, _ = classify_native_depth(
            owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A,
            complete_hand_geometry=True, validated_hand_absence=False,
            tolerance_m=.005, uncertainty_padding_mask_px=1)
        self.assertTrue(eligible[1:-1, 1:-1].all())
        self.assertTrue((codes[1:-1, 1:-1] == CLASS_CODES["hand_clearly_behind"]).all())
        depth[3, 3] = np.nan
        codes, eligible, _ = classify_native_depth(
            owned_mask=mask, hand_z=hand, depth_z=depth, A_depth_from_mask=A,
            complete_hand_geometry=True, validated_hand_absence=False,
            tolerance_m=.005, uncertainty_padding_mask_px=1)
        self.assertEqual(codes[3, 3], CLASS_CODES["unsupported_depth"])
        self.assertFalse(eligible[3, 3])

    def test_missing_hand_surface_is_never_metric_but_appearance_survives(self):
        mask = np.ones((5, 7), bool)
        codes, eligible, _ = classify_native_depth(
            owned_mask=mask, hand_z=np.full(mask.shape, np.inf), depth_z=np.ones(mask.shape),
            A_depth_from_mask=np.eye(3), complete_hand_geometry=False,
            validated_hand_absence=False, tolerance_m=.005,
            uncertainty_padding_mask_px=4)
        self.assertFalse(eligible.any())
        self.assertTrue((codes == CLASS_CODES["unresolved_hand_geometry"]).all())
        self.assertTrue(mask.all())

    def make_contract(self, root, frame_ids):
        contract = {
            "schema": "v19_camera_image_transform_contract_v2",
            "calibration_authority": "estimated_from_rgb_depth_model",
            "intrinsics_model": "constant_pinhole_fx_fy_cx_cy",
            "intrinsics_coordinate_plane": "calibration",
            "intrinsics_fx_fy_cx_cy": [4., 4., 3., 3.],
            "K": [[4., 0., 3.], [0., 4., 3.], [0., 0., 1.]],
            "frame_ids": frame_ids,
            "calibration_plane": {"width": 8, "height": 8},
            "image_planes": {},
        }
        for name in ("source_rgb", "sam2_mask"):
            contract["image_planes"][name] = {
                "plane_name": name, "width": 8, "height": 8,
                "A_plane_from_calibration": np.eye(3).tolist(),
                "transform_kind": "identity", "interpolation": "nearest_for_masks",
                "pixel_center_convention": "integer_pixel_centers_opencv",
            }
        path = root / "camera.json"
        path.write_text(json.dumps(contract))
        return path

    def test_preprocessor_drives_owned_track_with_arbitrary_ids(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            frame_ids = [17, 42, 77, 88]
            contract = self.make_contract(root, frame_ids)
            K4 = np.tile(np.array([4., 4., 3., 3.]), (4, 1))
            depth_path = root / "depth.npz"
            np.savez(depth_path, frame_idx=frame_ids, depth=np.ones((4, 8, 8), np.float32),
                     confidence=np.ones((4, 8, 8), np.float32), intrinsics_fx_fy_cx_cy=K4,
                     camera_contract_path=str(contract), camera_contract_sha256=digest(contract),
                     camera_contract_plane="source_rgb", A_depth_from_calibration=np.eye(3),
                     intrinsics_source="fixture", calibration_authority="estimated_from_rgb_depth_model",
                     depth_ray_geometry_reprojected=True, metadata_only_ray_relabel_override=False,
                     source_estimated_intrinsics_fx_fy_cx_cy=K4)
            cover_uv = np.array([[-100., -100.], [200., -100.], [-100., 200.]])
            vertices = np.stack([np.c_[cover_uv * z, np.full(3, z)] for z in (.5, 2., 1.003)])
            bridge_path = root / "bridge.npz"
            np.savez(bridge_path, frame_idx=frame_ids[:3], hand_side=np.array(["left", "left", "left"]),
                     vertices_current_v18_camera_m=vertices)
            hawor_path = root / "hawor.npz"
            np.savez(hawor_path, left_faces=np.array([[0, 1, 2]], np.int32),
                     right_faces=np.array([[0, 1, 2]], np.int32))
            track = {}
            frames = []
            for pos, idx in enumerate(frame_ids):
                mask_path = root / f"raw_{idx}.png"
                cv2.imwrite(str(mask_path), np.ones((8, 8), np.uint8) * 255)
                track[str(idx)] = {"visible": True, "mask_path": str(mask_path), "area_px": 64.}
                ref = {"bridge_npz": str(bridge_path), "source_hawor_npz": str(hawor_path),
                       "bridge_row_index": pos,
                       "bridge_vertices_camera_array": "vertices_current_v18_camera_m"}
                rgb_path = root / f"rgb_{idx}.png"
                cv2.imwrite(str(rgb_path), np.full((8, 8, 3), 80 + pos * 20, np.uint8))
                frames.append({"frame_idx": idx, "raw_frame_path": str(rgb_path),
                              "camera": {"T_world_camera_metric": np.eye(4).tolist()},
                              "hands": ([{"hand_side": "left",
                              "metric_mano_state": {"vertices_reference": ref}}] if pos < 3 else [])})
            base = root / "base.json"
            base.write_text(json.dumps({"case": "Case.9", "frames": frames}))
            raw = root / "raw_track.json"
            raw.write_text(json.dumps(track))
            out = root / "owned"
            args = argparse.Namespace(base_annotations=base, raw_sam2_track=raw, depth_npz=depth_path,
                calibration_contract=contract, mano_bridge=bridge_path, hawor_npz=hawor_path,
                output_dir=out, case="Case.9", object_id="object-A.2", track_id="track_Z-7",
                depth_image_plane="source_rgb", mask_image_plane="sam2_mask",
                depth_order_tolerance_m=.005, near_m=1e-6, uncertainty_padding_mask_px=4)
            report = ownership.build(args)
            owned_track = json.loads((out / "sam2_track.json").read_text())
            front = cv2.imread(owned_track["17"]["mask_path"], 0) > 0
            behind = cv2.imread(owned_track["42"]["mask_path"], 0) > 0
            near = cv2.imread(owned_track["77"]["mask_path"], 0) > 0
            self.assertLess(front.sum(), 64)
            self.assertEqual(behind.sum(), 64)
            self.assertEqual(near.sum(), 64)
            self.assertGreater(report["totals"]["removed_front_pixels"], 0)
            self.assertGreater(report["totals"]["preserved_behind_pixels"], 0)
            self.assertEqual(report["object_id"], "object-A.2")
            eligibility_path = out / "native_depth_metric_eligibility.json"
            self.assertEqual(report["outputs"]["native_depth_metric_eligibility"]["sha256"], digest(eligibility_path))
            raw_manifest = root / "raw_manifest.json"
            raw_manifest.write_text(json.dumps({"frames": [{"frame_idx": idx,
                "raw_frame_path": str(root / f"rgb_{idx}.png"), "source_width": 8, "source_height": 8}
                for idx in frame_ids]}))
            plan = root / "plan.json"
            plan.write_text(json.dumps({"objects": [{"object_id": "object-A.2", "track_id": "track_Z-7",
                                                       "description": "fixture"}]}))

            def p09(frame_idx, destination, *, eligibility=True):
                argv = ["p09", "--case", "Case.9", "--track-id", "track_Z-7",
                        "--object-id", "object-A.2", "--raw-frame-manifest", str(raw_manifest),
                        "--sam2-track-json", str(out / "sam2_track.json"),
                        "--depth-npz", str(depth_path), "--output-dir", str(destination),
                        "--base-annotations", str(base), "--calibration-contract", str(contract),
                        "--depth-image-plane", "source_rgb", "--mask-image-plane", "sam2_mask",
                        "--object-plan", str(plan), "--frame-start", str(frame_idx), "--frame-end", str(frame_idx),
                        "--propose-anchor-candidates-only", "--pixel-stride", "1", "--min-valid-points", "1",
                        "--no-robust-first-surface-depth-ownership", "--no-exclude-hand-bboxes"]
                if eligibility:
                    argv.extend(["--metric-eligibility-manifest", str(eligibility_path)])
                with patch.object(sys, "argv", argv):
                    return visible_geometry.build(visible_geometry.parse_args())

            proposal = p09(42, root / "p09_behind")
            self.assertEqual(proposal["status"], "anchor_candidates_proposed_only")
            proposed = json.loads((root / "p09_behind/anchor_candidate_proposals.json").read_text())
            eligibility_summary = proposed["ranked_candidates"][0]["native_depth_metric_eligibility"]
            self.assertEqual(eligibility_summary["counts"]["hand_clearly_behind"], 64)
            with self.assertRaisesRegex(RuntimeError, "no visible metric geometry"):
                p09(77, root / "p09_near")
            legacy = p09(77, root / "p09_near_legacy_default", eligibility=False)
            self.assertEqual(legacy["status"], "anchor_candidates_proposed_only")
            legacy_proposal = json.loads((root / "p09_near_legacy_default/anchor_candidate_proposals.json").read_text())
            self.assertGreater(legacy_proposal["ranked_candidates"][0]["valid_depth_mask_pixels"], 0)
            clear = p09(88, root / "p09_clear_nonoverlap")
            self.assertEqual(clear["status"], "anchor_candidates_proposed_only")
            clear_proposal = json.loads((root / "p09_clear_nonoverlap/anchor_candidate_proposals.json").read_text())
            self.assertEqual(clear_proposal["ranked_candidates"][0]["native_depth_metric_eligibility"]["counts"]
                             ["clear_no_hand_overlap_candidate"], 64)

            manifest_bytes = eligibility_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            self.assertGreater(manifest["frames"][0]["counts"]["hand_clearly_front"], 0)
            mutations = []
            bad_depth = json.loads(manifest_bytes); bad_depth["inputs"]["depth_npz"]["sha256"] = "0" * 64
            mutations.append(("depth input hash", eligibility_path, json.dumps(bad_depth).encode(), "depth_npz"))
            bad_affine = json.loads(manifest_bytes); bad_affine["frames"][1]["A_depth_from_mask"][0][2] += .25
            mutations.append(("depth affine", eligibility_path, json.dumps(bad_affine).encode(), "affine"))
            raster = Path(manifest["frames"][1]["raster"]["path"])
            mutations.append(("eligibility raster", raster, raster.read_bytes() + b"changed", "raster"))
            owned_mask_path = Path(owned_track["42"]["mask_path"])
            changed_owned = cv2.imread(str(owned_mask_path), cv2.IMREAD_GRAYSCALE)
            changed_owned[0, 0] = 0
            changed_owned_path = root / "changed_owned.png"
            cv2.imwrite(str(changed_owned_path), changed_owned)
            mutations.append(("owned mask", owned_mask_path, changed_owned_path.read_bytes(), "owned mask"))
            changed_base = json.loads(base.read_text()); changed_base["mutation"] = True
            mutations.append(("MANO/base lineage", base, json.dumps(changed_base).encode(), "recorded input"))
            for number, (label, path, changed, message) in enumerate(mutations):
                with self.subTest(consumption_mutation=label):
                    original = path.read_bytes(); path.write_bytes(changed)
                    try:
                        with self.assertRaisesRegex((ValueError, RuntimeError), message):
                            p09(42, root / f"p09_mutated_{number}")
                    finally:
                        path.write_bytes(original)
            self.assertEqual(eligibility_path.read_bytes(), manifest_bytes)
            self.assertEqual(report["outputs"]["owned_sam2_track"]["sha256"], digest(out / "sam2_track.json"))

    def test_corrected_p09_uses_owned_but_appearance_uses_raw(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "input").mkdir()
            (root / "state").mkdir()
            raw_rgb = root / "input" / "rgb.png"
            image = np.indices((16, 16)).sum(axis=0).astype(np.uint8) * 8
            cv2.imwrite(str(raw_rgb), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            raw_mask = root / "input" / "raw.png"
            owned_mask = root / "input" / "owned.png"
            cv2.imwrite(str(raw_mask), np.ones((16, 16), np.uint8) * 255)
            owned_array = np.ones((16, 16), np.uint8) * 255
            owned_array[:, :8] = 0
            cv2.imwrite(str(owned_mask), owned_array)
            raw_track = root / "input" / "raw_track.json"
            owned_track = root / "state" / "owned_track.json"
            raw_track.write_text(json.dumps({"0": {"visible": True, "mask_path": str(raw_mask)}}))
            owned_track.write_text(json.dumps({"0": {"visible": True, "mask_path": str(owned_mask)}}))
            base, depth, camera = (root / "state" / name for name in ("base.json", "depth.bin", "camera.json"))
            for path in (base, depth, camera):
                path.write_text(path.name)
            raw_manifest = root / "input" / "manifest.json"
            raw_manifest.write_text(json.dumps({"frames": [{"frame_idx": 0, "raw_frame_path": str(raw_rgb)}]}))
            adapter = {"case": "caseX", "object_id": "objX", "track_id": "trackX",
                       "sam2_track": str(owned_track), "base_annotations": str(base),
                       "depth_npz": str(depth), "calibration_contract": str(camera)}
            annotations = root / "state" / "p09.json"
            points = [[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.]]
            annotations.write_text(json.dumps({"v19_visible_geometry_adapter": adapter, "frames": [{
                "frame_idx": 0, "time_s": 0., "camera": {"T_world_camera_metric": np.eye(4).tolist()},
                "objects": [{"object_id": "objX", "mask_path": str(owned_mask),
                    "visible_geometry_candidate": {"world_vertices_sample_m": points,
                    "camera_vertices_sample_m": points, "centroid_world_m": [0.03, .03, 1.]}}]}]}))
            def rec(path):
                return {"path": str(path), "sha256": digest(path)}
            ownership_report = root / "state" / "ownership.json"
            ownership_report.write_text(json.dumps({"schema": ownership.SCHEMA, "status": "ok",
                "case": "caseX", "object_id": "objX", "track_id": "trackX",
                "inputs": {"raw_sam2_track": rec(raw_track), "base_annotations": rec(base),
                           "depth_npz": rec(depth), "calibration_contract": rec(camera)},
                "outputs": {"owned_sam2_track": rec(owned_track)}}))
            out = root / "measurements" / "demo_visual" / "objX" / "appearance_evidence"
            appearance_args = argparse.Namespace(
                run_root=root, object_id="objX", case_id="caseX", track_id="trackX",
                raw_frame_manifest=raw_manifest, annotations=annotations,
                raw_sam2_track=raw_track, depth_order_report=ownership_report, output_dir=out,
                min_area_fraction=0, min_border_fraction=0, min_sharpness=0)
            manifest, _ = build_appearance(appearance_args)
            copied = Path(manifest["frames"][0]["appearance_mask_path"])
            self.assertEqual(digest(copied), digest(raw_mask))
            self.assertNotEqual(digest(copied), digest(owned_mask))
            self.assertEqual(manifest["depth_order_binding"]["owned_sam2_track"]["path"], str(owned_track))
            self.assertTrue(manifest["raw_appearance_and_metric_eligibility_are_distinct"])
            self.assertIs(manifest["metric_owned_masks_consumed"], False)
            malformed = json.loads(ownership_report.read_text())
            malformed["inputs"]["raw_sam2_track"]["sha256"] = "0" * 64
            ownership_report.write_text(json.dumps(malformed))
            appearance_args.output_dir = root / "measurements" / "demo_visual" / "objX" / "bad"
            with self.assertRaisesRegex(ValueError, "raw SAM2 track path/hash mismatch"):
                build_appearance(appearance_args)

    def test_standalone_wiring_has_no_second_blind_subtraction(self):
        frontend = (SCRIPTS / "run_rgb_scene_frontend.py").read_text()
        stage = (SCRIPTS / "run_rgb_scene_stage.py").read_text()
        self.assertIn("--sam2-track-json',owned_track", frontend)
        self.assertIn("--no-exclude-hand-bboxes", frontend)
        self.assertNotIn("'--exclude-hand-regions'", frontend)
        self.assertIn("rgb_scene_depth_order_appearance.py", stage)
        self.assertIn("--depth-order-report',ownership", stage)
        self.assertIn("--raw-sam2-track',sam2", stage)


if __name__ == "__main__":
    unittest.main()
