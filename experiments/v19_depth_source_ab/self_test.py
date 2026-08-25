#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


freeze_builder = load_module(
    "depth_ab_freeze_builder", ROOT / "scripts/build_v19_depth_source_ab_freeze_contract.py"
)
pair_verifier = load_module(
    "depth_ab_pair_verifier", ROOT / "scripts/verify_v19_depth_source_ab_pair.py"
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class DepthSourceABContractTest(unittest.TestCase):
    def make_shared(self, root: Path, anchor_roles: list[str] | None = None) -> SimpleNamespace:
        root = root.resolve()
        video = root / "input.mp4"
        video.write_bytes(b"synthetic-video")
        rgb_dir = root / "rgb"
        mask_dir = root / "masks"
        rgb_dir.mkdir(); mask_dir.mkdir()
        frames = []
        track = {}
        for idx in range(2):
            rgb = rgb_dir / f"{idx:06d}.jpg"; rgb.write_bytes(f"rgb-{idx}".encode())
            mask = mask_dir / f"{idx:06d}.png"; mask.write_bytes(f"mask-{idx}".encode())
            frames.append({
                "frame_idx": idx, "rgb": str(rgb), "raw_frame_path": str(rgb),
                "source_video": str(video), "source_width": 4, "source_height": 3,
            })
            track[str(idx)] = {"visible": True, "mask_path": str(mask)}
        manifest = root / "manifest.json"
        write_json(manifest, {"status": "ok", "frames": frames})
        camera = root / "camera.json"
        write_json(camera, {
            "status": "ok", "frame_ids": [0, 1],
            "source_plane_intrinsics_fx_fy_cx_cy": [8.0, 9.0, 2.0, 1.5],
            "calibration_authority": "prediction_side_sensor_metadata",
        })
        camera_npz = root / "camera.npz"
        np.savez_compressed(camera_npz, frame_idx=np.asarray([0, 1], dtype=np.int32))
        hawor = root / "hawor.npz"
        R = np.repeat(np.eye(3)[None], 2, axis=0)
        t = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        K = np.repeat(np.asarray([[8.0, 9.0, 2.0, 1.5]]), 2, axis=0)
        np.savez_compressed(
            hawor, frame_idx=np.asarray([0, 1], dtype=np.int32), R_c2w=R, t_c2w=t,
            camera_intrinsics_fx_fy_cx_cy=K,
            video_sha256=np.asarray(freeze_builder.sha256_file(video)),
        )
        plan = root / "plan.json"
        write_json(plan, {
            "status": "ok", "case": "case", "objects": [{"object_id": "object", "track_id": "object"}],
        })
        prompt = root / "owl_prompt.json"
        write_json(prompt, {
            "case_id": "case", "object_id": "object", "track_id": "object",
            "prompt_source": "owlv2_text_grounded_detector_boxes", "point_prompts": [{"frame_idx": 0}],
        })
        owl_report = root / "owl_report.json"
        write_json(owl_report, {
            "status": "ok", "case_id": "case", "object_id": "object", "track_id": "object",
            "raw_frame_manifest": str(manifest), "output_prompt_json": str(prompt),
            "prompt_frames": [0], "missing_prompt_frames": [],
        })
        sam2 = root / "sam2_track.json"; write_json(sam2, track)
        mano = root / "mano.npz"; np.savez_compressed(mano, frame_idx=np.asarray([0, 1]))
        base = root / "base.json"
        write_json(base, {
            "status": "ok", "frames": [{"frame_idx": i, "raw_frame_path": frames[i]["rgb"]} for i in range(2)],
            "v19_inputs": {
                "raw_frame_manifest": str(manifest), "calibration_contract": str(camera),
                "hawor_npz": str(hawor), "object_plan": str(plan), "mano_bridge_npz": str(mano),
            },
        })
        base_report = root / "base_report.json"
        write_json(base_report, {
            "status": "ok", "case": "case", "frame_count": 2,
            "outputs": {"annotations": str(base), "mano_bridge": str(mano)},
        })
        review = root / "anchor_review.jpg"; review.write_bytes(b"synthetic-depth-independent-review")
        anchor = root / "anchor.json"
        write_json(anchor, {
            "schema": freeze_builder.ANCHOR_SCHEMA, "status": "ok", "object_id": "object",
            "selected_anchor_frame_idx": 1, "depth_source_independent": True,
            "selection_evidence_roles": anchor_roles or ["rgb", "sam2_mask", "semantic_identity"],
            "forbidden_evidence_roles_acknowledged": [
                "metric_depth", "depth_confidence", "provider_specific_p09_score",
            ],
            "review_image": str(review),
            "review_image_sha256": freeze_builder.sha256_file(review),
        })
        return SimpleNamespace(
            case_id="case", object_id="object", manifest=manifest, camera_contract=camera,
            camera_intrinsics=camera_npz, hawor_npz=hawor, object_plan=plan,
            owlv2_prompt=prompt, owlv2_report=owl_report, sam2_track=sam2,
            base_annotations=base, base_report=base_report, mano_bridge=mano,
            anchor_decision=anchor, output=root / "freeze.json", replace=False,
        )

    def write_depths(self, root: Path, freeze: dict, da3_scale_mode: str = "nested_metric_branch", w2c_delta: float = 0.0):
        K = np.repeat(np.asarray([freeze["official_source_intrinsics_fx_fy_cx_cy"]]), 2, axis=0)
        common = dict(
            frame_idx=np.asarray([0, 1], dtype=np.int32),
            confidence=np.ones((2, 3, 4), dtype=np.float16), source_size=np.asarray([4, 3], dtype=np.int32),
            intrinsics_fx_fy_cx_cy=K, depth_ray_geometry_reprojected=np.asarray(True),
            inference_camera_contract_sha256=np.asarray(freeze["camera_contract_sha256"]),
            inference_camera_output_plane=np.asarray("source_rgb"),
            camera_contract_sha256=np.asarray(freeze["camera_contract_sha256"]),
            camera_contract_plane=np.asarray("source_rgb"),
            metadata_only_ray_relabel_override=np.asarray(False),
        )
        uni = root / "uni.npz"
        np.savez_compressed(
            uni, **common, depth=np.ones((2, 3, 4), dtype=np.float16), depth_provider=np.asarray("unidepth"),
            camera_conditioning_mode=np.asarray("provided_pinhole_intrinsics"),
        )
        with np.load(self.args.hawor_npz, allow_pickle=False) as h:
            c2w = np.repeat(np.eye(4)[None], 2, axis=0)
            c2w[:, :3, :3] = h["R_c2w"]; c2w[:, :3, 3] = h["t_c2w"]
        w2c = np.linalg.inv(c2w); w2c[1, 0, 3] += w2c_delta
        da3 = root / "da3.npz"
        np.savez_compressed(
            da3, **common, depth=np.full((2, 3, 4), 1.2, dtype=np.float16),
            depth_provider=np.asarray("depth_anything_3"),
            camera_conditioning_mode=np.asarray("provided_pinhole_intrinsics_and_metric_extrinsics"),
            pose_conditioning_mode=np.asarray("fixed_prediction_side_hawor_metric_w2c"),
            camera_trajectory_source=np.asarray(str(self.args.hawor_npz.resolve())),
            inference_camera_extrinsics_w2c=w2c.astype(np.float32),
            confidence_role=np.asarray("predicted_error_proxy_higher_is_worse"),
            overlap_consistency_passed=np.asarray(True), overlap_consistency_required=np.asarray(True),
            metric_scale_mode=np.asarray(da3_scale_mode), metric_scale_source=np.asarray("synthetic"),
        )
        return uni, da3

    def pair_args(self, root: Path, uni: Path, da3: Path) -> SimpleNamespace:
        return SimpleNamespace(
            freeze_contract=self.args.output, unidepth_depth=uni, da3_depth=da3,
            output=root / "pair.json", replace=False,
        )

    def test_successful_pair_binds_every_shared_byte_and_fixed_w2c(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_ok_") as temp:
            root = Path(temp); self.args = self.make_shared(root)
            freeze = freeze_builder.build(self.args)
            self.assertEqual(freeze["asset_counts"], {"fixed": 14, "raw_rgb": 2, "sam2_mask": 2, "total": 18})
            uni, da3 = self.write_depths(root, freeze)
            pair = pair_verifier.verify(self.pair_args(root, uni, da3))
            self.assertEqual(pair["status"], "ready_for_frozen_upstream_depth_provider_branches")
            self.assertEqual(pair["anchor"]["frame_idx"], 1)
            self.assertLess(
                pair["branches"]["da3_nested_official_K_hawor_conditioned"]["fixed_w2c_max_abs_error"],
                1.0e-8,
            )

    def test_failed_owlv2_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_owl_") as temp:
            self.args = self.make_shared(Path(temp))
            payload = json.loads(self.args.owlv2_report.read_text()); payload["status"] = "failed"
            self.args.owlv2_report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "OWLv2 report"):
                freeze_builder.build(self.args)

    def test_provider_dependent_anchor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_anchor_") as temp:
            self.args = self.make_shared(Path(temp), ["rgb", "metric_depth"])
            with self.assertRaisesRegex(RuntimeError, "provider-dependent"):
                freeze_builder.build(self.args)

    def test_changed_frozen_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_mask_") as temp:
            root = Path(temp); self.args = self.make_shared(root)
            freeze = freeze_builder.build(self.args); uni, da3 = self.write_depths(root, freeze)
            Path(freeze["sam2_mask_assets"][0]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "frozen shared upstream changed"):
                pair_verifier.verify(self.pair_args(root, uni, da3))

    def test_tampered_freeze_contract_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_id_") as temp:
            root = Path(temp); self.args = self.make_shared(root)
            freeze = freeze_builder.build(self.args); uni, da3 = self.write_depths(root, freeze)
            payload = json.loads(self.args.output.read_text()); payload["contract_id"] = "0" * 64
            self.args.output.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "contract ID"):
                pair_verifier.verify(self.pair_args(root, uni, da3))

    def test_da3_umeyama_scale_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_scale_") as temp:
            root = Path(temp); self.args = self.make_shared(root)
            freeze = freeze_builder.build(self.args); uni, da3 = self.write_depths(root, freeze, "input_trajectory_umeyama")
            with self.assertRaisesRegex(RuntimeError, "metric scale mode is not eligible"):
                pair_verifier.verify(self.pair_args(root, uni, da3))

    def test_da3_different_w2c_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="depth_ab_w2c_") as temp:
            root = Path(temp); self.args = self.make_shared(root)
            freeze = freeze_builder.build(self.args); uni, da3 = self.write_depths(root, freeze, w2c_delta=0.1)
            with self.assertRaisesRegex(RuntimeError, "conditioned W2C differs"):
                pair_verifier.verify(self.pair_args(root, uni, da3))


if __name__ == "__main__":
    unittest.main()
