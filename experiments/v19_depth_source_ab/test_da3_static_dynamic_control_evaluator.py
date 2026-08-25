#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module(
    "da3_static_dynamic_control_evaluator",
    ROOT / "scripts/evaluate_v19_hot3d_da3_static_dynamic_control.py",
)


class Da3StaticDynamicControlEvaluatorTest(unittest.TestCase):
    def test_motion_classification_is_fail_closed_at_fixed_boundaries(self) -> None:
        self.assertEqual(evaluator.classify_motion(0.001, 0.5, 0.010), "motion_inactive")
        self.assertEqual(evaluator.classify_motion(0.030, 1.0, 0.010), "motion_active")
        self.assertEqual(evaluator.classify_motion(0.005, 6.0, 0.002), "motion_active")
        self.assertEqual(evaluator.classify_motion(0.030, 6.0, 0.030), "motion_intermediate")

    def test_exclusive_ranges_partition_timeline_and_window_clusters_are_unique(self) -> None:
        windows = [
            {"frame_start": 0},
            {"frame_start": 3},
            {"frame_start": 6},
        ]
        self.assertEqual(evaluator.exclusive_evaluation_ranges(windows, 9), [(0, 2), (3, 5), (6, 8)])
        rows = [
            {
                "window_index": 0,
                "motion_class": "motion_active",
                "dynamic_object": "a",
                "da3_dynamic_static_excess_absolute_log_ratio": 0.1,
                "difference_in_differences_absolute_log_ratio": 0.2,
            },
            {
                "window_index": 0,
                "motion_class": "motion_active",
                "dynamic_object": "b",
                "da3_dynamic_static_excess_absolute_log_ratio": 0.3,
                "difference_in_differences_absolute_log_ratio": 0.4,
            },
            {
                "window_index": 1,
                "motion_class": "motion_inactive",
                "dynamic_object": "b",
                "da3_dynamic_static_excess_absolute_log_ratio": -0.1,
                "difference_in_differences_absolute_log_ratio": -0.2,
            },
        ]
        clusters = evaluator.cluster_active_rows_by_window(rows)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["active_object_count"], 2)
        self.assertAlmostEqual(clusters[0]["mean_da3_dynamic_static_excess_absolute_log_ratio"], 0.2)

    def test_gt_only_cell_matching_uses_equal_counts_and_expected_error(self) -> None:
        labels = np.asarray([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ], dtype=np.uint16)
        gt = np.full((4, 4), 0.50, dtype=np.float64)
        gt[labels == 0] = 0.0
        predictions = {
            "unidepth": np.where(labels > 0, gt, np.nan),
            "depth_anything_3": np.where(labels == 1, 1.5 * gt, gt),
        }
        rows = evaluator.matched_cell_rows(
            frame_idx=0,
            dynamic_object="dynamic",
            dynamic_label=1,
            static_labels=[2],
            labels=labels,
            gt_depth_m=gt,
            predictions=predictions,
            principal_point_xy=(1.5, 1.5),
            depth_bin_m=1.0,
            radius_bin_px=10.0,
            max_samples_per_cell=100,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_count_per_role"], 5)
        aggregate = evaluator.aggregate_window(rows)
        da3 = aggregate["providers"]["depth_anything_3"]
        self.assertAlmostEqual(da3["dynamic_absolute_log_ratio"], np.log(1.5), places=12)
        self.assertAlmostEqual(da3["static_absolute_log_ratio"], 0.0, places=12)
        self.assertAlmostEqual(
            aggregate["difference_in_differences_absolute_log_ratio"], np.log(1.5), places=12
        )

    def test_prediction_acceptance_rejects_archive_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="da3_static_dynamic_binding_") as temp:
            root = Path(temp)
            unidepth = root / "unidepth.npz"
            da3 = root / "da3.npz"
            qc = root / "qc.json"
            np.savez(unidepth, value=np.asarray([1]))
            np.savez(da3, value=np.asarray([2]))
            qc.write_text(json.dumps({
                "status": "ok",
                "frame_count": 150,
                "pose_conditioning": {"metric_scale_mode": "nested_metric_branch"},
                "windowing": {"overlap_consistency": {"passed": True}},
            }))
            acceptance = root / "acceptance.json"
            acceptance.write_text(json.dumps({
                "status": "accepted_prediction_frozen_for_evaluator_only_static_dynamic_test",
                "runtime": {"sam3d_invoked": False},
                "assets": {
                    "unidepth": {"path": str(unidepth), "sha256": evaluator.sha256_file(unidepth)},
                    "da3_camera_bound": {"path": str(da3), "sha256": evaluator.sha256_file(da3)},
                    "qc": {"path": str(qc), "sha256": evaluator.sha256_file(qc)},
                },
            }))
            evaluator.validate_prediction_binding(acceptance, unidepth, da3, qc)
            da3.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "differs from frozen prediction acceptance"):
                evaluator.validate_prediction_binding(acceptance, unidepth, da3, qc)

    def test_exact_randomization_detects_consistent_active_excess(self) -> None:
        active = [0.10, 0.11, 0.12, 0.13, 0.14]
        inactive = [-0.01, 0.0, 0.01]
        self.assertEqual(evaluator.exact_sign_flip_p_greater(active), 1.0 / 32.0)
        self.assertLessEqual(evaluator.exact_partition_p_greater(active, inactive), 0.05)
        interval = evaluator.bootstrap_mean_ci(active, seed=1901, draws=1000)
        assert interval is not None
        self.assertGreater(interval[0], 0.0)


if __name__ == "__main__":
    unittest.main()
