"""Timing parser edge cases; real CFR/VFR files are covered by the video tests."""

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v19_video_timing import (
    _frame_pts,
    read_cfr_video_timing,
    validate_package_cfr_times,
)


class VideoTimingTests(unittest.TestCase):
    def probe(self, pts, time_base="1/1000", last_duration=33):
        payload = {
            "streams": [
                {
                    "avg_frame_rate": "30/1",
                    "time_base": time_base,
                    "width": 640,
                    "height": 480,
                }
            ],
            "frames": [{"pts": p, "pkt_duration": last_duration} for p in pts],
        }
        result = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
        with patch("v19_video_timing.subprocess.run", return_value=result):
            return read_cfr_video_timing("fixture.mp4", expected_frame_count=len(pts))

    def test_actual_clock_quantization_not_frame_fraction(self):
        timing = self.probe([1000, 1033, 1067])
        self.assertEqual(timing.frame_rate, Fraction(30))
        self.assertEqual(timing.presentation_times_s[0], 1.0)
        self.assertAlmostEqual(timing.max_grid_error_s, 1 / 3000)
        self.assertAlmostEqual(timing.report()["duration_s"], 0.1)
        with self.assertRaisesRegex(ValueError, "not CFR"):
            self.probe([1000, 1045, 1067])

    def test_coarse_integral_clock_does_not_allow_a_one_tick_frame_shift(self):
        timing = self.probe([0, 1, 2], "1/30", 1)
        self.assertEqual(timing.max_grid_error_s, 0)
        with self.assertRaisesRegex(ValueError, "not CFR"):
            self.probe([0, 2, 3], "1/30", 1)

    def test_no_inferred_or_repeated_presentation_timestamps(self):
        with self.assertRaisesRegex(ValueError, "presentation timestamp"):
            self.probe([0, None, 67])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.probe([0, 0, 67])

    def test_final_frame_duration_cannot_extend_or_shorten_output_silently(self):
        with self.assertRaisesRegex(ValueError, "final frame duration"):
            self.probe([0, 512, 1024], "1/15360", 700)

    def test_explicit_ffprobe_pts_aliases_not_best_effort(self):
        self.assertEqual(_frame_pts({"pkt_pts": 512}), 512)
        self.assertEqual(_frame_pts({"pts": 512, "pkt_pts": 512}), 512)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            _frame_pts({"pts": 512, "pkt_pts": 513})
        with self.assertRaisesRegex(ValueError, "missing"):
            _frame_pts({"best_effort_timestamp": 512})

    def test_package_tolerance_is_numeric_not_source_clock_resolution(self):
        self.assertLess(
            validate_package_cfr_times([10.0, 10.033333, 10.066667], Fraction(30)), 1e-6
        )
        with self.assertRaisesRegex(ValueError, "constant-rate"):
            validate_package_cfr_times([0.0, 0.045, 2 / 30], Fraction(30))


if __name__ == "__main__":
    unittest.main()
