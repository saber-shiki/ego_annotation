from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from envs.observation import HISTORY_FEATURE_SIZE, ObservationLayout, _history_features


class CausalHistoryTests(unittest.TestCase):
    def test_history_is_fixed_width_left_padded_and_finite(self):
        layout = ObservationLayout(action_size=4, history_frames=3)
        self.assertEqual(layout.size, 3 + 20 + 2 * (1 + 3 + 45 + 3 + 3 + 45 + 3 + 1 + 21 + 15 + 3 + 2)
                         + 1 * (3 + 3 + 3 + 3 + 3 + 8 * 3) + 2 * 1 + 4 + 3 * HISTORY_FEATURE_SIZE)
        state = SimpleNamespace(history=[np.arange(HISTORY_FEATURE_SIZE, dtype=np.float32)])
        values = _history_features(state, 3)
        self.assertEqual(values.shape, (3 * HISTORY_FEATURE_SIZE,))
        np.testing.assert_array_equal(values[:2 * HISTORY_FEATURE_SIZE], 0.0)
        np.testing.assert_array_equal(values[-HISTORY_FEATURE_SIZE:], np.arange(HISTORY_FEATURE_SIZE, dtype=np.float32))

    def test_history_does_not_accept_malformed_or_nonfinite_rows(self):
        state = SimpleNamespace(history=[
            np.ones(HISTORY_FEATURE_SIZE + 1, dtype=np.float32),
            np.full(HISTORY_FEATURE_SIZE, np.nan, dtype=np.float32),
        ])
        values = _history_features(state, 2)
        np.testing.assert_array_equal(values, 0.0)


if __name__ == "__main__":
    unittest.main()
