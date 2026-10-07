from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data.v2_rl_adapter import inspect_v2_manifest, require_trainable_v2_adapter


class V2RLAdapterContractTests(unittest.TestCase):
    def _write(self, payload: dict) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        path = Path(handle.name)
        handle.close()
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(path.unlink)
        return path

    def test_preprocessing_only_v2_is_explicitly_blocked(self):
        path = self._write({
            "schema": "ego.annotation.observation_bundle.v2",
            "preprocessing_only": True,
        })
        contract = inspect_v2_manifest(path)
        self.assertEqual(contract.status, "blocked_not_trainable")
        self.assertFalse(contract.document()["trainable"])
        self.assertIn("reward_profile", contract.missing_contract_fields)
        with self.assertRaisesRegex(RuntimeError, "not trainable"):
            require_trainable_v2_adapter(path)

    def test_declared_contract_still_requires_runtime_promotion(self):
        fields = {
            name: {"declared": True}
            for name in (
                "action_layout_version", "observation_layout_version", "reward_profile",
                "evidence_qualification", "visibility_uncertainty",
                "factor_activation_provenance", "geometry_roles", "source_hashes",
            )
        }
        path = self._write({
            "schema": "ego.annotation.observation_bundle.v2",
            "preprocessing_only": True,
            "rl_training_contract": fields,
        })
        contract = inspect_v2_manifest(path)
        self.assertEqual(contract.status, "ready_pending_runtime_adapter")
        with self.assertRaises(NotImplementedError):
            require_trainable_v2_adapter(path)


if __name__ == "__main__":
    unittest.main()
