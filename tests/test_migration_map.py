from __future__ import annotations

import importlib
import json
from pathlib import Path
import unittest


class ActiveMigrationOwnerTests(unittest.TestCase):
    def test_active_symbol_owners_import_and_exist(self):
        payload = json.loads((Path(__file__).parents[1] / "migration-map.json").read_text())
        owners = payload.get("active_symbol_ownership")
        self.assertIsInstance(owners, dict)
        self.assertTrue(owners)
        for name, record in owners.items():
            with self.subTest(name=name):
                self.assertIn("historical_source_owner", record)
                module_name, symbol_name = record["active_owner"]
                self.assertTrue(hasattr(importlib.import_module(module_name), symbol_name))


if __name__ == "__main__":
    unittest.main()
