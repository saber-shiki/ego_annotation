#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The wrapper imports the sibling verifier by module name, matching bundled CLI use.
verifier = load_module("verify_v19_depth_source_ab_pair", ROOT / "scripts/verify_v19_depth_source_ab_pair.py")
wrapper = load_module("depth_ab_p09_pair", ROOT / "scripts/run_v19_depth_source_ab_p09_pair.py")


class P09PairWrapperTest(unittest.TestCase):
    def branch(self, root: Path, name: str, mask_payloads: list[bytes], states: list[str]):
        masks = {}
        for idx, value in enumerate(mask_payloads):
            path = root / name / f"{idx:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            masks[idx] = {
                "path": str(path), "bytes": path.stat().st_size, "sha256": wrapper.sha256_file(path),
            }
        return {
            "object_owned_masks": masks,
            "shared_frame_state_sha256": {idx: state for idx, state in enumerate(states)},
        }

    def test_identical_masks_and_shared_state_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p09_pair_ok_") as temp:
            root = Path(temp)
            branches = {
                wrapper.BRANCH_ORDER[0]: self.branch(root, "a", [b"m0", b"m1"], ["s0", "s1"]),
                wrapper.BRANCH_ORDER[1]: self.branch(root, "b", [b"m0", b"m1"], ["s0", "s1"]),
            }
            result = wrapper.compare_branch_outputs(branches, [0, 1])
            self.assertTrue(result["object_owned_masks_byte_identical"])
            self.assertTrue(result["camera_hand_shared_frame_state_identical"])

    def test_mask_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p09_pair_mask_") as temp:
            root = Path(temp)
            branches = {
                wrapper.BRANCH_ORDER[0]: self.branch(root, "a", [b"m0", b"m1"], ["s0", "s1"]),
                wrapper.BRANCH_ORDER[1]: self.branch(root, "b", [b"m0", b"different"], ["s0", "s1"]),
            }
            with self.assertRaisesRegex(RuntimeError, "object-owned masks differ"):
                wrapper.compare_branch_outputs(branches, [0, 1])

    def test_camera_hand_state_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p09_pair_state_") as temp:
            root = Path(temp)
            branches = {
                wrapper.BRANCH_ORDER[0]: self.branch(root, "a", [b"m0", b"m1"], ["s0", "s1"]),
                wrapper.BRANCH_ORDER[1]: self.branch(root, "b", [b"m0", b"m1"], ["s0", "changed"]),
            }
            with self.assertRaisesRegex(RuntimeError, "camera/hand shared state differs"):
                wrapper.compare_branch_outputs(branches, [0, 1])


if __name__ == "__main__":
    unittest.main()
