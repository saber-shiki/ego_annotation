#!/usr/bin/env python3
"""Compose and validate the three controlled P15 geometry render branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
import render_v19_rigid_state_artifact as canonical  # noqa: E402

SCHEMA = "v19_experimental_p15_three_branch_video_ab_v1"
EXPECTED_ORDER = [
    "sam3d_owned_dual_mesh",
    "sam3d_owned_legacy_cut",
    "trellis_frozen_legacy_cut",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def prepare_output(path: Path, replace: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not replace:
            raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read render frame: {path}")
    return image


def resize_exact(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def summary(values: list[float | int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def manifest_branch(manifest: dict[str, Any]) -> str:
    branch = manifest.get("branch") if isinstance(manifest.get("branch"), dict) else {}
    return str(branch.get("branch_id", ""))


def render_frame_paths(manifest: dict[str, Any], output_index: int) -> tuple[Path, Path, Path]:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    name = f"{output_index:06d}.jpg"
    return (
        require_file(require_dir(Path(str(outputs.get("overlay_frames", ""))), "overlay frame directory") / name, "overlay frame"),
        require_file(require_dir(Path(str(outputs.get("world_frames", ""))), "world frame directory") / name, "world frame"),
        require_file(require_dir(Path(str(outputs.get("side_world_frames", ""))), "side-world frame directory") / name, "side-world frame"),
    )


def validate_manifests(
    adapter: dict[str, Any], manifests: list[dict[str, Any]], manifest_paths: list[Path]
) -> dict[str, Any]:
    if adapter.get("status") != "ok":
        raise RuntimeError("adapter report is not ok")
    branches = [manifest_branch(manifest) for manifest in manifests]
    if branches != EXPECTED_ORDER:
        raise RuntimeError(f"manifest branch order must be {EXPECTED_ORDER}, got {branches}")
    for path, manifest in zip(manifest_paths, manifests):
        if manifest.get("status") != "ok" or manifest.get("schema") != "v19_experimental_p14_p15_full_mano_layered_render_v1":
            raise RuntimeError(f"not a completed layered render manifest: {path}")
        if int(manifest.get("frame_count", 0)) <= 0:
            raise RuntimeError(f"manifest has no rendered frames: {path}")
        shared = manifest.get("shared_state_consumption") if isinstance(manifest.get("shared_state_consumption"), dict) else {}
        if shared.get("generated_faces_collision_eligible") is not False or shared.get("signed_geometry_ready") is not False:
            raise RuntimeError(f"physical quarantine missing from render: {path}")
        mano = manifest.get("mano_full_surface") if isinstance(manifest.get("mano_full_surface"), dict) else {}
        if int(mano.get("vertices_per_hand", 0)) != 778 or int(mano.get("full_surface_rows_read", 0)) != 2 * int(manifest["frame_count"]):
            raise RuntimeError(f"full MANO surface coverage is incomplete: {path}")
        if float(mano.get("sample_reproduction_max_error_m") or 0.0) > 1.0e-5:
            raise RuntimeError(f"full MANO/sample reproduction mismatch: {path}")

    ids = [manifest.get("source_frame_ids") for manifest in manifests]
    if any(value != ids[0] for value in ids[1:]):
        raise RuntimeError("source frame IDs differ across branches")
    frame_rows = [manifest.get("frame_rows") for manifest in manifests]
    if any(not isinstance(rows, list) or len(rows) != len(ids[0]) for rows in frame_rows):
        raise RuntimeError("manifest frame-row coverage is incomplete")

    invariant_fields = {
        "annotations_sha256": [manifest["inputs"]["annotations_sha256"] for manifest in manifests],
        "shared_observed_physical_surface_sha256": [
            manifest["inputs"]["shared_observed_physical_surface_sha256"] for manifest in manifests
        ],
        "generated_face_budget": [manifest["render_contract"]["generated_face_budget"] for manifest in manifests],
        "observed_face_budget": [manifest["render_contract"]["observed_face_budget"] for manifest in manifests],
        "mano_face_budget": [manifest["render_contract"]["mano_face_budget"] for manifest in manifests],
        "observed_object_ownership_override": [
            manifest["render_contract"].get("observed_object_ownership_override") for manifest in manifests
        ],
        "mano_archive_contract": [value_sha256(manifest["mano_full_surface"]["archives"]) for manifest in manifests],
    }
    for field, values in invariant_fields.items():
        if len(set(json.dumps(value, sort_keys=True) for value in values)) != 1:
            raise RuntimeError(f"controlled render invariant differs across branches: {field} -> {values}")

    intrinsics_hashes: list[str] = []
    bounds_hashes: list[str] = []
    for branch_rows in frame_rows:
        intrinsics_hashes.append(value_sha256([row["intrinsics"] for row in branch_rows]))
        bounds_hashes.append(value_sha256([row["shared_world_bounds_m"] for row in branch_rows]))
    if len(set(intrinsics_hashes)) != 1:
        raise RuntimeError("per-frame camera intrinsics differ across branches")
    if len(set(bounds_hashes)) != 1:
        raise RuntimeError("per-frame shared world bounds differ across branches")

    state_blocks = adapter.get("shared_state_value_sha256")
    if not isinstance(state_blocks, dict) or not state_blocks:
        raise RuntimeError("adapter report lacks shared-state hashes")
    return {
        "branch_order": branches,
        "frame_count": len(ids[0]),
        "source_frame_ids": ids[0],
        "controlled_invariants": invariant_fields,
        "per_frame_intrinsics_value_sha256": intrinsics_hashes[0],
        "per_frame_world_bounds_value_sha256": bounds_hashes[0],
        "adapter_shared_state_value_sha256": state_blocks,
    }


def compose(args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = require_file(args.adapter_report, "P14/P15 adapter report")
    manifest_paths = [require_file(path, "branch render manifest") for path in args.manifests]
    if len(manifest_paths) != 3:
        raise RuntimeError("exactly three --manifests are required")
    adapter = load_json(adapter_path)
    manifests = [load_json(path) for path in manifest_paths]
    validation = validate_manifests(adapter, manifests, manifest_paths)
    output_dir = prepare_output(args.output_dir, bool(args.replace))
    matrix_frames = output_dir / "matrix_frames"
    overlay_frames = output_dir / "overlay_ab_frames"
    matrix_frames.mkdir(parents=True, exist_ok=True)
    overlay_frames.mkdir(parents=True, exist_ok=True)

    source_ids = [int(value) for value in validation["source_frame_ids"]]
    review_idx = int(args.review_source_frame)
    if review_idx not in source_ids:
        raise RuntimeError(f"review source frame {review_idx} is not in rendered source IDs")
    review_output_index = source_ids.index(review_idx)
    review_matrix: np.ndarray | None = None
    for output_index, source_idx in enumerate(source_ids):
        overlays: list[np.ndarray] = []
        worlds: list[np.ndarray] = []
        sides: list[np.ndarray] = []
        for manifest in manifests:
            overlay_path, world_path, side_path = render_frame_paths(manifest, output_index)
            overlays.append(resize_exact(read_image(overlay_path), 640, 640))
            worlds.append(resize_exact(read_image(world_path), 640, 360))
            sides.append(resize_exact(read_image(side_path), 640, 360))
        overlay_row = np.hstack(overlays)
        world_row = np.hstack(worlds)
        side_row = np.hstack(sides)
        matrix = np.vstack([overlay_row, world_row, side_row])
        matrix_path = matrix_frames / f"{output_index:06d}.jpg"
        overlay_path = overlay_frames / f"{output_index:06d}.jpg"
        if not cv2.imwrite(str(matrix_path), matrix, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {matrix_path}")
        if not cv2.imwrite(str(overlay_path), overlay_row, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {overlay_path}")
        if output_index == review_output_index:
            review_matrix = matrix
        if output_index % 50 == 0 or output_index + 1 == len(source_ids):
            print(f"composed {output_index + 1}/{len(source_ids)} source_frame={source_idx}", flush=True)
    if review_matrix is None:
        raise RuntimeError("review matrix was not produced")
    review_path = output_dir / f"frame_{review_idx:06d}_three_branch_overlay_world_side.png"
    if not cv2.imwrite(str(review_path), review_matrix):
        raise RuntimeError(f"failed to write {review_path}")

    raw_video = manifests[0].get("frame_rows", [{}])[0].get("intrinsics", {})
    del raw_video  # FPS is explicit in the renderer outputs; use caller override/default below.
    fps = float(args.fps)
    matrix_video = output_dir / "p15_three_branch_overlay_world_side_ab.mp4"
    overlay_video = output_dir / "p15_three_branch_camera_overlay_ab.mp4"
    canonical.encode_video(matrix_frames, matrix_video, fps, frame_count=len(source_ids))
    canonical.encode_video(overlay_frames, overlay_video, fps, frame_count=len(source_ids))

    role_summaries: dict[str, Any] = {}
    for manifest in manifests:
        branch = manifest_branch(manifest)
        per_role: dict[str, list[int]] = {}
        for row in manifest["frame_rows"]:
            for role, count in row.get("overlay_visible_pixels_by_role", {}).items():
                per_role.setdefault(str(role), []).append(int(count))
        role_summaries[branch] = {role: summary(values) for role, values in per_role.items()}

    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "compose_experimental_p15_three_branch_video_ab",
        "claim_scope": (
            "Controlled visual A/B only. All branches share observed-only object trajectory, camera, full source MANO, "
            "projection, render budgets, and observed collision surface. Geometry source/integration is the sole branch "
            "variable. The videos do not establish signed contact, collision, nonpenetration, or GT accuracy."
        ),
        "inputs": {
            "adapter_report": str(adapter_path),
            "adapter_report_sha256": sha256_file(adapter_path),
            "render_manifests": [
                {"path": str(path), "sha256": sha256_file(path), "branch": manifest_branch(manifest)}
                for path, manifest in zip(manifest_paths, manifests)
            ],
        },
        "validation": validation,
        "visible_pixel_summary_by_branch_and_role": role_summaries,
        "outputs": {
            "camera_overlay_ab_video": str(overlay_video),
            "overlay_world_side_ab_video": str(matrix_video),
            "review_frame": str(review_path),
            "matrix_frames": str(matrix_frames),
            "overlay_ab_frames": str(overlay_frames),
        },
        "visual_inspection_required": True,
    }
    report_path = output_dir / "p15_three_branch_video_ab_report.json"
    report["outputs"]["report"] = str(report_path)
    write_path = report_path
    write_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "frame_count": validation["frame_count"],
        "branch_order": validation["branch_order"],
        "outputs": report["outputs"],
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-source-frame", type=int, default=109)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    compose(parse_args())
