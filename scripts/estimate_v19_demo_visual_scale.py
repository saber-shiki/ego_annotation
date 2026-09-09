#!/usr/bin/env python3
"""Estimate one bounded uniform demo scale from timeline triangle silhouettes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_v19_demo_pose_binding import _pose
from v19_demo_reconditioning import (AUTHORITY, load_context, appearance_evidence, mesh_arrays,
                                     record, bound_file, digest_json)
from v19_runtime_contracts import load
from v20_demo_geometry import resized_K


def silhouette(vertices, faces, K, shape):
    if np.any(vertices[:, 2] <= 1e-5) or not np.isfinite(vertices).all():
        raise ValueError("scale hypothesis crosses camera near plane")
    uv = vertices[:, :2] / vertices[:, 2, None] * K[:2] + K[2:]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e7:
        raise ValueError("unsafe projection")
    triangles = np.rint(uv[faces]).astype(np.int32)
    result = np.zeros(shape, np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(result, triangle, 1)
    return result.astype(bool)


def estimate_scale(vertices, faces, observations, bounds=(.8, 1.2), samples=41):
    """Mean triangle-silhouette 1-IoU, fixed poses/K and vertex-mean pivot.

    Observations bind an image mask, K, world-from-camera T and initializer R/t.
    Every triangle is retained; the low-resolution raster is an image objective,
    not reconstructed geometry or contact evidence.
    """
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]
            or type(samples) is not int or not 3 <= samples <= 101):
        raise ValueError("unsafe uniform scale bounds/grid")
    if len(observations) < 3:
        raise ValueError("at least three valid appearance frames required for scale alignment")
    pivot = vertices.mean(axis=0)
    grid = np.unique(np.r_[np.linspace(*bounds, samples), 1.])
    def objective(scale):
        scaled = (vertices - pivot) * scale + pivot
        errors = []
        for obs in observations:
            world = scaled @ obs["R"].T + obs["t"]
            T = obs["T"]
            camera = (world - T[:3, 3]) @ T[:3, :3]
            pred = silhouette(camera, faces, obs["K"], obs["mask"].shape)
            target = obs["mask"]
            union = np.count_nonzero(pred | target)
            errors.append(1 - np.count_nonzero(pred & target) / max(1, union))
        return float(np.mean(errors))
    values = [objective(float(value)) for value in grid]
    initial = objective(1.)
    if np.ptp(values) < 1e-8:
        raise ValueError("scale image objective is uninformative; no aligned hypothesis produced")
    # Deterministic ties prefer the least intervention, then the smaller scale.
    best = min(range(len(grid)), key=lambda i: (values[i], abs(grid[i] - 1), grid[i]))
    scale = float(grid[best])
    return {
        "schema": "v19_demo_image_uniform_scale_v1", **AUTHORITY,
        "uniform_scale": scale, "bounds": bounds.tolist(), "pivot_canonical_m": pivot.tolist(),
        "selection_source": "deterministic_multiframe_full_triangle_silhouette_grid",
        "static_across_timeline": True, "initial_objective": initial, "result_objective": values[best],
        "objective": "mean_1_minus_IoU_full_triangle_silhouette_at_declared_raster",
        "boundary_hit": bool(scale in bounds), "grid_scales": grid.tolist(), "grid_objectives": values,
        "uncertainty": "image_only_occlusion_mask_camera_pose_and_finite_raster_grid_uncertainty",
        "boundary_policy": "report_no_expansion_no_retry", "sample_count": len(observations),
    }


def align(contract_path, manifest_path, output, visual_completion=None):
    contract = load(contract_path)
    root = Path(contract["run_root"]).resolve()
    namespace = root / "measurements/demo_visual" / contract["object_id"] / contract["demo_id"]
    if output.resolve() != namespace / ".D00c_visual_scale.staging" or output.exists() or namespace.resolve() != namespace:
        raise ValueError("new confined scale output required")
    mode = contract.get("visual_scale_alignment", {})
    if mode.get("mode") != "image_uniform_grid" or mode.get("enabled") is not True:
        raise ValueError("explicit image scale mode required")
    inputs, frames, appearances, poses, summary = load_context(contract, manifest_path)
    completion_path = inputs["completion"]
    if contract["sam3d_reconditioning"]["enabled"]:
        expected = namespace / "visual_sam3d/visual_completion.json"
        if visual_completion is None or visual_completion.resolve() != expected:
            raise ValueError("enabled reconditioning requires visual completion; no core fallback")
        completion_path = visual_completion
        from v19_demo_reconditioning import validate_visual_completion
        validate_visual_completion(completion_path, contract, manifest_path)
    elif visual_completion is not None:
        raise ValueError("unexpected visual completion for disabled reconditioning")
    completion = load(completion_path)
    mesh = completion["render_mesh_contract"]
    mesh_path = bound_file({"path": mesh["required_mesh_path"], "sha256": mesh["required_mesh_sha256"]}, root)
    vertices, faces = mesh_arrays(mesh_path)
    candidates, excluded = [], []
    for idx in sorted(frames, key=lambda key: frames[key]["time_s"]):
        row = appearances[idx]
        if row.get("status") != "available" or row.get("eligible") is not True:
            excluded.append({"frame_idx": idx, "reason": "appearance_quality_unavailable"})
            continue
        if summary["p15_readiness"] != "explicitly_ready" and idx not in summary["direct_frame_ids"]:
            excluded.append({"frame_idx": idx, "reason": "latent_initializer_not_scale_measurement"})
            continue
        candidates.append(idx)
    count = mode["max_frames"]
    width = mode["raster_width"]
    if type(count) is not int or not 3 <= count <= 15 or type(width) is not int or not 32 <= width <= 512:
        raise ValueError("unsafe image scale sampling budget")
    chosen = [candidates[i] for i in np.unique(np.linspace(0, len(candidates) - 1, min(len(candidates), count), dtype=int))] if candidates else []
    observations, evidence = [], []
    for idx in chosen:
        rgb, mask_path, mask, K, T = appearance_evidence(appearances[idx], frames[idx], root, contract)
        size = [width, max(1, round(width * mask.shape[0] / mask.shape[1]))]
        Kr = resized_K(K, [mask.shape[1], mask.shape[0]], size)
        target = cv2.resize(mask.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
        if not target.any():
            excluded.append({"frame_idx": idx, "reason": "empty_at_objective_raster"})
            continue
        R, t = _pose(poses[idx])
        observations.append({"mask": target, "K": Kr, "T": T, "R": R, "t": t})
        evidence.append({"frame_idx": idx, "time_s": frames[idx]["time_s"], "rgb": record(rgb),
                         "appearance_mask": record(mask_path), "raster_size_wh": size,
                         "intrinsics_raster": Kr.tolist(), "T_world_camera": T.tolist(),
                         "initializer_pose": poses[idx], "initializer_pose_sha256": digest_json(poses[idx])})
    output.mkdir(parents=True)
    report = estimate_scale(vertices, faces, observations, mode["bounds"], mode["samples"])
    report.update(source_mesh=record(mesh_path), source_completion=record(completion_path),
                  source_pose_initializer=summary, evidence_frames=evidence, excluded_frames=excluded,
                  appearance_manifest=record(manifest_path),
                  inputs={key: record(inputs[key]) for key in ("annotations", "p14", "p15", "completion")})
    (output / "visual_scale_record.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "appearance-mask-manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--visual-completion", type=Path)
    a = p.parse_args()
    align(a.contract, a.appearance_mask_manifest, a.output_dir, a.visual_completion)


if __name__ == "__main__":
    main()
