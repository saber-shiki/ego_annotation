#!/usr/bin/env python3
"""Execute native SAM3D for an explicit demo appearance anchor, never core P11."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from v19_runtime_assets import verify_bundle, verify_external_assets, runtime_asset_subset
from v19_runtime_contracts import load, runtime_environment
from run_v19_additive_demo_branch import (ROOT, gpu_probe, python, verify_execution_bundle,
                                         verify_hashes, expected_contract_paths)
from v19_demo_reconditioning import (AUTHORITY, record, anchor_evidence, mesh_arrays,
                                     canonical_gauge, scene_similarity, write_mesh)


def native_command(config, name, image, mask, seed, output):
    if any("|" in str(value) for value in (name, image, mask)):
        raise ValueError("native case fields may not contain pipe delimiters")
    return list(map(str, [python(config["sam3d_python"]), ROOT / "scripts/remote_run_sam3d_objects_mesh_v7.py",
        "--repo", config["sam3d_repo"], "--config", config["sam3d_config"],
        "--moge-checkpoint", config["sam3d_moge_checkpoint"],
        "--dinov2-repo", config["sam3d_dinov2_repo"], "--dinov2-checkpoint", config["sam3d_dinov2_checkpoint"],
        "--case", f"{name}|{image}|{mask}|{seed}", "--output-dir", output]))


def verify_model_assets(config):
    manifest = verify_bundle(ROOT, asset_role="geometry")
    subset = runtime_asset_subset(manifest["external_assets"], config, "geometry")
    # Native loaders recheck these checkpoints too. Here the explicit optional
    # stage additionally hashes them before launch rather than target-only.
    verify_external_assets({"files": subset["resolved_targets"], "trees": {}})
    return {**subset["files"], **subset["resolved_targets"]}


def reconstruct(contract_path, manifest_path, anchor_path, output, published):
    contract = load(contract_path)
    root = Path(contract["run_root"]).resolve()
    from run_v19_additive_demo_branch import SAFE_ID
    if not all(isinstance(contract.get(k), str) and SAFE_ID.fullmatch(contract[k]) for k in ("object_id", "demo_id", "case_id")):
        raise ValueError("unsafe demo identity")
    namespace = root / "measurements/demo_visual" / contract["object_id"] / contract["demo_id"]
    launch = root / "state/demo_visual" / contract["object_id"] / contract["demo_id"] / "receipts/launch_contract.json"
    if contract_path.resolve() != launch or launch.resolve() != launch:
        raise ValueError("unexpected native launch contract namespace")
    if (output.resolve() != namespace / ".D00b_visual_sam3d.staging"
            or published.resolve() != namespace / "visual_sam3d" or output.exists() or published.exists()
            or namespace.resolve() != namespace):
        raise ValueError("new confined reconditioning staging/publication required")
    mode = contract.get("sam3d_reconditioning", {})
    if mode.get("enabled") is not True or mode.get("mode") != "native_demo_appearance_sam3d" or mode.get("core_completion_fallback") is not False:
        raise ValueError("explicit native reconditioning mode required; no core fallback")
    seed = mode.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("configured integer SAM3D seed required")
    _, bundle_expected = verify_execution_bundle(contract)
    inputs, rgb, mask_path, mask, K, T, surfels, pose, evidence = anchor_evidence(contract, manifest_path, anchor_path)
    config = load(contract["runtime_config"])
    import socket
    if socket.gethostname() != contract["compute_host"]:
        raise ValueError("demo native compute host mismatch")
    assets = verify_model_assets(config)
    expected = expected_contract_paths(contract, root)
    for path in (contract_path, manifest_path, anchor_path, rgb, mask_path):
        expected[path.resolve()] = record(path)["sha256"]
    probe = gpu_probe(int(contract["physical_gpu"]), int(contract["minimum_free_mib"]))
    output.mkdir(parents=True)
    name = f"{contract['case_id']}_{contract['object_id']}_frame_{evidence['selected_frame_idx']}"
    argv = native_command(config, name, rgb, mask_path, seed, output / "native")
    overlay = config.get("sam3d_environment", {})
    if not isinstance(overlay, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in overlay.items()):
        raise ValueError("sam3d_environment must contain strings")
    env = {**runtime_environment(config, contract["physical_gpu"]), **overlay,
           "CUDA_VISIBLE_DEVICES": str(contract["physical_gpu"]),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    execution = {"argv": argv, "gpu_preflight": probe, "asset_role": "geometry",
                 "code_sha256": {str(path): digest for path, digest in bundle_expected.items()},
                 "model_assets": assets, "seed": seed, "physical_gpu": contract["physical_gpu"],
                 "returncode": None, **AUTHORITY}
    execution_path = output / "native_execution.json"
    execution_path.write_text(json.dumps(execution, indent=2) + "\n")
    verify_hashes(expected, "demo native input before launch")
    verify_hashes(bundle_expected, "demo native code before launch")
    with (output / "native.log").open("x") as log:
        done = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    execution["returncode"] = done.returncode
    execution_path.write_text(json.dumps(execution, indent=2) + "\n")
    if done.returncode:
        raise RuntimeError(f"native demo SAM3D failed: {done.returncode}; no fallback")
    verify_hashes(expected, "demo native input after launch")
    verify_hashes(bundle_expected, "demo native code after launch")
    verify_model_assets(config)
    qc_path = output / "native" / name / "qc_sam3d_objects_mesh_v7.json"
    qc = load(qc_path)
    if (qc.get("status") != "ok" or qc.get("name") != name or qc.get("image") != str(rgb)
            or qc.get("mask") != str(mask_path) or qc.get("seed") != seed
            or qc.get("native_pose_contract", {}).get("quaternion_order") != "wxyz_scalar_first_pytorch3d"
            or qc.get("native_pose_contract", {}).get("metric_status") != "native_monocular_scene_units_not_sensor_meters"):
        raise ValueError("native QC conditioning/pose identity mismatch")
    raw_paths = {}
    for key in ("mesh", "native_pose_mesh_opencv_camera", "native_pose_mesh_pytorch3d_camera", "glb"):
        path = Path(qc[key]).resolve()
        if not path.is_relative_to((output / "native" / name).resolve()) or not path.is_file():
            raise ValueError("native output outside own case namespace")
        raw_paths[key] = path
    raw, faces = mesh_arrays(raw_paths["mesh"])
    from remote_run_sam3d_objects_mesh_v7 import apply_native_pose
    p = qc["pose"]
    p3d, cv, _ = apply_native_pose(raw, np.asarray(p["rotation"]), np.asarray(p["translation"]), np.asarray(p["scale"]))
    replay_errors = {}
    for key, expected_vertices in (("native_pose_mesh_opencv_camera", cv), ("native_pose_mesh_pytorch3d_camera", p3d)):
        v, f = mesh_arrays(raw_paths[key])
        if v.shape != expected_vertices.shape or not np.array_equal(f, faces):
            raise ValueError("native mesh topology changed")
        error = float(np.max(np.abs(v - expected_vertices)))
        if error > 1e-5:
            raise ValueError("native pose transform replay failed")
        replay_errors[key] = error
    scale, depth_receipt = scene_similarity(cv, surfels, K, mask)
    canonical, gauge = canonical_gauge(cv, scale, T, pose)
    mesh_path = output / "original_canonical_mesh.ply"
    topology = write_mesh(mesh_path, canonical, faces)
    def published_record(path):
        value = record(path)
        value["path"] = str(published / path.relative_to(output))
        return value
    receipt = {
        "schema": "v19_demo_native_visual_completion_v1", **AUTHORITY,
        "source_model": "sam3d_objects",
        "launch_contract": record(contract_path), "runtime_config": record(Path(contract["runtime_config"])),
        "object_id": contract["object_id"], "case_id": contract["case_id"],
        "render_mesh_contract": {"required_mesh_path": str(published / mesh_path.name),
                                 "required_mesh_sha256": record(mesh_path)["sha256"], "renderer_must_match": True},
        "inputs": {k: record(v) for k, v in inputs.items()},
        "appearance_manifest": record(manifest_path), "anchor_report": record(anchor_path),
        "conditioning": {"rgb": record(rgb), "mask": record(mask_path)},
        "native_outputs": {k: published_record(v) for k, v in raw_paths.items()},
        "native_qc": published_record(qc_path), "native_execution": published_record(execution_path),
        "native_qc_staging_paths_are_provenance_only": True,
        "native_pose_replay_max_abs_errors": replay_errors,
        "scene_depth_similarity": depth_receipt, "original_canonical_gauge": gauge,
        "intrinsics_appearance_fx_fy_cx_cy": K.tolist(), "topology": topology,
        "anchor_evidence": evidence, "core_completion_used_only_as_provenance": True,
        "core_sources_modified": False,
    }
    (output / "visual_completion.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "appearance-mask-manifest", "anchor-report", "output-dir", "published-output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    reconstruct(a.contract, a.appearance_mask_manifest, a.anchor_report, a.output_dir, a.published_output_dir)


if __name__ == "__main__":
    main()
