"""Demo-only full-topology reconditioning contracts; no physical authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from build_v19_demo_pose_binding import build_demo_pose_rows, _pose
from build_v19_demo_visual_inputs import indexed_rows, metric_alignment, metric_object
from v19_demo_timeline import validate_transforms
from v19_runtime_contracts import load, sha
from v20_demo_geometry import resized_K

AUTHORITY = {
    "demo_only": True, "annotation_ready": False, "physical_authority": False,
    "authority": "demo_only_visual_hypothesis",
    "demo_visual_proximity_fit_eligible": True, "metric_scale_authority": False,
    "physical_contact_authority_eligible": False,
    **{f"generated_faces_{role}_eligible": False for role in
       ("contact", "collision", "sign", "sdf", "nonpenetration")},
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def bound_file(item, root):
    path = Path(item["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != item["sha256"]:
        raise ValueError(f"missing/outside/changed source: {path}")
    return path


def mesh_arrays(path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("one complete triangular mesh required")
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError("invalid complete vertices/topology")
    return v, f.astype(np.int64)


def write_mesh(path, vertices, faces):
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
    v, f = mesh_arrays(path)
    if not np.array_equal(f, faces) or v.shape != vertices.shape or not np.allclose(v, vertices, atol=1e-6, rtol=0):
        raise ValueError("serialized mesh changed topology/vertex order or precision")
    return {
        "vertex_count": len(v), "face_count": len(f),
        "face_index_sha256": hashlib.sha256(np.ascontiguousarray(f, dtype=np.int64).tobytes()).hexdigest(),
        "vertices_float64_sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
        "faces_exactly_preserved": True, "serialized_vertex_tolerance_m": 1e-6,
    }


def canonical_gauge(native_cv, scale, T, pose):
    T = validate_transforms([T])[0]
    R, t = _pose(pose)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid scene similarity scale")
    world = (native_cv * scale) @ T[:3, :3].T + T[:3, 3]
    canonical = (world - t) @ R
    replay = ((canonical @ R.T + t - T[:3, 3]) @ T[:3, :3]) / scale
    error = float(np.max(np.abs(replay - native_cv)))
    if error > 1e-5:
        raise ValueError("native/canonical gauge roundtrip failed")
    # Column-vector homogeneous similarity; vertices above use its transpose.
    transform = np.eye(4)
    transform[:3, :3] = scale * R.T @ T[:3, :3]
    transform[:3, 3] = R.T @ (T[:3, 3] - t)
    return canonical, {"T_original_canonical_from_native_opencv": transform.tolist(),
                       "T_world_camera_metric": T.tolist(), "initializer_pose": pose,
                       "initializer_pose_sha256": digest_json(pose),
                       "roundtrip_max_abs_native_scene_error": error,
                       "formula": "Vcanonical=((s*Vnative)@Rcamera.T+tcamera-tinitializer)@Rinitializer"}


def load_context(contract, manifest_path):
    from run_v19_additive_demo_branch import verified_contract_inputs, verify_live_run_identity
    root = Path(contract["run_root"]).resolve()
    from run_v19_additive_demo_branch import SAFE_ID
    if (contract.get("mode") != "additive_demo_visual_v2"
            or contract.get("authority") != "demo_only_visual_hypothesis"
            or contract.get("core_outputs_are_read_only") is not True
            or not all(isinstance(contract.get(k), str) and SAFE_ID.fullmatch(contract[k])
                       for k in ("object_id", "demo_id", "case_id"))):
        raise ValueError("invalid demo source authority/identity")
    verify_live_run_identity(contract, root)
    inputs = verified_contract_inputs(contract, root)
    expected = root / "measurements/demo_visual" / contract["object_id"] / contract["demo_id"] / "appearance_evidence/appearance_mask_manifest.json"
    if manifest_path.resolve() != expected.resolve() or not expected.resolve().is_relative_to(root / "measurements/demo_visual"):
        raise ValueError("unexpected appearance manifest namespace")
    manifest = load(manifest_path)
    if (manifest.get("object_id") != contract["object_id"]
            or manifest.get("source_annotations") != str(inputs["annotations"])
            or manifest.get("source_annotations_sha256") != sha(inputs["annotations"])
            or manifest.get("source_track_sha256") != sha(inputs["sam2_track"])
            or manifest.get("raw_frame_manifest_sha256") != sha(inputs["raw_frame_manifest"])):
        raise ValueError("appearance source binding mismatch")
    annotations = load(inputs["annotations"])
    rows, summary = build_demo_pose_rows(load(inputs["p15"]), load(inputs["p14"]),
                                        annotations["frames"], object_id=contract["object_id"])
    frames = indexed_rows(annotations["frames"], "annotations")
    appearances = indexed_rows(manifest["frames"], "appearance")
    if set(frames) != set(appearances):
        raise ValueError("appearance timeline mismatch")
    return inputs, frames, appearances, {int(r["frame_idx"]): r for r in rows}, summary


def appearance_evidence(row, frame, root, contract):
    idx = str(row["frame_idx"])
    rgb_item = {"path": row["raw_rgb_path"], "sha256": row["raw_rgb_sha256"]}
    if rgb_item != contract["raw_frame_inputs"].get(idx):
        raise ValueError("appearance RGB wrong-frame binding")
    source = {"path": row["source_mask_path"], "sha256": row["source_mask_sha256"]}
    if source != contract["sam2_mask_inputs"].get(idx):
        raise ValueError("appearance mask wrong-frame binding")
    bound_file(source, root)
    rgb = bound_file(rgb_item, root)
    mask_path = bound_file({"path": row["appearance_mask_path"], "sha256": row["appearance_mask_sha256"]}, root)
    if row["appearance_mask_sha256"] != source["sha256"]:
        raise ValueError("appearance mask is not byte-identical to SAM2")
    image, mask = cv2.imread(str(rgb)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if (image is None or mask is None or image.shape[:2] != mask.shape or not mask.any()
            or row.get("source_size_wh") != [mask.shape[1], mask.shape[0]]
            or int(row["frame_idx"]) != int(frame["frame_idx"])
            or not np.isclose(row["time_s"], frame["time_s"], atol=1e-6)):
        raise ValueError("malformed/wrong-frame appearance image/mask")
    T = validate_transforms([frame["camera"]["T_world_camera_metric"]])[0]
    K = resized_K(frame["camera"]["intrinsics_fx_fy_cx_cy"],
                  (frame["source_width"], frame["source_height"]), (mask.shape[1], mask.shape[0]))
    return rgb, mask_path, mask > 0, K, T


def anchor_evidence(contract, manifest_path, anchor_path):
    inputs, frames, appearance, poses, summary = load_context(contract, manifest_path)
    if anchor_path.resolve() != manifest_path.with_name("anchor_decision.json").resolve():
        raise ValueError("unexpected anchor report path")
    anchor = load(anchor_path)
    idx = int(anchor["selected_frame_idx"])
    if (anchor.get("object_id") != contract["object_id"] or idx not in frames
            or anchor.get("selection") != appearance[idx] or not appearance[idx].get("anchor_eligible")):
        raise ValueError("selected appearance anchor mismatch")
    if summary["p15_readiness"] != "explicitly_ready" and idx not in summary["direct_frame_ids"]:
        raise ValueError("reconditioning anchor requires direct P14, not latent interpolation")
    frame = frames[idx]
    obj = metric_object(frame, contract["object_id"])
    alignment = metric_alignment(obj, frame)
    if alignment is None or anchor.get("metric_alignment") != alignment:
        raise ValueError("missing/wrong-frame retained metric alignment")
    geom = obj["visible_geometry_candidate"]
    if int(geom.get("frame_idx", -1)) != idx or geom.get("object_id") != contract["object_id"]:
        raise ValueError("metric geometry wrong-frame/object")
    owner = geom.get("first_surface_depth_ownership", {})
    fraction = float(owner.get("retained_fraction", 0))
    if (owner.get("enabled") is not True or owner.get("fail_closed") is True
            or owner.get("failure_reasons") or not np.isfinite(fraction) or not .90 <= fraction <= 1):
        raise ValueError("robust retained metric support unavailable")
    rgb, mask_path, mask, K, T = appearance_evidence(appearance[idx], frame, Path(contract["run_root"]).resolve(), contract)
    expected_conditioning = {"raw_rgb": str(rgb), "raw_rgb_sha256": sha(rgb),
                            "raw_sam2_mask": str(mask_path), "raw_sam2_mask_sha256": sha(mask_path)}
    if anchor.get("appearance_conditioning") != expected_conditioning:
        raise ValueError("anchor conditioning binding mismatch")
    cam = np.asarray(alignment["camera_surfels_m"], float)
    world = np.asarray(alignment["world_surfels_m"], float)
    if (len(cam) < 100 or not np.isfinite(cam).all() or np.any(cam[:, 2] <= 0)
            or not np.allclose(cam @ T[:3, :3].T + T[:3, 3], world, atol=1e-5, rtol=0)):
        raise ValueError("invalid positive metric surfels/camera-world alignment")
    return inputs, rgb, mask_path, mask, K, T, cam, poses[idx], {
        "selected_frame_idx": idx, "pose_initializer": summary,
        "retained_support": owner, "metric_alignment_sha256": digest_json(alignment),
        "appearance_mask_role": "conditioning_and_projection_only_never_trusted_depth",
        "metric_depth_role": "same_frame_retained_metric_surfels_only",
    }


def scene_similarity(native_cv, observed_cam, K, mask):
    # Reuse the unchanged P13 front-depth numerical implementation, not its
    # core owned-mask producer/lineage or any face-authority contract.
    from run_v19_sam3d_reconstruction import module
    bridge = module(Path(__file__).resolve().parents[1] / "experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py", "demo_native_depth_numerics")
    projection = bridge.convex_projection_and_front_depth(native_cv, K, mask)
    if (projection["positive_depth_vertex_fraction"] < .99
            or projection["owned_zbuffer_pixel_count"] < 100
            or projection["convex_projection_iou"] < .05):
        raise ValueError("native positive/projected front-depth support insufficient")
    depth = float(np.median(observed_cam[:, 2]))
    scale = depth / projection["owned_zbuffer_median_native_depth"]
    if not np.isfinite(scale) or not .05 <= scale <= 5.:
        raise ValueError("scene similarity outside [0.05,5.0]")
    return scale, {"scene_similarity_scale": scale, "bounds": [.05, 5.],
                   "metric_depth_median_m": depth,
                   "metric_depth_p05_p95_m": np.percentile(observed_cam[:, 2], [5, 95]).tolist(),
                   "metric_surfel_count": len(observed_cam), "native_front_projection": projection,
                   "uncertainty": "single_view_native_front_depth_and_retained_monocular_metric_surfel_uncertainty"}


def validate_visual_completion(path, contract, manifest_path):
    """Replay typed visual binding from published hashes, never raw QC paths."""
    root = Path(contract["run_root"]).resolve()
    expected = root / "measurements/demo_visual" / contract["object_id"] / contract["demo_id"] / "visual_sam3d/visual_completion.json"
    if path.resolve() != expected or expected.resolve() != expected:
        raise ValueError("unexpected visual completion namespace")
    value = load(path)
    if value.get("schema") != "v19_demo_native_visual_completion_v1" or any(value.get(k) != v for k, v in AUTHORITY.items()):
        raise ValueError("invalid demo visual completion authority/schema")
    if value.get("object_id") != contract["object_id"] or value.get("case_id") != contract["case_id"]:
        raise ValueError("visual completion object/case mismatch")
    launch = root / "state/demo_visual" / contract["object_id"] / contract["demo_id"] / "receipts/launch_contract.json"
    if value.get("launch_contract") != record(launch) or value.get("runtime_config") != record(Path(contract["runtime_config"])):
        raise ValueError("visual completion launch/config binding mismatch")
    for key in contract["inputs"]:
        if value.get("inputs", {}).get(key) != contract["inputs"][key]:
            raise ValueError("visual completion core source identity mismatch")
    if value.get("appearance_manifest") != record(manifest_path):
        raise ValueError("visual completion appearance identity mismatch")
    anchor_path = bound_file(value["anchor_report"], root)
    _, rgb, mask_path, mask, K, T, surfels, pose, _ = anchor_evidence(contract, manifest_path, anchor_path)
    if value.get("conditioning") != {"rgb": record(rgb), "mask": record(mask_path)}:
        raise ValueError("visual completion conditioning mismatch")
    outputs = {k: bound_file(v, path.parent) for k, v in value["native_outputs"].items()}
    bound_file(value["native_execution"], path.parent)
    qc = load(bound_file(value["native_qc"], path.parent))
    raw, faces = mesh_arrays(outputs["mesh"])
    from remote_run_sam3d_objects_mesh_v7 import apply_native_pose
    p = qc["pose"]
    p3d, cv, _ = apply_native_pose(raw, np.asarray(p["rotation"]), np.asarray(p["translation"]), np.asarray(p["scale"]))
    for key, expected_vertices in (("native_pose_mesh_opencv_camera", cv), ("native_pose_mesh_pytorch3d_camera", p3d)):
        v, f = mesh_arrays(outputs[key])
        if v.shape != expected_vertices.shape or not np.array_equal(f, faces) or not np.allclose(v, expected_vertices, atol=1e-5, rtol=0):
            raise ValueError("published native transform/topology replay failed")
    scale, _ = scene_similarity(cv, surfels, K, mask)
    expected_vertices, gauge = canonical_gauge(cv, scale, T, pose)
    if value["original_canonical_gauge"] != gauge or value["scene_depth_similarity"]["scene_similarity_scale"] != scale:
        raise ValueError("visual completion gauge binding changed")
    mesh = value["render_mesh_contract"]
    mesh_path = bound_file({"path": mesh["required_mesh_path"], "sha256": mesh["required_mesh_sha256"]}, path.parent)
    v, f = mesh_arrays(mesh_path)
    if v.shape != expected_vertices.shape or not np.array_equal(f, faces) or not np.allclose(v, expected_vertices, atol=1e-6, rtol=0):
        raise ValueError("published complete canonical mesh replay failed")
    topology = value["topology"]
    if (topology["vertex_count"] != len(v) or topology["face_count"] != len(f)
            or topology["face_index_sha256"] != hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()
            or topology["vertices_float64_sha256"] != hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()):
        raise ValueError("published canonical topology/vertex digest mismatch")
    return mesh_path, value
