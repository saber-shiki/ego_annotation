#!/usr/bin/env python3
"""Build a demo-only pose initializer and topology-preserving visual mesh.

Ready P15 poses pass through unchanged. Explicitly unready P15 poses are not
trusted: eligible direct P14 observations are retained and only absent rows or
explicit empty missing-surface placeholders are interpolated on the real
annotation timeline. Core reports and meshes are read
only and no generated surface receives physical/contact authority.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from v19_runtime_contracts import load, sha
from v20_prediction_contracts import assert_prediction_only


ROTATION_KEY = "rotation_world_from_completed_canonical_matrix"
TRANSLATION_KEY = "translation_world_m"
DIRECT_STATUS = "fit_to_visible_depth_samples"
MISSING_STATUS = "missing_accepted_metric_surface"
GENERATED_EVIDENCE_KEYS = ("generated_geometry_pose_evidence_consumed",
                          "generated_geometry_pose_evidence")


def _no_positive_evidence(row: dict, keys: tuple[str, ...]) -> bool:
    # Fail closed for malformed flag values too (e.g. strings or numeric true).
    return all(row.get(key) is None or row.get(key) is False for key in keys)


def _missing_placeholder(row: dict) -> bool:
    return (row.get("status") == MISSING_STATUS
            and row.get(ROTATION_KEY) is None and row.get(TRANSLATION_KEY) is None
            and _no_positive_evidence(row, GENERATED_EVIDENCE_KEYS + (
                "rigid_pose_observation_eligible", "direct_visible_measurement",
                "direct_metric_observation_added", "observed", "annotation_ready"))
            and not any(row.get(key) for key in (
                "direct_pose_observation_source", "pose_measurement_status", "pose_source")))


def _object_identity(p15: dict, p14: dict, object_id: str | None) -> str | None:
    identities = [object_id, p15.get("object_id"), p14.get("object_id")]
    for report in (p15, p14):
        rows = _rows({"pose_rows": report.get("pose_rows", [])}, "pose source")
        identities.extend(row.get("object_id") for row in rows.values())
    present = [value for value in identities if value is not None]
    if any(not isinstance(value, str) or not value or value != present[0] for value in present):
        raise ValueError("pose source object mismatch")
    return present[0] if present else None


def readiness(p15: dict) -> str:
    annotation_ready = p15.get("annotation_ready")
    sufficient = (p15.get("graph_support") or {}).get("sufficient")
    if annotation_ready is False or sufficient is False:
        return "explicitly_unready"
    if annotation_ready is True and sufficient is not False:
        return "explicitly_ready"
    return "unknown"


def _timeline(annotation_frames: list[dict]) -> list[tuple[int, float]]:
    if not isinstance(annotation_frames, list) or not annotation_frames:
        raise ValueError("annotations need nonempty frames")
    values = []
    seen = set()
    for frame in annotation_frames:
        idx = int(frame["frame_idx"])
        time_s = float(frame["time_s"])
        if idx in seen or not np.isfinite(time_s):
            raise ValueError("duplicate frame ID or invalid annotation time")
        seen.add(idx)
        values.append((idx, time_s))
    values.sort(key=lambda row: row[1])
    if any(b[1] <= a[1] for a, b in zip(values, values[1:])):
        raise ValueError("annotation times must be strictly increasing")
    return values


def _pose(row: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(row.get(ROTATION_KEY), dtype=float)
    t = np.asarray(row.get(TRANSLATION_KEY), dtype=float)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
        raise ValueError(f"invalid rigid pose at frame {row.get('frame_idx')}")
    return R, t


def _rows(report: dict, label: str) -> dict[int, dict]:
    rows = report.get("pose_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} lacks pose_rows")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid {label} pose row")
        idx = int(row["frame_idx"])
        if idx in indexed:
            raise ValueError(f"duplicate {label} frame {idx}")
        indexed[idx] = row
    return indexed


def _eligible_direct(row: dict) -> bool:
    return (row.get("status") == DIRECT_STATUS
            and row.get("rigid_pose_observation_eligible") is True
            and _no_positive_evidence(row, GENERATED_EVIDENCE_KEYS))


def build_demo_pose_rows(p15: dict, p14: dict, annotation_frames: list[dict],
                         *, object_id: str | None = None) -> tuple[list[dict], dict]:
    timeline = _timeline(annotation_frames)
    state = readiness(p15)
    object_id = _object_identity(p15, p14 if state == "explicitly_unready" else {"pose_rows": []}, object_id)
    if state == "explicitly_ready":
        p15_rows = _rows(p15, "P15")
        if set(p15_rows) != {idx for idx, _ in timeline}:
            raise ValueError("ready P15 does not cover annotation timeline")
        for row in p15_rows.values():
            _pose(row)
        return copy.deepcopy(p15["pose_rows"]), {
            "mode": "ready_P15_pass_through",
            "p15_readiness": state,
            "direct_frame_ids": [],
            "interpolated_frame_ids": [],
            "unresolved_frame_ids": [],
            "core_pose_values_modified": False,
        }
    if state != "explicitly_unready":
        raise ValueError("P15 readiness is unknown; demo fallback is not authorized")

    if not _no_positive_evidence(p14, GENERATED_EVIDENCE_KEYS):
        raise ValueError("P14 generated geometry pose evidence is not an eligible direct source")
    p14_rows = _rows(p14, "P14")
    timeline_ids = {idx for idx, _ in timeline}
    unexpected = set(p14_rows) - timeline_ids
    if unexpected:
        raise ValueError(f"P14 rows outside annotation timeline: {sorted(unexpected)}")
    direct = {}
    for idx, row in p14_rows.items():
        if _missing_placeholder(row):
            continue
        if not _eligible_direct(row):
            # Only the producer's explicit empty missing-surface status is absent evidence.
            raise ValueError(f"existing P14 frame {idx} is not an eligible direct observation")
        _pose(row)
        direct[idx] = row
    if len(direct) < 2:
        raise ValueError("at least two eligible direct P14 observations required")

    time_by_id = dict(timeline)
    ordered_direct = sorted(direct, key=lambda idx: time_by_id[idx])
    result = []
    interpolated = []
    for idx, time_s in timeline:
        if idx in direct:
            row = copy.deepcopy(direct[idx])
            row["demo_visual_provenance"] = {
                "source": "immutable_direct_P14_observation",
                "direct_visible_measurement": True,
                "authority": "demo_only_visual_hypothesis",
            }
            result.append(row)
            continue
        before = [anchor for anchor in ordered_direct if time_by_id[anchor] < time_s]
        after = [anchor for anchor in ordered_direct if time_by_id[anchor] > time_s]
        if not before or not after:
            raise ValueError(f"frame {idx} is not bracketed by direct P14 observations")
        left, right = before[-1], after[0]
        t0, t1 = time_by_id[left], time_by_id[right]
        alpha = (time_s - t0) / (t1 - t0)
        R0, x0 = _pose(direct[left])
        R1, x1 = _pose(direct[right])
        rotation = Slerp([t0, t1], Rotation.from_matrix(np.stack([R0, R1])))([time_s]).as_matrix()[0]
        translation = (1.0 - alpha) * x0 + alpha * x1
        result.append({
            "frame_idx": idx,
            "time_s": time_s,
            **({"object_id": object_id} if object_id is not None else {}),
            "status": "demo_visual_interpolated_between_direct_P14",
            ROTATION_KEY: rotation.tolist(),
            TRANSLATION_KEY: translation.tolist(),
            "rigid_pose_observation_eligible": False,
            "pose_measurement_status": None,
            "direct_visible_measurement": False,
            "annotation_ready": False,
            "generated_geometry_pose_evidence_consumed": False,
            "demo_visual_provenance": {
                "source": "SO3_Slerp_and_linear_translation_between_direct_P14",
                "source_placeholder_row": copy.deepcopy(p14_rows.get(idx)),
                "latent_interpolated": True,
                "bracket_frame_ids": [left, right],
                "bracket_time_s": [t0, t1],
                "interpolation_alpha": alpha,
                "authority": "demo_only_visual_hypothesis",
                "uncertainty": "interpolation_not_observation_or_contact_authority",
            },
        })
        interpolated.append(idx)
    return result, {
        "mode": "explicitly_unready_P15_direct_P14_interpolation",
        "p15_readiness": state,
        "direct_frame_ids": ordered_direct,
        "interpolated_frame_ids": interpolated,
        "unresolved_frame_ids": [],
        "core_pose_values_modified": False,
        "annotation_ready": False,
        "authority": "demo_only_visual_hypothesis",
    }


def scale_mesh(source_mesh: Path, output_mesh: Path, scale_record: dict | None) -> dict:
    import trimesh

    if output_mesh.exists():
        raise ValueError("visual mesh output already exists")
    source_mesh = source_mesh.resolve()
    mesh = trimesh.load(source_mesh, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("source visual mesh must contain finite vertices and valid integer triangle topology")
    if scale_record is None:
        scalar, bounds, pivot, selection = 1.0, [0.8, 1.2], np.zeros(3), "identity_without_visual_scale_record"
    else:
        scalar = float(scale_record["uniform_scale"])
        bounds = [float(value) for value in scale_record.get("bounds", [0.8, 1.2])]
        pivot = np.asarray(scale_record.get("pivot_canonical_m", [0, 0, 0]), dtype=float)
        selection = scale_record.get("selection_source")
        if not isinstance(selection, str) or not selection:
            raise ValueError("visual scale selection_source required")
    if (len(bounds) != 2 or not np.isfinite([scalar, *bounds]).all() or bounds[0] <= 0
            or bounds[1] < bounds[0] or not bounds[0] <= scalar <= bounds[1]
            or pivot.shape != (3,) or not np.isfinite(pivot).all()):
        raise ValueError("invalid/out-of-bound uniform visual scale")
    scaled = (vertices - pivot) * scalar + pivot
    if not np.isfinite(scaled).all():
        raise ValueError("scaled visual mesh vertices are nonfinite")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=scaled, faces=faces.copy(), process=False).export(output_mesh)
    written = trimesh.load(output_mesh, process=False)
    written_vertices = np.asarray(written.vertices, dtype=float)
    written_faces = np.asarray(written.faces)
    serialization_tolerance_m = 1e-6
    if (len(written_vertices) != len(scaled) or not np.isfinite(written_vertices).all()
            or not np.allclose(written_vertices, scaled, atol=serialization_tolerance_m, rtol=0)):
        output_mesh.unlink(missing_ok=True)
        raise ValueError("serialized visual mesh vertices changed beyond PLY tolerance")
    if not np.array_equal(written_faces, faces):
        output_mesh.unlink(missing_ok=True)
        raise ValueError("visual mesh topology changed during export")
    face_digest = __import__("hashlib").sha256(np.ascontiguousarray(faces).tobytes()).hexdigest()
    return {
        "source_mesh_path": str(source_mesh),
        "source_mesh_sha256": sha(source_mesh),
        "output_mesh_path": str(output_mesh.resolve()),
        "output_mesh_sha256": sha(output_mesh),
        "uniform_scale": scalar,
        "pivot_canonical_m": pivot.tolist(),
        "bounds": bounds,
        "selection_source": selection,
        "source_vertex_count": int(len(vertices)),
        "output_vertex_count": int(len(written.vertices)),
        "source_face_count": int(len(faces)),
        "output_face_count": int(len(written_faces)),
        "face_index_digest": face_digest,
        "faces_exactly_preserved": True,
        "vertices_serialized_in_source_order": True,
        "serialized_vertex_tolerance_m": serialization_tolerance_m,
        "source_aabb_extent_m": np.ptp(vertices, axis=0).tolist(),
        "scaled_aabb_extent_m": np.ptp(scaled, axis=0).tolist(),
        "static_across_timeline": True,
        "demo_only": True,
        "demo_visual_proximity_fit_eligible": True,
        "metric_scale_authority": False,
        "physical_contact_authority_eligible": False,
        "generated_faces_contact_eligible": False,
        "generated_faces_collision_eligible": False,
        "generated_faces_sign_eligible": False,
        "generated_faces_sdf_eligible": False,
        "generated_faces_nonpenetration_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("p15-report", "p14-report", "annotations", "output-mesh"):
        parser.add_argument("--" + name, type=Path)
    for name in ("completion-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pose-report", type=Path,
                        help="P15 source for explicit legacy ready-only binding")
    parser.add_argument("--legacy-ready-p15-binding", action="store_true")
    parser.add_argument("--object-id")
    parser.add_argument("--appearance-mask-manifest", type=Path)
    parser.add_argument("--visual-scale-record", type=Path)
    parser.add_argument("--visual-completion-report", type=Path)
    parser.add_argument("--demo-launch-contract", type=Path)
    parser.add_argument("--published-output-mesh", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("demo binding already exists")
    if args.legacy_ready_p15_binding:
        if args.pose_report is None or any(value is not None for value in
                                           (args.p15_report, args.p14_report, args.annotations,
                                            args.output_mesh, args.appearance_mask_manifest)):
            raise ValueError("legacy ready binding requires only --pose-report, completion and output")
        source, completion = load(args.pose_report), load(args.completion_report)
        if readiness(source) != "explicitly_ready":
            raise ValueError("legacy binding accepts only explicitly ready P15")
        visual = copy.deepcopy(source)
        assert_prediction_only(source, label="legacy demo source", allow_annotation_ready=True,
                               allow_nondiagnostic_prediction=True)
        visual["render_mesh_contract"] = completion["render_mesh_contract"]
        visual["visual_mesh_receipt"] = {
            "output_mesh_path": completion["render_mesh_contract"]["required_mesh_path"],
            "output_mesh_sha256": completion["render_mesh_contract"]["required_mesh_sha256"],
            "demo_visual_proximity_fit_eligible": True,
            "physical_contact_authority_eligible": False,
            "generated_faces_contact_eligible": False,
            "generated_faces_collision_eligible": False,
            "generated_faces_sign_eligible": False,
            "generated_faces_sdf_eligible": False,
            "generated_faces_nonpenetration_eligible": False,
        }
        visual["core_source_identity"] = {
            "pose_report": str(args.pose_report.resolve()), "sha256": sha(args.pose_report),
            "source_readiness": source.get("annotation_ready"),
            "source_graph_support": source.get("graph_support"),
        }
        visual.update({
            "schema": "v19_additive_demo_legacy_pose_binding_v2",
            "annotation_ready": False,
            "diagnostic_only": True,
            "authority": "demo_only_visual_hypothesis",
            "physical_authority": False,
            "legacy_binding": True,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(visual, indent=2) + "\n")
        return
    required = {
        "p15_report": args.p15_report, "p14_report": args.p14_report,
        "annotations": args.annotations, "output_mesh": args.output_mesh,
        "object_id": args.object_id, "appearance_mask_manifest": args.appearance_mask_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or args.pose_report is not None:
        raise ValueError("visual-v2 binding arguments missing or mixed: " + ", ".join(missing))
    p15, p14, annotations, completion = (load(path) for path in (
        args.p15_report, args.p14_report, args.annotations, args.completion_report))
    assert_prediction_only(p15, label="demo P15 source", object_id=args.object_id,
                           allow_annotation_ready=True, allow_nondiagnostic_prediction=True)
    appearance = load(args.appearance_mask_manifest)
    if appearance.get("object_id") != args.object_id:
        raise ValueError("appearance manifest object mismatch")
    rows, summary = build_demo_pose_rows(p15, p14, annotations["frames"], object_id=args.object_id)
    source_mesh = Path(completion["render_mesh_contract"]["required_mesh_path"])
    if sha(source_mesh) != completion["render_mesh_contract"]["required_mesh_sha256"]:
        raise ValueError("core completion mesh identity mismatch")
    visual_completion = None
    if args.visual_completion_report:
        if args.demo_launch_contract is None:
            raise ValueError("visual completion requires exact demo launch contract")
        from v19_demo_reconditioning import validate_visual_completion, record
        contract = load(args.demo_launch_contract)
        for key, path in (("p14", args.p14_report), ("p15", args.p15_report),
                          ("annotations", args.annotations), ("completion", args.completion_report)):
            if contract["inputs"].get(key) != record(path):
                raise ValueError("demo binding contract core source mismatch")
        if contract.get("sam3d_reconditioning", {}).get("enabled") is not True:
            raise ValueError("visual completion requires enabled reconstruction")
        source_mesh, visual_completion = validate_visual_completion(
            args.visual_completion_report, contract, args.appearance_mask_manifest)
    elif args.demo_launch_contract and load(args.demo_launch_contract).get("sam3d_reconditioning", {}).get("enabled") is True:
        raise ValueError("enabled reconstruction missing visual completion; no core fallback")
    scale_record = load(args.visual_scale_record) if args.visual_scale_record else None
    if scale_record and scale_record.get("schema") == "v19_demo_image_uniform_scale_v1":
        if (scale_record.get("source_mesh") != {"path": str(source_mesh.resolve()), "sha256": sha(source_mesh)}
                or scale_record.get("appearance_manifest") != {"path": str(args.appearance_mask_manifest.resolve()), "sha256": sha(args.appearance_mask_manifest)}):
            raise ValueError("estimated scale source binding mismatch")
        for key, path in (("p14", args.p14_report), ("p15", args.p15_report),
                          ("annotations", args.annotations), ("completion", args.completion_report)):
            if scale_record.get("inputs", {}).get(key) != {"path": str(path.resolve()), "sha256": sha(path)}:
                raise ValueError("estimated scale initializer source mismatch")
    mesh_receipt = scale_mesh(source_mesh, args.output_mesh, scale_record)
    if scale_record:
        mesh_receipt["scale_alignment"] = scale_record
        mesh_receipt["scale_record"] = {"path": str(args.visual_scale_record.resolve()), "sha256": sha(args.visual_scale_record)}
    if visual_completion:
        mesh_receipt["visual_completion"] = {
            "path": str(args.visual_completion_report.resolve()), "sha256": sha(args.visual_completion_report)}
        mesh_receipt["original_canonical_gauge"] = visual_completion["original_canonical_gauge"]
    if args.published_output_mesh:
        mesh_receipt["output_mesh_path"] = str(args.published_output_mesh.resolve())
    visual = copy.deepcopy(p15)
    visual.update({
        "schema": "v19_additive_demo_visual_pose_binding_v2",
        "pose_rows": rows,
        "diagnostic_only": True,
        "annotation_ready": False,
        "authority": "demo_only_visual_hypothesis",
        "physical_authority": False,
        "render_mesh_contract": {
            "required_mesh_path": mesh_receipt["output_mesh_path"],
            "required_mesh_sha256": mesh_receipt["output_mesh_sha256"],
            "renderer_must_match": True,
        },
        "demo_pose_initializer": summary,
        "visual_mesh_receipt": mesh_receipt,
        "visual_completion_provenance": (None if visual_completion is None else {
            "path": str(args.visual_completion_report.resolve()), "sha256": sha(args.visual_completion_report),
            "schema": visual_completion["schema"], "source_model": "native_SAM3D_appearance_conditioned"}),
        "appearance_mask_manifest": {
            "path": str(args.appearance_mask_manifest.resolve()),
            "sha256": sha(args.appearance_mask_manifest),
        },
        "core_source_identity": {
            "p15_report": str(args.p15_report.resolve()), "p15_sha256": sha(args.p15_report),
            "p14_report": str(args.p14_report.resolve()), "p14_sha256": sha(args.p14_report),
            "completion_report": str(args.completion_report.resolve()),
            "completion_sha256": sha(args.completion_report),
            "source_readiness": p15.get("annotation_ready"),
            "source_graph_support": p15.get("graph_support"),
        },
        "core_sources_modified": False,
        "demo_visual_proximity_fit_eligible": True,
        "generated_faces_contact_eligible": False,
        "generated_faces_collision_eligible": False,
        "generated_faces_sign_eligible": False,
        "generated_faces_sdf_eligible": False,
        "generated_faces_nonpenetration_eligible": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(visual, indent=2) + "\n")


if __name__ == "__main__":
    main()
