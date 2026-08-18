#!/usr/bin/env python3
"""Finalize the read-only HOT3D upstream mask/depth/pose trace.

This finalizer does not rerun prediction and does not modify finalized case
roots, the five-case collection, or the prior P14/P15 visual A/B.  It validates
mechanical evidence, explicitly authored image-read review, source hashes, and
inherited P14/P15 direct-pose identity before writing final reports inside the
new repair-validation root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CASE_ORDER = ("milk", "soup", "mug", "bbq", "spatula")
SCHEMA = "hot3d_upstream_mask_depth_pose_final_review_v1"
FINAL_STATUS = "complete_with_upstream_P03_P09_depth_surface_conflicts_and_case_specific_limits"
MECHANICAL_REPORT_NAME = "UPSTREAM_MASK_DEPTH_POSE_TRACE_REPORT.json"
MECHANICAL_DONE_NAME = "UPSTREAM_TRACE_MECHANICAL_DONE.json"
MANUAL_REVIEW_RELATIVE = Path(
    "manual_visual_review/MANUAL_UPSTREAM_IMAGE_READ_REVIEW.json"
)
NUMERIC_RELATIVE = Path("numeric/UPSTREAM_CAUSAL_AUDIT.json")
FINAL_REPORT_NAME = "FINAL_UPSTREAM_TRACE_REPORT.json"
FINAL_REPORT_ZH_NAME = "FINAL_UPSTREAM_TRACE_REPORT_ZH.md"
FINAL_INDEX_NAME = "FINAL_ARTIFACT_SHA256_INDEX.json"
FINAL_DONE_NAME = "FINAL_UPSTREAM_TRACE_DONE.json"
FINALIZE_COMMAND_RELATIVE = Path("provenance/FINALIZE_COMMAND.sh")
EXPECTED_IMAGE_COUNTS = {
    "depth_component_neighborhood_sheets": 10,
    "full_sequence_timeline_sheets": 5,
    "strict_rgb_pnp_depth_error_overlays": 17,
}
NPZ_IDENTITY_MEMBERS = (
    "frame_idx.npy",
    "depth.npy",
    "confidence.npy",
    "source_size.npy",
    "intrinsics_fx_fy_cx_cy.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--diagnostic-code-commit", required=True)
    parser.add_argument("--replace-final-reports", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path | str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"missing {description}: {resolved}")
    return resolved


def run_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def validate_committed_clean_code(
    script_path: Path, expected_commit: str
) -> dict[str, Any]:
    git_root = Path(
        run_text(["git", "-C", str(script_path.parent), "rev-parse", "--show-toplevel"])
    ).resolve(strict=True)
    head = run_text(["git", "-C", str(git_root), "rev-parse", "HEAD"])
    if head != expected_commit:
        raise RuntimeError(
            f"finalizer commit mismatch: expected {expected_commit}, current {head}"
        )
    status = run_text(["git", "-C", str(git_root), "status", "--porcelain"])
    if status:
        raise RuntimeError(f"finalizer requires a clean worktree:\n{status}")
    return {
        "git_root": str(git_root),
        "commit": head,
        "worktree_clean": True,
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
    }


def validate_ancestor(git_root: Path, ancestor: str, descendant: str) -> None:
    subprocess.run(
        ["git", "-C", str(git_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def strict_pnp_pass(probe: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    if probe.get("status") != "pnp_solution_available":
        return False
    pnp = probe.get("pnp")
    if not isinstance(pnp, dict):
        return False
    reprojection = pnp["reprojection_error_px"]
    return bool(
        int(pnp["target_track_count"]) >= int(thresholds["minimum_tracked_points"])
        and int(pnp["pnp_inlier_count"]) >= int(thresholds["minimum_pnp_inliers"])
        and float(pnp["pnp_inlier_fraction"])
        >= float(thresholds["minimum_pnp_inlier_fraction"])
        and float(reprojection["median"])
        <= float(thresholds["maximum_reprojection_median_px"])
        and float(reprojection["p95"])
        <= float(thresholds["maximum_reprojection_p95_px"])
        and float(pnp["positive_camera_depth_fraction"])
        >= float(thresholds["minimum_positive_camera_depth_fraction"])
    )


def strict_gate_failures(
    probe: dict[str, Any], thresholds: dict[str, Any]
) -> list[str]:
    if probe.get("status") != "pnp_solution_available" or not isinstance(
        probe.get("pnp"), dict
    ):
        return ["pnp_solution_unavailable"]
    pnp = probe["pnp"]
    reprojection = pnp["reprojection_error_px"]
    failures: list[str] = []
    if int(pnp["target_track_count"]) < int(thresholds["minimum_tracked_points"]):
        failures.append("minimum_tracked_points")
    if int(pnp["pnp_inlier_count"]) < int(thresholds["minimum_pnp_inliers"]):
        failures.append("minimum_pnp_inliers")
    if float(pnp["pnp_inlier_fraction"]) < float(
        thresholds["minimum_pnp_inlier_fraction"]
    ):
        failures.append("minimum_pnp_inlier_fraction")
    if float(reprojection["median"]) > float(
        thresholds["maximum_reprojection_median_px"]
    ):
        failures.append("maximum_reprojection_median_px")
    if float(reprojection["p95"]) > float(
        thresholds["maximum_reprojection_p95_px"]
    ):
        failures.append("maximum_reprojection_p95_px")
    if float(pnp["positive_camera_depth_fraction"]) < float(
        thresholds["minimum_positive_camera_depth_fraction"]
    ):
        failures.append("minimum_positive_camera_depth_fraction")
    return failures


def classify_image(path: Path) -> str:
    name = path.name
    if "depth_component_sheets" in path.parts:
        return "depth_component_neighborhood_sheets"
    if name.endswith("_upstream_trace_timeline.jpg"):
        return "full_sequence_timeline_sheets"
    if name.endswith("_strict_rgb_pnp_depth_error.jpg"):
        return "strict_rgb_pnp_depth_error_overlays"
    raise RuntimeError(f"unrecognized reviewed-image category: {path}")


def validate_manual_review(
    trace_root: Path,
    manual_path: Path,
    mechanical_path: Path,
) -> dict[str, Any]:
    manual = load_json(manual_path)
    if manual.get("status") != "manual_image_read_complete" or not bool(
        manual.get("image_read_review_complete")
    ):
        raise RuntimeError("manual image-read review is not complete")
    if Path(manual["mechanical_report"]).resolve(strict=True) != mechanical_path:
        raise RuntimeError("manual review does not bind the current mechanical report")
    if manual["mechanical_report_sha256"] != sha256_file(mechanical_path):
        raise RuntimeError("manual review mechanical-report hash mismatch")
    rows = manual.get("reviewed_images")
    if not isinstance(rows, list) or len(rows) != 32:
        raise RuntimeError("manual review must bind exactly 32 images")
    paths: list[Path] = []
    category_counts = {key: 0 for key in EXPECTED_IMAGE_COUNTS}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("image_read_opened")):
            raise RuntimeError(f"image was not marked opened: {row}")
        relative = Path(row["path"])
        path = (trace_root / relative).resolve(strict=True)
        if not path.is_relative_to(trace_root):
            raise RuntimeError(f"reviewed image escaped trace root: {path}")
        if path in paths:
            raise RuntimeError(f"duplicate reviewed image: {path}")
        paths.append(path)
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != row["sha256"] or actual_size != int(row["size_bytes"]):
            raise RuntimeError(f"reviewed image hash/size mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise RuntimeError(f"reviewed image cannot be decoded: {path}")
        category = classify_image(path)
        category_counts[category] += 1
        normalized_rows.append(
            {
                "path": str(relative),
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "decoded_shape": list(image.shape),
                "category": category,
                "image_read_opened": True,
            }
        )
    if category_counts != EXPECTED_IMAGE_COUNTS:
        raise RuntimeError(
            f"reviewed-image category mismatch: {category_counts} != {EXPECTED_IMAGE_COUNTS}"
        )
    normalized_rows.sort(key=lambda row: row["path"])
    set_blob = "".join(
        f"{row['path']}\t{row['sha256']}\t{row['size_bytes']}\n"
        for row in normalized_rows
    ).encode("utf-8")
    set_hash = hashlib.sha256(set_blob).hexdigest()
    if set_hash != manual["reviewed_image_set_sha256"]:
        raise RuntimeError("manual reviewed-image set hash mismatch")
    actual_paths = sorted(
        list(trace_root.glob("cases/*/depth_component_sheets/*.jpg"))
        + list(trace_root.glob("cases/*/*_upstream_trace_timeline.jpg"))
        + list(trace_root.glob("cases/*/rgb_pnp/*_strict_rgb_pnp_depth_error.jpg"))
    )
    if sorted(paths) != actual_paths:
        missing = sorted(str(path) for path in set(actual_paths) - set(paths))
        undeclared = sorted(str(path) for path in set(paths) - set(actual_paths))
        raise RuntimeError(
            f"manual reviewed-image set mismatch; missing={missing}, undeclared={undeclared}"
        )
    return {
        "status": "passed",
        "manual_review": str(manual_path),
        "manual_review_sha256": sha256_file(manual_path),
        "reviewed_image_count": len(normalized_rows),
        "reviewed_image_category_counts": category_counts,
        "reviewed_image_set_sha256": set_hash,
        "all_images_decoded": True,
        "rows": normalized_rows,
        "authored_verdict": manual,
    }


def validate_source_immutability(mechanical: dict[str, Any]) -> dict[str, Any]:
    source = mechanical.get("source_immutability")
    if not isinstance(source, dict) or int(source.get("mismatch_count", -1)) != 0:
        raise RuntimeError("mechanical source-immutability audit did not pass")
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != int(source["entry_count"]):
        raise RuntimeError("malformed mechanical source-immutability rows")
    validated: list[dict[str, Any]] = []
    for row in rows:
        path = require_file(row["path"], "audited source artifact")
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        expected_hash = row["sha256"]
        expected_size = int(row["size_bytes"])
        matched = actual_hash == expected_hash and actual_size == expected_size
        if not matched:
            raise RuntimeError(f"source artifact changed after mechanical audit: {path}")
        validated.append(
            {
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "size_bytes": actual_size,
                "matched": True,
            }
        )
    return {
        "status": "passed",
        "entry_count": len(validated),
        "mismatch_count": 0,
        "finalized_case_roots_modified": False,
        "collection_modified": False,
        "rows": validated,
    }


def validate_prior_stage_ab(stage_ab_root: Path) -> dict[str, Any]:
    done_path = require_file(stage_ab_root / "FINAL_REVIEW_DONE.json", "prior A/B sentinel")
    done = load_json(done_path)
    if bool(done.get("visual_review_pending")) or not bool(
        done.get("manual_image_read_review_complete")
    ):
        raise RuntimeError("prior P14/P15 A/B is not finalized")
    direct_path = require_file(
        stage_ab_root / "numeric/p14_vs_p15_exact_se3_delta.json",
        "prior P14/P15 exact-SE3 audit",
    )
    direct = load_json(direct_path)
    required = {
        "status": direct.get("status") == "passed",
        "case_count": int(direct.get("case_count", -1)) == 5,
        "direct_rows": int(direct.get("total_direct_row_count", -1)) == 679,
        "completed_rows": int(direct.get("total_completed_row_count", -1)) == 71,
        "direct_identity": bool(direct.get("all_direct_rows_exact_array_identity")),
        "P15_direct_smoothing_false": not bool(
            direct.get("physical_trajectory_smoothed_directly_by_P15")
        ),
    }
    if not all(required.values()):
        raise RuntimeError(f"prior P14/P15 identity contract failed: {required}")
    source_path = require_file(
        stage_ab_root / "numeric/source_immutability_audit.json",
        "prior A/B source immutability audit",
    )
    source = load_json(source_path)
    if source.get("status") != "passed" or int(source.get("mismatch_count", -1)) != 0:
        raise RuntimeError("prior A/B source audit did not pass")
    current_mismatch: list[str] = []
    for row in source.get("rows", []):
        path = require_file(row["path"], "prior A/B source artifact")
        if sha256_file(path) != row["expected_sha256"]:
            current_mismatch.append(str(path))
    if current_mismatch:
        raise RuntimeError(f"prior A/B sources changed: {current_mismatch}")
    return {
        "status": "passed",
        "prior_final_status": done["status"],
        "prior_done": str(done_path),
        "prior_done_sha256": sha256_file(done_path),
        "exact_se3_audit": str(direct_path),
        "exact_se3_audit_sha256": sha256_file(direct_path),
        "direct_row_count": 679,
        "completed_row_count": 71,
        "all_direct_rows_exact_array_identity": True,
        "P15_physical_trajectory_smoothed_directly": False,
        "prior_source_immutability_entry_count": int(source["entry_count"]),
        "prior_source_immutability_mismatch_count": 0,
    }


def validate_collection_links(
    collection_root: Path, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_case = {row["case"]: row for row in cases}
    for case_name in CASE_ORDER:
        link = collection_root / "runs" / case_name
        if not link.is_symlink():
            raise RuntimeError(f"collection run link is not a symlink: {link}")
        resolved = link.resolve(strict=True)
        expected = Path(by_case[case_name]["run_root"]).resolve(strict=True)
        if resolved != expected:
            raise RuntimeError(
                f"collection run link changed for {case_name}: {resolved} != {expected}"
            )
        rows.append(
            {
                "case": case_name,
                "link": str(link),
                "relative_link_target": os.readlink(link),
                "resolved_target": str(resolved),
                "matches_mechanical_run_root": True,
            }
        )
    return {
        "status": "passed",
        "case_count": len(rows),
        "all_links_relative": all(
            not Path(row["relative_link_target"]).is_absolute() for row in rows
        ),
        "rows": rows,
    }


def validate_p03_and_p11(mechanical: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in mechanical["cases"]:
        name = case["case"]
        identities = case["p03c_source_to_active_array_identity"]
        if not all(bool(value) for value in identities.values()):
            raise RuntimeError(f"{name}: P03/P03c array identity failed")
        source_path = require_file(case["inputs"]["p03_source_depth"], f"{name} P03 source")
        active_path = require_file(case["inputs"]["p03c_active_depth"], f"{name} P03c active")
        member_rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(source_path, "r") as source_archive, zipfile.ZipFile(
            active_path, "r"
        ) as active_archive:
            source_names = set(source_archive.namelist())
            active_names = set(active_archive.namelist())
            for member in NPZ_IDENTITY_MEMBERS:
                if member not in source_names or member not in active_names:
                    raise RuntimeError(f"{name}: missing NPZ member {member}")
                source_hash = sha256_zip_member(source_archive, member)
                active_hash = sha256_zip_member(active_archive, member)
                source_size = source_archive.getinfo(member).file_size
                active_size = active_archive.getinfo(member).file_size
                if source_hash != active_hash or source_size != active_size:
                    raise RuntimeError(f"{name}: P03/P03c member differs: {member}")
                member_rows.append(
                    {
                        "member": member,
                        "source_member_sha256": source_hash,
                        "active_member_sha256": active_hash,
                        "uncompressed_size_bytes": source_size,
                        "byte_identical_npy_member": True,
                    }
                )
        with np.load(source_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            metadata = {
                "camera_conditioning_mode": str(archive["camera_conditioning_mode"].item()),
                "depth_ray_geometry_reprojected": bool(
                    archive["depth_ray_geometry_reprojected"].item()
                ),
                "depth_output_quantity": str(archive["depth_output_quantity"].item()),
                "inference_camera_input_plane": str(
                    archive["inference_camera_input_plane"].item()
                ),
                "inference_camera_output_plane": str(
                    archive["inference_camera_output_plane"].item()
                ),
                "inference_camera_contract_sha256": str(
                    archive["inference_camera_contract_sha256"].item()
                ),
                "saved_preconversion_metric_radius_tensor": any(
                    "radius" in key.lower() for key in keys
                ),
            }
        if metadata["depth_output_quantity"] != (
            "camera_z_m_from_metric_radius_on_exact_contract_rays"
        ):
            raise RuntimeError(f"{name}: unexpected P03 output quantity")
        p11 = case["p11_binding"]
        exact_checks = p11["exact_transfer_checks"]
        anchor_checks = p11["canonical_anchor_checks"]
        if not all(bool(value) for value in exact_checks.values()) or not all(
            bool(value) for value in anchor_checks.values()
        ):
            raise RuntimeError(f"{name}: P11 exact binding failed")
        mode = p11["atomic_binding_mode"]
        if name == "milk":
            if mode != "legacy_schema_without_saved_selected_anchor_atomic_binding":
                raise RuntimeError("Milk must retain its legacy P11 schema limitation")
            if p11["saved_atomic_binding_checks"].get(
                "legacy_binding_replayed_or_upgraded"
            ):
                raise RuntimeError("Milk legacy P11 binding was improperly upgraded")
        elif mode != "modern_saved_selected_anchor_atomic_binding":
            raise RuntimeError(f"{name}: modern atomic P11 binding is absent")
        cases.append(
            {
                "case": name,
                "p03_source_depth": str(source_path),
                "p03c_active_depth": str(active_path),
                "array_identity": identities,
                "NPZ_member_identity": member_rows,
                "P03_metadata": metadata,
                "P11_selected_frame_idx": int(p11["selected_frame_idx"]),
                "P11_atomic_binding_mode": mode,
                "P11_exact_transfer_check_count": len(exact_checks),
                "P11_exact_transfer_checks_passed": True,
                "P11_canonical_anchor_check_count": len(anchor_checks),
                "P11_canonical_anchor_checks_passed": True,
                "P11_owned_mask_sha256": p11["owned_mask_sha256"],
            }
        )
    return {
        "status": "passed",
        "case_count": len(cases),
        "P03c_depth_confidence_frame_size_and_intrinsics_npy_members_byte_identical": True,
        "P11_exact_binding_passed_all_cases": True,
        "earliest_persisted_depth_quantity": (
            "P03 source archive camera_z_m_from_metric_radius_on_exact_contract_rays"
        ),
        "preconversion_metric_radius_tensor_available": False,
        "can_separate_UniDepth_radius_head_from_radius_to_ray_conversion": False,
        "cases": cases,
    }


def find_focus(
    mechanical: dict[str, Any], case_name: str, frame_idx: int
) -> dict[str, Any]:
    case = next(row for row in mechanical["cases"] if row["case"] == case_name)
    return next(
        row for row in case["focus_frames"] if int(row["frame_idx"]) == frame_idx
    )


def find_probe(focus: dict[str, Any], source_idx: int) -> dict[str, Any]:
    return next(
        row
        for row in focus["rgb_pnp_probes"]
        if int(row["source_frame_idx"]) == source_idx
    )


def accepted_components(focus: dict[str, Any]) -> list[dict[str, Any]]:
    target = next(
        row
        for row in focus["temporal_neighborhood"]
        if int(row["frame_idx"]) == int(focus["frame_idx"])
    )
    components: list[dict[str, Any]] = []
    for row in target.get("p09_components", []):
        retained = int(row.get("retained_pixels") or 0)
        summary = row.get("accepted_depth_summary_m")
        if retained > 0 and isinstance(summary, dict):
            components.append(
                {
                    "retained_pixels": retained,
                    "median_depth_m": float(summary["median"]),
                    "state": row.get("state"),
                }
            )
    return components


def validate_probes_and_stages(mechanical: dict[str, Any]) -> dict[str, Any]:
    thresholds = mechanical["strict_pnp_thresholds"]
    probe_rows: list[dict[str, Any]] = []
    modern_pairs: list[tuple[float, float, float]] = []
    p09_metric_rows = 0
    p09_neighborhood_rows = 0
    unique_neighborhood: set[tuple[str, int]] = set()
    for case in mechanical["cases"]:
        for focus in case["focus_frames"]:
            for row in focus["temporal_neighborhood"]:
                p09_neighborhood_rows += 1
                unique_neighborhood.add((case["case"], int(row["frame_idx"])))
                if bool(row.get("metric_surface_available")):
                    p09_metric_rows += 1
                    if not bool(row.get("p09_exact_accepted_raster_reconstructed")):
                        raise RuntimeError("P09 accepted raster did not reconstruct exactly")
            for probe in focus["rgb_pnp_probes"]:
                recomputed = strict_pnp_pass(probe, thresholds)
                recorded = bool(probe["production_strict_pass"])
                if recomputed != recorded:
                    raise RuntimeError(
                        f"strict PnP gate mismatch: {case['case']} "
                        f"{probe['source_frame_idx']}->{probe['target_frame_idx']}"
                    )
                row: dict[str, Any] = {
                    "case": case["case"],
                    "focus_frame_idx": int(focus["frame_idx"]),
                    "source_frame_idx": int(probe["source_frame_idx"]),
                    "target_frame_idx": int(probe["target_frame_idx"]),
                    "strict_pass": recorded,
                    "gate_failures": strict_gate_failures(probe, thresholds),
                }
                if recorded:
                    discrepancy = probe[
                        "target_p09_accepted_depth_minus_rgb_pnp_predicted_z_m"
                    ]
                    row.update(
                        {
                            "target_P09_depth_minus_RGB_PnP_predicted_z_median_m": float(
                                discrepancy["median"]
                            ),
                            "PnP_inlier_fraction": float(
                                probe["pnp"]["pnp_inlier_fraction"]
                            ),
                            "PnP_reprojection_median_px": float(
                                probe["pnp"]["reprojection_error_px"]["median"]
                            ),
                            "saved_stage_vs_RGB_PnP": probe[
                                "saved_stage_vs_rgb_pnp"
                            ],
                        }
                    )
                    stages = probe["saved_stage_vs_rgb_pnp"]
                    if "pairwise_chain" in stages:
                        if "pre_temporal_measurement" not in stages:
                            raise RuntimeError("modern pairwise comparison lacks measurement")
                        pairwise = float(
                            stages["pairwise_chain"][
                                "mapped_source_centroid_difference_m"
                            ]
                        )
                        measurement = float(
                            stages["pre_temporal_measurement"][
                                "mapped_source_centroid_difference_m"
                            ]
                        )
                        final = float(
                            stages["p14_final_regularized"][
                                "mapped_source_centroid_difference_m"
                            ]
                        )
                        modern_pairs.append((pairwise, measurement, final))
                probe_rows.append(row)
    if p09_neighborhood_rows != 50 or len(unique_neighborhood) != 50:
        raise RuntimeError("expected 50 unique focus-neighborhood rows")
    if p09_metric_rows != 45:
        raise RuntimeError("expected 45 reconstructed P09 metric rows")
    strict_rows = [row for row in probe_rows if row["strict_pass"]]
    failed_rows = [row for row in probe_rows if not row["strict_pass"]]
    if len(probe_rows) != 20 or len(strict_rows) != 17 or len(failed_rows) != 3:
        raise RuntimeError("unexpected strict PnP probe counts")
    expected_failures = {
        ("mug", 142, 143),
        ("mug", 144, 143),
        ("bbq", 60, 58),
    }
    actual_failures = {
        (row["case"], row["source_frame_idx"], row["target_frame_idx"])
        for row in failed_rows
    }
    if actual_failures != expected_failures:
        raise RuntimeError(f"unexpected strict PnP failures: {actual_failures}")
    if any(row["gate_failures"] != ["minimum_pnp_inlier_fraction"] for row in failed_rows):
        raise RuntimeError("all three expected PnP failures must fail inlier fraction")
    if len(modern_pairs) != 15:
        raise RuntimeError("expected 15 modern-schema stage comparisons")
    final_closer_pairwise = sum(final < pairwise for pairwise, _, final in modern_pairs)
    final_closer_measurement = sum(
        final < measurement for _, measurement, final in modern_pairs
    )
    if final_closer_pairwise != 14 or final_closer_measurement != 14:
        raise RuntimeError("unexpected P14 stage-vs-PnP comparison counts")

    focus_evidence: dict[str, Any] = {}
    for case_name, frame_idx, sources in (
        ("milk", 92, (91, 93)),
        ("soup", 87, (86, 88)),
        ("soup", 146, (145, 147)),
        ("mug", 33, (32, 34)),
        ("mug", 118, (117, 119)),
        ("mug", 143, (142, 144)),
        ("bbq", 58, (57, 60)),
        ("spatula", 10, (9, 11)),
        ("spatula", 37, (36, 38)),
        ("spatula", 117, (116, 118)),
    ):
        focus = find_focus(mechanical, case_name, frame_idx)
        target = next(
            row
            for row in focus["temporal_neighborhood"]
            if int(row["frame_idx"]) == frame_idx
        )
        probe_summary: list[dict[str, Any]] = []
        for source_idx in sources:
            probe = find_probe(focus, source_idx)
            compact: dict[str, Any] = {
                "source_frame_idx": source_idx,
                "target_frame_idx": frame_idx,
                "strict_pass": bool(probe["production_strict_pass"]),
                "gate_failures": strict_gate_failures(probe, thresholds),
            }
            if compact["strict_pass"]:
                compact.update(
                    {
                        "target_P09_depth_minus_RGB_PnP_predicted_z_median_m": float(
                            probe[
                                "target_p09_accepted_depth_minus_rgb_pnp_predicted_z_m"
                            ]["median"]
                        ),
                        "mapped_centroid_difference_m": {
                            key: float(value["mapped_source_centroid_difference_m"])
                            for key, value in probe[
                                "saved_stage_vs_rgb_pnp"
                            ].items()
                        },
                    }
                )
            probe_summary.append(compact)
        focus_evidence[f"{case_name}_f{frame_idx:03d}"] = {
            "case": case_name,
            "frame_idx": frame_idx,
            "focus_reason": focus["focus_reason"],
            "mask_log_area_temporal_curvature": focus[
                "target_log_mask_area_temporal_curvature"
            ],
            "accepted_depth_temporal_curvature_m": focus[
                "target_depth_temporal_curvature_m"
            ],
            "target_accepted_depth_m": target.get("p03_source_camera_z_m"),
            "rigid_pose_observation_eligible": target.get(
                "rigid_pose_observation_eligible"
            ),
            "accepted_components": accepted_components(focus),
            "RGB_PnP_probes": probe_summary,
        }

    array = np.asarray(modern_pairs, dtype=np.float64)
    return {
        "status": "passed",
        "P09_focus_neighborhood_row_count": p09_neighborhood_rows,
        "P09_focus_neighborhood_unique_row_count": len(unique_neighborhood),
        "P09_exact_reconstructed_metric_row_count": p09_metric_rows,
        "PnP_probe_count": len(probe_rows),
        "strict_PnP_pass_count": len(strict_rows),
        "strict_PnP_fail_count": len(failed_rows),
        "strict_PnP_failure_rows": failed_rows,
        "modern_stage_comparison_count": len(modern_pairs),
        "final_closer_to_RGB_PnP_than_pairwise_count": final_closer_pairwise,
        "final_closer_to_RGB_PnP_than_measurement_count": final_closer_measurement,
        "pairwise_mapped_centroid_difference_median_m": float(np.median(array[:, 0])),
        "measurement_mapped_centroid_difference_median_m": float(
            np.median(array[:, 1])
        ),
        "final_mapped_centroid_difference_median_m": float(np.median(array[:, 2])),
        "median_pairwise_to_final_improvement_m": float(
            np.median(array[:, 0] - array[:, 2])
        ),
        "median_measurement_to_final_improvement_m": float(
            np.median(array[:, 1] - array[:, 2])
        ),
        "final_is_ground_truth": False,
        "probe_rows": probe_rows,
        "focus_evidence": focus_evidence,
    }


def write_reproduction_command(
    path: Path,
    *,
    python: Path,
    script: Path,
    script_hash: str,
    git_root: Path,
    commit: str,
    trace_root: Path,
) -> None:
    stdout_path = Path("/tmp/hot3d_upstream_trace_finalizer_stdout.json")
    if stdout_path.is_relative_to(trace_root):
        raise RuntimeError("finalizer stdout must remain outside trace root")
    content = f"""#!/usr/bin/env bash
set -euo pipefail
PY={json.dumps(str(python))}
SCRIPT={json.dumps(str(script))}
TRACE_ROOT={json.dumps(str(trace_root))}
GIT_ROOT={json.dumps(str(git_root))}
EXPECTED_COMMIT={json.dumps(commit)}
EXPECTED_SCRIPT_SHA256={json.dumps(script_hash)}
ACTUAL_SCRIPT_SHA256=$(sha256sum \"$SCRIPT\" | awk '{{print $1}}')
test \"$ACTUAL_SCRIPT_SHA256\" = \"$EXPECTED_SCRIPT_SHA256\"
test \"$(git -C \"$GIT_ROOT\" rev-parse HEAD)\" = \"$EXPECTED_COMMIT\"
test -z \"$(git -C \"$GIT_ROOT\" status --porcelain)\"
\"$PY\" \"$SCRIPT\" \\
  --trace-root \"$TRACE_ROOT\" \\
  --diagnostic-code-commit \"$EXPECTED_COMMIT\" \\
  --replace-final-reports \\
  > {json.dumps(str(stdout_path))}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def selected_artifact_paths(trace_root: Path) -> list[Path]:
    paths = sorted(
        list(trace_root.glob("cases/*/depth_component_sheets/*.jpg"))
        + list(trace_root.glob("cases/*/*_upstream_trace_timeline.jpg"))
        + list(trace_root.glob("cases/*/rgb_pnp/*_strict_rgb_pnp_depth_error.jpg"))
    )
    paths.extend(
        [
            trace_root / MECHANICAL_REPORT_NAME,
            trace_root / MECHANICAL_DONE_NAME,
            trace_root / MANUAL_REVIEW_RELATIVE,
            trace_root / NUMERIC_RELATIVE,
            trace_root / FINAL_REPORT_NAME,
            trace_root / FINAL_REPORT_ZH_NAME,
            trace_root / FINALIZE_COMMAND_RELATIVE,
        ]
    )
    resolved: list[Path] = []
    for path in paths:
        candidate = require_file(path, "selected final artifact")
        if not candidate.is_relative_to(trace_root):
            raise RuntimeError(f"selected artifact escaped trace root: {candidate}")
        resolved.append(candidate)
    if len(resolved) != 39 or len(set(resolved)) != 39:
        raise RuntimeError(f"expected 39 unique selected artifacts, got {len(resolved)}")
    return sorted(resolved)


def build_artifact_index(trace_root: Path, paths: list[Path]) -> dict[str, Any]:
    rows = [
        {
            "path": str(path.relative_to(trace_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "schema": "hot3d_upstream_trace_selected_artifact_sha256_index_v1",
        "status": "passed",
        "scope": (
            "Selected deliverables only: 32 reviewed images, mechanical report/sentinel, "
            "manual review, numeric audit, final JSON/ZH reports, and reproduction command. "
            "This index and the final sentinel are excluded to avoid self-reference."
        ),
        "entry_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "missing_count": 0,
        "undeclared_count": 0,
        "hash_mismatch_count": 0,
        "duplicate_count": 0,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    trace_root = args.trace_root.expanduser().resolve(strict=True)
    if "repair_validation" not in trace_root.parts:
        raise RuntimeError("trace root must remain under repair_validation")
    script_path = Path(__file__).resolve(strict=True)
    code = validate_committed_clean_code(script_path, args.diagnostic_code_commit)
    git_root = Path(code["git_root"])

    final_paths = [
        trace_root / NUMERIC_RELATIVE,
        trace_root / FINAL_REPORT_NAME,
        trace_root / FINAL_REPORT_ZH_NAME,
        trace_root / FINAL_INDEX_NAME,
        trace_root / FINAL_DONE_NAME,
        trace_root / FINALIZE_COMMAND_RELATIVE,
    ]
    existing = [path for path in final_paths if path.exists()]
    if existing and not args.replace_final_reports:
        raise RuntimeError(f"final reports already exist; use --replace-final-reports: {existing}")
    if args.replace_final_reports:
        for path in existing:
            path.unlink()

    mechanical_path = require_file(
        trace_root / MECHANICAL_REPORT_NAME, "mechanical trace report"
    )
    mechanical_done_path = require_file(
        trace_root / MECHANICAL_DONE_NAME, "mechanical trace sentinel"
    )
    manual_path = require_file(trace_root / MANUAL_REVIEW_RELATIVE, "manual review")
    mechanical = load_json(mechanical_path)
    mechanical_done = load_json(mechanical_done_path)
    if mechanical.get("status") != "mechanical_trace_complete_visual_review_pending":
        raise RuntimeError("unexpected mechanical trace status")
    if not bool(mechanical_done.get("visual_review_pending")):
        raise RuntimeError("mechanical sentinel provenance was unexpectedly rewritten")
    if mechanical_done["report_sha256"] != sha256_file(mechanical_path):
        raise RuntimeError("mechanical sentinel report hash mismatch")
    if int(mechanical.get("case_count", -1)) != 5 or int(
        mechanical.get("focus_frame_count", -1)
    ) != 10:
        raise RuntimeError("unexpected mechanical case/focus counts")
    audit_script = require_file(
        mechanical["diagnostic_script"], "mechanical trace script"
    )
    if sha256_file(audit_script) != mechanical["diagnostic_script_sha256"]:
        raise RuntimeError("mechanical trace script hash changed")
    validate_ancestor(
        git_root, mechanical["diagnostic_code_commit"], args.diagnostic_code_commit
    )

    manual_audit = validate_manual_review(
        trace_root, manual_path, mechanical_path
    )
    p03_p11_audit = validate_p03_and_p11(mechanical)
    probe_audit = validate_probes_and_stages(mechanical)
    source_audit = validate_source_immutability(mechanical)
    stage_ab_root = Path(mechanical["stage_ab_root"]).resolve(strict=True)
    prior_stage_ab = validate_prior_stage_ab(stage_ab_root)
    collection_root = Path(mechanical["collection_root"]).resolve(strict=True)
    collection_links = validate_collection_links(collection_root, mechanical["cases"])

    numeric = {
        "schema": "hot3d_upstream_mask_depth_pose_causal_audit_v1",
        "status": "passed",
        "claim_scope": (
            "Read-only causal audit. P09 and RGB-PnP are prediction-side evidence, not GT. "
            "Generated SAM3D/TRELLIS geometry is not pose evidence."
        ),
        "mechanical_report": str(mechanical_path),
        "mechanical_report_sha256": sha256_file(mechanical_path),
        "manual_review": str(manual_path),
        "manual_review_sha256": sha256_file(manual_path),
        "P03_P11_audit": p03_p11_audit,
        "P09_P14_RGB_PnP_audit": probe_audit,
        "manual_visual_audit": {
            key: value
            for key, value in manual_audit.items()
            if key not in {"rows", "authored_verdict"}
        },
        "prior_P14_P15_audit": prior_stage_ab,
        "collection_link_audit": collection_links,
        "source_immutability": {
            key: value for key, value in source_audit.items() if key != "rows"
        },
        "causal_boundary": manual_audit["authored_verdict"]["causal_boundary"],
        "limitations": manual_audit["authored_verdict"]["limitations"],
    }
    numeric_path = trace_root / NUMERIC_RELATIVE
    write_json(numeric_path, numeric)

    focus = probe_audit["focus_evidence"]
    final_report = {
        "schema": SCHEMA,
        "status": FINAL_STATUS,
        "visual_review_pending": False,
        "manual_image_read_review_complete": True,
        "claim_scope": (
            "The earliest persisted discrepancy for the supported focus frames is inside the "
            "P03 camera-z raster/P09 accepted surface, after a visually coherent owned mask. "
            "This is not a claim of GT depth or GT pose."
        ),
        "primary_answer": {
            "mask_is_primary_origin": False,
            "earliest_persisted_problem_stage": (
                "P03 source camera-z raster, subsequently accepted by P09"
            ),
            "P03c_adapter_is_origin": False,
            "P11_anchor_builder_is_origin": False,
            "P14_pairwise_can_propagate_problem": True,
            "P14_temporal_translation_is_primary_origin": False,
            "P14_temporal_translation_interpretation": (
                "Across 15 modern-schema strict RGB-PnP comparisons, final P14 is closer in "
                "mapped-centroid distance than pairwise and measurement in 14 directions. "
                "It usually compensates for upstream depth/partial-surface conflict on these "
                "focus transitions, but remains a prior rather than ground truth."
            ),
            "P15_is_origin_for_direct_rows": False,
            "generated_geometry_is_origin": False,
        },
        "corrected_prior_interpretation": {
            "superseded_statement": (
                "A higher residual to the current P09 surface after temporal translation is by "
                "itself evidence that temporal regularization made physical pose worse."
            ),
            "replacement": (
                "The current P09 surface can itself be temporally inconsistent. On reviewed "
                "transitions such as Soup f87, Mug f33/f118, and Spatula f117, strict RGB-PnP "
                "shows that pairwise/measurement follow the discrepant depth surface while final "
                "P14 moves closer to independent RGB motion. Residual-to-P09 is therefore not GT error."
            ),
            "remaining_warning": (
                "Temporal smoothing can still be wrong, can hide a genuine rapid motion, and is "
                "not a substitute for direct evidence; Soup 147->146 is one direction where final "
                "is not closer than pairwise."
            ),
        },
        "causal_chain": [
            {
                "stage": "P07/P09 owned RGB mask",
                "finding": (
                    "Reviewed contours are visually coherent. P11 selected masks are file-level "
                    "byte-identical copies of the corresponding P09 owned masks."
                ),
                "causal_role": "not_primary_for_reviewed_focus_anomalies",
            },
            {
                "stage": "P03 UniDepth source archive",
                "finding": (
                    "Object-local camera-z conflicts are already present in the earliest persisted "
                    "camera-z raster. Only post-conversion camera-z is saved."
                ),
                "causal_role": "earliest_persisted_discrepant_metric_evidence",
            },
            {
                "stage": "P03c camera-contract adapter",
                "finding": (
                    "frame_idx, depth, confidence, source_size, and intrinsics .npy members are "
                    "byte-identical between P03 source and P03c active archives for all five cases."
                ),
                "causal_role": "excluded_as_origin",
            },
            {
                "stage": "P09 ownership/eligibility",
                "finding": (
                    "The accepted raster reconstructs exactly, but eligibility lacks temporal-depth, "
                    "calibrated RGB-PnP, and cross-component rigidity gates."
                ),
                "causal_role": "allows_discrepant_or_mixed_surfaces_to_remain_pose_eligible",
            },
            {
                "stage": "P11 atomic anchor",
                "finding": (
                    "Selected frame, mask, official camera, intrinsics, camera/world surfels, centroid, "
                    "and canonical anchor bind exactly. Milk retains an explicit legacy schema limit."
                ),
                "causal_role": "excluded_as_new_origin",
            },
            {
                "stage": "P14 pairwise observed-surface ICP",
                "finding": (
                    "Local trimmed mutual-nearest ICP has match/residual gates but no RGB, cross-component, "
                    "or maximum-SE3-step consistency gate. It can propagate partial/mixed depth evidence."
                ),
                "causal_role": "downstream_propagation_and_ambiguity",
            },
            {
                "stage": "P14 temporal translation",
                "finding": (
                    "Usually closer to strict RGB-PnP than pairwise/measurement for reviewed modern-schema "
                    "transitions (14/15), while not being GT."
                ),
                "causal_role": "usually_corrective_on_reviewed_focus_transitions",
            },
            {
                "stage": "P15 completion",
                "finding": (
                    "Inherited finalized audit confirms 679 direct P14/P15 rows are exact array-identical; "
                    "P15 only adds 71 uncertain completed rows."
                ),
                "causal_role": "excluded_for_direct_frame_origin",
            },
        ],
        "case_conclusions": {
            "milk": {
                "classification": "legacy_schema_control_with_directional_RGB_depth_disagreement",
                "evidence": focus["milk_f092"],
                "conclusion": (
                    "Mask and dominant component are coherent. 91->92 agrees at +0.45 mm while "
                    "93->92 differs by -11.74 mm; old schema prevents strict historical pairwise reconstruction."
                ),
            },
            "soup": {
                "classification": "upstream_depth_transition_supported_at_f87",
                "evidence": {
                    "f87": focus["soup_f087"],
                    "f146_control": focus["soup_f146"],
                },
                "conclusion": (
                    "At f87, strict target-depth-minus-RGB-PnP is +17.96/+3.09 mm. For 86->87, "
                    "mapped-centroid discrepancy falls from 18.53 mm pairwise / 18.99 mm measurement "
                    "to 6.63 mm final. f146 is a stable control."
                ),
            },
            "mug": {
                "classification": "upstream_depth_conflict_at_f33_f118; unresolved_at_f143",
                "evidence": {
                    "f33": focus["mug_f033"],
                    "f118": focus["mug_f118"],
                    "f143": focus["mug_f143"],
                },
                "conclusion": (
                    "f33 is deeper than bidirectional RGB prediction by +16.29/+18.00 mm; f118 is "
                    "shallower by -27.94/-39.47 mm. Both f143 probes fail only the inlier-fraction gate, "
                    "so f143 remains information-insufficient."
                ),
            },
            "bbq": {
                "classification": "not_primarily_explained_by_P03_depth_on_available_strict_evidence",
                "evidence": focus["bbq_f058"],
                "conclusion": (
                    "The only strict direction is -3.29 mm; the reverse available probe fails inlier fraction "
                    "and adjacent f59 has no accepted metric row. Downstream partial-surface translation/chain "
                    "ambiguity is more plausible than a demonstrated large local depth bias."
                ),
            },
            "spatula": {
                "classification": "strong_upstream_depth_component_failure_at_f117",
                "evidence": {
                    "f10_control": focus["spatula_f010"],
                    "f37_mixed_surface": focus["spatula_f037"],
                    "f117": focus["spatula_f117"],
                },
                "conclusion": (
                    "At f117 the coherent mask accompanies 74.46 mm depth curvature and accepted components "
                    "near 0.814/0.394 m. Bidirectional strict RGB-PnP shows +46.30/+112.27 mm target depth error. "
                    "Final P14 partly compensates but does not make the evidence ground truth."
                ),
            },
        },
        "mechanical_counts": {
            "case_count": 5,
            "focus_frame_count": 10,
            "focus_neighborhood_row_count": 50,
            "exact_reconstructed_P09_metric_row_count": 45,
            "PnP_probe_count": 20,
            "strict_PnP_pass_count": 17,
            "strict_PnP_fail_count": 3,
            "modern_stage_comparison_count": 15,
            "final_closer_than_pairwise_count": 14,
            "final_closer_than_measurement_count": 14,
            "manual_reviewed_image_count": 32,
            "P11_exact_transfer_check_count": 80,
            "P11_canonical_anchor_check_count": 10,
            "source_hash_entry_count": source_audit["entry_count"],
            "prior_P14_P15_direct_identity_rows": 679,
            "prior_P15_completed_rows": 71,
        },
        "numeric_summary": {
            "pairwise_mapped_centroid_difference_median_m": probe_audit[
                "pairwise_mapped_centroid_difference_median_m"
            ],
            "measurement_mapped_centroid_difference_median_m": probe_audit[
                "measurement_mapped_centroid_difference_median_m"
            ],
            "final_mapped_centroid_difference_median_m": probe_audit[
                "final_mapped_centroid_difference_median_m"
            ],
            "median_pairwise_to_final_improvement_m": probe_audit[
                "median_pairwise_to_final_improvement_m"
            ],
            "median_measurement_to_final_improvement_m": probe_audit[
                "median_measurement_to_final_improvement_m"
            ],
        },
        "provenance": {
            "finalizer": code,
            "mechanical_trace_script": {
                "path": str(audit_script),
                "sha256": sha256_file(audit_script),
                "commit": mechanical["diagnostic_code_commit"],
            },
            "mechanical_report": str(mechanical_path),
            "mechanical_report_sha256": sha256_file(mechanical_path),
            "mechanical_sentinel": str(mechanical_done_path),
            "mechanical_sentinel_sha256": sha256_file(mechanical_done_path),
            "manual_review": str(manual_path),
            "manual_review_sha256": sha256_file(manual_path),
            "numeric_audit": str(numeric_path),
            "numeric_audit_sha256": sha256_file(numeric_path),
            "prior_stage_ab_root": str(stage_ab_root),
            "collection_root": str(collection_root),
        },
        "immutability": {
            "mechanical_source_hash_mismatch_count": 0,
            "prior_stage_ab_source_hash_mismatch_count": 0,
            "collection_links_changed": False,
            "finalized_case_roots_modified": False,
            "collection_modified": False,
            "prior_stage_ab_modified": False,
        },
        "limitations": manual_audit["authored_verdict"]["limitations"],
    }
    final_report_path = trace_root / FINAL_REPORT_NAME
    write_json(final_report_path, final_report)

    pair_mm = probe_audit["pairwise_mapped_centroid_difference_median_m"] * 1000.0
    measurement_mm = (
        probe_audit["measurement_mapped_centroid_difference_median_m"] * 1000.0
    )
    final_mm = probe_audit["final_mapped_centroid_difference_median_m"] * 1000.0
    zh = f"""# HOT3D mask→depth→pose 上游只读溯源最终报告

## 最终状态

`{FINAL_STATUS}`

- 人工 image-read：完成（32 张：10 张 mask/P03/P09 neighborhood、5 张全时序、17 张 strict RGB-PnP overlay）。
- 机械审计：通过。
- finalized case roots、统一 collection、上一轮 P14/P15 A/B：均未修改。
- 本报告不使用 GT pose、GT depth、GT MANO、CAD 或 generated SAM3D/TRELLIS geometry。

## 直接回答

异常**不是首先从 2D mask 开始**。在已审阅 focus frames 中，owned mask 轮廓保持视觉连续；最早可持久化定位的异常位于：

`P03 source camera-z raster → P09 accepted metric surface/component`

P03c 没有制造这些差异：五例 `frame_idx/depth/confidence/source_size/intrinsics` 的 `.npy` members 在 source 与 active archives 间均 byte-identical。P11 也没有制造新差异：五例 selected frame、mask、official camera、K、camera/world surfels、centroid 和 canonical anchor 均精确绑定；其中 P11 mask copy 与 P09 mask 文件级 byte-identical。Milk 保留旧 schema，未伪造现代 atomic-binding 字段。

P09 的局部 component ownership 可精确重建，但 eligibility 没有 temporal-depth、calibrated RGB-PnP 或 cross-component rigidity gate，所以异常或互不一致的局部表面仍可能成为 `eligible`。P14 pairwise ICP 又只按局部 match count 与 trimmed residual 放行，没有 RGB、cross-component 或最大 SE(3) step gate，因而会传播 partial/mixed surface 冲突。

## 对上一轮 P14 结论的修正

不能再把“final P14 相对当帧 P09 surface residual 上升”直接等价成“temporal regularizer 制造了物理错误”。当帧 P09 surface 自身可能已经错。

在 15 个可比较的现代 schema + strict RGB-PnP directions 中：

- final P14 比 pairwise 更接近 RGB-PnP：`14/15`；
- final P14 比 pre-temporal measurement 更接近 RGB-PnP：`14/15`；
- mapped-centroid discrepancy 中位数：pairwise `{pair_mm:.2f} mm`，measurement `{measurement_mm:.2f} mm`，final `{final_mm:.2f} mm`。

因此，在这些 focus transitions 上，temporal translation 多数是在**补偿**上游 depth/partial-surface conflict，而不是首次制造异常。但 final 仍是 prior，不是 GT；Soup `147→146` 是 final 不比 pairwise 更近的方向，说明平滑不能自动升级为真值。

## 逐例结论

### Milk f92

- mask 和单一 dominant accepted component 连续。
- strict RGB-PnP：`91→92 = +0.45 mm`，`93→92 = −11.74 mm`。
- 方向间不一致；Milk 又缺现代历史 pairwise-stage 字段，因此只能作为 legacy/control，不作强上游故障判定。

### Soup f87 / f146 control

- f87 mask 连续；target P09 depth 相对 strict RGB-PnP 为 `+17.96/+3.09 mm`。
- `86→87` mapped-centroid discrepancy：pairwise `18.53 mm`、measurement `18.99 mm`、final `6.63 mm`。
- f146 control 仅约 `+3.81/+0.20 mm`。
- 结论：f87 有上游 depth transition 支持，regularizer 在该 transition 上是纠偏而非初始原因。

### Mug f33 / f118 / f143

- f33：target 比 RGB prediction 深 `+16.29/+18.00 mm`。
- f118：target 比 RGB prediction 浅 `−27.94/−39.47 mm`。
- 两帧均为双向 strict support，且 final 明显比 pairwise/measurement 更接近 RGB-PnP。
- f143：两个方向都只因 PnP inlier fraction `<0.5` fail closed；保持 evidence-insufficient，不把失败解释为 depth 正确。

### BBQ f58

- 唯一 strict direction `57→58` 仅 `−3.29 mm`；`60→58` 因 inlier fraction 不足失败，f59 又无 accepted metric row。
- 结论：当前证据不支持“大 P03 object-local depth bias”是 f58 主因；更像 downstream partial-surface chain/translation ambiguity，但仍受单向 strict evidence 限制。

### Spatula f37 / f117

- f37：同一 owned mask 内 accepted components 约 `0.370/0.671 m`，是 mixed/partial-surface ambiguity，不能简化为单一 z-bias。
- f117：mask-area temporal curvature 仅约 `0.0030`，但 accepted-depth curvature 为 `74.46 mm`；accepted components 约 `0.814/0.394 m`。
- 双向 strict RGB-PnP 指向 target P09 过深：`+46.30/+112.27 mm`。
- pairwise/measurement mapped-centroid discrepancy 为 `58.83/58.73 mm`（从 f116）及 `115.30/118.08 mm`（从 f118），final 降为 `18.35/51.90 mm`。
- 结论：这是五例中最强的上游 P03/P09 depth-component conflict 证据；P14 final 只部分补偿，不能把其升级为 GT。

## P15 与 generated geometry

上一轮 finalized exact-SE(3) 审计继续成立：`679` 个 P14/P15 direct rows 逐数组完全相同，P15 仅增加 `71` 个 uncertain completed rows。因此 P15 不是 direct-frame 异常来源。Generated SAM3D/TRELLIS geometry 未作为本审计的 pose/depth/contact 证据。

## 因果边界与限制

- 最早可定位的是 P03 archive 中已经存在的 post-conversion camera-z raster。
- P03 不保存 pre-conversion metric-radius tensor，因此不能无证据细分为 UniDepth radius head 错误还是 radius→exact-ray conversion 内部错误。
- RGB-PnP 继承 source-frame metric depth；双向支持强于单向，但仍不是 GT。
- PnP fail 表示信息不足，不表示 target depth 正确。
- 静态背景 probe 显示部分帧约 1–2% frame-level bias，但不能解释全部 object-local component 结构。
- partial/disconnected observed surface 仍然不能提供 signed contact、penetration 或 nonpenetration 结论。

## 审计计数

- cases：5
- focus frames：10
- focus neighborhood rows：50（其中 45 个 P09 metric rows 精确重建）
- RGB-PnP probes：20（strict pass 17，fail 3）
- modern stage comparisons：15
- 人工打开并绑定的图：32
- P11 exact transfer checks：80/80
- P11 canonical-anchor checks：10/10
- 当前 source hashes：{source_audit['entry_count']}/{source_audit['entry_count']}
- prior direct P14/P15 identity：679/679

权威结构化数据见 `{FINAL_REPORT_NAME}` 与 `{NUMERIC_RELATIVE}`。
"""
    final_zh_path = trace_root / FINAL_REPORT_ZH_NAME
    final_zh_path.write_text(zh, encoding="utf-8")

    command_path = trace_root / FINALIZE_COMMAND_RELATIVE
    write_reproduction_command(
        command_path,
        python=Path(sys.executable).resolve(strict=True),
        script=script_path,
        script_hash=code["script_sha256"],
        git_root=git_root,
        commit=args.diagnostic_code_commit,
        trace_root=trace_root,
    )

    artifact_paths = selected_artifact_paths(trace_root)
    index = build_artifact_index(trace_root, artifact_paths)
    index_path = trace_root / FINAL_INDEX_NAME
    write_json(index_path, index)
    sentinel = {
        "schema": SCHEMA,
        "status": FINAL_STATUS,
        "visual_review_pending": False,
        "manual_image_read_review_complete": True,
        "case_count": 5,
        "focus_frame_count": 10,
        "strict_PnP_pass_count": 17,
        "strict_PnP_fail_count": 3,
        "reviewed_image_count": 32,
        "selected_artifact_count": int(index["entry_count"]),
        "source_hash_mismatch_count": 0,
        "finalized_case_roots_modified": False,
        "collection_modified": False,
        "prior_stage_ab_modified": False,
        "diagnostic_code_commit": args.diagnostic_code_commit,
        "finalizer_sha256": code["script_sha256"],
        "final_report": str(final_report_path),
        "final_report_sha256": sha256_file(final_report_path),
        "final_report_zh": str(final_zh_path),
        "final_report_zh_sha256": sha256_file(final_zh_path),
        "numeric_audit": str(numeric_path),
        "numeric_audit_sha256": sha256_file(numeric_path),
        "manual_review": str(manual_path),
        "manual_review_sha256": sha256_file(manual_path),
        "artifact_index": str(index_path),
        "artifact_index_sha256": sha256_file(index_path),
        "reproduction_command": str(command_path),
        "reproduction_command_sha256": sha256_file(command_path),
    }
    done_path = trace_root / FINAL_DONE_NAME
    write_json(done_path, sentinel)

    # Independent final consistency before success is emitted.
    reloaded = load_json(done_path)
    for path_key, hash_key in (
        ("final_report", "final_report_sha256"),
        ("final_report_zh", "final_report_zh_sha256"),
        ("numeric_audit", "numeric_audit_sha256"),
        ("manual_review", "manual_review_sha256"),
        ("artifact_index", "artifact_index_sha256"),
        ("reproduction_command", "reproduction_command_sha256"),
    ):
        path = require_file(reloaded[path_key], path_key)
        if sha256_file(path) != reloaded[hash_key]:
            raise RuntimeError(f"sentinel hash mismatch: {path_key}")
    print(json.dumps(sentinel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
