#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_physical_state_schema"
CLAIM = (
    "This artifact converts model-produced physical notes into a structured object physical-state schema: primary "
    "whole-object state, part/relative-motion requirement, deformable/optical secondary evidence, and unresolved "
    "surface-change flags. It centralizes the text adapter so downstream gates do not each parse free text. It does "
    "not infer geometry, pose, articulation parameters, or contact ownership."
)

PRIMARY_STATES = {"rigid", "deformable", "articulated", "unknown", "unknown_optically_difficult"}
RIGID_TERMS = (
    "rigid",
    "does not deform",
    "does not change shape",
    "no object shape change",
)
DEFORMABLE_TERMS = (
    "deformable",
    "non-rigid",
    "nonrigid",
    "flexible",
    "changes shape",
    "curled",
    "folded",
    "crumpled",
    "bunched",
    "draped",
)
PRIMARY_ARTICULATION_TERMS = ("articulated", "hinged", "hinge", "lever")
PART_OR_RELATIVE_MOTION_TERMS = (
    "articulated",
    "hinged",
    "hinge",
    "lever",
    "relative motion",
    "moves relative",
    "relative to",
    "position changes",
    "changes position",
    "opens",
    "closes",
)
OPTICAL_TERMS = ("transparent", "translucent", "reflective", "reflections", "clear plastic")
SURFACE_CHANGE_TERMS = ("surface appearance changes", "skin is removed", "peeled", "detached", "surface texture")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def source_from_measurement_manifest(manifest: dict[str, Any], key: str) -> Path:
    raw = manifest.get(key)
    if isinstance(raw, str):
        return Path(raw)
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, str):
            return Path(first)
        if isinstance(first, dict) and isinstance(first.get("path"), str):
            return Path(first["path"])
    raise RuntimeError(f"measurement manifest has no usable source for {key}")


def matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def structured_from_notes(notes: Any) -> dict[str, Any]:
    if not isinstance(notes, str) or not notes.strip():
        return {
            "model_physical_state_type": "unknown",
            "physical_state_source": "unknown_no_model_notes",
            "requires_part_or_relative_motion_model": False,
            "part_or_relative_motion_evidence_terms": [],
            "secondary_deformable_or_surface_component": False,
            "secondary_deformable_evidence_terms": [],
            "optical_difficulty": False,
            "optical_evidence_terms": [],
            "surface_change_without_pose_state": False,
            "surface_change_evidence_terms": [],
            "schema_confidence": "low",
            "schema_blockers": ["missing_model_physical_notes"],
        }
    text = notes.lower()
    rigid_matches = matched_terms(text, RIGID_TERMS)
    if "non-rigid" in text or "nonrigid" in text:
        rigid_matches = [term for term in rigid_matches if term != "rigid"]
    deform_matches = matched_terms(text, DEFORMABLE_TERMS)
    primary_articulation_matches = matched_terms(text, PRIMARY_ARTICULATION_TERMS)
    part_motion_matches = matched_terms(text, PART_OR_RELATIVE_MOTION_TERMS)
    optical_matches = matched_terms(text, OPTICAL_TERMS)
    surface_change_matches = matched_terms(text, SURFACE_CHANGE_TERMS)

    has_rigid = bool(rigid_matches)
    has_deform = bool(deform_matches) or "no fixed geometry should be assumed" in text
    has_articulation = bool(primary_articulation_matches)
    has_part_or_relative_motion = bool(part_motion_matches)
    has_optical = bool(optical_matches)
    has_surface_change = bool(surface_change_matches)

    blockers: list[str] = []
    if has_articulation:
        primary = "articulated"
    elif has_deform and not has_rigid:
        primary = "deformable"
    elif has_rigid:
        primary = "rigid"
    elif has_optical:
        primary = "unknown_optically_difficult"
        blockers.append("optical_terms_without_primary_physical_state")
    else:
        primary = "unknown"
        blockers.append("no_explicit_primary_physical_state_term")
    if primary not in PRIMARY_STATES:
        raise RuntimeError(f"unexpected primary physical state: {primary}")
    secondary_deformable = bool(has_deform and primary in {"rigid", "articulated"})
    if secondary_deformable:
        blockers.append("secondary_deformable_or_surface_component_present")
    if has_surface_change and primary == "unknown":
        blockers.append("surface_change_without_pose_model")
    confidence = "high" if primary in {"rigid", "deformable", "articulated"} and not blockers else "medium" if primary != "unknown" else "low"
    return {
        "model_physical_state_type": primary,
        "physical_state_source": "vlm_physical_notes_structured_schema_v1",
        "requires_part_or_relative_motion_model": has_part_or_relative_motion,
        "primary_articulation_evidence_terms": primary_articulation_matches,
        "part_or_relative_motion_evidence_terms": part_motion_matches,
        "secondary_deformable_or_surface_component": secondary_deformable,
        "secondary_deformable_evidence_terms": deform_matches if secondary_deformable else [],
        "optical_difficulty": has_optical,
        "optical_evidence_terms": optical_matches,
        "surface_change_without_pose_state": has_surface_change and primary == "unknown",
        "surface_change_evidence_terms": surface_change_matches,
        "schema_confidence": confidence,
        "schema_blockers": blockers,
    }


def legacy_physical_state_from_notes(notes: Any) -> str:
    if not isinstance(notes, str) or not notes.strip():
        return "unknown"
    text = notes.lower()
    if any(word in text for word in ("hinged", "lever", "articulated")):
        return "articulated"
    if any(word in text for word in ("deformable", "non-rigid", "nonrigid", "flexible", "thin", "changes shape", "curled")):
        return "deformable"
    if "rigid" in text:
        return "rigid"
    if any(word in text for word in ("transparent", "translucent", "reflective")):
        return "unknown_optically_difficult"
    return "unknown"


def timeline_objects(timeline: dict[str, Any], roster_by_object_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for raw in require_list(timeline.get("objects"), "timeline objects"):
        row = require_dict(raw, "timeline object")
        object_id = require_str(row.get("object_id"), "object_id")
        roster_row = roster_by_object_id.get(object_id, {})
        merged = {**row}
        for key in ("physical_notes", "role_status", "source"):
            if key not in merged or merged.get(key) is None:
                merged[key] = roster_row.get(key)
        objects.append(merged)
    return objects


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    measurement_manifest_path = existing(args.measurement_store_root / case / "v17_measurement_manifest.json", f"{case} measurement manifest")
    measurement_manifest = require_dict(load_json(measurement_manifest_path), f"{case} measurement manifest")
    timeline_path = existing(args.multi_object_timeline_root / case / "v17_multi_object_timeline.json", f"{case} object timeline")
    timeline = require_dict(load_json(timeline_path), f"{case} object timeline")
    roster_path = existing(Path(require_str(measurement_manifest.get("object_roster"), f"{case} object_roster")), f"{case} object roster")
    roster_rows = [require_dict(row, "object roster row") for row in require_list(load_json(roster_path), f"{case} object roster")]
    roster_by_object_id = {require_str(row.get("object_id"), "roster object_id"): row for row in roster_rows}
    object_rows: list[dict[str, Any]] = []
    primary_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    part_motion_count = 0
    secondary_deformable_count = 0
    optical_count = 0
    surface_change_count = 0
    changed_from_legacy: list[dict[str, Any]] = []
    for obj in timeline_objects(timeline, roster_by_object_id):
        notes = obj.get("physical_notes")
        structured = structured_from_notes(notes)
        legacy = legacy_physical_state_from_notes(notes)
        primary = require_str(structured.get("model_physical_state_type"), "model physical state")
        primary_counts[primary] += 1
        legacy_counts[legacy] += 1
        if bool(structured.get("requires_part_or_relative_motion_model")):
            part_motion_count += 1
        if bool(structured.get("secondary_deformable_or_surface_component")):
            secondary_deformable_count += 1
        if bool(structured.get("optical_difficulty")):
            optical_count += 1
        if bool(structured.get("surface_change_without_pose_state")):
            surface_change_count += 1
        row = {
            "object_id": obj.get("object_id"),
            "track_id": obj.get("track_id"),
            "name": obj.get("name"),
            "physical_notes": notes,
            "role_status": obj.get("role_status"),
            "roster_source": obj.get("source"),
            "legacy_keyword_physical_state_type": legacy,
            **structured,
            "structured_schema_ready": True,
            "part_pose_ready": False,
            "object_pose_requirement_met": False,
        }
        if legacy != primary:
            changed_from_legacy.append(
                {
                    "object_id": obj.get("object_id"),
                    "legacy_keyword_physical_state_type": legacy,
                    "model_physical_state_type": primary,
                    "schema_reason": structured.get("schema_blockers"),
                }
            )
        object_rows.append(row)
    report = {
        "method": "build_v18_physical_state_schema",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v17_measurement_manifest": str(measurement_manifest_path),
            "v17_multi_object_timeline": str(timeline_path),
            "object_roster": str(roster_path),
        },
        "object_count": len(object_rows),
        "model_physical_state_type_counts": dict(sorted(primary_counts.items())),
        "legacy_keyword_physical_state_type_counts": dict(sorted(legacy_counts.items())),
        "part_or_relative_motion_required_count": part_motion_count,
        "secondary_deformable_or_surface_component_count": secondary_deformable_count,
        "optical_difficulty_count": optical_count,
        "surface_change_without_pose_state_count": surface_change_count,
        "changed_from_legacy_keyword_count": len(changed_from_legacy),
        "changed_from_legacy_keyword_rows": changed_from_legacy,
        "object_rows": object_rows,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_physical_state_schema_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    primary_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    for report in reports:
        primary_counts.update(report["model_physical_state_type_counts"])
        legacy_counts.update(report["legacy_keyword_physical_state_type_counts"])
    summary = {
        "method": "build_v18_physical_state_schema",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "object_count": sum(int(report["object_count"]) for report in reports),
        "model_physical_state_type_counts": dict(sorted(primary_counts.items())),
        "legacy_keyword_physical_state_type_counts": dict(sorted(legacy_counts.items())),
        "part_or_relative_motion_required_count": sum(int(report["part_or_relative_motion_required_count"]) for report in reports),
        "secondary_deformable_or_surface_component_count": sum(int(report["secondary_deformable_or_surface_component_count"]) for report in reports),
        "optical_difficulty_count": sum(int(report["optical_difficulty_count"]) for report in reports),
        "surface_change_without_pose_state_count": sum(int(report["surface_change_without_pose_state_count"]) for report in reports),
        "changed_from_legacy_keyword_count": sum(int(report["changed_from_legacy_keyword_count"]) for report in reports),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_physical_state_schema_report.json"),
                "object_count": report["object_count"],
                "model_physical_state_type_counts": report["model_physical_state_type_counts"],
                "changed_from_legacy_keyword_count": report["changed_from_legacy_keyword_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_physical_state_schema_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--multi-object-timeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
