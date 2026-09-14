"""Immutable prediction provenance and run containment."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
import hashlib

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def in_run(value, root, *, must_exist=True):
    root = Path(root).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f'clip-specific path must be absolute: {value}')
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'clip-specific input escapes run root: {value}')
    if must_exist and not resolved.is_file():
        raise ValueError(f'input file missing: {value}')
    return resolved


_GT_KEYS = {
    "gt_consumed",
    "gt_used_as_solver_input",
    "gt_used_for_initialization",
    "gt_used_for_pose_initialization",
    "gt_used_for_candidate_generation",
    "gt_used_for_solver_input",
    "hot3d_gt_consumed",
    "uses_gt",
    "used_gt",
    "used_ground_truth",
    "ground_truth_consumed",
}


def _key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def find_positive_gt_flags(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _key(key) in _GT_KEYS and child is True:
                found.append(child_path)
            found.extend(find_positive_gt_flags(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(find_positive_gt_flags(child, f"{path}[{index}]"))
    return found


def assert_prediction_only(
    report: Mapping[str, Any],
    *,
    label: str,
    object_id: str | None = None,
    allow_annotation_ready: bool = False,
    allow_nondiagnostic_prediction: bool = False,
    allowed_schema_prefixes: tuple[str, ...] = (),
    allowed_statuses: tuple[str, ...] = (),
) -> None:
    flags = find_positive_gt_flags(report)
    if flags:
        raise RuntimeError(f"{label} has positive GT provenance flags: {flags[:20]}")
    if not allow_annotation_ready and report.get("annotation_ready") is True:
        raise RuntimeError(f"{label} cannot be annotation-ready input")
    if not allow_nondiagnostic_prediction and report.get("diagnostic_only") is False:
        raise RuntimeError(f"{label} is not diagnostic_only")
    if object_id is not None:
        actual = report.get("object_id")
        if actual is not None and str(actual) != str(object_id):
            raise RuntimeError(f"{label} object_id mismatch: {actual!r} != {object_id!r}")
    if allowed_schema_prefixes:
        schema = str(report.get("schema") or "")
        if not any(schema.startswith(prefix) for prefix in allowed_schema_prefixes):
            raise RuntimeError(f"{label} schema is not allowlisted: {schema!r}")
    if allowed_statuses:
        status = str(report.get("status") or "")
        if status not in allowed_statuses:
            raise RuntimeError(f"{label} status is not allowlisted: {status!r}")


def require_equal_object_ids(*named_reports: tuple[str, Mapping[str, Any]]) -> str:
    values: dict[str, str] = {}
    for label, report in named_reports:
        value = report.get("object_id")
        if value is None or not str(value).strip():
            raise RuntimeError(f"{label} has no nonempty object_id")
        values[label] = str(value)
    if len(set(values.values())) != 1:
        raise RuntimeError(f"object_id mismatch: {values}")
    return next(iter(values.values()))


def append_prediction_stage(
    report: dict[str, Any],
    *,
    stage: str,
    input_hashes: Mapping[str, str | None],
    notes: Mapping[str, Any] | None = None,
) -> None:
    chain = report.get("prediction_provenance_chain")
    chain = list(chain) if isinstance(chain, list) else []
    chain.append(
        {
            "stage": stage,
            "gt_consumed": False,
            "input_sha256": dict(input_hashes),
            "notes": dict(notes or {}),
        }
    )
    report["prediction_provenance_chain"] = chain
    report["gt_consumed"] = False
    report["annotation_ready"] = False
    report["diagnostic_only"] = True
