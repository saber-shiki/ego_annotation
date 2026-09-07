#!/usr/bin/env python3
"""Audit stored V20 generated signed-depth evidence without changing a pose.

This is deliberately a post-hoc report reader.  It never loads HOT3D GT, never
reruns optimization, and never promotes generated geometry to physical
authority.  It catches a sustained late negative signed-depth segment that an
all-frame aggregate median can hide.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from v20_prediction_contracts import assert_prediction_only
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v20_prediction_contracts import assert_prediction_only


def import_solver() -> Any:
    path = Path(__file__).with_name("fit_v20_late_window_prediction_only_se3.py")
    spec = importlib.util.spec_from_file_location("v20_signed_front_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import signed-front helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def stored_metrics(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    """Find the final stored raw metric block in pose or shape reports."""
    final = report.get("final_metrics")
    if isinstance(final, Mapping):
        return final, "final_metrics"
    solver = report.get("solver")
    if isinstance(solver, Mapping):
        final_validation = solver.get("final_generated_validation")
        if isinstance(final_validation, Mapping):
            raw = final_validation.get("raw_metrics")
            if isinstance(raw, Mapping):
                return raw, "solver.final_generated_validation.raw_metrics"
        iterations = solver.get("outer_iterations")
        if isinstance(iterations, list):
            for outer in reversed(iterations):
                if not isinstance(outer, Mapping):
                    continue
                trials = outer.get("trials")
                if not isinstance(trials, list):
                    continue
                for trial in reversed(trials):
                    if not isinstance(trial, Mapping):
                        continue
                    validation = trial.get("generated_validation")
                    if isinstance(validation, Mapping) and isinstance(validation.get("raw_metrics"), Mapping):
                        return validation["raw_metrics"], "solver.outer_iterations[].trials[].generated_validation.raw_metrics"
    raise RuntimeError("report contains no stored generated first-hit metrics")


def audit_report(
    path: Path,
    solver: Any,
    *,
    threshold_m: float,
    max_per_frame_bias_m: float | None,
    max_segment_length: int | None,
) -> dict[str, Any]:
    report = load(path)
    assert_prediction_only(report, label=f"signed-front audit report {path}", object_id=report.get("object_id"))
    metrics, source = stored_metrics(report)
    diagnostics = solver.signed_front_bias_diagnostics(
        None, metrics, negative_threshold_m=threshold_m
    )
    passed, reasons = solver.signed_front_bias_gate(
        diagnostics,
        max_front_bias_m=None,
        max_front_fraction=None,
        max_per_frame_front_bias_m=max_per_frame_bias_m,
        max_negative_front_segment_length=max_segment_length,
    )
    return {
        "report": str(path.expanduser().resolve()),
        "report_schema": report.get("schema"),
        "report_status": report.get("status"),
        "metrics_source": source,
        "visible_pose_render_gate": {
            "pass": bool(passed),
            "reasons": reasons,
            "negative_threshold_m": float(threshold_m),
            "max_per_frame_front_bias_m": max_per_frame_bias_m,
            "max_negative_front_segment_length": max_segment_length,
        },
        "signed_front_diagnostics": diagnostics,
        "gt_consumed": bool(report.get("gt_consumed", False)),
        "generated_mesh_authority": report.get("generated_mesh_role")
        or report.get("claim_scope")
        or "diagnostic_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-threshold-m", type=float, default=0.005)
    parser.add_argument("--max-per-frame-front-bias-m", type=float, default=0.005)
    parser.add_argument("--max-negative-front-segment-length", type=int, default=2)
    args = parser.parse_args()
    if args.negative_threshold_m < 0 or args.max_per_frame_front_bias_m < 0 or args.max_negative_front_segment_length < 0:
        raise RuntimeError("signed-front thresholds must be non-negative")
    solver = import_solver()
    audits = [
        audit_report(
            path,
            solver,
            threshold_m=float(args.negative_threshold_m),
            max_per_frame_bias_m=float(args.max_per_frame_front_bias_m) if args.max_per_frame_front_bias_m is not None else None,
            max_segment_length=int(args.max_negative_front_segment_length) if args.max_negative_front_segment_length is not None else None,
        )
        for path in args.report
    ]
    result = {
        "schema": "v20_signed_front_validation_audit_v1",
        "status": "pass" if all(item["visible_pose_render_gate"]["pass"] for item in audits) else "incomplete_signed_front_validation",
        "annotation_ready": False,
        "diagnostic_only": True,
        "gt_consumed": False,
        "authority_scope": "visible-pose/render evidence only; not collision/contact/SDF/sign/signed-volume/nonpenetration authority",
        "audits": audits,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "reports": len(audits), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
