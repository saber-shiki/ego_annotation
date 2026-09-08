#!/usr/bin/env python3
"""Run the validated milk demo fit/render or replay a frozen demo state.

This is a demo postprocessing entry point, NOT a V19 fresh-run orchestrator.
It never installs dependencies, starts agents, or chooses a compute host.
Launch fitting on the explicitly selected server/GPU in managed tmux.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


FIT_INPUTS = (
    "annotations",
    "pose_report",
    "mano_bridge",
    "hawor_params",
    "mano_models",
)
PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "smplx",
    "chumpy",
    "trimesh",
    "open3d",
    "opencv-python-headless",
    "rerun-sdk",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("fit-render", "replay"), required=True)
    result.add_argument("--source-video", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--state", type=Path, help="Frozen demo_state.npz; replay only")
    for name in FIT_INPUTS:
        result.add_argument("--" + name.replace("_", "-"), type=Path)
    result.add_argument("--object-id", default="carton_milk")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--iterations", type=int, default=240)
    result.add_argument("--detail-yaw-deg", type=float, default=20.0)
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Check input paths and print commands; do not load models or write outputs",
    )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {args.output_dir}")
    if not args.source_video.is_file():
        raise ValueError(f"source video not found: {args.source_video}")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.expanduser().resolve()
    steps: list[tuple[str, list[str]]] = []
    if args.mode == "fit-render":
        if args.state is not None:
            raise ValueError("--state is only accepted in replay mode")
        missing = [name for name in FIT_INPUTS if getattr(args, name) is None]
        if missing:
            raise ValueError(f"fit-render requires inputs: {', '.join(missing)}")
        for name in FIT_INPUTS[:-1]:
            if not getattr(args, name).is_file():
                raise ValueError(f"input not found: {name}={getattr(args, name)}")
        for side in ("LEFT", "RIGHT"):
            path = args.mano_models / f"MANO_{side}.pkl"
            if not path.is_file():
                raise ValueError(f"MANO asset not found: {path}")
        fit = [sys.executable, str(script_dir / "fit_v20_visual_contact_demo.py")]
        for name in FIT_INPUTS:
            fit.extend(
                ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
            )
        fit.extend(
            [
                "--object-id",
                args.object_id,
                "--device",
                args.device,
                "--iterations",
                str(args.iterations),
                "--output-dir",
                str(output_dir / "fit"),
            ]
        )
        steps.append(("fit", fit))
        state_path = output_dir / "fit" / "demo_state.npz"
    else:
        if args.state is None or not args.state.is_file():
            raise ValueError("replay requires an existing --state demo_state.npz")
        if any(getattr(args, name) is not None for name in FIT_INPUTS):
            raise ValueError("replay does not consume fit inputs; pass only --state")
        state_path = args.state.resolve()
    render = [
        sys.executable,
        str(script_dir / "render_v20_visual_contact_demo.py"),
        "--state",
        str(state_path),
        "--source-video",
        str(args.source_video.resolve()),
        "--output-dir",
        str(output_dir / "renders"),
        "--detail-yaw-deg",
        str(args.detail_yaw_deg),
    ]
    steps.append(("render", render))
    return steps


def environment() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def execute(args: argparse.Namespace, plan: list[tuple[str, list[str]]]) -> int:
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = {"source_video": args.source_video.resolve()}
    if args.state is not None:
        inputs["state"] = args.state.resolve()
    if args.mode == "fit-render":
        inputs.update({name: getattr(args, name).resolve() for name in FIT_INPUTS[:-1]})
        for side in ("LEFT", "RIGHT"):
            inputs[f"mano_{side.lower()}"] = args.mano_models / f"MANO_{side}.pkl"
    report = {
        "schema": "v20_demo_postprocess_run_v1",
        "status": "running",
        "demo_only": True,
        "annotation_ready": False,
        "is_v19_fresh_run": False,
        "mode": args.mode,
        "environment": environment(),
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in inputs.items()
        },
        "script_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "run_v20_visual_contact_demo.py",
                "fit_v20_visual_contact_demo.py",
                "render_v20_visual_contact_demo.py",
                "v20_demo_geometry.py",
                "v20_prediction_contracts.py",
            )
        },
        "steps": [],
    }
    report_path = root / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for key, command in plan:
        start = time.monotonic()
        step = {"name": key, "argv": command, "log": f"{key}.log"}
        try:
            with (root / step["log"]).open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            code = process.returncode
        except OSError as exc:
            code = 1
            step["launch_error"] = str(exc)
        step.update(returncode=code, elapsed_s=time.monotonic() - start)
        report["steps"].append(step)
        if code:
            report["status"] = "failed"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if code:
            print(f"{key} failed; inspect {root / step['log']}", file=sys.stderr)
            return code if code > 0 else 1
    report["status"] = "completed_pending_visual_review"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Demo outputs: {root / 'renders'}")
    return 0


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    # Expand user paths consistently for validation and execution.
    for name in ("output_dir", "source_video", "state", *FIT_INPUTS):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser())
    try:
        plan = build_plan(args)
    except ValueError as exc:
        cli.error(str(exc))
    if args.dry_run:
        print(
            "Dry run: path validation only; model/decoder compatibility is not checked."
        )
        for key, command in plan:
            print(f"{key}: {shlex.join(command)}")
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
