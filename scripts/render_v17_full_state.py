#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata for {path}")
    return VideoInfo(fps=fps, width=width, height=height, frame_count=frame_count)


def check_video(path: Path, raw: VideoInfo) -> dict[str, Any]:
    info = video_info(path)
    return {
        "path": str(path),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "frame_count": info.frame_count,
        "raw_frame_count": raw.frame_count,
        "frame_count_match": info.frame_count == raw.frame_count,
    }


def run_command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=str(cwd), check=True)


def render_case(args: argparse.Namespace, case_manifest: Path, output_root: Path) -> dict[str, Any]:
    state = load_json(case_manifest)
    v16 = load_json(Path(state["v16_manifest"]))
    clip = Path(v16["clip"])
    raw = video_info(clip)
    if int(state["raw_frame_count"]) != raw.frame_count:
        raise RuntimeError(f"{case_manifest} raw_frame_count does not match clip metadata")
    case_dir = output_root / str(state["case"])
    render_dir = case_dir / "render_tmp"
    final_dir = case_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(args.python),
            "scripts/fuse_v1_full_fidelity.py",
            "--clip",
            str(clip),
            "--output-dir",
            str(render_dir),
            "--render-only-annotations",
            str(state["annotations"]),
            "--object-mesh-npz",
            str(state["object_mesh_archive"]),
            "--render-width",
            str(args.render_width),
        ],
        args.repo_root,
    )
    names = {
        "overlay": ("overlay_mano_object.mp4", "overlay_mano_object_multi.mp4"),
        "world": ("reconstruction_3d_world.mp4", "world_reconstruction_3d_v17.mp4"),
        "side_by_side": ("side_by_side.mp4", "side_by_side_v17.mp4"),
    }
    render_qc: dict[str, Any] = {}
    for key, (src_name, dst_name) in names.items():
        src = render_dir / src_name
        dst = final_dir / dst_name
        if not src.exists():
            raise RuntimeError(f"renderer did not produce {src}")
        shutil.copy2(src, dst)
        render_qc[key] = check_video(dst, raw)
    frame_count_match = all(row["frame_count_match"] for row in render_qc.values())
    report = {
        "case": state["case"],
        "status": "ok" if frame_count_match else "failed",
        "method": "render_v17_full_state",
        "clip": str(clip),
        "annotations": state["annotations"],
        "object_mesh_archive": state["object_mesh_archive"],
        "raw_video": raw.__dict__,
        "render_qc": render_qc,
        "frame_count_match": frame_count_match,
        "solver_status": state.get("solver_status"),
    }
    write_json(case_dir / "v17_render_manifest.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.repo_root = Path(args.repo_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = [render_case(args, manifest, args.output_root) for manifest in args.case_manifests]
    summary = {
        "status": "ok" if all(row["status"] == "ok" for row in reports) else "failed",
        "method": "render_v17_full_state",
        "cases": reports,
    }
    write_json(args.output_root / "v17_render_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_full_state"))
    parser.add_argument(
        "--case-manifests",
        type=Path,
        nargs="+",
        default=[
            Path("/data2/ego_annotation_outputs/v17_full_state/trash_1050/v17_full_state_manifest.json"),
            Path("/data2/ego_annotation_outputs/v17_full_state/task5_tomato_960/v17_full_state_manifest.json"),
        ],
    )
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
