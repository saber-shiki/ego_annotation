#!/usr/bin/env python3
"""Read-only MANO/video preflight for a local ego_annotation deployment.

The filesystem scan never modifies discovered assets. It writes only a new
preflight report/contact sheets under the requested output root. Rigidity is
left as a visual-review decision; filenames and category names are not
accepted as physical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MANO_RE = re.compile(r"^mano_(left|right)\.(pkl|npz|pth|pt)$", re.IGNORECASE)
VIDEO_RE = re.compile(r"\.((mp4)|(mov)|(avi)|(mkv))$", re.IGNORECASE)
ABS_VIDEO_RE = re.compile(r"(?<![\w])(/[^\s`'\")>]+\.(?:mp4|mov|avi|mkv))", re.IGNORECASE)
PRUNE_DIRS = {
    ".git", ".venv", ".runtime", "__pycache__", "node_modules", "site-packages",
    "frames", "world_frames", "overlay_frames", ".cache", "build", "dist",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, PermissionError):
        return None


def file_info(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    try:
        resolved = path.resolve()
        stat = path.stat()
    except (OSError, PermissionError):
        return {"path": str(path), "exists": False, "error": "stat_failed"}
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": True,
        "is_file": path.is_file(),
        "bytes": int(stat.st_size) if path.is_file() else None,
        "readable": os.access(path, os.R_OK),
        "mode": oct(stat.st_mode & 0o777),
        "owner_uid": int(stat.st_uid),
        "owner_gid": int(stat.st_gid),
        "sha256": sha256_file(path) if hash_file and path.is_file() else None,
    }


def referenced_mano_paths(repo: Path) -> set[Path]:
    found: set[Path] = set()
    for path in repo.rglob("*"):
        if not path.is_file() or any(part in PRUNE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in {".md", ".py", ".sh", ".json", ".toml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in re.finditer(r"(?:/[^\s`'\")]+|[^\s`'\")]+)MANO_(?:LEFT|RIGHT)\.pkl", text, re.IGNORECASE):
            raw = match.group(0).rstrip(".,:;)")
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = repo / raw
            found.add(candidate)
    return found


def search_mano_files(roots: list[Path], max_seconds: float, max_hits: int) -> tuple[list[Path], bool]:
    start = time.monotonic()
    hits: set[Path] = set()
    timed_out = False
    for root in roots:
        if not root.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
                for filename in filenames:
                    if MANO_RE.match(filename):
                        hits.add(Path(dirpath) / filename)
                        if len(hits) >= max_hits:
                            return sorted(hits), timed_out
                if time.monotonic() - start >= max_seconds:
                    timed_out = True
                    return sorted(hits), timed_out
        except (OSError, PermissionError):
            continue
    return sorted(hits), timed_out


def read_video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "exists": True, "opened": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "path": str(path),
        "exists": True,
        "opened": True,
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_s": float(frames / fps) if fps > 0 else None,
    }


def extract_contact_sheet(path: Path, output_dir: Path, label: str, sample_count: int = 8) -> dict[str, Any]:
    info = read_video_info(path)
    if not info.get("opened"):
        return {"status": "video_open_failed", "info": info}
    count = max(1, int(info["frame_count"]))
    indices = sorted(set(int(round(v)) for v in np.linspace(0, count - 1, sample_count)))
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(str(path))
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok or image is None:
                continue
            frames.append((index, image))
            cv2.imwrite(str(frame_dir / f"frame_{index:06d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    finally:
        cap.release()
    if not frames:
        return {"status": "no_frames_read", "info": info, "sample_indices": indices}

    tile_w = 480
    tile_h = max(2, int(round(frames[0][1].shape[0] * tile_w / frames[0][1].shape[1])))
    caption_h = 34
    cols = 4
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.zeros((rows * (tile_h + caption_h), cols * tile_w, 3), dtype=np.uint8)
    for slot, (index, image) in enumerate(frames):
        tile = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(slot, cols)
        y = row * (tile_h + caption_h)
        x = col * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
        cv2.rectangle(sheet, (x, y + tile_h), (x + tile_w, y + tile_h + caption_h), (0, 0, 0), -1)
        cv2.putText(sheet, f"{label}  frame={index}", (x + 8, y + tile_h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    sheet_path = output_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return {
        "status": "ok",
        "info": info,
        "sample_indices": indices,
        "frames_read": len(frames),
        "frame_dir": str(frame_dir),
        "contact_sheet": str(sheet_path),
    }


def repo_video_files(repo: Path) -> list[str]:
    result: list[str] = []
    for path in repo.rglob("*"):
        if not path.is_file() or any(part in PRUNE_DIRS for part in path.parts):
            continue
        if VIDEO_RE.search(path.name):
            result.append(str(path))
    return sorted(result)


def referenced_video_paths(repo: Path) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for path in repo.rglob("*"):
        if not path.is_file() or any(part in PRUNE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in {".md", ".py", ".sh", ".json", ".toml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for raw in ABS_VIDEO_RE.findall(text):
            paths.add(Path(raw.rstrip(".,:;")))
    return [file_info(path, hash_file=False) for path in sorted(paths)]


def maybe_network_fallback(output_root: Path, found: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if found.get("MANO_LEFT.pkl") and found.get("MANO_RIGHT.pkl"):
        return {"needed": False, "status": "skipped_assets_found", "downloads": []}
    left_url = os.environ.get("MANO_LEFT_URL")
    right_url = os.environ.get("MANO_RIGHT_URL")
    if not left_url or not right_url:
        return {
            "needed": True,
            "status": "blocked_no_explicit_licensed_source",
            "downloads": [],
            "note": "No complete local pair was found and no user-supplied official/download URL was configured. MANO files are license-controlled; no source-unknown mirror was guessed.",
        }
    # Deliberately require an explicit opt-in even when URLs are supplied.
    if os.environ.get("ALLOW_MANO_NETWORK_DOWNLOAD") != "1":
        return {"needed": True, "status": "explicit_opt_in_required", "downloads": [], "urls_present": True}
    import urllib.request
    target = output_root / "downloaded_mano"
    target.mkdir(parents=True, exist_ok=True)
    downloads = []
    for side, url in (("LEFT", left_url), ("RIGHT", right_url)):
        destination = target / f"MANO_{side}.pkl"
        urllib.request.urlretrieve(url, destination)
        downloads.append(file_info(destination, hash_file=True))
    return {"needed": True, "status": "downloaded_from_explicit_urls", "downloads": downloads}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_json(args.candidate_config)

    direct_refs = referenced_mano_paths(repo)
    roots = [Path(p).expanduser() for p in config.get("mano_search_roots", [])]
    roots = [p for p in roots if p.exists()]
    discovered, timed_out = search_mano_files(roots, float(args.max_search_seconds), int(args.max_hits))
    all_assets = set(discovered) | {p for p in direct_refs if p.exists()}
    by_side: dict[str, list[dict[str, Any]]] = {"MANO_LEFT.pkl": [], "MANO_RIGHT.pkl": []}
    for path in sorted(all_assets):
        normalized = path.name.casefold()
        side = "MANO_LEFT.pkl" if normalized == "mano_left.pkl" else "MANO_RIGHT.pkl" if normalized == "mano_right.pkl" else None
        if side is not None:
            by_side[side].append(file_info(path, hash_file=True))

    network = maybe_network_fallback(output_root, by_side)
    if network.get("downloads"):
        for item in network["downloads"]:
            side = Path(item["path"]).name
            by_side.setdefault(side, []).append(item)

    candidates = []
    for item in sorted(config.get("video_candidates", []), key=lambda x: int(x.get("priority", 999))):
        if not item.get("enabled", True):
            continue
        path = Path(str(item["video_path"])).expanduser()
        record = dict(item)
        record["video"] = read_video_info(path) if path.exists() else {"path": str(path), "exists": False}
        if path.exists() and record["video"].get("opened"):
            case_dir = output_root / "video_review" / str(item["case_id"])
            record["visual_review"] = extract_contact_sheet(path, case_dir, str(item["target_object"]))
            record["rigidity_status"] = "pending_parent_visual_review"
            record["rigidity_decision_rule"] = "Accept only after visual review confirms one rigid body and rejects deformation/articulation/identity leakage; do not infer from filename."
        else:
            record["rigidity_status"] = "blocked_video_missing_or_unreadable"
        candidates.append(record)

    report = {
        "status": "ok",
        "method": "read_only_mano_and_rigid_video_preflight",
        "claim_scope": "filesystem asset discovery plus visual-review preparation; no MANO asset mutation and no physical prediction run",
        "repo": str(repo),
        "repo_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "mano": {
            "search_roots": [str(p) for p in roots],
            "search_timed_out": timed_out,
            "direct_reference_paths": [str(p) for p in sorted(direct_refs)],
            "MANO_LEFT.pkl": by_side["MANO_LEFT.pkl"],
            "MANO_RIGHT.pkl": by_side["MANO_RIGHT.pkl"],
            "complete_readable_pair": bool(by_side["MANO_LEFT.pkl"] and by_side["MANO_RIGHT.pkl"]),
            "network_fallback": network,
        },
        "videos": {
            "files_inside_git_worktree": repo_video_files(repo),
            "absolute_video_references_in_repo": referenced_video_paths(repo),
            "candidates": candidates,
        },
        "elapsed_s": float(time.monotonic() - started),
    }
    (output_root / "preflight_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# MANO and rigid-video preflight",
        "",
        "- Scope: read-only discovery and visual-review preparation.",
        f"- Complete readable MANO pair: `{report['mano']['complete_readable_pair']}`",
        f"- MANO left hits: `{len(by_side['MANO_LEFT.pkl'])}`; right hits: `{len(by_side['MANO_RIGHT.pkl'])}`",
        f"- Videos inside git worktree: `{len(report['videos']['files_inside_git_worktree'])}`",
        "",
        "## MANO candidates",
    ]
    for side in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        for hit in by_side[side]:
            lines.append(f"- {side}: `{hit['path']}` readable=`{hit['readable']}` sha256=`{hit['sha256']}`")
    lines += ["", "## Video candidates and rigidity review"]
    for candidate in candidates:
        lines.append(f"- `{candidate['case_id']}` target=`{candidate['target_object']}` status=`{candidate['rigidity_status']}` video=`{candidate['video'].get('path')}`")
        if candidate.get("visual_review", {}).get("contact_sheet"):
            lines.append(f"  - contact sheet: `{candidate['visual_review']['contact_sheet']}`")
        lines.append("  - decision: parent must visually review the contact sheet before accepting rigid branch")
    (output_root / "PREFLIGHT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-search-seconds", type=float, default=90.0)
    parser.add_argument("--max-hits", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
