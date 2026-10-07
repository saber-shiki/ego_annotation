from __future__ import annotations

"""Discover observation manifests and build a source-video grouped index."""

from hashlib import sha256
from pathlib import Path
from typing import Iterable
import argparse
import json

from data.bundle import SCHEMA, validate_manifest

INDEX_SCHEMA = "ego.annotation.episode_index.v1"


def build_index(roots: Iterable[str | Path], output: str | Path | None = None) -> dict:
    records = []
    seen_ids: set[str] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("preprocessing_only") is True:
                if root.is_file():
                    raise ValueError(
                        "preprocessing-only v2 bundle requires the explicit v2->v3_rl "
                        "training adapter; inspect with data.v2_rl_adapter.inspect_v2_manifest"
                    )
                continue
            if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
                continue
            validate_manifest(payload, path)
            bundle_id = str(payload["bundle_id"])
            if bundle_id in seen_ids:
                raise ValueError(f"duplicate bundle_id {bundle_id!r}")
            seen_ids.add(bundle_id)
            records.append({
                "bundle_id": bundle_id,
                "clip_id": str(payload["clip_id"]),
                "source_video_id": str(payload["source_video_id"]),
                "manifest": str(path),
                "frame_count": len(payload["frame_ids"]) if "frame_ids" in payload else int(payload["frame_range"][1] - payload["frame_range"][0] + 1),
                "manifest_sha256": sha256(path.read_bytes()).hexdigest(),
            })
    records.sort(key=lambda row: (row["source_video_id"], row["clip_id"], row["bundle_id"]))
    index = {"schema": INDEX_SCHEMA, "episodes": records}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="source-observation manifest files or directories")
    parser.add_argument("--output", type=Path, required=True, help="create-only episode index JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"episode index is create-only: {args.output}")
    result = build_index(args.roots, args.output)
    print(json.dumps({"output": str(args.output), "episode_count": len(result["episodes"])}, indent=2))
    return result


if __name__ == "__main__":
    main()
