#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(st.st_mtime))}


def frame_count_key(output_key: str) -> str:
    mapping = {
        "overlay_video": "overlay",
        "world_video": "world",
        "side_by_side_video": "side_by_side",
        "video": "video",
    }
    return mapping.get(output_key, output_key.removesuffix("_video"))


def video_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    frame_counts = report.get("frame_counts", {}) if isinstance(report.get("frame_counts"), dict) else {}
    expected = report.get("frame_count")
    out = []
    for key, value in outputs.items():
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        actual = frame_counts.get(frame_count_key(key))
        out.append({**file_info(path), "role": key, "frame_count": actual, "expected_frame_count": expected, "frame_count_matches": actual == expected})
    return out


def case_bundle(case: str, root: Path) -> dict[str, Any]:
    graph = load_json(root / case / "v18_corrective_state_report.json")
    rigid = load_json(root / case / "rigid_se3_attempt" / "v18_rigid_se3_attempt_report.json")
    visible = load_json(root / case / "visible_surface_state" / "v18_visible_surface_state_report.json")
    hawor = load_json(root / case / "hawor_ghost_attempt" / "v18_hawor_ghost_attempt_report.json")
    owner = load_json(root / case / "occlusion_owner_best_effort" / "v18_occlusion_owner_best_effort_report.json")
    contact = load_json(root / case / "contact_nonpenetration_state" / "v18_contact_nonpenetration_state_report.json")
    montage = load_json(root / case / "corrective_montage" / "v18_corrective_montage_report.json")
    ann_path = root / case / "annotations_v18_corrective_state.json"
    ann = load_json(ann_path)
    review_sheets = sorted((root / "review_sheets").glob(f"{case}_*_corrective_review.jpg"))
    reports = {
        "graph_corrective_render": graph,
        "generic_rigid_se3_attempt": rigid,
        "frame_local_visible_surface": visible,
        "hawor_ghost_or_failure": hawor,
        "tentative_occlusion_owner": owner,
        "contact_nonpenetration": contact,
        "corrective_montage": montage,
    }
    videos: list[dict[str, Any]] = []
    for name, report in reports.items():
        for entry in video_entries(report):
            entry["mechanism"] = name
            videos.append(entry)
    return {
        "case": case,
        "annotation_state": {**file_info(ann_path), "frame_count": ann.get("frame_count"), "counts": ann.get("counts"), "claim_scope": ann.get("claim_scope")},
        "mechanisms": {
            "graph_corrective_render": {"draw_counts": graph.get("draw_counts"), "jitter_probe": graph.get("jitter_probe"), "claim_scope": graph.get("claim_scope")},
            "generic_rigid_se3_attempt": {"candidate_objects": rigid.get("candidate_objects"), "claim_scope": rigid.get("claim_scope")},
            "frame_local_visible_surface": {"candidate_objects": visible.get("candidate_objects"), "claim_scope": visible.get("claim_scope")},
            "hawor_ghost_or_failure": {"measurement_rows": hawor.get("measurement_rows"), "draw_counts": hawor.get("draw_counts"), "execution_failure_logs": hawor.get("execution_failure_logs"), "claim_scope": hawor.get("claim_scope")},
            "tentative_occlusion_owner": {"selected_tentative_owner_rows": owner.get("selected_tentative_owner_rows"), "strict_accepted_owner_rows": owner.get("strict_accepted_owner_rows"), "owner_object_counts": owner.get("owner_object_counts"), "acceptance_blocker_counts": owner.get("acceptance_blocker_counts"), "claim_scope": owner.get("claim_scope")},
            "contact_nonpenetration": {"contact_graph_selected_rows": contact.get("contact_graph_selected_rows"), "contact_graph_accepted_rows_before_nonpenetration_veto": contact.get("contact_graph_accepted_rows_before_nonpenetration_veto"), "signed_local_penetration_rows": contact.get("signed_local_penetration_rows"), "triangle_local_penetration_rows": contact.get("triangle_local_penetration_rows"), "mesh_watertight_rows": contact.get("mesh_watertight_rows"), "claim_scope": contact.get("claim_scope")},
            "corrective_montage": {"panels": montage.get("panels"), "claim_scope": montage.get("claim_scope")},
        },
        "videos": videos,
        "review_sheets": [file_info(p) for p in review_sheets],
        "all_listed_video_frame_counts_match": all(v.get("frame_count_matches") for v in videos),
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# V18 corrective bundle manifest",
        "",
        "This manifest indexes changed V18 corrective artifacts. It does not claim full V18 closure.",
        "",
        f"Root: `{manifest['output_root']}`",
        f"All listed video frame counts match: `{manifest['all_listed_video_frame_counts_match']}`",
        "",
    ]
    for case in manifest["cases"]:
        lines += [f"## {case['case']}", ""]
        ann = case["annotation_state"]
        lines += [f"Annotation state: `{ann['path']}`", f"Counts: `{ann.get('counts')}`", ""]
        owner = case["mechanisms"]["tentative_occlusion_owner"]
        hawor = case["mechanisms"]["hawor_ghost_or_failure"]
        contact = case["mechanisms"]["contact_nonpenetration"]
        lines += [
            f"Tentative owner rows: `{owner.get('selected_tentative_owner_rows')}`; strict accepted: `{owner.get('strict_accepted_owner_rows')}`",
            f"Contact rows selected: `{contact.get('contact_graph_selected_rows')}`; graph-accepted before local veto: `{contact.get('contact_graph_accepted_rows_before_nonpenetration_veto')}`; signed/triangle penetration rows: `{contact.get('signed_local_penetration_rows')}` / `{contact.get('triangle_local_penetration_rows')}`",
            f"HaWoR measurement rows: `{hawor.get('measurement_rows')}`",
            "",
            "Videos:",
        ]
        for video in case["videos"]:
            lines.append(f"- `{video['path']}` — {video['mechanism']} / {video['role']} / frames {video.get('frame_count')}/{video.get('expected_frame_count')}")
        lines += ["", "Review sheets:"]
        for sheet in case["review_sheets"]:
            lines.append(f"- `{sheet['path']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_bundle(case, args.output_root) for case in args.cases]
    manifest = {
        "method": "build_v18_corrective_bundle_manifest",
        "status": "corrective_bundle_index_not_full_v18_closure",
        "output_root": str(args.output_root),
        "cases": cases,
        "all_listed_video_frame_counts_match": all(case["all_listed_video_frame_counts_match"] for case in cases),
        "claim_scope": "indexes actual changed corrective V18 artifacts and failure evidence; not a readiness ledger or version-closure claim",
    }
    write_json(args.output_root / "v18_corrective_bundle_manifest.json", manifest)
    write_markdown(args.output_root / "V18_CORRECTIVE_BUNDLE.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
