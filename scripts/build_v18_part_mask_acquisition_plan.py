#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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

STATUS = "v18_part_mask_acquisition_plan"
CLAIM = (
    "This artifact records what evidence is needed to acquire missing or improved part masks for V18 "
    "part/relative-motion objects, and whether the current local environment can run the available mask-generation "
    "scripts. It does not create masks, geometry, contact ownership, or pose."
)


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


def module_available(module: str, extra_paths: list[Path] | None = None) -> bool:
    added: list[str] = []
    for path in extra_paths or []:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
    try:
        return importlib.util.find_spec(module) is not None
    finally:
        for text in added:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def env_probe(args: argparse.Namespace) -> dict[str, Any]:
    cv2_available = module_available("cv2")
    torch_available = module_available("torch")
    cuda_available = False
    torch_version = None
    if torch_available:
        try:
            import torch  # type: ignore[import-not-found]

            torch_version = str(torch.__version__)
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - diagnostic only
            torch_version = f"import_error:{type(exc).__name__}:{exc}"
            cuda_available = False
    samwise_repo_candidates = [Path(raw) for raw in args.samwise_repo_candidates]
    samwise_checkpoint_candidates = [Path(raw) for raw in args.samwise_checkpoint_candidates]
    existing_samwise_repos = [str(path) for path in samwise_repo_candidates if path.exists()]
    existing_samwise_checkpoints = [str(path) for path in samwise_checkpoint_candidates if path.exists()]
    samwise_blockers: list[str] = []
    if not cv2_available:
        samwise_blockers.append("python_cv2_unavailable_for_existing_samwise_runner")
    if not cuda_available:
        samwise_blockers.append("cuda_unavailable_for_existing_samwise_runner")
    if not existing_samwise_repos:
        samwise_blockers.append("samwise_repo_not_found_in_known_paths")
    if not existing_samwise_checkpoints:
        samwise_blockers.append("samwise_checkpoint_not_found_in_known_paths")
    samwise_ready = not samwise_blockers

    sam2_repo_candidates = [Path(raw) for raw in args.sam2_repo_candidates]
    sam2_checkpoint_candidates = [Path(raw) for raw in args.sam2_checkpoint_candidates]
    existing_sam2_repos = [str(path) for path in sam2_repo_candidates if path.exists()]
    existing_sam2_checkpoints = [str(path) for path in sam2_checkpoint_candidates if path.exists()]
    sam2_import_available = module_available("sam2.build_sam", sam2_repo_candidates) and module_available("sam2.sam2_image_predictor", sam2_repo_candidates)
    segment_anything_available = module_available("segment_anything")
    sam_v1_checkpoint_candidates = [Path(raw) for raw in args.sam_v1_checkpoint_candidates]
    existing_sam_v1_checkpoints = [str(path) for path in sam_v1_checkpoint_candidates if path.exists()]
    groundingdino_available = module_available("groundingdino")
    transformers_available = module_available("transformers")
    ultralytics_available = module_available("ultralytics")
    promptable_sam2_ready = cuda_available and sam2_import_available and bool(existing_sam2_checkpoints)
    promptable_sam_v1_ready = cuda_available and segment_anything_available and bool(existing_sam_v1_checkpoints)
    promptable_segmentation_backend_available = promptable_sam2_ready or promptable_sam_v1_ready
    open_vocab_or_referring_prompt_backend_available = samwise_ready or groundingdino_available
    local_new_mask_generation_ready = samwise_ready or (promptable_segmentation_backend_available and open_vocab_or_referring_prompt_backend_available)

    blockers: list[str] = []
    blockers.extend(samwise_blockers)
    if promptable_segmentation_backend_available and not open_vocab_or_referring_prompt_backend_available:
        blockers.append("promptable_sam_backend_available_but_no_open_vocab_or_referring_part_prompt_backend")
    if not promptable_segmentation_backend_available and not samwise_ready:
        blockers.append("no_promptable_sam_backend_ready")
    return {
        "cv2_available": cv2_available,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "samwise_repo_candidates_checked": [str(path) for path in samwise_repo_candidates],
        "samwise_checkpoint_candidates_checked": [str(path) for path in samwise_checkpoint_candidates],
        "existing_samwise_repos": existing_samwise_repos,
        "existing_samwise_checkpoints": existing_samwise_checkpoints,
        "existing_samwise_runner_locally_ready": samwise_ready,
        "sam2_repo_candidates_checked": [str(path) for path in sam2_repo_candidates],
        "sam2_checkpoint_candidates_checked": [str(path) for path in sam2_checkpoint_candidates],
        "existing_sam2_repos": existing_sam2_repos,
        "existing_sam2_checkpoints": existing_sam2_checkpoints,
        "sam2_import_available": sam2_import_available,
        "promptable_sam2_ready": promptable_sam2_ready,
        "segment_anything_available": segment_anything_available,
        "sam_v1_checkpoint_candidates_checked": [str(path) for path in sam_v1_checkpoint_candidates],
        "existing_sam_v1_checkpoints": existing_sam_v1_checkpoints,
        "promptable_sam_v1_ready": promptable_sam_v1_ready,
        "groundingdino_available": groundingdino_available,
        "transformers_available": transformers_available,
        "ultralytics_available": ultralytics_available,
        "promptable_segmentation_backend_available": promptable_segmentation_backend_available,
        "open_vocab_or_referring_prompt_backend_available": open_vocab_or_referring_prompt_backend_available,
        "local_new_mask_generation_ready": local_new_mask_generation_ready,
        "local_generation_blockers": blockers,
    }


def acquisition_state(row: dict[str, Any]) -> tuple[str, list[str]]:
    state = str(row.get("part_object_blocker_state"))
    blockers = set(str(item) for item in row.get("blockers", []) if isinstance(row.get("blockers"), list))
    if state == "blocked_missing_part_mask_evidence":
        return "requires_new_model_produced_part_masks", sorted(blockers | {"no_accepted_part_mask_evidence"})
    if state == "partial_visible_subset_only_blocked_no_pose":
        return "requires_improved_sparse_part_masks_or_visible_subset_only_model", sorted(
            blockers | {"partial_visible_subset_not_full_part_model"}
        )
    return "requires_manual_triage", sorted(blockers | {"unclassified_part_object_blocker_state"})


def case_report(case: str, args: argparse.Namespace, env: dict[str, Any]) -> dict[str, Any]:
    blocker_path = args.part_object_blockers_root / case / "v18_part_object_blocker_manifest_report.json"
    blocker_report = require_dict(load_json(blocker_path), f"{case} blocker report")
    object_rows: list[dict[str, Any]] = []
    for raw_row in require_list(blocker_report.get("object_rows"), "blocker object rows"):
        row = require_dict(raw_row, "blocker row")
        state, blockers = acquisition_state(row)
        locally_runnable = bool(env.get("local_new_mask_generation_ready"))
        next_actions = list(row.get("required_next_evidence", [])) if isinstance(row.get("required_next_evidence"), list) else []
        if not locally_runnable:
            if env.get("promptable_segmentation_backend_available") is True:
                next_actions.append("provision referring/open-vocabulary part prompt backend or provide precomputed part tracks; promptable SAM assets are present")
            else:
                next_actions.append("provision runnable open-vocabulary/referring video segmentation backend or provide precomputed part tracks")
        object_rows.append(
            {
                "object_id": row.get("object_id"),
                "track_id": row.get("track_id"),
                "case": case,
                "source_blocker_state": row.get("part_object_blocker_state"),
                "part_mask_acquisition_state": state,
                "accepted_part_track_count": row.get("accepted_part_track_count"),
                "visible_subset_candidate_count": row.get("visible_subset_candidate_count"),
                "local_new_mask_generation_ready": locally_runnable,
                "acquisition_blockers": blockers + list(env.get("local_generation_blockers", [])),
                "required_next_actions": sorted(set(str(item) for item in next_actions)),
                "mask_evidence_created": False,
                "part_geometry_created": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_mask_acquisition_plan",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"part_object_blockers": str(blocker_path)},
        "environment": env,
        "object_count": len(object_rows),
        "local_new_mask_generation_ready_count": sum(1 for row in object_rows if row["local_new_mask_generation_ready"]),
        "mask_evidence_created_count": 0,
        "object_rows": object_rows,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_mask_acquisition_plan_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    env = env_probe(args)
    reports = [case_report(case, args, env) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_part_mask_acquisition_plan",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "environment": env,
        "object_count": sum(int(report["object_count"]) for report in reports),
        "local_new_mask_generation_ready_count": sum(int(report["local_new_mask_generation_ready_count"]) for report in reports),
        "mask_evidence_created_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_mask_acquisition_plan_report.json"),
                "object_count": report["object_count"],
                "local_new_mask_generation_ready_count": report["local_new_mask_generation_ready_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_mask_acquisition_plan_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-object-blockers-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument(
        "--samwise-repo-candidates",
        nargs="+",
        default=["/home/yiwen/SAMWISE", "/home/yiwen/samwise", "/data2/SAMWISE", "/data2/samwise"],
    )
    parser.add_argument(
        "--samwise-checkpoint-candidates",
        nargs="+",
        default=[
            "/data2/checkpoints/samwise.pth",
            "/data2/ego_annotation_models/samwise.pth",
            "/home/yiwen/models/samwise.pth",
            "/home/yiwen/checkpoints/samwise.pth",
        ],
    )
    parser.add_argument(
        "--sam2-repo-candidates",
        nargs="+",
        default=[
            "third_party/sam2",
            "/home/yiwen/ego_annotation/third_party/sam2",
            "/home/yiwen/sam2",
            "/data2/sam2",
        ],
    )
    parser.add_argument(
        "--sam2-checkpoint-candidates",
        nargs="+",
        default=[
            "/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt",
            "/data2/checkpoints/sam2.1_hiera_small.pt",
            "/home/yiwen/ego_annotation/checkpoints/sam2.1_hiera_small.pt",
        ],
    )
    parser.add_argument(
        "--sam-v1-checkpoint-candidates",
        nargs="+",
        default=[
            "/home/yiwen/ego_annotation/checkpoints/sam_vit_b_01ec64.pth",
            "/data2/checkpoints/sam_vit_b_01ec64.pth",
            "/data2/ego_annotation_outputs/checkpoints/sam_vit_b_01ec64.pth",
        ],
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
