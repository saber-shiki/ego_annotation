#!/usr/bin/env python3
"""CPU-only preflight for the next HOT3D R2 correspondence slice.

This script does not run SAM2, UniDepth, CoTracker, TRELLIS, or any GPU/model
inference. It verifies the local raw inputs for a HOT3D keyboard slice, derives
selection/scoring facts from the HOT3D GT sidecar, and writes a run contract for
an A800/server R2 correspondence-first object-pose experiment.

Boundary enforced by the generated contract:
- prediction/runtime inputs: raw pinhole frames/video, prediction-side masks,
  prediction-side metric depth, prediction-side camera annotations;
- evaluation-only inputs: HOT3D GT object/hand/camera sidecar and P37-style
  constant-transform scorer;
- forbidden runtime inputs: GT object pose, GT hand state, GT masks, GT contact
  labels, and evaluator targets.

The goal is to prevent the next run from becoming another container/status
artifact: the A800 run must produce correspondence tracks, rigidity windows,
correspondence-derived object pose rows, GT residuals after freeze, and a render
whose keyboard placement changes from the depth-ICP baseline if the mechanism is
live.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path("/data2/ego_annotation_outputs")
A800_ROOT = Path("/mnt/truenas-user-home/yiwen/ego_annotation_outputs")
HOT3D_ROOT = ROOT / "v19_benchmarks" / "hot3d_clips"
DEFAULT_TARGET_CLIP = "001851"
DEFAULT_CONTROL_CLIP = "001849"
DEFAULT_OBJECT_ID = "28"  # keyboard in these HOT3D clips
DEFAULT_STREAM_ID = "214-1"
SCHEMA = "hot3d_r2_correspondence_preflight/v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return data


def _summary(vals: list[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"count": 0}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        i = (len(xs) - 1) * p / 100.0
        lo = math.floor(i)
        hi = math.ceil(i)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - i) + xs[hi] * (i - lo)

    return {
        "count": len(xs),
        "min": xs[0],
        "p05": pct(5),
        "median": pct(50),
        "mean": sum(xs) / len(xs),
        "p95": pct(95),
        "max": xs[-1],
    }


def _rle_area(mask_obj: Any) -> int | None:
    if not isinstance(mask_obj, dict):
        return None
    rle = mask_obj.get("rle")
    if not isinstance(rle, list):
        return None
    # HOT3D sidecar stores start,length pairs.
    total = 0
    for i in range(1, len(rle), 2):
        try:
            total += int(rle[i])
        except Exception:
            return None
    return total


def _box_area(box: Any) -> float | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _rel_to_abs(manifest_path: Path, value: str | None) -> str | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return str(p)
    # v19 raw-frame manifests store paths relative to the clip root
    # (.../clip-001851), not relative to the manifest file itself.
    clip_root = manifest_path.parents[2]
    return str((clip_root / p).resolve())


def _to_a800_path(value: str | Path) -> str:
    """Map the local verification mount to the A800/truenas runtime mount."""
    s = str(value)
    local_prefix = str(ROOT)
    remote_prefix = str(A800_ROOT)
    if s == local_prefix:
        return remote_prefix
    if s.startswith(local_prefix + "/"):
        return remote_prefix + s[len(local_prefix) :]
    return s


def _ffprobe_video(path: Path) -> dict[str, Any]:
    if not path.exists() or shutil.which("ffprobe") is None:
        return {"exists": path.exists(), "ffprobe_available": shutil.which("ffprobe") is not None}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=20)
        data = json.loads(out.stdout)
        stream = (data.get("streams") or [{}])[0]
        return {"exists": True, "ffprobe_available": True, **stream}
    except Exception as exc:  # pragma: no cover - external tool failure path
        return {"exists": True, "ffprobe_available": True, "error": str(exc)}


def _clip_paths(clip_id: str) -> dict[str, Path]:
    clip_name = f"clip-{clip_id}"
    pinhole_root = HOT3D_ROOT / "v19_inputs_pinhole" / clip_name
    gt_root = HOT3D_ROOT / "v19_inputs" / clip_name
    return {
        "pinhole_root": pinhole_root,
        "raw_manifest": pinhole_root / "input" / "raw_frame_manifest" / "manifest.json",
        "raw_video": pinhole_root / "input" / f"{clip_name}_214_1_pinhole.mp4",
        "gt_sidecar": gt_root / "evaluation" / "hot3d_gt" / "hot3d_clip_gt_sidecar.json",
        "hawor_hands": pinhole_root / "measurements" / "hawor_world_pinhole_f610" / "hawor_world_hands.npz",
    }


def _object_row(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    objects = frame.get("json", {}).get("objects.json", {})
    rows = objects.get(str(object_id)) if isinstance(objects, dict) else None
    if isinstance(rows, list) and rows:
        return rows[0]
    if isinstance(rows, dict):
        return rows
    return None


def analyze_clip(clip_id: str, object_id: str, stream_id: str) -> dict[str, Any]:
    paths = _clip_paths(clip_id)
    raw_manifest = paths["raw_manifest"]
    gt_sidecar = paths["gt_sidecar"]
    if not raw_manifest.exists():
        raise FileNotFoundError(raw_manifest)
    if not gt_sidecar.exists():
        raise FileNotFoundError(gt_sidecar)

    manifest = _read_json(raw_manifest)
    sidecar = _read_json(gt_sidecar)
    frames = sidecar.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"sidecar has no frames: {gt_sidecar}")

    translations: list[list[float]] = []
    vis: list[float] = []
    box_areas: list[float] = []
    modal_areas: list[int] = []
    amodal_areas: list[int] = []
    candidates: list[dict[str, Any]] = []
    missing_object_frames: list[int] = []

    for fr in frames:
        frame_idx = int(fr["frame_idx"])
        row = _object_row(fr, object_id)
        if row is None:
            missing_object_frames.append(frame_idx)
            continue
        T = row.get("T_world_from_object", {})
        t = T.get("translation_xyz")
        if isinstance(t, list) and len(t) == 3:
            translations.append([float(v) for v in t])
        v = row.get("visibilities_modeled", {}).get(stream_id)
        if v is not None:
            vis.append(float(v))
        ba = _box_area(row.get("boxes_amodal", {}).get(stream_id))
        if ba is not None:
            box_areas.append(ba)
        modal = _rle_area(row.get("masks_modal", {}).get(stream_id))
        amodal = _rle_area(row.get("masks_amodal", {}).get(stream_id))
        if modal is not None:
            modal_areas.append(modal)
        if amodal is not None:
            amodal_areas.append(amodal)
        # Query-frame score uses GT visibility/mask area only for choosing and
        # documenting a research run. The generated run contract forbids passing
        # this mask/pose to prediction; the A800 run must use SAM2/prediction masks.
        score = float(v or 0.0) * math.sqrt(float(modal or amodal or ba or 0.0))
        candidates.append(
            {
                "frame_idx": frame_idx,
                "visibility_modeled": float(v) if v is not None else None,
                "modal_mask_area_px": modal,
                "amodal_mask_area_px": amodal,
                "amodal_box_area_px": ba,
                "query_score_visibility_sqrt_area": score,
            }
        )

    if not translations:
        raise RuntimeError(f"no object translations for object {object_id} in {gt_sidecar}")
    axes = list(zip(*translations))
    ranges_m = [max(a) - min(a) for a in axes]
    range_norm_m = math.sqrt(sum(r * r for r in ranges_m))
    step_norms = [
        math.sqrt(sum((b[j] - a[j]) ** 2 for j in range(3)))
        for a, b in zip(translations[:-1], translations[1:])
    ]
    candidates_sorted = sorted(candidates, key=lambda r: r["query_score_visibility_sqrt_area"], reverse=True)

    video_path = paths["raw_video"]
    if not video_path.exists():
        # Fall back to manifest video if naming changes.
        mv = manifest.get("video")
        if isinstance(mv, str):
            alt = Path(_rel_to_abs(raw_manifest, mv) or "")
            if alt.exists():
                video_path = alt

    # Find local v19 prediction roots without touching remote/truenas paths.
    local_runs = sorted(str(p) for p in (ROOT / "v19_runs").glob(f"*clip{clip_id}*"))
    tmp_mirrors = sorted(str(p) for p in Path("/tmp").glob(f"v19_clip{clip_id}_*"))

    return {
        "clip_id": clip_id,
        "object_id": object_id,
        "stream_id": stream_id,
        "paths": {
            "raw_manifest": str(raw_manifest),
            "raw_video": str(video_path),
            "gt_sidecar_evaluation_only": str(gt_sidecar),
            "hawor_hands_prediction_candidate": str(paths["hawor_hands"]),
        },
        "local_input_status": {
            "raw_manifest_exists": raw_manifest.exists(),
            "raw_video_exists": video_path.exists(),
            "gt_sidecar_exists": gt_sidecar.exists(),
            "hawor_hands_exists": paths["hawor_hands"].exists(),
            "frame_count_manifest": int(manifest.get("frame_count", len(manifest.get("frames", [])))),
            "frame_count_sidecar": len(frames),
            "width": int(manifest.get("width", 0) or 0),
            "height": int(manifest.get("height", 0) or 0),
            "fps": float(manifest.get("fps", 0.0) or 0.0),
            "video_probe": _ffprobe_video(video_path),
            "local_v19_prediction_roots": local_runs,
            "tmp_mirror_artifacts_count": len(tmp_mirrors),
            "tmp_mirror_artifacts_sample": tmp_mirrors[:20],
        },
        "gt_selection_metrics": {
            "object_translation_range_m_xyz": ranges_m,
            "object_translation_range_norm_m": range_norm_m,
            "object_translation_range_mm_xyz": [1000.0 * r for r in ranges_m],
            "object_translation_range_norm_mm": 1000.0 * range_norm_m,
            "object_step_norm_m": _summary(step_norms),
            "visibility_modeled": _summary(vis),
            "frames_visibility_gt_0p2": int(sum(v > 0.2 for v in vis)),
            "amodal_box_area_px": _summary(box_areas),
            "modal_mask_area_px": _summary([float(v) for v in modal_areas]),
            "amodal_mask_area_px": _summary([float(v) for v in amodal_areas]),
            "missing_object_frames": missing_object_frames,
            "query_frame_candidates_from_gt_for_run_planning_not_prediction": candidates_sorted[:8],
        },
    }


def _a800_contract(summary: dict[str, Any]) -> str:
    target = summary["target_clip"]
    target_paths = target["paths"]
    target_metrics = target["gt_selection_metrics"]
    q = target_metrics["query_frame_candidates_from_gt_for_run_planning_not_prediction"][0]
    qidx = int(q["frame_idx"])
    run_root = summary["recommended_run_root"]
    raw_manifest = _to_a800_path(target_paths["raw_manifest"])
    gt_sidecar = _to_a800_path(target_paths["gt_sidecar_evaluation_only"])
    return f"""# A800 R2 correspondence run contract — clip{target['clip_id']} keyboard

This file is generated by `scripts/prepare_hot3d_r2_correspondence_preflight.py`.
It is a contract for the next heavy run, not evidence that the run has happened.
Do not execute SAM2/UniDepth/CoTracker locally.

## Mechanism

Artifact defect: clip{target['clip_id']} has a depth-ICP keyboard trajectory whose
pose residual lives in depth/silhouette null-space directions. The keyboard moves
{target_metrics['object_translation_range_norm_mm']:.1f} mm over 150 frames and is
visible in {target_metrics['frames_visibility_gt_0p2']}/150 frames; temporal
surface correspondence can observe in-plane translation and normal-axis rotation
that depth-ICP/silhouette cannot.

Prediction variable: keyboard SE(3) trajectory over frames 0-149.
Evaluation variable: median GT residual after freezing the prediction trajectory.

## Boundary

Runtime/prediction may consume:
- raw pinhole frames/video: `{raw_manifest}`
- prediction-side SAM2/modal keyboard masks
- prediction-side metric depth NPZ
- prediction-side camera annotations (`vggt_source_intrinsics_fx_fy_cx_cy`,
  `T_world_camera_metric`) from the approved v19 sensor/depth stack
- optional prediction-side HaWoR hands for occlusion review, not GT hands

Runtime/prediction must not consume:
- HOT3D GT object pose, GT hands, GT masks, GT contact/proximity labels
- evaluator targets or P37/P46 residual summaries

Evaluation after prediction freeze may consume:
- `{gt_sidecar}`
- the P37-style constant-transform object-pose evaluator

## Required A800 artifacts before CoTracker

Write all outputs under:
`{run_root}`

Required prediction artifacts:
1. `prediction_masks/keyboard_sam2_manifest.json`
   - frames 0-149, dense, each row has `frame_idx`, absolute `rgb`, absolute
     `mask` for the predicted/modal keyboard mask.
   - GT masks are forbidden here.
2. `prediction_depth/unidepth_metric_depth.npz`
   - arrays `frame_idx` and `depth`, frames 0-149, metric meters.
3. `prediction_annotations/annotations_for_cotracker.json`
   - frames 0-149, each row has `frame_idx` and camera fields required by
     `run_cotracker_object_tracks_v5.py`.

GT-derived query-frame suggestion for planning/debug only: frame {qidx}
(visibility {q['visibility_modeled']}, modal area {q['modal_mask_area_px']}).
Do not let this suggestion substitute for prediction evidence. After SAM2 masks
exist, choose `QUERY_FRAME_INDEX` from the prediction-mask manifest (e.g. largest
clean predicted modal mask, visually reviewed) and record that prediction-side
selection evidence. `run_cotracker_object_tracks_v5.py` expects the dense window
index, not an arbitrary source frame id; because this contract uses frames 0-149,
the index equals `frame_idx` here.

## Commands after masks/depth/annotations exist

```bash
set -euo pipefail
cd /home/yiwen/ego_annotation
RUN_ROOT="{run_root}"
# Set this from prediction-mask evidence, not from GT. The preflight's GT-derived
# diagnostic suggestion is frame {qidx}; it is not a runtime input.
: "${{QUERY_FRAME_INDEX:?set QUERY_FRAME_INDEX from prediction-mask evidence}}"

python3 scripts/run_cotracker_object_tracks_v5.py \\
  --manifest "$RUN_ROOT/prediction_masks/keyboard_sam2_manifest.json" \\
  --annotations "$RUN_ROOT/prediction_annotations/annotations_for_cotracker.json" \\
  --metric-depth-npz "$RUN_ROOT/prediction_depth/unidepth_metric_depth.npz" \\
  --output-dir "$RUN_ROOT/r2_cotracker_keyboard" \\
  --frame-start 0 --frame-end 149 \\
  --query-frame-index "$QUERY_FRAME_INDEX" \\
  --grid-step-px 24 --max-points 384 \\
  --still-frames 0 30 75 95 106 120 140 149 \\
  --backward-tracking --require-cuda

python3 scripts/build_visible_surfel_compat_archive.py \\
  --annotations "$RUN_ROOT/prediction_annotations/annotations_for_cotracker.json" \\
  --object-id keyboard \\
  --output-npz "$RUN_ROOT/r3_visible_surfel_compat/visible_surfel_compat_archive.npz" \\
  --output-report "$RUN_ROOT/r3_visible_surfel_compat/visible_surfel_compat_archive_report.json" \\
  --required-frame-npz "$RUN_ROOT/r2_cotracker_keyboard/cotracker_object_tracks_v5.npz" \\
  --distance-threshold-m 0.004

python3 scripts/build_cotracker_sparse_correspondence_edges_v5.py \\
  --cotracker-npz "$RUN_ROOT/r2_cotracker_keyboard/cotracker_object_tracks_v5.npz" \\
  --visible-surfel-archive "$RUN_ROOT/r3_visible_surfel_compat/visible_surfel_compat_archive.npz" \\
  --output-json "$RUN_ROOT/r2_cotracker_keyboard/sparse_correspondence_edges_v5.json" \\
  --max-visible-sample-distance-m 0.004
```

Stop here and inspect `visible_surfel_compat_archive_report.json` plus
`sparse_correspondence_edges_v5.json` before running rigidity or pairwise pose
fitting. The compatibility archive is prediction-side visible surfel samples
only; its faces are loader padding and are not geometry evidence. If the sparse
edge report has trivial edge count, poor temporal continuity, or sample spacing
comparable to the 4 mm proximity threshold, do not run rigidity as pose support.
Do not fabricate mesh edges from GT.

## Acceptance evidence

The A800 run is useful only if it writes:
- `correspondence_tracks`: track lifetimes, mask/depth hit rates, held-out
  residuals, admissible/rejected reasons.
- `rigidity_windows`: Kabsch/pairwise-distance residuals, certified rigid/static
  windows, member tracks.
- `object_pose` candidate rows from correspondence with covariance/observability.
- GT residual report after prediction freeze, compared to the 38.85 mm P46
  depth-ICP baseline.
- overlay/world/side-by-side render whose keyboard body placement is driven by
  the correspondence pose rows.

Discriminating outcomes:
- residual drops materially below 38.85 mm: adopt correspondence pose as R2
  support and recompute near-contact/source-gap state.
- translation improves but rotation remains ambiguous: represent the ambiguous
  DOF as high covariance; do not tune factors to hide symmetry.
- tracks die under hand occlusion/contact frames: blocker is correspondence
  robustness; add re-detection/stitching or multi-view before any pose claim.
"""


def _causal_card(summary: dict[str, Any]) -> str:
    t = summary["target_clip"]
    c = summary["control_clip"]
    tm = t["gt_selection_metrics"]
    cm = c["gt_selection_metrics"]
    return f"""# clip{t['clip_id']} R2 correspondence preflight

## Decision

Primary slice: **clip{t['clip_id']} keyboard, R2 correspondence-first object pose**.
Control slice: clip{c['clip_id']} keyboard rigidity statistic.

clip{t['clip_id']} has {tm['object_translation_range_norm_mm']:.1f} mm GT keyboard
translation range and visibility >0.2 on {tm['frames_visibility_gt_0p2']}/150
frames. clip{c['clip_id']} has {cm['object_translation_range_norm_mm']:.1f} mm range.
The moving clip is the artifact-changing slice because temporal correspondence can
change the pose trajectory and the near-contact/source-gap attribution; the control
clip mostly tests whether the rigidity statistic stays low on a weaker-motion case.

## Causal card

Artifact defect: existing depth-ICP/silhouette keyboard pose leaves a 38.85 mm
median residual on clip{t['clip_id']} and the near-contact state is unresolved.
The failed stationary and silhouette repairs show the residual is in pose directions
that local depth/silhouette do not observe.

Physical variable: keyboard SE(3) trajectory over frames 0-149, especially in-plane
translation and normal-axis rotation.

Live mechanisms:
- M1: temporal surface correspondence observes the null-space DOFs and lowers the
  GT residual.
- M2: keyboard gauge/symmetry leaves a rotation mode ambiguous; residual remains as
  calibrated covariance, not as a crisp pose.
- M3: hand occlusion kills tracks near the contact interval; the blocker becomes
  correspondence robustness or multi-view stitching.

Discriminating measurement: A800 CoTracker over prediction SAM2 keyboard masks,
metric-depth lift, sparse correspondence edges, Kabsch rigidity windows, and a
post-freeze GT residual report. GT is used only for scoring and slice selection,
not as a prediction input.

Prediction per outcome:
- M1: residual < 38.85 mm, high track survival on visible frames, pose render moves
  coherently with the keyboard.
- M2: translation residual improves but rotation residual remains structured around
  keyboard symmetry; covariance grows in that DOF.
- M3: accepted-track count collapses in occluded/near-contact frames; render remains
  unknown/diffuse there until re-detection or multi-view evidence exists.

Expected state/render change: `ego.hoi` gains correspondence/rigidity support rows
and a correspondence-derived object-pose candidate with covariance. The world render
must place the keyboard from those rows. Contact remains a follow-on: it is recomputed
only after the pose is live and geometry face provenance is admissible.

## Generated files

- `preflight_summary.json` — machine-readable selection/input/run contract.
- `a800_r2_run_contract.md` — exact heavy-run boundary and commands after prediction
  masks/depth/annotations exist.
"""


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    target = analyze_clip(args.clip_id, args.object_id, args.stream_id)
    control = analyze_clip(args.control_clip_id, args.object_id, args.stream_id)
    out_root = args.output_root.resolve()
    run_root = args.recommended_run_root.resolve()
    summary = {
        "schema": SCHEMA,
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "decision": "clip001851_r2_correspondence_primary" if args.clip_id == "001851" else "hot3d_r2_correspondence_primary",
        "target_clip": target,
        "control_clip": control,
        "recommended_run_root": str(run_root),
        "a800_path_mapping": {
            "local_verified_prefix": str(ROOT),
            "a800_runtime_prefix": str(A800_ROOT),
            "target_raw_manifest_a800": _to_a800_path(target["paths"]["raw_manifest"]),
            "target_gt_sidecar_a800_evaluation_only": _to_a800_path(target["paths"]["gt_sidecar_evaluation_only"]),
        },
        "prediction_evaluation_boundary": {
            "runtime_allowed": [
                "raw pinhole frames/video",
                "prediction-side SAM2/modal object masks",
                "prediction-side metric depth NPZ",
                "prediction-side camera annotations from approved sensor/depth stack",
                "optional prediction-side HaWoR hands for occlusion review",
            ],
            "evaluation_only_after_prediction_freeze": [
                target["paths"]["gt_sidecar_evaluation_only"],
                control["paths"]["gt_sidecar_evaluation_only"],
                "P37/P46-style constant-transform object-pose evaluator",
            ],
            "forbidden_runtime_inputs": [
                "GT object pose",
                "GT hand pose/MANO",
                "GT modal/amodal masks",
                "GT proximity/contact labels",
                "evaluator residual targets",
            ],
        },
        "strict_blockers": [
            {
                "id": "prediction_masks_depth_missing_locally",
                "mechanism": "CoTracker requires prediction masks and metric depth; current bounded roots only prove raw video+GT are local.",
                "resolution": "produce SAM2 keyboard masks and UniDepth/depth archive on A800, or mount/pull prior prediction artifacts; do not substitute GT masks/depth as prediction input.",
            },
            {
                "id": "mesh_archive_for_sparse_edges",
                "mechanism": "sparse correspondence edges need a visible-surface mesh archive; without it, CoTracker tracks can be measured but not mapped to mesh vertices.",
                "resolution": "build R3 visible surface archive from prediction masks/depth or stop after track/rigidity evidence; do not fabricate mesh support from GT.",
            },
            {
                "id": "keyboard_cad_for_r3_chamfer_not_local",
                "mechanism": "R3 GT chamfer scoring needs HOT3D keyboard CAD; R2 pose scoring does not.",
                "resolution": "source CAD before R3 chamfer; keep R2 residual scoring on local GT object pose.",
            },
        ],
        "baseline_to_beat": {
            "clip001851_depth_icp_median_residual_mm_documented": 38.85,
            "clip001851_stationary_repair_worse_mm_documented": 55.75,
            "clip001849_depth_icp_residual_mm_documented": 77.1,
            "note": "documented v19 P37/P46 baselines; recompute with local evaluator after prediction freeze before final claim.",
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "preflight_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / "a800_r2_run_contract.md").write_text(_a800_contract(summary), encoding="utf-8")
    (out_root / "causal_card.md").write_text(_causal_card(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    date_tag = "20260706"
    default_out = ROOT / f"research_clip001851_r2_correspondence_{date_tag}" / "preflight"
    default_run = A800_ROOT / "v19_runs" / f"{date_tag}_hot3d_clip001851_r2_correspondence_a800_v1"
    ap = argparse.ArgumentParser(description="CPU-only HOT3D R2 correspondence preflight/run-spec generator")
    ap.add_argument("--clip-id", default=DEFAULT_TARGET_CLIP, help="target HOT3D clip suffix, e.g. 001851")
    ap.add_argument("--control-clip-id", default=DEFAULT_CONTROL_CLIP, help="control HOT3D clip suffix, e.g. 001849")
    ap.add_argument("--object-id", default=DEFAULT_OBJECT_ID, help="HOT3D/BOP object id; 28 is keyboard")
    ap.add_argument("--stream-id", default=DEFAULT_STREAM_ID, help="HOT3D camera stream id; 214-1 is pinhole input")
    ap.add_argument("--output-root", type=Path, default=default_out)
    ap.add_argument("--recommended-run-root", type=Path, default=default_run)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_preflight(args)
    t = summary["target_clip"]
    c = summary["control_clip"]
    out = Path(args.output_root).resolve()
    print(json.dumps({
        "decision": summary["decision"],
        "target_clip": t["clip_id"],
        "target_motion_mm": t["gt_selection_metrics"]["object_translation_range_norm_mm"],
        "target_visibility_frames_gt_0p2": t["gt_selection_metrics"]["frames_visibility_gt_0p2"],
        "control_clip": c["clip_id"],
        "control_motion_mm": c["gt_selection_metrics"]["object_translation_range_norm_mm"],
        "output_root": str(out),
        "a800_contract": str(out / "a800_r2_run_contract.md"),
        "prediction_boundary": summary["prediction_evaluation_boundary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
