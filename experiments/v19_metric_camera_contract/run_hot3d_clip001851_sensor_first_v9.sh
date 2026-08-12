#!/usr/bin/env bash
# Reproducible additive run for HOT3D clip-001851, scheme-A steps 1-2.
# It consumes camera calibration metadata only; no object/hand/pose/contact GT.
set -uo pipefail

WT=/mnt/user-home/kupingxin/ego_annotation_worktrees/v19_metric_camera_contract
PY=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP="$RUN/experiments/v19_metric_camera_contract_visible_geometry_v9"
SENSOR=/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_benchmarks/hot3d_clips/v19_inputs_pinhole/clip-001851/input/raw_frame_manifest/manifest_for_eval.json
SENSOR_SOURCE_VIDEO=/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_benchmarks/hot3d_clips/v19_inputs_pinhole/clip-001851/input/clip-001851_214_1_pinhole.mp4
PREDICTION_SOURCE_VIDEO=/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/hot3d_clip001851_keyboard_pinhole/input.mp4
MANIFEST="$RUN/input/raw_frame_manifest/manifest.json"
SOURCE_DEPTH="$RUN/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
OLD_BASE="$RUN/state/base_annotations"
OLD_VISIBLE="$RUN/measurements/object_geometry/visible_geometry/keyboard"
SENTINEL="$RUN/experiments/v19_metric_camera_contract_visible_geometry_v9_exit.json"
STARTED=$(date +%s)

if [[ -e "$EXP" || -e "$SENTINEL" ]]; then
  printf 'refusing non-fresh output: %s or %s\n' "$EXP" "$SENTINEL" >&2
  exit 73
fi
mkdir -p "$EXP"

finalize() {
  local rc=$?
  local ended status
  ended=$(date +%s)
  status=failed
  if [[ $rc -eq 0 ]]; then status=ok; fi
  "$PY" - "$SENTINEL" "$status" "$rc" "$STARTED" "$ended" "$EXP" <<'PY'
import json, sys
from pathlib import Path
path, status, rc, started, ended, output = sys.argv[1:]
payload = {
    "status": status,
    "exit_code": int(rc),
    "started_unix_s": int(started),
    "ended_unix_s": int(ended),
    "elapsed_s": int(ended) - int(started),
    "output_root": output,
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  printf 'METRIC_CAMERA_VISIBLE_GEOMETRY_V9_EXIT_%s\n' "$rc"
}
trap finalize EXIT
set -e
cd "$WT"

"$PY" scripts/resolve_v19_camera_contract.py \
  --case hot3d_clip001851_keyboard_pinhole_sensor_first_v9 \
  --raw-frame-manifest "$MANIFEST" \
  --sensor-calibration-contract "$SENSOR" \
  --sensor-calibration-authority prediction_side_sensor_metadata \
  --sensor-frame-intrinsics-key hot3d_pinhole_fx_fy_cx_cy \
  --prediction-source-video "$PREDICTION_SOURCE_VIDEO" \
  --sensor-source-video "$SENSOR_SOURCE_VIDEO" \
  --fixed-intrinsics-tolerance-px 0.01 \
  --output-dir "$EXP/camera_contract_sensor_first" \
  --dataset-name HOT3D-Clips \
  --sensor-stream 214-1 \
  > "$EXP/camera_contract_sensor_first_stdout.json"

"$PY" scripts/resolve_v19_camera_contract.py \
  --case hot3d_clip001851_keyboard_pinhole_estimated_fallback_control_v9 \
  --raw-frame-manifest "$MANIFEST" \
  --unidepth-npz "$SOURCE_DEPTH" \
  --output-dir "$EXP/camera_contract_estimated_fallback_control" \
  --dataset-name HOT3D-Clips \
  --sensor-stream 214-1 \
  --fallback-reason 'controlled fallback path: sensor metadata intentionally omitted' \
  --aggregation median \
  --square-focal \
  > "$EXP/camera_contract_estimated_fallback_stdout.json"

CONTRACT="$EXP/camera_contract_sensor_first/v19_camera_calibration_contract.json"
BOUND_DEPTH="$EXP/depth_sensor_first/unidepth_full_frame_depth_camera_contract_v2.npz"
"$PY" scripts/adapt_v19_depth_to_camera_contract.py \
  --source-depth-npz "$SOURCE_DEPTH" \
  --camera-contract "$CONTRACT" \
  --depth-plane source_rgb \
  --output-dir "$EXP/depth_sensor_first" \
  > "$EXP/depth_sensor_first_stdout.json"

NEW_BASE="$EXP/base_annotations_sensor_first"
"$PY" scripts/build_v19_base_annotations.py \
  --case hot3d_clip001851_keyboard_pinhole_sensor_first_v9 \
  --raw-frame-manifest "$MANIFEST" \
  --output-dir "$NEW_BASE" \
  --depth-npz "$BOUND_DEPTH" \
  --calibration-contract "$CONTRACT" \
  --hawor-npz "$RUN/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --object-plan "$RUN/measurements/object_candidates/object_plan_agent.json" \
  --sam2-output-root "$RUN/measurements/object_tracks/sam2_owlv2_box_points" \
  > "$EXP/base_annotations_sensor_first_stdout.json"

# Strict shared-state check before visible geometry is allowed to run.
"$PY" - "$OLD_BASE" "$NEW_BASE" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
old, new = map(Path, sys.argv[1:])
a = json.loads((old / "annotations_v19_base.json").read_text())
b = json.loads((new / "annotations_v19_base.json").read_text())
if len(a["frames"]) != 150 or len(b["frames"]) != 150:
    raise SystemExit("base annotations must both contain 150 frames")
for left, right in zip(a["frames"], b["frames"], strict=True):
    if int(left["frame_idx"]) != int(right["frame_idx"]):
        raise SystemExit("base frame timeline changed")
    if not np.array_equal(
        np.asarray(left["camera"]["T_world_camera_metric"]),
        np.asarray(right["camera"]["T_world_camera_metric"]),
    ):
        raise SystemExit(f"camera pose changed at frame {left['frame_idx']}")
with np.load(old / "v19_mano_bridge_from_hawor_world.npz", allow_pickle=False) as x, np.load(
    new / "v19_mano_bridge_from_hawor_world.npz", allow_pickle=False
) as y:
    if x.files != y.files:
        raise SystemExit("MANO bridge keys changed")
    for key in x.files:
        if x[key].dtype.kind in "fc":
            equal = np.array_equal(x[key], y[key], equal_nan=True)
        else:
            equal = np.array_equal(x[key], y[key])
        if not equal:
            raise SystemExit(f"MANO bridge array changed: {key}")
print("BASE_CAMERA_POSES_AND_FULL_MANO_BRIDGE_EXACT")
PY

NEW_VISIBLE="$EXP/visible_geometry_sensor_first_full"
"$PY" scripts/build_v19_visible_geometry_from_sam2_depth.py \
  --case hot3d_clip001851_keyboard_pinhole_sensor_first_v9 \
  --track-id keyboard \
  --object-id keyboard \
  --raw-frame-manifest "$MANIFEST" \
  --sam2-track-json "$RUN/measurements/object_tracks/sam2_owlv2_box_points/keyboard/sam2/sam2_track.json" \
  --depth-npz "$BOUND_DEPTH" \
  --output-dir "$NEW_VISIBLE" \
  --base-annotations "$NEW_BASE/annotations_v19_base.json" \
  --calibration-contract "$CONTRACT" \
  --depth-image-plane source_rgb \
  --mask-image-plane sam2_mask \
  --pixel-center-convention integer_pixel_centers_opencv \
  --object-plan "$RUN/measurements/object_candidates/object_plan_agent.json" \
  --frame-start 0 \
  --frame-end 149 \
  --anchor-frame 109 \
  --require-anchor-frame \
  > "$EXP/visible_geometry_sensor_first_full_stdout.json"

AB="$EXP/visible_geometry_camera_contract_ab"
"$PY" experiments/v19_metric_camera_contract/render_visible_geometry_camera_contract_ab.py \
  --old-annotations "$OLD_VISIBLE/annotations_v19_visible_geometry.json" \
  --new-annotations "$NEW_VISIBLE/annotations_v19_visible_geometry.json" \
  --raw-frame-manifest "$MANIFEST" \
  --camera-contract "$CONTRACT" \
  --object-id keyboard \
  --output-dir "$AB" \
  --review-frames 0 50 109 149 \
  > "$EXP/visible_geometry_camera_contract_ab_stdout.json"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate \
  -of json "$AB/old_estimated_k_vs_sensor_k_visible_geometry_reprojection.mp4" \
  > "$AB/video_ffprobe.json"
ffmpeg -v error -i "$AB/old_estimated_k_vs_sensor_k_visible_geometry_reprojection.mp4" -f null -

"$PY" - "$RUN" "$EXP" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
run, exp = map(Path, sys.argv[1:])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

sensor_path = exp / "camera_contract_sensor_first/v19_camera_calibration_contract.json"
fallback_path = exp / "camera_contract_estimated_fallback_control/v19_camera_calibration_contract.json"
depth_report_path = exp / "depth_sensor_first/v19_depth_camera_contract_adapter_report.json"
base_report_path = exp / "base_annotations_sensor_first/v19_base_annotations_report.json"
visible_report_path = exp / "visible_geometry_sensor_first_full/v19_visible_geometry_adapter_report.json"
ab_report_path = exp / "visible_geometry_camera_contract_ab/camera_contract_visible_geometry_ab_report.json"
sensor = load(sensor_path)
fallback = load(fallback_path)
depth = load(depth_report_path)
base = load(base_report_path)
visible = load(visible_report_path)
ab = load(ab_report_path)
old_visible = load(run / "measurements/object_geometry/visible_geometry/keyboard/v19_visible_geometry_adapter_report.json")
old_by = {int(row["frame_idx"]): row for row in old_visible["rows"]}
new_by = {int(row["frame_idx"]): row for row in visible["rows"]}
ids = sorted(set(old_by) & set(new_by))
centroid_delta = np.asarray([
    np.linalg.norm(np.asarray(new_by[idx]["centroid_world_m"]) - np.asarray(old_by[idx]["centroid_world_m"]))
    for idx in ids
])
with np.load(
    run / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz", allow_pickle=False
) as old_bridge, np.load(
    exp / "base_annotations_sensor_first/v19_mano_bridge_from_hawor_world.npz", allow_pickle=False
) as new_bridge:
    bridge_equal = bool(
        old_bridge.files == new_bridge.files
        and all(
            np.array_equal(old_bridge[key], new_bridge[key], equal_nan=True)
            if old_bridge[key].dtype.kind in "fc"
            else np.array_equal(old_bridge[key], new_bridge[key])
            for key in old_bridge.files
        )
    )
old_base = load(run / "state/base_annotations/annotations_v19_base.json")
new_base = load(exp / "base_annotations_sensor_first/annotations_v19_base.json")
pose_equal_count = sum(
    np.array_equal(
        np.asarray(left["camera"]["T_world_camera_metric"]),
        np.asarray(right["camera"]["T_world_camera_metric"]),
    )
    for left, right in zip(old_base["frames"], new_base["frames"], strict=True)
)
frame_camera_sensor_k_count = sum(
    np.allclose(np.asarray(frame["camera"]["intrinsics_fx_fy_cx_cy"]), sensor["source_plane_intrinsics_fx_fy_cx_cy"], atol=1e-6, rtol=0)
    for frame in new_base["frames"]
)
visible_annotations = load(exp / "visible_geometry_sensor_first_full/annotations_v19_visible_geometry.json")
visible_frame_camera_sensor_k_count = sum(
    np.allclose(np.asarray(frame["camera"]["intrinsics_fx_fy_cx_cy"]), sensor["source_plane_intrinsics_fx_fy_cx_cy"], atol=1e-6, rtol=0)
    for frame in visible_annotations["frames"]
)
visible_sensor_k_count = sum(
    row.get("intrinsics_source") == "camera_contract:prediction_side_sensor_calibration"
    for row in visible["rows"]
)
mask_depth_contract_count = sum(
    row.get("mask_depth_transform_contract", {}).get("camera_contract_consistent") is True
    for row in visible["rows"]
)
mask_depth_exact_sampler_count = sum(
    row.get("mask_depth_transform_contract", {}).get("interpolation") == "cv2.INTER_NEAREST_EXACT"
    for row in visible["rows"]
)
hand_alignment = base["camera_hand_contract_alignment"]
camera_alignment_warning_count = sum(
    frame.get("camera", {}).get("inherited_hand_camera_contract_alignment", {}).get("active_contract_reinference_required") is True
    for frame in new_base["frames"]
)
hand_alignment_warning_count = sum(
    hand.get("metric_mano_state", {}).get("camera_contract_alignment", {}).get("active_contract_reinference_required") is True
    for frame in new_base["frames"]
    for hand in frame.get("hands", [])
)
camera_alignment_payload_match_count = sum(
    frame.get("camera", {}).get("inherited_hand_camera_contract_alignment") == hand_alignment
    for frame in new_base["frames"]
)
hand_alignment_payload_match_count = sum(
    hand.get("metric_mano_state", {}).get("camera_contract_alignment") == hand_alignment
    and hand.get("camera_contract_alignment") == hand_alignment
    for frame in new_base["frames"]
    for hand in frame.get("hands", [])
)
hand_uncertainty_warning = "hawor_camera_mano_intrinsics_differ_from_active_camera_contract_reinference_required"
hand_uncertainty_warning_count = sum(
    hand_uncertainty_warning in hand.get("uncertainty", [])
    for frame in new_base["frames"]
    for hand in frame.get("hands", [])
)
source_hawor_k = np.asarray(hand_alignment["source_hawor_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
active_camera_k = np.asarray(sensor["source_plane_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
hand_source_k_count = sum(
    np.allclose(
        np.asarray(hand.get("metric_mano_state", {}).get("current_v18_camera_intrinsics_fx_fy_cx_cy"), dtype=np.float64),
        source_hawor_k,
        atol=1e-6,
        rtol=0,
    )
    for frame in new_base["frames"]
    for hand in frame.get("hands", [])
)
hand_active_k_count = sum(
    np.allclose(
        np.asarray(hand.get("metric_mano_state", {}).get("v19_camera_intrinsics_fx_fy_cx_cy"), dtype=np.float64),
        active_camera_k,
        atol=1e-6,
        rtol=0,
    )
    for frame in new_base["frames"]
    for hand in frame.get("hands", [])
)
sensor_source_validation = sensor["provenance"]["source_timeline_validation"]
depth_contract_binding = visible["depth_camera_contract_binding_validation"]
summary = {
    "status": "ok",
    "method": "v19_metric_camera_contract_visible_geometry_scheme_a_steps_1_2",
    "claim_scope": (
        "prediction-side camera contract, camera-bound depth metadata, and visible metric geometry only; "
        "no object/hand/pose/contact evaluator GT consumed; no object trajectory or backend rerun"
    ),
    "experiment_version": "v9_final_source_video_bound_sensor_contract_depth_bound_exact_sampler_hawor_k_mismatch",
    "sensor_first": {
        "calibration_authority": sensor["calibration_authority"],
        "intrinsics_coordinate_plane": sensor["intrinsics_coordinate_plane"],
        "calibration_intrinsics_fx_fy_cx_cy": sensor["intrinsics_fx_fy_cx_cy"],
        "source_rgb_intrinsics_fx_fy_cx_cy": sensor["source_plane_intrinsics_fx_fy_cx_cy"],
        "fallback": sensor["fallback"],
        "evaluator_gt_consumed": sensor["provenance"]["evaluator_gt_consumed"],
        "source_timeline_validation": sensor_source_validation,
    },
    "estimated_fallback_control": {
        "calibration_authority": fallback["calibration_authority"],
        "intrinsics_fx_fy_cx_cy": fallback["intrinsics_fx_fy_cx_cy"],
        "fallback": fallback["fallback"],
    },
    "depth_array_invariants": depth["array_invariants"],
    "shared_state_invariants": {
        "camera_pose_exact_frame_count": int(pose_equal_count),
        "camera_pose_frame_count": len(new_base["frames"]),
        "full_mano_bridge_all_arrays_exact": bridge_equal,
        "base_frame_camera_sensor_k_count": int(frame_camera_sensor_k_count),
        "visible_frame_camera_sensor_k_count": int(visible_frame_camera_sensor_k_count),
        "inherited_hawor_camera_mano_alignment": hand_alignment,
        "camera_rows_with_reinference_warning": int(camera_alignment_warning_count),
        "hand_rows_with_reinference_warning": int(hand_alignment_warning_count),
        "camera_rows_with_exact_alignment_payload": int(camera_alignment_payload_match_count),
        "hand_rows_with_exact_alignment_payload": int(hand_alignment_payload_match_count),
        "hand_rows_with_uncertainty_warning": int(hand_uncertainty_warning_count),
        "hand_rows_preserving_source_hawor_k": int(hand_source_k_count),
        "hand_rows_recording_active_v19_k": int(hand_active_k_count),
    },
    "visible_geometry": {
        "frame_count": visible["visible_metric_frame_count"],
        "sensor_k_row_count": int(visible_sensor_k_count),
        "mask_depth_contract_consistent_row_count": int(mask_depth_contract_count),
        "mask_depth_exact_sampler_row_count": int(mask_depth_exact_sampler_count),
        "depth_camera_contract_binding_validation": depth_contract_binding,
        "anchor_frame_idx": visible["anchor_frame_idx"],
        "annotation_adapter_anchor_frame_idx": visible_annotations["v19_visible_geometry_adapter"].get("anchor_frame_idx"),
        "anchor_centroid_world_m": visible["anchor_centroid_world_m"],
        "anchor_extent_world_m": visible["anchor_extent_world_m"],
        "old_to_new_centroid_delta_m": {
            "median": float(np.median(centroid_delta)),
            "p95": float(np.percentile(centroid_delta, 95)),
            "max": float(np.max(centroid_delta)),
        },
    },
    "prediction_only_reprojection_review": ab["inside_owned_mask_fraction"],
    "all_object_owned_masks_byte_equal": ab["all_object_owned_masks_byte_equal"],
    "outputs": {
        "camera_contract": str(sensor_path),
        "fallback_contract": str(fallback_path),
        "depth_report": str(depth_report_path),
        "base_report": str(base_report_path),
        "visible_report": str(visible_report_path),
        "ab_report": str(ab_report_path),
        "ab_video": ab["outputs"]["video"],
        "multiframe_qc": ab["outputs"]["multiframe_qc"],
    },
    "immutable_legacy_artifacts": {
        str(run / "state/calibration/v19_camera_calibration_contract.json"): digest(run / "state/calibration/v19_camera_calibration_contract.json"),
        str(run / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz"): digest(run / "state/base_annotations/v19_mano_bridge_from_hawor_world.npz"),
        str(run / "measurements/object_geometry/visible_geometry/keyboard/v19_visible_geometry_adapter_report.json"): digest(run / "measurements/object_geometry/visible_geometry/keyboard/v19_visible_geometry_adapter_report.json"),
    },
}
checks = {
    "sensor_fallback_unused": sensor["fallback"]["used"] is False,
    "sensor_source_video_and_timeline_exact": (
        sensor_source_validation["status"] == "exact_source_video_hash_and_frame_time_timeline_match"
        and sensor_source_validation["prediction_source_video"]["sha256"]
        == sensor_source_validation["sensor_metadata_source_video"]["sha256"]
        == "7a9baf0553e5dcfb4411b6cfabbe3a734f815ee4c5b014bdd965b9e547ec2310"
        and sensor_source_validation["frame_count"] == 150
        and sensor_source_validation["time_s_rows_checked"] == 150
        and sensor_source_validation["time_s_max_abs_delta"] == 0.0
    ),
    "fallback_control_used": fallback["fallback"]["used"] is True,
    "depth_invariants_all": all(depth["array_invariants"].values()),
    "depth_archive_contract_binding_exact": (
        depth_contract_binding["status"] == "exact_camera_contract_hash_plane_intrinsics_and_affine_match"
        and depth_contract_binding["sha256"] == depth_contract_binding["supplied_contract_sha256"]
        and depth_contract_binding["plane"] == "source_rgb"
        and np.allclose(
            np.asarray(depth_contract_binding["A_depth_from_calibration"], dtype=np.float64),
            np.eye(3),
            atol=1e-9,
            rtol=0,
        )
    ),
    "camera_poses_exact": pose_equal_count == 150,
    "full_mano_bridge_exact": bridge_equal,
    "inherited_hawor_intrinsics_mismatch_explicit_without_silent_relabel": (
        hand_alignment["active_contract_reinference_required"] is True
        and hand_alignment["source_hawor_state_intrinsics_match_active_contract"] is False
        and hand_alignment["source_hawor_focal_matches_active_contract"] is False
        and hand_alignment["source_hawor_principal_point_matches_active_contract"] is False
        and hand_alignment["builder_reestimated_hawor_camera_or_mano"] is False
        and np.allclose(source_hawor_k, [537.0213012695312, 537.0213012695312, 704.0, 704.0], atol=1e-6, rtol=0)
        and np.allclose(active_camera_k, [609.85009765625, 609.85009765625, 707.4874877929688, 702.32177734375], atol=1e-6, rtol=0)
        and not np.allclose(source_hawor_k, active_camera_k, atol=1e-6, rtol=0)
        and camera_alignment_warning_count == 150
        and hand_alignment_warning_count == 300
        and camera_alignment_payload_match_count == 150
        and hand_alignment_payload_match_count == 300
        and hand_uncertainty_warning_count == 300
        and hand_source_k_count == 300
        and hand_active_k_count == 300
    ),
    "base_frame_camera_sensor_k_all": frame_camera_sensor_k_count == 150,
    "visible_frame_camera_sensor_k_all": visible_frame_camera_sensor_k_count == 150,
    "visible_sensor_k_all": visible_sensor_k_count == 150,
    "mask_depth_contract_all": mask_depth_contract_count == 150,
    "mask_depth_exact_sampler_all": mask_depth_exact_sampler_count == 150,
    "adapter_anchor_frame_preserved": visible_annotations["v19_visible_geometry_adapter"].get("anchor_frame_idx") == 109,
    "owned_masks_unchanged": ab["all_object_owned_masks_byte_equal"] is True,
    "new_reprojection_better_all_frames": ab["inside_owned_mask_fraction"]["new_better_frame_count"] == 150,
}
summary["required_checks"] = checks
if not all(checks.values()):
    raise SystemExit(f"final mechanism check failed: {checks}")
output = exp / "v19_metric_camera_visible_geometry_v9_summary.json"
output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": summary["status"], "required_checks": checks, "output": str(output)}, indent=2))
PY

echo METRIC_CAMERA_VISIBLE_GEOMETRY_V9_DONE
