#!/usr/bin/env bash
# Reproduce the isolated milk P14/P15 pose-repair experiment.
# This script never writes formal P14/P15/D19 paths; all outputs live below
# experiments/pose_repair_20260904.
set -euo pipefail

ROOT="/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/milk_depth_order_corrected_fullrun_20260903T101508Z"
CASE="P0014_84ea2dcc_carton_milk_f2370_2519"
R="$ROOT/run/$CASE"
E="$R/experiments/pose_repair_20260904"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python"
P3D_PY="/mnt/user-home/kupingxin/GHOST/.venv/bin/python"
GPU="${GPU:-5}"
OBJ="carton_milk"
ANN="$R/measurements/object_geometry/visible_geometry/$OBJ/annotations_v19_visible_geometry.json"
P14_FORMAL="$R/experiments/sam3d_trellis_controlled/P14_observed_pose_fit/v18_compact_rigid_object_pose_fit_report.json"
P15_FORMAL="$R/experiments/sam3d_trellis_controlled/P15_observed_pose_graph/v19_rigid_object_pose_graph_report.json"
COMP="$R/experiments/sam3d_trellis_controlled/P14_observed_completion/observed_only_completion_report.json"
MESH="$R/experiments/sam3d_trellis_controlled/P13_sam3d_dual/collision_eligible_observed_surface.ply"
HAND="$R/state/base_annotations/v19_mano_bridge_from_hawor_world.npz"
MANO="/mnt/user-home/kupingxin/ego_annotation_runtime/milk_unidepth_sam3d_ghost_lite_bundle_20260903T035100Z/third_party/WiLoR/mano_data/MANO_RIGHT.pkl"
RGB_NPZ="$E/rgb_pnp_edge_factors/rgb_pnp_edges.npz"

mkdir -p "$E/logs" "$E/rgb_pnp_edge_factors"
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

log "experiment worktree=$WT head=$($PY -c 'import subprocess,sys; print(subprocess.check_output(["git","-C",sys.argv[1],"rev-parse","HEAD"],text=True).strip())' "$WT") gpu=$GPU"

if [[ ! -s "$RGB_NPZ" ]]; then
  log "build all accepted adjacent RGB/PnP edges"
  "$PY" "$WT/scripts/build_v19_rgb_pnp_edge_factors.py" \
    --annotations "$ANN" --pose-report "$P14_FORMAL" --object-id "$OBJ" \
    --p14-script "$WT/scripts/fit_v18_compact_rigid_object_pose.py" \
    --output-npz "$RGB_NPZ" --output-json "$E/rgb_pnp_edge_factors/rgb_pnp_edges.json" \
    --frame-start 0 --frame-end 149 > "$E/logs/A_rgb_pnp_edges.log" 2>&1
fi

P14_GLOBAL="$E/p14_global_final"
if [[ ! -s "$P14_GLOBAL/v18_compact_rigid_object_pose_fit_report.json" ]]; then
  log "run global observed-only P14"
  "$PY" "$WT/scripts/fit_v19_global_observed_pose_graph.py" \
    --annotations "$ANN" --pose-report "$P14_FORMAL" \
    --pose-mesh "$MESH" --rgb-edge-npz "$RGB_NPZ" --object-id "$OBJ" \
    --output-dir "$P14_GLOBAL" --frame-start 0 --frame-end 149 --anchor-frame 0 \
    --loop-stride 1000 --max-observed-points 600 --max-anchor-points 800 \
    --max-anchor-pairs 160 --max-edge-pairs 100 --max-mesh-points 2500 --max-nfev 60 \
    > "$E/logs/B_p14_global_final.log" 2>&1
fi

build_factors() {
  local pose="$1" out="$2" log_path="$3"
  if [[ -s "$out/first_hit_silhouette_factors.npz" ]]; then return; fi
  mkdir -p "$out"
  log "build strict image factors: $out"
  CUDA_VISIBLE_DEVICES="$GPU" "$P3D_PY" \
    "$WT/experiments/sam3d_native_ghost_lite_20260826/build_p15_first_hit_silhouette_factors.py" \
    --annotations "$ANN" --pose-report "$pose" --factor-pose-report "$pose" \
    --completed-mesh "$MESH" --hand-npz "$HAND" --mano-faces-pkl "$MANO" \
    --object-id "$OBJ" --output-npz "$out/first_hit_silhouette_factors.npz" \
    --output-json "$out/first_hit_silhouette_factors.json" --frame-start 0 --frame-end 149 \
    --device cuda:0 --raster-size 256 --render-batch-size 16 > "$log_path" 2>&1
}
build_factors "$P14_FORMAL" "$E/p15_image_factors_final_baseline" "$E/logs/A_strict_factors_final_baseline.log"
build_factors "$P14_GLOBAL/v18_compact_rigid_object_pose_fit_report.json" "$E/p15_image_factors_final_global" "$E/logs/A_strict_factors_final_global.log"

run_p15_image() {
  local pose="$1" factor="$2" out="$3" log_path="$4"
  if [[ -s "$out/v19_rigid_object_pose_graph_report.json" ]]; then return; fi
  mkdir -p "$out"
  log "run strict P15 image factors: $out"
  "$PY" "$WT/scripts/solve_v19_rigid_object_pose_graph.py" \
    --annotations "$ANN" --pose-report "$pose" --completion-report "$COMP" \
    --completed-mesh "$MESH" --object-id "$OBJ" --output-dir "$out" \
    --complete-full-timeline-rigid-pose --image-factor-npz "$factor" \
    --require-strict-image-factor-contract --image-first-hit-weight 1.0 \
    --image-silhouette-weight 1.0 --sigma-image-first-hit-m 0.008 \
    --sigma-image-silhouette-px 4.0 --max-nfev 80 > "$log_path" 2>&1
}
run_p15_image "$P14_FORMAL" "$E/p15_image_factors_final_baseline/first_hit_silhouette_factors.npz" "$E/p15_strict_baseline_image" "$E/logs/B_p15_final_baseline_image.log"
run_p15_image "$P14_GLOBAL/v18_compact_rigid_object_pose_fit_report.json" "$E/p15_image_factors_final_global/first_hit_silhouette_factors.npz" "$E/p15_final_global_image" "$E/logs/B_p15_final_global_image.log"

for label_pose in \
  "formal:$P15_FORMAL" \
  "strict_baseline:$E/p15_final_baseline_image/v19_rigid_object_pose_graph_report.json" \
  "global:$P14_GLOBAL/v19_global_observed_pose_graph_report.json" \
  "global_image:$E/p15_final_global_image/v19_rigid_object_pose_graph_report.json"; do
  label="${label_pose%%:*}"; pose="${label_pose#*:}"
  out="$E/eval_${label}_mask_iou.json"
  if [[ ! -s "$out" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$P3D_PY" "$WT/scripts/evaluate_pose_mask_iou.py" \
      --annotations "$ANN" --pose-report "$pose" --mesh "$MESH" --object-id "$OBJ" \
      --output-json "$out" --label "$label" --raster-size 256 --device cuda:0 --batch-size 8 \
      > "$E/logs/C_eval_${label}.log" 2>&1
  fi
done

"$PY" "$WT/scripts/summarize_pose_repair_ab.py" \
  --formal-p15 "$P15_FORMAL" \
  --strict-baseline-p15 "$E/p15_strict_baseline_image/v19_rigid_object_pose_graph_report.json" \
  --global-p14 "$P14_GLOBAL/v19_global_observed_pose_graph_report.json" \
  --final-p15 "$E/p15_final_global_image/v19_rigid_object_pose_graph_report.json" \
  --formal-eval "$E/eval_formal_mask_iou.json" \
  --strict-eval "$E/eval_strict_baseline_mask_iou.json" \
  --final-eval "$E/eval_global_image_mask_iou.json" \
  --factor-baseline "$E/p15_image_factors_final_baseline/first_hit_silhouette_factors.json" \
  --factor-global "$E/p15_image_factors_final_global/first_hit_silhouette_factors.json" \
  --code-worktree "$WT" --output-json "$E/P15_POSE_REPAIR_AB.json" --output-md "$E/P15_POSE_REPAIR_AB.md"
log "experiment complete; formal D19 untouched"
