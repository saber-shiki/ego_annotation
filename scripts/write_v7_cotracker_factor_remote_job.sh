#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_cotracker_factor_outputs}
REPO_DIR=${REPO_DIR:-$REMOTE_ROOT/repo}
ENV_DIR=${ENV_DIR:-$REMOTE_ROOT/cotracker_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_cotracker_factor_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$OUT_ROOT" "$REPO_DIR" "$GPU_LOCK_DIR"
if [[ ! -x "$ENV_PY" ]]; then
  python3 -m pip install --user virtualenv
  python3 -m virtualenv "$ENV_DIR"
  "$ENV_PY" -m pip install --upgrade pip setuptools wheel
  "$ENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
"$ENV_PY" -m pip install -q numpy scipy opencv-python-headless pillow trimesh tqdm imageio imageio-ffmpeg
"$ENV_PY" - <<'PY'
import cv2, numpy, scipy, torch, trimesh
print("cotracker_env", "torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("imports_ok", cv2.__version__, numpy.__version__, scipy.__version__, trimesh.__version__)
PY
date '+%Y-%m-%d %H:%M:%S setup complete' > "$OUT_ROOT/setup_complete.marker"
EOF
chmod +x "$OUT_ROOT/setup_cotracker_factor_v7.sh"

cat > "$OUT_ROOT/run_cotracker_factor_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?GPU_ID is required}"
cd "$REPO_DIR"
if [[ ! -f "$OUT_ROOT/setup_complete.marker" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_cotracker_factor_v7.sh"
fi
mkdir -p "$OUT_ROOT"

run_case() {
  local name="\$1"
  local manifest="\$2"
  local annotations="\$3"
  local depth_npz="\$4"
  local mesh_archive="\$5"
  local frame_start="\$6"
  local frame_end="\$7"
  local query_index="\$8"
  local stills="\$9"
  local case_root="$OUT_ROOT/\$name"
  mkdir -p "\$case_root"
  "$ENV_PY" scripts/run_cotracker_object_tracks_v5.py \\
    --manifest "\$manifest" \\
    --annotations "\$annotations" \\
    --metric-depth-npz "\$depth_npz" \\
    --frame-start "\$frame_start" \\
    --frame-end "\$frame_end" \\
    --query-frame-index "\$query_index" \\
    --grid-step-px 24 \\
    --max-points 384 \\
    --output-fps 6 \\
    --still-frames \$stills \\
    --torchhub-repo facebookresearch/co-tracker \\
    --torchhub-model cotracker3_offline \\
    --torchhub-source github \\
    --backward-tracking \\
    --require-cuda \\
    --output-dir "\$case_root/tracks"
  "$ENV_PY" scripts/build_cotracker_sparse_correspondence_edges_v5.py \\
    --cotracker-npz "\$case_root/tracks/cotracker_object_tracks_v5.npz" \\
    --mesh-archive "\$mesh_archive" \\
    --output-json "\$case_root/sparse_edges/cotracker_sparse_correspondence_edges_v6.json" \\
    --min-track-frames 4 \\
    --max-surface-distance-m 0.004 \\
    --max-world-step-m 0.040 \\
    --max-frame-gap 1
  "$ENV_PY" scripts/fit_cotracker_pairwise_rigid_factors_v6.py \\
    --cotracker-npz "\$case_root/tracks/cotracker_object_tracks_v5.npz" \\
    --sparse-edges-json "\$case_root/sparse_edges/cotracker_sparse_correspondence_edges_v6.json" \\
    --output-json "\$case_root/pair_factors/qc_cotracker_pairwise_rigid_factors_v6.json" \\
    --min-pair-tracks 12 \\
    --min-inlier-tracks 12 \\
    --huber-delta-m 0.010 \\
    --max-inlier-residual-m 0.012 \\
    --accept-inlier-p95-m 0.010
}

run_case \\
  trash_865_870 \\
  /data2/ego_annotation_outputs/representative_trash/v3_samwise_pink_lid_bundlesdf_dataset_858_880/manifest.json \\
  /data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/annotations_side_metric_refit.json \\
  /data2/ego_annotation_outputs/representative_trash/v3_unidepth_metric_source_858_880/unidepth_metric_depth_v3.npz \\
  /data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_perframe_thick001_865_870/solidified_sheet_object_meshes_world.npz \\
  865 870 3 "865 868 870"

run_case \\
  mop_759_765 \\
  /data2/ego_annotation_outputs/representative_mop/v3_mop_depth_manifold_dataset_735_765/manifest.json \\
  /data2/ego_annotation_outputs/representative_mop/v3_vggt_object_skeleton_735_765/annotations_v3_vggt_object_skeleton.json \\
  /data2/ego_annotation_outputs/representative_mop/v3_unidepth_dense_735_765/unidepth_full_frame_depth_v3.npz \\
  /data2/ego_annotation_outputs/representative_mop/v7_mop_observed_surface_contract_unidepth_vggt_759_765/observed_mask_depth_meshes_world.npz \\
  759 765 3 "759 762 765"

"$ENV_PY" - <<'PY'
import json
from pathlib import Path

root = Path("$OUT_ROOT")
cases = {}
for name in ("trash_865_870", "mop_759_765"):
    pair = root / name / "pair_factors" / "qc_cotracker_pairwise_rigid_factors_v6.json"
    track = root / name / "tracks" / "qc_cotracker_object_tracks_v5.json"
    sparse = root / name / "sparse_edges" / "cotracker_sparse_correspondence_edges_v6.json"
    payload = json.loads(pair.read_text())
    cases[name] = {
        "track_report": str(track),
        "sparse_edges_json": str(sparse),
        "pair_factors_json": str(pair),
        "pair_count": payload.get("pair_count"),
        "ready_pair_count": len(payload.get("rigid_factor_ready_pairs", [])),
        "ready_pair_inlier_residual_m": payload.get("ready_pair_inlier_residual_m"),
    }
report = {"status": "ok", "method": "v7_cotracker_factor_remote_job", "cases": cases}
(root / "qc_v7_cotracker_factor_outputs.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
EOF
chmod +x "$OUT_ROOT/run_cotracker_factor_v7.sh"

cat > "$OUT_ROOT/wait_and_run_cotracker_factor_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_cotracker_factor_v7.sh"
MAX_USED_MB="\${MAX_USED_MB:-$MAX_USED_MB}"
POLL_SECONDS="\${POLL_SECONDS:-$POLL_SECONDS}"
GPU_SELECT_LOCK="\${GPU_SELECT_LOCK:-$GPU_SELECT_LOCK}"
GPU_LOCK_DIR="\${GPU_LOCK_DIR:-$GPU_LOCK_DIR}"
mkdir -p "\$GPU_LOCK_DIR"
while true; do
  GPU_ID=""
  exec 9>"\$GPU_SELECT_LOCK"
  flock -x 9
  while IFS=, read -r gpu_idx used_mb; do
    gpu_idx="\${gpu_idx//[[:space:]]/}"
    used_mb="\${used_mb//[[:space:]]/}"
    if [[ -n "\$gpu_idx" && -n "\$used_mb" && "\$used_mb" -le "\$MAX_USED_MB" ]]; then
      exec 8>"\$GPU_LOCK_DIR/gpu_\${gpu_idx}.lock"
      if flock -n 8; then
        GPU_ID="\$gpu_idx"
        break
      fi
      exec 8>&-
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [[ -n "\$GPU_ID" ]]; then
    export GPU_ID
    flock -u 9
    exec 9>&-
    date '+%Y-%m-%d %H:%M:%S selected GPU '"\$GPU_ID"
    exec bash "\$RUN_SCRIPT"
  fi
  flock -u 9
  exec 9>&-
  date '+%Y-%m-%d %H:%M:%S no GPU below memory threshold; sleeping'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep "\$POLL_SECONDS"
done
EOF
chmod +x "$OUT_ROOT/wait_and_run_cotracker_factor_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_cotracker_factor_v7.sh" \
  "$OUT_ROOT/run_cotracker_factor_v7.sh" \
  "$OUT_ROOT/wait_and_run_cotracker_factor_v7.sh"
