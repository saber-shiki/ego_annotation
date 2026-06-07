#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_hunyuan_prior_outputs}
ENV_PY=${ENV_PY:-$REMOTE_ROOT/hunyuan3d_v3_env/bin/python}
REPO=${REPO:-$REMOTE_ROOT/Hunyuan3D-2}
RUNNER=${RUNNER:-$REMOTE_ROOT/remote_run_hunyuan3d_shape_v3.py}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}

mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/run_hunyuan_v7_frame2539_2545.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
cd "$REMOTE_ROOT"
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --mode single \\
  --image "$REMOTE_ROOT/v7_sam3d_prior_inputs_frame2539/frame_002539_crop_rgba.png" \\
  --output-dir "$OUT_ROOT/frame2539_single" \\
  --model tencent/Hunyuan3D-2mini \\
  --subfolder hunyuan3d-dit-v2-mini-fast \\
  --steps 5 \\
  --octree-resolution 256 \\
  --num-chunks 12000 \\
  --seed 2539 \\
  --mesh-name mesh.glb
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --mode single \\
  --image "$REMOTE_ROOT/v7_sam3d_prior_inputs_frame2545/frame_002545_crop_rgba.png" \\
  --output-dir "$OUT_ROOT/frame2545_single" \\
  --model tencent/Hunyuan3D-2mini \\
  --subfolder hunyuan3d-dit-v2-mini-fast \\
  --steps 5 \\
  --octree-resolution 256 \\
  --num-chunks 12000 \\
  --seed 2545 \\
  --mesh-name mesh.glb
EOF
chmod +x "$OUT_ROOT/run_hunyuan_v7_frame2539_2545.sh"
cat > "$OUT_ROOT/wait_and_run_hunyuan_v7_frame2539_2545.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_hunyuan_v7_frame2539_2545.sh"
MAX_USED_MB="\${MAX_USED_MB:-$MAX_USED_MB}"
POLL_SECONDS="\${POLL_SECONDS:-$POLL_SECONDS}"
while true; do
  GPU_ID=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v max="\$MAX_USED_MB" '{gsub(/[[:space:]]/, "", \$1); gsub(/[[:space:]]/, "", \$2); if ((\$2 + 0) <= (max + 0)) {print \$1; exit}}')
  if [[ -n "\$GPU_ID" ]]; then
    export GPU_ID
    date '+%Y-%m-%d %H:%M:%S selected GPU '"\$GPU_ID"
    exec bash "\$RUN_SCRIPT"
  fi
  date '+%Y-%m-%d %H:%M:%S no GPU below memory threshold; sleeping'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep "\$POLL_SECONDS"
done
EOF
chmod +x "$OUT_ROOT/wait_and_run_hunyuan_v7_frame2539_2545.sh"
printf '%s\n%s\n' "$OUT_ROOT/run_hunyuan_v7_frame2539_2545.sh" "$OUT_ROOT/wait_and_run_hunyuan_v7_frame2539_2545.sh"
