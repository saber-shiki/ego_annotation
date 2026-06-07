#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_hunyuan_prior_outputs}
ENV_PY=${ENV_PY:-$REMOTE_ROOT/hunyuan3d_v3_env/bin/python}
REPO=${REPO:-$REMOTE_ROOT/Hunyuan3D-2}
RUNNER=${RUNNER:-$REMOTE_ROOT/remote_run_hunyuan3d_shape_v3.py}
GPU_ID=${GPU_ID:-0}

mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/run_hunyuan_v7_frame2539_2545.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$GPU_ID
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
printf '%s\n' "$OUT_ROOT/run_hunyuan_v7_frame2539_2545.sh"
