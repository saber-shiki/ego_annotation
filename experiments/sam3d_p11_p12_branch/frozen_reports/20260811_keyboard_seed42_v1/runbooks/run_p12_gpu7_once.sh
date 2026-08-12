#!/usr/bin/env bash
set -o pipefail

REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1
GPU_ID=7

mkdir -p "$EXP"
exec > "$EXP/p12_job_console.log" 2>&1
printf '%s selected physical GPU %s for corrected-mask SAM3D P12\n' "$(date -Iseconds)" "$GPU_ID"
nvidia-smi --id="$GPU_ID" --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

source /mnt/user-home/kupingxin/sam3d-objects/activate.sh
set -u
SAM_PYTHON=$CONDA_PREFIX/bin/python

set +e
"$REPO/.venv/bin/python" \
  "$REPO/experiments/sam3d_p11_p12_branch/run_p12_parallel_geometry_priors.py" \
  --p11-report "$EXP/p11_dual_inputs/p11_dual_geometry_inputs_report.json" \
  --output-dir "$EXP/p12_parallel_priors" \
  --sam3d-python "$SAM_PYTHON" \
  --sam3d-runner "$REPO/scripts/remote_run_sam3d_objects_mesh_v7.py" \
  --sam3d-repo /mnt/user-home/kupingxin/sam3d-objects/src \
  --sam3d-config /mnt/user-home/kupingxin/sam3d-objects/src/checkpoints/modelscope/pipeline.yaml \
  --cuda-visible-device "$GPU_ID" \
  --trellis-report \
    "$RUN/measurements/geometry_completion/trellis_keyboard_seed42/qc_trellis_shape_v3.json" \
  --seed 42 \
  > "$EXP/p12_orchestrator_stdout.txt" \
  2> "$EXP/p12_orchestrator_stderr.txt"
code=$?
set -e

printf '%s\n' "$code" > "$EXP/p12_exit_code.txt"
printf '%s P12 exit code %s\n' "$(date -Iseconds)" "$code"
if [ "$code" -eq 0 ]; then
  touch "$EXP/P12_DONE"
else
  touch "$EXP/P12_FAILED"
  if [ -f "$EXP/p12_orchestrator_stderr.txt" ]; then
    tail -80 "$EXP/p12_orchestrator_stderr.txt"
  fi
fi
exit "$code"
