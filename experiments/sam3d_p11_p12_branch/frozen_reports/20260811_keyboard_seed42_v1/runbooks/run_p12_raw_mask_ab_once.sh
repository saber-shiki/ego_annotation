#!/usr/bin/env bash
set -uo pipefail
REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1
OLD=$RUN/experiments/sam3d_vs_trellis_20260803/sam3d_raw/keyboard_anchor/qc_sam3d_objects_mesh_v7.json
NEW=$EXP/p12_parallel_priors/p12_parallel_geometry_priors_report.json
OUT=$EXP/p12_raw_mask_ab

set +e
"$REPO/.venv/bin/python" \
  "$REPO/experiments/sam3d_p11_p12_branch/render_p12_raw_sam_mask_ab.py" \
  --old-sam-report "$OLD" \
  --new-p12-report "$NEW" \
  --output-dir "$OUT" \
  --target-faces 60000 \
  --surface-samples 60000 \
  --profile-bins 24 \
  --panel-size 560 \
  --seed 42 \
  > "$EXP/p12_raw_mask_ab_stdout.txt" \
  2> "$EXP/p12_raw_mask_ab_stderr.txt"
code=$?
set -e
printf '%s\n' "$code" > "$EXP/p12_raw_mask_ab_exit_code.txt"
if [ "$code" -eq 0 ]; then
  touch "$EXP/P12_RAW_MASK_AB_DONE"
else
  touch "$EXP/P12_RAW_MASK_AB_FAILED"
  tail -80 "$EXP/p12_raw_mask_ab_stderr.txt"
fi
exit "$code"
