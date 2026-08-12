#!/usr/bin/env bash
set -uo pipefail
REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1
CONTROLLED=$EXP/p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json
OUT=$EXP/p13_controlled_render

set +e
"$REPO/.venv/bin/python" \
  "$REPO/experiments/sam3d_p11_p12_branch/render_p13_controlled_geometry_prior_ab.py" \
  --controlled-report "$CONTROLLED" \
  --output-dir "$OUT" \
  --target-faces 60000 \
  --panel-size 520 \
  > "$EXP/p13_controlled_render_stdout.txt" \
  2> "$EXP/p13_controlled_render_stderr.txt"
code=$?
set -e
printf '%s\n' "$code" > "$EXP/p13_controlled_render_exit_code.txt"
if [ "$code" -eq 0 ]; then
  touch "$EXP/P13_CONTROLLED_RENDER_DONE"
else
  touch "$EXP/P13_CONTROLLED_RENDER_FAILED"
  tail -100 "$EXP/p13_controlled_render_stderr.txt"
fi
exit "$code"
