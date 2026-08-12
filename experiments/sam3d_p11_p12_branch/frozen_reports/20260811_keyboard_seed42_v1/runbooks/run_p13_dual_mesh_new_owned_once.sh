#!/usr/bin/env bash
set -uo pipefail
REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1
CONTROLLED=$EXP/p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json
OUT=$EXP/p13_dual_mesh_new_owned

set +e
"$REPO/.venv/bin/python" \
  "$REPO/experiments/sam3d_p11_p12_branch/build_p13_dual_mesh_geometry_prior.py" \
  --controlled-report "$CONTROLLED" \
  --candidate sam3d_new_object_owned_mask \
  --output-dir "$OUT" \
  --target-faces 60000 \
  --panel-size 540 \
  > "$EXP/p13_dual_mesh_new_owned_stdout.txt" \
  2> "$EXP/p13_dual_mesh_new_owned_stderr.txt"
code=$?
set -e
printf '%s\n' "$code" > "$EXP/p13_dual_mesh_new_owned_exit_code.txt"
if [ "$code" -eq 0 ]; then
  touch "$EXP/P13_DUAL_MESH_NEW_OWNED_DONE"
else
  touch "$EXP/P13_DUAL_MESH_NEW_OWNED_FAILED"
  tail -100 "$EXP/p13_dual_mesh_new_owned_stderr.txt"
fi
exit "$code"
