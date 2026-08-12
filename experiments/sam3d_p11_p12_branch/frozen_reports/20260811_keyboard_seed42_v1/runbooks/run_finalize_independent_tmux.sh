#!/usr/bin/env bash
set -u -o pipefail

REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1
AB=$EXP/p14_p15_layered_geometry_ab_v1
COMPARE=$AB/full_video_three_branch_ab
FINAL=$AB/final_qc
LOG=$AB/finalize_independent_tmux_stdout.txt
SENTINEL=$AB/finalize_independent_tmux_exit.json
START=$(date +%s)

run_all() {
  set -e
  echo "[1/5] compose controlled three-branch full-video A/B"
  "$REPO/.venv/bin/python" \
    "$REPO/experiments/sam3d_p11_p12_branch/compose_p15_layered_geometry_ab.py" \
    --adapter-report "$AB/p14_p15_layered_render_state_adapter_report.json" \
    --manifests \
      "$AB/sam3d_owned_dual_mesh/full_video/p14_p15_layered_full_mano_render_manifest.json" \
      "$AB/sam3d_owned_legacy_cut/full_video/p14_p15_layered_full_mano_render_manifest.json" \
      "$AB/trellis_frozen_legacy_cut/full_video/p14_p15_layered_full_mano_render_manifest.json" \
    --output-dir "$COMPARE" \
    --review-source-frame 109

  echo "[2/5] ffprobe/decode all videos, verify geometry invariants, build QC/report"
  "$REPO/.venv/bin/python" \
    "$REPO/experiments/sam3d_p11_p12_branch/finalize_p15_layered_geometry_ab.py" \
    --experiment-root "$EXP" \
    --ab-root "$AB" \
    --comparison-report "$COMPARE/p15_three_branch_video_ab_report.json" \
    --output-dir "$FINAL" \
    --sam3d-repo /mnt/user-home/kupingxin/sam3d-objects/src

  echo "[3/5] CPU self-test"
  cd "$REPO"
  "$REPO/.venv/bin/python" experiments/sam3d_p11_p12_branch/self_test.py \
    > "$FINAL/self_test_stdout.txt" 2>&1

  echo "[4/5] compile and whitespace checks"
  "$REPO/.venv/bin/python" -m py_compile experiments/sam3d_p11_p12_branch/*.py
  git diff --check -- experiments/sam3d_p11_p12_branch
  git status --short -- experiments/sam3d_p11_p12_branch > "$FINAL/experiment_git_status.txt"
  git -C /mnt/user-home/kupingxin/sam3d-objects/src diff --check
  git -C /mnt/user-home/kupingxin/sam3d-objects/src status --short > "$FINAL/sam3d_upstream_git_status.txt"
  test ! -s "$FINAL/sam3d_upstream_git_status.txt"

  echo "[5/5] final artifact inventory"
  find "$AB" -maxdepth 4 -type f -printf '%P\t%s\n' | sort > "$FINAL/artifact_inventory.tsv"
  echo "FINALIZATION_DONE"
}

set +e
run_all > >(tee "$LOG") 2>&1
CODE=$?
set -e
END=$(date +%s)
"$REPO/.venv/bin/python" - "$SENTINEL" "$CODE" "$START" "$END" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
code, start, end = map(int, sys.argv[2:])
payload = {
    "status": "ok" if code == 0 else "failed",
    "exit_code": code,
    "started_unix_s": start,
    "ended_unix_s": end,
    "elapsed_s": end - start,
    "sentinel": "FINALIZATION_DONE" if code == 0 else "FINALIZATION_FAILED",
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
exit "$CODE"
