#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${EGO_REPO_ROOT:-/mnt/user-home/kupingxin/ego_annotation}"
BUNDLE="${EGO_RUNTIME_BUNDLE:-/mnt/user-home/kupingxin/ego_annotation_runtime/v19_bundle_a800_0c8e6a9_local1}"
INPUT_VIDEO="${1:-${EGO_INPUT_VIDEO:-/mnt/truenas-user-home/kupingxin/ego_annotation_inputs/hot3d_clip001851_keyboard_pinhole/input.mp4}}"
RUN_ROOT="${2:-${EGO_RUNTIME_RUN_ROOT:-/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v1}}"
CASE_ID="${3:-${EGO_CASE_ID:-hot3d_clip001851_keyboard_pinhole}}"
PREFLIGHT_REPORT="${4:-${EGO_RUNTIME_PREFLIGHT_REPORT:-}}"
SESSION="${EGO_RUNTIME_TMUX_SESSION:-ego_annotation_runtime_$(basename "$RUN_ROOT")}"
SESSION_ID="${EGO_RUNTIME_PI_SESSION_ID:-v19-runtime-${CASE_ID}-$(basename "$RUN_ROOT")}"
MODEL_PROVIDER="${EGO_RUNTIME_PI_PROVIDER:-dexgem-responses}"
MODEL_ID="${EGO_RUNTIME_PI_MODEL:-gpt-5.6-sol}"
TARGET_HINT="${EGO_TARGET_OBJECT:-keyboard body only}"
TARGET_EXCLUSIONS="${EGO_TARGET_EXCLUSIONS:-hands, table, and unrelated scene objects}"
SESSION_DIR="${EGO_RUNTIME_SESSION_DIR:-/mnt/user-home/kupingxin/ego_annotation_runtime/sessions}"
LAUNCH_ROOT="${EGO_RUNTIME_LAUNCH_ROOT:-/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/runtime_launch_logs}"
LAUNCH_LOG="$LAUNCH_ROOT/${CASE_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
DONE="$LAUNCH_LOG.done"

[ -d "$BUNDLE" ] || { echo "missing isolated runtime bundle: $BUNDLE" >&2; exit 1; }
[ -s "$INPUT_VIDEO" ] || { echo "missing input video: $INPUT_VIDEO" >&2; exit 1; }
[ ! -e "$RUN_ROOT" ] || { echo "run root already exists; refusing overwrite: $RUN_ROOT" >&2; exit 1; }
[ -f "$BUNDLE/runtime/v19_runtime_spec.md" ] || { echo "missing runtime spec in bundle" >&2; exit 1; }
[ -n "$PREFLIGHT_REPORT" ] || { echo "preflight report must be supplied as argument 4 or EGO_RUNTIME_PREFLIGHT_REPORT" >&2; exit 1; }
[ -s "$PREFLIGHT_REPORT" ] || { echo "missing launch preflight report: $PREFLIGHT_REPORT" >&2; exit 1; }
"${EGO_MAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}" - "$PREFLIGHT_REPORT" "$BUNDLE" "$INPUT_VIDEO" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
report = json.load(open(sys.argv[1]))
if report.get("status") != "ready_for_runtime_agent_launch":
    raise SystemExit(f"launch preflight is not ready: {report.get('status')}")
for key, expected in zip(("bundle", "input_video", "run_root"), sys.argv[2:]):
    declared = report.get(key)
    if declared is None or Path(declared).resolve(strict=False) != Path(expected).resolve(strict=False):
        raise SystemExit(f"launch preflight {key} mismatch: report={declared!r} launch={expected!r}")
PY
mkdir -p "$SESSION_DIR" "$LAUNCH_ROOT"
rm -f "$DONE"

PROMPT="/v19-run $INPUT_VIDEO $RUN_ROOT $CASE_ID

User-approved physical target hypothesis: $TARGET_HINT. Re-confirm the target and rigid branch from the raw video and your own P05/P07 visual evidence; exclude $TARGET_EXCLUSIONS. Continue with uncertainty if measurements are weak, and do not treat this hint as a pixel mask or ground truth label."

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "runtime tmux session already exists: $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -n v19_runtime -c "$BUNDLE"
CMD="cd '$BUNDLE' && export PYTHONUNBUFFERED=1 TORCH_HOME=/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub && pi --print --approve --no-context-files --no-skills --no-extensions --provider '$MODEL_PROVIDER' --model '$MODEL_ID' --thinking max --mode text --tools read,bash,edit,write --system-prompt \"\$(cat '$BUNDLE/configs/v19_agent_system_prompt.md')\" --prompt-template '$BUNDLE/.pi/prompts/v19-run.md' --session-dir '$SESSION_DIR' --session-id '$SESSION_ID' '$PROMPT' 2>&1 | tee '$LAUNCH_LOG'; rc=\${PIPESTATUS[0]}; printf 'exit_code=%s\\ncase_id=%s\\nrun_root=%s\\ninput_video=%s\\nmodel=%s/%s\\n' \"\$rc\" '$CASE_ID' '$RUN_ROOT' '$INPUT_VIDEO' '$MODEL_PROVIDER' '$MODEL_ID' > '$DONE'; exit \$rc"
tmux send-keys -t "$SESSION:v19_runtime" "$CMD" C-m
printf 'runtime_session=%s\nruntime_window=v19_runtime\nbundle=%s\ninput=%s\nrun_root=%s\ncase_id=%s\ntarget=%s\nmodel=%s/%s\nlog=%s\nsentinel=%s\n' "$SESSION" "$BUNDLE" "$INPUT_VIDEO" "$RUN_ROOT" "$CASE_ID" "$TARGET_HINT" "$MODEL_PROVIDER" "$MODEL_ID" "$LAUNCH_LOG" "$DONE"
