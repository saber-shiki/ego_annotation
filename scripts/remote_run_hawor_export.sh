#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${EGO_HAWOR_ROOT:-$PROJECT_ROOT/.runtime/hawor_work}"
CASE="${EGO_HAWOR_CASE:-runtime_case}"
IMG_FOCAL="${EGO_HAWOR_IMG_FOCAL:-2304}"
CAMERA_INTRINSICS="${EGO_HAWOR_CAMERA_INTRINSICS_FX_FY_CX_CY:-}"
FORCE_FOCAL_CACHE_REFRESH="${EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH:-0}"

if [ -z "${EGO_HAWOR_CLIP:-}" ]; then
  echo "EGO_HAWOR_CLIP must name the explicitly authorized, isolated input video" >&2
  exit 2
fi
CLIP="$EGO_HAWOR_CLIP"
OUTPUT_DIR="${EGO_HAWOR_OUTPUT_DIR:-$ROOT/outputs/${CASE}_hawor_world}"
HAWOR_ROOT="$ROOT/third_party/HaWoR"
CHECKPOINT="$HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt"
INFILLER="$HAWOR_ROOT/weights/hawor/checkpoints/infiller.pt"
CONFIG="$HAWOR_ROOT/weights/hawor/model_config.yaml"

for required in \
  "$ROOT/.venv_hawor/bin/activate" \
  "$HAWOR_ROOT/.git" \
  "$CHECKPOINT" \
  "$INFILLER" \
  "$CONFIG" \
  "$CLIP" \
  "$SCRIPT_DIR/export_hawor_world.py"; do
  if [ ! -e "$required" ]; then
    echo "missing required HaWoR export input: $required" >&2
    exit 1
  fi
done

if [ -n "${EGO_HAWOR_CLIP_SHA256:-}" ]; then
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "EGO_HAWOR_CLIP_SHA256 was provided but sha256sum is unavailable" >&2
    exit 1
  fi
  ACTUAL_CLIP_SHA256="$(sha256sum "$CLIP" | awk '{print $1}')"
  if [ "$ACTUAL_CLIP_SHA256" != "$EGO_HAWOR_CLIP_SHA256" ]; then
    echo "task clip sha256 mismatch for $CLIP" >&2
    echo "expected: $EGO_HAWOR_CLIP_SHA256" >&2
    echo "actual:   $ACTUAL_CLIP_SHA256" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
source .venv_hawor/bin/activate
export PYTHONPATH="$HAWOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

EXTRA_ARGS=()
if [ "$FORCE_FOCAL_CACHE_REFRESH" = "1" ] || [ "$FORCE_FOCAL_CACHE_REFRESH" = "true" ]; then
  EXTRA_ARGS+=(--force-focal-cache-refresh)
fi
if [ -n "$CAMERA_INTRINSICS" ]; then
  read -r -a CAMERA_K_VALUES <<< "$CAMERA_INTRINSICS"
  if [ "${#CAMERA_K_VALUES[@]}" -ne 4 ]; then
    echo "EGO_HAWOR_CAMERA_INTRINSICS_FX_FY_CX_CY must contain exactly four values: fx fy cx cy" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--camera-intrinsics "${CAMERA_K_VALUES[@]}")
fi

echo "running HaWoR export case=$CASE clip=$CLIP output=$OUTPUT_DIR img_focal=$IMG_FOCAL camera_intrinsics=${CAMERA_INTRINSICS:-legacy_image_center} force_focal_cache_refresh=$FORCE_FOCAL_CACHE_REFRESH" >&2
python "$SCRIPT_DIR/export_hawor_world.py" \
  --hawor-root "$HAWOR_ROOT" \
  --video_path "$CLIP" \
  --checkpoint "$CHECKPOINT" \
  --infiller_weight "$INFILLER" \
  --model_config "$CONFIG" \
  --img_focal "$IMG_FOCAL" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR"
