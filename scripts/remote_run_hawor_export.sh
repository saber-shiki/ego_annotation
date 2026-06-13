#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_HAWOR_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/hawor_work}"
CASE="${EGO_HAWOR_CASE:-trash_1050}"
IMG_FOCAL="${EGO_HAWOR_IMG_FOCAL:-2304}"

case "$CASE" in
  trash_1050)
    DEFAULT_CLIP="/mnt/user-home/yiwen/ego_annotation_remote/data/clip/20260108_1057_Recf94e_P0_S994da4_task_9.mp4"
    DEFAULT_OUTPUT_DIR="$ROOT/outputs/trash_hawor_world"
    ;;
  task5_tomato_960)
    DEFAULT_CLIP="/mnt/user-home/yiwen/ego_annotation_remote/data/clip/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4"
    DEFAULT_OUTPUT_DIR="$ROOT/outputs/task5_tomato_960_hawor_world"
    ;;
  *)
    if [ -z "${EGO_HAWOR_CLIP:-}" ]; then
      echo "unknown EGO_HAWOR_CASE=$CASE and EGO_HAWOR_CLIP is not set" >&2
      exit 2
    fi
    DEFAULT_CLIP="$EGO_HAWOR_CLIP"
    DEFAULT_OUTPUT_DIR="$ROOT/outputs/${CASE}_hawor_world"
    ;;
esac

CLIP="${EGO_HAWOR_CLIP:-$DEFAULT_CLIP}"
OUTPUT_DIR="${EGO_HAWOR_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
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
  "$CLIP"; do
  if [ ! -e "$required" ]; then
    echo "missing required HaWoR export input: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
source .venv_hawor/bin/activate
export PYTHONPATH="$HAWOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "running HaWoR export case=$CASE clip=$CLIP output=$OUTPUT_DIR" >&2
python repo/scripts/export_hawor_world.py \
  --hawor-root "$HAWOR_ROOT" \
  --video_path "$CLIP" \
  --checkpoint "$CHECKPOINT" \
  --infiller_weight "$INFILLER" \
  --img_focal "$IMG_FOCAL" \
  --output-dir "$OUTPUT_DIR"
