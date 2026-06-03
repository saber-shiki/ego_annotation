#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_HAWOR_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/hawor_work}"
CLIP="${EGO_HAWOR_CLIP:-/mnt/user-home/yiwen/ego_annotation_remote/data/clip/20260108_1057_Recf94e_P0_S994da4_task_9.mp4}"
OUTPUT_DIR="${EGO_HAWOR_OUTPUT_DIR:-$ROOT/outputs/trash_hawor_world}"
IMG_FOCAL="${EGO_HAWOR_IMG_FOCAL:-2304}"

cd "$ROOT"
source .venv_hawor/bin/activate
export PYTHONPATH="$ROOT/third_party/HaWoR${PYTHONPATH:+:$PYTHONPATH}"

python repo/scripts/export_hawor_world.py \
  --hawor-root "$ROOT/third_party/HaWoR" \
  --video_path "$CLIP" \
  --checkpoint "$ROOT/third_party/HaWoR/weights/hawor/checkpoints/hawor.ckpt" \
  --infiller_weight "$ROOT/third_party/HaWoR/weights/hawor/checkpoints/infiller.pt" \
  --img_focal "$IMG_FOCAL" \
  --output-dir "$OUTPUT_DIR"
