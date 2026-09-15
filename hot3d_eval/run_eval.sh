#!/usr/bin/env bash
# HOT3D 评测：把标注 JSON 路径填进 ANNOTATIONS，然后执行本脚本即可。
#
#   bash hot3d_eval/run_eval.sh
#
# 也可以不改文件、直接把路径作为参数传入：
#   bash hot3d_eval/run_eval.sh /path/to/run/final/annotation.json ...
#
# 报告默认写到 hot3d_eval/reports/（已 gitignore）。
set -euo pipefail

# ============================================================
# 1) 在这里填标注 JSON 路径（一行一个，可以填多个）
# ============================================================
ANNOTATIONS=(
  # /mnt/user-home/zjh/ego_annotation_runtime/grouped_astra_20260914/smoke_runs/mug_white_48f_verified_slam_59cb75a1/run/final/annotation.json
)

# ============================================================
# 2) 可选覆盖（默认值即本服务器约定，通常不用改；
#    也可用同名 HOT3D_EVAL_* 环境变量覆盖，不必改文件）
# ============================================================
RUNTIME_DEFAULT=/mnt/user-home/zjh/ego_annotation_runtime/grouped_astra_20260914
PYTHON="${HOT3D_EVAL_PYTHON:-$RUNTIME_DEFAULT/env/py_main}"
MODEL_CONFIG="${HOT3D_EVAL_MODEL_CONFIG:-$RUNTIME_DEFAULT/builds/59cb75a1/config/model_envs.json}"
DATASET_ROOT="${HOT3D_EVAL_DATASET_ROOT:-/mnt/nas-106/hot3d/dataset}"
OUTPUT="${HOT3D_EVAL_OUTPUT:-}"

# ============================================================
# 以下无需修改
# ============================================================
ANNOTATIONS+=("$@")
if [ "${#ANNOTATIONS[@]}" -eq 0 ]; then
  echo "错误：没有标注 JSON。编辑本脚本顶部的 ANNOTATIONS 填入路径，或作为参数传入：" >&2
  echo "  bash hot3d_eval/run_eval.sh /path/to/run/final/annotation.json" >&2
  exit 1
fi
for a in "${ANNOTATIONS[@]}"; do
  [ -f "$a" ] || { echo "错误：标注文件不存在：$a" >&2; exit 1; }
done
[ -x "$PYTHON" ] || { echo "错误：解释器不可用：$PYTHON（可用 HOT3D_EVAL_PYTHON=... 覆盖）" >&2; exit 1; }
[ -f "$MODEL_CONFIG" ] || { echo "错误：模型配置不存在：$MODEL_CONFIG（可用 HOT3D_EVAL_MODEL_CONFIG=... 覆盖）" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$OUTPUT" ]; then
  mkdir -p "$HERE/reports"
  OUTPUT="$HERE/reports/hot3d_eval_report_$(date +%Y%m%d_%H%M%S).md"
fi

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON" "$HERE/run_eval.py" \
  --annotations "${ANNOTATIONS[@]}" \
  --model-config "$MODEL_CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUTPUT"

echo
echo "报告：$OUTPUT"
