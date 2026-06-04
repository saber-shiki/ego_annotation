#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 REMOTE_ROOT DATASET_DIR OUTPUT_DIR GPU_ID ZFAR DEBUG_LEVEL" >&2
  exit 2
fi

ROOT="$1"
DATASET_DIR="$2"
OUTPUT_DIR="$3"
GPU_ID="$4"
ZFAR="$5"
DEBUG_LEVEL="$6"

BUNDLE="$ROOT/BundleSDF"
PREFIX="$ROOT/micromamba_root/envs/bundlesdf_py311"
CONFIG="$BUNDLE/BundleTrack/config_ho3d.yml"
BACKUP="$CONFIG.ego_backup"

if [ ! -d "$BUNDLE" ]; then
  echo "missing BundleSDF checkout: $BUNDLE" >&2
  exit 1
fi
if [ ! -x "$PREFIX/bin/python" ]; then
  echo "missing BundleSDF Python env: $PREFIX" >&2
  exit 1
fi
for rel in rgb depth masks cam_K.txt; do
  if [ ! -e "$DATASET_DIR/$rel" ]; then
    echo "missing BundleSDF dataset entry: $DATASET_DIR/$rel" >&2
    exit 1
  fi
done

cleanup() {
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$CONFIG"
  fi
}
trap cleanup EXIT

cp "$CONFIG" "$BACKUP"
"$PREFIX/bin/python" - "$CONFIG" "$ZFAR" <<'PY'
from pathlib import Path
import sys
from ruamel.yaml import YAML

config = Path(sys.argv[1])
zfar = float(sys.argv[2])
if zfar <= 0.0:
    raise SystemExit("zfar must be positive")
yaml = YAML()
data = yaml.load(config.read_text(encoding="utf-8"))
data["depth_processing"]["zfar"] = zfar
yaml.dump(data, config.open("w", encoding="utf-8"))
print(f"set {config} depth_processing.zfar={zfar}")
PY

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="$BUNDLE/BundleTrack/build:$PREFIX/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$BUNDLE/BundleTrack/build:$BUNDLE/BundleTrack:$BUNDLE${PYTHONPATH:+:$PYTHONPATH}"

cd "$BUNDLE"
"$PREFIX/bin/python" - <<'PY'
import my_cpp
import torch
import open3d
print("my_cpp", my_cpp.__file__)
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("open3d", open3d.__version__)
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA unavailable")
PY

"$PREFIX/bin/python" run_custom.py \
  --mode run_video \
  --video_dir "$DATASET_DIR" \
  --out_folder "$OUTPUT_DIR" \
  --use_segmenter 0 \
  --use_gui 0 \
  --stride 1 \
  --debug_level "$DEBUG_LEVEL"

"$PREFIX/bin/python" run_custom.py \
  --mode global_refine \
  --video_dir "$DATASET_DIR" \
  --out_folder "$OUTPUT_DIR" \
  --use_segmenter 0 \
  --use_gui 0 \
  --stride 1 \
  --debug_level "$DEBUG_LEVEL"

test -f "$OUTPUT_DIR/config_bundletrack.yml"
test -d "$OUTPUT_DIR/ob_in_cam"
test -f "$OUTPUT_DIR/mesh/mesh_real_scale.obj"

echo BUNDLESDF_RUN_V3_OK
