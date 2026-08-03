#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${EGO_REPO_ROOT:-/mnt/user-home/kupingxin/ego_annotation}"
MAIN_PYTHON="${EGO_MAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MODEL_ROOT="${EGO_MODEL_ROOT:-/mnt/truenas-user-home/kupingxin/ego_annotation_models}"
WORK_ROOT="${EGO_TRELLIS_WORK_ROOT:-/var/tmp/kupingxin/ego_annotation_envs/trellis_work}"
SOURCE_ROOT="$WORK_ROOT/TRELLIS"
VENV="$WORK_ROOT/.venv_trellis"
MODEL_REVISION="25e0d31ffbebe4b5a97464dd851910efc3002d96"
MODEL_DIR="${EGO_TRELLIS_MODEL:-$MODEL_ROOT/trellis-image-large-${MODEL_REVISION:0:8}}"
TORCH_HOME="${EGO_TORCH_HOME:-$MODEL_ROOT/torch_hub}"
export TORCH_HOME
RUNTIME_LINK="$REPO_ROOT/.runtime/trellis_work"
LOG_ROOT="${EGO_TRELLIS_SETUP_LOG_ROOT:-$MODEL_ROOT/setup_logs}"
LOG="$LOG_ROOT/setup_local_trellis_runtime.log"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
TRELLIS_REPO_URL="https://github.com/microsoft/TRELLIS.git"
TRELLIS_COMMIT="442aa1e1afb9014e80681d3bf604e8d728a86ee7"
UTILS3D_COMMIT="9a4eb15e4021b67b12c460c7057d642626897ec8"
ENV_MARKER="trellis-${TRELLIS_COMMIT}-torch2.4.0-cu121-py310-v2"
PYTHON_BIN="${TRELLIS_PYTHON_BIN:-/usr/bin/python3.10}"

mkdir -p "$WORK_ROOT" "$MODEL_DIR" "$TORCH_HOME" "$LOG_ROOT" "$(dirname "$RUNTIME_LINK")"
exec > >(tee -a "$LOG") 2>&1
printf '\n=== TRELLIS setup start %s ===\n' "$(date -u +%FT%TZ)"
printf 'work_root=%s\nmodel_dir=%s\ntorch_home=%s\n' "$WORK_ROOT" "$MODEL_DIR" "$TORCH_HOME"
for path in "$UV" "$PYTHON_BIN" "$MAIN_PYTHON"; do
  [ -s "$path" ] || { echo "missing required input: $path" >&2; exit 1; }
done

if [ ! -d "$SOURCE_ROOT/.git" ]; then
  rm -rf "$SOURCE_ROOT"
  mkdir -p "$SOURCE_ROOT"
  git -C "$SOURCE_ROOT" init
  git -C "$SOURCE_ROOT" remote add origin "$TRELLIS_REPO_URL"
  git -C "$SOURCE_ROOT" fetch --depth 1 origin "$TRELLIS_COMMIT"
  git -C "$SOURCE_ROOT" checkout --detach FETCH_HEAD
  git -C "$SOURCE_ROOT" submodule update --init --recursive --depth 1
fi
actual_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[ "$actual_commit" = "$TRELLIS_COMMIT" ] || {
  echo "TRELLIS checkout mismatch: expected $TRELLIS_COMMIT got $actual_commit" >&2
  exit 1
}
git -C "$SOURCE_ROOT" submodule update --init --recursive --depth 1
ln -sfn "$WORK_ROOT" "$RUNTIME_LINK"

if [ -f "$VENV/.ego_env_marker" ] && [ "$(cat "$VENV/.ego_env_marker")" != "$ENV_MARKER" ]; then
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV"
  "$UV" venv --python "$PYTHON_BIN" "$VENV"
  "$UV" pip install --python "$VENV/bin/python" \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  "$UV" pip install --python "$VENV/bin/python" pip setuptools wheel
  "$UV" pip install --python "$VENV/bin/python" \
    numpy==2.2.6 pillow==12.2.0 scipy==1.15.3 trimesh==4.12.2 \
    easydict==1.13 omegaconf==2.3.0 tqdm==4.67.3 \
    imageio==2.37.3 imageio-ffmpeg==0.6.0 \
    opencv-python==4.13.0.92 rembg==2.0.69 onnxruntime==1.23.2 \
    transformers==4.46.3 tokenizers==0.20.3 \
    plyfile==1.1.4 pygltflib==1.16.5 warp-lang==1.14.0 ipyevents==2.0.4
  "$UV" pip install --python "$VENV/bin/python" --no-deps \
    xformers==0.0.27.post2 spconv-cu120==2.3.6 cumm-cu120==0.4.11 \
    ccimport==0.4.4 pccm==0.4.16 pybind11==3.0.4 fire==0.7.1 \
    lark==1.3.1 portalocker==3.2.0 ninja==1.13.0
  "$UV" pip install --python "$VENV/bin/python" --no-deps \
    kaolin==0.18.0 \
    --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
  UV_CACHE_DIR="${UV_CACHE_DIR:-/var/tmp/kupingxin/uv-cache-trellis}" \
    "$UV" pip install --python "$VENV/bin/python" \
    "git+https://github.com/EasternJournalist/utils3d.git@$UTILS3D_COMMIT"
fi

"$MAIN_PYTHON" - "$MODEL_DIR" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
model_dir, revision = sys.argv[1:3]
path = snapshot_download(
    repo_id="microsoft/TRELLIS-image-large",
    revision=revision,
    local_dir=model_dir,
)
print("trellis_model_snapshot", path)
PY

ATTN_BACKEND=xformers SPCONV_ALGO=native "$VENV/bin/python" - "$SOURCE_ROOT" <<'PY'
import importlib.util
import os
import sys
import types
from pathlib import Path
repo = Path(sys.argv[1])
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
sys.path.insert(0, str(repo))
import kaolin
import spconv
import torch
import transformers
import trimesh
import xformers
root = repo / "trellis"
pipelines = root / "pipelines"
pkg = types.ModuleType("trellis")
pkg.__path__ = [str(root)]
pkg.__package__ = "trellis"
sys.modules["trellis"] = pkg
sub = types.ModuleType("trellis.pipelines")
sub.__path__ = [str(pipelines)]
sub.__package__ = "trellis.pipelines"
sys.modules["trellis.pipelines"] = sub
path = pipelines / "trellis_image_to_3d.py"
spec = importlib.util.spec_from_file_location("trellis.pipelines.trellis_image_to_3d", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert hasattr(module, "TrellisImageTo3DPipeline")
assert torch.cuda.is_available(), "CUDA unavailable to TRELLIS interpreter"
print("trellis_import_smoke_ok", torch.__version__, torch.version.cuda, transformers.__version__, kaolin.__version__, spconv.__version__)
PY

printf '%s\n' "$ENV_MARKER" > "$VENV/.ego_env_marker"
"$VENV/bin/python" - "$SOURCE_ROOT" "$MODEL_DIR" "$VENV" "$TRELLIS_COMMIT" "$MODEL_REVISION" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
source, model, venv = map(Path, sys.argv[1:4])
expected_commit, revision = sys.argv[4:6]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = sorted(p for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts)
manifest = {
    'status': 'installed_import_validated_gpu_execution_not_yet_smoked',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'source_root': str(source),
    'source_commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
    'expected_commit': expected_commit,
    'venv': str(venv),
    'model': {'repo_id': 'microsoft/TRELLIS-image-large', 'revision': revision, 'path': str(model)},
    'model_files': [{'path': str(p.relative_to(model)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
}
(model/'TRELLIS_RUNTIME_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
"$VENV/bin/python" -m pip freeze | sort > "$MODEL_DIR/trellis_requirements.freeze.txt"
printf '=== TRELLIS setup complete %s ===\n' "$(date -u +%FT%TZ)"
printf 'source=%s\npython=%s\nmodel=%s\n' "$SOURCE_ROOT" "$VENV/bin/python" "$MODEL_DIR"
