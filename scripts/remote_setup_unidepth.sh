#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${EGO_UNIDEPTH_ROOT:-$PROJECT_ROOT/.runtime/unidepth_work}"
REPO="$ROOT/UniDepth"
MODEL_ENV="${EGO_MODEL_ENV:-$PROJECT_ROOT/.runtime/model_envs/unidepth_sam2}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-3.10}"
UNIDEPTH_REPO_URL="${EGO_UNIDEPTH_REPO_URL:-https://github.com/lpiccinelli-eth/UniDepth.git}"
# This is the source revision used by the V19 A800 runtime that this branch documents.
UNIDEPTH_REVISION="${EGO_UNIDEPTH_REVISION:-8d8cfe4c7ee15297099983607febf0d4f32eb3d6}"
UNIDEPTH_MODEL_ID="${EGO_UNIDEPTH_MODEL_ID:-lpiccinelli/unidepth-v2-vitl14}"
UNIDEPTH_MODEL_REVISION="${EGO_UNIDEPTH_MODEL_REVISION:-52b349b514bd8b47642f67ac78cb7b5dc5c51dd9}"
UNIDEPTH_MODEL_DIR="${EGO_UNIDEPTH_MODEL_DIR:-$ROOT/models/unidepth-v2-vitl14-52b349b5}"
UNIDEPTH_MODEL_SHA256="${EGO_UNIDEPTH_MODEL_SHA256:-ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6}"

mkdir -p "$ROOT"/{logs,outputs,data}
cd "$ROOT"

if [ ! -x "$UV_BIN" ]; then
  python3 -m pip install --user uv==0.12.1
  UV_BIN="$HOME/.local/bin/uv"
fi
[ -x "$UV_BIN" ] || { echo "uv is not executable: $UV_BIN" >&2; exit 1; }

if [ ! -d "$REPO/.git" ]; then
  git clone --filter=blob:none "$UNIDEPTH_REPO_URL" "$REPO"
fi

git_repo() {
  git -c safe.directory="$REPO" -C "$REPO" "$@"
}

if [ -n "$(git_repo status --porcelain)" ]; then
  echo "UniDepth checkout is dirty; refusing to replace its revision: $REPO" >&2
  exit 1
fi
if [ -n "$UNIDEPTH_REVISION" ]; then
  if ! git_repo cat-file -e "$UNIDEPTH_REVISION^{commit}" 2>/dev/null; then
    git_repo fetch --depth 1 origin "$UNIDEPTH_REVISION" || git_repo fetch --depth 1 origin main
  fi
  git_repo checkout --detach "$UNIDEPTH_REVISION"
else
  git_repo pull --ff-only
fi

cd "$REPO"
mkdir -p "$(dirname "$MODEL_ENV")"
if [ -d "$MODEL_ENV" ] && [ ! -x "$MODEL_ENV/bin/python" ]; then
  rm -rf "$MODEL_ENV"
fi
if [ ! -x "$MODEL_ENV/bin/python" ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$MODEL_ENV"
fi
if [ ! -x "$MODEL_ENV/bin/python" ]; then
  echo "UniDepth/SAM2 model env python is not executable after venv creation: $MODEL_ENV/bin/python" >&2
  exit 1
fi

"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" --upgrade pip setuptools wheel
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" \
  opencv-python==4.13.0.92 pillow numpy==2.2.6 scipy==1.15.3 tqdm \
  open3d==0.19.0 trimesh==4.12.2 smplx==0.1.28 pyrender==0.1.45 plyfile rtree
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" hydra-core omegaconf iopath
"$UV_BIN" pip check --python "$MODEL_ENV/bin/python"

"$MODEL_ENV/bin/python" - <<'PY'
import importlib
import torch
from unidepth.models import UniDepthV2

for name in [
    "torch", "cv2", "PIL", "numpy", "unidepth", "hydra", "omegaconf", "iopath",
    "tqdm", "open3d", "trimesh", "smplx", "pyrender", "plyfile", "rtree",
]:
    importlib.import_module(name)
if not hasattr(UniDepthV2, "from_pretrained"):
    raise RuntimeError("UniDepthV2.from_pretrained is unavailable")
if not torch.cuda.is_available():
    raise RuntimeError("UniDepth setup requires CUDA")
print("unidepth_setup_ok", "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

if [ ! -s "$UNIDEPTH_MODEL_DIR/model.safetensors" ]; then
  mkdir -p "$UNIDEPTH_MODEL_DIR"
  HF_HOME="${HF_HOME:-$ROOT/.hf_cache}" \
    "$MODEL_ENV/bin/python" - "$UNIDEPTH_MODEL_ID" "$UNIDEPTH_MODEL_REVISION" "$UNIDEPTH_MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, output = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=output,
    allow_patterns=["model.safetensors", "config.json"],
)
PY
fi
actual="$(sha256sum "$UNIDEPTH_MODEL_DIR/model.safetensors" | awk '{print $1}')"
[ "$actual" = "$UNIDEPTH_MODEL_SHA256" ] || {
  echo "UniDepth model SHA256 mismatch: $UNIDEPTH_MODEL_DIR/model.safetensors (got $actual)" >&2
  exit 1
}
printf 'unidepth_model=%s@%s\n' "$UNIDEPTH_MODEL_DIR" "$UNIDEPTH_MODEL_REVISION"
