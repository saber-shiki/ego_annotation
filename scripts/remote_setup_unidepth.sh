#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_UNIDEPTH_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/unidepth_work}"
REPO="$ROOT/UniDepth"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

mkdir -p "$ROOT"/{logs,outputs,data}
cd "$ROOT"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/lpiccinelli-eth/UniDepth.git "$REPO"
else
  git -C "$REPO" pull --ff-only
fi

cd "$REPO"
if [ ! -x .venv/bin/python ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" .venv
fi

"$UV_BIN" pip install --python .venv/bin/python --upgrade pip setuptools wheel
"$UV_BIN" pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python .venv/bin/python -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python .venv/bin/python opencv-python pillow numpy scipy

.venv/bin/python - <<'PY'
import importlib
import torch

for name in ["torch", "cv2", "PIL", "unidepth"]:
    importlib.import_module(name)
if not torch.cuda.is_available():
    raise RuntimeError("UniDepth setup requires CUDA")
print("unidepth_setup_ok", "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
