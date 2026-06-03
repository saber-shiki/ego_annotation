#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/user-home/yiwen/ego_annotation_remote/sam3_work
mkdir -p "$ROOT/repo/scripts" "$ROOT/data" "$ROOT/checkpoints" "$ROOT/third_party" "$ROOT/outputs"
cd "$ROOT"

if [ ! -d third_party/SAMWISE/.git ]; then
  git clone https://github.com/ClaudiaCuttano/SAMWISE.git third_party/SAMWISE
else
  git -C third_party/SAMWISE pull --ff-only
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.10 .venv_samwise
source .venv_samwise/bin/activate
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r third_party/SAMWISE/requirements.txt
uv pip install -e third_party/SAMWISE/models/sam2 || true
uv pip install gdown

cd "$ROOT/checkpoints"
if [ ! -f final_model_mevis.pth ]; then
  gdown --fuzzy 'https://drive.google.com/file/d/1Molt2up2bP41ekeczXWQU-LWTskKJOV2/view?usp=sharing' -O final_model_mevis.pth
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
