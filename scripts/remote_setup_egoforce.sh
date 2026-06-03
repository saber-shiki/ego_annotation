#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/egoforce_work}"
REPO="${ROOT}/EgoForce"
LOG_DIR="${ROOT}/logs"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"

mkdir -p "${ROOT}" "${LOG_DIR}"
cd "${ROOT}"

if [ ! -x "${UV_BIN}" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="/mnt/user-home/yiwen/.local/bin/uv"
fi

if [ ! -d "${REPO}/.git" ]; then
  git clone https://github.com/dfki-av/EgoForce.git "${REPO}"
else
  git -C "${REPO}" pull --ff-only
fi

cd "${REPO}"
"${UV_BIN}" venv --python 3.10 .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools==81.0.0 wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torch_tensorrt==2.8.0+cu126 --find-links https://download.pytorch.org/whl/torch-tensorrt
python -m pip install -r scripts/requirements.txt
python -m pip install mmcv==2.1.0 --no-build-isolation
python -m pip install git+https://github.com/javrtg/AnyCalib.git --no-build-isolation
python -m pip install git+https://github.com/mattloper/chumpy.git --no-build-isolation
python -m pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
python -m pip install thirdparty/datapipes
python -m pip install thirdparty/mmdetection --no-build-isolation
python -m pip install numpy==1.26.4 projectaria_client_sdk==1.1.0 --no-cache-dir

if [ ! -f "${REPO}/_DATA/model_weights.pth" ]; then
  bash scripts/download_model_weights.sh
fi

python - <<'PY'
import importlib
mods = ["torch", "torch_tensorrt", "cv2", "mmdet", "ultralytics", "pytorch3d", "anycalib"]
for name in mods:
    importlib.import_module(name)
print("egoforce_setup_imports_ok")
PY

echo "egoforce_setup_ok"
