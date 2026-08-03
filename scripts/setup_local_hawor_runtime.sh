#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${EGO_REPO_ROOT:-/mnt/user-home/kupingxin/ego_annotation}"
MODEL_ROOT="${EGO_MODEL_ROOT:-/mnt/truenas-user-home/kupingxin/ego_annotation_models}"
MANO_ROOT="${EGO_MANO_ROOT:-$MODEL_ROOT/mano}"
WORK_ROOT="${EGO_HAWOR_WORK_ROOT:-/var/tmp/kupingxin/ego_annotation_envs/hawor_work}"
SOURCE_ROOT="$WORK_ROOT/HaWoR"
VENV="$WORK_ROOT/.venv_hawor"
ASSET_ROOT="${EGO_HAWOR_ASSET_ROOT:-$MODEL_ROOT/hawor-66c7d410}"
RUNTIME_LINK="$REPO_ROOT/.runtime/hawor_work"
LOG_ROOT="${EGO_HAWOR_SETUP_LOG_ROOT:-$MODEL_ROOT/setup_logs}"
LOG="$LOG_ROOT/setup_local_hawor_runtime.log"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
HAWOR_REPO_URL="https://github.com/ThunderVVV/HaWoR.git"
HAWOR_COMMIT="66c7d4108d58a716deccd192cb7645170cdc7bd7"
ENV_MARKER="hawor-${HAWOR_COMMIT}-torch2.6.0-cu126-py310-v2"
PYTHON_BIN="${HAWOR_PYTHON_BIN:-/usr/bin/python3.10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "$WORK_ROOT" "$ASSET_ROOT" "$LOG_ROOT" "$(dirname "$RUNTIME_LINK")"
exec > >(tee -a "$LOG") 2>&1
printf '\n=== HaWoR setup start %s ===\n' "$(date -u +%FT%TZ)"
printf 'work_root=%s\nasset_root=%s\nmano_root=%s\n' "$WORK_ROOT" "$ASSET_ROOT" "$MANO_ROOT"

for path in "$MANO_ROOT/MANO_LEFT.pkl" "$MANO_ROOT/MANO_RIGHT.pkl" "$UV" "$PYTHON_BIN"; do
  [ -s "$path" ] || { echo "missing required input: $path" >&2; exit 1; }
done

if [ ! -d "$SOURCE_ROOT/.git" ]; then
  rm -rf "$SOURCE_ROOT"
  mkdir -p "$SOURCE_ROOT"
  git -C "$SOURCE_ROOT" init
  git -C "$SOURCE_ROOT" remote add origin "$HAWOR_REPO_URL"
  git -C "$SOURCE_ROOT" fetch --depth 1 origin "$HAWOR_COMMIT"
  git -C "$SOURCE_ROOT" checkout --detach FETCH_HEAD
  git -C "$SOURCE_ROOT" submodule update --init --recursive --depth 1
fi
actual_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[ "$actual_commit" = "$HAWOR_COMMIT" ] || {
  echo "HaWoR checkout mismatch: expected $HAWOR_COMMIT got $actual_commit" >&2
  exit 1
}
git -C "$SOURCE_ROOT" submodule update --init --recursive --depth 1
ln -sfn "$WORK_ROOT" "$RUNTIME_LINK"
mkdir -p "$WORK_ROOT/third_party"
ln -sfn ../HaWoR "$WORK_ROOT/third_party/HaWoR"

if [ -f "$VENV/.ego_env_marker" ] && [ "$(cat "$VENV/.ego_env_marker")" != "$ENV_MARKER" ]; then
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV"
  "$UV" venv --python "$PYTHON_BIN" "$VENV"
  "$UV" pip install --python "$VENV/bin/python" \
    torch==2.6.0+cu126 torchvision==0.21.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
  "$UV" pip install --python "$VENV/bin/python" 'setuptools<70' wheel ninja packaging pip
  "$UV" pip install --python "$VENV/bin/python" \
    torch-scatter==2.1.2 \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu126.html

  PINNED_REQ="$WORK_ROOT/requirements.pinned.txt"
  "$PYTHON_BIN" - "$SOURCE_ROOT/requirements.txt" "$PINNED_REQ" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1]).read_text()
src = src.replace(
    "git+https://github.com/facebookresearch/pytorch3d.git@stable",
    "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47",
)
src = src.replace(
    "chumpy@git+https://github.com/mattloper/chumpy",
    "chumpy@git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da",
)
src = "\n".join(line for line in src.splitlines() if line.strip() != "torch-scatter==2.1.2") + "\n"
Path(sys.argv[2]).write_text(src)
PY
  "$UV" pip install --python "$VENV/bin/python" --no-build-isolation -r "$PINNED_REQ"
  "$UV" pip install --python "$VENV/bin/python" pytorch-lightning==2.2.4 --no-deps
  "$UV" pip install --python "$VENV/bin/python" lightning-utilities torchmetrics==1.4.0 gdown
fi
export PATH="$VENV/bin:$PATH"

mkdir -p \
  "$ASSET_ROOT/weights/external" \
  "$ASSET_ROOT/weights/hawor/checkpoints" \
  "$ASSET_ROOT/weights/hawor" \
  "$ASSET_ROOT/thirdparty/Metric3D/weights"

download_http() {
  local url="$1" destination="$2"
  if [ ! -s "$destination" ]; then
    local temporary="${destination}.partial"
    curl -fL --retry 5 --retry-delay 2 --continue-at - -o "$temporary" "$url"
    mv "$temporary" "$destination"
  fi
}

download_gdrive() {
  local id="$1" destination="$2"
  if [ ! -s "$destination" ]; then
    local download_dir="$WORK_ROOT/downloads"
    local local_tmp="$download_dir/$(basename "$destination").partial"
    local nas_tmp="${destination}.partial"
    mkdir -p "$download_dir"
    rm -f "$local_tmp" "$nas_tmp"
    "$VENV/bin/gdown" --continue "https://drive.google.com/uc?id=${id}" -O "$local_tmp"
    cp "$local_tmp" "$nas_tmp"
    mv "$nas_tmp" "$destination"
    rm -f "$local_tmp"
  fi
}

download_http \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
  "$ASSET_ROOT/weights/external/detector.pt"
download_http \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt \
  "$ASSET_ROOT/weights/hawor/checkpoints/hawor.ckpt"
download_http \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt \
  "$ASSET_ROOT/weights/hawor/checkpoints/infiller.pt"
download_http \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml \
  "$ASSET_ROOT/weights/hawor/model_config.yaml"
download_gdrive 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh "$ASSET_ROOT/weights/external/droid.pth"
download_gdrive 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr "$ASSET_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"

verify_hash() {
  local expected="$1" path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [ "$actual" = "$expected" ] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 1
  }
}
verify_hash 5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033 "$ASSET_ROOT/weights/external/detector.pt"
verify_hash 4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2 "$ASSET_ROOT/weights/hawor/checkpoints/hawor.ckpt"
verify_hash 30715e7e72e91d4e164bb762c7ea613dcff5448dbda5fabf40b4054e408cc5c2 "$ASSET_ROOT/weights/hawor/checkpoints/infiller.pt"
verify_hash edfe12dc14ce371d698da722b59acfed5b4a38a7f8f5116cbc1fce459a07dd2d "$ASSET_ROOT/weights/hawor/model_config.yaml"
verify_hash 46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526 "$ASSET_ROOT/weights/external/droid.pth"
verify_hash 15328ffc42b528b95f188687418f6f03b3f123eb34ccdbd686c112abbea6d972 "$ASSET_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"

link_file() {
  local source="$1" destination="$2"
  mkdir -p "$(dirname "$destination")"
  if [ -e "$destination" ] && [ ! -L "$destination" ]; then
    rm -f "$destination"
  fi
  ln -sfn "$source" "$destination"
}
link_file "$ASSET_ROOT/weights/external/detector.pt" "$SOURCE_ROOT/weights/external/detector.pt"
link_file "$ASSET_ROOT/weights/external/droid.pth" "$SOURCE_ROOT/weights/external/droid.pth"
link_file "$ASSET_ROOT/weights/hawor/checkpoints/hawor.ckpt" "$SOURCE_ROOT/weights/hawor/checkpoints/hawor.ckpt"
link_file "$ASSET_ROOT/weights/hawor/checkpoints/infiller.pt" "$SOURCE_ROOT/weights/hawor/checkpoints/infiller.pt"
link_file "$ASSET_ROOT/weights/hawor/model_config.yaml" "$SOURCE_ROOT/weights/hawor/model_config.yaml"
link_file "$ASSET_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth" "$SOURCE_ROOT/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"
link_file "$MANO_ROOT/MANO_RIGHT.pkl" "$SOURCE_ROOT/_DATA/data/mano/MANO_RIGHT.pkl"
link_file "$MANO_ROOT/MANO_LEFT.pkl" "$SOURCE_ROOT/_DATA/data_left/mano_left/MANO_LEFT.pkl"

DROID="$SOURCE_ROOT/thirdparty/DROID-SLAM"
sed -i -E \
  "/-gencode=arch=compute_(60|61|70|75|80|86),code=(sm_|compute_)(60|61|70|75|80|86)/d" \
  "$DROID/setup.py" "$DROID/thirdparty/lietorch/setup.py"
sed -i \
  -e 's/volume\.type()/volume.scalar_type()/g' \
  -e 's/fmap1\.type()/fmap1.scalar_type()/g' \
  "$DROID/src/correlation_kernels.cu" \
  "$DROID/src/altcorr_kernel.cu" \
  "$DROID/thirdparty/lietorch/lietorch/extras/corr_index_kernel.cu"
sed -i \
  -e 's/::detail::scalar_type(the_type)/the_type.scalarType()/g' \
  "$DROID/thirdparty/lietorch/lietorch/include/dispatch.h"

if ! "$VENV/bin/python" -c 'import torch; import droid_backends, lietorch' >/dev/null 2>&1; then
  rm -rf "$DROID/build" "$DROID/droid_backends.egg-info" \
    "$DROID/thirdparty/lietorch/build" "$DROID/thirdparty/lietorch/lietorch.egg-info"
  (cd "$DROID" && "$VENV/bin/python" setup.py install)
fi

(cd "$SOURCE_ROOT" && "$VENV/bin/python" - <<'PY'
import inspect
import numpy as np
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for name, value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
import chumpy
import cv2
import torch
import droid_backends
import lietorch
import mmcv
import pytorch3d
import smplx
assert torch.cuda.is_available(), "CUDA is unavailable to the HaWoR interpreter"
print("hawor_import_smoke_ok", torch.__version__, torch.version.cuda, cv2.__version__)
PY
)

printf '%s\n' "$ENV_MARKER" > "$VENV/.ego_env_marker"
"$VENV/bin/python" - "$SOURCE_ROOT" "$ASSET_ROOT" "$VENV" "$HAWOR_COMMIT" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
source, assets, venv, expected_commit = map(Path, sys.argv[1:5])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = [
    assets/'weights/external/detector.pt',
    assets/'weights/external/droid.pth',
    assets/'weights/hawor/checkpoints/hawor.ckpt',
    assets/'weights/hawor/checkpoints/infiller.pt',
    assets/'weights/hawor/model_config.yaml',
    assets/'thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth',
]
manifest = {
    'status': 'installed_import_validated_gpu_execution_not_yet_smoked',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'source_root': str(source),
    'source_commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
    'expected_commit': str(expected_commit),
    'venv': str(venv),
    'python': sys.version,
    'assets': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
}
(assets/'HAWOR_RUNTIME_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
"$VENV/bin/python" -m pip freeze | sort > "$ASSET_ROOT/hawor_requirements.freeze.txt"
printf '=== HaWoR setup complete %s ===\n' "$(date -u +%FT%TZ)"
printf 'source=%s\npython=%s\nassets=%s\n' "$SOURCE_ROOT" "$VENV/bin/python" "$ASSET_ROOT"
