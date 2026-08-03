#!/usr/bin/env bash
# Bootstrap the local, project-scoped runtime needed by the V19 SAM2/depth front end.
# Heavy model environments (HaWoR, UniDepth, TRELLIS) remain separate because they
# require incompatible PyTorch/CUDA stacks and licensed/model-specific assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV="${VENV:-$REPO_ROOT/.venv}"
SAM2_ROOT="${SAM2_ROOT:-$REPO_ROOT/third_party/sam2}"
SAM2_REPO="${SAM2_REPO:-https://github.com/facebookresearch/sam2.git}"
# Pinned source contains the SAM 2.1 configs/API used by the repository scripts.
SAM2_COMMIT="${SAM2_COMMIT:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$REPO_ROOT/checkpoints/sam2.1_hiera_small.pt}"
SAM2_CHECKPOINT_URL="${SAM2_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt}"
SAM2_CHECKPOINT_SHA256="${SAM2_CHECKPOINT_SHA256:-6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38}"
OWLV2_REPO="${OWLV2_REPO:-google/owlv2-base-patch16-ensemble}"
OWLV2_REVISION="${OWLV2_REVISION:-cfd3195ba4ea9592eec887ded089f4c08eff231d}"
OWLV2_MODEL="${OWLV2_MODEL:-$REPO_ROOT/checkpoints/owlv2-base-patch16-ensemble}"
OWLV2_WEIGHTS_SHA256="${OWLV2_WEIGHTS_SHA256:-e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7}"
GPU_ID="${GPU_ID:-0}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"

fail() {
  echo "setup_local_runtime: $*" >&2
  exit 1
}

[ -x "$UV_BIN" ] || fail "uv not found at $UV_BIN; install it with: python3 -m pip install --user uv"

if [ ! -x "$VENV/bin/python" ]; then
  "$UV_BIN" python install "$PYTHON_VERSION"
  "$UV_BIN" venv --python "$PYTHON_VERSION" --seed "$VENV"
else
  "$VENV/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"expected Python 3.11 in project venv, got {sys.version}")
PY
  # uv-created environments do not necessarily include pip; seed it without
  # replacing the already-installed project packages.
  "$UV_BIN" venv --python "$PYTHON_VERSION" --seed --allow-existing "$VENV"
fi

cd "$REPO_ROOT"
"$UV_BIN" sync --locked --inexact
"$VENV/bin/python" -m pip check

mkdir -p "$(dirname "$SAM2_ROOT")"
if [ ! -d "$SAM2_ROOT/.git" ]; then
  git clone --filter=blob:none "$SAM2_REPO" "$SAM2_ROOT"
fi
if [ -n "$(git -C "$SAM2_ROOT" status --porcelain)" ]; then
  fail "SAM2 checkout is dirty; clean it before changing the pinned revision: $SAM2_ROOT"
fi
if ! git -C "$SAM2_ROOT" cat-file -e "$SAM2_COMMIT^{commit}" 2>/dev/null; then
  git -C "$SAM2_ROOT" fetch --depth 1 origin "$SAM2_COMMIT" || git -C "$SAM2_ROOT" fetch --depth 1 origin main
fi
git -C "$SAM2_ROOT" checkout --detach "$SAM2_COMMIT" >/dev/null
test -f "$SAM2_ROOT/sam2/configs/sam2.1/sam2.1_hiera_s.yaml" || fail "SAM2 config is missing"

mkdir -p "$(dirname "$SAM2_CHECKPOINT")"
if [ -s "$SAM2_CHECKPOINT" ]; then
  actual="$(sha256sum "$SAM2_CHECKPOINT" | awk '{print $1}')"
  [ "$actual" = "$SAM2_CHECKPOINT_SHA256" ] || fail "checkpoint SHA256 mismatch: $SAM2_CHECKPOINT (got $actual)"
else
  tmp="${SAM2_CHECKPOINT}.part"
  rm -f "$tmp"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --connect-timeout 20 --output "$tmp" "$SAM2_CHECKPOINT_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --output-document="$tmp" "$SAM2_CHECKPOINT_URL"
  else
    fail "neither curl nor wget is available for checkpoint download"
  fi
  actual="$(sha256sum "$tmp" | awk '{print $1}')"
  [ "$actual" = "$SAM2_CHECKPOINT_SHA256" ] || { rm -f "$tmp"; fail "downloaded checkpoint SHA256 mismatch (got $actual)"; }
  mv "$tmp" "$SAM2_CHECKPOINT"
fi

if [ ! -s "$OWLV2_MODEL/model.safetensors" ]; then
  mkdir -p "$OWLV2_MODEL"
  HF_HOME="${HF_HOME:-$(dirname "$OWLV2_MODEL")/.hf_cache}" \
    "$VENV/bin/python" - "$OWLV2_REPO" "$OWLV2_REVISION" "$OWLV2_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, output = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=output,
    allow_patterns=[
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ],
)
PY
fi
actual="$(sha256sum "$OWLV2_MODEL/model.safetensors" | awk '{print $1}')"
[ "$actual" = "$OWLV2_WEIGHTS_SHA256" ] || fail "OWLv2 weights SHA256 mismatch: $OWLV2_MODEL/model.safetensors (got $actual)"

# The tracking scripts add third_party/sam2 to sys.path themselves.  This check
# therefore validates the actual import contract instead of only package metadata.
PYTHONPATH="$SAM2_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$VENV/bin/python" - <<'PY'
import importlib
import inspect
import numpy as np
import open3d
import torch
for name in ("iopath", "sam2"):
    importlib.import_module(name)
# Chumpy 0.70 predates Python 3.11 and NumPy 1.24. Project scripts apply
# this same narrow shim before unpickling legacy MANO data.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for name, value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if name not in np.__dict__:
        setattr(np, name, value)
importlib.import_module("chumpy")
print("runtime_imports_ok", "open3d", open3d.__version__, "torch", torch.__version__)
PY

if [ "$RUN_GPU_SMOKE" = "1" ]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$SAM2_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$VENV/bin/python" - "$SAM2_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
checkpoint = Path(sys.argv[1])
model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint), device="cuda")
predictor = SAM2ImagePredictor(model)
image = np.zeros((128, 192, 3), dtype=np.uint8)
image[32:96, 56:136] = (180, 90, 30)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray([[96.0, 64.0]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        multimask_output=False,
    )
if masks.shape != (1, 128, 192) or not np.isfinite(scores).all():
    raise SystemExit("SAM2 GPU smoke test returned an invalid result")
print("sam2_gpu_smoke_ok", "device", torch.cuda.get_device_name(0), "mask_pixels", int(masks[0].sum()))
PY
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$VENV/bin/python" - "$OWLV2_MODEL" <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
model_path = Path(sys.argv[1])
processor = Owlv2Processor.from_pretrained(str(model_path), local_files_only=True)
model = Owlv2ForObjectDetection.from_pretrained(str(model_path), local_files_only=True).to("cuda").eval()
image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8) + 90)
inputs = processor(text=[["an object."]], images=image, return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model(**inputs)
if tuple(outputs.logits.shape[:2]) != (1, 3600) or not torch.isfinite(outputs.logits).all():
    raise SystemExit("OWLv2 GPU smoke test returned an invalid result")
print("owlv2_gpu_smoke_ok", "device", torch.cuda.get_device_name(0), "queries", outputs.logits.shape[1])
PY
fi

printf '\nLocal runtime ready.\n'
printf 'repo:       %s\n' "$REPO_ROOT"
printf 'python:     %s\n' "$VENV/bin/python"
printf 'sam2:       %s @ %s\n' "$SAM2_ROOT" "$(git -C "$SAM2_ROOT" rev-parse HEAD)"
printf 'checkpoint: %s\n' "$SAM2_CHECKPOINT"
printf 'owlv2:     %s @ %s\n' "$OWLV2_MODEL" "$OWLV2_REVISION"
printf 'activate:   source %s/bin/activate\n' "$VENV"
