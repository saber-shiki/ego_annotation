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
TORCH_LIB="$PREFIX/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$BUNDLE/BundleTrack/build:$PREFIX/lib:$TORCH_LIB:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
P3D_TRANSFORMS="${EGO_PYTORCH3D_TRANSFORMS_PATH:-/mnt/user-home/yiwen/.cache/uv/archive-v0/GQ7Vw61ILrlJefOs}"
export PYTHONPATH="$P3D_TRANSFORMS:$BUNDLE/mycuda:$BUNDLE/BundleTrack/build:$BUNDLE/BundleTrack:$BUNDLE${PYTHONPATH:+:$PYTHONPATH}"

cd "$BUNDLE"
"$PREFIX/bin/python" - <<'PY'
import my_cpp
import kaolin
import torch
import open3d
import pyrender
import xatlas
from skimage import measure
print("my_cpp", my_cpp.__file__)
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("open3d", open3d.__version__)
print("pyrender", pyrender.__version__)
print("xatlas", xatlas.__name__)
print("skimage_marching_cubes", measure.marching_cubes.__module__)
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA unavailable")
print("kaolin", kaolin.__version__)
pts = torch.rand((16, 3), device="cuda") * 2.0 - 1.0
quantized = kaolin.ops.spc.quantize_points(pts.contiguous(), level=4)
octree = kaolin.ops.spc.unbatched_points_to_octree(quantized, 4, sorted=False)
print("kaolin_spc_octree_bytes", len(octree))
PY

"$PREFIX/bin/python" - <<'PY'
from pathlib import Path

path = Path("bundlesdf.py")
text = path.read_text(encoding="utf-8")
needle = "      pdb.set_trace()\n"
if needle in text:
    path.write_text(text.replace(needle, ""), encoding="utf-8")
if "pdb.set_trace()" in path.read_text(encoding="utf-8"):
    raise SystemExit("interactive pdb breakpoint remains in bundlesdf.py")
PY

"$PREFIX/bin/python" - <<'PY'
from pathlib import Path

utils = Path("Utils.py")
text = utils.read_text(encoding="utf-8")
helper = """
def remove_duplicate_faces_compat(mesh):
  if hasattr(mesh, "remove_duplicate_faces"):
    mesh.remove_duplicate_faces()
  else:
    unique, inverse = trimesh.grouping.unique_rows(mesh.faces)
    mesh.update_faces(unique)
  return mesh

"""
if "def remove_duplicate_faces_compat(mesh):" not in text:
    marker = "\n\ndef trimesh_clean(mesh):\n"
    if marker not in text:
        raise SystemExit("cannot find trimesh_clean insertion point in Utils.py")
    text = text.replace(marker, "\n" + helper + "\ndef trimesh_clean(mesh):\n")
text = text.replace("  mesh.remove_duplicate_faces()\n", "  remove_duplicate_faces_compat(mesh)\n")
utils.write_text(text, encoding="utf-8")

nerf = Path("nerf_runner.py")
text = nerf.read_text(encoding="utf-8")
text = text.replace("    mesh.remove_duplicate_faces()\n", "    remove_duplicate_faces_compat(mesh)\n")
nerf.write_text(text, encoding="utf-8")
PY

"$PREFIX/bin/python" run_custom.py \
  --mode run_video \
  --video_dir "$DATASET_DIR" \
  --out_folder "$OUTPUT_DIR" \
  --use_segmenter 0 \
  --use_gui 0 \
  --stride 1 \
  --debug_level "$DEBUG_LEVEL"

test -f "$OUTPUT_DIR/config_bundletrack.yml"
test -d "$OUTPUT_DIR/ob_in_cam"
test -f "$OUTPUT_DIR/mesh_cleaned.obj"
test -f "$OUTPUT_DIR/textured_mesh.obj"

echo BUNDLESDF_RUN_V3_OK
