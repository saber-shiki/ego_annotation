#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/instantmesh_work}
REPO=${REPO:-$WORK_ROOT/InstantMesh}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_instantmesh_prior_outputs}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/instantmesh_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_instantmesh_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/TencentARC/InstantMesh.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin main
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/instantmesh_git_head.txt"
python3 -m pip install --user virtualenv
rm -rf "$ENV_DIR"
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools==69.5.1 wheel ninja
"$ENV_PY" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
"$ENV_PY" -m pip install xformers==0.0.22.post7
cat > "$OUT_ROOT/instantmesh_constraints.txt" <<'CONSTRAINTS'
accelerate==0.23.0
bitsandbytes==0.41.1
huggingface-hub==0.17.3
numpy==1.26.4
torch==2.1.0
torchaudio==2.1.0
torchvision==0.16.0
triton==2.1.0
xformers==0.0.22.post7
CONSTRAINTS
"$ENV_PY" -m pip install --no-build-isolation -c "$OUT_ROOT/instantmesh_constraints.txt" -r requirements.txt
"$ENV_PY" - <<'PY'
import diffusers, torch, torchvision, trimesh
from diffusers import DiffusionPipeline
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("torchvision", torchvision.__version__, "diffusers", diffusers.__version__)
print("pipeline_import", DiffusionPipeline.__name__)
PY
EOF
chmod +x "$OUT_ROOT/setup_instantmesh_v7.sh"

cat > "$OUT_ROOT/run_instantmesh_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
cd "$REPO"
if [[ ! -x "$ENV_PY" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_instantmesh_v7.sh"
fi
INPUT_DIR="$OUT_ROOT/input_images"
RUN_ROOT="$OUT_ROOT/generated"
mkdir -p "\$INPUT_DIR" "\$RUN_ROOT"
cp "$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_crop_rgba.png" "\$INPUT_DIR/wild_rice_2539.png"
cp "$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_crop_rgba.png" "\$INPUT_DIR/wild_rice_2545.png"
cp "$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_crop_rgba.png" "\$INPUT_DIR/trash_0880.png"
cp "$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame750/frame_000750_crop_rgba.png" "\$INPUT_DIR/mop_0750.png"
"$ENV_PY" run.py configs/instant-mesh-large.yaml "\$INPUT_DIR" \\
  --output_path "\$RUN_ROOT" \\
  --no_rembg \\
  --diffusion_steps 75
"$ENV_PY" - <<'PY'
import json
from pathlib import Path
import trimesh

root = Path("$OUT_ROOT")
mesh_dir = root / "generated" / "instant-mesh-large" / "meshes"
cases = []
for mesh_path in sorted(mesh_dir.glob("*.obj")):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid InstantMesh output: {mesh_path}")
    cases.append({
        "name": mesh_path.stem,
        "mesh": str(mesh_path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "extent_model_units": [float(x) for x in mesh.extents],
    })
if not cases:
    raise RuntimeError(f"no InstantMesh OBJ outputs under {mesh_dir}")
report = {
    "status": "ok",
    "method": "instantmesh_v7_remote_job",
    "repo": "$REPO",
    "output_root": str(root),
    "cases": cases,
}
(root / "qc_instantmesh_shape_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
EOF
chmod +x "$OUT_ROOT/run_instantmesh_v7.sh"

cat > "$OUT_ROOT/wait_and_run_instantmesh_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_instantmesh_v7.sh"
MAX_USED_MB="\${MAX_USED_MB:-$MAX_USED_MB}"
POLL_SECONDS="\${POLL_SECONDS:-$POLL_SECONDS}"
GPU_SELECT_LOCK="\${GPU_SELECT_LOCK:-$GPU_SELECT_LOCK}"
GPU_LOCK_DIR="\${GPU_LOCK_DIR:-$GPU_LOCK_DIR}"
mkdir -p "\$GPU_LOCK_DIR"
while true; do
  GPU_ID=""
  exec 9>"\$GPU_SELECT_LOCK"
  flock -x 9
  while IFS=, read -r gpu_idx used_mb; do
    gpu_idx="\${gpu_idx//[[:space:]]/}"
    used_mb="\${used_mb//[[:space:]]/}"
    if [[ -n "\$gpu_idx" && -n "\$used_mb" && "\$used_mb" -le "\$MAX_USED_MB" ]]; then
      exec 8>"\$GPU_LOCK_DIR/gpu_\${gpu_idx}.lock"
      if flock -n 8; then
        GPU_ID="\$gpu_idx"
        break
      fi
      exec 8>&-
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [[ -n "\$GPU_ID" ]]; then
    export GPU_ID
    flock -u 9
    exec 9>&-
    date '+%Y-%m-%d %H:%M:%S selected GPU '"\$GPU_ID"
    exec bash "\$RUN_SCRIPT"
  fi
  flock -u 9
  exec 9>&-
  date '+%Y-%m-%d %H:%M:%S no GPU below memory threshold; sleeping'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep "\$POLL_SECONDS"
done
EOF
chmod +x "$OUT_ROOT/wait_and_run_instantmesh_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_instantmesh_v7.sh" \
  "$OUT_ROOT/run_instantmesh_v7.sh" \
  "$OUT_ROOT/wait_and_run_instantmesh_v7.sh"
