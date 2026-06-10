#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 REMOTE_ROOT INPUT_ROOT OUTPUT_ROOT JOB_ID [JOB_ID ...]" >&2
  exit 2
fi

ROOT="$1"
INPUT_ROOT="$2"
OUTPUT_ROOT="$3"
shift 3

LOG_ROOT="$ROOT/logs"
RUNNER="$ROOT/remote_run_bundlesdf_v3.sh"
LOCK="$ROOT/.v17_bundlesdf_runner.lock"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

choose_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      gsub(/ /, "", $3);
      if ($2 < 2000 && $3 < 20) {
        print $1;
        exit
      }
    }'
}

wait_for_gpu() {
  local gpu=""
  while true; do
    gpu="$(choose_gpu)"
    if [ -n "$gpu" ]; then
      printf '%s\n' "$gpu"
      return 0
    fi
    log "waiting_for_free_gpu"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2
    sleep 300
  done
}

if [ ! -x "$RUNNER" ]; then
  echo "missing executable BundleSDF runner: $RUNNER" >&2
  exit 1
fi

for job_id in "$@"; do
  dataset="$INPUT_ROOT/$job_id"
  out="$OUTPUT_ROOT/$job_id"
  run_log="$LOG_ROOT/v17_bundlesdf_${job_id}.log"
  if [ -f "$out/mesh_cleaned.obj" ] && [ -d "$out/ob_in_cam" ]; then
    log "skip_existing $job_id"
    continue
  fi
  for rel in rgb depth masks cam_K.txt manifest.json v17_geometry_reconstruction_job.json; do
    if [ ! -e "$dataset/$rel" ]; then
      echo "missing job input $dataset/$rel" >&2
      exit 1
    fi
  done
  gpu="$(wait_for_gpu)"
  log "start $job_id gpu=$gpu"
  rm -rf "$out"
  mkdir -p "$out"
  if flock "$LOCK" "$RUNNER" "$ROOT" "$dataset" "$out" "$gpu" 2.5 1 >"$run_log" 2>&1; then
    log "done $job_id gpu=$gpu log=$run_log"
  else
    log "failed $job_id gpu=$gpu log=$run_log"
    exit 1
  fi
done

log "queue_complete"
