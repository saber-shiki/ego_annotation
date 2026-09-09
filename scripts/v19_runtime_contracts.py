"""Same-run and subprocess contracts for the RGB/SAM3D prediction entrypoints."""
from pathlib import Path
import hashlib
import json
import os
import math
import socket
import subprocess
import time

CLIP_SUFFIXES = {'.json', '.npz', '.npy', '.ply', '.obj', '.glb', '.png', '.jpg', '.jpeg', '.mp4'}


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def in_run(value, root, *, must_exist=True):
    root = Path(root).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f'clip-specific path must be absolute: {value}')
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'clip-specific input escapes run root: {value}')
    if must_exist and not resolved.is_file():
        raise ValueError(f'input file missing: {value}')
    return resolved


def validate_references(value, root, *, must_exist=False, field=''):
    """Reject outside/symlink/relative clip paths, including embedded provenance.

    Model binaries (.pkl/.pt/.pth) are not clip predictions. Their separate
    config/preflight allowlist is not a license to reuse clip JSON/NPZ/meshes.
    Future output paths inside reports may not exist yet.
    """
    if isinstance(value, dict):
        for key,child in value.items():
            validate_references(child, root, must_exist=must_exist, field=str(key))
    elif isinstance(value, list):
        for child in value:
            validate_references(child, root, must_exist=must_exist, field=field)
    elif isinstance(value, str) and '\n' not in value:
        if Path(value).suffix.lower() in CLIP_SUFFIXES:
            prose = any(word in field.lower() for word in ('note','description','rationale','claim','interpretation','uncertainty','message','semantics','rule','policy'))
            if prose and any(c.isspace() for c in value) and not value.startswith(('/', './', '../')):
                return
            in_run(value, root, must_exist=must_exist)


def interpreter(value):
    # Resolving a virtualenv executable symlink would select system Python.
    path = Path(os.path.abspath(Path(value).expanduser()))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'interpreter unavailable: {path}')
    return path


def runtime_environment(config, gpu):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONDONTWRITEBYTECODE': '1'}
    if config.get('lightglue_torch_home'):
        env['TORCH_HOME'] = str(config['lightglue_torch_home'])
    return env


def run_command(key, argv, outputs, *, root, cwd, env, logdir=None, gpu=None, lineage=None, extra_inputs=(), capture_roots=()):
    """One fresh invocation; preserve failures and never continue after failure."""
    root = Path(root).resolve()
    argv = list(map(str, argv))
    outputs = [in_run(path, root, must_exist=False) for path in outputs]
    for path in outputs:
        if path.exists():
            raise ValueError(f'{key} output already exists; not a fresh stage: {path}')
    output_roots=[]
    for flag in ('--output-dir', '--output-root', '--review-dir'):
        if flag in argv:
            directory = in_run(argv[argv.index(flag) + 1], root, must_exist=False)
            output_roots.append(directory)
            if directory.exists():
                raise ValueError(f'{key} output directory already exists: {directory}')
    for value in capture_roots:
        directory=in_run(value,root,must_exist=False)
        if directory.exists():raise ValueError(f'{key} output directory already exists: {directory}')
        output_roots.append(directory)
    logdir = Path(logdir) if logdir else root / 'logs/integration'
    in_run(logdir, root, must_exist=False)
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f'{key}.log'
    start = time.monotonic()
    event = {'phase': key, 'argv': argv, 'log': str(log), 'outputs': list(map(str, outputs)),
             'execution_environment':{k:v for k,v in env.items() if k.startswith('EGO_HAWOR_') or k in {'CUDA_VISIBLE_DEVICES','TORCH_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'}}}
    try:
        inputs=lineage.inputs(argv,output_roots,extra_inputs) if lineage else {}
        with log.open('x') as handle:
            if gpu is not None:
                event['gpu_probe'] = live_gpu_probe(gpu)
                handle.write(json.dumps(event['gpu_probe']) + '\n')
                handle.flush()
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        event['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'{key} failed: {log}')
        for path in outputs:
            in_run(path, root)
        if lineage:
            lineage.success(event,inputs,outputs,output_roots)
    except Exception as exc:
        event['error'] = str(exc)
        recovery_failure = key == 'P09' and getattr(lineage, 'recovery', None) is not None
        blockers = root / ('state/runtime_recovery/P09' if recovery_failure else 'state/runtime_blockers')
        in_run(blockers, root, must_exist=False)
        blockers.mkdir(parents=True, exist_ok=True)
        failure_path = blockers / ('failure.json' if recovery_failure else f'{key}.json')
        # A stopped run's original P09 blocker is a protected recovery input.
        # Retain any recovery-attempt failure separately and never replace it.
        with failure_path.open('x' if recovery_failure else 'w') as handle:
            handle.write(json.dumps({
                **event, 'blocked_state_variable': key, 'next_required_repair': 'repair failed command/input before a new run',
            }, indent=2) + '\n')
        raise
    finally:
        event['elapsed_s'] = time.monotonic() - start
        with (logdir / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(event) + '\n')


def validate_timeline(annotations, manifest):
    a = annotations.get('frames', [])
    b = manifest.get('frames', [])
    def index(rows):
        out = {}
        for row in rows:
            idx = int(row['frame_idx'])
            if idx in out:
                raise ValueError('duplicate frame ID')
            stamp = float(row['time_s'])
            if not math.isfinite(stamp):
                raise ValueError('nonfinite frame time')
            out[idx] = stamp
        return out
    aa, bb = index(a), index(b)
    if not aa or aa.keys() != bb.keys() or any(abs(aa[k] - bb[k]) > 1e-6 for k in aa):
        raise ValueError('full annotation/manifest frame-time mismatch')
    ids = sorted(aa)
    if any(aa[b] <= aa[a] for a,b in zip(ids,ids[1:])):
        raise ValueError('frame timestamps are not strictly increasing')
    return ids


def execution_target(config, gpu):
    if config.get('compute_host') != socket.gethostname() or config.get('gpu') != gpu:
        raise ValueError('execution host/GPU differs from the preflighted target')


def live_gpu_probe(gpu, minimum_free_mib=0):
    result = subprocess.run(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=index,uuid,name,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    fields = [s.strip() for s in result.stdout.strip().split(',')]
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    if len(fields) != 5 or int(fields[0]) != gpu:
        raise RuntimeError('invalid GPU probe response')
    busy = [line for line in processes.stdout.splitlines() if fields[1] in line]
    record = {'gpu': fields, 'compute_processes': busy}
    if busy or int(fields[4]) > 0 or int(fields[3]) < minimum_free_mib:
        raise RuntimeError('configured GPU unavailable: ' + json.dumps(record))
    return record
