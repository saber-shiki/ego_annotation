"""Bounded server CPU launch; two independent case writers, one immutable snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--input-list', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    a = p.parse_args(); root = a.output_root.resolve(); root.mkdir(parents=True, exist_ok=False)
    items = json.loads(a.input_list.read_text())
    snapshot = a.snapshot.resolve(); code = {str(f.relative_to(snapshot)): sha(f) for f in sorted(snapshot.rglob('*.py'))}
    inputs = [{'case': x['case'], 'source': x['source'], 'baseline': str(Path(x['output'])/'coarse_scene.json'),
               'output': str(root/x['case'])} for x in items]
    (root/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    (root/'launch_contract.json').write_text(json.dumps({'snapshot': str(snapshot), 'python': str(a.python),
        'compute': 'dexrobot-Standard-PC-Q35-ICH9-2009 / CPU-only', 'threads_per_case': 1,
        'maximum_case_seconds': 900, 'maximum_concurrent_cases': 2, 'code_sha256': code}, indent=2)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def run(item):
        command = [str(a.python), '-m', 'preprocessing.surface_ablation', '--source-observations', item['source'],
                   '--baseline', item['baseline'], '--output-dir', item['output'], '--max-nfev', '24']
        try:
            with (root/(item['case']+'.log')).open('w') as log:
                result = subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        (root/(item['case']+'.exit')).write_text(str(code)+'\n')
        return code

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(run, inputs))
    changed = [f for f, value in code.items() if sha(snapshot/f) != value]
    status = next((c for c in codes if c), 0) or (1 if changed else 0)
    (root/'batch.exit').write_text(str(status)+'\n')
    if status == 0:
        receipts = []
        for item in inputs:
            path = Path(item['output'])
            done = json.loads((path/'completed.json').read_text())
            receipts.append({'case': item['case'], 'completed_sha256': sha(path/'completed.json'),
                'seconds': done['seconds'], 'arms': done['arms'],
                'render_sha256': {f.name: sha(f) for f in sorted((path/'renders').glob('*.mp4'))}})
        (root/'prediction_freeze.json').write_text(json.dumps({'code_unchanged': True, 'cases': receipts,
            'all_predictions_completed_before_evaluation': True}, indent=2)+'\n')
    sys.exit(status)


if __name__ == '__main__':
    main()
