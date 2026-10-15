"""Read-only object-first outcome review. No modification of prediction exports."""
from pathlib import Path
import argparse,json,hashlib,sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.rl_experiments.object_only_learning import geometry_check
from tools.rl_experiments.preprocessed_learning import export_npz


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def inspect(run,output):
    summary=json.loads((run/'summary.json').read_text())
    zero=json.loads((run/'zero/evaluation.json').read_text())
    cfg=json.loads((run/'run_config.json').read_text())['config']
    assessments=[]
    for row in summary['checkpoints']:
        checkpoint=Path(row['checkpoint']);provenance=json.loads(checkpoint.with_name('agent.provenance.json').read_text())
        if sha(checkpoint)!=provenance['checkpoint']['sha256']:raise RuntimeError('checkpoint provenance hash mismatch')
        evaluation=json.loads(Path(row['evaluation']).read_text())
        manifest_path=Path(evaluation['manifest']);manifest=json.loads(manifest_path.read_text())
        if manifest['executed_frame_count']!=150 or not manifest['preprocessors_frozen_during_evaluation']:raise RuntimeError('incomplete/nonfrozen policy rollout')
        for artifact in manifest['episodes'][0]['artifacts']:
            path=Path(artifact['path']);path=path if path.is_absolute() else manifest_path.parent/path
            if sha(path)!=artifact['sha256']:raise RuntimeError('rollout export hash mismatch')
        g=geometry_check(export_npz(row),zero['npz'],cfg)
        named=evaluation['named_families'];jn=evaluation['objective']
        contact_delta=sum(v-zero['named_families'].get(k,0) for k,v in named.items() if k.startswith('contact/'))
        assessments.append({'steps':row['steps'],'raw_J':jn,'raw_delta_vs_zero':jn-zero['objective'],
            'relative_return':evaluation['relative_return'],'factor_delta_vs_zero':{k:v-zero['named_families'].get(k,0) for k,v in named.items()},
            'contact_delta':contact_delta,'object_J':sum(v for k,v in named.items() if k.startswith('object/')),
            'geometry':g,'policy_latent_std':row['policy_latent_std'],'artifact_hashes_verified':True})
    metrics=json.loads((run/'renders/render_measurements.json').read_text())
    out={'scope':'same_video_posthoc_no_GT','zero_raw_J':zero['objective'],
        'zero_object_J':sum(v for k,v in zero['named_families'].items() if k.startswith('object/')),
        'checkpoints':assessments,'render_measurements':{k:v for k,v in metrics.items() if k!='per_frame_raw_IoU'},
        'baseline_wins':not summary['improves_zero'],'source_freezing_receipt':json.loads((run/'completed.json').read_text())}
    output.mkdir(parents=True,exist_ok=False)
    (output/'assessment.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'raw_objectives':[(r['steps'],r['raw_J']) for r in assessments],'baseline_wins':out['baseline_wins']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.run,a.output)
