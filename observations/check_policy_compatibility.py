"""Read-only milestone3 readiness check; never trains or bypasses v2 guards."""
import argparse
import json
from pathlib import Path
from data.bundle import EpisodeBundle
from data.build_index import build_index,INDEX_SCHEMA
from data.dataset import EpisodeDataset
from observations.contracts import SCHEMA_V2
from envs.v2_consumption import CONTRACT
from actions.space import ActionLayout
from envs.observation import ObservationLayout
from utils.state import sha


def check_compatibility(source,checkpoint_provenance):
    source=Path(source).resolve();provenance_path=Path(checkpoint_provenance).resolve()
    # Verifies the existing prediction package's hashes; no env reset/model load.
    bundle=EpisodeBundle.open(source)
    if bundle.manifest['schema']!=SCHEMA_V2:raise ValueError('milestone3 check requires real v2 source')
    before=sha(source);provenance=json.loads(provenance_path.read_text())
    if provenance.get('schema')!='ego.annotation.skrl_checkpoint/2.0.0':raise ValueError('unsupported checkpoint provenance')
    result={'source':str(source),'source_sha256':before,'checkpoint_provenance':str(provenance_path),
        'checkpoint_provenance_sha256':sha(provenance_path),'checks':[],'blockers':[],
        'training_executed':False,'checkpoint_loaded':False,'PPO_constructed':False}
    for name,call in [('official_index',lambda:build_index([source])),
        ('dataset_direct_index',lambda:EpisodeDataset({'schema':INDEX_SCHEMA,'episodes':[{
            'source_video_id':bundle.source_video_id,'manifest':str(source),'manifest_sha256':before}]})[0])]:
        try:call()
        except ValueError as error:
            result['checks'].append({'name':name,'status':'refused','reason':str(error)})
            result['blockers'].append(name+': explicit v2 training adapter not implemented')
        else:
            result['checks'].append({'name':name,'status':'unexpectedly_accepted','requires_review':True})
            result['blockers'].append(name+': existing guard unexpectedly allowed v2; not proof of feature consumption')
    compatibility=provenance.get('compatibility',{});declared=compatibility.get('dataset',{}).get('bundle_schemas',[])
    result['old_checkpoint']={'source_schemas':declared,'observation_space':compatibility.get('observation_space'),
        'action_space':compatibility.get('action_space'),'environment_class':compatibility.get('environment_class')}
    if SCHEMA_V2 not in declared:result['blockers'].append('old checkpoint declares v1 data/environment, not v2')
    layout=ActionLayout(object_slots=('object:0',),contact_variables_per_hand=5)
    result['unchanged_default_spaces']={'observation':ObservationLayout(layout.size,contact_variables_per_hand=5).size,'actions':layout.size}
    result['controlled_adapter_contract']=CONTRACT
    for key in ('policy_observation_layout','policy_action_layout','reward_profile'):
        if CONTRACT[key] is None:result['blockers'].append('controlled adapter has no reviewed '+key)
    result['blockers']+=['full visual hand decoder / shared object scale not integrated',
        'existing actor input packing does not consume v2 motion/geometry/typed semantics',
        'controlled physical-unit injections are not a PPO action contract']
    result['status']='blocked_not_trainable'
    result['source_unchanged']=sha(source)==before
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-observations',type=Path,required=True)
    p.add_argument('--checkpoint-provenance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError(a.output)
    result=check_compatibility(a.source_observations,a.checkpoint_provenance)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'blocker_count':len(result['blockers']),'report':str(a.output),'training_executed':False},ensure_ascii=False))
    return 2


if __name__=='__main__':raise SystemExit(main())
