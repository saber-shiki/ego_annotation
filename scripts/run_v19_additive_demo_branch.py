#!/usr/bin/env python3
"""Run the additive visual demo branch from a byte-bound current-run contract."""
from __future__ import annotations
import argparse,hashlib,json,os,socket,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def load(path):
 value=json.loads(Path(path).read_text())
 if not isinstance(value,dict):raise ValueError('JSON object required')
 return value
def file(path,root):
 p=Path(path).resolve()
 if not p.is_relative_to(root) or not p.is_file():raise ValueError(f'missing/out-of-run demo input: {path}')
 return p
def python(path):
 p=Path(os.path.abspath(Path(path).expanduser()))
 if not p.is_file() or not os.access(p,os.X_OK):raise ValueError('demo interpreter unavailable')
 return p

def gpu_probe(index,minimum_free):
 q=['nvidia-smi',f'--id={index}','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits']
 row=subprocess.run(q,check=True,capture_output=True,text=True).stdout.strip();parts=[p.strip() for p in row.split(',')]
 if len(parts)!=7 or int(parts[0])!=index:raise RuntimeError('invalid demo GPU probe')
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.splitlines()
 record={'gpu':parts,'existing_processes':[x for x in apps if parts[1] in x],
         'policy':'shared idle card explicitly user-authorized; require utilization<=5 and free memory floor'}
 if int(parts[6])>5 or int(parts[4])<minimum_free:raise RuntimeError('authorized demo GPU unavailable: '+json.dumps(record))
 return record

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);a=p.parse_args()
 contract=load(a.contract);root=Path(contract['run_root']).resolve()
 if a.contract.resolve()!=root/'state/additive_demo/launch_contract.json':raise ValueError('unexpected demo contract path')
 if contract.get('mode')!='additive_demo' or contract.get('authority')!='demo_only_visual_hypothesis':raise ValueError('invalid demo authority')
 if socket.gethostname()!=contract['compute_host']:raise ValueError('demo host mismatch')
 inputs={k:file(v['path'],root) for k,v in contract['inputs'].items()}
 for k,pth in inputs.items():
  if sha(pth)!=contract['inputs'][k]['sha256']:raise ValueError(f'demo input changed: {k}')
 for pth in [root/'state/demo_inputs/milk_carton_pose_mesh.json',root/'measurements/demo_visual/milk_carton',root/'renders/milk_carton_demo',root/'state/additive_demo/branch_result.json']:
  if pth.exists():raise ValueError(f'demo output already exists: {pth}')
 cfg=load(contract['runtime_config']);py=python(cfg['main_python']);gpu=int(contract['physical_gpu'])
 probe=gpu_probe(gpu,int(contract['minimum_free_mib']))
 logdir=root/'logs/demo_visual';logdir.mkdir(parents=True,exist_ok=True);events=[]
 env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
 demo_pose=root/'state/demo_inputs/milk_carton_pose_mesh.json';demo=root/'measurements/demo_visual/milk_carton';render=root/'renders/milk_carton_demo'
 commands=[('D00_demo_binding',[py,ROOT/'scripts/build_v19_demo_pose_binding.py','--pose-report',inputs['p15'],'--completion-report',inputs['completion'],'--output',demo_pose],[demo_pose]),
 ('D01_demo_fit',[py,ROOT/'scripts/fit_v20_visual_contact_demo.py','--annotations',inputs['annotations'],'--pose-report',demo_pose,'--mano-bridge',inputs['mano_bridge'],'--hawor-params',inputs['hawor'],'--mano-models',cfg['mano_models'],'--object-id','milk_carton','--device','cuda:0','--interaction-judgments',inputs['judgment'],'--render-width',960,'--output-dir',demo],[demo/'demo_state.npz',demo/'fit_report.json']),
 ('D02_demo_render',[py,ROOT/'scripts/render_v20_visual_contact_demo.py','--state',demo/'demo_state.npz','--source-video',inputs['source_video'],'--output-dir',render],[render/'interaction_demo.mp4',render/'contact_detail.mp4',render/'interaction_demo.rrd',render/'render_report.json'])]
 try:
  for name,argv,outputs in commands:
   start=time.monotonic();log=logdir/(name+'.log')
   with log.open('x') as f:res=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   event={'phase':name,'argv':list(map(str,argv)),'returncode':res.returncode,'elapsed_s':time.monotonic()-start,'log':str(log),'outputs':list(map(str,outputs))};events.append(event)
   with (logdir/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
   if res.returncode:raise RuntimeError(f'{name} failed: {log}')
   for output in outputs:
    if not output.is_file():raise RuntimeError(f'{name} output missing: {output}')
 except Exception as exc:
  fail=root/'state/additive_demo/failure.json';fail.write_text(json.dumps({'error':str(exc),'events':events,'authority':'demo_only'},indent=2)+'\n');raise
 result={'schema':'v19_additive_demo_branch_v1','status':'completed_pending_visual_review','authority':'demo_only_visual_hypothesis','core_state_modified':False,'input_contract':str(a.contract),'input_contract_sha256':sha(a.contract),'gpu_preflight':probe,'events':events,'outputs':{str(p.relative_to(root)):sha(p) for _,_,paths in commands for p in paths},'final_package_assembled':False}
 target=root/'state/additive_demo/branch_result.json';target.write_text(json.dumps(result,indent=2)+'\n');print(target)
if __name__=='__main__':main()
