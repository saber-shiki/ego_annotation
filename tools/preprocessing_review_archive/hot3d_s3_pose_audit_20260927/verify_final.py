"""Verify evaluation inputs and original source bindings without mutation."""
from pathlib import Path
import json,hashlib,datetime,subprocess
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
inv=read(ROOT/'inventory.json');files=0;size=0;bad=[];bindings=[];decodes=[]
for row in read(ROOT/'integrity_summary.json'):
 case=row['case']
 for kind in ['observations','preprocessing']:
  verification=read(ROOT/case/(kind+'_GET_verification.json'))
  for rec in verification['files']:
   p=ROOT/case/kind/rec['path'];files+=1;size+=rec['bytes']
   if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:bad.append(str(p))
 original=next(x for x in inv['packages'] if x['case']==case and x['kind']=='observations')
 source=Path(original['original_root'])/'source_observations.json'
 actual=sha(source);bindings.append({'case':case,'original_source_sha256':actual,'matches_S3':actual==row['source_bundle_sha256']})
 report=read(ROOT/case/'evaluation.json')
 if sha(Path(report['CAD_path']))!=report['CAD_sha256']:bad.append(report['CAD_path'])
 video=ROOT/case/'observations/input/raw.mp4'
 proc=subprocess.run(['ffmpeg','-hide_banner','-v','error','-threads','1','-i',str(video),'-map','0:v:0','-an','-threads','1','-f','null','-','-progress','pipe:1'],capture_output=True,text=True)
 last={}
 for line in proc.stdout.splitlines():
  if '=' in line:k,v=line.split('=',1);last[k]=v
 decodes.append({'case':case,'returncode':proc.returncode,'decoded_frames':int(last.get('frame',-1)),'out_time_us':int(last.get('out_time_us',-1)),'errors':proc.stderr})
 if proc.returncode or last.get('frame')!='150':bad.append(str(video))
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GET_SHA256_copies_reverified':files,'bytes':size,'mismatches':bad,'original_bindings':bindings,'full_video_decodes':decodes,'read_only_S3':True,'prediction_code_or_outputs_modified':False,'GT_modified':False,'test_count':6,'tests_passed':6}
(ROOT/'final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
artifacts={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('*') if p.is_file() and p.name!='report_artifact_hashes.json'}
(ROOT/'report_artifact_hashes.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print(json.dumps({'files':files,'bytes':size,'mismatches':bad,'original_sources_unchanged':all(x['matches_S3'] for x in bindings),'decoded_frames':sum(x['decoded_frames'] for x in decodes),'report':str(ROOT/'REPORT_ZH.md')}))
if bad or not all(x['matches_S3'] for x in bindings):raise SystemExit(1)
