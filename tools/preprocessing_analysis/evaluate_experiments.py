"""Post-freeze experiment analysis. GT only for fixed-gauge diagnostic oracle, never prediction.
All candidates and chosen paths must already be immutable and completed.
"""
from pathlib import Path
import argparse,json,sys,hashlib,subprocess
import numpy as np
from scipy.spatial.transform import Rotation


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 return {'n':len(a),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max()),'mean':float(np.mean(a))} if len(a) else {'n':0}
def gaps(mask):
 result=[];a=None
 for i,v in enumerate(list(mask)+[False]):
  if v and a is None:a=i
  if not v and a is not None:result.append((a,i));a=None
 return result


def evaluate(run,snapshot,output):
 if (run/'batch.exit').read_text().strip()!='0':raise ValueError('prediction batch must be frozen')
 output.mkdir(parents=True,exist_ok=False)
 audit=Path('/mnt/user-home/kupingxin/ego_annotation_experiments/hot3d_s3_pose_audit_20260927');sys.path.insert(0,str(audit))
 from evaluate import load_case,pose
 reports=[]
 for item in read(run/'inputs.json'):
  case=item['case'];folder=Path(item['output']);done=read(folder/'completed.json');e=read(folder/'candidates.json');src=Path(item['source'])
  assert sha(folder/'experiment.npz')==done['experiment_archive_sha256']
  assert sha(src)==done['source_sha256']
  for p,r in done['source_reads'].items():assert sha(p)==r['sha256'],p
  for rel,h in done['code_sha256'].items():assert sha(snapshot/'preprocessing'/rel)==h
  z=np.load(folder/'experiment.npz',allow_pickle=False);baseline=np.load(Path(item['baseline'])/'coarse_scene.npz',allow_pickle=False)
  # Camera matrices are float32-derived and not bitwise orthogonal. The producer
  # converts camera->world once; quantify its round-trip tolerance explicitly.
  roundtrip_R=float(np.max(np.abs(z['R_candidates'][:,0]-baseline['object_rotation'])))
  roundtrip_t=float(np.max(np.abs(z['t_candidates'][:,0]-baseline['object_translation_world'])))
  np.testing.assert_allclose(z['R_candidates'][:,0],baseline['object_rotation'],rtol=0,atol=1e-6)
  np.testing.assert_allclose(z['t_candidates'][:,0],baseline['object_translation_world'],rtol=0,atol=1e-6)
  np.testing.assert_array_equal(z['faces'],baseline['faces']);np.testing.assert_array_equal(z['vertices_source'],baseline['source_vertices_canonical'])
  videos=[]
  for v in ['mechanism','leave_one','candidates']:
   p=folder/(v+'.mp4');probe=read_json_command(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration','-of','json',str(p)])['streams'][0]
   assert probe['codec_name']=='h264' and int(probe['nb_read_frames'])==len(e)
   decode=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(p),'-threads','1','-f','null','-'],capture_output=True,text=True)
   assert decode.returncode==0 and not decode.stderr,decode.stderr
   videos.append({'path':str(p),**probe})
  c=load_case(case);a=c['m'];A=c['D'][a]@np.linalg.inv(c['C'][a]);Panchor=pose(baseline['object_rotation'][a],baseline['object_translation_world'][a]);B=np.linalg.inv(A@Panchor)@c['G'][a]
  R=z['R_candidates'];t=z['t_candidates'];valid=z['candidate_valid'];ids=z['frame_idx'];n=len(ids);angle=np.full((n,4),np.nan);trans=np.full((n,4),np.nan)
  refit_angle=[];refit_trans=[]
  for i,fid in enumerate(ids):
   fid=int(fid);G=np.linalg.inv(c['D'][fid])@c['G'][fid]
   def errors(rr,tt):
    pred=np.linalg.inv(c['C'][fid])@pose(rr,tt)@B
    return float(np.degrees(Rotation.from_matrix(pred[:3,:3]@G[:3,:3].T).magnitude())),float(np.linalg.norm(pred[:3,3]-G[:3,3])*1000)
   for j in range(4):
    if valid[i,j]:angle[i,j],trans[i,j]=errors(R[i,j],t[i,j])
   ar,tr=errors(z['R_scale_refit'][i],z['t_scale_refit'][i]);refit_angle.append(ar);refit_trans.append(tr)
  eligible=np.array([c['eligible'][int(fid)] for fid in ids]);ii=np.arange(n);mode_metrics={};rows=[]
  oracle=np.nanargmin(angle,axis=1);oracle_error=angle[ii,oracle];multi=z['choice_multi'];selected=angle[ii,multi]
  for mode in done['modes']:
   if mode=='scale_refit':ar=np.array(refit_angle);tr=np.array(refit_trans);pos=z['t_scale_refit']
   else:
    choice=z['choice_'+mode];ar=angle[ii,choice];tr=trans[ii,choice];pos=t[ii,choice]
   metric={'rotation_error_deg':summary(ar[eligible]),'translation_error_mm':summary(tr[eligible]),
     'world_step_mm':summary(np.linalg.norm(np.diff(pos,axis=0),axis=1)*1000)}
   if mode!='scale_refit':
    metric['nonbaseline_frames']=int(np.count_nonzero(z['choice_'+mode]))
    metric['changed_from_multi_frames']=int(np.count_nonzero(z['choice_'+mode]!=multi))
    metric['selected_surface_coverage']=summary([e[i]['scores'][int(z['choice_'+mode][i])]['surface_coverage'] for i in range(n) if e[i]['metric_available']])
    metric['selected_mask_iou']=summary([1.-e[i]['scores'][int(z['choice_'+mode][i])]['mask'] for i in range(n) if e[i]['scores'][int(z['choice_'+mode][i])]['mask'] is not None])
   mode_metrics[mode]=metric
  opportunities=eligible&(angle[:,0]-oracle_error>10.)
  captured=opportunities&(angle[:,0]-selected>5.)
  harmed=eligible&(selected-angle[:,0]>5.)
  for i,fid in enumerate(ids):
   rows.append({'frame_idx':int(fid),'GT_eligible':bool(eligible[i]),'candidate_rotation_errors_deg':[float(v) if np.isfinite(v) else None for v in angle[i]],
     'diagnostic_oracle_index':int(oracle[i]),'baseline_error_deg':float(angle[i,0]),'selected_error_deg':float(selected[i]),'oracle_error_deg':float(oracle_error[i]),
     'selected_index':int(multi[i]),'metric_available':bool(z['source_metric_eligible'][i]),'candidate_rotation_spread_deg':e[i]['candidate_rotation_spread_deg'],
     'candidate_translation_spread_mm':e[i]['candidate_translation_spread_mm'],'binding':e[i]['binding']})
  gap_rows=[]
  for lo,hi in gaps(~z['source_metric_eligible']):
   gap_rows.append({'start_frame_idx':int(ids[lo]),'end_frame_idx':int(ids[hi-1]),'length':hi-lo,
      'candidate_rotation_spread_deg':summary([e[i]['candidate_rotation_spread_deg'] for i in range(lo,hi)]),
      'candidate_translation_spread_mm':summary([e[i]['candidate_translation_spread_mm'] for i in range(lo,hi)]),
      'baseline_error_deg':summary(angle[lo:hi,0]),'multi_error_deg':summary(selected[lo:hi]),'oracle_error_deg':summary(oracle_error[lo:hi]),
      'uncertainty_is_candidate_spread_not_calibrated_covariance':True})
  report={'case':case,'modes':mode_metrics,'oracle_rotation_error_deg':summary(oracle_error[eligible]),
    'baseline_over_30deg_frames':int(np.count_nonzero(eligible&(angle[:,0]>30.))),
    'oracle_over_30deg_frames':int(np.count_nonzero(eligible&(oracle_error>30.))),
    'oracle_improvement_over_10deg_opportunity_frames':int(opportunities.sum()),'captured_over_5deg_frames':int(captured.sum()),
    'multi_worse_than_baseline_over_5deg_frames':int(harmed.sum()),'retained_candidate_count':summary(valid.sum(1)),
    'train_support_removed_by_lifecycle':summary([x['binding']['train_observations']-x['binding']['lifecycle_train_observations'] for x in e]),
    'holdout_count':summary([x['binding']['holdout_observations'] for x in e]),
    'selected_score_margin':summary([x['best_second_score_margin'] for x in e if x['best_second_score_margin'] is not None]),
    'svd_min_over_max':summary([x['observability']['min_over_max'] for x in e if x['observability']['available']]),
    'baseline_scale':float(z['baseline_scale']),'refit_scale':float(z['refit_scale']),
    'refit_relative_scale_change_percent':float((z['refit_scale']/z['baseline_scale']-1.)*100.),
    'gaps':gap_rows,'videos':videos,'source_and_snapshot_verified':True,'baseline_roundtrip_max_R_difference':roundtrip_R,'baseline_roundtrip_max_t_difference_m':roundtrip_t,'prediction_archive_sha256':done['experiment_archive_sha256'],
    'fixed_gauge':'same baseline anchor coordinate mapping for every mode/candidate, no per-candidate GT fit',
    'oracle_scope':'per-frame diagnostic orientation lower bound within this finite bank, not a realizable trajectory, not global oracle, not symmetry-adjusted',
    'holdout_scope':done['holdout_scope'],'seconds':done['seconds'],'rows':rows}
  write(output/(case+'.json'),report);reports.append({k:v for k,v in report.items() if k!='rows'})
  print(json.dumps({'case':case,'baseline_rot':mode_metrics['baseline']['rotation_error_deg']['median'],'lifecycle_rot':mode_metrics['lifecycle']['rotation_error_deg']['median'],
    'multi_rot':mode_metrics['multi']['rotation_error_deg']['median'],'oracle_rot':report['oracle_rotation_error_deg']['median'],
    'opportunities':report['oracle_improvement_over_10deg_opportunity_frames'],'captured':report['captured_over_5deg_frames'],
    'harmed':report['multi_worse_than_baseline_over_5deg_frames'],'scale_percent':report['refit_relative_scale_change_percent']}),flush=True)
 write(output/'summary.json',reports)

def read_json_command(cmd):return json.loads(subprocess.check_output(cmd,text=True))
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();evaluate(a.run,a.snapshot,a.output)
