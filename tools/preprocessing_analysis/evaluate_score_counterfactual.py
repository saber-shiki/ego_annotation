"""Evaluation-only counterfactual of score/model-binding compatibility.
No counterfactual enters candidate generation, selection, or saved predictions.
One baseline anchor defines native<->GT gauge; not an absolute CAD/native fit.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import cv2


def main(a):
 if (a.run/'batch.exit').read_text().strip()!='0':raise ValueError('freeze prediction first')
 a.output.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(a.snapshot))
 from preprocessing.inputs import load_inputs,sha
 from preprocessing.geometry import TriangleMesh,camera_pose
 from preprocessing.factors import hand_depths
 from preprocessing.observation_experiment import tracklets,score_candidate,choose
 sys.path.insert(0,'/mnt/user-home/kupingxin/ego_annotation_experiments/hot3d_s3_pose_audit_20260927')
 from evaluate import load_case,pose
 cv2.setNumThreads(1);reports=[]
 for item in json.loads((a.run/'inputs.json').read_text()):
  if a.case and item['case']!=a.case:continue
  out=Path(item['output']);done=json.loads((out/'completed.json').read_text());rows=json.loads((out/'candidates.json').read_text());case=item['case']
  assert sha(out/'experiment.npz')==done['experiment_archive_sha256']
  inp=load_inputs(item['source'],480,read_owned_masks=True);b=np.load(Path(item['baseline'])/'coarse_scene.npz',allow_pickle=False)
  mesh=TriangleMesh(b['vertices_canonical'],b['faces']);hd=hand_depths(inp,128);e,_,tracks=tracklets(inp,mesh,b['object_rotation'],b['object_translation_world'])
  z=np.load(out/'tracklets.npz',allow_pickle=False);np.testing.assert_array_equal(tracks,z['rows'])
  c=load_case(case);anchor=c['m'];A=c['D'][anchor]@np.linalg.inv(c['C'][anchor]);B=np.linalg.inv(A@pose(b['object_rotation'][anchor],b['object_translation_world'][anchor]))@c['G'][anchor]
  measured=[];base_favored={k:[] for k in ['rgb','mask','surface','total']};all_costs={k:[] for k in ['rgb','mask','surface','total']}
  orientation_favored={k:[] for k in base_favored};orientation_costs={k:[] for k in base_favored}
  for i,fid in enumerate(inp.ids):
   fid=int(fid)
   if not c['eligible'][fid]:continue
   # Exact GT RELATIVE motion mapped through the single frozen native anchor gauge.
   P=np.linalg.inv(c['D'][fid])@c['G'][fid]@np.linalg.inv(B)
   gs=score_candidate(inp,mesh,hd,i,e[i],P[:3,:3],P[:3,3]);bs=rows[i]['scores'][0]
   _,base_t=camera_pose(b['object_rotation'][i],b['object_translation_world'][i],inp.cameras[i])
   rs=score_candidate(inp,mesh,hd,i,e[i],P[:3,:3],base_t)
   _,total=choose([bs,gs],'multi');_,rtotal=choose([bs,rs],'multi')
   d={'frame_idx':fid,'baseline_scores':bs,'GT_relative_motion_counterfactual_scores':gs,'totals':total.tolist(),
      'GT_relative_rotation_baseline_translation_scores':rs,'orientation_only_totals':rtotal.tolist()}
   measured.append(d)
   for key in ['rgb','mask','surface','total']:
    x,y=total if key=='total' else (bs[key],gs[key])
    if x is not None and y is not None:base_favored[key].append(bool(x<y));all_costs[key].append([float(x),float(y)])
    x,y=rtotal if key=='total' else (bs[key],rs[key])
    if x is not None and y is not None:orientation_favored[key].append(bool(x<y));orientation_costs[key].append([float(x),float(y)])
  report={'case':case,'frames':measured,'baseline_favored_fraction':{k:float(np.mean(v)) if v else None for k,v in base_favored.items()},
    'score_medians_baseline_and_counterfactual':{k:np.median(v,axis=0).tolist() if v else None for k,v in all_costs.items()},
    'orientation_only_baseline_favored_fraction':{k:float(np.mean(v)) if v else None for k,v in orientation_favored.items()},
    'orientation_only_score_medians':{k:np.median(v,axis=0).tolist() if v else None for k,v in orientation_costs.items()},
    'scope':'evaluation-only GT-relative-motion counterfactual under baseline native anchor gauge; full-motion and rotation-only-fixed-baseline-translation controls; shape mismatch, residual anchor error and depth bias remain; not absolute true pose and not a prediction candidate',
    'GT_consumed_only_after_prediction':True,'counterfactual_written_to_prediction':False,'baseline_archive_sha256':sha(Path(item['baseline'])/'coarse_scene.npz')}
  (a.output/(case+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');reports.append({k:v for k,v in report.items() if k!='frames'})
  inp.receipt.check();assert sha(out/'experiment.npz')==done['experiment_archive_sha256']
  print(json.dumps(reports[-1]),flush=True)
 (a.output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case');main(p.parse_args())
