"""Read-only comparison of completed predictions. No CAD/GT; actual rendered triangles."""
from pathlib import Path
import sys,json,argparse,subprocess
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
SNAP=Path('/mnt/user-home/kupingxin/ego_annotation_runtime/monocular_visible_surface_20260928_jump_r2')
sys.path.insert(0,str(SNAP))
from preprocessing.inputs import load_inputs,sha
ROOT=Path(__file__).resolve().parent
OLD=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_preprocess_runs/monocular_visible_surface_20260927_r2')
NEW=OLD.with_name('monocular_visible_surface_20260928_jump_r2')
read=lambda p:json.loads(Path(p).read_text())

def statistics(x):
 x=np.asarray(x,float);return {'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(np.max(x))}

def compare(item):
 case=item['case'];new=NEW/case;old=OLD/case
 if not (new/'completed.json').is_file():raise ValueError('case prediction has not completed')
 dest=ROOT/NEW.name/case;dest.mkdir(parents=True,exist_ok=False)
 inputs=load_inputs(item['source'],480,read_owned_masks=True)
 archives={label:dict(np.load(path/'coarse_scene.npz',allow_pickle=False)) for label,path in [('old',old),('repaired',new)]}
 metrics={};frames={};rows=[]
 for label,z in archives.items():
  step=np.r_[0.,np.linalg.norm(np.diff(z['object_translation_world'],axis=0),axis=1)*1000]
  angle=np.r_[0.,np.degrees(Rotation.from_matrix(z['object_rotation'][1:]@z['object_rotation'][:-1].transpose(0,2,1)).magnitude())]
  tc=np.einsum('nji,nj->ni',z['T_world_camera'][:,:3,:3],z['object_translation_world']-z['T_world_camera'][:,:3,3])
  surface=read((old if label=='old' else new)/'visible_surface_metrics.json')['after']['frames']
  tracking=read((old if label=='old' else new)/'tracking.json')
  frames[label]=[{'frame_idx':int(fid),'world_step_mm':float(step[i]),'rotation_step_deg':float(angle[i]),'camera_translation_m':tc[i].tolist(),
    'surface_distance_m':surface[i]['visible_surface_distance_m'],'first_hit_coverage':surface[i]['first_hit_coverage'],
    'fixed_visible_IoU':surface[i]['visible_projection_IoU'],'metric_samples':surface[i]['metric_samples'],'tracking':tracking[i]} for i,fid in enumerate(z['frame_idx'])]
  metrics[label]={'world_step_mm':statistics(step[1:]),'rotation_step_deg':statistics(angle[1:]),
    'largest_position_step_frame_idx':int(z['frame_idx'][np.argmax(step)]),'largest_rotation_step_frame_idx':int(z['frame_idx'][np.argmax(angle)]),
    'fixed_visible_IoU':statistics([f['visible_projection_IoU'] for f in surface]),
    'surface_frame_median_mm':statistics([f['visible_surface_distance_m']['median']*1000 for f in surface if f['metric_samples']]),
    'uniform_scale':float(z['uniform_object_scale'])}
  metrics[label]['weak_PnP_frames']=int(sum(r.get('pnp_inliers',0)<6 for r in tracking))
  if label=='repaired':
   legacy_R=np.asarray([r['unfiltered_contour_pose']['rotation_world'] for r in tracking])
   legacy_t=np.asarray([r['unfiltered_contour_pose']['translation_world'] for r in tracking])
   baseline=np.load(old/'bootstrap.npz',allow_pickle=False)
   dr=float(np.max(np.abs(legacy_R-baseline['object_rotation'])));dt=float(np.max(np.abs(legacy_t-baseline['object_translation_world'])))
   if dr>1e-9 or dt>1e-9:raise ValueError(f'proposal state changed: {dr} {dt}')
   metrics[label]['unfiltered_proposal_matches_original_to_1e_9']=True
   metrics[label]['unfiltered_max_R_difference']=dr;metrics[label]['unfiltered_max_t_difference_m']=dt
   metrics[label]['repaired_contour_frames']=[int(r['frame']) for r in tracking if r['fit'].get('contour_candidate_rejected')]

 for i,fid in enumerate(inputs.ids):rows.append({'frame_idx':int(fid),'old':frames['old'][i],'repaired':frames['repaired'][i]})
 n=len(rows);wanted=set([0,n//3,2*n//3,n-1]);wanted.update(range(max(0,n-10),n))
 for label in ['old','repaired']:
  for key in ['largest_position_step_frame_idx','largest_rotation_step_frame_idx']:
   p=int(np.flatnonzero(inputs.ids==metrics[label][key])[0]);wanted.update(range(max(0,p-1),min(n,p+2)))
 caps=[cv2.VideoCapture(str(path/'renders/before_after_overlay.mp4')) for path in (old,new)]
 first=None;pipe=None;contact={}
 try:
  for i,fid in enumerate(inputs.ids):
   panels=[]
   for cap in caps:
    ok,frame=cap.read()
    if not ok:raise ValueError(f'missing encoded frame {fid}')
    panels.append(frame[:,frame.shape[1]//2:].copy())
   height,width=panels[0].shape[:2];wh=(width,height-40)
   raw=cv2.resize(cv2.imread(inputs.raw_paths[i]),wh,interpolation=cv2.INTER_AREA)
   mask=cv2.resize(inputs.masks[i].astype('uint8'),wh,interpolation=cv2.INTER_NEAREST_EXACT)
   contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(raw,contours,-1,(0,255,255),1)
   panels.insert(0,np.vstack([np.zeros((40,width,3),np.uint8),raw]))
   for p,label in zip(panels,['raw RGB + source appearance mask','old: contour-only refinement','repair: joint RGB + metric-gap bridge']):
    p[:40]=0;cv2.putText(p,f'f{int(fid)} {label}',(4,15),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
    cv2.putText(p,'actual full mesh + unchanged source MANO' if p is not panels[0] else 'source pixels unchanged; yellow observed mask',(4,32),cv2.FONT_HERSHEY_SIMPLEX,.32,(205,205,205),1)
   image=np.hstack(panels)
   if pipe is None:
    fps=1./float(np.median(np.diff(inputs.times)))
    pipe=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{image.shape[1]}x{image.shape[0]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(dest/'raw_old_repaired.mp4')],stdin=subprocess.PIPE)
   pipe.stdin.write(image.tobytes())
   if i in wanted:cv2.imwrite(str(dest/f'frame_{int(fid):06d}.jpg'),image)
   if i in [0,n//3,2*n//3,n-1]:contact[i]=image
  for cap in caps:
   if cap.read()[0]:raise ValueError('unexpected additional encoded frame')
 finally:
  for cap in caps:cap.release()
  if pipe is not None:
   pipe.stdin.close()
   if pipe.wait()!=0:raise ValueError('comparison encoder failed')
 cv2.imwrite(str(dest/'four_frame_review.jpg'),np.vstack(list(contact.values())))
 report={'case':case,'old':str(old),'repaired':str(new),'old_archive_sha256':sha(old/'coarse_scene.npz'),'repaired_archive_sha256':sha(new/'coarse_scene.npz'),
  'metrics':metrics,'frames':rows,'comparison_video':str(dest/'raw_old_repaired.mp4'),
  'scope':'fixed-reference and continuity diagnostics, not GT accuracy; fast real movement is not an error by itself'}
 (dest/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'case':case,'metrics':metrics,'video':report['comparison_video']}),flush=True)

if __name__=='__main__':
 cv2.setNumThreads(1)
 parser=argparse.ArgumentParser();parser.add_argument('cases',nargs='*');args=parser.parse_args()
 for item in read(NEW/'inputs.json'):
  if not args.cases or item['case'] in args.cases:compare(item)
