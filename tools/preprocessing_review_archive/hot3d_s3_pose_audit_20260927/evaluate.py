"""CPU-only evaluation of verified S3 arrays against CAD/GT. No prediction mutation.
Pose scores are one-anchor gauge-aligned motion errors, NOT absolute CAD pose errors.
Surface scores are raw camera-space model discrepancies with no GT alignment.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,csv,hashlib,gc,argparse
import numpy as np
import cv2,trimesh,open3d as o3d
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
GTROOT=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/hot3d_pinhole_rgbd_selection_v1/clips')
DATASET=Path('/mnt/nas-106/hot3d/dataset')
cv2.setNumThreads(1)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def pose(R,t):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T

def gtpose(row):
    q=row['quaternion_wxyz']
    return pose(Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix(),row['translation_xyz'])
def apply(T,x):return np.asarray(x)@T[:3,:3].T+T[:3,3]
def angle(R):return float(np.degrees(Rotation.from_matrix(R).magnitude()))
def summ(v):
    x=np.asarray(v,float);x=x[np.isfinite(x)]
    return {'count':len(x),'median':float(np.median(x)),'p05':float(np.percentile(x,5)),'p95':float(np.percentile(x,95)),'max':float(x.max()),'mean':float(x.mean())} if len(x) else {'count':0}
def loadmesh(p):
    m=trimesh.load(p,process=False)
    if isinstance(m,trimesh.Scene):m=m.to_geometry()
    return np.asarray(m.vertices,float),np.asarray(m.faces,np.int64)
def sample(v,f,n=2048):
    rng=np.random.default_rng(20260927);tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    ids=rng.choice(len(f),size=n,p=areas/areas.sum());uv=rng.random((n,2));flip=uv.sum(axis=1)>1;uv[flip]=1-uv[flip];t=tri[ids]
    return t[:,0]+uv[:,0,None]*(t[:,1]-t[:,0])+uv[:,1,None]*(t[:,2]-t[:,0])
def scene(v,f):
    s=o3d.t.geometry.RaycastingScene(nthreads=1)
    s.add_triangles(o3d.core.Tensor(np.asarray(v,np.float32)),o3d.core.Tensor(np.asarray(f,np.uint32)))
    return s

def distance(s,points):return s.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32)),nthreads=1).numpy().astype(float)
def gauge_errors(P,C,G,D,anchor):
    """P predicted world/object, C predicted world/camera; G,D GT equivalents.
    Exactly one anchor object transform and one camera gauge alignment, no scale or per-frame fit.
    """
    A=D[anchor]@np.linalg.inv(C[anchor]);B=np.linalg.inv(A@P[anchor])@G[anchor]
    rows={}
    for i in sorted(P):
        Ew=A@P[i]@B;Eg=np.linalg.inv(D[i])@G[i];Ec=np.linalg.inv(C[i])@P[i]@B
        rows[i]={'world_rotation_deg':angle(Ew[:3,:3]@G[i][:3,:3].T),'world_translation_mm':float(np.linalg.norm(Ew[:3,3]-G[i][:3,3])*1000),
            'camera_rotation_deg':angle(Ec[:3,:3]@Eg[:3,:3].T),'camera_translation_mm':float(np.linalg.norm(Ec[:3,3]-Eg[:3,3])*1000)}
    return rows

def sequence_name(case):return 'P0014_84ea2dcc' if case.startswith('milk_') else '_'.join(case.removeprefix('hot3d_').split('_')[:2])
def clip_name(case):return 'P0014_84ea2dcc_carton_milk_f2370_2519' if case.startswith('milk_') else case.removeprefix('hot3d_').removesuffix('_rl_v2_low')
def load_case(case):
    out=ROOT/case;src=out/'observations';pre=out/'preprocessing';b=read(src/'source_observations.json')
    vis=read(src/b['sources']['visual_geometry_report']['path']);ann=read(src/b['sources']['annotations']['path']);frames={int(f['frame_idx']):f for f in ann['frames']}
    inventory=next(p for p in read(ROOT/'inventory.json')['packages'] if p['case']==case and p['kind']=='observations');original=Path(inventory['original_root'])
    gt=GTROOT/clip_name(case);dr=read(gt/'depth_gt/depth_export_report.json');sidepath=gt/'evaluation/hot3d_gt/hot3d_vrs_gt_sidecar.json';side=read(sidepath);target=dr['target'];uid=target['object_uid'];bop=str(target['object_bop_id'])
    G={};D={};gtframes={};Kgt={};times={}
    for f in side['frames']:
        i=int(f['frame_idx']);row=f['json'];gtframes[i]=row;times[i]=int(f['timestamp_ns'])
        objects=[o for o in row['objects.json'][bop] if str(o['object_uid'])==uid]
        if len(objects)!=1:raise ValueError('GT target uniqueness')
        G[i]=gtpose(objects[0]['T_world_from_object']);D[i]=gtpose(row['cameras.json']['214-1']['T_world_from_camera']);Kgt[i]=np.array(row['cameras.json']['214-1']['projection_params'],float)
    cam=np.load(pre/'arrays/camera_timeline.npz',allow_pickle=False);ids=list(map(int,cam['frame_idx']));C={i:np.array(t) for i,t in zip(ids,cam['T_world_camera'])};K={i:np.array(k) for i,k in zip(ids,cam['intrinsics_source'])};cam.close()
    if set(ids)!=set(frames) or set(ids)!=set(G):raise ValueError('timeline mismatch')
    source={};objects={}
    for i in ids:
        o=next((o for o in frames[i]['objects'] if o.get('object_id')==vis['object_id']),{});objects[i]=o;po=o.get('reconstructed_geometry_pose') or {}
        r=po.get('rotation_world_from_canonical_matrix',po.get('rotation_world_from_completed_canonical_matrix'));t=po.get('translation_world_m')
        if r is not None and t is not None:source[i]=pose(r,t)
    seedrec=read(pre/'measurement_seed.json');seed={int(r['frame_idx']):pose(r['rotation_world_from_canonical'],r['translation_world_m']) for r in seedrec['frames'] if r['available']}
    qa={};qaprovenance={}
    for name in ('mask_object_pose_available','mask_headset_pose_available','mask_qa_pass','mask_object_visible'):
        path=DATASET/sequence_name(case)/'masks'/f'{name}.csv'
        with path.open() as f:byts={int(r['timestamp[ns]']):r['mask'].lower()=='true' for r in csv.DictReader(f) if r['stream_id']=='214-1'}
        qa[name]={i:byts.get(times[i]) for i in ids};qaprovenance[name]={'path':str(path),'sha256':sha(path),'true':sum(v is True for v in qa[name].values()),'false':sum(v is False for v in qa[name].values()),'missing':sum(v is None for v in qa[name].values())}
    eligible={i:all(qa[k][i] is True for k in ('mask_object_pose_available','mask_headset_pose_available','mask_qa_pass')) for i in ids}
    record=dr['mesh_reports'][uid];cadpath=Path(record['path'])
    if sha(cadpath)!=record['sha256']:raise ValueError('CAD checksum')
    nv,nf=loadmesh(src/b['sources']['visual_mesh_camera']['path']);gv,gf=loadmesh(cadpath)
    a=int(vis['appearance_anchor_frame_idx']);m=int(seedrec['anchor_frame_idx'])
    if a not in source or a not in seed or m not in source or m not in seed:raise ValueError('missing evaluation/binding anchor')
    with np.load(pre/'arrays/complete_visual_geometry.npz',allow_pickle=False) as z:
        if not np.allclose(z['vertices_metric_camera'],nv,atol=1e-7):raise ValueError('S3 obs/preprocessed visual mismatch')
        if not np.array_equal(z['faces'],nf):raise ValueError('face mismatch')
        if not np.allclose(apply(seed[a],z['vertices_canonical_at_seed']),apply(C[a],nv),atol=1e-6):raise ValueError('S3 seed geometry binding mismatch')
    return dict(case=case,out=out,src=src,pre=pre,bundle=b,vis=vis,frames=frames,objects=objects,original=original,gt=gt,side=side,sidepath=sidepath,target=target,G=G,D=D,C=C,K=K,Kgt=Kgt,ids=ids,source=source,seed=seed,seedrec=seedrec,qa=qaprovenance,eligible=eligible,nv=nv,nf=nf,gv=gv,gf=gf,cadpath=cadpath,a=a,m=m)

def evaluate(case):
    c=load_case(case);out=c['out'];nv,nf,gv,gf=c['nv'],c['nf'],c['gv'],c['gf'];a,m=c['a'],c['m'];C,D,G=c['C'],c['D'],c['G']
    ns=sample(nv,nf);gs=sample(gv,gf);gscene=scene(gv,gf);nscene=scene(nv,nf)
    ga=np.linalg.inv(D[a])@G[a];gva=apply(ga,gv)
    sizes={'camera_AABB_native_mm':(np.ptp(nv,axis=0)*1000).tolist(),'camera_AABB_CAD_mm':(np.ptp(gva,axis=0)*1000).tolist(),'camera_AABB_ratio':(np.ptp(nv,axis=0)/np.ptp(gva,axis=0)).tolist(),
        'native_area_sample_RMS_radius_mm':float(np.sqrt(np.mean(np.sum((ns-ns.mean(axis=0))**2,axis=1)))*1000),'CAD_area_sample_RMS_radius_mm':float(np.sqrt(np.mean(np.sum((gs-gs.mean(axis=0))**2,axis=1)))*1000),
        'note':'AABB shared camera frame, depends on pose; RMS radius is rotation/translation invariant but still shape-dependent. Uniform surface samples n=2048, seed=20260927.'}
    pose_errors={name:gauge_errors(c[name],C,G,D,m) for name in ('source','seed')}
    rows=[];depth=[];firstA=D[0]@np.linalg.inv(C[0]);Kmax=0.
    for i in c['ids']:
        row={'frame':i,'GT_quality_eligible':c['eligible'][i]};Kmax=max(Kmax,float(np.abs(c['K'][i]-c['Kgt'][i]).max()))
        cam=firstA@C[i];row['camera_trajectory_rotation_error_deg']=angle(cam[:3,:3]@D[i][:3,:3].T);row['camera_trajectory_translation_error_mm']=float(np.linalg.norm(cam[:3,3]-D[i][:3,3])*1000)
        row['GT_object_motion_from_metric_anchor_deg']=angle(G[i][:3,:3]@G[m][:3,:3].T)
        go=np.linalg.inv(D[i])@G[i]
        for name in ('source','seed'):
            P=c[name];row[name+'_available']=i in P
            if i not in P:continue
            row.update({name+'_aligned_'+k:v for k,v in pose_errors[name][i].items()})
            H=np.linalg.inv(C[i])@P[i]@np.linalg.inv(P[a])@C[a]
            n_to_g=distance(gscene,apply(np.linalg.inv(go)@H,ns))*1000
            g_to_n=distance(nscene,apply(np.linalg.inv(H)@go,gs))*1000
            both=np.r_[n_to_g,g_to_n]
            row[name+'_raw_surface_symmetric_mean_mm']=float(both.mean());row[name+'_raw_surface_symmetric_median_mm']=float(np.median(both));row[name+'_raw_surface_symmetric_p95_mm']=float(np.percentile(both,95));row[name+'_raw_surface_fraction_le_5mm']=float(np.mean(both<=5))
        obj=c['objects'][i];surf=obj.get('visible_geometry_candidate') or {};cp=np.asarray(surf.get('camera_vertices_sample_m',[]),float)
        if cp.ndim==2 and cp.shape[1]==3 and len(cp):
            k=c['K'][i];uv=cp[:,:2]/cp[:,2,None]*k[:2]+k[2:];pix=np.rint(uv).astype(int)
            gz=cv2.imread(str(c['gt']/f'depth_gt/target_depth_mm/{i:06d}.png'),cv2.IMREAD_UNCHANGED)
            if gz is None:raise FileNotFoundError('GT depth missing')
            good=(pix[:,0]>=0)&(pix[:,0]<gz.shape[1])&(pix[:,1]>=0)&(pix[:,1]<gz.shape[0])&(cp[:,2]>0)
            ii=np.flatnonzero(good);z=gz[pix[ii,1],pix[ii,0]].astype(float)/1000;ii=ii[z>0];z=z[z>0]
            if len(ii):
                er=(cp[ii,2]-z)*1000;ratio=cp[ii,2]/z
                d={'frame':i,'samples':len(ii),'source_samples':len(cp),'metric_eligible':obj.get('rigid_pose_observation_eligible') is True,'GT_quality_eligible':c['eligible'][i],'depth_bias_mm':summ(er),'depth_ratio':summ(ratio)}
                depth.append(d);row['P09_GT_depth_samples']=len(ii);row['P09_depth_bias_median_mm']=float(np.median(er));row['P09_depth_ratio_median']=float(np.median(ratio))
        rows.append(row)
    accepted=[r for r in rows if r['GT_quality_eligible']]
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'frame_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={}
    for name in ('source','seed'):
        summary[name]={key:summ([r[name+'_'+key] for r in accepted if name+'_'+key in r]) for key in ('aligned_world_rotation_deg','aligned_world_translation_mm','aligned_camera_rotation_deg','aligned_camera_translation_mm','raw_surface_symmetric_mean_mm','raw_surface_symmetric_median_mm','raw_surface_symmetric_p95_mm','raw_surface_fraction_le_5mm')}
        keys=[r for r in accepted if r.get(name+'_available')]
        summary[name]['aligned_camera_10mm_10deg_count']=sum(r[name+'_aligned_camera_translation_mm']<=10 and r[name+'_aligned_camera_rotation_deg']<=10 for r in keys)
        summary[name]['valid_frames']=len(keys)
    common=[r for r in accepted if r['source_available'] and r['seed_available']]
    matched={n:{k:summ([r[n+'_'+k] for r in common]) for k in ('aligned_camera_rotation_deg','aligned_camera_translation_mm','raw_surface_symmetric_mean_mm')} for n in ('source','seed')}
    report={'case':case,'evaluation_only':True,'input_bundle_sha256':sha(c['src']/'source_observations.json'),'GT_sidecar_sha256':sha(c['sidepath']),'CAD_path':str(c['cadpath']),'CAD_sha256':sha(c['cadpath']),'GT_quality':c['qa'],'timeline_frames':len(rows),'GT_quality_frames':len(accepted),'GT_source_sequence':c['side']['source_sequence'],'timestamp_match_delta_summary_ns':c['side'].get('timestamp_match_delta_summary_ns'),
        'appearance_anchor':a,'motion_evaluation_anchor':m,'same_anchor_for_source_and_seed':True,'maximum_K_difference_px':Kmax,'native_vertices':len(nv),'native_faces':len(nf),'CAD_vertices':len(gv),'CAD_faces':len(gf),'size_at_appearance_anchor':sizes,
        'motion_and_surface_summary':summary,'same_common_frame_comparison':{'count':len(common),'metrics':matched},
        'camera_first_frame_aligned_rotation_deg':summ([r['camera_trajectory_rotation_error_deg'] for r in accepted]),'camera_first_frame_aligned_translation_mm':summ([r['camera_trajectory_translation_error_mm'] for r in accepted]),'GT_true_object_rotation_from_metric_anchor_deg':summ([r['GT_object_motion_from_metric_anchor_deg'] for r in accepted]),
        'P09_frame_median_depth_bias_mm':summ([r['depth_bias_mm']['median'] for r in depth if r['metric_eligible'] and r['GT_quality_eligible']]),'P09_frame_median_depth_ratio':summ([r['depth_ratio']['median'] for r in depth if r['metric_eligible'] and r['GT_quality_eligible']]),'depth_rows':depth,'frame_rows':rows,
        'metric_definitions':{'motion':'One fixed SE3 world gauge and one fixed object-frame transform from GT at the P09 anchor, common source/seed anchor; initial error set to zero; no scale fit or per-frame alignment. Measures motion tracking, not absolute initial CAD orientation/position. Raw rotation is NOT symmetry-aware.', 'surface':'Uniform area sampling, 2048 points each direction; exact point-to-triangle unsigned distances, independently to actual CAD/native triangle meshes. Mean per frame over 4096 distances, reported median/P95 across frames. Raw current-camera geometry; no GT alignment; includes pose+shape+scale. Not BOP MSSD/VSD.', 'depth':'P09 sampled camera-Z minus GT rendered target depth at corresponding rounded source pixels, median per eligible frame. Not dense-pixel depth; GT render is not sensor depth.', 'coverage':'GT official object/headset pose availability and QA masks applied. Missing prediction not filled. Source/seed also compared on identical common frame set.'},
        'production_unchanged':True}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':case,'QA':len(accepted),'a':a,'m':m,'depth':report['P09_frame_median_depth_bias_mm'],'size_ratio':sizes['camera_AABB_ratio'],'RMS_ratio':sizes['native_area_sample_RMS_radius_mm']/sizes['CAD_area_sample_RMS_radius_mm'],'motion':{n:{k:v for k,v in summary[n].items() if k in ['aligned_camera_rotation_deg','aligned_camera_translation_mm','raw_surface_symmetric_mean_mm','valid_frames']} for n in summary},'camera_translation':report['camera_first_frame_aligned_translation_mm']}),flush=True)
    del nscene,gscene,c;gc.collect()
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');a=p.parse_args()
    cases=[r['case'] for r in read(ROOT/'integrity_summary.json')]
    if a.case:cases=[a.case]
    for case in cases:evaluate(case)
