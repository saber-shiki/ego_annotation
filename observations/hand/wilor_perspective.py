"""Prediction-only perspective translation from frozen WiLoR image measurements.

No reference, dataset, metric targets, scale fit, anatomical rescaling or rotation
update. This is not a new independent detector: 2D/3D share a network source.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from .wilor_calibrated import project, sha


def fit_translation(local, image_points, K, initial):
    local, uv, K = np.asarray(local,float), np.asarray(image_points,float), np.asarray(K,float)
    if local.shape != (21,3) or uv.shape != (21,2) or not np.isfinite(np.r_[local.ravel(),uv.ravel(),K]).all():
        raise ValueError('invalid perspective fit measurements')
    min_z = max(.001, float(-local[:,2].min()+.001))
    t0 = np.asarray(initial,float).copy()
    t0[2] = max(t0[2], min_z + .001)
    def residual(t):
        return (project(local+t,K)-uv).reshape(-1)
    fit = least_squares(residual,t0,bounds=([-np.inf,-np.inf,min_z],[np.inf,np.inf,np.inf]),
                        loss='linear',max_nfev=80,xtol=1e-9,ftol=1e-9,gtol=1e-8)
    if not np.isfinite(fit.x).all() or not np.isfinite(fit.fun).all():
        raise ValueError('nonfinite camera solution')
    return fit.x, dict(success=bool(fit.success), evaluations=int(fit.nfev),
                       before_mean_px=float(np.linalg.norm(residual(t0).reshape(-1,2),axis=-1).mean()),
                       after_mean_px=float(np.linalg.norm(fit.fun.reshape(-1,2),axis=-1).mean()))


def run(a):
    completion=a.source.with_suffix('.completed.json')
    meta=json.loads(completion.read_text())
    if sha(a.source)!=meta['output_sha256']:
        raise ValueError('source measurement changed')
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.source,allow_pickle=False) as z:
        q={k:z[k] for k in z.files}
    started=time.monotonic(); traces=[]; native_K=None
    for side in ['left','right']:
        for i in np.flatnonzero(q[side+'_valid']):
            old=q[side+'_camera_translation_m'][i].astype(float)
            native_t=q[side+'_default_camera_translation_m'][i].astype(float)
            native_f=native_t[2]/old[2]*q['intrinsics'][i,0]
            K=np.r_[native_f,native_f,q['image_size_wh']/2]
            if native_K is None:
                native_K=np.tile(K,(len(q['frame_idx']),1))
            local=q[side+'_joints_camera_m'][i].astype(float)-old
            target=project(q[side+'_default_joints_camera_m'][i],K)
            t,report=fit_translation(local,target,q['intrinsics'][i],old)
            q[side+'_joints_camera_m'][i]=local+t
            q[side+'_vertices_camera_m'][i]+=np.asarray(t-old,np.float32)
            q[side+'_camera_translation_m'][i]=t
            traces.append(dict(side=side,frame_idx=int(q['frame_idx'][i]),**report))
    q['native_projection_intrinsics']=native_K
    path=a.output/'wilor_hands.npz';np.savez_compressed(path,**q)
    report=dict(schema='ego.annotation.wilor_perspective_translation.v1',output_sha256=sha(path),
                source_path=str(a.source),source_sha256=sha(a.source),elapsed_s=time.monotonic()-started,
                frames=len(q['frame_idx']),traces=traces,
                mechanism='Fixed original full MANO articulation/shape/orientation; only translation optimized against own native 21 image joints under source K.',
                GT_or_reference_read=False,anatomical_rescaling=False,temporal_smoothing=False,
                source_measurement_provenance=meta)
    path.with_suffix('.completed.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['traces','source_measurement_provenance']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
