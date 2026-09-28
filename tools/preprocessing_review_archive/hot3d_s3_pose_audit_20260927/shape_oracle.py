"""Evaluation-only native-to-CAD similarity registration, never prediction input.
24 PCA initializations + trimmed symmetric ICP. An achieved fit, NOT guaranteed optimum.
No aligned mesh is exported. Transform and residuals are explicitly GT-assisted diagnostics.
"""
from evaluate import *
from scipy.spatial import cKDTree
import itertools

def similarity(X,Y):
    xc=X.mean(0);yc=Y.mean(0);x=X-xc;y=Y-yc
    u,s,vh=np.linalg.svd(x.T@y);fix=np.ones(3);fix[-1]=np.linalg.det(vh.T@u.T)
    R=vh.T@np.diag(fix)@u.T;scale=float((s*fix).sum()/np.sum(x*x))
    return scale,R,yc-scale*(R@xc)

def pca(x):
    _,v=np.linalg.eigh(np.cov(x.T));return v[:,::-1]

def run(case):
    c=load_case(case);n=sample(c['nv'],c['nf'],4096);g=sample(c['gv'],c['gf'],4096)
    small=n[::2];target=g[::2];ntree=cKDTree(small);gtree=cKDTree(target);bn=pca(small);bg=pca(target)
    s0=float(np.sqrt(np.sum((target-target.mean(0))**2)/np.sum((small-small.mean(0))**2)))
    best=None;trials=[]
    for perm in itertools.permutations(range(3)):
        P=np.eye(3)[:,perm]
        for signs in itertools.product([-1.,1.],repeat=3):
            R=bg@P@np.diag(signs)@bn.T
            if np.linalg.det(R)<0:continue
            s=s0;t=target.mean(0)-s*(R@small.mean(0))
            for _ in range(25):
                moved=s*(small@R.T)+t
                d,j=gtree.query(moved,workers=1);rd,k=ntree.query(((target-t)@R)/s,workers=1)
                keep=d<=np.percentile(d,85);rkeep=rd<=np.percentile(rd,85)
                X=np.vstack([small[keep],small[k[rkeep]]]);Y=np.vstack([target[j[keep]],target[rkeep]])
                ns,nR,nt=similarity(X,Y)
                if not .3<=ns<=3:break
                step=np.max(np.abs(nt-t))+np.max(np.abs(nR-R))+abs(ns-s);s,R,t=ns,nR,nt
                if step<1e-7:break
            d,_=gtree.query(s*(small@R.T)+t,workers=1);rd,_=ntree.query(((target-t)@R)/s,workers=1)
            score=float(.5*(d.mean()+rd.mean()*s));trials.append(score)
            if best is None or score<best['score']:best={'score':score,'scale':s,'R':R,'t':t}
    s,R,t=best['scale'],best['R'],best['t'];gs=scene(c['gv'],c['gf']);ns=scene(c['nv'],c['nf'])
    d=distance(gs,s*(n@R.T)+t)*1000;rd=distance(ns,((g-t)@R)/s)*s*1000
    both=np.r_[d,rd]
    result={'case':case,'GT_assisted_evaluation_only':True,'source_modified':False,'aligned_mesh_exported':False,'method':'24 PCA rotations, 25 iterations of 85%-trimmed symmetric nearest-sample similarity ICP; final actual-triangle distances','sample_count_each_direction':len(n),'uniform_scale_native_to_CAD':s,'rotation_camera_native_to_CAD':R.tolist(),'translation_camera_native_to_CAD_m':t.tolist(),'achieved_symmetric_surface_distance_mm':summ(both),'fraction_le_2mm':float(np.mean(both<=2)),'fraction_le_5mm':float(np.mean(both<=5)),'initialization_count':len(trials),'limitations':'GT-assisted removes location/orientation/one scale for SHAPE diagnosis only. Not absolute pose accuracy, not a prediction calibration, not a certified global optimum.'}
    (c['out']/'shape_oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['case','uniform_scale_native_to_CAD','achieved_symmetric_surface_distance_mm','fraction_le_5mm']}),flush=True)
    return result
if __name__=='__main__':
    for r in read(ROOT/'integrity_summary.json'):run(r['case'])
