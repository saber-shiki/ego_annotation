"""Relative monocular depth scale from rigid material-point chord lengths.
No absolute metric claim; the source appearance-frame gauge remains fixed.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from preprocessing.geometry import project


def estimate_relative_depth(inputs):
    n=len(inputs.ids);pos={int(f):i for i,f in enumerate(inputs.ids)};edges=[];trees={}
    for i,points in enumerate(inputs.clouds):
        if len(points)>=6:trees[i]=cKDTree(project(points,inputs.K[i]))
    for e in inputs.rgb.edges:
        a=pos[e.source_frame_idx];b=pos[e.target_frame_idx]
        if b not in trees or not inputs.cloud_valid[a] or not inputs.cloud_valid[b]:continue
        uv=(e.target_uv+.5)*(np.asarray(inputs.wh)/e.target_size_wh)-.5
        dd,j=trees[b].query(uv);good=np.flatnonzero(dd<3.)
        if len(good)<6:continue
        if len(good)>40:good=good[np.linspace(0,len(good)-1,40).astype(int)]
        p=(e.source_world[good]-inputs.cameras[a,:3,3])@inputs.cameras[a,:3,:3]
        q=inputs.clouds[b][j[good]]
        da=np.linalg.norm(p[:,None]-p[None,:],axis=-1);db=np.linalg.norm(q[:,None]-q[None,:],axis=-1)
        up=np.triu(np.ones_like(da,dtype=bool),1)
        ok=up&(da>.012)&(db>.012)
        if ok.sum()<12:continue
        ratios=np.log(da[ok]/db[ok]);center=float(np.median(ratios));mad=float(np.median(np.abs(ratios-center)))
        quality=max(.05,float(e.diagnostics['inlier_fraction']))/(1.+(mad/.08)**2)
        edges.append({'source':a,'target':b,'log_scale_difference':center,'MAD_log_ratio':mad,'weight':quality,'chords':int(ok.sum())})
    rows=[];cols=[];values=[];target=[];row=0
    for e in edges:
        w=np.sqrt(e['weight'])
        rows.extend([row,row]);cols.extend([e['source'],e['target']]);values.extend([-w,w]);target.append(e['log_scale_difference']*w);row+=1
    # Weak preference for inherited scale, temporal regularity, one explicit fixed gauge.
    for i in range(n):rows.append(row);cols.append(i);values.append(.08);target.append(0.);row+=1
    for i in range(n-1):
        rows.extend([row,row]);cols.extend([i,i+1]);values.extend([-.3,.3]);target.append(0.);row+=1
    rows.append(row);cols.append(inputs.anchor);values.append(1000.);target.append(0.);row+=1
    A=coo_matrix((values,(rows,cols)),shape=(row,n)).tocsr();y=np.asarray(target)
    fit=least_squares(lambda x:A@x-y,np.zeros(n),jac=lambda x:A.copy(),loss='soft_l1',f_scale=.04,bounds=(-.35,.35),max_nfev=40,ftol=1e-8,gtol=1e-8)
    scale=np.exp(fit.x);support=np.zeros(n,int)
    for e in edges:support[e['source']]+=1;support[e['target']]+=1
    med=np.asarray([float(np.median(p[:,2])) if len(p) else np.nan for p in inputs.clouds]);known=np.isfinite(med)
    if known.any():med=np.interp(inputs.times,inputs.times[known],med[known])
    else:med=np.zeros(n)
    return {'scale':scale,'bias_diagnostic_m':med*(1-scale),'support':support,'report':{
        'method':'RGB_material_chord_rigidity_relative_depth_scale','absolute_metric_accuracy_certified':False,
        'reference_frame_idx':int(inputs.ids[inputs.anchor]),'reference_scale':float(scale[inputs.anchor]),
        'per_frame_scale':scale.tolist(),'support_edges_per_frame':support.tolist(),'edges':edges,
        'unmeasured_scale_frames':[int(inputs.ids[i]) for i in range(n) if support[i]==0],
        'bounds_are_relative_source_prior_not_GT_accuracy':True,'solver_nfev':fit.nfev,'cost':float(fit.cost)}}
