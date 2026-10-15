"""Anchored relative depth scale from immutable RGB material-point chord evidence.
No object template, additive depth bias, target-size input or reference annotations.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from preprocessing.geometry import project

def fit_scale_curve(times,edges,anchor=0,knots=8):
 times=np.asarray(times,float);n=len(times);kt=np.linspace(times[0],times[-1],min(knots,n))
 # Fixed zero first knot; each intermediate frame is interpolation of an 8-D curve.
 A=np.stack([np.interp(times,kt,np.eye(len(kt))[j]) for j in range(len(kt))],axis=1)
 a=A[anchor];free=np.arange(1,len(kt));F=A[:,free]-a[free][None,:]
 ea=np.array([e['source_position'] for e in edges],int);eb=np.array([e['target_position'] for e in edges],int)
 target=np.array([e['log_scale_difference'] for e in edges]);weight=np.sqrt(np.array([e['weight'] for e in edges]))
 def residual(x):
  curve=F@x
  return np.r_[((curve[eb]-curve[ea])-target)*weight/.05,np.diff(x,n=2)/.12 if len(x)>2 else np.empty(0),x/.20*.12]
 fit=least_squares(residual,np.zeros(len(free)),loss='soft_l1',bounds=(-.35,.35),max_nfev=40,ftol=1e-9,xtol=1e-9,gtol=1e-9)
 scale=np.exp(F@fit.x);scale[anchor]=1.
 return scale,dict(knots_s=kt.tolist(),free_log_knots=fit.x.tolist(),cost=float(fit.cost),nfev=int(fit.nfev),anchor_position=int(anchor),anchor_scale=1.,additive_bias=False,temporal_curve_parameters=len(free))

def estimate_calibration(inputs):
 positions={int(fid):i for i,fid in enumerate(inputs.ids)};trees={};clouds=inputs.clouds
 for i,q in enumerate(clouds):
  if len(q)>=8:trees[i]=cKDTree(project(q,inputs.K[i]))
 edges=[];dropped=[]
 for edge in inputs.rgb.edges:
  a=positions[edge.source_frame_idx];b=positions[edge.target_frame_idx]
  if a not in trees or b not in trees:continue
  uv0=(edge.source_uv+.5)*(np.asarray(inputs.wh)/edge.source_size_wh)-.5;uv1=(edge.target_uv+.5)*(np.asarray(inputs.wh)/edge.target_size_wh)-.5
  d0,i0=trees[a].query(uv0,workers=1);d1,i1=trees[b].query(uv1,workers=1)
  good=(d0<3.)&(d1<3.)&(np.asarray(edge.weights)>0)
  if len(edge.inlier_indices):good &= np.isin(np.arange(len(good)),edge.inlier_indices)
  ids=np.flatnonzero(good)
  if len(ids)>40:ids=ids[np.linspace(0,len(ids)-1,40).astype(int)]
  if len(ids)<6:dropped.append(dict(a=int(inputs.ids[a]),b=int(inputs.ids[b]),reason='insufficient_same_material_depth_support',support=len(ids)));continue
  P=clouds[a][i0[ids]];Q=clouds[b][i1[ids]]
  da=np.linalg.norm(P[:,None]-P[None,:],axis=-1);db=np.linalg.norm(Q[:,None]-Q[None,:],axis=-1);ok=np.triu(np.ones_like(da,bool),1)&(da>.012)&(db>.012)
  if ok.sum()<12:continue
  ratios=np.log(da[ok]/db[ok]);center=float(np.median(ratios));mad=float(np.median(np.abs(ratios-center)))
  weight=max(.05,float(edge.diagnostics['inlier_fraction']))/(1+(mad/.10)**2)
  edges.append(dict(source_position=a,target_position=b,source_frame_idx=int(inputs.ids[a]),target_frame_idx=int(inputs.ids[b]),log_scale_difference=center,MAD_log_ratio=mad,weight=weight,matched_pixels=int(len(ids)),rigid_chords=int(ok.sum()),depth_lookup_median_px=float(np.median(np.r_[d0[ids],d1[ids]]))))
 scale,fit=fit_scale_curve(inputs.times,edges,inputs.anchor)
 support=np.zeros(len(inputs.ids),int)
 for e in edges:support[e['source_position']]+=1;support[e['target_position']]+=1
 report=dict(method='anchored_low_dimensional_rigid_RGB_chord_relative_depth_scale',fit=fit,scale=scale.tolist(),support_edges=support.tolist(),edges=edges,dropped=dropped,raw_depth_unchanged=True,absolute_metric_accuracy_certified=False,weak_support_frames=[int(inputs.ids[i]) for i in np.flatnonzero(support<2)],no_mesh_binding=True,no_additive_bias=True,scope='Source depth ratios on image matched material identities, uncertain when motion/depth matching is weak; source-anchor metric error remains')
 print('relative calibration','edges',len(edges),'range',float(scale.min()),float(scale.max()),'last',float(scale[-1]),flush=True)
 return scale,report
