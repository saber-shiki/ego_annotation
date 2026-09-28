"""Post-hoc evidence summary; optional existing GT evaluations never feed prediction."""
from pathlib import Path
import argparse,json,hashlib,collections


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def summarize(diagnostics,run,evaluation):
 summaries=read(diagnostics/'summary.json');result=[]
 for summary in summaries:
  case=summary['case'];d=read(diagnostics/(case+'.json'));evaluation_case=read(evaluation/(case+'.json'))
  gt={r['frame']:r for r in evaluation_case['rows'] if r['GT_quality_eligible']}
  strong=[r for r in d['tracking_rows'] if r['PnP_available'] and r['points']>=20 and r.get('median_reprojection_px',float('inf'))<=2. and r['frame_idx'] in gt]
  wrong=[r for r in strong if gt[r['frame_idx']]['camera_rotation_deg']>30.]
  visible_bad=[r for r in d['tracking_rows'] if r.get('model_bound_points_behind_first_surface_fraction',0)>.1]
  row=dict(summary)
  row.update(longest_metric_gap_frames=max([b-a+1 for a,b in summary['metric_gaps']],default=0),
    hand_confidence_distinct_values=sorted(set(c for r in d['surface_rows'] for c in r['hand_confidences'] if c is not None)),
    metric_gap_reason_counts=dict(collections.Counter(str(r['metric_rejection_reason']) for r in d['surface_rows'] if not r['metric_eligible'])),
    frames_with_over_10percent_model_binding_self_visibility_conflict=[r['frame_idx'] for r in visible_bad],
    evaluation_only={'strong_PnP_proxy_frames':len(strong),'those_with_final_GT_rotation_error_over_30deg':len(wrong),
      'fraction':len(wrong)/len(strong) if strong else None,
      'GT_camera_rotation_median_deg':evaluation_case['camera_rotation_deg']['median'],
      'scope':'post-hoc correlation of pre-contour PnP support with FINAL pose error, not proof of which stage caused error; one-anchor-aligned, not symmetry-adjusted; thresholds diagnostic only'},
    evidence_files={'diagnostic':{'path':str(diagnostics/(case+'.json')),'sha256':sha(diagnostics/(case+'.json'))},
      'frozen_bootstrap':{'path':str(run/case/'bootstrap.npz'),'sha256':sha(run/case/'bootstrap.npz')},
      'frozen_prediction':{'path':str(run/case/'coarse_scene.npz'),'sha256':sha(run/case/'coarse_scene.npz')},
      'evaluation_only':{'path':str(evaluation/(case+'.json')),'sha256':sha(evaluation/(case+'.json'))}})
  result.append(row)
 return {'checkpoint_commit':'a42e974a8a7c51fd4484b1756ff4ff89f96977f5','diagnostics':str(diagnostics),
   'run':str(run),'prediction_or_source_changes':False,'new_model_inference':False,
   'thresholds':{'strong_PnP_proxy_min_inliers':20,'strong_PnP_proxy_max_median_px':2.,'posthoc_rotation_error_deg':30.,
     'model_binding_behind_first_surface_m':.005,'MANO_front_conflict_margin_m':.01,'background_flow_FB_max_px':1.},
   'scope':'observation reliability is conditional and variable-specific; all numeric thresholds describe diagnostics, not production acceptance; post-hoc GT correlation is segregated from the GT-free replay',
   'cases':result}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 report=summarize(a.diagnostics,a.run,a.evaluation)
 with a.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 for r in report['cases']:
  e=r['evaluation_only'];print(r['case'],'strong-proxy/final-error>30:',e['strong_PnP_proxy_frames'],e['those_with_final_GT_rotation_error_over_30deg'], 'self-visibility frames:',r['frames_with_over_10percent_model_binding_self_visibility_conflict'])
