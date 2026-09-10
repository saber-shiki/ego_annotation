#!/usr/bin/env python3
"""Execute the real MANO LBS/projection/unsigned-triangle fitter on a typed posterior."""
import argparse
from pathlib import Path
import runpy
import sys
import numpy as np
from rgb_scene_visibility import validate_visibility, apply_display_visibility, POLICY
from bind_rgb_scene_shape import validate_posterior
from v19_runtime_contracts import load,sha


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--pose-report',type=Path,required=True)
    p.add_argument('--object-id',required=True)
    a,remaining=p.parse_known_args()
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--interaction-judgments',type=Path,required=True)
    options.add_argument('--output-dir',type=Path,required=True)
    fit_args,_=options.parse_known_args(remaining)
    binding=load(a.pose_report)
    if binding.get('schema')!='rgb_scene_visual_refined_binding_v1':
        raise ValueError('typed visual-refined binding required, no readiness fallback')
    ref=binding['refined_posterior'];posterior=load(ref['path'])
    if sha(ref['path'])!=ref['sha256']:raise ValueError('refined posterior changed')
    validate_posterior(posterior,a.object_id,a.run_root)
    if binding['pose_rows']!=posterior['pose_rows']:raise ValueError('binding discarded refined trajectory')
    judgment=load(fit_args.interaction_judgments)
    validate_visibility(judgment,[row['frame_idx'] for row in binding['pose_rows']],a.object_id)
    script=Path(__file__).with_name('fit_v20_visual_contact_demo.py')
    sys.argv=[str(script),'--pose-report',str(a.pose_report),'--object-id',a.object_id,*remaining]
    runpy.run_path(str(script),run_name='__main__')
    state_path=fit_args.output_dir/'demo_state.npz'
    with np.load(state_path,allow_pickle=False) as archive:
        state=apply_display_visibility(dict(archive),judgment,a.object_id)
    np.savez_compressed(state_path,**state)
    report_path=fit_args.output_dir/'fit_report.json'
    report=load(report_path)
    report['display_visibility']={'policy':POLICY,'judgment_sha256':sha(fit_args.interaction_judgments),
                                  'source_geometry_mutated':False}
    import json
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
