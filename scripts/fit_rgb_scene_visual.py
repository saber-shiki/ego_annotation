#!/usr/bin/env python3
"""Execute the real MANO LBS/projection/unsigned-triangle fitter on a typed posterior."""
import argparse
from pathlib import Path
import runpy
import sys
from bind_rgb_scene_shape import validate_posterior
from v19_runtime_contracts import load,sha


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--pose-report',type=Path,required=True)
    p.add_argument('--object-id',required=True)
    a,remaining=p.parse_known_args()
    binding=load(a.pose_report)
    if binding.get('schema')!='rgb_scene_visual_refined_binding_v1':
        raise ValueError('typed visual-refined binding required, no readiness fallback')
    ref=binding['refined_posterior'];posterior=load(ref['path'])
    if sha(ref['path'])!=ref['sha256']:raise ValueError('refined posterior changed')
    validate_posterior(posterior,a.object_id,a.run_root)
    if binding['pose_rows']!=posterior['pose_rows']:raise ValueError('binding discarded refined trajectory')
    script=Path(__file__).with_name('fit_v20_visual_contact_demo.py')
    sys.argv=[str(script),'--pose-report',str(a.pose_report),'--object-id',a.object_id,*remaining]
    runpy.run_path(str(script),run_name='__main__')

if __name__=='__main__':main()
