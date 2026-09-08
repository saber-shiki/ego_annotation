#!/usr/bin/env python3
"""Bind the intact demo mesh without relabeling the source prediction state."""
import argparse
from pathlib import Path
import json
from v19_runtime_contracts import load, sha
from v20_prediction_contracts import assert_prediction_only


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pose-report','completion-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('demo binding already exists')
    source=load(a.pose_report);completion=load(a.completion_report)
    assert_prediction_only(source,label='demo source',allow_annotation_ready=True,allow_nondiagnostic_prediction=True)
    visual=dict(source);visual['render_mesh_contract']=completion['render_mesh_contract']
    visual['core_source_identity']={'pose_report':str(a.pose_report),'sha256':sha(a.pose_report),
        'source_readiness':source.get('annotation_ready'),'source_graph_support':source.get('graph_support')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(visual,indent=2)+'\n')


if __name__=='__main__':main()
