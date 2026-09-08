#!/usr/bin/env python3
"""Bind an unchanged numerical pose to a mesh for post-solve visual diagnosis.

This does NOT certify alignment or imply the mesh was part of the pose objective.
"""
from pathlib import Path
import argparse,json
import fit_v20_late_window_prediction_only_se3 as late


def bind(pose_path: Path, mesh_path: Path):
    report=late.load_json(pose_path)
    late.assert_prediction_only(report,label='frozen visual diagnostic')
    contract=late.generated_mesh_contract(mesh_path)
    result=dict(report)
    result.update(status='frozen_pose_mesh_visual_diagnostic',diagnostic_only=True,annotation_ready=False)
    result['render_mesh_contract']={'required_mesh_path':str(mesh_path.resolve()),'required_mesh_sha256':late.sha256_file(mesh_path),'renderer_must_match':True}
    result['render_evidence_role']='post_solve_fixed_mesh_diagnostic_not_pose_objective'
    result['visual_diagnostic_source']={'pose_path':str(pose_path.resolve()),'pose_sha256':late.sha256_file(pose_path),'poses_numerically_unchanged':True,'mesh_contract':contract}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-report',required=True,type=Path);p.add_argument('--mesh',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('refusing to overwrite visual diagnostic')
    result=bind(a.pose_report,a.mesh);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(late.json_safe(result),indent=2,allow_nan=False)+'\n')
    print(str(a.output))


if __name__=='__main__':main()
