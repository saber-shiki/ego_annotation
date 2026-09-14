"""Immutable shape/pose bindings; generated surfaces do not gain physical roles."""
from utils.data import load, RKEY, TKEY
from utils.state import sha, in_run
from observations.common import pose_from_values

def validate_posterior(report, object_id, root):
    if report.get('schema') != 'rgb_scene_refined_pose_v1' or report.get('object_id') != object_id:
        raise ValueError('explicit selected-object visual-refined posterior required')
    source=in_run(report['source_prior']['path'],root)
    if sha(source)!=report['source_prior']['sha256'] or load(source)['pose_rows']!=report['source_pose_rows']:
        raise ValueError('immutable source prior changed')
    for row in report['pose_rows']:
        if pose_from_values(row.get(RKEY),row.get(TKEY)) is None:
            raise ValueError('missing posterior hypothesis')
    return report
