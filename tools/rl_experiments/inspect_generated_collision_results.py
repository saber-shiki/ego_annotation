"""Read-only outcome synthesis and full collision-highlight RRD for a frozen run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.rl_experiments.export_world_comparison_rrd import load_candidates,export_rrd,verify_roundtrip,sha
from tools.rl_experiments.preprocessed_learning import export_npz


def summarize_collision(report):
    summary={k:report[k] for k in ['affected_frame_count','affected_intervals','hand_affected_intervals','max_inside_depth_mm','inside_squared_hinge_sum_m2','total_crossing_edges_over_all_hand_frames']}
    per_side={}
    for side in ['left','right']:
        rows=[r['hands'][side] for r in report['frames'] if r['hands'][side]['state']!='absent_geometry']
        per_side[side]={'affected_frames':len(report['hand_affected_frame_ids'][side]),
            'substantive_over_1mm_frames':len(report['hand_substantive_over_1mm_frame_ids'][side]),
            'inside_vertex_count_total':sum(r['inside_vertex_count_over_0_1mm'] for r in rows),
            'inside_face_center_count_total':sum(r['inside_face_center_count_over_0_1mm'] for r in rows),
            'crossing_edges_total':sum(r['edge_crossings'] for r in rows),
            'max_depth_mm':max(r['max_inside_depth_mm'] for r in rows)}
    summary['hands']=per_side
    return summary


def inspect(run,output):
    output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((run/'summary.json').read_text())
    zero=json.loads((run/'zero/evaluation.json').read_text())
    zero_audit=json.loads((run/'zero/collision_audit.json').read_text())
    base=summarize_collision(zero_audit); rows=[]
    for row in summary['checkpoints']:
        cp=Path(row['checkpoint']);prov=json.loads(cp.with_name('agent.provenance.json').read_text())
        assert sha(cp)==prov['checkpoint']['sha256']
        evalpath=Path(row['evaluation']);evaluation=json.loads(evalpath.read_text())
        manifestpath=Path(evaluation['manifest']);manifest=json.loads(manifestpath.read_text())
        assert manifest['executed_frame_count']==150 and manifest['preprocessors_frozen_during_evaluation']
        for artifact in manifest['episodes'][0]['artifacts']:
            path=Path(artifact['path']);path=path if path.is_absolute() else manifestpath.parent/path
            assert sha(path)==artifact['sha256']
        audit=json.loads((evalpath.parent/'collision_audit.json').read_text());metrics=summarize_collision(audit)
        rows.append({'steps':row['steps'],'J_total':row['objective'],'J_original':row['original_formal_objective'],
                     'J_generated_collision':row['generated_collision_objective'],
                     'delta_J_total':row['objective']-zero['objective'],'delta_J_original':row['original_formal_objective']-zero['original_formal_objective'],
                     'joint_pixel_cost':row['named_families'].get('hand/joint_pixel_deadband',0),
                     'collision':metrics,'delta_squared_inside_metric':metrics['inside_squared_hinge_sum_m2']-base['inside_squared_hinge_sum_m2'],
                     'artifact_hashes_verified':True})
    result={'scope':'single_video_operator_trusted_generated_geometry_not_GT','zero':{
            'J_total':zero['objective'],'J_original':zero['original_formal_objective'],'J_generated_collision':zero['generated_collision_objective'],'collision':base},
            'checkpoints':rows,'improves_zero_total':summary['improves_zero'],'full_118D_hands_unfrozen':not summary['hands_frozen'],
            'generated_source_authority_not_promoted':not summary['source_authority_promoted']}
    (output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    items,archives=load_candidates(run)
    reports=[zero_audit]
    for row in [summary['best_trained_checkpoint'],summary['checkpoints'][-1]]:
        reports.append(json.loads((Path(row['evaluation']).parent/'collision_audit.json').read_text()))
    rrd=run/'renders/zero_best_last_world_collision.rrd'
    receipt=export_rrd(items,archives,rrd,collision_reports=reports)
    verification=verify_roundtrip(rrd,items,archives)
    (output/'rrd_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({'rows':[(r['steps'],r['J_total'],r['J_generated_collision'],r['collision']['max_inside_depth_mm']) for r in rows],
                      'rrd':str(rrd),'bytes':receipt['bytes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inspect(a.run,a.output)
