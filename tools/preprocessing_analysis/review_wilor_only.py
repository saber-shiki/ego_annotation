"""Astra-only visual review of frozen independent full-pipeline comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from utils.runtime import astra_request


def main(a):
    contract=json.loads((a.renders/'render.json').read_text())
    schema={'type':'object','properties':{key:{'type':'string'} for key in [
        'hand_alignment','object_geometry_and_pose','independent_camera_effects',
        'visible_rendering_defects','occlusion_contact_limitations','replacement_assessment','sampling_limits']},
        'required':['hand_alignment','object_geometry_and_pose','independent_camera_effects','visible_rendering_defects',
                    'occlusion_contact_limitations','replacement_assessment','sampling_limits'],'additionalProperties':False}
    prompt=('Inspect actual native object and MANO comparison images. Reply in Chinese. '
        'The five columns are old HaWoR R6, old HaWoR ordered-repair final, independently rerun WiLoR R6, '
        'independently rerun WiLoR ordered-repair final, and GT. WiLoR uses its own new DROID/UniDepth camera; '
        'no HaWoR predictions informed it. Both predictions use the same source RGB. '
        'All world panels use the same observer, single first-camera SE3 per run, scale1, no per-frame/object alignment. '
        'Judge hand/object alignment, shape, position and how the complete repair changes the independent WiLoR state. '
        'Separate visible image/geometry differences from uncertainty about their cause. '
        'Do not claim collision-free contact, correct real-image occlusion, millimetre accuracy or inter-sample '
        'temporal smoothness from sampled sheets. Report defects honestly; no metrics are supplied to anchor you.')
    evidence={'render_contract':contract,'sampled_frame_ids':[0,30,60,90,120,149],
              'source_predictions_frozen_before_GT_or_baseline_read':True}
    result,provenance=astra_request(prompt,schema,evidence,[a.renders/'overlay_sheet.jpg',a.renders/'world_sheet.jpg'],
                                   audit_dir=a.output.with_suffix('.request'))
    a.output.write_text(json.dumps(dict(judgment=result,provenance=provenance),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--renders',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
