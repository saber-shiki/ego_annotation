"""Post-freeze Astra-only visual QC of actual full-surface comparison renders."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from utils.runtime import astra_request


def main(a):
    if not (a.renders/'render.json').is_file():
        raise ValueError('complete full-timeline renders required before QC')
    schema={'type':'object','properties':{
        'observed_2d_alignment':{'type':'string'},
        'observed_3d_geometry':{'type':'string'},
        'rendering_or_handedness_defects':{'type':'array','items':{'type':'string'}},
        'occlusion_and_contact_limitations':{'type':'string'},
        'sampling_limitations':{'type':'string'},
        'replacement_recommendation_from_visuals_only':{'type':'string'}},
        'required':['observed_2d_alignment','observed_3d_geometry','rendering_or_handedness_defects',
                    'occlusion_and_contact_limitations','sampling_limitations','replacement_recommendation_from_visuals_only'],
        'additionalProperties':False}
    images=[a.renders/'overlay_sheet.jpg',a.renders/'world_sheet.jpg']
    evidence=dict(columns=['HaWoR frozen RL hand state','WiLoR with own-2D perspective translation','Official MANO GT'],
                  rows_frame_ids=[0,30,60,90,120,149],
                  render_contract=json.loads((a.renders/'render.json').read_text()),
                  colors='left hand blue/orange-brown, right hand cyan/yellow; side fixed across columns',
                  prediction_frozen=True,GT_used_for_prediction=False,
                  limitation='Sheets are sampled QC; full150-frame/5s videos are separate artifacts. No metric labels are provided to avoid anchoring the visual judgment.')
    prompt=('Inspect actual images carefully as an independent visual QC, not a schema review. '
            'All columns use the same source RGB and fixed world observer. Report visible hand alignment, '
            'finger geometry and depth discrepancies; identify missing/flipped/degenerate geometry only if visible. '
            'Do not infer millimetre accuracy, temporal smoothness between sampled frames, collision-free contact '
            'or occlusion correctness from these sheets. Overlays are projected surfaces and may cover foreground '
            'objects. Distinguish real geometry changes from camera/viewpoint effects. Reply in Chinese.')
    result,provenance=astra_request(prompt,schema,evidence,images,audit_dir=a.output.with_suffix('.request'))
    a.output.write_text(json.dumps(dict(judgment=result,provenance=provenance),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--renders',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
