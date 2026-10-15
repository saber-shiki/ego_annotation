"""Local post-freeze full-state evaluation using the retained audited evaluator.

This intentionally depends on the original-host analysis helper, not a producer.
No evaluation module or inputs belong in the isolated prediction snapshot.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
from utils.state import sha


def main(a):
    if not (a.prediction.parent/'completed.json').is_file():
        raise ValueError('prediction must be complete and frozen before loading references')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['MILK_REPAIR_PRED']=str(a.prediction.resolve())
    os.environ['MILK_REPAIR_OUTPUT']=str(a.output.resolve())
    namespace=runpy.run_path(str(a.evaluator),run_name='post_freeze_hand_object_analysis')
    globals_=namespace['evaluate'].__globals__
    # The retained helper's OBS global is provenance-only, but must name this
    # actual source rather than historical HaWoR measurements.
    globals_['OBS']=a.source_bundle.resolve().parent
    hashes={str(p):sha(p) for p in [a.prediction,a.prediction.with_suffix('.json'),a.source_bundle,a.evaluator]}
    namespace['evaluate']()
    for p,digest in hashes.items():
        if sha(p)!=digest:raise RuntimeError('analysis mutated input '+p)
    (a.output/'analysis_provenance.json').write_text(json.dumps(dict(
        prediction_frozen_before_evaluation=True,source_bundle=str(a.source_bundle.resolve()),
        local_evaluator=str(a.evaluator.resolve()),input_hashes=hashes,
        all_HaWoR_or_GT_reads_after_prediction_freeze=True,
        limitation='Original-host evaluator helpers and licensed/reference assets are external dependencies.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['prediction','source-bundle','output','evaluator']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
