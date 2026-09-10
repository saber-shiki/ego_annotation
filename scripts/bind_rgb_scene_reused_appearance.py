#!/usr/bin/env python3
"""Bind a reused native QC to byte-identical current appearance inputs.

Only the QC path mapping is derived. Original model outputs and receipts remain
unchanged; this stage does not run SAM3D or transform any vertices.
"""
import argparse
import copy
import json
from pathlib import Path
from rgb_scene_candidates import load_candidate
from v19_runtime_contracts import sha, in_run


def bind(context, source_qc, image, mask, output):
    paths, candidate = load_candidate(context)
    root = paths.run_root
    source_qc, image, mask = (in_run(p, root) for p in (source_qc, image, mask))
    output = Path(output).resolve()
    if not output.is_relative_to(paths.root) or output.exists():
        raise ValueError('new candidate-local QC output required')
    qc = json.loads(source_qc.read_text())
    if qc.get('status') != 'ok':
        raise ValueError('successful imported native QC required')
    raw = json.loads((root/'input/raw_frame_manifest/manifest.json').read_text())
    selected = [r for r in raw['frames'] if r['frame_idx'] == candidate['selected_frame_idx']]
    if len(selected) != 1 or Path(selected[0]['raw_frame_path']).resolve() != image:
        raise ValueError('appearance RGB is not the selected current frame')
    for key, current in (('image', image), ('mask', mask)):
        original = in_run(qc[key], root)
        if sha(original) != sha(current):
            raise ValueError('reused native conditioning bytes differ: ' + key)
    derived = copy.deepcopy(qc)
    derived.update(image=str(image), mask=str(mask))
    derived['appearance_conditioning_mapping'] = {
        'source_qc':str(source_qc), 'source_qc_sha256':sha(source_qc),
        'candidate_id':paths.candidate_id, 'frame_idx':candidate['selected_frame_idx'],
        'image_sha256':sha(image), 'mask_sha256':sha(mask),
        'model_executed':False, 'mesh_vertices_modified':False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(derived, handle, indent=2)
        handle.write('\n')
    return derived


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-context','source-qc','image','mask','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a=parser.parse_args()
    bind(a.candidate_context,a.source_qc,a.image,a.mask,a.output)
    print(a.output)


if __name__ == '__main__':
    main()
