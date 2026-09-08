#!/usr/bin/env python3
"""Run the fixed RGB edge producer with explicitly pinned offline Torch weights."""
import argparse
from pathlib import Path
import runpy
import sys
from urllib.parse import urlparse
from v19_run_lineage import RunLineage
from v19_runtime_contracts import load, sha


def bind_offline_weights(records, torch):
    """Replace URL resolution, not model code; only declared local bytes load."""
    def offline_load(url, *unused, **kwargs):
        name = kwargs.get('file_name') or Path(urlparse(url).path).name
        if name not in records:
            raise ValueError(f'undeclared Torch checkpoint request: {name}')
        record = records[name]
        path = Path(record['path'])
        if not path.is_file() or sha(path) != record['sha256']:
            raise ValueError(f'offline LightGlue checkpoint missing/changed: {path}')
        return torch.load(path, map_location=kwargs.get('map_location'), weights_only=False)
    torch.hub.load_state_dict_from_url = offline_load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    lineage=RunLineage(args.run_root,Path(__file__).resolve().parents[1],args.runtime_config,asset_role='rgb_loader')
    for flag in ('--annotations','--depth-npz','--initial-pose-report'):
        lineage.require(remaining[remaining.index(flag)+1])
    config = load(args.runtime_config)
    import torch
    bind_offline_weights(config['lightglue_weights'], torch)
    script = Path(__file__).with_name('build_v20_global_rgb_orientation_edges.py')
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
