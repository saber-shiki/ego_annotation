"""Owner-opted shared GPU placement; exclusive default runtime is unchanged."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess


def snapshot():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),free_mib=int(c),utilization=int(d))
            for a,b,c,d in (r.split(',') for r in text.splitlines() if r.strip())]


def select(min_free_mib, max_utilization=25, records=None):
    records=snapshot() if records is None else records
    good=[r for r in records if r['free_mib']>=min_free_mib and r['utilization']<=max_utilization]
    if not good:
        raise RuntimeError(f'no GPU meets shared resource budget free>={min_free_mib}MiB util<={max_utilization}%; snapshot={records}')
    return max(good,key=lambda r:(r['free_mib'],-r['utilization']))


def run_shared(command, *, cwd, environment, selected, min_free_mib, max_utilization, stdout):
    current=next((r for r in snapshot() if r['uuid']==selected['uuid']),None)
    if current is None or current['free_mib']<min_free_mib or current['utilization']>max_utilization:
        raise RuntimeError('selected shared GPU no longer meets launch budget; no silent device substitution')
    env={**environment,'CUDA_VISIBLE_DEVICES':current['uuid']}
    subprocess.run(list(map(str,command)),cwd=cwd,env=env,check=True,stdout=stdout,stderr=subprocess.STDOUT)
    return {**current,'policy':'owner_authorized_resource_aware_shared','exclusive_compute_check':False,
            'minimum_free_mib':min_free_mib,'maximum_utilization':max_utilization}
