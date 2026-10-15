"""Reproducible MANO source replay / derivative smoke. No optimization/export."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from preprocessing.mano_repair import models_from_source,fk
from preprocessing.inputs import sha

def main():
 import torch
 source=Path('/mnt/truenas-user-home/kupingxin/ego_annotation_rl_runs/milk_observation_v2_20260921_r2/source_observations.json');manifest=json.loads(source.read_text());inputs=SimpleNamespace(bundle=SimpleNamespace(manifest=manifest));models,hashes=models_from_source(inputs)
 p=source.parent/manifest['sources']['mano_archive']['path'];results={}
 with np.load(p,allow_pickle=False) as z:
  for s in ('left','right'):
   tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
   rot=tensor(z[s+'_root_orient_axis_angle'][:3]);pose=tensor(z[s+'_hand_pose_axis_angle'][:3]);b=tensor(z[s+'_betas'][:3]);t=tensor(z[s+'_trans_world_m'][:3]).requires_grad_(True)
   V,J=fk(models[s],rot,pose,b,t);ve=float(torch.max(torch.abs(V-tensor(z[s+'_vertices_world_m'][:3]))));je=float(torch.max(torch.abs(J-tensor(z[s+'_joints_world_m'][:3]))));assert max(ve,je)<2e-6
   J[0,0,2].backward();np.testing.assert_allclose(t.grad[0].numpy(),[0,0,1],atol=1e-6)
   results[s]=dict(vertex_max_error_m=ve,joint_max_error_m=je,translation_derivative_correct=True)
 print(json.dumps(dict(models=hashes,source_archive_sha256=sha(p),checks=results),indent=2))
if __name__=='__main__':main()
