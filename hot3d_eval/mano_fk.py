"""CPU-only MANO FK and the exact official19/ours21 correspondence."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TIP_VERTICES = [744, 320, 443, 554, 671]
MANO21_TO_OURS21 = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MANO21_TO_CANONICAL20 = [16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
OURS21_TO_OFFICIAL19 = [4, 8, 12, 16, 20, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def find_models(model_config=None, mano_left=None, mano_right=None):
    """Use explicit files or the supplied runtime config; never search private homes."""
    cfg = json.loads(Path(model_config).read_text()) if model_config else {}
    if not model_config and os.environ.get('EGO_MANO_MODELS'):
        cfg = {'mano_models': os.environ['EGO_MANO_MODELS']}
    result = {}
    for side, override in [('left', mano_left), ('right', mano_right)]:
        candidates = [Path(override)] if override else []
        if not override and cfg.get('mano_models'):
            candidates += [Path(cfg['mano_models']) / ('MANO_' + side.upper() + '.pkl')]
        if not override and cfg.get('hawor_root'):
            suffix = 'data_left/mano_left/MANO_LEFT.pkl' if side == 'left' else 'data/mano/MANO_RIGHT.pkl'
            candidates += [Path(cfg['hawor_root']) / '_DATA' / suffix]
        for p in candidates:
            try:
                with p.open('rb') as f:
                    f.read(1)
                result[side] = p
                break
            except OSError:
                continue
        else:
            raise FileNotFoundError(f'MANO {side} unreadable; supply --mano-left/--mano-right, --model-config, or EGO_MANO_MODELS')
    return result


class ManoFK:
    def __init__(self, paths):
        # Imported only for real GT FK. Synthetic metric tests require no model assets.
        import torch
        import smplx
        self.torch = torch
        torch.set_num_threads(1)
        # Old MANO pickle/chumpy compatibility, matching the official toolkit.
        for name, value in {'bool': np.bool_, 'int': np.int64, 'float': np.float64,
                            'complex': np.complex128, 'object': np.object_,
                            'unicode': np.str_, 'str': np.str_}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        self.models = {side: smplx.create(str(p), 'mano', use_pca=True,
                                        num_pca_comps=15, is_rhand=side == 'right').cpu().eval()
                       for side, p in paths.items()}
        left, right = self.models['left'], self.models['right']
        self.left_shapedirs_fixed = bool(torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1)
        if self.left_shapedirs_fixed:
            left.shapedirs[:, 0, :] *= -1
        self.provenance = {side: {'filename': p.name, 'sha256': sha256(p)} for side, p in paths.items()}

    def forward(self, side, records):
        """Official wrist_xform = MANO global_orient + transl, not post-hoc R*v+t."""
        if not records:
            return np.empty((0, 21, 3))
        torch = self.torch
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device='cpu')
        rotvec = [Rotation.from_quat(np.asarray(r['wrist_xform']['q_wxyz'])[[1, 2, 3, 0]]).as_rotvec() for r in records]
        with torch.no_grad():
            out = self.models[side](betas=tensor([r['betas'] for r in records]),
                                    hand_pose=tensor([r['pose'] for r in records]),
                                    global_orient=tensor(rotvec),
                                    transl=tensor([r['wrist_xform']['t_xyz'] for r in records]),
                                    return_verts=True)
        joints = np.concatenate([out.joints[:, :16].numpy(), out.vertices[:, TIP_VERTICES].numpy()], axis=1)
        return joints[:, MANO21_TO_OURS21].astype(float)

    def shape_vertices(self, side, betas):
        """Zero actual articulation / zero wrist: v_template + shapedirs * beta.

        PCA coefficient zero would add MANO's nonzero hand mean. Shape-only MPVPE
        therefore uses the rest mesh directly, without pose blend shapes or LBS.
        """
        m = self.models[side]
        b = np.asarray(betas, float)
        if b.shape[-1] != 10 or not np.isfinite(b).all():
            raise ValueError('source_betas must contain ten finite coefficients')
        return m.v_template.detach().numpy() + np.einsum('vck,...k->...vc', m.shapedirs.detach().numpy(), b)
