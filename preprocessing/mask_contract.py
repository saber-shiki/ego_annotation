"""Source-fixed silhouette evidence. No candidate-dependent occlusion exclusions."""
from dataclasses import dataclass
import hashlib

import cv2
import numpy as np

from preprocessing.geometry import camera_pose, resized_intrinsics


def boundary(mask):
    return mask & ~cv2.erode(mask.astype('uint8'), np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


@dataclass(frozen=True)
class FixedMaskDomain:
    target: np.ndarray
    owned: np.ndarray
    domain: np.ndarray
    boundary_domain: np.ndarray
    digest: str
    boundary_tolerance_px: float = 2.

    @classmethod
    def create(cls, target, owned, hand_depth, boundary_tolerance_px=2.):
        target = np.asarray(target, bool).copy()
        owned = np.asarray(owned, bool).copy()
        hand_depth = np.asarray(hand_depth)
        if target.ndim != 2 or target.shape != owned.shape or target.shape != hand_depth.shape:
            raise ValueError('Mask evidence shape mismatch')
        if not np.isfinite(boundary_tolerance_px) or boundary_tolerance_px < 0:
            raise ValueError('Invalid boundary tolerance')
        hand = np.isfinite(hand_depth) & (hand_depth > 0)
        # Owned object evidence overrides a possibly inaccurate MANO projection.
        # Raw SAM2 recall is checked separately, including positives under MANO.
        domain = ~hand | owned
        bd = cv2.erode(domain.astype('uint8'), np.ones((3, 3), np.uint8)).astype(bool)
        bd[:2] = bd[-2:] = False
        bd[:, :2] = bd[:, -2:] = False
        h = hashlib.sha256(str((target.shape, boundary_tolerance_px)).encode())
        for array in (target, owned, domain, bd):
            h.update(array.tobytes()); array.setflags(write=False)
        return cls(target, owned, domain, bd, h.hexdigest(), float(boundary_tolerance_px))

    @classmethod
    def from_inputs(cls, inputs, hand_depth, i):
        wh = hand_depth.shape[1:][::-1]
        target = cv2.resize(inputs.masks[i].astype('uint8'), wh, interpolation=cv2.INTER_NEAREST_EXACT) > 0
        masks = getattr(inputs, 'owned_masks', None)
        owned = np.zeros_like(target) if masks is None else cv2.resize(
            masks[i].astype('uint8'), wh, interpolation=cv2.INTER_NEAREST_EXACT) > 0
        return cls.create(target, owned, hand_depth[i])

    def score(self, rendered):
        rendered = np.asarray(rendered, bool)
        if rendered.shape != self.target.shape:
            raise ValueError('Candidate raster shape mismatch')

        def metrics(domain):
            truth = int(np.count_nonzero(self.target & domain))
            pred = int(np.count_nonzero(rendered & domain))
            hit = int(np.count_nonzero(rendered & self.target & domain))
            union = truth + pred - hit
            return {'available': truth > 0, 'iou': hit / max(1, union),
                    'recall': hit / max(1, truth), 'precision': hit / max(1, pred),
                    'target_pixels': truth, 'prediction_pixels': pred,
                    'domain_pixels': int(np.count_nonzero(domain)), 'spill_pixels': pred - hit}

        # Do not create fake contours at the hand exclusion boundary.
        tb = boundary(self.target) & self.boundary_domain
        pb = boundary(rendered) & self.boundary_domain
        nt, npred = int(tb.sum()), int(pb.sum())
        precision = recall = None
        if nt:
            dt = cv2.distanceTransform((~tb).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            precision = float(np.mean(dt[pb] <= self.boundary_tolerance_px)) if npred else 0.
            if npred:
                dp = cv2.distanceTransform((~pb).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                recall = float(np.mean(dp[tb] <= self.boundary_tolerance_px))
            else:
                recall = 0.
        return {'raw': metrics(np.ones_like(self.domain)), 'visible': metrics(self.domain),
                'owned': metrics(self.domain & (~self.target | self.owned)),
                'boundary': {'precision': precision, 'recall': recall, 'target_pixels': nt,
                             'prediction_pixels': npred, 'tolerance_px': self.boundary_tolerance_px},
                'fixed_domain': True, 'domain_digest': self.digest,
                'hand_excluded_pixels': int(np.count_nonzero(~self.domain)),
                'occlusion_allowance_occupied_pixels': int(np.count_nonzero(rendered & ~self.domain))}

    def render_score(self, mesh, R, t, inputs, i):
        wh = self.target.shape[::-1]
        k = resized_intrinsics(inputs.K[i], inputs.wh, wh)
        rc, tc = camera_pose(R, t, inputs.cameras[i])
        return self.score(mesh.raster(k, rc, tc, wh)[2])


def nonregression(reference, candidate, tolerance=.015, boundary_tolerance=.03):
    """Compare to a FROZEN reference, not a reference reset after every step."""
    if not np.isfinite([tolerance, boundary_tolerance]).all() or min(tolerance, boundary_tolerance) < 0:
        raise ValueError('Invalid mask nonregression tolerance')
    if not reference or not candidate:
        return False, 'missing_evidence'
    if not reference.get('fixed_domain') or not candidate.get('fixed_domain'):
        return False, 'unfixed_domain'
    if reference['domain_digest'] != candidate['domain_digest']:
        return False, 'changed_comparison_population'
    if not reference['visible']['available']:
        return False, 'no_visible_source_support'
    checks = [('raw', 'recall', tolerance)]
    checks += [(domain, key, tolerance) for domain in ('visible', 'owned')
               if reference[domain]['available'] for key in ('iou', 'recall', 'precision')]
    checks += [('boundary', key, boundary_tolerance) for key in ('precision', 'recall')
               if reference['boundary'][key] is not None]
    for domain, key, allowed in checks:
        before, after = reference[domain][key], candidate[domain][key]
        if before is None or after is None or not np.isfinite([before, after]).all():
            return False, domain + '_' + key + '_unavailable'
        if after < before - allowed - 1e-12:
            return False, domain + '_' + key + '_regressed'
    return True, 'fixed_reference_nonregressed'
