"""Hand metrics in metres internally; temporal windows never bridge missing frames."""
import numpy as np
from .alignment import fit_points, transform_points
from .mano_fk import OURS21_TO_OFFICIAL19


def metric(value, valid, total, reason=''):
    return {'value': None if value is None else float(value), 'valid': int(valid),
            'total': int(total), 'reason': reason}


def mean_metric(values, total, factor=1, reason=''):
    a = np.asarray(values, float)
    return metric(a.mean() * factor if a.size else None, a.size, total, reason)


def pck_auc(errors_mm):
    """Inclusive PCK at 101 thresholds, trapezoid integral / 50 * 100."""
    e = np.asarray(errors_mm, float).reshape(-1)
    if not e.size:
        return None
    thresholds = np.linspace(0, 50, 101)
    pck = (e[:, None] <= thresholds).mean(axis=0)
    return float(np.sum((pck[:-1] + pck[1:]) * 0.25) / 50 * 100)


def consecutive_runs(valid, frame_ids):
    runs, run = [], []
    for i, ok in enumerate(valid):
        if not ok or (run and frame_ids[i] != frame_ids[run[-1]] + 1):
            if run:
                runs.append(np.asarray(run, int))
            run = []
        if ok:
            run.append(i)
    if run:
        runs.append(np.asarray(run, int))
    return runs


class BetaDriftError(ValueError):
    """Official shape metric cannot choose a sequence-constant prediction beta."""


def constant_betas(frames, side, valid):
    """Take first valid frame; reject beta drift anywhere in the annotation."""
    first = next((i for i, v in enumerate(valid) if v), None)
    if first is None:
        return None, 'no valid prediction/GT hand samples'
    arrays = [f.get('hands', {}).get(side, {}).get('parameters', {}).get('source_betas') for f in frames]
    if arrays[first] is None:
        return None, 'source_betas missing at first valid hand frame'
    beta = np.asarray(arrays[first], float)
    if beta.shape != (10,) or not np.isfinite(beta).all():
        raise ValueError(f'{side}: source_betas must be finite length 10')
    first_drift = None
    max_drift = 0.0
    for i, other in enumerate(arrays):
        if other is None:
            if valid[i]:
                return None, f'source_betas missing at valid frame {frames[i].get("frame_idx", i)}'
            continue
        other = np.asarray(other, float)
        if other.shape != (10,) or not np.isfinite(other).all():
            raise ValueError(f'{side}: invalid source_betas at frame {i}')
        if not np.array_equal(other, beta):
            if first_drift is None:
                first_drift = frames[i].get('frame_idx', i)
            max_drift = max(max_drift, float(np.max(np.abs(other - beta))))
    if first_drift is not None:
        raise BetaDriftError(f'{side}: source_betas not constant; first drift frame={first_drift}; max abs drift from first valid beta={max_drift:.12g}')
    return beta, ''


def evaluate_hand(pred, gt, valid, frame_ids):
    pred, gt, valid = np.asarray(pred, float), np.asarray(gt, float), np.asarray(valid, bool)
    n = len(valid)
    errors = np.linalg.norm(pred[valid][:, OURS21_TO_OFFICIAL19] - gt[valid][:, OURS21_TO_OFFICIAL19], axis=-1) * 1000
    result = {'MPJPE_mm': mean_metric(errors, n * 19),
              'PCK_AUC_pct': metric(pck_auc(errors), errors.size, n * 19),
              'FINGERTIP_PCK_AUC_pct': metric(pck_auc(errors[:, :5]), len(errors) * 5, n * 5)}
    pa, reasons = [], []
    for p, g in zip(pred[valid], gt[valid]):
        try:
            pa.extend(np.linalg.norm(transform_points(fit_points(p, g, True), p) - g, axis=-1))
        except ValueError as e:
            reasons.append(str(e))
    result['PA_MPJPE_mm'] = mean_metric(pa, n * 21, 1000, '; '.join(sorted(set(reasons))))
    w, wa, accel, jerk = [], [], [], []
    dropped, segment_errors, segment_lengths = 0, [], []
    runs = consecutive_runs(valid, frame_ids)
    for run in runs:
        p, g = pred[run], gt[run]
        if len(run) >= 3:
            accel.extend(np.linalg.norm((np.diff(p, n=2, axis=0) - np.diff(g, n=2, axis=0)) * 30 ** 2, axis=-1).reshape(-1))
        if len(run) >= 4:
            jerk.extend(np.linalg.norm(np.diff(p, n=3, axis=0), axis=-1).reshape(-1))
        for start in range(0, len(run), 100):
            pp, gg = p[start:start + 100], g[start:start + 100]
            if len(pp) < 2:
                dropped += len(pp)
                continue
            segment_lengths.append(len(pp))
            for output, fit_n in [(w, 2), (wa, len(pp))]:
                try:
                    T = fit_points(pp[:fit_n], gg[:fit_n], True)
                    output.extend(np.linalg.norm(transform_points(T, pp) - gg, axis=-1).reshape(-1))
                except ValueError as e:
                    segment_errors.append(str(e))
    segment_reason = f'{len(segment_lengths)} segments, lengths={segment_lengths}; {dropped} singleton frames discarded'
    if segment_errors:
        segment_reason += '; ' + '; '.join(sorted(set(segment_errors)))
    result['W_MPJPE_mm'] = mean_metric(w, n * 21, 1000, segment_reason)
    result['WA_MPJPE_mm'] = mean_metric(wa, n * 21, 1000, segment_reason)
    result['Accel_m_s2'] = mean_metric(accel, max(0, n - 2) * 21)
    result['Jerk_3rd_diff_mm'] = mean_metric(jerk, max(0, n - 3) * 21, 1000)
    result['RTE_pct'] = metric(None, 0, n, 'needs >=2 valid wrists and nonzero GT path length')
    if valid.sum() >= 2:
        p, g = pred[valid, 0], gt[valid, 0]
        # No bridge over missing frames in the path length denominator.
        length = sum(np.linalg.norm(np.diff(gt[run, 0], axis=0), axis=-1).sum() for run in runs)
        if length > 1e-12:
            aligned = transform_points(fit_points(p, g, False), p)
            result['RTE_pct'] = metric(np.linalg.norm(aligned - g, axis=-1).mean() / length * 100, len(p), n)
    return result
