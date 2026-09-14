"""Grouped observation/numerical implementation."""
from __future__ import annotations
from utils.data import write_json_payload
from utils.data import read_json_mapping
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    (ax1, ay1, ax2, ay2) = [float(x) for x in a]
    (bx1, by1, bx2, by2) = [float(x) for x in b]
    (ix1, iy1) = (max(ax1, bx1), max(ay1, by1))
    (ix2, iy2) = (min(ax2, bx2), min(ay2, by2))
    (iw, ih) = (max(0.0, ix2 - ix1), max(0.0, iy2 - iy1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else float('nan')

def center_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=float)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=float)
    return float(np.linalg.norm(ac - bc))

def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {'count': 0}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'mean': float(np.mean(arr)), 'p10': float(np.percentile(arr, 10.0)), 'p90': float(np.percentile(arr, 90.0)), 'min': float(np.min(arr)), 'max': float(np.max(arr))}

def gt_box(row: dict[str, Any], side: str, stream_id: str, visibility_threshold: float) -> tuple[np.ndarray | None, float | None]:
    hands = row.get('json', {}).get('hands.json')
    if not isinstance(hands, dict):
        return (None, None)
    hand = hands.get(side)
    if not isinstance(hand, dict):
        return (None, None)
    visibility = None
    vis = hand.get('visibilities_modeled')
    if isinstance(vis, dict) and stream_id in vis:
        visibility = float(vis[stream_id])
        if visibility < visibility_threshold:
            return (None, visibility)
    boxes = hand.get('boxes_amodal')
    if not isinstance(boxes, dict) or stream_id not in boxes:
        return (None, visibility)
    arr = np.asarray(boxes[stream_id], dtype=float)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        return (None, visibility)
    return (arr, visibility)

def pred_box(npz: Any, side: str, idx: int, score_threshold: float) -> tuple[np.ndarray | None, float | None, bool]:
    box_key = f'{side}_det_box_xyxyscore'
    valid_key = f'{side}_valid'
    detected_key = f'{side}_detected_same_frame'
    if box_key not in npz.files:
        return (None, None, False)
    valid = bool(npz[valid_key][idx]) if valid_key in npz.files and idx < len(npz[valid_key]) else True
    detected = bool(npz[detected_key][idx]) if detected_key in npz.files and idx < len(npz[detected_key]) else valid
    raw = np.asarray(npz[box_key][idx], dtype=float)
    if raw.shape[0] < 4 or not np.isfinite(raw[:4]).all():
        return (None, None, detected)
    score = float(raw[4]) if raw.shape[0] >= 5 and np.isfinite(raw[4]) else None
    if score is not None and score < score_threshold:
        return (None, score, detected)
    return (raw[:4], score, detected)

def evaluate_boxes(args: argparse.Namespace) -> dict[str, Any]:
    gt = read_json_mapping(args.hot3d_gt)
    frames = gt.get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError('HOT3D GT sidecar has no frames')
    npz = np.load(args.hawor_npz, allow_pickle=True)
    frame_idx = np.asarray(npz['frame_idx'], dtype=int) if 'frame_idx' in npz.files else np.arange(len(frames), dtype=int)
    row_by_frame = {int(row['frame_idx']): row for row in frames if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    for (local_i, frame) in enumerate(frame_idx.tolist()):
        if local_i >= len(frame_idx):
            continue
        gt_row = row_by_frame.get(int(frame))
        if gt_row is None:
            continue
        for side in ('left', 'right'):
            (gbox, visibility) = gt_box(gt_row, side, args.stream_id, args.visibility_threshold)
            (pbox, score, detected) = pred_box(npz, side, local_i, args.score_threshold)
            measurable = gbox is not None
            matched = measurable and pbox is not None
            rows.append({'frame_idx': int(frame), 'side': side, 'gt_visibility': visibility, 'gt_box_xyxy': gbox.tolist() if gbox is not None else None, 'pred_box_xyxy': pbox.tolist() if pbox is not None else None, 'pred_score': score, 'pred_detected_same_frame': bool(detected), 'measurable': bool(measurable), 'matched': bool(matched), 'iou': iou_xyxy(gbox, pbox) if matched else None, 'center_distance_px': center_distance_px(gbox, pbox) if matched else None})
    measurable_rows = [r for r in rows if r['measurable']]
    matched_rows = [r for r in measurable_rows if r['matched']]
    by_side: dict[str, Any] = {}
    for side in ('left', 'right'):
        sr = [r for r in measurable_rows if r['side'] == side]
        sm = [r for r in matched_rows if r['side'] == side]
        by_side[side] = {'measurable_frames': len(sr), 'matched_frames': len(sm), 'match_rate': float(len(sm) / max(1, len(sr))), 'iou': summarize([float(r['iou']) for r in sm if r['iou'] is not None]), 'center_distance_px': summarize([float(r['center_distance_px']) for r in sm if r['center_distance_px'] is not None]), 'pred_score': summarize([float(r['pred_score']) for r in sm if r['pred_score'] is not None])}
    report = {'status': 'ok', 'method': 'evaluate_v19_hot3d_hawor_boxes', 'claim_scope': '2D hand-box localization against HOT3D amodal boxes; not 3D MANO/contact/object-pose scoring', 'hot3d_gt': str(args.hot3d_gt), 'hawor_npz': str(args.hawor_npz), 'stream_id': args.stream_id, 'visibility_threshold': float(args.visibility_threshold), 'score_threshold': float(args.score_threshold), 'frame_count_gt': len(frames), 'rows': rows, 'summary': {'measurable_rows': len(measurable_rows), 'matched_rows': len(matched_rows), 'match_rate': float(len(matched_rows) / max(1, len(measurable_rows))), 'iou': summarize([float(r['iou']) for r in matched_rows if r['iou'] is not None]), 'center_distance_px': summarize([float(r['center_distance_px']) for r in matched_rows if r['center_distance_px'] is not None]), 'by_side': by_side}}
    write_json_payload(args.output_report, report)
    print(json.dumps({k: v for (k, v) in report.items() if k != 'rows'}, indent=2)[:20000])
    return report

def boxes_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hot3d-gt', type=Path, required=True)
    parser.add_argument('--hawor-npz', type=Path, required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    parser.add_argument('--stream-id', default='214-1')
    parser.add_argument('--visibility-threshold', type=float, default=0.1)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    return parser.parse_args(argv)
import argparse
import inspect
import json
import math
import tarfile
from pathlib import Path
from typing import Any
import cv2
import numpy as np
MANO_FINGERTIP_VERT_INDICES = [744, 320, 443, 554, 671]
HAWOR_MANO_TO_OPENPOSE_MAPPING = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

def patch_legacy_numpy() -> None:
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for (name, value) in [('bool', np.bool_), ('int', int), ('float', float), ('complex', complex), ('object', object), ('unicode', str), ('str', str)]:
        if not hasattr(np, name):
            setattr(np, name, value)

def quat_wxyz_to_matrix(q: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    (w, x, y, z) = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        raise RuntimeError('zero quaternion')
    (w, x, y, z) = (w / n, x / n, y / n, z / n)
    return np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)

def se3_from_hot3d_dict(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (quat_wxyz_to_matrix(d['quaternion_wxyz']), np.asarray(d['translation_xyz'], dtype=np.float64))

def world_to_camera(points_world: np.ndarray, R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (np.asarray(points_world, dtype=np.float64) - np.asarray(t_c2w, dtype=np.float64)[None, :]) @ np.asarray(R_c2w, dtype=np.float64)

def load_hand_shape(gt: dict[str, Any], sidecar_path: Path) -> tuple[np.ndarray, str]:
    if isinstance(gt.get('hand_shapes'), dict) and 'mano' in gt['hand_shapes']:
        return (np.asarray(gt['hand_shapes']['mano'], dtype=np.float32), 'sidecar:hand_shapes')
    sibling = sidecar_path.parent / 'hot3d_hand_shapes.json'
    if sibling.exists():
        shape = read_json_mapping(sibling)
        if 'mano' not in shape:
            raise RuntimeError(f'{sibling} lacks mano hand shape')
        return (np.asarray(shape['mano'], dtype=np.float32), str(sibling))
    tar_path = Path(str(gt.get('source_tar', ''))).expanduser()
    if tar_path.exists():
        with tarfile.open(tar_path, 'r') as tar:
            if '__hand_shapes.json__' not in tar.getnames():
                raise RuntimeError(f'{tar_path} lacks __hand_shapes.json__')
            shape = json.load(tar.extractfile('__hand_shapes.json__'))
        return (np.asarray(shape['mano'], dtype=np.float32), str(tar_path))
    raise RuntimeError('HOT3D hand shape missing: sidecar has no hand_shapes, no sibling hot3d_hand_shapes.json, and source_tar is unavailable')

def load_smplx_mano(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    patch_legacy_numpy()
    try:
        import smplx
    except Exception as exc:
        raise RuntimeError('smplx is required to replay HOT3D MANO GT. Run this script in the HaWoR/remote environment or install smplx.') from exc
    left_path = args.mano_left.expanduser()
    right_path = args.mano_right.expanduser()
    if not left_path.exists():
        raise FileNotFoundError(f'missing MANO_LEFT model: {left_path}')
    if not right_path.exists():
        raise FileNotFoundError(f'missing MANO_RIGHT model: {right_path}')
    left = smplx.create(str(left_path), 'mano', use_pca=True, is_rhand=False, num_pca_comps=15)
    right = smplx.create(str(right_path), 'mano', use_pca=True, is_rhand=True, num_pca_comps=15)
    if torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
        left.shapedirs[:, 0, :] *= -1
    return {'left': left, 'right': right}

def replay_hot3d_mano(layer: Any, beta: np.ndarray, theta: np.ndarray, wrist_xform: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    beta_t = torch.as_tensor(beta[None, :], dtype=torch.float32, device=device)
    theta_t = torch.as_tensor(theta[None, :], dtype=torch.float32, device=device)
    wrist_t = torch.as_tensor(wrist_xform[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = layer(betas=beta_t, global_orient=wrist_t[:, :3], hand_pose=theta_t, transl=wrist_t[:, 3:], return_verts=True)
    vertices = out.vertices.detach().cpu().numpy()[0].astype(np.float64)
    joints = out.joints.detach().cpu().numpy()[0].astype(np.float64)
    if joints.shape[0] != 21:
        tips = vertices[np.asarray(MANO_FINGERTIP_VERT_INDICES, dtype=int)]
        joints = np.concatenate([joints, tips], axis=0)
    if joints.shape[0] != 21:
        raise RuntimeError(f'expected 21 GT MANO joints after fingertip append, got {joints.shape}')
    faces = np.asarray(layer.faces, dtype=np.int32)
    return (vertices, joints, faces)

def mano_eval_summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {'count': 0}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'mean': float(np.mean(arr)), 'p10': float(np.percentile(arr, 10)), 'p90': float(np.percentile(arr, 90)), 'p95': float(np.percentile(arr, 95)), 'min': float(np.min(arr)), 'max': float(np.max(arr))}

def project_pinhole(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    z = points_cam[:, 2]
    out = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float64)
    valid = z > 1e-06
    out[valid, 0] = points_cam[valid, 0] / z[valid] * fx + cx
    out[valid, 1] = points_cam[valid, 1] / z[valid] * fy + cy
    return out

def get_k_for_frame(gt_row: dict[str, Any], stream_id: str) -> tuple[float, float, float, float]:
    params = gt_row['json']['cameras.json'][stream_id]['calibration']['projection_params']
    (f, cx, cy) = [float(v) for v in params[:3]]
    return (f, f, cx, cy)

def per_joint_errors(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float, float, float]:
    e = np.linalg.norm(pred - gt, axis=1)
    wrist = float(e[0])
    mpjpe = float(np.mean(e))
    med = float(np.median(e))
    pred_rel = pred - pred[0:1]
    gt_rel = gt - gt[0:1]
    er = np.linalg.norm(pred_rel - gt_rel, axis=1)
    return (wrist, mpjpe, med, float(np.mean(er)), float(np.median(er)), float(np.percentile(er, 95)))

def draw_points(image: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    (h, w) = image.shape[:2]
    for p in pts:
        if not np.isfinite(p).all():
            continue
        (x, y) = (int(round(float(p[0]))), int(round(float(p[1]))))
        if -20 <= x < w + 20 and -20 <= y < h + 20:
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

def localize_prediction_path(path: str | Path, args: argparse.Namespace) -> Path:
    direct = Path(path).expanduser()
    if direct.exists():
        return direct
    remote_root = getattr(args, 'remote_root', None)
    local_root = getattr(args, 'local_root', None)
    if remote_root is not None and local_root is not None:
        remote_root = Path(remote_root)
        local_root = Path(local_root)
        for (src, dst) in ((remote_root, local_root), (local_root, remote_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(str(path))

def load_hawor_prediction(args: argparse.Namespace, frames_list: list[dict[str, Any]]) -> dict[str, Any]:
    if args.hawor_npz is None:
        raise RuntimeError('--hawor-npz is required unless --interval-state is supplied')
    npz_path = localize_prediction_path(args.hawor_npz, args)
    npz = np.load(npz_path, allow_pickle=True)
    frame_idx = np.asarray(npz['frame_idx'], dtype=int) if 'frame_idx' in npz.files else np.arange(len(frames_list), dtype=int)
    sides: dict[str, dict[str, Any]] = {}
    for side in ('left', 'right'):
        sides[side] = {'valid': np.asarray(npz[f'{side}_valid']).astype(bool) if f'{side}_valid' in npz.files else np.ones(frame_idx.shape[0], dtype=bool), 'joints_world_m': np.asarray(npz[f'{side}_joints_world_m'], dtype=np.float64), 'vertices_world_m': np.asarray(npz[f'{side}_vertices_world_m'], dtype=np.float64), 'detected_same_frame': np.asarray(npz[f'{side}_detected_same_frame']).astype(bool) if f'{side}_detected_same_frame' in npz.files else None, 'row_meta': [None] * int(frame_idx.shape[0])}
    return {'kind': 'hawor_npz', 'path': str(npz_path), 'frame_idx': frame_idx, 'R_c2w': np.asarray(npz['R_c2w'], dtype=np.float64), 't_c2w': np.asarray(npz['t_c2w'], dtype=np.float64), 'sides': sides, 'full_vertices_available': True, 'source_hawor_npz': str(npz_path)}

def camera_trajectory_from_annotations(annotation_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the V19 render/annotation camera trajectory.

    Interval states are not intrinsically HaWoR predictions: newer V19 states may
    store optimized world-frame joints without a source NPZ path.  In that case
    the physically correct camera trajectory is the one consumed by the renderer
    from ``annotations_v19_*.json``.  The renderer interprets
    ``T_world_camera[:3,:3]`` as camera-to-world rotation and ``[:3,3]`` as camera
    position, and projects world points as ``(p_world - t_c2w) @ R_c2w``.  The
    evaluator uses the same convention.
    """
    annotations = read_json_mapping(localize_prediction_path(annotation_path, args))
    frames = annotations.get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f'{annotation_path} lacks frames for interval-state camera trajectory')
    frame_ids: list[int] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for (pos, frame) in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        camera = frame.get('camera') if isinstance(frame.get('camera'), dict) else {}
        raw_T = camera.get('T_world_camera_metric') or camera.get('T_world_camera')
        if raw_T is None:
            continue
        T = np.asarray(raw_T, dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"invalid annotation camera transform for frame {frame.get('frame_idx', pos)} in {annotation_path}")
        frame_ids.append(int(frame.get('frame_idx', pos)))
        rotations.append(T[:3, :3].copy())
        translations.append(T[:3, 3].copy())
    if not frame_ids:
        raise RuntimeError(f'{annotation_path} has no usable camera transforms')
    return (np.asarray(frame_ids, dtype=int), np.stack(rotations), np.stack(translations))

def load_interval_state_prediction(args: argparse.Namespace, frames_list: list[dict[str, Any]]) -> dict[str, Any]:
    if args.interval_state is None:
        raise RuntimeError('internal error: interval-state path missing')
    interval_path = localize_prediction_path(args.interval_state, args)
    state = read_json_mapping(interval_path)
    rows = state.get('per_frame_states')
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f'{interval_path} lacks nonempty per_frame_states')
    source_candidates = [r.get('source_hawor_npz') for r in rows if isinstance(r, dict) and r.get('source_hawor_npz')]
    camera_source: str
    npz: Any | None = None
    source_npz_path: Path | None = None
    if source_candidates:
        source_npz_path = localize_prediction_path(source_candidates[0], args)
        npz = np.load(source_npz_path, allow_pickle=True)
        frame_idx = np.asarray(npz['frame_idx'], dtype=int) if 'frame_idx' in npz.files else np.arange(len(frames_list), dtype=int)
        R_c2w = np.asarray(npz['R_c2w'], dtype=np.float64)
        t_c2w = np.asarray(npz['t_c2w'], dtype=np.float64)
        camera_source = f'source_hawor_npz:{source_npz_path}'
    else:
        inputs = state.get('inputs') if isinstance(state.get('inputs'), dict) else {}
        annotation_value = inputs.get('annotations') or state.get('annotations')
        if not annotation_value:
            raise RuntimeError(f'{interval_path} lacks source_hawor_npz and inputs.annotations; camera trajectory is required')
        (frame_idx, R_c2w, t_c2w) = camera_trajectory_from_annotations(Path(str(annotation_value)), args)
        camera_source = f'annotations:{annotation_value}'
    frame_to_i = {int(f): i for (i, f) in enumerate(frame_idx.tolist())}
    sides: dict[str, dict[str, Any]] = {}
    for side in ('left', 'right'):
        sides[side] = {'valid': np.zeros(frame_idx.shape[0], dtype=bool), 'joints_world_m': np.full((frame_idx.shape[0], 21, 3), np.nan, dtype=np.float64), 'vertices_world_m': None, 'detected_same_frame': np.asarray(npz[f'{side}_detected_same_frame']).astype(bool) if npz is not None and f'{side}_detected_same_frame' in npz.files else None, 'row_meta': [None] * int(frame_idx.shape[0])}
    for row in rows:
        if not isinstance(row, dict):
            continue
        side = str(row.get('hand_side'))
        if side not in sides:
            continue
        frame = int(row.get('frame_idx'))
        if frame not in frame_to_i:
            continue
        joints = np.asarray(row.get('optimized_joints_world_m') if row.get('optimized_joints_world_m') is not None else [], dtype=np.float64)
        if joints.shape != (21, 3) or not np.isfinite(joints).all():
            continue
        i = frame_to_i[frame]
        sides[side]['joints_world_m'][i] = joints
        sides[side]['valid'][i] = True
        sides[side]['row_meta'][i] = {'temporal_mano_state': row.get('temporal_mano_state'), 'contact_patch_state_optimized': row.get('contact_patch_state_optimized'), 'visible_surface_depth_order_selected_vertex_count': row.get('visible_surface_depth_order_selected_vertex_count'), 'optimized_vertices_world_sample_count': len(row.get('optimized_vertices_world_sample_m') or []), 'source_frame_index': row.get('source_frame_index')}
    return {'kind': 'interval_state_optimized_joints', 'path': str(interval_path), 'frame_idx': frame_idx, 'R_c2w': np.asarray(R_c2w, dtype=np.float64), 't_c2w': np.asarray(t_c2w, dtype=np.float64), 'sides': sides, 'full_vertices_available': False, 'source_hawor_npz': str(source_npz_path) if source_npz_path is not None else None, 'camera_trajectory_source': camera_source, 'interval_summary': state.get('summary')}

def load_prediction(args: argparse.Namespace, frames_list: list[dict[str, Any]]) -> dict[str, Any]:
    if args.interval_state is not None and args.hawor_npz is not None:
        raise RuntimeError('supply exactly one prediction source: --hawor-npz or --interval-state')
    if args.interval_state is not None:
        return load_interval_state_prediction(args, frames_list)
    return load_hawor_prediction(args, frames_list)

def row_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        value = float(row[key])
        if np.isfinite(value):
            values.append(value)
    return values

def render_review(args: argparse.Namespace, rows: list[dict[str, Any]], gt_frames: dict[int, dict[str, Any]], pred_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]) -> str | None:
    if args.review_output is None or args.image_manifest is None:
        return None
    manifest = read_json_mapping(args.image_manifest)
    frame_paths = {int(r['frame_idx']): Path(r['raw_frame_path']) for r in manifest.get('frames', []) if isinstance(r, dict) and r.get('raw_frame_path')}
    by_frame_side = {(int(r['frame_idx']), str(r['side'])): r for r in rows if r.get('matched')}
    tiles: list[np.ndarray] = []
    for frame_idx in args.review_frames:
        path = frame_paths.get(int(frame_idx))
        gt_row = gt_frames.get(int(frame_idx))
        if path is None or gt_row is None or (not path.exists()):
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        vis = image.copy()
        (fx, fy, cx, cy) = get_k_for_frame(gt_row, args.stream_id)
        lines = [f'frame {frame_idx:06d}  GT green, pred yellow/cyan']
        for (side, color) in [('left', (0, 255, 255)), ('right', (255, 255, 0))]:
            row = by_frame_side.get((int(frame_idx), side))
            key = (int(frame_idx), side)
            if row is None or key not in pred_cache:
                lines.append(f'{side}: no matched 3D row')
                continue
            (gt_cam, pred_cam) = pred_cache[key]
            gt_uv = project_pinhole(gt_cam, fx, fy, cx, cy)
            pred_uv = project_pinhole(pred_cam, fx, fy, cx, cy)
            draw_points(vis, gt_uv, (0, 255, 0), 4)
            draw_points(vis, pred_uv, color, 3)
            lines.append(f"{side}: wrist {row['wrist_error_m'] * 1000:.1f}mm, MPJPE {row['joint_mpjpe_m'] * 1000:.1f}mm, wrist-sub {row['root_aligned_mpjpe_m'] * 1000:.1f}mm")
        banner = np.zeros((74, vis.shape[1], 3), dtype=np.uint8)
        banner[:] = (8, 8, 8)
        for (j, text) in enumerate(lines[:3]):
            cv2.putText(banner, text, (12, 23 + 23 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([banner, vis]))
    if not tiles:
        return None
    width = min(1600, max((t.shape[1] for t in tiles)))
    resized = []
    for t in tiles:
        if t.shape[1] != width:
            scale = width / t.shape[1]
            t = cv2.resize(t, (width, int(round(t.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        resized.append(t)
    sheet = np.vstack(resized)
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.review_output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f'failed to write {args.review_output}')
    return str(args.review_output)

def evaluate_mano(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    gt = read_json_mapping(args.hot3d_gt)
    frames_list = gt.get('frames')
    if not isinstance(frames_list, list) or not frames_list:
        raise RuntimeError('HOT3D GT sidecar has no frames')
    gt_frames = {int(row['frame_idx']): row for row in frames_list if isinstance(row, dict)}
    (beta, hand_shape_source) = load_hand_shape(gt, args.hot3d_gt)
    prediction = load_prediction(args, frames_list)
    layers = load_smplx_mano(args)
    device = torch.device(args.device)
    for layer in layers.values():
        layer.to(device)
        layer.eval()
    frame_idx = np.asarray(prediction['frame_idx'], dtype=int)
    rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (local_i, frame) in enumerate(frame_idx.tolist()):
        gt_row = gt_frames.get(int(frame))
        if gt_row is None:
            continue
        hot3d_cam = gt_row.get('json', {}).get('cameras.json', {}).get(args.stream_id)
        if not isinstance(hot3d_cam, dict):
            continue
        (R_hot3d_c2w, t_hot3d_c2w) = se3_from_hot3d_dict(hot3d_cam['T_world_from_camera'])
        R_pred_c2w = np.asarray(prediction['R_c2w'][local_i], dtype=np.float64)
        t_pred_c2w = np.asarray(prediction['t_c2w'][local_i], dtype=np.float64)
        for side in ('left', 'right'):
            hand = gt_row.get('json', {}).get('hands.json', {}).get(side)
            if not isinstance(hand, dict) or 'mano_pose' not in hand:
                rows.append({'frame_idx': int(frame), 'side': side, 'measurable': False, 'matched': False, 'reason': 'missing_hot3d_mano_pose'})
                continue
            side_pred = prediction['sides'][side]
            pred_valid = bool(side_pred['valid'][local_i])
            if not pred_valid:
                rows.append({'frame_idx': int(frame), 'side': side, 'measurable': True, 'matched': False, 'reason': f"missing_{prediction['kind']}_prediction"})
                continue
            mano = hand['mano_pose']
            theta = np.asarray(mano['thetas'], dtype=np.float32)
            wrist_xform = np.asarray(mano['wrist_xform'], dtype=np.float32)
            (gt_verts_w, gt_joints_raw_w, _) = replay_hot3d_mano(layers[side], beta, theta, wrist_xform, device)
            pred_joints_w = np.asarray(side_pred['joints_world_m'][local_i], dtype=np.float64)
            gt_joints_raw_cam = world_to_camera(gt_joints_raw_w, R_hot3d_c2w, t_hot3d_c2w)
            gt_joints_hawor_order_cam = gt_joints_raw_cam[np.asarray(HAWOR_MANO_TO_OPENPOSE_MAPPING, dtype=int)]
            pred_joints_cam = world_to_camera(pred_joints_w, R_pred_c2w, t_pred_c2w)
            (wrist, mpjpe, med, rel_mpjpe, rel_med, rel_p95) = per_joint_errors(pred_joints_cam, gt_joints_hawor_order_cam)
            vertex_centroid_err = None
            pred_vertices = side_pred.get('vertices_world_m')
            if pred_vertices is not None:
                pred_verts_w = np.asarray(pred_vertices[local_i], dtype=np.float64)
                pred_verts_cam = world_to_camera(pred_verts_w, R_pred_c2w, t_pred_c2w)
                vertex_centroid_err = float(np.linalg.norm(np.mean(pred_verts_cam, axis=0) - np.mean(world_to_camera(gt_verts_w, R_hot3d_c2w, t_hot3d_c2w), axis=0)))
            detected_same = side_pred.get('detected_same_frame')
            row_meta = side_pred.get('row_meta', [None] * len(frame_idx))[local_i]
            row = {'frame_idx': int(frame), 'side': side, 'measurable': True, 'matched': True, 'gt_visibility_modeled': hand.get('visibilities_modeled', {}).get(args.stream_id) if isinstance(hand.get('visibilities_modeled'), dict) else None, 'pred_detected_same_frame': bool(detected_same[local_i]) if detected_same is not None else None, 'wrist_error_m': wrist, 'joint_mpjpe_m': mpjpe, 'joint_median_error_m': med, 'root_aligned_mpjpe_m': rel_mpjpe, 'root_aligned_median_error_m': rel_med, 'root_aligned_p95_error_m': rel_p95, 'joint_order': 'HaWoR 21-joint export order: MANO joints plus fingertips reordered by [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]', 'prediction_source_kind': prediction['kind']}
            if vertex_centroid_err is not None:
                row['vertex_centroid_error_m'] = vertex_centroid_err
            if row_meta is not None:
                row['prediction_row_meta'] = row_meta
            rows.append(row)
            if int(frame) in args.review_frames:
                pred_cache[int(frame), side] = (gt_joints_hawor_order_cam, pred_joints_cam)
    matched = [r for r in rows if r.get('matched')]
    measurable = [r for r in rows if r.get('measurable')]
    by_side: dict[str, Any] = {}
    for side in ('left', 'right'):
        sm = [r for r in matched if r['side'] == side]
        sr = [r for r in measurable if r['side'] == side]
        by_side[side] = {'measurable_rows': len(sr), 'matched_rows': len(sm), 'match_rate': float(len(sm) / max(1, len(sr))), 'wrist_error_m': mano_eval_summarize(row_metric_values(sm, 'wrist_error_m')), 'joint_mpjpe_m': mano_eval_summarize(row_metric_values(sm, 'joint_mpjpe_m')), 'root_aligned_mpjpe_m': mano_eval_summarize(row_metric_values(sm, 'root_aligned_mpjpe_m')), 'vertex_centroid_error_m': mano_eval_summarize(row_metric_values(sm, 'vertex_centroid_error_m'))}
    review = render_review(args, rows, gt_frames, pred_cache)
    vertex_metric_scope = 'full predicted MANO vertices' if prediction.get('full_vertices_available') else 'not reported: interval state stores optimized joints and sampled vertices, not full predicted MANO vertices'
    report = {'status': 'ok', 'method': 'evaluate_v19_hot3d_hawor_mano3d', 'claim_scope': '3D MANO hand joint localization/articulation against HOT3D MANO in camera coordinates; root_aligned metrics are wrist-subtracted translation-aligned only; not contact, occlusion, nonpenetration, or object-pose scoring', 'hot3d_gt': str(args.hot3d_gt), 'prediction_source': {'kind': prediction['kind'], 'path': prediction['path'], 'source_hawor_npz': prediction.get('source_hawor_npz'), 'camera_trajectory_source': prediction.get('camera_trajectory_source', prediction.get('source_hawor_npz')), 'full_vertices_available': bool(prediction.get('full_vertices_available')), 'vertex_metric_scope': vertex_metric_scope}, 'hawor_npz': str(args.hawor_npz) if args.hawor_npz is not None else None, 'interval_state': str(args.interval_state) if args.interval_state is not None else None, 'stream_id': args.stream_id, 'mano_left': str(args.mano_left), 'mano_right': str(args.mano_right), 'hand_shape_source': hand_shape_source, 'review': review, 'summary': {'measurable_rows': len(measurable), 'matched_rows': len(matched), 'match_rate': float(len(matched) / max(1, len(measurable))), 'root_aligned_metric_definition': "Subtract each hand's wrist joint translation before computing joint errors; no rotation or scale alignment is applied.", 'wrist_error_m': mano_eval_summarize(row_metric_values(matched, 'wrist_error_m')), 'joint_mpjpe_m': mano_eval_summarize(row_metric_values(matched, 'joint_mpjpe_m')), 'joint_median_error_m': mano_eval_summarize(row_metric_values(matched, 'joint_median_error_m')), 'root_aligned_mpjpe_m': mano_eval_summarize(row_metric_values(matched, 'root_aligned_mpjpe_m')), 'root_aligned_median_error_m': mano_eval_summarize(row_metric_values(matched, 'root_aligned_median_error_m')), 'vertex_centroid_error_m': mano_eval_summarize(row_metric_values(matched, 'vertex_centroid_error_m')), 'by_side': by_side}, 'rows': rows}
    write_json_payload(args.output_report, report)
    print(json.dumps({k: v for (k, v) in report.items() if k != 'rows'}, indent=2)[:20000])
    return report

def mano_eval_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hot3d-gt', type=Path, required=True)
    parser.add_argument('--hawor-npz', type=Path, default=None, help='HaWoR-like prediction NPZ baseline. Mutually exclusive with --interval-state.')
    parser.add_argument('--interval-state', type=Path, default=None, help='V19 interval MANO correction state JSON; evaluates optimized joints only.')
    parser.add_argument('--output-report', type=Path, required=True)
    parser.add_argument('--stream-id', default='214-1')
    parser.add_argument('--image-manifest', type=Path, default=None)
    parser.add_argument('--review-output', type=Path, default=None)
    parser.add_argument('--review-frames', type=int, nargs='*', default=[0, 50, 100, 149])
    parser.add_argument('--mano-left', type=Path, required=True)
    parser.add_argument('--mano-right', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--remote-root', type=Path, default=None, help='Optional path prefix remapping source root for copied prediction artifacts.')
    parser.add_argument('--local-root', type=Path, default=None, help='Optional path prefix remapping destination root for copied prediction artifacts.')
    args = parser.parse_args(argv)
    if (args.hawor_npz is None) == (args.interval_state is None):
        parser.error('supply exactly one prediction source: --hawor-npz or --interval-state')
    return args
import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable
import numpy as np

def _as_np4(m: Any) -> np.ndarray:
    arr = np.asarray(m, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f'expected 4x4 transform, got {arr.shape}')
    return arr

def quat_wxyz_to_R(q: Iterable[float]) -> np.ndarray:
    (w, x, y, z) = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        raise ValueError('zero quaternion')
    (w, x, y, z) = (w / n, x / n, y / n, z / n)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)

def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q

def quat_average_wxyz(quats: list[np.ndarray]) -> np.ndarray:
    if not quats:
        raise ValueError('no quaternions')
    ref = quats[0]
    A = np.zeros((4, 4), dtype=float)
    for q in quats:
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        if float(np.dot(q, ref)) < 0:
            q = -q
        A += np.outer(q, q)
    (vals, vecs) = np.linalg.eigh(A)
    q = vecs[:, int(np.argmax(vals))]
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q

def make_T(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(list(t), dtype=float)
    return T

def transform_from_hot3d(x: dict[str, Any]) -> np.ndarray:
    return make_T(quat_wxyz_to_R(x['quaternion_wxyz']), x['translation_xyz'])

def transform_angle_rad(T: np.ndarray) -> float:
    R = np.asarray(T[:3, :3], dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.acos(c)

def object_eval_summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    vals.sort()
    if not vals:
        return {'count': 0}

    def q(p: float) -> float:
        return vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]
    return {'count': len(vals), 'min': vals[0], 'median': float(np.median(vals)), 'mean': float(np.mean(vals)), 'p90': q(0.9), 'p95': q(0.95), 'max': vals[-1]}

def load_v19_rows(render_state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = render_state.get('object_pose_trajectory', {}).get('pose_rows') or []
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict) or 'frame_idx' not in r:
            continue
        R = np.asarray(r.get('rotation_world_from_completed_canonical_matrix'), dtype=float)
        t = np.asarray(r.get('translation_world_m'), dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            continue
        out[int(r['frame_idx'])] = r
    return out

def v19_T_world_obj(row: dict[str, Any]) -> np.ndarray:
    return make_T(row['rotation_world_from_completed_canonical_matrix'], row['translation_world_m'])

def load_v19_cameras(annotation_path: Path) -> dict[int, np.ndarray]:
    d = json.loads(annotation_path.read_text())
    cams: dict[int, np.ndarray] = {}
    for fr in d.get('frames', []):
        cam = fr.get('camera') or {}
        T = cam.get('T_world_camera_metric') or cam.get('T_world_camera')
        if T is None:
            continue
        cams[int(fr['frame_idx'])] = _as_np4(T)
    return cams

def load_hot3d_camera_object(sidecar_path: Path, object_bop_id: str, stream_id: str) -> dict[int, dict[str, Any]]:
    d = json.loads(sidecar_path.read_text())
    out: dict[int, dict[str, Any]] = {}
    for fr in d.get('frames', []):
        idx = int(fr['frame_idx'])
        j = fr.get('json') or {}
        cams = j.get('cameras.json') or {}
        objs = j.get('objects.json') or {}
        cam = cams.get(stream_id)
        entries = objs.get(str(object_bop_id)) or []
        if not cam or not entries:
            continue
        obj = entries[0]
        T_world_cam = transform_from_hot3d(cam['T_world_from_camera'])
        T_world_obj = transform_from_hot3d(obj['T_world_from_object'])
        out[idx] = {'T_world_camera': T_world_cam, 'T_world_object': T_world_obj, 'object_name': obj.get('object_name'), 'visibility_modeled': (obj.get('visibilities_modeled') or {}).get(stream_id), 'box_amodal': (obj.get('boxes_amodal') or {}).get(stream_id)}
    return out

def evaluate_object_trajectory(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-state', required=True, type=Path)
    ap.add_argument('--hot3d-gt', required=True, type=Path)
    ap.add_argument('--object-bop-id', required=True)
    ap.add_argument('--stream-id', default='214-1')
    ap.add_argument('--output-report', required=True, type=Path)
    ap.add_argument('--min-visibility', type=float, default=0.0)
    ap.add_argument('--direct-only', action='store_true', help='Evaluate only direct visible object-pose observations, excluding temporal holds.')
    args = ap.parse_args(argv)
    render_state = json.loads(args.render_state.read_text())
    ann_path = Path(render_state['inputs']['annotations'])
    v19_rows = load_v19_rows(render_state)
    v19_cams = load_v19_cameras(ann_path)
    gt_rows = load_hot3d_camera_object(args.hot3d_gt, str(args.object_bop_id), args.stream_id)
    rows: list[dict[str, Any]] = []
    Xs: list[np.ndarray] = []
    for idx in sorted(set(v19_rows) & set(v19_cams) & set(gt_rows)):
        vrow = v19_rows[idx]
        if args.direct_only and (not (vrow.get('temporal_pose_graph') or {}).get('direct_visible_measurement', True)):
            continue
        gt = gt_rows[idx]
        vis = gt.get('visibility_modeled')
        if vis is not None and float(vis) < args.min_visibility:
            continue
        T_v19_cam_obj = np.linalg.inv(v19_cams[idx]) @ v19_T_world_obj(vrow)
        T_gt_cam_obj = np.linalg.inv(gt['T_world_camera']) @ gt['T_world_object']
        X = np.linalg.inv(T_v19_cam_obj) @ T_gt_cam_obj
        Xs.append(X)
        rows.append({'frame_idx': idx, 'v19_pose_status': vrow.get('status'), 'v19_pose_measurement_status': vrow.get('pose_measurement_status'), 'v19_direct_visible_measurement': (vrow.get('temporal_pose_graph') or {}).get('direct_visible_measurement'), 'gt_visibility_modeled': vis, 'gt_box_amodal': gt.get('box_amodal'), 'relative_transform_translation_m': X[:3, 3].tolist(), 'relative_transform_rotation_angle_rad': transform_angle_rad(X)})
    if not rows:
        raise SystemExit('no common evaluable frames')
    q_mean = quat_average_wxyz([R_to_quat_wxyz(X[:3, :3]) for X in Xs])
    R_mean = quat_wxyz_to_R(q_mean)
    t_mean = np.mean([X[:3, 3] for X in Xs], axis=0)
    X_fit = make_T(R_mean, t_mean)
    rel_t = [float(np.linalg.norm(X[:3, 3] - t_mean)) for X in Xs]
    rel_rot = [transform_angle_rad(np.linalg.inv(X_fit) @ X) for X in Xs]
    residual_t: list[float] = []
    residual_rot: list[float] = []
    camera_depth_delta: list[float] = []
    for (row, X) in zip(rows, Xs):
        idx = int(row['frame_idx'])
        T_v19_cam_obj = np.linalg.inv(v19_cams[idx]) @ v19_T_world_obj(v19_rows[idx])
        T_gt_cam_obj = np.linalg.inv(gt_rows[idx]['T_world_camera']) @ gt_rows[idx]['T_world_object']
        T_res = np.linalg.inv(T_gt_cam_obj) @ (T_v19_cam_obj @ X_fit)
        residual_t.append(float(np.linalg.norm(T_res[:3, 3])))
        residual_rot.append(transform_angle_rad(T_res))
        pred_origin_cam = (T_v19_cam_obj @ X_fit)[:3, 3]
        gt_origin_cam = T_gt_cam_obj[:3, 3]
        camera_depth_delta.append(float(pred_origin_cam[2] - gt_origin_cam[2]))
        row['constant_transform_residual_translation_m'] = residual_t[-1]
        row['constant_transform_residual_rotation_rad'] = residual_rot[-1]
        row['aligned_object_origin_depth_delta_m'] = camera_depth_delta[-1]
    report = {'status': 'ok', 'method': 'evaluate_v19_hot3d_object_trajectory_alignment', 'claim_scope': 'camera-coordinate object-pose trajectory consistency after fitting one fixed transform between predicted and HOT3D object canonical frames; does not score object shape/contact/nonpenetration', 'inputs': {'render_state': str(args.render_state), 'annotations': str(ann_path), 'hot3d_gt': str(args.hot3d_gt), 'object_bop_id': str(args.object_bop_id), 'stream_id': args.stream_id, 'min_visibility': args.min_visibility, 'direct_only': args.direct_only}, 'row_count': len(rows), 'frame_min': min((r['frame_idx'] for r in rows)), 'frame_max': max((r['frame_idx'] for r in rows)), 'hot3d_object_name': next((gt_rows[int(r['frame_idx'])].get('object_name') for r in rows), None), 'fitted_constant_transform_pred_object_to_hot3d_object': {'rotation_quaternion_wxyz': q_mean.tolist(), 'translation_xyz_m': t_mean.tolist()}, 'relative_transform_variability': {'translation_deviation_from_mean_m': object_eval_summarize(rel_t), 'rotation_deviation_from_mean_rad': object_eval_summarize(rel_rot), 'rotation_deviation_from_mean_deg': object_eval_summarize([math.degrees(v) for v in rel_rot])}, 'constant_transform_residual': {'translation_m': object_eval_summarize(residual_t), 'rotation_rad': object_eval_summarize(residual_rot), 'rotation_deg': object_eval_summarize([math.degrees(v) for v in residual_rot]), 'aligned_object_origin_depth_delta_m': object_eval_summarize(camera_depth_delta)}, 'interpretation_notes': ['A stable relative transform means the predicted and GT object trajectories are temporally consistent up to object-frame convention.', 'This cannot certify object shape, contact, or signed nonpenetration because the HOT3D CAD model is not used here.', 'Large residuals after one fixed transform indicate object pose/camera trajectory inconsistency, not ordinary per-frame contact measurement noise.'], 'rows': rows}
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'row_count', 'frame_min', 'frame_max', 'hot3d_object_name', 'relative_transform_variability', 'constant_transform_residual']}, indent=2))
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np

def box_summary_summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {'count': 0}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'mean': float(np.mean(arr)), 'p10': float(np.percentile(arr, 10.0)), 'p90': float(np.percentile(arr, 90.0)), 'min': float(np.min(arr)), 'max': float(np.max(arr))}

def aggregate_boxes(args: argparse.Namespace) -> dict[str, Any]:
    clip_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for report_path in args.reports:
        report = read_json_mapping(report_path)
        if report.get('method') != 'evaluate_v19_hot3d_hawor_boxes':
            raise RuntimeError(f"unexpected method in {report_path}: {report.get('method')}")
        clip_id = report_path.parent.parent.name if report_path.parent.name == 'evaluation' else report_path.stem
        report_rows = report.get('rows')
        if not isinstance(report_rows, list):
            raise RuntimeError(f'{report_path} has no rows list; rerun evaluator with full report output')
        for row in report_rows:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched['clip_id'] = clip_id
                rows.append(enriched)
        clip_summaries.append({'clip_id': clip_id, 'report': str(report_path), 'frame_count_gt': report.get('frame_count_gt'), 'summary': report.get('summary')})
    measurable = [r for r in rows if r.get('measurable')]
    matched = [r for r in measurable if r.get('matched')]
    by_side: dict[str, Any] = {}
    for side in ('left', 'right'):
        sr = [r for r in measurable if r.get('side') == side]
        sm = [r for r in matched if r.get('side') == side]
        by_side[side] = {'measurable_rows': len(sr), 'matched_rows': len(sm), 'match_rate': float(len(sm) / max(1, len(sr))), 'iou': box_summary_summarize([float(r['iou']) for r in sm if r.get('iou') is not None]), 'center_distance_px': box_summary_summarize([float(r['center_distance_px']) for r in sm if r.get('center_distance_px') is not None]), 'pred_score': box_summary_summarize([float(r['pred_score']) for r in sm if r.get('pred_score') is not None])}
    payload = {'status': 'ok', 'method': 'aggregate_v19_hot3d_box_evals', 'claim_scope': 'aggregate 2D HOT3D hand-box localization only; not 3D MANO/contact/object-pose scoring', 'reports': [str(p) for p in args.reports], 'clip_summaries': clip_summaries, 'summary': {'clip_count': len(args.reports), 'measurable_rows': len(measurable), 'matched_rows': len(matched), 'match_rate': float(len(matched) / max(1, len(measurable))), 'iou': box_summary_summarize([float(r['iou']) for r in matched if r.get('iou') is not None]), 'center_distance_px': box_summary_summarize([float(r['center_distance_px']) for r in matched if r.get('center_distance_px') is not None]), 'by_side': by_side}}
    write_json_payload(args.output_report, payload)
    print(json.dumps(payload['summary'], indent=2))
    return payload

def box_summary_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    return parser.parse_args(argv)
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
METRIC_KEYS = ['wrist_error_m', 'joint_mpjpe_m', 'joint_median_error_m', 'root_aligned_mpjpe_m', 'root_aligned_median_error_m', 'vertex_centroid_error_m']

def mano_summary_summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {'count': 0}
    return {'count': int(arr.size), 'median': float(np.median(arr)), 'mean': float(np.mean(arr)), 'p10': float(np.percentile(arr, 10)), 'p90': float(np.percentile(arr, 90)), 'p95': float(np.percentile(arr, 95)), 'min': float(np.min(arr)), 'max': float(np.max(arr))}

def aggregate_mano(args: argparse.Namespace) -> dict[str, Any]:
    clip_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for report_path in args.reports:
        report = read_json_mapping(report_path)
        if report.get('status') != 'ok':
            raise RuntimeError(f"report {report_path} is not ok: {report.get('status')}")
        clip_id = next((part for part in reversed(report_path.parts) if part.startswith('clip-')), report_path.parent.name)
        summary = report.get('summary', {})
        clip_reports.append({'clip_id': clip_id, 'report': str(report_path), 'matched_rows': summary.get('matched_rows'), 'measurable_rows': summary.get('measurable_rows'), 'match_rate': summary.get('match_rate'), 'wrist_error_m_median': summary.get('wrist_error_m', {}).get('median') if isinstance(summary.get('wrist_error_m'), dict) else None, 'joint_mpjpe_m_median': summary.get('joint_mpjpe_m', {}).get('median') if isinstance(summary.get('joint_mpjpe_m'), dict) else None, 'root_aligned_mpjpe_m_median': summary.get('root_aligned_mpjpe_m', {}).get('median') if isinstance(summary.get('root_aligned_mpjpe_m'), dict) else None})
        for row in report.get('rows', []):
            if isinstance(row, dict):
                row = dict(row)
                row['clip_id'] = clip_id
                rows.append(row)
    measurable = [r for r in rows if r.get('measurable')]
    matched = [r for r in rows if r.get('matched')]
    by_side: dict[str, Any] = {}
    for side in ('left', 'right'):
        sm = [r for r in matched if r.get('side') == side]
        sr = [r for r in measurable if r.get('side') == side]
        by_side[side] = {'measurable_rows': len(sr), 'matched_rows': len(sm), 'match_rate': float(len(sm) / max(1, len(sr))), **{key: mano_summary_summarize([float(r[key]) for r in sm if key in r and r[key] is not None]) for key in METRIC_KEYS}}
    detected = [r for r in matched if r.get('pred_detected_same_frame') is True]
    infilled = [r for r in matched if r.get('pred_detected_same_frame') is False]
    payload = {'status': 'ok', 'method': 'aggregate_v19_hot3d_mano3d_evals', 'claim_scope': 'Fixed-slice aggregate of 3D MANO hand localization/articulation against HOT3D MANO in camera coordinates; root_aligned metrics are wrist-subtracted translation-aligned only; not contact, occlusion, nonpenetration, or object-pose scoring', 'reports': [str(p) for p in args.reports], 'clip_count': len(args.reports), 'clip_summaries': clip_reports, 'summary': {'measurable_rows': len(measurable), 'matched_rows': len(matched), 'match_rate': float(len(matched) / max(1, len(measurable))), 'same_frame_detector_supported_rows': len(detected), 'infilled_or_not_same_frame_rows': len(infilled), 'same_frame_detector_supported_rate_among_matched': float(len(detected) / max(1, len(matched))), 'root_aligned_metric_definition': "Subtract each hand's wrist joint translation before computing joint errors; no rotation or scale alignment is applied.", **{key: mano_summary_summarize([float(r[key]) for r in matched if key in r and r[key] is not None]) for key in METRIC_KEYS}, 'by_detection_support': {'same_frame_detector_supported': {'rows': len(detected), **{key: mano_summary_summarize([float(r[key]) for r in detected if key in r and r[key] is not None]) for key in METRIC_KEYS}}, 'infilled_or_not_same_frame': {'rows': len(infilled), **{key: mano_summary_summarize([float(r[key]) for r in infilled if key in r and r[key] is not None]) for key in METRIC_KEYS}}}, 'by_side': by_side}}
    write_json_payload(args.output_report, payload)
    print(json.dumps(payload, indent=2)[:20000])
    return payload

def mano_summary_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    return parser.parse_args(argv)

def main(argv=None):
    """Evaluate frozen predictions; this entry is never imported by prediction main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('boxes', 'mano', 'object', 'aggregate-boxes', 'aggregate-mano'))
    (selected, remaining) = parser.parse_known_args(argv)
    if selected.task == 'boxes':
        return evaluate_boxes(boxes_parse_args(remaining))
    if selected.task == 'mano':
        return evaluate_mano(mano_eval_parse_args(remaining))
    if selected.task == 'object':
        return evaluate_object_trajectory(remaining)
    if selected.task == 'aggregate-boxes':
        return aggregate_boxes(box_summary_parse_args(remaining))
    return aggregate_mano(mano_summary_parse_args(remaining))
if __name__ == '__main__':
    main()
