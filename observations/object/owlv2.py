"""Build V19 object prompt JSON from OWLv2 text-grounded detections.

This script restores the V18-style grounded front-end for V19 object masks:
text query -> OWLv2 boxes -> SAM2 box prompts. It deliberately does not ask a
VLM/agent for pixel-accurate click coordinates. The output is the same prompt
root consumed by scripts/run_sam2_vlm_points_multiobject.py, with per-frame
box_xyxy entries and optional detector-derived center points for diagnostics.
"""
from __future__ import annotations
from utils.data import read_json
import argparse
import json
import math
import hashlib
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f'{label} must be a JSON object')
    return value

def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise RuntimeError('empty frame list')
    return out

def manifest_rows(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = manifest.get('frames')
    if not isinstance(rows, list):
        raise RuntimeError('raw frame manifest missing frames list')
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = int(row.get('frame_idx', row.get('index')))
        out[idx] = row
    return out

def prompt_texts(raw: list[str]) -> list[str]:
    if not raw or any(not isinstance(item, str) or not item.strip() for item in raw):
        raise ValueError('at least one nonempty text prompt is required')
    return [item.strip().lower() if item.strip().endswith('.') else item.strip().lower() + '.' for item in raw]

def resolve_model_path(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f'OWLv2 model path missing: {path}')
    snapshots = path / 'snapshots'
    if snapshots.is_dir():
        candidates = sorted([p for p in snapshots.iterdir() if p.is_dir()])
        if not candidates:
            raise RuntimeError(f'OWLv2 model cache has no snapshots: {path}')
        return candidates[-1]
    return path

def finite_box(box: list[float], width: int, height: int) -> list[float] | None:
    if len(box) != 4:
        return None
    vals = [float(v) for v in box]
    if not all((math.isfinite(v) for v in vals)):
        return None
    (x1, y1, x2, y2) = vals
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def text_budget_report(processor, config, prompts):
    """Count real CLIP tokens, including BOS/EOS, against encoder positions."""
    budget = int(config.text_config.max_position_embeddings)
    effective = prompt_texts(prompts)
    encoded = processor.tokenizer(effective, add_special_tokens=True, padding=False, truncation=False)
    token_ids = encoded['input_ids']
    rows = [{'original_prompt': text, 'effective_prompt': effective_text, 'token_ids': list(map(int, ids)), 'token_count_including_special': len(ids), 'truncated': False, 'within_budget': len(ids) <= budget} for (text, effective_text, ids) in zip(prompts, effective, token_ids)]
    return {'schema': 'owlv2_text_budget_v1', 'max_position_embeddings': budget, 'special_token_count': processor.tokenizer.num_special_tokens_to_add(pair=False), 'padding': 'max_length', 'truncation': False, 'valid': all((r['within_budget'] for r in rows)), 'prompts': rows}

def encode_detector_inputs(processor, model_config, image, prompts):
    report = text_budget_report(processor, model_config, prompts)
    if not report['valid']:
        raise ValueError('OWLv2 query exceeds real tokenizer/model position budget; shorten with Astra before inference: ' + json.dumps(report))
    inputs = processor(images=image, return_tensors='pt')
    text_inputs = processor.tokenizer([row['effective_prompt'] for row in report['prompts']], return_tensors='pt', padding='max_length', max_length=report['max_position_embeddings'], truncation=False, add_special_tokens=True)
    inputs.update(text_inputs)
    if inputs['input_ids'].shape[-1] != report['max_position_embeddings']:
        raise ValueError('OWLv2 processor violated padded model-position budget')
    return (inputs, report)

def run_owlv2_frame(processor: Any, model: Any, image: Image.Image, prompts: list[str], threshold: float) -> list[dict[str, Any]]:
    import torch
    (inputs, _) = encode_detector_inputs(processor, model.config, image, prompts)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = [(image.height, image.width)]
    detections: list[dict[str, Any]] = []
    if hasattr(processor, 'post_process_grounded_object_detection'):
        try:
            results = processor.post_process_grounded_object_detection(outputs, threshold=float(threshold), target_sizes=target_sizes, text_labels=[prompts])[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(outputs, threshold=float(threshold), target_sizes=target_sizes)[0]
        labels = results.get('text_labels') or results.get('labels') or []
        for (box_tensor, score_tensor, label) in zip(results['boxes'], results['scores'], labels):
            box = finite_box([float(v) for v in box_tensor.detach().cpu().tolist()], image.width, image.height)
            if box is None:
                continue
            detections.append({'bbox_xyxy': box, 'score': float(score_tensor.detach().cpu().item()), 'text_label': str(label)})
    else:
        results = processor.post_process_object_detection(outputs=outputs, threshold=float(threshold), target_sizes=target_sizes)[0]
        for (box_tensor, score_tensor, label_tensor) in zip(results['boxes'], results['scores'], results['labels']):
            box = finite_box([float(v) for v in box_tensor.detach().cpu().tolist()], image.width, image.height)
            if box is None:
                continue
            label_idx = int(label_tensor.detach().cpu().item())
            label = prompts[label_idx] if 0 <= label_idx < len(prompts) else str(label_idx)
            detections.append({'bbox_xyxy': box, 'score': float(score_tensor.detach().cpu().item()), 'text_label': str(label)})
    detections.sort(key=lambda row: float(row['score']), reverse=True)
    return detections

def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(1e-9, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection))


class CandidateIdentity:
    """One track, no category reacquisition after identity loss.

    Costs use IoU, center displacement in previous-box diagonals, log side
    ratios and a small detector-confidence term. Three missed frames may bridge
    occlusion; a fourth, or competing distinct boxes of equal cost, fails closed.
    Duplicate detector boxes do not count as distinct competing instances.
    """
    def __init__(self):
        self.last = None
        self.last_frame = None
        self.failed = False

    def select(self, frame, detections):
        diagnostics = {'frame_idx': frame, 'previous_frame_idx': self.last_frame,
                       'policy': 'iou_center_log_size_v1', 'candidates': []}
        if self.failed:
            return None, {**diagnostics, 'state': 'identity_lost'}
        ranked = []
        for index, detection in enumerate(detections):
            box = np.asarray(detection['bbox_xyxy'], float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError('invalid identity candidate box')
            score = float(detection['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid identity candidate score')
            if self.last is None:
                iou, center, size, allowed = None, None, None, True
                cost = 1. - score
            else:
                previous = np.asarray(self.last['bbox_xyxy'], float)
                gap = frame - self.last_frame
                if gap <= 0:
                    raise ValueError('identity frames must be strictly increasing')
                sides = previous[2:] - previous[:2]
                center = float(np.linalg.norm((box[:2] + box[2:] - previous[:2] - previous[2:]) / 2) / np.linalg.norm(sides))
                size = float(np.max(np.abs(np.log((box[2:] - box[:2]) / sides))))
                iou = box_iou(box, previous)
                allowed = gap <= 4 and center <= .35 + .10 * (gap - 1) and size <= math.log(1.8) and iou >= .10
                cost = (1. - iou) + 2. * center + size + .10 * (1. - score)
            diagnostics['candidates'].append({'candidate_index': index, 'score': score, 'iou': iou,
                'center_distance_box_diagonals': center, 'max_log_side_ratio': size, 'allowed': allowed, 'cost': cost})
            if allowed:
                ranked.append((cost, index))
        ranked.sort()
        if not ranked:
            self.failed = self.last_frame is not None and frame - self.last_frame > 3
            return None, {**diagnostics, 'state': 'identity_lost' if self.failed else 'occluded_or_out_of_frame'}
        cost, index = ranked[0]
        distinct = [(c, i) for c, i in ranked[1:] if box_iou(detections[index]['bbox_xyxy'], detections[i]['bbox_xyxy']) < .7]
        margin = None if not distinct else distinct[0][0] - cost
        if margin is not None and margin < (.05 if self.last is None else .12):
            self.failed = True
            return None, {**diagnostics, 'state': 'identity_ambiguous', 'distinct_cost_margin': margin}
        state = 'initial_category_seed_requires_visual_review' if self.last is None else 'temporally_associated'
        self.last, self.last_frame = detections[index], frame
        return self.last, {**diagnostics, 'state': state, 'selected_candidate_index': index,
            'selected_score': self.last['score'], 'selected_cost': cost, 'distinct_cost_margin': margin}


def check_candidate_mask_identity(report, track, source_size_wh):
    """Compare same-run SAM masks with associated boxes in detector pixels."""
    checks = []
    for row in report['detections']:
        selected = row['selected_detection']
        if selected is None:
            continue  # absence is explicit, not a fabricated box during occlusion
        mask_row = track.get(str(row['frame_idx']), {})
        mask_path = mask_row.get('mask_path')
        iou = 0.; mask_hash = None
        if mask_row.get('visible') is True and mask_path:
            raw_mask = Path(mask_path).read_bytes()
            mask_hash = hashlib.sha256(raw_mask).hexdigest()
            mask = cv2.imdecode(np.frombuffer(raw_mask, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None: raise ValueError('candidate identity mask unreadable')
            ys, xs = np.where(mask > 0)
            if len(xs):
                scale = np.array([row['image_width'] / mask.shape[1], row['image_height'] / mask.shape[0]] * 2)
                box = np.array([xs.min(), ys.min(), xs.max()+1, ys.max()+1]) * scale
                iou = box_iou(box, selected['bbox_xyxy'])
        checks.append({'frame_idx': row['frame_idx'], 'selected_candidate_index': row['association']['selected_candidate_index'],
                       'mask_path': mask_path, 'mask_sha256': mask_hash,
                       'mask_candidate_iou': iou, 'identity_supported': iou >= .10})
    return {'schema': 'candidate_mask_identity_v1', 'case_id': report['case_id'], 'object_id': report['object_id'],
            'track_id': report['track_id'], 'status': 'ok' if checks and all(r['identity_supported'] for r in checks) else 'failed', 'frames': checks}


def box_center(box: list[float]) -> dict[str, float | str]:
    (x1, y1, x2, y2) = box
    return {'x': float((x1 + x2) * 0.5), 'y': float((y1 + y2) * 0.5), 'label': 'detector_box_center_diagnostic'}

def negative_ring(box: list[float], width: int, height: int, pad: float) -> list[dict[str, float | str]]:
    (x1, y1, x2, y2) = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    points = [(max(0.0, x1 - pad), cy, 'left_of_detector_box'), (min(float(width - 1), x2 + pad), cy, 'right_of_detector_box'), (cx, max(0.0, y1 - pad), 'above_detector_box'), (cx, min(float(height - 1), y2 + pad), 'below_detector_box')]
    return [{'x': float(x), 'y': float(y), 'label': label} for (x, y, label) in points]

def render_review(image_bgr: np.ndarray, detections: list[dict[str, Any]], selected: dict[str, Any] | None, frame_idx: int, out_path: Path) -> None:
    canvas = image_bgr.copy()
    for (i, det) in enumerate(detections[:8]):
        (x1, y1, x2, y2) = [int(round(v)) for v in det['bbox_xyxy']]
        color = (0, 220, 255) if selected is det else (80, 80, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{i + 1}:{det['text_label']} {float(det['score']):.3f}"
        cv2.putText(canvas, label[:70], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f'OWLv2 object boxes frame {frame_idx:04d}', (10, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f'failed to write review image {out_path}')

def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import Owlv2ForObjectDetection
    from transformers import Owlv2Processor, Owlv2Config
    manifest = require_dict(read_json(args.raw_frame_manifest), 'raw frame manifest')
    rows_by_idx = manifest_rows(manifest)
    prompt_frames = sorted(set(parse_int_list(args.prompt_frames)))
    active_start = int(args.active_start if args.active_start is not None else min(prompt_frames))
    active_end = int(args.active_end if args.active_end is not None else int(manifest.get('frame_end', max(prompt_frames))))
    frames = list(range(active_start, active_end + 1))
    if not set(prompt_frames) <= set(frames):
        raise ValueError('detector prompt frames outside active timeline')
    association = CandidateIdentity()
    prompts = list(args.text_prompt)
    model_path = resolve_model_path(args.owlv2_model)
    device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested for OWLv2 but torch.cuda.is_available() is false')
    processor = Owlv2Processor.from_pretrained(str(model_path), local_files_only=True)
    budget = text_budget_report(processor, Owlv2Config.from_pretrained(str(model_path), local_files_only=True), prompts)
    if not budget['valid']:
        raise ValueError('OWLv2 queries exceed tokenizer position budget; regenerate short detector noun phrases before model inference: ' + json.dumps(budget))
    prompts = [row['effective_prompt'] for row in budget['prompts']]
    model = Owlv2ForObjectDetection.from_pretrained(str(model_path), local_files_only=True).to(device)
    model.eval()
    prompt_rows: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    missing_frames: list[int] = []
    for frame_idx in frames:
        row = rows_by_idx.get(frame_idx)
        if row is None:
            raise RuntimeError(f'frame {frame_idx} not present in manifest')
        image_path = Path(str(row.get('raw_frame_path') or row.get('rgb')))
        if not image_path.exists():
            raise RuntimeError(f'frame image missing: {image_path}')
        image = Image.open(image_path).convert('RGB')
        detections = run_owlv2_frame(processor, model, image, prompts, float(args.box_threshold))
        selected, association_diagnostic = association.select(frame_idx, detections)
        record = {'frame_idx': int(frame_idx), 'image_path': str(image_path), 'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(), 'image_width': int(image.width), 'image_height': int(image.height), 'text_prompts': prompts, 'detections': detections, 'selected_detection': selected, 'association': association_diagnostic}
        detection_records.append(record)
        if frame_idx not in prompt_frames:
            continue
        image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        render_review(image_bgr, detections, selected, frame_idx, args.review_dir / f'owlv2_boxes_{frame_idx:06d}.jpg')
        if selected is None:
            missing_frames.append(frame_idx)
            prompt_rows.append({'frame_idx': int(frame_idx), 'target_visible': False, 'detection_state': 'no_owlv2_box'})
            continue
        box = [float(v) for v in selected['bbox_xyxy']]
        min_side = min(float(box[2] - box[0]), float(box[3] - box[1]))
        neg_pad = max(float(args.negative_ring_min_pad_px), float(args.negative_ring_pad_fraction) * min_side)
        prompt_rows.append({'frame_idx': int(frame_idx), 'target_visible': True, 'detection_state': 'owlv2_selected_box', 'box_xyxy': box, 'positive_points': [] if args.box_only else [box_center(box)], 'negative_points': negative_ring(box, image.width, image.height, neg_pad), 'candidate_index': association_diagnostic['selected_candidate_index'], 'association': association_diagnostic, 'owlv2_score': float(selected['score']), 'owlv2_text_label': str(selected['text_label']), 'owlv2_text_prompts': prompts})
    identity_report = {'schema': 'owlv2_temporal_identity_v1', 'case_id': args.case_id, 'object_id': args.object_id, 'track_id': args.track_id, 'raw_frame_manifest': str(args.raw_frame_manifest), 'raw_frame_manifest_sha256': hashlib.sha256(args.raw_frame_manifest.read_bytes()).hexdigest(), 'detections': detection_records}
    write_json(args.output_root / args.object_id / 'candidate_identity.json', identity_report)
    if association.failed:
        raise RuntimeError('mask-track identity association failed; candidate_identity.json retains ambiguity/loss evidence')
    if len(prompt_rows) == 0 or all((not row.get('target_visible') for row in prompt_rows)):
        raise RuntimeError('OWLv2 produced no usable prompt boxes')
    if len(missing_frames) > int(args.max_missing_prompt_frames):
        raise RuntimeError(f'OWLv2 missing too many prompt frames: {missing_frames}')
    first_visible = next((row for row in prompt_rows if row.get('target_visible')))
    width = int(detection_records[0]['image_width'])
    height = int(detection_records[0]['image_height'])
    active_start = int(args.active_start if args.active_start is not None else min(frames))
    active_end = int(args.active_end if args.active_end is not None else int(manifest.get('frame_end', max(frames))))
    payload = {'case_id': args.case_id, 'object_id': args.object_id, 'track_id': args.track_id, 'description': args.description, 'prompt_source': 'owlv2_text_grounded_detector_boxes', 'point_coordinate_frame': f'manifest_pixels_{width}x{height}', 'prompt_image_width': width, 'prompt_image_height': height, 'object_surface_policy': 'OWLv2 text-grounded box prompts seed SAM2; masks containing broad hand/table/arm support remain P07 failures.', 'object_plan_payload': {'object_id': args.object_id, 'track_id': args.track_id, 'active_intervals': [{'start_frame': active_start, 'end_frame': active_end}], 'expected_visible_intervals': [{'start_frame': active_start, 'end_frame': active_end, 'visibility': 'visible_or_hand_occluded'}]}, 'owlv2_model': str(model_path), 'owlv2_box_threshold': float(args.box_threshold), 'point_prompts': prompt_rows}
    out_path = args.output_root / args.object_id / 'object_point_prompts_vlm.json'
    write_json(out_path, payload)
    report = {'status': 'ok', 'method': 'build_v19_owlv2_object_box_prompts', 'claim_scope': 'text-grounded OWLv2 boxes for SAM2 prompting only; not masks, geometry, or pose', 'raw_frame_manifest': str(args.raw_frame_manifest), 'output_prompt_json': str(out_path), 'review_dir': str(args.review_dir), 'case_id': args.case_id, 'object_id': args.object_id, 'track_id': args.track_id, 'text_prompts': prompts, 'text_tokenization': budget, 'prompt_frames': prompt_frames, 'association_frames': frames, 'missing_prompt_frames': missing_frames, 'selected_first_visible_prompt': first_visible, 'detections': detection_records}
    write_json(args.output_root / args.object_id / 'v19_owlv2_object_box_prompt_report.json', report)
    print(json.dumps({k: v for (k, v) in report.items() if k != 'detections'}, indent=2))
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-frame-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--track-id', required=True)
    parser.add_argument('--description', default='text-grounded rigid object')
    parser.add_argument('--text-prompt', action='append', required=True, help='Open-vocabulary detector query, e.g. keyboard. May repeat.')
    parser.add_argument('--prompt-frames', required=True, help='Comma-separated source frame indices for detector keyframes.')
    parser.add_argument('--active-start', type=int)
    parser.add_argument('--active-end', type=int)
    parser.add_argument('--owlv2-model', type=Path, required=True)
    parser.add_argument('--box-threshold', type=float, default=0.03)
    parser.add_argument('--max-missing-prompt-frames', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--box-only', action='store_true', help='Write box prompts without detector-center positive points.')
    parser.add_argument('--negative-ring-pad-fraction', type=float, default=0.25)
    parser.add_argument('--negative-ring-min-pad-px', type=float, default=24.0)
    return parser.parse_args()

def main() -> None:
    import sys
    if sys.argv[1:2] == ['text-contract']:
        from transformers import Owlv2Config, Owlv2Processor
        parser = argparse.ArgumentParser(description='CPU tokenizer/model budget check; no learned inference')
        parser.add_argument('--model', type=Path, required=True)
        parser.add_argument('--prompts', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        processor = Owlv2Processor.from_pretrained(str(args.model), local_files_only=True)
        config = Owlv2Config.from_pretrained(str(args.model), local_files_only=True)
        value = text_budget_report(processor, config, read_json(args.prompts)['prompts'])
        with args.output.open('x') as handle:
            json.dump(value, handle, indent=2)
        return
    run(parse_args())
if __name__ == '__main__':
    main()
