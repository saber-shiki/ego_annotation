#!/usr/bin/env python3
"""Build and freeze the GEOMETRY-BLIND visual contact-prior packet for clip001851.

Produces a hash-frozen prior packet (packet_version 1.0.0) with:
  - prior_id, prior_hash (sha256 over canonical JSON excluding prior_hash)
  - source.input_modality = "rgb_frames_plus_prediction_mask_overlay" + attestation
  - geometry_blind section enumerating forbidden inputs
  - validator_report that recomputes prior_hash and walks forbidden fields

Mechanism invariant: the per-frame / per-hand contact prior is derived ONLY from
RGB / video visual-semantic evidence -- visible hand pose, finger splay, key
occlusion, edge grip, and reach/hover gaps in the raw frames plus the SAM2
keyboard-mask overlay. It does NOT use any geometry, distance, depth, object
pose, MANO parameters, GT data, or solver correction.

Usage:
    python scripts/build_clip001851_visual_contact_prior_packet.py \
        --run-root /data2/ego_annotation_outputs/research_clip001851_visual_contact_prior_20260706 \
        [--overlay-video /tmp/v19_clip001851_p07_rerun_review/sam2_multiobject_overlay.mp4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

CLIP_ID = "clip001851"
OBJECT = "keyboard"
PACKET_VERSION = "1.0.0"
PRIOR_ID = f"{CLIP_ID}_visual_contact_prior_v{PACKET_VERSION}"

# Control frames + visually plausible interaction interval 88-118 (dense).
CONTROL_FRAMES = [0, 30, 60, 75, 125, 149]
INTERVAL_FRAMES = list(range(88, 119))
TARGET_FRAMES = sorted(set(CONTROL_FRAMES + INTERVAL_FRAMES))

# ---------------------------------------------------------------------------
# Validator: geometry-blind enforcement
# ---------------------------------------------------------------------------

# Patterns that identify meta-negation boolean keys (allowed when value is False).
# These keys contain substrings that would otherwise trigger forbidden-key alarms,
# but their semantics is to DECLARE the absence of the forbidden input.
META_NEGATION_SUFFIXES = ("_used", "_applied", "_emitted")

# Keys that are NEVER allowed in the packet regardless of value.
# Substring match against lowercased key.
FORBIDDEN_KEY_PATTERNS = [
    "confirmed_contact",   # but confirmed_contact_emitted is meta-negation OK
    "distance",
    "distance_mm",
    "proximity",
    "residual",
    "min_distance",
    "mano_object_distance",
    "surfel_distance",
    "surfel_proximity",
    "surfel_residual",
    "object_pose",
    "object_se3",
    "object_transform",
    "pose_matrix",
    "translation",
    "depth",
    "point_cloud",
    "gt_contact",
    "gt_pose",
    "gt_hand",
    "gt_skeleton",
    "solver_correction",
    "mano_betas",
    "mano_pose",
    "mano_global_orient",
    "mano_transl",
]


def is_meta_negation_key(key: str) -> bool:
    """True if key is a meta-negation boolean (e.g. geometry_distance_used)."""
    lk = key.lower()
    return any(lk.endswith(s) for s in META_NEGATION_SUFFIXES)


def is_forbidden_key(key: str) -> tuple[bool, str | None]:
    """Returns (is_forbidden, matched_pattern) for a key that is NOT meta-negation."""
    if is_meta_negation_key(key):
        return False, None
    lk = key.lower()
    for pat in FORBIDDEN_KEY_PATTERNS:
        if pat in lk:
            return True, pat
    return False, None


def canonical_json_bytes(obj: dict) -> bytes:
    """Serialize to canonical JSON (sorted keys, compact) for hashing."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def compute_prior_hash(packet: dict) -> str:
    """sha256 of canonical JSON with prior_hash and validator_report excluded."""
    canonical = {k: v for k, v in packet.items() if k not in ("prior_hash", "validator_report")}
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def validate_and_report(packet: dict, recompute_hash: bool = True) -> dict:
    """Run the full geometry-blind validator and produce a report.

    Returns a validator_report dict suitable for embedding in the packet.
    """
    report: dict = {
        "validator_version": "1.0.0",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    # --- Hash check ---
    if recompute_hash and "prior_hash" in packet:
        expected = packet["prior_hash"]
        computed = compute_prior_hash(packet)
        report["prior_hash"] = computed
        report["hash_recomputed"] = True
        report["hash_match"] = (computed == expected)
    else:
        report["hash_recomputed"] = False
        report["hash_match"] = None

    # --- Geometry-blind tree walk ---
    forbidden_found: list[dict] = []
    meta_violations: list[dict] = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                cur = f"{path}.{k}" if path else k
                if is_meta_negation_key(k):
                    # Meta-negation: allowed only if value is False
                    if v is not False:
                        meta_violations.append({
                            "path": cur,
                            "key": k,
                            "value": v,
                            "reason": f"meta-negation boolean must be False, got {type(v).__name__}={v}",
                        })
                else:
                    is_forb, matched = is_forbidden_key(k)
                    if is_forb:
                        forbidden_found.append({
                            "path": cur,
                            "key": k,
                            "matched_pattern": matched,
                        })
                walk(v, cur)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(packet)

    report["checks"]["forbidden_fields"] = {
        "passed": len(forbidden_found) == 0,
        "count": len(forbidden_found),
        "items": forbidden_found,
    }
    report["checks"]["meta_negation"] = {
        "passed": len(meta_violations) == 0,
        "count": len(meta_violations),
        "items": meta_violations,
    }

    # --- Structural checks ---
    structural_issues: list[str] = []
    if packet.get("packet_version") != PACKET_VERSION:
        structural_issues.append(f"packet_version must be '{PACKET_VERSION}', got '{packet.get('packet_version')}'")
    if packet.get("prior_id") != PRIOR_ID:
        structural_issues.append(f"prior_id mismatch: expected '{PRIOR_ID}', got '{packet.get('prior_id')}'")
    if "prior_hash" not in packet:
        structural_issues.append("missing prior_hash")
    if "source" not in packet:
        structural_issues.append("missing source section")
    elif packet["source"].get("input_modality") != "rgb_frames_plus_prediction_mask_overlay":
        structural_issues.append(f"source.input_modality expected 'rgb_frames_plus_prediction_mask_overlay'")
    if "geometry_blind" not in packet:
        structural_issues.append("missing geometry_blind section")

    report["checks"]["structural"] = {
        "passed": len(structural_issues) == 0,
        "issues": structural_issues,
    }

    # --- Overall ---
    all_passed = (
        (report["hash_match"] is True or report["hash_recomputed"] is False)
        and report["checks"]["forbidden_fields"]["passed"]
        and report["checks"]["meta_negation"]["passed"]
        and report["checks"]["structural"]["passed"]
    )
    report["overall"] = "valid" if all_passed else "invalid"

    return report


# ---------------------------------------------------------------------------
# Cue templates -- short, concrete visual observations (no geometry terms).
# ---------------------------------------------------------------------------
CUES = {
    "R_keybed": [
        "right/upper hand fingers splayed flat on the key-bed",
        "fingertips rest on / occlude the salmon-masked keys",
        "relaxed palm-down typing/pressing posture directly over the keyboard mask",
    ],
    "R_topgrip": [
        "right/upper hand fingers curl over the keyboard's top edge",
        "fingertips contact the top key-row / frame edge (keys occluded)",
        "hand posture consistent with holding the top edge",
    ],
    "R_topgrip_borderline": [
        "right/upper hand fingertips at the keyboard top-left corner",
        "possible light pinch of the corner; small visible gap makes touch ambiguous",
    ],
    "R_hover_contactlean": [
        "right/upper hand fingers extended just above the top edge",
        "very small gap to the surface; approaching/settling posture",
        "no clear key occlusion to confirm touch",
    ],
    "R_hover_sep": [
        "right/upper hand fingers extended above the keyboard in open air",
        "clear gap between fingertips and the keyboard surface (hover/reach)",
        "no key occlusion; not a pressing/grip posture",
    ],
    "R_absent": [
        "right/upper hand withdrawn from the keyboard",
        "fingers directed at a separate small object on the desk (pen/phone)",
        "hand is well separated from the salmon keyboard mask",
    ],
    "L_bottomgrip": [
        "left/lower hand cups/curls around the keyboard's bottom edge/corner",
        "thumb over the edge, fingers under -- support grip",
        "posture consistent with holding/supporting the keyboard",
    ],
    "L_cornergrip": [
        "left/lower hand fingers wrap the keyboard's bottom-right corner",
        "clear enclosing grip on the keyboard frame",
    ],
    "L_open_contactlean": [
        "left/lower hand open palm just below the keyboard's bottom edge",
        "keyboard edge near the fingers; light support plausible",
        "small gap keeps firm contact unconfirmed",
    ],
    "L_open_sep": [
        "left/lower hand open/spread below the bottom edge with a visible gap",
        "posture consistent with catching/framing rather than gripping",
    ],
    "L_occluded": [
        "left/lower hand at the bottom edge is partially out of frame / in shadow",
        "finger-surface contact cannot be visually resolved",
    ],
}

# ---------------------------------------------------------------------------
# Agent visual reading (direct image reasoning over extracted frames).
# frame -> {"right": (state, confidence, lean, cue_key),
#           "left":  (state, confidence, lean, cue_key)}
# state in {asserted, absent, unresolved}; lean in {contact, separated, none}.
# confidence: ordinal subjective visual-prior score in [0,1], geometry-free.
# ---------------------------------------------------------------------------
READING = {
    0:   {"right": ("asserted", 0.80, "none", "R_keybed"),   "left": ("asserted", 0.68, "none", "L_bottomgrip")},
    30:  {"right": ("asserted", 0.80, "none", "R_keybed"),   "left": ("asserted", 0.65, "none", "L_bottomgrip")},
    60:  {"right": ("asserted", 0.75, "none", "R_keybed"),   "left": ("asserted", 0.60, "none", "L_bottomgrip")},
    75:  {"right": ("asserted", 0.85, "none", "R_keybed"),   "left": ("asserted", 0.72, "none", "L_bottomgrip")},
    88:  {"right": ("asserted", 0.80, "none", "R_keybed"),   "left": ("asserted", 0.68, "none", "L_bottomgrip")},
    89:  {"right": ("asserted", 0.80, "none", "R_keybed"),   "left": ("asserted", 0.68, "none", "L_bottomgrip")},
    90:  {"right": ("asserted", 0.75, "none", "R_keybed"),   "left": ("asserted", 0.60, "none", "L_bottomgrip")},
    91:  {"right": ("asserted", 0.75, "none", "R_keybed"),   "left": ("asserted", 0.62, "none", "L_bottomgrip")},
    92:  {"right": ("asserted", 0.70, "none", "R_topgrip"),  "left": ("asserted", 0.65, "none", "L_bottomgrip")},
    93:  {"right": ("asserted", 0.70, "none", "R_topgrip"),  "left": ("asserted", 0.65, "none", "L_bottomgrip")},
    94:  {"right": ("asserted", 0.68, "none", "R_topgrip"),  "left": ("unresolved", 0.50, "contact", "L_occluded")},
    95:  {"right": ("asserted", 0.65, "none", "R_topgrip"),  "left": ("unresolved", 0.50, "contact", "L_occluded")},
    96:  {"right": ("asserted", 0.65, "none", "R_topgrip"),  "left": ("asserted", 0.60, "none", "L_bottomgrip")},
    97:  {"right": ("asserted", 0.65, "none", "R_topgrip"),  "left": ("asserted", 0.60, "none", "L_bottomgrip")},
    98:  {"right": ("asserted", 0.62, "none", "R_topgrip"),  "left": ("asserted", 0.58, "none", "L_bottomgrip")},
    99:  {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("asserted", 0.58, "none", "L_bottomgrip")},
    100: {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("asserted", 0.60, "none", "L_bottomgrip")},
    101: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("asserted", 0.58, "none", "L_bottomgrip")},
    102: {"right": ("unresolved", 0.40, "separated", "R_hover_sep"),       "left": ("asserted", 0.58, "none", "L_bottomgrip")},
    103: {"right": ("unresolved", 0.40, "separated", "R_hover_sep"),       "left": ("asserted", 0.58, "none", "L_bottomgrip")},
    104: {"right": ("unresolved", 0.40, "separated", "R_hover_sep"),       "left": ("unresolved", 0.50, "contact", "L_occluded")},
    105: {"right": ("unresolved", 0.40, "separated", "R_hover_sep"),       "left": ("unresolved", 0.50, "contact", "L_occluded")},
    106: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("asserted", 0.55, "none", "L_bottomgrip")},
    107: {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("asserted", 0.55, "none", "L_bottomgrip")},
    108: {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("asserted", 0.55, "none", "L_bottomgrip")},
    109: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("asserted", 0.52, "none", "L_bottomgrip")},
    110: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("unresolved", 0.50, "contact", "L_open_contactlean")},
    111: {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("unresolved", 0.50, "contact", "L_open_contactlean")},
    112: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("unresolved", 0.50, "contact", "L_open_contactlean")},
    113: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("unresolved", 0.48, "none", "L_open_sep")},
    114: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("unresolved", 0.48, "none", "L_open_sep")},
    115: {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("unresolved", 0.48, "none", "L_open_sep")},
    116: {"right": ("unresolved", 0.45, "separated", "R_hover_sep"),       "left": ("unresolved", 0.48, "none", "L_open_sep")},
    117: {"right": ("unresolved", 0.50, "contact", "R_hover_contactlean"), "left": ("unresolved", 0.50, "contact", "L_open_contactlean")},
    118: {"right": ("asserted", 0.68, "none", "R_keybed"),                 "left": ("unresolved", 0.50, "contact", "L_open_contactlean")},
    125: {"right": ("unresolved", 0.50, "contact", "R_topgrip_borderline"),"left": ("unresolved", 0.40, "separated", "L_open_sep")},
    149: {"right": ("absent", 0.75, "none", "R_absent"),                   "left": ("asserted", 0.80, "none", "L_cornergrip")},
}

HAND_IDENTITY = {
    "right": {
        "visual_descriptor": (
            "upper hand: enters from image-top; interacts with the keyboard key-bed "
            "and top edge (typing/pressing and top-edge grip/reach)."
        ),
        "anatomical_side_inference": "right",
        "side_inference_confidence": 0.65,
        "side_inference_basis": (
            "egocentric geometry only: frame is rotated ~90deg CW (ceiling rig at "
            "image-left); seated subject body sits at image-right facing image-left, "
            "so the upper hand maps to the right side when uprighted. NOT from GT."
        ),
    },
    "left": {
        "visual_descriptor": (
            "lower hand: enters from image-bottom-right; interacts with the keyboard "
            "bottom edge/corner (support grip and open-hand hold)."
        ),
        "anatomical_side_inference": "left",
        "side_inference_confidence": 0.65,
        "side_inference_basis": (
            "same egocentric-geometry inference as the right hand; the lower hand "
            "maps to the left side when uprighted. NOT from GT."
        ),
    },
}

STATE_DEFINITIONS = {
    "asserted": "visible cues strongly indicate hand-keyboard contact (finger splay on key-bed with key occlusion, or fingers wrapped over an edge/corner).",
    "absent": "hand is clearly separated from the keyboard (withdrawn, reaching a different object, or out of frame).",
    "unresolved": "hand is near the keyboard but contact cannot be visually distinguished from a small hover/approach gap, or the hand is occluded. `lean` records the weak cue direction.",
}

STATE_COLOR = {
    "asserted": (0, 175, 70),
    "unresolved": (235, 165, 0),
    "absent": (150, 150, 150),
}


def frame_path(run_root: str, idx: int) -> str:
    return os.path.join(run_root, "frames", f"sam2_overlay_{idx:03d}.png")


def load_frame(run_root: str, idx: int, overlay_video: str | None) -> Image.Image:
    p = frame_path(run_root, idx)
    if os.path.exists(p):
        return Image.open(p).convert("RGB")
    if overlay_video and os.path.exists(overlay_video):
        import cv2  # RGB read only; no geometry
        cap = cv2.VideoCapture(overlay_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not read frame {idx} from {overlay_video}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.save(p)
        return img
    raise FileNotFoundError(f"no frame source for idx {idx}: {p} missing and no overlay video")


def build_entry(state: str, conf: float, lean: str, cue_key: str) -> dict:
    return {
        "visual_prior_state": state,
        "confidence": round(float(conf), 2),
        "lean": lean,
        "visual_cues": list(CUES[cue_key]),
        "provenance": "visual_semantic_prior",
        "geometry_distance_used": False,
    }


def build_packet() -> dict:
    frames = []
    for idx in TARGET_FRAMES:
        r = READING[idx]["right"]
        l = READING[idx]["left"]
        frames.append({
            "frame_idx": idx,
            "role": "control" if idx in CONTROL_FRAMES else "interaction_interval",
            "hands": {
                "right": build_entry(*r),
                "left": build_entry(*l),
            },
        })

    packet = {
        "clip_id": CLIP_ID,
        "object": OBJECT,
        "packet_kind": "geometry_blind_visual_contact_prior",
        "packet_version": PACKET_VERSION,
        "prior_id": PRIOR_ID,
        "mechanism": "visual_semantic_contact_prior",
        "provenance": "visual_semantic_prior",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # --- Source attestation ---
        "source": {
            "input_modality": "rgb_frames_plus_prediction_mask_overlay",
            "attestation": (
                "Prediction-side visual segmentation only: SAM2 keyboard-mask overlay "
                "on raw RGB frames. No geometry, depth, pose, or GT data is present in "
                "the source frames. The overlay marks only the keyboard region; hand "
                "contact is read from bare visible hand pixels. This is a geometry-blind "
                "visual prior -- it does not incorporate any 3D reconstruction, MANO "
                "parameters, object pose, depth, surfel geometry, or GT labels."
            ),
            "visual_sources": [
                {
                    "kind": "sam2_keyboard_mask_overlay",
                    "path": "/tmp/v19_clip001851_p07_rerun_review/sam2_multiobject_overlay.mp4",
                    "note": "raw RGB + salmon keyboard-mask tint; hands fully visible; NO GT skeleton; NO distance/error HUD.",
                },
                {
                    "kind": "extracted_frames",
                    "path": "{run_root}/frames/sam2_overlay_{idx:03d}.png",
                },
            ],
            "excluded_sources": [
                {
                    "kind": "hot3d_benchmark_demo_video",
                    "reason": "bakes in GT white hand skeleton and mm-error HUD (GT + distance) -- forbidden for this geometry-blind prior.",
                },
            ],
        },
        # --- Geometry-blind declaration ---
        "geometry_blind": {
            "description": (
                "This prior is derived exclusively from visual-semantic evidence in RGB "
                "frames. No geometric measurements were used in its construction."
            ),
            "forbidden_inputs": [
                "MANO hand mesh parameters (betas, pose, global_orient, transl)",
                "MANO-object distance or proximity (mm or normalized)",
                "Object pose / SE(3) trajectory",
                "Surfel point cloud, mesh, proximity, or residual",
                "Depth maps or point clouds",
                "HOT3D GT (object pose, hand pose, masks, contact labels, skeleton)",
                "Solver or factor-graph corrections",
                "Confirmed contact labels or binary contact ground truth",
            ],
            "meta_negation_booleans": {
                "geometry_distance_used": False,
                "hot3d_gt_used": False,
                "mano_object_distance_used": False,
                "object_pose_used": False,
                "surfel_proximity_used": False,
                "confirmed_contact_emitted": False,
                "solver_correction_applied": False,
            },
        },
        "confidence_semantics": (
            "ordinal, subjective, geometry-free visual-prior score in [0,1]; NOT a "
            "calibrated probability and NOT derived from any distance/proximity."
        ),
        "hand_identity": HAND_IDENTITY,
        "state_definitions": STATE_DEFINITIONS,
        "target_frames": TARGET_FRAMES,
        "control_frames": CONTROL_FRAMES,
        "interaction_interval": [INTERVAL_FRAMES[0], INTERVAL_FRAMES[-1]],
        "frames": frames,
        "summary": summarize(frames),
        "limitations": [
            "Monocular egocentric view: contact vs a small hover/approach gap is not "
            "always visually separable -> many interval frames are 'unresolved' by design.",
            "The SAM2 overlay marks only the keyboard, not the hand; hand contact is "
            "read from the bare visible hand pixels, not from any mask/geometry.",
            "Left/right anatomical labels are an egocentric-geometry inference "
            "(confidence 0.65); the physical hand is pinned by `hand_identity.*.visual_descriptor`.",
            "This is a PRIOR, not an accepted contact label; it must not be treated as "
            "confirmed_contact and carries no solver correction.",
        ],
    }

    # -- Hash-freeze: compute and embed prior_hash --
    packet["prior_hash"] = compute_prior_hash(packet)

    # -- Validate geometry-blind invariant --
    report = validate_and_report(packet, recompute_hash=True)
    packet["validator_report"] = report

    if report["overall"] != "valid":
        import sys
        print("VALIDATION FAILED:", file=sys.stderr)
        json.dump(report, sys.stderr, indent=2)
        sys.exit(1)

    return packet


def summarize(frames: list[dict]) -> dict:
    out = {}
    for hand in ("right", "left"):
        counts = {"asserted": 0, "unresolved": 0, "absent": 0}
        for f in frames:
            counts[f["hands"][hand]["visual_prior_state"]] += 1
        out[hand] = counts
    return out


def render_contact_sheet(run_root: str, packet: dict, overlay_video: str | None, out_path: str) -> None:
    tile_w = 300
    header_h = 74
    cols = 6
    frames = packet["frames"]
    rows = (len(frames) + cols - 1) // cols
    # probe aspect from first frame
    probe = load_frame(run_root, frames[0]["frame_idx"], overlay_video)
    tile_h = int(tile_w * probe.height / probe.width)
    cell_w, cell_h = tile_w, tile_h + header_h
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        fsm = font
    for i, f in enumerate(frames):
        idx = f["frame_idx"]
        img = load_frame(run_root, idx, overlay_video).resize((tile_w, tile_h))
        cx = (i % cols) * cell_w
        cy = (i // cols) * cell_h
        sheet.paste(img, (cx, cy + header_h))
        draw.rectangle([cx, cy, cx + cell_w - 1, cy + header_h - 1], fill=(24, 24, 24))
        role_tag = "CTRL" if f["role"] == "control" else "INT"
        draw.text((cx + 6, cy + 3), f"f{idx:03d} [{role_tag}]", fill=(235, 235, 235), font=font)
        for j, hand in enumerate(("right", "left")):
            e = f["hands"][hand]
            st = e["visual_prior_state"]
            col = STATE_COLOR[st]
            y = cy + 26 + j * 23
            draw.rectangle([cx + 6, y + 2, cx + 20, y + 16], fill=col)
            lean = "" if e["lean"] == "none" else f"->{e['lean']}"
            label = f"{hand[0].upper()}: {st} {e['confidence']:.2f}{lean}"
            draw.text((cx + 26, y), label, fill=col, font=fsm)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="/data2/ego_annotation_outputs/research_clip001851_visual_contact_prior_20260706")
    ap.add_argument("--overlay-video", default="/tmp/v19_clip001851_p07_rerun_review/sam2_multiobject_overlay.mp4")
    ap.add_argument("--validate-only", action="store_true",
                    help="Re-validate existing packet without rebuilding (hash-check)")
    args = ap.parse_args()

    run_root = args.run_root
    os.makedirs(os.path.join(run_root, "packet"), exist_ok=True)
    os.makedirs(os.path.join(run_root, "review"), exist_ok=True)

    if args.validate_only:
        packet_path = os.path.join(run_root, "packet", "clip001851_visual_contact_prior_packet.json")
        with open(packet_path) as fh:
            existing = json.load(fh)
        report = validate_and_report(existing, recompute_hash=True)
        print(f"VALIDATE-ONLY: {report['overall']}")
        print(json.dumps(report, indent=2))
        return

    packet = build_packet()
    packet_path = os.path.join(run_root, "packet", "clip001851_visual_contact_prior_packet.json")
    with open(packet_path, "w") as fh:
        json.dump(packet, fh, indent=2, ensure_ascii=False)

    sheet_path = os.path.join(run_root, "review", "clip001851_visual_contact_prior_sheet.png")
    render_contact_sheet(run_root, packet, args.overlay_video, sheet_path)

    report_path = os.path.join(run_root, "packet", "clip001851_visual_contact_prior_validator_report.json")
    with open(report_path, "w") as fh:
        json.dump(packet["validator_report"], fh, indent=2, ensure_ascii=False)

    print(f"packet:   {packet_path}")
    print(f"sheet:    {sheet_path}")
    print(f"report:   {report_path}")
    print(f"prior_id: {packet['prior_id']}")
    print(f"prior_hash: {packet['prior_hash']}")
    print(f"source_modality: {packet['source']['input_modality']}")
    print(f"validation: {packet['validator_report']['overall']}")
    print(f"frames:   {len(packet['frames'])} (control {len(CONTROL_FRAMES)} + interval {len(INTERVAL_FRAMES)})")
    print(f"summary:  {json.dumps(packet['summary'])}")


if __name__ == "__main__":
    main()
