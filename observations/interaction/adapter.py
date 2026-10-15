"""Deterministic join of independent image judgments into the P17 input contract.

Only declared frame IDs are expanded. Image uncertainty remains text; the metric
support band is a configuration prior, not a distance measured by the judge.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import math

SCHEMA = "ego.annotation.interaction_evidence.v1"
CONTACT_SUPPORT_UNCERTAINTY_PRIOR_M = 0.06
SIDES = ("left", "right")
DIGITS = ("thumb", "index", "middle", "ring", "little")
OCCLUSION_MAP = {
    "hand_in_front": "hand_in_front_of_object",
    "object_in_front": "object_in_front_of_hand",
    "separate": "no_visible_occlusion",
    "unresolved": "unresolved",
}
JUDGMENTS = ("contact", "contact_pads", "occlusion", "depth_reliability", "visibility")
DIAGNOSTIC_ROLE = "diagnostic_only_not_a_visual_fit_factor"


def timeline_ids(values):
    ids = list(values)
    if not ids or any(type(value) is not int or value < 0 for value in ids) or ids != sorted(set(ids)):
        raise ValueError("interaction timeline requires sorted unique nonnegative integer frame IDs")
    return ids


def matches_object_identity(value, object_id):
    # Only the documented object: prefix is an alias, never arbitrary entity types.
    return value in {object_id, object_id.removeprefix("object:"), "object:" + object_id.removeprefix("object:")}


def _validated_judgment(name, result, ids, object_id):
    import jsonschema
    if not isinstance(result, dict) or result.get("object_id") != object_id:
        raise ValueError(f"{name}: foreign object identity")
    raw = {key: value for key, value in result.items() if key != "judgment_provenance"}
    module = importlib.import_module("observations.judgments.judge_" + name)
    jsonschema.validate(raw, module.SCHEMA)
    rows = raw.get("frames", raw.get("interaction_judgments", raw.get("intervals", [])))
    occupied = set()
    for position, row in enumerate(rows):
        if name == "contact" and not matches_object_identity(row["target_entity_id"], object_id):
            raise ValueError("contact: foreign target identity")
        start = row.get("frame_start", row.get("frame_idx"))
        end = row.get("frame_end", start)
        if start not in ids or end not in ids or start > end:
            raise ValueError(f"{name}: interval outside declared timeline or reversed")
        entity = row.get("hand_side", row.get("entity", "all"))
        for frame in ids:
            if start <= frame <= end:
                key = (frame, entity)
                if key in occupied:
                    raise ValueError(f"{name}: duplicate/overlapping evidence for {key}")
                occupied.add(key)
                yield key, row, position


def _depth_relation(hand, obj):
    """Lossy legacy vocabulary; retain both independent raw rows alongside it."""
    if hand is None and obj is None:
        return "not_evaluated"
    if hand is not None and hand["reliability"] in {"mixed_depth", "unsupported"}:
        return "hand_depth_unreliable"
    if hand is not None and hand["reliability"] == "supported" and obj is not None and obj["reliability"] == "supported":
        return "object_depth_reliable"
    return "mixed_or_unresolved"


def assemble_interaction_evidence(*, object_id, frame_ids, batches,
                                  support_uncertainty_prior_m=CONTACT_SUPPORT_UNCERTAINTY_PRIOR_M):
    """Join schema-valid results, preserving each raw result and optional receipt.

    A batch contains frame_ids, judgments keyed by judge name, and optional
    sources keyed by judge name (path/SHA256 receipts). Missing judges/rows are
    explicit unresolved evidence, not visible/contact observations.
    """
    ids = timeline_ids(frame_ids)
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("interaction object_id is required")
    if isinstance(support_uncertainty_prior_m, bool):
        raise ValueError("metric support prior must be a number, not a boolean")
    prior = float(support_uncertainty_prior_m)
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("metric support prior must be finite and nonnegative")
    target = "object:" + object_id.removeprefix("object:")
    indexes = {name: {} for name in JUDGMENTS}
    raw_sources = []
    seen_frames = set()
    for batch in batches:
        batch_ids = timeline_ids(batch["frame_ids"])
        if not set(batch_ids) <= set(ids) or seen_frames.intersection(batch_ids):
            raise ValueError("foreign or overlapping batch timeline")
        seen_frames.update(batch_ids)
        judgments = batch["judgments"]
        if set(judgments) - set(JUDGMENTS):
            raise ValueError("unknown interaction judgment")
        for name in JUDGMENTS:
            result = judgments.get(name)
            if result is None:
                continue
            source_index = len(raw_sources)
            raw_sources.append({"judgment": name, "frame_ids": batch_ids,
                                "role": DIAGNOSTIC_ROLE if name in {"occlusion", "depth_reliability"} else "independent_image_evidence",
                                "result": deepcopy(result),
                                "receipt": deepcopy(batch.get("sources", {}).get(name))})
            for key, row, position in _validated_judgment(name, result, batch_ids, object_id):
                indexes[name][key] = (row, {"source_index": source_index, "row_index": position})

    def evidence(name, frame, entity):
        return indexes[name].get((frame, entity), (None, None))

    rows, frames = [], []
    for frame in ids:
        visible, visibility_ref = evidence("visibility", frame, "all")
        visibility = deepcopy(visible["visibility"]) if visible else {
            entity: {"state": "unresolved", "occluder": None, "uncertainty": "missing independent visibility observation"}
            for entity in ("object", *SIDES)}
        frames.append({"frame_idx": frame, "visibility": visibility, "source_ref": visibility_ref})
        for side in SIDES:
            contact, contact_ref = evidence("contact", frame, side)
            pads, pads_ref = evidence("contact_pads", frame, side)
            occlusion, occlusion_ref = evidence("occlusion", frame, side)
            hand_depth, hand_depth_ref = evidence("depth_reliability", frame, side)
            object_depth, object_depth_ref = evidence("depth_reliability", frame, "object")
            refs = {"contact": contact_ref, "contact_pads": pads_ref, "occlusion": occlusion_ref,
                    "hand_depth_reliability": hand_depth_ref, "object_depth_reliability": object_depth_ref,
                    "visibility": visibility_ref}
            row = {
                "frame_start": frame, "frame_end": frame, "hand_side": side, "target_entity_id": target,
                "contact_state": contact["contact_state"] if contact else "unresolved",
                "contact_prior_probability": contact["contact_prior_probability"] if contact else 0.5,
                "contact_prior_source": "independent_contact_judgment" if contact else "unresolved_configuration_prior",
                "rationale": contact["rationale"] if contact else "missing independent contact observation",
                "uncertainty": contact["uncertainty"] if contact else "missing independent contact observation",
                "occlusion_relation": OCCLUSION_MAP[occlusion["relation"]] if occlusion else "unresolved",
                "depth_reliability": _depth_relation(hand_depth, object_depth),
                "contact_support_uncertainty_m": prior,
                "contact_support_uncertainty_source": {"kind": "conservative_configuration_prior",
                    "parameter": "contact_support_uncertainty_prior_m", "value_m": prior,
                    "is_metric_measurement": False},
                "source_refs": refs,
                "primary_pad_declarations": [
                    {"digit": digit, "declared": digit in pads["digits"] if pads else None,
                     "uncertainty": pads["uncertainty"] if pads else "missing independent pad observation"}
                    for digit in DIGITS],
            }
            if pads is not None:
                # Compatibility spelling; designation is not reachability or authority.
                row["primary_contact_pads"] = {"digits": list(pads["digits"]), "rationale": pads["rationale"]}
            rows.append(row)
    # Preserve the existing diagnostic/timeline views used by explicit rendering.
    diagnostics = {name: [] for name in ("occlusion", "depth_reliability")}
    timeline = {"contact": [], "pads": []}
    for source in raw_sources:
        name, result = source["judgment"], source["result"]
        entry = {"intervals": deepcopy(result.get("interaction_judgments", result.get("intervals", []))),
                 "provenance": deepcopy(result.get("judgment_provenance", {}))}
        if source["receipt"] is not None:
            entry["judgment_sha256"] = source["receipt"]["sha256"]
        if name in diagnostics:
            diagnostics[name].append({**entry, "role": DIAGNOSTIC_ROLE})
        elif name in {"contact", "contact_pads"}:
            timeline["contact" if name == "contact" else "pads"].append(entry)
    return {"schema": SCHEMA, "object_id": object_id, "status": "ok", "frame_ids": ids,
            "diagnostics": diagnostics, "timeline_evidence": timeline,
            "interaction_judgments": rows, "frames": frames, "source_judgments": raw_sources,
            "adapter_contract": {"occlusion_vocabulary": dict(OCCLUSION_MAP),
                "depth_reliability_role": DIAGNOSTIC_ROLE, "occlusion_role": DIAGNOSTIC_ROLE,
                "pad_role": "anatomical_declaration_not_reachability_or_contact_authority",
                "contact_support_uncertainty_prior_m": prior}}
