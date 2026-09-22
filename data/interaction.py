"""Immutable source declarations, separate from qualified geometric factors."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from observations.interaction.adapter import DIGITS, SCHEMA, SIDES, timeline_ids

ADAPTED_OCCLUSION_RELATIONS = {
    "hand_in_front_of_object", "object_in_front_of_hand", "no_visible_occlusion", "unresolved",
}
ADAPTED_DEPTH_RELIABILITY_STATES = {
    "hand_depth_unreliable", "object_depth_reliable", "mixed_or_unresolved", "not_evaluated",
}


def validate_adapted_interaction_row(row: Mapping[str, Any]) -> None:
    """Enforce provenance and mapped vocabularies at every adapted-data seam."""
    occlusion = row.get("occlusion_relation")
    if occlusion not in ADAPTED_OCCLUSION_RELATIONS:
        raise ValueError("adapted interaction evidence has an invalid occlusion_relation")
    depth = row.get("depth_reliability")
    if depth not in ADAPTED_DEPTH_RELIABILITY_STATES:
        raise ValueError("adapted interaction evidence has an invalid depth_reliability")
    support = row.get("contact_support_uncertainty_m")
    source = row.get("contact_support_uncertainty_source")
    if not isinstance(source, Mapping) or source.get("kind") != "conservative_configuration_prior":
        raise ValueError("adapted interaction evidence requires a conservative configuration prior")
    if source.get("parameter") != "contact_support_uncertainty_prior_m":
        raise ValueError("adapted interaction evidence prior parameter mismatch")
    value = source.get("value_m")
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            or support is None or float(value) != float(support)):
        raise ValueError("adapted interaction evidence prior value mismatch")
    if type(source.get("is_metric_measurement")) is not bool or source["is_metric_measurement"] is not False:
        raise ValueError("adapted interaction evidence prior cannot claim a metric measurement")

VISIBILITY_STATES = {"visible", "partially_visible", "occluded", "out_of_frame", "unresolved"}
CONTACT_STATES = {"likely_contact", "possible_contact", "no_contact", "unresolved"}


from data.immutable import freeze as freeze_evidence, thaw as thaw_evidence


@dataclass(frozen=True)
class DigitDeclaration:
    digit: str
    declared: bool | None
    uncertainty: str


@dataclass(frozen=True)
class InteractionDeclaration:
    frame_idx: int
    hand_side: str
    target_entity_id: str
    contact_state: str
    prior_probability: float | None
    uncertainty: str
    support_uncertainty_m: float | None
    support_uncertainty_source: Mapping[str, Any] | str | None
    occlusion_relation: str
    depth_reliability: str
    pads: tuple[DigitDeclaration, ...]
    source_refs: Mapping[str, Any]


@dataclass(frozen=True)
class VisibilityDeclaration:
    frame_idx: int
    entity_id: str
    state: str
    occluder: str | None
    uncertainty: str
    source_ref: Mapping[str, Any] | None


@dataclass(frozen=True)
class InteractionEvidence:
    schema: str | None
    declarations: Mapping[tuple[int, str, str], InteractionDeclaration]
    visibility: Mapping[tuple[int, str], VisibilityDeclaration]
    source: Mapping[str, Any]


def load_interaction_evidence(payload, frame_ids, object_ids):
    """Load declarations without assigning geometric reachability or authority.

    Existing v1 bundles without this source remain valid, with explicit missing
    declarations/visibility. Unversioned legacy inputs retain their raw evidence.
    """
    ids = timeline_ids(frame_ids)
    declarations, visibility = {}, {}
    for frame in ids:
        for entity in (*SIDES, *object_ids):
            visibility[frame, entity] = VisibilityDeclaration(frame, entity, "unresolved", None,
                "missing independent visibility observation", None)
        for target in object_ids:
            for side in SIDES:
                declarations[frame, side, target] = InteractionDeclaration(
                    frame, side, target, "unresolved", None, "missing independent contact observation",
                    None, None, "unresolved", "not_evaluated",
                    tuple(DigitDeclaration(digit, None, "missing independent pad observation") for digit in DIGITS),
                    MappingProxyType({}))
    if payload is None:
        return InteractionEvidence(None, MappingProxyType(declarations), MappingProxyType(visibility), MappingProxyType({}))
    if not isinstance(payload, dict) or payload.get("schema") not in {None, SCHEMA}:
        raise ValueError("unsupported interaction evidence schema")
    if payload.get("schema") == SCHEMA and "frame_ids" not in payload:
        raise ValueError("adapted interaction evidence requires explicit frame_ids")
    if payload.get("frame_ids", ids) != ids:
        raise ValueError("interaction evidence timeline mismatch")
    aliases = {alias: target for target in object_ids for alias in (target, target.removeprefix("object:"), "object:" + target.removeprefix("object:"))}
    target = aliases.get(payload.get("object_id"))
    if target is None:
        raise ValueError("interaction evidence foreign object identity")
    occupied = set()
    for row in payload.get("interaction_judgments", []):
        row_target = aliases.get(row.get("target_entity_id", payload["object_id"]))
        side, state = row.get("hand_side"), row.get("contact_state")
        if row_target != target or side not in SIDES or state not in CONTACT_STATES:
            raise ValueError("invalid interaction target/hand/state")
        start, end = row.get("frame_start"), row.get("frame_end")
        if type(start) is not int or type(end) is not int or start not in ids or end not in ids or start > end:
            raise ValueError("invalid interaction interval")
        pad_rows = row.get("primary_pad_declarations")
        if pad_rows is None:
            legacy = row.get("primary_contact_pads")
            digits = legacy.get("digits", []) if legacy else []
            if len(set(digits)) != len(digits) or set(digits) - set(DIGITS):
                raise ValueError("invalid anatomical pad declarations")
            pad_rows = [{"digit": digit, "declared": digit in digits if legacy else None,
                         "uncertainty": "legacy declaration has no per-digit uncertainty"} for digit in DIGITS]
        if [p.get("digit") for p in pad_rows] != list(DIGITS) or any(
                p.get("declared") is not None and type(p["declared"]) is not bool for p in pad_rows):
            raise ValueError("invalid per-digit declarations")
        if any(not isinstance(p.get("uncertainty"), str) or not p["uncertainty"].strip() for p in pad_rows):
            raise ValueError("per-digit uncertainty text is required")
        for key in ("contact_prior_probability", "contact_support_uncertainty_m"):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid numeric interaction {key}")
        probability = row.get("contact_prior_probability")
        if probability is not None and probability > 1:
            raise ValueError("contact probability outside [0, 1]")
        if payload.get("schema") == SCHEMA:
            validate_adapted_interaction_row(row)
        pads = tuple(DigitDeclaration(p["digit"], p["declared"], p["uncertainty"]) for p in pad_rows)
        for frame in ids:
            if not start <= frame <= end:
                continue
            key = (frame, side, target)
            if key in occupied:
                raise ValueError("duplicate/overlapping interaction evidence")
            occupied.add(key)
            declarations[key] = InteractionDeclaration(frame, side, target, state,
                row.get("contact_prior_probability"), row.get("uncertainty", "unspecified legacy uncertainty"),
                row.get("contact_support_uncertainty_m"), freeze_evidence(row.get("contact_support_uncertainty_source")),
                row.get("occlusion_relation", "unresolved"), row.get("depth_reliability", "not_evaluated"),
                pads, freeze_evidence(row.get("source_refs", {})))
    occupied = set()
    for row in payload.get("frames", []):
        frame = row.get("frame_idx")
        if type(frame) is not int or frame not in ids or frame in occupied:
            raise ValueError("duplicate/foreign visibility frame")
        occupied.add(frame)
        if set(row["visibility"]) != {"object", *SIDES}:
            raise ValueError("visibility requires object/left/right")
        for name, value in row["visibility"].items():
            entity = target if name == "object" else name
            if value.get("state") not in VISIBILITY_STATES or not isinstance(value.get("uncertainty"), str) or not value["uncertainty"].strip():
                raise ValueError("invalid visibility state/uncertainty")
            occluder = value.get("occluder")
            if occluder is not None and (not isinstance(occluder, str) or not occluder.strip()):
                raise ValueError("invalid visibility occluder")
            visibility[frame, entity] = VisibilityDeclaration(frame, entity, value["state"], occluder,
                value["uncertainty"], freeze_evidence(row.get("source_ref")))
    return InteractionEvidence(payload.get("schema"), MappingProxyType(declarations),
                               MappingProxyType(visibility), freeze_evidence(payload))
