"""Anatomical contact declarations and shared visual reachability qualification."""
import numpy as np
from utils.data import frame_ids

SIDES = ("left", "right")


PRIMARY_DIGITS = ("thumb", "index", "middle", "ring", "little")


def primary_contact_pads(payload, ids, object_id):
    """Optional P17 interval declarations; missing designation grants no reach.

    Names select existing MANO fingertip pads, never caller-supplied vertices.
    Expanded reach requires likely_contact and visible image support at fit time.
    """
    ids = frame_ids(ids)
    states = interaction_states(payload, ids, object_id)
    selected = np.zeros((len(ids), 2, 5), bool)
    if payload is None:
        return selected
    rows = payload.get('interaction_judgments', payload.get('judgments', payload.get('segments', [])))
    for row in rows:
        declaration = row.get('primary_contact_pads')
        if declaration is None:
            continue
        if (not isinstance(declaration, dict) or set(declaration) != {'digits', 'rationale'}
                or not isinstance(declaration['digits'], list)
                or any(not isinstance(d, str) or d not in PRIMARY_DIGITS for d in declaration['digits'])
                or len(set(declaration['digits'])) != len(declaration['digits'])
                or not isinstance(declaration['rationale'], str) or not declaration['rationale'].strip()):
            raise ValueError('primary_contact_pads requires unique anatomical digits and nonempty rationale')
        target = row.get('target_entity_id', row.get('object_id', row.get('track_id')))
        if target and target.split(':', 1)[-1] != object_id.split(':', 1)[-1]:
            continue
        side = row.get('hand_side', row.get('side'))
        si = SIDES.index(side)
        start = int(row.get('frame_start', row.get('start_frame', int(min(ids)))))
        end = int(row.get('frame_end', row.get('end_frame', int(max(ids)))))
        mask = (ids >= start) & (ids <= end) & (states[:, si] == 'likely_contact')
        for digit in declaration['digits']:
            selected[mask, si, PRIMARY_DIGITS.index(digit)] = True
    # Standalone P17 has explicit visibility; occluded or unresolved poses are
    # retained but do not gain expanded attraction from an image-only hypothesis.
    for row in payload.get('frames', []):
        at = np.flatnonzero(ids == row['frame_idx'])
        visibility = row.get('visibility', {})
        for si, side in enumerate(SIDES):
            if any(visibility.get(entity, {}).get('state') not in {'visible', 'partially_visible'}
                   for entity in (side, 'object')):
                selected[at, si] = False
    return selected


def interaction_states(payload, ids, object_id):
    states = np.full((len(ids), 2), "legacy_visual_hypothesis", dtype="U32")
    if payload is None:
        return states
    if not isinstance(payload, dict):
        raise ValueError("interaction judgments payload is not an object")
    ids = frame_ids(ids)
    if payload.get("schema") == "v19_visible_contact_ownership_factor_v2":
        target = payload.get("target_entity_id")
        if not isinstance(target, str) or target.split(":", 1)[-1] != object_id.split(":", 1)[-1]:
            raise ValueError("P17 interaction report target mismatch")
        rows = payload.get("factor_rows")
        if not isinstance(rows, list):
            raise ValueError("P17 interaction report lacks factor_rows")
        states[:] = "unresolved"
        lookup = {int(value): position for position, value in enumerate(ids)}
        occupied = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid P17 factor row")
            idx = int(row["frame_idx"])
            side = row.get("hand_side", row.get("side"))
            state = row.get("agent_contact_state")
            if idx not in lookup or side not in SIDES or state not in {"likely_contact", "possible_contact", "no_contact", "unresolved"}:
                raise ValueError("invalid P17 frame/side/contact state")
            key = (idx, side)
            if key in occupied:
                raise ValueError("duplicate P17 frame/side state")
            occupied.add(key)
            states[lookup[idx], SIDES.index(side)] = state
        return states
    if payload.get("status") not in (None, "ok"):
        raise ValueError("interaction judgments status is not ok")
    rows = payload.get("interaction_judgments", payload.get("judgments", payload.get("segments")))
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing interaction_judgments")
    states[:] = "unresolved"
    occupied = np.zeros(states.shape, bool)
    matched_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("interaction judgment segment is not an object")
        target = row.get("target_entity_id", row.get("object_id", row.get("track_id")))
        if target is not None and not isinstance(target, str):
            raise ValueError("interaction judgment target must be a string")
        if target and target.split(":", 1)[-1] != object_id.split(":", 1)[-1]:
            continue
        side = row.get("hand_side", row.get("side"))
        if side not in SIDES:
            raise ValueError("interaction judgment side must be left or right")
        state = row.get("contact_state")
        if not isinstance(state, str) or state not in {"likely_contact", "possible_contact", "no_contact", "unresolved"}:
            raise ValueError("invalid interaction contact_state")
        bounds = [row.get("frame_start", row.get("start_frame", int(min(ids)))),
                  row.get("frame_end", row.get("end_frame", int(max(ids))))]
        try:
            start, end = (int(value) for value in bounds)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("interaction interval bounds must be integer frame IDs") from exc
        if any(isinstance(value, bool) or (isinstance(value, float) and value != int(value)) for value in bounds):
            raise ValueError("interaction interval bounds must be integer frame IDs")
        if end < start:
            raise ValueError("inverted interaction interval")
        mask = (ids >= start) & (ids <= end)
        si = SIDES.index(side)
        if occupied[mask, si].any():
            raise ValueError("overlapping interaction judgments")
        occupied[mask, si] = True
        states[mask, si] = state
        matched_rows += int(mask.any())
    if not matched_rows:
        raise ValueError(f"interaction judgments contained no segments for target {object_id!r} on this timeline")
    return states


VISUAL_PROXIMITY_MAX_GAP_M = 0.012


PRIMARY_PROXIMITY_MAX_GAP_M = 0.045


HAND_TRANSLATION_BOUND_M = 0.025


HAND_SCALE_BOUNDS = (0.85, 1.18)


SHARED_TRANSLATION_TOLERANCE_M = 0.001


def primary_similarity_reachable(points_camera, targets_camera, active=None):
    """Test one shared camera-origin scale/translation for active primary pads.

    The same hand-level similarity correction must explain every declared
    primary pad.  A one-millimeter tolerance covers the fact that each closest
    triangle target is a sampled visual hypothesis; it is not a second free
    translation per pad.  Independent per-point feasibility is insufficient.
    """
    points = np.asarray(points_camera, float)
    targets = np.asarray(targets_camera, float)
    if points.shape != (5, 3) or targets.shape != (5, 3):
        raise ValueError('five camera-space pad/triangle points required')
    active = np.ones(5, bool) if active is None else np.asarray(active, bool)
    if active.shape != (5,):
        raise ValueError('five active primary-pad values required')
    reachable = np.zeros(5, bool)
    active_ids = np.flatnonzero(active)
    if not len(active_ids):
        return reachable
    low, high = HAND_SCALE_BOUNDS
    tolerance = SHARED_TRANSLATION_TOLERANCE_M
    # For each coordinate, q - s*p is the one common translation coordinate,
    # up to the declared closest-triangle target tolerance. Intersect its
    # bounded translation feasibility and all pairwise common-translation
    # inequalities before activating any of the declared pads.
    for axis in range(3):
        for i in active_ids:
            p, q = points[i, axis], targets[i, axis]
            if not np.isfinite([p, q]).all():
                return reachable
            bound = HAND_TRANSLATION_BOUND_M + tolerance
            if abs(p) < 1e-12:
                if abs(q) > bound:
                    return reachable
                continue
            a, b = sorted(((q-bound) / p, (q+bound) / p))
            low, high = max(low, a), min(high, b)
            if low > high:
                return reachable
        for i in active_ids:
            for j in active_ids:
                if i == j:
                    continue
                coefficient = points[i, axis] - points[j, axis]
                lower = targets[i, axis] - targets[j, axis] - 2.0*tolerance
                if abs(coefficient) < 1e-12:
                    if lower > 0.0:
                        return reachable
                elif coefficient > 0.0:
                    low = max(low, lower / coefficient)
                else:
                    high = min(high, lower / coefficient)
                if low > high:
                    return reachable
    reachable[active_ids] = True
    return reachable


def visual_proximity_weights(inside, image_distances, triangle_gaps,
                             *, state="likely_contact", primary=None, reachable=None):
    """Demo-only image/triangle proximity weights; never a contact measurement."""
    inside = np.asarray(inside, dtype=bool)
    gaps = np.asarray(triangle_gaps, dtype=float)
    if inside.shape != (5,) or gaps.shape != (5,) or len(image_distances) != 5:
        raise ValueError("five fingertip proximity values required")
    result = np.zeros(5, np.float32)
    if state not in {"likely_contact", "possible_contact", "legacy_visual_hypothesis"}:
        return result
    primary = np.zeros(5, bool) if primary is None else np.asarray(primary, bool)
    reachable = np.zeros(5, bool) if reachable is None else np.asarray(reachable, bool)
    if primary.shape != (5,) or reachable.shape != (5,):
        raise ValueError('five primary/reachability values required')
    image_bounds = [16.0, 8.0, 8.0, 8.0, 8.0]
    strengths = [1.0, 0.12, 0.12, 0.12, 0.12]
    for finger in range(5):
        expanded = state == 'likely_contact' and primary[finger] and reachable[finger]
        bound = PRIMARY_PROXIMITY_MAX_GAP_M if expanded else VISUAL_PROXIMITY_MAX_GAP_M
        distance = image_distances[finger]
        if (inside[finger] and distance is not None and np.isfinite(distance)
                and float(distance) < (16. if expanded else image_bounds[finger])
                and np.isfinite(gaps[finger]) and 0 <= gaps[finger] < bound):
            result[finger] = 1. if expanded else strengths[finger]
    return result


def validate_visual_proximity_contract(rep, appearance_manifest_path):
    embedded = rep.get("appearance_mask_manifest")
    receipt = rep.get("visual_mesh_receipt")
    if appearance_manifest_path is not None and (not isinstance(embedded, dict)
            or Path(embedded.get("path", "")).resolve() != appearance_manifest_path.resolve()
            or embedded.get("sha256") != sha(appearance_manifest_path)):
        raise ValueError("appearance manifest does not match pose binding identity")
    if (not isinstance(receipt, dict)
            or receipt.get("demo_visual_proximity_fit_eligible") is not True
            or receipt.get("physical_contact_authority_eligible") is not False
            or any(receipt.get(key) is not False for key in (
                "generated_faces_contact_eligible", "generated_faces_collision_eligible",
                "generated_faces_sign_eligible", "generated_faces_sdf_eligible", "generated_faces_nonpenetration_eligible"))):
        raise ValueError("visual mesh lacks demo visual-proximity-only role contract")
    mesh_contract = rep.get("render_mesh_contract") or {}
    if (not receipt.get("output_mesh_path") or not mesh_contract.get("required_mesh_path")
            or Path(receipt["output_mesh_path"]).resolve() != Path(mesh_contract["required_mesh_path"]).resolve()
            or receipt.get("output_mesh_sha256") != mesh_contract.get("required_mesh_sha256")):
        raise ValueError("visual proximity role does not bind the rendered mesh identity")


from pathlib import Path
from utils.state import sha
