from __future__ import annotations

"""Exact prefix objectives for named annotation reward compositions.

The default ``formal_annotation`` profile combines the non-overlapping formal
MANO interval and formal object-trajectory objectives. A transition reward is
``-reward_scale * (J(prefix_t) - J(prefix_{t-1}))``. With the documented empty
prefix ``J(empty)=0`` and ``gamma=1``, the undiscounted return is exactly
``-reward_scale * J(final)``. No optimizer is invoked by reward evaluation.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import math
import numpy as np

from envs.state import AnnotationState
from rewards.aggregation import aggregate_temporal, scipy_least_squares_cost
from rewards.profiles import RewardProfile, resolve_profile


@dataclass(frozen=True)
class FactorLedger:
    end_position: int
    terms: Mapping[str, float]
    statistics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    activation: Mapping[str, Any] = field(default_factory=dict)
    baseline_constant: float = 0.0
    profile: str = "formal_annotation"
    reward_scale: float = 1.0

    @property
    def objective(self) -> float:
        return float(self.baseline_constant + sum(self.terms.values()))

    def document(self) -> dict[str, Any]:
        return {
            "end_position": self.end_position,
            "profile": self.profile,
            "reward_scale": self.reward_scale,
            "terms": dict(self.terms),
            "statistics": {key: dict(value) for key, value in self.statistics.items()},
            "activation": dict(self.activation),
            "baseline_constant": self.baseline_constant,
            "objective": self.objective,
            "return_convention": "sum(reward)=-reward_scale*J(final), J(empty)=0, gamma=1",
        }


def _cfg(cfg: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _tensor(value: Any, device: str):
    import torch
    # Episode evidence is deliberately read-only. ``torch.tensor`` copies it,
    # avoiding undefined behavior warnings from aliasing a non-writable NumPy
    # buffer while keeping reward evaluation pure.
    return torch.tensor(value, dtype=torch.float32, device=device)


def _axis_angle_matrix(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        K = np.asarray([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]],
                        [-vector[1], vector[0], 0.0]])
        return np.eye(3) + K
    x, y, z = vector / angle
    K = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    R = np.asarray(matrix, dtype=np.float64)
    cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    axis = np.asarray([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= max(2.0 * math.sin(angle), 1e-12)
    return axis * angle


def _project(points: np.ndarray, T_world_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (np.asarray(points) - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]
    z = np.maximum(camera[:, 2], 1e-6)
    fx, fy, cx, cy = intrinsics
    return np.column_stack([fx * camera[:, 0] / z + cx, fy * camera[:, 1] / z + cy]), camera[:, 2]


def _as_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(row for row in value if isinstance(row, Mapping))
    return ()


def _factor_rows(episode: Any, families: set[str], end: int, *, side: str | None = None,
                 target: str | None = None) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    source = getattr(episode, "factor_rows", {})
    for key, values in source.items():
        if not isinstance(key, tuple) or len(key) != 4:
            continue
        frame, row_side, row_target, family = key
        if family not in families or int(frame) not in set(map(int, episode.frame_idx[:end])):
            continue
        if side is not None and row_side != side:
            continue
        if target is not None and row_target != target:
            continue
        result.extend(_as_rows(values))
    return result


def _row_frame(row: Mapping[str, Any]) -> int | None:
    value = row.get("frame_idx")
    return int(value) if value is not None else None


def _frame_position(episode: Any, frame: int) -> int | None:
    where = np.flatnonzero(np.asarray(episode.frame_idx) == int(frame))
    return int(where[0]) if len(where) else None


def _object_entity(obj: Mapping[str, Any], slot: str) -> str:
    return str(obj.get("entity_id") or slot)


def _contact_rows(episode: Any, end: int, side: str) -> list[tuple[int, str, Mapping[str, Any]]]:
    result: list[tuple[int, str, Mapping[str, Any]]] = []
    contacts = getattr(episode, "contacts", None)
    if contacts is None:
        contacts = getattr(episode, "contact", {})
    for key, row in contacts.items():
        if not isinstance(row, Mapping) or not isinstance(key, tuple):
            continue
        if len(key) == 3:
            frame, row_side, target = key
        elif len(key) == 2:
            frame, row_side = key
            target = str(row.get("target_entity_id") or "")
        else:
            continue
        pos = _frame_position(episode, int(frame))
        if row_side == side and pos is not None and pos < end:
            result.append((pos, str(target), row))
    return sorted(result, key=lambda item: (item[0], item[1]))


def _array_or_none(container: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    if key not in container:
        return None
    value = np.asarray(container[key])
    return value if value.shape == shape and np.isfinite(value).all() else None


class RewardEvaluator:
    def __init__(self, cfg: Mapping[str, Any] | Any, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = str(device)
        self.profile = resolve_profile(cfg)

    def objective(self, state: AnnotationState, end_position: int) -> FactorLedger:
        end = int(end_position)
        if end < 0 or end > state.episode.frame_count:
            raise ValueError("invalid objective prefix")
        scale = float(self.profile.get("reward_scale"))
        if end == 0:
            return FactorLedger(0, {}, {}, {"prefix": "empty_J=0"}, profile=self.profile.name,
                                reward_scale=scale)
        terms: dict[str, float] = {}
        statistics: dict[str, dict[str, Any]] = {}
        activation: dict[str, Any] = {"composition": self.profile.name}

        def add(name: str, value: Any, **stats: Any) -> None:
            number = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            if not np.isfinite(number):
                raise FloatingPointError(f"nonfinite factor {name}: {number}")
            if name in terms:
                terms[name] += number
                if stats:
                    previous = statistics.get(name)
                    if previous is None:
                        statistics[name] = {"contributions": [stats]}
                    elif "contributions" in previous:
                        previous["contributions"].append(stats)
                    else:
                        statistics[name] = {"contributions": [previous, stats]}
                return
            terms[name] = number
            if stats:
                statistics[name] = stats

        if self.profile.formal_hand:
            self._formal_hand(state, end, add, activation)
            self._formal_contact(state, end, add, activation)
            self._signed_physics(state, end, add, activation)
            if not self.profile.formal_object and bool(_cfg(self.cfg, "optimize_object_translation", False)):
                self._formal_interval_object_translation(state, end, add, activation)
        if self.profile.formal_object:
            self._formal_object(state, end, add, activation)
        if self.profile.visual_joint_fixed_scale:
            self._visual_joint_fixed_scale(state, end, add, activation)
        return FactorLedger(end, terms, statistics, activation, profile=self.profile.name,
                            reward_scale=scale)

    def _formal_hand(self, state: AnnotationState, end: int, add: Any,
                     activation: dict[str, Any]) -> None:
        import torch
        from rewards.terms.common import magnitude_hinge
        from rewards.terms.hand.observation import joint_depth_deadband, joint_pixel_deadband
        from rewards.terms.hand.priors import (formal_articulation, formal_root_orientation,
                                        formal_translation, hand_ray_translation_prior)
        from rewards.terms.interaction.physics import visible_surface_depth_order
        from rewards.terms.motion.temporal import parameter_acceleration, parameter_velocity

        episode = state.episode
        frames = np.asarray(episode.frame_idx[:end])
        for side in episode.layout.hand_slots:
            source = episode.hands[side]
            valid = np.asarray(source["valid"][:end], dtype=bool)
            trans_np = np.asarray(state.hand_translation[side][:end] - source["translation"][:end], dtype=np.float64)
            root_np = np.stack([_rotation_vector(_axis_angle_matrix(state.hand_root[side][i]) @
                                                 _axis_angle_matrix(source["root_orient"][i]).T)
                                for i in range(end)]).reshape(end, 1, 3)
            pose_np = np.stack([np.stack([
                _rotation_vector(_axis_angle_matrix(state.hand_articulation[side][i].reshape(15, 3)[j]) @
                                 _axis_angle_matrix(source["articulation"][i].reshape(15, 3)[j]).T)
                for j in range(15)]) for i in range(end)])
            trans, root, pose = (_tensor(trans_np, self.device), _tensor(root_np, self.device),
                                 _tensor(pose_np, self.device))

            row_vis = _array_or_none(source, "hand_observation_visibility_weight_multiplier",
                                     (episode.frame_count,))
            pose_vis = _array_or_none(source, "pose_visibility", (episode.frame_count, 15))
            joint_vis = _array_or_none(source, "joint_visibility", (episode.frame_count, 21))
            if valid.any() and row_vis is None:
                raise ValueError(f"hand {side}: valid rows require measured hand_observation_visibility_weight_multiplier shape ({episode.frame_count},)")
            if valid.any() and joint_vis is None:
                raise ValueError(f"hand {side}: valid rows require measured joint_visibility shape ({episode.frame_count}, 21)")
            if valid.any() and pose_vis is None:
                raise ValueError(f"hand {side}: valid rows require measured pose_visibility shape ({episode.frame_count}, 15)")
            if row_vis is None:
                activation[f"hand/{side}/row_visibility"] = "inactive_hand_absent"
            else:
                observed = np.asarray(row_vis[:end], dtype=np.float32) * valid.astype(np.float32)
                observed_t = _tensor(observed, self.device)
                for name, value, weight, kernel, historical_dim in (
                    ("translation_prior", trans, self.profile.get("translation_prior_weight"), formal_translation, 3),
                    ("root_prior", root, self.profile.get("root_prior_weight"), formal_root_orientation, 9),
                ):
                    numerator = float(weight) * float(np.sum(np.square(value.detach().cpu().numpy()) *
                                                              observed.reshape((end,) + (1,) * (value.ndim - 1))))
                    denominator = max(float(historical_dim) * float(observed.sum()), 1.0)
                    add(f"hand/{side}/{name}", kernel(value, observed_t, float(weight)),
                        numerator=numerator, denominator=denominator,
                        active_count=int(np.count_nonzero(observed > 0)), reduction="weighted_interval_mean")
            if pose_vis is None:
                activation[f"hand/{side}/pose_visibility"] = "inactive_hand_absent"
            else:
                pvis = np.asarray(pose_vis[:end], dtype=np.float32) * valid[:, None]
                weight = float(self.profile.get("pose_prior_weight"))
                numerator = weight * float(np.sum(np.square(pose_np) * pvis[..., None]))
                denominator = max(3.0 * float(pvis.sum()), 1.0)
                add(f"hand/{side}/articulation_prior", formal_articulation(pose, _tensor(pvis, self.device), weight),
                    numerator=numerator, denominator=denominator,
                    active_count=int(np.count_nonzero(pvis > 0)), reduction="weighted_interval_mean")

            for name, value in (("translation", trans), ("root", root), ("articulation", pose)):
                velocity, velocity_value, velocity_den, edge_count = aggregate_temporal(
                    value, valid, frames, float(self.profile.get("smooth_weight")), parameter_velocity, 1)
                acceleration, acceleration_value, acceleration_den, triple_count = aggregate_temporal(
                    value, valid, frames, float(self.profile.get("accel_weight")), parameter_acceleration, 2)
                add(f"hand/{side}/{name}_velocity", velocity, numerator=velocity_value * velocity_den,
                    denominator=velocity_den, active_count=edge_count, reduction="mean_over_all_valid_edges")
                add(f"hand/{side}/{name}_acceleration", acceleration,
                    numerator=acceleration_value * acceleration_den, denominator=acceleration_den,
                    active_count=triple_count, reduction="mean_over_all_valid_triples")

            selected = np.flatnonzero(valid)
            if len(selected):
                ids = torch.as_tensor(selected, dtype=torch.long, device=self.device)
                for name, value, onset in (
                    ("translation", trans, self.profile.get("max_translation_m")),
                    ("root", root, self.profile.get("max_root_delta_rad")),
                    ("articulation", pose, self.profile.get("max_pose_delta_rad")),
                ):
                    add(f"hand/{side}/{name}_soft_bound", magnitude_hinge(
                        value[ids], float(onset), float(self.profile.get("bound_hinge_weight"))),
                        active_count=len(selected), reduction="mean_squared_vector_hinge")

            if bool(self.profile.get("use_hand_ray_shift_prior")):
                targets = _array_or_none(source, "hand_ray_shift_prior_world_m", (episode.frame_count, 3))
                weights = _array_or_none(source, "hand_ray_shift_prior_weight", (episode.frame_count,))
                if targets is None or weights is None:
                    raise ValueError(f"{side}: hand-ray prior requested but measured ray fields are missing or malformed")
                else:
                    active = valid & (np.asarray(weights[:end]) > 0) & (np.linalg.norm(targets[:end], axis=1) > 1e-9)
                    if active.any():
                        ids = torch.as_tensor(np.flatnonzero(active), dtype=torch.long, device=self.device)
                        add(f"hand/{side}/hand_ray_prior", hand_ray_translation_prior(
                            trans[ids], _tensor(targets[:end][active], self.device),
                            _tensor(weights[:end][active], self.device)), active_count=int(active.sum()),
                            denominator=int(active.sum()) * 3, reduction="weighted_sum_over_difference_numel")
                        activation[f"hand/{side}/hand_ray_prior"] = "active_configured_measured_ray_prior"
                    else:
                        activation[f"hand/{side}/hand_ray_prior"] = "inactive_no_positive_measured_ray_rows"
            else:
                activation[f"hand/{side}/hand_ray_prior"] = "inactive_profile_switch_disabled"

            for i in np.flatnonzero(valid):
                frame = int(episode.frame_idx[i])
                if joint_vis is None:
                    activation[f"hand/{side}/joint_observation/frame={frame}"] = "inactive_hand_absent"
                else:
                    weights_np = np.asarray(joint_vis[i], dtype=np.float32)
                    if not np.any(weights_np > 0):
                        activation[f"hand/{side}/joint_observation/frame={frame}"] = "inactive_zero_measured_joint_visibility"
                    else:
                        base_uv, base_depth = _project(source["joints"][i], episode.T_world_camera[i], episode.intrinsics[i])
                        uv, depth = _project(state.hand_joints[side][i], episode.T_world_camera[i], episode.intrinsics[i])
                        weights_t = _tensor(weights_np, self.device)
                        add(f"hand/{side}/joint_pixel_deadband/frame={frame}", joint_pixel_deadband(
                            _tensor(uv, self.device), _tensor(base_uv, self.device), weights_t,
                            float(self.profile.get("visible_shift_limit_px")),
                            float(self.profile.get("visible_hinge_weight"))),
                            denominator=float(max(weights_np.sum(), 1.0)), active_count=int(np.count_nonzero(weights_np > 0)),
                            reduction="visibility_weighted_row_mean")
                        add(f"hand/{side}/joint_depth_deadband/frame={frame}", joint_depth_deadband(
                            _tensor(depth, self.device), _tensor(base_depth, self.device), weights_t,
                            float(self.profile.get("depth_shift_limit_m")),
                            float(self.profile.get("depth_hinge_weight"))),
                            denominator=float(max(weights_np.sum(), 1.0)), active_count=int(np.count_nonzero(weights_np > 0)),
                            reduction="visibility_weighted_row_mean")
                        activation[f"hand/{side}/joint_observation/frame={frame}"] = "active_measured_joint_visibility"

                depth_rows = source.get("depth_order")
                evidence = depth_rows[i] if isinstance(depth_rows, Sequence) and len(depth_rows) > i else None
                if evidence is None and all(name in source for name in
                    ("depth_order_vertex_ids", "depth_order_surface_depth_m", "depth_order_state")):
                    ids_rows = source["depth_order_vertex_ids"]
                    surface_rows = source["depth_order_surface_depth_m"]
                    state_rows = source["depth_order_state"]
                    evidence = {
                        "vertex_ids": ids_rows[i], "surface_depth_m": surface_rows[i],
                        "state": state_rows[i],
                    }
                state_name = str(evidence.get("state", "")) if isinstance(evidence, Mapping) else "inactive_missing_independent_first_surface"
                enabled_by_config = bool(_cfg(self.cfg, "visible_surface_depth_order_term", False))
                enabled = enabled_by_config or state_name.startswith("active_")
                if not enabled:
                    activation[f"depth_order/{side}/frame={frame}"] = "inactive_profile_switch_and_evidence"
                    continue
                if not isinstance(evidence, Mapping):
                    if enabled_by_config:
                        raise ValueError(f"{side} frame {frame}: depth-order factor requested but evidence is missing")
                    activation[f"depth_order/{side}/frame={frame}"] = "inactive_missing_independent_first_surface"
                    continue
                ids_np = np.asarray(evidence.get("vertex_ids", []), dtype=np.int64)
                surface = np.asarray(evidence.get("surface_depth_m", []), dtype=np.float64)
                if len(ids_np) != len(surface):
                    raise ValueError(f"{side} frame {frame}: malformed active depth-order evidence")
                if not len(ids_np):
                    activation[f"depth_order/{side}/frame={frame}"] = state_name or "inactive_no_supported_vertices"
                    continue
                margin = float(self.profile.get("visible_surface_depth_order_margin_m"))
                source_depth = _project(source["vertices"][i, ids_np], episode.T_world_camera[i], episode.intrinsics[i])[1]
                initial_violation = np.maximum(0.0, surface - margin - source_depth)
                order = np.argsort(initial_violation)[::-1]
                cap = int(self.profile.get("max_visible_surface_depth_vertices"))
                if cap > 0:
                    order = order[:cap]
                ids_np, surface = ids_np[order], surface[order]
                candidate_depth = _project(state.hand_vertices[side][i, ids_np], episode.T_world_camera[i], episode.intrinsics[i])[1]
                residual = np.maximum(0.0, surface - margin - candidate_depth)
                weight = float(self.profile.get("visible_surface_depth_order_weight"))
                positive = int(np.count_nonzero(residual > 0))
                add(f"hand/{side}/visible_surface_depth_order", visible_surface_depth_order(
                    _tensor(candidate_depth, self.device), _tensor(surface, self.device), margin, weight),
                    numerator=weight * float(np.square(residual).sum()), denominator=max(positive, 1),
                    active_count=len(ids_np), positive_count=positive, reduction="positive_residual_count")
                activation[f"depth_order/{side}/frame={frame}"] = state_name

    def _formal_contact(self, state: AnnotationState, end: int, add: Any,
                        activation: dict[str, Any]) -> None:
        import torch
        from rewards.terms.interaction.contact import (contact_geometry_likelihood, contact_state_continuity,
            contact_state_prior, normal_deadband, normal_scaled_attraction)

        episode = state.episode
        for side_index, side in enumerate(episode.layout.hand_slots):
            candidates = _contact_rows(episode, end, side)
            active_rows: list[tuple[int, str, Mapping[str, Any]]] = []
            for i, target, row in candidates:
                key = f"contact/{side}/{target or 'unbound'}/frame={int(episode.frame_idx[i])}"
                active = (str(row.get("state")) == "active_contact_patch" and
                          bool(episode.hands[side]["valid"][i]) and float(row.get("weight", 0.0)) > 0)
                if not active:
                    activation[key] = str(row.get("state") or "inactive_missing_weight_or_hand")
                    continue
                slot = next((name for name in episode.layout.object_slots
                             if _object_entity(episode.objects[name], name) == target), None)
                refreshed_contact = None
                if hasattr(episode, "contacts"):
                    refreshed = state.term_cache.get(i)
                    if refreshed is None:
                        from envs.dynamics import refresh_candidate_terms
                        refreshed = refresh_candidate_terms(state, self.cfg, i)
                        state.term_cache[i] = refreshed
                    refreshed_contact = refreshed.get("contact_by_key", {}).get((side, target))
                ids = np.asarray((refreshed_contact or row).get("vertex_ids", []), dtype=np.int64)
                targets_object = np.asarray(row.get("targets_object", []), dtype=np.float64).reshape(-1, 3)
                normals_object = np.asarray(row.get("normals_object", []), dtype=np.float64).reshape(-1, 3)
                if slot is None:
                    activation[key] = "inactive_target_entity_unavailable"
                    continue
                if not (len(ids) == len(targets_object) == len(normals_object)):
                    raise ValueError(f"{key}: malformed frozen contact geometry")
                if len(ids) == 0:
                    activation[key] = str(row.get("membership_state") or "inactive_missing_frozen_contact_geometry")
                    continue
                active_rows.append((i, target, row))
                R = state.object_rotation[slot][i]
                t = state.object_translation[slot][i]
                if refreshed_contact is not None:
                    targets = np.asarray(refreshed_contact.get("targets_world", []), dtype=np.float64).reshape(-1, 3)
                    normals = np.asarray(refreshed_contact.get("normals_world", []), dtype=np.float64).reshape(-1, 3)
                    if not (len(ids) == len(targets) == len(normals)):
                        raise ValueError(f"{key}: malformed refreshed contact geometry")
                else:
                    targets = targets_object @ R.T + t[None, :]
                    normals = normals_object @ R.T
                gap = np.sum((state.hand_vertices[side][i, ids] - targets) * normals, axis=1)
                prior = float(row.get("prior", 0.0))
                if episode.layout.contact_variables_per_hand:
                    probability = torch.sigmoid(_tensor(state.contact_logits[i, side_index], self.device)).mean()
                else:
                    probability = _tensor(prior, self.device)
                weight = float(row.get("weight", 0.0))
                margin = float(row.get("margin_m", 0.0))
                uncertainty = float(row.get("support_uncertainty_m", 0.0))
                if str(self.profile.get("contact_patch_residual_mode")) == "support_scaled_attraction":
                    value = normal_scaled_attraction(_tensor(gap, self.device), _tensor(margin, self.device),
                        _tensor(uncertainty, self.device), weight, probability)
                    mode = "support_scaled_attraction"
                else:
                    value = normal_deadband(_tensor(gap, self.device), margin + uncertainty, weight, probability)
                    mode = "normal_deadband"
                add(f"contact/{side}/{mode}", value, active_count=len(ids),
                    reduction="sum_of_row_means_over_frozen_patch_vertices")
                activation[key] = str(row.get("membership_state") or "active_frozen_source_patch")

            if not episode.layout.contact_variables_per_hand:
                activation[f"contact/{side}/latent_state"] = "inactive_profile_has_no_contact_variable"
                continue
            if not active_rows:
                activation[f"contact/{side}/latent_state"] = "inactive_no_active_contact_rows"
                continue
            positions = np.asarray([i for i, _, _ in active_rows], dtype=np.int64)
            priors = np.asarray([float(row.get("prior", 0.0)) for _, _, row in active_rows], dtype=np.float32)
            weights = np.asarray([float(row.get("weight", 0.0)) for _, _, row in active_rows], dtype=np.float32)
            strength = weights * float(self.profile.get("contact_state_prior_residual_scale_m")) ** 2
            probability = torch.stack([torch.sigmoid(_tensor(state.contact_logits[i, side_index], self.device)).mean()
                                       for i in positions])
            active_t = torch.ones(len(active_rows), dtype=torch.bool, device=self.device)
            add(f"contact/{side}/state_prior", contact_state_prior(probability, _tensor(priors, self.device),
                _tensor(strength, self.device), active_t), numerator=float(np.sum(strength *
                np.square(probability.detach().cpu().numpy() - priors))), denominator=len(active_rows),
                active_count=len(active_rows), reduction="active_row_mean")
            if bool(self.profile.get("contact_state_geometry_likelihood")):
                targets = []
                for _, _, row in active_rows:
                    frozen_target = row.get("initial_contact_geometry_target_probability")
                    if frozen_target is not None:
                        targets.append(float(np.clip(float(frozen_target), 0.0001, 1.0 - 0.0001)))
                        continue
                    deadband = max(float(row.get("margin_m", 0.0)) + float(row.get("support_uncertainty_m", 0.0)), 1e-6)
                    distances = np.asarray(row.get("distances_m", []), dtype=np.float64)
                    targets.append(float(np.exp(-0.5 * (float(np.median(distances)) / deadband) ** 2))
                                   if len(distances) else float(row.get("prior", 0.0)))
                add(f"contact/{side}/geometry_likelihood", contact_geometry_likelihood(probability,
                    _tensor(targets, self.device), _tensor(strength, self.device), active_t),
                    denominator=len(active_rows), active_count=len(active_rows), reduction="active_row_mean")
                activation[f"contact/{side}/geometry_likelihood"] = "active_configured_frozen_source_geometry_target"
            else:
                activation[f"contact/{side}/geometry_likelihood"] = "inactive_profile_switch_disabled"
            pairs = []
            for target in sorted({target for _, target, _ in active_rows}):
                indexed = [(index, item) for index, item in enumerate(active_rows) if item[1] == target]
                for (a, (i0, _, _)), (b, (i1, _, _)) in zip(indexed[:-1], indexed[1:]):
                    if int(episode.frame_idx[i1]) == int(episode.frame_idx[i0]) + 1:
                        pairs.append((a, b))
            if pairs:
                pair_t = torch.as_tensor(pairs, dtype=torch.long, device=self.device)
                add(f"contact/{side}/state_continuity", contact_state_continuity(probability,
                    _tensor(strength, self.device), pair_t,
                    float(self.profile.get("contact_state_temporal_strength"))), active_count=len(pairs),
                    denominator=len(pairs), reduction="adjacent_active_row_mean")
            else:
                activation[f"contact/{side}/state_continuity"] = "inactive_no_adjacent_active_same_target_rows"

    def _signed_physics(self, state: AnnotationState, end: int, add: Any,
                        activation: dict[str, Any]) -> None:
        from rewards.terms.interaction.physics import signed_active_penetration, signed_dense_penetration
        episode = state.episode
        for i in range(end):
            needs_refresh = any(bool(episode.objects[slot].get("watertight", False) and
                                     episode.objects[slot].get("signed_geometry_ready", False) and
                                     np.any(np.asarray(episode.objects[slot].get("signed_authority", []), dtype=bool)))
                                for slot in episode.layout.object_slots)
            if not needs_refresh:
                for slot in episode.layout.object_slots:
                    reason = "inactive_nonwatertight_or_missing_local_face_authority"
                    frame = int(episode.frame_idx[i])
                    activation[f"signed/{slot}/{frame}"] = reason
                    activation[f"signed/{slot}/frame={frame}"] = reason
                continue
            refreshed = state.term_cache.get(i)
            if refreshed is None:
                from envs.dynamics import refresh_candidate_terms
                refreshed = refresh_candidate_terms(state, self.cfg, i)
                state.term_cache[i] = refreshed
            for object_slot, trace in refreshed.get("signed", {}).items():
                frame = int(episode.frame_idx[i])
                activation[f"signed/{object_slot}/{frame}"] = trace.get("state")
                activation[f"signed/{object_slot}/frame={frame}"] = trace.get("state")
                if not str(trace.get("state", "")).startswith("active_"):
                    continue
                for side, row in trace.get("hands", {}).items():
                    for family, kernel, weight_name in (
                        ("active", signed_active_penetration, "observed_penetration_weight"),
                        ("dense", signed_dense_penetration, "dense_observed_penetration_weight"),
                    ):
                        if family == "dense" and not bool(self.profile.get("dense_observed_surface_barrier")):
                            activation[f"signed/{object_slot}/{side}/dense/frame={frame}"] = "inactive_profile_switch_disabled"
                            continue
                        ids = np.asarray(row.get(f"{family}_ids", []), dtype=np.int64)
                        if not len(ids):
                            activation[f"signed/{object_slot}/{side}/{family}/frame={frame}"] = "inactive_empty_candidate_set"
                            continue
                        moved_np = state.hand_vertices[side][i, ids] - episode.hands[side]["vertices"][i, ids]
                        normals_np = np.asarray(row[f"{family}_normals"])
                        depth_np = np.asarray(row[f"{family}_depths"])
                        uncertainty = float(row["uncertainty_m"])
                        residual = np.maximum(0.0, depth_np - uncertainty - np.sum(normals_np * moved_np, axis=1))
                        weight = float(self.profile.get(weight_name))
                        denominator = len(residual) if family == "active" else max(int(np.count_nonzero(residual > 0)), 1)
                        add(f"signed/{object_slot}/{side}/{family}/frame={frame}", kernel(
                            _tensor(moved_np, self.device), _tensor(normals_np, self.device),
                            _tensor(depth_np, self.device), uncertainty, weight),
                            numerator=weight * float(np.square(residual).sum()), denominator=denominator,
                            active_count=len(residual), positive_count=int(np.count_nonzero(residual > 0)),
                            reduction="selected_mean" if family == "active" else "positive_residual_count")

    def _formal_interval_object_translation(self, state: AnnotationState, end: int, add: Any,
                                            activation: dict[str, Any]) -> None:
        """Legacy formal-interval optional translation-only object block."""
        import torch
        from rewards.terms.common import magnitude_hinge
        from rewards.terms.motion.temporal import parameter_acceleration, parameter_velocity
        from rewards.terms.object.pose import object_translation_prior

        episode = state.episode
        frames = np.asarray(episode.frame_idx[:end])
        for slot in episode.layout.object_slots:
            obj = episode.objects[slot]
            valid = np.asarray(obj.get("valid", obj.get("present"))[:end], dtype=bool)
            delta = _tensor(state.object_translation[slot][:end] - np.asarray(obj["translation"][:end]), self.device)
            ids_np = np.flatnonzero(valid)
            if not len(ids_np):
                activation[f"object/{slot}/formal_interval_translation"] = "inactive_no_present_object_rows"
                continue
            ids = torch.as_tensor(ids_np, dtype=torch.long, device=self.device)
            add(f"object/{slot}/formal_interval_translation_prior", object_translation_prior(
                delta[ids], float(self.profile.get("object_translation_prior_weight"))),
                active_count=len(ids_np), reduction="elementwise_mean")
            velocity, value1, den1, n1 = aggregate_temporal(delta, valid, frames,
                float(self.profile.get("object_smooth_weight")), parameter_velocity, 1)
            acceleration, value2, den2, n2 = aggregate_temporal(delta, valid, frames,
                float(self.profile.get("accel_weight")), parameter_acceleration, 2)
            add(f"object/{slot}/formal_interval_translation_velocity", velocity,
                numerator=value1 * den1, denominator=den1, active_count=n1,
                reduction="mean_over_all_valid_edges")
            add(f"object/{slot}/formal_interval_translation_acceleration", acceleration,
                numerator=value2 * den2, denominator=den2, active_count=n2,
                reduction="mean_over_all_valid_triples")
            add(f"object/{slot}/formal_interval_translation_soft_bound", magnitude_hinge(
                delta[ids], float(self.profile.get("max_object_translation_m")),
                float(self.profile.get("bound_hinge_weight"))), active_count=len(ids_np),
                reduction="mean_squared_vector_hinge")
            activation[f"object/{slot}/formal_interval_translation"] = "active_optimize_object_translation"

    def _formal_object(self, state: AnnotationState, end: int, add: Any,
                       activation: dict[str, Any]) -> None:
        from rewards.terms.interaction.physics import nonpenetration_translation_target
        from rewards.terms.motion.temporal import correction_acceleration, correction_velocity
        from rewards.terms.object.pose import source_pose_rotation, source_pose_translation

        episode = state.episode
        loss = str(self.profile.get("formal_object_loss"))
        for slot in episode.layout.object_slots:
            obj = episode.objects[slot]
            entity = _object_entity(obj, slot)
            pose_valid = np.asarray(obj.get("pose_valid", obj.get("valid", obj.get("present")))[:end], dtype=bool)
            valid = pose_valid
            trans = np.asarray(state.object_translation[slot][:end] - obj["translation"][:end], dtype=np.float64)
            rot = np.stack([_rotation_vector(state.object_rotation[slot][i] @ np.asarray(obj["rotation"][i]).T)
                            for i in range(end)])
            sigma_t_raw = obj.get("translation_sigma_m", _cfg(self.cfg, "object_translation_sigma_m", None))
            sigma_r_raw = obj.get("rotation_sigma_rad", _cfg(self.cfg, "object_rotation_sigma_rad", None))
            if (sigma_t_raw is None) != (sigma_r_raw is None):
                raise ValueError(f"object {slot}: partial sigma declaration requires both translation_sigma_m and rotation_sigma_rad")
            object_rows = _factor_rows(episode, {"object_pose_observation", "rigid_pose_observation",
                                                "formal_object_pose"}, end, target=entity)
            row_sigma_declared = None
            if (sigma_t_raw is None or sigma_r_raw is None) and object_rows:
                row_sigma_t = np.full(end, np.nan, dtype=np.float64)
                row_sigma_r = np.full(end, np.nan, dtype=np.float64)
                row_sigma_declared = np.zeros(end, dtype=bool)
                for row in object_rows:
                    pos = _frame_position(episode, int(row.get("frame_idx", -1)))
                    if pos is None or pos >= end:
                        continue
                    has_t = row.get("translation_sigma_m") is not None
                    has_r = row.get("rotation_sigma_rad") is not None
                    if has_t != has_r:
                        raise ValueError(f"object {slot} frame {episode.frame_idx[pos]}: partial row sigma declaration")
                    if has_t:
                        row_sigma_t[pos] = float(row["translation_sigma_m"])
                        row_sigma_r[pos] = float(row["rotation_sigma_rad"])
                        row_sigma_declared[pos] = True
                if sigma_t_raw is None and row_sigma_declared[valid].any():
                    sigma_t_raw = row_sigma_t
                    sigma_r_raw = row_sigma_r
            sigma_t = self._broadcast_sigma(sigma_t_raw, end)
            sigma_r = self._broadcast_sigma(sigma_r_raw, end)
            if sigma_t_raw is not None and sigma_t is None:
                raise ValueError(f"object {slot}: malformed translation_sigma_m")
            if sigma_r_raw is not None and sigma_r is None:
                raise ValueError(f"object {slot}: malformed rotation_sigma_rad")
            if sigma_t is None or sigma_r is None:
                activation[f"object/{slot}/source_sigma_priors"] = "inactive_missing_measured_or_explicit_sigma"
            else:
                active = valid if row_sigma_declared is None else valid & row_sigma_declared
                if (not np.isfinite(sigma_t[active]).all() or not np.isfinite(sigma_r[active]).all()
                        or np.any(sigma_t[active] <= 0) or np.any(sigma_r[active] <= 0)):
                    raise ValueError(f"object {slot}: declared formal object row has invalid sigma")
                if active.any():
                    rt = np.concatenate([np.asarray(source_pose_translation(trans[i], sigma_t[i])).reshape(-1)
                                         for i in np.flatnonzero(active)])
                    rr = np.concatenate([np.asarray(source_pose_rotation(rot[i], sigma_r[i])).reshape(-1)
                                         for i in np.flatnonzero(active)])
                    add(f"object/{slot}/translation_sigma_prior", scipy_least_squares_cost(rt, loss),
                        residual_count=len(rt), active_count=int(active.sum()), reduction=f"scipy_0.5_rho/{loss}")
                    add(f"object/{slot}/rotation_sigma_prior", scipy_least_squares_cost(rr, loss),
                        residual_count=len(rr), active_count=int(active.sum()), reduction=f"scipy_0.5_rho/{loss}")
                    activation[f"object/{slot}/source_sigma_priors"] = "active_formal_object_rows"
                else:
                    activation[f"object/{slot}/source_sigma_priors"] = "inactive_no_eligible_positive_sigma_rows"

            target_raw = obj.get("nonpenetration_target_world_m")
            weight_raw = obj.get("nonpenetration_weight")
            if (target_raw is None) != (weight_raw is None):
                raise ValueError(f"object {slot}: partial nonpenetration declaration requires target and weight")
            if target_raw is None and weight_raw is None and object_rows:
                targets = np.zeros((end, 3), dtype=np.float64)
                weights = np.zeros(end, dtype=np.float64)
                declared = np.zeros(end, dtype=bool)
                for row in object_rows:
                    has_target = row.get("nonpenetration_target_world_m") is not None
                    has_weight = row.get("nonpenetration_weight") is not None
                    if has_target != has_weight:
                        raise ValueError(f"object {slot} frame {row.get('frame_idx')}: partial row nonpenetration declaration")
                    if not has_target:
                        continue
                    pos = _frame_position(episode, int(row.get("frame_idx", -1)))
                    if pos is None or pos >= end:
                        continue
                    target = np.asarray(row["nonpenetration_target_world_m"], dtype=np.float64)
                    if target.shape != (3,) or not np.isfinite(target).all():
                        raise ValueError(f"object {slot}: malformed row nonpenetration target")
                    targets[pos] = target
                    weights[pos] = float(row["nonpenetration_weight"])
                    declared[pos] = True
                if declared[valid].any():
                    target_raw, weight_raw = targets, weights
            if target_raw is not None and weight_raw is not None:
                targets = np.asarray(target_raw, dtype=np.float64)
                if targets.ndim != 2 or targets.shape[0] < end or targets.shape[1:] != (3,) or not np.isfinite(targets[:end]).all():
                    raise ValueError(f"object {slot}: malformed nonpenetration target")
                targets = targets[:end]
                weights = np.asarray(weight_raw, dtype=np.float64)
                weights = np.full(end, float(weights)) if weights.ndim == 0 else weights[:end]
                if weights.shape != (end,) or not np.isfinite(weights).all():
                    raise ValueError(f"object {slot}: malformed nonpenetration weight")
                active = valid & (weights > 0)
                residuals = [nonpenetration_translation_target(trans[i], targets[i], weights[i],
                             float(self.profile.get("sigma_nonpenetration_target_m"))) for i in np.flatnonzero(active)]
                if residuals:
                    vector = np.concatenate([np.asarray(v).reshape(-1) for v in residuals])
                    add(f"object/{slot}/nonpenetration_translation_target", scipy_least_squares_cost(vector, loss),
                        residual_count=len(vector), active_count=int(active.sum()), reduction=f"scipy_0.5_rho/{loss}")
                    activation[f"object/{slot}/nonpenetration_translation_target"] = "active_declared_soft_target"
            else:
                activation[f"object/{slot}/nonpenetration_translation_target"] = "inactive_no_declared_soft_target"

            ids = np.flatnonzero(valid)
            velocity_t: list[np.ndarray] = []
            velocity_r: list[np.ndarray] = []
            accel_t: list[np.ndarray] = []
            accel_r: list[np.ndarray] = []
            for a, b in zip(ids[:-1], ids[1:]):
                gap = max(1, int(episode.frame_idx[b]) - int(episode.frame_idx[a]))
                velocity_t.append(correction_velocity(trans[b], trans[a], gap,
                    float(self.profile.get("sigma_translation_delta_step_m"))))
                velocity_r.append(correction_velocity(rot[b], rot[a], gap,
                    float(self.profile.get("sigma_rotation_delta_step_rad"))))
            for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
                gap0 = max(1, int(episode.frame_idx[b]) - int(episode.frame_idx[a]))
                gap1 = max(1, int(episode.frame_idx[c]) - int(episode.frame_idx[b]))
                accel_t.append(correction_acceleration(trans[a], trans[b], trans[c], gap0, gap1,
                    float(self.profile.get("sigma_translation_delta_accel_m"))))
                accel_r.append(correction_acceleration(rot[a], rot[b], rot[c], gap0, gap1,
                    float(self.profile.get("sigma_rotation_delta_accel_rad"))))
            for name, blocks in (("translation_velocity", velocity_t), ("rotation_velocity", velocity_r),
                                 ("translation_acceleration", accel_t), ("rotation_acceleration", accel_r)):
                if blocks:
                    vector = np.concatenate([np.asarray(v).reshape(-1) for v in blocks])
                    add(f"object/{slot}/{name}", scipy_least_squares_cost(vector, loss),
                        residual_count=len(vector), active_count=len(blocks), reduction=f"scipy_0.5_rho/{loss}")
                else:
                    activation[f"object/{slot}/{name}"] = "inactive_insufficient_eligible_rows"

            if bool(self.profile.get("object_observed_terms")):
                self._object_observed_terms(state, end, slot, add, activation)
            else:
                activation[f"object/{slot}/observed_nn"] = "inactive_profile_switch_disabled"
            self._object_declared_image_terms(state, end, slot, entity, add, activation)

    @staticmethod
    def _broadcast_sigma(value: Any, end: int) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.full(end, float(array))
        if array.shape[0] < end:
            return None
        return array[:end].reshape(end)

    def _object_observed_terms(self, state: AnnotationState, end: int, slot: str, add: Any,
                               activation: dict[str, Any]) -> None:
        from rewards.terms.object.geometry import observed_point_to_plane, observed_point_to_point
        from utils.candidate_geometry import build_observed_point_factors
        from utils.data import PoseNode, SolverConfig
        from utils.geometry import inverse_pose

        episode, obj = state.episode, state.episode.objects[slot]
        nodes, poses = [], {}
        eligible = np.asarray(obj.get("metric_observation_eligible", obj.get("valid"))[:end], dtype=bool)
        for i in range(end):
            points = np.asarray(obj["observed_points_world"][i], dtype=np.float64)
            metric = bool(eligible[i] and len(points) >= 3)
            node = PoseNode(int(episode.frame_idx[i]), np.asarray(obj["rotation"][i]),
                            np.asarray(obj["translation"][i]), points if metric else None,
                            "source_visible_metric_surface", metric, 0.0)
            nodes.append(node)
            poses[node.frame_idx] = (state.object_rotation[slot][i], state.object_translation[slot][i])
        if sum(node.metric_observation for node in nodes) < 2:
            activation[f"object/{slot}/observed_nn"] = "inactive_fewer_than_two_metric_eligible_rows"
            return
        kernel = SolverConfig(max_points=int(self.profile.get("object_observed_max_points")),
                              min_pairs=int(self.profile.get("object_observed_min_pairs")),
                              max_correspondence_m=float(self.profile.get("object_observed_max_correspondence_m")),
                              seed=int(_cfg(self.cfg, "seed", 0)))
        factors, diagnostic = build_observed_point_factors(nodes, poses, nodes[0].frame_idx, kernel, end)
        loss = str(self.profile.get("object_observed_loss"))
        for factor in factors:
            source_node, target_node = nodes[factor.source_pos], nodes[factor.target_pos]
            delta = inverse_pose(factor.source_world, *poses[source_node.frame_idx]) - \
                    inverse_pose(factor.target_world, *poses[target_node.frame_idx])
            plane = observed_point_to_plane(delta, factor.target_normals, factor.weights,
                                            float(self.profile.get("object_observed_weight")))
            point = observed_point_to_point(delta, factor.weights,
                                            float(self.profile.get("object_observed_weight")))
            add(f"object/{slot}/observed_point_to_plane", scipy_least_squares_cost(plane, loss),
                residual_count=np.asarray(plane).size, active_count=len(factor.weights), reduction=f"scipy_0.5_rho/{loss}")
            add(f"object/{slot}/observed_point_to_point", scipy_least_squares_cost(point, loss),
                residual_count=np.asarray(point).size, active_count=len(factor.weights), reduction=f"scipy_0.5_rho/{loss}")
        activation[f"object/{slot}/observed_nn"] = diagnostic

    def _object_declared_image_terms(self, state: AnnotationState, end: int, slot: str, entity: str,
                                     add: Any, activation: dict[str, Any]) -> None:
        from rewards.terms.object.geometry import first_hit_depth
        from rewards.terms.object.image import rgb_reprojection_residual, silhouette_endpoint_residual
        from rewards.terms.object.pose import rgb_edge_residual

        episode, obj = state.episode, state.episode.objects[slot]
        loss = str(self.profile.get("formal_object_loss"))
        image_rows = _factor_rows(episode, {"object_image", "object_first_hit_silhouette",
                                           "first_hit_depth", "silhouette_endpoint"}, end, target=entity)
        if not image_rows:
            activation[f"object/{slot}/image_factors"] = "inactive_no_declared_image_evidence"
        for index, row in enumerate(image_rows):
            frame = _row_frame(row)
            pos = _frame_position(episode, frame) if frame is not None else None
            key = f"object/{slot}/image/factor={index}"
            if pos is None or pos >= end:
                activation[key] = "inactive_frame_outside_prefix"
                continue
            R, t = state.object_rotation[slot][pos], state.object_translation[slot][pos]
            row_state = str(row.get("state", ""))
            active_declared = bool(row.get("active") is True or row_state == "active" or row_state.startswith("active_"))
            depth_keys = ("depth_points_canonical", "depth_observed_z", "depth_weight")
            silhouette_keys = ("silhouette_points_canonical", "silhouette_target_uv")
            depth_present = [key in row and row[key] is not None for key in depth_keys]
            silhouette_present = [key in row and row[key] is not None for key in silhouette_keys]
            if any(depth_present) and not all(depth_present):
                raise ValueError(f"{key}: partial declared first-hit payload")
            if any(silhouette_present) and not all(silhouette_present):
                raise ValueError(f"{key}: partial declared silhouette payload")
            if active_declared and not (all(depth_present) or all(silhouette_present)):
                raise ValueError(f"{key}: active object image row requires first-hit or silhouette payload")
            if not any(depth_present) and not any(silhouette_present):
                activation[key] = str(row.get("state") or "inactive_declared_image_row_without_selected_subfamily")
                continue
            depth_points = (np.asarray(row["depth_points_canonical"], dtype=np.float64)
                            if all(depth_present) else np.empty((0, 3), dtype=np.float64))
            observed_z = (np.asarray(row["depth_observed_z"], dtype=np.float64)
                          if all(depth_present) else np.empty((0,), dtype=np.float64))
            depth_weight = (np.asarray(row["depth_weight"], dtype=np.float64)
                            if all(depth_present) else np.empty((0,), dtype=np.float64))
            if depth_points.size and (depth_points.ndim != 2 or depth_points.shape[1:] != (3,)):
                raise ValueError(f"{key}: malformed declared first-hit canonical points")
            depth_points = depth_points.reshape(-1, 3)
            if len(depth_points):
                if len(depth_points) != len(observed_z) or len(depth_points) != len(depth_weight):
                    raise ValueError(f"{key}: malformed declared first-hit factor")
                T = np.asarray(row.get("T_world_camera", episode.T_world_camera[pos]), dtype=np.float64)
                world = depth_points @ R.T + t
                camera = (world - T[:3, 3]) @ T[:3, :3]
                residual = first_hit_depth(camera[:, 2], observed_z, depth_weight,
                    sigma=float(self.profile.get("sigma_image_first_hit_m")),
                    clip=float(self.profile.get("max_image_first_hit_residual_m")),
                    weight=float(self.profile.get("image_first_hit_weight")))
                add(f"{key}/first_hit", scipy_least_squares_cost(residual, loss),
                    residual_count=len(residual), active_count=len(residual), reduction=f"scipy_0.5_rho/{loss}")
                activation[f"{key}/first_hit"] = "active_declared_first_hit"
            silhouette_points = (np.asarray(row["silhouette_points_canonical"], dtype=np.float64)
                                 if all(silhouette_present) else np.empty((0, 3), dtype=np.float64))
            silhouette_target = (np.asarray(row["silhouette_target_uv"], dtype=np.float64)
                                 if all(silhouette_present) else np.empty((0, 2), dtype=np.float64))
            if silhouette_points.size and (silhouette_points.ndim != 2 or silhouette_points.shape[1:] != (3,)):
                raise ValueError(f"{key}: malformed declared silhouette canonical points")
            if silhouette_target.size and (silhouette_target.ndim != 2 or silhouette_target.shape[1:] != (2,)):
                raise ValueError(f"{key}: malformed declared silhouette targets")
            silhouette_points = silhouette_points.reshape(-1, 3)
            silhouette_target = silhouette_target.reshape(-1, 2)
            if active_declared and not (len(depth_points) or len(silhouette_points)):
                raise ValueError(f"{key}: active object image row requires nonempty first-hit or silhouette payload")
            if not len(depth_points) and not len(silhouette_points):
                activation[key] = str(row.get("state") or "inactive_declared_image_row_without_selected_subfamily")
                continue
            if len(silhouette_points):
                if len(silhouette_points) != len(silhouette_target):
                    raise ValueError(f"{key}: malformed declared silhouette factor")
                T = np.asarray(row.get("T_world_camera", episode.T_world_camera[pos]), dtype=np.float64)
                K = np.asarray(row.get("K_raster", episode.intrinsics[pos]), dtype=np.float64)
                world = silhouette_points @ R.T + t
                camera = (world - T[:3, 3]) @ T[:3, :3]
                if K.shape == (3, 3):
                    intr = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                else:
                    intr = K.reshape(4)
                uv = np.column_stack([intr[0] * camera[:, 0] / np.maximum(camera[:, 2], 1e-9) + intr[2],
                                      intr[1] * camera[:, 1] / np.maximum(camera[:, 2], 1e-9) + intr[3]])
                residual = silhouette_endpoint_residual(uv, silhouette_target,
                    sigma=float(self.profile.get("sigma_image_silhouette_px")),
                    clip=float(self.profile.get("max_image_silhouette_residual_px")),
                    weight=float(self.profile.get("image_silhouette_weight")))
                add(f"{key}/silhouette", scipy_least_squares_cost(residual, loss),
                    residual_count=len(residual), active_count=len(silhouette_points), reduction=f"scipy_0.5_rho/{loss}")
                activation[f"{key}/silhouette"] = "active_declared_silhouette"

        edge_rows = _factor_rows(episode, {"object_rgb_edge", "rgb_rotation"}, end, target=entity)
        if not edge_rows:
            activation[f"object/{slot}/rgb_edges"] = "inactive_no_declared_rgb_edge_evidence"
        pose_by_frame = {int(episode.frame_idx[i]): (state.object_rotation[slot][i], state.object_translation[slot][i])
                         for i in range(end)}
        for index, row in enumerate(edge_rows):
            source_frame = int(row.get("source_frame_idx", -1)); target_frame = int(row.get("target_frame_idx", -1))
            key = f"object/{slot}/rgb_edge/factor={index}"
            if source_frame not in pose_by_frame or target_frame not in pose_by_frame:
                activation[key] = "inactive_edge_outside_prefix"
                continue
            required = ("rotation_source_to_target_world", "translation_source_to_target_world_m")
            if any(name not in row for name in required):
                raise ValueError(f"{key}: malformed declared RGB edge")
            edge = SimpleNamespace(source_frame_idx=source_frame, target_frame_idx=target_frame,
                rotation_source_to_target_world=np.asarray(row[required[0]], dtype=np.float64),
                translation_source_to_target_world_m=np.asarray(row[required[1]], dtype=np.float64),
                diagnostics=dict(row.get("diagnostics", {})),
                translation_reference_rotation=np.asarray(row["translation_reference_rotation"], dtype=np.float64)
                    if row.get("translation_reference_rotation") is not None else None,
                translation_source_support_world=np.asarray(row["translation_source_support_world"], dtype=np.float64)
                    if row.get("translation_source_support_world") is not None else None)
            dr, dt = rgb_edge_residual(edge, pose_by_frame)
            weight = float(row.get("weight", 1.0))
            rr = math.sqrt(weight * float(self.profile.get("rgb_rotation_weight"))) * dr / float(self.profile.get("rgb_rotation_rad"))
            add(f"{key}/rotation", scipy_least_squares_cost(rr, loss), residual_count=3,
                reduction=f"scipy_0.5_rho/{loss}")
            if not bool(row.get("rotation_only", False)):
                rt = math.sqrt(weight * float(self.profile.get("rgb_translation_weight"))) * dt / float(self.profile.get("rgb_translation_m"))
                add(f"{key}/translation", scipy_least_squares_cost(rt, loss), residual_count=3,
                    reduction=f"scipy_0.5_rho/{loss}")
            activation[key] = "active_declared_rgb_edge"

        reprojection_rows = _factor_rows(episode, {"object_rgb_reprojection", "rgb_reprojection"}, end, target=entity)
        if not reprojection_rows:
            activation[f"object/{slot}/rgb_reprojection"] = "inactive_no_declared_rgb_reprojection_evidence"
        for index, row in enumerate(reprojection_rows):
            sf, tf = int(row.get("source_frame_idx", -1)), int(row.get("target_frame_idx", -1))
            key = f"object/{slot}/rgb_reprojection/factor={index}"
            if sf not in pose_by_frame or tf not in pose_by_frame:
                activation[key] = "inactive_edge_outside_prefix"
                continue
            required = ("source_world", "target_uv", "weights", "target_camera_rotation",
                        "target_camera_translation", "target_intrinsics")
            if any(name not in row for name in required):
                raise ValueError(f"{key}: malformed declared RGB reprojection")
            source_pos, target_pos = _frame_position(episode, sf), _frame_position(episode, tf)
            nodes = [SimpleNamespace(frame_idx=int(frame), initial_rotation=np.asarray(obj["rotation"][i]),
                                     initial_translation=np.asarray(obj["translation"][i]))
                     for i, frame in enumerate(episode.frame_idx[:end])]
            factor = SimpleNamespace(source_pos=source_pos, target_pos=target_pos, source_frame_idx=sf,
                target_frame_idx=tf, source_world=np.asarray(row["source_world"], dtype=np.float64),
                target_uv=np.asarray(row["target_uv"], dtype=np.float64), weights=np.asarray(row["weights"], dtype=np.float64),
                target_camera_rotation=np.asarray(row["target_camera_rotation"], dtype=np.float64),
                target_camera_translation=np.asarray(row["target_camera_translation"], dtype=np.float64),
                target_intrinsics=np.asarray(row["target_intrinsics"], dtype=np.float64),
                edge_weight=float(row.get("edge_weight", 1.0)), rotation_eligible=bool(row.get("rotation_eligible", True)))
            config = SimpleNamespace(rgb_reprojection_weight_scale=float(self.profile.get("rgb_reprojection_weight")),
                sigma_rgb_reprojection_px=float(self.profile.get("sigma_rgb_reprojection_px")),
                max_rgb_reprojection_residual_px=float(self.profile.get("max_rgb_reprojection_residual_px")))
            residual = rgb_reprojection_residual(factor, pose_by_frame, nodes, config)
            add(key, scipy_least_squares_cost(residual, loss), residual_count=len(residual),
                active_count=len(factor.weights), reduction=f"scipy_0.5_rho/{loss}")
            activation[key] = "active_declared_rgb_reprojection"

    def _visual_joint_fixed_scale(self, state: AnnotationState, end: int, add: Any,
                      activation: dict[str, Any]) -> None:
        """Selected fixed-scale RL slice of the legacy visual composition.

        It deliberately has no hand-log-scale action or term; legacy_optimize
        retains the full visual solver objective including hand scale.
        """
        from rewards.terms.hand.observation import visual_joint_reprojection, visual_surface_reprojection
        from rewards.terms.hand.priors import visual_articulation, visual_root_orientation, visual_translation
        from rewards.terms.interaction.contact import unsigned_pad_loss
        from rewards.terms.motion.temporal import parameter_acceleration, parameter_velocity
        from utils.mano import TIP_VERTICES

        episode = state.episode
        frames = np.asarray(episode.frame_idx[:end])
        for side in episode.layout.hand_slots:
            source = episode.hands[side]
            valid = np.asarray(source["valid"][:end], dtype=bool)
            trans_world = np.asarray(state.hand_translation[side][:end] - source["translation"][:end])
            trans_camera = np.stack([trans_world[i] @ episode.T_world_camera[i, :3, :3] for i in range(end)])
            root = np.stack([_rotation_vector(_axis_angle_matrix(state.hand_root[side][i]) @
                                              _axis_angle_matrix(source["root_orient"][i]).T) for i in range(end)])
            pose = np.stack([np.stack([_rotation_vector(_axis_angle_matrix(state.hand_articulation[side][i].reshape(15, 3)[j]) @
                                                   _axis_angle_matrix(source["articulation"][i].reshape(15, 3)[j]).T)
                                       for j in range(15)]) for i in range(end)])
            ids = np.flatnonzero(valid)
            if len(ids):
                add(f"visual/hand/{side}/articulation_prior", visual_articulation(_tensor(pose[ids], self.device)))
                add(f"visual/hand/{side}/root_prior", visual_root_orientation(_tensor(root[ids], self.device)))
                add(f"visual/hand/{side}/translation_prior", visual_translation(_tensor(trans_camera[ids], self.device)))
            for i in ids:
                frame = int(episode.frame_idx[i])
                base_uv, _ = _project(source["joints"][i], episode.T_world_camera[i], episode.intrinsics[i])
                uv, _ = _project(state.hand_joints[side][i], episode.T_world_camera[i], episode.intrinsics[i])
                add(f"visual/hand/{side}/joint_reprojection/frame={frame}",
                    float(self.profile.get("visual_image_weight")) * visual_joint_reprojection(
                        _tensor(uv, self.device), _tensor(base_uv, self.device)))
                sample = np.asarray(source.get("visual_surface_vertex_ids",
                    np.unique(np.r_[np.arange(0, 778, 4), TIP_VERTICES])), dtype=np.int64)
                base_pix, _ = _project(source["vertices"][i, sample], episode.T_world_camera[i], episode.intrinsics[i])
                pix, _ = _project(state.hand_vertices[side][i, sample], episode.T_world_camera[i], episode.intrinsics[i])
                add(f"visual/hand/{side}/surface_reprojection/frame={frame}",
                    float(self.profile.get("visual_image_weight")) * visual_surface_reprojection(
                        _tensor(pix, self.device), _tensor(base_pix, self.device)))
            for name, value, sigma in (("articulation", pose, .05), ("root", root, .035),
                                       ("translation", trans_camera, .006)):
                normalized = _tensor(value / sigma, self.device)
                velocity, _, den1, n1 = aggregate_temporal(normalized, valid, frames,
                    float(self.profile.get("visual_temporal_weight")), parameter_velocity, 1)
                acceleration, _, den2, n2 = aggregate_temporal(normalized, valid, frames,
                    float(self.profile.get("visual_temporal_weight")), parameter_acceleration, 2)
                add(f"visual/hand/{side}/{name}_velocity", velocity, denominator=den1, active_count=n1,
                    reduction="mean_over_all_valid_edges")
                add(f"visual/hand/{side}/{name}_acceleration", acceleration, denominator=den2, active_count=n2,
                    reduction="mean_over_all_valid_triples")

        # The visual fitter uses a distinct camera-basis object prior and a
        # single six-DoF temporal scale. These replace, rather than augment,
        # formal object sigma priors in this profile.
        for slot in episode.layout.object_slots:
            obj = episode.objects[slot]
            valid = np.asarray(obj.get("valid", obj.get("present"))[:end], dtype=bool)
            rot = np.stack([_rotation_vector(state.object_rotation[slot][i] @ np.asarray(obj["rotation"][i]).T)
                            for i in range(end)])
            trans_world = np.asarray(state.object_translation[slot][:end] - obj["translation"][:end])
            trans_camera = np.stack([trans_world[i] @ episode.T_world_camera[i, :3, :3] for i in range(end)])
            ids = np.flatnonzero(valid)
            if len(ids):
                rot_t = _tensor(rot[ids], self.device)
                trans_t = _tensor(trans_camera[ids], self.device)
                add(f"visual/object/{slot}/rotation_prior", rot_t.square().mean() / 0.05 ** 2)
                add(f"visual/object/{slot}/translation_xy_prior", trans_t[:, :2].square().mean() / 0.006 ** 2)
                add(f"visual/object/{slot}/translation_z_prior", trans_t[:, 2].square().mean() / 0.025 ** 2)
            six = _tensor(np.concatenate([rot, trans_camera], axis=1) / 0.008, self.device)
            velocity, _, den1, n1 = aggregate_temporal(six, valid, frames,
                float(self.profile.get("visual_temporal_weight")), parameter_velocity, 1)
            acceleration, _, den2, n2 = aggregate_temporal(six, valid, frames,
                float(self.profile.get("visual_temporal_weight")), parameter_acceleration, 2)
            add(f"visual/object/{slot}/se3_velocity", velocity, denominator=den1, active_count=n1,
                reduction="mean_over_all_valid_edges")
            add(f"visual/object/{slot}/se3_acceleration", acceleration, denominator=den2, active_count=n2,
                reduction="mean_over_all_valid_triples")

        boundary_rows = _factor_rows(episode, {"visual_silhouette_endpoint"}, end)
        if not boundary_rows:
            activation["visual/silhouette"] = "inactive_no_declared_visual_silhouette_evidence"
        for index, row in enumerate(boundary_rows):
            frame = _row_frame(row)
            pos = _frame_position(episode, frame) if frame is not None else None
            target = str(row.get("target_entity_id") or "")
            slot = next((name for name in episode.layout.object_slots
                         if _object_entity(episode.objects[name], name) == target), None)
            key = f"visual/silhouette/factor={index}"
            points = np.asarray(row.get("points_canonical", []), dtype=np.float64).reshape(-1, 3)
            targets = np.asarray(row.get("target_uv", []), dtype=np.float64).reshape(-1, 2)
            if pos is None or pos >= end or slot is None:
                activation[key] = "inactive_declared_entity_or_frame_unavailable"
                continue
            if len(points) != len(targets):
                raise ValueError(f"{key}: malformed declared visual silhouette factor")
            if not len(points):
                activation[key] = "inactive_empty_declared_visual_silhouette_factor"
                continue
            R, t = state.object_rotation[slot][pos], state.object_translation[slot][pos]
            uv, _ = _project(points @ R.T + t, episode.T_world_camera[pos], episode.intrinsics[pos])
            difference = _tensor((uv - targets) / 5.0, self.device)
            add(key, float(self.profile.get("visual_boundary_weight")) * difference.square().mean() / max(end, 1),
                active_count=len(points), reduction="legacy_boundary_row_mean_divided_by_B")
            activation[key] = "active_declared_visual_silhouette"

        rows = _factor_rows(episode, {"visual_unsigned_pad"}, end)
        if not rows:
            activation["visual/unsigned_pad"] = "inactive_selected_visual_profile_missing_declared_pad_evidence"
        B = max(end, 1)
        for index, row in enumerate(rows):
            frame, side, target = _row_frame(row), row.get("hand_side"), row.get("target_entity_id")
            pos = _frame_position(episode, frame) if frame is not None else None
            slot = next((name for name in episode.layout.object_slots
                         if _object_entity(episode.objects[name], name) == str(target)), None)
            key = f"visual/unsigned_pad/factor={index}"
            ids = np.asarray(row.get("vertex_ids", []), dtype=np.int64)
            targets = np.asarray(row.get("targets_object", []), dtype=np.float64).reshape(-1, 3)
            if pos is None or pos >= end:
                activation[key] = "inactive_frame_outside_prefix"
                continue
            if side not in episode.layout.hand_slots or slot is None:
                activation[key] = "inactive_declared_entity_unavailable"
                continue
            if not (len(ids) == len(targets)):
                raise ValueError(f"{key}: malformed declared visual pad factor")
            if len(ids) < 3:
                activation[key] = "inactive_fewer_than_three_declared_pad_vertices"
                continue
            R, t = state.object_rotation[slot][pos], state.object_translation[slot][pos]
            target_world = targets @ R.T + t
            delta = state.hand_vertices[side][pos, ids] - target_world
            value = unsigned_pad_loss(_tensor(delta, self.device), float(row.get("weight", 0.0))) / (B * 2)
            add(key, float(self.profile.get("visual_unsigned_pad_weight")) * value,
                active_count=len(ids), reduction="legacy_three_nearest_divided_by_B_times_2")
            activation[key] = "active_declared_visual_unsigned_pad"

    def transition(self, state: AnnotationState, end_position: int) -> tuple[float, FactorLedger]:
        ledger = self.objective(state, end_position)
        reward = -ledger.reward_scale * (ledger.objective - float(state.previous_objective))
        state.previous_objective = ledger.objective
        state.ledger = ledger
        return float(reward), ledger
