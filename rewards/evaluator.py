from __future__ import annotations

"""Exact-prefix objective accounting over shared legacy factor kernels.

A transition reward is ``-(J(prefix_t) - J(prefix_{t-1}))``. Therefore the
undiscounted return is ``-J(final)`` from the documented empty-prefix baseline
``J(empty)=0``. Re-evaluating the prefix preserves interval denominators;
summing frame-local means would not.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping
import math
import numpy as np

from envs.state import AnnotationState
from envs.dynamics import refresh_candidate_terms


@dataclass(frozen=True)
class FactorLedger:
    end_position: int
    terms: Mapping[str, float]
    statistics: Mapping[str, Mapping[str, float | int | None]] = field(default_factory=dict)
    activation: Mapping[str, Any] = field(default_factory=dict)
    baseline_constant: float = 0.0

    @property
    def objective(self) -> float:
        return float(self.baseline_constant + sum(self.terms.values()))

    def document(self) -> dict[str, Any]:
        return {
            "end_position": self.end_position,
            "terms": dict(self.terms),
            "statistics": {key: dict(value) for key, value in self.statistics.items()},
            "activation": dict(self.activation),
            "baseline_constant": self.baseline_constant,
            "objective": self.objective,
        }


def _cfg(cfg: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _tensor(value: Any, device: str):
    import torch
    return torch.as_tensor(value, dtype=torch.float32, device=device)


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
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= max(2.0 * math.sin(angle), 1e-12)
    return axis * angle


def _project(points: np.ndarray, T_world_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = (np.asarray(points) - T_world_camera[:3, 3][None, :]) @ T_world_camera[:3, :3]
    z = np.maximum(camera[:, 2], 1e-6)
    fx, fy, cx, cy = intrinsics
    uv = np.column_stack([fx * camera[:, 0] / z + cx, fy * camera[:, 1] / z + cy])
    return uv, camera[:, 2]


class RewardEvaluator:
    def __init__(self, cfg: Mapping[str, Any] | Any, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device

    def objective(self, state: AnnotationState, end_position: int) -> FactorLedger:
        import torch
        from factors.hand.priors import formal_translation, formal_root_orientation, formal_articulation
        from factors.hand.observation import joint_pixel_deadband, joint_depth_deadband
        from factors.motion.temporal import parameter_velocity, parameter_acceleration
        from factors.object.pose import object_translation_prior, source_pose_rotation
        from factors.common import magnitude_hinge
        from factors.interaction.contact import contact_state_prior, normal_deadband
        from factors.interaction.physics import signed_active_penetration, signed_dense_penetration

        end = int(end_position)
        if end < 0 or end > state.episode.frame_count:
            raise ValueError("invalid objective prefix")
        if end == 0:
            return FactorLedger(0, {}, {}, {"prefix": "empty"})
        terms: dict[str, float] = {}
        statistics: dict[str, dict[str, float | int | None]] = {}
        activation: dict[str, Any] = {}
        episode = state.episode
        smooth = float(_cfg(self.cfg, "smooth_weight", 0.06))
        accel = float(_cfg(self.cfg, "accel_weight", 0.06))

        def add(name: str, value: Any, *, numerator: float | None = None,
                denominator: float | None = None, active_count: int | None = None,
                positive_count: int | None = None) -> None:
            number = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
            if not np.isfinite(number):
                raise FloatingPointError(f"nonfinite factor {name}: {number}")
            terms[name] = terms.get(name, 0.0) + number
            if numerator is not None or denominator is not None or active_count is not None or positive_count is not None:
                statistics[name] = {"numerator": numerator, "denominator": denominator,
                                    "active_count": active_count, "positive_count": positive_count}

        for side in episode.layout.hand_slots:
            values = episode.hands[side]
            valid_np = np.asarray(values["valid"][:end], dtype=np.float32)
            visibility = _tensor(valid_np, self.device)
            trans_delta = _tensor(state.hand_translation[side][:end] - values["translation"][:end], self.device)
            root_delta_np = np.stack([
                _rotation_vector(_axis_angle_matrix(state.hand_root[side][i]) @
                                 _axis_angle_matrix(values["root_orient"][i]).T)
                for i in range(end)
            ])
            pose_delta_np = np.stack([np.stack([
                _rotation_vector(_axis_angle_matrix(state.hand_articulation[side][i].reshape(15, 3)[j]) @
                                 _axis_angle_matrix(values["articulation"][i].reshape(15, 3)[j]).T)
                for j in range(15)]) for i in range(end)])
            root_delta = _tensor(root_delta_np, self.device).reshape(end, 1, 3)
            pose_delta = _tensor(pose_delta_np, self.device).reshape(end, 15, 3)
            pose_visibility = visibility[:, None].expand(-1, 15)
            translation_weight = float(_cfg(self.cfg, "translation_prior_weight", 1.0))
            translation_den = max(3.0 * float(valid_np.sum()), 1.0)
            translation_num = translation_weight * float(np.sum(np.square(trans_delta.detach().cpu().numpy()) * valid_np[:, None]))
            add(f"hand/{side}/translation_prior", formal_translation(trans_delta, visibility, translation_weight),
                numerator=translation_num, denominator=translation_den, active_count=int(valid_np.sum()))
            root_weight = float(_cfg(self.cfg, "root_prior_weight", 1.0))
            root_den = max(9.0 * float(valid_np.sum()), 1.0)
            root_num = root_weight * float(np.sum(np.square(root_delta.detach().cpu().numpy()) * valid_np[:, None, None]))
            add(f"hand/{side}/root_prior", formal_root_orientation(root_delta, visibility, root_weight),
                numerator=root_num, denominator=root_den, active_count=int(valid_np.sum()))
            pose_weight = float(_cfg(self.cfg, "pose_prior_weight", 1.0))
            pose_visibility_np = np.broadcast_to(valid_np[:, None], (end, 15))
            pose_den = max(3.0 * float(pose_visibility_np.sum()), 1.0)
            pose_num = pose_weight * float(np.sum(np.square(pose_delta.detach().cpu().numpy()) * pose_visibility_np[..., None]))
            add(f"hand/{side}/articulation_prior", formal_articulation(pose_delta, pose_visibility, pose_weight),
                numerator=pose_num, denominator=pose_den, active_count=int(pose_visibility_np.sum()))
            valid_indices = np.flatnonzero(valid_np > 0)
            segments = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1) if len(valid_indices) else []
            for name, value in (("translation", trans_delta), ("root", root_delta), ("articulation", pose_delta)):
                for segment in segments:
                    if not len(segment):
                        continue
                    selected = value[torch.as_tensor(segment, dtype=torch.long, device=self.device)]
                    add(f"hand/{side}/{name}_velocity", parameter_velocity(selected, smooth))
                    add(f"hand/{side}/{name}_acceleration", parameter_acceleration(selected, accel))
            add(f"hand/{side}/translation_soft_bound", magnitude_hinge(trans_delta,
                float(_cfg(self.cfg, "max_translation_m", 0.025)),
                float(_cfg(self.cfg, "bound_hinge_weight", 10.0))))
            add(f"hand/{side}/root_soft_bound", magnitude_hinge(root_delta,
                float(_cfg(self.cfg, "max_root_delta_rad", 0.15)),
                float(_cfg(self.cfg, "bound_hinge_weight", 10.0))))
            add(f"hand/{side}/articulation_soft_bound", magnitude_hinge(pose_delta,
                float(_cfg(self.cfg, "max_pose_delta_rad", 0.35)),
                float(_cfg(self.cfg, "bound_hinge_weight", 10.0))))
            for i in np.flatnonzero(valid_np):
                base_uv, base_depth = _project(values["joints"][i], episode.T_world_camera[i], episode.intrinsics[i])
                uv, depth = _project(state.hand_joints[side][i], episode.T_world_camera[i], episode.intrinsics[i])
                weights = torch.ones(21, dtype=torch.float32, device=self.device)
                add(f"hand/{side}/joint_pixel_deadband", joint_pixel_deadband(
                    _tensor(uv, self.device), _tensor(base_uv, self.device), weights,
                    float(_cfg(self.cfg, "visible_shift_limit_px", 12.0)),
                    float(_cfg(self.cfg, "visible_hinge_weight", 1.0))))
                add(f"hand/{side}/joint_depth_deadband", joint_depth_deadband(
                    _tensor(depth, self.device), _tensor(base_depth, self.device), weights,
                    float(_cfg(self.cfg, "depth_shift_limit_m", 0.035)),
                    float(_cfg(self.cfg, "depth_hinge_weight", 1.0))))
                evidence = episode.contact.get((int(episode.frame_idx[i]), side), {})
                depth_ids = np.asarray(evidence.get("depth_order_vertex_ids", []), dtype=np.int64)
                surface_depth = np.asarray(evidence.get("depth_order_surface_depth_m", []), dtype=np.float64)
                if len(depth_ids):
                    margin = float(_cfg(self.cfg, "visible_surface_depth_order_margin_m", 0.01))
                    base_vertex_depth = _project(values["vertices"][i, depth_ids], episode.T_world_camera[i], episode.intrinsics[i])[1]
                    violation = np.maximum(0.0, surface_depth - margin - base_vertex_depth)
                    order = np.argsort(violation)[::-1]
                    cap = int(_cfg(self.cfg, "max_visible_surface_depth_vertices", 256))
                    if cap and len(order) > cap:
                        order = order[:cap]
                    selected_ids = depth_ids[order]
                    selected_surface = surface_depth[order]
                    candidate_depth = _project(state.hand_vertices[side][i, selected_ids], episode.T_world_camera[i], episode.intrinsics[i])[1]
                    residual = np.maximum(0.0, selected_surface - margin - candidate_depth)
                    from factors.interaction.physics import visible_surface_depth_order
                    term = visible_surface_depth_order(_tensor(candidate_depth, self.device),
                        _tensor(selected_surface, self.device), margin,
                        float(_cfg(self.cfg, "visible_surface_depth_order_weight", 20000.0)))
                    positive = int(np.count_nonzero(residual > 0.0))
                    numerator = float(_cfg(self.cfg, "visible_surface_depth_order_weight", 20000.0)) * float(np.square(residual).sum())
                    add(f"hand/{side}/visible_surface_depth_order", term, numerator=numerator,
                        denominator=float(max(positive, 1)), active_count=int(len(selected_ids)), positive_count=positive)
                    activation[f"depth_order/{side}/{i}"] = evidence.get("depth_order_state")
                else:
                    activation[f"depth_order/{side}/{i}"] = evidence.get("depth_order_state", "inactive_missing_independent_first_surface")

        for slot_name in episode.layout.object_slots:
            obj = episode.objects[slot_name]
            valid = np.asarray(obj["valid"][:end], dtype=bool)
            if not valid.any():
                activation[f"object/{slot_name}"] = "inactive_absent_slot"
                continue
            translation_delta_np = state.object_translation[slot_name][:end] - np.asarray(obj["translation"][:end])
            translation_delta = _tensor(translation_delta_np, self.device)
            rotation_delta_np = np.stack([
                _rotation_vector(state.object_rotation[slot_name][i] @ np.asarray(obj["rotation"][i]).T)
                for i in range(end)
            ])
            rotation_delta = _tensor(rotation_delta_np, self.device)
            add(f"object/{slot_name}/translation_prior", object_translation_prior(translation_delta,
                float(_cfg(self.cfg, "object_translation_prior_weight", 1.0))))
            rotation_sigma = float(_cfg(self.cfg, "object_rotation_prior_rad", 0.08))
            rotation_residual = source_pose_rotation(rotation_delta, rotation_sigma)
            add(f"object/{slot_name}/rotation_prior", rotation_residual.square().mean())
            object_indices = np.flatnonzero(valid)
            object_segments = np.split(object_indices, np.flatnonzero(np.diff(object_indices) > 1) + 1) if len(object_indices) else []
            for name, value in (("translation", translation_delta), ("rotation", rotation_delta)):
                for segment in object_segments:
                    if not len(segment):
                        continue
                    selected = value[torch.as_tensor(segment, dtype=torch.long, device=self.device)]
                    add(f"object/{slot_name}/{name}_velocity", parameter_velocity(selected,
                        float(_cfg(self.cfg, "object_smooth_weight", smooth))))
                    add(f"object/{slot_name}/{name}_acceleration", parameter_acceleration(selected, accel))
            add(f"object/{slot_name}/translation_soft_bound", magnitude_hinge(translation_delta,
                float(_cfg(self.cfg, "max_object_translation_m", 0.05)),
                float(_cfg(self.cfg, "bound_hinge_weight", 10.0))))
            add(f"object/{slot_name}/rotation_soft_bound", magnitude_hinge(rotation_delta,
                float(_cfg(self.cfg, "max_object_rotation_rad", 0.4)),
                float(_cfg(self.cfg, "bound_hinge_weight", 10.0))))
            self._object_observation_terms(state, end, slot_name, add, activation)

        # Contact state priors preserve row-specific activation and weights.
        for side_index, side in enumerate(episode.layout.hand_slots):
            active = []
            priors = []
            strengths = []
            positions = []
            for i in range(end):
                row = episode.contact.get((int(episode.frame_idx[i]), side), {})
                enabled = bool(row.get("state") == "active_contact_patch" and episode.hands[side]["valid"][i])
                active.append(enabled); priors.append(float(row.get("prior", 0.0)))
                strengths.append(float(row.get("weight", 0.0)) * float(_cfg(self.cfg, "contact_prior_scale_m", 0.004)) ** 2)
                positions.append(i)
            active_t = torch.as_tensor(active, dtype=torch.bool, device=self.device)
            if torch.any(active_t) and episode.layout.contact_variables_per_hand:
                probability = torch.sigmoid(_tensor(state.contact_logits[:end, side_index], self.device)).mean(dim=-1)
                prior_value = contact_state_prior(probability,
                    _tensor(priors, self.device), _tensor(strengths, self.device), active_t)
                prior_num = float(np.sum(np.asarray(strengths)[np.asarray(active)] *
                                         np.square(probability.detach().cpu().numpy()[np.asarray(active)] - np.asarray(priors)[np.asarray(active)])))
                prior_den = float(max(int(np.count_nonzero(active)), 1))
                add(f"contact/{side}/state_prior", prior_value, numerator=prior_num,
                    denominator=prior_den, active_count=int(np.count_nonzero(active)))
            for i, enabled in enumerate(active):
                if not enabled:
                    continue
                refreshed = state.term_cache.get(i)
                if refreshed is None:
                    refreshed = refresh_candidate_terms(state, self.cfg, i)
                    state.term_cache[i] = refreshed
                for object_slot, signed_trace in refreshed.get("signed", {}).items():
                    activation[f"signed/{object_slot}/{i}"] = signed_trace.get("state")
                contact = refreshed["contact"].get(side)
                if not contact or not len(contact["vertex_ids"]):
                    activation[f"contact/{side}/{i}"] = (contact or {}).get("state", "inactive_missing_geometry")
                    continue
                ids = contact["vertex_ids"]
                gap = np.sum((state.hand_vertices[side][i, ids] - contact["targets_world"]) * contact["normals_world"], axis=1)
                if episode.layout.contact_variables_per_hand:
                    probability = torch.sigmoid(_tensor(state.contact_logits[i, side_index], self.device)).mean()
                else:
                    probability = _tensor(priors[i], self.device)
                add(f"contact/{side}/normal_deadband", normal_deadband(
                    _tensor(gap, self.device),
                    float(contact["margin_m"] + contact["support_uncertainty_m"]),
                    float(contact["weight"]), probability))
                activation[f"contact/{side}/{i}"] = contact["state"]

        # Candidate-refreshed signed active and dense constraints from declared authority only.
        for i in range(end):
            refreshed = state.term_cache.get(i)
            if refreshed is None:
                refreshed = refresh_candidate_terms(state, self.cfg, i)
                state.term_cache[i] = refreshed
            for object_slot, signed_trace in refreshed.get("signed", {}).items():
                activation[f"signed/{object_slot}/{i}"] = signed_trace.get("state")
                if not str(signed_trace.get("state", "")).startswith("active_"):
                    continue
                for side, row in signed_trace.get("hands", {}).items():
                    for family in ("active", "dense"):
                        ids = np.asarray(row[f"{family}_ids"], dtype=np.int64)
                        if not len(ids):
                            continue
                        moved_np = state.hand_vertices[side][i, ids] - episode.hands[side]["vertices"][i, ids]
                        moved = _tensor(moved_np, self.device)
                        normals = _tensor(row[f"{family}_normals"], self.device)
                        depth = _tensor(row[f"{family}_depths"], self.device)
                        uncertainty = float(row["uncertainty_m"])
                        residual = np.maximum(0.0, np.asarray(row[f"{family}_depths"]) - uncertainty -
                                              np.sum(np.asarray(row[f"{family}_normals"]) * moved_np, axis=1))
                        if family == "active":
                            weight = float(_cfg(self.cfg, "observed_penetration_weight", 1.0))
                            term = signed_active_penetration(moved, normals, depth, uncertainty, weight)
                            denominator = float(max(len(residual), 1))
                        else:
                            weight = float(_cfg(self.cfg, "dense_observed_penetration_weight", 1.0))
                            term = signed_dense_penetration(moved, normals, depth, uncertainty, weight)
                            denominator = float(max(int(np.count_nonzero(residual > 0.0)), 1))
                        add(f"signed/{object_slot}/{side}/{family}", term,
                            numerator=weight * float(np.square(residual).sum()), denominator=denominator,
                            active_count=int(len(residual)), positive_count=int(np.count_nonzero(residual > 0.0)))

        # Optional explicit signed constraints used by authoritative bundle exporters.
        signed = _cfg(self.cfg, "signed_constraints", {})
        for side in episode.layout.hand_slots:
            for i in range(end):
                row = signed.get((int(episode.frame_idx[i]), side)) if isinstance(signed, Mapping) else None
                if not row:
                    continue
                if not bool(row.get("authority", False)):
                    activation[f"signed/{side}/{i}"] = "inactive_missing_local_face_authority"
                    continue
                ids = np.asarray(row["vertex_ids"], dtype=np.int64)
                moved = _tensor(state.hand_vertices[side][i, ids] - episode.hands[side]["vertices"][i, ids], self.device)
                normals = _tensor(row["normals_world"], self.device)
                depth = _tensor(row["depth_m"], self.device)
                add(f"signed/{side}/active", signed_active_penetration(moved, normals, depth,
                    float(row.get("uncertainty_m", 0.0)), float(row.get("active_weight", 1.0))))
                if bool(row.get("dense", False)):
                    add(f"signed/{side}/dense", signed_dense_penetration(moved, normals, depth,
                        float(row.get("uncertainty_m", 0.0)), float(row.get("dense_weight", 1.0))))
                activation[f"signed/{side}/{i}"] = "active_declared_authority"
        return FactorLedger(end, terms, statistics, activation)

    def _object_observation_terms(self, state: AnnotationState, end: int, slot_name: str,
                                  add: Any, activation: dict[str, Any]) -> None:
        from utils.data import PoseNode, SolverConfig
        from optimization.common.correspondences import build_observed_point_factors
        from factors.object.geometry import observed_point_to_plane, observed_point_to_point
        from utils.geometry import inverse_pose
        from factors.common import robust_cost
        obj = state.episode.objects[slot_name]
        nodes = []
        poses = {}
        for i in range(end):
            points = np.asarray(obj["observed_points_world"][i], dtype=np.float64)
            metric = bool(obj["valid"][i] and len(points) >= 3)
            node = PoseNode(int(state.episode.frame_idx[i]), np.asarray(obj["rotation"][i]),
                            np.asarray(obj["translation"][i]), points if metric else None,
                            "source_visible_metric_surface", metric, 0.0)
            nodes.append(node)
            poses[node.frame_idx] = (state.object_rotation[slot_name][i], state.object_translation[slot_name][i])
        if sum(node.metric_observation for node in nodes) < 2:
            activation[f"object/{slot_name}/observed_nn"] = "inactive_fewer_than_two_metric_rows"
            return
        kernel = SolverConfig(max_points=int(_cfg(self.cfg, "object_observed_max_points", 250)),
                              min_pairs=int(_cfg(self.cfg, "object_observed_min_pairs", 3)),
                              max_correspondence_m=float(_cfg(self.cfg, "object_observed_max_correspondence_m", 0.08)),
                              seed=int(_cfg(self.cfg, "seed", 0)))
        factors, diagnostic = build_observed_point_factors(nodes, poses, nodes[0].frame_idx, kernel, end)
        for factor in factors:
            source_node, target_node = nodes[factor.source_pos], nodes[factor.target_pos]
            source_pose = poses[source_node.frame_idx]
            target_pose = poses[target_node.frame_idx]
            delta = inverse_pose(factor.source_world, *source_pose) - inverse_pose(factor.target_world, *target_pose)
            plane = observed_point_to_plane(delta, factor.target_normals, factor.weights,
                                            float(_cfg(self.cfg, "object_observed_weight", 1.0)))
            point = observed_point_to_point(delta, factor.weights,
                                            float(_cfg(self.cfg, "object_observed_weight", 1.0)))
            loss_kind = str(_cfg(self.cfg, "object_observed_loss", "soft_l1"))
            add(f"object/{slot_name}/observed_point_to_plane", robust_cost(np.asarray(plane), loss_kind),
                denominator=float(max(1, len(factor.weights))), active_count=int(len(factor.weights)))
            add(f"object/{slot_name}/observed_point_to_point", robust_cost(np.asarray(point), loss_kind),
                denominator=float(max(1, len(factor.weights))), active_count=int(len(factor.weights)))
        activation[f"object/{slot_name}/observed_nn"] = diagnostic

    def transition(self, state: AnnotationState, end_position: int) -> tuple[float, FactorLedger]:
        ledger = self.objective(state, end_position)
        reward = -(ledger.objective - float(state.previous_objective))
        state.previous_objective = ledger.objective
        state.ledger = ledger
        return float(reward), ledger
