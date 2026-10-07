from __future__ import annotations

"""Fixed-width causal policy observations with source-state/evidence features."""

from dataclasses import dataclass
import numpy as np

from envs.state import AnnotationState, EpisodeStatic


HISTORY_FEATURE_SIZE = 12
OBSERVATION_LAYOUT_VERSION = "ego.annotation.observation_layout.v2"


@dataclass(frozen=True)
class ObservationLayout:
    action_size: int
    hand_slots: int = 2
    object_slots: int = 1
    contact_variables_per_hand: int = 1
    point_samples: int = 8
    history_frames: int = 0
    include_hand_scale: bool = False
    include_object_shared_scale: bool = False

    @property
    def version(self) -> str:
        return OBSERVATION_LAYOUT_VERSION

    @property
    def size(self) -> int:
        # time/camera + source hand state/qualification/candidate residual +
        # source object SE3, P09 summaries/samples/qualification + contact/action history.
        hand = self.hand_slots * (1 + 3 + 45 + 3 + 3 + 45 + 3 + 1 + 21 + 15 + 3 + 2
                                  + (1 if self.include_hand_scale else 0))
        obj = self.object_slots * (3 + 3 + 3 + 3 + 3 + self.point_samples * 3
                                   + (1 if self.include_object_shared_scale else 0))
        contact = self.hand_slots * self.contact_variables_per_hand
        history = max(0, int(self.history_frames)) * HISTORY_FEATURE_SIZE
        return 3 + 20 + hand + obj + contact + self.action_size + history


def observation_layout(episode: EpisodeStatic) -> ObservationLayout:
    return ObservationLayout(
        episode.layout.size,
        len(episode.layout.hand_slots),
        len(episode.layout.object_slots),
        episode.layout.contact_variables_per_hand,
        int(episode.provenance.get("observation_point_samples", 8)),
        int(episode.provenance.get("causal_history_frames", 0)),
        bool(episode.layout.include_hand_scale),
        bool(episode.layout.include_object_shared_scale),
    )


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    axis = np.asarray([matrix[2,1]-matrix[1,2], matrix[0,2]-matrix[2,0], matrix[1,0]-matrix[0,1]], float)
    return axis / 2 if angle < 1e-8 else axis / max(2 * np.sin(angle), 1e-12) * angle


def _fixed_points(points: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros((count, 3), np.float32)
    points = np.asarray(points, np.float32)
    if len(points):
        ids = np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int)
        result[:len(ids)] = points[ids]
    return result


def _contact_summary(episode: EpisodeStatic, frame: int, side: str) -> tuple[float, float]:
    rows = [row for (row_frame, row_side, _), row in episode.contacts.items() if row_frame == frame and row_side == side]
    if not rows:
        return 0.0, 0.0
    return (float(np.mean([row.get("prior", 0.0) for row in rows])),
            float(np.mean([row.get("support_uncertainty_m", 0.0) for row in rows])))


def history_row(state: AnnotationState, reward: float, ledger: object, position: int) -> np.ndarray:
    """Build a fixed-size causal summary after the current safe transition.

    The row contains no future-frame values.  It is intentionally a compact
    memory interface rather than a copy of the factor ledger, so changing the
    number of reward terms cannot silently change the policy observation size.
    """
    trace = state.last_gate_trace
    if trace is None:
        safe = np.zeros(state.episode.layout.size, dtype=np.float32)
        allowed_fraction = 0.0
        projected_fraction = 0.0
    else:
        safe = np.asarray(trace.normalized_executed_action, dtype=np.float32)
        allowed_fraction = float(np.mean(trace.allowed_mask))
        projected = np.asarray(trace.metadata.get("projected", []), dtype=bool)
        projected_fraction = float(np.mean(projected)) if projected.shape == safe.shape else 0.0
    i = max(0, min(int(position), state.episode.frame_count - 1))
    hand_valid = [bool(state.episode.hands[side]["valid"][i]) for side in state.episode.layout.hand_slots]
    object_valid = [bool(state.episode.objects[slot]["pose_valid"][i]) for slot in state.episode.layout.object_slots]
    metric_valid = [bool(state.episode.objects[slot]["metric_observation_eligible"][i])
                    for slot in state.episode.layout.object_slots]
    contact_rows = [row for (row_frame, _row_side, _target), row in state.episode.contacts.items()
                    if row_frame == int(state.episode.frame_idx[i])]
    contact_active = [str(row.get("state")) == "active_contact_patch" for row in contact_rows]
    hand_residuals = []
    for side in state.episode.layout.hand_slots:
        source = state.episode.hands[side]
        if hand_valid[state.episode.layout.hand_slots.index(side)]:
            hand_residuals.extend((state.hand_root[side][i] - source["root_orient"][i]).reshape(-1).tolist())
            hand_residuals.extend((state.hand_articulation[side][i] - source["articulation"][i]).reshape(-1).tolist())
            hand_residuals.extend(((state.hand_translation[side][i] - source["translation"][i]) / 0.01).tolist())
    object_residuals = []
    for slot in state.episode.layout.object_slots:
        obj = state.episode.objects[slot]
        if bool(obj["pose_valid"][i]):
            object_residuals.extend(((state.object_translation[slot][i] - obj["translation"][i]) / 0.01).tolist())
    objective = float(getattr(ledger, "objective", 0.0))
    denominator = max(1.0, abs(objective))
    row = np.asarray([
        float(np.sqrt(np.mean(np.square(safe)))) if safe.size else 0.0,
        float(np.max(np.abs(safe))) if safe.size else 0.0,
        allowed_fraction,
        projected_fraction,
        float(np.tanh(float(reward) / denominator)),
        float(np.tanh(np.log1p(abs(objective)) / 10.0)),
        float(np.mean(hand_valid)) if hand_valid else 0.0,
        float(np.mean(object_valid)) if object_valid else 0.0,
        float(np.mean(metric_valid)) if metric_valid else 0.0,
        float(np.mean(contact_active)) if contact_active else 0.0,
        float(np.sqrt(np.mean(np.square(hand_residuals)))) if hand_residuals else 0.0,
        float(np.sqrt(np.mean(np.square(object_residuals)))) if object_residuals else 0.0,
    ], dtype=np.float32)
    if row.shape != (HISTORY_FEATURE_SIZE,) or not np.isfinite(row).all():
        raise RuntimeError("causal history row must be finite and fixed width")
    return row


def _history_features(state: AnnotationState, frames: int) -> np.ndarray:
    if frames <= 0:
        return np.empty((0,), dtype=np.float32)
    rows = [np.asarray(row, dtype=np.float32) for row in state.history[-frames:]]
    rows = [row for row in rows if row.shape == (HISTORY_FEATURE_SIZE,) and np.isfinite(row).all()]
    padded = [np.zeros(HISTORY_FEATURE_SIZE, dtype=np.float32) for _ in range(max(0, frames - len(rows)))] + rows
    return np.concatenate(padded, axis=0) if padded else np.empty((0,), dtype=np.float32)


def pack_observation(state: AnnotationState, *, position: int | None = None, terminal: bool = False) -> np.ndarray:
    episode = state.episode
    position = state.position if position is None else int(position)
    position = max(0, min(position - 1 if terminal else position, episode.frame_count - 1))
    i = position; duration = max(float(episode.time_s[-1] - episode.time_s[0]), 1e-6); frame = int(episode.frame_idx[i])
    values: list[float] = [float(i / max(episode.frame_count - 1, 1)),
        float((episode.time_s[i] - episode.time_s[0]) / duration), 1.0 if terminal else 0.0]
    values.extend(np.asarray(episode.T_world_camera[i], np.float32).reshape(-1)); values.extend(np.asarray(episode.intrinsics[i], np.float32))
    for side in episode.layout.hand_slots:
        hand = episode.hands[side]; present = bool(hand["valid"][i]); values.append(float(present))
        if present:
            # Exogenous source MANO plus current candidate residual/history are both causal.
            values.extend(hand["root_orient"][i]); values.extend(hand["articulation"][i]); values.extend(hand["translation"][i] / .01)
            values.extend(state.hand_root[side][i] - hand["root_orient"][i])
            values.extend(state.hand_articulation[side][i] - hand["articulation"][i]); values.extend((state.hand_translation[side][i] - hand["translation"][i]) / .01)
            if observation_layout(episode).include_hand_scale:
                scale_values = state.hand_log_scale.get(side)
                values.append(float(scale_values[i]) if scale_values is not None and len(scale_values) > i else 0.0)
            values.append(float(hand["row_visibility"][i])); values.extend(hand["joint_visibility"][i]); values.extend(hand["pose_visibility"][i]); values.extend(hand["hand_ray_shift_prior_world_m"][i] / .01)
            prior, uncertainty = _contact_summary(episode, frame, side); values.extend([prior, min(uncertainty / .1, 10.)])
        else:
            values.extend([0.] * (3 + 45 + 3 + 3 + 45 + 3 + (1 if observation_layout(episode).include_hand_scale else 0) + 1 + 21 + 15 + 3 + 2))
    samples = observation_layout(episode).point_samples
    for slot in episode.layout.object_slots:
        obj = episode.objects[slot]; present = bool(obj.get("present", obj["valid"])[i]); pose_valid = bool(obj["pose_valid"][i]); metric = bool(obj["metric_observation_eligible"][i]); values.extend([float(present), float(pose_valid), float(metric)])
        if pose_valid:
            values.extend(_rotation_vector(obj["rotation"][i])); values.extend(np.asarray(obj["translation"][i]) / .01)
            raw = np.asarray(obj["raw_observed_points_world"][i], np.float32); admitted = np.asarray(obj["observed_points_world"][i], np.float32)
            centroid = raw.mean(0) if len(raw) else np.zeros(3, np.float32); extent = np.ptp(raw, axis=0) if len(raw) else np.zeros(3, np.float32)
            values.extend(centroid / .01); values.extend(extent / .01); values.extend((_fixed_points(admitted, samples) / .01).reshape(-1))
            if observation_layout(episode).include_object_shared_scale:
                values.append(float(state.object_log_scale[i]) if len(state.object_log_scale) > i else 0.0)
        else:
            values.extend([0.] * (3 + 3 + 3 + 3 + samples * 3 + (1 if observation_layout(episode).include_object_shared_scale else 0)))
    values.extend(state.contact_logits[i].reshape(-1))
    values.extend(np.zeros(episode.layout.size, np.float32) if state.last_gate_trace is None else state.last_gate_trace.normalized_executed_action)
    values.extend(_history_features(state, observation_layout(episode).history_frames))
    result = np.asarray(values, np.float32); expected = observation_layout(episode).size
    if result.shape != (expected,) or not np.isfinite(result).all(): raise RuntimeError(f"packed observation shape/finite failure: {result.shape} expected {(expected,)}")
    return result
