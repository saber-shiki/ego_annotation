from __future__ import annotations

"""Fixed-width causal policy observations with source-state/evidence features."""

from dataclasses import dataclass
import numpy as np

from envs.state import AnnotationState, EpisodeStatic


@dataclass(frozen=True)
class ObservationLayout:
    action_size: int
    hand_slots: int = 2
    object_slots: int = 1
    contact_variables_per_hand: int = 1
    point_samples: int = 8

    @property
    def size(self) -> int:
        # time/camera + source hand state/qualification/candidate residual +
        # source object SE3, P09 summaries/samples/qualification + contact/action history.
        hand = self.hand_slots * (1 + 3 + 45 + 3 + 3 + 45 + 3 + 1 + 21 + 15 + 3 + 2)
        obj = self.object_slots * (3 + 3 + 3 + 3 + 3 + self.point_samples * 3)
        contact = self.hand_slots * self.contact_variables_per_hand
        return 3 + 20 + hand + obj + contact + self.action_size


def observation_layout(episode: EpisodeStatic) -> ObservationLayout:
    return ObservationLayout(episode.layout.size, len(episode.layout.hand_slots), len(episode.layout.object_slots),
                             episode.layout.contact_variables_per_hand,
                             int(episode.provenance.get("observation_point_samples", 8)))


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
            values.append(float(hand["row_visibility"][i])); values.extend(hand["joint_visibility"][i]); values.extend(hand["pose_visibility"][i]); values.extend(hand["hand_ray_shift_prior_world_m"][i] / .01)
            prior, uncertainty = _contact_summary(episode, frame, side); values.extend([prior, min(uncertainty / .1, 10.)])
        else:
            values.extend([0.] * (3 + 45 + 3 + 3 + 45 + 3 + 1 + 21 + 15 + 3 + 2))
    samples = observation_layout(episode).point_samples
    for slot in episode.layout.object_slots:
        obj = episode.objects[slot]; present = bool(obj.get("present", obj["valid"])[i]); pose_valid = bool(obj["pose_valid"][i]); metric = bool(obj["metric_observation_eligible"][i]); values.extend([float(present), float(pose_valid), float(metric)])
        if pose_valid:
            values.extend(_rotation_vector(obj["rotation"][i])); values.extend(np.asarray(obj["translation"][i]) / .01)
            raw = np.asarray(obj["raw_observed_points_world"][i], np.float32); admitted = np.asarray(obj["observed_points_world"][i], np.float32)
            centroid = raw.mean(0) if len(raw) else np.zeros(3, np.float32); extent = np.ptp(raw, axis=0) if len(raw) else np.zeros(3, np.float32)
            values.extend(centroid / .01); values.extend(extent / .01); values.extend((_fixed_points(admitted, samples) / .01).reshape(-1))
        else:
            values.extend([0.] * (3 + 3 + 3 + 3 + samples * 3))
    values.extend(state.contact_logits[i].reshape(-1))
    values.extend(np.zeros(episode.layout.size, np.float32) if state.last_gate_trace is None else state.last_gate_trace.normalized_executed_action)
    result = np.asarray(values, np.float32); expected = observation_layout(episode).size
    if result.shape != (expected,) or not np.isfinite(result).all(): raise RuntimeError(f"packed observation shape/finite failure: {result.shape} expected {(expected,)}")
    return result
