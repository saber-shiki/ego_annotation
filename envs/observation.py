from __future__ import annotations

"""Fixed-width causal policy observations with explicit entity-slot masks."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from envs.state import AnnotationState, EpisodeStatic


@dataclass(frozen=True)
class ObservationLayout:
    action_size: int
    hand_slots: int = 2
    object_slots: int = 1
    contact_variables_per_hand: int = 1

    @property
    def size(self) -> int:
        # progress/time (3), camera T/K (20), masks, candidate residuals,
        # evidence counts/contact priors, and previous executed normalized action.
        hand = self.hand_slots * (1 + 3 + 45 + 3 + 2)
        obj = self.object_slots * (1 + 3 + 3 + 2)
        contact = self.hand_slots * self.contact_variables_per_hand
        return 3 + 20 + hand + obj + contact + self.action_size


def observation_layout(episode: EpisodeStatic) -> ObservationLayout:
    return ObservationLayout(episode.layout.size, len(episode.layout.hand_slots),
                             len(episode.layout.object_slots), episode.layout.contact_variables_per_hand)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                           matrix[1, 0] - matrix[0, 1]], dtype=np.float64) / 2.0
    axis = np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                       matrix[1, 0] - matrix[0, 1]], dtype=np.float64)
    return axis / max(2.0 * np.sin(angle), 1e-12) * angle


def pack_observation(state: AnnotationState, *, position: int | None = None, terminal: bool = False) -> np.ndarray:
    episode = state.episode
    if position is None:
        position = state.position
    if terminal:
        position = max(0, min(int(position) - 1, episode.frame_count - 1))
    else:
        position = min(int(position), episode.frame_count - 1)
    i = position
    duration = max(float(episode.time_s[-1] - episode.time_s[0]), 1e-6)
    values: list[float] = [
        float(i / max(episode.frame_count - 1, 1)),
        float((episode.time_s[i] - episode.time_s[0]) / duration),
        1.0 if terminal else 0.0,
    ]
    values.extend(np.asarray(episode.T_world_camera[i], dtype=np.float32).reshape(-1).tolist())
    values.extend(np.asarray(episode.intrinsics[i], dtype=np.float32).tolist())
    frame_id = int(episode.frame_idx[i])
    for side in episode.layout.hand_slots:
        hand = episode.hands[side]
        present = bool(hand["valid"][i])
        values.append(float(present))
        if present:
            values.extend((state.hand_root[side][i] - hand["root_orient"][i]).tolist())
            values.extend((state.hand_articulation[side][i] - hand["articulation"][i]).tolist())
            values.extend(((state.hand_translation[side][i] - hand["translation"][i]) / 0.01).tolist())
            contact = episode.contact.get((frame_id, side), {})
            values.extend([float(contact.get("prior", 0.0)),
                           min(float(contact.get("support_uncertainty_m", 0.0)) / 0.1, 10.0)])
        else:
            values.extend([0.0] * (3 + 45 + 3 + 2))
    for slot_name in episode.layout.object_slots:
        obj = episode.objects[slot_name]
        present = bool(obj["valid"][i])
        values.append(float(present))
        if present:
            residual_R = state.object_rotation[slot_name][i] @ np.asarray(obj["rotation"][i]).T
            values.extend(_rotation_vector(residual_R).tolist())
            values.extend(((state.object_translation[slot_name][i] - obj["translation"][i]) / 0.01).tolist())
            observed = obj["observed_points_world"][i]
            values.extend([min(len(observed) / 2000.0, 10.0), float(bool(obj.get("watertight", False)))])
        else:
            values.extend([0.0] * (3 + 3 + 2))
    values.extend(state.contact_logits[i].reshape(-1).tolist())
    if state.last_gate_trace is None:
        values.extend([0.0] * episode.layout.size)
    else:
        values.extend(state.last_gate_trace.normalized_executed_action.tolist())
    result = np.asarray(values, dtype=np.float32)
    expected = observation_layout(episode).size
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise RuntimeError(f"packed observation shape/finite failure: {result.shape} expected {(expected,)}")
    return result
