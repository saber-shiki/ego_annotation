from __future__ import annotations

"""Complete annotation episode state and immutable episode inputs."""

from dataclasses import dataclass, field
from typing import Any, Mapping
import copy
import numpy as np

from actions.scaling import GateTrace
from actions.space import ActionLayout


@dataclass(frozen=True)
class EpisodeStatic:
    bundle_id: str
    clip_id: str
    source_video_id: str
    frame_idx: np.ndarray
    time_s: np.ndarray
    T_world_camera: np.ndarray
    intrinsics: np.ndarray
    image_size_wh: np.ndarray
    hands: Mapping[str, Mapping[str, Any]]
    objects: Mapping[str, Mapping[str, Any]]
    contact: Mapping[tuple[int, str], Mapping[str, Any]]
    contacts: Mapping[tuple[int, str, str], Mapping[str, Any]]
    factor_rows: Mapping[tuple[int, str | None, str | None, str], tuple[Mapping[str, Any], ...]]
    layout: ActionLayout
    mano_models: Mapping[str, Any]
    provenance: Mapping[str, Any]
    identity: Mapping[str, Any]

    @property
    def frame_count(self) -> int:
        return len(self.frame_idx)


@dataclass
class AnnotationState:
    episode: EpisodeStatic
    hand_root: dict[str, np.ndarray]
    hand_articulation: dict[str, np.ndarray]
    hand_translation: dict[str, np.ndarray]
    hand_vertices: dict[str, np.ndarray]
    hand_joints: dict[str, np.ndarray]
    object_rotation: dict[str, np.ndarray]
    object_translation: dict[str, np.ndarray]
    contact_logits: np.ndarray
    applied: np.ndarray
    hand_log_scale: dict[str, np.ndarray] = field(default_factory=dict)
    object_log_scale: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    position: int = 0
    previous_objective: float = 0.0
    last_gate_trace: GateTrace | None = None
    ledger: Any = None
    term_cache: dict[int, Any] = field(default_factory=dict)
    history: list[np.ndarray] = field(default_factory=list)
    history_limit: int = 0

    @property
    def done(self) -> bool:
        return self.position >= self.episode.frame_count

    def copy(self) -> "AnnotationState":
        return AnnotationState(
            episode=self.episode,
            hand_root={k: v.copy() for k, v in self.hand_root.items()},
            hand_articulation={k: v.copy() for k, v in self.hand_articulation.items()},
            hand_translation={k: v.copy() for k, v in self.hand_translation.items()},
            hand_log_scale={k: v.copy() for k, v in self.hand_log_scale.items()},
            hand_vertices={k: v.copy() for k, v in self.hand_vertices.items()},
            hand_joints={k: v.copy() for k, v in self.hand_joints.items()},
            object_rotation={k: v.copy() for k, v in self.object_rotation.items()},
            object_translation={k: v.copy() for k, v in self.object_translation.items()},
            contact_logits=self.contact_logits.copy(), applied=self.applied.copy(),
            object_log_scale=self.object_log_scale.copy(),
            position=self.position, previous_objective=self.previous_objective,
            last_gate_trace=self.last_gate_trace, ledger=copy.deepcopy(self.ledger),
            term_cache=copy.deepcopy(self.term_cache),
            history=[row.copy() for row in self.history], history_limit=self.history_limit,
        )


def initial_state(episode: EpisodeStatic) -> AnnotationState:
    hands = episode.hands
    objects = episode.objects
    count = episode.frame_count
    contact_count = episode.layout.contact_variables_per_hand
    contact_logits = np.zeros((count, len(episode.layout.hand_slots), contact_count), dtype=np.float32)
    if contact_count:
        for i, frame in enumerate(episode.frame_idx):
            for side_index, side in enumerate(episode.layout.hand_slots):
                matches = [row for (row_frame, row_side, _target), row in episode.contacts.items()
                           if row_frame == int(frame) and row_side == side]
                prior = float(np.mean([row.get("prior", 0.5) for row in matches])) if matches else 0.5
                prior = float(np.clip(prior, 1e-4, 1.0 - 1e-4))
                contact_logits[i, side_index] = np.log(prior / (1.0 - prior))
    return AnnotationState(
        episode=episode,
        hand_root={side: np.asarray(value["root_orient"], dtype=np.float32).copy()
                   for side, value in hands.items()},
        hand_articulation={side: np.asarray(value["articulation"], dtype=np.float32).copy()
                           for side, value in hands.items()},
        hand_translation={side: np.asarray(value["translation"], dtype=np.float32).copy()
                          for side, value in hands.items()},
        hand_log_scale={side: np.zeros(count, dtype=np.float32)
                        for side in episode.layout.hand_slots},
        hand_vertices={side: np.asarray(value["vertices"], dtype=np.float32).copy()
                       for side, value in hands.items()},
        hand_joints={side: np.asarray(value["joints"], dtype=np.float32).copy()
                     for side, value in hands.items()},
        object_rotation={name: np.asarray(value["rotation"], dtype=np.float64).copy()
                         for name, value in objects.items()},
        object_translation={name: np.asarray(value["translation"], dtype=np.float64).copy()
                            for name, value in objects.items()},
        object_log_scale=np.zeros(count, dtype=np.float32),
        contact_logits=contact_logits,
        applied=np.zeros(count, dtype=bool),
    )
