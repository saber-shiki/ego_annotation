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
    position: int = 0
    previous_objective: float = 0.0
    last_gate_trace: GateTrace | None = None
    ledger: Any = None
    term_cache: dict[int, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.position >= self.episode.frame_count

    def copy(self) -> "AnnotationState":
        return AnnotationState(
            episode=self.episode,
            hand_root={k: v.copy() for k, v in self.hand_root.items()},
            hand_articulation={k: v.copy() for k, v in self.hand_articulation.items()},
            hand_translation={k: v.copy() for k, v in self.hand_translation.items()},
            hand_vertices={k: v.copy() for k, v in self.hand_vertices.items()},
            hand_joints={k: v.copy() for k, v in self.hand_joints.items()},
            object_rotation={k: v.copy() for k, v in self.object_rotation.items()},
            object_translation={k: v.copy() for k, v in self.object_translation.items()},
            contact_logits=self.contact_logits.copy(), applied=self.applied.copy(),
            position=self.position, previous_objective=self.previous_objective,
            last_gate_trace=self.last_gate_trace, ledger=copy.deepcopy(self.ledger),
            term_cache=copy.deepcopy(self.term_cache),
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
                prior = float(episode.contact.get((int(frame), side), {}).get("prior", 0.5))
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
        hand_vertices={side: np.asarray(value["vertices"], dtype=np.float32).copy()
                       for side, value in hands.items()},
        hand_joints={side: np.asarray(value["joints"], dtype=np.float32).copy()
                     for side, value in hands.items()},
        object_rotation={name: np.asarray(value["rotation"], dtype=np.float64).copy()
                         for name, value in objects.items()},
        object_translation={name: np.asarray(value["translation"], dtype=np.float64).copy()
                            for name, value in objects.items()},
        contact_logits=contact_logits,
        applied=np.zeros(count, dtype=bool),
    )
