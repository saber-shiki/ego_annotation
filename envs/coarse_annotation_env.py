"""Fresh-checkpoint environment for explicitly derived coarse/v2 episodes."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from data.coarse_training import ADAPTER, EXTRA_FEATURES, VISIBILITY, CONTACT
from envs.annotation_env import AnnotationEnv
from envs.observation import pack_observation


def _one_hot(values, vocabulary):
    if any(str(v) not in vocabulary for v in np.asarray(values).reshape(-1)):
        raise ValueError("unknown typed coarse interaction observation")
    return np.asarray([[float(v == label) for label in vocabulary] for v in values], np.float32).reshape(-1)


class CoarseAnnotationEnv(AnnotationEnv):
    def __init__(self, dataset, cfg, device="cpu"):
        if cfg.get("preprocessed_state_adapter") != ADAPTER:
            raise ValueError("coarse environment requires its explicit reviewed adapter config")
        if int(cfg.get("max_objects", 1)) != 1:
            raise ValueError("coarse adapter v1 supports one selected complete object")
        if cfg.get("reward_profile", "formal_annotation") != "formal_annotation":
            raise ValueError("coarse adapter v1 reviewed only with formal_annotation")
        self.observation_contract_version = "coarse-v2-features/1"
        super().__init__(dataset, cfg, device)

    def _observation_size(self, layout):
        return super()._observation_size(layout) + EXTRA_FEATURES

    def _pack_observation(self, state, *, terminal=False):
        episode = state.episode
        if episode.provenance.get("preprocessed_state_adapter", {}).get("name") != ADAPTER:
            raise ValueError("coarse environment cannot consume a raw v1/v2 episode")
        i = max(0, min(state.position - 1 if terminal else state.position, episode.frame_count - 1))
        obj = episode.objects["object:0"]
        features = episode.provenance["coarse_features"]
        interaction = features["interaction"]
        def residual(position):
            return np.r_[Rotation.from_matrix(state.object_rotation["object:0"][position] @ obj["rotation"][position].T).as_rotvec(),
                         (state.object_translation["object:0"][position] - obj["translation"][position]) / .01]
        history = residual(i - 1) if i > 0 and state.applied[i - 1] else np.zeros(6)
        extra = np.r_[residual(i), history,
            features["source_pose_available"][i], features["source_metric_eligible"][i], features["translation_depth_inferred"][i],
            _one_hot(interaction.visibility[i], VISIBILITY), _one_hot(interaction.contact_state[i], CONTACT),
            interaction.contact_prior[i], interaction.prior_available[i], interaction.primary_declared[i].reshape(-1),
            interaction.primary_known[i].reshape(-1), features["rgb_summary"][i]].astype(np.float32)
        if extra.shape != (EXTRA_FEATURES,):
            raise RuntimeError(f"coarse feature layout mismatch: {extra.shape}")
        result = np.r_[pack_observation(state, terminal=terminal), extra].astype(np.float32)
        if result.shape != self.observation_space.shape or not np.isfinite(result).all():
            raise RuntimeError("invalid coarse policy observations")
        return result
