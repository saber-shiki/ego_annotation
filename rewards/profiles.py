from __future__ import annotations

"""Reviewed reward compositions and their historical numerical defaults.

A profile selects one existing objective composition. Optional factor families are
activated by their original switches and evidence; profiles are not blindly
unioned, because the formal and visual objectives contain alternative priors and
temporal terms.
"""

from dataclasses import dataclass
from typing import Any, Mapping


FORMAL_DEFAULTS: dict[str, Any] = {
    "translation_prior_weight": 2000.0,
    "root_prior_weight": 150.0,
    "pose_prior_weight": 75.0,
    "smooth_weight": 5000.0,
    "accel_weight": 10000.0,
    "observed_penetration_weight": 300000.0,
    "dense_observed_penetration_weight": 300000.0,
    "visible_hinge_weight": 800.0,
    "depth_hinge_weight": 20000.0,
    "bound_hinge_weight": 3000.0,
    "max_translation_m": 0.045,
    "max_root_delta_rad": 0.30,
    "max_pose_delta_rad": 0.45,
    "max_object_translation_m": 0.015,
    "object_translation_prior_weight": 4000.0,
    "object_smooth_weight": 8000.0,
    "penetration_epsilon_m": 1.0e-5,
    "max_constraints_per_frame": 96,
    "max_visible_surface_depth_vertices": 160,
    "visible_shift_limit_px": 12.0,
    "depth_shift_limit_m": 0.035,
    "visible_surface_depth_order_margin_m": 0.01,
    "visible_surface_depth_order_weight": 20000.0,
    "contact_state_prior_residual_scale_m": 0.01,
    "contact_state_temporal_strength": 1.0,
    "contact_patch_residual_mode": "deadband_tube",
    "dense_observed_surface_barrier": True,
    "use_hand_ray_shift_prior": False,
    "contact_state_geometry_likelihood": False,
    "object_observed_terms": True,
    "object_observed_weight": 1.0,
    "object_observed_loss": "soft_l1",
    "object_observed_max_points": 250,
    "object_observed_min_pairs": 3,
    "object_observed_max_correspondence_m": 0.08,
    "formal_object_loss": "soft_l1",
    "sigma_nonpenetration_target_m": 0.03,
    "sigma_translation_delta_step_m": 0.01,
    "sigma_rotation_delta_step_rad": 0.08,
    "sigma_translation_delta_accel_m": 0.006,
    "sigma_rotation_delta_accel_rad": 0.05,
    "image_first_hit_weight": 1.0,
    "image_silhouette_weight": 1.0,
    "sigma_image_first_hit_m": 0.008,
    "sigma_image_silhouette_px": 4.0,
    "max_image_first_hit_residual_m": 0.03,
    "max_image_silhouette_residual_px": 16.0,
    "rgb_rotation_weight": 1.0,
    "rgb_translation_weight": 1.0,
    "rgb_rotation_rad": 0.05,
    "rgb_translation_m": 0.025,
    "rgb_reprojection_weight": 1.0,
    "sigma_rgb_reprojection_px": 2.0,
    "max_rgb_reprojection_residual_px": 40.0,
    # Scene-v7 visual boundary/pad weights are kept separate from formal
    # factor ratios and are only consumed by the named scene_v7_rl profile.
    "visual_boundary_weight": 3.0,
    "visual_unsigned_pad_weight": 1.7,
    "reward_scale": 1.0,
}

VISUAL_DEFAULTS: dict[str, Any] = {
    "visual_image_weight": 3.0,
    "visual_unsigned_pad_weight": 1.7,
    "visual_boundary_weight": 3.0,
    "visual_temporal_weight": 0.06,
    "reward_scale": 1.0,
}


@dataclass(frozen=True)
class RewardProfile:
    name: str
    formal_hand: bool
    formal_object: bool
    visual_joint_fixed_scale: bool
    values: Mapping[str, Any]
    required_families: tuple[str, ...] = ()
    enforce_activation: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def _get(cfg: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def resolve_profile(cfg: Mapping[str, Any] | Any) -> RewardProfile:
    """Resolve one named composition and apply only explicit numeric/switch overrides.

    ``visual_joint_fixed_scale`` is the RL-compatible fixed-scale slice of the
    legacy visual fitter: policy actions are MANO root/articulation/translation
    residuals only. The legacy log-scale objective remains in legacy_optimize;
    this profile never claims full legacy visual-objective equivalence.
    """
    name = str(_get(cfg, "reward_profile", "formal_annotation"))
    if name == "visual_joint":
        raise ValueError("reward_profile='visual_joint' was renamed to 'visual_joint_fixed_scale'; the RL profile has no hand-scale action")
    shapes = {
        # Full policy objective: formal MANO interval factors plus the distinct
        # formal object trajectory factors. The hand solver's optional
        # translation-only object block is omitted, avoiding duplicate object
        # priors/temporal penalties.
        "formal_annotation": (True, True, False, FORMAL_DEFAULTS, (), False),
        "formal_interval": (True, False, False, FORMAL_DEFAULTS, (), False),
        "formal_object": (False, True, False, FORMAL_DEFAULTS, (), False),
        "visual_joint_fixed_scale": (False, False, True, VISUAL_DEFAULTS, (), False),
        # This is intentionally fail-closed: the profile composes the formal
        # physical terms with explicit shape/pad terms and refuses a run whose
        # required evidence families are absent at the natural episode end.
        "scene_v7_rl": (True, True, False, FORMAL_DEFAULTS,
                         ("pose", "prior", "time", "depth", "shape", "hand", "pad"), True),
    }
    if name not in shapes:
        raise ValueError(f"unreviewed reward profile {name!r}; expected one of {sorted(shapes)}")
    hand, obj, visual, defaults, required_families, enforce_activation = shapes[name]
    values = dict(defaults)
    for key, default in defaults.items():
        value = _get(cfg, key, default)
        if isinstance(default, bool):
            values[key] = bool(value)
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            values[key] = type(default)(value)
        else:
            values[key] = value
    if float(values["reward_scale"]) <= 0:
        raise ValueError("reward_scale must be positive and is separate from factor weights")
    return RewardProfile(name, hand, obj, visual, values, required_families, enforce_activation)
