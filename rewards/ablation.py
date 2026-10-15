"""Strict, single-family reward selection for paired ablation experiments.

Full evaluation remains unmodified. This module never changes geometry, action
sampling, source observations or individual factor weights/reductions.
"""
from dataclasses import replace

GROUPS = (
    "hand_pixel", "hand_joint_depth", "hand_surface_depth", "hand_priors",
    "hand_temporal", "hand_bounds", "object_temporal", "object_geometry",
    "object_rgb_rotation", "object_rgb_reprojection", "contact", "generated_collision",
)
CONTRACT = "single_family_drop_full_eval.v1"


def classify(name):
    p = name.split('/')
    if p[0] == 'hypothesis':
        return 'generated_collision'
    if p[0] == 'signed':
        return 'optional_measured_signed'
    if p[0] == 'contact':
        return 'contact'
    if len(p) < 3:
        raise ValueError(f'unclassified full factor: {name}')
    family = p[2]
    if p[0] == 'hand':
        if family == 'joint_pixel_deadband':return 'hand_pixel'
        if family == 'joint_depth_deadband':return 'hand_joint_depth'
        if family == 'visible_surface_depth_order':return 'hand_surface_depth'
        if family in ('translation_prior','root_prior','articulation_prior','hand_ray_prior'):return 'hand_priors'
        if family.endswith(('_velocity','_acceleration')):return 'hand_temporal'
        if family.endswith('_soft_bound'):return 'hand_bounds'
    if p[0] == 'object':
        if family in ('translation_velocity','rotation_velocity','translation_acceleration','rotation_acceleration'):return 'object_temporal'
        if family in ('observed_point_to_plane','observed_point_to_point'):return 'object_geometry'
        if family == 'rgb_edge':
            if p[-1] != 'rotation':raise ValueError('this screen requires the current rotation-only RGB edge adapter')
            return 'object_rgb_rotation'
        if family == 'rgb_reprojection':return 'object_rgb_reprojection'
        if family in ('translation_sigma_prior','rotation_sigma_prior','nonpenetration_translation_target'):return 'optional_object_priors'
    raise ValueError(f'unclassified full factor: {name}')


def grouped(terms):
    result = {name:0. for name in GROUPS}
    for name, value in terms.items():
        group=classify(name)
        result[group]=result.get(group,0.)+float(value)
    return result


def select_ledger(full, drop_group):
    if drop_group not in ('full',*GROUPS):
        raise ValueError(f'unreviewed single reward-family ablation: {drop_group}')
    removed={k:v for k,v in full.terms.items() if classify(k)==drop_group}
    kept={k:v for k,v in full.terms.items() if k not in removed}
    selected=replace(full,terms=kept,statistics={k:v for k,v in full.statistics.items() if k in kept},
                     activation={**full.activation,'ablation_contract':CONTRACT,'removed_group':drop_group,
                                 'full_evaluation_objective':full.objective,'removed_cost':sum(removed.values())})
    return selected,removed
