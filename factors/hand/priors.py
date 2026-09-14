"""Visual and formal MANO prior normalizations remain distinct."""
from factors.common import mean_square


def visual_articulation(delta):
    return mean_square(delta, .18, .8)


def visual_root_orientation(delta):
    return mean_square(delta, .12)


def visual_translation(delta):
    return mean_square(delta, .018)


def hand_log_scale(delta):
    return mean_square(delta, .12, .3)


def formal_translation(delta, visibility, weight):
    return weight * (delta.square() * visibility[:, None]).sum() / (3 * visibility.sum()).clamp_min(1.)


def formal_root_orientation(delta, visibility, weight):
    # Historical interval objective divides the 3-D root residual by 9.
    return weight * (delta.square() * visibility[:, None, None]).sum() / (9 * visibility.sum()).clamp_min(1.)


def formal_articulation(delta, visibility, weight):
    return weight * (delta.square() * visibility[..., None]).sum() / (3 * visibility.sum()).clamp_min(1.)
