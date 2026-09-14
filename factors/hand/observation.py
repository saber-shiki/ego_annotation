"""MANO baseline image/depth preservation, with distinct visual/formal normalizers."""
from factors.common import mean_square


def visual_surface_reprojection(pixel, baseline):
    return mean_square(pixel-baseline, 9.)


def visual_joint_reprojection(pixel, baseline):
    return mean_square(pixel-baseline, 6.)


def joint_pixel_deadband(pixel, baseline, visibility, margin, weight):
    import torch
    error=torch.relu(torch.linalg.vector_norm(pixel-baseline, dim=-1)-margin).square()
    return weight*(visibility*error).sum()/visibility.sum().clamp_min(1.)


def joint_depth_deadband(depth, baseline, visibility, margin, weight):
    import torch
    error=torch.relu(torch.abs(depth-baseline)-margin).square()
    return weight*(visibility*error).sum()/visibility.sum().clamp_min(1.)
