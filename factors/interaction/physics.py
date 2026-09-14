"""Signed-authorized and visible-depth residuals; activation belongs to gates."""
import torch


def signed_active_penetration(moved, normals, depth, uncertainty, weight):
    residual=torch.relu(depth-uncertainty-(normals*moved).sum(dim=1))
    return weight*residual.square().mean()


def signed_dense_penetration(moved, normals, depth, uncertainty, weight):
    residual=torch.relu(depth-uncertainty-(normals*moved).sum(dim=1))
    return weight*residual.square().sum()/(residual>0).sum().clamp_min(1.)


def visible_surface_depth_order(camera_depth, surface_depth, margin, weight):
    residual=torch.relu(surface_depth-margin-camera_depth)
    return weight*residual.square().sum()/(residual>0).sum().clamp_min(1.)
