"""Metric geometry residual kernels; normal and count normalization are explicit."""
import numpy as np


def first_hit_depth(camera_z, observed_z, weights, *, sigma, clip, weight):
    residual=np.clip(camera_z-observed_z,-clip,clip)/sigma
    return np.sqrt(weight)*np.sqrt(weights)*residual/np.sqrt(max(float(np.sum(weights)),1.))


def observed_point_to_plane(delta,normals,weights,weight=1.):
    return np.sqrt(weights*weight/max(1,len(delta)))*np.einsum('ij,ij->i',normals,delta)/.008


def observed_point_to_point(delta,weights,weight=1.):
    return np.sqrt(weights*weight/max(1,len(delta)))[:,None]*delta/.020
