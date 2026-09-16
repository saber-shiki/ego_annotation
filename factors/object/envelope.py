"""Containment is a soft outside-only prior, never a reconstructed surface."""
import numpy as np

def local_coordinates(points_world,T_world_surface):
    # Exact inverse, preserving the source camera's tiny non-orthonormal rounding.
    return (np.asarray(points_world)-T_world_surface[:3,3])@np.linalg.inv(T_world_surface[:3,:3]).T

def outside_residual_m(points_world,T_world_surface,lower,upper):
    """Signed per-coordinate soft-bound residual, usable for sampled mesh R/t fitting.

Zero inside; negative below the lower bound; positive beyond upper. This is
only containment: no point-to-surface, first-hit or edge correspondence here.
"""
    q=local_coordinates(points_world,T_world_surface)
    return np.minimum(q-lower,0)+np.maximum(q-upper,0)

def conditional_translation_interval(canonical_vertices,R_candidate,T_world_surface,lower,upper):
    """Exact full-mesh vertex translation interval for a FIXED candidate rotation.

Returned t_local is (object canonical-origin world t - S origin) in S coords,
not object center. For a convex box, vertex containment also contains all mesh
triangles. The underlying box remains an approximate/uncertain hypothesis.
"""
    a=(canonical_vertices@R_candidate.T)@np.linalg.inv(T_world_surface[:3,:3]).T
    tlo=lower-a.min(axis=0);thi=upper-a.max(axis=0)
    return tlo,thi

def scaled_vertices(vertices, pivot, scale):
    return np.asarray(pivot) + scale * (np.asarray(vertices) - pivot)


def robust_box_loss(points_world, T_world_surface, lower, upper):
    residual = outside_residual_m(points_world, T_world_surface, lower, upper) / .015
    return float(np.mean(2 * (np.sqrt(1 + residual * residual) - 1)))
