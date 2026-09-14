"""Relative RGB world-motion residual; translation-only has zero rotation Jacobian."""
from __future__ import annotations
from typing import Mapping
import numpy as np
from scipy.spatial.transform import Rotation
from utils.data import RGBEdge

def rgb_edge_residual(edge: RGBEdge, pose_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    Rs, ts = pose_by_frame[edge.source_frame_idx]
    Rt, tt = pose_by_frame[edge.target_frame_idx]
    predicted_R = Rt @ Rs.T
    rotation_error = Rotation.from_matrix(predicted_R @ edge.rotation_source_to_target_world.T).as_rotvec()
    if edge.diagnostics.get("translation_only"):
        if edge.translation_reference_rotation is None or edge.translation_source_support_world is None:
            raise ValueError("translation-only edge requires a fixed orientation and source support")
        reference_R = edge.translation_reference_rotation
        c = edge.translation_source_support_world
        predicted_displacement = tt - ts + (reference_R - np.eye(3)) @ (c - ts)
        measured_displacement = (edge.rotation_source_to_target_world - np.eye(3)) @ c + edge.translation_source_to_target_world_m
        # Both orientation-dependent terms are immutable measurements; this
        # conditional translation residual has exactly zero rotation Jacobian.
        return np.zeros(3), predicted_displacement - measured_displacement
    predicted_t = tt - predicted_R @ ts
    return rotation_error, predicted_t - edge.translation_source_to_target_world_m
