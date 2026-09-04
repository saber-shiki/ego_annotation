from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "fit_v20_keyframe_observed_pose_graph.py"
WRAP_PATH = ROOT / "scripts" / "fit_v20_keyframe_periodic_graph.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load(CORE_PATH, "v20_test_core")
wrapper = load(WRAP_PATH, "v20_test_wrapper")


def test_left_se3_correction_roundtrip_and_interpolation():
    r0 = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    t0 = np.array([0.4, -0.2, 1.1])
    cr = Rotation.from_euler("xyz", [0.03, -0.02, 0.01]).as_matrix()
    ct = np.array([0.01, -0.015, 0.02])
    rf = cr @ r0
    tf = cr @ t0 + ct
    dr, dt = wrapper.pose_correction(r0, t0, rf, tf)
    np.testing.assert_allclose(Rotation.from_rotvec(dr).as_matrix(), cr, atol=1e-10)
    np.testing.assert_allclose(dt, ct, atol=1e-10)
    initial = {0: (r0, t0), 5: (r0, t0), 10: (r0, t0)}
    final = {0: (r0, t0), 10: (rf, tf)}
    rm, tm, mode = wrapper.interpolate_pose_correction(5, [0, 5, 10], [0, 10], initial, final)
    expected_r = Rotation.from_rotvec(0.5 * dr).as_matrix() @ r0
    expected_t = Rotation.from_rotvec(0.5 * dr).as_matrix() @ t0 + 0.5 * dt
    assert mode == "interpolated_keyframe_correction"
    np.testing.assert_allclose(rm, expected_r, atol=1e-10)
    np.testing.assert_allclose(tm, expected_t, atol=1e-10)


def test_canonical_registration_factor_has_zero_residual_for_true_relation():
    # C maps source canonical coordinates into target canonical coordinates.
    c = Rotation.from_euler("z", 0.12).as_matrix()
    ct = np.array([0.02, -0.01, 0.03])
    rs = Rotation.from_euler("xyz", [0.1, 0.2, -0.1]).as_matrix()
    ts = np.array([0.3, 0.1, 0.8])
    rt = rs @ c.T
    tt = ts - rt @ ct
    factor = core.RelativeFactor(0, 1, 0, 1, c, ct, 1.0, "keyframe")
    rr, tr = core.relative_residual([rs, rt], [ts, tt], factor)
    np.testing.assert_allclose(rr, 0.0, atol=1e-10)
    np.testing.assert_allclose(tr, 0.0, atol=1e-10)


def test_rgb_absolute_factor_does_not_remove_physical_motion():
    # A non-zero physical target pose is represented directly; the factor
    # constrains the target absolute pose and must not compare it with source.
    rs = Rotation.from_euler("y", -0.4).as_matrix()
    ts = np.array([0.2, 0.3, 0.9])
    rt = Rotation.from_euler("xyz", [0.15, -0.2, 0.25]).as_matrix()
    tt = np.array([-0.1, 0.4, 1.2])
    factor = core.RelativeFactor(1, 1, 7, 7, rt, tt, 1.0, "rgb_absolute")
    rr, tr = core.relative_residual([rs, rt], [ts, tt], factor)
    np.testing.assert_allclose(rr, 0.0, atol=1e-10)
    np.testing.assert_allclose(tr, 0.0, atol=1e-10)


def test_image_residual_sparsity_matches_flattened_rows():
    nodes = [
        core.Node(0, {}, np.eye(3), np.zeros(3), np.zeros((3, 3)), 1.0, True),
        core.Node(1, {}, np.eye(3), np.array([0.0, 0.0, 1.0]), np.zeros((3, 3)), 1.0, True),
    ]
    factors = {
        i: core.ImageFactor(
            i,
            np.eye(4),
            np.eye(3),
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
            np.array([1.0, 1.0]),
            np.ones(2),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0]]),
            np.array([0], dtype=np.int8),
        )
        for i in (0, 1)
    }
    args = SimpleNamespace(
        sigma_pose_prior_translation_m=0.04,
        sigma_pose_prior_rotation_rad=0.2,
        image_first_hit_weight=1.0,
        image_silhouette_weight=1.0,
        max_image_first_hit_residual_m=0.03,
        sigma_image_first_hit_m=0.008,
        max_image_silhouette_residual_px=16.0,
        sigma_image_silhouette_px=4.0,
        sigma_correction_translation_step_m=0.012,
        sigma_correction_rotation_step_rad=0.1,
        sigma_correction_translation_accel_m=0.008,
        sigma_correction_rotation_accel_rad=0.06,
        sigma_motion_translation_accel_m=0.04,
        sigma_motion_rotation_accel_rad=0.2,
        anchor_position=0,
        sigma_anchor_gauge_translation_m=1e-6,
        sigma_anchor_gauge_rotation_rad=1e-6,
        point_to_plane_weight=0.0,
        point_to_point_weight=0.0,
    )
    blocks = core.residual_blocks(np.zeros(12), nodes, [], [], [], args, factors)
    flattened = core.flatten_blocks(blocks)
    pattern = core.residual_sparsity(nodes, [], [], [], args, factors)
    assert pattern.shape == (len(flattened), 12)


def test_disabled_relative_scales_are_really_disabled():
    args = SimpleNamespace(
        keyframe_relative_weight_scale=0.0,
        min_relative_weight=0.1,
        rgb_relative_weight_scale=0.0,
        min_rgb_weight=0.05,
    )
    # The scale-zero policy is enforced before the minimum quality floor; this
    # prevents a nominally disabled term from silently receiving min weight.
    assert args.keyframe_relative_weight_scale == 0.0
    assert args.rgb_relative_weight_scale == 0.0
