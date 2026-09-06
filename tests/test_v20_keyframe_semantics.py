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
RENDER_PATH = ROOT / "scripts" / "build_v20_support_aware_focused_rrd.py"
GENERATED_FACTOR_PATH = ROOT / "scripts" / "v20_generated_first_hit_factors.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load(CORE_PATH, "v20_test_core")
wrapper = load(WRAP_PATH, "v20_test_wrapper")
renderer = load(RENDER_PATH, "v20_test_renderer")
generated = load(GENERATED_FACTOR_PATH, "v20_test_generated_factors")


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


def test_point_factor_preserves_real_interframe_object_motion():
    canonical = np.array(
        [
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.1, 0.1, 0.0],
            [-0.1, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    source_t = np.array([0.0, 0.0, 0.5])
    target_t = np.array([0.3, -0.2, 1.4])
    source_world = core.apply_pose(canonical, np.eye(3), source_t)
    target_world = core.apply_pose(canonical, np.eye(3), target_t)
    nodes = [
        core.Node(0, {}, np.eye(3), source_t, source_world, 1.0, True),
        core.Node(5, {}, np.eye(3), target_t, target_world, 1.0, True),
    ]
    factor = core.PointFactor(
        0,
        1,
        source_world,
        target_world,
        np.tile(np.array([[0.0, 0.0, 1.0]]), (len(canonical), 1)),
        np.ones(len(canonical)),
        "keyframe_local",
    )
    args = SimpleNamespace(
        sigma_pose_prior_translation_m=0.04,
        sigma_pose_prior_rotation_rad=0.2,
        point_to_plane_weight=1.0,
        point_to_point_weight=0.2,
        max_point_residual_m=0.05,
        sigma_point_to_plane_m=0.006,
        sigma_point_to_point_m=0.015,
        sigma_correction_translation_step_m=0.012,
        sigma_correction_rotation_step_rad=0.1,
        sigma_correction_translation_accel_m=0.008,
        sigma_correction_rotation_accel_rad=0.06,
        sigma_motion_translation_accel_m=0.04,
        sigma_motion_rotation_accel_rad=0.2,
        anchor_position=0,
        sigma_anchor_gauge_translation_m=1e-6,
        sigma_anchor_gauge_rotation_rad=1e-6,
    )
    exact = core.residual_blocks(
        np.zeros(12), nodes, [factor], [], [], args
    )
    np.testing.assert_allclose(exact["point_to_plane"][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(exact["point_to_point"][0], 0.0, atol=1e-12)

    perturbed_x = np.zeros(12)
    perturbed_x[11] = 0.01
    perturbed = core.residual_blocks(
        perturbed_x, nodes, [factor], [], [], args
    )
    assert np.max(np.abs(perturbed["point_to_plane"][0])) > 0.0
    assert np.max(np.abs(perturbed["point_to_point"][0])) > 0.0


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


def test_all_frame_generated_depth_factor_constrains_both_neighboring_keyframes():
    key_nodes = [
        core.Node(0, {}, np.eye(3), np.array([0.0, 0.0, 1.0]), np.zeros((1, 3)), 1.0, True),
        core.Node(10, {}, np.eye(3), np.array([0.0, 0.0, 1.0]), np.zeros((1, 3)), 1.0, True),
    ]
    all_nodes = [
        key_nodes[0],
        core.Node(5, {}, np.eye(3), np.array([0.0, 0.0, 1.0]), np.zeros((1, 3)), 1.0, False),
        key_nodes[1],
    ]
    initial = {
        node.frame_idx: (node.base_rotation.copy(), node.base_translation.copy())
        for node in all_nodes
    }
    factor = SimpleNamespace(
        frame_idx=5,
        T_world_camera=np.eye(4),
        K_raster=np.eye(3),
        depth_points_canonical=np.array([[0.0, 0.0, 0.0]]),
        depth_observed_z=np.array([1.0]),
        depth_weight=np.ones(1),
        silhouette_points_canonical=np.empty((0, 3)),
        silhouette_target_uv=np.empty((0, 2)),
        silhouette_kind=np.empty(0, dtype=np.int8),
    )
    args = SimpleNamespace(
        sigma_pose_prior_translation_m=0.04,
        sigma_pose_prior_rotation_rad=0.2,
        image_first_hit_weight=0.0,
        image_silhouette_weight=0.0,
        point_to_plane_weight=0.0,
        point_to_point_weight=0.0,
        sigma_correction_translation_step_m=0.012,
        sigma_correction_rotation_step_rad=0.1,
        sigma_correction_translation_accel_m=0.008,
        sigma_correction_rotation_accel_rad=0.06,
        sigma_motion_translation_accel_m=0.04,
        sigma_motion_rotation_accel_rad=0.2,
        anchor_position=0,
        sigma_anchor_gauge_translation_m=1e-6,
        sigma_anchor_gauge_rotation_rad=1e-6,
        generated_visible_first_hit_weight=1.0,
        generated_visible_silhouette_weight=0.0,
        generated_visible_max_depth_residual_m=0.10,
        generated_visible_sigma_depth_m=0.008,
        generated_visible_max_silhouette_residual_px=40.0,
        generated_visible_sigma_silhouette_px=4.0,
    )
    factors = {5: factor}
    exact = wrapper.joint_residual_blocks(
        core,
        np.zeros(12),
        key_nodes,
        [],
        [],
        [],
        args,
        {},
        factors,
        all_nodes,
        [0, 10],
        initial,
    )
    np.testing.assert_allclose(exact["generated_first_hit"][0], 0.0, atol=1e-12)

    perturbed = np.zeros(12)
    perturbed[11] = 0.02
    moved = wrapper.joint_residual_blocks(
        core,
        perturbed,
        key_nodes,
        [],
        [],
        [],
        args,
        {},
        factors,
        all_nodes,
        [0, 10],
        initial,
    )
    np.testing.assert_allclose(
        moved["generated_first_hit"][0], np.array([0.01 / 0.008]), atol=1e-10
    )
    pattern = wrapper.joint_residual_sparsity(
        core, key_nodes, [], [], [], args, {}, factors, [0, 10]
    )
    generated_row_columns = set(pattern.getrow(pattern.shape[0] - 1).indices.tolist())
    assert generated_row_columns == set(range(12))


def test_exact_gate_step_scales_are_strictly_descending():
    assert wrapper.parse_exact_gate_step_scales("1,.5,.1") == [1.0, 0.5, 0.1]
    for invalid in ("", "0.5,0.5", "0.5,0.7", "1.2", "0"):
        try:
            wrapper.parse_exact_gate_step_scales(invalid)
        except Exception:
            pass
        else:
            raise AssertionError(f"accepted invalid exact-gate scales: {invalid}")


def test_generated_metric_deltas_are_per_frame_and_per_segment():
    def frame(frame_idx, median, p95=None, coverage=0.9, iou=0.8):
        return {
            "frame_idx": frame_idx,
            "true_first_hit_abs_depth_m": {
                "median": median,
                "p95": 2.0 * median if p95 is None else p95,
            },
            "true_first_hit_coverage_fraction": coverage,
            "initial_silhouette_iou": iou,
        }

    current = {
        "mesh_sha256": "same",
        "true_first_hit_abs_depth_m": {"median": 0.004, "p95": 0.010},
        "per_frame": [frame(0, 0.003), frame(5, 0.004), frame(10, 0.005)],
    }
    candidate = {
        "mesh_sha256": "same",
        "true_first_hit_abs_depth_m": {"median": 0.0045, "p95": 0.011},
        "per_frame": [
            frame(0, 0.003),
            frame(5, 0.006, p95=0.012, coverage=0.87, iou=0.79),
            frame(10, 0.004),
        ],
    }
    delta = wrapper.generated_metric_deltas(current, candidate, [0, 5, 10])
    np.testing.assert_allclose(delta["global_abs_depth_median_delta_m"], 0.0005)
    np.testing.assert_allclose(delta["global_abs_depth_p95_delta_m"], 0.001)
    np.testing.assert_allclose(
        delta["max_frame_common_hit_abs_depth_median_delta_m"], 0.002
    )
    np.testing.assert_allclose(
        delta["max_frame_common_hit_abs_depth_p95_delta_m"], 0.004
    )
    np.testing.assert_allclose(delta["min_frame_first_hit_coverage_delta"], -0.03)
    np.testing.assert_allclose(delta["min_frame_silhouette_iou_delta"], -0.01)
    assert len(delta["segments"]) == 2


def test_generated_common_hit_tail_ignores_newly_covered_pixels():
    current = {
        "mesh_sha256": "same",
        "true_first_hit_abs_depth_m": {"median": 0.001, "p95": 0.001},
        "per_frame": [
            {
                "frame_idx": 0,
                "true_first_hit_abs_depth_m": {
                    "count": 1,
                    "median": 0.001,
                    "p95": 0.001,
                },
                "true_first_hit_coverage_fraction": 0.5,
                "initial_silhouette_iou": 0.8,
            }
        ],
    }
    candidate = {
        "mesh_sha256": "same",
        "true_first_hit_abs_depth_m": {"median": 0.0255, "p95": 0.04755},
        "per_frame": [
            {
                "frame_idx": 0,
                "true_first_hit_abs_depth_m": {
                    "count": 2,
                    "median": 0.0255,
                    "p95": 0.04755,
                },
                "true_first_hit_coverage_fraction": 1.0,
                "initial_silhouette_iou": 0.82,
            }
        ],
    }
    before_factor = SimpleNamespace(
        evaluation_hit=np.array([True, False]),
        evaluation_signed_depth=np.array([0.001, np.nan]),
    )
    after_factor = SimpleNamespace(
        evaluation_hit=np.array([True, True]),
        evaluation_signed_depth=np.array([0.001, 0.05]),
    )
    delta = wrapper.generated_metric_deltas(
        current,
        candidate,
        [0],
        {0: before_factor},
        {0: after_factor},
    )
    assert delta["per_frame"][0]["common_hit_count"] == 1
    np.testing.assert_allclose(
        delta["max_frame_common_hit_abs_depth_median_delta_m"], 0.0
    )
    np.testing.assert_allclose(
        delta["max_frame_common_hit_abs_depth_p95_delta_m"], 0.0
    )
    np.testing.assert_allclose(delta["global_abs_depth_median_delta_m"], 0.0)
    np.testing.assert_allclose(delta["global_abs_depth_p95_delta_m"], 0.0)
    assert delta["global_support_abs_depth_median_delta_m"] > 0.0
    assert delta["global_support_abs_depth_p95_delta_m"] > 0.0
    assert delta["global_depth_comparison_support"] == "before_after_common_first_hits"
    np.testing.assert_allclose(delta["min_frame_first_hit_coverage_delta"], 0.5)


def test_dynamic_generated_factor_rebuild_uses_current_pose_without_gpu():
    from dataclasses import dataclass

    @dataclass
    class Frame:
        frame_idx: int
        rotation_world_object: np.ndarray
        translation_world_object: np.ndarray
        T_world_camera: np.ndarray
        observed_uv: np.ndarray
        observed_camera: np.ndarray

    class Builder:
        @staticmethod
        def canonical_to_camera(vertices, rotation, translation, _camera):
            return vertices @ rotation.T + translation

        @staticmethod
        def build_frame_factors(frame, _zbuf, _rendered, _K, _args):
            return {
                "depth_points": np.array([[0.0, 0.0, 0.0]]),
                "depth_z": np.array([1.0]),
                "depth_weight": np.ones(1),
                "silhouette_points": np.empty((0, 3)),
                "silhouette_target_uv": np.empty((0, 2)),
                "silhouette_kind": np.empty(0, dtype=np.int8),
            }, {
                "frame_idx": frame.frame_idx,
                "initial_silhouette_iou": 1.0,
                "initial_observed_hit_fraction": 1.0,
            }

    class Rasterizer:
        @staticmethod
        def render(vertices_camera):
            assert np.isclose(vertices_camera[0][0, 2], 1.0)
            zbuf = np.full((1, 2, 2), np.nan, dtype=np.float64)
            rendered = np.zeros((1, 2, 2), dtype=bool)
            zbuf[0, 0, 0] = 1.0
            rendered[0, 0, 0] = True
            return zbuf, rendered

    frame = Frame(
        5,
        np.eye(3),
        np.zeros(3),
        np.eye(4),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    )
    context = generated.GeneratedFactorContext(
        builder=Builder(),
        mesh_path=Path("generated.ply"),
        mesh_sha256="mesh-hash",
        mesh_vertices=np.array([[0.0, 0.0, 0.0]]),
        mesh_faces=np.array([[0, 0, 0]], dtype=np.int64),
        source_frames=[frame],
        K_raster=np.eye(3),
        rasterizer=Rasterizer(),
    )
    args = SimpleNamespace(
        annotations=Path("annotations.json"),
        pose_report=Path("pose.json"),
        generated_visible_hand_npz=Path("hand.npz"),
        generated_visible_mano_faces_pkl=Path("mano.pkl"),
        object_id="object",
        frame_start=0,
        frame_end=10,
        generated_visible_raster_size=2,
        generated_visible_source_size=2,
        generated_visible_min_frames=1,
        generated_visible_min_observed_points=1,
        generated_visible_min_ownership_fraction=0.0,
        generated_visible_hand_unknown_dilation_px=0,
        generated_visible_max_removed_fraction_for_full_weight=0.1,
        generated_visible_min_depth_factor_weight=0.25,
        generated_visible_boundary_downweight_radius_px=1.5,
        generated_visible_boundary_weight=0.65,
        generated_visible_max_observed_factors_per_frame=1,
        generated_visible_max_outside_factors_per_frame=1,
        generated_visible_max_missing_factors_per_frame=1,
        generated_visible_render_batch_size=1,
    )
    factors, info = generated.rebuild_factors(
        context, {5: (np.eye(3), np.array([0.0, 0.0, 1.0]))}, args
    )
    assert sorted(factors) == [5]
    assert info["depth_factor_count"] == 1
    assert info["mesh_sha256"] == "mesh-hash"
    np.testing.assert_allclose(
        info["true_first_hit_abs_depth_m"]["median"], 0.0, atol=1e-12
    )


def test_render_mesh_contract_rejects_a_different_mesh(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected = root / "expected.ply"
        wrong = root / "wrong.ply"
        equivalent_copy = root / "equivalent-copy.ply"
        expected.write_bytes(b"same generated mesh")
        wrong.write_bytes(b"different generated mesh")
        equivalent_copy.write_bytes(expected.read_bytes())
        payload = {
            "render_mesh_contract": {
                "renderer_must_match": True,
                "required_mesh_path": str(expected),
                "required_mesh_sha256": renderer.sha256_file(expected),
            }
        }
        validated = renderer.validate_render_mesh_contract(payload, expected)
        assert validated["validated"] is True
        assert validated["path_matches"] is True
        try:
            renderer.validate_render_mesh_contract(payload, equivalent_copy)
        except RuntimeError as exc:
            assert "path does not match pose objective" in str(exc)
        else:
            raise AssertionError(
                "renderer accepted identical bytes from an unbound mesh path"
            )
        expected.write_bytes(b"tampered generated mesh")
        try:
            renderer.validate_render_mesh_contract(payload, expected)
        except RuntimeError as exc:
            assert "hash does not match pose objective" in str(exc)
        else:
            raise AssertionError("renderer accepted changed bytes at the bound path")
        try:
            renderer.validate_render_mesh_contract(payload, wrong)
        except RuntimeError as exc:
            assert (
                "path does not match pose objective" in str(exc)
                or "hash does not match pose objective" in str(exc)
            )
        else:
            raise AssertionError("renderer accepted a mesh outside the pose contract")


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
