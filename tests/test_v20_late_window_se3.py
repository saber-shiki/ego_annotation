from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_v20_late_window_prediction_only_se3.py"
RENDERER_SCRIPT = ROOT / "scripts" / "build_v20_support_aware_focused_rrd.py"
RGB_BUILDER_SCRIPT = ROOT / "scripts" / "build_v20_global_rgb_orientation_edges.py"
SHAPE_SCRIPT = ROOT / "scripts" / "fit_v20_generated_shape_alignment.py"
spec = importlib.util.spec_from_file_location("v20_late_window_se3_test_module", SCRIPT)
assert spec and spec.loader
late = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = late
spec.loader.exec_module(late)
renderer_spec = importlib.util.spec_from_file_location("v20_late_window_renderer_test_module", RENDERER_SCRIPT)
assert renderer_spec and renderer_spec.loader
renderer = importlib.util.module_from_spec(renderer_spec)
sys.modules[renderer_spec.name] = renderer
renderer_spec.loader.exec_module(renderer)
builder_spec = importlib.util.spec_from_file_location("v20_late_window_rgb_builder_test_module", RGB_BUILDER_SCRIPT)
assert builder_spec and builder_spec.loader
builder = importlib.util.module_from_spec(builder_spec)
sys.modules[builder_spec.name] = builder
builder_spec.loader.exec_module(builder)
shape_spec = importlib.util.spec_from_file_location("v20_late_window_shape_test_module", SHAPE_SCRIPT)
assert shape_spec and shape_spec.loader
shape = importlib.util.module_from_spec(shape_spec)
sys.modules[shape_spec.name] = shape
shape_spec.loader.exec_module(shape)


def node(frame_idx: int, R: np.ndarray | None = None, t: np.ndarray | None = None, points: np.ndarray | None = None) -> late.PoseNode:
    return late.PoseNode(
        frame_idx,
        np.eye(3) if R is None else R,
        np.zeros(3) if t is None else t,
        points,
        "test" if points is not None else "latent",
        points is not None,
        0.0 if points is not None else 1.0,
        {"frame_idx": frame_idx},
    )


def test_se3_interpolation_is_continuous_constant_twist():
    R0 = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    t0 = np.array([0.2, -0.1, 0.6])
    R1 = Rotation.from_euler("xyz", [1.1, 0.4, -0.2]).as_matrix()
    t1 = np.array([0.8, 0.4, 0.9])
    R_start, t_start = late.interpolate_se3_pose(R0, t0, R1, t1, 0.0)
    R_end, t_end = late.interpolate_se3_pose(R0, t0, R1, t1, 1.0)
    np.testing.assert_allclose(R_start, R0, atol=1e-10)
    np.testing.assert_allclose(t_start, t0, atol=1e-10)
    np.testing.assert_allclose(R_end, R1, atol=1e-10)
    np.testing.assert_allclose(t_end, t1, atol=1e-10)
    Rm, tm = late.se3_interpolate(R0, t0, R1, t1, 0.5)
    assert np.isclose(np.linalg.det(Rm), 1.0)
    assert np.linalg.norm(Rotation.from_matrix(Rm @ R0.T).as_rotvec()) < np.linalg.norm(Rotation.from_matrix(R1 @ R0.T).as_rotvec())
    # Pure translation is exactly linear under the SE(3) exponential.
    _, pure_tm = late.interpolate_se3_pose(np.eye(3), np.zeros(3), np.eye(3), np.array([2.0, 0.0, 0.0]), 0.5)
    np.testing.assert_allclose(pure_tm, [1.0, 0.0, 0.0], atol=1e-10)


def test_world_camera_conversion_roundtrip_uses_metric_contract():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    T[:3, 3] = [0.4, -0.2, 1.3]
    camera = np.array([[0.1, 0.2, 2.0], [-0.2, 0.1, 1.5]])
    world = late.camera_to_world(camera, T)
    np.testing.assert_allclose(late.world_to_camera(world, T), camera, atol=1e-12)


def test_rgb_world_source_to_target_direction_is_explicit():
    Rs = np.eye(3); ts = np.array([0.2, -0.1, 1.0])
    Rt = Rotation.from_euler("y", 0.3).as_matrix(); tt = np.array([-0.4, 0.2, 1.5])
    edge = late.RGBEdge(10, 11, Rt @ Rs.T, tt - (Rt @ Rs.T) @ ts, 1.0)
    rr, tr = late.rgb_edge_residual(edge, {10: (Rs, ts), 11: (Rt, tt)})
    np.testing.assert_allclose(rr, 0.0, atol=1e-10)
    np.testing.assert_allclose(tr, 0.0, atol=1e-10)


def test_existing_no_metric_143_row_is_not_interpolation_source():
    points = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]]
    annotations = {"frames": [
        {"frame_idx": 142, "objects": [{"object_id": "obj", "visible_geometry_candidate": {"world_vertices_sample_m": points}}]},
        {"frame_idx": 143, "objects": [{"object_id": "obj", "visible_geometry_candidate": {}}]},
        {"frame_idx": 144, "objects": [{"object_id": "obj", "visible_geometry_candidate": {"world_vertices_sample_m": points}}]},
    ]}
    R144 = Rotation.from_euler("z", np.pi / 2.0).as_matrix()
    rows = [
        {"frame_idx": 142, "rotation_world_from_completed_canonical_matrix": np.eye(3).tolist(), "translation_world_m": [0.0, 0.0, 0.0]},
        {"frame_idx": 143, "rotation_world_from_completed_canonical_matrix": np.eye(3).tolist(), "translation_world_m": [99.0, 99.0, 99.0]},
        {"frame_idx": 144, "rotation_world_from_completed_canonical_matrix": R144.tolist(), "translation_world_m": [2.0, 0.0, 0.0]},
    ]
    nodes, _, _, _ = late.build_pose_nodes(annotations, {"pose_rows": rows}, {"pose_rows": rows}, "obj", 142, 144)
    middle = next(item for item in nodes if item.frame_idx == 143)
    assert not np.allclose(middle.initial_translation, [99.0, 99.0, 99.0])
    assert "metric_missing_frame_143_latent_uncertain" in middle.observation_source



    R142 = np.eye(3); t142 = np.array([0.0, 0.0, 1.0])
    R144 = Rotation.from_euler("z", np.pi / 2.0).as_matrix(); t144 = np.array([2.0, 0.0, 1.0])
    R143, t143, provenance = late.initialize_missing_se3_pose(143, {142: (R142, t142), 144: (R144, t144)})
    assert "continuous_se3_interpolation_142_144" == provenance
    assert np.linalg.norm(Rotation.from_matrix(R143).as_rotvec()) > 0.0
    assert not np.array_equal(t143, t142)


def test_generated_factors_are_not_in_pose_objective():
    assert "generated" not in inspect.signature(late.pose_residual).parameters
    nodes = [node(120), node(121)]
    residual = late.pose_residual(np.zeros(12), nodes, [], [], 120)
    assert np.isfinite(residual).all()
    assert late.candidate_correction_diagnostics(nodes, np.zeros(12))["small_cumulative_rotation_cap_applied"] is False


def test_anchor_only_gauge_and_non_anchor_rotation_are_distinct():
    nodes = [node(120), node(121)]
    cfg = late.SolverConfig()
    zero = late.pose_residual(np.zeros(12), nodes, [], [], 120, cfg)
    moved = np.zeros(12); moved[6:9] = [0.4, 0.0, 0.0]
    moved_residual = late.pose_residual(moved, nodes, [], [], 120, cfg)
    assert len(zero) == len(moved_residual)
    assert not np.array_equal(zero, moved_residual)
    R, t = late.current_poses(nodes, moved)[120]
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t, np.zeros(3), atol=1e-12)


def test_rotation_allowance_is_broad_without_total_rotation_cap():
    R, t = late.apply_left_correction(np.eye(3), np.zeros(3), np.array([0.0, 0.0, 2.2]), np.zeros(3))
    np.testing.assert_allclose(R, Rotation.from_rotvec([0.0, 0.0, 2.2]).as_matrix(), atol=1e-12)
    assert late.candidate_correction_diagnostics([node(120)], np.r_[np.zeros(3), np.zeros(3)])["broad_rotation_allowance_rad"] >= 2 * np.pi


def test_fixed_initial_gate_and_state_selection_do_not_leak_rejected_candidate():
    current = np.array([1.0, 2.0, 3.0])
    candidate = np.array([9.0, 8.0, 7.0])
    selected = late.select_candidate_state(current, candidate, accepted=False)
    np.testing.assert_array_equal(selected, current)
    accepted, reasons = late.candidate_gate(
        {"observed_surface_abs_median_m": 0.2},
        {"observed_surface_abs_median_m": 0.01},
        {"observed_surface_abs_median_m": 0.01},
        late.SolverConfig(max_metric_degradation_m=0.001),
    )
    assert not accepted
    assert "observed_surface_degradation_vs_current" in reasons
    assert "observed_surface_degradation_vs_immutable_initial" in reasons


def test_signed_front_bias_is_diagnostic_and_gated_against_fixed_initial():
    ok, reasons = late.signed_front_bias_gate(
        {"front_bias_m": 0.03, "front_fraction": 0.8},
        {"front_bias_m": 0.01},
        {"front_bias_m": 0.01},
        max_front_bias_m=0.1,
        max_increase_m=0.005,
    )
    assert not ok
    assert "signed_front_bias_degradation_vs_current" in reasons
    assert "signed_front_bias_degradation_vs_immutable_initial" in reasons


def test_signed_front_per_frame_and_segment_gate_catches_late_tail():
    from types import SimpleNamespace
    factors = {
        140: SimpleNamespace(evaluation_signed_depth=np.array([-0.001, 0.001, 0.002])),
        141: SimpleNamespace(evaluation_signed_depth=np.array([-0.012, -0.010, -0.008])),
        142: SimpleNamespace(evaluation_signed_depth=np.array([-0.014, -0.011, -0.009])),
        143: SimpleNamespace(evaluation_signed_depth=np.array([-0.013, -0.010, -0.007])),
    }
    summary = late.signed_front_bias_diagnostics(factors, negative_threshold_m=0.005)
    assert summary["max_negative_front_segment_length"] == 3
    assert summary["sustained_negative_front_segments"][0]["left_frame"] == 141
    ok, reasons = late.signed_front_bias_gate(
        summary,
        max_front_bias_m=0.1,
        max_front_fraction=1.0,
        max_per_frame_front_bias_m=0.005,
        max_negative_front_segment_length=2,
    )
    assert not ok
    assert any(reason.startswith("signed_front_per_frame_absolute_gate:141") for reason in reasons)
    assert any(reason.startswith("signed_front_sustained_segment_gate:141-143") for reason in reasons)


def test_rgb_npz_json_edges_merge_once_and_keep_json_quality(tmp_path):
    npz_path = tmp_path / "edges.npz"
    json_path = tmp_path / "edges.json"
    R = np.eye(3); t = np.array([0.1, 0.0, 0.0])
    np.savez_compressed(
        npz_path,
        source_frame_idx=np.array([120]), target_frame_idx=np.array([121]),
        rotation_rgb=np.array([R]), translation_rgb_m=np.array([t]),
        accepted=np.array([True]), quality_weight=np.array([0.9]),
    )
    json_path.write_text(json.dumps({"rows": [{
        "source_frame_idx": 120, "target_frame_idx": 121, "status": "accepted",
        "rgb_rotation": R.tolist(), "rgb_translation_m": t.tolist(),
        "quality_weight": 0.2, "inlier_fraction": 1.0,
        "source_3d_conditioning": 1.0,
    }]}))
    edges, diagnostics = late.load_rgb_edges(
        npz_path,
        json_path=json_path,
        pose_by_frame={120: (np.eye(3), np.zeros(3)), 121: (np.eye(3), t)},
        frame_ids={120, 121},
        config=late.SolverConfig(min_rgb_edge_weight=0.01),
    )
    assert len(edges) == 1
    assert np.isclose(edges[0].diagnostics["quality_weight"], 0.2)
    assert edges[0].diagnostics["npz_json_transform_verified"] is True
    assert sum(d.get("status") == "accepted" for d in diagnostics) == 1


def test_rows_outside_window_are_preserved_exactly():
    original = {
        "schema": "base",
        "pose_rows": [
            {"frame_idx": 114, "status": "outside", "custom": {"keep": True}},
            {"frame_idx": 115, "rotation_world_from_completed_canonical_matrix": np.eye(3).tolist(), "translation_world_m": [0.0, 0.0, 0.0], "status": "inside"},
            {"frame_idx": 150, "status": "outside", "custom": [1, 2, 3]},
        ],
    }
    nodes = [node(115)]
    output = late._update_rows(original, nodes, {115: (np.eye(3), np.array([1.0, 0.0, 0.0]))}, 115, 149, {}, None)
    by_id = {row["frame_idx"]: row for row in output["pose_rows"]}
    assert by_id[114] == original["pose_rows"][0]
    assert by_id[150] == original["pose_rows"][2]
    assert by_id[115]["translation_world_m"] == [1.0, 0.0, 0.0]


def test_renderer_converts_camera_points_to_world_once():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 0.25).as_matrix()
    T[:3, 3] = [0.1, -0.2, 0.3]
    camera_points = np.array([[0.2, 0.4, 1.0], [-0.1, 0.3, 0.8]])
    expected = camera_points @ T[:3, :3].T + T[:3, 3]
    np.testing.assert_allclose(renderer.camera_points_to_world(camera_points, T), expected)


def test_conditioning_is_normalized_to_prediction_edge_scale():
    weight, diagnostics = late._conditioning_weight(
        {
            "source_3d_conditioning": 0.03,
            "inlier_fraction": 0.9,
            "reprojection_median_px": 0.8,
            "quality_weight": 1.0,
        },
        full_weight_scale=0.02,
    )
    assert diagnostics["conditioning_ratio"] == 1.0
    assert weight > 0.1


def test_rgb_builder_uses_declared_source_plane_and_multihop_pairs():
    depths = np.ones((2, 4, 6), dtype=np.float32)
    confidence = np.ones_like(depths)
    intrinsics = np.ones((2, 4), dtype=np.float64)
    assert builder.validate_depth_contract(
        np.array([0, 1]), depths, confidence, np.array([6, 4]), intrinsics
    ) == (6, 4)
    pairs = builder.build_edge_pairs([0, 1, 2, 3, 4, 5, 6], [0, 2, 4, 6], 2)
    assert (0, 4) in pairs and (2, 6) in pairs


def test_pose_residual_sparsity_matches_pose_residual_rows():
    nodes = [node(120), node(121), node(122)]
    edge = late.RGBEdge(120, 121, np.eye(3), np.zeros(3), 1.0)
    factor = late.PointFactor(
        0, 1, np.zeros((2, 3)), np.zeros((2, 3)),
        np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)), np.ones(2)
    )
    cfg = late.SolverConfig()
    residual = late.pose_residual(np.zeros(18), nodes, [edge], [factor], 120, cfg)
    sparsity = late.pose_residual_sparsity(nodes, [edge], [factor], 120)
    assert sparsity.shape == (len(residual), 18)
    assert sparsity.nnz > 0


def test_low_conditioning_edge_keeps_translation_but_disables_rotation():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "edges.npz"
        np.savez_compressed(
            p,
            source_frame_idx=np.array([120]),
            target_frame_idx=np.array([121]),
            rotation_source_to_target_world=np.array([np.eye(3)]),
            translation_source_to_target_world_m=np.array([[0.01, 0.0, 0.0]]),
            accepted=np.array([True]),
        )
        edges, diagnostics = late.load_rgb_edges(
            p,
            json_path=None,
            pose_by_frame={120: (np.eye(3), np.zeros(3)), 121: (np.eye(3), np.zeros(3))},
            frame_ids={120, 121},
            config=late.SolverConfig(
                min_rgb_edge_weight=0.01,
                min_rgb_translation_edge_weight=0.01,
                min_rgb_rotation_conditioning=0.01,
            ),
        )
        assert len(edges) == 1
        assert edges[0].diagnostics["rotation_eligible"] is True

        # A companion diagnostic row with an explicitly planar condition is
        # still accepted for translation, but its rotation residual is zeroed.
        j = Path(d) / "edges.json"
        j.write_text(json.dumps({"rows": [{
            "source_frame_idx": 120, "target_frame_idx": 121,
            "status": "accepted", "source_3d_conditioning": 0.001,
            "inlier_fraction": 0.9, "reprojection_median_px": 1.0,
        }]}))
        edges, _ = late.load_rgb_edges(
            p,
            json_path=j,
            pose_by_frame={120: (np.eye(3), np.zeros(3)), 121: (np.eye(3), np.zeros(3))},
            frame_ids={120, 121},
            config=late.SolverConfig(
                min_rgb_edge_weight=0.25,
                min_rgb_translation_edge_weight=0.01,
                min_rgb_rotation_conditioning=0.01,
            ),
        )
        assert len(edges) == 1
        assert edges[0].diagnostics["rotation_eligible"] is False
        assert edges[0].diagnostics["rotation_weight_scale"] == 0.0


def test_optional_rgb_reprojection_factor_is_observed_only_and_sparse():
    nodes = [node(120), node(121)]
    canonical = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]])
    uv = np.column_stack((100.0 * canonical[:, 0] / canonical[:, 2] + 320.0,
                          100.0 * canonical[:, 1] / canonical[:, 2] + 240.0))
    factor = late.RGBReprojectionFactor(
        120, 121, 0, 1, canonical, uv, np.array([100.0, 100.0, 320.0, 240.0]),
        np.eye(3), np.zeros(3), np.ones(len(canonical)), 1.0, True,
    )
    cfg = late.SolverConfig(rgb_reprojection_weight_scale=1.0)
    residual = late.pose_residual(np.zeros(12), nodes, [], [], 120, cfg, [factor])
    np.testing.assert_allclose(residual[:-6], 0.0, atol=1e-10)
    sparsity = late.pose_residual_sparsity(nodes, [], [], 120, [factor])
    assert sparsity.shape == (len(residual), 12)
    assert sparsity.nnz > 0

    low = late.RGBReprojectionFactor(
        120, 121, 0, 1, canonical, uv, np.array([100.0, 100.0, 320.0, 240.0]),
        np.eye(3), np.zeros(3), np.ones(len(canonical)), 1.0, False,
    )
    moved = np.zeros(12); moved[6:9] = [0.3, 0.0, 0.0]
    low_residual = late.pose_residual(moved, nodes, [], [], 120, cfg, [low])
    zero_low = late.pose_residual(np.zeros(12), nodes, [], [], 120, cfg, [low])
    # Correction priors/temporal rows correctly change with the moved state;
    # the optional pixel-evidence block (after the 12 prior rows) must not
    # react to rotation for a low-conditioning factor.
    np.testing.assert_allclose(low_residual[12:20], zero_low[12:20], atol=1e-10)


def test_rgb_reprojection_evidence_loader_validates_flattened_offsets():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "edges_v2.npz"
        pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]])
        uv = np.array([[320.0, 240.0], [330.0, 240.0], [320.0, 250.0], [330.0, 250.0]])
        np.savez_compressed(
            p,
            source_frame_idx=np.array([120]), target_frame_idx=np.array([121]),
            rotation_source_to_target_world=np.array([np.eye(3)]),
            translation_source_to_target_world_m=np.zeros((1, 3)),
            accepted=np.array([True]),
            reprojection_evidence_offsets=np.array([0, 4]),
            reprojection_canonical_points=pts,
            reprojection_target_uv=uv,
            reprojection_weights=np.ones(4),
            reprojection_target_intrinsics=np.array([[100.0, 100.0, 320.0, 240.0]]),
            reprojection_target_camera=np.array([np.eye(4)]),
        )
        nodes = [node(120), node(121)]
        edge = late.RGBEdge(120, 121, np.eye(3), np.zeros(3), 1.0, {"rotation_eligible": True})
        factors, diagnostics = late.load_rgb_reprojection_factors(
            p, edges=[edge], nodes=nodes,
            config=late.SolverConfig(rgb_reprojection_weight_scale=0.5),
        )
        assert diagnostics["available"] is True
        assert diagnostics["point_count"] == 4
        assert len(factors) == 1
        assert factors[0].canonical_points.shape == (4, 3)


def test_disabled_rgb_reprojection_factors_do_not_change_sparsity_rows():
    nodes = [node(120), node(121)]
    canonical = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]])
    factor = late.RGBReprojectionFactor(
        120, 121, 0, 1, canonical, np.zeros((4, 2)),
        np.array([100.0, 100.0, 320.0, 240.0]), np.eye(3), np.zeros(3),
        np.ones(4), 1.0, True,
    )
    cfg = late.SolverConfig(rgb_reprojection_weight_scale=0.0)
    residual = late.pose_residual(np.zeros(12), nodes, [], [], 120, cfg, [factor])
    sparsity = late.pose_residual_sparsity(nodes, [], [], 120, [])
    assert sparsity.shape == (len(residual), 12)


def test_rgb_rotation_observability_marks_translation_only_tail_uncertain():
    nodes = [node(140), node(141), node(142)]
    good = late.RGBEdge(
        140, 141, np.eye(3), np.zeros(3), 1.0,
        {"rotation_eligible": True, "rotation_weight_scale": 1.0, "conditioning_ratio_raw": 0.03},
    )
    weak = late.RGBEdge(
        141, 142, np.eye(3), np.zeros(3), 0.05,
        {"rotation_eligible": False, "rotation_weight_scale": 0.0, "translation_only": True, "conditioning_ratio_raw": 0.001},
    )
    summary = late.rgb_rotation_observability([good, weak], nodes)
    by_frame = {row["frame_idx"]: row for row in summary["per_frame"]}
    assert by_frame[140]["rotation_observability_uncertain"] is False
    assert by_frame[141]["rotation_eligible_edge_count"] == 1
    assert by_frame[142]["rotation_observability_uncertain"] is True
    assert 142 in summary["rotation_observability_uncertain_frames"]


def test_shared_shape_rotation_parameterization_is_explicit_and_bounded():
    points = np.array([[1.0, 0.0, 0.0]])
    rotated = shape.transform_points(
        points, np.zeros(3), np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-10)
    assert shape.parameter_size(type("Args", (), {"optimize_canonical_rotation": True})()) == 9
