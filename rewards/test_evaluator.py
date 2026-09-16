from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np

from actions.space import ActionLayout
from rewards.aggregation import scipy_least_squares_cost
from rewards.evaluator import RewardEvaluator, _axis_angle_matrix
from rewards.profiles import resolve_profile


class SharedFactorReductionTest(unittest.TestCase):
    def test_interval_and_positive_count_normalizers_match_legacy_helpers(self):
        import torch
        from rewards.terms.hand.priors import formal_translation, formal_root_orientation, formal_articulation
        from rewards.terms.interaction.physics import signed_active_penetration, signed_dense_penetration, visible_surface_depth_order
        translation = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        visibility = torch.tensor([1., .25])
        expected_translation = 2.0 * (translation.square() * visibility[:, None]).sum() / (3 * visibility.sum())
        self.assertTrue(torch.allclose(formal_translation(translation, visibility, 2.0), expected_translation))
        root = translation.reshape(2, 1, 3)
        expected_root = 3.0 * (root.square() * visibility[:, None, None]).sum() / (9 * visibility.sum())
        self.assertTrue(torch.allclose(formal_root_orientation(root, visibility, 3.0), expected_root))
        pose = torch.arange(18., dtype=torch.float32).reshape(2, 3, 3)
        pose_visibility = torch.tensor([[1., 0., .5], [.2, .3, 0.]])
        expected_pose = 4.0 * (pose.square() * pose_visibility[..., None]).sum() / (3 * pose_visibility.sum())
        self.assertTrue(torch.allclose(formal_articulation(pose, pose_visibility, 4.0), expected_pose))
        moved = torch.tensor([[0., 0., 0.], [0., 0., .8], [0., 0., 2.]])
        normals = torch.tensor([[0., 0., 1.]]).expand(3, -1)
        depth = torch.ones(3)
        residual = torch.relu(depth - .1 - moved[:, 2])
        self.assertTrue(torch.allclose(signed_active_penetration(moved, normals, depth, .1, 5.),
                                       5. * residual.square().mean()))
        self.assertTrue(torch.allclose(signed_dense_penetration(moved, normals, depth, .1, 5.),
                                       5. * residual.square().sum() / (residual > 0).sum()))
        camera_depth = torch.tensor([.5, 1., 1.5])
        surface_depth = torch.ones(3)
        order = torch.relu(surface_depth - .1 - camera_depth)
        self.assertTrue(torch.allclose(visible_surface_depth_order(camera_depth, surface_depth, .1, 7.),
                                       7. * order.square().sum() / (order > 0).sum()))

    def test_scipy_half_rho_convention(self):
        residual = np.asarray([0.0, 1.0, -2.0])
        self.assertAlmostEqual(scipy_least_squares_cost(residual, "soft_l1"),
            float(np.sum(np.sqrt(1.0 + residual ** 2) - 1.0)))
        self.assertAlmostEqual(scipy_least_squares_cost(residual, "linear"),
            0.5 * float(np.square(residual).sum()))


class RewardEvaluatorSemanticsTest(unittest.TestCase):
    @staticmethod
    def _hand(count: int, valid: np.ndarray, row_visibility: np.ndarray) -> dict:
        vertices = np.zeros((count, 778, 3), dtype=np.float32)
        vertices[..., 2] = 1.0
        vertices[..., 0] = np.arange(778, dtype=np.float32)[None, :] * 1e-4
        joints = vertices[:, :21].copy()
        joint_visibility = np.linspace(.1, 1.0, 21, dtype=np.float32)[None, :] * valid[:, None]
        pose_visibility = np.linspace(.2, 1.0, 15, dtype=np.float32)[None, :] * valid[:, None]
        return {
            "valid": valid.copy(), "root_orient": np.zeros((count, 3), np.float32),
            "articulation": np.zeros((count, 45), np.float32),
            "translation": np.zeros((count, 3), np.float32),
            "vertices": vertices, "joints": joints,
            "hand_observation_visibility_weight_multiplier": row_visibility.astype(np.float32),
            "joint_visibility": joint_visibility, "pose_visibility": pose_visibility,
            "hand_ray_shift_prior_world_m": np.zeros((count, 3), np.float32),
            "hand_ray_shift_prior_weight": np.zeros(count, np.float32),
            "depth_order_vertex_ids": tuple(np.zeros(0, np.int64) for _ in range(count)),
            "depth_order_surface_depth_m": tuple(np.zeros(0, np.float64) for _ in range(count)),
            "depth_order_state": tuple("inactive_no_supported_projected_vertices" for _ in range(count)),
        }

    @classmethod
    def _state(cls, count: int = 8, *, object_valid: bool = False, contact: bool = True):
        frames = np.arange(count, dtype=np.int64)
        valid_left = np.asarray([True, True, True, True, False, True, True, True][:count])
        valid_right = np.zeros(count, dtype=bool)
        visibility = np.asarray([1., .25, .5, .75, 0., .2, 1., .4][:count], dtype=np.float32)
        left = cls._hand(count, valid_left, visibility)
        right = cls._hand(count, valid_right, np.zeros(count, np.float32))
        observed = tuple(np.zeros((0, 3), np.float64) for _ in range(count))
        valid_obj = np.full(count, object_valid, dtype=bool)
        obj = {
            "entity_id": "object:0", "valid": valid_obj, "present": valid_obj,
            "metric_observation_eligible": valid_obj,
            "rotation": np.repeat(np.eye(3)[None], count, axis=0),
            "translation": np.zeros((count, 3), np.float64),
            "observed_points_world": observed, "watertight": False,
            "signed_geometry_ready": False, "signed_authority": np.zeros(0, bool),
            "faces": np.zeros((0, 3), np.int64), "vertices_canonical": np.zeros((0, 3)),
        }
        contacts = {}
        if contact and count >= 2:
            for frame, prior, weight, distance in ((0, .4, 2.0, .002), (1, .3, 4.0, .004)):
                ids = np.asarray([0, 1, 2], np.int64)
                contacts[(frame, "left", "object:0")] = {
                    "target_entity_id": "object:0", "state": "active_contact_patch",
                    "membership_state": "active_frozen_source_patch", "prior": prior,
                    "weight": weight, "band_m": .02, "margin_m": .01,
                    "support_uncertainty_m": 0.0, "vertex_ids": ids,
                    "targets_object": left["vertices"][frame, ids].astype(np.float64),
                    "normals_object": np.tile(np.asarray([[0., 0., 1.]]), (3, 1)),
                    "distances_m": np.full(3, distance),
                    "initial_contact_geometry_target_probability": .7 if frame == 0 else .2,
                }
        layout = ActionLayout(object_slots=("object:0",), contact_variables_per_hand=1)
        episode = SimpleNamespace(frame_count=count, frame_idx=frames, time_s=frames / 30.,
            T_world_camera=np.repeat(np.eye(4)[None], count, axis=0),
            intrinsics=np.repeat(np.asarray([[100., 100., 0., 0.]]) , count, axis=0),
            hands={"left": left, "right": right}, objects={"object:0": obj}, contact=contacts,
            factor_rows={}, layout=layout)
        translation_delta = np.column_stack([
            np.asarray([0., .01, .03, .06, 0., .02, .07, .08][:count]),
            np.zeros(count), np.asarray([.015, .02, .025, .03, 0., .012, .014, .016][:count])])
        hand_translation = {"left": translation_delta.copy(), "right": np.zeros((count, 3))}
        hand_vertices = {"left": left["vertices"] + translation_delta[:, None, :],
                         "right": right["vertices"].copy()}
        hand_joints = {"left": left["joints"] + translation_delta[:, None, :],
                       "right": right["joints"].copy()}
        contact_logits = np.zeros((count, 2, 1), np.float32)
        if count >= 2:
            contact_logits[0, 0, 0] = np.log(.8 / .2)
            contact_logits[1, 0, 0] = np.log(.6 / .4)
        state = SimpleNamespace(episode=episode,
            hand_root={"left": np.zeros((count, 3)), "right": np.zeros((count, 3))},
            hand_articulation={"left": np.zeros((count, 45)), "right": np.zeros((count, 45))},
            hand_translation=hand_translation, hand_vertices=hand_vertices, hand_joints=hand_joints,
            object_rotation={"object:0": np.repeat(np.eye(3)[None], count, axis=0)},
            object_translation={"object:0": np.zeros((count, 3))},
            contact_logits=contact_logits, term_cache={}, previous_objective=0.0, ledger=None)
        return state, translation_delta, visibility, valid_left

    def test_profile_defaults_preserve_formal_units_and_weights(self):
        profile = resolve_profile({})
        self.assertEqual(profile.name, "formal_annotation")
        expected = {
            "translation_prior_weight": 2000., "root_prior_weight": 150., "pose_prior_weight": 75.,
            "smooth_weight": 5000., "accel_weight": 10000.,
            "observed_penetration_weight": 300000., "dense_observed_penetration_weight": 300000.,
            "visible_hinge_weight": 800., "depth_hinge_weight": 20000., "bound_hinge_weight": 3000.,
            "max_translation_m": .045, "max_root_delta_rad": .3, "max_pose_delta_rad": .45,
            "max_object_translation_m": .015, "object_translation_prior_weight": 4000.,
            "object_smooth_weight": 8000., "penetration_epsilon_m": 1e-5,
            "max_constraints_per_frame": 96, "max_visible_surface_depth_vertices": 160,
            "contact_state_prior_residual_scale_m": .01,
        }
        for key, value in expected.items():
            self.assertEqual(profile.get(key), value, key)

    def test_actual_evaluator_uses_measured_visibility_and_global_temporal_counts(self):
        state, delta, visibility, valid = self._state()
        evaluator = RewardEvaluator({"reward_profile": "formal_interval",
            "contact_state_geometry_likelihood": True}, "cpu")
        ledger = evaluator.objective(state, state.episode.frame_count)
        expected_prior = 2000. * np.sum(delta ** 2 * visibility[:, None]) / (3. * visibility.sum())
        self.assertAlmostEqual(ledger.terms["hand/left/translation_prior"], expected_prior, places=5)

        edge_ids = [(0, 1), (1, 2), (2, 3), (5, 6), (6, 7)]
        edge = np.stack([delta[b] - delta[a] for a, b in edge_ids])
        expected_velocity = 5000. * float(np.square(edge).mean())
        self.assertAlmostEqual(ledger.terms["hand/left/translation_velocity"], expected_velocity, places=5)
        self.assertEqual(ledger.statistics["hand/left/translation_velocity"]["active_count"], 5)
        triple_ids = [(0, 1, 2), (1, 2, 3), (5, 6, 7)]
        triples = np.stack([delta[c] - 2 * delta[b] + delta[a] for a, b, c in triple_ids])
        expected_acceleration = 10000. * float(np.square(triples).mean())
        self.assertAlmostEqual(ledger.terms["hand/left/translation_acceleration"], expected_acceleration, places=5)
        self.assertEqual(ledger.statistics["hand/left/translation_acceleration"]["active_count"], 3)

        strength = np.asarray([2., 4.]) * .01 ** 2
        expected_contact = float(np.sum(strength * np.square(np.asarray([.8, .6]) - np.asarray([.4, .3]))) / 2.)
        self.assertAlmostEqual(ledger.terms["contact/left/state_prior"], expected_contact, places=7)
        expected_continuity = float(.5 * (strength[0] + strength[1]) * (.6 - .8) ** 2)
        self.assertAlmostEqual(ledger.terms["contact/left/state_continuity"], expected_continuity, places=7)
        self.assertEqual(ledger.activation["contact/left/geometry_likelihood"],
                         "active_configured_frozen_source_geometry_target")
        expected_geometry = float(np.sum(strength * np.square(np.asarray([.8, .6]) - np.asarray([.7, .2]))) / 2.)
        self.assertAlmostEqual(ledger.terms["contact/left/geometry_likelihood"], expected_geometry, places=7)

    def test_prefix_ledger_telescopes_from_valid_empty_baseline(self):
        state, _, _, _ = self._state()
        evaluator = RewardEvaluator({"reward_profile": "formal_interval"}, "cpu")
        rewards = []
        final = None
        for end in range(1, state.episode.frame_count + 1):
            reward, final = evaluator.transition(state, end)
            rewards.append(reward)
        self.assertIsNotNone(final)
        self.assertAlmostEqual(sum(rewards), -final.objective, places=5)
        self.assertEqual(evaluator.objective(state, 0).objective, 0.0)
        self.assertIn("J(empty)=0", final.document()["return_convention"])

    def test_formal_object_sigma_gap_temporal_and_soft_target_match_closure(self):
        state, _, _, _ = self._state(count=3, object_valid=True, contact=False)
        obj = state.episode.objects["object:0"]
        obj["translation_sigma_m"] = np.asarray([.02, .025, .04])
        obj["rotation_sigma_rad"] = np.asarray([.1, .2, .25])
        obj["nonpenetration_target_world_m"] = np.zeros((3, 3))
        obj["nonpenetration_weight"] = np.asarray([.5, 0., 2.])
        trans = np.asarray([[.01, 0., 0.], [.02, -.01, 0.], [.025, -.005, .01]])
        rotvec = np.asarray([[.02, 0., 0.], [.05, .01, 0.], [.09, .01, -.02]])
        state.object_translation["object:0"] = trans.copy()
        state.object_rotation["object:0"] = np.stack([_axis_angle_matrix(value) for value in rotvec])
        evaluator = RewardEvaluator({"reward_profile": "formal_object", "object_observed_terms": False}, "cpu")
        ledger = evaluator.objective(state, 3)
        expected_t = scipy_least_squares_cost((trans / obj["translation_sigma_m"][:, None]).reshape(-1), "soft_l1")
        expected_r = scipy_least_squares_cost((rotvec / obj["rotation_sigma_rad"][:, None]).reshape(-1), "soft_l1")
        self.assertAlmostEqual(ledger.terms["object/object:0/translation_sigma_prior"], expected_t, places=6)
        self.assertAlmostEqual(ledger.terms["object/object:0/rotation_sigma_prior"], expected_r, places=5)
        target_residual = np.concatenate([np.sqrt(.5) * trans[0] / .03, np.sqrt(2.) * trans[2] / .03])
        self.assertAlmostEqual(ledger.terms["object/object:0/nonpenetration_translation_target"],
                               scipy_least_squares_cost(target_residual, "soft_l1"), places=6)
        velocity = np.concatenate([(trans[1] - trans[0]) / .01, (trans[2] - trans[1]) / .01])
        self.assertAlmostEqual(ledger.terms["object/object:0/translation_velocity"],
                               scipy_least_squares_cost(velocity, "soft_l1"), places=6)

    def test_visual_fixed_scale_profile_selects_unsigned_pad_without_formal_double_counting(self):
        state, _, _, _ = self._state(count=1, object_valid=True, contact=False)
        left = state.episode.hands["left"]
        ids = np.asarray([0, 1, 2], np.int64)
        row = {"frame_idx": 0, "hand_side": "left", "target_entity_id": "object:0",
               "vertex_ids": ids, "targets_object": left["vertices"][0, ids].astype(float), "weight": 2.0}
        state.episode.factor_rows = {(0, "left", "object:0", "visual_unsigned_pad"): (row,)}
        ledger = RewardEvaluator({"reward_profile": "visual_joint_fixed_scale"}, "cpu").objective(state, 1)
        self.assertIn("visual/unsigned_pad/factor=0", ledger.terms)
        self.assertIn("visual/hand/left/translation_prior", ledger.terms)
        self.assertNotIn("hand/left/translation_prior", ledger.terms)
        delta = state.hand_vertices["left"][0, ids] - row["targets_object"]
        expected = 1.7 * (2.0 * np.sort(np.sum(delta ** 2, axis=1))[:3].mean() / .003 ** 2) / 2.0
        self.assertAlmostEqual(ledger.terms["visual/unsigned_pad/factor=0"], expected, places=4)

    def test_declared_first_hit_and_silhouette_use_formal_object_robust_reduction(self):
        from rewards.terms.object.geometry import first_hit_depth
        from rewards.terms.object.image import silhouette_endpoint_residual
        state, _, _, _ = self._state(count=1, object_valid=True, contact=False)
        obj = state.episode.objects["object:0"]
        obj["translation_sigma_m"] = np.asarray([.02])
        obj["rotation_sigma_rad"] = np.asarray([.1])
        state.object_translation["object:0"][0] = np.asarray([.01, 0., .02])
        depth_points = np.asarray([[0., 0., 1.], [.01, 0., 1.]])
        observed_z = np.asarray([.98, 1.01])
        depth_weight = np.asarray([1., .25])
        silhouette_points = np.asarray([[0., 0., 1.], [.02, 0., 1.]])
        targets = np.asarray([[1., 0.], [3., 0.]])
        row = {"frame_idx": 0, "target_entity_id": "object:0",
               "depth_points_canonical": depth_points, "depth_observed_z": observed_z,
               "depth_weight": depth_weight, "silhouette_points_canonical": silhouette_points,
               "silhouette_target_uv": targets, "T_world_camera": np.eye(4),
               "K_raster": np.asarray([100., 100., 0., 0.])}
        state.episode.factor_rows = {(0, None, "object:0", "object_image"): (row,)}
        ledger = RewardEvaluator({"reward_profile": "formal_object", "object_observed_terms": False}, "cpu").objective(state, 1)
        camera = depth_points + state.object_translation["object:0"][0]
        first = first_hit_depth(camera[:, 2], observed_z, depth_weight, sigma=.008, clip=.03, weight=1.)
        world = silhouette_points + state.object_translation["object:0"][0]
        uv = np.column_stack([100. * world[:, 0] / world[:, 2], 100. * world[:, 1] / world[:, 2]])
        silhouette = silhouette_endpoint_residual(uv, targets, sigma=4., clip=16., weight=1.)
        self.assertAlmostEqual(ledger.terms["object/object:0/image/factor=0/first_hit"],
                               scipy_least_squares_cost(first, "soft_l1"), places=6)
        self.assertAlmostEqual(ledger.terms["object/object:0/image/factor=0/silhouette"],
                               scipy_least_squares_cost(silhouette, "soft_l1"), places=6)

    def test_declared_rgb_rotation_and_reprojection_are_evaluated(self):
        state, _, _, _ = self._state(count=2, object_valid=True, contact=False)
        obj = state.episode.objects["object:0"]
        obj["translation_sigma_m"] = np.asarray([.02, .02])
        obj["rotation_sigma_rad"] = np.asarray([.1, .1])
        state.object_rotation["object:0"][1] = _axis_angle_matrix(np.asarray([0., .02, 0.]))
        state.object_translation["object:0"][1] = np.asarray([.01, 0., 0.])
        edge = {"frame_idx": 1, "target_entity_id": "object:0", "source_frame_idx": 0,
                "target_frame_idx": 1, "rotation_source_to_target_world": np.eye(3),
                "translation_source_to_target_world_m": np.zeros(3), "weight": 1.0}
        reprojection = {"frame_idx": 1, "target_entity_id": "object:0", "source_frame_idx": 0,
                "target_frame_idx": 1, "source_world": np.asarray([[0., 0., 1.], [.01, 0., 1.]]),
                "target_uv": np.asarray([[0., 0.], [1., 0.]]), "weights": np.ones(2),
                "target_camera_rotation": np.eye(3), "target_camera_translation": np.zeros(3),
                "target_intrinsics": np.asarray([100., 100., 0., 0.]), "edge_weight": 1.0,
                "rotation_eligible": True}
        state.episode.factor_rows = {
            (1, None, "object:0", "object_rgb_edge"): (edge,),
            (1, None, "object:0", "object_rgb_reprojection"): (reprojection,),
        }
        ledger = RewardEvaluator({"reward_profile": "formal_object", "object_observed_terms": False}, "cpu").objective(state, 2)
        self.assertIn("object/object:0/rgb_edge/factor=0/rotation", ledger.terms)
        self.assertIn("object/object:0/rgb_edge/factor=0/translation", ledger.terms)
        self.assertIn("object/object:0/rgb_reprojection/factor=0", ledger.terms)
        self.assertEqual(ledger.activation["object/object:0/rgb_edge/factor=0"], "active_declared_rgb_edge")
        self.assertEqual(ledger.activation["object/object:0/rgb_reprojection/factor=0"],
                         "active_declared_rgb_reprojection")

    def test_legacy_visual_profile_name_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "renamed to 'visual_joint_fixed_scale'"):
            resolve_profile({"reward_profile": "visual_joint"})

    def test_valid_hand_requires_complete_measured_visibility_arrays(self):
        for name in ("hand_observation_visibility_weight_multiplier", "joint_visibility", "pose_visibility"):
            state, _, _, _ = self._state(count=2, contact=False)
            del state.episode.hands["left"][name]
            with self.subTest(missing=name):
                with self.assertRaisesRegex(ValueError, name):
                    RewardEvaluator({"reward_profile": "formal_interval"}, "cpu").objective(state, 2)
        bad_shapes = {
            "hand_observation_visibility_weight_multiplier": np.ones(1),
            "joint_visibility": np.ones((2, 20)),
            "pose_visibility": np.ones((2, 14)),
        }
        for name, bad in bad_shapes.items():
            state, _, _, _ = self._state(count=2, contact=False)
            state.episode.hands["left"][name] = bad
            with self.subTest(malformed=name):
                with self.assertRaisesRegex(ValueError, name):
                    RewardEvaluator({"reward_profile": "formal_interval"}, "cpu").objective(state, 2)

    def test_pose_valid_object_keeps_pose_factors_when_metric_surface_is_ineligible(self):
        state, _, _, _ = self._state(count=2, object_valid=True, contact=False)
        obj = state.episode.objects["object:0"]
        obj["present"] = np.asarray([True, True])
        obj["pose_valid"] = np.asarray([True, True])
        obj["metric_observation_eligible"] = np.asarray([False, False])
        obj["translation_sigma_m"] = np.asarray([.02, .02])
        obj["rotation_sigma_rad"] = np.asarray([.1, .1])
        state.object_translation["object:0"] = np.asarray([[.01, 0., 0.], [.03, -.01, 0.]])
        ledger = RewardEvaluator({"reward_profile": "formal_object"}, "cpu").objective(state, 2)
        self.assertGreater(ledger.terms["object/object:0/translation_sigma_prior"], 0.)
        self.assertGreater(ledger.terms["object/object:0/translation_velocity"], 0.)
        self.assertEqual(ledger.activation["object/object:0/observed_nn"],
                         "inactive_fewer_than_two_metric_eligible_rows")

    def test_sparse_row_sigma_and_nonpenetration_apply_only_declared_pose_rows(self):
        state, _, _, _ = self._state(count=2, object_valid=True, contact=False)
        state.object_translation["object:0"] = np.asarray([[.01, 0., 0.], [.03, 0., 0.]])
        state.episode.factor_rows = {
            (0, None, "object:0", "object_pose_observation"): ({
                "frame_idx": 0, "target_entity_id": "object:0",
                "translation_sigma_m": .02, "rotation_sigma_rad": .1,
                "nonpenetration_target_world_m": [0., 0., 0.], "nonpenetration_weight": .5,
            },),
            (1, None, "object:0", "object_pose_observation"): ({
                "frame_idx": 1, "target_entity_id": "object:0",
            },),
        }
        ledger = RewardEvaluator({"reward_profile": "formal_object", "object_observed_terms": False}, "cpu").objective(state, 2)
        expected_sigma = scipy_least_squares_cost(np.asarray([.01, 0., 0.]) / .02, "soft_l1")
        self.assertAlmostEqual(ledger.terms["object/object:0/translation_sigma_prior"], expected_sigma, places=6)
        expected_target = scipy_least_squares_cost(np.sqrt(.5) * np.asarray([.01, 0., 0.]) / .03, "soft_l1")
        self.assertAlmostEqual(ledger.terms["object/object:0/nonpenetration_translation_target"], expected_target, places=6)
        self.assertGreater(ledger.terms["object/object:0/translation_velocity"], 0.0)

    def test_active_object_image_row_without_subfamily_payload_fails(self):
        state, _, _, _ = self._state(count=1, object_valid=True, contact=False)
        obj = state.episode.objects["object:0"]
        obj["translation_sigma_m"] = np.asarray([.02])
        obj["rotation_sigma_rad"] = np.asarray([.1])
        state.episode.factor_rows = {(0, None, "object:0", "object_image"): ({"frame_idx": 0,
            "target_entity_id": "object:0", "state": "active_object_image"},)}
        with self.assertRaisesRegex(ValueError, "active object image row requires"):
            RewardEvaluator({"reward_profile": "formal_object", "object_observed_terms": False}, "cpu").objective(state, 1)

    def test_partial_sigma_and_nonpenetration_declarations_fail(self):
        state, _, _, _ = self._state(count=1, object_valid=True, contact=False)
        state.episode.objects["object:0"]["translation_sigma_m"] = np.asarray([.02])
        with self.assertRaisesRegex(ValueError, "partial sigma declaration"):
            RewardEvaluator({"reward_profile": "formal_object"}, "cpu").objective(state, 1)
        state, _, _, _ = self._state(count=1, object_valid=True, contact=False)
        state.episode.objects["object:0"]["translation_sigma_m"] = np.asarray([.02])
        state.episode.objects["object:0"]["rotation_sigma_rad"] = np.asarray([.1])
        state.episode.objects["object:0"]["nonpenetration_target_world_m"] = np.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, "partial nonpenetration declaration"):
            RewardEvaluator({"reward_profile": "formal_object"}, "cpu").objective(state, 1)

    def test_configured_missing_hand_ray_evidence_fails_actionably(self):
        state, _, _, _ = self._state()
        del state.episode.hands["left"]["hand_ray_shift_prior_world_m"]
        evaluator = RewardEvaluator({"reward_profile": "formal_interval", "use_hand_ray_shift_prior": True}, "cpu")
        with self.assertRaisesRegex(ValueError, "hand-ray prior requested"):
            evaluator.objective(state, 1)


if __name__ == "__main__":
    unittest.main()
