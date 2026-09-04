#!/usr/bin/env python3
"""CPU contract and synthetic-regression tests for the pose-repair experiment."""
from __future__ import annotations
import importlib.util
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ROOT = "/mnt/user-home/kupingxin/ego_annotation_worktrees/milk_pose_repair_experiment_20260904"
solver = load("pose_repair_solver_test", ROOT + "/scripts/solve_v19_rigid_object_pose_graph.py")
global_pose = load("global_pose_test", ROOT + "/scripts/fit_v19_global_observed_pose_graph.py")
builder = load("factor_builder_test", ROOT + "/experiments/sam3d_native_ghost_lite_20260826/build_p15_first_hit_silhouette_factors.py")

# Half-pixel transforms compose exactly: calibration -> actual RGB/mask -> factor raster.
K = np.asarray([[974.3447, 0, 702.6607], [0, 974.3447, 706.1102], [0, 0, 1]], float)
two = builder.resize_intrinsics_xy(builder.resize_intrinsics_xy(K, (1408, 1408), (960, 960)), (960, 960), (256, 256))
one = builder.resize_intrinsics_xy(K, (1408, 1408), (256, 256))
assert np.allclose(two, one, atol=1e-10)

# Camera roundtrip under the row-vector contract.
rng = np.random.default_rng(7)
R = Rotation.from_rotvec(rng.normal(size=3) * .2).as_matrix()
t = rng.normal(size=3)
T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
p = rng.normal(size=(100, 3)); p[:, 2] += 2
c = builder.world_to_camera(p, T); w = builder.camera_to_world(c, T)
assert np.max(np.abs(p - w)) < 1e-10

# MANO triangle silhouette must produce bounded unknown support.
tri = np.asarray([[0, 1, 2]], dtype=np.int64)
hands = {"frame_idx": np.asarray([0]), "vertices_current_v18_world_from_hawor_projection_relift_m": np.asarray([[[[-.1,-.1,1.0],[.1,-.1,1.0],[0,.1,1.0]]]])[0]}
# Normalize the synthetic archive shape to (N,778,3).
v = np.zeros((1, 778, 3), dtype=np.float64); v[0, :3] = [[-.1,-.1,1.0],[.1,-.1,1.0],[0,.1,1.0]]
hands["vertices_current_v18_world_from_hawor_projection_relift_m"] = v
unknown = builder.build_hand_unknown_mask(hands, {0:[0]}, tri, 0, np.eye(4), np.asarray([[100.,0,128.],[0,100.,128.],[0,0,1.]]), 256, 1)
assert 0 < int(unknown.sum()) < 256 * 256

# Synthetic P15 image block has the expected residual/sparsity dimensions.
obs_pts = p[:100] * .03 + np.asarray([0,0,1.0]); obs = solver.PoseObservation(0, {}, np.eye(3), np.zeros(3), .02, .1, len(obs_pts), obs_pts, None, 0., 0)
factor = solver.ImageFactorObservation(0, np.eye(4), one, np.asarray([[0.,0.,1.]]), np.asarray([1.]), np.asarray([.5]), np.asarray([[0.,0.,1.]]), np.asarray([[128.,128.]]), np.asarray([0], dtype=np.int8))
a = SimpleNamespace(image_first_hit_weight=1., image_silhouette_weight=1., max_image_first_hit_residual_m=.03, sigma_image_first_hit_m=.008, max_image_silhouette_residual_px=16., sigma_image_silhouette_px=4., sigma_translation_delta_step_m=.01, sigma_rotation_delta_step_rad=.08, sigma_translation_delta_accel_m=.006, sigma_rotation_delta_accel_rad=.05)
r = solver.residual_vector(np.zeros(6), [obs], a, {0:factor})
J = solver.residual_sparsity([obs], {0:factor}, a)
assert J.shape == (len(r), 6) and np.isfinite(r).all()

# Synthetic global graph residual/sparsity remains finite and shape-consistent.
states=[]
for i in range(3):
    Ri=Rotation.from_euler("z", i*.03).as_matrix(); ti=np.asarray([i*.002,0.,0.])
    q=global_pose.apply_pose(p[:80], Ri, ti)
    s=global_pose.FrameState(i, {}, Ri, ti, q, p[:80].copy(), 1., 0., p[:20].copy(), q[:20].copy(), np.ones(20))
    states.append(s)
edges=[]
for i in range(2):
    D=states[i+1].rotation0 @ states[i].rotation0.T; dt=states[i+1].translation0-D@states[i].translation0
    edges.append(global_pose.SurfaceEdge(i,i+1,i,i+1,p[:20],p[:20],np.ones(20),D,dt,1.,1,"local"))
g = SimpleNamespace(sigma_pose_prior_translation_m=.035,sigma_pose_prior_rotation_rad=.18,sigma_anchor_point_m=.012,sigma_edge_point_m=.01,sigma_edge_rotation_rad=.1,sigma_edge_translation_m=.018,sigma_rgb_rotation_rad=.08,sigma_rgb_translation_m=.015,sigma_correction_translation_step_m=.012,sigma_correction_rotation_step_rad=.1,sigma_correction_translation_accel_m=.008,sigma_correction_rotation_accel_rad=.06,sigma_anchor_gauge_translation_m=1e-6,sigma_anchor_gauge_rotation_rad=1e-6,anchor_position=0,max_anchor_residual_m=.06,max_edge_residual_m=.06)
gr=global_pose.residual_vector(np.zeros(18), states, edges, [], g); gj=global_pose.residual_sparsity(states, edges, [], g)
assert gj.shape == (len(gr), 18) and np.isfinite(gr).all()
print("pose_repair_self_test: PASS")
