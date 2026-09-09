"""Full-topology native-gauge and image-scale kernels extracted from visual fitting."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np,trimesh
from v19_runtime_contracts import load,sha
from v19_demo_timeline import validate_transforms
from rgb_scene_pose_io import pose_from_values,RKEY,TKEY

def _pose(row): return pose_from_values(row[RKEY],row[TKEY])

AUTHORITY = {"demo_only": True,"annotation_ready": False,"physical_authority": False,
    "demo_visual_proximity_fit_eligible": True,"physical_contact_authority_eligible": False,
    **{f"generated_faces_{role}_eligible":False for role in ('contact','collision','sign','sdf','nonpenetration')}}

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}

def mesh_arrays(path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("one complete triangular mesh required")
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError("invalid complete vertices/topology")
    return v, f.astype(np.int64)

def write_mesh(path, vertices, faces):
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
    v, f = mesh_arrays(path)
    if not np.array_equal(f, faces) or v.shape != vertices.shape or not np.allclose(v, vertices, atol=1e-6, rtol=0):
        raise ValueError("serialized mesh changed topology/vertex order or precision")
    return {
        "vertex_count": len(v), "face_count": len(f),
        "face_index_sha256": hashlib.sha256(np.ascontiguousarray(f, dtype=np.int64).tobytes()).hexdigest(),
        "vertices_float64_sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
        "faces_exactly_preserved": True, "serialized_vertex_tolerance_m": 1e-6,
    }

def canonical_gauge(native_cv, scale, T, pose):
    T = validate_transforms([T])[0]
    R, t = _pose(pose)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid scene similarity scale")
    world = (native_cv * scale) @ T[:3, :3].T + T[:3, 3]
    canonical = (world - t) @ R
    replay = ((canonical @ R.T + t - T[:3, 3]) @ T[:3, :3]) / scale
    error = float(np.max(np.abs(replay - native_cv)))
    if error > 1e-5:
        raise ValueError("native/canonical gauge roundtrip failed")
    # Column-vector homogeneous similarity; vertices above use its transpose.
    transform = np.eye(4)
    transform[:3, :3] = scale * R.T @ T[:3, :3]
    transform[:3, 3] = R.T @ (T[:3, 3] - t)
    return canonical, {"T_original_canonical_from_native_opencv": transform.tolist(),
                       "T_world_camera_metric": T.tolist(), "initializer_pose": pose,
                       "initializer_pose_sha256": digest_json(pose),
                       "roundtrip_max_abs_native_scene_error": error,
                       "formula": "Vcanonical=((s*Vnative)@Rcamera.T+tcamera-tinitializer)@Rinitializer"}

def scene_similarity(native_cv, observed_cam, K, mask):
    # Reuse the unchanged P13 front-depth numerical implementation, not its
    # core owned-mask producer/lineage or any face-authority contract.
    from run_rgb_scene_reconstruction import module
    bridge = module(Path(__file__).resolve().parents[1] / "experiments/sam3d_p11_p12_branch/build_p13_sam3d_native_metric_bridge.py", "demo_native_depth_numerics")
    projection = bridge.convex_projection_and_front_depth(native_cv, K, mask)
    if (projection["positive_depth_vertex_fraction"] < .99
            or projection["owned_zbuffer_pixel_count"] < 100
            or projection["convex_projection_iou"] < .05):
        raise ValueError("native positive/projected front-depth support insufficient")
    depth = float(np.median(observed_cam[:, 2]))
    scale = depth / projection["owned_zbuffer_median_native_depth"]
    if not np.isfinite(scale) or not .05 <= scale <= 5.:
        raise ValueError("scene similarity outside [0.05,5.0]")
    return scale, {"scene_similarity_scale": scale, "bounds": [.05, 5.],
                   "metric_depth_median_m": depth,
                   "metric_depth_p05_p95_m": np.percentile(observed_cam[:, 2], [5, 95]).tolist(),
                   "metric_surfel_count": len(observed_cam), "native_front_projection": projection,
                   "uncertainty": "single_view_native_front_depth_and_retained_monocular_metric_surfel_uncertainty"}

def silhouette(vertices, faces, K, shape):
    if np.any(vertices[:, 2] <= 1e-5) or not np.isfinite(vertices).all():
        raise ValueError("scale hypothesis crosses camera near plane")
    uv = vertices[:, :2] / vertices[:, 2, None] * K[:2] + K[2:]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e7:
        raise ValueError("unsafe projection")
    triangles = np.rint(uv[faces]).astype(np.int32)
    result = np.zeros(shape, np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(result, triangle, 1)
    return result.astype(bool)

def estimate_scale(vertices, faces, observations, bounds=(.8, 1.2), samples=41):
    """Mean triangle-silhouette 1-IoU, fixed poses/K and vertex-mean pivot.

    Observations bind an image mask, K, world-from-camera T and initializer R/t.
    Every triangle is retained; the low-resolution raster is an image objective,
    not reconstructed geometry or contact evidence.
    """
    bounds = np.asarray(bounds, float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]
            or type(samples) is not int or not 3 <= samples <= 101):
        raise ValueError("unsafe uniform scale bounds/grid")
    if not observations:
        raise ValueError("at least one appearance frame required for scale alignment")
    pivot = vertices.mean(axis=0)
    grid = np.unique(np.r_[np.linspace(*bounds, samples), 1.])
    def objective(scale):
        scaled = (vertices - pivot) * scale + pivot
        errors = []
        for obs in observations:
            world = scaled @ obs["R"].T + obs["t"]
            T = obs["T"]
            camera = (world - T[:3, 3]) @ T[:3, :3]
            pred = silhouette(camera, faces, obs["K"], obs["mask"].shape)
            target = obs["mask"]
            union = np.count_nonzero(pred | target)
            errors.append(1 - np.count_nonzero(pred & target) / max(1, union))
        return float(np.mean(errors))
    values = [objective(float(value)) for value in grid]
    initial = objective(1.)
    # Deterministic ties prefer the least intervention, then the smaller scale.
    best = min(range(len(grid)), key=lambda i: (values[i], abs(grid[i] - 1), grid[i]))
    scale = float(grid[best])
    return {
        "schema": "v19_demo_image_uniform_scale_v1", **AUTHORITY,
        "uniform_scale": scale, "bounds": bounds.tolist(), "pivot_canonical_m": pivot.tolist(),
        "selection_source": "deterministic_multiframe_full_triangle_silhouette_grid",
        "static_across_timeline": True, "initial_objective": initial, "result_objective": values[best],
        "objective": "mean_1_minus_IoU_full_triangle_silhouette_at_declared_raster",
        "boundary_hit": bool(scale in bounds), "grid_scales": grid.tolist(), "grid_objectives": values,
        "uncertainty": "image_only_occlusion_mask_camera_pose_and_finite_raster_grid_uncertainty",
        "boundary_policy": "report_no_expansion_no_retry", "sample_count": len(observations),
    }
