"""Inside/crossing queries on actual generated triangles, not physical authority.

An operator-selected closed generated mesh can define a collision HYPOTHESIS.
Never promote source signed_geometry_ready or observed-face authority from this.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

CONTRACT = "generated_mesh_collision_hypothesis.v1"


def mesh_identity(vertices, faces):
    h = hashlib.sha256()
    for a in (np.asarray(vertices, np.float32), np.asarray(faces, np.int64)):
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


class GeneratedCollisionScene:
    def __init__(self, vertices, faces):
        import open3d as o3d
        import trimesh
        self.vertices = np.asarray(vertices, np.float32)
        self.faces = np.asarray(faces, np.int64)
        if not np.isfinite(self.vertices).all() or not len(self.faces) or self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("invalid native collision hypothesis mesh")
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        self.topology = {"vertices": len(self.vertices), "triangles": len(self.faces),
                         "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
                         "positive_volume": bool(mesh.is_volume), "volume_m3": float(mesh.volume),
                         "source_authority_promoted": False, "contract": CONTRACT}
        if not (self.topology["watertight"] and self.topology["winding_consistent"] and self.topology["positive_volume"]):
            raise ValueError("native generated mesh does not support this closed-body inside/outside hypothesis; no repair/substitution allowed")
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.faces.astype(np.uint32)))
        self._edge_cache = {}

    def query(self, vertices_world, mano_faces, rotation, translation, *, nsamples=5, epsilon_m=.0001, edges=False):
        import open3d as o3d
        v = np.asarray(vertices_world, float)
        f = np.asarray(mano_faces, np.int64)
        centers = v[f].mean(axis=1)
        samples_world = np.concatenate((v, centers))
        points = (samples_world - translation) @ rotation
        depth = -self.scene.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32)), nthreads=1, nsamples=nsamples).numpy().astype(float)
        if not np.isfinite(depth).all():
            raise FloatingPointError("nonfinite native-mesh signed query")
        result = {"depth_m": depth, "vertex_depth_m": depth[:len(v)], "face_center_depth_m": depth[len(v):]}
        if edges:
            key = hashlib.sha256(f.tobytes()).hexdigest()
            if key not in self._edge_cache:
                all_edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
                self._edge_cache[key] = np.unique(np.sort(all_edges, axis=1), axis=0)
            edge_ids = self._edge_cache[key]
            vo = (v - translation) @ rotation
            start, end = vo[edge_ids[:, 0]], vo[edge_ids[:, 1]]
            length = np.linalg.norm(end - start, axis=1)
            direction = (end - start) / np.maximum(length[:, None], 1e-12)
            # Only interior segment hits, excluding epsilon-sized endpoint touches.
            origins = start + epsilon_m * direction
            rays = np.concatenate((origins, direction), axis=1).astype(np.float32)
            hit = self.scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)
            distance = hit["t_hit"].numpy().astype(float)
            crossing = np.isfinite(distance) & (distance > epsilon_m) & (distance < length - 2 * epsilon_m)
            result["edge_crossings"] = int(crossing.sum())
            result["crossing_edge_vertex_ids"] = edge_ids[crossing].tolist()
            result["crossing_object_triangle_ids"] = hit["primitive_ids"].numpy()[crossing].astype(int).tolist()
        return result


def summarize_depth(row, epsilon_m=.0001):
    depths = np.asarray(row["depth_m"])
    penetrating = depths > epsilon_m
    v = np.asarray(row["vertex_depth_m"])
    c = np.asarray(row["face_center_depth_m"])
    positive = depths[penetrating]
    return {"inside_sample_count_over_0_1mm": int(penetrating.sum()),
            "inside_vertex_count_over_0_1mm": int((v > epsilon_m).sum()),
            "inside_face_center_count_over_0_1mm": int((c > epsilon_m).sum()),
            "inside_sample_count_over_1mm": int((depths > .001).sum()),
            "max_inside_depth_mm": float(max(0., depths.max()) * 1000),
            "positive_depth_mean_mm": float(positive.mean() * 1000) if len(positive) else 0.,
            "positive_depth_p95_mm": float(np.percentile(positive, 95) * 1000) if len(positive) else 0.,
            "inside_squared_hinge_sum_m2": float(np.square(np.maximum(depths - epsilon_m, 0)).sum()),
            "inside_vertex_ids": np.flatnonzero(v > epsilon_m).tolist(),
            "inside_face_ids_by_centroid": np.flatnonzero(c > epsilon_m).tolist(),
            **{k: row[k] for k in ("edge_crossings", "crossing_edge_vertex_ids", "crossing_object_triangle_ids") if k in row}}


def intervals(ids):
    result = []
    for fid in sorted(set(map(int, ids))):
        if not result or fid != result[-1][1] + 1:
            result.append([fid, fid])
        else:
            result[-1][1] = fid
    return result


def audit_archive(archive, *, scene=None, nsamples=11):
    if isinstance(archive, (str, Path)):
        with np.load(archive, allow_pickle=False) as z:
            archive = dict(z)
    a = archive
    scene = scene or GeneratedCollisionScene(a['object_vertices_canonical'], a['object_faces'])
    rows = []
    affected = {side: [] for side in ('left', 'right')}
    substantive = {side: [] for side in ('left', 'right')}
    total = 0.; max_depth = 0.; crossing_total = 0
    for i, fid in enumerate(a['frame_idx']):
        sides = {}
        for side in ('left', 'right'):
            if not (a['object_valid'][i] and a[side+'_valid'][i]):
                sides[side] = {'state': 'absent_geometry'}
                continue
            f = a[side+'_faces']; f = f[i] if f.ndim == 3 else f
            row = scene.query(a[side+'_vertices_world'][i], f, a['object_rotation'][i], a['object_translation_world'][i], nsamples=nsamples, edges=True)
            stats = summarize_depth(row)
            stats['state'] = 'generated_mesh_inside_or_surface_crossing' if stats['inside_sample_count_over_0_1mm'] or stats['edge_crossings'] else 'no_detected_generated_mesh_penetration'
            if stats['state'].startswith('generated_'):
                affected[side].append(int(fid))
            if stats['inside_sample_count_over_1mm']:
                substantive[side].append(int(fid))
            total += stats['inside_squared_hinge_sum_m2']; max_depth = max(max_depth, stats['max_inside_depth_mm']); crossing_total += stats['edge_crossings']
            sides[side] = stats
        rows.append({'frame_idx': int(fid), 'time_s': float(a['time_s'][i]), 'hands': sides})
    union = sorted(set(affected['left'] + affected['right']))
    return {'contract': CONTRACT, 'scope': 'intersection_with_trusted_generated_geometry_not_GT_or_certified_physics',
            'frame_count': len(rows), 'nsamples': nsamples, 'epsilon_m': .0001, 'topology': scene.topology,
            'affected_frame_ids': union, 'affected_intervals': intervals(union),
            'affected_frame_count': len(union), 'hand_affected_frame_ids': affected,
            'hand_affected_intervals': {k: intervals(v) for k, v in affected.items()},
            'hand_substantive_over_1mm_frame_ids': substantive,
            'max_inside_depth_mm': max_depth, 'inside_squared_hinge_sum_m2': total,
            'total_crossing_edges_over_all_hand_frames': crossing_total, 'frames': rows}
