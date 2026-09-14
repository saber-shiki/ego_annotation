"""Current local RGB rotation evidence and independent cycle decisions."""
from dataclasses import replace
import math
import numpy as np
from scipy.spatial.transform import Rotation

def local_rotation_evidence(nodes, times, edges, anchor, reprojections=(), max_gap_s=2.):
    positions = {n.frame_idx: i for i, n in enumerate(nodes)}
    ids = list(positions)
    sources = [i for i, n in enumerate(nodes) if n.metric_observation]
    local_pairs = {frozenset((a, b)) for a, b in zip(ids[:-1], ids[1:])
                   if times[positions[b]] - times[positions[a]] <= max_gap_s}
    # Bracket depth-missing targets without promoting their depth. Consecutive
    # valid sources may also bridge a missing interval, within the same time cap.
    for a, b in zip(sources[:-1], sources[1:]):
        if times[b] - times[a] <= max_gap_s:
            local_pairs.add(frozenset((ids[a], ids[b])))
    for i, n in enumerate(nodes):
        if n.metric_observation:
            continue
        before = [j for j in sources if j < i]
        after = [j for j in sources if j > i]
        for j in ([before[-1]] if before else []) + ([after[0]] if after else []):
            if abs(times[j] - times[i]) <= max_gap_s:
                local_pairs.add(frozenset((ids[i], ids[j])))

    def quality(e):
        d = e.diagnostics
        return float(np.clip(float(d.get('inlier_fraction', 1.)) *
                     math.exp(-float(d.get('reprojection_median_px', 0.)) / 2.), 0, 1))

    # Temporal-span-first forest, with quality and matrix tie breaks: invariant
    # to input edge order and frame-ID renaming, including reversed measurements.
    candidates = []
    for k, e in enumerate(edges):
        a, b = positions[e.source_frame_idx], positions[e.target_frame_idx]
        if e.weight <= 0 or e.diagnostics.get('translation_only') or frozenset((ids[a], ids[b])) not in local_pairs:
            continue
        rotation = e.rotation_source_to_target_world if a < b else e.rotation_source_to_target_world.T
        candidates.append((abs(times[b]-times[a]), -quality(e), -e.weight,
                           min(a,b), max(a,b), tuple(rotation.ravel()), k))
    parent = list(range(len(nodes)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    adjacency = {i: [] for i in ids}
    tree = set()
    for *_, k in sorted(candidates):
        e = edges[k]; a, b = e.source_frame_idx, e.target_frame_idx
        ra, rb = root(positions[a]), root(positions[b])
        if ra == rb:
            continue
        parent[rb] = ra
        tree.add(k)
        adjacency[a].append((b, e.rotation_source_to_target_world, k))
        adjacency[b].append((a, e.rotation_source_to_target_world.T, k))

    relative, paths, components = {}, {}, {}
    for start in [anchor] + [i for i in ids if i != anchor]:
        if start in relative:
            continue
        relative[start] = np.eye(3); paths[start] = []; components[start] = start
        queue = [start]
        for a in queue:
            for b, rotation, k in sorted(adjacency[a], key=lambda x: positions[x[0]]):
                if b in relative:
                    continue
                relative[b] = rotation @ relative[a]
                paths[b] = paths[a] + [k]; components[b] = start
                queue.append(b)
    anchor_rotation = nodes[positions[anchor]].initial_rotation
    seeded = []
    seed_rows = []
    for n, stamp in zip(nodes, times):
        supported = components[n.frame_idx] == anchor
        path = paths[n.frame_idx]
        strength = min((quality(edges[k]) for k in path), default=1.) if supported else 0.
        r = relative[n.frame_idx] @ anchor_rotation if supported else n.initial_rotation.copy()
        seeded.append(replace(n, initial_rotation=r.copy(), uncertainty=max(n.uncertainty, 1.-strength)))
        seed_rows.append({'frame_idx': n.frame_idx, 'time_s': float(stamp),
                          'rotation_world_from_completed_canonical_matrix': r.tolist(),
                          'anchor_connected': supported, 'local_path_edge_indices': path,
                          'support_quality': strength, 'uncertainty': max(n.uncertainty, 1.-strength),
                          'kind': 'current_local_RGB_seed' if supported and path else
                                  ('selected_anchor_gauge' if n.frame_idx == anchor else 'source_or_latent_disconnected')})
    filtered, decisions = [], []
    pair_scales = {}
    for k, e in enumerate(edges):
        a, b = e.source_frame_idx, e.target_frame_idx
        independent = sorted(set(paths[a]) ^ set(paths[b])) if components[a] == components[b] else []
        disagreement = None
        fraction = float(np.clip(float(e.diagnostics.get('source_3d_conditioning', .02))/.02, 0, 1))
        allowed = 6. + 14.*fraction
        scale, reason = 1., 'local_tree_measurement_not_cycle_validated'
        strength = min((quality(edges[j]) for j in independent), default=0.)
        if k not in tree and not e.diagnostics.get('translation_only') and e.weight > 0:
            if independent and strength >= .2:
                predicted = relative[b] @ relative[a].T
                disagreement = float(np.degrees(Rotation.from_matrix(predicted @ e.rotation_source_to_target_world.T).magnitude()))
                scale = 0. if disagreement > allowed else 1.
                reason = 'rejected_independent_local_cycle' if scale == 0 else 'consistent_independent_local_cycle'
            else:
                scale, reason = .25, 'unverified_weak_or_disconnected_local_path'
        filtered.append(replace(e, weight=e.weight*scale))
        # Production has one measurement per directed pair. Conservative minimum
        # also prevents an ambiguous duplicate's pixels from bypassing rejection.
        pair_scales[a,b] = min(pair_scales.get((a,b), 1.), scale)
        decisions.append({'edge_index': k, 'source': a, 'target': b,
                          'reason': reason, 'weight_scale': scale,
                          'independent_path_edge_indices': independent if k not in tree else [],
                          'independent_support_quality': strength if k not in tree else None,
                          'disagreement_deg': disagreement, 'allowed_deg': allowed,
                          'input_weight': e.weight, 'output_weight': e.weight*scale})
    projections = [replace(f, edge_weight=f.edge_weight*pair_scales.get((f.source_frame_idx,f.target_frame_idx), 1.))
                   for f in reprojections]
    return seeded, filtered, projections, {
        'policy': 'temporal_local_forest_then_independent_conditioned_cycles',
        'max_local_gap_s': max_gap_s, 'minimum_independent_path_quality': .2,
        'seed_rows': seed_rows, 'edges': decisions,
        'local_disconnected_ids': [i for i in ids if components[i] != anchor],
        'source_prior_mutated': False,
    }


def observed_rotation_authority(nodes, chain):
    """Partial near-planar NN surfaces constrain translation, not RGB rotation.

    Covariance ratio is a heuristic observability estimate, not a claim that all
    planar rotations are unobservable. Only locally RGB-supported nodes use it.
    """
    support = {r['frame_idx']: r for r in chain['seed_rows']}
    scales = {}
    for n in nodes:
        scale = 1.
        row = support[n.frame_idx]
        if row['anchor_connected'] and row['support_quality'] >= .2 and n.observed_world is not None:
            points = np.asarray(n.observed_world)
            eig = np.maximum(np.linalg.eigvalsh(np.cov(points.T)), 0.)
            scale = float(np.clip(eig[0] / max(eig[-1], 1e-12) / .02, 0, 1))
        scales[n.frame_idx] = scale
    return scales
