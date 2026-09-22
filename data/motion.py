"""Observation-side local motion qualification and explicitly separate R4 seed.

No least-squares, policy, reward, candidate update or generated geometry input.
The rotation/cycle kernels retain the V7 semantics. Translation is local relative
measurement propagation, not a full P14 trajectory fit.
"""
from dataclasses import dataclass
import math
import numpy as np
from data.immutable import freeze
from data.timeline import timeline_ids, timeline_times
from utils.data import PoseNode, RGBEdge, RGBReprojectionFactor
from observations.object.rgb_tracking import local_rotation_evidence, observed_rotation_authority
from envs.qualification.object_observation import p09_geometry_validation
from utils.data import RKEY, TKEY


@dataclass(frozen=True)
class MotionPreparation:
    qualification: object
    source_initialization: object
    rgb_seed_comparison: object

    @property
    def initialization(self):
        """Backward-compatible name for explicit RGB comparison only."""
        return self.rgb_seed_comparison


def source_pose_initialization(frames, object_id, *, unresolved='none'):
    """Return unpropagated source pose rows as a no-RGB-seed candidate baseline."""
    if unresolved not in {'none', 'previous_source_hold'}:
        raise ValueError('unknown source initialization missing-pose policy')
    rows=[];last=None
    for frame in frames:
        objects=[o for o in frame.get('objects',[]) if o.get('object_id')==object_id]
        if len(objects)>1:raise ValueError('duplicate source pose entity')
        obj=objects[0] if objects else None
        pose=(obj or {}).get('reconstructed_geometry_pose') or {}
        rotation_key='rotation_world_from_canonical_matrix' if 'rotation_world_from_canonical_matrix' in pose else RKEY
        R=np.asarray(pose.get(rotation_key),float) if rotation_key in pose else None
        t=np.asarray(pose.get(TKEY),float) if TKEY in pose else None
        geom=(obj or {}).get('visible_geometry_candidate') or {}
        if R is None and t is None and geom.get('centroid_world_m') is not None:
            R=np.eye(3);t=np.asarray(geom['centroid_world_m'],float)
            pose_source='source_visible_geometry_centroid_initializer'
        else:
            pose_source='source_visible_geometry_pose_row'
        valid=(R is not None and t is not None and R.shape==(3,3) and t.shape==(3,)
               and np.isfinite(R).all() and np.isfinite(t).all())
        metric=bool(valid and (obj or {}).get('rigid_pose_observation_eligible') is True
                    and geom.get('rigid_pose_observation_eligible') is True)
        if valid:
            last=(R.copy(),t.copy())
            rows.append({'frame_idx':int(frame['frame_idx']),'time_s':float(frame['time_s']),
                'available':True,'rotation_world_from_canonical':R.tolist(),'translation_world_m':t.tolist(),
                'pose_source':pose_source,'metric_observation':metric,
                'uncertainty':.1 if metric else .95,'missing_pose_policy':'none'})
        elif unresolved=='previous_source_hold' and last is not None:
            rows.append({'frame_idx':int(frame['frame_idx']),'time_s':float(frame['time_s']),
                'available':True,'rotation_world_from_canonical':last[0].tolist(),'translation_world_m':last[1].tolist(),
                'pose_source':'latent_previous_source_hold_not_observation','metric_observation':False,
                'uncertainty':1.,'missing_pose_policy':unresolved})
        else:
            rows.append({'frame_idx':int(frame['frame_idx']),'time_s':float(frame['time_s']),
                'available':False,'rotation_world_from_canonical':None,'translation_world_m':None,
                'pose_source':'unresolved_missing_source_pose','metric_observation':False,
                'uncertainty':1.,'missing_pose_policy':unresolved})
    return freeze({'role':'R4_source_pose_candidate_baseline_not_RGB_seed','object_id':object_id,
        'contains_rgb_propagation':False,'contains_optimized_state':False,
        'missing_pose_policy':unresolved,'frames':rows})


def prepare_motion(evidence, frames, object_id, anchor, *, include_rgb_seed=True):
    ids = timeline_ids([f['frame_idx'] for f in frames])
    times = timeline_times([f['time_s'] for f in frames],len(ids))
    if anchor not in ids:raise ValueError('motion anchor outside timeline')
    if evidence.object_id != object_id:raise ValueError('motion evidence object mismatch')
    nodes=[]
    for frame in frames:
        objects=[o for o in frame.get('objects',[]) if o.get('object_id')==object_id]
        if len(objects)>1:raise ValueError('duplicate motion entity')
        obj=objects[0] if objects else None
        points,reason,_=p09_geometry_validation(obj,frame)
        metric=bool(points is not None and obj.get('rigid_pose_observation_eligible') is True
                    and obj.get('visible_geometry_candidate',{}).get('rigid_pose_observation_eligible') is True)
        points=points if metric else None
        centroid=np.mean(points,axis=0) if metric else np.zeros(3)
        nodes.append(PoseNode(frame['frame_idx'],np.eye(3),centroid,points,reason,metric,.1 if metric else .95))
    by_id={n.frame_idx:n for n in nodes};positions={i:p for p,i in enumerate(ids)}
    if not by_id[anchor].metric_observation:raise ValueError('motion anchor needs retained metric support')
    edges=[];projections=[]
    for edge in evidence.edges:
        a,b=edge.source_frame_idx,edge.target_frame_idx
        if a not in positions or b not in positions:raise ValueError('motion edge outside timeline')
        if edge.source_time_s!=times[positions[a]] or edge.target_time_s!=times[positions[b]]:raise ValueError('motion edge timestamp mismatch')
        d=edge.diagnostics
        weight=float(d['inlier_fraction'])*math.exp(-float(d['reprojection_median_px'])/2.)
        weight*=max(.02,min(1.,float(d['source_3d_conditioning'])/.02))
        weight*=.45 if d['topological_role']=='adjacent' else 1.
        edges.append(RGBEdge(a,b,edge.rotation_source_to_target_world,edge.translation_source_to_target_world_m,max(.001,weight),dict(d)))
        projections.append(RGBReprojectionFactor(a,b,positions[a],positions[b],edge.source_world,edge.target_uv,
            edge.target_intrinsics,edge.target_camera[:3,:3],edge.target_camera[:3,3],edge.weights,max(.001,weight)))
    _,filtered,pixels,chain=local_rotation_evidence(nodes,times,edges,anchor,projections)
    conditioning=observed_rotation_authority(nodes,chain)
    # Only local-tree paths from the fixed anchor may establish a propagated seed.
    seeds=[]
    for item in chain['seed_rows']:
        fid=item['frame_idx'];available=item['anchor_connected'] or by_id[fid].metric_observation
        rotation=translation=None
        if item['anchor_connected']:
            rotation=np.eye(3);translation=by_id[anchor].initial_translation.copy();cursor=anchor
            for idx in item['local_path_edge_indices']:
                edge=edges[idx];D=edge.rotation_source_to_target_world;d=edge.translation_source_to_target_world_m
                if edge.source_frame_idx==cursor:cursor=edge.target_frame_idx
                elif edge.target_frame_idx==cursor:
                    cursor=edge.source_frame_idx;d=-D.T@d;D=D.T
                else:raise ValueError('broken local motion path')
                rotation=D@rotation;translation=D@translation+d
            if cursor!=fid:raise ValueError('seed path did not reach frame')
        elif available:
            rotation=np.eye(3);translation=by_id[fid].initial_translation.copy()
        seeds.append({'frame_idx':fid,'time_s':item['time_s'],'available':bool(available),
            'rotation_world_from_canonical':None if rotation is None else rotation.tolist(),
            'translation_world_m':None if translation is None else translation.tolist(),
            'kind':item['kind'] if available else 'unresolved_no_anchor_path_or_metric_source',
            'metric_observation':by_id[fid].metric_observation,'uncertainty':item['uncertainty'],
            'edge_path':item['local_path_edge_indices']})
    source_initialization=source_pose_initialization(frames,object_id)
    rgb_seed=None
    if include_rgb_seed:
        rgb_seed=freeze({'role':'R4_derived_rgb_measurement_seed_comparison_only','anchor_frame_idx':anchor,
            'canonical_frame':'selected_P09_anchor_centroid_world_aligned',
            'contains_optimized_state':False,'source_mutated':False,'comparison_only':True,
            'frames':seeds})
    return MotionPreparation(
        freeze({'role':'R2_derived_motion_qualification','anchor_frame_idx':anchor,'edges':chain['edges'],
                'rotation_authority':conditioning,'disconnected_ids':chain['local_disconnected_ids'],
                'relative_edge_weights':[e.weight for e in filtered],
                'reprojection_edge_weights':[f.edge_weight for f in pixels]}),
        source_initialization,rgb_seed)
