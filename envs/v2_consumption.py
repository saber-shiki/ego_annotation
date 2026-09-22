"""V2 environment-side consumer for zero/controlled-update validation, NOT PPO.

Source, measurement seed and candidate remain separate. This adapter provides
real geometry/evidence consumers without relaxing the existing training guard or
claiming that the old 480-feature actor has been upgraded. No dummy reward or
optimizer: callers inspect explicit measurements and candidate exports.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from data.bundle import EpisodeBundle
from data.geometry import bind_visual_to_candidate
from data.immutable import thaw
from observations.contracts import SCHEMA_V2,CAPABILITY_SOURCES
from utils.state import sha
from utils.mano import TIP_VERTICES

SCHEMA='ego.annotation.v2_rl_source_baseline_candidate.v1'
CONTRACT={
    'source_schema':SCHEMA_V2,'consumer':SCHEMA,'mode':'controlled_geometry_validation_only',
    'trainable':False,'policy_observation_layout':None,'policy_action_layout':None,'reward_profile':None,
    'default_initialization':'source_full_native_no_rgb_seed','comparison_initialization':'rgb_seed_comparison_only',
    'rgb_evidence_role':'retained_measurement_inputs_for_future_candidate_updates',
    'object_geometry_role':'complete_visual_mesh','collision_authority_changed':False,
    'seed_role':'R4_comparison_only_not_default_candidate','hand_decoder':'source_geometry_translation_only_for_control_test',
    'pending':['actor_encoding','52DoF_visual_MANO_decoder','shared_object_scale','reviewed_reward_profile','checkpoint_compatibility'],
}


class V2ConsumptionSession:
    """A real candidate/evidence adapter, deliberately not an RL training Env."""
    def __init__(self,source_observations, *, initialization='source_full_native', missing_source_pose='none'):
        self.bundle=EpisodeBundle.open(source_observations)
        if initialization not in {'source_full_native','rgb_seed_comparison'}:
            raise ValueError('unknown v2 candidate initialization')
        if missing_source_pose not in {'none','previous_source_hold'}:
            raise ValueError('unknown missing source pose policy')
        self.initialization=initialization;self.missing_source_pose=missing_source_pose
        if self.bundle.manifest['schema']!=SCHEMA_V2:raise ValueError('explicit v2 source required')
        if set(CAPABILITY_SOURCES)-set(self.bundle.manifest['capabilities']):raise ValueError('incomplete v2 capabilities')
        self.source=self.bundle.load_observations()
        self.ids=np.asarray(self.source['frame_idx']);self.position={int(x):i for i,x in enumerate(self.ids)}
        self.entity=self.bundle.manifest['entity_slots']['objects'][0]
        if len(self.bundle.manifest['entity_slots']['objects'])!=1:raise ValueError('one selected object required')
        self.interaction=self.source['interaction_timelines'][self.entity]
        motion=self.source['motion_preprocessing'];visual=self.source['visual_geometry']
        if motion is None or visual is None:raise ValueError('explicit source pose and full visual geometry required')
        self.qualifications=motion.qualification
        source_init=motion.source_initialization
        if missing_source_pose!='none':
            from data.motion import source_pose_initialization
            self.source_initialization=source_pose_initialization(self.source['annotation_frames'],self.entity,unresolved=missing_source_pose)
        else:
            self.source_initialization=source_init
        selected=self.source_initialization if initialization=='source_full_native' else motion.rgb_seed_comparison
        if selected is None:raise ValueError('RGB seed comparison was not retained')
        self.initialization_role=('R4_source_pose_candidate_baseline_not_RGB_seed' if initialization=='source_full_native'
                                  else 'R4_derived_rgb_measurement_seed_comparison_only')
        self.candidate_initialization=selected
        rows={r['frame_idx']:r for r in selected['frames']}
        self.seed_available=np.array([rows[int(fid)]['available'] for fid in self.ids],bool)
        self.seed_rotation=np.array([rows[int(fid)]['rotation_world_from_canonical'] or np.eye(3).tolist() for fid in self.ids])
        self.seed_translation=np.array([rows[int(fid)]['translation_world_m'] or [0,0,0] for fid in self.ids])
        self.rgb_seed_comparison=motion.rgb_seed_comparison
        self.anchor=int(selected.get('anchor_frame_idx',visual.anchor_frame_idx))
        self.seed=self.candidate_initialization
        at=self.position[visual.anchor_frame_idx]
        if not self.seed_available[at]:raise ValueError('visual binding requires an available appearance-anchor seed')
        self.binding=bind_visual_to_candidate(visual,self.seed_rotation[at],self.seed_translation[at],
            candidate_revision=self.initialization_role+'-'+sha(self.bundle.manifest_path)[:16])
        self.full_vertices=self.binding['vertices_canonical'];self.full_faces=self.binding['faces']
        self._scene=None
        self.reset()

    def reset(self):
        self.object_rotation=self.seed_rotation.copy();self.object_translation=self.seed_translation.copy()
        self.hand_vertices={s:np.array(h['vertices'],copy=True) for s,h in self.source['hands'].items()}
        self.hand_joints={s:np.array(h['joints'],copy=True) for s,h in self.source['hands'].items()}
        self.history=[];self.revision=0
        return {'schema':SCHEMA,'frames':len(self.ids),'candidate_source':self.initialization,
            'candidate_initialization_role':self.initialization_role,
            'source_manifest_sha256':sha(self.bundle.manifest_path),'trainable':False,
            'rgb_evidence_retained':self.source['rgb_evidence'] is not None}

    def object_world(self,frame_idx):
        i=self.position[int(frame_idx)]
        if not self.seed_available[i]:return None
        return self.full_vertices@self.object_rotation[i].T+self.object_translation[i]

    def display_validity(self):
        return {'object':self.seed_available & self.interaction.display_observation_eligible[:,0],
            **{s:np.asarray(self.source['hands'][s]['valid']) & self.interaction.display_observation_eligible[:,j+1]
               for j,s in enumerate(('left','right'))}}

    def apply_control(self,frame_idx,*,object_rotation_rad=None,object_translation_m=None,
                      left_translation_m=None,right_translation_m=None):
        """Bounded physical-unit injection for validation; not a policy action schema.

        All inputs/availability are checked before any state mutation. Fixed gauge
        anchor cannot move. No numeric update can disable source evidence.
        """
        if type(frame_idx) is not int or frame_idx not in self.position:raise ValueError('control frame not in source timeline')
        values={}
        for key,value,limit in [('object_rotation_rad',object_rotation_rad,.05),('object_translation_m',object_translation_m,.02),
                                 ('left_translation_m',left_translation_m,.02),('right_translation_m',right_translation_m,.02)]:
            vector=np.zeros(3) if value is None else np.asarray(value,dtype=float)
            if vector.shape!=(3,) or not np.isfinite(vector).all() or np.any(np.abs(vector)>limit):
                raise ValueError('invalid/beyond-validation-budget control: '+key)
            values[key]=vector
        i=self.position[frame_idx];object_changed=any(np.any(values[k]) for k in ('object_rotation_rad','object_translation_m'))
        if object_changed and (frame_idx==self.anchor or not self.seed_available[i]):
            raise ValueError('fixed gauge or unavailable candidate cannot receive object control')
        for side in ('left','right'):
            if np.any(values[side+'_translation_m']) and not self.source['hands'][side]['valid'][i]:
                raise ValueError('cannot manufacture a missing source hand')
        if object_changed:
            self.object_rotation[i]=Rotation.from_rotvec(values['object_rotation_rad']).as_matrix()@self.object_rotation[i]
            self.object_translation[i]+=values['object_translation_m']
        for side in ('left','right'):
            self.hand_vertices[side][i]+=values[side+'_translation_m']
            self.hand_joints[side][i]+=values[side+'_translation_m']
        self.revision+=1
        event={'frame_idx':frame_idx,'revision':self.revision,'units':'rad/world_meters','controls':{k:v.tolist() for k,v in values.items()},
               'source_observations_changed':False,'fixed_visual_binding_changed':False}
        self.history.append(event)
        return event

    def _closest(self,points):
        import open3d as o3d
        if self._scene is None:
            self._scene=o3d.t.geometry.RaycastingScene(nthreads=1)
            self._scene.add_triangles(o3d.core.Tensor(np.asarray(self.full_vertices,np.float32)),
                                      o3d.core.Tensor(np.asarray(self.full_faces,np.uint32)))
        return self._scene.compute_closest_points(o3d.core.Tensor(np.asarray(points,np.float32)))['points'].numpy()

    def measure(self,frame_idx):
        """Fresh candidate-dependent evidence values, never a stale source residual."""
        i=self.position[int(frame_idx)];T=self.source['T_world_camera'][i]
        source_views=self.source['source_visual_preprocessing'];K=source_views['intrinsics_fit'][i]
        appearance=self.source['appearance_evidence'].prepare_frame(frame_idx)
        result={'frame_idx':int(frame_idx),'revision':self.revision,'units':'m/rad/pixels',
            'candidate_available':bool(self.seed_available[i]),'metric_observation_eligible':bool(self.source['objects'][self.entity]['metric_observation_eligible'][i]),
            'visibility':self.interaction.visibility[i].tolist(),'primary_visual_eligible':self.interaction.primary_visual_eligible[i].tolist(),
            'hand_measurements':{},'RGB_edges':[],'physical_contact_authority':False}
        for j,side in enumerate(('left','right')):
            hand=self.source['hands'][side]
            if not hand['valid'][i]:result['hand_measurements'][side]={'available':False};continue
            vertices=self.hand_vertices[side][i];cam=(vertices-T[:3,3])@T[:3,:3]
            finite=np.isfinite(cam).all(axis=1)&(cam[:,2]>1e-5)
            uv=np.zeros((778,2));uv[finite]=cam[finite,:2]/cam[finite,2:]*K[:2]+K[2:]
            source_uv=source_views['hands'][side]['vertices_uv_fit'][i]
            delta=uv-source_uv;summary={'available':True,'projectable_vertices':int(finite.sum()),
                'source_projection_delta_median_px':float(np.median(np.linalg.norm(delta[finite],axis=1))) if finite.any() else None}
            pads=source_views['hands'][side]['pad_vertex_ids'][i]
            summary['fixed_pad_vertex_ids']=pads.tolist()
            if self.seed_available[i]:
                canonical=(vertices[pads]-self.object_translation[i])@self.object_rotation[i]
                near=self._closest(canonical.reshape(-1,3)).reshape(5,12,3)
                distance=np.linalg.norm(canonical-near,axis=-1)
                summary.update(top3_mean_unsigned_gap_m=np.sort(distance,axis=1)[:,:3].mean(axis=1).tolist(),
                    closest_targets_canonical=near.tolist(),closest_role='complete_generated_visual_mesh_unsigned_only')
            if appearance['available']:
                tips=uv[TIP_VERTICES];inside=finite[TIP_VERTICES]&(tips[:,0]>=0)&(tips[:,0]<appearance['fit_size_wh'][0])&(tips[:,1]>=0)&(tips[:,1]<appearance['fit_size_wh'][1])
                pixel=np.rint(tips).astype(int);pixel=np.clip(pixel,[0,0],appearance['fit_size_wh']-1)
                summary['pad_image_distance_px']=[float(appearance['distance_transform_px'][p[1],p[0]]) if ok else None for p,ok in zip(pixel,inside)]
            result['hand_measurements'][side]=summary
        for k,edge in enumerate(self.source['rgb_evidence'].edges):
            if frame_idx not in (edge.source_frame_idx,edge.target_frame_idx):continue
            a=self.position[edge.source_frame_idx];b=self.position[edge.target_frame_idx]
            if not self.seed_available[a] or not self.seed_available[b]:
                result['RGB_edges'].append({'edge_index':k,'candidate_available':False});continue
            source_canonical=(edge.source_world-self.object_translation[a])@self.object_rotation[a]
            world=source_canonical@self.object_rotation[b].T+self.object_translation[b]
            cam=(world-edge.target_camera[:3,3])@edge.target_camera[:3,:3]
            valid=np.isfinite(cam).all(axis=1)&(cam[:,2]>1e-5)
            if valid.all():
                uv=cam[:,:2]/cam[:,2:]*edge.target_intrinsics[:2]+edge.target_intrinsics[2:]
                residual=uv-edge.target_uv;mean=float(np.mean(np.sum(residual**2,axis=1)))
            else:mean=None
            result['RGB_edges'].append({'edge_index':k,'source_frame_idx':edge.source_frame_idx,'target_frame_idx':edge.target_frame_idx,
                'point_count':len(edge.source_world),'candidate_available':True,'positive_depth_count':int(valid.sum()),
                'mean_squared_reprojection_px2':mean,'qualified_weight':self.qualifications['reprojection_edge_weights'][k],
                'invalid_projection_is_not_dropped':True})
        return result

    def export(self,output_dir):
        """Complete geometry for existing triangle-render primitives; NOT policy rollout schema."""
        output=Path(output_dir).resolve()
        if output.is_relative_to(self.bundle.manifest_path.parent):raise ValueError('candidate output must be outside source run')
        output.mkdir(parents=True,exist_ok=False);valid=self.display_validity()
        arrays={'frame_idx':self.ids,'time_s':self.source['time_s'],'T_world_camera':self.source['T_world_camera'],
            'intrinsics_render':self.source['intrinsics'],'render_size_wh':self.source['image_size_wh'],
            'object_vertices_canonical':self.full_vertices,'object_faces':self.full_faces,
            'object_rotation':self.object_rotation,'object_translation_world':self.object_translation,
            'object_pose_valid':self.seed_available,'object_valid':valid['object'],
            'object_metric_observation_eligible':self.source['objects'][self.entity]['metric_observation_eligible']}
        for side in ('left','right'):
            arrays.update({side+'_vertices_world':self.hand_vertices[side],side+'_joints_world':self.hand_joints[side],
                side+'_faces':self.source['hands'][side]['faces'],side+'_source_valid':self.source['hands'][side]['valid'],
                side+'_valid':self.source['hands'][side]['valid'],side+'_display_valid':valid[side]})
        npz=output/'candidate.npz'
        with npz.open('xb') as handle:np.savez_compressed(handle,**arrays)
        document={'schema':SCHEMA,'contract':CONTRACT,'source_manifest':str(self.bundle.manifest_path),
            'source_manifest_sha256':sha(self.bundle.manifest_path),'candidate_revision':self.revision,
            'candidate_initialization':self.initialization,'candidate_initialization_role':self.initialization_role,
            'rgb_seed_comparison_available':self.rgb_seed_comparison is not None,
            'array_layer':{'path':str(npz),'sha256':sha(npz)},'frame_count':len(self.ids),
            'object_id':self.entity,'geometry_role':'complete_visual_mesh','visual_binding':{
                k:thaw(v) for k,v in self.binding.items() if k not in ('vertices_canonical','faces')},
            'visibility_policy':'source/candidate availability AND not out_of_frame; occluded remains uncertain',
            'visibility':self.interaction.visibility.tolist(),'raw_declarations_preserved':True,
            'source_arrays_mutated':False,'policy_executed':False,'optimization_executed':False,
            'physical_contact_authority':False,'annotation_ready':False,'history':self.history}
        path=output/'candidate.json';path.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
        receipt={'json_sha256':sha(path),'npz_sha256':sha(npz),'contract':SCHEMA}
        (output/'integrity.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return {'json':str(path),'npz':str(npz),'receipt':str(output/'integrity.json')}
