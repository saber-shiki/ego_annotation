"""Independent v2 observation/preprocessing acceptance (no PPO or inference).

Usage: python -m observations.inspect --source-observations SOURCE --output-dir NEW
All timeline data are checked; review images are selected-frame QC, not a final
annotation video. Missing capabilities return an incomplete report, not success.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys
import numpy as np
import cv2

from data.bundle import EpisodeBundle
from data.immutable import thaw
from data.geometry import bind_visual_to_candidate
from data.timeline import join_annotation_rows
from observations.contracts import SCHEMA_V2, ARTIFACT_CLASSES, CAPABILITY_SOURCES
from observations.inspection_io import SourceReads, stream_depth, numeric_summary, track_rows
from observations.inspection_render import panel, mask_view, source_view, depth_view, edge_view
from observations.object.surface import MetricEligibility, array_sha256, RASTER_SCHEMA, CLASS_CODES
from observations.object.rgb_tracking import reproduce_source_depth_support
from envs.qualification.camera import load_contract, plane_intrinsics, resize_affine, K_from_intrinsics
from envs.qualification.object_observation import retained_metric_alignment
from utils.state import sha

# Precisely distinguish legacy use from future consumption of the NEW v2 views.
CONSUMERS = {
    'input_metadata': ('原始视频、配置/模型身份、目标计划', 'metadata only; no model weights loaded', 'no new policy input implied'),
    'camera_timeline': ('K/T/时间轴', 'legacy env uses equivalent v1 fields', 'explicit v2 episode adapter'),
    'rgb_frames': ('逐帧RGB', 'existing perception uses images', 'actor image/texture encoder'),
    'depth_masks_surface': ('depth/confidence、raw/owned masks、eligibility、P09', 'legacy env uses some derived P09/first-surface values', 'full evidence/quality encoding; nonmetric candidate support'),
    'source_mano': ('完整源MANO、投影、12顶点pad', 'legacy decoder uses source parameters, not new static views', '52-DoF visual decoder; pad features; all-frame licensed LBS replay'),
    'appearance': ('raw appearance、轮廓、distance transform', 'not consumed by old policy geometry path', 'silhouette objective/features and candidate outline correspondences'),
    'interaction': ('typed contact、primary、visibility/occluder', 'legacy factors use reduced priors; new semantic arrays not wired', 'EpisodeStatic/actor activation and actual display-valid/clear'),
    'rgb_motion': ('RGB/PnP边、双端重投影、loop资格', 'not supplied to old actor/default reward', 'motion tokens/reprojection factors in new policy consumer'),
    'motion_seed': ('独立R4测量传播seed', 'old env still initializes from P09 centroid', 'explicit candidate initializer; missing-depth action semantics'),
    'visual_geometry': ('完整native metric-camera mesh/显式binding', 'old env/render still uses pose mesh', 'full visual mesh state/render binding; shared object scale'),
    'formal_factors': ('P17因子、独立first-surface查询', 'legacy v1 path consumes its own patch/factors', 'new v2 adapter; legacy frozen contact membership deliberately not prepared'),
}


ARTIFACT_GROUPS = {
    **{k:'input_metadata' for k in ('runtime_input_contract','code_manifest','model_asset_manifest','object_plan','candidate_identity','raw_video')},
    'raw_frames':'rgb_frames','camera_calibration':'camera_timeline','conditioned_depth':'depth_masks_surface',
    'hawor_world':'source_mano','mano_bridge':'source_mano',
    **{k:'depth_masks_surface' for k in ('raw_object_masks','owned_object_masks','metric_eligibility','visible_annotations','surface_report')},
    **{k:'visual_geometry' for k in ('completion_report','reconstructed_mesh','collision_mesh','visual_geometry_report','visual_mesh_camera','visual_native_mesh','visual_native_qc')},
    'appearance_masks':'appearance','appearance_anchor_decision':'appearance',
    'interaction_judgments':'interaction','interaction_factors':'formal_factors',
    'rgb_source_npz':'rgb_motion','rgb_source_report':'rgb_motion',
}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(thaw(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_npz(path, **arrays):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def write_csv(path, rows):
    names=list(dict.fromkeys(key for row in rows for key in row)) if rows else ['status']
    with Path(path).open('x', encoding='utf-8-sig', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=names);writer.writeheader();writer.writerows(rows)


class Inspection:
    def __init__(self, source, output, review_frames=None):
        self.source=Path(source).resolve();self.output=Path(output).resolve()
        if self.output.is_relative_to(self.source.parent):
            raise ValueError('inspection output must be outside the immutable source run')
        self.output.mkdir(parents=True,exist_ok=False)
        self.reads=SourceReads(self.source.parent)
        self.phase='open';self.groups={};self.warnings=[];self.missing=[]
        self.review_frames=review_frames;self.outputs=[]
        (self.output/'arrays').mkdir();(self.output/'review').mkdir()

    def artifact(self,name):
        rec=self.manifest['artifacts'].get(name)
        if rec is None:
            raise ValueError('missing observation artifact: '+name)
        return self.reads.path(rec['path'],rec['sha256'])

    def image_output(self,name,image):
        path=self.output/'review'/name
        if not cv2.imwrite(str(path),image):raise RuntimeError('failed to write QC image')
        self.outputs.append(str(path.relative_to(self.output)))

    def group(self,name,summary,outputs=()):
        self.groups[name]={'status':'read_and_preprocessed','summary':summary,
            'outputs':list(outputs),'legacy_consumption':CONSUMERS[name][1],
            'pending_consumers':CONSUMERS[name][2],'v2_policy_connected':False}

    def run(self):
        header=self.reads.json(self.source)
        if header.get('schema')!=SCHEMA_V2:
            raise ValueError('inspection requires explicit v2; no silent promotion of v1')
        self.bundle=EpisodeBundle.open(self.source)
        self.manifest=self.bundle.manifest
        if self.manifest['schema']!=SCHEMA_V2:
            raise ValueError('inspection requires explicit v2; no silent promotion of v1')
        self.missing=sorted(set(CAPABILITY_SOURCES)-set(self.manifest['capabilities']))
        missing_artifacts=sorted(set(ARTIFACT_CLASSES)-set(self.manifest.get('artifacts',{})))
        self.missing.extend('artifact:'+name for name in missing_artifacts)
        self.phase='read_bound_artifacts';inventory=[]
        for name,rec in self.manifest['artifacts'].items():
            path=self.artifact(name)
            meta=self.reads.json(path) if path.suffix=='.json' else None
            inventory.append({'name':name,'role':rec['role'],'path':str(path),'sha256':rec['sha256'],
                'bytes':path.stat().st_size,'classes':'/'.join(self.manifest['observation_roles'][name]['classes']),
                'read':'JSON decoded' if meta is not None else 'bytes hash checked; typed payload processing below'})
        write_csv(self.output/'artifact_inventory.csv',inventory)
        if self.missing:
            # No feature zero-fill or fictitious producer completion.
            return self.finish('incomplete_missing_capabilities')
        self.phase='load_observation_views'
        self.arrays=self.bundle.load_observations()
        self.annotation=self.reads.json(self.bundle.source_path('annotations'))
        self.ids,self.times,self.frames=join_annotation_rows(self.manifest,self.annotation['frames'])
        self.ids=self.ids.tolist();self.by_id={f['frame_idx']:f for f in self.frames}
        self.positions={fid:i for i,fid in enumerate(self.ids)}
        objects=list(self.manifest['entity_slots']['objects'])
        if len(objects)!=1:raise ValueError('current complete inspection requires one selected object')
        self.object_id=objects[0]
        if self.review_frames is None:
            self.reviews=set(self.ids[i] for i in np.unique(np.linspace(0,len(self.ids)-1,min(6,len(self.ids)),dtype=int)))
        else:
            if len(set(self.review_frames))!=len(self.review_frames) or not set(self.review_frames)<=set(self.ids):
                raise ValueError('review IDs must be unique members of the source timeline')
            self.reviews=set(self.review_frames)
        visual=self.arrays['visual_geometry']
        if visual is not None:self.reviews.add(visual.anchor_frame_idx)
        self.phase='camera_rgb';self.inspect_camera_rgb()
        self.phase='source_mano';self.inspect_mano()
        self.phase='depth_masks_surface';self.inspect_rasters()
        self.phase='appearance';self.inspect_appearance()
        self.phase='interaction';self.inspect_interaction()
        self.phase='rgb_motion';self.inspect_motion()
        self.phase='visual_geometry';self.inspect_geometry()
        self.phase='formal_factors';self.inspect_factors()
        return self.finish('complete_with_measurement_gaps' if self.warnings else 'complete_preprocessing_readback')

    def inspect_camera_rgb(self):
        path=self.artifact('camera_calibration')
        payload,normalized=load_contract(path,expected_frame_ids=self.ids)
        size=tuple(map(int,self.arrays['image_size_wh']))
        K,_=plane_intrinsics(payload,normalized,plane_name='source_rgb',actual_size_wh=size)
        if not np.allclose(self.arrays['intrinsics'],K,atol=.01,rtol=0):
            raise ValueError('annotation K differs from bound camera contract')
        self.camera_payload=payload;self.camera_normalized=normalized
        raw=self.reads.json(self.artifact('raw_frames'));raw_rows=raw['frames']
        if [r['frame_idx'] for r in raw_rows]!=self.ids:raise ValueError('raw RGB timeline mismatch')
        self.raw_rows={r['frame_idx']:r for r in raw_rows}
        self.raw_sizes={};self.review_images={}
        rgb_rows=[]
        for i,row in enumerate(raw_rows):
            if row['time_s']!=self.times[i] or (row['source_width'],row['source_height'])!=size:
                raise ValueError('raw frame time/source raster mismatch')
            p=row.get('raw_frame_path') or row.get('rgb')
            image=self.reads.image(p,row.get('raw_frame_sha256'))
            if image.shape[1::-1]!=(row['manifest_width'],row['manifest_height']):raise ValueError('RGB raster mismatch')
            self.raw_sizes[row['frame_idx']]=image.shape[1::-1]
            if row['frame_idx'] in self.reviews:self.review_images[row['frame_idx']]=image
            rgb_rows.append({'frame_idx':row['frame_idx'],'width':image.shape[1],'height':image.shape[0],'decoded':True})
        video=self.artifact('raw_video');capture=cv2.VideoCapture(str(video));count=0
        try:
            if not capture.isOpened():raise ValueError('raw video is not decodable')
            while True:
                ok,image=capture.read()
                if not ok:break
                if image.shape[1::-1]!=size:raise ValueError('raw video dimensions disagree with annotation')
                count+=1
        finally:capture.release()
        if count!=len(self.ids):raise ValueError('raw video decoded frame count mismatch')
        write_csv(self.output/'rgb_frames.csv',rgb_rows)
        write_npz(self.output/'arrays/camera_timeline.npz',frame_idx=self.ids,time_s=self.times,
            T_world_camera=self.arrays['T_world_camera'],intrinsics_source=self.arrays['intrinsics'],
            intrinsics_fit=self.arrays['source_visual_preprocessing']['intrinsics_fit'])
        runtime=self.reads.json(self.artifact('runtime_input_contract'))
        if runtime.get('input_video_sha256')!=sha(video) or runtime.get('run_id')!=self.bundle.bundle_id:
            raise ValueError('raw input contract differs from observation bundle identity')
        plan=self.reads.json(self.artifact('object_plan'))
        selected=[o for o in plan.get('objects',[]) if o.get('object_id')==self.object_id]
        if len(selected)!=1 or selected[0].get('track_id')!=self.manifest['track_id']:
            raise ValueError('object plan does not identify the selected object/track')
        candidate=self.reads.json(self.artifact('candidate_identity'))
        for key,value in [('object_id',self.object_id),('track_id',self.manifest['track_id'])]:
            if key in candidate and candidate[key]!=value:raise ValueError('detector identity disagrees with selected object')
        self.group('input_metadata',{'raw_video_decoded_frames':count,'metadata_artifacts':'artifact_inventory.csv',
            'weights_loaded':False,'selected_object_id':self.object_id,'selected_track_id':self.manifest['track_id'],
            'classification_is_accuracy_certificate':False},['artifact_inventory.csv'])
        self.group('camera_timeline',{'frame_count':len(self.ids),'calibration_authority':normalized['calibration_authority'],
            'intrinsics_source':K.tolist(),'world_camera_trajectory':'source prediction, not ground truth'},['arrays/camera_timeline.npz'])
        self.group('rgb_frames',{'decoded_frames':len(rgb_rows),'timeline_complete':True},['rgb_frames.csv'])

    def inspect_mano(self):
        views=self.arrays['source_visual_preprocessing'];summary={}
        for side in ('left','right'):
            hand=views['hands'][side]
            write_npz(self.output/f'arrays/{side}_source_preprocessing.npz',
                frame_idx=self.ids,**{key:value for key,value in hand.items() if isinstance(value,np.ndarray)},
                **{'source_'+key:self.arrays['hands'][side][key] for key in
                   ('vertices','joints','faces','root_orient','articulation','betas','translation')})
            summary[side]={'source_valid':int(hand['source_valid'].sum()),
                'pad_shape':list(hand['pad_vertex_ids'].shape),'projectable_vertices':int(hand['vertex_projection_valid'].sum()),
                'complete_source_faces':len(self.arrays['hands'][side]['faces'])}
        bridge=self.reads.npz(self.artifact('mano_bridge'))
        if not {'frame_idx','hand_side','vertices_current_v18_world_from_hawor_projection_relift_m'}<=set(bridge):
            raise ValueError('MANO bridge lacks source full-surface replay data')
        lookup={};max_error=0.
        for j,(frame,side) in enumerate(zip(bridge['frame_idx'],bridge['hand_side'])):
            key=(int(frame),str(side))
            if key in lookup:raise ValueError('duplicate MANO bridge row')
            lookup[key]=j
        for side in ('left','right'):
            for i,fid in enumerate(self.ids):
                if not self.arrays['hands'][side]['valid'][i]:continue
                if (fid,side) not in lookup:raise ValueError('MANO bridge misses valid source surface')
                j=lookup[fid,side]
                a=bridge['vertices_current_v18_world_from_hawor_projection_relift_m'][j]
                b=self.arrays['hands'][side]['vertices'][i]
                if a.shape!=b.shape or not np.allclose(a,b,atol=1e-6,rtol=0):raise ValueError('MANO bridge/source surface mismatch')
                max_error=max(max_error,float(np.max(np.abs(a-b))))
        summary.update(bridge_source_max_error_m=max_error,licensed_mano_lbs_replay=False,
            source_valid_visibility_and_display_are_distinct=True)
        self.group('source_mano',summary,['arrays/left_source_preprocessing.npz','arrays/right_source_preprocessing.npz'])

    def inspect_rasters(self):
        raw_track=track_rows(self.reads.json(self.artifact('raw_object_masks')),self.ids)
        owned_track=track_rows(self.reads.json(self.artifact('owned_object_masks')),self.ids)
        masks={};mask_rows=[]
        # Hash/decode all masks, retain only paths (not whole-video raster stacks).
        for kind,rows in [('raw',raw_track),('owned',owned_track)]:
            masks[kind]={}
            for fid,row in rows.items():
                path=row.get('mask_path')
                if path is None:
                    if row.get('visible') is True:raise ValueError('visible mask track has no image')
                    mask_rows.append({'kind':kind,'frame_idx':fid,'available':False,'area_px':None})
                    continue
                mask=self.reads.image(path,gray=True)
                masks[kind][fid]=Path(path)
                plane_intrinsics(self.camera_payload,self.camera_normalized,plane_name='sam2_mask',actual_size_wh=mask.shape[::-1])
                mask_rows.append({'kind':kind,'frame_idx':fid,'available':True,'area_px':int((mask>0).sum())})
        self.masks=masks
        depth_path=self.artifact('conditioned_depth');depth_rows=[];hashes={};depth_ids=[]
        for fid,depth,confidence,K in stream_depth(depth_path):
            if fid not in self.positions:raise ValueError('foreign depth frame ID')
            if depth.shape[::-1]!=tuple(self.arrays['image_size_wh']) or not np.allclose(K,self.arrays['intrinsics'][self.positions[fid]],atol=.01,rtol=0):
                raise ValueError('depth shape/K differs from source camera')
            hashes[fid]=array_sha256(depth);depth_ids.append(fid)
            good=np.isfinite(depth)&(depth>0)
            depth_rows.append({'frame_idx':fid,'pixels':depth.size,'positive_finite_depth_pixels':int(good.sum()),
                'positive_finite_confidence_pixels':int((np.isfinite(confidence)&(confidence>0)).sum()),
                'depth_median_m':float(np.median(depth[good])) if good.any() else None})
        absent=sorted(set(self.ids)-set(depth_ids))
        if absent:self.warnings.append({'kind':'missing_depth_observation','frame_ids':absent})
        if any(r['positive_finite_depth_pixels']==0 for r in depth_rows):
            self.warnings.append({'kind':'no_positive_depth','frame_ids':[r['frame_idx'] for r in depth_rows if not r['positive_finite_depth_pixels']]})
        eligibility_path=self.artifact('metric_eligibility');ep=self.reads.json(eligibility_path)
        for record in ep['inputs'].values():
            if record is not None:self.reads.path(record['path'],record['sha256'])
        base=ep['inputs'].get('base_annotations')
        eligibility=MetricEligibility(eligibility_path,case=self.manifest['case'],object_id=self.object_id,
            track_id=self.manifest['track_id'],sam2_track=self.artifact('owned_object_masks'),depth_npz=depth_path,
            base_annotations=Path(base['path']) if base else None,calibration_contract=self.artifact('camera_calibration'),
            frame_to_depth_i={fid:i for i,fid in enumerate(depth_ids)},depth_row_hashes=hashes)
        adapter=self.reads.json(self.artifact('surface_report'));surface_rows=[];eligibility_rows=[]
        for fid,depth,confidence,K in stream_depth(depth_path):
            erow=eligibility.rows[fid];rec=erow['raster'];blob=self.reads.npz(rec['path'],rec['sha256'])
            if fid in masks['owned']:
                mask=self.reads.image(masks['owned'][fid],gray=True)
                eligible,diag=eligibility.frame(fid,owned_mask_path=masks['owned'][fid],depth_shape=depth.shape,
                    A_depth_from_mask=resize_affine(mask.shape[::-1],depth.shape[::-1]),K_depth=K_from_intrinsics(K))
            else:
                eligible=blob['metric_candidate_eligible'];classes=blob['class_code']
                if (set(blob)!={'schema','frame_idx','class_code','metric_candidate_eligible'}
                        or str(blob['schema'].item())!=RASTER_SCHEMA or int(blob['frame_idx'].item())!=fid
                        or classes.shape!=depth.shape or classes.dtype!=np.uint8 or eligible.shape!=depth.shape or eligible.dtype!=bool
                        or array_sha256(classes)!=erow['class_code_array_sha256']
                        or array_sha256(eligible)!=erow['metric_candidate_array_sha256']
                        or not set(np.unique(classes))<=set(CLASS_CODES.values())):
                    raise ValueError('unavailable-frame eligibility raster is malformed')
                if erow.get('status')!='raw_track_not_visible' or eligible.any():raise ValueError('missing owned mask cannot grant metric support')
            eligibility_rows.append({'frame_idx':fid,'metric_candidate_pixels':int(eligible.sum()),
                'class_counts_json':json.dumps(dict(zip(*[x.tolist() for x in np.unique(blob['class_code'],return_counts=True)])))})
            frame=self.by_id[fid];obj=next((o for o in frame.get('objects',[]) if o.get('object_id')==self.object_id),None)
            points,reason,_=retained_metric_alignment(obj,frame)
            retained=None
            if points is not None:
                mask=self.reads.image(obj['mask_path'],gray=True)
                retained=reproduce_source_depth_support(obj,mask,depth,confidence,K,adapter['parameters'],metric_eligibility=eligibility)
                T=self.arrays['T_world_camera'][self.positions[fid]];cam=(points-T[:3,3])@T[:3,:3]
                uv=cam[:,:2]/cam[:,2:]*K[:2]+K[2:]
                pixel=np.rint(uv).astype(int);inside=(pixel[:,0]>=0)&(pixel[:,0]<depth.shape[1])&(pixel[:,1]>=0)&(pixel[:,1]<depth.shape[0])
                residual=cam[inside,2]-depth[pixel[inside,1],pixel[inside,0]]
                write_npz(self.output/f'arrays/p09_{fid:06d}.npz',source_world=points,source_camera=cam,source_uv=uv,inside_depth=inside,depth_residual_m=residual)
                surface_rows.append({'frame_idx':fid,'state':'retained_support_replayed','point_count':len(points),
                    'retained_pixels':int(retained.sum()),'median_abs_sample_depth_delta_m':float(np.median(np.abs(residual))) if len(residual) else None})
            else:
                surface_rows.append({'frame_idx':fid,'state':reason,'point_count':0,'retained_pixels':None})
            if fid in self.reviews:
                color,scale=depth_view(depth);image=self.review_images[fid]
                owned=self.reads.image(masks['owned'][fid],gray=True)>0 if fid in masks['owned'] else None
                raw=self.reads.image(masks['raw'][fid],gray=True)>0 if fid in masks['raw'] else None
                views=[panel(image,'SOURCE RGB - observation QC only'),panel(mask_view(image,raw,(50,200,50)),'raw predicted mask'),
                    panel(mask_view(image,owned,(255,100,50)),'depth-owned mask'),panel(cv2.resize(color,(image.shape[1],image.shape[0])),'depth camera-z (uncertain)')]
                self.image_output(f'input_{fid:06d}.jpg',np.concatenate(views,axis=1))
        for fid in absent:surface_rows.append({'frame_idx':fid,'state':'missing_depth_not_synthesized','point_count':0,'retained_pixels':None})
        write_csv(self.output/'depth_frames.csv',depth_rows);write_csv(self.output/'mask_frames.csv',mask_rows)
        write_csv(self.output/'eligibility_frames.csv',eligibility_rows);write_csv(self.output/'surface_frames.csv',surface_rows)
        self.group('depth_masks_surface',{'depth_frames_decoded':len(depth_rows),'depth_missing_ids':absent,
            'mask_images_decoded':sum(r['available'] for r in mask_rows),'eligibility_rasters_decoded':len(eligibility_rows),
            'P09_retained_frames_replayed':sum(r['state']=='retained_support_replayed' for r in surface_rows),
            'two_streaming_depth_passes':True,'depth_numeric_hash_dtype':'float32_same_as_P09_consumer',
            'never_treated_as_metric_ground_truth':True},
            ['depth_frames.csv','mask_frames.csv','eligibility_frames.csv','surface_frames.csv','arrays/p09_*.npz'])

    def inspect_appearance(self):
        appearance=self.arrays['appearance_evidence']
        if appearance is None:
            self.missing.append('appearance_preprocessing');return
        rows=[];render_counts={}
        for fid in self.ids:
            row=appearance.rows[fid]
            if row['status']=='available':
                self.reads.path(row['appearance_mask_path'],row['appearance_mask_sha256'])
                self.reads.path(row['source_mask_path'],row['source_mask_sha256'])
                self.reads.path(row['raw_rgb_path'],row['raw_rgb_sha256'])
            prepared=appearance.prepare_frame(fid)
            if not prepared['available']:
                rows.append({'frame_idx':fid,'available':False,'reason':prepared['reason']});continue
            rows.append({'frame_idx':fid,'available':True,'mask_pixels':int(prepared['mask'].sum()),
                'contour_points':len(prepared['contour_uv']),'max_mask_distance_px':float(prepared['distance_transform_px'].max())})
            if fid in self.reviews:
                write_npz(self.output/f'arrays/appearance_{fid:06d}.npz',**{k:v for k,v in prepared.items() if isinstance(v,np.ndarray)})
                canvas,counts=source_view(self.review_images[fid],self.arrays,self.positions[fid],prepared,self.arrays['visual_geometry'])
                render_counts[str(fid)]=counts
                label='SYNTHETIC FIXTURE QC' if self.manifest.get('synthetic_fixture') else 'SOURCE OBSERVATION QC (not policy result)'
                self.image_output(f'geometry_{fid:06d}.jpg',panel(canvas,label+' | triangles, pads, raw contour',width=960))
        unavailable=[r['frame_idx'] for r in rows if not r['available']]
        if unavailable:self.warnings.append({'kind':'appearance_unavailable','frame_ids':unavailable})
        write_csv(self.output/'appearance_frames.csv',rows);write_json(self.output/'review_geometry_counts.json',render_counts)
        self.group('appearance',{'frames_preprocessed':len(rows),'unavailable_ids':unavailable,
            'fit_width':960,'all_frames_processed_review_arrays_sampled_only':True},['appearance_frames.csv','review_geometry_counts.json','arrays/appearance_*.npz'])

    def inspect_interaction(self):
        evidence=self.arrays['interaction_evidence'];timeline=self.arrays['interaction_timelines'][self.object_id]
        if evidence.schema is None:self.missing.append('versioned_typed_interaction')
        for source in evidence.source.get('source_judgments',()):
            rec=source.get('receipt')
            if rec is not None:
                actual=self.reads.json(rec['path'],rec['sha256'])
                if actual!=thaw(source['result']):raise ValueError('typed judgment differs from original source JSON')
        write_npz(self.output/'arrays/interaction_timeline.npz',**{f.name:getattr(timeline,f.name) for f in fields(timeline)})
        declarations=[thaw(vars(v)) for v in evidence.declarations.values()]
        # Nested pad dataclasses are serialized explicitly, without flattening away uncertainty.
        for result,value in zip(declarations,evidence.declarations.values()):result['pads']=[vars(p) for p in value.pads]
        write_json(self.output/'interaction_declarations.json',declarations)
        write_json(self.output/'visibility_declarations.json',[vars(v) for v in evidence.visibility.values()])
        rows=[]
        for i,fid in enumerate(self.ids):
            for j,side in enumerate(('left','right')):
                rows.append({'frame_idx':fid,'hand':side,'contact_state':timeline.contact_state[i,j],
                    'visibility':timeline.visibility[i,j+1],'source_mano_valid':bool(self.arrays['hands'][side]['valid'][i]),
                    'observation_display_eligible':bool(timeline.display_observation_eligible[i,j+1]),
                    'primary_declared':int(timeline.primary_declared[i,j].sum()),
                    'primary_known':int(timeline.primary_known[i,j].sum()),
                    'primary_visual_eligible':int(timeline.primary_visual_eligible[i,j].sum()),
                    'actual_policy_display_changed':False})
        write_csv(self.output/'interaction_frames.csv',rows)
        unknown=int((timeline.visibility=='unresolved').sum())
        if unknown:self.warnings.append({'kind':'unresolved_visibility_entries','count':unknown})
        self.group('interaction',{'declarations':len(declarations),'visibility_entries':len(evidence.visibility),
            'unresolved_visibility_entries':unknown,'compiled_shape':list(timeline.primary_declared.shape),
            'display_eligibility_is_not_actual_policy_display':True},
            ['arrays/interaction_timeline.npz','interaction_frames.csv','interaction_declarations.json','visibility_declarations.json'])

    def inspect_motion(self):
        evidence=self.arrays['rgb_evidence'];motion=self.arrays['motion_preprocessing']
        if evidence is None:self.missing.append('rgb_measurement_reader');return
        for rec in evidence.report['inputs'].values():self.reads.path(rec['path'],rec['sha256'])
        rows=[]
        for i,edge in enumerate(evidence.edges):
            target_world=edge.source_world@edge.rotation_source_to_target_world.T+edge.translation_source_to_target_world_m
            camera=(target_world-edge.target_camera[:3,3])@edge.target_camera[:3,:3]
            predicted=camera[:,:2]/camera[:,2:]*edge.target_intrinsics[:2]+edge.target_intrinsics[2:]
            residual=predicted-edge.target_uv
            plane=evidence.report['image_planes'][str(edge.target_frame_idx)]
            matcher_error=np.linalg.norm(residual*(np.asarray(plane['manifest_size_wh'])/edge.target_size_wh),axis=1)
            write_npz(self.output/f'arrays/rgb_edge_{i:06d}.npz',source_world=edge.source_world,source_uv=edge.source_uv,
                target_uv=edge.target_uv,target_uv_projected=predicted,residual_native_px=residual,
                residual_matcher_norm_px=matcher_error,point_weights=edge.weights)
            rows.append({'edge_index':i,'source_frame_idx':edge.source_frame_idx,'target_frame_idx':edge.target_frame_idx,
                'source_points':len(edge.source_world),'median_matcher_error_px':float(np.median(matcher_error)),
                'p95_matcher_error_px':float(np.percentile(matcher_error,95)),
                'reprojection_edge_weight':motion.qualification['reprojection_edge_weights'][i] if motion else None,
                'loop_decision':motion.qualification['edges'][i]['reason'] if motion else 'not_prepared'})
            if i<6:
                a=self.reads.image(evidence.report['inputs'][f'rgb:{edge.source_frame_idx}']['path'])
                b=self.reads.image(evidence.report['inputs'][f'rgb:{edge.target_frame_idx}']['path'])
                self.image_output(f'rgb_edge_{i:06d}.jpg',edge_view(a,b,edge,predicted))
        write_csv(self.output/'rgb_edges.csv',rows)
        write_json(self.output/'rgb_measurement_diagnostics.json',evidence.report)
        if not rows:self.warnings.append({'kind':'measured_empty_rgb_edges','reason':'no identity/zero-motion substitution'})
        self.group('rgb_motion',{'edge_count':len(rows),'evidence_point_count':sum(r['source_points'] for r in rows),
            'empty_is_valid_measurement_result':not rows,'all_edge_reprojections_computed':True},
            ['rgb_edges.csv','rgb_measurement_diagnostics.json','arrays/rgb_edge_*.npz'])
        if motion is None:
            self.missing.append('motion_preprocessing_without_explicit_anchor');return
        write_json(self.output/'motion_qualification.json',motion.qualification)
        source_initialization=motion.source_initialization
        comparison=motion.rgb_seed_comparison
        write_json(self.output/'source_pose_initialization.json',source_initialization)
        write_json(self.output/'measurement_seed_comparison.json',comparison)
        # Historical filename remains, but is explicitly the comparison artifact.
        write_json(self.output/'measurement_seed.json',comparison)
        unknown=[r['frame_idx'] for r in source_initialization['frames'] if not r['available']]
        comparison_unknown=[r['frame_idx'] for r in comparison['frames'] if not r['available']]
        if unknown:self.warnings.append({'kind':'unresolved_source_pose_initialization','frame_ids':unknown})
        if comparison_unknown:self.warnings.append({'kind':'unresolved_measurement_seed_comparison','frame_ids':comparison_unknown})
        self.group('motion_seed',{'role':'R4_source_pose_candidate_baseline_not_RGB_seed','frames':len(self.ids),
            'unavailable_frame_ids':unknown,'rgb_seed_comparison_unavailable_frame_ids':comparison_unknown,
            'rgb_seed_is_comparison_only':True,'policy_state_updated':False},
            ['motion_qualification.json','source_pose_initialization.json','measurement_seed_comparison.json'])

    def inspect_geometry(self):
        geometry=self.arrays['visual_geometry']
        if geometry is None:self.missing.append('complete_visual_geometry_reader');return
        for rec in geometry.report['inputs'].values():self.reads.path(rec['path'],rec['sha256'])
        world=geometry.vertices_metric_camera@geometry.T_world_camera[:3,:3].T+geometry.T_world_camera[:3,3]
        values={'vertices_metric_camera':geometry.vertices_metric_camera,'vertices_world_at_anchor':world,'faces':geometry.faces}
        summary={'vertices':len(world),'faces':len(geometry.faces),'anchor_frame_idx':geometry.anchor_frame_idx,
            'physical_authority':False,'image_scale_search_performed':False,'seed_binding_performed':False}
        motion=self.arrays['motion_preprocessing']
        seed=next((r for r in motion.source_initialization['frames'] if r['frame_idx']==geometry.anchor_frame_idx),None) if motion else None
        if seed and seed['available']:
            binding=bind_visual_to_candidate(geometry,seed['rotation_world_from_canonical'],seed['translation_world_m'],
                candidate_revision='inspection-measurement-seed-'+sha(self.source)[:16])
            canonical=binding['vertices_canonical'];R=np.asarray(seed['rotation_world_from_canonical']);t=np.asarray(seed['translation_world_m'])
            values['vertices_canonical_at_seed']=canonical
            summary.update(seed_binding_performed=True,canonical_world_roundtrip_max_m=float(np.max(np.abs(canonical@R.T+t-world))),
                binding_role='R4_inspection_only_not_installed_in_policy')
        else:self.warnings.append({'kind':'visual_binding_not_evaluated_no_seed_at_appearance_anchor'})
        old=self.arrays['objects'][self.object_id]
        summary['separate_geometry_roles']={}
        for role,vkey,fkey in [('pose_hypothesis','pose_vertices_canonical','pose_faces'),
                              ('collision','collision_vertices_canonical','collision_faces')]:
            v=old[vkey];f=old[fkey]
            if not np.isfinite(v).all() or (f.size and (f.min()<0 or f.max()>=len(v))):
                raise ValueError('invalid source geometry in role '+role)
            write_npz(self.output/f'arrays/{role}_source_geometry.npz',vertices_canonical=v,faces=f)
            summary['separate_geometry_roles'][role]={'vertices':len(v),'faces':len(f)}
        write_npz(self.output/'arrays/complete_visual_geometry.npz',**values)
        write_json(self.output/'visual_geometry_check.json',summary)
        self.group('visual_geometry',summary,['arrays/complete_visual_geometry.npz','visual_geometry_check.json'])

    def inspect_factors(self):
        counts={k:len(v) for k,v in self.arrays['factor_rows_by_family'].items()}
        rows=[]
        for family,factors in self.arrays['factor_rows_by_family'].items():
            for row in factors:
                for key,value in row.items():
                    if key.endswith('_path') and isinstance(value,str) and Path(value).suffix.lower() in {'.png','.npz'}:
                        expected=row.get(key[:-5]+'_sha256')
                        if value.endswith('.png'):self.reads.image(value,expected,gray=True)
                        else:self.reads.npz(value,expected)
                rows.append({'frame_idx':row.get('frame_idx'),'side':row.get('hand_side'),'family':family,'state':row.get('state')})
        write_csv(self.output/'factor_rows.csv',rows)
        per_hand={}
        for side,hand in self.arrays['hands'].items():
            per_hand[side]=[{'frame_idx':fid,'state':q['state'],'sample_count':int(q.get('translation_support_count',0)),
                'initial_depth_delta_m':numeric_summary(q.get('initial_delta_m',[]))}
                for fid,q in zip(self.ids,hand['depth_order'])]
        write_json(self.output/'first_surface_queries.json',per_hand)
        query_samples=sum(r['sample_count'] for rows in per_hand.values() for r in rows)
        self.group('formal_factors',{'factor_families':counts,'source_first_surface_queries_consumed':query_samples>0,
            'source_first_surface_query_samples':query_samples,
            'legacy_contact_patch_prepared':self.arrays['legacy_contact_patch_prepared'],
            'no_new_signed_or_contact_authority':True},['factor_rows.csv','first_surface_queries.json'])

    def finish(self,status):
        self.phase='verify_source_unchanged'
        self.reads.verify_unchanged()
        if self.missing:status='incomplete_missing_capabilities'
        connections=[]
        for name,(description,legacy,pending) in CONSUMERS.items():
            item=self.groups.get(name)
            connections.append({'group':name,'observation':description,
                'inspection_consumer':item['status'] if item else 'not_consumed_missing_input',
                'inspection_outputs':'; '.join(item['outputs']) if item else '',
                'legacy_path':legacy,'not_connected_downstream':pending,'new_v2_policy_connected':False})
        write_csv(self.output/'consumer_matrix.csv',connections)
        artifact_consumers=[]
        for name,record in self.manifest['artifacts'].items():
            group=ARTIFACT_GROUPS[name];state=self.groups.get(group)
            artifact_consumers.append({'artifact':name,'role':record['role'],'group':group,
                'inspection_consumption':state['status'] if state else 'not_consumed_in_incomplete_inspection',
                'legacy_path':CONSUMERS[group][1],'pending_consumers':CONSUMERS[group][2],
                'v2_policy_connected':False})
        write_csv(self.output/'artifact_consumers.csv',artifact_consumers)
        report={'schema':'ego.annotation.observation_inspection.v1','status':status,
            'source_manifest':str(self.source),'source_manifest_sha256':sha(self.source),
            'source_schema':self.manifest['schema'],'synthetic_fixture':bool(self.manifest.get('synthetic_fixture',False)),
            'capabilities':self.manifest.get('capabilities',()),'missing':self.missing,'warnings':self.warnings,
            'groups':self.groups,'source_files_read_and_hash_rechecked':len(self.reads.files),
            'source_unchanged':True,'source_integrity_scope':'outer bundle receipts verified; transitive path-only auxiliaries are hashed before/after, not granted invented producer receipts',
            'training_or_policy_executed':False,'model_inference_executed':False,
            'annotation_quality_verified':False,'physical_authority_granted':False,
            'reviews_are_selected_frame_QC_not_complete_annotation':True,
            'review_images':self.outputs,'pending_consumers':'consumer_matrix.csv'}
        write_json(self.output/'source_reads.json',self.reads.files)
        write_json(self.output/'inspection.json',report)
        lines=['# V2 观测/前处理独立检查','',f'状态：`{status}`',f'源包：`{self.source}`',
            '', '**合成夹具，不是真实 milk 预测。**' if report['synthetic_fixture'] else '**只读观测检查，不是新预测或最终标注质量认证。**','',
            '所有声明时间轴的数据逐帧消费；检查图只覆盖选定帧，不能作为完整标注视频。',
            '未调用 PPO、policy、感知模型或优化器；原始源文件读取后再次核验哈希。','',
            '|观测组|独立读取/前处理|尚未接入的下游|','|---|---|---|']
        for r in connections:lines.append(f"|{r['observation']}|{r['inspection_consumer']}|{r['not_connected_downstream']}|")
        lines+=['','## 检查图']+[f'- [{p}]({p})' for p in self.outputs]
        lines+=['','详细数值：`inspection.json`、`arrays/`、逐帧CSV、`consumer_matrix.csv`。',
            '空RGB边、缺深度、未知visibility保留为测量缺口，不伪造为零运动或可见。']
        (self.output/'README.md').write_text('\n'.join(lines)+'\n')
        write_json(self.output/'output_hashes.json',{str(p.relative_to(self.output)):sha(p) for p in sorted(self.output.rglob('*')) if p.is_file()})
        return report


def inspect_bundle(source,output,review_frames=None):
    inspection=Inspection(source,output,review_frames)
    try:
        return inspection.run()
    except Exception as exc:
        write_json(inspection.output/'failure.json',{'status':'inspection_failed','phase':inspection.phase,
            'exception_type':type(exc).__name__,'message':str(exc),'source':str(inspection.source),
            'partial_outputs_are_not_acceptance':True,'training_or_policy_executed':False})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-observations',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review-frames',type=int,nargs='*',default=None,help='QC frame IDs only; numeric checks still cover all frames')
    args=parser.parse_args(argv)
    try:
        report=inspect_bundle(args.source_observations,args.output_dir,args.review_frames)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 1
    print(json.dumps({'status':report['status'],'report':str(args.output_dir/'inspection.json'),
        'source_unchanged':report['source_unchanged'],'training_or_policy_executed':False},ensure_ascii=False))
    return 2 if report['missing'] else 0


if __name__=='__main__':raise SystemExit(main())
