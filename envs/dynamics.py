from __future__ import annotations

"""Domain assembly, semantic action filtering, and physical candidate decoding."""

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import inspect
import math

import numpy as np

from actions.scaling import ActionScales, GateTrace, filter_scaled_action
from actions.space import ActionLayout
from data.bundle import EpisodeBundle, file_sha256
from envs.state import AnnotationState, EpisodeStatic


def _cfg(cfg: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _load_mano_models(bundle: EpisodeBundle, cfg: Mapping[str, Any] | Any, device: str) -> Mapping[str, Any]:
    supplied = _cfg(cfg, "mano_models")
    if supplied is not None:
        if set(supplied) != {"left", "right"}:
            raise ValueError("mano_models must contain left and right")
        return MappingProxyType(dict(supplied))
    assets = bundle.manifest["sources"].get("mano_assets")
    if not isinstance(assets, Mapping):
        raise RuntimeError("present hand slots require real MANO models or manifest-bound mano_assets")
    asset_root = Path(assets["path"])
    wrapper_path = Path(assets["wrapper_path"]).resolve()
    if file_sha256(wrapper_path) != assets["wrapper_sha256"]:
        raise ValueError("manifest-bound WiLoR wrapper changed before MANO model loading")
    expected_wilor_root = wrapper_path.parents[2]
    configured_root = _cfg(cfg, "wilor_root", None)
    if configured_root is not None and Path(configured_root).expanduser().resolve() != expected_wilor_root:
        raise ValueError("configured wilor_root differs from manifest-bound wrapper root")
    wilor_root = expected_wilor_root
    for name, value in (("bool", np.bool_), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("unicode", str), ("str", str)):
        if name not in np.__dict__: setattr(np, name, value)
    if not hasattr(inspect, "getargspec"): inspect.getargspec = inspect.getfullargspec
    from utils.mano import load_wilor_mano_class
    mano_cls = load_wilor_mano_class(wilor_root)
    models = {}
    for side in ("left", "right"):
        path = asset_root / f"MANO_{side.upper()}.pkl"
        if not path.is_file(): raise FileNotFoundError(f"missing licensed MANO asset: {path}")
        model = mano_cls(model_path=str(path), is_rhand=(side == "right"), use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
        if side == "left" and bool(_cfg(cfg, "hawor_left_shapedirs_x_fix", True)):
            import torch
            with torch.no_grad(): model.shapedirs[:, 0, :] *= -1
        model.eval(); models[side] = model
    return MappingProxyType(models)


def build_episode(bundle: EpisodeBundle, cfg: Mapping[str, Any] | Any, device: str = "cpu") -> EpisodeStatic:
    arrays = bundle.load(); max_objects = int(_cfg(cfg, "max_objects", 1))
    if max_objects < len(arrays["objects"]): raise ValueError("max_objects is smaller than declared object entities")
    layout = ActionLayout(object_slots=tuple(f"object:{i}" for i in range(max_objects)),
                          contact_variables_per_hand=int(_cfg(cfg, "contact_variables_per_hand", 1)))
    source_objects = list(arrays["objects"].items()); count = len(arrays["frame_idx"]); objects: dict[str, Mapping[str, Any]] = {}
    for index, slot in enumerate(layout.object_slots):
        if index < len(source_objects):
            entity, value = source_objects[index]; objects[slot] = MappingProxyType({**dict(value), "entity_id": entity})
        else:
            objects[slot] = MappingProxyType({"entity_id": None, "rotation": np.tile(np.eye(3),(count,1,1)), "translation": np.zeros((count,3)), "present": np.zeros(count,bool), "pose_valid": np.zeros(count,bool), "metric_observation_eligible": np.zeros(count,bool), "valid": np.zeros(count,bool), "pose_vertices_canonical": np.zeros((0,3),np.float32), "pose_faces": np.zeros((0,3),np.int64), "vertices_canonical": np.zeros((0,3),np.float32), "faces": np.zeros((0,3),np.int64), "collision_vertices_canonical": np.zeros((0,3),np.float32), "collision_faces": np.zeros((0,3),np.int64), "watertight": False, "signed_geometry_ready": False, "signed_authority": np.zeros((0,),bool), "observed_points_world": tuple(np.zeros((0,3)) for _ in range(count)), "raw_observed_points_world": tuple(np.zeros((0,3)) for _ in range(count)), "observed_point_state": tuple("inactive_absent_entity" for _ in range(count)), "pose_provenance": "absent_entity_slot", "geometry_provenance": "absent_entity_slot"})
    episode = EpisodeStatic(bundle_id=bundle.bundle_id, clip_id=str(bundle.manifest["clip_id"]), source_video_id=bundle.source_video_id,
        frame_idx=np.asarray(arrays["frame_idx"]), time_s=np.asarray(arrays["time_s"]), T_world_camera=np.asarray(arrays["T_world_camera"]), intrinsics=np.asarray(arrays["intrinsics"]), image_size_wh=np.asarray(arrays["image_size_wh"]), hands=arrays["hands"], objects=MappingProxyType(objects), contact=arrays["contact_legacy"], contacts=arrays["contact"], factor_rows=arrays["factor_rows"], layout=layout, mano_models=_load_mano_models(bundle,cfg,device), provenance=MappingProxyType({**dict(arrays["provenance"]),"sources":dict(bundle.manifest["sources"]),"observation_point_samples":int(_cfg(cfg,"observation_point_samples",8))}), identity=arrays["identity"])
    if bool(_cfg(cfg,"verify_mano_replay",True)): _verify_mano_replay(episode,device=device,tolerance_m=float(_cfg(cfg,"mano_replay_tolerance_m",1e-4)))
    return episode


def _model_forward(model: Any, root: np.ndarray, pose: np.ndarray, betas: np.ndarray, translation: np.ndarray, device: str) -> tuple[np.ndarray,np.ndarray]:
    import torch
    root_t=torch.as_tensor(np.array(root,copy=True),dtype=torch.float32,device=device).reshape(1,1,3); pose_t=torch.as_tensor(np.array(pose,copy=True),dtype=torch.float32,device=device).reshape(1,15,3); beta_t=torch.as_tensor(np.array(betas,copy=True),dtype=torch.float32,device=device).reshape(1,-1); trans_t=torch.as_tensor(np.array(translation,copy=True),dtype=torch.float32,device=device).reshape(1,3)
    def matrices(rotvec: Any):
        x,y,z=rotvec.unbind(dim=-1); zero=torch.zeros_like(x); skew=torch.stack([torch.stack([zero,-z,y],-1),torch.stack([z,zero,-x],-1),torch.stack([-y,x,zero],-1)],-2); return torch.linalg.matrix_exp(skew)
    with torch.no_grad():
        try: output=model(global_orient=matrices(root_t),hand_pose=matrices(pose_t),betas=beta_t,transl=trans_t,return_verts=True,pose2rot=False)
        except (TypeError,RuntimeError):
            try: output=model(global_orient=root_t.reshape(1,3),hand_pose=pose_t.reshape(1,45),betas=beta_t,transl=trans_t,return_verts=True)
            except TypeError:
                output=model(global_orient=root_t.reshape(1,3),hand_pose=pose_t.reshape(1,45),betas=beta_t); output.vertices=output.vertices+trans_t[:,None]; output.joints=output.joints+trans_t[:,None]
    vertices,joints=output.vertices,output.joints
    if joints.shape[1]<21 and vertices.shape[1]==778:
        from utils.mano import JOINT_ORDER,TIP_VERTICES
        joints=torch.cat([joints[:,:16],vertices[:,TIP_VERTICES]],1)[:,JOINT_ORDER]
    elif joints.shape[1]>21: joints=joints[:,:21]
    return vertices[0].detach().cpu().numpy().astype(np.float32),joints[0].detach().cpu().numpy().astype(np.float32)


def _verify_mano_replay(episode: EpisodeStatic, *, device: str, tolerance_m: float) -> None:
    for side,values in episode.hands.items():
        valid=np.flatnonzero(values["valid"])
        for i in (int(valid[0]),int(valid[-1])) if len(valid) else ():
            vertices,_=_model_forward(episode.mano_models[side],values["root_orient"][i],values["articulation"][i],values["betas"][i],values["translation"][i],device); error=np.linalg.norm(vertices-values["vertices"][i],axis=1)
            if not np.isfinite(error).all() or float(error.max())>tolerance_m: raise RuntimeError(f"{side} MANO replay mismatch max={float(error.max()):.6g}m")


def _rotvec_matrix(value: Sequence[float]) -> np.ndarray:
    vector=np.asarray(value,float); angle=float(np.linalg.norm(vector));
    if angle<1e-12:
        return np.eye(3)+np.asarray([[0.,-vector[2],vector[1]],[vector[2],0.,-vector[0]],[-vector[1],vector[0],0.]])
    x,y,z=vector/angle; K=np.asarray([[0.,-z,y],[z,0.,-x],[-y,x,0.]]); return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)


def _matrix_rotvec(matrix: np.ndarray) -> np.ndarray:
    R=np.asarray(matrix,float); angle=float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    axis=np.asarray([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    return axis/2 if angle<1e-8 else axis/max(2*np.sin(angle),1e-12)*angle


def _contact_rows(episode: EpisodeStatic, frame_id: int, side: str) -> list[tuple[str,Mapping[str,Any]]]:
    return [(target,row) for (frame,row_side,target),row in episode.contacts.items() if frame==frame_id and row_side==side]


def _slot_for_entity(episode: EpisodeStatic, entity_id: str) -> str | None:
    matches=[slot for slot in episode.layout.object_slots if episode.objects[slot].get("entity_id")==entity_id]
    return matches[0] if len(matches)==1 else None


def action_gate_mask(state: AnnotationState, cfg: Mapping[str,Any] | Any) -> tuple[np.ndarray,tuple[str,...]]:
    episode=state.episode; i=state.position; frame=int(episode.frame_idx[i]); hand_present=[bool(episode.hands[side]["valid"][i]) for side in episode.layout.hand_slots]; object_present=[bool(episode.objects[slot].get("present",episode.objects[slot]["valid"])[i]) and bool(episode.objects[slot]["pose_valid"][i]) for slot in episode.layout.object_slots]
    active=[]
    for side,present in zip(episode.layout.hand_slots,hand_present):
        active.append(bool(present and any(row.get("state")=="active_contact_patch" and (slot:=_slot_for_entity(episode,target)) is not None and object_present[episode.layout.object_slots.index(slot)] for target,row in _contact_rows(episode,frame,side))))
    allowed=episode.layout.presence_mask(hand_present,object_present,active); reasons=np.full(episode.layout.size,"allowed",object); reasons[~allowed]="absent_entity_or_inactive_contact"
    if bool(_cfg(cfg,"freeze_translation_without_visible_surface_support",False)):
        threshold=int(_cfg(cfg,"translation_gate_min_visible_surface_depth_vertices",0))
        for side,present in zip(episode.layout.hand_slots,hand_present):
            if present and int(episode.hands[side]["depth_order"][i].get("translation_support_count",0))<=threshold:
                block=episode.layout.block(f"hand.{side}.translation");allowed[block.slice]=False;reasons[block.slice]="missing_visible_surface_translation_support"
    explicit=_cfg(cfg,"hard_action_mask")
    if explicit is not None:
        mask=np.asarray(explicit,bool); mask=mask[i] if mask.ndim==2 else mask
        if mask.shape!=(episode.layout.size,): raise ValueError("hard_action_mask shape mismatch")
        newly=allowed&~mask;allowed&=mask;reasons[newly]="configured_semantic_hard_gate"
    return allowed,tuple(map(str,reasons.tolist()))


def filter_action(state: AnnotationState, raw_action: Sequence[float], cfg: Mapping[str,Any] | Any) -> tuple[np.ndarray,GateTrace]:
    value=_cfg(cfg,"action_scales",{}); scales=value if isinstance(value,ActionScales) else ActionScales(**dict(value)); allowed,reasons=action_gate_mask(state,cfg); trace=filter_scaled_action(raw_action,state.episode.layout,scales,allowed,reason_by_dimension=reasons); return trace.executed_delta.copy(),trace


def decode_candidate(state: AnnotationState, executed_delta: Sequence[float], cfg: Mapping[str,Any] | Any, device: str="cpu") -> None:
    episode=state.episode
    if state.done: raise RuntimeError("cannot decode action after episode completion")
    i=state.position; decoded=episode.layout.decode(executed_delta)
    for side in episode.layout.hand_slots:
        if not bool(episode.hands[side]["valid"][i]): continue
        root_delta,pose_delta,trans_delta=decoded[f"hand.{side}.root"],decoded[f"hand.{side}.articulation"],decoded[f"hand.{side}.translation"]; state.hand_root[side][i]=_matrix_rotvec(_rotvec_matrix(root_delta)@_rotvec_matrix(episode.hands[side]["root_orient"][i])); base=episode.hands[side]["articulation"][i].reshape(15,3); delta=pose_delta.reshape(15,3);state.hand_articulation[side][i]=np.stack([_matrix_rotvec(_rotvec_matrix(delta[j])@_rotvec_matrix(base[j])) for j in range(15)]).reshape(45);state.hand_translation[side][i]=episode.hands[side]["translation"][i]+trans_delta;state.hand_vertices[side][i],state.hand_joints[side][i]=_model_forward(episode.mano_models[side],state.hand_root[side][i],state.hand_articulation[side][i],episode.hands[side]["betas"][i],state.hand_translation[side][i],device)
    for slot in episode.layout.object_slots:
        obj=episode.objects[slot]
        if not bool(obj["valid"][i]): continue
        state.object_rotation[slot][i]=_rotvec_matrix(decoded[f"object.{slot}.rotation"])@np.asarray(obj["rotation"][i]); delta_t=decoded[f"object.{slot}.translation"].astype(float); state.object_translation[slot][i]=np.asarray(obj["translation"][i])+(episode.T_world_camera[i,:3,:3]@delta_t if _cfg(cfg,"object_translation_frame","world")=="camera" else delta_t)
    for index,side in enumerate(episode.layout.hand_slots):
        if episode.layout.contact_variables_per_hand:
            priors=[float(row.get("prior",.5)) for _,row in _contact_rows(episode,int(episode.frame_idx[i]),side)];prior=float(np.clip(np.mean(priors) if priors else .5,1e-4,1-1e-4));state.contact_logits[i,index]=np.log(prior/(1-prior))+decoded[f"contact.{side}"]
    state.applied[i]=True;state.term_cache.pop(i,None)


def transformed_object_geometry(state: AnnotationState, frame_position: int, slot_name: str) -> tuple[np.ndarray,np.ndarray]:
    obj=state.episode.objects[slot_name];return np.asarray(obj["pose_vertices_canonical"],float)@state.object_rotation[slot_name][frame_position].T+state.object_translation[slot_name][frame_position],np.asarray(obj["pose_faces"],np.int64)


def refresh_candidate_terms(state: AnnotationState, cfg: Mapping[str,Any] | Any, frame_position: int | None=None) -> dict[str,Any]:
    """Refresh only candidate-dependent object NN/signed sets.

    Contact membership is frozen at episode assembly. Default contact terms only
    transform stored object-coordinate anchors under the declared target SE(3).
    """
    i=state.position if frame_position is None else int(frame_position);episode=state.episode;frame=int(episode.frame_idx[i]);result={"contact":{},"contact_by_key":{},"signed":{},"object_observed":{}}
    for slot in episode.layout.object_slots:
        obj=episode.objects[slot]
        if not bool(obj["valid"][i]): result["signed"][slot]={"state":"inactive_absent_object","hands":{}};continue
        authority=np.asarray(obj.get("signed_authority",[]),bool); eligible=bool(obj.get("watertight",False) and obj.get("signed_geometry_ready",False) and len(authority)==len(obj["collision_faces"]) and np.any(authority)); signed={"state":"active_declared_mesh_bound_local_face_authority" if eligible else "inactive_nonwatertight_or_missing_local_face_authority","hands":{}};result["signed"][slot]=signed
        if eligible:
            from types import SimpleNamespace
            from utils.candidate_geometry import make_scene,active_constraints_from_vertices,dense_observed_surface_constraints_from_vertices
            scene=make_scene(obj["collision_vertices_canonical"],obj["collision_faces"])
            for side in episode.layout.hand_slots:
                if not bool(episode.hands[side]["valid"][i]): continue
                row=SimpleNamespace(object_rotation_world_from_object=state.object_rotation[slot][i],object_translation_world_m=state.object_translation[slot][i],face_strict_observed=authority,current_vertices_world=episode.hands[side]["vertices"][i],observed_surface_support_uncertainty_m=float(_cfg(cfg,"signed_surface_support_uncertainty_m",0.)))
                active=active_constraints_from_vertices(state.hand_vertices[side][i],row,scene,int(_cfg(cfg,"max_constraints_per_frame",96)),float(_cfg(cfg,"penetration_epsilon_m",1e-5)),episode.hands[side]["vertices"][i]);dense=dense_observed_surface_constraints_from_vertices(state.hand_vertices[side][i],row,scene,episode.hands[side]["vertices"][i]);signed["hands"][side]={"active_ids":active[0],"active_normals":active[1],"active_depths":active[2],"dense_ids":dense[0],"dense_normals":dense[1],"dense_depths":dense[2],"uncertainty_m":row.observed_surface_support_uncertainty_m}
        if bool(obj["metric_observation_eligible"][i]):
            observed=np.asarray(obj["observed_points_world"][i],float)
            if len(observed): result["object_observed"][slot]=(observed-state.object_translation[slot][i])@state.object_rotation[slot][i]
    pair_entries={}
    for target,row in _contact_rows(episode,frame,"left")+_contact_rows(episode,frame,"right"):
        side=row["factor_row"].get("hand_side");key=(side,target);slot=_slot_for_entity(episode,target)
        if slot is None or not bool(episode.objects[slot]["valid"][i]) or not bool(episode.hands[side]["valid"][i]): entry={**dict(row),"state":"inactive_contact_hand_or_target_absent","vertex_ids":np.zeros(0,np.int64),"targets_world":np.zeros((0,3)),"normals_world":np.zeros((0,3))}
        else:
            R=state.object_rotation[slot][i];t=state.object_translation[slot][i];targets=np.asarray(row["targets_object"],float);normals=np.asarray(row["normals_object"],float);entry={**dict(row),"state":"active_frozen_source_patch" if len(targets) else str(row.get("membership_state","inactive_no_source_patch_vertices")),"vertex_ids":np.asarray(row["vertex_ids"],np.int64),"targets_world":targets@R.T+t if len(targets) else targets,"normals_world":normals@R.T if len(normals) else normals,"object_slot":slot}
        result["contact_by_key"][key]=entry
        pair_entries.setdefault(side, []).append(entry)
    # Legacy two-key consumers are safe only for a single target. Multi-target
    # episodes must use contact_by_key so target selection cannot depend on map order.
    result["contact"]={side: entries[0] for side,entries in pair_entries.items() if len(entries)==1}
    return result
