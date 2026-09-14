"""Extracted formal interval mechanisms; no historical script imports."""
from __future__ import annotations

def fit_formal_interval(args) -> None:
    import json
    import hashlib
    from pathlib import Path
    from typing import Any
    import numpy as np
    import torch
    from utils.data import load_mesh
    from gates.object.trajectory import pose_graph_readiness
    from gates.object.geometry import resolve_physical_surface_contract
    from gates.output.acceptance import reject_rejected_annotation_path
    from gates.object.geometry import completion_report_completed_mesh
    from optimization.hand.mano import load_models
    from gates.hand.observation import validate_interval_camera_mano_contract
    from utils.data import load_json
    from utils.data import write_json
    from gates.object.geometry import load_signed_face_authority
    reject_rejected_annotation_path(args.annotations)
    pose_report = load_json(args.pose_report)
    pose_readiness = pose_graph_readiness(pose_report)
    surface_contract = resolve_physical_surface_contract(args.completed_mesh, args.physical_surface_mesh, args.completion_report)
    args.resolved_physical_surface_mesh = Path(surface_contract['physical_surface_mesh'])
    args.physical_surface_semantics = str(surface_contract['physical_surface_semantics'])
    args.completion_geometry_readiness = surface_contract['geometry_readiness']
    if pose_readiness['explicitly_unready'] and (not args.include_unready_object_pose_for_diagnostic_optimization):
        return quarantined_source_state(args, pose_readiness)
    args.camera_mano_contract_validation = validate_interval_camera_mano_contract(args) if bool(getattr(args, 'require_active_full_K_mano_contract', False)) else {'status': 'not_requested_generic_interval_compatibility', 'validated_before_mano_model_loading': False}
    physical_surface_probe = load_mesh(args.resolved_physical_surface_mesh)
    args.physical_surface_watertight = bool(physical_surface_probe.is_watertight)
    signed_face_authority = load_signed_face_authority(surface_contract.get('completion_report_payload') or {}, source_report_path=surface_contract.get('completion_report_path'), mesh_path=args.resolved_physical_surface_mesh, mesh_face_count=len(physical_surface_probe.faces))
    args.signed_face_authority_mask = np.asarray(signed_face_authority.pop('mask'), dtype=bool)
    signed_face_authority.pop('provenance_code', None)
    args.signed_face_authority_report = signed_face_authority
    signed_geometry_declared_ready = surface_contract['signed_geometry_ready'] is True
    signed_geometry_diagnostic_override_used = bool(args.physical_surface_watertight and (not signed_geometry_declared_ready) and args.allow_unready_signed_geometry_for_diagnostic_optimization)
    args.signed_object_surface_factor_active = bool(args.physical_surface_watertight and signed_face_authority.get('usable') is True and (signed_geometry_declared_ready or signed_geometry_diagnostic_override_used))
    args.signed_object_surface_factor_state = 'active_explicit_signed_geometry_ready' if args.signed_object_surface_factor_active and signed_geometry_declared_ready else 'active_diagnostic_override_unready_signed_geometry' if signed_geometry_diagnostic_override_used else 'inactive_signed_geometry_not_ready_nonwatertight_or_missing_local_face_authority'
    visible_surface_factor_supplied = args.visible_surface_track_factor_report is not None or bool(args.factor_report)
    if (bool(args.visible_object_mask_gate) or bool(args.visible_surface_depth_order_term)) and args.visible_object_mask_report is None and (not visible_surface_factor_supplied):
        raise ValueError('visible object mask terms require --visible-object-mask-report or a visible_surface_track factor report')
    device = torch.device(args.device)
    models = load_models(args, device)
    intervals: list[dict[str, Any]] = []
    per_frame_states: list[dict[str, Any]] = []
    full_archive_rows: list[dict[str, np.ndarray]] = []
    build_meta: dict[str, Any] = {}
    for side in args.sides:
        (rows, meta, scene) = build_rows(args, side)
        build_meta[side] = meta
        if not rows:
            intervals.append({'hand_side': side, 'start_frame': int(args.start_frame), 'end_frame': int(args.end_frame), 'frame_count': 0, 'solver': 'joint_root_translation_root_orientation_and_articulation', 'state': 'no_rows'})
            continue
        (interval, states, archive_rows) = optimize_rows(rows, models[side], args, device, scene)
        full_archive_rows.append(archive_rows)
        interval['interval_id'] = f'{side}_{rows[0].frame_idx:04d}_{rows[-1].frame_idx:04d}_joint_mano'
        for st in states:
            st['interval_id'] = interval['interval_id']
        intervals.append(interval)
        per_frame_states.extend(states)
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_archive_path = out_dir / 'v18_joint_mano_interval_full_vertices_world.npz'
    if full_archive_rows:
        frame_idx = np.concatenate([row['frame_idx'] for row in full_archive_rows], axis=0)
        hand_side = np.concatenate([row['hand_side'] for row in full_archive_rows], axis=0)
        vertices_world_m = np.concatenate([row['vertices_world_m'] for row in full_archive_rows], axis=0)
        joints_world_m = np.concatenate([row['joints_world_m'] for row in full_archive_rows], axis=0)
    else:
        frame_idx = np.zeros((0,), dtype=np.int32)
        hand_side = np.asarray([], dtype='<U5')
        vertices_world_m = np.zeros((0, 778, 3), dtype=np.float32)
        joints_world_m = np.zeros((0, 21, 3), dtype=np.float32)
    np.savez_compressed(full_archive_path, frame_idx=frame_idx, hand_side=hand_side, vertices_world_m=vertices_world_m, joints_world_m=joints_world_m)
    full_archive_sha256 = hashlib.sha256(full_archive_path.read_bytes()).hexdigest()
    report = {'method': 'solve_v18_joint_mano_interval_trajectory', 'status': 'completed_diagnostic_optimization_with_unready_object_pose_override' if pose_readiness['explicitly_unready'] else 'completed_diagnostic_optimization_with_unready_signed_geometry_override' if signed_geometry_diagnostic_override_used else 'ok' if args.signed_object_surface_factor_active else 'ok_with_signed_object_surface_factor_inactive', 'annotation_ready': False, 'physical_state_quarantined': bool(pose_readiness['explicitly_unready'] or signed_geometry_diagnostic_override_used), 'object_pose_input_ready_for_optimization': bool(pose_readiness['explicitly_ready']), 'object_pose_input_legacy_compatibility': bool(pose_readiness['legacy_readiness_fields_missing']), 'case': str(args.case), 'object_id': str(args.object_id), 'claim_scope': 'Diagnostic override ran continuous MANO/object-relative optimization despite an explicitly unready P15 trajectory; all output remains non-annotation-ready and must stay quarantined.' if pose_readiness['explicitly_unready'] else 'Historical diagnostic override activated signed object-surface terms without explicit signed-geometry readiness; all output remains quarantined and cannot be published as physical state.' if signed_geometry_diagnostic_override_used else 'Continuous interval MANO trajectory candidate uses collision-eligible unsigned surface/contact/depth-order evidence, while signed object-surface barriers are explicitly inactive because signed geometry is not ready.' if not args.signed_object_surface_factor_active else 'Continuous interval MANO trajectory correction candidate: root translation, root orientation, and finger articulation optimized jointly against visible/depth compatibility and explicitly ready signed object-surface geometry.', 'inputs': {'annotations': str(args.annotations), 'pose_report': str(args.pose_report), 'completed_mesh': str(args.completed_mesh), 'pose_hypothesis_mesh': str(args.completed_mesh), 'physical_surface_mesh': str(args.resolved_physical_surface_mesh), 'physical_surface_semantics': args.physical_surface_semantics, 'completion_report': None if args.completion_report is None else str(args.completion_report), 'completion_report_pose_hypothesis_mesh': None if args.completion_report is None else str(completion_report_completed_mesh(args.completion_report)), 'completion_geometry_readiness': args.completion_geometry_readiness, 'depth_npz': [str(p) for p in list(args.depth_npz)], 'visible_object_mask_report': None if args.visible_object_mask_report is None else str(args.visible_object_mask_report), 'visible_ownership_factor_report': None if args.visible_ownership_factor_report is None else str(args.visible_ownership_factor_report), 'surface_eligibility_factor_report': None if args.surface_eligibility_factor_report is None else str(args.surface_eligibility_factor_report), 'visible_surface_track_factor_report': None if args.visible_surface_track_factor_report is None else str(args.visible_surface_track_factor_report), 'factor_report': None if args.factor_report is None else [str(p) for p in args.factor_report]}, 'parameters': {k: [str(x) for x in v] if k == 'factor_report' and v is not None else str(v) if isinstance(v, Path) else v for (k, v) in vars(args).items() if k not in {'depth_npz', 'signed_face_authority_mask', 'signed_face_authority_report'}}, 'object_pose_readiness': pose_readiness, 'camera_mano_contract_validation': args.camera_mano_contract_validation, 'physical_surface_contract': {'pose_hypothesis_mesh': str(args.completed_mesh), 'physical_surface_mesh': str(args.resolved_physical_surface_mesh), 'physical_surface_semantics': args.physical_surface_semantics, 'physical_surface_watertight': args.physical_surface_watertight, 'geometry_readiness': args.completion_geometry_readiness, 'signed_geometry_declared_ready': signed_geometry_declared_ready, 'signed_object_surface_factor_active': args.signed_object_surface_factor_active, 'signed_object_surface_factor_state': args.signed_object_surface_factor_state, 'signed_face_authority': args.signed_face_authority_report, 'unready_signed_geometry_diagnostic_override_requested': bool(args.allow_unready_signed_geometry_for_diagnostic_optimization), 'unready_signed_geometry_diagnostic_override_used': signed_geometry_diagnostic_override_used}, 'unready_object_pose_diagnostic_optimization_override': bool(pose_readiness['explicitly_unready'] and args.include_unready_object_pose_for_diagnostic_optimization), 'build_meta': build_meta, 'summary': {'interval_count': int(len(intervals)), 'per_frame_state_count': int(len(per_frame_states)), 'frame_span': [int(args.start_frame), int(args.end_frame)], 'sides': list(args.sides)}, 'full_mano_vertices_world_archive': str(full_archive_path), 'full_mano_vertices_world_archive_sha256': full_archive_sha256, 'full_mano_vertices_world_archive_rows': int(len(frame_idx)), 'intervals': intervals, 'per_frame_states': per_frame_states, 'scientific_test': 'If this sequence still looks incoherent after rendering, the remaining failure is not that the optimizer was missing root/articulation coupling; it is a conflict among hand observation, observed object geometry, camera/depth alignment, and interval observability.'}
    out = out_dir / 'v18_joint_mano_interval_trajectory_state.json'
    write_json(out, report)
    print(json.dumps({'output': str(out), 'summary': report['summary'], 'intervals': intervals}, indent=2)[:6000])
    return report

def optimize_rows(rows: list[FrameHandRow], model: Any, args: argparse.Namespace, device: torch.device, scene: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    import math
    import argparse
    from typing import Any
    import numpy as np
    import torch
    from collections import Counter
    from utils.geometry import frame_camera_pose
    from optimization.common.correspondences import dense_observed_surface_constraints_from_vertices
    from gates.hand.observation import active_frame_intrinsics
    from utils.data import numeric_summary
    from optimization.common.correspondences import merge_constraints
    from gates.hand.observation import translation_support_gate_applies
    from optimization.hand.mano import rotvec_to_matrix
    from observations.interaction.evidence import project_world
    from observations.interaction.evidence import sample_ids
    from utils.geometry import world_to_camera
    from gates.interaction.contact import contact_patch_anchor_coherence
    from utils.data import FrameHandRow
    from observations.interaction.evidence import full_signed_topology_measure
    from observations.interaction.evidence import full_observed_surface_measure
    from observations.interaction.evidence import OPENPOSE_FINGER_GROUPS
    from optimization.common.correspondences import active_constraints_from_vertices
    from observations.interaction.evidence import inactive_signed_surface_measure
    from factors.hand.priors import formal_translation, formal_root_orientation, formal_articulation
    from factors.interaction.contact import contact_state_prior, contact_geometry_likelihood, contact_state_continuity, normal_deadband, normal_scaled_attraction
    from factors.motion.temporal import parameter_velocity, parameter_acceleration
    if not rows:
        return ({'status': 'no_rows'}, [], {'frame_idx': np.zeros((0,), dtype=np.int32), 'hand_side': np.asarray([], dtype='<U5'), 'vertices_world_m': np.zeros((0, 778, 3), dtype=np.float32), 'joints_world_m': np.zeros((0, 21, 3), dtype=np.float32)})
    b = len(rows)
    root = torch.tensor(np.stack([r.root_orient_axis_angle for r in rows]).reshape(b, 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.stack([r.hand_pose_axis_angle for r in rows]).reshape(b, 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.stack([r.betas for r in rows]), dtype=torch.float32, device=device)
    trans = torch.tensor(np.stack([r.trans_world_m for r in rows]), dtype=torch.float32, device=device)
    base_root_mat = rotvec_to_matrix(root)
    base_pose_mat = rotvec_to_matrix(pose)
    with torch.no_grad():
        base_out = model(global_orient=base_root_mat, hand_pose=base_pose_mat, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_base_vertices = base_out.vertices.detach().cpu().numpy().astype(float)
        raw_base_joints = base_out.joints.detach().cpu().numpy().astype(float)
    replay_vertex_err = [np.linalg.norm(raw_base_vertices[i] - rows[i].raw_vertices_world, axis=1) for i in range(b)]
    replay_joint_err = [np.linalg.norm(raw_base_joints[i] - rows[i].raw_joints_world, axis=1) for i in range(b)]
    replay_ok = max((float(np.median(e)) for e in replay_vertex_err)) <= 1e-05 and max((float(np.median(e)) for e in replay_joint_err)) <= 1e-05
    root_delta = torch.zeros((b, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    pose_delta = torch.zeros((b, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    hand_ray_shift_prior_t = torch.tensor(np.stack([r.hand_ray_shift_prior_world_m for r in rows]), dtype=torch.float32, device=device)
    hand_ray_shift_prior_weight_t = torch.tensor(np.asarray([float(r.hand_ray_shift_prior_weight) for r in rows], dtype=float), dtype=torch.float32, device=device)
    trans_init = hand_ray_shift_prior_t.detach().clone() if bool(args.initialize_hand_ray_shift) else torch.zeros((b, 3), dtype=torch.float32, device=device)
    trans_delta = trans_init.clone().detach().requires_grad_(True)
    translation_support_count_np = np.asarray([len(r.visible_surface_translation_support_vertex_indices) for r in rows], dtype=int)
    if bool(args.freeze_translation_without_visible_surface_support):
        translation_allowed_np = np.asarray([not translation_support_gate_applies(enabled=True, support_count=int(count), minimum_supported_vertices=int(args.translation_gate_min_visible_surface_depth_vertices)) for count in translation_support_count_np.tolist()], dtype=bool)
    else:
        translation_allowed_np = np.ones((b,), dtype=bool)
    translation_allowed_t = torch.tensor(translation_allowed_np.astype(np.float32), dtype=torch.float32, device=device).reshape(b, 1)
    translation_allowed_bool_t = torch.tensor(translation_allowed_np, dtype=torch.bool, device=device)
    object_trans_delta = torch.zeros((b, 3), dtype=torch.float32, device=device, requires_grad=bool(args.optimize_object_translation))
    contact_prior_np = np.asarray([float(np.clip(r.contact_patch_prior_probability, 0.0, 1.0)) for r in rows], dtype=float)
    contact_geometry_target_np = []
    for r in rows:
        if r.contact_patch_factor_state == 'active_contact_patch' and len(r.contact_patch_initial_distance_m):
            deadband = max(float(r.contact_patch_target_margin_m + r.contact_patch_support_uncertainty_m), 1e-06)
            distance = float(np.median(np.asarray(r.contact_patch_initial_distance_m, dtype=float)))
            target = float(np.exp(-0.5 * (distance / deadband) ** 2))
        else:
            target = float(np.clip(r.contact_patch_prior_probability, 0.0, 1.0))
        contact_geometry_target_np.append(float(np.clip(target, 0.0001, 1.0 - 0.0001)))
    contact_geometry_target_np = np.asarray(contact_geometry_target_np, dtype=float)
    contact_prior_t = torch.tensor(contact_prior_np, dtype=torch.float32, device=device)
    contact_geometry_target_t = torch.tensor(contact_geometry_target_np, dtype=torch.float32, device=device)
    contact_logit_init = torch.logit(torch.tensor(np.clip(contact_prior_np, 0.0001, 1.0 - 0.0001), dtype=torch.float32, device=device))
    contact_logit = contact_logit_init.clone().detach().requires_grad_(bool(args.optimize_contact_state))
    hand_ray_shift_prior_active = (torch.linalg.norm(hand_ray_shift_prior_t, dim=1) > 1e-09) & (hand_ray_shift_prior_weight_t > 0.0)
    optim_params = [root_delta, pose_delta, trans_delta]
    if bool(args.optimize_object_translation):
        optim_params.append(object_trans_delta)
    if bool(args.optimize_contact_state):
        optim_params.append(contact_logit)
    optimizer = torch.optim.LBFGS(optim_params, lr=0.35, max_iter=int(args.max_optimizer_iterations), line_search_fn='strong_wolfe')
    current_vertices_t = [torch.tensor(r.current_vertices_world, dtype=torch.float32, device=device) for r in rows]
    current_joints_t = [torch.tensor(r.current_joints_world, dtype=torch.float32, device=device) for r in rows]
    raw_base_vertices_t = torch.tensor(raw_base_vertices, dtype=torch.float32, device=device)
    raw_base_joints_t = torch.tensor(raw_base_joints, dtype=torch.float32, device=device)
    sim_scale_np = np.asarray([r.similarity_scale for r in rows], dtype=float).reshape(b, 1, 1)
    sim_rot_np = np.stack([r.similarity_rotation_raw_to_current for r in rows]).astype(float)
    sim_trans_np = np.stack([r.similarity_translation_raw_to_current for r in rows]).astype(float)
    sim_scale_t = torch.tensor(sim_scale_np, dtype=torch.float32, device=device)
    sim_rot_t = torch.tensor(sim_rot_np, dtype=torch.float32, device=device)
    sim_trans_t = torch.tensor(sim_trans_np, dtype=torch.float32, device=device).reshape(b, 1, 3)
    zero_surface_mode = str(args.zero_surface_mode)
    if zero_surface_mode == 'similarity_mapped_raw':
        reference_vertices_np = sim_scale_np * np.matmul(raw_base_vertices, np.transpose(sim_rot_np, (0, 2, 1))) + sim_trans_np[:, None, :]
        reference_joints_np = sim_scale_np * np.matmul(raw_base_joints, np.transpose(sim_rot_np, (0, 2, 1))) + sim_trans_np[:, None, :]
    else:
        reference_vertices_np = np.stack([r.current_vertices_world for r in rows]).astype(float)
        reference_joints_np = np.stack([r.current_joints_world for r in rows]).astype(float)
    reference_vertices_t = [torch.tensor(reference_vertices_np[i], dtype=torch.float32, device=device) for i in range(b)]
    active_constraint_indices: list[np.ndarray] = []
    active_constraint_normals: list[np.ndarray] = []
    active_constraint_depths: list[np.ndarray] = []
    dense_constraint_indices: list[np.ndarray] = []
    dense_constraint_normals: list[np.ndarray] = []
    dense_constraint_depths: list[np.ndarray] = []
    reference_observed_measures: list[dict[str, Any]] = []
    reference_raw_observed_measures: list[dict[str, Any]] = []
    signed_surface_active = bool(getattr(args, 'signed_object_surface_factor_active', False))
    for (i, r) in enumerate(rows):
        if signed_surface_active:
            (c_idx, c_normals, c_depths) = active_constraints_from_vertices(reference_vertices_np[i], r, scene, int(args.max_constraints_per_frame), float(args.penetration_epsilon_m), reference_vertices_np[i])
        else:
            c_idx = np.zeros((0,), dtype=np.int64)
            c_normals = np.zeros((0, 3), dtype=float)
            c_depths = np.zeros((0,), dtype=float)
        active_constraint_indices.append(c_idx)
        active_constraint_normals.append(c_normals)
        active_constraint_depths.append(c_depths)
        if bool(args.dense_observed_surface_barrier) and signed_surface_active:
            (d_idx, d_normals, d_depths) = dense_observed_surface_constraints_from_vertices(reference_vertices_np[i], r, scene, reference_vertices_np[i])
        else:
            d_idx = np.zeros((0,), dtype=np.int64)
            d_normals = np.zeros((0, 3), dtype=float)
            d_depths = np.zeros((0,), dtype=float)
        dense_constraint_indices.append(d_idx)
        dense_constraint_normals.append(d_normals)
        dense_constraint_depths.append(d_depths)
        if signed_surface_active:
            reference_observed_measures.append(full_observed_surface_measure(reference_vertices_np[i], r, scene, float(args.penetration_epsilon_m)))
            reference_raw_observed_measures.append(full_observed_surface_measure(reference_vertices_np[i], r, scene, float(args.penetration_epsilon_m), face_strict_observed=r.face_strict_observed_raw))
        else:
            reference_observed_measures.append(inactive_signed_surface_measure(r.frame_idx, r.observed_surface_support_uncertainty_m))
            reference_raw_observed_measures.append(inactive_signed_surface_measure(r.frame_idx, r.observed_surface_support_uncertainty_m))
    pose_joint_finger_groups = infer_pose_joint_finger_groups(model, base_root_mat, base_pose_mat, betas, trans)
    joint_visibility_weights_np = np.stack([r.joint_visibility_weights for r in rows]).astype(float)
    pose_visibility_weights_np = np.ones((b, 15), dtype=float)
    for (pose_i, group_i) in enumerate(pose_joint_finger_groups.astype(int)):
        pose_visibility_weights_np[:, pose_i] = np.mean(joint_visibility_weights_np[:, OPENPOSE_FINGER_GROUPS[group_i]], axis=1)
    hand_observation_weight_multiplier_np = np.asarray([float(r.hand_observation_visibility_weight_multiplier) if r.hand_observation_visibility_factor_state == 'active_hand_observation_visibility' else 1.0 for r in rows], dtype=float)
    joint_visibility_weights_t = torch.tensor(joint_visibility_weights_np, dtype=torch.float32, device=device)
    pose_visibility_weights_t = torch.tensor(pose_visibility_weights_np, dtype=torch.float32, device=device)
    hand_observation_weight_multiplier_t = torch.tensor(hand_observation_weight_multiplier_np, dtype=torch.float32, device=device)
    intr_t: list[torch.Tensor | None] = []
    base_uv: list[torch.Tensor | None] = []
    r_c2w_t: list[torch.Tensor] = []
    t_c2w_t: list[torch.Tensor] = []
    base_depth: list[torch.Tensor] = []
    visible_surface_depth_order_indices_t: list[torch.Tensor] = []
    visible_surface_depth_order_depth_t: list[torch.Tensor] = []
    visible_surface_depth_order_initial_counts: list[int] = []
    contact_patch_indices_t: list[torch.Tensor] = []
    contact_patch_targets_t: list[torch.Tensor] = []
    contact_patch_normals_t: list[torch.Tensor] = []
    contact_patch_weights_t: list[torch.Tensor] = []
    contact_patch_margins_t: list[torch.Tensor] = []
    contact_patch_support_uncertainty_t: list[torch.Tensor] = []
    for row in rows:
        intr = active_frame_intrinsics(row.frame, row.side, require_full_k=bool(getattr(args, 'require_active_full_K_mano_contract', False)))
        intr_t.append(torch.tensor(intr, dtype=torch.float32, device=device))
        uv0 = project_world(row.current_joints_world, row.frame, row.side)
        base_uv.append(None if uv0 is None else torch.tensor(uv0, dtype=torch.float32, device=device))
        (r_c2w, t_c2w) = frame_camera_pose(row.frame)
        r_c2w_t.append(torch.tensor(r_c2w, dtype=torch.float32, device=device))
        t_c2w_t.append(torch.tensor(t_c2w, dtype=torch.float32, device=device))
        base_depth.append(torch.tensor(world_to_camera(row.current_joints_world, row.frame)[:, 2], dtype=torch.float32, device=device))
        visible_surface_depth_order_indices_t.append(torch.tensor(row.visible_surface_depth_order_vertex_indices, dtype=torch.long, device=device))
        visible_surface_depth_order_depth_t.append(torch.tensor(row.visible_surface_depth_order_depth_m, dtype=torch.float32, device=device))
        visible_surface_depth_order_initial_counts.append(int((row.visible_surface_depth_order_initial_measure or {}).get('finite_inside_count', 0)))
        contact_patch_indices_t.append(torch.tensor(row.contact_patch_vertex_indices, dtype=torch.long, device=device))
        contact_patch_targets_t.append(torch.tensor(row.contact_patch_target_world_m, dtype=torch.float32, device=device))
        contact_patch_normals_t.append(torch.tensor(row.contact_patch_normal_world, dtype=torch.float32, device=device))
        contact_patch_weights_t.append(torch.tensor(float(row.contact_patch_weight), dtype=torch.float32, device=device))
        contact_patch_margins_t.append(torch.tensor(float(row.contact_patch_target_margin_m), dtype=torch.float32, device=device))
        contact_patch_support_uncertainty_t.append(torch.tensor(float(row.contact_patch_support_uncertainty_m), dtype=torch.float32, device=device))
    contact_active_np = np.asarray([bool(r.contact_patch_factor_state == 'active_contact_patch' and len(r.contact_patch_vertex_indices) and (float(r.contact_patch_weight) > 0.0)) for r in rows], dtype=bool)
    contact_active_t = torch.tensor(contact_active_np, dtype=torch.bool, device=device)
    contact_weight_np = np.asarray([float(r.contact_patch_weight) for r in rows], dtype=float)
    contact_prior_strength_np = contact_weight_np * float(args.contact_state_prior_residual_scale_m) ** 2
    contact_prior_strength_t = torch.tensor(contact_prior_strength_np, dtype=torch.float32, device=device)
    contact_temporal_pairs: list[tuple[int, int]] = []
    by_side: dict[str, list[tuple[int, int]]] = {}
    for (i, r) in enumerate(rows):
        by_side.setdefault(str(r.side), []).append((int(r.frame_idx), i))
    for items in by_side.values():
        items = sorted(items)
        for ((f0, i0), (f1, i1)) in zip(items[:-1], items[1:]):
            if f1 == f0 + 1 and contact_active_np[i0] and contact_active_np[i1]:
                contact_temporal_pairs.append((i0, i1))
    contact_temporal_pairs_t = torch.tensor(contact_temporal_pairs, dtype=torch.long, device=device) if contact_temporal_pairs else torch.zeros((0, 2), dtype=torch.long, device=device)

    def effective_trans_delta() -> torch.Tensor:
        return trans_delta * translation_allowed_t

    def hypothesis() -> tuple[torch.Tensor, torch.Tensor]:
        new_root = rotvec_to_matrix(root_delta) @ base_root_mat
        new_pose = rotvec_to_matrix(pose_delta) @ base_pose_mat
        out = model(global_orient=new_root, hand_pose=new_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        eff_trans_delta = effective_trans_delta()
        if zero_surface_mode == 'similarity_mapped_raw':
            mapped_vertices = sim_scale_t * torch.matmul(out.vertices, sim_rot_t.transpose(1, 2)) + sim_trans_t
            mapped_joints = sim_scale_t * torch.matmul(out.joints, sim_rot_t.transpose(1, 2)) + sim_trans_t
            verts = mapped_vertices + eff_trans_delta[:, None, :]
            joints = mapped_joints + eff_trans_delta[:, None, :]
        else:
            raw_delta_vertices = out.vertices - raw_base_vertices_t
            raw_delta_joints = out.joints - raw_base_joints_t
            mapped_vertices = sim_scale_t * torch.matmul(raw_delta_vertices, sim_rot_t.transpose(1, 2))
            mapped_joints = sim_scale_t * torch.matmul(raw_delta_joints, sim_rot_t.transpose(1, 2))
            verts = torch.stack(current_vertices_t, dim=0) + mapped_vertices + eff_trans_delta[:, None, :]
            joints = torch.stack(current_joints_t, dim=0) + mapped_joints + eff_trans_delta[:, None, :]
        return (verts, joints)

    def project_torch(points_world: torch.Tensor, i: int) -> torch.Tensor | None:
        intr = intr_t[i]
        if intr is None:
            return None
        cam = torch.matmul(points_world - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
        z = cam[:, 2].clamp_min(1e-05)
        (fx, fy, cx, cy) = intr
        return torch.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], dim=-1)

    def temporal_terms(x: torch.Tensor, weight: float) -> torch.Tensor:
        return parameter_velocity(x, float(weight)) + parameter_acceleration(x, float(args.accel_weight))

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        (hyp_vertices, hyp_joints) = hypothesis()
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        contact_prob = torch.sigmoid(contact_logit) if bool(args.optimize_contact_state) else torch.ones((b,), dtype=torch.float32, device=device)
        if bool(args.optimize_contact_state) and torch.any(contact_active_t):
            loss = loss + contact_state_prior(contact_prob, contact_prior_t, contact_prior_strength_t, contact_active_t)
            if bool(args.contact_state_geometry_likelihood):
                loss = loss + contact_geometry_likelihood(contact_prob, contact_geometry_target_t, contact_prior_strength_t, contact_active_t)
            if contact_temporal_pairs_t.numel() > 0:
                loss = loss + contact_state_continuity(contact_prob, contact_prior_strength_t, contact_temporal_pairs_t, float(args.contact_state_temporal_strength))
        obs_mult = hand_observation_weight_multiplier_t
        eff_trans_delta = effective_trans_delta()
        loss = loss + formal_translation(eff_trans_delta, obs_mult, float(args.translation_prior_weight))
        loss = loss + formal_root_orientation(root_delta, obs_mult, float(args.root_prior_weight))
        loss = loss + formal_articulation(pose_delta, pose_visibility_weights_t, float(args.pose_prior_weight))
        loss = loss + temporal_terms(eff_trans_delta, float(args.smooth_weight))
        loss = loss + temporal_terms(root_delta, float(args.smooth_weight))
        loss = loss + temporal_terms(pose_delta, float(args.smooth_weight))
        if bool(args.optimize_object_translation):
            loss = loss + float(args.object_translation_prior_weight) * torch.mean(object_trans_delta * object_trans_delta)
            loss = loss + temporal_terms(object_trans_delta, float(args.object_smooth_weight))
        hand_ray_shift_prior_active_supported = hand_ray_shift_prior_active & translation_allowed_bool_t
        if torch.any(hand_ray_shift_prior_active_supported):
            diff = eff_trans_delta[hand_ray_shift_prior_active_supported] - hand_ray_shift_prior_t[hand_ray_shift_prior_active_supported]
            weights = hand_ray_shift_prior_weight_t[hand_ray_shift_prior_active_supported].reshape(-1, 1)
            denom = torch.clamp(torch.tensor(float(diff.numel()), dtype=torch.float32, device=device), min=1.0)
            loss = loss + torch.sum(weights * diff * diff) / denom
        trans_norm = torch.linalg.norm(eff_trans_delta, dim=1)
        object_trans_norm = torch.linalg.norm(object_trans_delta, dim=1)
        root_norm = torch.linalg.norm(root_delta.reshape(b, 3), dim=1)
        pose_norm = torch.linalg.norm(pose_delta, dim=2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(trans_norm - float(args.max_translation_m)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(object_trans_norm - float(args.max_object_translation_m)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(root_norm - float(args.max_root_delta_rad)) ** 2)
        loss = loss + float(args.bound_hinge_weight) * torch.mean(torch.relu(pose_norm - float(args.max_pose_delta_rad)) ** 2)
        for (i, row) in enumerate(rows):
            if len(active_constraint_indices[i]):
                ids = torch.tensor(active_constraint_indices[i], dtype=torch.long, device=device)
                normals = torch.tensor(active_constraint_normals[i], dtype=torch.float32, device=device)
                depths = torch.tensor(active_constraint_depths[i], dtype=torch.float32, device=device)
                moved = hyp_vertices[i, ids] - reference_vertices_t[i][ids] - object_trans_delta[i].reshape(1, 3)
                residual = torch.relu(depths - float(rows[i].observed_surface_support_uncertainty_m) - torch.sum(normals * moved, dim=1))
                loss = loss + float(args.observed_penetration_weight) * torch.mean(residual * residual)
            if len(dense_constraint_indices[i]):
                ids = torch.tensor(dense_constraint_indices[i], dtype=torch.long, device=device)
                normals = torch.tensor(dense_constraint_normals[i], dtype=torch.float32, device=device)
                depths = torch.tensor(dense_constraint_depths[i], dtype=torch.float32, device=device)
                moved = hyp_vertices[i, ids] - reference_vertices_t[i][ids] - object_trans_delta[i].reshape(1, 3)
                residual = torch.relu(depths - float(rows[i].observed_surface_support_uncertainty_m) - torch.sum(normals * moved, dim=1))
                active_count = torch.clamp(torch.sum((residual > 0.0).to(torch.float32)), min=1.0)
                loss = loss + float(args.dense_observed_penetration_weight) * torch.sum(residual * residual) / active_count
            uv = project_torch(hyp_joints[i], i)
            base_uv_i = base_uv[i]
            obs_w = joint_visibility_weights_t[i]
            obs_den = torch.clamp(torch.sum(obs_w), min=1.0)
            if uv is not None and base_uv_i is not None:
                shift = torch.linalg.norm(uv - base_uv_i, dim=1)
                visible_hinge = torch.relu(shift - float(args.visible_shift_limit_px)) ** 2
                loss = loss + float(args.visible_hinge_weight) * torch.sum(obs_w * visible_hinge) / obs_den
            cam = torch.matmul(hyp_joints[i] - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
            depth_shift = torch.abs(cam[:, 2] - base_depth[i])
            depth_hinge = torch.relu(depth_shift - float(args.depth_shift_limit_m)) ** 2
            loss = loss + float(args.depth_hinge_weight) * torch.sum(obs_w * depth_hinge) / obs_den
            visible_depth_order_enabled = bool(args.visible_surface_depth_order_term) or rows[i].visible_surface_track_factor_state == 'active_visible_surface'
            if visible_depth_order_enabled and len(visible_surface_depth_order_indices_t[i]):
                ids = visible_surface_depth_order_indices_t[i]
                surface_depth = visible_surface_depth_order_depth_t[i]
                cam_v = torch.matmul(hyp_vertices[i, ids] - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
                residual = torch.relu(surface_depth - float(args.visible_surface_depth_order_margin_m) - cam_v[:, 2])
                active_count = torch.clamp(torch.sum((residual > 0.0).to(torch.float32)), min=1.0)
                loss = loss + float(args.visible_surface_depth_order_weight) * torch.sum(residual * residual) / active_count
            if rows[i].contact_patch_factor_state == 'active_contact_patch' and len(contact_patch_indices_t[i]) and (float(rows[i].contact_patch_weight) > 0.0):
                ids = contact_patch_indices_t[i]
                targets = contact_patch_targets_t[i] + object_trans_delta[i].reshape(1, 3)
                normals = contact_patch_normals_t[i]
                normal_gap = torch.sum((hyp_vertices[i, ids] - targets) * normals, dim=1)
                deadband = contact_patch_margins_t[i] + contact_patch_support_uncertainty_t[i]
                if str(args.contact_patch_residual_mode) == 'support_scaled_attraction':
                    loss = loss + normal_scaled_attraction(normal_gap, contact_patch_margins_t[i], contact_patch_support_uncertainty_t[i], contact_patch_weights_t[i], contact_prob[i])
                else:
                    loss = loss + normal_deadband(normal_gap, deadband, contact_patch_weights_t[i], contact_prob[i])
        loss.backward()
        return loss
    active_set_added_counts: list[int] = []
    active_set_closed = False
    active_set_pass_count = 0
    if replay_ok:
        for active_iter in range(max(1, int(args.active_set_iterations))):
            active_set_pass_count = active_iter + 1
            if active_iter > 0:
                optimizer = torch.optim.LBFGS(optim_params, lr=0.25, max_iter=int(args.max_optimizer_iterations), line_search_fn='strong_wolfe')
            optimizer.step(closure)
            with torch.no_grad():
                (hyp_vertices_t, _hyp_joints_t) = hypothesis()
                hyp_vertices_np = hyp_vertices_t.detach().cpu().numpy().astype(float)
                object_trans_np_active = object_trans_delta.detach().cpu().numpy().astype(float)
            added_total = 0
            for (i, row) in enumerate(rows):
                if signed_surface_active:
                    (new_idx, new_normals, new_depths) = active_constraints_from_vertices(hyp_vertices_np[i], row, scene, int(args.max_constraints_per_frame), float(args.penetration_epsilon_m), reference_vertices_np[i], object_trans_np_active[i])
                else:
                    new_idx = np.zeros((0,), dtype=np.int64)
                    new_normals = np.zeros((0, 3), dtype=float)
                    new_depths = np.zeros((0,), dtype=float)
                before = len(active_constraint_indices[i])
                merged = merge_constraints(active_constraint_indices[i], active_constraint_normals[i], active_constraint_depths[i], new_idx, new_normals, new_depths, max(int(args.max_constraints_per_frame), before))
                (active_constraint_indices[i], active_constraint_normals[i], active_constraint_depths[i]) = merged
                added_total += max(0, len(active_constraint_indices[i]) - before)
                if bool(args.dense_observed_surface_barrier) and signed_surface_active:
                    (dense_constraint_indices[i], dense_constraint_normals[i], dense_constraint_depths[i]) = dense_observed_surface_constraints_from_vertices(hyp_vertices_np[i], row, scene, reference_vertices_np[i], object_trans_np_active[i])
            active_set_added_counts.append(int(added_total))
            if added_total == 0:
                active_set_closed = True
                break
    with torch.no_grad():
        (hyp_vertices_t, hyp_joints_t) = hypothesis()
        hyp_vertices = hyp_vertices_t.detach().cpu().numpy().astype(float)
        hyp_joints = hyp_joints_t.detach().cpu().numpy().astype(float)
        trans_np = effective_trans_delta().detach().cpu().numpy().astype(float)
        latent_trans_np = trans_delta.detach().cpu().numpy().astype(float)
        object_trans_np = object_trans_delta.detach().cpu().numpy().astype(float)
        contact_posterior_np = (torch.sigmoid(contact_logit) if bool(args.optimize_contact_state) else contact_prior_t).detach().cpu().numpy().astype(float)
        root_np = root_delta.detach().cpu().numpy().reshape(b, 3).astype(float)
        pose_np = pose_delta.detach().cpu().numpy().astype(float)
    render_ids = sample_ids(778, int(args.sample_vertex_count_for_render))
    states: list[dict[str, Any]] = []
    initial_obs_max: list[float] = []
    final_linear_residual_max: list[float] = []
    final_full_observed_max: list[float] = []
    final_raw_observed_max: list[float] = []
    visible_max: list[float] = []
    depth_max: list[float] = []
    trans_max: list[float] = []
    object_trans_max: list[float] = []
    root_max: list[float] = []
    pose_max: list[float] = []
    visible_surface_depth_order_selected_count: list[float] = []
    visible_surface_depth_order_initial_in_front_count: list[float] = []
    visible_surface_depth_order_final_in_front_count: list[float] = []
    visible_surface_depth_order_final_delta_min: list[float] = []
    contact_patch_final_abs_normal_gap: list[float] = []
    output_translation_gate_applied_count = 0
    output_translation_gate_shift_norm: list[float] = []
    output_translation_gate_support_count: list[float] = []
    raw_optimizer_full_observed_max: list[float] = []
    raw_optimizer_visible_max: list[float] = []
    raw_optimizer_depth_max: list[float] = []
    full_state_vertices_world: list[np.ndarray] = []
    full_state_joints_world: list[np.ndarray] = []
    corrected_frames = 0
    for (i, row) in enumerate(rows):
        if len(active_constraint_indices[i]):
            moved = hyp_vertices[i, active_constraint_indices[i]] - reference_vertices_np[i, active_constraint_indices[i]] - object_trans_np[i][None, :]
            residual = np.maximum(0.0, active_constraint_depths[i] - float(row.observed_surface_support_uncertainty_m) - np.sum(active_constraint_normals[i] * moved, axis=1))
        else:
            residual = np.zeros((0,), dtype=float)
        init_measure = reference_observed_measures[i]
        init_raw_measure = reference_raw_observed_measures[i]
        init_max = float((init_measure.get('observed_supported_penetration_m') or {}).get('max') or 0.0)
        final_max = float(np.max(residual)) if residual.size else 0.0
        if signed_surface_active:
            full_post = full_observed_surface_measure(hyp_vertices[i], row, scene, float(args.penetration_epsilon_m), object_trans_np[i])
            raw_optimizer_topology_post = full_signed_topology_measure(hyp_vertices[i], row, scene, float(args.penetration_epsilon_m), object_trans_np[i])
            full_raw_post = full_observed_surface_measure(hyp_vertices[i], row, scene, float(args.penetration_epsilon_m), object_trans_np[i], face_strict_observed=row.face_strict_observed_raw)
        else:
            full_post = inactive_signed_surface_measure(row.frame_idx, row.observed_surface_support_uncertainty_m)
            raw_optimizer_topology_post = {'state': 'inactive_signed_geometry_not_ready', 'topology_penetrating_vertex_count': 0, 'authorized_penetrating_vertex_count': 0, 'unauthorized_penetrating_vertex_count': 0, 'topology_penetration_m': numeric_summary(np.asarray([], dtype=float)), 'authorized_penetration_m': numeric_summary(np.asarray([], dtype=float)), 'unauthorized_penetration_m': numeric_summary(np.asarray([], dtype=float))}
            full_raw_post = inactive_signed_surface_measure(row.frame_idx, row.observed_surface_support_uncertainty_m)
        full_post_max = float((full_post.get('observed_supported_penetration_m') or {}).get('max') or 0.0)
        full_raw_post_max = float((full_raw_post.get('observed_supported_penetration_m') or {}).get('max') or 0.0)
        raw_optimizer_full_post = full_post
        raw_optimizer_full_raw_post = full_raw_post
        raw_optimizer_active_residual = residual.copy()
        uv0 = project_world(row.current_joints_world, row.frame, row.side)
        uv1 = project_world(hyp_joints[i], row.frame, row.side)
        if uv0 is not None and uv1 is not None:
            shift = np.linalg.norm(uv1 - uv0, axis=1)
            shift_max = float(np.max(shift))
            shift_med = float(np.median(shift))
        else:
            shift = np.zeros((0,), dtype=float)
            shift_max = float('nan')
            shift_med = float('nan')
        cam0 = world_to_camera(row.current_joints_world, row.frame)
        cam1 = world_to_camera(hyp_joints[i], row.frame)
        dshift = np.abs(cam1[:, 2] - cam0[:, 2])
        raw_optimizer_shift = shift.copy()
        raw_optimizer_dshift = dshift.copy()
        surface_ids = row.visible_surface_depth_order_vertex_indices.astype(int)
        translation_support_ids = row.visible_surface_translation_support_vertex_indices.astype(int)
        if surface_ids.size:
            cam_v_final = world_to_camera(hyp_vertices[i, surface_ids], row.frame)[:, 2]
            surface_final_delta = cam_v_final.astype(float) - row.visible_surface_depth_order_depth_m.astype(float)
            surface_initial_delta = row.visible_surface_depth_order_initial_delta_m.astype(float)
            surface_initial_in_front = int(np.count_nonzero(surface_initial_delta < -float(args.visible_surface_depth_order_margin_m)))
            surface_final_in_front = int(np.count_nonzero(surface_final_delta < -float(args.visible_surface_depth_order_margin_m)))
            surface_final_summary = numeric_summary(surface_final_delta)
            additional_camera_z_to_clear = float(max(0.0, -float(args.visible_surface_depth_order_margin_m) - float(np.min(surface_final_delta))))
        else:
            surface_initial_delta = np.zeros((0,), dtype=float)
            surface_final_delta = np.zeros((0,), dtype=float)
            surface_initial_in_front = 0
            surface_final_in_front = 0
            surface_final_summary = numeric_summary(np.zeros((0,), dtype=float))
            additional_camera_z_to_clear = 0.0
        (r_c2w_frame, _t_c2w_frame) = frame_camera_pose(row.frame)
        camera_z_axis_world = np.asarray(r_c2w_frame[:, 2], dtype=float)
        optimized_camera_z_shift_m = float(np.dot(trans_np[i], camera_z_axis_world))
        lateral_translation = trans_np[i] - optimized_camera_z_shift_m * camera_z_axis_world
        lateral_norm = float(np.linalg.norm(lateral_translation))
        max_translation = max(0.0, float(args.max_translation_m))
        max_camera_z_inside_translation_bound = math.sqrt(max(0.0, max_translation * max_translation - lateral_norm * lateral_norm))
        translation_bound_remaining_camera_z_m = float(max_camera_z_inside_translation_bound - optimized_camera_z_shift_m)
        if len(row.contact_patch_vertex_indices):
            cp_targets = row.contact_patch_target_world_m + object_trans_np[i][None, :]
            cp_gap = np.sum((hyp_vertices[i, row.contact_patch_vertex_indices.astype(int)] - cp_targets) * row.contact_patch_normal_world, axis=1)
            cp_gap_summary = numeric_summary(cp_gap)
        else:
            cp_gap_summary = numeric_summary(np.zeros((0,), dtype=float))
        tnorm = float(np.linalg.norm(trans_np[i]))
        otnorm = float(np.linalg.norm(object_trans_np[i]))
        rnorm = float(np.linalg.norm(root_np[i]))
        pnorm = float(np.max(np.linalg.norm(pose_np[i], axis=1)))
        changed = tnorm > 0.0001 or rnorm > 0.0001 or pnorm > 0.0001
        if changed:
            corrected_frames += 1
        initial_obs_max.append(init_max)
        trans_max.append(tnorm)
        object_trans_max.append(otnorm)
        root_max.append(rnorm)
        pose_max.append(pnorm)
        state_joints_world = hyp_joints[i].astype(float).copy()
        state_vertices_world = hyp_vertices[i].astype(float).copy()
        state_translation_world = trans_np[i].astype(float).copy()
        optimizer_translation_support_gate = {'enabled': bool(args.freeze_translation_without_visible_surface_support), 'frozen': translation_support_gate_applies(enabled=bool(args.freeze_translation_without_visible_surface_support), support_count=int(translation_support_ids.size), minimum_supported_vertices=int(args.translation_gate_min_visible_surface_depth_vertices)), 'min_visible_surface_depth_vertices': int(args.translation_gate_min_visible_surface_depth_vertices), 'selected_visible_surface_depth_vertices': int(translation_support_ids.size), 'support_semantics': 'nearby_P09_accepted_first_surface_grounding_not_depth_order_residual', 'policy': 'global MANO translation optimized only for rows with visible-surface support above threshold' if bool(args.freeze_translation_without_visible_surface_support) else 'global MANO translation optimized normally'}
        output_translation_gate = {'enabled': bool(args.gate_translation_with_visible_surface_support), 'applied': False, 'reason': 'disabled' if not bool(args.gate_translation_with_visible_surface_support) else 'support_count_above_threshold', 'min_visible_surface_depth_vertices': int(args.translation_gate_min_visible_surface_depth_vertices), 'selected_visible_surface_depth_vertices': int(translation_support_ids.size), 'support_semantics': 'nearby_P09_accepted_first_surface_grounding_not_depth_order_residual', 'articulation_policy': 'raw optimizer output'}
        if translation_support_gate_applies(enabled=bool(args.gate_translation_with_visible_surface_support), support_count=int(translation_support_ids.size), minimum_supported_vertices=int(args.translation_gate_min_visible_surface_depth_vertices)):
            gate_shift = row.current_joints_world[0].astype(float) - state_joints_world[0].astype(float)
            state_joints_world = state_joints_world + gate_shift[None, :]
            state_vertices_world = state_vertices_world + gate_shift[None, :]
            state_translation_world = state_translation_world + gate_shift
            output_translation_gate_applied_count += 1
            output_translation_gate_shift_norm.append(float(np.linalg.norm(gate_shift)))
            output_translation_gate_support_count.append(float(translation_support_ids.size))
            output_translation_gate = {'enabled': True, 'applied': True, 'reason': 'visible_surface_support_at_or_below_threshold', 'min_visible_surface_depth_vertices': int(args.translation_gate_min_visible_surface_depth_vertices), 'selected_visible_surface_depth_vertices': int(translation_support_ids.size), 'support_semantics': 'nearby_P09_accepted_first_surface_grounding_not_depth_order_residual', 'baseline_wrist_world_m': row.current_joints_world[0].astype(float).tolist(), 'raw_optimizer_wrist_world_m': hyp_joints[i, 0].astype(float).tolist(), 'applied_world_shift_m': gate_shift.astype(float).tolist(), 'applied_world_shift_norm_m': float(np.linalg.norm(gate_shift)), 'articulation_policy': 'preserve optimized wrist-relative MANO articulation; preserve source HaWoR wrist/root translation'}
        if len(active_constraint_indices[i]):
            state_moved = state_vertices_world[active_constraint_indices[i]] - reference_vertices_np[i, active_constraint_indices[i]] - object_trans_np[i][None, :]
            residual = np.maximum(0.0, active_constraint_depths[i] - float(row.observed_surface_support_uncertainty_m) - np.sum(active_constraint_normals[i] * state_moved, axis=1))
        else:
            residual = np.zeros((0,), dtype=float)
        final_max = float(np.max(residual)) if residual.size else 0.0
        if signed_surface_active:
            full_post = full_observed_surface_measure(state_vertices_world, row, scene, float(args.penetration_epsilon_m), object_trans_np[i])
            full_topology_post = full_signed_topology_measure(state_vertices_world, row, scene, float(args.penetration_epsilon_m), object_trans_np[i])
            full_raw_post = full_observed_surface_measure(state_vertices_world, row, scene, float(args.penetration_epsilon_m), object_trans_np[i], face_strict_observed=row.face_strict_observed_raw)
        else:
            full_post = inactive_signed_surface_measure(row.frame_idx, row.observed_surface_support_uncertainty_m)
            full_topology_post = {'state': 'inactive_signed_geometry_not_ready', 'topology_penetrating_vertex_count': 0, 'authorized_penetrating_vertex_count': 0, 'unauthorized_penetrating_vertex_count': 0, 'topology_penetration_m': numeric_summary(np.asarray([], dtype=float)), 'authorized_penetration_m': numeric_summary(np.asarray([], dtype=float)), 'unauthorized_penetration_m': numeric_summary(np.asarray([], dtype=float))}
            full_raw_post = inactive_signed_surface_measure(row.frame_idx, row.observed_surface_support_uncertainty_m)
        full_post_max = float((full_post.get('observed_supported_penetration_m') or {}).get('max') or 0.0)
        full_raw_post_max = float((full_raw_post.get('observed_supported_penetration_m') or {}).get('max') or 0.0)
        uv1 = project_world(state_joints_world, row.frame, row.side)
        if uv0 is not None and uv1 is not None:
            shift = np.linalg.norm(uv1 - uv0, axis=1)
            shift_max = float(np.max(shift))
            shift_med = float(np.median(shift))
        else:
            shift = np.zeros((0,), dtype=float)
            shift_max = float('nan')
            shift_med = float('nan')
        cam1 = world_to_camera(state_joints_world, row.frame)
        dshift = np.abs(cam1[:, 2] - cam0[:, 2])
        if surface_ids.size:
            state_cam_v = world_to_camera(state_vertices_world[surface_ids], row.frame)[:, 2]
            surface_final_delta = state_cam_v.astype(float) - row.visible_surface_depth_order_depth_m.astype(float)
            surface_final_in_front = int(np.count_nonzero(surface_final_delta < -float(args.visible_surface_depth_order_margin_m)))
            surface_final_summary = numeric_summary(surface_final_delta)
            additional_camera_z_to_clear = float(max(0.0, -float(args.visible_surface_depth_order_margin_m) - float(np.min(surface_final_delta))))
        if len(row.contact_patch_vertex_indices):
            cp_targets = row.contact_patch_target_world_m + object_trans_np[i][None, :]
            cp_gap = np.sum((state_vertices_world[row.contact_patch_vertex_indices.astype(int)] - cp_targets) * row.contact_patch_normal_world, axis=1)
            cp_gap_summary = numeric_summary(cp_gap)
        final_linear_residual_max.append(final_max)
        final_full_observed_max.append(full_post_max)
        raw_optimizer_full_observed_max.append(float((raw_optimizer_full_post.get('observed_supported_penetration_m') or {}).get('max') or 0.0))
        if raw_optimizer_shift.size:
            raw_optimizer_visible_max.append(float(np.max(raw_optimizer_shift)))
        if raw_optimizer_dshift.size:
            raw_optimizer_depth_max.append(float(np.max(raw_optimizer_dshift)))
        final_raw_observed_max.append(full_raw_post_max)
        if np.isfinite(shift_max):
            visible_max.append(shift_max)
        depth_max.append(float(np.max(dshift)))
        if surface_ids.size:
            visible_surface_depth_order_selected_count.append(float(surface_ids.size))
            visible_surface_depth_order_initial_in_front_count.append(float(surface_initial_in_front))
            visible_surface_depth_order_final_in_front_count.append(float(surface_final_in_front))
            visible_surface_depth_order_final_delta_min.append(float(np.min(surface_final_delta)))
        if len(row.contact_patch_vertex_indices):
            contact_patch_final_abs_normal_gap.extend(np.abs(cp_gap).astype(float).tolist())
        state_camera_z_shift_m = float(np.dot(state_translation_world, camera_z_axis_world))
        state_lateral_translation = state_translation_world - state_camera_z_shift_m * camera_z_axis_world
        state_lateral_norm = float(np.linalg.norm(state_lateral_translation))
        state_max_camera_z_inside_translation_bound = math.sqrt(max(0.0, max_translation * max_translation - state_lateral_norm * state_lateral_norm))
        state_translation_bound_remaining_camera_z_m = float(state_max_camera_z_inside_translation_bound - state_camera_z_shift_m)
        full_state_vertices_world.append(state_vertices_world.astype(np.float32))
        full_state_joints_world.append(state_joints_world.astype(np.float32))
        states.append({'frame_idx': int(row.frame_idx), 'hand_side': row.side, 'temporal_mano_state': 'joint_continuous_mano_trajectory_correction', 'source_hawor_npz': str(row.source_hawor_npz), 'source_frame_index': int(row.source_frame_index), 'optimized_translation_world_m': state_translation_world.astype(float).tolist(), 'raw_optimizer_translation_world_m': trans_np[i].astype(float).tolist(), 'latent_optimizer_translation_world_m': latent_trans_np[i].astype(float).tolist(), 'optimizer_translation_support_gate': optimizer_translation_support_gate, 'output_translation_gate': output_translation_gate, 'raw_optimizer_candidate_audit': {'active_constraint_residual_m': numeric_summary(raw_optimizer_active_residual), 'full_observed_surface_penetration_m': raw_optimizer_full_post.get('observed_supported_penetration_m'), 'full_raw_observed_surface_penetration_m': raw_optimizer_full_raw_post.get('observed_supported_penetration_m'), 'full_signed_topology_audit': raw_optimizer_topology_post, 'visible_joint_shift_px': numeric_summary(raw_optimizer_shift), 'joint_camera_depth_shift_m': numeric_summary(raw_optimizer_dshift), 'published_candidate_was_translation_gated': bool(output_translation_gate.get('applied'))}, 'hand_ray_shift_prior_translation_world_m': row.hand_ray_shift_prior_world_m.astype(float).tolist(), 'hand_ray_shift_prior_source_m': row.hand_ray_shift_prior_source_m, 'hand_ray_shift_prior_weight': float(row.hand_ray_shift_prior_weight), 'optimized_object_translation_world_m': object_trans_np[i].astype(float).tolist(), 'joint_visibility_weights': row.joint_visibility_weights.astype(float).tolist(), 'joint_depth_residual_m': [None if not np.isfinite(x) else float(x) for x in row.joint_depth_residual_m], 'pose_visibility_weights': pose_visibility_weights_np[i].astype(float).tolist(), 'optimized_root_delta_axis_angle_rad': root_np[i].astype(float).tolist(), 'optimized_hand_pose_delta_axis_angle_rad': pose_np[i].reshape(-1).astype(float).tolist(), 'optimized_joints_world_m': state_joints_world.astype(float).tolist(), 'optimized_vertices_world_sample_m': state_vertices_world[render_ids].astype(float).tolist(), 'optimized_vertices_sample_ids': render_ids.astype(int).tolist(), 'signed_object_surface_factor_state': str(getattr(args, 'signed_object_surface_factor_state', 'active_explicit_signed_geometry_ready' if signed_surface_active else 'inactive_signed_geometry_not_ready')), 'physical_surface_mesh': str(getattr(args, 'resolved_physical_surface_mesh', args.completed_mesh)), 'initial_observed_surface_measurement_state': init_measure.get('state', 'active_signed_surface_measurement'), 'final_observed_surface_measurement_state': full_post.get('state', 'active_signed_surface_measurement'), 'initial_observed_surface_penetration_m': init_measure.get('observed_supported_penetration_m'), 'initial_raw_observed_surface_penetration_m': init_raw_measure.get('observed_supported_penetration_m'), 'current_bridge_observed_surface_penetration_m': row.observed_initial_measure.get('observed_supported_penetration_m'), 'hand_owned_quarantined_face_count': int(row.hand_owned_quarantined_face_count), 'surface_eligibility_npz_path': row.surface_eligibility_npz_path, 'surface_eligibility_mode': row.surface_eligibility_mode, 'observed_surface_support_uncertainty_m': float(row.observed_surface_support_uncertainty_m), 'surface_eligible_face_count': int(row.surface_eligible_face_count), 'surface_input_face_count': int(row.surface_input_face_count), 'surface_applied_face_delta': int(row.surface_applied_face_delta), 'signed_face_authority_required': bool(row.signed_face_authority_required), 'signed_face_authority_state': row.signed_face_authority_state, 'signed_face_authority_face_count': int(row.signed_face_authority_face_count), 'signed_face_authority_eligible_face_count': int(row.signed_face_authority_eligible_face_count), 'visible_ownership_non_object_mask_path': row.visible_ownership_non_object_mask_path, 'visible_ownership_object_owned_mask_path': row.visible_ownership_object_owned_mask_path, 'visible_ownership_constraint_eligible_mask_path': row.visible_ownership_constraint_eligible_mask_path, 'visible_ownership_non_object_owned_px': int(row.visible_ownership_non_object_owned_px), 'visible_ownership_object_owned_px': int(row.visible_ownership_object_owned_px), 'visible_ownership_constraint_eligible_px': int(row.visible_ownership_constraint_eligible_px), 'visible_ownership_quarantined_face_count': int(row.visible_ownership_quarantined_face_count), 'visible_mask_A_mask_from_source_coordinate_model': row.visible_mask_A_mask_from_source.astype(float).tolist(), 'visible_mask_image_plane_contract': row.visible_mask_image_plane_contract, 'visible_object_mask_path': row.visible_object_mask_path, 'visible_object_mask_face_count_raw': int(row.visible_object_mask_face_count_raw), 'visible_object_mask_face_count': int(row.visible_object_mask_face_count), 'visible_surface_track_factor_state': row.visible_surface_track_factor_state, 'visible_surface_track_mask_path': row.visible_surface_track_mask_path, 'visible_surface_track_npz_path': row.visible_surface_track_npz_path, 'visible_surface_track_valid_depth_pixels': int(row.visible_surface_track_valid_depth_pixels), 'visible_surface_track_quarantined_face_count': int(row.visible_surface_track_quarantined_face_count), 'visible_surface_depth_order_initial': row.visible_surface_depth_order_initial_measure, 'visible_surface_translation_support': row.visible_surface_translation_support_measure, 'visible_surface_translation_support_vertex_count': int(translation_support_ids.size), 'visible_surface_translation_support_vertex_ids': translation_support_ids.astype(int).tolist(), 'visible_surface_translation_support_depth_m': row.visible_surface_translation_support_depth_m.astype(float).tolist(), 'visible_surface_translation_support_initial_delta_hand_minus_surface_m': row.visible_surface_translation_support_initial_delta_m.astype(float).tolist(), 'visible_surface_depth_order_selected_vertex_count': int(surface_ids.size), 'visible_surface_depth_order_selected_vertex_ids': surface_ids.astype(int).tolist(), 'visible_surface_depth_order_selected_surface_depth_m': row.visible_surface_depth_order_depth_m.astype(float).tolist(), 'visible_surface_depth_order_selected_initial_delta_hand_minus_surface_m': surface_initial_delta.astype(float).tolist(), 'visible_surface_depth_order_selected_final_in_front_count': int(surface_final_in_front), 'visible_surface_depth_order_selected_initial_in_front_count': int(surface_initial_in_front), 'visible_surface_depth_order_selected_final_delta_hand_minus_surface_m': surface_final_summary, 'visible_surface_depth_order_selected_final_delta_values_m': surface_final_delta.astype(float).tolist(), 'visible_surface_depth_order_additional_camera_z_to_clear_selected_m': float(additional_camera_z_to_clear), 'optimized_translation_camera_z_m': float(state_camera_z_shift_m), 'optimized_translation_lateral_norm_m': float(state_lateral_norm), 'translation_bound_remaining_camera_z_m': float(state_translation_bound_remaining_camera_z_m), 'raw_optimizer_translation_camera_z_m': float(optimized_camera_z_shift_m), 'raw_optimizer_translation_lateral_norm_m': float(lateral_norm), 'raw_optimizer_translation_bound_remaining_camera_z_m': float(translation_bound_remaining_camera_z_m), 'hand_observation_visibility_factor_state': row.hand_observation_visibility_factor_state, 'hand_observation_visibility_candidate_px': int(row.hand_observation_visibility_candidate_px), 'hand_observation_visibility_weight_multiplier': float(row.hand_observation_visibility_weight_multiplier), 'contact_patch_factor_state': row.contact_patch_factor_state, 'contact_patch_vertex_count': int(len(row.contact_patch_vertex_indices)), 'contact_patch_vertex_ids': row.contact_patch_vertex_indices.astype(int).tolist(), 'contact_patch_target_world_m': row.contact_patch_target_world_m.astype(float).tolist(), 'contact_patch_normal_world': row.contact_patch_normal_world.astype(float).tolist(), 'contact_patch_initial_distance_values_m': row.contact_patch_initial_distance_m.astype(float).tolist(), 'contact_patch_initial_distance_m': numeric_summary(row.contact_patch_initial_distance_m), 'contact_patch_weight': float(row.contact_patch_weight), 'contact_patch_prior_probability': float(row.contact_patch_prior_probability), 'contact_patch_geometry_target_probability': float(contact_geometry_target_np[i]), 'contact_patch_posterior_probability': float(contact_posterior_np[i]), 'contact_patch_state_optimized': bool(args.optimize_contact_state), 'contact_patch_band_m': float(row.contact_patch_band_m), 'contact_patch_target_margin_m': float(row.contact_patch_target_margin_m), 'contact_patch_support_uncertainty_m': float(row.contact_patch_support_uncertainty_m), 'contact_patch_support_uncertainty_source': row.contact_patch_support_uncertainty_source, 'local_patch_support_state': row.local_patch_support_state, 'local_patch_support_consumed': bool(row.local_patch_support_consumed), 'local_patch_support_uncertainty_m': row.local_patch_support_uncertainty_m, 'global_object_support_uncertainty_m': row.global_object_support_uncertainty_m, 'local_patch_sample_count': int(row.local_patch_sample_count), 'local_patch_temporal_sample_count': int(row.local_patch_temporal_sample_count), 'contact_patch_deadband_m': float(row.contact_patch_target_margin_m + row.contact_patch_support_uncertainty_m), 'contact_patch_residual_mode': str(args.contact_patch_residual_mode), 'contact_anchor_state': row.contact_anchor_state, 'contact_anchor_residual_allowed': bool(row.contact_anchor_residual_allowed), 'contact_anchor_blockers': list(row.contact_anchor_blockers), 'contact_pose_anchor_key': row.contact_pose_anchor_key, 'contact_patch_final_normal_gap_m': cp_gap_summary, 'final_active_constraint_residual_after_solver_m': numeric_summary(residual), 'full_observed_surface_penetration_after_solver_m': full_post.get('observed_supported_penetration_m'), 'full_raw_observed_surface_penetration_after_solver_m': full_raw_post.get('observed_supported_penetration_m'), 'full_signed_topology_audit_after_solver': full_topology_post, 'full_unauthorized_surface_penetration_after_solver_m': full_topology_post.get('unauthorized_penetration_m'), 'full_unauthorized_surface_penetrating_vertex_count_after_solver': int(full_topology_post.get('unauthorized_penetrating_vertex_count', 0)), 'full_observed_supported_penetrating_vertex_count_after_solver': int(full_post.get('observed_supported_penetrating_vertex_count', 0)), 'visible_joint_shift_px': {'count': int(len(shift)), 'median': shift_med, 'max': shift_max}, 'joint_camera_depth_shift_m': {'count': int(len(dshift)), 'median': float(np.median(dshift)), 'max': float(np.max(dshift))}, 'delta_norms': {'translation_m': tnorm, 'object_translation_m': otnorm, 'root_rad': rnorm, 'max_pose_joint_rad': pnorm}})
    interval = {'hand_side': rows[0].side, 'start_frame': int(rows[0].frame_idx), 'end_frame': int(rows[-1].frame_idx), 'frame_count': int(len(rows)), 'solver': 'joint_root_translation_root_orientation_and_articulation', 'zero_surface_mode': zero_surface_mode, 'optimizer_ran': bool(replay_ok), 'replay_ok': bool(replay_ok), 'raw_replay_vertex_error_median_m': numeric_summary(np.asarray([float(np.median(e)) for e in replay_vertex_err], dtype=float)), 'raw_replay_joint_error_median_m': numeric_summary(np.asarray([float(np.median(e)) for e in replay_joint_err], dtype=float)), 'corrected_frame_count': int(corrected_frames), 'initial_observed_surface_penetration_max_m': numeric_summary(np.asarray(initial_obs_max, dtype=float)), 'final_active_constraint_residual_after_solver_max_m': numeric_summary(np.asarray(final_linear_residual_max, dtype=float)), 'full_observed_surface_penetration_after_solver_max_m': numeric_summary(np.asarray(final_full_observed_max, dtype=float)), 'full_raw_observed_surface_penetration_after_solver_max_m': numeric_summary(np.asarray(final_raw_observed_max, dtype=float)), 'hand_owned_quarantined_face_count': numeric_summary(np.asarray([r.hand_owned_quarantined_face_count for r in rows], dtype=float)), 'visible_object_mask_face_count_raw': numeric_summary(np.asarray([r.visible_object_mask_face_count_raw for r in rows], dtype=float)), 'visible_object_mask_face_count': numeric_summary(np.asarray([r.visible_object_mask_face_count for r in rows], dtype=float)), 'visible_surface_track_active_row_count': int(sum((r.visible_surface_track_factor_state == 'active_visible_surface' for r in rows))), 'visible_surface_track_quarantined_face_count': numeric_summary(np.asarray([r.visible_surface_track_quarantined_face_count for r in rows], dtype=float)), 'visible_surface_depth_order_selected_vertex_count': numeric_summary(np.asarray(visible_surface_depth_order_selected_count, dtype=float)), 'translation_optimization_support_gate_enabled': bool(args.freeze_translation_without_visible_surface_support), 'translation_optimization_support_gate_min_visible_surface_depth_vertices': int(args.translation_gate_min_visible_surface_depth_vertices), 'translation_optimization_support_gate_frozen_count': int(np.count_nonzero(~translation_allowed_np)) if bool(args.freeze_translation_without_visible_surface_support) else 0, 'visible_surface_depth_order_selected_initial_in_front_count': numeric_summary(np.asarray(visible_surface_depth_order_initial_in_front_count, dtype=float)), 'visible_surface_depth_order_selected_final_in_front_count': numeric_summary(np.asarray(visible_surface_depth_order_final_in_front_count, dtype=float)), 'visible_surface_depth_order_selected_final_delta_min_m': numeric_summary(np.asarray(visible_surface_depth_order_final_delta_min, dtype=float)), 'visible_joint_shift_max_px': numeric_summary(np.asarray(visible_max, dtype=float)), 'joint_camera_depth_shift_max_m': numeric_summary(np.asarray(depth_max, dtype=float)), 'translation_delta_norm_m': numeric_summary(np.asarray(trans_max, dtype=float)), 'output_translation_gate_enabled': bool(args.gate_translation_with_visible_surface_support), 'output_translation_gate_min_visible_surface_depth_vertices': int(args.translation_gate_min_visible_surface_depth_vertices), 'output_translation_gate_applied_count': int(output_translation_gate_applied_count), 'output_translation_gate_shift_norm_m': numeric_summary(np.asarray(output_translation_gate_shift_norm, dtype=float)), 'output_translation_gate_selected_support_count': numeric_summary(np.asarray(output_translation_gate_support_count, dtype=float)), 'raw_optimizer_candidate_audit': {'full_observed_surface_penetration_max_m': numeric_summary(np.asarray(raw_optimizer_full_observed_max, dtype=float)), 'visible_joint_shift_max_px': numeric_summary(np.asarray(raw_optimizer_visible_max, dtype=float)), 'joint_camera_depth_shift_max_m': numeric_summary(np.asarray(raw_optimizer_depth_max, dtype=float)), 'published_translation_gate_applied_count': int(output_translation_gate_applied_count)}, 'object_translation_delta_norm_m': numeric_summary(np.asarray(object_trans_max, dtype=float)), 'root_delta_norm_rad': numeric_summary(np.asarray(root_max, dtype=float)), 'pose_delta_max_joint_norm_rad': numeric_summary(np.asarray(pose_max, dtype=float)), 'active_set_added_constraint_counts': active_set_added_counts, 'active_set_pass_count': int(active_set_pass_count), 'active_set_closed': bool(active_set_closed), 'active_constraint_count_final': numeric_summary(np.asarray([len(x) for x in active_constraint_indices], dtype=float)), 'visibility_weighted_hand_observation_enabled': bool(args.visibility_weighted_hand_observation), 'hand_observation_visibility_factor_active_row_count': int(sum((r.hand_observation_visibility_factor_state == 'active_hand_observation_visibility' for r in rows))), 'hand_observation_visibility_candidate_px': numeric_summary(np.asarray([r.hand_observation_visibility_candidate_px for r in rows], dtype=float)), 'contact_patch_factor_active_row_count': int(sum((r.contact_patch_factor_state == 'active_contact_patch' for r in rows))), 'contact_patch_state_optimized': bool(args.optimize_contact_state), 'contact_patch_residual_mode': str(args.contact_patch_residual_mode), 'contact_patch_prior_probability': numeric_summary(np.asarray([r.contact_patch_prior_probability for r in rows if r.contact_patch_factor_state == 'active_contact_patch'], dtype=float)), 'contact_patch_geometry_target_probability': numeric_summary(contact_geometry_target_np[[i for (i, r) in enumerate(rows) if r.contact_patch_factor_state == 'active_contact_patch']].astype(float) if any((r.contact_patch_factor_state == 'active_contact_patch' for r in rows)) else np.asarray([], dtype=float)), 'contact_patch_posterior_probability': numeric_summary(contact_posterior_np[[i for (i, r) in enumerate(rows) if r.contact_patch_factor_state == 'active_contact_patch']].astype(float) if any((r.contact_patch_factor_state == 'active_contact_patch' for r in rows)) else np.asarray([], dtype=float)), 'contact_patch_posterior_minus_prior': numeric_summary(np.asarray([float(contact_posterior_np[i]) - float(r.contact_patch_prior_probability) for (i, r) in enumerate(rows) if r.contact_patch_factor_state == 'active_contact_patch'], dtype=float)), 'contact_patch_temporal_pair_count': int(contact_temporal_pairs_t.shape[0]), 'contact_patch_vertex_count': numeric_summary(np.asarray([len(r.contact_patch_vertex_indices) for r in rows], dtype=float)), 'contact_patch_initial_distance_m': numeric_summary(np.concatenate([r.contact_patch_initial_distance_m for r in rows if len(r.contact_patch_initial_distance_m)]).astype(float) if any((len(r.contact_patch_initial_distance_m) for r in rows)) else np.asarray([], dtype=float)), 'contact_patch_final_abs_normal_gap_m': numeric_summary(np.asarray(contact_patch_final_abs_normal_gap, dtype=float)), 'contact_patch_support_uncertainty_m': numeric_summary(np.asarray([r.contact_patch_support_uncertainty_m for r in rows if r.contact_patch_factor_state == 'active_contact_patch'], dtype=float)), 'contact_patch_support_uncertainty_source_counts': dict(Counter((str(r.contact_patch_support_uncertainty_source) for r in rows if r.contact_patch_factor_state == 'active_contact_patch'))), 'local_patch_support_state_counts': dict(Counter((str(r.local_patch_support_state) for r in rows if r.contact_patch_factor_state == 'active_contact_patch'))), 'local_patch_support_consumed_count': int(sum((bool(r.local_patch_support_consumed) for r in rows if r.contact_patch_factor_state == 'active_contact_patch'))), 'local_patch_support_uncertainty_m': numeric_summary(np.asarray([r.local_patch_support_uncertainty_m for r in rows if r.contact_patch_factor_state == 'active_contact_patch' and r.local_patch_support_uncertainty_m is not None], dtype=float)), 'global_object_support_uncertainty_m': numeric_summary(np.asarray([r.global_object_support_uncertainty_m for r in rows if r.contact_patch_factor_state == 'active_contact_patch' and r.global_object_support_uncertainty_m is not None], dtype=float)), 'local_patch_sample_count': numeric_summary(np.asarray([r.local_patch_sample_count for r in rows if r.contact_patch_factor_state == 'active_contact_patch'], dtype=float)), 'local_patch_temporal_sample_count': numeric_summary(np.asarray([r.local_patch_temporal_sample_count for r in rows if r.contact_patch_factor_state == 'active_contact_patch'], dtype=float)), 'contact_patch_anchor_coherence': contact_patch_anchor_coherence(rows), 'hand_observation_weight_multiplier': numeric_summary(hand_observation_weight_multiplier_np), 'joint_visibility_weight': numeric_summary(joint_visibility_weights_np.reshape(-1)), 'pose_visibility_weight': numeric_summary(pose_visibility_weights_np.reshape(-1)), 'pose_joint_finger_groups': pose_joint_finger_groups.astype(int).tolist(), 'hand_ray_shift_prior_enabled': bool(args.use_hand_ray_shift_prior), 'hand_ray_shift_prior_count': int(sum((np.linalg.norm(r.hand_ray_shift_prior_world_m) > 1e-09 for r in rows))), 'object_translation_optimized': bool(args.optimize_object_translation), 'hand_owned_object_depth_quarantine_enabled': bool(args.hand_owned_object_depth_quarantine), 'surface_eligibility_factor_enabled': args.surface_eligibility_factor_report is not None or any((r.surface_eligibility_mode is not None for r in rows)), 'surface_eligibility_mode': str(args.surface_eligibility_mode), 'observed_surface_support_uncertainty_m': numeric_summary(np.asarray([r.observed_surface_support_uncertainty_m for r in rows], dtype=float)), 'visible_ownership_factor_enabled': args.visible_ownership_factor_report is not None or any((r.visible_ownership_non_object_mask_path is not None or r.visible_ownership_object_owned_mask_path is not None or r.visible_ownership_constraint_eligible_mask_path is not None for r in rows)), 'visible_surface_track_factor_enabled': args.visible_surface_track_factor_report is not None or any((r.visible_surface_track_factor_state is not None for r in rows)), 'visible_object_mask_gate_enabled': bool(args.visible_object_mask_gate), 'visible_mask_quarantine_signed_mesh_enabled': bool(args.visible_mask_quarantine_signed_mesh), 'visible_surface_depth_order_term_enabled': bool(args.visible_surface_depth_order_term), 'visible_object_mask_report': None if args.visible_object_mask_report is None else str(args.visible_object_mask_report), 'dense_observed_surface_barrier_configured': bool(args.dense_observed_surface_barrier), 'dense_observed_surface_barrier_enabled': bool(args.dense_observed_surface_barrier and signed_surface_active), 'signed_object_surface_factor_active': signed_surface_active, 'dense_observed_constraint_count_final': numeric_summary(np.asarray([len(x) for x in dense_constraint_indices], dtype=float))}
    full_archive_rows = {'frame_idx': np.asarray([int(row.frame_idx) for row in rows], dtype=np.int32), 'hand_side': np.asarray([str(row.side) for row in rows], dtype='<U5'), 'vertices_world_m': np.stack(full_state_vertices_world).astype(np.float32), 'joints_world_m': np.stack(full_state_joints_world).astype(np.float32)}
    return (interval, states, full_archive_rows)

def infer_pose_joint_finger_groups(model: Any, base_root_mat: torch.Tensor, base_pose_mat: torch.Tensor, betas: torch.Tensor, trans: torch.Tensor) -> np.ndarray:
    """Map each MANO internal pose joint to the output finger group it moves most.

    The WiLoR wrapper returns OpenPose-ordered 21 hand joints, while MANO's 15
    hand_pose rotations use MANO's internal kinematic order.  A small local
    perturbation gives a model-specific mapping without relying on undocumented
    joint-name assumptions.
    """
    from typing import Any
    import numpy as np
    import torch
    from optimization.hand.mano import rotvec_to_matrix
    from observations.interaction.evidence import OPENPOSE_FINGER_GROUPS
    device = base_pose_mat.device
    out_groups: list[int] = []
    with torch.no_grad():
        base = model(global_orient=base_root_mat[:1], hand_pose=base_pose_mat[:1], betas=betas[:1], transl=trans[:1], return_verts=True, pose2rot=False)
        base_j = base.joints[0, :21]
        for pose_i in range(15):
            delta = torch.zeros((1, 15, 3), dtype=torch.float32, device=device)
            delta[0, pose_i, 0] = 0.08
            posed = rotvec_to_matrix(delta) @ base_pose_mat[:1]
            hyp = model(global_orient=base_root_mat[:1], hand_pose=posed, betas=betas[:1], transl=trans[:1], return_verts=True, pose2rot=False)
            disp = torch.linalg.norm(hyp.joints[0, :21] - base_j, dim=1).detach().cpu().numpy().astype(float)
            scores = [float(np.mean(disp[g])) for g in OPENPOSE_FINGER_GROUPS]
            out_groups.append(int(np.argmax(scores)))
    return np.asarray(out_groups, dtype=np.int64)

def build_rows(args: argparse.Namespace, side: str) -> tuple[list[FrameHandRow], dict[str, Any], Any]:
    from pathlib import Path
    import open3d as o3d
    import argparse
    from typing import Any
    import numpy as np
    from utils.data import load_mesh
    from utils.data import as_list
    from utils.geometry import frame_camera_pose
    from observations.interaction.evidence import load_hand_ray_shift_priors
    from observations.interaction.evidence import load_binary_mask
    from utils.data import pose_map
    from utils.geometry import finite_image_affine
    from gates.object.geometry import validate_completed_mesh_contract
    from observations.interaction.evidence import load_visible_object_mask_paths
    from observations.interaction.evidence import visible_object_mask_face_gate
    from gates.hand.observation import joint_visibility_from_metric_depth
    from observations.interaction.evidence import hand_owned_object_depth_quarantine
    from observations.interaction.evidence import load_depth_order_first_surface_for_row
    from gates.interaction.contact import load_surface_eligibility_rows
    from optimization.hand.mano import source_npz_for_hand
    from observations.object.surface import load_depth_sources
    from gates.interaction.contact import load_visible_surface_track_rows
    from observations.interaction.evidence import surface_eligibility_mask_for_row
    from gates.interaction.contact import contact_patch_for_row
    from observations.object.surface import face_provenance
    from observations.interaction.evidence import observed_constraints_for_hand
    from gates.interaction.contact import load_generic_factor_reports
    from gates.interaction.contact import merge_factor_row_maps
    from observations.interaction.evidence import visible_surface_depth_order_constraints
    from observations.interaction.evidence import visible_ownership_quarantine_faces
    from observations.object.surface import classify_object_vertices_against_depth
    from optimization.hand.mano import bridge_vertices_and_joints
    from optimization.hand.mano import similarity_from_to
    from observations.interaction.evidence import visible_surface_track_mask_for_row
    from observations.interaction.evidence import hand_depth_shift_prior_for_row
    from gates.interaction.contact import load_visible_ownership_rows
    from optimization.common.correspondences import contact_patch_targets_from_vertices
    from optimization.hand.mano import load_source_arrays
    from utils.data import load_json
    from gates.hand.observation import hand_observation_visibility_for_row
    from observations.interaction.evidence import visible_ownership_masks_for_row
    from utils.data import FrameHandRow
    from observations.interaction.evidence import inactive_signed_surface_measure
    annotations = load_json(args.annotations)
    frames = [f for f in as_list(annotations.get('frames')) if isinstance(f, dict)]
    frames_by_idx = {int(f['frame_idx']): f for f in frames}
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    validate_completed_mesh_contract(args.completed_mesh, args.completion_report)
    physical_surface_mesh = Path(getattr(args, 'resolved_physical_surface_mesh', args.completed_mesh))
    mesh = load_mesh(physical_surface_mesh)
    vertices_object = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri_obj = vertices_object[faces]
    face_normals_object = np.cross(tri_obj[:, 1] - tri_obj[:, 0], tri_obj[:, 2] - tri_obj[:, 0])
    face_normal_norms = np.linalg.norm(face_normals_object, axis=1)
    valid_face_normals = face_normal_norms > 1e-12
    face_normals_object[valid_face_normals] /= face_normal_norms[valid_face_normals, None]
    face_normals_object[~valid_face_normals] = 0.0
    signed_geometry_active_for_faces = bool(getattr(args, 'signed_object_surface_factor_active', False))
    signed_geometry_consumer_policy = str((getattr(args, 'completion_geometry_readiness', {}) or {}).get('signed_geometry_consumer_policy', ''))
    signed_face_authority_mask = np.asarray(getattr(args, 'signed_face_authority_mask', np.ones((len(faces),), dtype=bool)), dtype=bool)
    if signed_face_authority_mask.shape != (len(faces),):
        raise RuntimeError('mesh-bound signed face authority shape differs from physical surface faces')
    signed_face_authority_report = dict(getattr(args, 'signed_face_authority_report', {}) or {})
    local_signed_proxy_faces_eligible = bool(signed_geometry_active_for_faces and signed_geometry_consumer_policy == 'local_observation_authority_faces_only')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices_object.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    depth_paths = list(args.depth_npz)
    depth_rows = load_depth_sources(depth_paths)
    visible_mask_paths = load_visible_object_mask_paths(args.visible_object_mask_report)
    generic_factor_rows = load_generic_factor_reports(args.factor_report, target_entity_id=str(args.object_id))
    visible_ownership_rows = merge_factor_row_maps('visible_ownership', load_visible_ownership_rows(args.visible_ownership_factor_report, target_entity_id=str(args.object_id)), generic_factor_rows['visible_ownership'])
    surface_eligibility_rows = merge_factor_row_maps('surface_eligibility', load_surface_eligibility_rows(args.surface_eligibility_factor_report, target_entity_id=str(args.object_id)), generic_factor_rows['surface_eligibility'])
    visible_surface_track_rows = merge_factor_row_maps('visible_surface_track', load_visible_surface_track_rows(args.visible_surface_track_factor_report, target_entity_id=str(args.object_id)), generic_factor_rows['visible_surface_track'])
    hand_observation_visibility_rows = generic_factor_rows['hand_observation_visibility']
    hand_depth_shift_prior_rows = generic_factor_rows['hand_depth_shift_prior']
    contact_patch_rows = generic_factor_rows['contact_patch']
    visible_mask_cache: dict[Path, np.ndarray] = {}
    surface_eligibility_cache: dict[Path, np.ndarray] = {}
    hand_ray_shift_priors = load_hand_ray_shift_priors(args.hand_depth_repair_graph)
    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    rows: list[FrameHandRow] = []
    object_depth_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for frame_idx in range(int(args.start_frame), int(args.end_frame) + 1):
        frame = frames_by_idx.get(frame_idx)
        pose = poses.get(frame_idx)
        if frame is None or pose is None:
            skipped.append({'frame_idx': frame_idx, 'side': side, 'reason': 'missing_frame_or_object_pose'})
            continue
        (vertex_classes, obj_summary) = classify_object_vertices_against_depth(frame=frame, vertices_object=vertices_object, pose=pose, depth_row=depth_rows.get(frame_idx), support_margin_m=float(args.support_margin_m), free_space_margin_m=float(args.free_space_margin_m))
        object_depth_summaries.append(obj_summary)
        prov = face_provenance(vertex_classes, faces)
        strict_raw = signed_face_authority_mask.copy() if local_signed_proxy_faces_eligible else np.asarray(prov['observed_supported_strict'], dtype=bool)
        hand = None
        for h in as_list(frame.get('hands')):
            if isinstance(h, dict) and str(h.get('hand_side')) == side:
                hand = h
                break
        if hand is None:
            skipped.append({'frame_idx': frame_idx, 'side': side, 'reason': 'missing_hand'})
            continue
        arrays = bridge_vertices_and_joints(hand, bridge_cache)
        source_info = source_npz_for_hand(hand)
        if arrays is None or source_info is None:
            skipped.append({'frame_idx': frame_idx, 'side': side, 'reason': 'missing_bridge_or_source'})
            continue
        (current_vertices, current_joints) = arrays
        (strict, hand_owned_quarantined) = hand_owned_object_depth_quarantine(frame=frame, side=side, object_vertices=vertices_object, object_faces=faces, object_pose=pose, hand_vertices_world=current_vertices, face_strict_observed=strict_raw, depth_row=depth_rows.get(frame_idx), args=args)
        ownership_row = visible_ownership_rows.get((frame_idx, side))
        (ownership_non_object_mask, ownership_constraint_eligible_mask, ownership_diag) = visible_ownership_masks_for_row(ownership_row, visible_mask_cache)
        ownership_affine_raw = ownership_diag.get('A_mask_from_source_coordinate_model')
        if ownership_affine_raw is None:
            if ownership_non_object_mask is not None or ownership_constraint_eligible_mask is not None:
                raise RuntimeError(f'frame {frame_idx} {side}: ownership mask exists without an image-plane affine')
            A_mask_from_source = np.eye(3, dtype=np.float64)
            visible_mask_plane_contract: dict[str, Any] = {'state': 'not_consumed_missing_visible_ownership_factor_row', 'affine_placeholder_inert': True}
        else:
            A_mask_from_source = finite_image_affine(ownership_affine_raw, f'frame {frame_idx} {side} visible ownership A_mask_from_source')
            visible_mask_plane_contract = dict(ownership_diag.get('image_plane_transform') or {})
        (strict, visible_ownership_quarantined) = visible_ownership_quarantine_faces(frame=frame, side=side, object_vertices=vertices_object, object_faces=faces, object_pose=pose, face_strict_observed=strict, non_object_owned_mask=ownership_non_object_mask, A_mask_from_source=A_mask_from_source, args=args)
        visible_mask_path = visible_mask_paths.get(frame_idx)
        visible_mask = None if visible_mask_path is None else load_binary_mask(visible_mask_path, visible_mask_cache)
        visible_surface_row = visible_surface_track_rows.get((frame_idx, side))
        if (args.visible_surface_track_factor_report is not None or generic_factor_rows['visible_surface_track']) and visible_surface_row is None:
            raise ValueError(f'visible-surface track factor missing row for frame={frame_idx} side={side}')
        (visible_surface_mask, visible_surface_diag) = visible_surface_track_mask_for_row(visible_surface_row, visible_mask_cache)
        visible_surface_active = visible_surface_diag.get('state') == 'active_visible_surface' and visible_surface_mask is not None
        if visible_surface_active:
            visible_mask = visible_surface_mask
            visible_mask_path = Path(str(visible_surface_diag.get('surface_mask_path')))
        if ownership_constraint_eligible_mask is not None:
            visible_mask = ownership_constraint_eligible_mask if visible_mask is None else visible_mask & ownership_constraint_eligible_mask
        if visible_mask is not None and ownership_affine_raw is None:
            raise RuntimeError(f'frame {frame_idx} {side}: a visible mask would be consumed without an exact source-to-mask affine')
        (strict, visible_mask_face_count_raw, visible_mask_face_count) = visible_object_mask_face_gate(frame=frame, side=side, object_vertices=vertices_object, object_faces=faces, object_pose=pose, face_strict_observed=strict, mask=visible_mask, A_mask_from_source=A_mask_from_source, args=args)
        visible_surface_track_quarantined_face_count = 0
        if visible_surface_active and bool(visible_surface_diag.get('quarantine_hidden_volume', True)):
            visible_surface_track_quarantined_face_count = int(np.count_nonzero(strict))
            strict[:] = False
            visible_mask_face_count = 0
        surface_row = surface_eligibility_rows.get((frame_idx, side))
        (surface_mask, surface_diag) = surface_eligibility_mask_for_row(surface_row, len(faces), surface_eligibility_cache)
        surface_input_face_count = int(np.count_nonzero(strict))
        if args.surface_eligibility_factor_report is not None and surface_mask is None:
            raise ValueError(f'surface eligibility factor could not supply frame={frame_idx} side={side}: {surface_diag}')
        if surface_mask is not None:
            if str(args.surface_eligibility_mode) == 'replace':
                strict = surface_mask.astype(bool)
            else:
                strict = strict & surface_mask.astype(bool)
        if local_signed_proxy_faces_eligible:
            strict = signed_face_authority_mask.copy()
        surface_eligible_face_count = int(np.count_nonzero(strict))
        surface_applied_face_delta = surface_eligible_face_count - surface_input_face_count
        observed_surface_support_uncertainty_m = float(surface_diag.get('observed_surface_support_uncertainty_m', args.observed_surface_support_uncertainty_m) or 0.0)
        (source_path, source_frame) = source_info
        source = load_source_arrays(source_cache, source_path)
        required = [f'{side}_vertices_world_m', f'{side}_joints_world_m', f'{side}_root_orient_axis_angle', f'{side}_hand_pose_axis_angle', f'{side}_betas', f'{side}_trans_world_m']
        missing = [key for key in required if key not in source]
        if missing:
            skipped.append({'frame_idx': frame_idx, 'side': side, 'reason': 'source_missing_arrays', 'missing': missing})
            continue
        raw_vertices = np.asarray(source[f'{side}_vertices_world_m'][source_frame], dtype=float)
        raw_joints = np.asarray(source[f'{side}_joints_world_m'][source_frame], dtype=float)
        (scale, rot, sim_trans, _err) = similarity_from_to(raw_vertices, current_vertices)
        (cidx, normals, depths, measure) = observed_constraints_for_hand(vertices_world=current_vertices, pose=pose, scene=scene, face_strict_observed=strict, frame_idx=frame_idx, max_constraints=int(args.max_constraints_per_frame), eps=float(args.penetration_epsilon_m), support_uncertainty_m=observed_surface_support_uncertainty_m) if bool(getattr(args, 'signed_object_surface_factor_active', False)) else (np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), inactive_signed_surface_measure(frame_idx, observed_surface_support_uncertainty_m))
        if bool(args.visibility_weighted_hand_observation):
            (joint_visibility_weights, joint_depth_residual) = joint_visibility_from_metric_depth(frame, side, current_joints, depth_rows.get(frame_idx), args)
        else:
            joint_visibility_weights = np.ones((21,), dtype=float)
            joint_depth_residual = np.full((21,), np.nan, dtype=float)
        hand_visibility_diag = hand_observation_visibility_for_row(hand_observation_visibility_rows.get((frame_idx, side)), args)
        if hand_visibility_diag.get('state') == 'active_hand_observation_visibility':
            joint_visibility_weights = np.minimum(joint_visibility_weights, float(hand_visibility_diag.get('weight_multiplier', 1.0)))
        (depth_order_mask, depth_order_depth, depth_order_diag) = load_depth_order_first_surface_for_row(ownership_diag, visible_mask_cache, ownership_constraint_eligible_mask.shape if ownership_constraint_eligible_mask is not None else None)
        depth_order_enabled = bool(args.visible_surface_depth_order_term) or bool(visible_surface_active)
        if bool(args.visible_surface_depth_order_term) and ownership_row is not None and (depth_order_mask is None):
            raise ValueError(f'frame {frame_idx} {side}: P17 ownership row lacks independent P09 first-hit query support; ownership mask is not a valid substitute')
        hard_depth_order_mask = visible_mask & depth_order_mask if visible_mask is not None and depth_order_mask is not None else None
        (surface_depth_idx, surface_depth_m, surface_depth_delta, surface_depth_measure) = visible_surface_depth_order_constraints(frame=frame, side=side, vertices_world=current_vertices, mask=hard_depth_order_mask, depth_row={'depth': depth_order_depth} if depth_order_depth is not None else None, A_mask_from_source=A_mask_from_source, depth_is_mask_plane=True, mask_dilation_px=0, args=args, enabled=depth_order_enabled)
        surface_depth_measure['depth_source'] = 'P09_accepted_first_surface_not_raw_hand_pixel_depth'
        (translation_support_idx, translation_support_depth, translation_support_delta, translation_support_measure) = visible_surface_depth_order_constraints(frame=frame, side=side, vertices_world=current_vertices, mask=depth_order_mask, depth_row={'depth': depth_order_depth} if depth_order_depth is not None else None, A_mask_from_source=A_mask_from_source, depth_is_mask_plane=True, mask_dilation_px=0, args=args, enabled=depth_order_enabled)
        translation_support_measure['independent_query_support'] = depth_order_diag
        translation_support_measure['residual_policy'] = 'translation_gate_grounding_only_not_a_one_sided_depth_order_force'
        (r_obj, t_obj) = pose
        ray_shift = hand_ray_shift_priors.get((frame_idx, side))
        hand_depth_shift_diag = hand_depth_shift_prior_for_row(hand_depth_shift_prior_rows.get((frame_idx, side)))
        contact_patch_diag = contact_patch_for_row(contact_patch_rows.get((frame_idx, side)), args)
        contact_patch_idx = np.zeros((0,), dtype=np.int64)
        contact_patch_targets = np.zeros((0, 3), dtype=float)
        contact_patch_normals = np.zeros((0, 3), dtype=float)
        contact_patch_distances = np.zeros((0,), dtype=float)
        if contact_patch_diag.get('state') == 'active_contact_patch' and float(contact_patch_diag.get('weight', 0.0)) > 0.0:
            if bool(args.require_contact_patch_pose_anchor) and (not bool(contact_patch_diag.get('contact_anchor_residual_allowed'))):
                raise ValueError(f"active contact_patch row lacks stable contact-anchor support; refusing to upgrade local visible-surface contact to persistent A_t pose anchor for frame={frame_idx} side={side} state={contact_patch_diag.get('contact_anchor_state')} blockers={contact_patch_diag.get('contact_anchor_blockers')}")
            (contact_patch_idx, contact_patch_targets, contact_patch_normals, contact_patch_distances) = contact_patch_targets_from_vertices(current_vertices, scene, np.asarray(r_obj, dtype=float), np.asarray(t_obj, dtype=float), strict.astype(bool), face_normals_object.astype(float), max_vertices=int(contact_patch_diag.get('max_vertices', args.max_contact_patch_vertices)), band_m=float(contact_patch_diag.get('band_m', args.contact_patch_band_m)))
        (r_c2w_frame, _t_c2w_frame) = frame_camera_pose(frame)
        ray_prior_world = np.zeros(3, dtype=float)
        ray_prior_source: float | None = None
        ray_prior_weight = float(args.hand_ray_shift_prior_weight)
        if bool(args.use_hand_ray_shift_prior) and ray_shift is not None:
            ray_prior_world = -float(ray_shift) * np.asarray(r_c2w_frame[:, 2], dtype=float)
            ray_prior_source = float(ray_shift)
        if hand_depth_shift_diag.get('state') == 'active_hand_depth_shift_prior':
            camera_z_shift = float(hand_depth_shift_diag.get('camera_z_shift_m', 0.0))
            ray_prior_world = camera_z_shift * np.asarray(r_c2w_frame[:, 2], dtype=float)
            ray_prior_source = camera_z_shift
            if hand_depth_shift_diag.get('weight') is not None:
                ray_prior_weight = float(hand_depth_shift_diag['weight'])
        rows.append(FrameHandRow(frame_idx=frame_idx, side=side, frame=frame, current_vertices_world=current_vertices, current_joints_world=current_joints, raw_vertices_world=raw_vertices, raw_joints_world=raw_joints, root_orient_axis_angle=np.asarray(source[f'{side}_root_orient_axis_angle'][source_frame], dtype=float), hand_pose_axis_angle=np.asarray(source[f'{side}_hand_pose_axis_angle'][source_frame], dtype=float), betas=np.asarray(source[f'{side}_betas'][source_frame], dtype=float), trans_world_m=np.asarray(source[f'{side}_trans_world_m'][source_frame], dtype=float), similarity_scale=float(scale), similarity_rotation_raw_to_current=rot.astype(float), similarity_translation_raw_to_current=np.asarray(sim_trans, dtype=float), source_hawor_npz=source_path, source_frame_index=int(source_frame), constraint_indices=cidx, constraint_normals_world=normals, constraint_depths_m=depths, observed_initial_measure=measure, observed_constraint_count=int(len(depths)), object_rotation_world_from_object=np.asarray(r_obj, dtype=float), object_translation_world_m=np.asarray(t_obj, dtype=float), face_strict_observed_raw=strict_raw.astype(bool), face_strict_observed=strict.astype(bool), signed_face_authority_required=bool(signed_face_authority_report.get('required')), signed_face_authority_state=str(signed_face_authority_report.get('state') or 'legacy_not_required'), signed_face_authority_face_count=int(signed_face_authority_report.get('face_count', len(faces))), signed_face_authority_eligible_face_count=int(signed_face_authority_report.get('eligible_face_count', np.count_nonzero(strict_raw))), hand_owned_quarantined_face_count=int(hand_owned_quarantined), surface_eligibility_npz_path=surface_diag.get('face_state_npz_path'), surface_eligibility_mode=str(args.surface_eligibility_mode) if surface_mask is not None else None, observed_surface_support_uncertainty_m=float(observed_surface_support_uncertainty_m), surface_eligible_face_count=int(surface_eligible_face_count), surface_input_face_count=int(surface_input_face_count), surface_applied_face_delta=int(surface_applied_face_delta), visible_ownership_non_object_mask_path=ownership_diag.get('non_object_owned_mask_path'), visible_ownership_object_owned_mask_path=ownership_diag.get('visible_object_owned_mask_path'), visible_ownership_constraint_eligible_mask_path=ownership_diag.get('constraint_eligible_entity_mask_path'), visible_ownership_non_object_owned_px=int(ownership_diag.get('non_object_owned_px', 0)), visible_ownership_object_owned_px=int(ownership_diag.get('visible_object_owned_px', 0)), visible_ownership_constraint_eligible_px=int(ownership_diag.get('constraint_eligible_entity_px', 0)), visible_ownership_quarantined_face_count=int(visible_ownership_quarantined), visible_mask_A_mask_from_source=A_mask_from_source.astype(float), visible_mask_image_plane_contract=visible_mask_plane_contract, visible_object_mask_path=None if visible_mask_path is None else str(visible_mask_path), visible_object_mask_face_count_raw=int(visible_mask_face_count_raw), visible_object_mask_face_count=int(visible_mask_face_count), visible_surface_track_factor_state=visible_surface_diag.get('state') if args.visible_surface_track_factor_report is not None or generic_factor_rows['visible_surface_track'] else None, visible_surface_track_mask_path=visible_surface_diag.get('surface_mask_path'), visible_surface_track_npz_path=visible_surface_diag.get('visible_surface_npz_path'), visible_surface_track_valid_depth_pixels=int(visible_surface_diag.get('valid_depth_pixels', 0) or 0), visible_surface_track_quarantined_face_count=int(visible_surface_track_quarantined_face_count), visible_surface_depth_order_vertex_indices=surface_depth_idx.astype(np.int64), visible_surface_depth_order_depth_m=surface_depth_m.astype(float), visible_surface_depth_order_initial_delta_m=surface_depth_delta.astype(float), visible_surface_depth_order_initial_measure=surface_depth_measure, visible_surface_translation_support_vertex_indices=translation_support_idx.astype(np.int64), visible_surface_translation_support_depth_m=translation_support_depth.astype(float), visible_surface_translation_support_initial_delta_m=translation_support_delta.astype(float), visible_surface_translation_support_measure=translation_support_measure, hand_observation_visibility_factor_state=hand_visibility_diag.get('state') if hand_visibility_diag.get('state') != 'missing_hand_observation_visibility_row' else None, hand_observation_visibility_candidate_px=int(hand_visibility_diag.get('candidate_px', 0)), hand_observation_visibility_weight_multiplier=float(hand_visibility_diag.get('weight_multiplier', 1.0)), contact_patch_factor_state=contact_patch_diag.get('state') if contact_patch_diag.get('state') != 'missing_contact_patch_row' else None, contact_patch_vertex_indices=contact_patch_idx.astype(np.int64), contact_patch_target_world_m=contact_patch_targets.astype(float), contact_patch_normal_world=contact_patch_normals.astype(float), contact_patch_initial_distance_m=contact_patch_distances.astype(float), contact_patch_weight=float(contact_patch_diag.get('weight', 0.0)), contact_patch_prior_probability=float(contact_patch_diag.get('prior_probability', 0.0)), contact_patch_band_m=float(contact_patch_diag.get('band_m', args.contact_patch_band_m)), contact_patch_target_margin_m=float(contact_patch_diag.get('target_margin_m', args.contact_patch_target_margin_m)), contact_patch_support_uncertainty_m=float(contact_patch_diag.get('support_uncertainty_m', args.contact_patch_support_uncertainty_m)), contact_patch_support_uncertainty_source=contact_patch_diag.get('support_uncertainty_source'), local_patch_support_state=contact_patch_diag.get('local_patch_support_state'), local_patch_support_consumed=bool(contact_patch_diag.get('local_patch_support_consumed')), local_patch_support_uncertainty_m=contact_patch_diag.get('local_patch_support_uncertainty_m'), global_object_support_uncertainty_m=contact_patch_diag.get('global_object_support_uncertainty_m'), local_patch_sample_count=int(contact_patch_diag.get('local_patch_sample_count', 0)), local_patch_temporal_sample_count=int(contact_patch_diag.get('local_patch_temporal_sample_count', 0)), contact_anchor_state=contact_patch_diag.get('contact_anchor_state'), contact_anchor_residual_allowed=bool(contact_patch_diag.get('contact_anchor_residual_allowed')), contact_anchor_blockers=list(contact_patch_diag.get('contact_anchor_blockers', [])), contact_pose_anchor_key=contact_patch_diag.get('contact_pose_anchor_key'), joint_visibility_weights=joint_visibility_weights.astype(float), joint_depth_residual_m=joint_depth_residual.astype(float), hand_ray_shift_prior_world_m=ray_prior_world.astype(float), hand_ray_shift_prior_source_m=ray_prior_source, hand_ray_shift_prior_weight=float(ray_prior_weight)))
    meta = {'side': side, 'requested_start_frame': int(args.start_frame), 'requested_end_frame': int(args.end_frame), 'row_count': int(len(rows)), 'skipped': skipped, 'object_depth_summaries': object_depth_summaries[:5], 'pose_hypothesis_mesh': str(args.completed_mesh), 'physical_surface_mesh': str(physical_surface_mesh), 'physical_surface_semantics': getattr(args, 'physical_surface_semantics', 'legacy_unknown'), 'geometry_readiness': getattr(args, 'completion_geometry_readiness', {}), 'signed_object_surface_factor_active': bool(getattr(args, 'signed_object_surface_factor_active', False)), 'local_signed_proxy_faces_eligible': local_signed_proxy_faces_eligible, 'signed_face_authority': signed_face_authority_report, 'signed_geometry_consumer_policy': signed_geometry_consumer_policy}
    return (rows, meta, scene)

def quarantined_source_state(args: argparse.Namespace, pose_readiness: dict[str, Any]) -> dict[str, Any]:
    """Emit source-only hand rows without loading MANO models or physical factors."""
    import json
    from pathlib import Path
    import argparse
    from typing import Any
    import numpy as np
    from collections import Counter
    from utils.data import as_list
    from utils.data import load_json
    from utils.data import write_json
    annotations = load_json(args.annotations)
    rows: list[dict[str, Any]] = []
    side_counts: Counter[str] = Counter()
    for (pos, frame) in enumerate(as_list(annotations.get('frames'))):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get('frame_idx', pos))
        if frame_idx < int(args.start_frame) or frame_idx > int(args.end_frame):
            continue
        hands = {str(hand.get('hand_side')): hand for hand in as_list(frame.get('hands')) if isinstance(hand, dict) and str(hand.get('hand_side')) in {'left', 'right'}}
        for side in args.sides:
            hand = hands.get(side)
            if hand is None:
                continue
            metric = hand.get('metric_mano_state') if isinstance(hand.get('metric_mano_state'), dict) else {}
            joints = np.asarray(metric.get('joints_current_v18_world_m') or metric.get('joints_world_m') or hand.get('joints_current_v18_world_m') or [], dtype=float)
            joints_payload = joints.astype(float).tolist() if joints.shape == (21, 3) and np.isfinite(joints).all() else []
            rows.append({'frame_idx': frame_idx, 'hand_side': side, 'status': 'quarantined_unready_object_pose_trajectory', 'annotation_ready': False, 'physical_constraint_quarantine': 'p15_unready_object_pose_trajectory', 'temporal_mano_state': 'source_metric_mano_only_object_relative_optimization_skipped', 'joint_state_policy': 'source_metric_mano_preserved_due_to_unready_object_pose', 'optimized_joints_world_m': joints_payload, 'optimized_vertices_world_sample_m': [], 'contact_surface_vertices_world_sample_m': [], 'contact_state': 'unresolved_object_pose_trajectory', 'candidate_application_state': 'quarantined_unready_object_pose_trajectory', 'object_pose_readiness': pose_readiness, 'uncertainty': 'P15 trusted graph support is insufficient; no object-relative contact, nonpenetration, depth-order, or MANO optimization was run'})
            side_counts[side] += 1
    if not rows:
        raise RuntimeError('cannot build source-only P18 quarantine state: no hand rows in the requested interval')
    intervals = [{'hand_side': side, 'start_frame': int(args.start_frame), 'end_frame': int(args.end_frame), 'frame_count': int(side_counts.get(side, 0)), 'solver': 'not_run_unready_object_pose_trajectory', 'state': 'quarantined_source_metric_mano_only'} for side in args.sides]
    report = {'method': 'solve_v18_joint_mano_interval_trajectory', 'status': 'completed_quarantined_unready_object_pose_trajectory', 'annotation_ready': False, 'case': str(args.case), 'object_id': str(args.object_id), 'claim_scope': 'P15 explicitly reports an unready object trajectory. P18 therefore preserves source hand rows and skips all object-relative MANO/contact/nonpenetration optimization; this output is a quarantine state, not a physical solve.', 'inputs': {'annotations': str(args.annotations), 'pose_report': str(args.pose_report), 'completed_mesh': str(args.completed_mesh), 'pose_hypothesis_mesh': str(args.completed_mesh), 'physical_surface_mesh_not_loaded': str(getattr(args, 'resolved_physical_surface_mesh', args.completed_mesh)), 'physical_surface_semantics': getattr(args, 'physical_surface_semantics', 'legacy_unknown'), 'completion_geometry_readiness': getattr(args, 'completion_geometry_readiness', {}), 'completion_report': None if args.completion_report is None else str(args.completion_report), 'depth_npz_not_consumed': [str(path) for path in list(args.depth_npz)], 'factor_report': None if args.factor_report is None else [str(path) for path in args.factor_report]}, 'object_pose_readiness': pose_readiness, 'object_pose_input_ready_for_optimization': False, 'object_pose_input_legacy_compatibility': bool(pose_readiness['legacy_readiness_fields_missing']), 'optimization_skipped': True, 'physical_state_quarantined': True, 'physical_factor_families_quarantined': ['object_surface_nonpenetration', 'visible_surface_depth_order', 'contact_patch', 'object_relative_mano_correction'], 'physical_surface_contract': {'pose_hypothesis_mesh': str(args.completed_mesh), 'physical_surface_mesh': str(getattr(args, 'resolved_physical_surface_mesh', args.completed_mesh)), 'physical_surface_semantics': getattr(args, 'physical_surface_semantics', 'legacy_unknown'), 'geometry_readiness': getattr(args, 'completion_geometry_readiness', {}), 'mesh_not_loaded_due_to_pose_quarantine': True}, 'parameters': {key: [str(item) for item in value] if key == 'factor_report' and value is not None else str(value) if isinstance(value, Path) else value for (key, value) in vars(args).items() if key not in {'depth_npz', 'signed_face_authority_mask', 'signed_face_authority_report'}}, 'build_meta': {}, 'summary': {'interval_count': len(intervals), 'per_frame_state_count': len(rows), 'frame_span': [int(args.start_frame), int(args.end_frame)], 'sides': list(args.sides), 'optimization_skipped': True}, 'intervals': intervals, 'per_frame_states': rows, 'scientific_test': 'No object-relative scientific test was run because the object trajectory failed its support gate.'}
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'v18_joint_mano_interval_trajectory_state.json'
    write_json(out, report)
    print(json.dumps({'output': str(out), 'status': report['status'], 'summary': report['summary']}, indent=2))
    return report

def formal_interval_arguments(argv=None) -> argparse.Namespace:
    import torch
    from pathlib import Path
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--object-id', required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--pose-report', type=Path, required=True)
    p.add_argument('--include-unready-object-pose-for-diagnostic-optimization', action='store_true', help='Diagnostic/historical override: run object-relative MANO optimization even when P15 explicitly reports annotation_ready=false or insufficient graph support')
    p.add_argument('--completed-mesh', type=Path, help='P13 completed-canonical pose hypothesis mesh. This preserves the canonical frame contract; physical factors use --physical-surface-mesh or the completion report collision_eligible_mesh_labeled output.', required=True)
    p.add_argument('--physical-surface-mesh', type=Path, default=None, help='Optional explicit collision-eligible surface. With --completion-report it must match outputs.collision_eligible_mesh_labeled; generated hidden pose-hypothesis faces are not a physical fallback.')
    p.add_argument('--completion-report', type=Path, default=None, help='Optional P13/P14b completion report. --completed-mesh must match its pose hypothesis and physical factors use its collision-eligible surface.')
    p.add_argument('--allow-unready-signed-geometry-for-diagnostic-optimization', action='store_true', help='Historical diagnostic override: allow signed-mesh terms without geometry_readiness.signed_geometry_ready=true. Output remains non-annotation-ready and records the override.')
    p.add_argument('--depth-npz', type=Path, action='append', required=True)
    p.add_argument('--hand-depth-repair-graph', type=Path, default=None, help='Optional prior source with per-frame hand_ray_shift_m camera-ray observations from the V17 hand-depth repair graph.')
    p.add_argument('--use-hand-ray-shift-prior', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--initialize-hand-ray-shift', action=argparse.BooleanOptionalAction, default=False, help='Initialize MANO translation deltas from the hand-ray depth repair observation for a discriminating repair test.')
    p.add_argument('--hand-ray-shift-prior-weight', type=float, default=2500.0)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--wilor-root', type=Path, required=True)
    p.add_argument('--wilor-mano-right', type=Path, default=None)
    p.add_argument('--wilor-mano-left', type=Path, required=True)
    p.add_argument('--hawor-left-shapedirs-x-fix', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--start-frame', type=int, required=True)
    p.add_argument('--end-frame', type=int, required=True)
    p.add_argument('--sides', nargs='+', choices=('left', 'right'), default=['left', 'right'])
    p.add_argument('--support-margin-m', type=float, default=0.015)
    p.add_argument('--free-space-margin-m', type=float, default=0.025)
    p.add_argument('--penetration-epsilon-m', type=float, default=1e-05)
    p.add_argument('--max-constraints-per-frame', type=int, default=96)
    p.add_argument('--max-optimizer-iterations', type=int, default=120)
    p.add_argument('--visible-shift-limit-px', type=float, default=12.0)
    p.add_argument('--depth-shift-limit-m', type=float, default=0.035)
    p.add_argument('--max-translation-m', type=float, default=0.045)
    p.add_argument('--max-root-delta-rad', type=float, default=0.3)
    p.add_argument('--max-pose-delta-rad', type=float, default=0.45)
    p.add_argument('--translation-prior-weight', type=float, default=2000.0)
    p.add_argument('--root-prior-weight', type=float, default=150.0)
    p.add_argument('--pose-prior-weight', type=float, default=75.0)
    p.add_argument('--smooth-weight', type=float, default=5000.0)
    p.add_argument('--accel-weight', type=float, default=10000.0)
    p.add_argument('--observed-penetration-weight', type=float, default=300000.0)
    p.add_argument('--zero-surface-mode', choices=('bridge_delta', 'similarity_mapped_raw'), default='bridge_delta', help='bridge_delta uses the current V18 bridge as the zero surface and maps MANO deltas onto it; similarity_mapped_raw optimizes the similarity-transformed MANO surface directly.')
    p.add_argument('--dense-observed-surface-barrier', action=argparse.BooleanOptionalAction, default=True, help='Apply a tangent-plane nonpenetration barrier to every MANO vertex whose nearest object face is observed-supported, not only the current active penetrating subset.')
    p.add_argument('--dense-observed-penetration-weight', type=float, default=300000.0)
    p.add_argument('--optimize-object-translation', action=argparse.BooleanOptionalAction, default=False, help='Jointly solve a small per-frame object translation delta so hand/object residual can expose tomato pose-depth alignment error instead of forcing all correction into MANO.')
    p.add_argument('--hand-owned-object-depth-quarantine', action=argparse.BooleanOptionalAction, default=False, help='Do not treat object faces as trusted observed-surface constraints when their projected depth is plausibly owned by the visible/current hand surface.')
    p.add_argument('--hand-owned-quarantine-radius-px', type=float, default=3.0)
    p.add_argument('--hand-owned-quarantine-depth-margin-m', type=float, default=0.005, help='Required camera-depth foreground separation for hand-owned object-depth quarantine.')
    p.add_argument('--hand-owned-quarantine-hand-depth-support-m', type=float, default=0.03)
    p.add_argument('--max-object-translation-m', type=float, default=0.015, help='Object translation uncertainty bound; default equals the observed-depth support margin scale.')
    p.add_argument('--object-translation-prior-weight', type=float, default=4000.0)
    p.add_argument('--object-smooth-weight', type=float, default=8000.0)
    p.add_argument('--visibility-weighted-hand-observation', action=argparse.BooleanOptionalAction, default=False, help='Use metric depth support to reduce HaWoR joint/articulation anchoring for occluded or depth-inconsistent fingers while preserving visible joints.')
    p.add_argument('--visible-joint-depth-margin-m', type=float, default=0.03)
    p.add_argument('--occluded-joint-observation-weight', type=float, default=0.12)
    p.add_argument('--front-inconsistent-joint-observation-weight', type=float, default=0.35)
    p.add_argument('--invalid-joint-observation-weight', type=float, default=0.35)
    p.add_argument('--visible-hinge-weight', type=float, default=800.0)
    p.add_argument('--depth-hinge-weight', type=float, default=20000.0)
    p.add_argument('--bound-hinge-weight', type=float, default=3000.0)
    p.add_argument('--sample-vertex-count-for-render', type=int, default=160)
    p.add_argument('--active-set-iterations', type=int, default=6, help='Maximum active-set passes. Each pass optimizes, remeasures full observed-surface penetration, and expands constraints. A closed pass adds zero constraints.')
    p.add_argument('--visible-ownership-factor-report', type=Path, default=None, help='Optional reusable visible ownership factor report. Its non_object_owned masks quarantine hard object constraints; its visible_object_owned masks replace visible-object masks for depth-order/gating when present.')
    p.add_argument('--surface-eligibility-factor-report', type=Path, default=None, help='Optional reusable surface eligibility factor report. Its eligible_hard_observed face masks define which object faces may exert hard MANO constraints.')
    p.add_argument('--surface-eligibility-mode', choices=('replace', 'intersect'), default='intersect', help='How to apply surface eligibility to the current trusted face set when a factor report is supplied. Default intersects to preserve existing ownership/visibility quarantines unless replacement is explicitly justified.')
    p.add_argument('--observed-surface-support-uncertainty-m', type=float, default=0.0, help='Default support uncertainty slack for observed object surface nonpenetration. Row-level surface_support_uncertainty_m / observed_surface_support_uncertainty_m from surface_eligibility factors overrides this value.')
    p.add_argument('--visible-surface-track-factor-report', type=Path, default=None, help='Optional reusable visible-surface track factor report. Active rows provide model-mask/metric-depth first-surface observations for MANO depth-order and hidden-volume quarantine.')
    p.add_argument('--factor-report', type=Path, action='append', default=None, help='Generic reusable factor report(s). Rows are dispatched by factor_family, enabling ownership, surface_eligibility, visible_surface_track, hand_observation_visibility, hand_depth_shift_prior, and contact_patch factors through one interface.')
    p.add_argument('--contact-patch-weight', type=float, default=0.0, help='Default weight for active contact_patch factor rows. A row-level weight overrides this value. The residual keeps selected MANO vertices near an observed object surface patch; existing nonpenetration prevents crossing.')
    p.add_argument('--contact-patch-band-m', type=float, default=0.02, help='Current-state max distance for selecting MANO vertices that define the local contact patch.')
    p.add_argument('--contact-patch-target-margin-m', type=float, default=0.0025, help='Allowed hand-to-patch distance before the two-sided contact residual is active.')
    p.add_argument('--contact-patch-support-uncertainty-m', type=float, default=0.0, help='Independent object/patch support uncertainty added to the contact_patch deadband. Row-level object_support_uncertainty_m overrides this value. Use this to represent latent/sliding contact support rather than hard current-surface anchoring.')
    p.add_argument('--contact-patch-residual-mode', choices=('deadband_tube', 'support_scaled_attraction'), default='deadband_tube', help='deadband_tube preserves the original bounded-contact residual: no force inside target margin plus support uncertainty. support_scaled_attraction treats active contact as a soft normal-manifold likelihood whose precision is reduced by object/patch support uncertainty, so it can test whether contact evidence can move H_t without claiming millimetre object support.')
    p.add_argument('--max-contact-patch-vertices', type=int, default=96, help='Maximum MANO vertices selected for each contact_patch factor row.')
    p.add_argument('--optimize-contact-state', action=argparse.BooleanOptionalAction, default=False, help='Optimize a per-row latent contact probability C_t for contact_patch rows. The contact residual is weighted by posterior C_t, while observation priors and temporal continuity keep plausible contacts active.')
    p.add_argument('--contact-state-prior-residual-scale-m', type=float, default=0.01, help='Physical residual scale used to convert contact row weights into contact-state prior strength. Deviating C_t from its observation prior by 1 costs the same as this many metres of contact residual.')
    p.add_argument('--contact-state-temporal-strength', type=float, default=1.0, help='Multiplier on same-side adjacent-frame C_t smoothness, using the same physical weight scale as the contact-state prior.')
    p.add_argument('--contact-state-geometry-likelihood', action=argparse.BooleanOptionalAction, default=False, help="Add a metric contact-compatibility observation on C_t from the selected current MANO-to-patch distance relative to the row's target margin plus object/patch support uncertainty. This makes C_t an evidence-updated switch instead of only a prior/temporal scalar.")
    p.add_argument('--require-contact-patch-pose-anchor', action=argparse.BooleanOptionalAction, default=False, help='Fail loudly if an active contact_patch row is consumed without explicit stable contact-anchor support. Use only to test a persistent object-frame A_t/pose-anchor mechanism; local visible-surface contact rows must not be silently upgraded to point anchors.')
    p.add_argument('--visible-ownership-face-overlap-dilation-px', type=int, default=2, help='Pixel dilation for deciding whether any projected face support sample overlaps non-object-owned ownership pixels.')
    p.add_argument('--visible-object-mask-report', type=Path, default=None, help='Legacy visible entity mask report. Prefer --factor-report with factor_family=visible_surface_track; this path remains only for reproducing earlier mask/depth ablations.')
    p.add_argument('--visible-object-mask-gate', action=argparse.BooleanOptionalAction, default=False, help='Legacy gate: trust observed object mesh faces only when their projected center lies inside the model-produced visible entity mask.')
    p.add_argument('--visible-mask-quarantine-signed-mesh', action=argparse.BooleanOptionalAction, default=False, help='On frames with a visible entity mask or active visible-surface factor, do not use compact-mesh signed nonpenetration as a trusted force; rely on visible first-surface constraints instead.')
    p.add_argument('--visible-object-mask-dilation-px', type=int, default=2)
    p.add_argument('--visible-surface-depth-order-term', dest='visible_surface_depth_order_term', action=argparse.BooleanOptionalAction, default=False, help='Penalize MANO vertices that project inside a visible first-surface mask but remain in front of the observed metric depth beyond a margin. Active visible_surface_track factors enable this residual automatically.')
    p.add_argument('--visible-surface-depth-order-margin-m', dest='visible_surface_depth_order_margin_m', type=float, default=0.01)
    p.add_argument('--visible-surface-depth-order-weight', dest='visible_surface_depth_order_weight', type=float, default=20000.0)
    p.add_argument('--max-visible-surface-depth-vertices', dest='max_visible_surface_depth_vertices', type=int, default=160)
    p.add_argument('--gate-translation-with-visible-surface-support', action=argparse.BooleanOptionalAction, default=False, help='Output gate: when selected visible-surface depth-order support is at or below the threshold, preserve the source HaWoR wrist/root translation while keeping optimized wrist-relative MANO articulation. This prevents ungrounded contact/temporal terms from moving the global hand state.')
    p.add_argument('--freeze-translation-without-visible-surface-support', action=argparse.BooleanOptionalAction, default=False, help='In-solver gate: rows whose selected visible-surface depth-order support count is at or below --translation-gate-min-visible-surface-depth-vertices cannot use global MANO translation during optimization. This tests whether unsupported latent translation contaminates wrist-relative articulation before the output gate projects the wrist/root back to HaWoR.')
    p.add_argument('--translation-gate-min-visible-surface-depth-vertices', type=int, default=0, help='Rows with selected visible-surface depth-order vertex count <= this value are translation-gated when --gate-translation-with-visible-surface-support is enabled, and translation-frozen during optimization when --freeze-translation-without-visible-surface-support is enabled.')
    p.add_argument('--visible-lid-depth-order-term', dest='visible_surface_depth_order_term', action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument('--visible-lid-depth-order-margin-m', dest='visible_surface_depth_order_margin_m', type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument('--visible-lid-depth-order-weight', dest='visible_surface_depth_order_weight', type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument('--max-visible-lid-depth-vertices', dest='max_visible_surface_depth_vertices', type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument('--require-active-full-K-mano-contract', action=argparse.BooleanOptionalAction, default=False, help='Fail before MANO model loading unless every interval hand row proves that HaWoR regression, projected hand masks, and HaWoR SLAM consumed the exact affine-centered inference plane bound to active source [fx,fy,cx,cy]. The controlled HOT3D shared tail enables this explicitly.')
    return p.parse_args(argv)
