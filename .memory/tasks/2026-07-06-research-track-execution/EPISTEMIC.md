# Current epistemic model

The research task is a mechanism-resolution problem. The artifact that matters is renderable HOI annotation whose visible state is driven by measured geometry, pose, contact, occlusion, and hand-correction rows. A row or graph-health decision is useful only if it changes the contact/geometry state a renderer consumes.

The first concrete slice is HOT3D clip001850 keyboard, right hand, frames 26-46 in `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`.

Current observation: the same hand/object relation is triple-valued. Interval MANO reports up to ~10.7 cm penetration in its dense observed-surface channel; contact/nonpenetration reports zero penetrating vertices against a non-watertight 95.4%-TRELLIS completed mesh whose signed query has zero/near-zero usable candidates; the published render reports a +39 mm gap. The rigid pose graph is inert (`nfev=1`, cost 0, no NP targets) while marking the state annotation-ready.

The previous interpretation, “wire the observed-surface penetration into contact/NP,” is now suspect. The strongest critique shows that the interval channel is not known to be keyboard surface: it uses full-frame UniDepth, with hand-owned quarantine off, object-mask gating off, surface eligibility off, and the depth-order-vs-object term selecting zero vertices. Therefore the interval penetration may be table/background/hand-depth leakage. It remains evidence of cross-solver/source inconsistency, not yet evidence of keyboard contact.

Live mechanisms now separate into two layers:

1. Source-validity mechanism: the dense observed-surface penetration survives keyboard masking and hand-depth quarantine, or it collapses as full-frame depth leakage. This decides whether the observed-surface channel can become a contact factor.
2. Geometry-epoch mechanism: the completed TRELLIS mesh is inflated and non-watertight, so its zero contact/NP result is a disabled query, not a physical no-contact measurement. It cannot be used as a contact body until free-space/observed-face repair exists.
3. Pose mechanism: frames 37-44 lie in a missing-pose/interpolated gap between visible graph fits, so completed-mesh distances must be stratified by pose provenance.
4. Graph-liveness mechanism: `annotation_ready=true` with inert objective and zero NP targets is a false state; contact/NP factors are not coupled.
5. Render-consumer mechanism: `ego.hoi` rows currently do not affect the published render path; a real R0 artifact needs a minimal consumer that makes contact_state/provenance visible on frames 32/36 before full-video work.

The graph-health route `cross_solver_geometry_decoupled` remains necessary, but it must be derived from measured provenance hashes and filter states: interval `depth_npz` hash plus mask/quarantine/eligibility flags, contact/NP mesh and sign-mesh hashes plus watertightness, render consumed mesh/state hash, and non-max summary statistics. Authored source-family strings and max-of-max penetration are not enough.

Next decisive artifact is KT-1/KT-2/KT-3/KT-5/KT-6/KT-7 on frames 28-48. The desired outcome is per-frame `ego.hoi` contact state: `contact_candidate_keyboard_masked` only where keyboard-masked/quarantined evidence and coherent vertices survive; otherwise `unresolved_full_frame_depth_leak`, `geometry_epoch_contaminated`, or `unresolved_pose_gap`, rendered visibly in an ego.hoi-consuming review for frames 32 and 36.
