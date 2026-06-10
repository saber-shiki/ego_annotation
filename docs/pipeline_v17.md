# Pipeline V17: Measurement-Grounded Multi-Object HOI Annotation

## Status

V16 closed only as full-duration packaging. It produced full-length videos for two raw clips, but it did not satisfy the original V3 joint graph requirement and its annotations do not meet the quality requirement. V17 treats every detector output as a measurement with residuals, confidence, and source evidence before the solver can accept an annotation state.

V17 implementation has produced the measurement store, full-state integration, a latent contact-mode graph, and sparse full-timeline evidence-consistency graphs. The measurement store reads V16 full-video outputs, prior HaWoR/WiLoR artifacts, HaMeR repairs, SAM2 object masks, contact-state rows, local deformable contact patches, and tomato persistent-shape state as measurements, then emits anchor QC before any graph solver can accept or repair them. The full-state integration writes full-length V17 evidence/QC JSONs, and the current render step verifies duration only. The contact-mode graph estimates per-frame per-hand contact/no-contact/unobserved modes from graph-corrected hand-object gaps, object-mask proximity, selected anchors, and temporal switch cost. The sparse geometry graph optimizes per-active-frame object translation corrections, per-active-frame small-angle object rotation corrections, and per-valid-hand camera-ray depth corrections against either selected anchor contacts or contact-mode factor-ready rows, object priors, pose smoothness, and hand-ray smoothness. Its contact equality terms use a local nearest MANO surface patch and report broader hand-object distances separately. It keeps camera trajectory, MANO articulation and shape, object mesh topology, and contact mode labels fixed, so it is an integrated consistency solver for the current V17 evidence layer. The complete nonlinear V3 joint solver remains open.

The current generated graph annotations still carry one manipulated object stream per frame. V17 now materializes the object roster as a separate multi-object mask-evidence timeline, but it is not an object-pose solution: the timeline carries SAM2 mask evidence for simultaneous objects and marks geometry and pose unresolved. Trash has four active VLM objects over 1,633 object-frame rows, with 1,604 visible-mask rows and 29 active rows without visible masks. Tomato has nine active VLM objects over 1,149 object-frame rows, with 1,109 visible-mask rows and 40 active rows without visible masks. The contact-mode-factor graph still outputs one legacy object stream, with 816 object-variable frames for trash and 670 for tomato. Any artifact whose schema exposes only `object` rather than simultaneous `objects` is a single-manipulated-object QC artifact.

V17 now also materializes multi-object visible-surface evidence under `/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces`. This layer fuses the multi-object SAM2 masks with the V16 UniDepth metric archives and the graph annotation camera poses, then writes object-aware world-surface mesh archives. The trash clip has 1,417 depth-backed visible-surface rows and 187 rejected visible-mask rows; rejections are 156 missing metric-depth frames, 15 rows with too few sampled vertices, and 16 rows with too little surface connectivity. The tomato clip has 694 depth-backed visible-surface rows and 415 rejected visible-mask rows; rejections are 172 missing metric-depth frames, 230 rows with too few sampled vertices, one row with too few valid masked depth pixels, and 12 rows with too little surface connectivity. Every row is marked `visible_surface_only_not_canonical_mesh`, `no_object_pose_variable`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`. These surfaces are measurement evidence for the future solver, not reconstructed object poses.

V17 also writes a multi-object geometry-state diagnostic under `/data2/ego_annotation_outputs/v17_multi_object_geometry_state`. This diagnostic measures center-normalized visible-surface envelope repeatability from the object-aware RGBD surfaces. Trash has four envelope candidates and tomato has two, but both clips have zero persistent visible-surface candidates and zero rigid-pose candidates. The distinction matters: a center-normalized envelope can repeat when a deformable bag, a symmetric object, or a depth artifact produces similar silhouettes, so it cannot supply material correspondence, canonical object geometry, or SE(3) pose.

V17 now exports object-track datasets under `/data2/ego_annotation_outputs/v17_object_track_datasets`. These datasets contain per-object RGB, mask, metric-depth PNGs, and intrinsics at the 960 by 540 depth resolution. They are inputs for material-correspondence tracking, not tracking results. Trash exports four object datasets with 1,448 total frames and 185 rejected frames. Tomato exports four usable object datasets with 936 total frames and 213 rejected frames; five tomato context objects have no exported tracking dataset because visible masks and metric depth do not overlap enough. The material-correspondence tracker and motion-state diagnostic now consume these datasets.

V17 now also has a first material-correspondence measurement layer under `/data2/ego_annotation_outputs/v17_object_material_tracks`. The tracker consumes the V17 object-track dataset contract directly: RGB, SAM2 object mask, depth PNG, per-frame depth intrinsics, and graph annotation camera pose. It does not read legacy single-object intrinsics fields. The material-track summary now contains ten attempted windows, one for every exported object dataset in both representative clips, and 203 rigid-ready adjacent pairs. Trash has five windows across four tracked objects and 135 rigid-ready adjacent pairs: black trash bag 0160-0219 has 26, off-white trash can 0182-0241 has 25, pink-lid trash can 0800-0859 has 44, white bag 0720-0779 has 30, and white bag 0800-0859 has 10. Tomato has five windows across four tracked objects and 68 rigid-ready adjacent pairs: early tomato 0274-0333 has zero from only four query points, later tomato 0690-0749 has 9, tomato peel 0720-0779 has 59, faucet handle 0906-0939 has zero, and the two-frame plastic-container window 0938-0939 has zero. The rigid diagnostic requires enough inliers, local spatial support, plausible centroid displacement, and plausible rotation angle. Several windows have local adjacent-pair factors but zero tracks surviving the whole window, so they provide short-term surface-motion evidence rather than a persistent object pose. These tracks are local correspondence evidence; they are not a full-timeline object pose state, do not reconstruct complete object geometry, and do not satisfy the V3 solver requirement.

V17 now writes material-motion state diagnostics under `/data2/ego_annotation_outputs/v17_object_material_motion_state`. This diagnostic composes adjacent material-track rigid factors and measures whether the chained transform still explains tracked material points at later frames. The 203 local rigid-ready adjacent pairs produce one persistent window-motion candidate: the pink-lid trash can has three short candidate segments inside frames 0811-0844, with 8 to 13 composed adjacent pairs and chained p95 residuals below 15 mm. The other four trash windows have only local adjacent material motion. Tomato has zero persistent window-motion candidates: the tomato and peel windows have local adjacent factors that do not satisfy the persistent-track and chained-residual predicates, while the early tomato, faucet-handle, and two-frame plastic-container windows have no ready material motion. The diagnostic converts local correspondence evidence into a stricter object-motion question, but it still does not create canonical object geometry, full-timeline SE(3), or deformation variables.

V17 also writes partial material-pose candidate diagnostics under `/data2/ego_annotation_outputs/v17_object_material_pose_candidates`. This layer fits a canonical material-point cloud and per-frame SE(3) transforms only for persistent material-motion segments. Trash has three ready partial pose segments, all on the pink-lid trash can: 0811-0820 with 160 persistent tracks and p95 residual 9.80 mm, 0822-0830 with 161 tracks and p95 residual 9.57 mm, and 0831-0844 with 152 tracks and p95 residual 9.27 mm. Tomato has zero partial pose candidates. These candidates are object-motion evidence over tracked surface points, not reconstructed object geometry: they do not include hidden surfaces, complete topology, full active intervals, contact-coupled physical state, or a delivered per-object pose timeline.

V17 now writes visible-surface replay diagnostics under `/data2/ego_annotation_outputs/v17_object_material_surface_replay`. This layer takes each partial material-pose candidate, transforms the visible surface from the segment's first frame, and compares it against observed object-aware RGBD visible surfaces at later frames. Two of the three trash-can partial pose segments pass this stricter surface replay test: 0822-0830 and 0831-0844 have symmetric visible-surface p95 residuals about 10.8 mm and 10.5 mm. The 0811-0820 segment fails because frames 0814, 0819, and 0820 have replay p95 residuals above 30 mm, despite low median residuals. Tomato still has zero replay candidates. Surface replay is observed-surface evidence only; it does not fill hidden geometry or establish complete object topology.

V17 now writes full multi-object contact evidence under `/data2/ego_annotation_outputs/v17_multi_object_contact_evidence`. This table contains two hand-side rows for every active multi-object timeline row: 5,564 rows total, with 3,266 rows for trash and 2,298 for tomato. It measures MANO-to-visible-surface distances where both hand geometry and object visible-surface geometry exist; 4,213 rows are measured and 1,351 rows are unobserved because hand geometry or visible-surface geometry is absent. This layer reports zero factor-ready contact rows. The zero is not a no-contact conclusion: accepted sparse-graph contacts at trash frames 0182 and 0856 use local contact-patch meshes, and other sparse contacts use legacy corrected object surfaces, while the new multi-object table measures raw object-mask RGBD visible surfaces. The table exposes the unmerged geometry sources that a real hand-object contact solver must reconcile.

V17 now writes a geometry-source audit under `/data2/ego_annotation_outputs/v17_geometry_source_audit`. This audit names which source owns each current geometry claim and checks whether contact factors can be interpreted against the multi-object visible-surface state. The result is incompatible with a closed object-pose solver: the current contact-mode graph has 463 factor-ready rows across the two clips, while the multi-object visible-surface contact table has zero factor-ready rows and zero same-frame visible-surface contact candidates for those ready contact rows. Trash has two accepted local contact-patch states, and both conflict with the multi-object visible-surface distances: frame 0182 has a local patch minimum distance of 2.47 mm while the black-bag visible-surface distance is 342 mm; frame 0856 has a local patch minimum distance of 0.97 mm while the white-bag visible-surface distance is 120 mm. The audit also records that the two passing material-surface replay segments are short observed-surface checks, not complete object geometry. The audit now consumes depth-contact consistency rows for accepted reconstruction meshes. The accepted short-segment pink-lid reconstructions pass rasterized front-surface depth QC against their rectified RGBD observations, while the current hand/contact graph uses a different depth state: 23 evaluated reconstruction frames and 46 hand rows produce zero near reconstructed-mesh hand rows, zero reconstructed-mesh contact candidates, zero shared-depth ready frames, and 23 depth-owner incompatibilities. This localizes the current blocker to depth/state ownership after mesh existence has been established.

V17 now writes a per-object geometry-hypothesis state under `/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state`. This state groups every active object by object id and compares its mask evidence, visible RGBD surface rows, persistent visible-surface shape measurements, object-depth repair candidates, local contact patches, material-motion windows, material-pose segments, accepted short-segment reconstructions, and source-compatibility audit rows. Across the 13 active objects in the two clips, zero objects have complete geometry, zero can own contact factors, and zero are pose-factor-ready. Trash has two local-contact-patch conflict objects, one partial short-segment hidden-topology reconstruction object, and one visible-surface-only object. Tomato has five mask-only objects, one partial persistent visible-surface object, and three visible-surface-only objects. This table is the current object-geometry state owner for V17: a future solver must promote one of these hypotheses by adding the missing geometry/topology/pose variables, not by treating a contact patch or visible mask as object pose.

V17 now writes an object-geometry factor problem under `/data2/ego_annotation_outputs/v17_object_geometry_factor_problem`. This artifact converts every active object into explicit variable blocks and factor blocks for a future object solver: canonical mesh or SDF geometry, per-active-frame SE(3) pose, topology or deformation deltas, contact attachment state, visible-surface residuals, material-correspondence rigidity residuals, material-motion segments, partial material-pose priors, visible-surface replay checks, unknown-object RGBD reconstruction jobs, evaluated reconstruction results, depth-contact consistency, multi-object hand-contact distances, object-depth repair validation, and geometry-source compatibility. Across the two clips it materializes 13 object rows, 2,111 visible-surface factor rows, 203 material-rigidity pair factors, three partial material-pose segments, two visible-surface replay-ready segments, two solver-ready RGBD reconstruction jobs, two detected BundleSDF outputs, two scale-plausible recovered metric meshes, two projection/depth-passing short-segment reconstructions, two accepted short-segment hidden-topology reconstructions, 23 depth-contact evaluated frames, 46 evaluated hand rows, zero near reconstructed-mesh hand rows, zero reconstructed-mesh contact candidates, zero shared-depth ready frames, 23 depth-owner incompatibilities, zero contact-ready rows against multi-object geometry, and zero solve-activation-ready objects. This matches the object-level structure used by unknown-object RGBD tracking and reconstruction methods such as BundleSDF: object pose and object geometry must be solved together from RGBD/mask observations, while contact factors can enter only after they attach to the same mesh or SDF state.

V17 now writes observed-surface geometry seeds under `/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed`. This builder takes replay-passing material-pose segments, maps each segment's visible RGBD surfaces back into the material-pose canonical frame, and stores canonical observed-surface mesh archives. The current output has two pink-lid trash-can seed segments: frames 0822-0830 with 11,478 vertices and 21,009 faces, and frames 0831-0844 with 17,618 vertices and 32,390 faces. Their canonicalized surface centroids stay within about 3-4 mm median of the segment source surface, so the material-pose transforms are geometrically coherent for the observed surface. The seeds are still short-window observed-surface geometry: they do not reconstruct hidden topology, do not cover the full active interval, and cannot own contact factors. The object-hypothesis state therefore changes one trash object from partial pose segments to a partial observed-surface geometry seed while preserving `object_geometry_complete=false`, `object_pose_requirement_met=false`, and `v3_solver_complete=false`.

V17 now writes unknown-object RGBD reconstruction jobs under `/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs`. This layer takes the observed-surface seed windows, rectifies their per-frame metric RGBD observations into a constant-intrinsics 1,920 by 1,080 camera contract, and measures whether the rectified depth preserves the original object rays before any hidden-topology solver is allowed to consume the folder. The current output has two solver-ready pink-lid trash-can jobs, both from the two observed-surface seed segments. The rectification p95 residual over 23 job frames is about 0.00027 m, and the projected-inside fraction is at least 0.9967. These jobs are prepared solver inputs; reconstructed hidden surfaces, full active-interval meshes, and contact-compatible object geometry enter only through evaluated backend outputs and later object-geometry factors.

V17 now writes evaluated reconstruction-result reports under `/data2/ego_annotation_outputs/v17_geometry_reconstruction_results`. This layer consumes the prepared RGBD jobs and a BundleSDF output root, then checks whether each solver output has a mesh, a complete per-frame object-in-camera pose sequence, a mesh scale compatible with the rectified RGBD observations, plausible mesh topology, and projection/depth agreement against the job masks and depth maps. Missing backend output is recorded as `pending_solver_output`; a bad mesh is recorded as rejected evidence. The current local report has two detected BundleSDF outputs and two accepted short-segment reconstructions. Both backend runs produced complete pose sequences and `mesh_cleaned.obj`; source inspection shows `mesh_cleaned.obj` is exported in normalized NeRF coordinates before BundleSDF's `mesh_to_real_world` step. The texture path is nonessential for geometry QC and can hang or crash after the mesh exists, so the remote runner now preserves stable `mesh_cleaned.obj` evidence and terminates that path explicitly. V17 recovers a metric QC mesh through BundleSDF's persisted `sc_factor` and translation, records that coordinate contract, and runs scale, topology, projection, and rasterized front-surface depth tests over all projected mesh faces. The all-face condition matters: an earlier 120k-face cap skipped nearer triangles and created a false centimeter-scale front-surface depth tail. Under all-face QC, the recovered metric meshes pass scale, topology, mask projection, and front-surface depth checks. The 0822-0830 segment has median front-surface depth error 2.7 mm and median frame p95 13.7 mm; the 0831-0844 segment has median front-surface depth error 1.6 mm and median frame p95 6.8 mm. All-frame BundleSDF under `/data2/ego_annotation_outputs/v17_bundlesdf_outputs_allframes` also passes both segments and is slightly better on the first segment, with median frame p95 12.0 mm and 6.5 mm. The depth-weighted all-frame run under `/data2/ego_annotation_outputs/v17_bundlesdf_outputs_allframes_depth100` also passes; its second-segment median frame p95 is 8.3 mm, higher than the all-frame depth0 value of 6.5 mm. These accepted meshes cover only short pink-lid trash-can windows. Full active-interval geometry, simultaneous multi-object pose, and contact-factor ownership in one unified object-geometry state remain unresolved.

V17 now writes a depth-contact consistency audit under `/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit`. This layer places accepted BundleSDF meshes into graph world coordinates with their `ob_in_cam` poses, compares them to graph-corrected MANO vertices, and compares visible UniDepth, reconstructed mesh camera depth, accepted front-surface depth QC, legacy object-center depth, hand source-camera depth, and contact-mode gaps. The accepted trash windows have visible object depth median 0.386 m, reconstructed full-mesh camera-depth median 0.462 m, and rasterized front-surface depth p95 values already accepted by reconstruction QC, while the current MANO source-camera depth and legacy object depth sit near 1.3-1.8 m. The report records 23/23 depth-owner incompatibility frames, zero near reconstructed-mesh hand rows, zero reconstructed-mesh contact candidates, and zero shared-depth ready frames. The accepted mesh is geometry evidence; contact factors still require a joint camera, hand, object, and depth state.

V17 also now writes a joint solver problem contract under `/data2/ego_annotation_outputs/v17_joint_solver_problem`. That contract is a generated gap audit, not an optimizer. It binds the V3 requirement to concrete variable families and reports the still-missing families: camera trajectory, MANO articulation and shape, multi-object identity timeline, object geometry topology, object pose or deformation state, hand-object contact modes for every object pair, contact patch identity, dense depth and visible-surface state, and physical consistency terms. The contract now requires the multi-object visible-surface, geometry-state, object-track-dataset, object-material-track, object-material-motion-state, object-material-pose-candidate, object-material-surface-replay, multi-object-contact-evidence, geometry-source-audit, object-geometry-hypothesis-state, object-geometry-factor-problem, geometry-reconstruction-job, geometry-reconstruction-result, and depth-contact consistency reports and records their measured, rejected, envelope-candidate, rigid-candidate, exported tracking-input, material-track-window, rigid-factor, local-motion, persistent-motion-candidate, partial-pose, surface-replay, hand-object distance, source-incompatibility, object-hypothesis, object-factor, solver-job, evaluated solver-output, depth-owner incompatibility, and shared-depth readiness rows inside the object-geometry, object-pose, contact, dense-depth, and physical-consistency variable families. The contract makes the liability explicit: the sparse graph optimizes a small consistency layer over fixed central state variables, and the current object/contact sources do not yet define one unified object-geometry state.

V17 also corrects a version-accounting problem. V3 already identified the core requirement: solve or expose the metric contradiction between MANO hands and object geometry through a joint factor graph. Later versions implemented real component graphs, including object-pose, sparse object-track, and contact-dynamics graphs. Their scope stayed at selected windows or selected state variables. V16 then closed as a full-length delivery artifact with QC flags while the original joint graph requirement remained open. V17 therefore treats prior graph outputs as evidence modules and reports the current sparse full-timeline graph separately from the still-unimplemented complete nonlinear V3 solver.

The current measurement-store implementation is the evidence layer for V17 state estimation. Model outputs remain traceable measurements with confidence, residual, source, and failure fields; missing hands, missing objects, missing contact states, and incomplete HaWoR/WiLoR coverage become explicit QC failures.

The current HaWoR evidence path uses a compact full-video adapter input generated from V16 annotations. The compact file preserves frame indices, timestamps, source camera transforms, source intrinsics, measured V16 hand 2D keypoints, detector scores, and hand boxes, then reruns the HaWoR camera-local adapter against the full 0-1049 HaWoR NPZ. The adapter input therefore contains only the fields read by the HaWoR residual calculation.

HaWoR rows without current-frame observed hand support are stored as `hawor_motion_infill_candidate` measurements. They can support an occluded or detector-miss state after temporal, projection-contradiction, contact, and nonpenetration checks. Current-frame 2D evidence is required for observed-visible measurements, not for motion-infill candidates.

The trash hand-evidence path now includes VLM-localized visible hand boxes for anchor frames where RTMLib or full-frame HaMeR crops were broad, missing, or attached to the wrong region. The VLM boxes enter the measurement store as image-localization evidence and HaMeR crop inputs. Synthetic keypoints derived from those boxes have no metric meaning; the usable 3D evidence is the HaMeR MANO output and its source-camera reprojection residual.

The trash anchor repair path now materializes selected VLM-box HaMeR hypotheses as V17 repair hand states for frames 0182, 0260, 0764, 0856, 0949, and 0970. The measurement store also ingests contact evidence recomputed from those repaired states. This clears the hand-state blockers at the named trash anchors.

Frames 0182 and 0856 exposed the wrong object variable. Whole-object depth re-anchoring can force a hand-bag metric contact, but temporal validation rejects that variable because it shifts the entire deformable bag surface. V17 now records separate local deformable contact-patch meshes for those frames. Frame 0182 uses a 206-vertex, 302-face black-bag patch centered at the repaired right-hand contact support; the patch occupies 1.43 percent of the SAM2 bag mask and has a 2.47 mm nearest hand-surface distance. Frame 0856 uses a 638-vertex, 1,086-face white-bag patch; the patch occupies 12.77 percent of the SAM2 bag mask and has a 0.97 mm nearest hand-surface distance. The anchor contact graph v4 selects those local patch rows for 0182 and 0856, accepts contact at 0182, 0260, 0764, and 0856, and accepts no-contact at 0949 and 0970. The rejected whole-object depth candidates remain in the measurement store as failed evidence, while the accepted state is the local deformable surface geometry.

The tomato measurement path now has a persistent visible-surface mesh state for `object:obj_tomato`. The solver fuses SAM2 object masks with V16 metric-depth surface extraction over the full active mask interval, rejects 20 temporal surface-scale outliers, and writes a canonical mesh with 84,318 vertices and 136,902 faces. The robust 1-99 percent canonical extents are 9.85 cm, 9.67 cm, and 9.56 cm; raw min-max extent is reported separately because sparse tails can overstate object scale. Tomato anchors 0480, 0720, and 0760 pass persistent-shape QC with surface-to-canonical p95 residuals of 1.7 mm, 5.6 mm, and 1.3 mm. The full-state builder translates the canonical-local visible surfaces by their per-frame `object_center_world_m` before saving the V17 mesh archive, so the archive carries world-coordinate meshes. The tomato contact graph v1 selects left-hand contact rows at 0480, 0720, and 0760.

The current measurement store passes the named trash and tomato anchors. The evidence-layer full-state integration also renders full raw-video outputs for both representative clips: trash has 1,050 raw frames and 1,050 frames in overlay, world, and side-by-side renders; tomato has 960 raw frames and 960 frames in overlay, world, and side-by-side renders. The anchor-only sparse evidence-consistency graph remains a limited QC branch. Trash uses four selected contact factors at 0182, 0260, 0764, and 0856; after optimization its local contact p95-of-p95 is 4.56 mm, broader 80-nearest-point p95-of-p95 is 11.94 mm, max object translation correction is 6.89 mm, max object rotation correction is 0.0259 rad, and max hand camera-ray correction is 3.18 mm. Tomato uses three selected contact factors at 0480, 0720, and 0760; after optimization its local contact p95-of-p95 is 6.08 mm, broader 80-nearest-point p95-of-p95 is 22.44 mm, max object translation correction is 7.62 mm, max object rotation correction is 0.0105 rad, and max hand camera-ray correction is 10.69 mm. The tomato anchor-only branch therefore remains partial even under the sparse evidence-consistency metric. The graph uses local nearest MANO patches for the current contact correspondence selection; a previous 160-point stochastic hand subsample made tomato frame 0480 fail at 36.1 mm, which exposed a sampling artifact in the sparse contact linearization.

The contact-mode graph is implemented as a V17 contact-state layer; the full V3 solver remains open. Its default manifests read the anchor-only sparse graph outputs, so it estimates contact modes from fixed graph-corrected geometry while hand geometry, object geometry, object pose, and contact labels remain fixed inputs. It accepts both representative clips with zero anchor contradictions and explicit `v3_solver_complete=false` metadata. Trash has 2,100 hand-side rows, 1,625 active geometry observations, 473 unobserved rows, 172 contact-mode rows, and 81 contact-factor-ready rows after requiring positive contact evidence, explicit mask-distance evidence, and sparse-graph hand residual compatibility. Tomato has 1,920 hand-side rows, 1,340 active observations, 580 unobserved rows, 527 contact-mode rows, and 382 contact-factor-ready rows. Contact-mode reports now store row-level `contact_factor_readiness_checks`, including hand residual values, mask-distance availability, and threshold predicates, so readiness can be audited from the JSON. Contact-mode interval sheets are regenerated from solved rows with visible QC banners and report per-side timelines, counts, interval previews, and readiness state; they are contact-state QC summaries, and visual-quality rendering remains open. The long tomato left-hand interval from frame 0714 to 0939 matches hand-held washing/rinsing behavior in the graph-corrected side-by-side video, while the trash contact bursts match bag/trash-can manipulation. Rows without hand/object geometry are marked `unobserved` and cannot inherit temporal contact. Rows whose unary evidence opposes contact can remain temporally labeled as contact, but they cannot become geometry factors.

The contact-mode-factor sparse graph consumes only accepted contact-mode `contact_factor_ready` rows through an explicit `--contact-mode-graph-root` input. It structurally accepts both representative clips with converged local contact-patch correspondences. The modeled contact patch now uses 16 nearest MANO surface vertices. A support-size sweep showed why the broader metric cannot be the local contact predicate: trash/tomato p95-of-p95 stays below 5 mm through 18 nearest vertices, then rises to 12.84/13.22 mm at 80 vertices because non-contact hand surface is being included. The 80-nearest-point report remains a support-size sensitivity diagnostic, not a requirement that the whole nearby hand surface touch the object. Trash uses all 81 corrected contact-mode factors, with 1,296 linearized local-patch correspondences, contact-patch p95-of-p95 4.75 mm, broader 80-nearest-point p95-of-p95 12.84 mm, max object translation correction 6.32 mm, max object rotation correction 0.0295 rad, and max hand ray correction 3.36 mm. Tomato uses all 382 corrected contact-mode factors, with 6,112 local-patch correspondences, contact-patch p95-of-p95 5.00 mm, broader 80-nearest-point p95-of-p95 13.22 mm, max object translation correction 7.15 mm, max object rotation correction 0.0294 rad, and max hand ray correction 12.21 mm. The 5 mm numbers are local evidence-consistency diagnostics under the sparse graph's fixed camera, fixed MANO articulation, fixed object topology, fixed contact-label, and nearest-vertex support assumptions. They do not prove a physically valid contact patch until patch identity, image/depth support, and anatomical/contact-area stability are also estimated. The generated reports set `accuracy_target_met=false`, `annotation_ready=false`, and `deliverable_ready=false` because the complete V3-class nonlinear solver remains open.

Two contact-support experiments were rejected before this interpretation was adopted. A single contiguous MANO patch made the graph nonconvergent and raised p95-of-p95 to 15.8 mm for trash and 10.3 mm for tomato. A small multi-site anatomical support model remained nonconvergent at 8.77 mm and 8.07 mm while increasing runtime. These negative results show that support selection alone is not the remaining mechanism; full MANO articulation/surface fitting and object geometry remain fixed in the current sparse graph.

The graph output materializes the optimized state in the annotation files. Object translation and small-angle rotation corrections move `center_world_m`, nested V17 surface centers, local-patch world vertex arrays when present, and the corrected mesh archive around the solved object center. Hand camera-ray corrections move world-space MANO vertices and joints along the solved camera optical axis. Source-camera measurements remain unchanged as evidence. Corrected annotation JSONs now expose `annotation_ready=false`, `deliverable_ready=false`, `v3_solver_complete=false`, `multi_object_timeline_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false` at the root, in the graph metadata, and on every legacy `object` record. Corrected mesh NPZ archives also embed `v17_archive_metadata_json` and write an adjacent `.metadata.json` sidecar with the same object-geometry and multi-object limitations, so the mesh arrays do not stand alone as complete object-pose evidence.

Graph-corrected full-length renders pass duration QC only. Trash graph renders contain 1,050 frames for overlay, world, and side-by-side outputs; tomato graph renders contain 960 frames for all three outputs. The visual inspection sheets are sampled from the final graph-corrected side-by-side videos and recorded in the render summary. The current trash sheet shows selected contact frames 0182, 0260, 0764, and 0856 with displayed nearest gaps of 0.5 mm, 1.7 mm, 2.0 mm, and 0.9 mm. The current tomato sheet shows selected contact frames 0480, 0720, and 0760 with displayed nearest gaps of 2.4 mm, 0.5 mm, and 2.3 mm. The regenerated contact-mode-factor renders also pass duration QC: trash contains 1,050 frames for overlay, world, and side-by-side outputs; tomato contains 960 frames for all three outputs. New QC renders use `qc_` filenames and their render summary carries `render_qc_scope=duration_only_not_visual_quality`, `visual_quality_qc_pass=false`, `stage9_visual_deliverable_ready=false`, `annotation_ready=false`, `deliverable_ready=false`, and `accuracy_target_met=false`. These videos are duration evidence/QC renders rather than V17 closure deliverables. The current world videos and sheets still use the diagnostic V16/V17 renderer; the Stage 9 audience renderer with shaded MANO/object surfaces, image-plane context, close-up manipulation views, and uncertainty/rejection status remains open.

Current V17 evidence outputs:

```text
/data2/ego_annotation_outputs/v17_full_state/
/data2/ego_annotation_outputs/v17_full_timeline_factor_graph/
/data2/ego_annotation_outputs/v17_full_timeline_factor_graph_renders/
/data2/ego_annotation_outputs/v17_multi_object_timeline/v17_multi_object_timeline_summary.json
/data2/ego_annotation_outputs/v17_multi_object_timeline/{trash_1050,task5_tomato_960}/v17_multi_object_timeline.json
/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/v17_multi_object_visible_surface_summary.json
/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/{trash_1050,task5_tomato_960}/v17_multi_object_visible_surface_report.json
/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/{trash_1050,task5_tomato_960}/multi_object_visible_surfaces_world.npz
/data2/ego_annotation_outputs/v17_multi_object_geometry_state/v17_multi_object_geometry_state_summary.json
/data2/ego_annotation_outputs/v17_multi_object_geometry_state/{trash_1050,task5_tomato_960}/v17_multi_object_geometry_state_report.json
/data2/ego_annotation_outputs/v17_multi_object_geometry_state/{trash_1050,task5_tomato_960}/multi_object_center_normalized_visible_surface_points.npz
/data2/ego_annotation_outputs/v17_object_track_datasets/v17_object_track_dataset_summary.json
/data2/ego_annotation_outputs/v17_object_track_datasets/{trash_1050,task5_tomato_960}/v17_object_track_dataset_summary.json
/data2/ego_annotation_outputs/v17_object_track_datasets/{trash_1050,task5_tomato_960}/{track_id}/manifest.json
/data2/ego_annotation_outputs/v17_object_material_tracks/v17_object_material_track_summary.json
/data2/ego_annotation_outputs/v17_object_material_tracks/{trash_1050,task5_tomato_960}/v17_object_material_track_summary.json
/data2/ego_annotation_outputs/v17_object_material_tracks/{trash_1050,task5_tomato_960}/{window_id}/v17_object_material_track_report.json
/data2/ego_annotation_outputs/v17_object_material_tracks/{trash_1050,task5_tomato_960}/{window_id}/rigid_pair_factors.json
/data2/ego_annotation_outputs/v17_object_material_motion_state/v17_object_material_motion_state_summary.json
/data2/ego_annotation_outputs/v17_object_material_motion_state/{trash_1050,task5_tomato_960}/v17_object_material_motion_state_report.json
/data2/ego_annotation_outputs/v17_object_material_pose_candidates/v17_object_material_pose_candidate_summary.json
/data2/ego_annotation_outputs/v17_object_material_pose_candidates/{trash_1050,task5_tomato_960}/v17_object_material_pose_candidate_report.json
/data2/ego_annotation_outputs/v17_object_material_surface_replay/v17_object_material_surface_replay_summary.json
/data2/ego_annotation_outputs/v17_object_material_surface_replay/{trash_1050,task5_tomato_960}/v17_object_material_surface_replay_report.json
/data2/ego_annotation_outputs/v17_multi_object_contact_evidence/v17_multi_object_contact_evidence_summary.json
/data2/ego_annotation_outputs/v17_multi_object_contact_evidence/{trash_1050,task5_tomato_960}/v17_multi_object_contact_evidence_report.json
/data2/ego_annotation_outputs/v17_geometry_source_audit/v17_geometry_source_audit_summary.json
/data2/ego_annotation_outputs/v17_geometry_source_audit/{trash_1050,task5_tomato_960}/v17_geometry_source_audit_report.json
/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state/v17_object_geometry_hypothesis_state_summary.json
/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state/{trash_1050,task5_tomato_960}/v17_object_geometry_hypothesis_state_report.json
/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed/v17_observed_surface_geometry_seed_summary.json
/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed/{trash_1050,task5_tomato_960}/v17_observed_surface_geometry_seed_report.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/v17_geometry_reconstruction_jobs_summary.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/{trash_1050,task5_tomato_960}/v17_geometry_reconstruction_jobs_report.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/trash_1050/{job_id}/v17_geometry_reconstruction_job.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/trash_1050/{job_id}/{rgb,depth,masks}/
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/trash_1050/{job_id}/cam_K.txt
/data2/ego_annotation_outputs/v17_geometry_reconstruction_results/v17_geometry_reconstruction_results_summary.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_results/{trash_1050,task5_tomato_960}/v17_geometry_reconstruction_results_report.json
/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit/v17_depth_contact_consistency_audit_summary.json
/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit/{trash_1050,task5_tomato_960}/v17_depth_contact_consistency_audit_report.json
/data2/ego_annotation_outputs/v17_object_geometry_factor_problem/v17_object_geometry_factor_problem_summary.json
/data2/ego_annotation_outputs/v17_object_geometry_factor_problem/{trash_1050,task5_tomato_960}/v17_object_geometry_factor_problem.json
/data2/ego_annotation_outputs/v17_joint_solver_problem/v17_joint_solver_problem_summary.json
/data2/ego_annotation_outputs/v17_joint_solver_problem/{trash_1050,task5_tomato_960}/v17_joint_solver_problem.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/trash_1050/v17_contact_mode_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/trash_1050/contact_mode_interval_review_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_graph/task5_tomato_960/v17_contact_mode_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/task5_tomato_960/contact_mode_interval_review_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_graph/v17_contact_mode_graph_summary.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/v17_full_timeline_factor_graph_summary.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/v17_full_timeline_factor_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/annotations_v17_full_timeline_graph.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/object_meshes_v17_full_timeline_graph.npz
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_overlay_mano_object_multi.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_world_reconstruction_3d_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_side_by_side_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/v17_contact_mode_factor_side_by_side_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_overlay_mano_object_multi.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_world_reconstruction_3d_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_side_by_side_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/v17_contact_mode_factor_side_by_side_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/v17_render_summary.json
```

The first three roots above are legacy V17 evidence/QC outputs whose filenames predate the QC naming correction. The current contact-mode-factor render manifests use `qc_` render filenames.

## V16 Failure Analysis

The failures are not isolated rendering bugs.

Trash frame 0182:
- The right hand is visible in the raw image, but the delivered MANO state drifts away from it.
- The object mesh covers disconnected or incomplete bag/can regions.
- The final annotation record contains hand boxes but no retained confidence, source, or residual fields, so a bad hand measurement looks like an accepted hand state.

Trash frame 0260:
- Both rendered hands are attached to the wrong image region near the frame edge.
- This is a single-frame hand hypothesis failure under truncation and occlusion.
- The temporal state did not reject the hypothesis against mask, keypoint, depth, or motion evidence.

Trash frame 0764:
- The visible hand and bag are in contact, but the V16 manipulation view reports them as separated.
- Nearest hand-mesh distance is an insufficient contact model. Contact must be a latent state supported by image adjacency, depth ordering, 3D gap, surface motion, and temporal consistency.

Trash frame 0856:
- The object mask/mesh and both MANO hands are visibly wrong.
- The contact label becomes numerically small even though the underlying hand/object states are bad. A contact residual cannot rescue invalid state estimates.

Trash frame 0949:
- The hand annotation is wrong even without requiring object reasoning.
- V16 lacks an independent hand-state rejection path.

Trash frame 0970:
- A hand region is visible in the raw frame, but the delivered hand list is empty.
- V16 drops the state when measurements disappear; it does not maintain a prediction/update filter with uncertainty through occlusion or detector failure.

Tomato frames around 0480 to 0760:
- The tomato mesh changes shape frame to frame because V16 reconstructs a visible depth patch each frame.
- The tomato requires a persistent near-rigid canonical shape with per-frame pose and small allowed deformation.
- Bowls, plates, tray, sink, and containers can participate in manipulation context, but V16 only carries one object stream.

World reconstruction:
- Skeleton-only hands do not visually read as hands.
- The camera frustum and trajectory do not explain the camera-hand-object relation.
- The 3D panel looks like a diagnostic plot. V17 must render shaded MANO surfaces, shaded object meshes, the current camera image plane/frustum, a local manipulation close-up, and a stable world/camera relationship so the view reads as a reconstruction.

## Root Causes

1. V16 promoted measurements into annotations.

WiLoR hands, object masks, and depth-derived meshes enter the delivered timeline without a strong measurement residual contract. The final JSON often lacks score/source/residual fields for active hand states. The renderer cannot distinguish accepted state, low-confidence measurement, prediction, or rejection.

2. V16 is single-object.

The user-facing task is object pose annotation for manipulated objects. A clip can involve several manipulated or contact-relevant objects. V17 must keep a multi-object timeline where each object has identity, active interval, role, geometry state, and evidence status.

3. V16 has no persistent object shape state.

A tomato, bowl, plate, lid, and trash can are not independent depth patches per frame. Near-rigid objects need a canonical mesh plus per-frame pose. Deformable objects need a canonical or reference surface plus deformation state and temporal regularization.

4. V16 does not implement occlusion smoothing.

Missing hands or objects become predicted states with uncertainty. Bad measurements are rejected or downweighted by image, mask, keypoint, depth, and motion residuals.

5. V16 contact is a diagnostic distance, not contact reasoning.

Contact requires a state variable. A small nearest distance can be false when hand/object states are bad; a large distance can be false when depth or pose is wrong. V17 must estimate contact jointly with hand and object state.

6. V16's 3D view is not designed as a visual explanation.

The 3D panel must show what the annotation means. A line plot with labels cannot satisfy the V17 rendering contract.

## Research Conclusions

The current literature supports a measurement-and-state design over a single replacement model.

HaWoR is directly relevant because it targets egocentric world-space hand motion, combines camera trajectory estimation with hand reconstruction, and includes a motion infiller for missing frames. Its official repository also depends on masked DROID-SLAM and Metric3D, which matches the failure mode of moving egocentric cameras and missing hands. Source: https://github.com/ThunderVVV/HaWoR and https://arxiv.org/abs/2501.02973.

The previous HaWoR branch did not prove HaWoR wrong. It proved the integration was incomplete: the raw camera-local HaWoR hands were partly plausible, while the tested bridge into the existing DROID/object world used a global Sim(3) alignment that produced severe reprojection and hand-scale errors. V17 therefore treats HaWoR as a primary measurement source for world-space hand motion and missing-frame infilling, while making the coordinate bridge itself a residual-checked graph variable.

SAM 2 supports promptable video segmentation and mask propagation, including multi-object video tracking support in the official repository. V17 uses it as one segmentation measurement source; the graph estimates the final object state from masks, depth, tracks, geometry, and contact evidence. Source: https://github.com/facebookresearch/sam2 and https://arxiv.org/abs/2408.00714.

VGGT predicts camera parameters, depth maps, point maps, and 3D point tracks from multiple views. V17 uses it as a geometry and track source for camera/object consistency checks, especially when DROID or monocular depth is unstable. Source: https://github.com/facebookresearch/vggt and https://arxiv.org/abs/2503.11651.

FoundationPose and BundleSDF are relevant for rigid or near-rigid object pose and reconstruction. FoundationPose handles model-based and model-free 6D pose estimation/tracking for novel objects; BundleSDF reconstructs and tracks unknown rigid objects from RGBD sequences with pose-graph optimization. These methods are not suitable as a universal solution for deformable bags, but they are the right class of method for tomato, bowl, plate, lid, can, and container-like objects after masks and depth are available. Sources: https://github.com/NVlabs/FoundationPose, https://arxiv.org/abs/2312.08344, https://github.com/NVlabs/BundleSDF, and https://arxiv.org/abs/2303.14158.

TRELLIS, Hunyuan3D, and PartCrafter can propose complete meshes or structured object priors from images. V17 may use them as prior proposal sources, but any generated mesh must pass visible replay, temporal track support, and physical consistency before entering the delivered annotation. Sources: https://github.com/microsoft/TRELLIS, https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1, and https://github.com/wgsxm/PartCrafter.

WHOLE and EgoGrasp point toward the correct formulation: world-space hand-object interaction reconstruction must model hands and objects jointly over time, especially under occlusion and object entries/exits. V17 uses this formulation as the missing target. Current V17 artifacts provide measurement layers, contact-mode QC, and a sparse evidence-consistency graph, while the full joint hand-object-camera-depth-contact solver remains open. Sources: https://arxiv.org/abs/2602.22209 and https://arxiv.org/abs/2601.01050.

## V17 Pipeline Definition

### Stage 0: Representative Inputs

V17 will run on at least the same two full raw clips as V16:

- task5 tomato, 960 frames;
- trash, 1050 frames.

The named failure frames become required QC anchors for trash: 0182, 0260, 0764, 0856, 0949, 0970. Tomato must include frames around 0480, 0720, and 0760, plus frames where bowls/plates/trays are visible.

### Stage 1: Measurement Store

Every model output enters as a measurement record. The graph solver creates annotation states from residual-checked measurements.

Each measurement stores:

```text
frame_idx
entity_id
measurement_type
source_model
source_checkpoint
coordinate_frame
value
confidence
covariance_or_scale
residuals_against_other_sources
visibility_state
failure_reason
```

No hand, object, camera, or contact state can enter the delivered annotation unless its source measurements and residuals remain traceable.

### Stage 2: Camera And Metric Geometry

V17 will estimate camera/geometry with at least two independent sources:

- masked DROID/HaWoR-style egocentric SLAM for full trajectory continuity;
- VGGT for camera, depth, point maps, and tracks on selected windows;
- UniDepth or Metric3D as dense metric depth measurement.

Dynamic hand/object masks are excluded from static-background SLAM where the method supports masking. The camera state stores uncertainty and residual spikes. The renderer must project accepted hand/object states back through the same camera model, so an impossible camera-hand relation becomes visible in QC.

### Stage 3: Multi-Object Plan

V17 replaces the single object stream with an object roster.

A VLM/video-review agent proposes:

```text
object_id
natural_language_name
active_intervals
role_distribution
prompt_points_or_boxes
expected_physical_behavior
```

Roles are data, not code branches. Examples of roles are manipulated object, support object, container, tool, target surface, and distractor. The downstream pipeline treats all object records uniformly.

Open-vocabulary detection plus SAM 2/Cutie-style video segmentation produces masks for every active object. Object identities are checked by temporal mask propagation, point tracks, depth, and VLM review. A clip can deliver multiple object meshes and poses.

The current V17 multi-object timeline is the first concrete state owner for this stage. It consumes the measurement-store object roster and SAM2 per-object track JSONs, verifies local mask paths, and emits one full-frame `objects` array per raw frame. Each object state contains mask evidence, active interval, role notes, and explicit unresolved geometry/pose fields. It intentionally sets `object_geometry_complete=false`, `object_pose_requirement_met=false`, and `annotation_ready=false` until mesh reconstruction and pose/deformation variables are added.

### Stage 4: Hand Measurement And State Estimation

V17 uses a fixed measurement set:

- HaWoR world-space hand motion and infilled hand trajectory measurements as the primary temporal hand-motion source;
- WiLoR per-frame MANO measurements as an independent image-conditioned hand source;
- HaMeR per-frame MANO measurements from RTMLib crop evidence, with explicit source-coordinate intrinsics and metric-translation residuals;
- VLM-localized visible hand boxes for detector-miss or bad-crop anchors, used as HaMeR crop localization and as image-level hand evidence;
- RTMLib 2D keypoints;
- SAM 2 hand masks or another hand-mask source;
- metric depth over visible hand regions;
- learned hand-motion and hand-object interaction priors.

The delivered hand state is a fixed-lag smoothed MANO trajectory. The solver can choose among measurements because the objective defines residuals, not because code branches on visual cases.

HaWoR contributes motion continuity, world-space trajectory, and missing-frame infill. WiLoR and HaMeR contribute per-frame MANO image evidence through separate model families and crop contracts. The accepted hand state is the graph solution that best satisfies projection, mask, depth, temporal, and contact residuals. When hand sources disagree, V17 records the disagreement and either repairs the state or marks it unresolved.

For each hand and frame, the state can be:

```text
measured_accepted
measured_repaired
predicted_occluded
predicted_detector_miss
rejected_unresolved
outside_view
```

A visible hand cannot silently disappear. If all hand reconstructor measurements fail but the raw image/mask/keypoint evidence shows a hand, V17 emits a predicted state with high uncertainty and a QC flag, then tries local repair from neighboring frames.

### Stage 5: Object Geometry State

Each object has a canonical geometry state and a per-frame state.

Canonical geometry can come from:

- visible RGBD/depth fusion over accepted masks;
- BundleSDF-style rigid object reconstruction when the rigidity residual supports it;
- FoundationPose tracking when a reconstructed/generated mesh is accepted;
- TRELLIS/Hunyuan3D/PartCrafter mesh priors after replay acceptance.

Per-frame object state contains:

```text
T_wo_t        pose for rigid or near-rigid component
D_t           deformation field or surface offsets when needed
M_t           mask support
P_t           point-track support
Q_t           geometry and replay residuals
```

Rigidity is inferred from evidence. If a surface track set can be explained by one SE(3) transform with small residual, the object is near-rigid. If it cannot, the object uses a deformable surface state with smoothness and local area regularization. This is a physical distinction, not category branching.

The current visible-surface builder is the first multi-object RGBD measurement layer for this stage. It treats every visible object uniformly: resize the verified SAM2 mask to the metric-depth raster, keep finite metric depth inside the mask, backproject with the depth archive intrinsics, transform through the graph annotation camera pose, and save an object-frame surface mesh with object id and frame index. It rejects rows with missing depth, too few valid depth pixels, too few sampled vertices, or disconnected surface support. This produces world-coordinate visible surface evidence, but it does not estimate `G_o`, `T_wo,t`, or `D_o,t`; the V3 object-geometry and object-pose families remain unmet.

The current geometry-state diagnostic splits each object's center-normalized surfaces into even and odd frame partitions and measures cross-partition nearest-surface residuals. It is a measurement of envelope repeatability after subtracting the per-frame surface center. It is not a rigidity proof because it ignores material correspondences and removes translation by construction. Current reports therefore keep `persistent_visible_surface_candidate_count=0`, `rigid_pose_candidate_count=0`, `object_geometry_complete=false`, and `object_pose_requirement_met=false` even when an object has a repeatable envelope.

The current object-track dataset exporter prepares the next evidence source for this stage. It writes per-object frame datasets only where the multi-object timeline has a visible SAM2 mask and the UniDepth metric archive has depth for the same raw frame. The exported rows remain tracking inputs; they become object-pose evidence only after a material point tracker produces correspondences and a rigid/deformable motion test accepts them.

For tomato, V17 enforces a near-rigid shape prior: one canonical tomato mesh with pose and small deformation, plus visible-surface updates only when they do not contradict the canonical shape.

For trash bag, V17 uses deformable surface state and avoids pretending there is a rigid 6D pose.

### Stage 6: Contact State And Physical Consistency

Contact is a latent state per hand patch and object surface region:

```text
no_contact
candidate_contact
sticking
sliding
supporting
occluded_contact
rejected_unresolved
```

The graph estimates contact from:

- 2D hand/object mask adjacency;
- depth ordering and depth gap;
- 3D hand-object SDF gap;
- nonpenetration;
- relative motion of hand patches and object surface tracks;
- object acceleration consistency when object mass/rigidity evidence makes the test meaningful;
- deformation consistency for nonrigid surfaces.

Contact cannot be asserted from nearest distance alone. Contact cannot be rejected when image/depth/track evidence supports contact but the current 3D state is inconsistent; that case becomes a state repair target.

### Stage 7: Full Nonlinear Graph Target

This section defines the still-open full nonlinear solver target. The implemented V17 graphs are the contact-mode graph and sparse evidence-consistency graphs described in the Status section: the contact-mode graph estimates binary contact/no-contact/unobserved modes from fixed V17 geometry and image evidence, and the sparse geometry graph optimizes object translation, small-angle object rotation, local contact-patch correspondences, and hand camera-ray depth corrections. Camera trajectory, MANO articulation and shape, object mesh topology, full object geometry, and nonlinear contact physics stay fixed. The current sparse graph only shows consistency of the accepted evidence layer under that limited variable set; it is not full annotation closure.

The generated joint-problem contract is the current machine-readable definition of this target. It compares the current sparse graph against the required variable families and fails closed by setting `v3_solver_complete=false`, `annotation_ready=false`, and `deliverable_ready=false`. The next optimizer must either create graph variables for those families or record a source-backed reason for fixing one family. A graph over one legacy object stream cannot satisfy this stage even if its local contact residuals are small.

The full V3-class solver should implement the prediction/update idea as a fixed-lag nonlinear factor graph. A simple constant-velocity or constant-acceleration prior is not the process model for hand-object manipulation. It can appear only as a weak local smoothness regularizer. The actual process terms are learned priors and physically grounded residuals.

State variables:

```text
T_wc_t                    camera pose
H_h,t                     MANO hand pose/shape/global transform
V_h,t                     hand velocity latent
Z_h,t                     learned hand-motion latent
G_o                       object canonical geometry
T_wo,t                    object pose for near-rigid components
D_o,t                     object deformation state
C_h,o,t                   contact mode variables
U_*                       uncertainty/covariance variables
```

Residuals:

```text
camera motion and reprojection
hand MANO prior
hand 2D keypoint reprojection
hand mask silhouette
hand metric-depth support
hand temporal velocity/acceleration
object mask replay
object depth replay
object point-track consistency
rigidity or deformation energy
object shape-prior consistency
hand-object nonpenetration
contact equality / sliding / support residuals
measurement confidence calibration
HaWoR motion-infill prior
hand-object correspondence prior
hand-object generative plausibility prior
```

The learned process model predicts through missing measurements with growing uncertainty. For hands, HaWoR-style motion infilling provides the primary learned temporal proposal. For contact, TOCH-style spatio-temporal object-to-hand correspondence provides a learned contact refinement prior. For broader hand-object plausibility, a G-HOP-style diffusion prior can propose or score physically plausible hand-object states. These learned priors enter as factors in the objective; they do not override image, mask, depth, object-track, or nonpenetration evidence.

Measurements update the state only when their residuals are plausible. Outlier measurements remain in the measurement store but do not become accepted states.

### Stage 8: QC

V17 QC is frame-local and timeline-local.

Required frame anchors:

- trash 0182: right hand must stay on the visible hand or be flagged unresolved;
- trash 0260: wrong edge hand hypotheses must be rejected or repaired;
- trash 0764: hand-bag contact must be represented as contact or unresolved repair target, not isolated separation;
- trash 0856: bad hand/object state must not produce a confident contact label;
- trash 0949: visible hands must pass hand residual checks;
- trash 0970: visible hand evidence requires an accepted, predicted, or unresolved hand state;
- tomato 0480/0720/0760: tomato mesh must preserve persistent near-rigid shape;
- tomato context: bowls/plates/trays must appear in the multi-object roster when visible and relevant to the manipulation.

Aggregate QC is insufficient. Every deliverable manifest must include:

```text
named_anchor_status
measurement_source_coverage
accepted_state_coverage
rejected_unresolved_frames
hand_residual_summary
object_residual_summary
contact_residual_summary
render_frame_count_qc
```

### Stage 9: Rendering

The V17 world view must be rebuilt.

The 3D panel must show:

- shaded MANO hand meshes with skeleton overlays as secondary cues;
- shaded object meshes with separate colors per object;
- current camera frustum with textured image plane or raw-frame thumbnail plane;
- a line of sight from camera to hand-object region;
- stable world axes and scale;
- head trajectory as a subtle path;
- local manipulation close-up with contact patches;
- uncertainty/rejection status when state is predicted or unresolved.

The side-by-side video remains full-length and synchronized with the raw video. Debug plots can exist as QC artifacts, but the deliverable must read as a reconstruction.

### Stage 10: Deliverables And Closure

For every V17 sample:

```text
overlay_mano_object_multi.mp4
world_reconstruction_3d_v17.mp4
side_by_side_v17.mp4
annotations_v17_full.json
measurements_v17_full/
object_meshes_v17_full/
v17_manifest.json
v17_anchor_qc.json
```

Closure requires:

- output frame count and fps equal raw video;
- all named V16 failure frames pass or are explicitly marked unresolved with correct evidence;
- no visible hand disappears silently;
- no per-frame object depth patch is delivered as a persistent near-rigid object;
- multiple manipulated/context objects are represented where visible and relevant;
- contact labels are state estimates with supporting residuals;
- world reconstruction is visually legible as a 3D scene.
