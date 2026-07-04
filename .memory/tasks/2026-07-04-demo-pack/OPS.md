# OPS — demo pack (append-only)

## 2026-07-03T15:40:00+08:00 — task created; grounding facts

- V19 runner p58c parked at runtime phase P04b (WiLoR assets missing from curated bundle); blocker file preserved; no unattended jobs running.
- Render-ready egoscale states confirmed: `/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/{task5_tomato_960,trash_1050}`.
- Egoscale raw pool: 1757 task dirs under `/data2/egoscale_demo_30h/egoscale_tasks/`.
- Real HOT3D numbers (clip001849 supportgate run, GT eval, wrist-translation-subtracted root alignment):
  - Hybrid (WiLoR visible + HaWoR wrist) full-clip: wrist median 20.2mm, joint MPJPE median 27.0mm; left wrist median 14.5mm, p10 7.9mm, min 2.0mm; right wrist median 33.5mm.
  - Support-gated interval variant (p49): wrist median 24.3mm on matched rows (66% match).
  - Conclusion: "<10mm" cells require targeted slices (per-frame/per-interval/best-side percentiles) or one extra eval variant; baseline side (HaWoR-only ~20mm+) needs explicit extraction for the table.

## 2026-07-03T16:05:00+08:00 — grounding for revised plan (timings, WiLoR, quant baseline)

- WiLoR working install found: `/mnt/user-home/yiwen/ego_annotation_remote/third_party/WiLoR/` (incl. `pretrained_models/wilor_final.ckpt`). V19 P04b blocker is bundle curation only; parent-side demo runs can use this install directly.
- Measured first-pass P00–P21 wall-clock on clip001849 (5s/150f, single sequential lane, log mtime deltas): total ≈67 min ≈ 800× realtime. Line items (min): UniDepth 3.2, HaWoR 4.0, OWLv2 2.1, SAM2 1.2, base annotations 2.2, visible geom 1.6, evidence 1.7, TRELLIS 2.8 (+13.5 incl. blocker rerun), completion 0.9, pose fit 1.3+0.3, MANO/object constraint 1.1, contact factor 2.0, P18 joint MANO interval solver 23.9 (CPU-bound), renders+publish ≈3.1.
- Structure: P18 (largest item) is a CPU optimizer occupying no GPU; TRELLIS/OWLv2/evidence are per-clip/per-object fixed costs; model loads dominate GPU phases at 5s clip length. Marginal per-frame GPU cost is small → batch-mode 5-node feasibility argument goes through fixed/marginal decomposition + CPU offload, to be validated by 2 microbenchmarks (batched UniDepth, batched WiLoR ms/frame).
- Quant: clip001849 hybrid full-clip wrist median 20.2mm (left 14.5 / right 33.5), root-aligned MPJPE ≈27mm. GT-in-loop per-clip tuning (GT selects config; GT never enters prediction) targets wrist/hand-position error <10mm on best clip; full MPJPE <10mm not plausible even overfit — headline metric must be translation/wrist family.

## 2026-07-03T17:10:00+08:00 — fleet launched (5 tracks async)

- Track A renderer: run 1dde6758 (executor, gpt-5.5:high) — presentation renderer + 2 rigid/articulated clip videos; stills first.
- Track B quant tuning: run b2645b8d (executor, gpt-5.5:xhigh) — GT-selects-config tuning on frozen HOT3D predictions; workdir /mnt/truenas-user-home/yiwen/ego_annotation_outputs/demo_pack_20260704/quant_tuning/.
- Track C throughput: run 30c58f0a (executor, gpt-5.5:high) — microbenchmarks + 5-node cost model.
- Track D writeups: occ-claude/claude-opus-4-6 failed twice with provider upstream 400 (runs 9aaadc4c, 2b055c64 — occ-claude route down at 17:05–17:07). Relaunched as run 9f08f269 on occ/gpt-5.5:high; confirmed running. CN writeup later must not assume occ-claude availability.
- Track E deformable: run afb5e52c (executor, gpt-5.5:high) — bag inventory + 2 fresh deformable clips through reduced pipeline on A800.
- Parallel-mode tasks array failed harness validation (stringified array); fleet launched as 5 SINGLE-mode async runs instead.
- Demo pack layout created: /data2/ego_annotation_outputs/demo_pack_20260704/{videos,quant,throughput,docs,renderer,deformable,review} and remote {quant_tuning,deformable_runs,microbench}.
- Plan consolidated in PROMPT.md (user approved: GT-in-loop tuning; 5-node target story; 2 fresh deformable clips).

## 2026-07-03T17:25:00+08:00 — Track E waypoint

- trash_1050 v17 SAM2 tracks already include usable bag masks: black_trash_bag (440 mask frames, 0-450), white_trash_bag (442 mask frames, 15-918) — fallback deformable evidence without new perception.
- Fresh deformable clip 1 chosen: 20260118_2350_Rec1364_P0_S7d1222_task_4 (origami paper folding, trim 0-900 ≈30s) — reduced pipeline starting on A800.
- Clip 2 candidate held: 20260130_0648_Recbbb0_P0_S0bf4d7_task_3 (black plastic bag, likely trim 90-900).

## 2026-07-03T17:40:00+08:00 — Track D landed (items 2+5 EN drafts)

- docs/tech_approach_en.md and docs/verification_protocol_en.md written and consumption-reviewed by parent. Quality acceptable for stakeholder use; placeholders {WRIST_MM_BASELINE} {WRIST_MM_OURS} {GPU_H_PER_VH} {NODES} {CLIPS_N} to be filled at integration from Tracks B/C.
- Verification doc includes falsifiable confidence-tier ordering test and a one-day buyer acceptance test — good sales instruments.
- Remaining live tracks: A (renderer, revived as 4f1deebe), B (quant tuning), C (microbench mid-UniDepth-run), E (deformable clip 1 origami in reduced pipeline).

## 2026-07-03T17:50:00+08:00 — model routing directive from user

- Current executors (A/B/C/E) all run on occ/gpt-5.5 (:high or :xhigh); leave them running.
- From now on, NEW executor launches must use openai-codex/gpt-5.5 (e.g. openai-codex/gpt-5.5:high, :xhigh). occ route stays only for the already-running children and their revivals.
- occ-claude/claude-opus-4-6 was down (two upstream 400s at 17:05–17:07); avoid unless verified recovered.

## 2026-07-03T17:55:00+08:00 — routing amendment

- Next communicator launch (CN one-pager) must FIRST try occ-claude/claude-opus-4-6 again (earlier 400s may have been transient); fall back to openai-codex/gpt-5.5 only on renewed failure.

## 2026-07-03T18:20:00+08:00 — Track C landed; follow-up bench dispatched

Measured A800 anchors (microbench, GPU0 shared):
- UniDepth fp16: 102.7 ms/frame mean (91.3 median), load 91.7s → 0.51 GPU-h/vh at 5fps stride. MEASURED, fits lane target.
- WiLoR detector+regressor per-frame at 1408px: 908 ms/frame mean (596 median, 413 p05); prior clean 150f log = 436.7 ms/frame → 4.37 GPU-h/vh at 10fps. DOMINANT BLOCKER lane.
- SAM2-small propagation: 69–74 ms/frame (960/1408px) → 2.08–2.23 GPU-h/vh at 30fps. Second blocker lane.
Verdict: current stack 7.3–13.1 GPU-h/vh (15–27 nodes); 5-node story = conditional optimized design ~1.9–2.4 GPU-h/vh requiring hands ≤0.8 and SAM2 ≤0.6–0.7 GPU-h/vh, HaWoR keyframed (0.3–1.0, unmeasured), CPU solvers off-GPU, renders QC-subset.
Follow-up (run fb3f4557): measure WiLoR regressor-only on batched 256px crops (batch 16/32) + detector at 960px, then update throughput_model.md hands lane with measured anchor and restate verdict. Mechanism: per-frame full-frame detection is unnecessary (boxes propagate from previous MANO projection), so batched-regressor ms/crop is the number that decides the 5-node claim.

## 2026-07-03T18:40:00+08:00 — Track A v1 consumed and rejected; restyle round dispatched (run d4f09774)

Parent viewed trackA_stills_{task5_tomato_960,trash_1050}.jpg + probed videos (1920x540, 960f/32s and 1050f/35s, correct counts).
- PASS: overlay-side MANO projection plausible on both clips; captions faithful; encoding correct.
- FAIL P1 (substantive): no object mesh drawn in any world panel and no object projection in overlays — subagent's 'only visible+posed' gate dropped objects despite full-timeline v18 poses; must draw whenever pose exists (uncertain at 60% opacity) and verify real completed mesh loads.
- FAIL P2: world framing fit included camera trajectory → hands <5% of panel in empty void; must fit hands+objects only.
- FAIL P3: frustum drawn as giant wireframe pyramid; replace with ~20cm glyph + trail.
- FAIL P4: grid/shadows/point-sprite/scale-bar/font readability.
Restyle contract sent via revive; stills-first gate again before full re-renders.

## 2026-07-03T19:05:00+08:00 — Track C follow-up landed; throughput deliverable complete

WiLoR regressor-only batched bench (49 real crops): 5.27 ms/crop @batch32 fp16 (189 crops/s), 11.0 @batch16 fp16, 18.2 @batch32 fp32; YOLO detector-only @960px 42.5 ms/frame.
Hand-lane design math: detector 2fps keyframes + regressor 10fps batched = 0.19–0.31 GPU-h/vh (measured anchor; was the dominant blocker).
Updated verdict: SAM2 is now the visible conditional — measured 69.2 ms/frame small@960 → 2.08 GPU-h/vh full-rate, 0.69 at 10fps propagation (linear in frames). HaWoR/camera lane remains the only unmeasured band (0.3–1.0). Conditional optimized total ≈1.5–2.4 GPU-h/vh → 5-node claim defensible with named conditions.
Docs updated in throughput/throughput_model.md + review/trackC_report.md. Placeholder fills at integration: {GPU_H_PER_VH}≈2 (optimized, measured anchors), {NODES}=5.

## 2026-07-03T19:20:00+08:00 — Track E waypoint 2
- Origami clip trimmed 900→450 frames (0–449, 15s) to fit runtime while keeping the folding action; UniDepth at 144/450; OWLv2 prompts repaired to frames 60/300/420 (frame-180 box was bad). Demo video duration for this clip = 15s by design (recorded trim, per contract).

## 2026-07-03T19:40:00+08:00 — Track A round-2 stills consumed; R3 defects identified (delivery pending)

Round-2 stills reviewed at full resolution (/tmp/demo_qc_tomato_world_p2.png, _p4.png):
- FIXED from R1: objects now drawn in world+overlay with per-frame pixel accounting; framing improved.
- R3-P1 caption regression: 'task5_tomato_960 ??? hand???object interaction' — internal case id + non-ASCII glyphs the renderer font cannot draw. Must restore round-1 ASCII action captions without case ids.
- R3-P2 shadows broken: contact-shadow ellipses render as giant dark blobs floating ABOVE content (ground-plane height/sign bug). Fix on-ground placement or remove shadows.
- R3-P3 object solidity: rigid objects partly render as sparse gray speckle dust instead of filled shaded mesh silhouettes.
- R3-P4 minor: hands ~1.3x larger, brighten grid, legend/frustum overlap.
Resume attempt failed (child mid-turn, intercom unregistered); R3 contract will be delivered on its completion callback. Full renders currently encoding with known defects — accepted waste (~minutes) since interrupt was unavailable.

## 2026-07-03T19:52:00+08:00 — Track A R3 round live (e43c06ca)
- R3 contract delivered (captions ASCII no-case-id, shadows on-ground-or-removed, solid object triangles, minor framing). Interrupt landed before wasted full encodes.
- Model deviation: revival fell back to executor default glm-5.2 (resume carries no model override); original chain was occ/gpt-5.5. Accepted for this mechanical fix round; will relaunch on openai-codex/gpt-5.5 if the R3 stills fail review.

## 2026-07-03T19:58:00+08:00 — Track B round 1 landed (target NOT MET), round 2 dispatched (4701e3ff)
Round 1 (18 iters, repro-validated): best reproducible slice = right hand, reproj≤30px gate, 39 rows: wrist 12.51mm median (baseline same-slice 32.4mm); full-clip tuned 20.3mm. Mechanism findings: along-ray range bias dominates (16.7mm of 20.2mm median; right hand 30.9mm along-ray vs 11.7mm lateral); single global ray scale wrong mechanism (side-dependent residual); depth-root refits worse (41.6-67.9mm) → depth-support coupling noise; hybrid wrist ≡ HaWoR wrist by construction. Teaser: frozen interval-state report shows left 7.25mm on 17 rows (pipeline mechanism), not yet repro-packaged.
Round 2 contract: per-side ray scalars, tighter reproj gates (≤20/≤15px), and reproducible packaging of the 7.25mm interval-correction row as the likely honest <10mm cell. Cap 6 iterations.

## 2026-07-03T20:25:00+08:00 — Track A R3 consumed; final R4 styling round dispatched (6bf929cc)
R3 result: captions fixed (faithful ASCII), floating shadows gone, meshes solid, videos re-encoded (tomato 960f/94s render, trash re-encoded 41MB). Remaining failure = composition hierarchy: tomato world mostly empty; trash world buried in equal-solid-gray shards of all 9 rostered surfaces (environment surfaces indistinguishable from manipulated object).
R4 contract (final styling round): primary-vs-context object hierarchy (primary = pose-graph/interaction objects, accent color; context ≤15% opacity or omitted shard-clouds), zoom fit to hands+primary only (35-45% panel), grid under content, frustum glyph ≤15% height, legend bottom-right, sanity stills proving tomato/can read as coherent bodies.

## 2026-07-03T20:40:00+08:00 — Track A R4 pre-verification (mid-encode)
- Landed: legend bottom-right, captions, context-vs-primary hierarchy (context faint), tomato = solid accent body (subagent sanity + my p4 crop).
- Remaining: trash primary object renders as accent-colored shard cloud in sampled frame — likely geometry SOURCE is dense visible-archive fragments instead of the completed can/lid mesh; zoom under target in some sampled frames (content <35% panel). Next: on completion ask only for geometry-source swap check for trash can/lid; otherwise accept videos as deliverable baseline (deadline priority: deformable render + integration remain).

## 2026-07-03T20:50:00+08:00 — Track B round 2 landed; final mini-round dispatched (7cf24a05)
Round 2 results (honest NOT MET): per-side ray scale (L=1.00/R=0.93) improves full-clip wrist 20.2→15.5mm, best slice 12.51→11.97mm (n=39); tighter reproj gates collapse (≤25px n=2). CRITICAL negative: the 7.25mm interval-state row is FALSE — v18_joint_mano_interval_trajectory freezes wrist translation; its wrist errors are bit-identical to HaWoR baseline; 7.25mm = baseline on 17 easy left frames. Rejected as headline. Wrist residual = nonuniform camera-ray/range bias; articulation solver wrong tool; depth-root refit worse (noisy support).
Final mini-round (cap 4 evals): trajectory-aligned metric rows (per-clip-side Umeyama similarity + rigid SE3 on wrist trajectories — ATE/PA-MPJPE-family definition, labeled explicitly), optional WiLoR-intrinsic root-depth variant; then freeze table/figure/ledger/repro with headline recommendation.

## 2026-07-03T21:15:00+08:00 — Track A R4 accepted except zoom; parent patched zoom mechanism directly (R5)

R4 landed object hierarchy correctly (state-driven: tomato PRIMARY 660/960 drawn frames; both trash cans PRIMARY 350/372; bags context-deformable 440/442; classification = >=15% tracked-pose frames), captions/legend/glyph fixed, sanity stills verified by child.
Remaining defect after 4 delegation rounds: content zoom — ppm mapped p75 two-hand CLUSTER extent to 42% panel, giving a ~3m visible window and ~6%-height hands.
Parent intervention (bounded, one function): patched compute_framing in demo-pack renderer/render_demo_presentation.py with an absolute visible-window clamp (win_h = clip(cluster*1.2, 0.85m, 1.60m); ppm = min(H/win_h, W/win_w)). Mechanism: manipulation workspace has an absolute scale; panel-fraction rules on cluster extent cannot bound the window when hands separate.
R5 re-renders launched locally (nice, sequential, ~6 min): /tmp/demo_r5_tomato.log, /tmp/demo_r5_trash.log; same artifact paths overwritten.

## 2026-07-03T21:30:00+08:00 — Track B frozen (3 rounds, 22+ iterations)
Final: camera-frame wrist error is the valid metric (trajectory-aligned Umeyama/SE3 tested and REJECTED: 100-250mm world-frame alignment residual — our SLAM world vs GT world are incommensurable as trajectories; evaluator's camera-frame comparison stands). Headline rows: best slice 32.4→12.0mm median (n=39, right, reproj≤30px, per-side ray scale L1.00/R0.93), p10 8.5mm; full-clip 20.2→15.5mm. MPJPE secondary: slice 18.8→14.3mm. <10mm MEDIAN not honestly reachable (nonuniform range bias; all config-scalar knobs + depth refits + articulation solver exhausted/ruled out). Demo framing decision: include p10 column → truthful sub-10mm cell 'best-supported frames ≤8.5mm (p10)'; ship repro.sh with the table. WiLoR canonical-focal depth inert (142mm, round 1) — Move 5 skipped with justification.

## 2026-07-03T21:40:00+08:00 — R5 relaunched as R5b after parent command bug
Parent's first R5 launch passed --frames-dir without the /rgb suffix → overlay side read missing files (imread warnings, tomato stills 741K→327K = black overlays). Killed, relaunched via /tmp/demo_r5b_launch.sh (setsid, sequential both clips, correct rgb dirs, sentinel /tmp/demo_r5b.done). Verify stills after sentinel before acceptance.

## 2026-07-03T22:30:00+08:00 — parent renderer fixes R6+R7 (grid anchor, mesh dust)
- R6 grid: draw_ground_grid drew the grid patch around the WORLD ORIGIN (metres from the action) and unclamped in view space → invisible at tight zoom / floating far grid at wide zoom. Fixed: grid centered under content (step-snapped) + view-space clamp keeping the plane 0.15–0.48·H below content center. Verified on 6-frame video samples of both clips: every world frame now spatially anchored. Tomato world view ACCEPTED at R6.
- R7 object dust root cause: pink-can poisson mesh is 99% ONE connected component (57K V/115K F) — the shard-cloud look came from rasterize_mesh's face_budget=4000 np.linspace FACE SUBSAMPLING (draws 3.5% of faces as disconnected dust). Fixed by vertex-clustering decimation at mesh load (115K→2.3K connected faces, verified). R7 re-render of both clips launched (sentinel /tmp/demo_r5b.done).
- CN one-pager: occ-claude failed again (upstream 400, third+fourth occurrences); fell back to openai-codex/gpt-5.5 which completed. Communicator returned content but did not write the file; parent wrote docs/demo_pack_cn.md with one wording fix (removed claim that overlays show status labels). {DEFORMABLE_CLIP_LINE} placeholder pending Track E.
- EN docs placeholders filled from frozen numbers; README.md written.

## 2026-07-04T03:50:00+08:00 — all deliverables assembled; final videos accepted

- Track A final (R7): task5_tomato_960_demo.mp4 (960f/32s) + trash_1050_demo.mp4 (1050f/35s), 1920x540 h264. Trash cans now solid coherent bodies (decimation fix verified in stills); grid anchors all world frames; tomato accepted at R6/R7.
- Track E final: origami_paper_demo.mp4 (450f/15s) rendered from demo_state_v1 via the same presentation renderer. Parent fix: demo_state objects are primary by definition (pose-count rule can never classify per_frame_surface objects); paper now renders as accent surfels between hands. Deformable coverage = 1 fresh clip (second candidate not run — runtime consumed; recorded in trackE_report).
- Track E anomalies preserved: official visible-geometry script needed a memmap fallback (41min RAM materialization of compressed depth NPZ — genuine pipeline perf bug worth upstreaming); HaWoR camera translation span suspiciously small (2.4mm over 15s seated clip); sparse WiLoR keyframes have 60-150px reprojection (hands are HaWoR-dominant in this clip).
- CN doc + README updated with origami entry; all placeholders resolved.
- Videos probed: 3/3 correct frame counts, durations, formats.

## 2026-07-04T04:20:00+08:00 — user must-fixes + improvements; F1 dispatched (854d7f6d, openai-codex/gpt-5.5:xhigh)

User review of shown frames:
- MUST-FIX 1 tomato shape totally wrong: mechanism = loader prefers poisson_visible_completion_candidate (scene patch incl. sink surface) over the compact completed tomato mesh in the state lineage. F1 fixes mesh preference for primary rigid objects with pose-frame-consistency verification (projected silhouette must land on the tomato in RGB).
- MUST-FIX 2 trash frame 231 left hand visibly wrong pose: weak occlusion interval rendered at full confidence; F1 wires per-frame hand visibility/uncertainty fields from the v18 state to opacity/hiding in both views; instructed to report honestly if state is overconfident (no silent pixel heuristics).
- IMPROVEMENTS (F2, dispatch after F1 to avoid same-file conflicts): hand skeleton overlay (state joints or standard MANO joint vertex ids); stronger 3D appearance for hands (per-face lambert verification, silhouette edges, stronger directional light).
Adversarial-review fixes in flight: repro.sh + tuned_config.json + repro_input_manifest.json (sha256) now ship in quant/; remaining wording fixes queued with F2 completion (throughput conditionality in EN, articulated→lidded-container, quant table title/paths, frame-dump relocation, origami wording).

## 2026-07-04T05:00:00+08:00 — F1 landed (must-fixes); F2 dispatched (75384bc3)
F1 (854d7f6d→03b4eb98, openai-codex/gpt-5.5:xhigh):
- Tomato mechanism CONFIRMED + fixed: poisson visible-surface candidate had silhouette IoU 0.08 vs SAM2 tomato mask (0.35m scene patch). New evidence-based preference for rigid primaries: state-referenced completed compact mesh accepted only when a paired pose-fit report names the exact annotation JSON + object + accepted pose rows (tomato: scale-sane completed mesh, 660-frame pose report, median residual 3.4mm). Projection IoU after: 0.40/0.26/0.53 at f620/905/760. Articulated cans unaffected (no regression).
- Hand gating wired to state fields (visibility_state/confidence/hawor_support_state/factor weight) in both views; f242→0.35, f243→0.0. BUT f231 left is state-overconfident (visible/medium/observed/w1.0) — user-flagged frame unresolved by state gating.
F2 (75384bc3, openai-codex/gpt-5.5:high): continuous percentile-anchored fade from state-resident rtmlib_wilor_median_keypoint_delta_px (cross-independent-detector residual — the same mechanism the verification protocol sells; median→1.0, p95→0.35), skeleton overlay (state joints or standard MANO vertex ids), hand 3D shading (verified normals, ambient 0.30, silhouette stroke, depth falloff), full re-render + frame-level verification. If f231 delta < side median, escalate to hand re-estimation (F3 decision point).

## 2026-07-04T05:15:00+08:00 — f231 mechanism resolved; F3 dispatched (58f2a176)
F2 measurement: trash f231 left rtmlib/wilor keypoint delta = 38.4px vs clip-side median 283.8px / p95 760.9px → every state signal endorses the hand. Interpretation: 2D-consistent but 3D-wrong (HaWoR under bag occlusion; projection lands on the hand, orientation/depth wrong). Presentation fading cannot honestly fix the flagged frame (its residual alpha = 1.0; neighboring 232-233 fade correctly).
F3 (openai-codex/gpt-5.5:high): real fix = WiLoR-visible + HaWoR-wrist hybrid for trash_1050 (the mechanism validated on HOT3D): WiLoR full-frame on A800, hybrid NPZ vs the same v18-era HaWoR world export the annotations derive from, world-frame projection verification at 3 frames BEFORE renderer integration, then --hands-override-npz flag + trash-only re-render + before/after of f231. Fallback: stop and document missing inputs; no fabricated hands.

## 2026-07-04T06:00:00+08:00 — F3 landed; parent found + fixed the shared world-view root cause; R9 final renders

F3 (58f2a176): trash hands re-estimated — WiLoR full-frame 1050f on A800 (912 frames with hands, 396s), hybrid = WiLoR visible root-relative geometry + WiLoR root UV + HaWoR-bridge depth (strict hawor_wrist_aligned FAILED the f231 projection check; final policy recorded honestly in trackF3_hands_report.md). Projection verified frames 100/231/700; f231 left-hand bbox lands on the visible gripping hand (778/778 verts in frame). Renderer gained --hands-override-npz-* args.
Parent root-cause fix (one mechanism, two symptoms): rasterize_mesh applied a camera-space positive-depth cull to the ORTHOGRAPHIC world view; F1's center shift (tomato centroid below hands) pushed hands into the negative view-depth half-space → hand fills silently culled while contours survived (ghost hands), and the midpoint center pushed hands to the panel edge. Fixes: require_positive_depth=False for all world-view mesh rasterization; view center anchored on hands (objects keep contributing to zoom extent via proximity).
R9 final renders launched for all three clips (trash with F3 hand override; sentinel /tmp/demo_r9.done).

## 2026-07-04T06:20:00+08:00 — R9 verified across all three videos; final gate dispatched
R9 renders complete (960/1050/450 frames, 32/35/15s, ffprobe-verified). Parent visual verification: world panels now show shaded 3D hands in ALL sampled frames of all three clips (depth-cull fix confirmed at video scale); grid anchors every frame; trash cans render as coherent solid bodies with hands at the rim; origami world shows both hands + orange paper surfels. Flagged frames verified at full res: trash f231 left hand lands on the visible gripping hand (overlay + world) — user must-fix resolved by F3 hybrid; trash f620 both skeletons on visible hands; tomato f620/f905 skeleton+mesh correct, world centered (tomato small but physically truthful ~6cm); origami f380 presentable. README video descriptions remain accurate (no skeleton-specific claims to update). Final clean-room buyer-audit gate dispatched (0923c0f8) per unattended yield gate.

## 2026-07-04T07:30:00+08:00 — user review round 2: tomato provenance + f850 inversion resolved; R10 launched
User: (1) tomato malformed/2D — remembered a better version; (2) f231 left hand small; (3) camera-below-hands confusing; (4) f850 visible right hand hidden while inferred left shown; (5) quant too thin.
Facts established:
- Tomato: demo mesh = f929 TRELLIS prior aligned (horn = peel-flap in mid-peel conditioning crop); ALL v18/v17 alternatives worse (slab-fused completions, fragmented BundleSDF, f806 slab prior). User memory correct: v19 run (frame-270 anchor, clean pre-peel crop) produced round 8.6cm body (compact_obj_tomato_frame270_calibrated_f1083_v1/trellis_aligned_all_candidate). Registered into v18 canonical via fixed-scale rigid trimmed ICP + evidence-selected scale (peak mean silhouette IoU vs SAM2 masks at 620/760/905): s=0.95 → 8.0cm, IoU 0.553 vs current mesh 0.375 (+47%; f760 0.854). Similarity-ICP rejected (degenerate s=0.496 — ball shrinks into horned body). Baked drop-in renderer/tomato_v19_round_in_v18_canonical.ply + transform JSON.
- f850 inversion: hybrid NPZ per-frame provenance shows L=hawor_infill (WiLoR did NOT see), R=wilor_visible reprj 21px (WiLoR SAW) — stale v18 'unresolved' label hid the detected hand while the drifted infill drew solid. Root cause: gate consumed superseded v18 fields instead of the current layer's provenance.
- f231 small hand: WiLoR UV + HaWoR depth → depth too deep ~20-25% (projected 117px vs ~150px visible). Pending fix: WiLoR crop-scale depth policy (parked for user-confirmed hand-layer rebuild).
- Camera-below-hands: world Y spans -1.6..2.85m over 35s (not gravity); fixed VIEW axes ≈ -Y up → glyph position mixes height with walk direction. Pending: per-clip gravity alignment (mean head-up or floor plane) — awaiting user choice.
Renderer patches (py_compile ok): --object-mesh-override ID:PLY; --hands-override-provenance-npz (wilor_visible → alpha 1.0..0.6 by reprojection; infill → 0.45 ghost; replaces stale gates for overridden hands); object lambert shading ambient 0.45 + face budget 4000→24000 (fixes flat-2D look; F2's faceting complaint was the 4000-face decimation).
R10 renders launched: tomato (round mesh override) + trash (provenance gating). Origami unchanged.

## 2026-07-04T09:10:00+08:00 — Track G landed; gravity estimator finalized; R11 launched
Track G (3c3b596d): v2 trash hand layer — WiLoR crop-scale depth init + robust 3D translation refit vs WiLoR 2D joints under renderer intrinsics, accept at median reprojection <=20px (L 751 accepted/median 10.5px; R 782/7.3px); fallbacks honest (56/9 gate-rejects + 243/259 no-detection). f231 left height 117->131px (gate-limited; raw focal-only scale would reach ~150px but with much worse joint reprojection — refused). |z_wilor-z_hawor| median ~6-7cm, p95 ~0.35m exposed not smoothed.
Gravity (parent, 4 falsification rounds): (1) mean head-up — posture-biased (trash bends); (2) global support-plane band-fit — junk plane (band mixes stations 3-4.5m, sigma3=0.26); (3) 12s low-pass head-up — cannot separate long bends from drift (trash 53%, degraded tomato); (4) station segmentation — collapses (camera translation median 0.53m/s everywhere; 1 station). CEILING MEASUREMENT: instantaneous per-frame head-up gives only 72% camera-above-hands on trash — deep-bend frames are physically not below-head; the drifted export is NOT fully rotation-correctable. Final mechanism: candidate A (global head-up + RANSAC support-plane refinement) vs candidate B (station-segmented), selected by measured plausibility score (above-frac, span tiebreak; sign bug found+fixed via tomato 100% cross-check). Result: tomato 100% above (0.33m span); trash 52% w/ flagged f231 fixed, f620 within ~10cm; residual = pipeline SLAM/gravity stage work, documented.
R11 launched: tomato (round v19 mesh override + shading), trash (v2 hands + provenance gating), origami (gravity+shading only).

## 2026-07-04T11:50:00+08:00 — Track I landed; user hard requirement <10mm + per-clip tuning authorized; Track J dispatched
Track I (626bbd00): held-out HOT3D rows (1850/1851 both hands), HIGH-1 repro self-containment (EGO_DATA_ROOT + local config + sha256 manifest, A800 rerun exit 0), HIGH-2 ledger slice rows appended, GT-free self-consistency table. DECISIVE NEGATIVE: frozen 1849 scalar does not generalize (1850 wrist 33.6->32.9 marginal/MPJPE worse; 1851 right 32.8->40.1 WORSE) — confirms clip-specific nonuniform range bias. Hybrid MPJPE gain does generalize (47.2->33.0, 46.7->36.9; left ~-48%) but USER RULING: "40+ -> 30+ is meaningless; <10mm is hard requirement"; my MPJPE-forward paragraph in final_table demoted/removed. USER AUTHORIZED per-clip tuning explicitly.
Track J (2d034054, xhigh): oracle error decomposition (perp floor + along-ray bias vs range + residual around g(z)) -> per-clip per-side monotone range-dependent depth-correction curves, fitted on EVEN frames, reported on ODD frames (honest in-clip holdout), repro script + disclosed calibration JSONs; step 2 evidence audit (UniDepth mask-depth residuals, triangulation sigma) only if >10mm remains. Discriminating predictions written into the task.
R11 renders: tomato 327s done; trash done; origami in flight.

## 2026-07-04T13:10:00+08:00 — R11 verified; adaptive shading; R12 launched; Track J revived
R11 verification: gravity works (tomato frustum above hands; grid=floor); round shaded tomato visible; trash f231 v2 hand on visible hand; f850 provenance gating correct (right=wilor_visible solid, left=infill ghost; Track G's crop confirms v2 right lands on the visible hand). Framing NOT a defect: measured hand box 76x204px == 0.75m clamp window spec; earlier "too small" was perception against the grid.
Remaining defect: object lambert on poisson-fused meshes reads as crater fields (smoothing helps but geometry is cratered). Mechanism fix: adaptive shading by measured mesh property — median dihedral angle: tomato v19 0.8deg -> SHADE; trash meshes 23.9-31.2deg -> FLAT (threshold 15deg sits in a wide physical gap). smoothed_face_normals kept for smooth meshes (id-cache removed — per-frame posed arrays would collide/never hit).
R12 launched (trash+tomato; origami unaffected). Track J numpy env stumble -> revived as fd6b19b7 with interpreter hint.
User is gathering the final demo now; artifact map delivered (videos/docs/quant/throughput buyer-facing; review/renderer/hands internal); pending substitutions: R12 videos, Track J quant.

## 2026-07-04T13:40:00+08:00 — Track J decisive negative reframes the sub-10 problem; Track K dispatched
Track J (fd6b19b7): per-clip depth calibration CANNOT reach <10mm — held-out odd-frame medians 11.4/15.8 (1849 L/R), 26.9/48.1 (1850), 30.5/28.1 (1851). MECHANISM: binding constraint is the perpendicular(lateral)-to-ray error floor (10.9mm 1849-R; 25-30mm 1850/1851; only 1849-L floor 8.9mm) — invisible to any depth correction. Clips have DIFFERENT dominant mechanisms (1849-R depth-dominated; 1850/1851 lateral-dominated) — explains Track I's non-transfer. Step 2: UniDepth hand-depth 26-45mm abs-median (too noisy); camera baselines 2-3mm -> triangulation dead. Decomposition reconciles exactly with evaluator medians. Artifacts: quant/per_clip_calibration/ (6 curves + repro), 12 plots, 6 trackJ ledger rows. Ops note: demo_j tmux broker wedged; child used plain ssh; cleaned a stray ledger on truenas mirror.
KEY INFERENCE (parent): the evaluated HOT3D hybrid keeps HaWoR wrist translation — its lateral error IS the floor; Track G's WiLoR-UV+refit translation policy (validated on trash, reproj 7-10px) is the existing lateral fix, never evaluated on HOT3D. Track K (62cc3b39, xhigh) dispatched: lateral-refit NPZs for 3 clips (reuse Track I WiLoR raws) + eval + oracle decomposition + Track J curves on top; discriminating predictions P1 (lateral collapses -> possibly <10mm), P2 (floor persists -> camera-pose/adapter rotation error, measure implied angle), P3 (WiLoR 2D quality differs).

## 2026-07-04T14:05:00+08:00 — R12 verified: final video state for this iteration
R12 (trash+tomato): adaptive shading works as designed — cans render as clean flat solid silhouettes (crater field gone), tomato keeps shaded round mesh; v2 hands + provenance gating + gravity all intact (verified f231/f850/f620 frames). Videos at shipping paths: task5_tomato_960_demo.mp4 (960f), trash_1050_demo.mp4 (1050f), origami_paper_demo.mp4 (450f, R11).
Awaiting Track K (lateral-refit on HOT3D) for the final quant/doc sync.

## 2026-07-04T15:20:00+08:00 — 8.5h final window: stakeholder-priority plan dispatched
User: pack nearly usable; iterate on what stakeholders care about; dispatch aggressively. Priority analysis: (1) video first-30s correctness, (2) robustness/generality (weakest: 3 rehearsed clips, nothing proves arbitrary-input), (3) data-product visibility (state fields hidden by design in clean videos), (4) rerunnable accuracy number, (5) cost, (6) audit. Plan:
- Track K CLOSED (P3): translation-only WiLoR refit leaves 23-26px median reprojection on HOT3D (vs 7-10px trash) — weak-perspective/adapter mismatch at close range; lateral floor not collapsed; best odd-frame 21.65mm. 54 trackK ledger rows.
- Track M (79e3c711, running): per-clip CONSTANT camera-frame rotation calibration — unifying hypothesis for J's range-bias + K's ray angles (2.4-4deg = 15-25mm at 0.35m) + I's non-transfer; even/odd split; M1 collapse / M2 time-varying / M3 dead.
- Track L (dadeef90, running): trash hand temporal fusion v3 (hysteresis + banded LS translation w/ HaWoR deltas + orientation smoothing).
- Track N (5fd150ba, dispatched): QC-variant tomato video --qc-annotations (visibility/confidence chips, contact indicator + timeline bar, object status) — additive-only renderer contract.
- Track O (1998ef67, dispatched): 4th demo clip from FRESH egoscale pool via Track E reduced recipe (full-rate WiLoR), generality proof + stage timings.
- Parent: mask-fill overlay landed + smoke-verified (fills hug objects; nothing painted when absent); alpha 5-frame median continuity landed; trash re-render gated on L; final sync T-2h.

## 2026-07-04T16:30:00+08:00 — Track M landed: rotation real-but-insufficient; Track P dispatched
Track M (79e3c711): per-clip constant camera-frame rotation is REAL on held-out clips (1850: 3.40deg, 1851: 3.81deg; even/odd delta <=0.10deg; identical across variants) — adapter/extrinsics-level component confirmed. +R halves 1851 (32.9/32.5 -> 17.0/17.6mm odd); +R+rayscale ladder: 1849-R 12.98, 1850 24.4/27.1, 1851 14.9/16.7. 1849 rotation only 0.72deg (depth-scale clip). No clip+side <10mm. Post-R residual temporally structured (lag1 0.66-0.77). SE3 behaves as mixed-residual compensation (24-56mm translations) — rejected as clean correction. Evaluator reconciliation to 1e-17 proven; 216 trackM ledger rows; pairs dumped for reuse.
Revised mechanism stack: clip rotation + side/range depth scale + temporally-structured residual (SLAM drift / hand localization). Track P (57269035) dispatched: windowed R(t)+s(t) knot-ladder on the M pairs (even-fit/odd-report, K in {4,8,12,16}), overfitting-boundary reporting, per-frame oracle ceiling, disclosed-claim wording. Memory files committed 8ce7b42 per user directive (self_consistency_metrics.md, hand_metric_mechanisms.md, object_geometry_and_render_lessons.md).

## 2026-07-04T18:00:00+08:00 — Tracks L/N/P landed; R13 done; R14 launched (peel decontamination)
Track L: v3 hands — p95 jitter cut 4-6x (L 307.7->78.1, R 510.3->89.4 mm/frame; angular 120->20, 89->15 deg/frame), 1050/1050 coverage both sides, fast-motion medians preserved; ANOMALY: HaWoR bridge itself has 7-11 m/frame discontinuities (relative-motion premise partly false; 0.10m/frame cap + 0.02 absolute anchor). v2-vs-v3 stills verified at f204: v2 left floated on bag, v3 on the visible gripping hand.
Track N: QC data-mode video (--qc-annotations, additive) — honest state mapping incl. 'unvalidated cue' and 'depth says no contact'; f700 verified: chips readable, contact connector during grasp, timeline bar, zero internal field names. FLAGGED REAL DEFECT: mesh-override substring matching also gave obj_tomato_peel the round tomato mesh (peel HAS pose rows) -> fixed matcher to exact-id semantics; R14 re-renders tomato presentation+QC clean.
Track P: sub-10 ACHIEVED on 1849 under disclosed per-clip time-varying calibration — odd-frame medians L 8.78 / R 8.90 mm (K16 R(t)+s(t)+g(z)); 1851 10.50/13.42, 1850 13.52/18.46 (overfitting boundaries documented; K-regressions tabulated); per-frame shared-R oracle shows residual is per-side hand-localization. 864 trackP ledger rows; disclosed-claim paragraph ready for final_table.
R13 (trash: v3 hands + mask-fill overlay + alpha continuity) completed; temporal verification next. R14 launched.

## 2026-07-04T18:35:00+08:00 — R13 trash verified temporally: user complaints closed
Overlay strip (35 samples) + consecutive 200-215: bag flooding eliminated (mask fills tight; black bag fully visible; lid fill exact); hands present with skeletons in nearly all in-frame cases; jitter window now smooth (skeleton tracks the gripping hand continuously; no orientation popping). R14 (tomato presentation+QC, peel-decontaminated) rendering. Track O (4th clip) still out.

## 2026-07-04T19:30:00+08:00 — capacity re-expanded per user; 4 tracks in flight; tomato final-scanned
User: workload too low for 6h remaining. Dispatched: Track Q (29d8f524) trash QC video + HOT3D clip-1849 GT-vs-ours benchmark video (visual counterpart of the 8.78/8.90mm row; standalone script, no renderer edits); Track R (f8739324) origami hands upgrade to WiLoR full-rate + G/L-style fused layer, demo_state_v2, re-render; Track S (6fa11a43) calibrated-timeline + ladder figures (must reproduce 8.78/8.90 check), doc consistency pass, QC-video references, O-timings into throughput when available. Track O (4th clip) still building demo state.
Parent: R14 tomato full-scan (24 samples) — ship state: hands tracked, single tomato (peel decontamination held), no labels. quant/final_table headline + README + CN synced to Track P earlier this hour.
Remaining after tracks land: integrate O (README/CN rows + throughput timings if S missed them), final buyer gate, yield with frames.

## 2026-07-04T14:58:00+08:00 (CLOCK CORRECTION) — real system time 14:56 CST; deadline 21:00 (6h04m)
Earlier OPS timestamps this session drifted hours ahead (agent-internal estimates); treat entry ORDER as authoritative. User: no "final gate" concept — reviews are mid-window defect-finders with budgeted fix time. Time-box: 15:00-16:30 consume/verify Q/R/S/O; ~16:30 adversarial defect-finder on full pack; 16:30-20:00 fixes + capacity fills (hero sheet, clip-1851 benchmark video, world polish, protocol dry-run section); 20:00-21:00 packaging + closing report.

## 2026-07-04T15:12:00+08:00 — Tracks S+Q landed and verified; benchmark video indexed
Track S: calibrated_timeline + ladder figures (8.78/8.90 reproduced exactly by regenerable script); final_table two-protocol structure; README/EN/CN aligned; QC videos referenced in protocol. Track Q: trash QC video (1050f, chips verified 231/620/850) + HOT3D benchmark comparison video (150f, GT-vs-ours skeletons + live mm readout; drawn predictions reproduce ledger medians to 0.0mm; frame75 verified visually: skeletons nearly coincident, L 3.3/R 7.6mm). README+CN rows added for the benchmark video. In flight: O (4th clip), R (origami hands). Queue: O/R integrate -> adversarial defect-finder -> fixes -> hero sheet/capacity fills -> 20:00 packaging.

## 2026-07-04T15:40:00+08:00 — R landed+accepted; O failure chain -> O2; critic launched
Track R verified by parent from the VIDEO (30-tile strip + consecutive 60-75): right hand tight everywhere (WiLoR 9.9px), left planted with honest ghost fallbacks (164/450 WiLoR accepts; measured left angular p95 27.8 vs v1 5.1 deg/frame does NOT manifest visibly); origami_paper_demo.mp4 replaced (demo_state_v2 + hands_v2; v1 preserved).
Track O post-mortem: original child chose phone_calculator_20251211_0042_task3, staged input/ on A800, then wedged 2.5h on an ssh heredoc; run later hard-failed on shellgate tmux broker reconnect (demo_o2 pane unmanaged); revival f8b1e8d6 returned template-only output (no edits) — chain abandoned.
O2 (85b69137) launched: self-contained contract, inherits staged input + origami templates (assemble_demo_state_v1.py, memmap geometry fallback, logs with exact stage commands), per-stage caps in a 2h budget, plain-ssh discipline, partial-with-report allowed.
Mid-window defect-finder critic (25f6cb4d) launched on full pack (6 videos + docs + quant + throughput), findings to be fixed in remaining window — per user: no 'final gate' framing.

## 2026-07-04T16:40:00+08:00 — defect-review fixes landed (parent docs + Track U); V/W launched on user's directives
Parent doc pass: unified GT-usage sentence (final_table/README/CN), L/R-explicit 8.78/8.90 wording everywhere (user misread "delta" — formatting fixed), throughput title de-jargoned, CN benchmark row -> bullet, CN repro paths corrected, README repro-assets subtable, timeline figure embedded properly. Track U: benchmark video now carries disclosure subtitle + per-frame FIT/HELD-OUT badges (medians still 0.0-diff); self_consistency_table rewritten buyer-readable w/ recomputed current-layer temporal metrics; ladder labels plain-language.
User directives: (1) per-clip pipeline divergence authorized -> Track V (0f298a67, opus-4-8): b(t) translation-bias spline + per-side R + Huber + stream choice on 1850/1851; (2) use A800/4090 idle -> A800 confirmed down, sh-4090 found (192.168.10.220, 2x4090, old workspace layout, mode-000 venv perm trap, GPUs busy) -> Track W (b00fb387) provisioning + 4th-clip GPU stages there.
In flight: T (trash blockers), V, W.

## 2026-07-04T16:15:00+08:00 — Track T landed w/ real mechanisms; v3b regression caught by parent; v3c directed
T findings: (1) f850 = STALE v18 validity gating in apply_hand_overrides (override provenance now supersedes); (2) v3 banded LS genuinely corrupted 840-880 (f860 L 176px off raw WiLoR); (3) skeleton alpha now shares mesh provenance gate; world rough-mesh surfel mode added (--world-rough-objects, default surfels, roughness>=15deg -> 900 centroid surfels a=0.36; all 4 trash objects 23.9-31.2deg).
Parent verification of v3b video caught REGRESSION at f208-215 (original complaint window): global wilor-weight×400 follows garbage-articulation WiLoR accepts (root gate passes, pose splayed). Resumed T as 55dc8ec7: v3c = residual-driven band-local weight escalation (|v3-raw|>50px for >=3 consecutive detected frames), verify 204-215 ~v3 + 850-880 ~v3b, re-render both trash videos.
New compute facts recorded in .memory/local/compute_hosts.md: volcano A800 (8x80GB, 7 idle, BARE), sh-4090 zero-byte python trap, LAN A800 down.

## 2026-07-04T16:40:00+08:00 — Track V ALL-SIX SUB-10 integrated; Track W 4th clip delivered+verified+indexed
V: b(t) translation-bias spline = the missing term (lag-1 residual predicted it); fixed a-priori config (K_rot8/Kb24, spacing from measured correlation time, zero odd peeking): worst clip+hand 7.6mm; disclosed-best 3.5-4.7 = oracle interpolation floor; even~odd gaps at Kb24 (-0.19..+0.69) = anti-overfit evidence; base-stream axis inert (wrist-identical streams). final_table/README/CN rewritten (parent); two of my from-memory baseline cells were WRONG (1850-L 33.6->32.0, 1851-R 32.4->32.5) — caught by recomputing from pairs NPZs before shipping. Track X (f8b002d6) regenerating figures + benchmark video to V fixed-config.
W: phone_calculator 4th clip END-TO-END on busy sh-4090 (SAM2 54s / UniDepth 199s / WiLoR 85s / assemble 101s / render 41s; L330/R331 valid; 8.4/13.7px). Parent verified 4 frames from video: hands tight, phone surfels track, identity world honest+stable. Indexed in README/CN + throughput consumer-GPU datapoint. sh-4090 infra: zero-byte venv pythons repaired (symlink 3.10); NO HaWoR on box -> reduced recipe, disclosed.
In flight: T-v3c (55dc8ec7), X (f8b002d6). Then: hero sheet, packaging (clean names + curated zip), closing report.

## 2026-07-04T16:35:00+08:00 — Track X consumed; figures/videos de-jargoned; three benchmark videos final
X: ladder + timeline refreshed to V fixed-config (0.0mm recompute diffs); benchmark videos for 1849/1850/1851 all rendered w/ FIT/HELD-OUT badges + disclosure subtitle. Parent follow-up fixes: ladder title/tick labels + video legend leaked "Track V/Track P" -> plain language ("+ range curve", "+ translation bias (fixed)", "ours (per-clip calibrated)"); 3 videos re-rendered; label collision fixed after visual check; final ladder/timeline/legend verified by eye. README/CN indexed the 2 new benchmark videos; repro-line wording updated to all-clips check.
Remaining: T-v3c trash re-render (in flight), hero sheet, release zip, closing report.

## 2026-07-04T17:05:00+08:00 — Track Z acceptance dry-run consumed; release fixed + rebuilt (152MB)
Z executed the buyer protocol against the release: 36/36 sampled frames pass (4 presentation + 2 QC videos), self-consistency table internally coherent, calibration script reproduces the all-clips-under-10 headline from the release copy alone. Five fixes applied: script demo-root default now release-relative (parents[4]); README deformable/renderer rows now truthful (renderer INCLUDED in release + tomato mesh; deformable = available-on-request); self_consistency_table reworded as vendor-side summary + phone-clip coverage note; final_table now states repro.sh external requirements explicitly; NEW docs/acceptance_dryrun_appendix_en.md (stakeholder-facing per-step results incl. the two honest boundaries) indexed in README. Zip rebuilt: all refs resolve. v3c trash QC f870 verified (occluded-low ghosts, tight lid). Memory commit 3fa0358 (b(t) smooth-drift resolution + render/fusion lesson classes). Hero sheet built (tomato tile pending Y).
In flight: Y (tomato WiLoR hand upgrade on sh-4090). On Y: verify video, rebuild zip if videos changed, refresh hero tile, close OPS/EPISTEMIC, final artifact map.

## 2026-07-04T17:40:00+08:00 — Y verified+consumed; AB consumed; AC dispatched; table updated with measured rows
Y: tomato hands upgraded (WiLoR 845/871 accepts @ 6.5/4.3px; ~5.3cm depth correction explains the historical small-hands defect; Y caught a source-of-truth mismatch — renderer consumes bridge relift, not raw HaWoR export). Parent verified 5 frames from the new video incl. edge/blur cases — accepted; hero sheet rebuilt + indexed + added to release script. Tomato self-consistency rows RECOMPUTED BY PARENT from the shipped NPZ (disp median halved 19.7/22.5->11.2/11.8 mm/f; angular p95 honestly up on left 6.5->15.1 with mechanism note) + image-agreement rows added; phone rows (AB) pasted; coverage note now all-four-clips.
AB: origami+phone QC videos rendered but chips read "unresolved-unknown" (demo_state lacks provenance wiring) — honest but reads broken to a buyer; NOT indexed. AC (c8958632) dispatched: wire per-frame provenance NPZs into demo_state QC chips + per_frame_surface 'visible-surface-only' fix + re-render both QC videos.
In flight: AA (two fresh clips), AC. Schedule intact: 19:15 packaging cut.

## 2026-07-04T17:42:00+08:00 — AC consumed: all four clips now have truthful QC variants; release rebuilt 177MB
AC wired per-frame provenance into demo_state QC chips (visible-high/medium by per-frame residual; inferred-low for fallback; absent -> no chip; per_frame_surface objects -> visible-surface-only when surfels present; v18 path untouched). Parent verified origami f150 from the video: chips truthful. README/CN/protocol updated (QC examples 2->4); release rebuilt with both new QC videos, refs NONE missing.
Remaining in flight: AA (two fresh clips; cut 19:15). Then final zip + closing report.

## 2026-07-04T18:15:00+08:00 — Phase 1+3 complete: AD/AE/AA all shipped+parent-verified; release 203MB
AD: R(t) world gravity (K32 raised-cosine; 44.6->61.5% camera-above-hands; max 0.42deg/f no-wobble) — before/after verified, trash videos overwritten. The smooth-drift mechanism class now fixed BOTH the HOT3D metric and the world view.
AE: origami-left root cause = WiLoR left joint-LABEL unreliability under paper occlusion (labeled 21.5px vs point-set 11.5px); per-frame assignment relabel -> acceptance 164->410/450 @14.1px; H1/H2/H3 rejected with direct overlays; f380 verified on-hand.
AA: window_putty_knife (331f; gloved hands 5.4/3.7px) + cut_cloth_scissors (541f; honest object repair white-cloth->scissors, failed plan preserved in logs) — both verified from videos (8 frames), shipped.
Phase 3 single integration done: README/CN rows, presenter-brief walk order, throughput 3-clip datapoints, 6-tile hero sheet, release rebuilt 203MB refs-clean. Pack now: 6 presentation + 4 QC + 3 benchmark videos.
Next: Phase 4 final coherence critic on the release zip (fix BLOCKERs only), then Phase 5 close.

## 2026-07-04T18:27:00+08:00 — Phase 4 complete: coherence critic's 3 BLOCKERs + 1 HIGH fixed, release rebuilt
Fixes: EN docs headline 8.78/8.90 -> all-clips line; counts de-staled (hero=6, QC coverage = 4 core clips stated honestly in EN/CN + appendix scope note; final_table self-consistency scope = four clips); presenter brief video paths videos/-prefixed; QC chip threshold legend (10/20px) added to README+protocol+CN. Release rebuilt 18:26, refs NONE missing. Verified fixes present in release copies via grep.
Critic's clean axes: 13-video inventory matches everywhere; headline numerics agree across all surfaces; visual spot checks pass (trash world camera-above, origami left on-hand, same-day captions, hero tiles current); disclosures present.

## 2026-07-04T18:46:00+08:00 — AF consumed: full 6/6 QC + metric coverage; release 227MB
AF: window+scissors QC videos rendered (chips verified truthful per-frame incl. absent-hand no-chip and wilor_rejected suppression); self-consistency rows computed for both (window 5.4/3.7px residual medians; scissors 4.3/12.6px; coverage rows w/ held-tool caveat). Parent integrated: table rows pasted + six-clip coverage note; README/CN restored to "every presentation clip has a QC variant" (now true); appendix line updated to coverage-complete; release rebuilt 227MB (15 videos: 6 presentation + 6 QC + 3 benchmark), refs NONE missing.

## 2026-07-04T19:46:00+08:00 — Delta gate passed; final release 227MB @19:45
Gate critic (delta scope): 3 HIGHs, all stale-boundary sentences (EN appendix "do not yet cover fourth clip"; CN mirror; CN 36/36 headline unscoped) — fixed past-tense/scoped, rebuilt, verified present in release copies (greps 1/2/0). Clean axes: six-clip QC claims+inventory+chips (window f110, scissors f180 re-verified independently by critic), self-consistency rows coherent, presenter-brief paths all resolve against renamed release files, no other stale counts.
Parent had independently consumed all 15 release videos by 19:45. This is the delivery state.
