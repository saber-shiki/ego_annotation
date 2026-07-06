# 35 — Fresh-context critic on the parent's package-surface cleanup

Review only. No edits, nothing staged. Branch `yiwen_research`. Case: HOT3D
clip001850 keyboard, right hand. All claims verified on disk this session
(commands in §Reproduce), not taken from OPS/EPISTEMIC or reports 31–33.

Package under review:
`/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706/`
Sidecar:
`.../ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/`

---

## Bottom line

**All seven named blockers are genuinely fixed.** The pose tables, the sidecar
`tables/`, the sidecar root, the graph-health lineage, the row-count/hash
integrity, and the contact state all pass independent verification. The parent's
per-blocker validation claims in OPS are each true.

**But the cleanup is still additive, not substitutive, at the *root-manifest
render surface*.** The exact F1 laundering that subagent 33 flagged — a render
that draws a fabricated keyboard body on the 142 `unknown` frames from the P15
nearest-hold/interp pose graph — is only *partially* retracted. The parent fixed
the `render_artifacts` pointer and demoted **one** of the two P15-posed renders
in the **sidecar** manifest, but:

- the **root `artifact_manifest.json`** still advertises **two** P15-posed
  laundered renders (`full_duration_render` and `generic_contact_state_render`)
  as top-level blocks with every `acceptance_gates` = `true` and **no
  supersession marker** (§R1);
- the **generic** P15-posed render is flagged superseded **nowhere** (not in
  root `superseded_artifacts`, not in sidecar `superseded_render_artifacts`, not
  in `render_consumption.ndjson`) (§R1, §R2);
- the pose-provenance-purity audit and the graph-health render-lineage that are
  supposed to catch this **only scan the one clean render**, so they report PASS
  while two laundered renders ship advertised (§R3 — the root cause of survival).

Both laundered renders were confirmed to draw the body on unknown frames three
ways: their consuming scripts load the 150-row all-numeric P15 pose report; their
own per-frame manifests report `world_body_edges=1649` + overlay body faces on
unknown frames; and the pixels of `v19_contact_state_generic_render` world f090
show a full amber keyboard wireframe under a banner reading "no contact evidence
is fabricated."

Severity: **HIGH** (not a full BLOCKER of the specific `render_artifacts` key,
which is clean, but a HIGH-severity residual laundering surface because the
package still ships and advertises fabricated-localization renders as accepted
deliverables).

---

## Part A — the seven named blockers: all FIXED (verified)

| # | Blocker | Verdict | Evidence |
|---|---|---|---|
| F1 | primary `render_artifacts` unknown-preserving | **FIXED** | Sidecar + root `render_artifacts.{overlay,world,side_by_side}.path` all `= unknown_preserving_pose_render/v19_*.mp4`; each carries `consumed_object_pose_observations_sha256=67dabc79…` (clean pose) and `body_drawn_only_on_measured_frames=true`. `render_consumption.ndjson` rows 6–8 = `status: current_object_pose_observation_render`, `body_drawn_only_on_measured_frames=True`. |
| F2 | static-gauge rows absent from sidecar `tables/` | **FIXED** | `tables/` contains no `*static*`/`*gauge*` file; `object_pose_visibility_ledger.ndjson` (the current ledger) has 0 `static_gauge`/`T_rest` tokens; `tables[]` lists neither `object_pose_static_gauge` nor `object_visibility_ledger`. The two static files now live under `superseded_prior_laundering/sidecar_tables/`. |
| F3 | `static_pose_hypothesis` deleted | **FIXED** | `find … -name '*static_pose_hypothesis*'` → 0 hits; 0 references in root/sidecar manifests. The `superseded_prior_laundering/orphan_pose_dir/` is an empty directory (file deleted, not relocated). |
| F4 | `static_gauge_summary.json` absent from sidecar root | **FIXED** | Sidecar root holds only `manifest.json`, `rest_cluster_reference_for_outlier_rejection.json`, `tables/`. The summary moved to `superseded_prior_laundering/sidecar_root/static_gauge_summary.json`. |
| F5 | graph_health lineage non-null + purity row | **FIXED** | `graph_health.ndjson` = 2 rows. Row 0 `render_state_hash_lineage.consumed_pose_source = object_pose_observations.ndjson`, `…_sha256=67dabc79…`, `body_drawn_only_on_measured_frames=true` (non-null). Row 1 = `pose_provenance_purity` (decision `pose_channel_pure_current__original_laundering_detected`, non-null `render_state_hash=c0ac8ab9…`). |
| F6 | table row counts/hashes consistent | **FIXED** | Every `tables[]` `row_count` == `wc -l` (contact_frame_detail 150, render_consumption 9, object_pose_observations 150, object_pose_visibility_ledger 150, motion_coupling 1, geometry_epochs 6, graph_health 2). Every shipped table sha256 == its `input_hashes` entry. Contact split disambiguated: `input_hashes` records `contact_frame_detail.ndjson=b51c3c11` (150-row shipped) and `contact_frame_detail_source_window.ndjson=de7ebc27` (21-row consumed). |
| — | contact state unchanged | **FIXED** | f32=`geometry_epoch_contaminated`, f36=`full_frame_depth_leak`, f45=`unresolved_incoherent_evidence` in **both** the 21-row and 150-row tables. 0 `confirmed_contact`/`static_gauge`/`T_rest` tokens in either contact table. `object_pose_observations.ndjson` = 8 numeric (f30-36,46) / 142 null, with **0** null-translation-but-non-null-rotation rows (no partial laundering). |

The pose channel itself is clean: numeric R/t exists only on the 8
`observed_measured` frames; all 137 `unknown` + f45 + f60/75/76/77 carry null
pose. This is exactly the doctrine (report 31) contract, enforced.

---

## Part B — residual laundering (the attack result)

### R1 — HIGH: root manifest advertises two P15-posed renders that draw a body on 142 unknown frames

The root `artifact_manifest.json` has three top-level render surfaces:

- `render_artifacts` → `unknown_preserving_pose_render/` — **clean.**
- `full_duration_render` → `v19_contact_state_full_duration/v19_*.mp4` —
  **P15-posed, laundered.** `acceptance_gates` = all `true`
  (`full_duration_150_frames`, `rendered_body_replaces_trellis_by_hash`, …). **No
  `status` field, not in `superseded_artifacts`.**
- `generic_contact_state_render` → `v19_contact_state_generic_render/` —
  **P15-posed, laundered.** `acceptance_gates` = all `true`. **No `status`, in no
  superseded list, absent from `render_consumption.ndjson`.**

Both consuming scripts load the P15 pose graph, which has 150 all-numeric rows
(nearest-hold f0–29←f30, f78–149←f77 the ~197 mm outlier, Slerp/lerp between):

- `scripts/render_clip001850_v19_contact_state_full_duration.py:48`
  `POSE_REPORT = …/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`;
  `:529` `load_pose_rows()`; requires a pose row every frame.
- `scripts/render_v19_contact_state_full_duration.py:622`
  `load_pose_rows(args.pose_report)`; `:645-651` draws overlay+world with
  `pose_rows[frame_idx]` for every frame; the generic render manifest's
  `inputs.pose_report` = the same P15 report.

The renders' **own per-frame manifests** confirm the body is drawn on unknown
frames (world edges = 1649 is the observed body outline; overlay faces > 0 is the
projected body):

| frame (state) | generic world edges / overlay faces | full_duration world edges / overlay faces | unknown_preserving |
|---|---|---|---|
| f000 unknown | 1649 / 0 | 1649 / 7 | body_drawn=**false**, edges 0 |
| f090 unknown | 1649 / 1484 | 1649 / 2289 | body_drawn=**false**, edges 0 |
| f120 unknown | 1649 / 2031 | 1649 / 2031 | body_drawn=**false**, edges 0 |
| f149 unknown | 1649 / 1956 | 1649 / 1956 | body_drawn=**false**, edges 0 |
| f045 unresolved | 1649 / 1463 | 1649 / 2433 | body_drawn=**false**, edges 0 |
| f060 rejected | 1649 / 93 | 1649 / 2966 | body_drawn=**false**, edges 0 |

Visual proof (decisive), `v19_contact_state_generic_render/v19_world.mp4` f090
(state `unresolved_evidence_incomplete`): banner reads "no contact evidence is
fabricated," yet a full amber keyboard wireframe is rendered in world space with
a legend "amber dashed body = repaired observed open object patch." The contact
label is honest; the **body localization is fabricated** from the held pose. The
clean `unknown_preserving_pose_render/v19_world.mp4` f090 draws **no** body and
states "OBJECT BODY NOT DRAWN (pose unknown) … Body drawn only on measured
f30-36,f46." (Crops: `/tmp/clip001850_surface_review_35/*_world_f090.png`.)

Mechanism: identical to report 33 F1. A consumer that reads the root manifest's
`full_duration_render.overlay` or `generic_contact_state_render` plays a video
that places the keyboard at a nearest-hold/interp pose on 142 frames where pose
is `unknown` — the doctrine §4 (report 31) "on unknown draw no metric body"
violation the cleanup claimed to remove. `unknown_preserving_pose_render` already
provides the honest full-duration render, so both P15-posed renders are
**redundant and laundered**, not just laundered.

### R2 — MEDIUM: `render_consumption.ndjson` demotes only the static-gauge render; the P15 full_duration render has no status and the generic render is untracked

`render_consumption.ndjson` (9 rows):

- rows 0–2 `v19_contact_state_full_duration/v19_*.mp4` — **no `status` key**;
  full `acceptance_gates`; reads as accepted.
- rows 3–5 `static_gauge_pose_render/v19_*.mp4` —
  `status=superseded_rejected_prior_laundering`. ✓
- rows 6–8 `unknown_preserving_pose_render/v19_*.mp4` —
  `status=current_object_pose_observation_render`. ✓
- `v19_contact_state_generic_render` — **absent** (0 rows).

So the demotion is inconsistent: static-gauge is flagged, unknown-preserving is
flagged current, but the P15-posed full_duration render is left status-less
(indistinguishable from accepted), and the generic laundered render is not
tracked at all.

### R3 — MEDIUM (root cause of survival): the purity audit + graph-health render-lineage scan only the current render, not the package render surface

- `pose_provenance_purity_audit/summary.json` references none of
  `v19_contact_state_full_duration`, `generic_render`, `full_duration`.
- `graph_health.ndjson` row 0 `render_state_hash_lineage.consumed_render_manifest`
  = `unknown_preserving_pose_render/manifest.json` only.

The audit correctly proves the *current pose table* is pure and the *current
render* is body-clean. It does **not** enumerate the other advertised renders,
so it cannot detect that two P15-posed renders still ship. This is exactly the
report-33 §5 intervention ("flag any body drawn from a non-`observed_measured`
pose") applied to only one render. The audit's PASS therefore does not license
"the package render surface is clean"; it licenses "the current render is clean."

### R4 — LOW: two physical files both named `contact_frame_detail.ndjson`

`contact_state_table/contact_frame_detail.ndjson` (21 rows, de7ebc27, the render's
consumed source window; also `key_files.canonical_contact_rows`) and
`…/tables/contact_frame_detail.ndjson` (150 rows, b51c3c11, the sidecar's shipped
table) share a filename, disambiguated only by directory + the `input_hashes`
alias. Not a laundering path — both carry identical honest states (f32/f36/f45
match, no `confirmed_contact`, 150-row = 21 kill-test + 129 defaulted
`unresolved_evidence_incomplete`) — but the name collision means "canonical
contact rows" resolves to 21 or 150 rows depending on which key a consumer reads.
Recommend a distinct filename for the source-window table on disk (e.g.
`contact_frame_detail_source_window.ndjson`, matching the `input_hashes` alias).

### R5 — LOW: `motion_coupling` sidecar table is a placeholder while the real verdict exists elsewhere

`tables[]` lists `motion_coupling` `row_count:1, status:placeholder_not_produced`,
yet the real negative motion-coupling verdict
(`motion_coupling_absent_no_impulse_above_motion_floor`) exists under
`motion_coupling_contact/`. This under-ships (not launders) a real result; the
sidecar table should carry the produced verdict or point to it, so the sidecar is
not read as "motion coupling not evaluated."

---

## Exact next intervention (ordered)

The root cause is that the cleanup and its audit treat "the `render_artifacts`
pointer" as the render surface, not "every advertised render block + every
`.mp4` dir." Fix both the surface and the audit that missed it.

1. **Retire both P15-posed renders from the root manifest surface.** In the
   builder that writes `artifact_manifest.json`
   (`scripts/build_clip001850_ego_hoi_sidecar_package.py` and/or the top-level
   manifest writer): delete the `full_duration_render` and
   `generic_contact_state_render` top-level blocks, or replace each with a
   superseded stub (`status: superseded_rejected_pose_laundering`,
   `failure_mechanism: "draws object body from P15 nearest-hold/interp pose
   (v19_rigid_object_pose_graph_report.json) on 142 unknown frames"`,
   `must_not_use_for: [current_rendered_annotation, object_pose_state,
   contact_geometry, occlusion_geometry]`). Add both render dirs to root
   `superseded_artifacts`.

2. **Physically relocate** `v19_contact_state_full_duration/` and
   `v19_contact_state_generic_render/` under `superseded_prior_laundering/`
   (same structural demotion used for the static-gauge tables/summary). Structure
   beats flags — the report-31 lesson: a status string a player ignores does not
   prevent a consumer from playing the file; moving it off the consumable path
   does. `unknown_preserving_pose_render/` remains the single canonical
   full-duration render.

3. **Fix `render_consumption.ndjson`:** add
   `status: superseded_rejected_pose_laundering` to rows 0–2 (full_duration), and
   add three rows for the generic render with the same status (currently
   untracked). No consumption row should read as accepted while lacking a status
   that its siblings carry.

4. **Add the generic render to the sidecar `superseded_render_artifacts`**
   (currently only the clip-specific full_duration render is listed).

5. **Extend the render-lineage purity audit to scan the whole render surface.**
   In `audit_clip001850_pose_provenance_purity.py` /
   `build_ego_hoi_sidecar_and_graph_health.py`: enumerate every render manifest
   under the research root; for each, flag a violation if any per-frame record has
   `body_drawn` (or `world_edges_drawn>0`/`overlay_faces_drawn>0`) on a
   non-`observed_measured` frame, or if its consumed pose-source sha ≠
   `67dabc79…` (`object_pose_observations.ndjson`). Run it over the *existing*
   artifacts first: it must report the two P15-posed renders as violations
   **before** they are removed, proving the audit now covers the render surface
   (the report-33 #5 discipline, applied to all renders not one).

6. **(LOW) Rename the on-disk 21-row source-window contact table** to
   `contact_frame_detail_source_window.ndjson` to end the filename collision
   (R4), and **populate the sidecar `motion_coupling` table** with the produced
   negative verdict or a pointer (R5).

Preferred alternative to 1+2: rebuild both renders to consume
`object_pose_observations.ndjson` (null pose ⇒ no body). This is unnecessary here
because `unknown_preserving_pose_render/` already is that render; retiring the
redundant laundered pair (1–4) plus closing the audit gap (5) is the minimal,
sufficient intervention.

---

## Residual risks / boundaries

- The **contact channel is clean and pose-independent**: contact states derive
  from masked depth (kill-tests), reference no `static_gauge`/`T_rest`, and are
  identical across both contact tables. No live path was found where a held pose
  feeds a contact or nonpenetration decision. The laundering is confined to
  **body-localization rendering** advertised at the root manifest.
- **The seven named blockers are truly fixed** — I re-verified each against file
  contents and hashes, not against the manifest's self-description. The parent's
  per-blocker OPS validation claims are accurate; their *scope* omitted the two
  non-`render_artifacts` render blocks.
- R1 was verified three independent ways (consuming-script pose source, the
  renders' own per-frame body-draw manifests, and world f090 pixels for the
  generic render). The full_duration render's overlay/world use the same
  unconditional pose→body call, laundered identically; I extracted world f090
  pixels for the generic render specifically and read per-frame manifest counts
  for both.
- Read-only inspection; no pipeline stage rerun, no edits, nothing staged. The
  interventions are the parent's to apply. A crude HSV amber-fraction detector was
  discarded as non-discriminating (it caught shared banner/legend pixels); the
  per-frame body-draw manifest counts + the direct image read are the load-bearing
  evidence.
- The audit gap (R3) is the mechanism to fix, not just the symptom (R1/R2):
  without a package-wide render-surface audit, the next artifact that adds a
  render will recreate this class of residual.

---

## Reproduce (read-only)

```bash
cd /home/yiwen/ego_annotation
R=/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706
SC=$R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research

# Part A — blockers fixed
python3 -c "import json;m=json.load(open('$SC/manifest.json'));print(m['render_artifacts']['world']['path']);print([t['name'] for t in m['tables']])"
grep -c "static_gauge_held_rest\|T_rest" $SC/tables/object_pose_visibility_ledger.ndjson   # 0
find $R -name '*static_pose_hypothesis*'                                                     # empty
ls $SC/ | grep static_gauge_summary || echo ABSENT                                          # ABSENT
python3 -c "import json;[print(list(json.loads(l))) for l in open('$SC/tables/graph_health.ndjson')]" | head
# row counts + hashes: see script below
for f in $R/contact_state_table/contact_frame_detail.ndjson $SC/tables/contact_frame_detail.ndjson; do echo "$(wc -l <$f) $(sha256sum $f|cut -c1-12) $f"; done

# Part B — residual laundering
python3 -c "import json;m=json.load(open('$R/artifact_manifest.json'));print('render_artifacts.world =',m['render_artifacts']['world']['path']);print('full_duration_render.world =',m['full_duration_render']['world']);print('generic root =',m['generic_contact_state_render']['root']);print('superseded_artifacts names =',[a.get('name') for a in m['superseded_artifacts']])"
# both laundered renders draw body on unknown frames (their own manifests):
python3 -c "import json;m=json.load(open('$R/v19_contact_state_generic_render/manifest.json'));fr={x['frame_idx']:x for x in m['frames']};print({fi:(fr[fi]['world_render_info']['observed_body_edges_drawn'],fr[fi]['overlay_render_info']['observed_body_faces_drawn']) for fi in (0,90,120,149)})"
sed -n '48p;529p' scripts/render_clip001850_v19_contact_state_full_duration.py
sed -n '622p;645,651p' scripts/render_v19_contact_state_full_duration.py
# render_consumption: rows 0-2 have no status, generic absent
python3 -c "import json;[print(i,r.get('status','<NO STATUS>'),'|', (r.get('render') or r.get('consumed_render_manifest') or '')[-55:]) for i,r in enumerate(json.loads(l) for l in open('$SC/tables/render_consumption.ndjson'))]"
# audit does not scan laundered renders
python3 -c "import json;b=json.dumps(json.load(open('$R/pose_provenance_purity_audit/summary.json')));print('full_duration in audit:', 'full_duration' in b, '| generic in audit:', 'generic' in b)"

# visual: amber keyboard body on unknown f090 in generic render vs none in unknown-preserving
mkdir -p /tmp/clip001850_surface_review_35
for n in unknown_preserving_pose_render v19_contact_state_generic_render; do ffmpeg -y -loglevel error -i $R/$n/v19_world.mp4 -vf "select=eq(n\,90)" -vframes 1 /tmp/clip001850_surface_review_35/${n}_world_f090.png; done
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Verified all seven named blockers FIXED against file contents/hashes (not manifest self-description): F1 render_artifacts in sidecar+root point to unknown_preserving_pose_render with consumed_object_pose_observations_sha256=67dabc79 and body_drawn_only_on_measured_frames=true, render_consumption rows 6-8 status current; F2 sidecar tables/ has no static-gauge file and object_pose_visibility_ledger.ndjson has 0 static_gauge/T_rest tokens (statics moved to superseded_prior_laundering/sidecar_tables/); F3 find -name '*static_pose_hypothesis*' = 0 hits and 0 manifest refs (orphan_pose_dir empty); F4 static_gauge_summary.json absent from sidecar root (moved to superseded_prior_laundering/sidecar_root/); F5 graph_health.ndjson 2 rows, row0 render_state_hash_lineage non-null consuming object_pose_observations 67dabc79, row1 pose_provenance_purity decision pose_channel_pure_current__original_laundering_detected; F6 all tables[] row_count==wc-l and all shipped-table sha256==input_hashes; contact unchanged f32=geometry_epoch_contaminated/f36=full_frame_depth_leak/f45=unresolved_incoherent_evidence identical in 21-row and 150-row tables, object_pose_observations 8 numeric/142 null with 0 partial-laundering rows. Then attacked remaining laundering and found a HIGH residual with file paths+severity: root artifact_manifest.json still advertises full_duration_render (v19_contact_state_full_duration/) and generic_contact_state_render (v19_contact_state_generic_render/) as top-level render blocks with all acceptance_gates=true and no supersession; both consume the 150-row all-numeric P15 pose graph (render_clip001850_v19_contact_state_full_duration.py:48 POSE_REPORT; render_v19_contact_state_full_duration.py:622/:645-651) and draw the keyboard body on 142 unknown frames (their own per-frame manifests: world_body_edges=1649 + overlay faces up to 2966 on f0/f90/f120/f149/f45/f60; generic world f090 pixels show a full amber keyboard wireframe under a 'no contact fabricated' banner). The generic render is flagged superseded nowhere and is absent from render_consumption; the full_duration render's render_consumption rows 0-2 carry no status while siblings do; the pose-provenance-purity audit and graph-health render-lineage scan only the clean render (audit summary references neither laundered render), which is why the residual survived. Provided the exact ordered next intervention."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read PROMPT.md/EPISTEMIC.md/TASK_PACK.md/graph_liveness_and_motion_coupling.md/OPS.md/subagents 31,32,33; enumerate package tree",
      "result": "passed",
      "summary": "Loaded full task + cleanup context; enumerated package dirs incl superseded_prior_laundering/{sidecar_tables,sidecar_root,orphan_pose_dir}."
    },
    {
      "command": "verify F1-F6 + contact regression against file contents/hashes (sidecar manifest, root manifest, tables/, graph_health.ndjson, contact tables, object_pose_observations)",
      "result": "passed",
      "summary": "All seven named blockers verified fixed: render_artifacts unknown-preserving; static-gauge/visibility-ledger/summary/hypothesis absent from consumable surface; graph_health lineage non-null + purity row; all row_count==wc-l and sha256==input_hashes; contact f32/f36/f45 unchanged and pose-independent; pose table 8 numeric/142 null with no partial laundering."
    },
    {
      "command": "attack render surface: root manifest render blocks, both render scripts pose source, both laundered renders' per-frame body-draw manifests, render_consumption statuses, audit scope, ffmpeg world f090 crops + image read",
      "result": "passed",
      "summary": "Found HIGH residual: two P15-posed renders (full_duration + generic) advertised in root manifest with green gates and no supersession, drawing body on 142 unknown frames (triple-confirmed: script pose source, per-frame manifest counts, generic world-f090 pixels). Generic render superseded nowhere + untracked in render_consumption; full_duration render_consumption rows 0-2 status-less; purity audit/graph-health lineage scan only the clean render."
    }
  ],
  "validationOutput": [
    "F1 FIXED: sidecar+root render_artifacts.{overlay,world,side_by_side}.path = unknown_preserving_pose_render/v19_*.mp4; consumed_object_pose_observations_sha256=67dabc7998766d6c...; body_drawn_only_on_measured_frames=true; render_consumption rows6-8 status=current_object_pose_observation_render.",
    "F2 FIXED: tables/ has no static/gauge file; object_pose_visibility_ledger.ndjson 0 static_gauge/T_rest tokens; tables[] omits object_pose_static_gauge + object_visibility_ledger.",
    "F3 FIXED: find -name '*static_pose_hypothesis*' -> 0; 0 manifest references; orphan_pose_dir empty.",
    "F4 FIXED: sidecar root = {manifest.json, rest_cluster_reference_for_outlier_rejection.json, tables/}; static_gauge_summary.json only under superseded_prior_laundering/sidecar_root/.",
    "F5 FIXED: graph_health.ndjson 2 rows; row0 render_state_hash_lineage.consumed_pose_source=object_pose_observations.ndjson sha 67dabc79 (non-null); row1 pose_provenance_purity decision pose_channel_pure_current__original_laundering_detected render_state_hash=c0ac8ab9.",
    "F6 FIXED: contact_frame_detail 150==150, render_consumption 9==9, object_pose_observations 150==150, object_pose_visibility_ledger 150==150, motion_coupling 1==1, geometry_epochs 6, graph_health 2; all shipped sha256==input_hashes; contact split de7ebc27(21)/b51c3c11(150) both hashed.",
    "contact unchanged: f32=geometry_epoch_contaminated, f36=full_frame_depth_leak, f45=unresolved_incoherent_evidence in both tables; 0 confirmed_contact/static_gauge/T_rest; object_pose_observations 8 numeric(f30-36,46)/142 null, 0 null-t-nonnull-R rows.",
    "R1 HIGH: root full_duration_render + generic_contact_state_render advertised (acceptance_gates all true, no status); both load P15 pose report (150 numeric rows); per-frame manifests show world_body_edges=1649 + overlay faces on unknown f0/f90/f120/f149/f45/f60; generic world f090 pixels = amber keyboard wireframe under 'no contact fabricated' banner; unknown_preserving f090 body_drawn=false, 'OBJECT BODY NOT DRAWN'.",
    "R2 MEDIUM: render_consumption rows0-2 (full_duration) have no status; rows3-5 static_gauge superseded; rows6-8 unknown_preserving current; generic render absent (0 rows).",
    "R3 MEDIUM root-cause: pose_provenance_purity_audit/summary.json references neither laundered render; graph_health render lineage consumes only unknown_preserving manifest -> audit scoped to current render, cannot catch R1.",
    "R4 LOW: two physical files named contact_frame_detail.ndjson (21-row de7ebc27 canonical vs 150-row b51c3c11 sidecar), disambiguated only by dir+input_hashes alias; honest states, no laundering.",
    "R5 LOW: sidecar motion_coupling table status=placeholder_not_produced while real negative verdict exists under motion_coupling_contact/."
  ],
  "residualRisks": [
    "Contact channel is clean and pose-independent; no live path found where a held pose feeds a contact/nonpenetration decision. Residual laundering is confined to body-localization renders advertised at the root manifest (R1/R2).",
    "The audit gap (R3) is the mechanism, not just the symptom: without a package-wide render-surface purity audit, the next added render will recreate this residual class. The exact fix must extend the audit to enumerate every render manifest and flag body_drawn on non-observed_measured frames or a consumed pose-source sha != 67dabc79.",
    "R1 verified three ways (consuming-script pose source, both renders' per-frame body-draw manifests, generic world-f090 pixels). full_duration overlay/world use the same unconditional pose->body call so are laundered identically; pixels extracted for generic f090 specifically.",
    "Read-only inspection; no pipeline rerun, no edits, nothing staged. Interventions (retire/relocate both P15-posed renders, add statuses in render_consumption, add generic to sidecar superseded_render_artifacts, extend audit to full render surface, rename source-window contact table, populate motion_coupling) are the parent's to apply.",
    "A crude HSV amber-fraction heuristic was discarded as non-discriminating (caught shared banner/legend pixels); load-bearing evidence is per-frame body-draw manifest counts + direct image read."
  ],
  "noStagedFiles": true,
  "diffSummary": "No code or artifact changes. Findings written only to the authoritative report path 35_clean_package_surface_review.md.",
  "reviewFindings": [
    "no blocker on the seven named items: F1 render_artifacts unknown-preserving (sidecar+root), F2 static-gauge rows absent from sidecar tables/, F3 static_pose_hypothesis deleted, F4 static_gauge_summary absent from sidecar root, F5 graph_health render lineage non-null + pose_provenance_purity row present, F6 all row counts/hashes consistent, contact f32/f36/f45 unchanged and pose-independent - all verified against file contents and hashes.",
    "high: root artifact_manifest.json still advertises full_duration_render (v19_contact_state_full_duration/) and generic_contact_state_render (v19_contact_state_generic_render/) as top-level render blocks with all acceptance_gates=true and no supersession marker; both consume the 150-row all-numeric P15 pose graph (render_clip001850_v19_contact_state_full_duration.py:48; render_v19_contact_state_full_duration.py:622/:645-651) and draw the keyboard body on 142 unknown frames (per-frame manifests world_body_edges=1649 + overlay faces on f0/f90/f120/f149/f45/f60; generic world-f090 pixels show an amber keyboard wireframe under a 'no contact evidence is fabricated' banner). This is report-33 F1 only partially retracted - the correction fixed the render_artifacts pointer but not the other two advertised render surfaces.",
    "medium: render_consumption.ndjson rows 0-2 (v19_contact_state_full_duration) carry no status field while sibling rows 3-5 (static_gauge=superseded) and 6-8 (unknown_preserving=current) do; the generic laundered render is absent from render_consumption entirely and from every superseded list (root superseded_artifacts, sidecar superseded_render_artifacts).",
    "medium (root cause): pose_provenance_purity_audit + graph_health render_state_hash_lineage scan only the current unknown_preserving render (audit summary references neither laundered render), so they report PASS while two P15-posed renders ship advertised; the audit must be extended to the whole package render surface.",
    "low: two physical files named contact_frame_detail.ndjson (21-row de7ebc27 canonical/key_files vs 150-row b51c3c11 sidecar) - honest states, name collision only; rename the source-window file to match its input_hashes alias.",
    "low: sidecar motion_coupling table is status=placeholder_not_produced while the real negative motion-coupling verdict exists under motion_coupling_contact/ - under-ships a produced result."
  ],
  "manualNotes": "Verdict: the seven named blockers are genuinely fixed (verified against file contents/hashes, not manifest self-claims); the pose channel is pure (8 numeric/142 null) and contact is unchanged and pose-independent. The remaining problem is the SAME F1 laundering, only partially retracted: the parent fixed the render_artifacts pointer and demoted one of two P15-posed renders in the SIDECAR manifest, but the ROOT artifact_manifest.json still advertises BOTH the clip-specific (full_duration_render) and the generic (generic_contact_state_render) P15-posed renders as accepted top-level deliverables that draw a fabricated keyboard body on 142 unknown frames. Root cause of survival: the purity audit + graph-health render-lineage only scan the one clean render. Exact next intervention, ordered: (1) delete or superseded-stub the full_duration_render + generic_contact_state_render blocks in the root manifest and add both dirs to root superseded_artifacts; (2) physically move both render dirs under superseded_prior_laundering/ (structure beats flags); (3) add status=superseded_rejected_pose_laundering to render_consumption rows 0-2 and add the generic render's rows; (4) add the generic render to sidecar superseded_render_artifacts; (5) extend audit_clip001850_pose_provenance_purity.py to enumerate EVERY render manifest and flag body_drawn on non-observed_measured frames or consumed pose sha != object_pose_observations (67dabc79), and prove it flags the two P15 renders before removal; (6 low) rename the 21-row source-window contact file and populate the motion_coupling sidecar table. unknown_preserving_pose_render/ already is the honest full-duration render, so the two P15-posed renders are redundant as well as laundered - retire them."
}
```
