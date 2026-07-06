# 33 — Fresh-context attack on the corrected object-pose semantics + R0/R1 provenance-purity audit

Review only. No edits, nothing staged. Branch `yiwen_research`. Case: HOT3D
clip001850 keyboard, right hand. All claims below were verified on disk this
session (commands in §Reproduce), not taken from reports 30/31.

## Bottom line

The correction is **additive, not substitutive**, and it is **not gated**. Subagent 30
built a clean parallel pose table + clean render, and subagent 31 wrote the
doctrine. But the dirty artifacts were left in place **and remain the
authoritative package surface**:

- The sidecar's **primary** `render_artifacts` (top-level, the one a consumer
  plays) is still the P15-posed contact render, which **draws a localized
  keyboard body on all 150 frames**, including the 142 the corrected pose state
  calls `unknown`. Verified visually at f090 (§F1).
- The durable `ego_hoi_sidecar/.../tables/` directory **still ships the
  static-gauge pose table (150 all-numeric T_rest rows)** and the old visibility
  ledger, listed in the manifest `tables[]` (§F2).
- An **orphan `static_pose_hypothesis.json`** (schema `ego.hoi.static_pose_hypothesis/0.1.0`,
  role "candidate prior for a future factor graph", embedded T_rest) sits inside
  the *current* pose dir and is referenced by the root manifest. No current
  script writes it; subagent 30 never reported it (§F3).
- `static_gauge_summary.json` lives unflagged in the sidecar root and advertises
  "static-held gauge (KT-L1 route)" with `render_lineage.pose_source_consumed_by_renderer =
  object_pose_static_gauge.ndjson` and frames the static fill as `defect_repaired` (§F4).
- The **R0/R1 provenance-purity audit does not exist** and the render-lineage
  hook that would catch F1 is a null stub. That is *why* F1–F4 survived: nothing
  measures them (§F5).

What is genuinely clean (scope precisely): `object_pose_observations.ndjson`
(8 numeric / 142 null), the `unknown_preserving_pose_render/` videos (body only
on measured frames, world-view amber 1.66% measured vs 0.022% unknown),
`rest_cluster_reference_for_outlier_rejection.json` (properly demoted), and the
contact **state labels** (`contact_frame_detail`) which are derived from masked
depth and are pose-graph-independent (§F6 note). The defect is that these clean
artifacts did not *replace* the laundered ones; they coexist, and the laundered
ones are advertised as primary.

---

## Findings

### F1 — BLOCKER: sidecar's primary render draws a localized body on 142 unknown frames from the P15 pose graph

The sidecar `manifest.json` top-level `render_artifacts.{overlay,world,side_by_side}`
points at `v19_contact_state_full_duration/v19_*.mp4`. That render is produced by
`scripts/render_clip001850_v19_contact_state_full_duration.py`, which:

- loads pose from the **P15 graph report** (`:48` `POSE_REPORT = .../keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`; `:143` `load_pose_rows`),
- calls `observed_body_world(body, pose_row)` **unconditionally every frame**
  (`:294` overlay, `:349` world) with **no visibility / is_measured / null
  gate** (`:167-170` uses only `rotation_world_from_completed_canonical_matrix`
  + `translation_world_m`),
- **requires** a pose row for every frame (`:546-547` raises if missing).

The P15 report has **150 pose rows, all numeric** (verified). f78–149 all hold
`t=[0.390,-0.110,0.344]` — the f77 mask-drift outlier, **~197 mm off rest**
(nearest_visible_pose_hold). f0–29 hold the f30 pose.

**Visual proof (decisive).** `v19_contact_state_full_duration/v19_world.mp4`
frame f090 (corrected state = `unknown`, null pose, "no body"): banner reads
`contact_state=unresolved_evidence_incomplete` / "no contact evidence is
fabricated", yet a large **amber dashed keyboard body** is rendered (legend:
"amber dashed body = repaired observed open keyboard patch"). The banner is
honest about contact; the **body localization is fabricated** — the keyboard is
placed at a world pose on a frame where pose is unknown. Amber-pixel fraction
(world view): laundered render f090=0.090%, f120=0.191%, f149=0.170% vs the clean
`unknown_preserving_pose_render` f090=f120=f149=0.022% (background). This is
exactly what doctrine §4 (report 31) forbids: "On unknown / observed_rejected /
unresolved: draw no metric body."

Severity BLOCKER: this is the sidecar's *advertised primary render*. A consumer
taking the canonical `render_artifacts` sees the same prior-laundering the
correction claims to have removed — the correction only added a *second* clean
render (`unknown_preserving_render_artifacts`, a secondary manifest block).

### F2 — HIGH: sidecar `tables/` still ships the static-gauge pose table (150 numeric rows) + old visibility ledger, listed in `tables[]`

`ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/tables/`
contains `object_pose_static_gauge.ndjson` (208 KB, **150 rows, all with numeric
`translation_world_m = T_rest` and `pose_source=static_gauge_held_rest`,
`is_measured=false`**) and `object_visibility_ledger.ndjson`. The sidecar
`manifest.json` `tables[]` lists both as package tables
(`status: superseded_rejected_prior_laundering`, `may_use_for: negative_evidence_only`).

Why the status flag is insufficient: report 31 finding A (re-verified) proved the
delivered renderers key on `translation_world_m` and **read no status/is_measured
field**. Any consumer that iterates `tables/*.ndjson` or picks the
`object_pose_static_gauge` table gets 150 confident numeric poses. The rows carry
no in-file poison marker beyond `is_measured:false` (which is ignored) — the row
schema string is a benign `ego.hoi.object_pose_static_gauge/0.1.0`. Two pose
tables and two visibility ledgers coexist in the same canonical `tables/` dir,
distinguished only by filename and a manifest status string.

### F3 — HIGH: orphan `static_pose_hypothesis.json` in the current pose dir, named "hypothesis / candidate prior", referenced by the root manifest

`unknown_preserving_pose_render/static_pose_hypothesis.json` (mtime 09:17:31,
sha 42d59383…): schema `ego.hoi.static_pose_hypothesis/0.1.0`,
`hypothesis_not_measurement:true`, `role_statement`: "This object is a
static-rest **POSE HYPOTHESIS**… a **candidate prior for a future factor
graph**", embedded T_rest value.

- **No current script writes it** (`grep -rl static_pose_hypothesis scripts/`
  empty). The current `render_clip001850_unknown_preserving_pose_artifact.py`
  emits only `object_pose_observations / rest_cluster_reference_for_outlier_rejection /
  visibility_ledger / manifest`. It is a stale leftover from an earlier script
  iteration, never cleaned.
- It is referenced by the **root `artifact_manifest.json`** `files[]` as a plain
  hashed package file (no superseded flag).
- Subagent 30's report does not mention it at all — it claims the only retained
  reference is `rest_cluster_reference_for_outlier_rejection.json` "explicitly not
  named hypothesis or prior". This file directly contradicts that claim and
  doctrine §1/§5.3 ("static-ness must emerge as an R4 posterior, not be injected
  as T_rest"; priors are "chosen by the model, not inherited local patches").

Severity HIGH: a pre-computed T_rest carrying the literal status "candidate prior
for a future factor graph" is the exact object the doctrine bans, sitting in the
*current* pose directory, blessed by the root manifest. If R4/R7 (or any
consumer) globs the pose dir for a prior, this is the laundering entry point.

### F4 — HIGH: `static_gauge_summary.json` in the sidecar root advertises static fill as the render pose source and as a "repair"

`ego_hoi_sidecar/.../static_gauge_summary.json` (sha e0e01c25…, **no in-file
superseded marker**):
- `regime_verdict: "static-held gauge (KT-L1 route)"`,
- `render_lineage.pose_source_consumed_by_renderer: "object_pose_static_gauge.ndjson"`,
  `lineage_statement: "the renderer places the object body exclusively from
  object_pose_static_gauge.ndjson (T_rest held gauge)"`,
- `scope.pose_rows_are_static_gauge: true`,
- `defect_repaired`: frames replacing the f77-outlier hold with T_rest as a
  *repair* (`body_displacement_change_mm: 196.8`).

The sidecar manifest `input_hashes` records this only as
`static_gauge_summary.json_superseded_negative_evidence`, but the **file itself,
present in the sidecar root, reads as a live verdict** that static-gauge is the
render pose source. The root `artifact_manifest.json` lists it as a plain file.
This is a manifest/summary field that still advertises static fill.

### F5 — BLOCKER (for the audit itself): the R0/R1 provenance-purity audit is unimplemented; the render-lineage hook is a null stub

`graph_health.ndjson` (1 row) union keys: support_fraction_min_critical,
active_residual_count_total, factor_family_health, stale_dependency_count,
gauge_declaration_count, input_output_deltas, cross_solver_geometry_consistency,
geometry_epoch_lineage, render_state_hash_lineage, mechanism_decision, … . It
contains **none** of: provenance_purity, held/nearest/interp/imputed,
static_gauge, t_rest, measurement_factor_provenance, pose_source,
observed_measured, solver_estimate, unknown. `scripts/build_ego_hoi_sidecar_and_graph_health.py`
has no such logic either (grep empty).

`render_state_hash_lineage` = `{graph_output_hash: …, render_state_hash: null,
pending: "renderer must write its consumed-state fingerprint here"}`. The
render-lineage-purity check proposed in report 31 §5.7/R1 — "flag any body drawn
from a non-observed_measured / non-solver_estimate pose" — cannot fire because
the renderer never records its consumed pose source and the audit does not read
it.

Severity BLOCKER for the R0/R1 milestone: the *entire mechanism whose job is to
detect F1–F4* is absent. The correction added clean artifacts but not the gate
that would have flagged the dirty artifacts still being advertised. This is the
root cause of F1–F4 surviving.

### F6 — MEDIUM/HIGH: schema does not structurally separate observation / estimate / unknown; several count and hash inconsistencies

- **No observation-vs-estimate record type.** Doctrine §3.2 requires observation
  rows and solver-estimate rows to be different record types. The schema still
  has a single `object_pose*` shape; `solver_estimate` (doctrine class) has no
  table or record type. Separation is by filename + manifest status only.
- **Numeric pose with `is_measured:false`.** The static-gauge rows carry full
  numeric R/t plus `is_measured:false` — the "flagged placeholder" the doctrine
  says is unrepresentable-as-unknown because renderers ignore the flag (proven).
- **Contact-table provenance split.** Canonical
  `contact_state_table/contact_frame_detail.ndjson` = **21 rows**, sha de7ebc27
  (kill-test window). Sidecar `tables/contact_frame_detail.ndjson` = **150 rows**,
  sha b51c3c11 (full-duration expansion). The sidecar `input_hashes` and every
  render's `consumed_contact_state_table_sha256` reference **de7ebc27 (21-row)**,
  but the sidecar ships **b51c3c11 (150-row)** under the same logical table name.
  Two different contact tables, same name; the shipped one is not the one the
  renders consumed.
- **Manifest count integrity.** Sidecar `tables[]` declares `render_consumption`
  `row_count: 3`; the file has **9 rows** (3 contact + 3 static-gauge superseded
  + 3 unknown-preserving).

---

## Exact checks (runnable gates the parent can enforce)

1. **No render places a body from the P15 pose graph on unknown frames.**
   ```bash
   grep -n "v19_rigid_object_pose_graph_report.json" scripts/render_clip001850_v19_contact_state_full_duration.py scripts/render_v19_contact_state_full_duration.py
   # EXPECT: empty (renders must consume object_pose_observations.ndjson, which is null on unknown frames)
   ```
2. **Authoritative render is body-clean on unknown frames (pixel gate).**
   ```bash
   # world-view amber fraction on f0/f90/f120 must match background (~0.02%), not 0.09-0.19%
   # laundered render currently: f090=0.090% f120=0.191%  ->  FAIL
   # clean render currently:     f090=0.022% f120=0.022%  ->  PASS
   ```
3. **Authoritative pose table: numeric rows == observed_measured count.**
   ```bash
   python - <<'PY'
   import json
   rows=[json.loads(l) for l in open('.../tables/object_pose_observations.ndjson')]
   num=sum(1 for r in rows if r.get('translation_world_m') is not None)
   assert num==8 and len(rows)==150, (num,len(rows))
   PY
   ```
4. **No static-gauge / held pose in any current-status table.**
   ```bash
   grep -l "static_gauge_held_rest" .../tables/*.ndjson   # EXPECT: no CURRENT-status table
   # and object_pose_static_gauge.ndjson / object_visibility_ledger.ndjson must NOT be listed in tables[] as tables
   ```
5. **No `static_pose_hypothesis` object in any current output dir.**
   ```bash
   find /data2/.../research_clip001850_contact_state_20260706 -name '*static_pose_hypothesis*'   # EXPECT: empty
   grep -rn "static_pose_hypothesis" .../artifact_manifest.json                                   # EXPECT: empty
   ```
6. **graph_health render-lineage purity is live, not a stub.**
   ```bash
   python -c "import json;r=json.loads(open('.../tables/graph_health.ndjson').readline());\
   l=r['render_state_hash_lineage'];assert l.get('render_state_hash') and 'pending' not in l"
   # and the recorded consumed pose-source sha must equal object_pose_observations.ndjson sha, not the P15 report sha
   ```
7. **Manifest count/hash integrity.**
   ```bash
   # for every tables[] entry: declared row_count == wc -l (render_consumption currently 3 vs 9 -> FAIL)
   # contact_frame_detail: input_hashes sha must equal the shipped tables/ file sha (currently de7ebc27 vs b51c3c11 -> FAIL)
   ```

---

## Required next interventions (ordered)

1. **Retract the laundered primary render.** Either (a) rebuild
   `v19_contact_state_full_duration` to place the body only on `observed_measured`
   frames by consuming `object_pose_observations.ndjson` (null pose ⇒ no body)
   instead of the P15 `pose_report`; or (b) repoint the sidecar top-level
   `render_artifacts` to `unknown_preserving_pose_render/` and demote the P15-posed
   render to `superseded_artifacts`. Option (a) is preferred so the *contact*
   annotation and *pose* annotation live in one honest render. Until then the
   sidecar's advertised render fabricates keyboard localization on 142 frames.
2. **Remove static-gauge rows from the canonical `tables/` dir.** Delete
   `tables/object_pose_static_gauge.ndjson` and `tables/object_visibility_ledger.ndjson`,
   or move them under a `superseded/` subtree that is not named `tables/` and not
   listed in `tables[]`. Superseded negative evidence belongs in OPS/EPISTEMIC or
   a clearly non-consumable path, not in the schema table directory.
3. **Delete the orphan `static_pose_hypothesis.json`** and drop its root-manifest
   reference. A static prior, if wanted, is an R4 model factor with calibrated
   noise that produces a `solver_estimate` posterior — not a precomputed T_rest
   file named "hypothesis".
4. **Remove or in-file-flag `static_gauge_summary.json`** in the sidecar root so
   no live document advertises `object_pose_static_gauge.ndjson` as the render
   pose source or frames the static fill as a repair.
5. **Implement the R0/R1 provenance-purity audit** in
   `build_ego_hoi_sidecar_and_graph_health.py`: (a) measurement-factor-provenance
   count (must be 0 for non-`observed_measured`), (b) held/imputed/T_rest pose
   count (must be 0), (c) render-lineage-purity: capture each render's consumed
   pose-source sha (wire the currently-null `render_state_hash`) and flag any body
   drawn from a pose whose provenance is not `observed_measured`/`solver_estimate`.
   Run it over the *existing* artifacts first — it must report F1–F4 as nonzero,
   proving the audit works.
6. **Add the observation/estimate/unknown record-type separation to the R0
   schema** (doctrine §3): `unknown` = null pose only; numeric pose forbidden
   unless `observed_measured` or `solver_estimate`; `solver_estimate` is a
   distinct record type with posterior covariance.
7. **Fix count/hash integrity** (F6): one authoritative contact table; manifest
   `row_count` == file rows; `input_hashes` sha == shipped file sha.

---

## Residual risks / boundaries

- Contact **state labels** are clean and pose-graph-independent (no
  contact/killtest/body-repair script references static_gauge/T_rest; contact is
  `unresolved` everywhere). The laundering is confined to (i) body-localization
  rendering and (ii) stored pose/prior/manifest artifacts — I did not find a live
  path where T_rest feeds a contact or nonpenetration *decision*.
- I verified F1 visually only in the world view at f090/f120; the overlay view of
  the same render uses the same unconditional `observed_body_world` call, so it is
  laundered by the same mechanism, but I did not extract overlay pixels for every
  unknown frame.
- I did not re-run any pipeline stage; all findings are read-only inspection of
  on-disk artifacts + code regions. Deleting/rebuilding per §Interventions is the
  parent's call (I made no edits).
- The clean artifacts (`object_pose_observations.ndjson`,
  `unknown_preserving_pose_render`, `rest_cluster_reference_for_outlier_rejection.json`)
  are correct; the risk is purely that they coexist with, and are outranked by,
  the dirty ones in the authoritative package.

## Reproduce (read-only)

```bash
cd /home/yiwen/ego_annotation
R=/data2/ego_annotation_outputs/research_clip001850_contact_state_20260706
# F1 pose source + unconditional body draw
sed -n '48p;143p;167,170p;294p;349p;546,547p' scripts/render_clip001850_v19_contact_state_full_duration.py
python -c "import json;pr=json.load(open('/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1/measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json'))['pose_rows'];print(len(pr),[r['translation_world_m'] for r in pr if r['frame_idx'] in (90,120)])"
# F1 visual: /tmp/pose_attack_crops/CONTACT_world_f090.jpg (amber body on unknown frame)
# F2 static-gauge table still in tables[]
python -c "import json;m=json.load(open('$R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/manifest.json'));print([t['name'] for t in m['tables']])"
head -1 $R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/tables/object_pose_static_gauge.ndjson
# F3 orphan hypothesis
cat $R/unknown_preserving_pose_render/static_pose_hypothesis.json | python -m json.tool | head -8
grep -rl static_pose_hypothesis scripts/   # empty
# F4 static_gauge_summary advertises render pose source
python -c "import json;print(json.load(open('$R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/static_gauge_summary.json'))['render_lineage']['pose_source_consumed_by_renderer'])"
# F5 audit absent + null render lineage
grep -ni "provenance_pur\|measurement_factor_provenance\|observed_measured\|solver_estimate" scripts/build_ego_hoi_sidecar_and_graph_health.py  # empty
python -c "import json;print(json.loads(open('$R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/tables/graph_health.ndjson').readline())['render_state_hash_lineage'])"
# F6 contact table split + render_consumption count
wc -l $R/contact_state_table/contact_frame_detail.ndjson $R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/tables/contact_frame_detail.ndjson
wc -l $R/ego_hoi_sidecar/extensions/org.ego.hoi/0.1.0/clip001850_research/tables/render_consumption.ndjson
```

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Six findings with file paths and severity. F1 BLOCKER: sidecar primary render_artifacts = v19_contact_state_full_duration/*.mp4, produced by scripts/render_clip001850_v19_contact_state_full_duration.py which loads P15 pose graph (:48/:143) and draws observed_body_world unconditionally every frame (:294/:349, no visibility gate :167-170, requires pose row :546); P15 report has 150 numeric rows, f78-149 hold f77 outlier ~197mm off rest; visual proof /tmp/pose_attack_crops/CONTACT_world_f090.jpg shows amber keyboard body on unknown frame f090 (amber world-view f090=0.090% vs clean render 0.022%). F2 HIGH: sidecar tables/object_pose_static_gauge.ndjson (150 numeric T_rest rows) + object_visibility_ledger.ndjson still listed in manifest tables[]. F3 HIGH: orphan unknown_preserving_pose_render/static_pose_hypothesis.json (schema ego.hoi.static_pose_hypothesis/0.1.0, role 'candidate prior for a future factor graph', embedded T_rest) referenced by root artifact_manifest.json, written by no current script, unreported by subagent 30. F4 HIGH: ego_hoi_sidecar/.../static_gauge_summary.json advertises render_lineage.pose_source_consumed_by_renderer=object_pose_static_gauge.ndjson and defect_repaired framing, no in-file superseded marker. F5 BLOCKER: R0/R1 provenance-purity audit absent from build_ego_hoi_sidecar_and_graph_health.py and graph_health.ndjson; render_state_hash_lineage is a null stub. F6 MEDIUM/HIGH schema ambiguity: no observation/solver_estimate record type; static-gauge rows carry numeric R/t with ignored is_measured=false; contact table split (canonical 21-row de7ebc27 vs sidecar 150-row b51c3c11 same name); render_consumption declared row_count 3 vs actual 9."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read PROMPT.md/EPISTEMIC.md/TASK_PACK.md/graph_liveness_and_motion_coupling.md/subagents 30,31; ls research output tree",
      "result": "passed",
      "summary": "Loaded full task context; enumerated 181-file research output tree."
    },
    {
      "command": "grep pose source + body-draw logic in render_clip001850_v19_contact_state_full_duration.py; inspect P15 report pose_rows",
      "result": "passed",
      "summary": "Confirmed contact render loads P15 pose graph and draws observed_body_world unconditionally on all 150 frames; P15 has 150 numeric rows, f90/f120/f149 hold f77 outlier ~197mm off rest."
    },
    {
      "command": "python amber-fraction on world videos + extract /tmp/pose_attack_crops/CONTACT_world_f090.jpg",
      "result": "passed",
      "summary": "Laundered render draws amber keyboard body on unknown f090 (0.090%) vs clean unknown-preserving render (0.022%); visual crop confirms localized body on unknown frame."
    },
    {
      "command": "read sidecar manifest.json, static_gauge_summary.json, render_consumption.ndjson, artifact_manifest.json, static_pose_hypothesis.json",
      "result": "passed",
      "summary": "Sidecar tables[] still lists static-gauge tables; static_gauge_summary advertises static render lineage; render_consumption has 9 rows (manifest says 3); root manifest lists static/hypothesis files unflagged; hypothesis file is orphan (no writer) referenced by root manifest."
    },
    {
      "command": "grep provenance-purity in build_ego_hoi_sidecar_and_graph_health.py + graph_health.ndjson keys; wc contact tables",
      "result": "passed",
      "summary": "Provenance-purity audit absent; render_state_hash_lineage is null stub; contact table split 21-row de7ebc27 vs 150-row b51c3c11; contact-state scripts do not reference static_gauge/T_rest (contact labels clean)."
    }
  ],
  "validationOutput": [
    "F1: render_clip001850_v19_contact_state_full_duration.py:48 loads P15 pose_report; :294/:349 call observed_body_world every frame; :167-170 no visibility gate; :546 requires pose row. P15 pose_rows=150 all numeric; f90/f120/f149 t=[0.390,-0.110,0.344] (~197mm off rest). Visual: amber body on unknown f090.",
    "F2: sidecar tables/ ships object_pose_static_gauge.ndjson (150 rows, translation_world_m=T_rest, pose_source=static_gauge_held_rest, is_measured=false) + object_visibility_ledger.ndjson; both in manifest tables[] with status superseded but present and numeric.",
    "F3: unknown_preserving_pose_render/static_pose_hypothesis.json schema ego.hoi.static_pose_hypothesis/0.1.0, role='candidate prior for a future factor graph', embeds T_rest; no script writes it; referenced by root artifact_manifest.json; not mentioned in subagent 30 report.",
    "F4: ego_hoi_sidecar/.../static_gauge_summary.json regime_verdict 'static-held gauge (KT-L1 route)', render_lineage.pose_source_consumed_by_renderer=object_pose_static_gauge.ndjson, defect_repaired framing, no in-file superseded flag.",
    "F5: build_ego_hoi_sidecar_and_graph_health.py has no provenance_purity/measurement_factor_provenance/observed_measured/solver_estimate logic; graph_health.ndjson lacks these keys; render_state_hash_lineage={render_state_hash:null, pending:'renderer must write its consumed-state fingerprint here'}.",
    "F6: no observation/solver_estimate record type; static-gauge numeric rows with ignored is_measured flag; contact table split (21-row de7ebc27 canonical vs 150-row b51c3c11 sidecar, same logical name); render_consumption row_count declared 3, actual 9."
  ],
  "residualRisks": [
    "Contact STATE labels (contact_frame_detail) are clean and pose-graph-independent; no live path found where T_rest/static-gauge feeds a contact or nonpenetration decision. Laundering is confined to body-localization render (F1) and stored pose/prior/manifest artifacts (F2-F4).",
    "F1 verified visually in world view at f090/f120 only; overlay view uses the same unconditional observed_body_world call so is laundered identically, but overlay pixels were not extracted per unknown frame.",
    "Read-only inspection; no pipeline stage rerun and no edits. The interventions in the report (retract laundered render, remove static-gauge tables, delete orphan hypothesis, flag static_gauge_summary, implement provenance-purity audit, add observation/estimate schema separation, fix count/hash integrity) are the parent's to apply.",
    "The clean artifacts (object_pose_observations.ndjson, unknown_preserving_pose_render, rest_cluster_reference) are correct; the risk is only that they coexist with and are outranked by the dirty artifacts in the authoritative package."
  ],
  "noStagedFiles": true,
  "diffSummary": "No code or artifact changes. Findings written only to the authoritative report path 33_pose_semantics_attack.md.",
  "reviewFindings": [
    "blocker: sidecar primary render_artifacts (v19_contact_state_full_duration/*.mp4 via render_clip001850_v19_contact_state_full_duration.py:48/:294/:349) draws a localized keyboard body on all 150 frames from the P15 pose graph (nearest-hold, f78-149 = f77 outlier ~197mm off rest); visual proof at unknown frame f090 (/tmp/pose_attack_crops/CONTACT_world_f090.jpg). Same laundering the correction claims to remove, still advertised as primary.",
    "high: ego_hoi_sidecar/.../tables/object_pose_static_gauge.ndjson (150 numeric T_rest rows) + object_visibility_ledger.ndjson still shipped in the canonical tables/ dir and listed in manifest tables[]; protected only by a status string the renderer proven to ignore.",
    "high: orphan unknown_preserving_pose_render/static_pose_hypothesis.json (schema ego.hoi.static_pose_hypothesis/0.1.0, role 'candidate prior for a future factor graph', embedded T_rest) referenced by root artifact_manifest.json, written by no current script, unreported by subagent 30 - reintroduces the doctrine-forbidden prior object in the current pose dir.",
    "high: ego_hoi_sidecar/.../static_gauge_summary.json (no in-file superseded marker) advertises render_lineage.pose_source_consumed_by_renderer=object_pose_static_gauge.ndjson and frames the static fill as defect_repaired.",
    "blocker: the R0/R1 provenance-purity audit does not exist (build_ego_hoi_sidecar_and_graph_health.py + graph_health.ndjson lack all provenance-purity/held/observed_measured/solver_estimate fields) and render_state_hash_lineage is a null stub - nothing detects F1-F4.",
    "medium: schema ambiguity - no observation vs solver_estimate record type; numeric pose rows with ignored is_measured=false; contact table split (21-row de7ebc27 canonical vs 150-row b51c3c11 sidecar under same name); render_consumption declared row_count 3 vs actual 9."
  ],
  "manualNotes": "Core message: the pose-semantics correction is additive and ungated, not substitutive. object_pose_observations.ndjson and the unknown_preserving render are clean, but they were added ALONGSIDE the dirty artifacts rather than replacing them, and the sidecar still advertises the P15-posed contact render as its primary render_artifacts (draws a keyboard body on 142 unknown frames - visual proof at f090). The static-gauge pose table, an orphan static_pose_hypothesis.json ('candidate prior'), and static_gauge_summary.json all persist in the authoritative package. None were caught because the R0/R1 provenance-purity audit is unimplemented and its render-lineage hook is a null stub. Fix order: (1) retract/rebuild the primary render to draw body only on observed_measured frames (consume object_pose_observations, not the P15 pose_report), or repoint render_artifacts to the unknown-preserving render; (2) remove static-gauge tables from tables/; (3) delete orphan static_pose_hypothesis.json; (4) remove/flag static_gauge_summary.json; (5) implement the provenance-purity + render-lineage audit and prove it flags the current artifacts before cleanup; (6) add observation/estimate/unknown schema record types; (7) fix contact-table and count/hash integrity."
}
```
