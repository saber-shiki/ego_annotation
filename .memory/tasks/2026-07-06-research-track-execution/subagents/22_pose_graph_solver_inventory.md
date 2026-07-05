# 22 — Pose-graph solver inventory & causal mechanism for clip001850 keyboard inertness

Read-only causal inventory. Branch `yiwen_research`. No edits, nothing staged.
Run: `/data2/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`
Script: `scripts/solve_v19_rigid_object_pose_graph.py` (611 lines). Report: `measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`.

---

## 0. Headline (the mechanism in one paragraph)

The pose graph is inert by construction, not by a bug. Its optimization variables are
**per-direct-frame correction deltas** `x = [rotvec_delta_i, trans_delta_i]` for the 13 fit
frames, **seeded at zero** (`x0 = zeros`, `scripts/solve_v19_rigid_object_pose_graph.py:487`).
The residual function `residual_vector` (`:295-333`) contains **no measurement term that
compares a predicted pose to an observed pose** — the per-frame ICP pose is treated as an
exact observation anchored at `delta=0`. The only term that can push a delta away from zero
is the **nonpenetration target** term, and that term is absent here: the run invoked the
solver **without `--constraint-report`** (`report.inputs.constraint_report = null`),
**and** the constraint state that does exist (P16) carries **zero rows in the solver's
accepted state** `candidate_coordinate_correction_visible_2d_compatible` (all 300 rows are
`uncertainty_only_nonwatertight_mesh_no_signed_correction` or
`no_penetration_no_coordinate_change_needed`, because the keyboard completion mesh is
non-watertight). Every remaining term — the anchor (`delta/sigma`), the temporal step
(`delta_i - delta_{i-1}`), and the acceleration second-difference — is a **homogeneous
linear function of the deltas**, so the entire residual vector is identically zero at
`x0=0`, the gradient `Jᵀr = 0`, and `scipy.optimize.least_squares` terminates on the `gtol`
condition in **one function evaluation** (`nfev=1`, `cost=0.0`). The 13 raw ICP poses pass
through the optimizer unchanged (verified: output translation == input ICP translation to
<0.001 mm on f30-34), and the 137 non-direct frames are filled afterwards by pure
geometric nearest-hold / Slerp+lerp interpolation (`build_pose_rows`, `:377-463`),
independent of the optimizer. The graph therefore neither smooths the jumpy ICP trajectory
nor localizes the object on the 137 unmeasured frames.

---

## 1. Provenance — exact command/log that created the report

- **Phase:** P15 `rigid_pose_graph_created`, `harness_events.jsonl`,
  `timestamp_utc = 2026-06-26T09:34:28Z`, `frame_fit_count` upstream = 13.
- **Log:** `logs/P15_rigid_pose_graph_keyboard.log` — contains only the JSON summary
  printed by `run()` (`:570`); matches the report's `optimizer`/`correction_summary` block.
- **Report path written:** `measurements/pose_fits/keyboard_rigid_pose_graph/v19_rigid_object_pose_graph_report.json`
  (`write_json`, `:566`) **and** a byte-identical copy at
  `.../v18_compact_rigid_object_pose_fit_report.json` for legacy consumers (`:568`).
- **Canonical command (runtime spec):** `runtime/v19_runtime_spec.md:479-494` (§P15). The
  spec command does **not** pass `--constraint-report`:
  `solve_v19_rigid_object_pose_graph.py --annotations … --pose-report <P14 ICP fit> --completion-report … --object-id keyboard --complete-full-timeline-rigid-pose --output-dir …/keyboard_rigid_pose_graph`.
  This matches `report.inputs.constraint_report = null`. The orchestration doc
  (`docs/v19_english_orchestration.md:623-637`, §9.5) makes the no-constraint run the
  default: *"If the MANO/object constraint state does not exist yet, run this graph without
  `--constraint-report`."* P16 (constraint state) runs **after** P15.
- **`scripts/build_v19_static_rigid_pose_report.py` is NOT involved** in this run — the
  report's `method` field is `solve_v19_rigid_object_pose_graph`, and
  `build_v19_static_rigid_pose_report` is referenced only in an unrelated older OPS
  (`.memory/tasks/2026-06-23-pipeline-v19/OPS.md`). It is a separate anchor-copy tool.

---

## 2. Why `optimizer.nfev=1`, `cost=0.0` (the causal mechanism)

Trace of `residual_vector(x0=0, …)` (`solve_v19_rigid_object_pose_graph.py:295-333`), term by term:

| Term | Code line | Form at `x0=0` | Drives motion? |
|---|---|---|---|
| Translation anchor | `:300` `trans_delta[i]/sigma_t` | `0` | No — pulls delta→0 |
| Rotation anchor | `:301` `rot_delta[i]/sigma_r` | `0` | No — pulls delta→0 |
| Nonpenetration target | `:303-307` (conditional) | **absent** | Would, but `target is None` for all 13 obs |
| Temporal step (trans) | `:316` `(delta_i-delta_{i-1})/σ·√gap` | `0` | No — homogeneous in delta |
| Temporal step (rot) | `:317` | `0` | No |
| Accel 2nd diff (trans) | `:328` | `0` | No |
| Accel 2nd diff (rot) | `:329` | `0` | No |

- **No observed→mesh / depth re-fit residual exists inside the optimization.** `surface_metrics`
  (`:335-353`) computes `apply_pose` + `nearest_summary` (observed↔mesh) but is called
  **after** the solve (`:491-492`) purely to populate `surface_before`/`surface_after` for
  the report. It never enters `residual_vector`. So the optimizer never re-fits the mesh to
  the visible surface; it only ever perturbs deltas.
- The anchor + temporal + accel terms are all **linear and homogeneous in `delta`**
  (no constant offset), so `r(x0)=0`, the Jacobian `J` is constant, and the least-squares
  gradient `Jᵀr = 0`. `scipy`'s `gtol` test (`‖Jᵀr‖_∞ < gtol`) is satisfied at the seed →
  `nfev=1`, `cost=0.0`, `message="gtol termination condition is satisfied."`
- **Result:** `correction_summary` is all zeros (`translation_delta_norm_m.max=0`,
  `rotation_delta_norm_rad.max=0`), `surface_before == surface_after` byte-for-byte
  (`surface_observed_to_mesh_median_degradation_m = 0.0`), and the 13 direct frames'
  output translations equal the input ICP translations exactly.

**Two independent reasons the nonpenetration channel is closed for this clip:**
1. The P15 command omitted `--constraint-report` (`report.inputs.constraint_report = null`)
   → `constraint_targets()` returns `{}` (`:163` early-return on `path is None`).
2. **Even if it had been passed**, the constraint state
   (`measurements/contact_nonpenetration/keyboard_mano_object_constraint/v18_mano_object_constraint_state.json`)
   has **0 of 300 rows** in state `candidate_coordinate_correction_visible_2d_compatible`
   (172 `uncertainty_only_nonwatertight_mesh_no_signed_correction` +
   128 `no_penetration_no_coordinate_change_needed`). So `nonpenetration_target_frame_count`
   would be 0 regardless. The non-watertight keyboard completion mesh (EPISTEMIC: 95.4%
   inferred hidden surface, non-watertight) is why P16 produced no usable signed
   corrections.

---

## 3. Factor families — supposed to exist vs actually active

The report's `parameters` block advertises five factor families. Their actual support:

| Factor family | Parameter | Constructed in residual? | Active for clip001850? |
|---|---|---|---|
| Pose prior / anchor (delta→0) | `min/max_pose_sigma` | Yes (`:300-301`) | **Yes (dominant, trivially min)** |
| Temporal smoothness (1st diff of correction) | `sigma_translation_delta_step_m`, `sigma_rotation_delta_step_rad` | Yes (`:316-317`) | Constructed but **homogeneous → inert** (no driver) |
| Acceleration (2nd diff of correction) | `sigma_translation_delta_accel_m`, `sigma_rotation_delta_accel_rad` | Yes (`:328-329`) | Constructed but **homogeneous → inert** |
| Nonpenetration target | `max_nonpenetration_target_m`, `sigma_nonpenetration_target_m`, `nonpenetration_states` | Yes (`:303-307`, conditional) | **Zero support** (`nonpenetration_target_frame_count=0`) |
| **Observed-surface / depth re-fit** | — | **Not present in residual at all** | N/A — does not exist in the solver |

**Key structural fact:** the comment at `:310-311` states the design intent explicitly:
*"Smooth the correction field, not the physical object trajectory, so real object motion
measured by ICP is preserved."* The graph is therefore a **nonpenetration-driven
corrector**, not a trajectory smoother. With no nonpenetration driver, it is a pure
pass-through of the raw ICP seeds. There is no factor that smooths the **pose trajectory**
`T_world_object(t)`; the design intentionally preserves ICP motion (including the 166 mm
jumps).

---

## 4. Are temporal residuals actually constructed? — Yes, but inert

The temporal step and acceleration residuals **are constructed** (`:314-330`). They are
genuine code, with a matching Jacobian sparsity pattern (`residual_sparsity`, `:336-371`,
the `for i in range(1,n)` and `for i in range(1,n-1)` blocks). They are **not dead code**.
However they are **differences of the correction field** `delta_i - delta_{i-1}`, not
differences of the pose. Because they are homogeneous in `delta`:
- at `x0=0` they are exactly 0 (contributing to `cost=0`);
- they can only *redistribute* corrections if a nonpenetration target first creates them;
- with no driver, they exert no force and the gradient is zero everywhere along the
  constant-delta manifold.

So "temporal residuals constructed = yes; temporal residuals doing work = no." This is the
crux of the D5 (graph-inertness) diagnosis from subagent 21: the graph *contains* temporal
factors but they are **unsupported** in the sense that matters — they have no input that
gives them nonzero value, so the optimizer's output equals its input.

---

## 5. Intervention point (if wiring is to be made live)

The single function that would have to change is **`residual_vector`**
(`scripts/solve_v19_rigid_object_pose_graph.py:295-333`). Two distinct interventions,
depending on which mechanism is desired:

1. **Make the graph a trajectory smoother** (addresses the actual measured defect:
   13 jumpy poses, 137 unlocalized frames). Add a **measurement/trajectory** term that
   couples the corrected pose to either (a) the visible surface (observed↔mesh nearest
   distance under the corrected pose — the machinery already exists in `surface_metrics`
   `:335-353` and `apply_pose` `:112`) or (b) a smoothness prior on the **pose** itself
   (not the correction). This would let the optimizer redistribute the 13 fits into a
   smoothed trajectory and propagate corrections toward the held frames. The natural
   insertion is a new block in `residual_vector` mirroring `surface_metrics`, plus
   extending `residual_sparsity` (`:336`) with the new residual rows.
2. **Activate the nonpenetration driver** — but this is **structurally closed** for this
   clip until the keyboard completion mesh is made watertight (P16 currently routes 100% of
   rows to `uncertainty_only_nonwatertight_mesh_no_signed_correction`). Even passing
   `--constraint-report` (contrary to the runtime spec) would yield `targets={}`.

**Already-implemented alternative:** `scripts/smooth_v19_rigid_object_pose_trajectory.py`
explicitly smooths the **physical SE(3) trajectory** (acceleration prior, treats measured
pose as a noisy observation) and its header states it "differs from
`solve_v19_rigid_object_pose_graph.py` because it smooths the physical trajectory itself."
It is **not wired into this run** (no reference in the run root or harness events). It is
the existing candidate for intervention #1.

**Trajectory-completion point (separate from the optimizer):** `build_pose_rows`
(`:377-463`) is where the 137 non-direct frames get nearest-hold / Slerp+lerp fill. This is
geometric only; it does not consult any factor and is where the "frozen plateau" behavior
originates. A live-trajectory intervention would also need to revise how these gaps are
filled (e.g., extrapolate the smoothed trajectory rather than hold the nearest direct pose).

---

## 6. Per-frame classification table (direct / held / interpolated / ineligible)

Source: `v19_rigid_object_pose_graph_report.json` `pose_rows` (150 rows).
Classification keys: `pose_measurement_status` (upstream ICP eligibility) +
`temporal_pose_graph.pose_source` (graph fill mode).

| Category | Count | Frames | Translation behavior |
|---|---|---|---|
| **Direct fit** (`fit_to_visible_depth_samples`, `direct_visible_pose_observation_corrected`) | 13 | 30,31,32,33,34,35,36,45,46,60,75,76,77 | Raw per-frame ICP pose, **passed through unchanged** (delta=0). Jumpy. |
| **Interpolated** (`missing_initial_graph_pose` or ineligible, `interpolated_between_visible_pose_observations`) | 35 | 37-44, 47-59, 61-74 | Slerp rotation + linear translation between bracketing direct frames. |
| **Nearest-hold → f30** (`missing_initial_graph_pose`) | 30 | 0-29 | **Frozen at f30 pose** `[0.5776, -0.0958, 0.3385]` |
| **Nearest-hold → f77** (`visible_surface_ineligible_for_rigid_pose_fit`) | 72 | 78-149 | **Frozen at f77 pose** `[0.3897, -0.1101, 0.3444]` |
| **Total** | 150 | | 102 nearest-hold collapse to **2 unique translations** |

`pose_measurement_status` totals (upstream P14 ICP eligibility, carried through unchanged):
`missing_initial_graph_pose = 65` (f0-29, 37-44, 47-59, 61-74 — no visible mask/depth before
f30 or in the active gaps), `visible_surface_ineligible_for_rigid_pose_fit = 72`
(f78-149 — mask present but `< min_visible_points` / near-planar / failed rigid-fit
eligibility), `fit_to_visible_depth_samples = 13`.

### Direct-fit jump magnitudes (consecutive DIRECT→DIRECT edges)

| Edge | Δtranslation (mm) | Gap (frames) |
|---|---|---|
| f30→f31 | 20.2 | 1 |
| f31→f32 | 28.2 | 1 |
| f32→f33 | 21.1 | 1 |
| f33→f34 | 4.8 | 1 |
| f34→f35 | 4.1 | 1 |
| f35→f36 | 8.6 | 1 |
| f36→f45 | 40.8 | 9 |
| f45→f46 | 57.2 | 1 |
| f46→f60 | 74.0 | 14 |
| f60→f75 | 121.9 | 15 |
| **f75→f76** | **166.4** | 1 |
| f76→f77 | 23.0 | 1 |

Window 28-48 (the contact-test slice) has 7 direct edges; median 20.2 mm, max 57.2 mm (f46).
Full-timeline motion floor (median+3·MAD over 9 edges) = 76.9 mm trans / 18.2° rot (subagent
19); no edge clears it. The 166.4 mm jump (f75→f76) is physically impossible for a desk-rested
keyboard and is ICP/SAM2 mask re-acquisition noise — it passes through the optimizer untouched.

---

## 7. Precision corrections to prior subagent claims

- Subagent 21 / `graph_liveness_and_motion_coupling.md` state "65 frozen missing-pose frames
  all share one identical frozen translation `[0.578,-0.096,0.339]`." This is imprecise: of
  the 65 `missing_initial_graph_pose` frames, only **30 (f0-29) are frozen** at that
  translation; the other **35 (f37-74) are interpolated** and vary. Separately, the 72
  **ineligible** frames (f78-149) are frozen at a *different* pose `[0.3897,-0.1101,0.3444]`.
  Net: 102 nearest-hold frames collapse to **2** unique translations, not 1. The qualitative
  verdict (inert, jumpy, unlocalized) is unchanged and correct; the count is corrected here.
- The phrase "temporal residuals zero / not constructed" should read "constructed but
  homogeneous and unsupported → identically zero at the seed." The code does build them.

---

## 8. Causal conclusion (what an intervention must change)

The inertness is not a wiring bug to be "found"; it is the designed behavior of a
nonpenetration-driven corrector run with no nonpenetration driver and no trajectory
objective. To change the artifact (a live, de-jumped, gap-filled object trajectory), an
intervention must add a **measurement or trajectory-smoothness term on the pose itself**
inside `residual_vector` (or replace P15 with `smooth_v19_rigid_object_pose_trajectory.py`).
Re-passing `--constraint-report` alone will not help: the constraint state has zero accepted
rows because the keyboard completion mesh is non-watertight. This is consistent with
subagent 21's card-1 decision: the next mechanism is graph-factor liveness / object-pose
trajectory, and the discriminating first probe is whether a trajectory/measurement term
exists in the residual — it does not.
