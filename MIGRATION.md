# Grouped source capability and verification boundaries

The active source is the grouped package. Baseline commit
`12aede8eaa2a945150b3de9bd3d0f81461d408b4` preserves the retired source for audit;
the original worktree was not modified. `migration-map.json` names the final
static owners of extracted source symbols. No production import or subprocess
calls a retired ego-annotation script.

## Callable capabilities

| Entry | Implemented behavior | Verification boundary |
|---|---|---|
| `main.py annotate` | Fresh raw video, fixed/estimated K, conditioned camera-z depth, private-input HaWoR, OWLv2/SAM2, metric ownership, full native SAM3D shape, observed/RGB SO3+SE3, full-MANO unsigned visual fit, batched Astra judgments, complete JSON/RRD and triangle MP4 | CPU contracts and real worker-interface checks; full fresh-video acceptance requires the separately executed runtime cases |
| `main.py refine-pose` | Same-run role/raw/object/timeline-bound prepared observations and paired RGB producer inputs | Numerical/gradient/restoration/PnP tests; empty accepted RGB is a valid measured outcome, absent producer is an error |
| `main.py fit-visual` | Prepared full MANO/object candidate; immutable source replay, scale/rotation/articulation bounds, 12-step triangle/outline refresh | Real MANO CPU smoke and baseline loss parity; physical acceptance remains false |
| `main.py fit-formal` | Named per-hand interval profiles; optional contact geometry, object delta, hand-ray, depth-order, visibility and residual variants | Value/gradient parity, actual consumer spies and candidate perturbation; no claim that every optional real-data factor is nonempty |
| `main.py formal-tail` | Prepared formal or RGB-initialized object graph, P17 evidence, P18 solve, full-MANO acceptance, separate physical/display completion roles, layered JSON/RRD/MP4 | Unit/contract fixtures; full real-data shared-tail integration is not yet established by this refactor's tests |
| `main.py render-dual` | Two bound SAM3D/TRELLIS display geometries, one frozen shared camera/pose/MANO state, separate RRD/MP4 outputs | Shared-array/hash and role fixtures; no new physical optimization or generated signed authority |
| `main.py prepare` | Frozen object image targets, visible-surface track, face eligibility, hand visibility, signed proxy, multiview support | Retained computations and loader contracts; optional heavy rasterizers require their declared environments |
| `main.py judge` | Thirteen separate prompts/schemas/entrypoints, one Astra-only transport | Live image/structured-output probes and rejection tests; visibility receives all requested corresponding frames |
| `observations.hand.wilor` | Optional raw WiLoR hand prediction | Provider import/API retained; does not replace HaWoR's world-camera/archive contract |
| `observations.object.trellis` | Optional complete-shape model worker with exact offline DINO bindings | Provisioned model worker; dual display accepts a separately canonical-bound result |
| `evaluate.py` | Hand boxes, MANO3D, object trajectory and two summaries on frozen predictions | Isolated evaluation entry; ground truth is never imported by prediction main |

## Preserved distinctions

- Source, candidate, accepted full-MANO and display arrays are not interchangeable.
  Accepted formal MANO is rendered only with its gate result and archive hash;
  standalone visual fitting retains explicit false physical/contact authority.
- Generated complete triangles support appearance/display and unsigned proximity.
  Signed factors require declared readiness and local per-face authority.
- Camera K, affine image planes and radius-to-camera-z conversion retain their
  source conventions. HaWoR receives an isolated raw copy before upstream code
  writes detector/SLAM caches.
- Formal active penetration uses mean squared residual; dense penetration and
  visible-depth-order use squared sum divided by positive residual count. Contact
  geometry retains gradients through the optimizable probability multiplier.
- The object graph refreshes observed correspondences per outer pass, the formal
  hand solver refreshes signed sets only between LBFGS passes, and visual fitting
  refreshes triangle/outline correspondences every 12 Adam iterations.

## Explicit remaining gaps

The new raw entry is create-only. The former version-specific automatic
resume/rewind, cross-version candidate-prefix reuse and native-reuse launch flows
were not migrated into a public recovery API. Producer receipts and immutable
input checks preserve evidence, but are not a substitute for that missing
execution capability. Prepared numerical entries can write a new candidate from
same-run inputs; they do not claim automatic recovery of a failed cold run.

Real three-case outputs and full formal/shared-tail execution are runtime
acceptance obligations outside the CPU/build evidence. A build, an import, or an
empty signed active set does not establish those outcomes. The refactor must not
be described as fully equivalent until this recovery gap is resolved or an
explicit scope decision is recorded.

## Retired independent historical families

Historical VLM click/candidate-selection flows, alternative depth/SLAM/hand/shape
providers, part/deformable/variable-camera/SDF-deformation/contact-dynamics
problems, and independent global/keyframe/periodic/late-window experiments remain
in baseline Git history. Their actual shared numerical helpers were extracted
when used by a retained path; their old CLIs and rejected experimental modes were
not copied into the active package.
