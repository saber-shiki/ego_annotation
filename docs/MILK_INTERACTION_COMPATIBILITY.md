# Milk observation → P17 compatibility patch

Scope: keep the original RL pipeline and P17 consumer unchanged. Independent
strict contact judgments cannot supply P17's required occlusion/metric-uncertainty
fields. The observations-owned `interaction.assembly` adapter joins the five
independent judgments on selected object, declared frame IDs and hand/entity.
It expands only supplied IDs, rejects overlapping/foreign/misaligned evidence,
and never edits the saved judgment files.

- Occlusion maps only `hand_in_front`, `object_in_front`, `separate`, `unresolved`
  to the corresponding existing P17 vocabulary. Visibility is retained as
  evidence, not used to invent contact or depth ordering.
- Joint depth reduces to `object_depth_reliable` only for supported object and
  hand; supported object with mixed/unsupported hand becomes
  `hand_depth_unreliable`; all other or missing combinations remain
  `mixed_or_unresolved`. Original per-entity rows and uncertainty remain attached.
- Missing contact becomes explicit `unresolved`, with a separately labeled
  neutral configuration probability of .5. No anatomical pads are inferred.
  This does **not** disable P17's existing image-candidate activation behavior.
- `assets_manifest.contact_support_uncertainty_m` optionally configures a finite,
  positive support uncertainty; default .06 m. It is labeled a conservative
  configuration prior, never a VLM metric measurement or signed authority.
- The versioned adapter/prior contract is bound in the runtime input contract,
  checked on resume, and included in assembly-stage parameters. Existing code,
  asset and stage recovery checks remain intact. Old-code runs cannot resume
  under this patch; use a fresh immutable snapshot/run after review.
- Main preserves the diagnostic, visibility and timeline-evidence views. Joined
  rows retain source rows/text and source file hashes; `assembly_sources` retains
  the original judgment provenance once per batch. P17 carries the row's
  `evidence` into its existing provenance field.

Validation: `observations.test_interaction_assembly` uses current strict schemas,
real P17 loading, and the actual main assembly AST under real StageRecovery with
synthetic producers. Optional `EGO_INTERACTION_JUDGMENT_BATCH` reads five saved
JSON judgments and hashes them before/after; it writes only a temporary test
assembly, never a prediction or old-run artifact.

Static downstream check: main supplies every required and dereferenced source
manifest artifact key; mesh output keys match the shape producer. A remaining
conditional prerequisite is `observations.manifest`'s exact MANO left/right files
and WiLoR wrapper. Main passes optional `assets['wilor_root']`, otherwise the
writer looks under `mano_models.parent/wilor/models/mano_wrapper.py`; cold
preflight does not guarantee this wrapper location (its optional WiLoR provider
uses `wilor_prediction_root`). No asset provisioning/path policy is changed here.
No full P17 geometry build, source-bundle production, train/rollout or visual
annotation improvement is claimed by these compatibility tests.
