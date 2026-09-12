# v7 publication record — developer only

This record must not be included in a prediction agent's runtime context. It
identifies the frozen numerical baseline, not a quality pass or a new prediction.
The local packaging commits on `research/rgb-scene-v7-baseline` add only the
README, baseline docs, parent-side helper and tests. Numerical source remains
`fa9a84cbfb6836b92e35a6316073a6855e6b4d22`. No push is performed by preparation.
Canonical remote: `origin`, `git@github.com:DexGEM-Lab/ego_annotation.git`.

## Frozen identities

External experiment root `E` (path abbreviation used below):
`/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/integration_validation/rgb_sam3d_visual_annotation_20260908/standalone_pipeline_20260910`.

- Candidate: `E/cold_raw_rgb_v1/candidates/rgb_chain_contact_repair_v7`.
- Executed bundle: `/mnt/user-home/kupingxin/ego_annotation_runtime/rgb_scene_chain_contact_gpu4_20260911_v4`.
- Execution bundle manifest SHA256:
  `2072c5a7e7636da7d3f0d28ece045be736e664848a86810c3aa5faaedfd94ba4`.
- Birth bundle manifest SHA256 (distinct; not relabeled as execution):
  `0e40e24dfd2bf61a8484a82f94e40ae69490238cf055c8d5084a7d14c21254aa`.
- Candidate contract SHA256:
  `dcbdced6c2e71fd9c140da128123c71fec395a237f241dd657f5b1d92dfa1ecc`.
- Source input contract SHA256:
  `3bc7e0a7d27c9525ce5f501fd7f7f1c984c8e8ae905025440c2fa149d8c17676`.
- `candidate/final/annotation.json`: 4,026,607 bytes, SHA256
  `3e704c3a8708df122ed14bcaf334543253e2f10392f41fec4476fba9c4a0c202`.
- `candidate/final/world.rrd`: 39,959,814 bytes, SHA256
  `41c9143c3c67a41b5e076898ca0b86e9e8977b42e701ea313d8f4625d4233f17`.

All 15 recorded candidate events have rc0, including native import. The candidate
reused the verified P01–P08 prefix and strict current-video frame-0 native cache;
P09 onward was computed in the candidate, with no SAM3D neural rerun. The native
cache's historical wrapper failed during post-model printing despite a valid
model receipt. That failure is not rewritten as success. The native actor
exported, then encountered managed transport/import issues; pair-only readback
was parent-authored, not a fictitious native completion.

Parent readback evidence: `E/candidate_pipeline_20260911/recharge_v7/FINAL_PAIR_READBACK.json`.
It reports 85 chunks, all 150 camera/K/RGB/object-pose frames, full object geometry
(238,288 vertices / 476,588 triangles), and both complete MANO surfaces
(778 vertices / 1,538 triangles each). Final delivery is JSON/RRD only, not MP4.
These counts establish content/readability, not physical annotation quality.

## External prerequisites / portability

Use existing, preflighted server environments; no installation is implied:

- Main/export Python: `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`.
  Preserve this lexical venv executable, not its shared uv symlink target.
- RGB/LightGlue Python: `/mnt/user-home/kupingxin/GHOST/.venv/bin/python`;
  bundle `third_party/LightGlue` and pinned weights.
- UniDepth/SAM2 Python: `/mnt/user-home/kupingxin/ego_annotation/.runtime/model_envs/unidepth_sam2/bin/python`.
- HaWoR work/assets: `/mnt/user-home/kupingxin/ego_annotation/.runtime/hawor_work`;
  licensed MANO and WiLoR assets referenced by the executed runtime config.
- SAM3D Python: `/mnt/user-home/kupingxin/sam3d-objects/mamba/envs/sam3d-objects/bin/python`,
  with its pinned source/config/checkpoints and exact model receipt for reuse.
- Installed native `ego-annotation.rgb-scene-runtime` profile and its declared
  ShellGate extension/broker; pinned Rerun version from runtime config. Inspect
  that config/manifest for full asset identities before any fresh run.

Paths are deployment-specific. Neither checkout nor a pip command reproduces
these external environments, licensed assets or native provider. There is no
unattended VLM planner. Physical GPU must match configuration; logical device
stays `cuda:0`. Resume and GPU scheduling/sharing remain limitations, not solved
by this packaging patch. Image dimensions in the helper are metadata only.
Raw-new-video and validated-reuse commands are linked from the baseline guide.

## Actual publication validation (2026-09-12)

Evidence directory: `E/baseline_publication_20260912/review_fix_7a803443/retry_validation`.
These are fresh nonzero-count runs after the gateway-concurrency interruption;
earlier evidence remains untouched in the parent directory.
On the authorized server, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''`, each command was:

```bash
PREP_PYTHON=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
"$PREP_PYTHON" -m unittest discover -s tests -p '<filename>.py' -v
```

| Filename without `.py` (also log basename) | Result |
| --- | --- |
| test_rgb_scene_candidate_launch | 12 passed |
| test_rgb_scene_primary_pads | 5 passed |
| test_rgb_scene_rotation_chain | 5 passed |
| test_rgb_scene_pose_graph | 6 passed |
| test_v19_demo_timeline | 13 passed |
| test_export_rgb_scene | 5 passed |
| test_rgb_scene_edge_conventions | 3 passed |
| test_rgb_scene_metric_replay | 2 passed |
| test_rgb_scene_runtime | 5 passed, 1 error |

**56 passed / 1 known error**, not all-green. The unchanged runtime test
`test_byte_bound_successful_resume_rejects_tamper` expects `ValueError`, but
`rgb_scene_lineage_base.py:73` raises `FileNotFoundError` for the removed result.
Its complete traceback is in `test_rgb_scene_runtime.log`. Failed preparation
fixture validation is preserved in `../helper_before_fixture_update.log`.

`smoke_and_verify.py` / `smoke.log` / `actual_smoke_and_hashes.json` contain the
exact producer-free CLI smoke and read-only hash commands/results. A fresh
external plan (path containing spaces) was prepared from the actual frozen v7
bundle/candidate. All 12 stage argv match the successful scripts semantically;
15 shell bodies pass `bash -n`. Only the three **read-only inspection** scripts
were executed (startup/proposal/P09 outputs: 498 / 928 / 914 bytes). No model,
producer, optimizer, prediction, or quality evaluator ran. All 59 declared
source scripts/helpers match both the executed bundle and frozen Git source.
Prepublication snapshot files, candidate launch bodies and context were hashed
before/after and remained unchanged, including the final pair.

## Unchanged quality limits

`annotation_ready=false`; demo-only / diagnostic-only. Parent post-freeze
assessment reports rotation errors after fixed first-frame camera-relative
canonical alignment: v7 median/mean/p90/max 11.16/9.50/13.72/23.31 degrees;
v13 14.89/40.56/89.58/103.65. Full-mesh q05 median gaps: v7 left/right
6.33/14.80 mm, v13 26.11/11.45 mm, GT 1.91/3.12 mm. These are unsigned
geometric diagnostics, not measured contact, collision, sign/SDF or
nonpenetration. They do not establish a causal explanation for the hand gap.
GT was used only in post-freeze evaluation, never as prediction input.
Parent assessment paths:
`E/comparisons/rgb_chain_contact_repair_v7_vs_hot3d_gt_20260911/HOT3D_GT_ASSESSMENT.md`
and `E/comparisons/rgb_chain_contact_repair_v7_vs_v13_old_20260911/QUALITY_ASSESSMENT.md`.
No new quality evaluation or hand-discrepancy investigation is claimed here.
