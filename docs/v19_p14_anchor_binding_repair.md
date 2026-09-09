# P13 → P14 selected-anchor compatibility repair

## Defect and causal prediction

The completed SAM3D P13 wrapper emitted `inputs.evidence_report` and
`input_hashes.evidence_report`, but P14 reads `inputs.candidate_evidence_report`,
`candidate_evidence_report_sha256`, and `selected_anchor_atomic_binding`.
P14 therefore stopped before fitting, despite an original producer-declared P11
hash and validated frame-92 atomic binding. This is a disconnected input
contract, not a reason to rerun SAM3D or relax a physical gate.

If the aliases are the only input defect, a verified copy must pass the actual
P14 hash/frame/centroid code. A changed P11 hash means the old completion cannot
be rebound. A frame/object/surfel/centroid conflict means the current P09 and
original P11 no longer describe the same anchor. Passing these checks says
nothing about subsequent pose quality or full-video render quality.

## Mechanism

`bind_v19_completion_anchor.py` requires the original producer-declared P11
path and SHA. It verifies the P11 bytes before using its binding, compares the
current P09 object/frame/surfels/centroid with P11, validates P11's canonical
anchor, rejects positive GT flags and conflicting existing aliases, and writes
only a new report. The output preserves every existing completion field and
adds the three P14 aliases plus source-report/annotation hash provenance.
Geometry, readiness, uncertainty, face support, and intact render-mesh contracts
are unchanged. This is a compatibility repair, not a new geometry estimate.

The fresh SAM3D wrapper shares the normalizer and now requires `--annotations`.
It captures the P11 SHA before production and verifies it again at emission;
it cannot replace that digest with a changed evidence file's current digest.

The pose dispatcher now runs:

1. `P13_anchor_binding`: original `completion_report.json` →
   `anchor_bound_completion_report.json` in the same geometry directory.
2. `P14_anchor_bound_visible_pose`: new completion →
   `pose_fits/<object>/visible_anchor_bound/v18_compact_rigid_object_pose_fit_report.json`.
3. Existing RGB edges and initializer consume that new pose path.
4. P14b consumes the bound completion and preserves its input aliases; P15
   consumes P14b's report and checks equality with the pose binding/digest.
5. Demo consumes the bound completion's unchanged explicit intact-mesh contract.
   P19 continues to consume the separate P14b physical-support report.

The old failed `visible/` directory and `P14_visible_pose.log` are not reused,
removed, or overwritten. There is one fixed repair path, no retry counter.
Existing repair outputs are refused rather than overwritten.

## Focused validation (no prediction)

Use deployed Python with thread caps:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python
"$PY" -m unittest discover -s tests -p 'test_bind_v19_completion_anchor.py' -v
"$PY" -m unittest discover -s tests -p 'test_v19_sam3d_reconstruction.py' -v
"$PY" -m unittest discover -s tests -p 'test_v19_sam3d_runtime_closure.py' -v
```

For actual read-only evidence, set `P14_BINDING_REAL_RUN` to the failed run and
`P14_BINDING_CHECK_OUTPUT` to a fresh directory **outside** it, then run the first
test command. That test invokes the normalizer as a real subprocess, reproduces
the old P14 error, and runs unchanged P14 `main` through its actual anchor checks.
Only pre-check diagnostic mesh loading/sampling is stubbed. A trace stops before
pose fitting. It also reaches P15's actual binding gate and stops before mesh
loading. No model, solver fit, or render is executed.

Actual check on 2026-09-09: frame 92, centroid error 0.0 m, P11 SHA
`cf46b0814c87004067ea00ea7625f55f6f9299f665add054dcadf9c77d57d138`.
Original completion/P11/P09 bytes stayed unchanged. Evidence copy and check
results: `/tmp/p14-anchor-binding-check-20260909-final/`.

## Deployment boundary / remaining work

The same-run recovery transition, profile/asset bindings, runtime bundle and
current prediction state are deliberately not edited here. The parent must
incorporate the new helper and changed wrapper/dispatcher into its explicit
recovery transition before launching the fixed phases. Do not rerun completed
P00–P13 models. Full P14/P14b/P15 prediction, later physical constraints, and final
video rendering/visual inspection remain unexecuted by this repair check.
