#!/usr/bin/env bash
set -uo pipefail
REPO=/mnt/user-home/kupingxin/ego_annotation
RUN=/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/v19_runs/20260803_hot3d_clip001851_keyboard_pinhole_kupingxin_v2
EXP=$RUN/experiments/sam3d_p11_p12_native_owned_mask_v1
P13_OUT=$EXP/p13_controlled_geometry_prior_ab
EVIDENCE=$RUN/measurements/geometry_completion/rigid_evidence/hot3d_clip001851_keyboard_pinhole/keyboard/evidence_bundle/evidence_bundle_report.json
TRELLIS_MESH=$RUN/measurements/geometry_completion/trellis_keyboard_seed42/trellis_mesh.ply
TRELLIS_REPORT=$RUN/measurements/geometry_completion/trellis_keyboard_seed42/qc_trellis_shape_v3.json
OLD_SAM_MESH=$RUN/experiments/sam3d_vs_trellis_20260803/sam3d_raw/keyboard_anchor/sam3d_mesh.ply
OLD_SAM_REPORT=$RUN/experiments/sam3d_vs_trellis_20260803/sam3d_raw/keyboard_anchor/qc_sam3d_objects_mesh_v7.json
NEW_SAM_MESH=$EXP/p12_parallel_priors/sam3d_objects_native/hot3d_clip001851_keyboard_pinhole_keyboard_frame_000109/sam3d_mesh.ply
NEW_SAM_REPORT=$EXP/p12_parallel_priors/p12_parallel_geometry_priors_report.json

set +e
"$REPO/.venv/bin/python" \
  "$REPO/experiments/sam3d_p11_p12_branch/run_p13_controlled_geometry_prior_ab.py" \
  --evidence-report "$EVIDENCE" \
  --builder-script "$REPO/scripts/build_v18_compact_rigid_trellis_completion.py" \
  --python "$REPO/.venv/bin/python" \
  --candidate "trellis_frozen|TRELLIS|$TRELLIS_MESH|$TRELLIS_REPORT" \
  --candidate "sam3d_old_raw_sam2_mask|SAM3D Objects raw-SAM2-mask|$OLD_SAM_MESH|$OLD_SAM_REPORT" \
  --candidate "sam3d_new_object_owned_mask|SAM3D Objects object-owned-mask|$NEW_SAM_MESH|$NEW_SAM_REPORT" \
  --output-dir "$P13_OUT" \
  --observed-band-scale 1.7320508075688772 \
  --silhouette-dilate-px 16 \
  --planar-slab-eigenvalue-ratio-max 0.04 \
  --planar-slab-min-band-m 0.018 \
  --planar-slab-max-band-m 0.055 \
  > "$EXP/p13_controlled_geometry_prior_ab_stdout.txt" \
  2> "$EXP/p13_controlled_geometry_prior_ab_stderr.txt"
code=$?
set -e
printf '%s\n' "$code" > "$EXP/p13_controlled_geometry_prior_ab_exit_code.txt"
if [ "$code" -eq 0 ]; then
  touch "$EXP/P13_CONTROLLED_AB_DONE"
else
  touch "$EXP/P13_CONTROLLED_AB_FAILED"
  tail -100 "$EXP/p13_controlled_geometry_prior_ab_stderr.txt"
fi
exit "$code"
