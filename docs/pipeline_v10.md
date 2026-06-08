# Pipeline V10: Video-Conditioned Object Mesh Priors

## Starting Point

V9 added a real generated-mesh path, then accepted a small hidden component on the trash sample. The rejection mechanism was visible-shell competition: single-image complete priors usually produced surfaces that rendered in front of the measured object mask or depth. V10 therefore changes the object-prior source, while keeping the acceptance evidence unchanged.

The question V10 tests is:

```text
Can a video-conditioned generated mesh stream explain the same observed object across time better than a single-image complete prior while preserving image replay, surface tracks, MANO contact, and hand-object nonpenetration?
```

## State

For each object window, V10 keeps the measured-visible mesh archive as the visible evidence:

```text
M_t       measured visible object mesh from mask, depth, camera pose
P_t       video-conditioned generated object mesh for frame t
A_t       metric Sim3 or SE3 alignment from generated coordinates to world coordinates
G_t       delivered object mesh candidate after evidence filtering
H_t       MANO hand mesh and contact patches
T_wc_t    head camera pose
```

The generated stream `P_t` can come from Mesh4D, another video-conditioned reconstruction model, or a multi-view point/mesh completion model. The downstream path stays the same for every object: align, replay, track, physics, render.

## Evidence Contract

V10 delivery still requires the same observable agreement used by V7 through V9:

1. measured target replay must pass before generated geometry is evaluated;
2. generated visible surface must cover the measured visible mesh under the 10 mm p95 threshold;
3. full-fidelity z-buffer replay must pass silhouette, visible-inside, and depth p95 thresholds;
4. CoTracker surface factors must remain accepted on the generated candidate surface;
5. mesh-surface contact and selected-contact SDF must agree where contact evidence exists;
6. full-window hand-object SDF must keep hand penetration within threshold;
7. stakeholder render must show head camera, MANO hand, object mesh, contact marker, axes, scale, and caption clearly.

Single-image priors remain useful as proposals and negative controls. They become delivered geometry through the same checks.

## Implementation Plan

The first V10 executable path uses the existing Mesh4D wrappers:

```text
local RGBA sequence input
  -> A800 Mesh4D generation in tmux
  -> sync generated qc_mesh4d_sequence_v7.json and meshes
  -> archive_mesh4d_sequence_prior_v7.py
  -> run_v7_generated_prior_replay_qc.py
  -> run_v7_candidate_physics_qc.py
  -> render_v7_candidate_deliverables.py
```

Current target windows:

- wild rice frames 2538 to 2540, with Mesh4D input prepared from frames 2538 to 2548;
- trash frames 865 to 870;
- mop frames 759 to 765.

The local acceptance wrapper is `scripts/run_v7_mesh4d_sequence_batch.py`. It discovers remote Mesh4D reports after sync, aligns each generated six-frame stream to the measured target, runs full-fidelity replay, then runs physics and render for accepted replay outputs.

## Remote Execution Status

A800 host `192.168.11.220` is reachable. The 4090 host `192.168.9.220` timed out on SSH during the current V10 run. Mesh4D is running under tmux session `ego_v10_mesh4d` on A800 GPU 6.

SAM 3D Objects currently lacks checkpoint access: Hugging Face returned HTTP 403 for `facebook/sam-3d-objects`. That route needs checkpoint authorization before it can generate object meshes.

Mesh4D setup required two environment repairs:

- install `torch-cluster` for `torch_cluster.fps` in Mesh4D autoencoder blocks;
- install `plyfile` for `im2mesh.utils.io`;
- replace the bundled `im2mesh` pykdtree C extension with a SciPy `cKDTree` wrapper, because the bundled extension fails under Python 3.10 while Mesh4D uses KDTree query semantics.

## Acceptance for Closing V10

V10 closes after at least one generated video-conditioned mesh stream is evaluated through replay, track, physics, and visual inspection on the representative samples. Accepted delivery requires actual videos. Rejection is useful evidence when the generated stream fails for a localized mechanism, because that mechanism decides the V11 design.

The likely V11 branch depends on the V10 failure mode:

- if Mesh4D fails visible replay like single-image priors, move to multi-view VGGT/MoGe point fusion and learned completion constrained by masks;
- if Mesh4D passes visible replay but fails track factors, optimize a temporal surface graph over `A_t` and surfel offsets;
- if Mesh4D passes replay and tracks but fails contact physics, add contact-aware object-surface deformation before rendering;
- if setup or checkpoint access remains the blocker, prioritize accessible video-conditioned or multi-view reconstruction repos over additional single-image generators.
