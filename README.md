# Ego Annotation — visual-contact demo branch

This branch preserves the V19/V20 research pipeline and adds a **demo-only
hand–object visual fitting and rendering path**. The delivered example is a
150-frame milk-carton clip. It is not a claim of measured physical contact,
annotation readiness, or arbitrary-video support.

## Start here

- **[Demo replay and refitting](docs/V20_VISUAL_CONTACT_DEMO_REPRODUCE.md)**:
  required inputs, verified case limits, environment notes, commands and outputs.
- [Demo design](docs/V20_VISUAL_CONTACT_DEMO_DESIGN.md) and
  [visual/numerical results](docs/V20_VISUAL_CONTACT_DEMO_RESULTS.md).
- **[V19 fresh-run reproducibility plan](docs/V19_FRESH_TO_DEMO_VALIDATION.md)**:
  distinguishes raw-video prediction, cached demo refitting, render replay and
  cross-video transfer.

The independent delivered artifacts are stored outside Git:

```text
/mnt/truenas-user-home/kupingxin/ego_annotation_outputs/demos/milk_visual_contact_20260908/
  milk_hand_object_demo.mp4
  milk_contact_detail.mp4
  milk_hand_object_demo.rrd
  contact_keyframes.jpg
  contact_before_after_120.jpg
  state/demo_state.npz
  SHA256SUMS
```

Both videos are 1920×960, 30 FPS, 150 frames / 5 seconds. The detail video's
reconstruction is explicitly a +20° inspection view, not a second observed
camera. The original V16 archive and formal annotation state remain unchanged.
Raw videos, predictions, licensed MANO models, model weights and MP4/RRD files
are **not distributed through this branch**.

## Code entry points

```text
scripts/run_v20_visual_contact_demo.py     # explicit replay or fit-render launcher
scripts/fit_v20_visual_contact_demo.py     # actual MANO/object fitting
scripts/render_v20_visual_contact_demo.py  # combined triangle-depth rendering + RRD
scripts/v20_demo_geometry.py               # camera, MANO and triangle utilities
```

The launcher is **not** a replacement for the V19 phase graph. It never
installs environments or starts an inference agent. Fitting must run on an
explicitly selected server/GPU in managed tmux; use a fresh output directory.

## Focused tests

Run with the preflighted demo Python environment:

```bash
python -m unittest discover -s tests -p 'test_v20_visual_contact_demo.py' -v
python -m unittest discover -s tests -p 'test_v20_demo_runner.py' -v
```

The first suite checks geometry/projection/occlusion and Torch–NumPy transform
parity. The second checks launcher arguments, fresh-output protection, dry-run
behavior and subprocess failure reporting. Tests are necessary but do not
replace inspection of full videos and contact close-ups.
