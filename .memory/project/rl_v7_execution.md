# Active RL V7 implementation

User approved implementation on 2026-09-17, not another planning-only pass.

Read `docs/RL_V7_IMPLEMENTATION_PLAN.md` and
`.memory/tasks/2026-09-17-rl-v7-parity/README.md` before continuing.

Preserve this order: (1) real evidence chain → (2) offline object PPO refinement →
(3) full mesh/scale/MANO/visual pads → (4) portable JSON/RRD output →
(5) authorized-server real training and full-video comparison.

The reference is standalone RGB scene V7 (`fa9a84c`), not `docs/pipeline_v7.md`.
Do not hide legacy solvers in the RL environment, claim unsigned visual geometry
has signed authority, or close stages from CPU/schema/checkpoint evidence alone.
Do not run local heavy jobs or alter existing prediction outputs/model environments.
