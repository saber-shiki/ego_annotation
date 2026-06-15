# V19 Design Proposal: Pi-Harness Physical Annotation

V19 changes the annotation authority, not just the model stack.

The core shift is from a compiled outer pipeline to a Pi-harness annotation process. Scripts may still produce measurements, renders, masks, depth maps, MANO states, geometry candidates, residuals, and review artifacts, but they are instruments. They do not own the final physical-state decision.

The Pi agent owns the decision loop: inspect visual and geometric evidence, form competing explanations, request targeted measurements, revise beliefs when evidence conflicts, and write the renderable hand/object/contact/occlusion/pose state with explicit uncertainty. This makes the decision procedure open-world and context-sensitive rather than limited to branches anticipated in an outer script.

Native visual understanding is a side benefit, not the main design point. In V19, visual reasoning is part of the agent's cognition while inspecting videos/renders/artifacts; it is not a separate script that emits a `vlm_contact=true` field. This avoids laundering visual judgment through another brittle JSON channel that can be mis-weighted by fixed gates.

The practical implication is that cases like task5 tomato contact should not be decided by a hard-coded depth veto. The agent should weigh visible manipulation, temporal continuity, metric geometry, depth reliability, occlusion, and nonpenetration evidence together, then record the supported claim and its uncertainty.

V18 remains the current executable script-based pipeline. V19 is a different orchestration paradigm: Pi harness as annotation engine; scripts as evidence instruments.
