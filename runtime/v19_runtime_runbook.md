# V19 Runtime Prediction Runbook

This is the runtime prediction runbook. It delegates exact step execution to `runtime/v19_runtime_phase_graph.md`.

## Runtime workspace contract

The runtime workspace contains only runtime files and allowed scripts. Use:

- `runtime/v19_runtime_ontology.md` for state-variable meanings;
- `runtime/v19_runtime_phase_graph.md` for exact phase order, scripts, command templates, inputs, outputs, and stop conditions;
- `configs/v19_agent_system_prompt.md` for runtime behavior;
- `.pi/prompts/v19-run.md` for launch argument binding;
- scripts named in the phase graph.

If a required file is absent, record `missing_runtime_bundle_component` in the run root and stop the affected branch. Do not search outside this workspace for project history or development instructions.

## Execution policy

1. Execute phase graph phases in order.
2. Do not discover or substitute scripts. The phase graph names the script for each script phase.
3. For an `agent writes` phase, write only the specified JSON/Markdown artifact and preserve uncertainty.
4. If a command template has a placeholder, bind it from launch arguments, phase outputs, or the named runtime ontology. If a placeholder cannot be bound without searching outside the bundle, record the unresolved placeholder as a blocker.
5. Heavy model phases run on the declared server target after probe and bundle sync. Light metadata/state phases may run locally.
6. Do not run scoring or comparisons inside the runtime run.

## Completion condition

The runtime run is complete when the run root contains render-consumed physical state, full-duration renders, and evidence showing either the physical mechanism works or the exact missing component prevents the next state variable. Created containers or valid JSON alone are not completion.
