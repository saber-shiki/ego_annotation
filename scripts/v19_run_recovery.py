"""Explicit P09 continuation of one stopped run; never relabel old outputs.

The immutable birth input contract and original producer records are retained.
Only the exact recorded event prefix can use the original code identity after a
reviewed verification-only bundle transition. This is operator-approved run
lineage, not a security sandbox or evidence of prediction quality.
"""

from pathlib import Path
import hashlib
import json

from v19_runtime_contracts import in_run, load, sha

RECOVERY_DIR = Path("state/runtime_recovery/P09")
RECOVERY_FILE = RECOVERY_DIR / "recovery.json"
SCHEMA = "v19_same_run_P09_verification_recovery_v1"
ALLOWED_CHANGED_FILES = {
    "scripts/v19_run_lineage.py",
    "scripts/v19_run_recovery.py",
    "scripts/v19_runtime_contracts.py",  # reviewed recovery-only failure destination
    "runtime/rgb_sam3d_runtime_spec.md",
    "runtime/rgb_sam3d_system_prompt.md",
    ".pi/agents/ego-annotation.rgb-sam3d-runtime.md",
    "runtime/runtime_config.json",
}


def run_identity(contract):
    return {
        k: contract[k]
        for k in ("run_id", "run_root", "input_video_sha256", "bundle_manifest_sha256")
    }


def code_identity(manifest):
    return {k: v for k, v in manifest["files"].items() if k.endswith((".py", ".sh"))}


def event_digest(event):
    return hashlib.sha256(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized_config(value, bundle):
    if isinstance(value, dict):
        return {k: normalized_config(v, bundle) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized_config(v, bundle) for v in value]
    if isinstance(value, str) and value.startswith(str(bundle) + "/"):
        return "$BUNDLE/" + value[len(str(bundle)) + 1 :]
    return value


def validate_transition(old, new, old_config, new_config, old_bundle, new_bundle):
    changed = {
        k
        for k in set(old["files"]) | set(new["files"])
        if old["files"].get(k) != new["files"].get(k)
    }
    if not changed <= ALLOWED_CHANGED_FILES:
        raise ValueError(
            "P09 recovery changed prediction/model code or assets: "
            + ", ".join(sorted(changed - ALLOWED_CHANGED_FILES))
        )
    if old.get("external_assets") != new.get("external_assets"):
        raise ValueError("P09 recovery external model/tool identities differ")
    if normalized_config(old_config, old_bundle) != normalized_config(
        new_config, new_bundle
    ):
        raise ValueError("P09 recovery model/execution config changed")
    return sorted(changed)


def recovery_context(root, bundle, contract, manifest):
    """Default stays strict. A different execution bundle needs this exact seam."""
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    target_hash = sha(bundle / "runtime/bundle_manifest.json")
    if target_hash == contract["bundle_manifest_sha256"]:
        return None
    recovery_path = in_run(root / RECOVERY_FILE, root)
    recovery = load(recovery_path)
    identity = run_identity(contract)
    if (
        recovery.get("schema") != SCHEMA
        or recovery.get("mode") != "same_run_recovery"
        or recovery.get("resume_phase") != "P09"
        or recovery.get("run_identity") != identity
        or recovery.get("target_bundle") != str(bundle)
        or recovery.get("target_bundle_manifest_sha256") != target_hash
        or recovery.get("input_contract_sha256")
        != sha(root / "input/runtime_input_contract.json")
    ):
        raise ValueError("P09 recovery identity/phase/bundle mismatch")
    directory = root / RECOVERY_DIR
    old_manifest_path = in_run(directory / "source_bundle_manifest.json", root)
    old_config_path = in_run(directory / "source_runtime_config.json", root)
    old = load(old_manifest_path)
    if sha(old_manifest_path) != contract["bundle_manifest_sha256"]:
        raise ValueError("P09 recovery original bundle does not match birth contract")
    if sha(old_config_path) != old["files"]["runtime/runtime_config.json"]:
        raise ValueError("P09 recovery original config snapshot changed")
    changed = validate_transition(
        old,
        manifest,
        load(old_config_path),
        load(bundle / "runtime/runtime_config.json"),
        Path(recovery["source_bundle"]),
        bundle,
    )
    if changed != recovery.get("changed_bundle_files"):
        raise ValueError("P09 recovery change set differs")
    prefix_path = in_run(directory / "source_events.jsonl", root)
    prefix = prefix_path.read_bytes()
    if (
        hashlib.sha256(prefix).hexdigest() != recovery["source_events_sha256"]
        or len(prefix) != recovery["source_events_bytes"]
    ):
        raise ValueError("P09 recovery event snapshot changed")
    with in_run(root / "logs/integration/events.jsonl", root).open("rb") as handle:
        if handle.read(len(prefix)) != prefix:
            raise ValueError("P09 recovery original event prefix changed")
    if (
        recovery.get("object_id")
        != load(
            in_run(root / "measurements/object_candidates/object_plan_agent.json", root)
        )["objects"][0]["object_id"]
    ):
        raise ValueError("P09 recovery selected object differs")
    obj = recovery.get("object_id")
    if (
        not isinstance(obj, str)
        or not obj
        or obj in {".", ".."}
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in obj
        )
    ):
        raise ValueError("unsafe P09 recovery object identity")
    protected = recovery.get("protected_inputs", {})
    anchor = str(root / "state/anchor_decisions" / (obj + ".json"))
    plan = str(root / "measurements/object_candidates/object_plan_agent.json")
    blocker = str(root / "state/runtime_blockers/P09.json")
    if not {anchor, plan, blocker} <= set(protected):
        raise ValueError("P09 recovery missing protected decision/failure evidence")
    for name, digest in protected.items():
        if sha(in_run(name, root)) != digest:
            raise ValueError("P09 recovery protected input changed: " + name)
    source_code = code_identity(old)
    previous = set()
    for line in prefix.splitlines():
        event = json.loads(line)
        if event.get("returncode") == 0 and "error" not in event:
            if (
                event.get("run_identity") != identity
                or event.get("code_sha256") != source_code
            ):
                raise ValueError(
                    "P09 recovery event does not belong to original run/code"
                )
            previous.add(event_digest(event))
    if not previous:
        raise ValueError("P09 recovery has no original successful producers")
    return {
        "contract": recovery,
        "contract_sha256": sha(recovery_path),
        "execution_bundle_sha256": target_hash,
        "source_code": source_code,
        "previous_event_digests": previous,
    }


def event_code_is_valid(event, current_code, context):
    if context is None:
        return event.get("code_sha256") == current_code
    if event_digest(event) in context["previous_event_digests"]:
        return event.get("code_sha256") == context["source_code"]
    return (
        event.get("code_sha256") == current_code
        and event.get("execution_bundle_manifest_sha256")
        == context["execution_bundle_sha256"]
        and event.get("recovery_contract_sha256") == context["contract_sha256"]
    )
