#!/usr/bin/env python3
"""Build an isolated current-user V19 runtime bundle from a curated script closure."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)")
FORBIDDEN_RUNTIME_PATTERNS = {
    "/mnt/user-home/yiwen": "other-user home path",
    "/mnt/truenas-user-home/yiwen": "other-user NAS path",
    "/home/yiwen": "other-user home path",
    ".memory": "parent task memory",
    "Workbench": "development workbench language",
    "evaluation": "evaluator-phase language",
    "evaluator": "evaluator-phase language",
    "ablation": "evaluator/research objective",
    "EPISTEMIC": "parent task-memory language",
}
FORBIDDEN_BUNDLE_PATHS = tuple(
    pattern for pattern, label in FORBIDDEN_RUNTIME_PATTERNS.items() if label.startswith("other-user")
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
EXCLUDED_NAMED_SCRIPTS = {"render_v18_compact_rigid_tomato_temporal_mano_attempt.py"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_required(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} replacement, found {count}")
    return text.replace(old, new)


def adapt_runtime_spec(source: str, bundle_root: Path) -> str:
    declared_old = """- Runtime host: A800 compute host `yiwen@192.168.11.220`; Pi is launched inside a tmux session on this host.
- Runtime workspace: `/mnt/user-home/yiwen/ego_annotation_runtime/v19_bundle_a800`.
- Run roots and runtime inputs are A800-local/truenas paths under `/mnt/truenas-user-home/yiwen/ego_annotation_outputs`.
- HaWoR work root: `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work`
- HaWoR Python: `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/.venv_hawor/bin/python`
- SAM2 checkpoint: `/mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt`
- OWLv2 Python: `/mnt/user-home/yiwen/ego_annotation_remote/hunyuan3d_v3_env/bin/python`; this interpreter must import `transformers`, `torch`, `PIL`, and `cv2` before P06.
- OWLv2 model cache: `/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble`; this is a parent-preflighted local cache, not a runtime download.
- UniDepth checkout: `/mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work/UniDepth`
- Model Python for UniDepth/SAM2 on the A800 host: `/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2/bin/python`; this is a launch-preflighted contract."""
    declared_new = f"""- Runtime host: current A800 compute host `dexrobot-Standard-PC-Q35-ICH9-2009`; Pi is launched in a dedicated runtime tmux session on this host.
- Runtime workspace: `{bundle_root}`.
- Run roots and runtime inputs are current-user paths under `/mnt/truenas-user-home/kupingxin/ego_annotation_outputs` and `/mnt/truenas-user-home/kupingxin/ego_annotation_inputs`.
- General runtime Python: `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`.
- HaWoR work root: `/mnt/user-home/kupingxin/ego_annotation/.runtime/hawor_work`.
- HaWoR Python: `/mnt/user-home/kupingxin/ego_annotation/.runtime/hawor_work/.venv_hawor/bin/python`.
- MANO assets: `{bundle_root}/third_party/WiLoR/mano_data/MANO_LEFT.pkl` and `{bundle_root}/third_party/WiLoR/mano_data/MANO_RIGHT.pkl`.
- SAM2 source: `{bundle_root}/third_party/sam2`; checkpoint: `/mnt/user-home/kupingxin/ego_annotation/checkpoints/sam2.1_hiera_small.pt`.
- OWLv2 Python: `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`; this interpreter must import `transformers`, `torch`, `PIL`, and `cv2` before P06.
- OWLv2 model: `/mnt/truenas-user-home/kupingxin/ego_annotation_models/owlv2-base-patch16-ensemble-cfd3195b`; this is a launch-preflighted local snapshot, not a runtime download.
- UniDepth checkout: `/mnt/user-home/kupingxin/ego_annotation/.runtime/UniDepth`; model: `/mnt/user-home/kupingxin/ego_annotation/.runtime/models/unidepth-v2-vitl14`.
- UniDepth Python: `/mnt/user-home/kupingxin/ego_annotation/.runtime/model_envs/unidepth_sam2/bin/python`.
- TRELLIS checkout: `/mnt/user-home/kupingxin/ego_annotation/.runtime/trellis_work/TRELLIS`.
- TRELLIS Python: `/mnt/user-home/kupingxin/ego_annotation/.runtime/trellis_work/.venv_trellis/bin/python`.
- TRELLIS model: `/mnt/truenas-user-home/kupingxin/ego_annotation_models/trellis-image-large-25e0d31f`.
- Torch Hub cache for TRELLIS/DINOv2: `/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub`."""
    text = replace_required(source, declared_old, declared_new, "declared target block")

    placeholders_old = """- `{REMOTE_MODEL_PYTHON}`: `/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2/bin/python`, a launch-preflighted A800 model interpreter used for UniDepth/SAM2 Python phases.
- `{OWLV2_PYTHON}`: `/mnt/user-home/yiwen/ego_annotation_remote/hunyuan3d_v3_env/bin/python`, a launch-preflighted A800 interpreter used only for OWLv2 detector-box prompting."""
    placeholders_new = """- `{REMOTE_MODEL_PYTHON}`: `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`, the launch-preflighted general runtime interpreter.
- `{UNIDEPTH_PYTHON}`: `/mnt/user-home/kupingxin/ego_annotation/.runtime/model_envs/unidepth_sam2/bin/python`, the launch-preflighted UniDepth interpreter.
- `{OWLV2_PYTHON}`: `/mnt/user-home/kupingxin/ego_annotation/.venv/bin/python`, used only for OWLv2 detector-box prompting.
- `{TRELLIS_PYTHON}`: `/mnt/user-home/kupingxin/ego_annotation/.runtime/trellis_work/.venv_trellis/bin/python`.
- `{TRELLIS_REPO}`: `/mnt/user-home/kupingxin/ego_annotation/.runtime/trellis_work/TRELLIS`.
- `{TRELLIS_MODEL}`: `/mnt/truenas-user-home/kupingxin/ego_annotation_models/trellis-image-large-25e0d31f`."""
    text = replace_required(text, placeholders_old, placeholders_new, "placeholder block")

    text = replace_required(
        text,
        "CUDA_VISIBLE_DEVICES='{GPU_ID}' '{REMOTE_MODEL_PYTHON}' scripts/run_unidepth_full_frame_v3.py \\",
        "CUDA_VISIBLE_DEVICES='{GPU_ID}' '{UNIDEPTH_PYTHON}' scripts/run_unidepth_full_frame_v3.py \\",
        "UniDepth interpreter",
    )
    text = replace_required(
        text,
        "  --unidepth-repo /mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work/UniDepth \\",
        "  --unidepth-repo /mnt/user-home/kupingxin/ego_annotation/.runtime/UniDepth \\\n  --model-id /mnt/user-home/kupingxin/ego_annotation/.runtime/models/unidepth-v2-vitl14 \\",
        "UniDepth checkout",
    )
    text = replace_required(
        text,
        "EGO_HAWOR_ROOT=/mnt/user-home/yiwen/ego_annotation_remote/hawor_work \\",
        "HAWOR_CLIP='{RUN_ROOT}/input/hawor_sequence/{CASE_ID}.mp4'\nmkdir -p \"$(dirname \"$HAWOR_CLIP\")\"\nif [ ! -s \"$HAWOR_CLIP\" ]; then cp -- '{INPUT_VIDEO}' \"$HAWOR_CLIP\"; fi\n[ \"$(sha256sum '{INPUT_VIDEO}' | awk '{print $1}')\" = \"$(sha256sum \"$HAWOR_CLIP\" | awk '{print $1}')\" ]\nCUDA_VISIBLE_DEVICES='{GPU_ID}' \\\nEGO_HAWOR_ROOT=/mnt/user-home/kupingxin/ego_annotation/.runtime/hawor_work \\",
        "HaWoR root",
    )
    text = replace_required(
        text,
        "EGO_HAWOR_CLIP='{INPUT_VIDEO}' \\",
        "EGO_HAWOR_CLIP=\"$HAWOR_CLIP\" \\",
        "HaWoR isolated input",
    )
    text = replace_required(
        text,
        "  --owlv2-model /home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble \\",
        "  --owlv2-model /mnt/truenas-user-home/kupingxin/ego_annotation_models/owlv2-base-patch16-ensemble-cfd3195b \\",
        "OWLv2 model",
    )
    text = replace_required(
        text,
        "  --checkpoint /mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt \\",
        "  --checkpoint /mnt/user-home/kupingxin/ego_annotation/checkpoints/sam2.1_hiera_small.pt \\",
        "SAM2 checkpoint",
    )
    trellis_old = """PYTHONPATH="/mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/trellis_work/.venv_trellis/lib/python3.10/site-packages:${PYTHONPATH:-}" \\
"{REMOTE_MODEL_PYTHON}" scripts/remote_run_trellis_shape_v3.py \\
  --repo /mnt/user-home/yiwen/ego_annotation_remote/trellis_work/TRELLIS \\
  --image "$EVIDENCE_CROP_RGBA" \\
  --output-dir "{RUN_ROOT}/measurements/geometry_completion/trellis_{OBJECT_ID}_seed42" \\
  --seed 42"""
    trellis_new = """CUDA_VISIBLE_DEVICES='{GPU_ID}' TORCH_HOME=/mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub ATTN_BACKEND=xformers SPCONV_ALGO=native \\
"{TRELLIS_PYTHON}" scripts/remote_run_trellis_shape_v3.py \\
  --repo "{TRELLIS_REPO}" \\
  --model "{TRELLIS_MODEL}" \\
  --dinov2-repo /mnt/truenas-user-home/kupingxin/ego_annotation_models/torch_hub/hub/facebookresearch_dinov2_main \\
  --image "$EVIDENCE_CROP_RGBA" \\
  --output-dir "{RUN_ROOT}/measurements/geometry_completion/trellis_{OBJECT_ID}_seed42" \\
  --seed 42"""
    text = replace_required(text, trellis_old, trellis_new, "TRELLIS command")
    text = replace_required(
        text,
        "Those hypotheses must not automatically overwrite the metric MANO joint/root state used for evaluation.",
        "Those hypotheses must not automatically overwrite the canonical metric MANO joint/root state consumed downstream.",
        "runtime metric-state wording",
    )
    text = replace_required(
        text,
        "### P19c presentation rerender for Workbench item 4",
        "### P19c optional presentation rerender",
        "presentation heading",
    )
    text = replace_required(
        text,
        "8. Do not use sleep, polling loops, or idle waits. Long-running jobs need durable command logs/status files and inspectable job handles.",
        "8. Before each GPU-heavy phase, recheck live GPU memory/utilization and compute processes; if the P02 card is no longer safe, select another idle A800 and append the change to `harness_events.jsonl` before launching that phase.\n9. Do not use sleep, polling loops, or idle waits. Long-running jobs need durable command logs/status files and inspectable job handles.",
        "dynamic GPU recheck policy",
    )
    text = text.replace("parent-preflighted", "launch-preflighted")
    return text


def imported_local_modules(path: Path, available: dict[str, Path]) -> set[str]:
    found: set[str] = set()
    if path.suffix == ".py":
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            return found
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for name in names:
                if name in available:
                    found.add(name)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in SCRIPT_RE.findall(text):
            name = Path(match).stem
            if name in available:
                found.add(name)
    return found


def script_closure(source_root: Path, spec: str) -> list[Path]:
    scripts_root = source_root / "scripts"
    available = {p.stem: p for p in scripts_root.iterdir() if p.is_file() and p.suffix in {".py", ".sh"}}
    queue: list[str] = []
    for match in sorted(set(SCRIPT_RE.findall(spec))):
        path = Path(match)
        if path.name in EXCLUDED_NAMED_SCRIPTS:
            continue
        if path.stem not in available:
            raise FileNotFoundError(source_root / path)
        queue.append(path.stem)
    selected: set[str] = set()
    while queue:
        name = queue.pop()
        if name in selected:
            continue
        selected.add(name)
        for dependency in imported_local_modules(available[name], available):
            if dependency not in selected:
                queue.append(dependency)
    return sorted((available[name] for name in selected), key=lambda p: p.name)


def copy_tree_filtered(source: Path, destination: Path) -> None:
    def ignore(_: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {".git", "__pycache__", "demo", "notebooks", "training", "sav_dataset"}}
    shutil.copytree(source, destination, ignore=ignore)


def all_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def assert_bundle_path_isolation(root: Path) -> None:
    findings: list[str] = []
    for path in all_files(root):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_BUNDLE_PATHS:
            if pattern.lower() in text.lower():
                findings.append(f"{path.relative_to(root)}: {pattern}")
    if findings:
        raise RuntimeError("runtime bundle contains other-user absolute paths: " + "; ".join(findings))


def source_revision(path: Path, explicit_revision: str | None = None) -> tuple[str, str]:
    if explicit_revision:
        return explicit_revision, "explicit_revision"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return revision, "git_rev_parse"
    except (subprocess.CalledProcessError, FileNotFoundError):
        digest = hashlib.sha256()
        for file_path in all_files(path):
            relative = str(file_path.relative_to(path)).encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            file_digest = bytes.fromhex(sha256_file(file_path))
            digest.update(file_digest)
        return f"tree-sha256:{digest.hexdigest()}", "deterministic_tree_hash_no_git_metadata"


def run(args: argparse.Namespace) -> dict:
    source_root = args.source_root.resolve()
    bundle_root = args.bundle_root.resolve()
    if bundle_root.exists():
        if not args.replace:
            raise FileExistsError(f"bundle already exists: {bundle_root}")
        shutil.rmtree(bundle_root)
    (bundle_root / "runtime").mkdir(parents=True)
    (bundle_root / "configs").mkdir(parents=True)
    (bundle_root / ".pi" / "prompts").mkdir(parents=True)
    (bundle_root / "scripts").mkdir(parents=True)

    source_spec = (source_root / "runtime" / "v19_runtime_spec.md").read_text(encoding="utf-8")
    runtime_spec = adapt_runtime_spec(source_spec, bundle_root)
    for pattern, label in FORBIDDEN_RUNTIME_PATTERNS.items():
        if pattern.lower() in runtime_spec.lower():
            raise RuntimeError(f"runtime spec still exposes {label}: {pattern}")
    (bundle_root / "runtime" / "v19_runtime_spec.md").write_text(runtime_spec, encoding="utf-8")

    system_prompt = (source_root / "configs" / "v19_agent_system_prompt.md").read_text(encoding="utf-8")
    prompt_template = (source_root / ".pi" / "prompts" / "v19-run.md").read_text(encoding="utf-8")
    for name, text in (("system prompt", system_prompt), ("prompt template", prompt_template)):
        for pattern, label in FORBIDDEN_RUNTIME_PATTERNS.items():
            if pattern.lower() in text.lower():
                raise RuntimeError(f"{name} exposes {label}: {pattern}")
    (bundle_root / "configs" / "v19_agent_system_prompt.md").write_text(system_prompt, encoding="utf-8")
    (bundle_root / ".pi" / "prompts" / "v19-run.md").write_text(prompt_template, encoding="utf-8")

    scripts = script_closure(source_root, runtime_spec)
    for source in scripts:
        destination = bundle_root / "scripts" / source.name
        shutil.copy2(source, destination)
        if source.suffix == ".sh":
            destination.chmod(destination.stat().st_mode | stat.S_IXUSR)

    sam2_source = source_root / "third_party" / "sam2"
    copy_tree_filtered(sam2_source / "sam2", bundle_root / "third_party" / "sam2" / "sam2")
    for name in ("LICENSE", "LICENSE_cctorch"):
        if (sam2_source / name).exists():
            shutil.copy2(sam2_source / name, bundle_root / "third_party" / "sam2" / name)

    wilor_source = args.wilor_source.resolve()
    shutil.copytree(wilor_source / "wilor", bundle_root / "third_party" / "WiLoR" / "wilor", ignore=shutil.ignore_patterns("__pycache__"))
    (bundle_root / "third_party" / "WiLoR" / "mano_data").mkdir(parents=True)
    shutil.copy2(wilor_source / "mano_data" / "mano_mean_params.npz", bundle_root / "third_party" / "WiLoR" / "mano_data" / "mano_mean_params.npz")
    shutil.copy2(args.mano_left, bundle_root / "third_party" / "WiLoR" / "mano_data" / "MANO_LEFT.pkl")
    shutil.copy2(args.mano_right, bundle_root / "third_party" / "WiLoR" / "mano_data" / "MANO_RIGHT.pkl")
    shutil.copy2(wilor_source / "license.txt", bundle_root / "third_party" / "WiLoR" / "license.txt")

    subprocess.run([str(args.python), "-m", "compileall", "-q", str(bundle_root / "scripts")], check=True)
    for cache_dir in sorted(bundle_root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir)
    assert_bundle_path_isolation(bundle_root)
    files = [
        {"path": str(path.relative_to(bundle_root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in all_files(bundle_root)
    ]
    wilor_revision, wilor_revision_source = source_revision(wilor_source, args.wilor_source_revision)
    manifest = {
        "status": "curated_runtime_bundle_built",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "source_repo": str(source_root),
        "source_revision": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "wilor_source_revision": wilor_revision,
        "wilor_source_revision_source": wilor_revision_source,
        "wilor_source": str(wilor_source),
        "scripts": [path.name for path in scripts],
        "file_count": len(files),
        "files": files,
        "bundle_path_isolation": {
            "status": "ok",
            "forbidden_other_user_path_pattern_count": len(FORBIDDEN_BUNDLE_PATHS),
        },
        "excluded_runtime_context": [
            "AGENTS.md", ".memory/", "development docs", "evaluator/GT sidecars", "prior prediction outputs", "parent dirty working tree",
        ],
    }
    (bundle_root / "RUNTIME_BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--wilor-source", type=Path, required=True)
    parser.add_argument("--wilor-source-revision", default=None, help="Optional immutable revision inherited from a parent bundle manifest when --wilor-source is a curated non-git tree")
    parser.add_argument("--mano-left", type=Path, required=True)
    parser.add_argument("--mano-right", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
