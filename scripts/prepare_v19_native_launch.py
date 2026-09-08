#!/usr/bin/env python3
"""Prepare exact native-subagent requests; never launch Pi or prediction code.

The caller executes the generated request with the native subagent tool. A
successful tool probe verifies the actual registered actor/provider/cwd path,
not model inference or an operating-system filesystem sandbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import uuid
import zlib

from v19_video_timing import read_cfr_video_timing
from v19_runtime_assets import verify_bundle

AGENT = "ego-annotation.rgb-sam3d-runtime"
PROFILE = Path(".pi/agents/ego-annotation.rgb-sam3d-runtime.md")
REQUIRED_TOOLS = {
    "read",
    "write",
    "bash",
    "shellgate_connect",
    "shellgate_input",
    "contact_supervisor",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def profile_contract(path):
    text = Path(path).read_text()
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("native agent frontmatter missing")
    header = text.split("---", 2)[1]
    fields = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    expected = {
        "name": "rgb-sam3d-runtime",
        "package": "ego-annotation",
        "systemPromptMode": "replace",
        "defaultContext": "fresh",
        "inheritProjectContext": "false",
        "inheritGlobalContext": "false",
        "inheritSkills": "false",
    }
    allowed_fields = set(expected) | {
        "description",
        "tools",
        "thinking",
        "subagentOnlyExtensions",
    }
    if set(fields) - allowed_fields:
        raise ValueError(
            "undeclared native profile fields could alter supplied context"
        )
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(f"native profile context mismatch: {key}")
    if {s.strip() for s in fields.get("tools", "").split(",")} != REQUIRED_TOOLS:
        raise ValueError("native tool allowlist mismatch")
    extension = Path(fields.get("subagentOnlyExtensions", ""))
    if not extension.is_absolute() or not extension.is_file():
        raise ValueError("native ShellGate provider is not declared/available")
    providers = [
        extension,
        extension.with_name("shellgate-broker.py"),
        extension.parent.parent / "package.json",
        extension.parent.parent / "package-lock.json",
    ]
    if any(not p.is_file() for p in providers):
        raise ValueError("native ShellGate provider dependency is missing")
    return {
        "agent": AGENT,
        "profile_sha256": sha(path),
        "context": expected,
        "tools": sorted(REQUIRED_TOOLS),
        "provider_files_sha256": {str(p): sha(p) for p in providers},
    }


def verify_workspace(bundle):
    bundle = Path(bundle).resolve(strict=True)
    manifest_path = bundle / "runtime/bundle_manifest.json"
    manifest = load(manifest_path)
    if manifest.get("preflight_status") != "imports_checked":
        raise ValueError("candidate import preflight is not checked")
    if str(PROFILE) not in manifest.get("files", {}):
        raise ValueError(
            "project native agent profile is not registered in bundle manifest"
        )
    for relative, digest in manifest["files"].items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("invalid bundle manifest path")
        path = bundle / rel
        if (
            not path.resolve().is_relative_to(bundle)
            or path.is_symlink()
            or not path.is_file()
            or sha(path) != digest
        ):
            raise ValueError(f"changed/missing/escaped candidate file: {relative}")
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in bundle.rglob("*")) or actual != set(
        manifest["files"]
    ) | {"runtime/bundle_manifest.json"}:
        raise ValueError("candidate contains unregistered files or symlinks")
    forbidden = {"AGENTS.md", ".memory", ".git", "tests", "docs"}
    if any((bundle / name).exists() for name in forbidden):
        raise ValueError("candidate contains non-runtime context")
    registration = profile_contract(bundle / PROFILE)
    declared = manifest.get("external_assets", {}).get("files", {})
    for name, digest in registration["provider_files_sha256"].items():
        record = declared.get(name)
        if (
            not record
            or record.get("sha256") != digest
            or record.get("resolved_path") != str(Path(name).resolve())
        ):
            raise ValueError("native provider differs from declared candidate identity")
    registration.update(bundle=str(bundle), bundle_manifest_sha256=sha(manifest_path))
    return registration


def synthetic_probe_png(path, width=960, height=640):
    """Non-case image with enough bytes to exercise real image-tool transport."""
    pixels = bytearray(random.Random(739).randbytes(width * height * 3))
    for y in range(80, height - 80):
        for x in range(70, 380):
            p = (y * width + x) * 3
            pixels[p : p + 3] = bytes((235, 50, 40))
    for y in range(90, height - 90):
        half = int((y - 90) * 0.55)
        for x in range(max(420, 680 - half), min(width - 40, 680 + half + 1)):
            p = (y * width + x) * 3
            pixels[p : p + 3] = bytes((30, 220, 70))

    def chunk(kind, value):
        return (
            struct.pack("!I", len(value))
            + kind
            + value
            + struct.pack("!I", zlib.crc32(kind + value))
        )

    raw = b"".join(
        b"\0" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def write_request(bundle, root, mode, contract_path, output_dir):
    bundle, root, output_dir = (Path(p).resolve() for p in (bundle, root, output_dir))
    registration = verify_workspace(bundle)
    if output_dir.exists() or output_dir.is_relative_to(bundle):
        raise ValueError("new developer-side request directory outside bundle required")
    task = json.dumps(
        {
            "mode": mode,
            "runtime_cwd": str(bundle),
            "contract_path": str(Path(contract_path).resolve()),
            "run_root" if mode == "prediction" else "probe_root": str(root),
            "tmux_target": "ego_annotation_rgb_sam3d_" + uuid.uuid4().hex[:12],
        },
        sort_keys=True,
    )
    if mode == "tool_probe":
        # The exact target is carried by the probe contract as well.
        payload = load(contract_path)
        task_data = json.loads(task)
        task_data["tmux_target"] = payload["tmux_target"]
        task = json.dumps(task_data, sort_keys=True)
    script = (
        "const result = await runs.run('native-"
        + mode.replace("_", "-")
        + "', {agent: "
        + json.dumps(AGENT)
        + ", task: "
        + json.dumps(task)
        + ", output: "
        + json.dumps("native/" + mode + ".md")
        + "}); return {result, outputReference: result.outputReference, artifactPaths: result.artifactPaths};\n"
    )
    request = {
        "cwd": str(bundle),
        "agentScope": "project",
        "context": "fresh",
        "async": True,
        "worktree": False,
        "timeoutMs": 600000 if mode == "tool_probe" else 3600000,
        "toolTimeoutMs": 120000 if mode == "tool_probe" else 1800000,
        "workflowScriptPath": str(output_dir / "native.workflow.js"),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "native.workflow.js").write_text(script)
    (output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (output_dir / "registration.json").write_text(
        json.dumps(
            {
                **registration,
                "mode": mode,
                "contract_path": str(Path(contract_path).resolve()),
                "contract_sha256": sha(contract_path),
                "request_sha256": sha(output_dir / "request.json"),
                "workflow_sha256": sha(output_dir / "native.workflow.js"),
                "native_harness_execution": "pending",
                "prediction_executed": False,
                "boundary": "explicit supplied context and role/tool contract; not an OS filesystem sandbox",
            },
            indent=2,
        )
        + "\n"
    )
    return request


def prepare_probe(bundle, probe_root, output_dir):
    bundle, root = Path(bundle).resolve(), Path(probe_root).resolve()
    output_dir = Path(output_dir).resolve()
    verify_workspace(bundle)
    if root.exists() or root.is_relative_to(bundle):
        raise ValueError("new isolated probe root required")
    if (
        output_dir.exists()
        or output_dir.is_relative_to(bundle)
        or root.is_relative_to(output_dir)
    ):
        raise ValueError("new separate developer request directory required")
    root.mkdir(parents=True)
    marker = root / "input_marker.txt"
    marker.write_text("native-tool-probe-" + uuid.uuid4().hex + "\n")
    image = root / "transport_probe.png"
    synthetic_probe_png(image)
    contract = {
        "mode": "tool_probe",
        "runtime_cwd": str(bundle),
        "probe_root": str(root),
        "tmux_target": "ego_annotation_rgb_sam3d_probe_" + uuid.uuid4().hex[:12],
        "input_marker_file": str(marker),
        "input_marker_sha256": sha(marker),
        "input_image": str(image),
        "input_image_sha256": sha(image),
        "output_json": str(root / "tool_probe_result.json"),
        "required_output_fields": [
            "mode",
            "observed_cwd",
            "marker_text",
            "image_description",
            "managed_shell_verified",
            "prediction_executed",
        ],
        "allowed_execution": "cwd/printf plus file/image read-write only; no prediction scripts or GPU",
        "prediction_executed": False,
    }
    target = root / "probe_contract.json"
    target.write_text(json.dumps(contract, indent=2) + "\n")
    return write_request(bundle, root, "tool_probe", target, output_dir)


def prepare_prediction_request(bundle, run_root, output_dir):
    bundle, root = Path(bundle).resolve(), Path(run_root).resolve()
    registration = verify_workspace(bundle)
    contract_path = root / "input/runtime_input_contract.json"
    contract = load(contract_path)
    if (
        not contract.get("run_id")
        or contract.get("bundle_manifest_sha256")
        != registration["bundle_manifest_sha256"]
    ):
        raise ValueError("current prepared run identity/bundle binding missing")
    video = Path(contract["input_video"]).resolve(strict=True)
    if not video.is_relative_to(root) or sha(video) != contract["input_video_sha256"]:
        raise ValueError("prepared raw input identity mismatch")
    read_cfr_video_timing(video)
    for name in ("measurements", "renders"):
        if any(p.is_file() for p in (root / name).rglob("*")):
            raise ValueError(
                "initial native launch cannot consume pre-existing predictions"
            )
    # Authorized initial prediction launch uses the full asset path. The
    # lightweight synthetic probe checks only the tooling it actually uses.
    verify_bundle(bundle)
    return write_request(bundle, root, "prediction", contract_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tool-probe", "prediction-request"])
    for name in ("bundle-root", "run-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    fn = prepare_probe if args.mode == "tool-probe" else prepare_prediction_request
    print(json.dumps(fn(args.bundle_root, args.run_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
