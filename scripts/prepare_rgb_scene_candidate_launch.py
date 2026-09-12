#!/usr/bin/env python3
"""Create-only launch materialization for a current RGB candidate contract.

This parent-side utility emits direct absolute /bin/bash invocations. It never
runs a producer, authors judgments, or copies a historical launch template.
"""
from __future__ import annotations
import argparse, hashlib, json, shlex
from pathlib import Path

STAGES = (
    ("frontend-P09_DEPTH_ORDER", "frontend", "P09_DEPTH_ORDER"),
    ("frontend-P09-propose", "frontend", "P09-propose"),
    ("frontend-P09", "frontend", "P09"),
    ("frontend-P11", "frontend", "P11"),
    ("geometry", "geometry", None), ("initial-pose", "initial-pose", None),
    ("refined-pose", "refined-pose", None), ("appearance", "appearance", None),
    ("native-shape", "native-shape", None), ("shape", "shape", None),
    ("visual-fit", "visual-fit", None), ("export", "export", None),
)

def _safe(p: Path) -> Path:
    p = p.expanduser().resolve(strict=True)
    if p.is_symlink() or not p.is_dir(): raise ValueError(f"not a directory: {p}")
    return p

def materialize(bundle, candidate, output, *, gpu, device="cuda:0", image_width=None, image_height=None, python=None):
    bundle, candidate, output = _safe(Path(bundle)), Path(candidate).expanduser().resolve(strict=True), Path(output).expanduser().resolve()
    if not candidate.is_file(): raise ValueError("candidate contract is not a file")
    contract = json.loads(candidate.read_text())
    run_root = Path(contract.get("run_root", "")).expanduser().resolve(strict=True)
    runtime = bundle / "runtime/runtime_config.json"
    if not runtime.is_file(): raise ValueError("runtime config missing")
    if output.exists(): raise ValueError(f"create-only output already exists: {output}")
    if output == bundle or output.is_relative_to(bundle) or output.is_relative_to(run_root):
        raise ValueError("output must be separate from bundle and run")
    if not isinstance(gpu, int) or gpu < 0: raise ValueError("gpu must be non-negative integer")
    if image_width is not None and image_width <= 0 or image_height is not None and image_height <= 0:
        raise ValueError("image dimensions must be positive")
    py = Path(python).expanduser().resolve(strict=True) if python else Path("/usr/bin/env")
    if python and (not py.is_file() or not py.name.startswith("python")):
        raise ValueError("pinned Python must be an existing python executable")
    output.mkdir(parents=True)
    scripts = {}; hashes = {}
    common = [str(py)] if python else ["/usr/bin/env", "python3"]
    entry = bundle / "scripts/run_rgb_scene_stage.py"
    if not entry.is_file(): raise ValueError("current stage CLI missing")
    for name, stage, phase in STAGES:
        args = common + [str(entry), "--stage", stage, "--run-root", str(run_root),
                         "--runtime-config", str(runtime), "--gpu", str(gpu),
                         "--candidate-context", str(candidate)]
        if phase: args += ["--phase", phase]
        body = "#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(bundle)) + "\nexec " + " ".join(shlex.quote(x) for x in args) + "\n"
        path = output / f"{name}.sh"; path.write_text(body); hashes[str(path)] = hashlib.sha256(body.encode()).hexdigest(); scripts[name] = str(path)
    metadata = {"kind":"rgb_scene_candidate_launch_plan", "bundle":str(bundle), "candidate_context":str(candidate), "run_root":str(run_root), "candidate_id":contract.get("candidate_id"), "object_id":contract.get("object_id"), "selected_frame_idx":contract.get("selected_frame_idx"), "python":str(py), "gpu":gpu, "device":device, "image_width":image_width, "image_height":image_height, "stages":scripts, "stage_count":len(STAGES), "producer_execution":"not performed", "judgment_paths": {"P09_anchor":"state/anchor_decisions/<object_id>.json", "P10_selection":"state/object_selection.json", "appearance_anchor":"state/appearance_decisions/<object_id>.json", "P17":"state/agent_interaction_judgments/<object_id>.json"}}
    (output / "inspection_recipes.json").write_text(json.dumps({"raw_appearance":"inspect current RGB/mask evidence", "metric_eligibility":"inspect depth-owned metric eligibility", "P17":"inspect exact integer frame_idx and visibility", "final_pair":"inspect final annotation.json and world.rrd only", "dimensions":{"width":image_width,"height":image_height}, "device":device}, indent=2)+"\n")
    metadata["script_sha256"] = hashes
    (output / "launch_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle",type=Path,required=True); p.add_argument("--candidate-context",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--gpu",type=int,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--image-width",type=int); p.add_argument("--image-height",type=int); p.add_argument("--python",required=True)
    a=p.parse_args()
    print(json.dumps(materialize(a.bundle,a.candidate_context,a.output,gpu=a.gpu,device=a.device,image_width=a.image_width,image_height=a.image_height,python=a.python),indent=2))
if __name__ == "__main__": main()
