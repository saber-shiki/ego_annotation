#!/usr/bin/env python3
"""Snapshot P19 physical state and additive demo geometry into a new final package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v19_annotation_layers import (SCHEMA, SIDES, Snapshot, core_arrays, demo_arrays,
                                   load_json, load_npz, load_package, require,
                                   annotation_hawor_references, validate_hawor_bindings,
                                   object_target_matches, resolve_object, demo_mesh_contract,
                                   load_geometry_mesh, sha256)


def assemble(args):
    output = Path(args.output_dir).resolve()
    require(not output.exists(), f"output-dir must be new: {output}")
    require(not args.demo_report or args.demo_state, "demo-report requires demo-state")
    require(not args.demo_state or args.demo_report, "required component: demo-report with demo-state")
    snap = Snapshot(output, args.run_root, args.allow_external_replay)
    source_paths = {k: snap.source(getattr(args, k)) for k in
                    ("render_state", "annotations", "hawor_npz", "interaction_judgments")}
    core = load_json(source_paths["render_state"])
    ann = load_json(source_paths["annotations"])
    judgments = load_json(source_paths["interaction_judgments"])
    require(core.get("method") == "build_v19_rigid_render_state", "required component: P19 rigid render state")
    require("mano_constraint_state" in core and "constraint_rows" in core["mano_constraint_state"], "required component: P16 constraints")
    require(core.get("temporal_mano_state", {}).get("payload") is not None, "required component: P18/P18b temporal MANO payload")
    rows = judgments.get("interaction_judgments", judgments.get("judgments", judgments.get("segments")))
    require(isinstance(rows, list) and len(rows) > 0, "required component: explicit P17 interaction_judgments/judgments/segments schema")
    require(judgments.get("case") in (None, core.get("case")), "P17 case mismatch")
    require(judgments.get("status") in (None, "ok"), "P17 status is not ok")
    for row in rows:
        require(isinstance(row, dict), "invalid P17 judgment row")
        require(row.get("hand_side", row.get("side")) in SIDES, "required component: P17 hand side")
        require(row.get("contact_state") in {"likely_contact", "possible_contact", "no_contact", "unresolved"}, "required component: P17 contact state")
        require(row.get("occlusion_relation", row.get("visibility_relation")) in {
            "hand_in_front_of_object", "object_in_front_of_hand", "object_partially_occluded_by_hand", "no_visible_occlusion", "unresolved"}, "required component: P17 occlusion relation")
    core_dir = source_paths["render_state"].parent
    annotation_source = snap.source(core["inputs"]["annotations"], core_dir)
    require(annotation_source == source_paths["annotations"], "P19 annotation source mismatch")
    canonical = snap.source(core["object_geometry"]["completed_mesh_path"], core_dir)
    hawor_paths = {ref: snap.source(ref, source_paths["annotations"].parent)
                   for ref in annotation_hawor_references(ann)}
    hawor_bindings = {ref: {"asset": p, "sha256": sha256(p)} for ref, p in hawor_paths.items()}
    validate_hawor_bindings(ann, source_paths["hawor_npz"], hawor_bindings, lambda p: p)
    prediction = core_arrays(core, ann, load_npz(source_paths["hawor_npz"]), canonical)
    source_paths["completion_report"] = snap.source(core["inputs"]["completion_report"], core_dir)
    completion = load_json(source_paths["completion_report"])
    demo_source = demo_report = demo = None
    demo_files = []
    if args.demo_state:
        demo_source = snap.source(args.demo_state)
        demo_report = snap.source(args.demo_report)
        report = load_json(demo_report)
        demo_mesh_reference, digest = demo_mesh_contract(core, completion)
        demo_mesh = snap.source(demo_mesh_reference, source_paths["completion_report"].parent) if digest else canonical
        require(sha256(demo_mesh) == (digest or sha256(canonical)), "demo bound canonical mesh hash mismatch")
        demo = demo_arrays(load_npz(demo_source), prediction, load_geometry_mesh(demo_mesh))
        # Model weights are not render dependencies. Prediction NPZ/mesh/report
        # inputs are, and their producer-declared hashes must still match.
        for key, digest in report.get("input_hashes", {}).items():
            value = report.get("inputs", {}).get(key)
            if value is None:
                continue
            p = snap.source(value, demo_report.parent)
            require(sha256(p) == digest, f"demo report input hash mismatch: {key}")
            demo_files.append(p)
        if report.get("mesh"):
            p = snap.source(report["mesh"], demo_report.parent)
            require(report.get("mesh_sha256") == sha256(p), "demo report mesh hash mismatch")
            require(sha256(p) == sha256(demo_mesh), "demo report canonical mesh mismatch")
            demo_files.append(p)
        for key, expected in (("annotations", source_paths["annotations"]), ("hawor_params", source_paths["hawor_npz"])):
            if key in report.get("inputs", {}):
                actual = snap.source(report["inputs"][key], demo_report.parent)
                require(sha256(actual) == sha256(expected), f"demo report source mismatch: {key}")
    # Validate paths before making the output directory. Original JSON strings remain
    # verbatim evidence; renderers never dereference those provenance strings.
    core_files = [snap.source(v, core_dir) for k, v in core["inputs"].items() if v]
    physical = core["object_geometry"].get("physical_surface", {}).get("mesh")
    if physical:
        core_files.append(snap.source(physical, core_dir))
    # P18b can reference a full-surface NPZ outside P19.inputs; preserve it too.
    def archives(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and value.get(key + "_sha256"):
                    p = snap.source(child, core_dir)
                    require(sha256(p) == value[key + "_sha256"], f"core declared asset hash mismatch: {key}")
                yield from archives(child)
        elif isinstance(value, list):
            for child in value:
                yield from archives(child)
        elif isinstance(value, str) and value.lower().endswith((".npz", ".ply", ".obj", ".glb", ".stl")):
            yield snap.source(value, core_dir)
    core_files.extend(archives(core))
    core_files.extend(archives(ann.get("v19_inputs", {})))
    core_files.extend(demo_files)
    output.mkdir(parents=True)
    sources = {k: snap.copy(p) for k, p in source_paths.items()}
    for p in core_files:
        snap.copy(p)
    mesh_asset = snap.copy(canonical)
    metric = {"render_state": core, "annotations": ann, "interaction_judgments": judgments,
              "source_assets": sources, "canonical_mesh_asset": mesh_asset,
              "completion_report_binding": {"reference": core["inputs"]["completion_report"],
                                            "sha256": sha256(source_paths["completion_report"])},
              "hawor_source_bindings": {ref: {"asset": snap.copy(binding["asset"]), "sha256": binding["sha256"]}
                                        for ref, binding in hawor_bindings.items()},
              "render_arrays_asset": snap.generated("prediction_render_arrays.npz", prediction),
              "authority": "unchanged source metric prediction; no demo corrections or collision promotion"}
    demo_block = None
    if demo is not None:
        demo_block = {"source_assets": {"state": snap.copy(demo_source), "report": snap.copy(demo_report)},
                      "report": load_json(demo_report),
                      "canonical_mesh_binding": {"reference": demo_mesh_reference, "asset": snap.copy(demo_mesh)},
                      "render_arrays_asset": snap.generated("demo_render_arrays.npz", demo),
                      "authority": "visual hypothesis only; not contact/collision/sign evidence",
                      "annotation_ready": False, "collision_eligible": False}
    timeline = []
    for i, frame in enumerate(ann["frames"]):
        fid = int(prediction["frame_idx"][i])
        frame_judgments = []
        for row in rows:
            require(isinstance(row, dict), "invalid P17 judgment row")
            start = row.get("frame_start", row.get("start_frame", prediction["frame_idx"][0]))
            end = row.get("frame_end", row.get("end_frame", prediction["frame_idx"][-1]))
            require(int(end) >= int(start), "inverted P17 interval")
            target_id = row.get("target_entity_id", row.get("object_id", row.get("track_id")))
            matches = object_target_matches(frame, core["object_id"], target_id)
            if matches and int(start) <= fid <= int(end):
                frame_judgments.append(row)
        hands = {}
        for side in SIDES:
            raw = next((h for h in frame.get("hands", []) if h.get("hand_side") == side), {})
            hands[side] = {"valid": bool(prediction[f"{side}_valid"][i]),
                           "visibility_state": raw.get("visibility_state", "unresolved"),
                           "occluder": raw.get("occluder", "unresolved"),
                           "source_state": raw,
                           "uncertainty": raw.get("uncertainty", ["source_uncertainty_not_quantified"])}
        obj = resolve_object(frame, core["object_id"])
        timeline.append({"frame_idx": fid, "time_s": float(prediction["time_s"][i]),
                         "camera": {"T_world_camera": prediction["T_world_camera"][i].tolist(),
                                    "intrinsics_fx_fy_cx_cy": prediction["intrinsics_render"][i].tolist(),
                                    "source_size_wh": prediction["render_size_wh"][i].tolist()},
                         "object": {"pose_valid": bool(prediction["object_valid"][i]),
                                    "visibility_state": obj.get("visibility_state", "unresolved"),
                                    "occluder": obj.get("occluder", "unresolved"), "source_state": obj,
                                    "uncertainty": ["source_physical_uncertainty_preserved"] if prediction["object_valid"][i] else ["missing_object_pose"]},
                         "hands": hands, "interaction_judgments": frame_judgments,
                         "interaction_uncertainty": [] if frame_judgments else ["no_P17_judgment_for_frame"]})
    package = {"schema": SCHEMA, "run_mode": "external_replay_not_fresh" if args.allow_external_replay else "fresh_run_root_inputs",
               "coordinate_contract": {"world": "sequence_local_estimated_metric", "units": "m", "camera_axes": "RDF",
                                       "object": "V_world = V_completed_canonical @ R.T + t",
                                       "hands": "source MANO world surfaces; demo uses its own corrected surfaces"},
               "metric_prediction": metric, "demo_visual": demo_block, "timeline": timeline,
               "assets": snap.assets, "source_asset_map": snap.by_source}
    target = output / "annotation_layers.json"
    target.write_text(json.dumps(package, indent=2, allow_nan=False) + "\n")
    load_package(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("render-state", "annotations", "hawor-npz", "interaction-judgments", "output-dir", "run-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--demo-state", type=Path)
    p.add_argument("--demo-report", type=Path)
    p.add_argument("--allow-external-replay", action="store_true")
    print(assemble(p.parse_args()))


if __name__ == "__main__":
    main()
