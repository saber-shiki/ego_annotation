#!/usr/bin/env python3
"""Render only a validated final annotation package to layer-separated world RRD."""
from __future__ import annotations

import argparse
from pathlib import Path

from v19_annotation_layers import SIDES, load_package, require


def render(package_path, output, layer="both"):
    import rerun as rr
    import rerun.blueprint as rrb

    require(layer in ("prediction", "demo", "both"), "invalid layer selection")
    package, layers = load_package(package_path)
    selected = list(layers) if layer == "both" else [layer]
    require(all(s in layers for s in selected), "required component: demo layer not supplied")
    output = Path(output).resolve()
    require(not output.exists(), f"RRD output must be new: {output}")
    require(not output.is_relative_to(Path(package_path).resolve().parent / "assets"), "RRD may not modify snapshot assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("v19_annotation_layers", spawn=False)
    rr.save(str(output))
    rr.log("/world", rr.ViewCoordinates.RDF, static=True)
    rr.log("/world/authority", rr.TextDocument(package["run_mode"] + "\nPrediction: source metric state. Demo: visual hypothesis, not collision/contact authority."), static=True)
    p = layers["prediction"]
    rr.log("/world/camera_trajectory", rr.LineStrips3D([p["T_world_camera"][:, :3, 3]], colors=[150, 150, 150]), static=True)
    for i, fid in enumerate(p["frame_idx"]):
        rr.set_time("frame", sequence=int(fid))
        rr.set_time("time_s", duration=float(p["time_s"][i]))
        T = p["T_world_camera"][i]
        K = p["intrinsics_render"][i]
        rr.log("/world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log("/world/camera/image", rr.Pinhole(image_from_camera=[[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]], resolution=p["render_size_wh"][i]))
        rr.log("/world/camera/timestamp", rr.Scalars(float(p["time_s"][i])))
        for name in selected:
            state = layers[name]
            root = "/world/" + name
            if state["object_valid"][i]:
                rr.log(root + "/object", rr.Transform3D(translation=state["object_translation_world"][i], mat3x3=state["object_rotation"][i]))
                if i == 0 or not state["object_valid"][i - 1]:
                    # Canonical topology is constant. Dynamic logging at each
                    # valid-segment start restores it after recursive clears.
                    rr.log(root + "/object/mesh", rr.Mesh3D(vertex_positions=state["object_vertices_canonical"], triangle_indices=state["object_faces"], albedo_factor=[80, 170, 240, 255] if name == "prediction" else [110, 220, 110, 255]))
                    if i > 0:
                        rr.log(root + "/object/status", rr.Clear(recursive=True))
            else:
                rr.log(root + "/object", rr.Clear(recursive=True))
                rr.log(root + "/object/status", rr.TextLog("missing object pose; unresolved"))
            for side in SIDES:
                entity = root + "/hands/" + side
                if state[f"{side}_valid"][i]:
                    rr.log(entity, rr.Mesh3D(vertex_positions=state[f"{side}_vertices_world"][i], triangle_indices=state[f"{side}_faces"], albedo_factor=[235, 183, 158, 255]))
                    rr.log(entity + "/joints", rr.Points3D(state[f"{side}_joints_world"][i], radii=0.002))
                else:
                    rr.log(entity, rr.Clear(recursive=True))
            # P17/P16 remain prediction evidence, never inferred from visual fit.
            if name == "prediction":
                import json
                rr.log(root + "/interaction_state", rr.TextDocument(json.dumps(package["timeline"][i])))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*[
        rrb.Spatial3DView(origin="/world", name=name, contents=[f"+ /world/{name}/**", "+ /world/camera/**", "+ /world/camera_trajectory"])
        for name in selected
    ])))
    rr.get_global_data_recording().flush()
    rr.disconnect()
    require(output.is_file() and output.stat().st_size > 0, "RRD writer produced no data")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", choices=["prediction", "demo", "both"], default="both")
    a = p.parse_args()
    print(render(a.package, a.output, a.layer))


if __name__ == "__main__":
    main()
