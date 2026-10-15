"""Export the zero/best-trained/last comparison as portable interactive world meshes.

Read-only posthoc visualization: no policies, perception, fitting, GT, decimation,
world offsets, or source updates. Requires the provisioned Rerun SDK 0.37.2.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import uuid

import numpy as np

SIDES = ("left", "right")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_candidates(run: Path):
    summary = json.loads((run / "summary.json").read_text())
    zero_document = run / "zero/candidate.json"
    zero = json.loads(zero_document.read_text())["policy_rollout_array_layer"]
    items = [{"key": "zero", "name": "Zero R6", "steps": 0,
              "objective": summary["zero_objective"], "path": zero["path"],
              "sha256": zero["sha256"], "source_document": str(zero_document)}]
    for key, row in (("best", summary["best_trained_checkpoint"]),
                     ("last", summary["checkpoints"][-1])):
        evaluation_path = Path(row["evaluation"])
        manifest_path = Path(json.loads(evaluation_path.read_text())["manifest"])
        manifest = json.loads(manifest_path.read_text())
        if manifest["episode_count"] != 1 or manifest["episodes"][0]["completion"] != "natural_termination":
            raise ValueError("comparison requires one complete naturally terminated episode")
        checkpoint = Path(row["checkpoint"])
        if sha(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("evaluation checkpoint hash mismatch")
        artifact = next(a for a in manifest["episodes"][0]["artifacts"] if a["format"] == "npz")
        path = Path(artifact["path"])
        path = path if path.is_absolute() else manifest_path.parent / path
        items.append({"key": key, "name": f'{"Best trained" if key == "best" else "Last"} {row["steps"]}',
                      "steps": row["steps"], "objective": row["objective"], "path": str(path),
                      "sha256": artifact["sha256"], "source_document": str(manifest_path),
                      "checkpoint_sha256": manifest["checkpoint_sha256"]})
    archives = []
    for item in items:
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"candidate input hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            archives.append(dict(archive))
    validate_archives(archives)
    return items, archives


def validate_archives(archives):
    if len(archives) != 3:
        raise ValueError("comparison needs exactly zero, best trained and last")
    zero = archives[0]
    ids, times = zero["frame_idx"], zero["time_s"]
    n = len(ids)
    if n < 2 or len(set(ids)) != n or np.any(np.diff(times) <= 0):
        raise ValueError("invalid full timeline")
    for a in archives:
        for key in ("frame_idx", "time_s", "T_world_camera", "object_vertices_canonical", "object_faces"):
            np.testing.assert_array_equal(a[key], zero[key])
        for key in ("object_rotation", "object_translation_world", "T_world_camera"):
            if len(a[key]) != n or not np.isfinite(a[key]).all():
                raise ValueError(f"invalid temporal geometry: {key}")
        np.testing.assert_allclose(np.linalg.det(a["object_rotation"]), 1, atol=1e-6)
        np.testing.assert_allclose(a["object_rotation"].transpose(0, 2, 1) @ a["object_rotation"],
                                   np.broadcast_to(np.eye(3), (n, 3, 3)), atol=1e-6)
        for prefix in ("object", "left", "right"):
            vertex_key = "object_vertices_canonical" if prefix == "object" else prefix + "_vertices_world"
            vertices, faces = a[vertex_key], a[prefix + "_faces"]
            if vertices.shape[-1] != 3 or faces.shape[-1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("invalid mesh")
            if faces.size == 0 or faces.min() < 0 or faces.max() >= vertices.shape[-2]:
                raise ValueError("invalid triangle index")
            if prefix != "object":
                if len(vertices) != n:
                    raise ValueError("incomplete MANO timeline")
                # This exporter mirrors the frozen world MP4 validity, not new
                # acceptance decisions. Reject if that would discard a display mesh.
                np.testing.assert_array_equal(a[prefix + "_valid"], a[prefix + "_display_valid"])


def fixed_eye(archives):
    """Same initial eye/target/up used by preprocessed_learning.render_comparison."""
    zero = archives[0]
    target = np.median(zero["object_translation_world"], axis=0)
    C0 = zero["T_world_camera"][0]
    radius = max(.2, max(float(np.max(np.linalg.norm(a["object_translation_world"] - target, axis=1)))
                          for a in archives) + .12)
    position = target + C0[:3, :3] @ np.array([radius * .7, -radius * .25, -radius * 2.5])
    up = C0[:3, :3] @ np.array([0., -1., 0.])
    return {"position": position.tolist(), "look_target": target.tolist(), "eye_up": up.tolist()}


def export_rrd(items, archives, output: Path, *, collision_reports=None):
    import rerun as rr
    import rerun.blueprint as rrb
    if collision_reports is not None:
        if len(collision_reports) != len(archives):
            raise ValueError("one collision audit per rendered candidate required")
        for report, a in zip(collision_reports, archives):
            np.testing.assert_array_equal([r["frame_idx"] for r in report["frames"]], a["frame_idx"])
    if output.exists() or output.with_suffix(".rrd.json").exists():
        raise ValueError("RRD and receipt destinations must be fresh")
    validate_archives(archives)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name("." + output.name + ".staging-" + uuid.uuid4().hex)
    eye = fixed_eye(archives)
    views = [rrb.Spatial3DView(
        origin=f'/comparison/{item["key"]}/world', name=item["name"],
        contents="$origin/**", background=[242, 242, 242], line_grid=False,
        eye_controls=rrb.EyeControls3D(kind="Orbital", **eye),
    ) for item in items]
    fps = 1. / float(np.median(np.diff(archives[0]["time_s"])))
    blueprint = rrb.Blueprint(
        rrb.Horizontal(*views),
        rrb.TimePanel(state="expanded", timeline="frame", fps=fps, play_state="Paused"),
        rrb.SelectionPanel(state="collapsed"), rrb.BlueprintPanel(state="collapsed"),
        auto_layout=False, auto_views=False,
    )
    recording_id = str(uuid.uuid4())
    recording = rr.RecordingStream("milk_zero_best_last_world", recording_id=recording_id)
    recording.save(staging, default_blueprint=blueprint)
    receipt = {
        "schema": "ego.annotation.world_comparison_rrd.v1", "rerun_sdk": importlib.metadata.version("rerun-sdk"),
        "recording_id": recording_id, "units": "meters", "world_offsets_applied": False,
        "transform_convention": "column vectors; world_from_object; MANO vertices already in world",
        "timeline": {"sequence": "frame", "duration": "time_s", "frames": len(archives[0]["frame_idx"]),
                     "fps": fps, "first_frame": int(archives[0]["frame_idx"][0]),
                     "last_frame": int(archives[0]["frame_idx"][-1])},
        "initial_eye": eye, "view_layout": "Zero R6 / Best trained / Last; same initial camera and world gauge",
        "candidates": items, "native_vertices": len(archives[0]["object_vertices_canonical"]),
        "native_triangles": len(archives[0]["object_faces"]), "decimation_applied": False,
        "physical_contact_authority": False, "signed_geometry_ready": False,
        "scope": "posthoc geometry-only interactive visualization, not a new prediction or improvement claim",
        "collision_highlight": "red MANO vertices >0.1mm inside generated mesh plus actual crossing hand edges; hypothesis only" if collision_reports is not None else None,
        "render_notes": "Actual mesh surfaces, not video billboards. Interactive Rerun lighting/perspective may differ from MP4. No coordinate re-alignment or per-candidate normalization.",
    }
    counts = {item["key"]: {"object": 0, "left": 0, "right": 0} for item in items}
    try:
        recording.log("/comparison/provenance", rr.TextDocument(json.dumps(receipt, indent=2)), static=True)
        for item in items:
            recording.log(f'/comparison/{item["key"]}/world', rr.ViewCoordinates.RDF, static=True)
            recording.log(f'/comparison/{item["key"]}/notes', rr.TextDocument(
                f'{item["name"]}; raw J={item["objective"]:.9f}. '
                "Best trained is not necessarily better than zero. Full native object / metric MANO. No signed-contact claim."
            ), static=True)
        for i, fid in enumerate(archives[0]["frame_idx"]):
            recording.set_time("frame", sequence=int(fid))
            recording.set_time("time_s", duration=float(archives[0]["time_s"][i]))
            for candidate_index, (item, a) in enumerate(zip(items, archives)):
                root = f'/comparison/{item["key"]}/world'
                valid = bool(a["object_valid"][i])
                if valid:
                    recording.log(root + "/object", rr.Transform3D(
                        translation=a["object_translation_world"][i], mat3x3=a["object_rotation"][i]))
                    # Log on frame 0 and after a clear; latest-at then retains
                    # complete rigid topology while only the transform changes.
                    if i == 0 or not bool(a["object_valid"][i - 1]):
                        recording.log(root + "/object", rr.Mesh3D(
                            vertex_positions=a["object_vertices_canonical"], triangle_indices=a["object_faces"],
                            albedo_factor=[245, 180, 90, 255], face_rendering="DoubleSided"))
                    counts[item["key"]]["object"] += 1
                else:
                    recording.log(root + "/object", rr.Clear(recursive=True))
                for side, color in (("left", [75, 150, 225, 255]), ("right", [225, 185, 95, 255])):
                    if bool(a[side + "_valid"][i]):
                        faces = a[side + "_faces"]
                        faces = faces[i] if faces.ndim == 3 else faces
                        appearance = {"albedo_factor": color}
                        if collision_reports is not None:
                            stats = collision_reports[candidate_index]["frames"][i]["hands"][side]
                            colors = np.tile(np.asarray(color, np.uint8), (len(a[side + "_vertices_world"][i]), 1))
                            colors[stats.get("inside_vertex_ids", [])] = [245, 30, 60, 255]
                            appearance = {"vertex_colors": colors, "albedo_factor": [255, 255, 255, 255]}
                            edge_ids = np.asarray(stats.get("crossing_edge_vertex_ids", []), dtype=np.int64).reshape(-1, 2)
                            edge_path = root + "/collision/" + side
                            if len(edge_ids):
                                recording.log(edge_path, rr.LineStrips3D(a[side + "_vertices_world"][i][edge_ids],
                                    colors=[245, 30, 60], radii=.00025))
                            else:
                                recording.log(edge_path, rr.Clear(recursive=True))
                        recording.log(root + "/hands/" + side, rr.Mesh3D(
                            vertex_positions=a[side + "_vertices_world"][i], triangle_indices=faces,
                            **appearance, face_rendering="DoubleSided"))
                        counts[item["key"]][side] += 1
                    else:
                        recording.log(root + "/hands/" + side, rr.Clear(recursive=True))
        recording.flush()
    finally:
        recording.disconnect()
    # A visualization export must never mutate the recorded training inputs.
    for item in items:
        if sha(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("candidate changed during visualization export")
    if output.exists():
        raise ValueError("destination appeared during export")
    staging.rename(output)
    receipt.update({"rrd": str(output), "rrd_sha256": sha(output), "bytes": output.stat().st_size,
                    "geometry_frames_logged": counts})
    output.with_suffix(".rrd.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def verify_roundtrip(output: Path, items, archives):
    """Read actual RRD chunks back; compare all triangles, MANO frames and poses."""
    from rerun.experimental import RrdReader
    reader = RrdReader(output)
    if len(reader.recordings()) != 1 or len(reader.blueprints()) != 1:
        raise ValueError("RRD must contain one portable recording and its default blueprint")
    meshes = {}
    poses = {}
    by_key = {item["key"]: a for item, a in zip(items, archives)}
    positions = {int(fid): i for i, fid in enumerate(archives[0]["frame_idx"])}
    time_ns = np.round(archives[0]["time_s"] * 1e9).astype(np.int64)
    for chunk in reader.stream().to_chunks():
        path = chunk.entity_path
        parts = path.strip("/").split("/")
        if len(parts) < 4 or parts[0] != "comparison" or parts[1] not in by_key or parts[2] != "world":
            continue
        a = by_key[parts[1]]
        batch = chunk.to_record_batch()
        if "frame" not in batch.schema.names:
            continue
        def component(name, row):
            if name not in batch.schema.names:
                return None
            scalar = batch.column(name)[row]
            if not scalar.is_valid or len(scalar.values) == 0:
                return None
            # FixedSizeListArray.values exposes the full backing buffer, even
            # for a sliced row. flatten() respects the per-row Arrow offset.
            return scalar.values.flatten().to_numpy(zero_copy_only=False)
        for row in range(batch.num_rows):
            fid = int(batch.column("frame")[row].as_py())
            i = positions[fid]
            np.testing.assert_allclose(batch.column("time_s").cast("int64")[row].as_py(), time_ns[i], atol=1, rtol=0)
            if parts[3] == "object":
                v = component("Mesh3D:vertex_positions", row)
                faces = component("Mesh3D:triangle_indices", row)
                if v is not None:
                    np.testing.assert_array_equal(v.reshape(-1, 3), a["object_vertices_canonical"])
                    np.testing.assert_array_equal(faces.reshape(-1, 3), a["object_faces"])
                    meshes.setdefault(path, []).append(fid)
                rotation = component("Transform3D:mat3x3", row)
                translation = component("Transform3D:translation", row)
                if rotation is not None:
                    np.testing.assert_array_equal(rotation.reshape(3, 3, order="F"), a["object_rotation"][i].astype(np.float32))
                    np.testing.assert_array_equal(translation, a["object_translation_world"][i].astype(np.float32))
                    poses.setdefault(path, []).append(fid)
            elif parts[3] == "hands" and len(parts) == 5:
                side = parts[4]
                v = component("Mesh3D:vertex_positions", row)
                if v is not None:
                    faces = component("Mesh3D:triangle_indices", row)
                    expected_faces = a[side + "_faces"]
                    if expected_faces.ndim == 3:
                        expected_faces = expected_faces[i]
                    np.testing.assert_array_equal(v.reshape(-1, 3), a[side + "_vertices_world"][i])
                    np.testing.assert_array_equal(faces.reshape(-1, 3), expected_faces)
                    meshes.setdefault(path, []).append(fid)
    for item, a in zip(items, archives):
        root = f'/comparison/{item["key"]}/world'
        np.testing.assert_array_equal(sorted(poses.get(root + "/object", [])), a["frame_idx"][a["object_valid"]])
        starts = a["object_valid"] & np.r_[True, ~a["object_valid"][:-1]]
        np.testing.assert_array_equal(sorted(meshes.get(root + "/object", [])), a["frame_idx"][starts])
        for side in SIDES:
            np.testing.assert_array_equal(sorted(meshes.get(root + "/hands/" + side, [])), a["frame_idx"][a[side + "_valid"]])
    return {"all_mesh_vertices_and_triangle_indices_equal": True,
            "all_object_transforms_equal_after_sdk_float32_encoding": True,
            "time_s_verified_to_nanosecond": True,
            "mesh_frame_counts": {k: len(v) for k, v in meshes.items()},
            "object_pose_frame_counts": {k: len(v) for k, v in poses.items()},
            "embedded_blueprint": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    output = args.output or args.run / "renders/zero_best_last_world.rrd"
    items, archives = load_candidates(args.run)
    receipt = export_rrd(items, archives, output)
    verification = verify_roundtrip(output, items, archives)
    output.with_suffix(".rrd.verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("rrd", "rrd_sha256", "bytes", "timeline", "geometry_frames_logged")}, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
