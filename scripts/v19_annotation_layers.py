"""Validated, self-contained V19 annotation layers (no physical-state fitting)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil

import numpy as np

SCHEMA = "v19_annotation_layers_v1"
SIDES = ("left", "right")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    data = json.loads(Path(path).read_text())
    require(isinstance(data, dict), f"required JSON object: {path}")
    return data


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def array(data, key, shape=None):
    require(key in data, f"required component: {key}")
    a = np.asarray(data[key])
    require(shape is None or a.shape == shape, f"invalid shape: {key}: {a.shape}, expected {shape}")
    return a


def rigid(rotation, translation, label):
    require(np.isfinite(rotation).all() and np.isfinite(translation).all(), f"nonfinite transform: {label}")
    require(np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5)
            and np.allclose(np.linalg.det(rotation), 1, atol=1e-5), f"invalid rigid rotation: {label}")


def same(a, b, label, atol=1e-5):
    require(np.shape(a) == np.shape(b) and np.allclose(a, b, atol=atol, rtol=0), f"mismatch: {label}")


def mesh(vertices, faces, label):
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0
            and np.isfinite(vertices).all(), f"invalid mesh vertices: {label}")
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0
            and faces.max() < len(vertices), f"invalid mesh faces: {label}")


def load_geometry_mesh(path):
    """Load only embedded geometry; no material, texture, or buffer dependency reads."""
    import trimesh
    path = Path(path)
    file_type = path.suffix.lower().lstrip(".")
    require(file_type in {"ply", "obj", "stl", "glb"}, "unsupported self-contained mesh format")
    # BytesIO has no filename from which Trimesh could infer a filesystem
    # resolver. The empty mapping denies every external dependency by default.
    m = trimesh.load(io.BytesIO(path.read_bytes()), file_type=file_type,
                     resolver={}, allow_remote=False, force="mesh", process=False,
                     skip_materials=True)
    require(isinstance(m, trimesh.Trimesh), "required component: canonical triangle mesh")
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    mesh(v, f, "canonical object")
    return {"object_vertices_canonical": v, "object_faces": f}


def object_alias(value):
    text = str(value)
    return text.removeprefix("object:")


def resolve_object(frame, object_id):
    """Resolve established object:/track aliases, rejecting competing rows."""
    target = object_alias(object_id)
    matches = [row for row in frame.get("objects", [])
               if any(value is not None and object_alias(value) == target
                      for value in (row.get("object_id"), row.get("track_id")))]
    require(len(matches) <= 1, f"ambiguous object/track identity: {object_id}")
    return matches[0] if matches else {}


def object_target_matches(frame, object_id, target):
    if target in (None, ""):
        return True
    source = resolve_object(frame, object_id)
    if source:
        return resolve_object(frame, target) is source
    return object_alias(target) == object_alias(object_id)


def annotation_hawor_references(annotations):
    """Archive identity binds all rows, including invalid rows without joints."""
    references = set()
    declared = annotations.get("v19_inputs", {}).get("hawor_npz")
    if declared:
        references.add(declared)
    for frame in annotations.get("frames", []):
        for hand in frame.get("hands", []):
            ref = (hand.get("metric_mano_state") or {}).get("vertices_reference")
            if ref:
                require(ref.get("source_hawor_npz"), "required component: vertices_reference.source_hawor_npz")
                references.add(ref["source_hawor_npz"])
    return sorted(references)


def validate_hawor_bindings(annotations, supplied_path, bindings, resolve_asset):
    require(set(bindings) == set(annotation_hawor_references(annotations)), "annotation HaWoR source bindings mismatch")
    digest = sha256(supplied_path)
    for reference, binding in bindings.items():
        require(sha256(resolve_asset(binding["asset"])) == binding["sha256"] == digest,
                f"annotation HaWoR source archive mismatch: {reference}")


def hand_arrays(source, n, demo=False):
    out = {}
    for side in SIDES:
        suffix = "world" if demo else "world_m"
        vk, jk = f"{side}_vertices_{suffix}", f"{side}_joints_{suffix}"
        valid_key = f"{side}_valid"
        # An absent side is explicitly invalid, never a zero mesh at the origin.
        if vk not in source and jk not in source:
            require(valid_key not in source or not np.any(source[valid_key]), f"missing valid hand geometry: {side}")
            out[valid_key] = np.zeros(n, dtype=bool)
            continue
        valid = array(source, valid_key, (n,))
        require(np.isin(valid, [0, 1]).all(), f"invalid hand validity: {side}")
        valid = valid.astype(bool)
        v = array(source, vk, (n, 778, 3))
        j = array(source, jk, (n, 21, 3))
        f = array(source, f"{side}_faces")
        require(np.isfinite(v[valid]).all() and np.isfinite(j[valid]).all(), f"nonfinite valid hand: {side}")
        mesh(np.zeros((778, 3)), f, side)
        out.update({valid_key: valid, f"{side}_vertices_world": v,
                    f"{side}_joints_world": j, f"{side}_faces": f})
    require(any(f"{s}_vertices_world" in out for s in SIDES), "required component: source MANO surfaces/joints")
    return out


class Snapshot:
    def __init__(self, output, run_root, allow_external):
        self.output = Path(output).resolve()
        self.run_root = Path(run_root).resolve(strict=True)
        self.allow_external = allow_external
        self.assets = {}
        self.by_source = {}

    def source(self, path, relative_to=None):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (relative_to or self.run_root) / p
        p = p.resolve(strict=True)
        require(p.is_file(), f"required asset is not a file: {p}")
        require(self.allow_external or p.is_relative_to(self.run_root), f"fresh input escapes run-root: {p}")
        return p

    def copy(self, path, relative_to=None):
        p = self.source(path, relative_to)
        if str(p) in self.by_source:
            return self.by_source[str(p)]
        digest = sha256(p)
        rel = f"assets/{len(self.assets):04d}_{p.name}"
        target = self.output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        require(sha256(target) == digest == sha256(p), f"source changed during snapshot: {p}")
        self.assets[rel] = {"sha256": digest, "size_bytes": target.stat().st_size, "source": str(p)}
        self.by_source[str(p)] = rel
        return rel

    def generated(self, name, data):
        rel = f"assets/{name}"
        np.savez_compressed(self.output / rel, **data)
        p = self.output / rel
        self.assets[rel] = {"sha256": sha256(p), "size_bytes": p.stat().st_size, "source": "derived_lossless_render_arrays"}
        return rel


def core_arrays(core, annotations, hawor, mesh_path):
    frames = annotations.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, "required component: annotation frames")
    n = len(frames)
    ids = np.asarray([f["frame_idx"] for f in frames])
    require(np.issubdtype(ids.dtype, np.integer) and (np.diff(ids) > 0).all(), "reordered/duplicate annotation frame IDs")
    same(ids, core["annotation_backbone"]["frame_ids"], "P19 timeline", atol=0)
    same(ids, array(hawor, "frame_idx", (n,)), "HaWoR timeline", atol=0)
    time = np.asarray([f["time_s"] for f in frames], dtype=float)
    require(np.isfinite(time).all() and (np.diff(time) > 0).all(), "invalid timestamp timeline")
    sizes, intrinsics, transforms = [], [], []
    for f in frames:
        cam = f.get("camera", {})
        T = np.asarray(cam.get("T_world_camera_metric", cam.get("T_world_camera")))
        require(T.shape == (4, 4), "required component: camera T_world_camera_metric")
        same(T[3], [0, 0, 0, 1], "camera homogeneous row")
        rigid(T[:3, :3], T[:3, 3], "camera")
        K = cam.get("intrinsics_fx_fy_cx_cy", cam.get("current_v18_camera_intrinsics_fx_fy_cx_cy"))
        K = np.asarray(K, dtype=float)
        require(K.shape == (4,) and np.isfinite(K).all() and (K[:2] > 0).all(), "required component: camera intrinsics")
        raw = annotations.get("raw_video") or {}
        wh = [f.get("source_width") or raw.get("width"), f.get("source_height") or raw.get("height")]
        require(all(isinstance(x, (int, float)) and x > 0 and int(x) == x for x in wh), "required component: source image dimensions")
        sizes.append(wh); intrinsics.append(K); transforms.append(T)
    T = np.asarray(transforms)
    same(array(hawor, "R_c2w", (n, 3, 3)), T[:, :3, :3], "HaWoR world camera rotation")
    same(array(hawor, "t_c2w", (n, 3)), T[:, :3, 3], "HaWoR world camera translation")
    require(core["object_geometry"].get("mesh_frame") == "completed_canonical", "required component: completed_canonical mesh convention")
    geometry = load_geometry_mesh(mesh_path)
    v, faces = geometry["object_vertices_canonical"], geometry["object_faces"]
    R, t = np.full((n, 3, 3), np.nan), np.full((n, 3), np.nan)
    valid = np.zeros(n, dtype=bool)
    index = {int(f): i for i, f in enumerate(ids)}
    for row in core["object_pose_trajectory"]["pose_rows"]:
        idx = row["frame_idx"]
        require(idx in index and not valid[index[idx]], "duplicate/out-of-timeline object pose")
        i = index[idx]
        R[i] = array(row, "rotation_world_from_completed_canonical_matrix", (3, 3))
        t[i] = array(row, "translation_world_m", (3,))
        rigid(R[i], t[i], f"object frame {idx}")
        valid[i] = True
    out = dict(frame_idx=ids, time_s=time, render_size_wh=np.asarray(sizes), intrinsics_render=np.asarray(intrinsics),
               T_world_camera=T, object_vertices_canonical=v, object_faces=faces,
               object_rotation=R, object_translation_world=t, object_valid=valid)
    out.update(hand_arrays(hawor, n))
    for i, frame in enumerate(frames):
        for hand in frame.get("hands", []):
            side = hand.get("hand_side")
            metric = hand.get("metric_mano_state") or {}
            reference = metric.get("vertices_reference")
            if reference and side in SIDES:
                pos = reference.get("source_frame_index")
                require(isinstance(pos, int) and 0 <= pos < n and ids[pos] == ids[i], "annotation HaWoR source row mismatch")
                for kind in ("vertices", "joints"):
                    declared = reference.get(f"source_{kind}_world_array")
                    require(declared in hawor, f"required annotation source array: {declared}")
                    actual = array(hawor, declared)[pos]
                    expected = out[f"{side}_{kind}_world"][i]
                    require(actual.shape == expected.shape and np.array_equal(actual, expected, equal_nan=True), f"annotation HaWoR source {kind} mismatch")
            joints = metric.get("joints_current_v18_world_m", metric.get("joints_world_m"))
            if side in SIDES and joints is not None:
                require(out[f"{side}_valid"][i], f"annotation/HaWoR validity mismatch: {side}")
                same(joints, out[f"{side}_joints_world"][i], f"annotation/HaWoR source joints: {side}")
    return out


def demo_mesh_contract(core, completion):
    if "render_mesh_contract" not in completion:
        return core["object_geometry"]["completed_mesh_path"], None
    contract = completion["render_mesh_contract"]
    require(isinstance(contract, dict) and contract.get("required_mesh_path")
            and isinstance(contract.get("required_mesh_sha256"), str)
            and len(contract["required_mesh_sha256"]) == 64,
            "required component: completion render_mesh_contract path/hash")
    return contract["required_mesh_path"], contract["required_mesh_sha256"]


def demo_arrays(demo, prediction, canonical_geometry=None):
    n = len(prediction["frame_idx"])
    for key in ("frame_idx", "time_s", "T_world_camera"):
        same(array(demo, key), prediction[key], f"demo {key}")
    size = array(demo, "render_size_wh")
    if size.shape == (2,):
        size = np.broadcast_to(size, (n, 2))
    require(size.shape == (n, 2) and np.isfinite(size).all() and (size > 0).all()
            and np.equal(size, np.floor(size)).all(), "invalid demo render_size_wh")
    scale = size / prediction["render_size_wh"]
    expected = prediction["intrinsics_render"].copy()
    expected[:, :2] *= scale
    expected[:, 2:] = (expected[:, 2:] + 0.5) * scale - 0.5
    same(array(demo, "intrinsics_render", (n, 4)), expected, "demo intrinsics/render size")
    geometry = prediction if canonical_geometry is None else canonical_geometry
    for key in ("object_vertices_canonical", "object_faces"):
        same(array(demo, key), geometry[key], f"demo canonical mesh {key}",
             atol=0 if key == "object_faces" else 1e-5)
    valid = prediction["object_valid"]
    for key in ("object_rotation", "object_translation_world"):
        base = array(demo, "base_" + key, prediction[key].shape)
        same(base[valid], prediction[key][valid], f"demo base {key}")
    R = array(demo, "object_rotation", (n, 3, 3))
    t = array(demo, "object_translation_world", (n, 3))
    dv = np.asarray(demo.get("object_valid", np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1)))
    require(dv.shape == (n,) and np.isin(dv, [0, 1]).all(), "invalid demo object validity")
    dv = dv.astype(bool)
    rigid(R[dv], t[dv], "demo object")
    hands = hand_arrays(demo, n, demo=True)
    for side in SIDES:
        same(hands[f"{side}_valid"], prediction[f"{side}_valid"], f"demo {side} validity", atol=0)
        hv = hands[f"{side}_valid"]
        if f"{side}_vertices_world" in hands:
            for kind in ("vertices", "joints"):
                key = f"{side}_{kind}_world"
                same(array(demo, f"{side}_base_{kind}_world")[hv], prediction[key][hv], f"demo source {side} {kind}")
            same(hands[f"{side}_faces"], prediction[f"{side}_faces"], f"demo {side} topology", atol=0)
    return {**{k: prediction[k] for k in ("frame_idx", "time_s", "T_world_camera")},
            **{k: geometry[k] for k in ("object_vertices_canonical", "object_faces")},
            "render_size_wh": size, "intrinsics_render": expected, "object_rotation": R,
            "object_translation_world": t, "object_valid": dv, **hands}


def package_asset(root, relative):
    require(isinstance(relative, str) and "\\" not in relative, "invalid relative asset path")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and ".." not in p.parts and p.parts and p.parts[0] == "assets", "asset path traversal/absolute path")
    resolved = (root / relative).resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), "asset escapes package or is missing")
    return resolved


def load_package(path):
    path = Path(path).resolve(strict=True)
    package = load_json(path)
    require(package.get("schema") == SCHEMA, "unsupported final package schema")
    root = path.parent
    assets = package["assets"]
    for relative, spec in assets.items():
        p = package_asset(root, relative)
        require(p.stat().st_size == spec["size_bytes"] and sha256(p) == spec["sha256"], f"asset hash mismatch: {relative}")
    def asset(relative):
        require(relative in assets, f"unregistered asset: {relative}")
        return package_asset(root, relative)
    metric = package["metric_prediction"]
    # Rebuild and compare: hashes alone do not prove internal timeline/camera consistency.
    core = load_json(asset(metric["source_assets"]["render_state"]))
    ann = load_json(asset(metric["source_assets"]["annotations"]))
    require(core == metric["render_state"] and ann == metric["annotations"], "verbatim source document mismatch")
    require(load_json(asset(metric["source_assets"]["interaction_judgments"])) == metric["interaction_judgments"], "P17 source document mismatch")
    hawor_path = asset(metric["source_assets"]["hawor_npz"])
    validate_hawor_bindings(ann, hawor_path, metric.get("hawor_source_bindings", {}), asset)
    expected = core_arrays(core, ann, load_npz(hawor_path), asset(metric["canonical_mesh_asset"]))
    pred = load_npz(asset(metric["render_arrays_asset"]))
    compare_arrays(pred, expected, "prediction")
    timeline = package["timeline"]
    require(len(timeline) == len(pred["frame_idx"]), "package timeline length mismatch")
    judgments = metric["interaction_judgments"]
    source_rows = judgments.get("interaction_judgments", judgments.get("judgments", judgments.get("segments")))
    for i, row in enumerate(timeline):
        fid = int(pred["frame_idx"][i])
        require(row["frame_idx"] == fid and row["time_s"] == pred["time_s"][i], "package timeline/time mismatch")
        for key, expected_key in (("T_world_camera", "T_world_camera"), ("intrinsics_fx_fy_cx_cy", "intrinsics_render"), ("source_size_wh", "render_size_wh")):
            same(row["camera"][key], pred[expected_key][i], f"package camera {key}")
        require(row["object"]["pose_valid"] == bool(pred["object_valid"][i]), "package object validity mismatch")
        for side in SIDES:
            require(row["hands"][side]["valid"] == bool(pred[f"{side}_valid"][i]), "package hand validity mismatch")
            source_hand = next((h for h in ann["frames"][i].get("hands", []) if h.get("hand_side") == side), {})
            require(row["hands"][side]["source_state"] == source_hand, "package source hand mismatch")
            for field, default in (("visibility_state", "unresolved"), ("occluder", "unresolved"), ("uncertainty", ["source_uncertainty_not_quantified"])):
                require(row["hands"][side][field] == source_hand.get(field, default), f"package hand {field} mismatch")
        source_object = resolve_object(ann["frames"][i], core["object_id"])
        require(row["object"]["source_state"] == source_object, "package source object mismatch")
        for field in ("visibility_state", "occluder"):
            require(row["object"][field] == source_object.get(field, "unresolved"), f"package object {field} mismatch")
        uncertainty = ["source_physical_uncertainty_preserved"] if pred["object_valid"][i] else ["missing_object_pose"]
        require(row["object"]["uncertainty"] == uncertainty, "package object uncertainty mismatch")
        selected = []
        for judgment in source_rows:
            target = judgment.get("target_entity_id", judgment.get("object_id", judgment.get("track_id")))
            matches = object_target_matches(ann["frames"][i], core["object_id"], target)
            start = judgment.get("frame_start", judgment.get("start_frame", pred["frame_idx"][0]))
            end = judgment.get("frame_end", judgment.get("end_frame", pred["frame_idx"][-1]))
            if matches and int(start) <= fid <= int(end):
                selected.append(judgment)
        require(row["interaction_judgments"] == selected, "package P17 timeline mismatch")
        require(row["interaction_uncertainty"] == ([] if selected else ["no_P17_judgment_for_frame"]), "package interaction uncertainty mismatch")
    layers = {"prediction": pred}
    demo = package.get("demo_visual")
    if demo is not None:
        require(demo.get("annotation_ready") is False and demo.get("collision_eligible") is False, "demo physical authority promotion forbidden")
        require(load_json(asset(demo["source_assets"]["report"])) == demo["report"], "demo report source mismatch")
        report_binding = metric["completion_report_binding"]
        require(report_binding["reference"] == core["inputs"]["completion_report"], "completion report binding mismatch")
        completion_path = asset(metric["source_assets"]["completion_report"])
        require(sha256(completion_path) == report_binding["sha256"], "completion report binding hash mismatch")
        completion = load_json(completion_path)
        reference, digest = demo_mesh_contract(core, completion)
        binding = demo["canonical_mesh_binding"]
        require(binding["reference"] == reference, "demo canonical mesh binding mismatch")
        mesh_path = asset(binding["asset"])
        expected_digest = digest or sha256(asset(metric["canonical_mesh_asset"]))
        require(sha256(mesh_path) == expected_digest, "demo bound canonical mesh hash mismatch")
        if demo["report"].get("mesh"):
            require(demo["report"].get("mesh_sha256") == expected_digest, "demo report canonical mesh mismatch")
        geometry = load_geometry_mesh(mesh_path)
        source = load_npz(asset(demo["source_assets"]["state"]))
        expected_demo = demo_arrays(source, pred, geometry)
        actual = load_npz(asset(demo["render_arrays_asset"]))
        compare_arrays(actual, expected_demo, "demo")
        layers["demo"] = actual
    return package, layers


def compare_arrays(actual, expected, label):
    require(actual.keys() == expected.keys(), f"{label} render array keys mismatch")
    for key in expected:
        a, b = actual[key], expected[key]
        require(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-7, equal_nan=True), f"{label} render array mismatch: {key}")
