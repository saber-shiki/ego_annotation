#!/usr/bin/env python3
"""Demo-only image-preserving MANO/SAM3D interaction fitting, no GT/P09 gate."""
from pathlib import Path
import argparse, json, time
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
from v20_demo_geometry import (
    JOINT_ORDER,
    TIP_VERTICES,
    camera,
    closest,
    load,
    make_scene,
    mano_models,
    project,
    resized_K,
    sha,
    torch_project,
    torch_rodrigues,
)
from v20_prediction_contracts import assert_prediction_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "annotations",
        "pose-report",
        "mano-bridge",
        "hawor-params",
        "mano-models",
        "output-dir",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--object-id", default="carton_milk")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=240)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise ValueError("new output directory required")
    out.mkdir(parents=True)
    start = time.monotonic()
    device = torch.device(args.device)
    import trimesh

    ann = load(args.annotations)
    rep = load(args.pose_report)
    assert_prediction_only(rep, label="demo initial pose", object_id=args.object_id)
    rows = {r["frame_idx"]: r for r in rep["pose_rows"]}
    meshpath = Path(rep["render_mesh_contract"]["required_mesh_path"])
    if sha(meshpath) != rep["render_mesh_contract"]["required_mesh_sha256"]:
        raise ValueError("mesh identity mismatch")
    mesh = trimesh.load(meshpath, process=False)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    scene = make_scene(V, F)
    z = np.load(args.hawor_params, allow_pickle=False)
    bridge = np.load(args.mano_bridge, allow_pickle=False)
    ids = np.asarray(z["frame_idx"], int)
    N = len(ids)
    assert N == 150 and np.array_equal(ids, np.arange(N))
    key = np.unique(np.r_[np.arange(0, N, 5), N - 1])
    B = len(key)
    T = np.asarray([f["camera"]["T_world_camera_metric"] for f in ann["frames"]])
    K = np.asarray(
        [
            resized_K(
                f["camera"]["intrinsics_fx_fy_cx_cy"],
                (f["source_width"], f["source_height"]),
            )
            for f in ann["frames"]
        ]
    )
    R0 = np.asarray(
        [rows[i]["rotation_world_from_completed_canonical_matrix"] for i in ids]
    )
    t0 = np.asarray([rows[i]["translation_world_m"] for i in ids])
    models = mano_models(args.mano_models, device)
    for model in models.values():
        model.requires_grad_(False)
    to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    # Check full MANO replay against the original 778-vertex geometry, not just parameters/schema.
    data = {}
    maxreplay = {}
    for side in ["left", "right"]:
        root = z[side + "_root_orient_axis_angle"]
        pose = z[side + "_hand_pose_axis_angle"]
        beta = z[side + "_betas"]
        trans = z[side + "_trans_world_m"]
        with torch.no_grad():
            rr = models[side](
                global_orient=to(root), hand_pose=to(pose), betas=to(beta)
            )
            replay = rr.vertices + to(trans)[:, None]
        source = z[side + "_vertices_world_m"]
        bi = np.asarray(
            [
                np.flatnonzero(
                    (bridge["frame_idx"] == i) & (bridge["hand_side"] == side)
                )[0]
                for i in ids
            ]
        )
        if not np.allclose(
            bridge["vertices_current_v18_world_from_hawor_projection_relift_m"][bi],
            source,
            atol=1e-6,
        ):
            raise ValueError("MANO bridge/source mismatch")
        err = np.linalg.norm(replay.cpu().numpy() - source, axis=-1)
        maxreplay[side] = float(err.max())
        if err.max() > 1e-4:
            raise ValueError(f"MANO replay inconsistent {side}: {err.max()}")
        joints = z[side + "_joints_world_m"]
        joints_replay = (
            (
                torch.cat([rr.joints[:, :16], rr.vertices[:, TIP_VERTICES]], dim=1)[
                    :, JOINT_ORDER
                ]
                + to(trans)[:, None]
            )
            .detach()
            .cpu()
            .numpy()
        )
        if np.max(np.linalg.norm(joints_replay - joints, axis=-1)) > 1e-4:
            raise ValueError(f"MANO joint order inconsistent {side}")
        # Surface pad samples: fixed vertex ids around each fingertip in actual MANO topology.
        sample = np.asarray(
            [cKDTree(source[i]).query(source[i, TIP_VERTICES], k=12)[1] for i in key]
        )
        pix = np.asarray([project(camera(source[i], T[i]), K[i]) for i in key])
        jpix = np.asarray([project(camera(joints[i], T[i]), K[i]) for i in key])
        data[side] = dict(
            root=to(root[key]),
            pose=to(pose[key]),
            beta=to(beta[key]),
            trans=to(trans[key]),
            base_vertices=to(source[key]),
            base_joints=to(joints[key]),
            base_pixel=to(pix),
            base_joint_pixel=to(jpix),
            patch=sample,
        )
    # Convex silhouette SUPPORT of the actual mesh, used only as an image loss
    # (not substituted for object reconstruction). Full triangles are rendered.
    mask_contours = []
    contact_gate = np.zeros((B, 2, 5), np.float32)
    for k, i in enumerate(key):
        obj = next(
            o for o in ann["frames"][i]["objects"] if o["object_id"] == args.object_id
        )
        m = cv2.imread(obj["mask_path"], 0)
        if m is None or m.shape != (960, 960):
            raise ValueError("expected object mask on960 plane")
        ys, xs = np.where(m > 0)
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype("float32")).reshape(
            -1, 2
        )
        edge = np.roll(hull, -1, axis=0) - hull
        length = np.linalg.norm(edge, axis=1)
        cum = np.r_[0, np.cumsum(length)]
        positions = np.linspace(0, cum[-1], 160, endpoint=False)
        ix = np.searchsorted(cum, positions, side="right") - 1
        contour = (
            hull[ix] + (positions - cum[ix])[:, None] / length[ix, None] * edge[ix]
        )
        mask_contours.append(contour)
        dist = cv2.distanceTransform((m == 0).astype("uint8"), cv2.DIST_L2, 5)
        for si, side in enumerate(["left", "right"]):
            uv = data[side]["base_joint_pixel"][k, [4, 8, 12, 16, 20]].cpu().numpy()
            p = np.rint(uv).astype(int)
            inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < 960) & (p[:, 1] < 960)
            p = np.clip(p, 0, 959)
            # Thumbs seen supporting the carton are primary demo hypotheses.
            contact_gate[k, si, 0] = float(inside[0] and dist[p[0, 1], p[0, 0]] < 16)
            base = data[side]["base_vertices"][k].cpu().numpy()
            q = (base[TIP_VERTICES] - t0[i]) @ R0[i]
            near, n = closest(scene, q)
            gap = np.linalg.norm(q - near, axis=1)
            for finger in range(1, 5):
                contact_gate[k, si, finger] = 0.12 * float(
                    inside[finger]
                    and dist[p[finger, 1], p[finger, 0]] < 8
                    and gap[finger] < 0.012
                )
    (out / "contact_hypotheses.json").write_text(
        json.dumps(
            {
                "kind": "demo_visual_grasp_hypothesis_not_measured_contact",
                "keyframes": key.tolist(),
                "weights": contact_gate.tolist(),
                "tip_vertices": TIP_VERTICES.tolist(),
                "raw_frames_inspected": [0, 35, 65, 95, 120, 140, 149],
            },
            indent=2,
        )
        + "\n"
    )
    # Only ~1000 surface points are needed to derive silhouette indices. The
    # original full mesh is used for every contact query and output render.
    vid = np.linspace(0, len(V) - 1, 5000, dtype=int)
    ov = to(V[vid])
    base_R = to(R0[key])
    base_t = to(t0[key])
    cam_R = to(T[key, :3, :3])
    cam_t = to(T[key, :3, 3])
    intr = to(K[key])[:, None]
    hand_delta = torch.nn.Parameter(torch.zeros((B, 2, 45), device=device))
    hand_root = torch.nn.Parameter(torch.zeros((B, 2, 3), device=device))
    hand_t = torch.nn.Parameter(torch.zeros((B, 2, 3), device=device))
    hand_scale = torch.nn.Parameter(torch.zeros((B, 2, 1), device=device))
    obj_delta = torch.nn.Parameter(torch.zeros((B, 6), device=device))
    params = [hand_delta, hand_root, hand_t, hand_scale, obj_delta]
    opt = torch.optim.Adam(params, lr=0.008)
    weights = to(contact_gate)
    targets = {}
    outline = {}
    trace = []
    samp = np.unique(np.r_[np.arange(0, 778, 4), TIP_VERTICES])
    sample_indices = to(samp).long()

    def forward():
        R = torch_rodrigues(obj_delta[:, :3]) @ base_R
        t = base_t + torch.einsum("bij,bj->bi", cam_R, obj_delta[:, 3:])
        hands = {}
        for si, side in enumerate(["left", "right"]):
            d = data[side]
            o = models[side](
                global_orient=d["root"],
                hand_pose=d["pose"] + hand_delta[:, si],
                betas=d["beta"],
            )
            vw = o.vertices + d["trans"][:, None]
            jw = (
                torch.cat([o.joints[:, :16], o.vertices[:, TIP_VERTICES]], dim=1)[
                    :, JOINT_ORDER
                ]
                + d["trans"][:, None]
            )
            pivot = jw[:, :1]
            dR = torch_rodrigues(hand_root[:, si])
            vw = (vw - pivot) @ dR.transpose(1, 2) + pivot
            jw = (jw - pivot) @ dR.transpose(1, 2) + pivot
            vc = (vw - cam_t[:, None]) @ cam_R
            jc = (jw - cam_t[:, None]) @ cam_R
            scale = torch.exp(hand_scale[:, si])[:, None]
            vc = vc * scale + hand_t[:, si, None]
            jc = jc * scale + hand_t[:, si, None]
            vw = vc @ cam_R.transpose(1, 2) + cam_t[:, None]
            jw = jc @ cam_R.transpose(1, 2) + cam_t[:, None]
            hands[side] = (vw, jw, vc, jc)
        return R, t, hands

    for it in range(args.iterations):
        opt.zero_grad()
        R, t, hands = forward()
        if it % 12 == 0:
            with torch.no_grad():
                for si, side in enumerate(["left", "right"]):
                    vw = hands[side][0]
                    qc = (vw - t[:, None]) @ R
                    q, n = closest(scene, qc.cpu().numpy())
                    targets[side] = (to(q), to(n))
                for k, i in enumerate(key):
                    vc = (ov @ R[k].T + t[k] - cam_t[k]) @ cam_R[k]
                    uv = torch_project(vc, intr[k]).cpu().numpy()
                    h = cv2.convexHull(
                        uv.astype("float32"), returnPoints=False
                    ).reshape(-1)
                    # Sample the full projected mesh boundary, not only hull
                    # corner vertices. Corner-only reverse matching falsely
                    # pulled long carton edges toward corners.
                    hu = uv[h]
                    length = np.linalg.norm(np.roll(hu, -1, axis=0) - hu, axis=1)
                    cum = np.r_[0, np.cumsum(length)]
                    pos = np.linspace(0, cum[-1], 160, endpoint=False)
                    edge = np.searchsorted(cum, pos, side="right") - 1
                    frac = (pos - cum[edge]) / np.maximum(length[edge], 1e-6)
                    pa = h[edge]
                    pb = h[(edge + 1) % len(h)]
                    boundary = uv[pa] * (1 - frac[:, None]) + uv[pb] * frac[:, None]
                    contour = mask_contours[k]
                    nearest = cKDTree(contour).query(boundary)[1]
                    reverse = cKDTree(boundary).query(contour)[1]
                    outline[k] = (
                        torch.tensor(pa, device=device),
                        torch.tensor(pb, device=device),
                        to(frac)[:, None],
                        to(contour[nearest]),
                        torch.tensor(reverse, device=device),
                        to(contour),
                    )
        loss = (
            hand_delta.square().mean() / 0.18**2 * 0.8
            + hand_root.square().mean() / 0.12**2
            + hand_t.square().mean() / 0.018**2
            + hand_scale.square().mean() / 0.12**2 * 0.3
        )
        loss = (
            loss
            + obj_delta[:, :3].square().mean() / 0.05**2
            + obj_delta[:, 3:5].square().mean() / 0.006**2
            + obj_delta[:, 5].square().mean() / 0.025**2
        )
        image_loss = 0
        contact_loss = 0
        intrusion = 0
        for si, side in enumerate(["left", "right"]):
            vw, jw, vc, jc = hands[side]
            d = data[side]
            pix = torch_project(vc[:, sample_indices], intr)
            jpix = torch_project(jc, intr)
            image_loss = (
                image_loss
                + ((pix - d["base_pixel"][:, sample_indices]) / 9.0).square().mean()
                + ((jpix - d["base_joint_pixel"]) / 6.0).square().mean()
            )
            qc = (vw - t[:, None]) @ R
            q, n = targets[side]
            signed = ((qc - q) * n).sum(-1)
            intrusion = (
                intrusion
                + (torch.relu(-signed[:, sample_indices] - 0.002) / 0.004)
                .square()
                .mean()
            )
            for k in range(B):
                for finger in range(5):
                    if contact_gate[k, si, finger] == 0:
                        continue
                    ix = torch.tensor(d["patch"][k, finger], device=device)
                    delta = qc[k, ix] - (q[k, ix] + 0.0008 * n[k, ix])
                    dist = delta.square().sum(-1)
                    contact_loss = contact_loss + weights[k, si, finger] * torch.topk(
                        dist, 3, largest=False
                    ).values.mean() / 0.003**2 / (B * 2)
        boundary_loss = 0
        for k in range(B):
            vc = (ov @ R[k].T + t[k] - cam_t[k]) @ cam_R[k]
            uv = torch_project(vc, intr[k])
            ia, ib, fraction, targ, rev, contour = outline[k]
            boundary = uv[ia] * (1 - fraction) + uv[ib] * fraction
            boundary_loss += (
                ((boundary - targ) / 5.0).square().mean()
                + ((boundary[rev] - contour) / 5.0).square().mean()
            ) / B
        temporal = 0
        for p, scale in [
            (hand_delta, 0.05),
            (hand_root, 0.035),
            (hand_t, 0.006),
            (hand_scale, 0.025),
            (obj_delta, 0.008),
        ]:
            temporal += ((p[1:] - p[:-1]) / scale).square().mean() * 0.06
            if B > 2:
                temporal += (
                    (p[2:] - 2 * p[1:-1] + p[:-2]) / scale
                ).square().mean() * 0.06
        loss = (
            loss
            + image_loss * 3
            + contact_loss * 1.7
            + intrusion * 0.35
            + boundary_loss * 3
            + temporal
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 20.0)
        opt.step()
        with torch.no_grad():
            hand_delta.clamp_(-0.35, 0.35)
            hand_root.clamp_(-0.15, 0.15)
            hand_t.clamp_(-0.025, 0.025)
            hand_scale.clamp_(np.log(0.85), np.log(1.18))
            obj_delta[:, :3].clamp_(-0.08, 0.08)
            obj_delta[:, 3:5].clamp_(-0.015, 0.015)
            obj_delta[:, 5].clamp_(-0.035, 0.035)
        if it % 30 == 0 or it == args.iterations - 1:
            row = {
                "iteration": it,
                "total": float(loss.detach()),
                "image": float(image_loss.detach()),
                "contact": float(contact_loss.detach()),
                "intrusion": float(intrusion.detach()),
                "boundary": float(boundary_loss.detach()),
            }
            trace.append(row)
            print(row, flush=True)
    # Parameter corrections, not vertex morphing: replay source MANO at each frame.
    interp = {}
    for name, p in zip(
        [
            "hand_articulation",
            "hand_root",
            "hand_translation_camera",
            "hand_log_scale",
            "object_delta",
        ],
        params,
    ):
        a = p.detach().cpu().numpy()
        flat = a.reshape(B, -1)
        full = np.stack(
            [np.interp(ids, key, flat[:, j]) for j in range(flat.shape[1])], 1
        ).reshape(N, *a.shape[1:])
        interp[name] = full
    state = {
        "frame_idx": ids,
        "T_world_camera": T,
        "intrinsics_960": K,
        "object_vertices_canonical": V,
        "object_faces": F,
        "keyframe_idx": key,
        **interp,
    }
    from scipy.spatial.transform import Rotation

    RR = Rotation.from_rotvec(interp["object_delta"][:, :3]).as_matrix() @ R0
    tt = t0 + np.einsum("bij,bj->bi", T[:, :3, :3], interp["object_delta"][:, 3:])
    state["object_rotation"] = RR
    state["object_translation_world"] = tt
    measurements = []
    for si, side in enumerate(["left", "right"]):
        with torch.no_grad():
            o = models[side](
                global_orient=to(z[side + "_root_orient_axis_angle"]),
                hand_pose=to(
                    z[side + "_hand_pose_axis_angle"]
                    + interp["hand_articulation"][:, si]
                ),
                betas=to(z[side + "_betas"]),
            )
            vw = (o.vertices + to(z[side + "_trans_world_m"])[:, None]).cpu().numpy()
            jw = (
                (
                    torch.cat([o.joints[:, :16], o.vertices[:, TIP_VERTICES]], dim=1)[
                        :, JOINT_ORDER
                    ]
                    + to(z[side + "_trans_world_m"])[:, None]
                )
                .cpu()
                .numpy()
            )
        dR = Rotation.from_rotvec(interp["hand_root"][:, si]).as_matrix()
        pivot = jw[:, :1]
        vw = (vw - pivot) @ dR.transpose(0, 2, 1) + pivot
        jw = (jw - pivot) @ dR.transpose(0, 2, 1) + pivot
        vc = (vw - T[:, :3, 3][:, None]) @ T[:, :3, :3]
        jc = (jw - T[:, :3, 3][:, None]) @ T[:, :3, :3]
        sc = np.exp(interp["hand_log_scale"][:, si])[:, None]
        vc = vc * sc + interp["hand_translation_camera"][:, si, None]
        jc = jc * sc + interp["hand_translation_camera"][:, si, None]
        vw = vc @ T[:, :3, :3].transpose(0, 2, 1) + T[:, :3, 3][:, None]
        jw = jc @ T[:, :3, :3].transpose(0, 2, 1) + T[:, :3, 3][:, None]
        state[side + "_vertices_world"] = vw.astype("float32")
        state[side + "_joints_world"] = jw.astype("float32")
        state[side + "_faces"] = z[side + "_faces"]
        state[side + "_base_vertices_world"] = z[side + "_vertices_world_m"]
        state[side + "_base_joints_world"] = z[side + "_joints_world_m"]
        for i in ids:
            qc = (vw[i] - tt[i]) @ RR[i]
            q, n = closest(scene, qc)
            bc = (z[side + "_vertices_world_m"][i] - t0[i]) @ R0[i]
            qb, nb = closest(scene, bc)
            tips = []
            for tip in TIP_VERTICES:
                patch = cKDTree(z[side + "_vertices_world_m"][i]).query(
                    z[side + "_vertices_world_m"][i, tip], k=12
                )[1]
                tips.append(
                    {
                        "tip_vertex": int(tip),
                        "base_gap_mm": float(
                            np.sort(np.linalg.norm(bc[patch] - qb[patch], axis=1))[
                                :3
                            ].mean()
                            * 1000
                        ),
                        "fitted_gap_mm": float(
                            np.sort(np.linalg.norm(qc[patch] - q[patch], axis=1))[
                                :3
                            ].mean()
                            * 1000
                        ),
                    }
                )
            before = project(camera(z[side + "_joints_world_m"][i], T[i]), K[i])
            after = project(jc[i], K[i])
            measurements.append(
                {
                    "frame": int(i),
                    "side": side,
                    "pad_gaps": tips,
                    "joint_projection_delta_median_px": float(
                        np.median(np.linalg.norm(after - before, axis=1))
                    ),
                    "normal_intrusion_fraction_gt3mm": float(
                        np.mean(np.sum((qc - q) * n, axis=1) < -0.003)
                    ),
                }
            )
    state["base_object_rotation"] = R0
    state["base_object_translation_world"] = t0
    np.savez_compressed(out / "demo_state.npz", **state)
    report = {
        "schema": "v20_visual_contact_demo_fit_v1",
        "status": "demo_fitted_pending_visual_review",
        "demo_only": True,
        "diagnostic_only": True,
        "annotation_ready": False,
        "gt_consumed": False,
        "generated_mesh_role": "demo_visual_contact_reference_not_physical_truth",
        "inputs": {
            k: str(getattr(args, k).resolve())
            for k in [
                "annotations",
                "pose_report",
                "mano_bridge",
                "hawor_params",
                "mano_models",
            ]
        },
        "input_hashes": {
            **{
                k: sha(getattr(args, k))
                for k in ["annotations", "pose_report", "mano_bridge", "hawor_params"]
            },
            **{
                "mano_" + side: sha(args.mano_models / f"MANO_{side.upper()}.pkl")
                for side in ["left", "right"]
            },
        },
        "mesh": str(meshpath),
        "mesh_sha256": sha(meshpath),
        "device": str(device),
        "frame_count": N,
        "zero_state_max_vertex_error_m": maxreplay,
        "trace": trace,
        "measurements": measurements,
        "elapsed_s": time.monotonic() - start,
        "parameters": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "claim": "image-fitted demo contact hypothesis; no measured collision/contact/SDF claim",
    }
    (out / "fit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
