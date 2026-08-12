#!/usr/bin/env python3
"""Finalize P11-P15 experiment evidence after the full P15 renders complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

SCHEMA = "v19_experimental_sam3d_p11_p15_final_evidence_v1"
BRANCH_ORDER = [
    "sam3d_owned_dual_mesh",
    "sam3d_owned_legacy_cut",
    "trellis_frozen_legacy_cut",
]
VIDEO_KEYS = ("overlay_video", "world_video", "side_world_video", "side_by_side_video")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def prepare_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float | int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,pix_fmt",
        "-show_entries", "format=duration,size", "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {path}")
    stream = streams[0]
    fmt = payload.get("format") or {}
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "frame_count": frames,
        "duration_s": float(fmt.get("duration", 0.0)),
        "pix_fmt": stream.get("pix_fmt"),
    }
    if frames != 150:
        raise RuntimeError(f"video frame count is {frames}, expected 150: {path}")
    if abs(report["duration_s"] - 5.0) > 0.08:
        raise RuntimeError(f"video duration is {report['duration_s']}, expected about 5.0s: {path}")
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"unexpected video codec/pixel format: {path} -> {stream}")
    return report


def decode_video(path: Path) -> dict[str, Any]:
    command = [
        "ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or completed.stderr.strip():
        raise RuntimeError(
            f"video decode failed or emitted errors: {path}\nreturn={completed.returncode}\n{completed.stderr[-4000:]}"
        )
    return {"command": command, "exit_code": completed.returncode, "stderr_empty": True}


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    geometry = trimesh.load(path, force="mesh", process=False)
    if not isinstance(geometry, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh: {path}")
    return np.asarray(geometry.vertices, dtype=np.float64), np.asarray(geometry.faces, dtype=np.int64)


def selected_faces(faces: np.ndarray, selector: dict[str, Any]) -> np.ndarray:
    mode = str(selector.get("mode", "all"))
    if mode == "all":
        return faces
    if mode == "contiguous_range":
        return faces[int(selector["start"]):int(selector["stop"])]
    raise RuntimeError(f"unsupported face selector: {selector}")


def triangle_soup_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    triangles = vertices[faces]
    canonical_triangles: list[np.ndarray] = []
    for triangle in triangles:
        order = np.lexsort((triangle[:, 2], triangle[:, 1], triangle[:, 0]))
        canonical_triangles.append(triangle[order].reshape(-1))
    rows = np.asarray(canonical_triangles, dtype="<f8")
    order = np.lexsort(tuple(rows[:, index] for index in reversed(range(rows.shape[1]))))
    return hashlib.sha256(np.ascontiguousarray(rows[order]).tobytes()).hexdigest()


def verify_observed_layer_identity(adapter: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for branch in adapter.get("branches", []):
        observed = [
            row for row in branch.get("render_layers", [])
            if row.get("role") == "observed_metric_surface_overlay"
        ]
        if len(observed) != 1:
            raise RuntimeError(f"branch {branch.get('branch_id')} lacks one observed layer")
        layer = observed[0]
        path = require_file(Path(str(layer["mesh"])), "observed render layer")
        vertices, faces = load_mesh(path)
        selected = selected_faces(faces, layer.get("face_selection") or {"mode": "all"})
        rows.append(
            {
                "branch_id": branch["branch_id"],
                "mesh": str(path),
                "selected_faces": int(len(selected)),
                "triangle_soup_sha256_f64": triangle_soup_digest(vertices, selected),
            }
        )
    hashes = {row["triangle_soup_sha256_f64"] for row in rows}
    if len(hashes) != 1 or any(row["selected_faces"] != 4598 for row in rows):
        raise RuntimeError(f"observed render layers differ: {rows}")
    return {"identical": True, "shared_triangle_soup_sha256_f64": next(iter(hashes)), "branches": rows}


def git_state(path: Path) -> dict[str, Any]:
    path = require_dir(path, "git repository")
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"path": str(path), "commit": commit, "porcelain": status, "clean": not status}


def create_contact_sheet(matrix_dir: Path, source_ids: list[int], output: Path) -> dict[str, Any]:
    requested = [0, 25, 50, 75, 109, 125, 149]
    tiles: list[np.ndarray] = []
    mapping: list[dict[str, int]] = []
    for source_idx in requested:
        if source_idx not in source_ids:
            continue
        output_idx = source_ids.index(source_idx)
        path = require_file(matrix_dir / f"{output_idx:06d}.jpg", "comparison matrix frame")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        tile = cv2.resize(image, (960, 680), interpolation=cv2.INTER_AREA)
        layer = tile.copy()
        cv2.rectangle(layer, (0, 0), (300, 42), (0, 0, 0), -1)
        tile = cv2.addWeighted(layer, 0.68, tile, 0.32, 0.0)
        cv2.putText(
            tile, f"source frame {source_idx:03d}", (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(tile)
        mapping.append({"source_frame_idx": source_idx, "output_frame_index": output_idx})
    if not tiles:
        raise RuntimeError("no contact-sheet frames selected")
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 2:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[index:index + 2]) for index in range(0, len(tiles), 2)])
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write contact sheet: {output}")
    return {"path": str(output), "sha256": sha256_file(output), "frames": mapping, "size_wh": [sheet.shape[1], sheet.shape[0]]}


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def mm(value: float) -> str:
    return f"{1000.0 * float(value):.2f}"


def p13_table(controlled: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in controlled["candidates"]:
        alignment = candidate["metric_alignment"]
        faces = candidate["face_semantics"]
        total = int(faces["total_raw_candidate_faces"])
        rows.append(
            {
                "name": candidate["name"],
                "source_model": candidate["source_model"],
                "observed_to_generated_median_m": alignment["observed_to_generated_final"]["median_m"],
                "observed_to_generated_p95_m": alignment["observed_to_generated_final"]["p95_m"],
                "generated_to_observed_p95_m": alignment["generated_to_observed_final"]["p95_m"],
                "free_space_rejected_fraction": int(faces["free_space_rejected_faces"]) / total,
                "retained_hidden_fraction": int(faces["generated_hidden_faces_in_pose_hypothesis"]) / total,
                "retained_hidden_faces": int(faces["generated_hidden_faces_in_pose_hypothesis"]),
            }
        )
    return rows


def seam_table(seams: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "symmetric_median_m": row["symmetric_boundary_to_counterpart_surface"]["median_m"],
            "symmetric_p95_m": row["symmetric_boundary_to_counterpart_surface"]["p95_m"],
            "observed_to_generated_median_m": row["observed_boundary_to_generated_surface"]["median_m"],
            "observed_to_generated_within_10mm": row["observed_boundary_to_generated_surface"]["fraction_within"]["10mm"],
        }
        for row in seams["candidates"]
    ]


def markdown_report(summary: dict[str, Any]) -> str:
    p11 = summary["p11"]
    p12 = summary["p12"]
    raw = summary["p12_raw_mask_ab"]
    dual = summary["p13_dual_mesh"]
    mano = summary["p15_full_mano_unsigned"]
    videos = summary["p15_video_qc"]
    lines = [
        "# SAM3D Objects 替换 TRELLIS 几何先验：P11–P15 全流程总结",
        "",
        "## 1. 结论摘要",
        "",
        "- 当前证据支持把 **SAM3D Objects 继续作为 P12 完整形状/render prior 的首选实验候选**；尚不支持替换物体时序、MANO、contact、collision 或 signed nonpenetration。",
        "- object-owned mask 的新 SAM3D 结果保持了旧结果的主体形状，同时减少组件并改善 P13 tail/hidden retention；但仅有 keyboard、单 anchor、seed 42。",
        "- P13 legacy observed-band face deletion 是约 11 mm seam 和拓扑破坏的直接来源。Topology-preserving dual mesh 保留原始 275,208 faces、0 boundary edges 和 watertight 状态。",
        "- 150 帧 P15 可视结果把完整 SAM prior、共享 observed surface、完整 778-vertex MANO 和同一 observed-only trajectory/camera 放在同一 artifact 中；TRELLIS 的隐藏形状在世界/侧视图中仍明显偏离 keyboard，SAM 两路更接近完整矩形主体。",
        "- 当前最硬的物理 blocker 已从“没有完整手物视频”转为 **MANO/object/camera/trajectory 的度量冲突**：共享 observed surface 上，仅 "
        + percent(mano["observed_any_vertex_within_5mm_row_fraction"])
        + " 的 frame-side hand rows 有任一完整 MANO vertex 位于 5 mm 内；因此不能把生成网格的近距离当成 contact。",
        "",
        "## 2. 不变量与实验边界",
        "",
        "- 所有工作位于 additive experiment；canonical V19 与冻结 TRELLIS 产物未改。",
        "- SAM3D 原生输入：full RGB + prediction-side binary object-owned mask；`pointmap=None`；无外部 crop/rotation/rectification。",
        "- P14/P15 三路只有 geometry source/integration 不同；annotation/camera、150-frame observed-only pose、MANO payload、temporal MANO payload、projection contract 的 value hash 完全相同。",
        "- 三路共享 2,394 vertices / 4,598 faces 的 observed-only physical surface；generated faces 的 collision/contact eligibility=false，signed geometry=false。",
        "- 这里的 painter 是可视 review，不是 metric z-buffer evaluator；旧 1408→960 K 合同仍需在 official HOT3D pinhole/P03c 上重做。",
        "",
        "## 3. P11：公平输入合同与 mask provenance",
        "",
        f"- anchor frame：{p11['frame_idx']}；object-owned mask：{p11['owned_pixels']:,} px。",
        f"- 旧 raw SAM2 mask：{p11['raw_pixels']:,} px；XOR {p11['xor_pixels']:,} px；IoU {p11['mask_iou']:.6f}。",
        "- ownership mask 与 visible-geometry candidate byte-identical；来源是 prediction-side OWLv2→SAM2→hand ownership subtraction，无 HOT3D GT mask/CAD/pose leakage。",
        "- TRELLIS 保留 object-isolated RGBA crop；SAM3D 保留 full-scene native contract。公平性来自同 anchor/owned mask，而不是强迫相同 crop。",
        "",
        "## 4. P12：SAM3D 原生生成与 mask A/B",
        "",
        f"- 新 raw mesh：{p12['vertices']:,} vertices / {p12['faces']:,} faces；extents={p12['extents']}；watertight={p12['watertight']}。",
        f"- raw mesh SHA256：`{p12['mesh_sha256']}`；seed={p12['seed']}；native uniform scale={p12['native_scale']:.6f}。",
        f"- old/new normalized symmetric surface mean distance：{raw['symmetric_chamfer']:.6f} longest-axis units。",
        f"- silhouette IoU：top={raw['view_iou'][0]:.4f}，long-side={raw['view_iou'][1]:.4f}，end={raw['view_iou'][2]:.4f}。",
        f"- components：{raw['old_components']}→{raw['new_components']}；native rotation delta={raw['rotation_delta_deg']:.2f}°；scale ratio={raw['scale_ratio']:.4f}。",
        "",
        "## 5. P13：共同下游 controlled A/B",
        "",
        "| Candidate | obs→gen median (mm) | obs→gen p95 (mm) | gen→obs p95 (mm) | free-space reject | retained hidden | hidden faces |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["p13_controlled"]:
        lines.append(
            f"| {row['name']} | {mm(row['observed_to_generated_median_m'])} | "
            f"{mm(row['observed_to_generated_p95_m'])} | {mm(row['generated_to_observed_p95_m'])} | "
            f"{percent(row['free_space_rejected_fraction'])} | {percent(row['retained_hidden_fraction'])} | "
            f"{row['retained_hidden_faces']:,} |"
        )
    lines.extend([
        "",
        "单一 obs→gen median 会偏向 TRELLIS，但它同时有高 free-space reject、极低 hidden retention 和错误的完整主轴形状；因此不能用单一 nearest residual 选 backend。",
        "",
        "### Legacy seam",
        "",
        "| Candidate | symmetric median (mm) | symmetric p95 (mm) | observed→generated median (mm) | observed boundary within 10 mm |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["p13_seams"]:
        lines.append(
            f"| {row['name']} | {mm(row['symmetric_median_m'])} | {mm(row['symmetric_p95_m'])} | "
            f"{mm(row['observed_to_generated_median_m'])} | {percent(row['observed_to_generated_within_10mm'])} |"
        )
    lines.extend([
        "",
        f"Legacy observed band={dual['observed_band_mm']:.3f} mm；new SAM legacy seam median≈11.31 mm，与该删除带一致。",
        "",
        "### Topology-preserving dual mesh",
        "",
        f"- intact aligned SAM：{dual['raw_vertices']:,} vertices / {dual['raw_faces']:,} faces，boundary edges={dual['raw_boundary_edges']}，components={dual['raw_components']}，watertight={dual['raw_watertight']}。",
        f"- legacy cut：{dual['legacy_vertices']:,} vertices / {dual['legacy_faces']:,} faces，boundary edges={dual['legacy_boundary_edges']:,}，components={dual['legacy_components']}，watertight={dual['legacy_watertight']}。",
        "- dual mesh 不填洞、不 stitch、不形变、不删 generated faces；observed surface 独立覆盖，collision 保持 observed-only。",
        "",
        "## 6. P14/P15：150-frame object + full MANO + camera",
        "",
        f"- 三路均完成 {videos['frame_count']} frames / {videos['duration_s']:.3f} s；所有输出通过 ffprobe frame/duration/codec 检查及完整 decode。",
        f"- 每个 frame/side 使用完整 778 MANO vertices + 1,538 faces；共 {mano['query_vertices']:,} 个 vertex-surface queries；64-vertex inline sample reproduction max error={mano['sample_error_m']:.1e} m。",
        "- camera overlay 中共享 observed green layer一致；geometry 差异主要出现在 observed silhouette 外缘、隐藏底面与世界/侧视图。",
        "- frame 50/109 等代表帧中：SAM dual 与 SAM legacy 均维持 keyboard-like 完整长方体；dual 保留更多连续 purple underlay；TRELLIS 在 X-Z/Y-Z 中表现为明显 wedge/偏轴完整先验。",
        "- 完整 MANO 在 2D overlay 大体贴合可见手，但 world/side 中与 object 的大距离暴露了现有 hand/object/camera/trajectory 度量冲突。",
        "",
        "### 完整 MANO unsigned proximity（仅 observed surface 可作物理 proximity）",
        "",
        f"- observed surface：all-vertex median={mm(mano['observed_all_vertex_median_m'])} mm；per-row minimum median={mm(mano['observed_row_min_median_m'])} mm；任一 vertex within 5/10 mm 的 row fraction={percent(mano['observed_any_vertex_within_5mm_row_fraction'])}/{percent(mano['observed_any_vertex_within_10mm_row_fraction'])}。",
        f"- intact SAM 对同一 MANO 比 legacy-cut 更近的 vertex fraction={percent(mano['dual_closer_fraction'])}；median delta={mm(mano['dual_minus_cut_median_m'])} mm。该结果只说明 face deletion 删除了邻近 render surface，**不是 contact 改善**。",
        "- observed surface 非 watertight，不能给 sign；generated mesh 即使 watertight 也没有 metric/collision validation。",
        "",
        "## 7. 当前判断",
        "",
        "1. **继续保留 SAM3D object-owned-mask + topology-preserving dual mesh 作为实验 render prior 主候选。**",
        "2. **TRELLIS 暂时保留 frozen fallback，不改默认 canonical backend。** 当前单样本视觉/hidden-tail 证据偏向 SAM3D，但不足以正式推广。",
        "3. **不复用 SAM3D native pose 代替 P15 trajectory。** 当前视频使用 observed-only trajectory 是正确隔离；native quaternion/scale 仍需独立 reprojection/GT convention 验证。",
        "4. **contact/collision/signed state 继续禁用。** 当前完整 MANO proximity 明确显示度量冲突，生成表面的 near-zero distance不能升级为物理事实。",
        "",
        "## 8. 还需要做的工作（优先级）",
        "",
        "### P0：正式替换前必做",
        "- 在 official HOT3D pinhole K/P03c camera contract 上重复 P11–P15，消除旧 1408→960 K 合同的外推风险。",
        "- 解决/重新估计 MANO–object–camera–trajectory 的共同 metric state；用完整 MANO、可见深度和 mask reprojection 同时验证，而不是让 generated mesh 吸收冲突。",
        "- 对三路固定预测做真正 perspective-correct metric z-buffer evaluator：held-out silhouette、first-hit depth median/p95、free-space contradiction、hand/object depth order。",
        "- 扩展到多 object、多 anchor、多 seed；至少报告均值、方差和失败类型，不能以 keyboard/seed42 决定默认 backend。",
        "",
        "### P1：模型与 pose 归因",
        "- 独立验证 SAM3D native quaternion/axis/translation/scale convention；HOT3D CAD/pose 如可用，只能在预测冻结后作为 evaluator。",
        "- 做 generator-only 共同下游 A/B 与 SAM3D-native capability 分离报告；legacy `trellis_*` 兼容字段不得进入 source attribution。",
        "- 评估多 anchor 或多视角支持是否能稳定完整 prior，而不是 seed/anchor 挑优。",
        "",
        "### P2：物理状态与生产接入",
        "- 独立构建 conservative collision proxy，并验证 watertight、metric front surface、free-space 与多视角一致性；通过前 generated mesh 永远 render-only。",
        "- 在可信 proxy 后，才重做完整 778 MANO contact/nonpenetration；继续保留 unsigned/signed 语义隔离。",
        "- 只有多样本视频与 evaluator 均改善后，才把 source-neutral geometry contract 接到 runtime spec 并考虑修改默认 backend。",
        "",
        "## 9. 关键产物",
        "",
        f"- 三路 camera overlay A/B：`{videos['comparison_overlay_video']}`",
        f"- 三路 overlay/world/side A/B：`{videos['comparison_matrix_video']}`",
        f"- 多帧 QC sheet：`{videos['contact_sheet']}`",
        f"- 完整 MANO unsigned report：`{mano['report']}`",
        f"- 最终 machine-readable report：`{summary['output_report']}`",
        "",
        "本报告中的视觉判断必须结合上述视频/QC sheet；JSON/schema 仅是 backing evidence。",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = require_dir(args.experiment_root, "experiment root")
    ab_root = require_dir(args.ab_root, "P14/P15 A/B root")
    output_dir = prepare_output(args.output_dir)

    paths = {
        "p11": require_file(experiment_root / "p11_dual_inputs/p11_dual_geometry_inputs_report.json", "P11 report"),
        "p12": require_file(experiment_root / "p12_parallel_priors/p12_parallel_geometry_priors_report.json", "P12 report"),
        "raw": require_file(experiment_root / "p12_raw_mask_ab/p12_raw_sam_mask_ab_report.json", "P12 mask A/B report"),
        "controlled": require_file(experiment_root / "p13_controlled_geometry_prior_ab_retry1/p13_controlled_geometry_prior_ab_report.json", "controlled P13 report"),
        "seams": require_file(experiment_root / "p13_controlled_seams/p13_controlled_seam_report.json", "P13 seam report"),
        "dual": require_file(experiment_root / "p13_dual_mesh_new_owned/p13_dual_mesh_geometry_prior_report.json", "dual-mesh report"),
        "adapter": require_file(ab_root / "p14_p15_layered_render_state_adapter_report.json", "P14/P15 adapter report"),
        "mano": require_file(ab_root / "full_mano_unsigned_distance/p15_full_mano_unsigned_surface_distance_report.json", "full MANO unsigned report"),
        "comparison": require_file(args.comparison_report, "P15 comparison report"),
    }
    reports = {key: load_json(path) for key, path in paths.items()}
    for key, report in reports.items():
        if report.get("status") != "ok":
            raise RuntimeError(f"input report {key} is not ok: {paths[key]}")

    branch_manifests: dict[str, dict[str, Any]] = {}
    video_reports: list[dict[str, Any]] = []
    for branch in BRANCH_ORDER:
        manifest_path = require_file(
            ab_root / branch / "full_video/p14_p15_layered_full_mano_render_manifest.json",
            f"{branch} full-video manifest",
        )
        manifest = load_json(manifest_path)
        if manifest.get("status") != "ok" or int(manifest.get("frame_count", 0)) != 150:
            raise RuntimeError(f"incomplete branch manifest: {manifest_path}")
        branch_manifests[branch] = manifest
        for key in VIDEO_KEYS:
            path = require_file(Path(str(manifest["outputs"].get(key, ""))), f"{branch} {key}")
            probe = ffprobe_video(path)
            probe.update({"branch": branch, "output_key": key, "decode": decode_video(path)})
            video_reports.append(probe)

    comparison = reports["comparison"]
    comparison_video_specs = [
        ("camera_overlay_ab_video", (1920, 640)),
        ("overlay_world_side_ab_video", (1920, 1360)),
    ]
    for key, expected_size in comparison_video_specs:
        path = require_file(Path(str(comparison["outputs"].get(key, ""))), f"comparison {key}")
        probe = ffprobe_video(path)
        if (probe["width"], probe["height"]) != expected_size:
            raise RuntimeError(f"comparison size mismatch for {key}: {probe}")
        probe.update({"branch": "three_branch_comparison", "output_key": key, "decode": decode_video(path)})
        video_reports.append(probe)

    source_ids = [int(value) for value in comparison["validation"]["source_frame_ids"]]
    matrix_dir = require_dir(Path(str(comparison["outputs"]["matrix_frames"])), "comparison matrix frames")
    contact_sheet = create_contact_sheet(matrix_dir, source_ids, output_dir / "p15_three_branch_multiframe_qc.jpg")
    observed_identity = verify_observed_layer_identity(reports["adapter"])
    sam_git = git_state(args.sam3d_repo)
    if not sam_git["clean"]:
        raise RuntimeError(f"SAM3D upstream checkout is dirty: {sam_git}")

    p11 = reports["p11"]
    p12 = reports["p12"]
    raw = reports["raw"]
    dual = reports["dual"]
    mano = reports["mano"]
    sam_native = p12["candidates"]["sam3d_objects"]["native_outputs"]
    view_iou = [row["normalized_silhouette_iou"] for row in raw["shape_comparison"]["per_view_normalized_silhouette"]]
    surfaces = {row["surface_id"]: row for row in mano["surfaces"]}
    observed_surface = surfaces["shared_observed_metric_surface"]
    observed_proximity = observed_surface["per_frame_side_hand_proximity"]
    proximity_fraction = observed_proximity["frame_side_row_fraction_with_any_vertex_within_threshold"]
    comparison_delta = mano["comparisons"]["sam3d_dual_minus_same_prior_legacy_cut_unsigned_distance_m"]
    branch_elapsed = {
        branch: float(manifest["total_elapsed_s"])
        for branch, manifest in branch_manifests.items()
    }

    compact_summary = {
        "p11": {
            "frame_idx": int(p11["selected_frame_idx"]),
            "owned_pixels": int(p11["selected_input_summary"]["mask_area_px"]),
            "raw_pixels": int(p11["comparison_mask_diagnostic"]["comparison_pixels"]),
            "xor_pixels": int(p11["comparison_mask_diagnostic"]["xor_pixels"]),
            "mask_iou": float(p11["comparison_mask_diagnostic"]["iou"]),
        },
        "p12": {
            "seed": int(p12["seed"]),
            "vertices": int(sam_native["mesh_stats"]["vertices"]),
            "faces": int(sam_native["mesh_stats"]["faces"]),
            "extents": sam_native["mesh_stats"]["extent_model_units"],
            "watertight": bool(sam_native["mesh_stats"]["watertight"]),
            "mesh_sha256": sam_native["raw_mesh"]["sha256"],
            "native_scale": float(sam_native["native_pose"]["scale"][0]),
        },
        "p12_raw_mask_ab": {
            "symmetric_chamfer": float(raw["shape_comparison"]["symmetric_chamfer_mean_normalized"]),
            "view_iou": view_iou,
            "old_components": int(raw["old_raw_sam2_mask_mesh"]["summary"]["connected_components"]),
            "new_components": int(raw["new_object_owned_mask_mesh"]["summary"]["connected_components"]),
            "rotation_delta_deg": float(raw["native_layout_delta_not_metric_v19"]["quaternion_angular_delta_degrees"]),
            "scale_ratio": float(raw["native_layout_delta_not_metric_v19"]["uniform_scale_ratio_new_over_old"]),
        },
        "p13_controlled": p13_table(reports["controlled"]),
        "p13_seams": seam_table(reports["seams"]),
        "p13_dual_mesh": {
            "observed_band_mm": 12.119223035953178,
            "raw_vertices": int(dual["topology"]["aligned_raw_render_prior"]["vertices"]),
            "raw_faces": int(dual["topology"]["aligned_raw_render_prior"]["faces"]),
            "raw_boundary_edges": int(dual["topology"]["aligned_raw_render_prior"]["boundary_edges"]),
            "raw_components": int(dual["topology"]["aligned_raw_render_prior"]["connected_components"]),
            "raw_watertight": bool(dual["topology"]["aligned_raw_render_prior"]["watertight"]),
            "legacy_vertices": int(dual["topology"]["legacy_cut_pose_hypothesis"]["vertices"]),
            "legacy_faces": int(dual["topology"]["legacy_cut_pose_hypothesis"]["faces"]),
            "legacy_boundary_edges": int(dual["topology"]["legacy_cut_pose_hypothesis"]["boundary_edges"]),
            "legacy_components": int(dual["topology"]["legacy_cut_pose_hypothesis"]["connected_components"]),
            "legacy_watertight": bool(dual["topology"]["legacy_cut_pose_hypothesis"]["watertight"]),
        },
        "p14_p15_shared_state": {
            "pose_frame_count": int(reports["adapter"]["source_validation"]["pose_frame_count"]),
            "pose_nonpenetration_target_frame_count": int(reports["adapter"]["source_validation"]["pose_nonpenetration_target_frame_count"]),
            "shared_state_hashes": reports["adapter"]["shared_state_value_sha256"],
            "shared_observed_layer_identity": observed_identity,
        },
        "p15_full_mano_unsigned": {
            "report": str(paths["mano"]),
            "query_vertices": int(mano["distance_contract"]["query_vertex_count"]),
            "sample_error_m": float(mano["mano_full_surface"]["sample_reproduction_max_error_m"]),
            "observed_all_vertex_median_m": float(observed_surface["all_full_mano_vertices_unsigned_distance_m"]["median"]),
            "observed_row_min_median_m": float(observed_proximity["minimum_vertex_distance_m_across_frame_side_rows"]["median"]),
            "observed_any_vertex_within_5mm_row_fraction": float(proximity_fraction["5mm"]),
            "observed_any_vertex_within_10mm_row_fraction": float(proximity_fraction["10mm"]),
            "dual_closer_fraction": float(comparison_delta["dual_is_closer_fraction"]),
            "dual_minus_cut_median_m": float(comparison_delta["summary"]["median"]),
        },
        "p15_video_qc": {
            "frame_count": 150,
            "duration_s": 5.0,
            "branch_elapsed_s": branch_elapsed,
            "videos": video_reports,
            "comparison_overlay_video": comparison["outputs"]["camera_overlay_ab_video"],
            "comparison_matrix_video": comparison["outputs"]["overlay_world_side_ab_video"],
            "contact_sheet": contact_sheet["path"],
        },
        "sam3d_upstream": sam_git,
        "input_reports": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()
        },
        "contact_sheet": contact_sheet,
        "output_report": str(output_dir / "sam3d_p11_p15_final_evidence_report.json"),
    }
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "method": "finalize_experimental_sam3d_p11_p15_evidence",
        "claim_scope": (
            "Consolidated additive experiment evidence and video integrity checks. Visual conclusions remain review judgments; "
            "generated geometry stays render-only and no signed contact/collision promotion is made."
        ),
        **compact_summary,
    }
    report_path = output_dir / "sam3d_p11_p15_final_evidence_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact_summary["output_report"] = str(report_path)
    markdown = markdown_report(compact_summary)
    markdown_path = output_dir / "SAM3D_P11_P15_FULL_REPORT_ZH.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "markdown": str(markdown_path),
        "contact_sheet": contact_sheet,
        "video_count": len(video_reports),
        "sam3d_upstream": sam_git,
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--ab-root", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3d-repo", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
