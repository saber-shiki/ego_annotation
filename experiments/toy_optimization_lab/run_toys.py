#!/usr/bin/env python3
"""Visual toy problems for the optimization stages used by the V19 pipeline.

The toys intentionally use 2-D/low-dimensional state so every variable, residual,
and failure mode can be plotted. They are not replacements for the production
solvers and do not consume prediction artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import LinearConstraint, least_squares, minimize
from scipy.spatial import cKDTree


PROBLEMS = ("sim2", "pose_graph", "halfspace", "mano", "joint_gauge")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def rotation_2d(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def apply_sim2(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return float(scale) * (np.asarray(points, dtype=float) @ np.asarray(rotation, dtype=float).T) + np.asarray(translation, dtype=float)[None, :]


def nearest_stats(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    distances, _ = cKDTree(np.asarray(target, dtype=float)).query(np.asarray(query, dtype=float), k=1)
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90.0)),
        "p95": float(np.percentile(distances, 95.0)),
        "max": float(np.max(distances)),
    }


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if np.linalg.det(vectors) < 0.0:
        vectors[:, -1] *= -1.0
    return center, vectors, values


def sim2_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 2:
        raise ValueError("sim2_umeyama requires matched Nx2 arrays")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(2, dtype=float)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.sum(src_centered * src_centered) / len(src))
    scale = float(np.sum(singular_values * sign) / max(variance, 1.0e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def keyboard_points_2d() -> np.ndarray:
    """Asymmetric keyboard-like point set, including a small right keypad."""
    width, height = 2.0, 0.72
    x_edge = np.linspace(-width / 2.0, width / 2.0, 70)
    y_edge = np.linspace(-height / 2.0, height / 2.0, 28)
    parts = [
        np.column_stack([x_edge, np.full_like(x_edge, -height / 2.0)]),
        np.column_stack([x_edge, np.full_like(x_edge, height / 2.0)]),
        np.column_stack([np.full_like(y_edge, -width / 2.0), y_edge]),
        np.column_stack([np.full_like(y_edge, width / 2.0), y_edge]),
    ]
    for y in np.linspace(-0.22, 0.22, 4):
        parts.append(np.column_stack([np.linspace(-0.84, 0.42, 38), np.full(38, y)]))
    for x in np.linspace(0.58, 0.88, 4):
        parts.append(np.column_stack([np.full(16, x), np.linspace(-0.24, 0.24, 16)]))
    # A small asymmetric protrusion removes the exact 180-degree ambiguity.
    parts.append(np.column_stack([np.linspace(0.78, 1.10, 15), np.full(15, 0.30)]))
    return np.concatenate(parts, axis=0)


def sim2_rotation_candidates(observed_basis: np.ndarray, model_basis: np.ndarray) -> list[np.ndarray]:
    permutations = (np.eye(2), np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    candidates: list[np.ndarray] = []
    for permutation in permutations:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                candidate = observed_basis @ permutation @ np.diag([sx, sy]) @ model_basis.T
                if np.linalg.det(candidate) > 0.0:
                    candidates.append(candidate)
    return candidates


def wrapped_angle_error(rotation: np.ndarray, truth: np.ndarray) -> float:
    relative = rotation @ truth.T
    angle = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
    return abs(math.atan2(math.sin(angle), math.cos(angle)))


def run_sim2(output_dir: Path, seed: int) -> dict[str, Any]:
    """P13/P14 analogue: discrete PCA initialization followed by Sim(2) ICP."""
    rng = np.random.default_rng(seed + 11)
    model = keyboard_points_2d()
    true_scale = 1.28
    true_rotation = rotation_2d(math.radians(31.0))
    true_translation = np.asarray([0.58, -0.22], dtype=float)
    truth_world = apply_sim2(model, true_scale, true_rotation, true_translation)

    keep = rng.random(len(model)) > 0.12
    observed = truth_world[keep] + rng.normal(scale=0.012, size=(int(np.count_nonzero(keep)), 2))
    outliers = rng.uniform(low=[-0.85, -1.0], high=[1.8, 0.75], size=(12, 2))
    observed = np.concatenate([observed, outliers], axis=0)

    observed_center, observed_basis, _ = pca_basis(observed)
    model_center, model_basis, _ = pca_basis(model)
    observed_radius = math.sqrt(float(np.mean(np.sum((observed - observed_center) ** 2, axis=1))))
    model_radius = math.sqrt(float(np.mean(np.sum((model - model_center) ** 2, axis=1))))
    initial_scale = observed_radius / max(model_radius, 1.0e-12)

    candidate_rows: list[dict[str, Any]] = []
    for candidate_rotation in sim2_rotation_candidates(observed_basis, model_basis):
        candidate_translation = observed_center - initial_scale * (candidate_rotation @ model_center)
        transformed = apply_sim2(model, initial_scale, candidate_rotation, candidate_translation)
        stats = nearest_stats(observed, transformed)
        candidate_rows.append(
            {
                "score": stats["median"] + 0.25 * stats["p90"],
                "scale": initial_scale,
                "rotation": candidate_rotation,
                "translation": candidate_translation,
                "stats": stats,
            }
        )
    best = min(candidate_rows, key=lambda row: float(row["score"]))
    scale = float(best["scale"])
    rotation = np.asarray(best["rotation"], dtype=float)
    translation = np.asarray(best["translation"], dtype=float)
    initial_transformed = apply_sim2(model, scale, rotation, translation)

    trace: list[dict[str, float]] = []
    for iteration in range(8):
        transformed = apply_sim2(model, scale, rotation, translation)
        _, nearest_ids = cKDTree(transformed).query(observed, k=1)
        source_matches = model[np.asarray(nearest_ids, dtype=int)]
        scale, rotation, translation = sim2_umeyama(source_matches, observed)
        stats = nearest_stats(observed, apply_sim2(model, scale, rotation, translation))
        trace.append({"iteration": iteration + 1, **stats})
    final_transformed = apply_sim2(model, scale, rotation, translation)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].scatter(model[:, 0], model[:, 1], s=8, alpha=0.65, label="canonical model")
    axes[0].set_title("Canonical toy keyboard")
    axes[0].axis("equal")
    axes[0].legend(loc="best")

    axes[1].scatter(observed[:, 0], observed[:, 1], s=9, alpha=0.55, label="partial/noisy observation")
    axes[1].scatter(initial_transformed[:, 0], initial_transformed[:, 1], s=6, alpha=0.45, label="PCA candidate init")
    axes[1].plot(truth_world[:, 0], truth_world[:, 1], ".", ms=2, alpha=0.25, label="hidden truth")
    axes[1].set_title("Discrete initialization")
    axes[1].axis("equal")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].scatter(observed[:, 0], observed[:, 1], s=9, alpha=0.45, label="observation")
    axes[2].scatter(final_transformed[:, 0], final_transformed[:, 1], s=6, alpha=0.65, label="final Sim(2) ICP")
    axes[2].plot(truth_world[:, 0], truth_world[:, 1], ".", ms=2, alpha=0.20, label="hidden truth")
    axes[2].set_title("Final alignment")
    axes[2].axis("equal")
    axes[2].legend(loc="best", fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    alignment_path = output_dir / "sim2_alignment.png"
    figure.savefig(alignment_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    iterations = [row["iteration"] for row in trace]
    axis.plot(iterations, [row["mean"] for row in trace], "o-", label="mean NN distance")
    axis.plot(iterations, [row["median"] for row in trace], "o-", label="median NN distance")
    axis.plot(iterations, [row["p95"] for row in trace], "o-", label="p95 NN distance")
    axis.set_xlabel("ICP iteration")
    axis.set_ylabel("distance (toy units)")
    axis.set_title("P13-like ICP diagnostic trace")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    trace_path = output_dir / "sim2_trace.png"
    figure.savefig(trace_path, dpi=180)
    plt.close(figure)

    report = {
        "problem": "sim2_partial_point_alignment",
        "analogue": ["P13 PCA/Sim(3) alignment", "P14 nearest-neighbor rigid fit"],
        "truth": {"scale": true_scale, "rotation": true_rotation, "translation": true_translation},
        "initial": {
            "candidate_count": len(candidate_rows),
            "scale": float(best["scale"]),
            "rotation": best["rotation"],
            "translation": best["translation"],
            "stats": best["stats"],
        },
        "final": {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "stats": nearest_stats(observed, final_transformed),
            "scale_abs_error": abs(scale - true_scale),
            "rotation_abs_error_rad": wrapped_angle_error(rotation, true_rotation),
            "translation_error": float(np.linalg.norm(translation - true_translation)),
        },
        "trace": trace,
        "outputs": {"alignment": alignment_path, "trace": trace_path},
        "interpretation": "Nearest-neighbor fit can reduce its one-way objective despite partial observations and outliers; bidirectional/ground-truth errors need not improve identically.",
    }
    write_json(output_dir / "sim2_report.json", report)
    return report


def pose_graph_residual(
    flat: np.ndarray,
    frame_idx: np.ndarray,
    target: np.ndarray,
    target_active: np.ndarray,
    *,
    sigma_translation: float,
    sigma_rotation: float,
    sigma_target: float,
    sigma_step_translation: float,
    sigma_step_rotation: float,
    sigma_accel_translation: float,
    sigma_accel_rotation: float,
) -> np.ndarray:
    n = len(frame_idx)
    state = np.asarray(flat, dtype=float).reshape(n, 3)
    translation_delta = state[:, :2]
    rotation_delta = state[:, 2]
    residuals: list[np.ndarray] = [translation_delta.reshape(-1) / sigma_translation, rotation_delta / sigma_rotation]
    if np.any(target_active):
        residuals.append((translation_delta[target_active] - target[target_active]).reshape(-1) / sigma_target)
    for i in range(1, n):
        gap = max(1, int(frame_idx[i] - frame_idx[i - 1]))
        residuals.append((translation_delta[i] - translation_delta[i - 1]) / (sigma_step_translation * math.sqrt(gap)))
        residuals.append(np.asarray([(rotation_delta[i] - rotation_delta[i - 1]) / (sigma_step_rotation * math.sqrt(gap))]))
    for i in range(1, n - 1):
        gap0 = max(1, int(frame_idx[i] - frame_idx[i - 1]))
        gap1 = max(1, int(frame_idx[i + 1] - frame_idx[i]))
        previous_t = (translation_delta[i] - translation_delta[i - 1]) / gap0
        next_t = (translation_delta[i + 1] - translation_delta[i]) / gap1
        previous_r = (rotation_delta[i] - rotation_delta[i - 1]) / gap0
        next_r = (rotation_delta[i + 1] - rotation_delta[i]) / gap1
        scale = math.sqrt(max(gap0, gap1))
        residuals.append((next_t - previous_t) / (sigma_accel_translation * scale))
        residuals.append(np.asarray([(next_r - previous_r) / (sigma_accel_rotation * scale)]))
    return np.concatenate([np.asarray(row, dtype=float).reshape(-1) for row in residuals])


def run_pose_graph(output_dir: Path, seed: int) -> dict[str, Any]:
    """P15 analogue: an inert zero-correction graph and an externally pressured graph."""
    rng = np.random.default_rng(seed + 23)
    frame_idx = np.arange(60, dtype=int)
    phase = frame_idx / max(1, frame_idx[-1])
    truth_translation = np.column_stack(
        [0.55 * np.sin(1.35 * math.pi * phase), 0.16 * np.cos(2.0 * math.pi * phase) + 0.12 * phase]
    )
    truth_rotation = 0.35 * np.sin(1.8 * math.pi * phase)
    observation_translation = truth_translation + rng.normal(scale=0.018, size=truth_translation.shape)
    observation_rotation = truth_rotation + rng.normal(scale=0.025, size=len(frame_idx))

    active = (frame_idx >= 21) & (frame_idx <= 37)
    pressure_target = np.zeros_like(observation_translation)
    pressure_target[active, 1] = 0.075
    common = {
        "sigma_translation": 0.030,
        "sigma_rotation": 0.090,
        "sigma_target": 0.045,
        "sigma_step_translation": 0.018,
        "sigma_step_rotation": 0.070,
        "sigma_accel_translation": 0.012,
        "sigma_accel_rotation": 0.045,
    }
    x0 = np.zeros((len(frame_idx), 3), dtype=float).reshape(-1)

    inactive_mask = np.zeros_like(active)
    result_inert = least_squares(
        lambda x: pose_graph_residual(x, frame_idx, pressure_target, inactive_mask, **common),
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=100,
    )
    result_active = least_squares(
        lambda x: pose_graph_residual(x, frame_idx, pressure_target, active, **common),
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=150,
    )
    inert_state = result_inert.x.reshape(len(frame_idx), 3)
    active_state = result_active.x.reshape(len(frame_idx), 3)
    corrected_translation = observation_translation + active_state[:, :2]
    corrected_rotation = observation_rotation + active_state[:, 2]

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(frame_idx, truth_translation[:, 1], "k--", label="hidden true y")
    axes[0].plot(frame_idx, observation_translation[:, 1], color="tab:gray", alpha=0.7, label="P14-like observation y")
    axes[0].plot(frame_idx, corrected_translation[:, 1], color="tab:blue", label="active graph corrected y")
    axes[0].fill_between(frame_idx, axes[0].get_ylim()[0], axes[0].get_ylim()[1], where=active, alpha=0.10, color="tab:red", label="soft pressure interval")
    axes[0].set_ylabel("translation y")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    axes[1].plot(frame_idx, inert_state[:, 1], label="no target: correction")
    axes[1].plot(frame_idx, active_state[:, 1], label="with target: correction")
    axes[1].plot(frame_idx, pressure_target[:, 1], "--", label="target correction")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("delta y")
    axes[1].legend(loc="upper right")

    axes[2].plot(frame_idx, truth_rotation, "k--", label="hidden true rotation")
    axes[2].plot(frame_idx, observation_rotation, color="tab:gray", alpha=0.7, label="observation")
    axes[2].plot(frame_idx, corrected_rotation, color="tab:orange", label="corrected")
    axes[2].set_ylabel("rotation (rad)")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("P15-like correction field: zero-target graph is exactly inert")
    figure.tight_layout()
    plot_path = output_dir / "pose_graph.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = {
        "problem": "temporal_pose_correction_graph",
        "analogue": "P15 rigid object pose graph",
        "inert_case": {
            "success": bool(result_inert.success),
            "nfev": int(result_inert.nfev),
            "cost": float(result_inert.cost),
            "max_correction": float(np.max(np.abs(inert_state))),
        },
        "active_case": {
            "success": bool(result_active.success),
            "nfev": int(result_active.nfev),
            "cost": float(result_active.cost),
            "translation_delta_norm": {
                "median": float(np.median(np.linalg.norm(active_state[:, :2], axis=1))),
                "max": float(np.max(np.linalg.norm(active_state[:, :2], axis=1))),
            },
            "target_residual_before": float(np.mean(np.linalg.norm(pressure_target[active], axis=1))),
            "target_residual_after": float(np.mean(np.linalg.norm(active_state[active, :2] - pressure_target[active], axis=1))),
            "hidden_truth_rmse_before": float(np.sqrt(np.mean((observation_translation - truth_translation) ** 2))),
            "hidden_truth_rmse_after": float(np.sqrt(np.mean((corrected_translation - truth_translation) ** 2))),
        },
        "parameters": common,
        "outputs": {"plot": plot_path},
        "interpretation": "With correction priors centered at zero, a graph without a nonzero external factor is minimized exactly by zero. A pressure factor can move the graph, but it can also worsen hidden truth because the graph optimizes declared factors, not reality.",
    }
    write_json(output_dir / "pose_graph_report.json", report)
    return report


def circle_escape_constraints(points: np.ndarray, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    closest = points / np.maximum(norms[:, None], 1.0e-12) * radius
    displacement = closest - points
    depths = np.linalg.norm(displacement, axis=1)
    normals = displacement / np.maximum(depths[:, None], 1.0e-12)
    return closest, normals, depths


def solve_halfspace_escape(points: np.ndarray) -> dict[str, Any]:
    closest, normals, depths = circle_escape_constraints(points)
    x0 = np.average(closest - points, axis=0, weights=np.maximum(depths, 1.0e-12))
    constraint = LinearConstraint(normals, depths, np.full_like(depths, np.inf))
    result = minimize(
        lambda x: 0.5 * float(np.dot(x, x)),
        x0=x0,
        jac=lambda x: np.asarray(x, dtype=float),
        constraints=[constraint],
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 300, "disp": False},
    )
    translation = np.asarray(result.x, dtype=float)
    slack = normals @ translation - depths
    return {
        "success": bool(result.success and np.all(np.isfinite(translation)) and float(np.min(slack)) >= -1.0e-7),
        "message": str(result.message),
        "translation": translation,
        "translation_norm": float(np.linalg.norm(translation)),
        "closest": closest,
        "normals": normals,
        "depths": depths,
        "slack": slack,
        "min_slack": float(np.min(slack)),
    }


def draw_circle_geometry(axis: plt.Axes, points: np.ndarray, solution: dict[str, Any], title: str) -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, 300)
    axis.plot(np.cos(angles), np.sin(angles), color="black", label="object boundary")
    axis.fill(np.cos(angles), np.sin(angles), color="tab:gray", alpha=0.12)
    axis.scatter(points[:, 0], points[:, 1], color="tab:red", label="penetrating hand points")
    for point, closest in zip(points, solution["closest"]):
        axis.arrow(point[0], point[1], closest[0] - point[0], closest[1] - point[1], width=0.004, color="tab:orange", length_includes_head=True)
    if solution["success"]:
        moved = points + solution["translation"][None, :]
        axis.scatter(moved[:, 0], moved[:, 1], color="tab:green", marker="x", s=55, label="after rigid translation")
        axis.arrow(0.0, 0.0, solution["translation"][0], solution["translation"][1], width=0.008, color="tab:blue", length_includes_head=True, label="least-norm t")
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.4, 1.4)
    axis.set_ylim(-1.3, 1.3)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower left", fontsize=7)


def draw_translation_feasible_set(axis: plt.Axes, solution: dict[str, Any], title: str) -> None:
    grid = np.linspace(-0.65, 0.65, 260)
    tx, ty = np.meshgrid(grid, grid)
    candidates = np.stack([tx.reshape(-1), ty.reshape(-1)], axis=1)
    feasible = np.all(candidates @ solution["normals"].T >= solution["depths"][None, :] - 1.0e-8, axis=1)
    feasible_image = feasible.reshape(tx.shape)
    axis.contourf(tx, ty, feasible_image.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#f6d5d5", "#d8f1d8"], alpha=0.85)
    objective = 0.5 * (tx * tx + ty * ty)
    axis.contour(tx, ty, objective, levels=[0.01, 0.03, 0.07, 0.14, 0.24], colors="gray", linewidths=0.7)
    if solution["success"]:
        axis.scatter([solution["translation"][0]], [solution["translation"][1]], color="tab:blue", s=65, label="optimum")
    axis.scatter([0.0], [0.0], color="black", marker="+", s=80, label="zero")
    axis.set_title(title)
    axis.set_xlabel("translation x")
    axis.set_ylabel("translation y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper left", fontsize=8)


def run_halfspace(output_dir: Path, seed: int) -> dict[str, Any]:
    """P16 analogue: feasible and conflicting least-norm escape translations."""
    del seed
    feasible_points = np.asarray([[0.78, -0.16], [0.82, -0.04], [0.76, 0.08], [0.84, 0.17]], dtype=float)
    conflicting_points = np.asarray([[0.78, 0.0], [-0.78, 0.0], [0.0, 0.82]], dtype=float)
    feasible_solution = solve_halfspace_escape(feasible_points)
    conflicting_solution = solve_halfspace_escape(conflicting_points)

    figure, axes = plt.subplots(2, 2, figsize=(11, 10))
    draw_circle_geometry(axes[0, 0], feasible_points, feasible_solution, "Feasible local escape")
    draw_translation_feasible_set(axes[0, 1], feasible_solution, "Feasible halfspace intersection")
    draw_circle_geometry(axes[1, 0], conflicting_points, conflicting_solution, "Conflicting rigid escape directions")
    draw_translation_feasible_set(axes[1, 1], conflicting_solution, "Empty/infeasible intersection")
    figure.suptitle("P16-like least-norm halfspace escape")
    figure.tight_layout()
    plot_path = output_dir / "halfspace_escape.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = {
        "problem": "least_norm_halfspace_escape",
        "analogue": "P16 MANO/object rigid translation candidate",
        "feasible": feasible_solution,
        "conflicting": conflicting_solution,
        "outputs": {"plot": plot_path},
        "interpretation": "A local cluster can be cleared by one least-norm translation. Opposite penetration normals can make a rigid translation infeasible; optimizer failure is then a model-capacity/sign-support diagnosis, not a reason to invent a correction.",
    }
    write_json(output_dir / "halfspace_report.json", report)
    return report


def torch_hand_forward(
    root: torch.Tensor,
    root_angle: torch.Tensor,
    pose: torch.Tensor,
    *,
    samples_per_link: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple differentiable 3-link planar hand/finger."""
    lengths = torch.tensor([0.14, 0.11, 0.09], dtype=root.dtype, device=root.device)
    angles = torch.stack(
        [root_angle, root_angle + pose[:, 0], root_angle + pose[:, 0] + pose[:, 1]], dim=1
    )
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    joints = [root]
    current = root
    segment_points: list[torch.Tensor] = []
    fractions = torch.linspace(1.0 / samples_per_link, 1.0, samples_per_link, dtype=root.dtype, device=root.device)
    for link in range(3):
        vector = lengths[link] * directions[:, link]
        segment_points.append(current[:, None, :] + fractions[None, :, None] * vector[:, None, :])
        current = current + vector
        joints.append(current)
    return torch.stack(joints, dim=1), torch.cat(segment_points, dim=1)


def temporal_loss(value: torch.Tensor, smooth_weight: float, accel_weight: float) -> torch.Tensor:
    if value.shape[0] <= 1:
        return torch.zeros((), dtype=value.dtype, device=value.device)
    velocity = value[1:] - value[:-1]
    loss = float(smooth_weight) * torch.mean(velocity * velocity)
    if value.shape[0] > 2:
        acceleration = value[2:] - 2.0 * value[1:-1] + value[:-2]
        loss = loss + float(accel_weight) * torch.mean(acceleration * acceleration)
    return loss


def mano_toy_terms(
    translation_delta: torch.Tensor,
    root_delta: torch.Tensor,
    pose_delta: torch.Tensor,
    contact_logit: torch.Tensor,
    base_root: torch.Tensor,
    base_root_angle: torch.Tensor,
    base_pose: torch.Tensor,
    base_joints: torch.Tensor,
    contact_active: torch.Tensor,
    contact_prior: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    root = base_root + translation_delta
    root_angle = base_root_angle + root_delta
    pose = base_pose + pose_delta
    joints, surface_points = torch_hand_forward(root, root_angle, pose)
    contact_probability = torch.sigmoid(contact_logit)

    terms: dict[str, torch.Tensor] = {}
    terms["translation_prior"] = 28.0 * torch.mean(translation_delta * translation_delta)
    terms["root_prior"] = 6.0 * torch.mean(root_delta * root_delta)
    terms["pose_prior"] = 3.0 * torch.mean(pose_delta * pose_delta)
    terms["temporal"] = (
        temporal_loss(translation_delta, 70.0, 130.0)
        + temporal_loss(root_delta[:, None], 35.0, 70.0)
        + temporal_loss(pose_delta, 28.0, 55.0)
    )

    penetration = torch.relu(0.004 - surface_points[:, :, 1])
    positive_count = torch.clamp(torch.sum((penetration > 0.0).to(translation_delta.dtype)), min=1.0)
    terms["nonpenetration"] = 1800.0 * torch.sum(penetration * penetration) / positive_count

    joint_shift = torch.linalg.norm(joints - base_joints, dim=2)
    terms["self_projection_hinge"] = 220.0 * torch.mean(torch.relu(joint_shift - 0.055) ** 2)
    joint_depth_shift = torch.abs(joints[:, :, 1] - base_joints[:, :, 1])
    terms["self_depth_hinge"] = 500.0 * torch.mean(torch.relu(joint_depth_shift - 0.045) ** 2)

    translation_norm = torch.linalg.norm(translation_delta, dim=1)
    root_norm = torch.abs(root_delta)
    pose_norm = torch.abs(pose_delta)
    terms["soft_bounds"] = 300.0 * (
        torch.mean(torch.relu(translation_norm - 0.085) ** 2)
        + torch.mean(torch.relu(root_norm - 0.35) ** 2)
        + torch.mean(torch.relu(pose_norm - 0.50) ** 2)
    )

    active_float = contact_active.to(translation_delta.dtype)
    active_count = torch.clamp(torch.sum(active_float), min=1.0)
    terms["contact_state_prior"] = 8.0 * torch.sum(active_float * (contact_probability - contact_prior) ** 2) / active_count
    pair_active = contact_active[1:] & contact_active[:-1]
    pair_count = torch.clamp(torch.sum(pair_active.to(translation_delta.dtype)), min=1.0)
    terms["contact_state_temporal"] = 5.0 * torch.sum(
        pair_active.to(translation_delta.dtype) * (contact_probability[1:] - contact_probability[:-1]) ** 2
    ) / pair_count

    fingertip_height = joints[:, -1, 1]
    contact_residual = torch.relu(torch.abs(fingertip_height) - 0.015)
    terms["contact_patch"] = 650.0 * torch.sum(
        active_float * contact_probability * contact_residual * contact_residual
    ) / active_count
    total = torch.stack(list(terms.values())).sum()
    return terms, joints, surface_points, contact_probability


def tensor_terms_to_float(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in terms.items()}


def run_mano(output_dir: Path, seed: int) -> dict[str, Any]:
    """P18 analogue: coupled priors, temporal terms, barriers, contact switch and output gate."""
    torch.manual_seed(seed + 37)
    dtype = torch.float64
    frame_count = 48
    frame = torch.arange(frame_count, dtype=dtype)
    contact_active = (frame >= 10) & (frame <= 37)
    base_root_x = -0.52 + 1.04 * frame / (frame_count - 1)
    base_root_y = torch.where(contact_active, torch.full_like(frame, 0.255), torch.full_like(frame, 0.405))
    base_root_y = base_root_y + 0.012 * torch.sin(2.0 * math.pi * frame / frame_count)
    base_root = torch.stack([base_root_x, base_root_y], dim=1)
    base_root_angle = -0.5 * math.pi + 0.05 * torch.sin(2.0 * math.pi * frame / frame_count)
    base_pose = torch.stack(
        [0.10 * torch.sin(2.0 * math.pi * frame / frame_count), -0.08 * torch.cos(2.0 * math.pi * frame / frame_count)],
        dim=1,
    )
    with torch.no_grad():
        base_joints, base_surface = torch_hand_forward(base_root, base_root_angle, base_pose)

    translation_delta = torch.zeros((frame_count, 2), dtype=dtype, requires_grad=True)
    root_delta = torch.zeros((frame_count,), dtype=dtype, requires_grad=True)
    pose_delta = torch.zeros((frame_count, 2), dtype=dtype, requires_grad=True)
    contact_prior = torch.where(contact_active, torch.full_like(frame, 0.88), torch.full_like(frame, 0.05))
    contact_logit = torch.logit(torch.clamp(contact_prior, 1.0e-4, 1.0 - 1.0e-4)).detach().clone().requires_grad_(True)
    parameters = [translation_delta, root_delta, pose_delta, contact_logit]

    with torch.no_grad():
        initial_terms_t, initial_joints_t, initial_surface_t, initial_contact_t = mano_toy_terms(
            translation_delta,
            root_delta,
            pose_delta,
            contact_logit,
            base_root,
            base_root_angle,
            base_pose,
            base_joints,
            contact_active,
            contact_prior,
        )
        initial_terms = tensor_terms_to_float(initial_terms_t)

    optimizer = torch.optim.LBFGS(parameters, lr=0.45, max_iter=140, line_search_fn="strong_wolfe")
    closure_trace: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        terms, _, _, _ = mano_toy_terms(
            translation_delta,
            root_delta,
            pose_delta,
            contact_logit,
            base_root,
            base_root_angle,
            base_pose,
            base_joints,
            contact_active,
            contact_prior,
        )
        total = torch.stack(list(terms.values())).sum()
        total.backward()
        closure_trace.append(float(total.detach().cpu()))
        return total

    optimizer.step(closure)
    with torch.no_grad():
        final_terms_t, raw_joints_t, raw_surface_t, final_contact_t = mano_toy_terms(
            translation_delta,
            root_delta,
            pose_delta,
            contact_logit,
            base_root,
            base_root_angle,
            base_pose,
            base_joints,
            contact_active,
            contact_prior,
        )
        final_terms = tensor_terms_to_float(final_terms_t)
        gate_shift = base_root - raw_joints_t[:, 0]
        gated_joints_t = raw_joints_t + gate_shift[:, None, :]
        gated_surface_t = raw_surface_t + gate_shift[:, None, :]

    base_joints_np = initial_joints_t.detach().cpu().numpy()
    base_surface_np = initial_surface_t.detach().cpu().numpy()
    raw_joints_np = raw_joints_t.detach().cpu().numpy()
    raw_surface_np = raw_surface_t.detach().cpu().numpy()
    gated_joints_np = gated_joints_t.detach().cpu().numpy()
    gated_surface_np = gated_surface_t.detach().cpu().numpy()
    contact_probability_np = final_contact_t.detach().cpu().numpy()
    contact_prior_np = contact_prior.detach().cpu().numpy()
    translation_np = translation_delta.detach().cpu().numpy()

    def penetration_per_frame(surface: np.ndarray) -> np.ndarray:
        return np.max(np.maximum(0.0, 0.004 - surface[:, :, 1]), axis=1)

    base_penetration = penetration_per_frame(base_surface_np)
    raw_penetration = penetration_per_frame(raw_surface_np)
    gated_penetration = penetration_per_frame(gated_surface_np)

    figure, axes = plt.subplots(3, 2, figsize=(13, 13))
    selected_frames = [8, 16, 29, 41]
    colors = ["tab:purple", "tab:blue", "tab:orange", "tab:green"]
    for idx, color in zip(selected_frames, colors):
        axes[0, 0].plot(base_joints_np[idx, :, 0], base_joints_np[idx, :, 1], "o--", color=color, alpha=0.45, label=f"base f{idx}")
        axes[0, 0].plot(raw_joints_np[idx, :, 0], raw_joints_np[idx, :, 1], "o-", color=color, label=f"raw f{idx}")
        axes[0, 1].plot(base_joints_np[idx, :, 0], base_joints_np[idx, :, 1], "o--", color=color, alpha=0.45, label=f"base f{idx}")
        axes[0, 1].plot(gated_joints_np[idx, :, 0], gated_joints_np[idx, :, 1], "o-", color=color, label=f"gated f{idx}")
    for axis, title in zip(axes[0], ["Raw LBFGS hypothesis", "After no-support wrist gate"]):
        axis.axhspan(-0.20, 0.0, color="tab:gray", alpha=0.18, label="object interior")
        axis.axhline(0.0, color="black", lw=1.0)
        axis.set_title(title)
        axis.axis("equal")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)

    frame_np = np.arange(frame_count)
    axes[1, 0].plot(frame_np, translation_np[:, 1], label="raw root delta y")
    axes[1, 0].axhline(0.085, color="tab:red", ls="--", label="soft translation bound")
    axes[1, 0].axhline(-0.085, color="tab:red", ls="--")
    axes[1, 0].fill_between(frame_np, -0.12, 0.12, where=contact_active.detach().cpu().numpy(), alpha=0.10, color="tab:green")
    axes[1, 0].set_ylabel("translation delta y")
    axes[1, 0].set_title("Raw optimized global translation")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(frame_np, base_joints_np[:, -1, 1], label="base fingertip y")
    axes[1, 1].plot(frame_np, raw_joints_np[:, -1, 1], label="raw fingertip y")
    axes[1, 1].plot(frame_np, gated_joints_np[:, -1, 1], label="post-gate fingertip y")
    axes[1, 1].axhspan(-0.015, 0.015, color="tab:green", alpha=0.13, label="contact deadband")
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_ylabel("fingertip height")
    axes[1, 1].set_title("Contact objective versus saved state")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(frame_np, base_penetration, label="base")
    axes[2, 0].plot(frame_np, raw_penetration, label="raw solver")
    axes[2, 0].plot(frame_np, gated_penetration, label="after output gate")
    axes[2, 0].set_yscale("symlog", linthresh=1.0e-4)
    axes[2, 0].set_ylabel("max penetration")
    axes[2, 0].set_xlabel("frame")
    axes[2, 0].set_title("Re-measure after every state transform")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(frame_np, contact_prior_np, "--", label="contact prior")
    axes[2, 1].plot(frame_np, contact_probability_np, label="contact posterior")
    axes[2, 1].set_ylim(-0.02, 1.02)
    axes[2, 1].set_xlabel("frame")
    axes[2, 1].set_ylabel("probability")
    axes[2, 1].set_title("Latent contact switch")
    axes[2, 1].legend(fontsize=8)
    for axis in axes[1:].reshape(-1):
        axis.grid(alpha=0.22)
    figure.suptitle("P18-like toy: coupled losses and a post-optimization support gate")
    figure.tight_layout()
    trajectory_path = output_dir / "mano_trajectory.png"
    figure.savefig(trajectory_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    keys = list(initial_terms)
    x = np.arange(len(keys))
    axes[0].bar(x - 0.19, [initial_terms[key] for key in keys], width=0.38, label="initial")
    axes[0].bar(x + 0.19, [final_terms[key] for key in keys], width=0.38, label="final raw")
    axes[0].set_xticks(x, keys, rotation=55, ha="right", fontsize=8)
    axes[0].set_yscale("symlog", linthresh=1.0e-5)
    axes[0].set_ylabel("weighted term value")
    axes[0].set_title("Loss decomposition")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(np.arange(len(closure_trace)), closure_trace)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("LBFGS closure evaluation")
    axes[1].set_ylabel("total loss")
    axes[1].set_title("Strong-Wolfe closure trace (not necessarily monotonic per call)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    losses_path = output_dir / "mano_losses.png"
    figure.savefig(losses_path, dpi=180)
    plt.close(figure)

    report = {
        "problem": "planar_articulated_hand_interval",
        "analogue": "P18 joint MANO interval trajectory",
        "frame_count": frame_count,
        "variables": {
            "translation_delta": [frame_count, 2],
            "root_delta": [frame_count],
            "pose_delta": [frame_count, 2],
            "contact_logit": [frame_count],
        },
        "optimizer": {"kind": "torch.optim.LBFGS", "max_iter": 140, "closure_evaluations": len(closure_trace)},
        "initial_loss_terms": initial_terms,
        "final_raw_loss_terms": final_terms,
        "total_loss_initial": float(sum(initial_terms.values())),
        "total_loss_final_raw": float(sum(final_terms.values())),
        "raw_state": {
            "translation_delta_norm_median": float(np.median(np.linalg.norm(translation_np, axis=1))),
            "translation_delta_norm_max": float(np.max(np.linalg.norm(translation_np, axis=1))),
            "penetration_median": float(np.median(raw_penetration)),
            "penetration_max": float(np.max(raw_penetration)),
            "contact_gap_active_median": float(np.median(np.abs(raw_joints_np[contact_active.detach().cpu().numpy(), -1, 1]))),
        },
        "output_gate": {
            "policy": "no visible support on every row; re-anchor wrist/root to the source while preserving articulation",
            "applied_count": frame_count,
            "shift_norm_median": float(np.median(np.linalg.norm(gate_shift.detach().cpu().numpy(), axis=1))),
            "penetration_median_after_gate": float(np.median(gated_penetration)),
            "penetration_max_after_gate": float(np.max(gated_penetration)),
            "contact_gap_active_median_after_gate": float(np.median(np.abs(gated_joints_np[contact_active.detach().cpu().numpy(), -1, 1]))),
        },
        "contact_state": {
            "prior_active_median": float(np.median(contact_prior_np[contact_active.detach().cpu().numpy()])),
            "posterior_active_median": float(np.median(contact_probability_np[contact_active.detach().cpu().numpy()])),
            "posterior_minus_prior_abs_max": float(np.max(np.abs(contact_probability_np - contact_prior_np))),
        },
        "outputs": {"trajectory": trajectory_path, "losses": losses_path},
        "interpretation": "The raw optimizer can trade root motion, articulation, contact and penetration. A post-hoc support gate changes the state after optimization, so all physical residuals must be re-measured on the gated/canonical state.",
    }
    write_json(output_dir / "mano_report.json", report)
    return report


def build_joint_linear_system(frame_count: int, anchored: bool) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    t = np.linspace(0.0, 1.0, frame_count)
    camera = 0.35 * np.sin(2.0 * math.pi * t)
    obj = camera + 0.55 + 0.08 * np.sin(1.5 * math.pi * t)
    hand = obj + 0.03 * np.cos(3.0 * math.pi * t)
    observations = {
        "camera": camera,
        "object": obj,
        "hand": hand,
        "object_image": obj - camera,
        "hand_image": hand - camera,
        "contact": hand - obj,
    }
    columns = 3 * frame_count
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def row_for(entries: list[tuple[int, float]], value: float, weight: float) -> None:
        row = np.zeros(columns, dtype=float)
        for index, coefficient in entries:
            row[index] = coefficient * math.sqrt(weight)
        rows.append(row)
        rhs.append(value * math.sqrt(weight))

    # Variables are [camera_0..T, object_0..T, hand_0..T].
    for i in range(frame_count):
        row_for([(i, -1.0), (frame_count + i, 1.0)], observations["object_image"][i], 1.0)
        row_for([(i, -1.0), (2 * frame_count + i, 1.0)], observations["hand_image"][i], 1.0)
        row_for([(frame_count + i, -1.0), (2 * frame_count + i, 1.0)], observations["contact"][i], 0.5)
    for block in range(3):
        offset = block * frame_count
        for i in range(1, frame_count):
            row_for([(offset + i - 1, -1.0), (offset + i, 1.0)], 0.0, 0.08)
    if anchored:
        row_for([(0, 1.0)], camera[0], 4.0)
    return np.stack(rows), np.asarray(rhs), observations


def run_joint_gauge(output_dir: Path, seed: int) -> dict[str, Any]:
    """Monolithic MAP analogue: expose a common-translation gauge direction."""
    del seed
    frame_count = 24
    matrix_free, rhs_free, truth = build_joint_linear_system(frame_count, anchored=False)
    matrix_anchor, rhs_anchor, _ = build_joint_linear_system(frame_count, anchored=True)
    free_solution, *_ = np.linalg.lstsq(matrix_free, rhs_free, rcond=None)
    anchored_solution, *_ = np.linalg.lstsq(matrix_anchor, rhs_anchor, rcond=None)
    singular_free = np.linalg.svd(matrix_free, compute_uv=False)
    singular_anchor = np.linalg.svd(matrix_anchor, compute_uv=False)

    common_direction = np.ones(3 * frame_count, dtype=float)
    common_direction /= np.linalg.norm(common_direction)
    alpha = np.linspace(-2.5, 2.5, 240)

    def energy(matrix: np.ndarray, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        candidates = solution[None, :] + alpha[:, None] * common_direction[None, :]
        residual = candidates @ matrix.T - rhs[None, :]
        return 0.5 * np.sum(residual * residual, axis=1)

    energy_free = energy(matrix_free, rhs_free, free_solution)
    energy_anchor = energy(matrix_anchor, rhs_anchor, anchored_solution)
    frame = np.arange(frame_count)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(alpha, energy_free - energy_free.min(), label="relative-only factors")
    axes[0].plot(alpha, energy_anchor - energy_anchor.min(), label="with camera gauge anchor")
    axes[0].set_yscale("symlog", linthresh=1.0e-10)
    axes[0].set_xlabel("common camera/object/hand shift")
    axes[0].set_ylabel("energy above minimum")
    axes[0].set_title("Loss along global gauge direction")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].semilogy(np.arange(len(singular_free)), np.sort(singular_free), ".", label="relative-only")
    axes[1].semilogy(np.arange(len(singular_anchor)), np.sort(singular_anchor), ".", label="anchored")
    axes[1].set_xlabel("sorted singular-value index")
    axes[1].set_ylabel("singular value")
    axes[1].set_title("One near-zero mode is a gauge, not optimizer failure")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(frame, truth["camera"], "k--", label="true camera")
    axes[2].plot(frame, truth["object"], "k-.", label="true object")
    axes[2].plot(frame, truth["hand"], "k:", label="true hand")
    axes[2].plot(frame, anchored_solution[:frame_count], label="estimated camera")
    axes[2].plot(frame, anchored_solution[frame_count : 2 * frame_count], label="estimated object")
    axes[2].plot(frame, anchored_solution[2 * frame_count :], label="estimated hand")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("1-D world position")
    axes[2].set_title("Anchored joint MAP estimate")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    plot_path = output_dir / "joint_gauge.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = {
        "problem": "joint_camera_object_hand_map_gauge",
        "analogue": "a monolithic camera + object + hand + contact optimization",
        "relative_only": {
            "matrix_shape": list(matrix_free.shape),
            "smallest_singular_value": float(np.min(singular_free)),
            "largest_singular_value": float(np.max(singular_free)),
            "energy_range_along_common_shift": float(np.max(energy_free) - np.min(energy_free)),
        },
        "anchored": {
            "matrix_shape": list(matrix_anchor.shape),
            "smallest_singular_value": float(np.min(singular_anchor)),
            "largest_singular_value": float(np.max(singular_anchor)),
            "energy_range_along_common_shift": float(np.max(energy_anchor) - np.min(energy_anchor)),
        },
        "outputs": {"plot": plot_path},
        "interpretation": "A large joint problem can be sparse and solvable, but relative observations leave gauge freedoms. Adding more losses does not automatically add information; explicit anchors and uncertainty are required.",
    }
    write_json(output_dir / "joint_gauge_report.json", report)
    return report


def parse_problem_list(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PROBLEMS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(PROBLEMS))
    if unknown:
        raise ValueError(f"unknown problems {unknown}; choices={PROBLEMS} or all")
    return requested


def write_index_html(output_dir: Path, reports: dict[str, Any], report_names: dict[str, str]) -> Path:
    cards: list[str] = []
    root = output_dir.resolve()
    for problem, report in reports.items():
        images: list[str] = []
        for raw_path in (report.get("outputs") or {}).values():
            path = Path(str(raw_path)).resolve()
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = path.as_posix()
            images.append(f'<a href="{relative}"><img src="{relative}" alt="{problem}" loading="lazy"></a>')
        report_relative = f"{problem}/{report_names[problem]}"
        interpretation = str(report.get("interpretation") or "")
        cards.append(
            f'<section><h2>{problem}</h2><p>{interpretation}</p>'
            f'<p><a href="{report_relative}">JSON report</a></p>'
            f'<div class="images">{"".join(images)}</div></section>'
        )
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V19 Toy Optimization Lab</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.45;background:#fafafa;color:#222}
section{background:white;padding:1rem 1.25rem;margin:1rem 0;border:1px solid #ddd;border-radius:10px}
.images{display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start}.images img{max-width:min(100%,720px);height:auto;border:1px solid #ddd}
code{background:#eee;padding:.1rem .25rem}a{color:#1459a6}
</style></head><body><h1>V19 Toy Optimization Lab</h1>
<p>Synthetic visual analogues only; not production inference or metric evaluation.</p>
""" + "\n".join(cards) + "</body></html>\n"
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="all", help="Comma-separated subset: sim2,pose_graph,halfspace,mano,joint_gauge, or all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy_optimization_lab"))
    parser.add_argument("--seed", type=int, default=1907)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runners: dict[str, Callable[[Path, int], dict[str, Any]]] = {
        "sim2": run_sim2,
        "pose_graph": run_pose_graph,
        "halfspace": run_halfspace,
        "mano": run_mano,
        "joint_gauge": run_joint_gauge,
    }
    reports: dict[str, Any] = {}
    for problem in parse_problem_list(args.problems):
        problem_dir = args.output_dir / problem
        problem_dir.mkdir(parents=True, exist_ok=True)
        print(f"[toy-optimization] running {problem} -> {problem_dir}", flush=True)
        reports[problem] = runners[problem](problem_dir, int(args.seed))
    summary = {
        "method": "v19_toy_optimization_lab",
        "seed": int(args.seed),
        "problems": list(reports),
        "output_dir": str(args.output_dir),
        "reports": {key: str(args.output_dir / key / f"{key}_report.json") for key in reports},
        "claim_scope": "Low-dimensional visual analogues for understanding optimizer behavior; not production inference, not metric evaluation, and not a replacement for V19 states.",
    }
    # Correct report names that are intentionally more descriptive.
    report_names = {
        "sim2": "sim2_report.json",
        "pose_graph": "pose_graph_report.json",
        "halfspace": "halfspace_report.json",
        "mano": "mano_report.json",
        "joint_gauge": "joint_gauge_report.json",
    }
    index_path = write_index_html(args.output_dir, reports, report_names)
    summary["reports"] = {key: str(args.output_dir / key / report_names[key]) for key in reports}
    summary["index_html"] = str(index_path)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
