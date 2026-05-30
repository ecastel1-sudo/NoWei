#!/usr/bin/env python3
"""Monitoring dashboard for the MoE load-balancing submission.

This dashboard has TWO complementary halves:

A) LOCAL SIMULATOR HALF (uses the committed LmSys traces under trace/).

   For every requested case it runs the exact simulator math from
   dynamic_lb_simulator.py / quickstart.py against three methods:
     - default     : no redeploy, simulator's static placement.
     - deepseek    : the DeepSeek EPLB baseline (eplb_algorithms/deepseek.py).
     - submission  : the LATEST `submission*.py` found at the repo root,
                     auto-detected by file mtime so you do not have to remember
                     which version is current. Use --submission-file to override.

   Per case outputs (output/figure/<plot>_<case>.png):
     - PAR_over_iterations_<case>.png      mean PAR per iteration
     - Cumulative_Transmit_<case>.png      cumulative #expert moves
     - Algo_Time_Per_Call_<case>.png       wallclock per rebalance() call
     - Score_Summary_<case>.png            final mean PAR / transmit / score
     - Constraint_Check_<case>.png         PASS/FAIL competition constraints
     - PAR_Distribution_<case>.png         per-iter PAR distribution box+violin
     - Time_Decomposition_<case>.png       balanced-compute vs transmission

   With --compare-versions every `submission_v*.py` plus the active
   `submission.py` is also run side-by-side, producing a single
   Submission_Versions_<case>.png that shows score / mean PAR / total
   transit / wall time per version.

B) GRADER-FEEDBACK HALF (uses ./feedbacks/ written by the official grader).

   Walks `feedbacks/prediction_result_*` (per-case PAR + transit) and the
   matched `feedbacks/scoring_result_*` (composite score, totals) for every
   version we ever submitted, then produces:
     - Grader_Score_Evolution.png          composite score / total time /
                                           transit / mean PAR per version
     - Energy_Score_Evolution.png          per-version EnergyScore (bit-energy
                                           variant from ENERGY_COMPOSITE_SCORE
                                           .pdf), absolute MJ, time-score
                                           comparison, and compute-share-of-
                                           total-energy. Re-ranks versions
                                           under the joules-based model where
                                           movement is priced at NVLink+HBM
                                           datapath cost only.
     - Grader_Per_Case_Heatmap_<vN>.png    PAR + transit heatmaps for the
                                           latest version (rows=model+dataset,
                                           cols=EP)
     - Grader_EP_Scaling_<vN>.png          PAR vs EP and transit vs EP curves
                                           per dataset (one panel per model)
     - Grader_Time_Decomposition_<vN>.png  per-case balanced-compute vs
                                           transmission stacked bars
                                           (which cases are PAR-bound vs
                                           transit-bound on the real grader)
     - Grader_Latest_vs_Previous.png       per-case delta of PAR + transit
                                           (latest vs previous version);
                                           catches regressions at a glance

Usage:
  python dashboard.py                          # default: Qwen3 LmSys EP32
                                               # + grader-feedback dashboard
  python dashboard.py --model DS-R1 --ep 64
  python dashboard.py --all-samples            # all model x ep cases on the
                                               # committed traces
  python dashboard.py --compare-versions       # also run every submission_vN
                                               # locally for side-by-side
  python dashboard.py --feedback-only          # just the grader plots, no sim
  python dashboard.py --no-feedback            # local sim only
  python dashboard.py --check-only             # fast PASS/FAIL, exit 0/1
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend; we only save PNGs
import matplotlib.pyplot as plt
import matplotlib.ticker  # explicit import: used by EP-scaling x-axis ticks

from eplb_algorithms import rebalance as build_rebalance


# ---------------------------------------------------------------------------
# Constants and shapes that match the simulator exactly.
# ---------------------------------------------------------------------------
MODEL_SHAPES = {
    "DS-R1": (58, 256),    # (n_layers, n_experts)
    "Qwen3": (94, 128),
}

BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
TRANSFER_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000

# Energy-composite constants (see ENERGY_COMPOSITE_SCORE.pdf and the matching
# block in grader_sim.py). H100-SXM at 700 W with NVLink 4.0 bandwidth =
# 900 GB/s — the simulator's published bandwidth — gives:
#   1 PAR-point per case window  ~  42_000 J of compute (=H100_TDP_W * 60s)
#   1 expert move (TDP method)   ~  0.069 J  (conservative; equivalent to
#                                             the existing time-based score)
#   1 expert move (bit method)   ~  0.0011 J (NVLink + HBM3 datapath cost,
#                                             physically minimal)
# We use the bit-energy variant for the new EnergyScore so movement is priced
# at its true datapath cost, not the worst-case TDP-window upper bound.
H100_TDP_W = 700.0
NVLINK_PJ_PER_BIT = 1.3
HBM3_PJ_PER_BIT = 0.29
E_PAR_POINT_J = H100_TDP_W * BALANCED_COMPUTE_SECONDS
E_MOVE_J_TDP = H100_TDP_W * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND
E_MOVE_J_BITS = EXPERT_BYTES * 8 * (NVLINK_PJ_PER_BIT + HBM3_PJ_PER_BIT) * 1e-12

METHOD_COLORS = {
    "default": "#888888",
    "deepseek": "#1f77b4",
    "submission": "#d62728",
}

# Generic palette for arbitrary submission_vN entries when --compare-versions
# is on. Picked from matplotlib's tab10 minus the colours already used above
# so the plots stay readable.
_VERSION_PALETTE = [
    "#d62728",  # submission (the active / latest one — keep red)
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # grey
]

# ---------------------------------------------------------------------------
# Competition constraints (read from dynamic_lb_simulator.py).
#   - PER_ITER_BUDGET_S: each iter is 0.08 s; rebalance() runtime is converted
#     into algo_execute_iters = ceil(runtime / 0.08). Any per-call runtime
#     above this wastes that many iterations on the redeploy schedule.
#   - DQ_TOTAL_WALL_MULTIPLIER: the grader DQs submissions whose total wall
#     time exceeds 3x the deepseek baseline (use deepseek wall time as the
#     reference for the check; for "default" we compare to itself).
# Both are surfaced per case as PASS / FAIL so it is obvious at a glance
# whether the submission is inside the rules before uploading.
# ---------------------------------------------------------------------------
PER_ITER_BUDGET_S = 0.08
DQ_TOTAL_WALL_MULTIPLIER = 3.0
CONSTRAINT_BASELINE_METHOD = "deepseek"


# ---------------------------------------------------------------------------
# Helpers (copies of quickstart.py / simulator math).
# ---------------------------------------------------------------------------
def init_deploy_table(n_devices: int, n_experts: int, n_red_experts: int,
                      default: bool) -> np.ndarray:
    n_exp_per_dev = (n_experts + n_red_experts) // n_devices
    deploy = np.zeros((n_devices, n_exp_per_dev), dtype=np.int64)
    for device in range(n_devices):
        if default:
            for slot in range(n_exp_per_dev):
                deploy[device, slot] = (device * n_exp_per_dev + slot) % n_experts
        else:
            for slot in range(n_exp_per_dev - 1):
                deploy[device, slot] = (device * (n_exp_per_dev - 1) + slot) % n_experts
            deploy[device, -1] = deploy[device, -2]
    return deploy


def calculate_par(hotness: np.ndarray, deployment: np.ndarray) -> float:
    n_experts = hotness.shape[0]
    cut = np.bincount(deployment.reshape(-1), minlength=n_experts)
    if np.any(cut == 0):
        raise ValueError("deployment must contain every logical expert at least once")
    weights = hotness / cut
    loads = weights[deployment.reshape(-1)].reshape(deployment.shape).sum(-1)
    return float(loads.max() / loads.mean())


def cal_par_per_iter(cur_hotness: np.ndarray, cur_deploy_table: np.ndarray) -> np.ndarray:
    pars = np.zeros((cur_hotness.shape[0],), dtype=np.float64)
    for layer_idx in range(cur_hotness.shape[0]):
        pars[layer_idx] = calculate_par(cur_hotness[layer_idx], cur_deploy_table[layer_idx])
    return pars


def compute_redeploy_cost(old: np.ndarray, new: np.ndarray) -> int:
    return int(np.sum(old != new))


def transmission_time_seconds(transmit_amount: int) -> float:
    return transmit_amount * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND


def modeled_runtime_seconds(mean_par: float, transmit_amount: int) -> float:
    return BALANCED_COMPUTE_SECONDS * mean_par + transmission_time_seconds(transmit_amount)


def composite_energy_joules(mean_par: float, transmit_amount: int, n_cases: int,
                            move_energy_j: float = E_MOVE_J_BITS) -> float:
    """Total joules across n_cases windows. Mirrors modeled_runtime_seconds but
    in energy units. With move_energy_j = E_MOVE_J_TDP the resulting score is
    identical to composite_score; with E_MOVE_J_BITS movement is weighted at
    NVLink+HBM datapath cost only.
    """
    return E_PAR_POINT_J * float(mean_par) * int(n_cases) + move_energy_j * float(transmit_amount)


def load_trace(repo_root: Path, model: str, dataset: str, max_iters: int) -> np.ndarray:
    path = repo_root / "trace" / model / f"{dataset}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing sample trace: {path}")
    trace = np.load(path, mmap_mode="r")[:max_iters]
    trace = np.array(trace, copy=True)
    trace[trace == 0] = 1
    return trace


# ---------------------------------------------------------------------------
# Submission version discovery.
#
# discover_submission_versions() returns every `submission*.py` at the repo
# root, sorted oldest -> newest by mtime, with a stable label used in plots
# and an algorithm alias understood by eplb_algorithms.rebalance().
#
# The "active" submission (i.e. the one the grader will pick up if you zip
# right now) is always submission.py. We also surface the LATEST file by
# mtime, which may be a versioned WIP like submission_v7.py — that is the
# one you most likely want to look at when iterating, hence we use it as
# the default "submission" entry and label its filename in plot legends so
# there is no ambiguity about what was actually run.
# ---------------------------------------------------------------------------
@dataclass
class SubmissionVersion:
    label: str        # short label used in plots, e.g. 'submission_v7'
    path: Path        # absolute path to the .py file
    algorithm: str    # alias accepted by eplb_algorithms.rebalance()
    mtime: float      # unix mtime, used for sorting


def discover_submission_versions(repo_root: Path) -> list[SubmissionVersion]:
    """Return every submission*.py at repo root, oldest-first by mtime."""
    versions: list[SubmissionVersion] = []
    for path in sorted(repo_root.glob("submission*.py")):
        stem = path.stem  # 'submission' or 'submission_vN'
        algorithm = "proposed" if stem == "submission" else stem
        versions.append(SubmissionVersion(
            label=stem,
            path=path,
            algorithm=algorithm,
            mtime=path.stat().st_mtime,
        ))
    versions.sort(key=lambda v: v.mtime)
    return versions


def latest_submission_version(repo_root: Path) -> SubmissionVersion:
    """Most recently modified `submission*.py`. Falls back to submission.py."""
    versions = discover_submission_versions(repo_root)
    if not versions:
        raise FileNotFoundError(f"no submission*.py at {repo_root}")
    return versions[-1]


# ---------------------------------------------------------------------------
# Generic per-iteration runner.
#
# When rebalance_fn is None we run the "default" baseline: no redeploys,
# n_red_expert=0 packing.
#
# When rebalance_fn is provided we run the simulator's exact one-layer-per-iter
# redeploy loop (see quickstart.run_ds_eplb / dynamic_lb_simulator.forward).
# ---------------------------------------------------------------------------
def run_method(
    method_name: str,
    hotness: np.ndarray,
    ep: int,
    n_layers: int,
    n_experts: int,
    collection_interval: int,
    rebalance_fn: Optional[Callable] = None,
) -> dict:
    n_iterations = len(hotness)
    per_iter_par = np.zeros(n_iterations, dtype=np.float64)
    per_iter_transmit = np.zeros(n_iterations, dtype=np.int64)
    algo_call_iters: list[int] = []
    algo_call_seconds: list[float] = []

    use_red = rebalance_fn is not None
    n_red = ep if use_red else 0
    cur_deploy_table = np.array([
        init_deploy_table(ep, n_experts, n_red, default=not use_red)
        for _ in range(n_layers)
    ])
    next_deploy_table = np.zeros_like(cur_deploy_table)

    redeploy_finish_iter = 0
    expert_ready = True
    cur_layers_priority: list[int] = []

    wall_start = time.time()
    for i in range(1, n_iterations + 1):
        cur_hotness = hotness[i - 1]
        pars = cal_par_per_iter(cur_hotness, cur_deploy_table)
        per_iter_par[i - 1] = float(pars.mean())

        if not use_red:
            continue

        # Pop one queued layer redeploy per iter, matching simulator exactly.
        if not expert_ready and cur_layers_priority:
            adjust_layer_idx = cur_layers_priority.pop(0)
            per_iter_transmit[i - 1] += compute_redeploy_cost(
                cur_deploy_table[adjust_layer_idx],
                next_deploy_table[adjust_layer_idx],
            )
            cur_deploy_table[adjust_layer_idx] = next_deploy_table[adjust_layer_idx]

        if len(cur_layers_priority) == 0 and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + collection_interval + 1 and expert_ready:
            train_window = hotness[i - collection_interval:i]
            call_start = time.time()
            change, layers_priority, deployment_table, _ = rebalance_fn(train_window)
            call_elapsed = time.time() - call_start
            algo_call_iters.append(i)
            algo_call_seconds.append(call_elapsed)
            if change:
                selected_layers = np.asarray(layers_priority, dtype=np.int64)
                cur_layers_priority = selected_layers.tolist()
                expert_ready = False
                next_deploy_table[selected_layers] = deployment_table[selected_layers]

    wall_elapsed = time.time() - wall_start
    mean_par = float(per_iter_par.mean())
    total_transmit = int(per_iter_transmit.sum())
    return {
        "method": method_name,
        "mean_par": mean_par,
        "total_transmit": total_transmit,
        "transmission_time_seconds": transmission_time_seconds(total_transmit),
        "total_time_seconds": modeled_runtime_seconds(mean_par, total_transmit),
        "wall_runtime_seconds": wall_elapsed,
        "per_iter_par": per_iter_par,
        "per_iter_transmit": per_iter_transmit,
        "algo_call_iters": np.asarray(algo_call_iters, dtype=np.int64),
        "algo_call_seconds": np.asarray(algo_call_seconds, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
def _legend_label(r: dict) -> str:
    """Method label that also surfaces the actual submission filename when
    we ran a versioned submission_vN.py instead of plain submission.py.

    Keeps the leaderboard-side plots honest: at a glance you can tell
    'submission (submission_v7)' rather than wondering which file the
    figure came from.
    """
    base = r["method"]
    extra = r.get("display_label")
    if extra and extra != base:
        return f"{base} ({extra})"
    return base


def _save_par_plot(results: list[dict], case_name: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(
            np.arange(len(r["per_iter_par"])),
            r["per_iter_par"],
            label=f"{_legend_label(r)} (mean={r['mean_par']:.4f})",
            color=METHOD_COLORS.get(r["method"]),
            linewidth=1.5,
            alpha=0.85,
        )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean PAR across layers", fontsize=12)
    plt.title(f"PAR over iterations: {case_name}", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="best", frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_transmit_plot(results: list[dict], case_name: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    for r in results:
        cumulative = np.cumsum(r["per_iter_transmit"])
        if len(cumulative):
            label = f"{_legend_label(r)} (total={int(cumulative[-1])})"
        else:
            label = _legend_label(r)
        plt.plot(
            np.arange(len(cumulative)),
            cumulative,
            label=label,
            color=METHOD_COLORS.get(r["method"]),
            linewidth=1.8,
            alpha=0.9,
        )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cumulative expert moves (transit)", fontsize=12)
    plt.title(f"Cumulative transit over iterations: {case_name}", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="best", frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_algo_time_plot(results: list[dict], case_name: str, budget_s: float,
                         out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    plotted_any = False
    for r in results:
        if len(r["algo_call_iters"]) == 0:
            continue
        plt.plot(
            r["algo_call_iters"],
            r["algo_call_seconds"],
            label=f"{r['method']} (max={r['algo_call_seconds'].max():.3f}s, "
                  f"mean={r['algo_call_seconds'].mean():.3f}s)",
            color=METHOD_COLORS.get(r["method"]),
            marker="o",
            linewidth=1.6,
            alpha=0.9,
        )
        plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return
    plt.axhline(
        budget_s,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"per-iter budget = {budget_s:.3f}s",
    )
    plt.xlabel("Iteration at which rebalance() was called", fontsize=12)
    plt.ylabel("Algorithm runtime (seconds)", fontsize=12)
    plt.title(f"Algorithm runtime per rebalance call: {case_name}", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="best", frameon=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_summary_plot(results: list[dict], case_name: str, baseline_method: str,
                       out_path: Path) -> None:
    # Reference total time for the score is the chosen baseline method
    # (matches dynamic_lb_simulator.run: 100 * baseline_time / total_time).
    baseline = next((r for r in results if r["method"] == baseline_method), results[0])
    baseline_total_time = baseline["total_time_seconds"]

    methods = [r["method"] for r in results]
    pars = [r["mean_par"] for r in results]
    transits = [r["total_transmit"] for r in results]
    scores = [100.0 * baseline_total_time / r["total_time_seconds"] for r in results]
    colors = [METHOD_COLORS.get(m, "#444") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(methods, pars, color=colors)
    axes[0].set_title("Mean PAR (lower is better)")
    axes[0].set_ylabel("Mean PAR")
    for i, v in enumerate(pars):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)

    axes[1].bar(methods, transits, color=colors)
    axes[1].set_title("Total expert moves (lower is better)")
    axes[1].set_ylabel("Cumulative transit")
    for i, v in enumerate(transits):
        axes[1].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=10)
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)

    axes[2].bar(methods, scores, color=colors)
    axes[2].set_title(f"Score vs {baseline_method} (higher is better)")
    axes[2].set_ylabel("Score")
    axes[2].axhline(100.0, color="black", linestyle="--", linewidth=1.0)
    for i, v in enumerate(scores):
        axes[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    axes[2].grid(axis="y", linestyle=":", alpha=0.7)

    fig.suptitle(f"Score summary: {case_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_par_distribution_plot(results: list[dict], case_name: str,
                                out_path: Path) -> None:
    """Per-iter PAR distribution per method as a violin + boxplot overlay.

    Why this is useful and not redundant with PAR_over_iterations:
      The line plot shows the time series. This plot shows the SHAPE of the
      distribution (median, IQR, p1/p99 tails, outliers). A method that has
      the same mean PAR but a much fatter upper tail is clearly worse for
      worst-case latency. For the leaderboard score (driven by mean PAR)
      they look equivalent in the line plot; here they don't.
    """
    if not results:
        return
    methods = [r["method"] for r in results]
    data = [r["per_iter_par"] for r in results]
    colors = [METHOD_COLORS.get(m, "#444") for m in methods]

    fig, ax = plt.subplots(figsize=(11, 6))
    parts = ax.violinplot(data, showmeans=False, showmedians=False,
                          showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.45)

    bp = ax.boxplot(data, widths=0.18, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.6)
    for whisker in bp["whiskers"]:
        whisker.set_color("#222")
    for median in bp["medians"]:
        median.set_color("#222")
        median.set_linewidth(1.8)

    means = [float(np.mean(d)) for d in data]
    p99s = [float(np.quantile(d, 0.99)) for d in data]
    ax.scatter(range(1, len(methods) + 1), means, marker="D",
               color="black", zorder=3, label="mean")
    for i, (m, p99) in enumerate(zip(means, p99s), start=1):
        ax.text(i, m, f"  mean={m:.3f}\n  p99={p99:.3f}",
                fontsize=9, va="center")

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean PAR across layers (per iter)", fontsize=12)
    ax.set_title(f"PAR distribution per iter: {case_name}", fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_time_decomposition_plot(results: list[dict], case_name: str,
                                  out_path: Path) -> None:
    """Stacked bar showing modeled compute time vs transmission time.

    The competition score is composed of:
        total_time = 60s * mean_par + transmit_amount * 88MB / 900GBps
                   = balanced_compute_time + transmission_time
    Splitting them visually answers a critical iteration question:
      * If a method's bar is dominated by balanced_compute, you are PAR-bound
        and should prioritise lowering mean PAR (more aggressive redeploys).
      * If transmission dominates, you should prioritise cutting moves
        (better anchoring, harsher gate, lower cadence).
    The dashed line shows the deepseek baseline so you immediately see the
    score = 100 * baseline / total_time relationship.
    """
    if not results:
        return
    methods = [r["method"] for r in results]
    par_seconds = [BALANCED_COMPUTE_SECONDS * r["mean_par"] for r in results]
    transit_seconds = [r["transmission_time_seconds"] for r in results]
    totals = [p + t for p, t in zip(par_seconds, transit_seconds)]

    fig, ax = plt.subplots(figsize=(11, 6))
    par_bar = ax.bar(methods, par_seconds, color="#4c72b0", edgecolor="black",
                     label=f"balanced compute = {BALANCED_COMPUTE_SECONDS}s * mean_par")
    transit_bar = ax.bar(methods, transit_seconds, bottom=par_seconds,
                         color="#dd8452", edgecolor="black",
                         label="transmission = moves * 88MB / 900GBps")

    for i, (p, t, total) in enumerate(zip(par_seconds, transit_seconds, totals)):
        share_t = 100.0 * t / total if total > 0 else 0.0
        ax.text(i, total, f"{total:.1f}s\n(transit={share_t:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    deepseek = next((r for r in results if r["method"] == "deepseek"), None)
    if deepseek is not None:
        ds_total = (BALANCED_COMPUTE_SECONDS * deepseek["mean_par"]
                    + deepseek["transmission_time_seconds"])
        ax.axhline(ds_total, color="black", linestyle="--", linewidth=1.0,
                   label=f"deepseek total = {ds_total:.1f}s (score=100 line)")

    ax.set_ylabel("Modeled total time (s) — lower is better", fontsize=12)
    ax.set_title(f"Time decomposition: {case_name}", fontsize=14)
    ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_versions_comparison_plot(results: list[dict], case_name: str,
                                   baseline_method: str, out_path: Path) -> None:
    """Compare every submission version side-by-side on a single case.

    `results` is expected to contain default + deepseek + one entry per
    submission version (method_name = the version label). Renders a 2x2
    panel showing score / mean_par / total transit / max algo time so a
    single glance tells you which version is winning AND why.
    """
    if not results:
        return
    baseline = next((r for r in results if r["method"] == baseline_method), results[0])
    baseline_total = baseline["total_time_seconds"]

    methods = [r["method"] for r in results]
    pars = [r["mean_par"] for r in results]
    transits = [r["total_transmit"] for r in results]
    scores = [100.0 * baseline_total / r["total_time_seconds"] for r in results]
    max_algo = [float(r["algo_call_seconds"].max())
                if len(r["algo_call_seconds"]) else 0.0
                for r in results]

    def _color(m: str, idx: int) -> str:
        if m in METHOD_COLORS:
            return METHOD_COLORS[m]
        return _VERSION_PALETTE[idx % len(_VERSION_PALETTE)]

    colors = [_color(m, i) for i, m in enumerate(methods)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [
        (axes[0, 0], scores,    f"Score vs {baseline_method} (higher is better)",
         "Score", lambda v: f"{v:.2f}", 100.0),
        (axes[0, 1], pars,      "Mean PAR (lower is better)",
         "Mean PAR", lambda v: f"{v:.4f}", None),
        (axes[1, 0], transits,  "Total expert moves (lower is better)",
         "Cumulative transit", lambda v: f"{int(v)}", None),
        (axes[1, 1], max_algo,  "Max rebalance() runtime (≤ 0.08s budget)",
         "seconds", lambda v: f"{v:.4f}s", PER_ITER_BUDGET_S),
    ]
    for ax, values, title, ylabel, fmt, ref in panels:
        ax.bar(methods, values, color=colors, edgecolor="black")
        if ref is not None:
            ax.axhline(ref, color="black", linestyle="--", linewidth=1.0)
        for i, v in enumerate(values):
            ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Submission version comparison: {case_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grader-feedback ingestion + plots.
#
# The official grader writes two directory trees per submission to
# `feedbacks/`:
#   prediction_result_<suffix>/metrics.json   (per-case PAR + transit)
#   scoring_result_<suffix>/scores.json       (composite score and totals)
#
# Suffix shape varies (`v3`, `+ EWMA + two-level anchor`,
# `109_ + EWMA + two-level anchor_4_gate`, ...). We canonicalise the
# directory name to match the prediction/scoring pair, then sort all the
# versions by mtime to get an iteration timeline. The five plots below
# turn that timeline plus the per-case data into actually-actionable
# diagnostics that the local LmSys-only sim cannot produce, because the
# LmSys trace is not the case set the grader actually scores.
# ---------------------------------------------------------------------------
GRADER_PRED_PREFIX = "prediction_result"
GRADER_SCORE_PREFIX = "scoring_result"
GRADER_LABEL_MAX_LEN = 22


@dataclass
class GraderVersion:
    label: str                    # canonical suffix (e.g. 'v6', '+ EWMA + ...')
    short_label: str              # truncated label for tight x-axes
    cases: list[dict]             # rows from prediction metrics.json
    composite_score: float
    composite_mean_par: float
    composite_transmit: float
    composite_total_time: float
    composite_transmission_time: float
    mtime: float
    pred_dir: Path
    score_dir: Path


def _canonicalize_grader_suffix(name: str, prefix: str) -> str:
    """Strip the prefix + leading underscore/space and trailing period/space.

    Handles all of the inconsistent naming conventions in feedbacks/, e.g.
       'prediction_result + EWMA + two-level anchor'
       'scoring_result_+ EWMA + two-level anchor.'
       'prediction_result_v6'
    All canonicalise to the same key so we can pair them up.
    """
    if not name.startswith(prefix):
        return name
    return name[len(prefix):].lstrip("_ ").rstrip(". ")


def _short_label(label: str, max_len: int = GRADER_LABEL_MAX_LEN) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _approx_modeled_time_per_case(c: dict) -> dict:
    """Add modeled-time columns to a per-case prediction row.

    Mirrors dynamic_lb_simulator.modeled_runtime_seconds:
       compute_s = 60 * mean_par
       transit_s = transmit * 88MB / 900GBps
       total_s   = compute_s + transit_s
    Useful for the time-decomposition plot.
    """
    compute_s = BALANCED_COMPUTE_SECONDS * float(c["mean_par"])
    transit_s = transmission_time_seconds(int(c["transmit_amount"]))
    return {
        **c,
        "balanced_compute_s": compute_s,
        "transmission_s": transit_s,
        "modeled_total_s": compute_s + transit_s,
    }


def discover_grader_versions(repo_root: Path) -> list[GraderVersion]:
    """Pair prediction_result_* with scoring_result_*; return mtime-sorted list."""
    feedback_dir = repo_root / "feedbacks"
    if not feedback_dir.exists():
        return []
    pred_by_key: dict[str, Path] = {}
    score_by_key: dict[str, Path] = {}
    for entry in feedback_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith(GRADER_PRED_PREFIX):
            pred_by_key[
                _canonicalize_grader_suffix(entry.name, GRADER_PRED_PREFIX)
            ] = entry
        elif entry.name.startswith(GRADER_SCORE_PREFIX):
            score_by_key[
                _canonicalize_grader_suffix(entry.name, GRADER_SCORE_PREFIX)
            ] = entry

    versions: list[GraderVersion] = []
    for key, pred_dir in pred_by_key.items():
        score_dir = score_by_key.get(key)
        if score_dir is None:
            print(f"  [feedback] {pred_dir.name}: no matching scoring_result, skipping")
            continue
        try:
            with (pred_dir / "metrics.json").open() as f:
                metrics = json.load(f)
            with (score_dir / "scores.json").open() as f:
                scores = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"  [feedback] {pred_dir.name}: failed to parse ({exc}), skipping")
            continue
        cases = [_approx_modeled_time_per_case(c) for c in metrics.get("cases", [])]
        if not cases:
            continue
        mtime = max(pred_dir.stat().st_mtime, score_dir.stat().st_mtime)
        versions.append(GraderVersion(
            label=key or pred_dir.name,
            short_label=_short_label(key or pred_dir.name),
            cases=cases,
            composite_score=float(scores.get("composite_score", scores.get("score", 0.0))),
            composite_mean_par=float(scores.get("mean_par", 0.0)),
            composite_transmit=float(scores.get("transmit_amount", 0.0)),
            composite_total_time=float(scores.get("total_time_seconds", 0.0)),
            composite_transmission_time=float(scores.get("transmission_time_seconds", 0.0)),
            mtime=mtime,
            pred_dir=pred_dir,
            score_dir=score_dir,
        ))
    versions.sort(key=lambda v: v.mtime)
    return versions


def _case_key(c: dict) -> tuple[str, str, int]:
    """(model, dataset, ep) — stable per-case key for cross-version joins."""
    return (str(c["model"]), str(c["dataset"]), int(c["ep"]))


def _case_label(c: dict) -> str:
    return f"{c['model']} {c['dataset']} EP{c['ep']}"


def _save_grader_score_evolution(versions: list[GraderVersion], out_path: Path) -> None:
    """4-panel timeline of composite score / mean PAR / transit / total time.

    The composite score panel is the headline. Mean PAR and total transmit
    panels reveal which axis is moving the score. The total-time panel is
    just composite_score restated as seconds; useful for sanity checking
    against the simulator math (score = 100 * baseline / total_time).
    """
    if len(versions) < 1:
        return
    labels = [v.short_label for v in versions]
    scores = [v.composite_score for v in versions]
    pars = [v.composite_mean_par for v in versions]
    transits = [v.composite_transmit for v in versions]
    times = [v.composite_total_time for v in versions]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [
        (axes[0, 0], scores,   "Composite score (higher is better)",
         "Score",          "{:.2f}", "#2ca02c", 100.0),
        (axes[0, 1], pars,     "Composite mean PAR (lower is better)",
         "Mean PAR",       "{:.4f}", "#1f77b4", None),
        (axes[1, 0], transits, "Total transit (lower is better)",
         "Expert moves",   "{:.0f}", "#dd8452", None),
        (axes[1, 1], times,    "Total modeled time (lower is better)",
         "Seconds",        "{:.1f}", "#9467bd", None),
    ]
    for ax, values, title, ylabel, fmt, color, ref in panels:
        ax.plot(labels, values, marker="o", color=color, linewidth=2.0)
        if ref is not None:
            ax.axhline(ref, color="black", linestyle="--", linewidth=1.0,
                       label=f"deepseek baseline = {ref}")
            ax.legend(loc="best", fontsize=9)
        for x, v in zip(labels, values):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        # highlight the best version on each axis
        best_idx = (int(np.argmax(values)) if title.startswith("Composite score")
                    else int(np.argmin(values)))
        ax.scatter([labels[best_idx]], [values[best_idx]], s=160, marker="*",
                   color="gold", edgecolor="black", zorder=5,
                   label=f"best: {labels[best_idx]}")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        f"Grader-feedback timeline ({len(versions)} versions, "
        f"latest = '{versions[-1].label}')", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_energy_score_evolution(versions: list[GraderVersion], out_path: Path) -> None:
    """Re-rank every grader version under the joules-based composite score.

    Why this exists alongside Grader_Score_Evolution.png:
    The competition score is a ratio of modeled times. Multiplying both
    terms by H100_TDP_W converts time -> energy without changing the
    ranking (TDP-energy score == composite_score). The physically minimal
    interpretation is to charge each move at the actual NVLink+HBM datapath
    cost (~0.0011 J/move) instead of the conservative TDP-window cost
    (~0.069 J/move). That re-pricing weighs movement ~63x lighter, so a
    transit-heavy version that the time score punishes can rank better
    here, and a PAR-heavy version that the time score rewards can pull
    even further ahead. See ENERGY_COMPOSITE_SCORE.pdf.

    The earliest version is the 100 reference (the PDF's "v1 = 100"
    convention) because feedbacks/scoring_result_*/scores.json reports
    only the submission's mean_par + transmit, not DS-EPLB's, so we
    cannot reconstruct DS-EPLB's energy directly from the feedback files.
    Use the local simulator (grader_sim.aggregate_composite) for an
    absolute DS-EPLB-anchored energy score.
    """
    if len(versions) < 1:
        return
    labels = [v.short_label for v in versions]

    e_total_mj_bits: list[float] = []
    e_total_mj_tdp: list[float] = []
    compute_share: list[float] = []
    for v in versions:
        n_cases = len(v.cases) or 1
        e_compute = E_PAR_POINT_J * float(v.composite_mean_par) * n_cases
        e_move_bits = E_MOVE_J_BITS * float(v.composite_transmit)
        e_move_tdp = E_MOVE_J_TDP * float(v.composite_transmit)
        e_total_mj_bits.append((e_compute + e_move_bits) / 1e6)
        e_total_mj_tdp.append((e_compute + e_move_tdp) / 1e6)
        compute_share.append(100.0 * e_compute / (e_compute + e_move_bits))

    # Score = 100 * E_first / E_v. Rises when total energy drops.
    e_first_bits = e_total_mj_bits[0] or 1.0
    energy_scores_bits = [100.0 * e_first_bits / max(e, 1e-9) for e in e_total_mj_bits]
    time_scores = [v.composite_score for v in versions]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [
        (axes[0, 0], energy_scores_bits,
         f"EnergyScore (bits, ref = '{versions[0].short_label}' = 100)",
         "EnergyScore",   "{:.2f}", "#2ca02c", 100.0, "max"),
        (axes[0, 1], e_total_mj_bits,
         "Total energy (MJ, bit-energy)",
         "MJ",            "{:.2f}", "#9467bd", None,  "min"),
        (axes[1, 0], time_scores,
         "Time-based composite_score (for comparison)",
         "Score",         "{:.2f}", "#1f77b4", 100.0, "max"),
        (axes[1, 1], compute_share,
         "Compute share of total energy (bit-energy)",
         "%",             "{:.2f}", "#dd8452", None,  "max"),
    ]
    for ax, values, title, ylabel, fmt, color, ref, mode in panels:
        ax.plot(labels, values, marker="o", color=color, linewidth=2.0)
        if ref is not None:
            ax.axhline(ref, color="black", linestyle="--", linewidth=1.0,
                       label=f"reference = {ref}")
            ax.legend(loc="best", fontsize=9)
        for x, v in zip(labels, values):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        best_idx = int(np.argmax(values)) if mode == "max" else int(np.argmin(values))
        ax.scatter([labels[best_idx]], [values[best_idx]], s=160, marker="*",
                   color="gold", edgecolor="black", zorder=5,
                   label=f"best: {labels[best_idx]}")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        f"Energy-based composite (bits): {len(versions)} versions, "
        f"1 PAR-pt = {E_PAR_POINT_J/1000:.0f} kJ, "
        f"1 move = {E_MOVE_J_BITS*1000:.4f} mJ "
        f"(see ENERGY_COMPOSITE_SCORE.pdf)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grader_per_case_heatmap(version: GraderVersion, out_path: Path) -> None:
    """Two heatmaps for the latest version: PAR per (model+dataset, EP) and
    transit per (model+dataset, EP). Cells are annotated with raw values.

    This plot is the single most useful "where am I bleeding score" view:
    bright PAR cells = layers that need better placement; bright transit
    cells = the gate / anchor is too soft for that EP.
    """
    if not version.cases:
        return
    # Build the (row, col) -> value tables.
    rows_set: set[tuple[str, str]] = set()
    cols_set: set[int] = set()
    for c in version.cases:
        rows_set.add((str(c["model"]), str(c["dataset"])))
        cols_set.add(int(c["ep"]))
    rows = sorted(rows_set, key=lambda r: (r[0], r[1]))
    cols = sorted(cols_set)
    par_grid = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
    tx_grid = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
    for c in version.cases:
        i = rows.index((str(c["model"]), str(c["dataset"])))
        j = cols.index(int(c["ep"]))
        par_grid[i, j] = float(c["mean_par"])
        tx_grid[i, j] = float(c["transmit_amount"])
    row_labels = [f"{m} / {d}" for m, d in rows]
    col_labels = [f"EP{ep}" for ep in cols]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, len(rows) * 0.6 + 2)))

    def _draw(ax, grid, title, cmap, fmt):
        im = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title, fontsize=12)
        # Pick text colour based on perceived luminance of each cell so the
        # number is readable on both ends of the colourmap. We normalise the
        # cell value into [0, 1] using the colourmap's vmin/vmax, then pick
        # white text on dark cells (luminance < 0.5) and black on light cells.
        finite = np.isfinite(grid)
        vmin = float(np.nanmin(grid)) if finite.any() else 0.0
        vmax = float(np.nanmax(grid)) if finite.any() else 1.0
        span = max(vmax - vmin, 1e-9)
        cm = plt.get_cmap(cmap)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    rgba = cm((v - vmin) / span)
                    # Rec. 709 luminance.
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    text_color = "white" if lum < 0.5 else "black"
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=9, color=text_color)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    _draw(axes[0], par_grid, "Mean PAR (lower better)", "viridis", "{:.3f}")
    _draw(axes[1], tx_grid,  "Transit (lower better)",  "magma",   "{:.0f}")

    fig.suptitle(
        f"Grader per-case breakdown: '{version.label}' "
        f"(score={version.composite_score:.2f})", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grader_ep_scaling(version: GraderVersion, out_path: Path) -> None:
    """For each model, plot mean PAR and transmit as functions of EP, with
    one curve per dataset. EP scaling is the structural lens on the data:
    PAR usually grows superlinearly with EP, transit grows roughly linearly,
    and the gap between datasets tells you how much trace stationarity helps.
    """
    if not version.cases:
        return
    by_model: dict[str, dict[str, list[dict]]] = {}
    for c in version.cases:
        m = str(c["model"])
        d = str(c["dataset"])
        by_model.setdefault(m, {}).setdefault(d, []).append(c)

    models = sorted(by_model)
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 5 * len(models)))
    if len(models) == 1:
        axes = np.asarray([axes])  # normalise to 2D indexing

    for mi, model in enumerate(models):
        ax_par = axes[mi, 0]
        ax_tx = axes[mi, 1]
        datasets = sorted(by_model[model])
        for di, dataset in enumerate(datasets):
            rows = sorted(by_model[model][dataset], key=lambda c: int(c["ep"]))
            eps = [int(c["ep"]) for c in rows]
            pars = [float(c["mean_par"]) for c in rows]
            txs = [float(c["transmit_amount"]) for c in rows]
            ax_par.plot(eps, pars, marker="o", linewidth=1.8, label=dataset)
            ax_tx.plot(eps, txs, marker="o", linewidth=1.8, label=dataset)
            for x, y in zip(eps, pars):
                ax_par.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                xytext=(0, 6), fontsize=8, ha="center")
            for x, y in zip(eps, txs):
                ax_tx.annotate(f"{int(y)}", (x, y), textcoords="offset points",
                               xytext=(0, 6), fontsize=8, ha="center")
        ax_par.set_title(f"{model}: mean PAR vs EP")
        ax_par.set_xlabel("Number of devices (EP)")
        ax_par.set_ylabel("Mean PAR")
        ax_par.set_xscale("log", base=2)
        ax_par.set_xticks(sorted({int(c["ep"]) for c in version.cases
                                   if c["model"] == model}))
        ax_par.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_par.grid(True, linestyle=":", alpha=0.6)
        ax_par.legend(fontsize=9)

        ax_tx.set_title(f"{model}: transit vs EP")
        ax_tx.set_xlabel("Number of devices (EP)")
        ax_tx.set_ylabel("Total expert moves")
        ax_tx.set_xscale("log", base=2)
        ax_tx.set_xticks(sorted({int(c["ep"]) for c in version.cases
                                  if c["model"] == model}))
        ax_tx.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_tx.grid(True, linestyle=":", alpha=0.6)
        ax_tx.legend(fontsize=9)

    fig.suptitle(
        f"Grader EP-scaling curves: '{version.label}' "
        f"(score={version.composite_score:.2f})", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grader_time_decomposition(version: GraderVersion,
                                    out_path: Path) -> None:
    """Per-case stacked bars: balanced_compute_seconds vs transmission_seconds.

    Cases are sorted by total modeled time (largest on the left), so the
    expensive cases that drag the composite score down jump out immediately.
    The transit-share label on each bar tells you whether the case is
    PAR-bound (low transit share -> push for more redeploys) or
    transit-bound (high transit share -> tighten the gate / anchor harder).
    """
    if not version.cases:
        return
    cases = sorted(version.cases, key=lambda c: -c["modeled_total_s"])
    labels = [_case_label(c) for c in cases]
    par_s = [c["balanced_compute_s"] for c in cases]
    tx_s = [c["transmission_s"] for c in cases]
    totals = [c["modeled_total_s"] for c in cases]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.55), 7))
    ax.bar(labels, par_s, color="#4c72b0", edgecolor="black",
           label=f"balanced compute = {BALANCED_COMPUTE_SECONDS}s * mean_par")
    ax.bar(labels, tx_s, bottom=par_s, color="#dd8452", edgecolor="black",
           label="transmission = moves * 88MB / 900GBps")

    for i, (t_total, t_tx) in enumerate(zip(totals, tx_s)):
        share = 100.0 * t_tx / t_total if t_total else 0.0
        ax.text(i, t_total, f"{t_total:.1f}s\n(tx={share:.0f}%)",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Modeled total time per case (s) — lower is better")
    ax.set_title(
        f"Grader time decomposition per case: '{version.label}'  "
        f"(sum={version.composite_total_time:.1f}s, "
        f"transit_total={version.composite_transmission_time:.1f}s, "
        f"score={version.composite_score:.2f})",
        fontsize=12)
    ax.tick_params(axis="x", rotation=70)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grader_latest_vs_previous(versions: list[GraderVersion],
                                    out_path: Path) -> None:
    """Per-case diff bars: latest mean_par - previous mean_par, and the same
    for transmit. Negative bars are improvements; positive bars are
    regressions. The plot is colour-coded so a regressed case lights up red,
    making "did we just break a case while improving the rest?" obvious.
    """
    if len(versions) < 2:
        return
    latest = versions[-1]
    previous = versions[-2]
    prev_by_key = {_case_key(c): c for c in previous.cases}

    rows: list[tuple[str, float, float]] = []
    for c in latest.cases:
        key = _case_key(c)
        if key not in prev_by_key:
            continue
        p = prev_by_key[key]
        rows.append((
            _case_label(c),
            float(c["mean_par"]) - float(p["mean_par"]),
            float(c["transmit_amount"]) - float(p["transmit_amount"]),
        ))
    if not rows:
        return
    # Sort by magnitude of PAR delta (worst regression first)
    rows.sort(key=lambda r: -r[1])

    labels = [r[0] for r in rows]
    par_delta = [r[1] for r in rows]
    tx_delta = [r[2] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * 0.55), 9))
    par_colors = ["#d62728" if v > 0 else "#2ca02c" for v in par_delta]
    tx_colors = ["#d62728" if v > 0 else "#2ca02c" for v in tx_delta]

    axes[0].bar(labels, par_delta, color=par_colors, edgecolor="black")
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title(
        f"Δ mean PAR: '{latest.label}' minus '{previous.label}' "
        "(green = improvement, red = regression)", fontsize=12)
    axes[0].set_ylabel("Δ mean PAR")
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(par_delta):
        axes[0].text(i, v, f"{v:+.4f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)

    axes[1].bar(labels, tx_delta, color=tx_colors, edgecolor="black")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Δ transit (expert moves)", fontsize=12)
    axes[1].set_ylabel("Δ transmit_amount")
    axes[1].tick_params(axis="x", rotation=70)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(tx_delta):
        axes[1].text(i, v, f"{int(v):+d}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle(
        f"Latest vs previous on grader  "
        f"(score: {previous.composite_score:.2f} -> "
        f"{latest.composite_score:.2f}, "
        f"Δ={latest.composite_score - previous.composite_score:+.2f})",
        fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _print_grader_table(versions: list[GraderVersion]) -> None:
    """Compact terminal table of every grader version we have feedback for."""
    if not versions:
        print("  [feedback] no grader versions found under feedbacks/")
        return
    headers = ["version", "score", "mean_par", "transmit",
               "compute_s", "transit_s", "total_s", "n_cases"]
    rendered: list[dict[str, str]] = []
    for v in versions:
        compute_s = v.composite_total_time - v.composite_transmission_time
        rendered.append({
            "version": v.short_label,
            "score": f"{v.composite_score:.3f}",
            "mean_par": f"{v.composite_mean_par:.4f}",
            "transmit": f"{int(v.composite_transmit)}",
            "compute_s": f"{compute_s:.2f}",
            "transit_s": f"{v.composite_transmission_time:.2f}",
            "total_s": f"{v.composite_total_time:.2f}",
            "n_cases": str(len(v.cases)),
        })
    widths = {h: len(h) for h in headers}
    for row in rendered:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rendered:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


def run_grader_feedback_dashboard(repo_root: Path, output_dir: Path) -> bool:
    """Top-level entry for half (B). Returns True iff at least one plot was
    written. Safe to call when feedbacks/ is missing or empty."""
    versions = discover_grader_versions(repo_root)
    print(f"\n=== Grader-feedback dashboard "
          f"({len(versions)} versions found) ===")
    _print_grader_table(versions)
    if not versions:
        return False

    figure_dir = output_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    latest = versions[-1]
    safe_label = re.sub(r"[^A-Za-z0-9._+-]+", "_", latest.label) or "latest"

    _save_grader_score_evolution(versions,
                                 figure_dir / "Grader_Score_Evolution.png")
    _save_energy_score_evolution(versions,
                                 figure_dir / "Energy_Score_Evolution.png")
    _save_grader_per_case_heatmap(latest,
                                  figure_dir / f"Grader_Per_Case_Heatmap_{safe_label}.png")
    _save_grader_ep_scaling(latest,
                            figure_dir / f"Grader_EP_Scaling_{safe_label}.png")
    _save_grader_time_decomposition(latest,
                                    figure_dir / f"Grader_Time_Decomposition_{safe_label}.png")
    if len(versions) >= 2:
        _save_grader_latest_vs_previous(versions,
                                        figure_dir / "Grader_Latest_vs_Previous.png")

    print(f"  -> grader figures: {figure_dir}/Grader_*.png")
    print(f"  -> latest version on grader: '{latest.label}'  "
          f"(score={latest.composite_score:.3f}, "
          f"mean_par={latest.composite_mean_par:.4f}, "
          f"transit={int(latest.composite_transmit)})")
    return True


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _print_table(rows: list[dict], baseline_method: str) -> None:
    baseline = next((r for r in rows if r["method"] == baseline_method), rows[0])
    baseline_total = baseline["total_time_seconds"]
    headers = [
        "method", "mean_par", "total_transit",
        "transmission_time_s", "total_time_s", "algo_calls",
        "algo_max_s", "algo_mean_s", "score_vs_" + baseline_method,
    ]
    rendered: list[dict[str, str]] = []
    for r in rows:
        algo_max = float(r["algo_call_seconds"].max()) if len(r["algo_call_seconds"]) else 0.0
        algo_mean = float(r["algo_call_seconds"].mean()) if len(r["algo_call_seconds"]) else 0.0
        score = 100.0 * baseline_total / r["total_time_seconds"]
        rendered.append({
            "method": r["method"],
            "mean_par": f"{r['mean_par']:.6f}",
            "total_transit": str(r["total_transmit"]),
            "transmission_time_s": f"{r['transmission_time_seconds']:.6f}",
            "total_time_s": f"{r['total_time_seconds']:.6f}",
            "algo_calls": str(len(r["algo_call_seconds"])),
            "algo_max_s": f"{algo_max:.4f}",
            "algo_mean_s": f"{algo_mean:.4f}",
            "score_vs_" + baseline_method: f"{score:.3f}",
        })
    widths = {h: len(h) for h in headers}
    for row in rendered:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rendered:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


# ---------------------------------------------------------------------------
# Constraint check.
#
# Important framing: only submission.py is being graded. The deepseek and
# default rows are shown as REFERENCE only — they get no PASS/FAIL label.
#
# Two checks we run against submission.py:
#   1. PER-CALL ALGO TIME (verified): max rebalance() runtime should stay
#      <= PER_ITER_BUDGET_S. The simulator converts longer runtimes into
#      ceil(runtime/0.08) wasted iterations before the redeploy can start
#      (see dynamic_lb_simulator.py line 177). Going over is not a hard DQ
#      but it silently hurts the score.
#   2. TOTAL WALL TIME (UNVERIFIED): the brief mentions a "3x baseline"
#      timeout, but no code path in this repo enforces it. We surface it as
#      a sanity bound only — labelled "[unverified]" so we don't pretend it
#      is rule-of-thumb.
#
# The default row has no rebalance() calls so its algo metric is N/A.
# ---------------------------------------------------------------------------
JUDGED_METHOD = "submission"


def _check_constraints(rows: list[dict]) -> list[dict]:
    baseline = next(
        (r for r in rows if r["method"] == CONSTRAINT_BASELINE_METHOD),
        rows[0],
    )
    wall_limit_s = baseline["wall_runtime_seconds"] * DQ_TOTAL_WALL_MULTIPLIER
    # Competition score = 100 * baseline_total_time / total_time. Same formula
    # as dynamic_lb_simulator.run; we surface it here so the constraint readout
    # also tells us "are we actually winning?", not only "are we legal?".
    baseline_total_time = float(baseline["total_time_seconds"])
    checks: list[dict] = []
    for r in rows:
        n_calls = len(r["algo_call_seconds"])
        max_algo_s = float(r["algo_call_seconds"].max()) if n_calls > 0 else 0.0
        algo_na = n_calls == 0
        wall_s = float(r["wall_runtime_seconds"])
        score = 100.0 * baseline_total_time / float(r["total_time_seconds"])

        is_judged = r["method"] == JUDGED_METHOD
        if is_judged and not algo_na:
            algo_pass = bool(max_algo_s <= PER_ITER_BUDGET_S)
            wall_pass = bool(wall_s <= wall_limit_s)
            overall_pass = algo_pass and wall_pass
        else:
            algo_pass = True
            wall_pass = True
            overall_pass = True

        checks.append({
            "method": r["method"],
            "is_judged": is_judged,
            "max_algo_s": max_algo_s,
            "budget_s": PER_ITER_BUDGET_S,
            "algo_pass": algo_pass,
            "algo_na": algo_na,
            "wall_s": wall_s,
            "wall_limit_s": wall_limit_s,
            "wall_pass": wall_pass,
            "score": score,
            "score_baseline": CONSTRAINT_BASELINE_METHOD,
            "overall_pass": overall_pass,
        })
    return checks


def _algo_label(c: dict) -> str:
    """Verdict text for the per-call algo budget cell."""
    if c["algo_na"]:
        return "n/a"
    if not c["is_judged"]:
        return "ref"
    return "PASS" if c["algo_pass"] else "FAIL"


def _wall_label(c: dict) -> str:
    """Verdict text for the wall-time cell (unverified rule)."""
    if not c["is_judged"]:
        return "ref"
    return ("PASS" if c["wall_pass"] else "FAIL") + " [unverified]"


def _verdict_label(c: dict) -> str:
    """Combined verdict, only meaningful for the judged method."""
    if not c["is_judged"]:
        return "(ref)"
    return "PASS" if c["overall_pass"] else "FAIL"


def _print_constraint_block(checks: list[dict]) -> None:
    """Compact PASS/FAIL block, easy to spot in terminal output.
    Only `submission` is judged; deepseek / default are shown as reference.
    Score column = 100 * baseline_total_time / total_time (baseline = deepseek
    by default), so higher is better and `deepseek` itself sits at 100."""
    score_baseline = checks[0]["score_baseline"]
    print(f"\n  Constraint check (judging method: {JUDGED_METHOD})")
    print(f"    per-call budget = {PER_ITER_BUDGET_S}s  (hard simulator rule)")
    print(f"    wall-time bound = {DQ_TOTAL_WALL_MULTIPLIER}x "
          f"{CONSTRAINT_BASELINE_METHOD} wall  [unverified]")
    print(f"    score           = 100 * {score_baseline}_time / total_time "
          "(higher is better)")
    score_header = f"score_vs_{score_baseline}"
    headers = [
        "method", "max_algo_s", "budget_s", "algo", "wall_s",
        "wall_limit_s", "wall", score_header, "verdict",
    ]
    rendered: list[dict[str, str]] = []
    for c in checks:
        rendered.append({
            "method": c["method"],
            "max_algo_s": f"{c['max_algo_s']:.4f}",
            "budget_s": f"{c['budget_s']:.4f}",
            "algo": _algo_label(c),
            "wall_s": f"{c['wall_s']:.3f}",
            "wall_limit_s": f"{c['wall_limit_s']:.3f}",
            "wall": _wall_label(c),
            score_header: f"{c['score']:.2f}",
            "verdict": _verdict_label(c),
        })
    widths = {h: len(h) for h in headers}
    for row in rendered:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print("    " + " | ".join(h.ljust(widths[h]) for h in headers))
    print("    " + "-+-".join("-" * widths[h] for h in headers))
    for row in rendered:
        print("    " + " | ".join(row[h].ljust(widths[h]) for h in headers))


def _save_constraint_plot(checks: list[dict], case_name: str, out_path: Path) -> None:
    """Three side-by-side panels: algo_max vs budget, wall vs limit, score.

    Bar colours:
      - GREEN  : the judged method PASSes the constraint (or wins on score)
      - RED    : the judged method FAILs the constraint (or loses on score)
      - GREY   : reference method (deepseek/default), NOT being judged
    Dashed reference lines mark the per-iter budget, the (unverified) wall
    bound, and the score=100 line."""
    methods = [c["method"] for c in checks]
    algo_vals = [c["max_algo_s"] for c in checks]
    wall_vals = [c["wall_s"] for c in checks]
    score_vals = [c["score"] for c in checks]
    score_baseline = checks[0]["score_baseline"]

    def _bar_color(c: dict, ok: bool) -> str:
        if not c["is_judged"] or c["algo_na"]:
            return "#bbbbbb"  # reference, no verdict
        return "#2ca02c" if ok else "#d62728"

    algo_colors = [_bar_color(c, c["algo_pass"]) for c in checks]
    wall_colors = [_bar_color(c, c["wall_pass"]) for c in checks]
    score_colors = [_bar_color(c, c["score"] >= 100.0) for c in checks]
    budget_s = checks[0]["budget_s"]
    wall_limit_s = checks[0]["wall_limit_s"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].bar(methods, algo_vals, color=algo_colors)
    axes[0].axhline(budget_s, color="black", linestyle="--", linewidth=1.0,
                    label=f"per-iter budget = {budget_s}s")
    axes[0].set_title(f"Max rebalance() runtime per call\n"
                      f"(only `{JUDGED_METHOD}` is judged; others = reference)")
    axes[0].set_ylabel("seconds")
    for i, c in enumerate(checks):
        axes[0].text(i, c["max_algo_s"], f"{c['max_algo_s']:.3f}s\n{_algo_label(c)}",
                     ha="center", va="bottom", fontsize=10)
    axes[0].legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)

    axes[1].bar(methods, wall_vals, color=wall_colors)
    axes[1].axhline(wall_limit_s, color="black", linestyle="--", linewidth=1.0,
                    label=f"wall bound = {DQ_TOTAL_WALL_MULTIPLIER}x "
                          f"{CONSTRAINT_BASELINE_METHOD} = {wall_limit_s:.2f}s  "
                          f"[unverified]")
    axes[1].set_title(f"Total wall-clock runtime\n"
                      f"(only `{JUDGED_METHOD}` is judged; others = reference)")
    axes[1].set_ylabel("seconds")
    for i, c in enumerate(checks):
        axes[1].text(i, c["wall_s"], f"{c['wall_s']:.2f}s\n{_wall_label(c)}",
                     ha="center", va="bottom", fontsize=10)
    axes[1].legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)

    axes[2].bar(methods, score_vals, color=score_colors)
    axes[2].axhline(100.0, color="black", linestyle="--", linewidth=1.0,
                    label=f"{score_baseline} reference = 100")
    axes[2].set_title(f"Composite score vs {score_baseline} (higher is better)\n"
                      f"score = 100 * {score_baseline}_time / total_time")
    axes[2].set_ylabel("score")
    for i, c in enumerate(checks):
        delta = c["score"] - 100.0
        sign = "+" if delta >= 0 else ""
        axes[2].text(i, c["score"],
                     f"{c['score']:.1f}\n({sign}{delta:.1f} vs {score_baseline})",
                     ha="center", va="bottom", fontsize=10)
    axes[2].legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    axes[2].grid(axis="y", linestyle=":", alpha=0.7)

    fig.suptitle(f"Constraint check: {case_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def check_submission(
    repo_root: Path,
    model: str,
    dataset: str,
    ep: int,
    max_iters: int,
    submission_collection_interval: int,
    deepseek_collection_interval: int = 1024,
    method_name: str = "submission",
) -> bool:
    """Fast standalone check: run submission.py (and deepseek for the wall-time
    reference + score baseline) on one case and return True iff both constraints
    PASS. Used by the --check-only CLI flag; no plots / CSVs are written.

    deepseek_collection_interval defaults to 1024 (the simulator's reference
    cadence) so the printed score matches the full-dashboard score for the
    same case. If max_iters is too short for deepseek to even rebalance once,
    we fall back to a smaller interval so the wall-time and score numbers are
    not vacuous; in that case we print a warning so the user knows the score
    is not directly comparable to the default cadence."""
    n_layers, n_experts = MODEL_SHAPES[model]
    hotness = load_trace(repo_root, model, dataset, max_iters)
    deepseek_interval = deepseek_collection_interval
    if max_iters <= deepseek_interval:
        deepseek_interval = max(64, max_iters // 4)
        print(f"  [warn] max_iters={max_iters} <= deepseek interval "
              f"{deepseek_collection_interval}; falling back to "
              f"deepseek interval={deepseek_interval} so the reference "
              f"actually rebalances. Score not directly comparable to "
              f"the {deepseek_collection_interval}-cadence run.")
    deepseek_fn = build_rebalance(ep, ep, "deepseek")
    deepseek_result = run_method(
        method_name="deepseek",
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=deepseek_interval, rebalance_fn=deepseek_fn,
    )
    sub_fn = build_rebalance(ep, ep, "proposed")
    sub_result = run_method(
        method_name=method_name,
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=submission_collection_interval, rebalance_fn=sub_fn,
    )
    checks = _check_constraints([deepseek_result, sub_result])
    case_name = f"{model}_{dataset}_EP{ep}"
    print(f"\n=== Constraint check: {case_name} ===")
    _print_constraint_block(checks)
    sub_check = next(c for c in checks if c["method"] == method_name)
    verdict = "PASS" if sub_check["overall_pass"] else "FAIL"
    delta = sub_check["score"] - 100.0
    sign = "+" if delta >= 0 else ""
    print(f"\n  -> {method_name}: {verdict}  "
          f"(score {sub_check['score']:.2f}, {sign}{delta:.2f} vs "
          f"{sub_check['score_baseline']})")
    return bool(sub_check["overall_pass"])


def _save_csv(rows: list[dict], baseline_method: str, out_path: Path) -> None:
    baseline = next((r for r in rows if r["method"] == baseline_method), rows[0])
    baseline_total = baseline["total_time_seconds"]
    cols = [
        "method", "mean_par", "total_transit",
        "transmission_time_s", "total_time_s", "algo_calls",
        "algo_max_s", "algo_mean_s", "score_vs_" + baseline_method,
    ]
    with out_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            algo_max = float(r["algo_call_seconds"].max()) if len(r["algo_call_seconds"]) else 0.0
            algo_mean = float(r["algo_call_seconds"].mean()) if len(r["algo_call_seconds"]) else 0.0
            score = 100.0 * baseline_total / r["total_time_seconds"]
            f.write(",".join([
                r["method"],
                f"{r['mean_par']:.6f}",
                str(r["total_transmit"]),
                f"{r['transmission_time_seconds']:.6f}",
                f"{r['total_time_seconds']:.6f}",
                str(len(r["algo_call_seconds"])),
                f"{algo_max:.6f}",
                f"{algo_mean:.6f}",
                f"{score:.6f}",
            ]) + "\n")


# ---------------------------------------------------------------------------
# Top-level case runner.
# ---------------------------------------------------------------------------
def run_case(
    repo_root: Path,
    model: str,
    dataset: str,
    ep: int,
    max_iters: int,
    deepseek_collection_interval: int,
    submission_collection_interval: int,
    baseline_method: str,
    output_dir: Path,
    submission_version: Optional[SubmissionVersion] = None,
    extra_versions: Optional[list[SubmissionVersion]] = None,
) -> None:
    """Run default + deepseek + submission for one case and emit all plots.

    submission_version
      The `submission*.py` to run as the "submission" entry. Defaults to the
      most recently modified file, which is the one you most likely want to
      look at while iterating.
    extra_versions
      If non-empty, additionally run each listed `submission_vN.py` and emit
      a Submission_Versions_<case>.png comparing all of them on this case.
      The active submission_version is included automatically; we de-dup.
    """
    n_layers, n_experts = MODEL_SHAPES[model]
    hotness = load_trace(repo_root, model, dataset, max_iters)
    case_name = f"{model}_{dataset}_EP{ep}"
    if submission_version is None:
        submission_version = latest_submission_version(repo_root)
    print(f"\n=== {case_name}  (iters={len(hotness)}, "
          f"deepseek_int={deepseek_collection_interval}, "
          f"submission_int={submission_collection_interval}, "
          f"submission_file={submission_version.path.name}) ===")

    if len(hotness) <= max(deepseek_collection_interval, submission_collection_interval):
        raise ValueError("max_iters must exceed both collection intervals")

    results: list[dict] = []

    # 1) "default" baseline: no redeploy, n_red_expert=0 (matches the
    #    simulator's static placement).
    results.append(run_method(
        method_name="default",
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=0, rebalance_fn=None,
    ))

    # 2) "deepseek" baseline through the official adapter
    #    (eplb_algorithms/deepseek.py).
    deepseek_fn = build_rebalance(ep, ep, "deepseek")
    results.append(run_method(
        method_name="deepseek",
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=deepseek_collection_interval, rebalance_fn=deepseek_fn,
    ))

    # 3) The chosen "submission" entry through its alias.
    submission_fn = build_rebalance(ep, ep, submission_version.algorithm)
    sub_result = run_method(
        method_name="submission",
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=submission_collection_interval,
        rebalance_fn=submission_fn,
    )
    sub_result["display_label"] = submission_version.label
    results.append(sub_result)

    _print_table(results, baseline_method=baseline_method)

    # PASS/FAIL block — quick read-out of competition constraints.
    checks = _check_constraints(results)
    _print_constraint_block(checks)

    figure_dir = output_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_par_plot(results, case_name, figure_dir / f"PAR_over_iterations_{case_name}.png")
    _save_transmit_plot(results, case_name, figure_dir / f"Cumulative_Transmit_{case_name}.png")
    _save_algo_time_plot(
        results, case_name,
        budget_s=PER_ITER_BUDGET_S,
        out_path=figure_dir / f"Algo_Time_Per_Call_{case_name}.png",
    )
    _save_summary_plot(
        results, case_name, baseline_method=baseline_method,
        out_path=figure_dir / f"Score_Summary_{case_name}.png",
    )
    _save_constraint_plot(
        checks, case_name,
        out_path=figure_dir / f"Constraint_Check_{case_name}.png",
    )
    _save_par_distribution_plot(
        results, case_name,
        out_path=figure_dir / f"PAR_Distribution_{case_name}.png",
    )
    _save_time_decomposition_plot(
        results, case_name,
        out_path=figure_dir / f"Time_Decomposition_{case_name}.png",
    )

    # Optional cross-version run on this exact case. Ordered by mtime so
    # bars on the comparison plot read left-to-right oldest -> newest.
    if extra_versions:
        version_results: list[dict] = [results[0], results[1]]  # default + deepseek
        seen_labels: set[str] = set()
        # All versions selected, in mtime order; include the active one too.
        merged_versions = list(extra_versions)
        if submission_version.label not in {v.label for v in merged_versions}:
            merged_versions.append(submission_version)
        merged_versions.sort(key=lambda v: v.mtime)
        for v in merged_versions:
            if v.label in seen_labels:
                continue
            seen_labels.add(v.label)
            print(f"  [versions] running {v.label} ({v.path.name})")
            fn = build_rebalance(ep, ep, v.algorithm)
            r = run_method(
                method_name=v.label,
                hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
                collection_interval=submission_collection_interval,
                rebalance_fn=fn,
            )
            version_results.append(r)
        _save_versions_comparison_plot(
            version_results, case_name, baseline_method=baseline_method,
            out_path=figure_dir / f"Submission_Versions_{case_name}.png",
        )

    csv_dir = output_dir / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(results, baseline_method, csv_dir / f"{case_name}.csv")

    print(f"  -> figures: {figure_dir}/*_{case_name}.png")
    print(f"  -> summary: {csv_dir / (case_name + '.csv')}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _resolve_submission_version(repo_root: Path,
                                override: Optional[str]) -> SubmissionVersion:
    """Pick the submission_*.py to run as the 'submission' entry. CLI override
    accepts either a bare label ('submission_v7') or a path to a .py file."""
    if not override:
        return latest_submission_version(repo_root)
    # path?
    p = Path(override)
    if not p.is_absolute():
        p = (repo_root / override).resolve()
    if p.exists() and p.suffix == ".py":
        algorithm = "proposed" if p.stem == "submission" else p.stem
        return SubmissionVersion(label=p.stem, path=p,
                                 algorithm=algorithm,
                                 mtime=p.stat().st_mtime)
    # label (e.g. 'submission_v7')
    candidate = repo_root / f"{override}.py"
    if candidate.exists():
        algorithm = "proposed" if candidate.stem == "submission" else candidate.stem
        return SubmissionVersion(label=candidate.stem, path=candidate,
                                 algorithm=algorithm,
                                 mtime=candidate.stat().st_mtime)
    raise FileNotFoundError(
        f"--submission-file '{override}' is neither a .py path nor a known "
        f"submission*.py at {repo_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run default / deepseek / submission on the committed "
                    "sample traces and emit a score table, PAR/transit/time "
                    "graphs, a PASS/FAIL constraint check, and (if "
                    "feedbacks/ is populated) a grader-feedback dashboard.",
    )
    p.add_argument("--model", choices=sorted(MODEL_SHAPES), default="Qwen3")
    p.add_argument("--dataset", default="LmSys")
    p.add_argument("--ep", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=1200)
    p.add_argument("--deepseek-interval", type=int, default=1024,
                   help="deepseek collection interval (default 1024, "
                        "baseline-equivalent).")
    p.add_argument("--submission-interval", type=int, default=128,
                   help="submission collection interval (default 128, "
                        "fast cadence).")
    p.add_argument("--baseline", choices=["default", "deepseek", "submission"],
                   default="deepseek",
                   help="Reference method for "
                        "score = 100 * baseline_time / total_time.")
    p.add_argument("--output-dir", default="output",
                   help="Where to write figures/ and summary/ subfolders.")
    p.add_argument("--all-samples", action="store_true",
                   help="Run both committed sample models (DS-R1 and Qwen3).")
    p.add_argument("--check-only", action="store_true",
                   help="Fast PASS/FAIL constraint check for submission.py "
                        "(no PAR / transit / score plots). Process exits 0 "
                        "on PASS and 1 on FAIL for easy use in scripts / "
                        "pre-commit.")
    p.add_argument("--submission-file", default=None,
                   help="Override the file used as the 'submission' entry. "
                        "Accepts a path ('submission_v7.py') or a label "
                        "('submission_v7'). Default = the most-recently "
                        "modified submission*.py at the repo root.")
    p.add_argument("--compare-versions", action="store_true",
                   help="Also run every submission_v*.py side-by-side on "
                        "each case and emit Submission_Versions_<case>.png.")
    p.add_argument("--feedback-only", action="store_true",
                   help="Skip the local-trace simulator and only build the "
                        "grader-feedback dashboard from feedbacks/. Useful "
                        "after you upload a new version.")
    p.add_argument("--no-feedback", action="store_true",
                   help="Skip the grader-feedback dashboard.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    models = sorted(MODEL_SHAPES) if args.all_samples else [args.model]

    if args.check_only:
        all_pass = True
        for model in models:
            ok = check_submission(
                repo_root=repo_root,
                model=model,
                dataset=args.dataset,
                ep=args.ep,
                max_iters=args.max_iters,
                submission_collection_interval=args.submission_interval,
                deepseek_collection_interval=args.deepseek_interval,
            )
            all_pass = all_pass and ok
        print(f"\n=== Overall: {'PASS' if all_pass else 'FAIL'} ===")
        raise SystemExit(0 if all_pass else 1)

    sub_version = _resolve_submission_version(repo_root, args.submission_file)
    print(f"[dashboard] active submission file: {sub_version.path.name}  "
          f"(label={sub_version.label}, alias={sub_version.algorithm}, "
          f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sub_version.mtime))})")
    extra_versions = (discover_submission_versions(repo_root)
                      if args.compare_versions else None)

    if not args.feedback_only:
        for model in models:
            run_case(
                repo_root=repo_root,
                model=model,
                dataset=args.dataset,
                ep=args.ep,
                max_iters=args.max_iters,
                deepseek_collection_interval=args.deepseek_interval,
                submission_collection_interval=args.submission_interval,
                baseline_method=args.baseline,
                output_dir=output_dir,
                submission_version=sub_version,
                extra_versions=extra_versions,
            )

    if not args.no_feedback:
        run_grader_feedback_dashboard(repo_root, output_dir)


if __name__ == "__main__":
    main()
