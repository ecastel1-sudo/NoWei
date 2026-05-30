#!/usr/bin/env python3
"""Monitoring dashboard for the MoE load-balancing submission.

Runs three methods on the committed sample traces using the exact simulator
math from dynamic_lb_simulator.py / quickstart.py:
  - default    : no redeploy, simulator's static placement (no rebalancing).
  - deepseek   : the DeepSeek EPLB baseline from eplb_algorithms/deepseek.py.
  - submission : whatever is in submission.py at the repo root (our entry).

Outputs per case:
  - PAR_over_iterations_<case>.png         : mean PAR per iteration
  - Cumulative_Transmit_<case>.png         : cumulative #expert moves
  - Algo_Time_Per_Call_<case>.png          : wallclock per rebalance() call
  - Score_Summary_<case>.png               : final mean PAR / transmit / score
  - Constraint_Check_<case>.png            : PASS/FAIL panels for the two
                                             competition constraints

Why this lives outside quickstart.py:
  - Keeps quickstart.py untouched (per the user's "avoid_overwriting" rule).
  - Tracks per-iteration data and per-call algo time, which quickstart.py
    does not need to do for its smoke-test purpose.

Usage:
  python dashboard.py                       # default: Qwen3 / LmSys / EP32
  python dashboard.py --model DS-R1 --ep 64
  python dashboard.py --all-samples         # both committed traces
  python dashboard.py --check-only          # fast PASS/FAIL only, exit 0/1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend; we only save PNGs
import matplotlib.pyplot as plt

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

METHOD_COLORS = {
    "default": "#888888",
    "deepseek": "#1f77b4",
    "submission": "#d62728",
}

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


def load_trace(repo_root: Path, model: str, dataset: str, max_iters: int) -> np.ndarray:
    path = repo_root / "trace" / model / f"{dataset}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing sample trace: {path}")
    trace = np.load(path, mmap_mode="r")[:max_iters]
    trace = np.array(trace, copy=True)
    trace[trace == 0] = 1
    return trace


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
def _save_par_plot(results: list[dict], case_name: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(
            np.arange(len(r["per_iter_par"])),
            r["per_iter_par"],
            label=r["method"],
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
        plt.plot(
            np.arange(len(cumulative)),
            cumulative,
            label=f"{r['method']} (total={int(cumulative[-1])})" if len(cumulative) else r["method"],
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
) -> None:
    n_layers, n_experts = MODEL_SHAPES[model]
    hotness = load_trace(repo_root, model, dataset, max_iters)
    case_name = f"{model}_{dataset}_EP{ep}"
    print(f"\n=== {case_name}  (iters={len(hotness)}, "
          f"deepseek_int={deepseek_collection_interval}, "
          f"submission_int={submission_collection_interval}) ===")

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

    # 3) Our "submission" via the 'proposed' adapter (loads submission.py at
    #    the repo root).
    submission_fn = build_rebalance(ep, ep, "proposed")
    results.append(run_method(
        method_name="submission",
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=submission_collection_interval,
        rebalance_fn=submission_fn,
    ))

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

    csv_dir = output_dir / "summary"
    csv_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(results, baseline_method, csv_dir / f"{case_name}.csv")

    print(f"  -> figures: {figure_dir}/*_{case_name}.png")
    print(f"  -> summary: {csv_dir / (case_name + '.csv')}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run default / deepseek / submission on the committed "
                    "sample traces and emit a score table, PAR/transit/time "
                    "graphs, and a PASS/FAIL constraint check."
    )
    p.add_argument("--model", choices=sorted(MODEL_SHAPES), default="Qwen3")
    p.add_argument("--dataset", default="LmSys")
    p.add_argument("--ep", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=1200)
    p.add_argument("--deepseek-interval", type=int, default=1024,
                   help="deepseek collection interval (default 1024, "
                        "baseline-equivalent).")
    p.add_argument("--submission-interval", type=int, default=128,
                   help="submission.py collection interval (default 128, "
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
        )


if __name__ == "__main__":
    main()
