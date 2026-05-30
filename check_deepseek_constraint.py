#!/usr/bin/env python3
"""Check whether the reference DeepSeek EPLB respects the per-call budget.

The competition rule (see ``dashboard.PER_ITER_BUDGET_S``) is that a single
``rebalance()`` call must finish within 0.08 s, otherwise the simulator
charges ``ceil(runtime / 0.08)`` wasted iterations.

The dashboard already reports DeepSeek's max ``rebalance()`` runtime in the
"Constraint check" figure (left panel of
``output/figure/Constraint_Check_*.png``). On our machine DeepSeek lands
well above 0.08 s, which strongly suggests it was timed on a different
(faster) machine for the official baseline and is therefore not directly
comparable on this hardware.

This script reproduces that timing in isolation so it's easy to confirm
or contradict the "different setup" hypothesis without rebuilding the
full dashboard. It is purposely self-contained: same trace loader, same
rebalance wrapper, same per-call ``time.time()`` measurement as
``dashboard.run_method``.

Usage:
    python check_deepseek_constraint.py
    python check_deepseek_constraint.py --model Qwen3 --dataset LmSys --ep 32
    python check_deepseek_constraint.py --collection-interval 512 --repeat 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from eplb_algorithms import rebalance as build_rebalance


# Mirror dashboard.py / quickstart.py so we stay aligned with the grader.
MODEL_SHAPES = {
    "DS-R1": (58, 256),
    "Qwen3": (94, 128),
}

# Hard simulator rule: one iteration = 0.08 s, anything above this gets
# charged as wasted iterations in dynamic_lb_simulator.py.
PER_ITER_BUDGET_S = 0.08

# How far over budget we treat as "not the same setup". 2x = silently
# painful but plausibly slow Python; >=5x = realistically a different
# machine / different runtime than the one the brief was tuned on.
DIFFERENT_SETUP_MULTIPLIER = 5.0


def load_trace(repo_root: Path, model: str, dataset: str, max_iters: int) -> np.ndarray:
    """Load a sample trace the same way quickstart / dashboard do."""
    path = repo_root / "trace" / model / f"{dataset}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing sample trace: {path}")
    trace = np.load(path, mmap_mode="r")[:max_iters]
    trace = np.array(trace, copy=True)
    # Replace exact zeros so calculate_par doesn't divide by zero downstream.
    trace[trace == 0] = 1
    return trace


def time_deepseek_calls(
    hotness: np.ndarray,
    ep: int,
    collection_interval: int,
    repeat: int,
) -> list[float]:
    """Time every DeepSeek ``rebalance()`` call over the trace.

    We slide a fixed-size window (matching the simulator's
    ``collection_interval``) across the trace and call the DeepSeek
    rebalancer on each window. We *do not* simulate the redeploy queue
    here; we only care about per-call wall time, which is what the
    constraint is about.

    ``repeat`` runs the whole loop multiple times so the reported
    statistics are not dominated by a single noisy outlier.
    """
    rebalance_fn = build_rebalance(ep, ep, "deepseek")
    n_iterations = len(hotness)
    if n_iterations <= collection_interval:
        raise ValueError(
            f"trace has {n_iterations} iters but collection_interval="
            f"{collection_interval}; pick a smaller window."
        )
    per_call_seconds: list[float] = []
    for _ in range(repeat):
        # Walk the trace in non-overlapping windows, mirroring the
        # simulator's "trigger once per collection_interval" cadence.
        i = collection_interval
        while i <= n_iterations:
            window = hotness[i - collection_interval:i]
            t0 = time.time()
            rebalance_fn(window)
            per_call_seconds.append(time.time() - t0)
            i += collection_interval
    return per_call_seconds


def classify(max_s: float, budget_s: float) -> tuple[str, str]:
    """Return (status, human-readable explanation)."""
    if max_s <= budget_s:
        return (
            "PASS",
            f"DeepSeek respects the {budget_s:.3f}s per-call budget on this "
            "machine - comparable setup.",
        )
    over_ratio = max_s / budget_s
    if over_ratio >= DIFFERENT_SETUP_MULTIPLIER:
        return (
            "DIFFERENT-SETUP",
            f"DeepSeek is {over_ratio:.1f}x over the {budget_s:.3f}s budget. "
            "The official baseline was almost certainly timed on a different "
            "(faster) machine - numbers are not directly comparable to ours.",
        )
    return (
        "OVER-BUDGET",
        f"DeepSeek is {over_ratio:.1f}x over the {budget_s:.3f}s budget but "
        "within the same order of magnitude - possibly comparable, possibly "
        "just slow Python on this machine.",
    )


def print_report(
    model: str,
    dataset: str,
    ep: int,
    collection_interval: int,
    per_call_seconds: list[float],
) -> int:
    n_calls = len(per_call_seconds)
    max_s = max(per_call_seconds)
    mean_s = statistics.fmean(per_call_seconds)
    median_s = statistics.median(per_call_seconds)
    over = [s for s in per_call_seconds if s > PER_ITER_BUDGET_S]
    status, explanation = classify(max_s, PER_ITER_BUDGET_S)

    print(f"\n=== DeepSeek constraint check: "
          f"{model}_{dataset}_EP{ep} (interval={collection_interval}) ===")
    print(f"  calls timed                : {n_calls}")
    print(f"  per-call budget            : {PER_ITER_BUDGET_S:.3f} s")
    print(f"  max  rebalance() runtime   : {max_s:.4f} s "
          f"({max_s / PER_ITER_BUDGET_S:.2f}x budget)")
    print(f"  mean rebalance() runtime   : {mean_s:.4f} s")
    print(f"  median rebalance() runtime : {median_s:.4f} s")
    print(f"  calls over budget          : {len(over)} / {n_calls} "
          f"({100.0 * len(over) / n_calls:.1f}%)")
    print(f"\n  verdict: {status}")
    print(f"  -> {explanation}")
    # Exit code helps when this is wired into CI / make targets.
    return 0 if status == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time DeepSeek's rebalance() per call and check it against the "
            "0.08s per-iter budget. Confirms whether the official baseline "
            "could plausibly be running on the same setup as ours."
        ),
    )
    parser.add_argument("--model", choices=sorted(MODEL_SHAPES), default="Qwen3")
    parser.add_argument("--dataset", default="LmSys")
    parser.add_argument("--ep", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=4096,
                        help="Max trace iterations to load (default: 4096).")
    parser.add_argument("--collection-interval", type=int, default=1024,
                        help="Window size fed to rebalance(), like the "
                             "simulator (default: 1024).")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat the whole sweep N times to smooth out "
                             "single-call noise (default: 1).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    hotness = load_trace(repo_root, args.model, args.dataset, args.max_iters)
    per_call = time_deepseek_calls(
        hotness=hotness,
        ep=args.ep,
        collection_interval=args.collection_interval,
        repeat=args.repeat,
    )
    return print_report(
        model=args.model,
        dataset=args.dataset,
        ep=args.ep,
        collection_interval=args.collection_interval,
        per_call_seconds=per_call,
    )


if __name__ == "__main__":
    raise SystemExit(main())
