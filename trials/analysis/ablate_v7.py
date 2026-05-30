#!/usr/bin/env python3
"""Ablate the three v7 knobs to find which combination wins on LmSys.

We import submission_v7 and toggle its module constants, run the grader
loop on the local traces, and tabulate per-case scores.

Variants:
  v7_full   : drift + tail + gain-order (current)
  v7_no_tail: drift + gain-order        (tail gate off)
  v7_drift  : drift only                (v6 ordering + window gate)
  v7_gain   : gain-order only           (v6 recency, no tail)
  v7_none   : everything off            (equivalent to v6)

This is a sanity layer over compare_v6_v7.py to pick the safest v7
config before sending anything to the grader.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np

import submission_v7

from grader_sim import (
    MODEL_SHAPES,
    COLLECTION_INTERVAL,
    enumerate_grader_grid,
    split_available_cases,
    load_trace,
    init_deploy_table,
    cal_par_per_iter,
    compute_redeploy_cost,
    modeled_runtime_seconds,
)


def run_one_with_v7(hotness: np.ndarray, ep: int,
                    n_layers: int, n_experts: int,
                    drift: bool, tail: bool, gain: bool) -> dict:
    """Reload submission_v7 with toggled knobs, then run the grader loop."""
    importlib.reload(submission_v7)
    submission_v7.DRIFT_ADAPT_ENABLED = drift
    submission_v7.TAIL_GATE_ENABLED = tail
    submission_v7.GAIN_ORDER_ENABLED = gain
    submission_v7._reset()

    n_iter = len(hotness)
    cur = np.array([init_deploy_table(ep, n_experts, ep, default_layout=False)
                    for _ in range(n_layers)])
    nxt = np.zeros_like(cur)
    pars = np.zeros(n_iter, dtype=np.float64)
    transmit = 0
    queue: list[int] = []
    ready = True
    finish_iter = 0
    call_times: list[float] = []
    for i in range(1, n_iter + 1):
        pars[i - 1] = float(cal_par_per_iter(hotness[i - 1], cur).mean())
        if not ready and queue:
            L = queue.pop(0)
            transmit += compute_redeploy_cost(cur[L], nxt[L])
            cur[L] = nxt[L]
        if not ready and not queue:
            ready = True
            finish_iter = i
        if i == finish_iter + COLLECTION_INTERVAL["submission"] + 1 and ready:
            train = hotness[i - COLLECTION_INTERVAL["submission"]:i]
            ct = time.time()
            change, prio, deploy_tbl, _ = submission_v7.rebalance(train, ep, ep)
            call_times.append(time.time() - ct)
            if change:
                sel = np.asarray(prio, dtype=np.int64)
                queue = sel.tolist()
                ready = False
                nxt[sel] = deploy_tbl[sel]
    mean_par = float(pars.mean())
    return {
        "mean_par": mean_par,
        "transmit": int(transmit),
        "total_time_s": modeled_runtime_seconds(mean_par, transmit),
        "call_max_ms": max(call_times) * 1000 if call_times else 0.0,
    }


VARIANTS = [
    ("v7_none",     False, False, False),
    ("v7_drift",    True,  False, False),
    ("v7_gain",     False, False, True),
    ("v7_no_tail",  True,  False, True),
    ("v7_no_drift", False, True,  True),
    ("v7_full",     True,  True,  True),
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    have, _ = split_available_cases(repo_root, enumerate_grader_grid())
    if not have:
        print("No local traces; nothing to ablate.")
        return

    # Per-case totals per variant, plus per-case PAR + transit for diff print.
    totals = {name: {"par": [], "tx": 0, "t": 0.0} for name, *_ in VARIANTS}
    per_case: list[dict] = []
    for ds, md, ep in have:
        n_layers, n_experts = MODEL_SHAPES[md]
        hotness = load_trace(repo_root, md, ds)
        case_label = f"{ds}/{md}/EP{ep}"
        print(f"\n[{case_label}] iters={len(hotness)}")
        case_row = {"case": case_label}
        for name, d, t, g in VARIANTS:
            r = run_one_with_v7(hotness, ep, n_layers, n_experts, d, t, g)
            print(f"  {name:12s}: PAR={r['mean_par']:.4f} "
                  f"tx={r['transmit']:>7d} "
                  f"total_s={r['total_time_s']:.2f}")
            totals[name]["par"].append(r["mean_par"])
            totals[name]["tx"] += r["transmit"]
            totals[name]["t"] += r["total_time_s"]
            case_row[name] = r
        per_case.append(case_row)

    print("\n=== AGGREGATE OVER LOCAL LMSYS CASES ===")
    # baseline is v7_none == v6 behaviour
    base_t = totals["v7_none"]["t"]
    print(f"{'variant':12s}  {'mean_par':>9s}  {'transmit':>9s}  "
          f"{'total_s':>9s}  {'rel_score':>9s}")
    for name, *_ in VARIANTS:
        a = totals[name]
        mean_par = float(np.mean(a["par"]))
        rel = base_t / a["t"] * 100.0
        print(f"{name:12s}  {mean_par:>9.5f}  {a['tx']:>9d}  "
              f"{a['t']:>9.2f}  {rel:>9.3f}")


if __name__ == "__main__":
    main()
