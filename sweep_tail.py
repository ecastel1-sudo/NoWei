#!/usr/bin/env python3
"""Sweep TAIL_FRACTION on v7 to find the noise/freshness sweet spot."""

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


def run_one(hotness, ep, n_layers, n_experts, tail_frac):
    importlib.reload(submission_v7)
    submission_v7.DRIFT_ADAPT_ENABLED = False
    submission_v7.TAIL_GATE_ENABLED = tail_frac is not None
    if tail_frac is not None:
        submission_v7.TAIL_FRACTION = tail_frac
    submission_v7.GAIN_ORDER_ENABLED = True
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
            change, prio, deploy_tbl, _ = submission_v7.rebalance(train, ep, ep)
            if change:
                sel = np.asarray(prio, dtype=np.int64)
                queue = sel.tolist()
                ready = False
                nxt[sel] = deploy_tbl[sel]
    mean_par = float(pars.mean())
    return mean_par, int(transmit), modeled_runtime_seconds(mean_par, transmit)


def main():
    repo_root = Path(__file__).resolve().parent
    have, _ = split_available_cases(repo_root, enumerate_grader_grid())
    SWEEP = [None, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]
    labels = {None: "no-tail", 0.125: "16/128", 0.25: "32/128",
              0.375: "48/128", 0.5: "64/128", 0.75: "96/128", 1.0: "128/128"}
    totals = {f: {"par": [], "tx": 0, "t": 0.0} for f in SWEEP}
    for ds, md, ep in have:
        n_layers, n_experts = MODEL_SHAPES[md]
        hotness = load_trace(repo_root, md, ds)
        print(f"\n[{ds}/{md}/EP{ep}]")
        for f in SWEEP:
            par, tx, t = run_one(hotness, ep, n_layers, n_experts, f)
            totals[f]["par"].append(par)
            totals[f]["tx"] += tx
            totals[f]["t"] += t
            print(f"  tail={labels[f]:9s}: PAR={par:.4f} tx={tx:>7d} t={t:.2f}")

    print("\n=== AGGREGATE (over local LmSys) ===")
    base_t = totals[None]["t"]
    print(f"{'tail':10s}  {'mean_par':>9s}  {'transmit':>9s}  {'total_s':>9s}  {'rel':>8s}")
    for f in SWEEP:
        a = totals[f]
        mp = float(np.mean(a["par"]))
        rel = base_t / a["t"] * 100.0
        print(f"{labels[f]:10s}  {mp:>9.5f}  {a['tx']:>9d}  {a['t']:>9.2f}  {rel:>8.3f}")


if __name__ == "__main__":
    main()
