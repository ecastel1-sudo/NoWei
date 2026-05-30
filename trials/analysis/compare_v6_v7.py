#!/usr/bin/env python3
"""Local A/B sanity check between v6 and v7 on whatever traces are committed.

This is NOT the grader -- we only have LmSys traces locally, so the
numbers here say "did v7 regress on stationary traces?". If v6 and v7
report nearly identical PAR + transit on LmSys, the v7 changes are
working as designed (no-op on stationary, only kicks in on drift).

Mix DS-R1 is where v7 is expected to actually help; that needs the
official grader.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from grader_sim import (
    MODEL_SHAPES,
    COLLECTION_INTERVAL,
    enumerate_grader_grid,
    split_available_cases,
    load_trace,
    init_deploy_table,
    cal_par_per_iter,
    compute_redeploy_cost,
    transmission_time_seconds,
    modeled_runtime_seconds,
)
from eplb_algorithms import rebalance as build_rebalance


def run_one(method_alias: str, hotness: np.ndarray, ep: int,
            n_layers: int, n_experts: int) -> dict:
    rebalance_fn = build_rebalance(ep, ep, method_alias)
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
    t_start = time.time()
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
            change, prio, deploy_tbl, _ = rebalance_fn(train)
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
        "wall_s": time.time() - t_start,
        "calls": len(call_times),
        "call_max_s": max(call_times) if call_times else 0.0,
        "call_mean_s": float(np.mean(call_times)) if call_times else 0.0,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    have, _missing = split_available_cases(repo_root, enumerate_grader_grid())
    if not have:
        print("No local traces; nothing to compare.")
        return

    methods = [("v6", "submission_v6"),
               ("v7", "submission_v7"),
               ("v9", "submission_v9")]
    rows: list[dict] = []
    print(f"Comparing {len(methods)} methods on {len(have)} local case(s).")
    for ds, md, ep in have:
        n_layers, n_experts = MODEL_SHAPES[md]
        hotness = load_trace(repo_root, md, ds)
        print(f"\n[{ds}/{md}/EP{ep}] iters={len(hotness)}")
        case_row = {"case": f"{ds}/{md}/EP{ep}"}
        for label, alias in methods:
            r = run_one(alias, hotness, ep, n_layers, n_experts)
            print(f"  {label}: PAR={r['mean_par']:.4f} "
                  f"tx={r['transmit']:>7d} "
                  f"total_s={r['total_time_s']:.2f} "
                  f"wall={r['wall_s']:.1f}s "
                  f"call_max={r['call_max_s']*1000:.1f}ms "
                  f"call_mean={r['call_mean_s']*1000:.1f}ms")
            case_row[f"{label}_par"] = r["mean_par"]
            case_row[f"{label}_tx"] = r["transmit"]
            case_row[f"{label}_t"] = r["total_time_s"]
            case_row[f"{label}_call_max_ms"] = r["call_max_s"] * 1000
        rows.append(case_row)

    print("\n=== SUMMARY (vs v6 baseline) ===")
    labels = [label for label, _ in methods]
    tots = {lbl: sum(r[f"{lbl}_t"] for r in rows) for lbl in labels}
    txs = {lbl: sum(r[f"{lbl}_tx"] for r in rows) for lbl in labels}
    pars = {lbl: float(np.mean([r[f"{lbl}_par"] for r in rows])) for lbl in labels}
    base_lbl = labels[0]
    print(f"{'variant':>6s}  {'mean_par':>10s}  {'transmit':>10s}  "
          f"{'total_s':>10s}  {'rel_score':>10s}")
    for lbl in labels:
        rel = tots[base_lbl] / tots[lbl] * 100.0
        print(f"{lbl:>6s}  {pars[lbl]:>10.5f}  {txs[lbl]:>10d}  "
              f"{tots[lbl]:>10.2f}  {rel:>10.3f}")


if __name__ == "__main__":
    main()
