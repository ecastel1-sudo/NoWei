#!/usr/bin/env python3
"""Competitive-ratio report for the current submission.

WHAT WE COMPUTE
===============
For every (model, dataset, ep) case whose trace file is on disk we run:

  - SUBMISSION     online,   cadence 128, past-window  -> grader_sim closed loop
  - DS-EPLB        online,   cadence 1024, past-window -> grader_sim closed loop
  - OPT-CV-DS      OFFLINE,  cadence 128, FUTURE window, vanilla DS-EPLB
                   Closed-loop simulator with DS-EPLB packing; the only
                   difference vs DS-EPLB online is that each rebalance
                   call sees hotness[i:i+W] instead of hotness[i-W:i].
                   "How well DS-EPLB itself would do with no uncertainty."
                   Universal reference, independent of which submission
                   we are scoring.
  - OPT-CV-SUB     OFFLINE,  cadence 128, FUTURE window, *same* submission
                   Same submission alias is invoked, but at each rebalance
                   call it receives hotness[i:i+W] (the future) instead
                   of hotness[i-W:i] (the past). This is the
                   "perfect-prediction twin" of the algorithm we score:
                   strips uncertainty out of the picture while keeping
                   the same anchoring / gating / packing logic.
                   Typical interpretation: total_time(submission) /
                   total_time(OPT-CV-SUB) ~ how much our prediction
                   error is costing us specifically.
  - OPT-LB         OFFLINE,  per-iteration optimal placement, FREE movement
                   For every (iter, layer) we run DS-EPLB packing on that
                   iteration's own hotness, then compute PAR. Transmit = 0.
                   This is a STRICT lower bound on any algorithm's cost:
                   no causal algorithm can produce lower PAR than the
                   per-iteration optimum, and transmit cannot be negative.
                   PAR_LB * 60 is the absolute floor on total_time.

COMPETITIVE RATIO
=================
For each case (and the composite over cases) we report

    CR_LB     = submission_total_time / OPT-LB_total_time
                (strict floor; ceiling on achievable grader score)
    CR_CV_DS  = submission_total_time / OPT-CV-DS_total_time
                (algorithm-family reference; can be < 1 if submission's
                 anchoring/gating beats vanilla DS-EPLB-with-future)
    CR_CV_SUB = submission_total_time / OPT-CV-SUB_total_time
                (submission-specific; isolates prediction error)

CR_LB >= 1 always. CR_CV_SUB is typically >= 1.

SPEED
=====
The batched DS-EPLB packing in `eplb_algorithms.deepseek` has a python
inner loop in `balanced_packing` -> ~120s/256 iters on Qwen3 EP32. That
is far too slow for OPT-LB which needs per-iter packing. We ship a
numpy-vectorised LPT packer that handles `(X, n_log)` in one Python loop
over items (independent of X). Replication is vectorised in torch
already, so we reuse the upstream implementation there.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Reuse simulator constants + per-case runner.
import grader_sim as gs
from eplb_algorithms import rebalance as build_rebalance
from eplb_algorithms.deepseek import replicate_experts as ds_replicate_experts


# ---------------------------------------------------------------------------
# Vectorised batched LPT packer (numpy). Identical algorithm to the
# DeepSeek reference; we only vectorise the outer-over-rows axis so the
# Python loop is over items, not over rows. Used by OPT-LB.
# ---------------------------------------------------------------------------
def vec_balanced_packing(weight: np.ndarray, num_packs: int) -> np.ndarray:
    """LPT pack for each independent row of `weight`.

    Parameters
    ----------
    weight : (X, n) numpy array of non-negative floats
    num_packs : int, must divide n

    Returns
    -------
    pack_index    : (X, n) int64, pack id for each item
    rank_in_pack  : (X, n) int64, position within its pack (0..items_per_pack-1)
    """
    X, n = weight.shape
    assert n % num_packs == 0, "n must be a multiple of num_packs"
    items_per_pack = n // num_packs

    order = np.argsort(-weight, axis=1, kind="stable")  # (X, n) sorted desc
    pack_load = np.zeros((X, num_packs), dtype=np.float64)
    pack_count = np.zeros((X, num_packs), dtype=np.int64)
    pack_index = np.empty((X, n), dtype=np.int64)
    rank_in_pack = np.empty((X, n), dtype=np.int64)
    rows = np.arange(X)
    INF = np.float64(np.inf)

    for k in range(n):
        item = order[:, k]                                  # (X,)
        w = weight[rows, item]                              # (X,)
        masked = np.where(pack_count < items_per_pack, pack_load, INF)
        chosen = np.argmin(masked, axis=1)                  # (X,)
        pack_index[rows, item] = chosen
        rank_in_pack[rows, item] = pack_count[rows, chosen]
        pack_load[rows, chosen] += w
        pack_count[rows, chosen] += 1

    return pack_index, rank_in_pack


def vec_pack_to_deploy(weight_per_phys: np.ndarray, phy2log: np.ndarray,
                       n_packs: int) -> np.ndarray:
    """Run the batched LPT packer on the per-replica weights and return
    a deployment table reshaped to (X, n_packs, items_per_pack)."""
    X, n_phys = weight_per_phys.shape
    items_per_pack = n_phys // n_packs
    pack_idx, rank = vec_balanced_packing(weight_per_phys, n_packs)
    deploy = np.empty((X, n_packs, items_per_pack), dtype=np.int64)
    rows = np.repeat(np.arange(X), n_phys)
    deploy[rows, pack_idx.reshape(-1), rank.reshape(-1)] = phy2log.reshape(-1)
    return deploy


# ---------------------------------------------------------------------------
# OPT-LB: per-(iter, layer) optimal PAR, free movement (strict lower bound).
# ---------------------------------------------------------------------------
def compute_opt_lb(
    hotness: np.ndarray,
    n_device: int,
    n_red_expert: int,
    chunk_iters: int = 256,
    verbose: bool = True,
) -> dict:
    T, L, E = hotness.shape
    n_phys = E + n_red_expert
    exp_per_dev = n_phys // n_device
    per_iter_par = np.empty(T, dtype=np.float64)

    for s in range(0, T, chunk_iters):
        e = min(T, s + chunk_iters)
        chunk = hotness[s:e]
        c = chunk.shape[0]
        flat_h = chunk.reshape(c * L, E).astype(np.float64)

        # 1) Replication: vectorised in torch (uses the upstream impl).
        weight_t = torch.from_numpy(flat_h.astype(np.float32))
        phy2log_t, _, _ = ds_replicate_experts(weight_t, n_phys)
        phy2log = phy2log_t.numpy()                       # (c*L, n_phys)

        # 2) Per-replica weight = h_e / replica_count_e.
        rows = np.arange(c * L)[:, None]
        cut = np.zeros_like(flat_h, dtype=np.int64)
        np.add.at(cut, (rows, phy2log), 1)
        cut = np.maximum(cut, 1)
        replica_weight = flat_h / cut                     # (c*L, E)
        weight_per_phys = replica_weight[rows, phy2log]   # (c*L, n_phys)

        # 3) Vectorised LPT pack -> deployment (c*L, n_dev, exp_per_dev).
        deploy = vec_pack_to_deploy(weight_per_phys, phy2log, n_device)

        # 4) PAR per (iter, layer) on the SAME hotness used for packing
        #    (that is the definition of the per-iter optimum).
        cut2 = np.zeros_like(flat_h, dtype=np.int64)
        np.add.at(cut2, (rows, deploy.reshape(c * L, -1)), 1)
        cut2 = np.maximum(cut2, 1)
        replica_weight2 = flat_h / cut2
        per_replica_load = replica_weight2[rows, deploy.reshape(c * L, -1)]
        loads = per_replica_load.reshape(c * L, n_device, exp_per_dev).sum(-1)
        mean = np.maximum(loads.mean(axis=1), 1e-12)
        par_pair = loads.max(axis=1) / mean               # (c*L,)
        per_iter_par[s:e] = par_pair.reshape(c, L).mean(axis=1)

        if verbose:
            print(f"    opt_lb chunk {s:6d}..{e:6d}/{T:6d}  "
                  f"running mean PAR = {per_iter_par[:e].mean():.4f}",
                  flush=True)

    mean_par = float(per_iter_par.mean())
    return {
        "method": "opt_lb",
        "mean_par": mean_par,
        "total_transmit": 0,
        "transmission_time_seconds": 0.0,
        "total_time_seconds": gs.BALANCED_COMPUTE_SECONDS * mean_par,
        "per_iter_par": per_iter_par,
    }


# ---------------------------------------------------------------------------
# OPT-CV: clairvoyant DS-EPLB. Same closed-loop simulator as the grader,
# same cadence as the submission, but the rebalance window is the FUTURE
# instead of the past. This is the "ideal offline" achievable under the
# simulator's redeploy rules.
# ---------------------------------------------------------------------------
def _ds_pack_future_window(window: np.ndarray, n_device: int,
                            n_red_expert: int) -> np.ndarray:
    """Run vanilla DS-EPLB packing on the SUM of `window` hotness."""
    L, E = window.shape[1], window.shape[2]
    n_phys = E + n_red_expert
    weight_t = torch.from_numpy(window.sum(axis=0).astype(np.float32))

    # Replication (vectorised over layers in torch).
    phy2log_t, _, _ = ds_replicate_experts(weight_t, n_phys)
    phy2log = phy2log_t.numpy()                           # (L, n_phys)

    # Per-replica weights.
    flat_h = window.sum(axis=0).astype(np.float64)        # (L, E)
    rows = np.arange(L)[:, None]
    cut = np.zeros_like(flat_h, dtype=np.int64)
    np.add.at(cut, (rows, phy2log), 1)
    cut = np.maximum(cut, 1)
    weight_per_phys = (flat_h / cut)[rows, phy2log]       # (L, n_phys)

    return vec_pack_to_deploy(weight_per_phys, phy2log, n_device)


# ---------------------------------------------------------------------------
# OPT-CV-SUB: same submission alias, but each rebalance call receives the
# FUTURE window instead of the past. Strips uncertainty out of the picture
# while keeping the submission's own anchoring / gating / packing.
# ---------------------------------------------------------------------------
def compute_opt_cv_sub(
    hotness: np.ndarray,
    ep: int,
    n_layers: int,
    n_experts: int,
    collection_interval: int,
    submission_alias: str,
    verbose: bool = True,
) -> dict:
    """Mirror of grader_sim.run_case_one_method, but with future-window
    instead of past-window rebalance calls."""
    T = len(hotness)
    n_phys = n_experts + ep
    exp_per_dev = n_phys // ep

    rebalance_fn = build_rebalance(ep, ep, submission_alias)

    per_iter_par = np.zeros(T, dtype=np.float64)
    cur_deploy = np.array([
        gs.init_deploy_table(ep, n_experts, ep, default_layout=False)
        for _ in range(n_layers)
    ])
    next_deploy = np.zeros_like(cur_deploy)
    cur_priority: list[int] = []
    expert_ready = True
    redeploy_finish_iter = 0
    total_transmit = 0
    algo_call_seconds: list[float] = []

    for i in range(1, T + 1):
        cur_hotness = hotness[i - 1]
        per_iter_par[i - 1] = gs.cal_par_per_iter(cur_hotness, cur_deploy).mean()

        if not expert_ready and cur_priority:
            layer = cur_priority.pop(0)
            total_transmit += gs.compute_redeploy_cost(
                cur_deploy[layer], next_deploy[layer]
            )
            cur_deploy[layer] = next_deploy[layer]

        if not cur_priority and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + collection_interval + 1 and expert_ready:
            # CLAIRVOYANT: pass the FUTURE window to the submission.
            end = min(T, (i - 1) + collection_interval)
            future_window = hotness[i - 1: end]
            if future_window.shape[0] >= 1:
                call_start = time.time()
                change, layers_priority, deployment_table, _ = rebalance_fn(
                    future_window
                )
                algo_call_seconds.append(time.time() - call_start)
                if change:
                    selected = np.asarray(layers_priority, dtype=np.int64)
                    cur_priority = selected.tolist()
                    expert_ready = False
                    next_deploy[selected] = deployment_table[selected]

        if verbose and i % 1024 == 0:
            print(f"    opt_cv_sub iter {i:6d}/{T:6d}  "
                  f"running mean PAR = {per_iter_par[:i].mean():.4f}  "
                  f"transmit = {total_transmit}",
                  flush=True)

    mean_par = float(per_iter_par.mean())
    return {
        "method": "opt_cv_sub",
        "mean_par": mean_par,
        "total_transmit": int(total_transmit),
        "transmission_time_seconds": gs.transmission_time_seconds(total_transmit),
        "total_time_seconds": gs.modeled_runtime_seconds(mean_par, total_transmit),
        "rebalance_calls": len(algo_call_seconds),
    }


def compute_opt_cv(
    hotness: np.ndarray,
    n_device: int,
    n_red_expert: int,
    collection_interval: int,
    verbose: bool = True,
) -> dict:
    T, L, E = hotness.shape
    n_phys = E + n_red_expert
    exp_per_dev = n_phys // n_device

    per_iter_par = np.zeros(T, dtype=np.float64)
    cur_deploy = np.array([
        gs.init_deploy_table(n_device, E, n_red_expert, default_layout=False)
        for _ in range(L)
    ])
    next_deploy = np.zeros_like(cur_deploy)
    cur_priority: list[int] = []
    expert_ready = True
    redeploy_finish_iter = 0
    total_transmit = 0
    n_calls = 0

    for i in range(1, T + 1):
        cur_hotness = hotness[i - 1]
        per_iter_par[i - 1] = gs.cal_par_per_iter(cur_hotness, cur_deploy).mean()

        if not expert_ready and cur_priority:
            layer = cur_priority.pop(0)
            total_transmit += gs.compute_redeploy_cost(
                cur_deploy[layer], next_deploy[layer]
            )
            cur_deploy[layer] = next_deploy[layer]

        if not cur_priority and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + collection_interval + 1 and expert_ready:
            # CLAIRVOYANT: feed the rebalance call the FUTURE window
            # (iters i .. i+W-1) rather than the past one.
            end = min(T, (i - 1) + collection_interval)
            future_window = hotness[i - 1: end]
            if future_window.shape[0] >= 1:
                proposed = _ds_pack_future_window(
                    future_window, n_device, n_red_expert
                )
                n_calls += 1
                # Same convention as DS-EPLB online: mark every layer as a
                # redeploy candidate. The redeploy loop will count transmit.
                changes = [ell for ell in range(L)
                           if not np.array_equal(proposed[ell], cur_deploy[ell])]
                if changes:
                    cur_priority = list(range(L))  # match DS-EPLB ordering
                    next_deploy = proposed
                    expert_ready = False

        if verbose and i % 1024 == 0:
            print(f"    opt_cv iter {i:6d}/{T:6d}  "
                  f"running mean PAR = {per_iter_par[:i].mean():.4f}  "
                  f"transmit = {total_transmit}",
                  flush=True)

    mean_par = float(per_iter_par.mean())
    return {
        "method": "opt_cv",
        "mean_par": mean_par,
        "total_transmit": int(total_transmit),
        "transmission_time_seconds": gs.transmission_time_seconds(total_transmit),
        "total_time_seconds": gs.modeled_runtime_seconds(mean_par, total_transmit),
        "rebalance_calls": n_calls,
    }


# ---------------------------------------------------------------------------
# Per-case driver.
# ---------------------------------------------------------------------------
def run_case(
    repo_root: Path,
    ds: str, md: str, ep: int,
    submission_alias: str,
    max_iters: Optional[int],
    skip_opt_lb: bool,
    cv_collection_interval: int,
) -> dict:
    n_layers, n_experts = gs.MODEL_SHAPES[md]
    hotness = gs.load_trace(repo_root, md, ds, max_iters=max_iters)
    print(f"\n[{ds}/{md}/EP{ep}]  trace shape {hotness.shape}", flush=True)

    out: dict = {"case": (ds, md, ep), "iters": int(hotness.shape[0])}

    # 1) Online DS-EPLB baseline (cadence 1024).
    print("  running DS-EPLB online ...", flush=True)
    t0 = time.time()
    deepseek_fn = build_rebalance(ep, ep, "deepseek")
    out["deepseek"] = gs.run_case_one_method(
        method_name="deepseek", hotness=hotness, ep=ep,
        n_layers=n_layers, n_experts=n_experts,
        collection_interval=gs.COLLECTION_INTERVAL["deepseek"],
        rebalance_fn=deepseek_fn,
    )
    print(f"    done in {time.time()-t0:.1f}s "
          f"(PAR={out['deepseek']['mean_par']:.4f}, "
          f"transmit={out['deepseek']['total_transmit']}, "
          f"total_t={out['deepseek']['total_time_seconds']:.2f}s)",
          flush=True)

    # 2) Submission online (cadence 128).
    print(f"  running submission '{submission_alias}' online ...", flush=True)
    t0 = time.time()
    submission_fn = build_rebalance(ep, ep, submission_alias)
    out["submission"] = gs.run_case_one_method(
        method_name="submission", hotness=hotness, ep=ep,
        n_layers=n_layers, n_experts=n_experts,
        collection_interval=gs.COLLECTION_INTERVAL["submission"],
        rebalance_fn=submission_fn,
    )
    print(f"    done in {time.time()-t0:.1f}s "
          f"(PAR={out['submission']['mean_par']:.4f}, "
          f"transmit={out['submission']['total_transmit']}, "
          f"total_t={out['submission']['total_time_seconds']:.2f}s)",
          flush=True)

    # 3) OPT-CV-DS (offline, clairvoyant vanilla DS-EPLB packing).
    print(f"  running OPT-CV-DS (clairvoyant DS-EPLB, cadence "
          f"{cv_collection_interval}) ...", flush=True)
    t0 = time.time()
    out["opt_cv_ds"] = compute_opt_cv(
        hotness=hotness, n_device=ep, n_red_expert=ep,
        collection_interval=cv_collection_interval, verbose=False,
    )
    print(f"    done in {time.time()-t0:.1f}s "
          f"(PAR={out['opt_cv_ds']['mean_par']:.4f}, "
          f"transmit={out['opt_cv_ds']['total_transmit']}, "
          f"total_t={out['opt_cv_ds']['total_time_seconds']:.2f}s)",
          flush=True)

    # 4) OPT-CV-SUB (offline, clairvoyant SAME submission).
    print(f"  running OPT-CV-SUB (clairvoyant {submission_alias}, cadence "
          f"{cv_collection_interval}) ...", flush=True)
    t0 = time.time()
    out["opt_cv_sub"] = compute_opt_cv_sub(
        hotness=hotness, ep=ep, n_layers=n_layers, n_experts=n_experts,
        collection_interval=cv_collection_interval,
        submission_alias=submission_alias, verbose=False,
    )
    print(f"    done in {time.time()-t0:.1f}s "
          f"(PAR={out['opt_cv_sub']['mean_par']:.4f}, "
          f"transmit={out['opt_cv_sub']['total_transmit']}, "
          f"total_t={out['opt_cv_sub']['total_time_seconds']:.2f}s)",
          flush=True)

    # 5) OPT-LB (offline, strict per-iter lower bound).
    if not skip_opt_lb:
        print("  running OPT-LB (per-iter optimal, free movement) ...",
              flush=True)
        t0 = time.time()
        opt_lb = compute_opt_lb(
            hotness=hotness, n_device=ep, n_red_expert=ep,
            verbose=False,
        )
        opt_lb.pop("per_iter_par", None)
        out["opt_lb"] = opt_lb
        print(f"    done in {time.time()-t0:.1f}s "
              f"(PAR={opt_lb['mean_par']:.4f}, "
              f"total_t={opt_lb['total_time_seconds']:.2f}s)",
              flush=True)

    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _row_fmt(row: dict, headers: list[str], widths: dict[str, int]) -> str:
    return " | ".join(row[h].ljust(widths[h]) for h in headers)


def print_per_case_table(records: list[dict]) -> None:
    headers = [
        "ds", "md", "ep", "iters",
        "sub_PAR", "cvS_PAR", "cvD_PAR", "lb_PAR",
        "sub_tx", "cvS_tx", "cvD_tx",
        "sub_t", "cvS_t", "cvD_t", "ds_t", "lb_t",
        "CR_LB", "CR_CV_SUB", "CR_CV_DS",
    ]
    rows: list[dict[str, str]] = []
    for rec in records:
        ds, md, ep = rec["case"]
        sub = rec["submission"]
        cvD = rec["opt_cv_ds"]
        cvS = rec.get("opt_cv_sub")
        lb = rec.get("opt_lb")
        dso = rec["deepseek"]
        cr_lb = (sub["total_time_seconds"] / lb["total_time_seconds"]
                 if lb is not None else float("nan"))
        cr_cvD = sub["total_time_seconds"] / cvD["total_time_seconds"]
        cr_cvS = (sub["total_time_seconds"] / cvS["total_time_seconds"]
                  if cvS is not None else float("nan"))
        rows.append({
            "ds": ds, "md": md, "ep": str(ep), "iters": str(rec["iters"]),
            "sub_PAR": f"{sub['mean_par']:.4f}",
            "cvS_PAR": f"{cvS['mean_par']:.4f}" if cvS else "-",
            "cvD_PAR": f"{cvD['mean_par']:.4f}",
            "lb_PAR": f"{lb['mean_par']:.4f}" if lb else "-",
            "sub_tx": str(sub["total_transmit"]),
            "cvS_tx": str(cvS["total_transmit"]) if cvS else "-",
            "cvD_tx": str(cvD["total_transmit"]),
            "sub_t": f"{sub['total_time_seconds']:.2f}",
            "cvS_t": f"{cvS['total_time_seconds']:.2f}" if cvS else "-",
            "cvD_t": f"{cvD['total_time_seconds']:.2f}",
            "ds_t": f"{dso['total_time_seconds']:.2f}",
            "lb_t": f"{lb['total_time_seconds']:.2f}" if lb else "-",
            "CR_LB": f"{cr_lb:.4f}" if lb else "-",
            "CR_CV_SUB": f"{cr_cvS:.4f}" if cvS else "-",
            "CR_CV_DS": f"{cr_cvD:.4f}",
        })
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print("\n=== Per-case competitive ratios ===")
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(_row_fmt(row, headers, widths))


def aggregate_composite(records: list[dict]) -> dict:
    methods = {"deepseek", "submission", "opt_cv_ds", "opt_cv_sub", "opt_lb"}
    agg = {m: {"par": [], "tx": 0, "t": 0.0, "n": 0} for m in methods}
    for rec in records:
        for m in methods:
            r = rec.get(m)
            if r is None:
                continue
            a = agg[m]
            a["par"].append(r["mean_par"])
            a["tx"] += int(r.get("total_transmit", 0))
            a["t"] += float(r["total_time_seconds"])
            a["n"] += 1
    out = {}
    for m, a in agg.items():
        if a["n"] == 0:
            continue
        out[m] = {
            "n_cases": a["n"],
            "mean_par": float(np.mean(a["par"])),
            "transmit": int(a["tx"]),
            "total_time_seconds": float(a["t"]),
        }
    return out


def print_composite(composite: dict) -> None:
    print("\n=== Composite over cases we ran ===")
    headers = ["method", "n_cases", "mean_par", "transmit", "total_time_s"]
    rows = []
    for m, a in composite.items():
        rows.append({
            "method": m,
            "n_cases": str(a["n_cases"]),
            "mean_par": f"{a['mean_par']:.6f}",
            "transmit": str(a["transmit"]),
            "total_time_s": f"{a['total_time_seconds']:.4f}",
        })
    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(r[h]))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(_row_fmt(r, headers, widths))

    if "submission" in composite:
        sub_t = composite["submission"]["total_time_seconds"]
        print()
        if "opt_lb" in composite:
            cr_lb = sub_t / composite["opt_lb"]["total_time_seconds"]
            print(f"  Composite CR_LB     (vs strict floor)        = "
                  f"{cr_lb:.4f}")
        if "opt_cv_sub" in composite:
            cr_cvS = sub_t / composite["opt_cv_sub"]["total_time_seconds"]
            print(f"  Composite CR_CV_SUB (vs perfect-pred twin)   = "
                  f"{cr_cvS:.4f}  <-- prediction-uncertainty gap")
        if "opt_cv_ds" in composite:
            cr_cvD = sub_t / composite["opt_cv_ds"]["total_time_seconds"]
            print(f"  Composite CR_CV_DS  (vs clairvoyant DS-EPLB) = "
                  f"{cr_cvD:.4f}")
        if "deepseek" in composite:
            score = (100.0 * composite["deepseek"]["total_time_seconds"]
                     / sub_t)
            print(f"  Grader-style score (DS-EPLB online = 100)    = "
                  f"{score:.4f}")
        if "opt_lb" in composite and "deepseek" in composite:
            max_score = (100.0 * composite["deepseek"]["total_time_seconds"]
                         / composite["opt_lb"]["total_time_seconds"])
            print(f"  Theoretical max grader score (1/CR_LB * 100) = "
                  f"{max_score:.4f}  <-- ceiling on attainable score")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--submission", default="submission",
                   help="Submission alias to score (e.g. submission, "
                        "submission_v13, submission_v14). Loads "
                        "<alias>.py at the repo root.")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Truncate every trace to this many iterations. "
                        "Use a small value (e.g. 512) for a quick smoke "
                        "test; default uses the full trace.")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to a list of ds/md/ep cases.")
    p.add_argument("--skip-opt-lb", action="store_true",
                   help="Skip the strict per-iter lower bound (the most "
                        "expensive step).")
    p.add_argument("--cv-collection-interval", type=int, default=128,
                   help="Cadence to use for OPT-CV. Default 128 matches "
                        "the submission. Use 1024 to compare to DS-EPLB "
                        "online apples-to-apples.")
    p.add_argument("--output-json", type=str, default=None,
                   help="Optional path to write the full result dict as "
                        "JSON (per-case + composite).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    full_grid = gs.enumerate_grader_grid()
    have, missing = gs.split_available_cases(repo_root, full_grid)
    if args.only:
        keep = gs.parse_only_filter(args.only)
        have = [c for c in have if c in keep]

    print(f"Grader grid total : {len(full_grid)} cases.")
    print(f"  Have trace locally  : {len(have)} case(s).")
    print(f"  Missing trace       : {len(missing)} case(s).")
    if args.max_iters is not None:
        print(f"  Truncating to       : {args.max_iters} iters per trace.")
    print(f"  Submission alias    : {args.submission}")
    print(f"  Skip OPT-LB?        : {args.skip_opt_lb}")
    print(f"  OPT-CV cadence      : {args.cv_collection_interval}")

    records: list[dict] = []
    for idx, (ds, md, ep) in enumerate(have, start=1):
        print(f"\n[{idx}/{len(have)}] {ds}/{md}/EP{ep}", flush=True)
        rec = run_case(
            repo_root=repo_root,
            ds=ds, md=md, ep=ep,
            submission_alias=args.submission,
            max_iters=args.max_iters,
            skip_opt_lb=args.skip_opt_lb,
            cv_collection_interval=args.cv_collection_interval,
        )
        records.append(rec)

    if not records:
        print("\nNo cases ran. Add at least one trace under trace/<model>/<dataset>.npy.")
        return

    print_per_case_table(records)
    composite = aggregate_composite(records)
    print_composite(composite)

    if args.output_json:
        payload = {
            "submission_alias": args.submission,
            "max_iters": args.max_iters,
            "cv_collection_interval": args.cv_collection_interval,
            "per_case": [
                {
                    "case": list(rec["case"]),
                    "iters": rec["iters"],
                    "deepseek": rec["deepseek"],
                    "submission": rec["submission"],
                    "opt_cv_ds": rec["opt_cv_ds"],
                    "opt_cv_sub": rec["opt_cv_sub"],
                    "opt_lb": rec.get("opt_lb"),
                }
                for rec in records
            ],
            "composite": composite,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\n[info] wrote {args.output_json}")


if __name__ == "__main__":
    main()
