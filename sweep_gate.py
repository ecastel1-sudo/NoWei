#!/usr/bin/env python3
"""Sweep GATE_SAFETY on submission_v10 across the local traces.

Goal: find the GATE_SAFETY value that minimises composite total_time on the
full grader grid we can simulate locally.

Background: at GATE_SAFETY=1 the per-move PAR threshold is exactly the
score-math break-even (delta_par/n_layers * 60s == moves * 9.787e-5s).
Our shipped value is 16x that. The leaderboard leader has ~1.45x more
transit than us at lower mean PAR, suggesting they redeploy much more
aggressively. This sweep tells us how low we can go locally before the
extra transit on stationary LmSys cases stops paying off.

Usage:
  python sweep_gate.py
  python sweep_gate.py --safety 0.5 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from grader_sim import (
    COLLECTION_INTERVAL,
    MODEL_SHAPES,
    cal_par_per_iter,
    compute_redeploy_cost,
    enumerate_grader_grid,
    init_deploy_table,
    load_trace,
    modeled_runtime_seconds,
    split_available_cases,
)


def _load_submission(path: Path):
  """Fresh import so module-level constants can be toggled per run."""
  spec = importlib.util.spec_from_file_location(path.stem, str(path))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[path.stem] = mod
  spec.loader.exec_module(mod)
  return mod


def run_one(submission_path: Path, hotness: np.ndarray, ep: int,
            n_layers: int, n_experts: int, gate_safety: float) -> dict:
  """Run a single (case, gate_safety) configuration end-to-end."""
  sub = _load_submission(submission_path)
  sub.GATE_SAFETY = float(gate_safety)
  if hasattr(sub, "_reset"):
    sub._reset()

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
  cadence = COLLECTION_INTERVAL["submission"]
  for i in range(1, n_iter + 1):
    pars[i - 1] = float(cal_par_per_iter(hotness[i - 1], cur).mean())
    if not ready and queue:
      L = queue.pop(0)
      transmit += compute_redeploy_cost(cur[L], nxt[L])
      cur[L] = nxt[L]
    if not ready and not queue:
      ready = True
      finish_iter = i
    if i == finish_iter + cadence + 1 and ready:
      train = hotness[i - cadence:i]
      ct = time.time()
      change, prio, deploy_tbl, _ = sub.rebalance(train, ep, ep)
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
      "total_s": modeled_runtime_seconds(mean_par, transmit),
      "call_max_ms": max(call_times) * 1000 if call_times else 0.0,
  }


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawTextHelpFormatter)
  p.add_argument("--submission", default="submissions/submission/submission.py",
                 help="Submission file to sweep (default: the one currently "
                      "uploaded under submissions/submission/).")
  p.add_argument("--safety", type=float, nargs="+",
                 default=[16.0, 8.0, 4.0, 2.0, 1.0, 0.5],
                 help="GATE_SAFETY values to test.")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  repo_root = Path(__file__).resolve().parent
  sub_path = (repo_root / args.submission).resolve()
  if not sub_path.exists():
    raise SystemExit(f"submission not found: {sub_path}")

  have, _ = split_available_cases(repo_root, enumerate_grader_grid())
  if not have:
    raise SystemExit("No local traces found under trace/")

  print(f"Sweeping GATE_SAFETY in {args.safety} on {sub_path.name}")
  print(f"Cases (locally available): {len(have)}")

  totals = {gs: {"par": [], "tx": 0, "t": 0.0, "call_max_ms": 0.0}
            for gs in args.safety}
  for ds, md, ep in have:
    n_layers, n_experts = MODEL_SHAPES[md]
    hotness = load_trace(repo_root, md, ds)
    print(f"\n[{ds}/{md}/EP{ep}] iters={len(hotness)}")
    for gs in args.safety:
      r = run_one(sub_path, hotness, ep, n_layers, n_experts, gs)
      totals[gs]["par"].append(r["mean_par"])
      totals[gs]["tx"] += r["transmit"]
      totals[gs]["t"] += r["total_s"]
      totals[gs]["call_max_ms"] = max(
          totals[gs]["call_max_ms"], r["call_max_ms"])
      print(f"  GATE={gs:>5}: PAR={r['mean_par']:.4f} "
            f"tx={r['transmit']:>7d} t={r['total_s']:.2f} "
            f"max_ms={r['call_max_ms']:.1f}")

  print("\n=== AGGREGATE OVER LOCAL CASES ===")
  base_t = totals[args.safety[0]]["t"]  # first entry == reference
  print(f"{'gate':>6s}  {'mean_par':>9s}  {'transmit':>9s}  "
        f"{'total_s':>9s}  {'rel':>8s}  {'max_ms':>8s}")
  for gs in args.safety:
    a = totals[gs]
    mp = float(np.mean(a["par"]))
    rel = base_t / a["t"] * 100.0
    print(f"{gs:>6}  {mp:>9.5f}  {a['tx']:>9d}  {a['t']:>9.2f}  "
          f"{rel:>8.3f}  {a['call_max_ms']:>7.1f}ms")


if __name__ == "__main__":
  main()
