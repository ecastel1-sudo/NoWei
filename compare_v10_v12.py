#!/usr/bin/env python3
"""Compare v10 (current submission.py) vs v12 (new) on local traces.

We only have LmSys traces locally, so the goal of this comparison is
narrow: confirm v12 is a NO-OP on stationary cases (drift ~ 0.02 there).
The Mix benefit cannot be measured locally; that's why we still need a
grader submission to confirm the win.
"""
from __future__ import annotations

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


def _load(path: Path):
  spec = importlib.util.spec_from_file_location(path.stem, str(path))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[path.stem] = mod
  spec.loader.exec_module(mod)
  return mod


def run_one(submission_path: Path, hotness: np.ndarray, ep: int,
            n_layers: int, n_experts: int) -> dict:
  sub = _load(submission_path)
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


def main() -> None:
  repo_root = Path(__file__).resolve().parent
  v10 = (repo_root / "submission_v10.py").resolve()
  v12 = (repo_root / "submission_v12.py").resolve()
  assert v10.exists() and v12.exists()

  have, _ = split_available_cases(repo_root, enumerate_grader_grid())
  if not have:
    raise SystemExit("No local traces found")

  totals = {"v10": {"par": [], "tx": 0, "t": 0.0, "max_ms": 0.0},
            "v12": {"par": [], "tx": 0, "t": 0.0, "max_ms": 0.0}}
  for ds, md, ep in have:
    n_layers, n_experts = MODEL_SHAPES[md]
    hotness = load_trace(repo_root, md, ds)
    print(f"\n[{ds}/{md}/EP{ep}] iters={len(hotness)}")
    for tag, path in (("v10", v10), ("v12", v12)):
      r = run_one(path, hotness, ep, n_layers, n_experts)
      totals[tag]["par"].append(r["mean_par"])
      totals[tag]["tx"] += r["transmit"]
      totals[tag]["t"] += r["total_s"]
      totals[tag]["max_ms"] = max(totals[tag]["max_ms"], r["call_max_ms"])
      print(f"  {tag}: PAR={r['mean_par']:.4f}  tx={r['transmit']:>7d}  "
            f"t={r['total_s']:.2f}  max_ms={r['call_max_ms']:.1f}")

  print("\n=== AGGREGATE OVER LOCAL CASES (LmSys only) ===")
  print(f"{'tag':>4s}  {'mean_par':>9s}  {'transmit':>9s}  "
        f"{'total_s':>9s}  {'rel':>8s}  {'max_ms':>8s}")
  base_t = totals["v10"]["t"]
  for tag in ("v10", "v12"):
    a = totals[tag]
    mp = float(np.mean(a["par"]))
    rel = base_t / a["t"] * 100.0
    print(f"{tag:>4}  {mp:>9.5f}  {a['tx']:>9d}  {a['t']:>9.2f}  "
          f"{rel:>8.3f}  {a['max_ms']:>7.1f}ms")
  d_par = totals["v12"]["par"]
  print(f"\nDelta (v12 - v10):")
  print(f"  mean_par: {np.mean(totals['v12']['par']) - np.mean(totals['v10']['par']):+.6f}")
  print(f"  transmit: {totals['v12']['tx'] - totals['v10']['tx']:+d}")
  print(f"  total_s:  {totals['v12']['t'] - totals['v10']['t']:+.3f}s")


if __name__ == "__main__":
  main()
