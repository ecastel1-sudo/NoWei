#!/usr/bin/env python3
"""Quick local comparison: v15 vs v18 on the LmSys traces we have on disk.

Loads each submission as a side-loaded module and replays both through
grader_sim.run_case_one_method, then prints PAR / transit / total time
per case and the composite delta. This is the smallest meaningful
sanity check before we package v18 for the grader.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

import grader_sim as gs


REPO_ROOT = Path(__file__).resolve().parent
CASES = [
  ("LmSys", "DS-R1", 32),
  ("LmSys", "DS-R1", 64),
  ("LmSys", "DS-R1", 128),
  ("LmSys", "DS-R1", 256),
  ("LmSys", "Qwen3", 32),
  ("LmSys", "Qwen3", 64),
  ("LmSys", "Qwen3", 128),
]


def _load_submission(name: str):
  path = REPO_ROOT / f"{name}.py"
  spec = importlib.util.spec_from_file_location(name, str(path))
  if spec is None or spec.loader is None:
    raise ImportError(name)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  if hasattr(mod, "_reset"):
    mod._reset()
  return mod


def _make_rebalance_fn(mod, ep: int):
  if hasattr(mod, "_reset"):
    mod._reset()
  def fn(hotness, _mod=mod, _ep=ep):
    return _mod.rebalance(hotness, _ep, _ep)
  return fn


def _run(name: str, ds: str, md: str, ep: int) -> dict:
  mod = _load_submission(name)
  fn = _make_rebalance_fn(mod, ep)
  n_layers, n_experts = gs.MODEL_SHAPES[md]
  hotness = gs.load_trace(REPO_ROOT, md, ds)
  return gs.run_case_one_method(
      method_name=name, hotness=hotness, ep=ep,
      n_layers=n_layers, n_experts=n_experts,
      collection_interval=gs.COLLECTION_INTERVAL["submission"],
      rebalance_fn=fn,
  )


def _run_deepseek(ds: str, md: str, ep: int) -> dict:
  from eplb_algorithms import rebalance as build_rebalance
  fn = build_rebalance(ep, ep, "deepseek")
  n_layers, n_experts = gs.MODEL_SHAPES[md]
  hotness = gs.load_trace(REPO_ROOT, md, ds)
  return gs.run_case_one_method(
      method_name="deepseek", hotness=hotness, ep=ep,
      n_layers=n_layers, n_experts=n_experts,
      collection_interval=gs.COLLECTION_INTERVAL["deepseek"],
      rebalance_fn=fn,
  )


def main() -> None:
  rows: list[dict[str, str]] = []
  tot = {n: {"par": [], "tx": 0, "t": 0.0} for n in ("deepseek", "v15", "v18")}
  for ds, md, ep in CASES:
    trace_path = REPO_ROOT / "trace" / md / f"{ds}.npy"
    if not trace_path.exists():
      print(f"skip {ds}/{md}/EP{ep} (no trace)")
      continue
    t0 = time.time()
    d = _run_deepseek(ds, md, ep)
    a = _run("submission_v15", ds, md, ep)
    b = _run("submission_v18", ds, md, ep)
    for n, r in (("deepseek", d), ("v15", a), ("v18", b)):
      tot[n]["par"].append(r["mean_par"])
      tot[n]["tx"] += r["total_transmit"]
      tot[n]["t"] += r["total_time_seconds"]
    delta_t = b["total_time_seconds"] - a["total_time_seconds"]
    delta_tx = b["total_transmit"] - a["total_transmit"]
    rows.append({
        "case": f"{ds}/{md}/EP{ep}",
        "v15_par": f"{a['mean_par']:.4f}",
        "v18_par": f"{b['mean_par']:.4f}",
        "v15_tx": f"{a['total_transmit']:>6d}",
        "v18_tx": f"{b['total_transmit']:>6d}",
        "Δtx": f"{delta_tx:+d}",
        "v15_t": f"{a['total_time_seconds']:7.2f}",
        "v18_t": f"{b['total_time_seconds']:7.2f}",
        "Δt": f"{delta_t:+6.2f}",
        "wall_s": f"{time.time()-t0:5.1f}",
    })
    print(f"done {ds}/{md}/EP{ep} in {time.time()-t0:.1f}s")
  headers = list(rows[0].keys())
  w = {h: max(len(h), max(len(r[h]) for r in rows)) for h in headers}
  print()
  print(" | ".join(h.ljust(w[h]) for h in headers))
  print("-+-".join("-" * w[h] for h in headers))
  for r in rows:
    print(" | ".join(r[h].ljust(w[h]) for h in headers))

  print()
  for n in ("deepseek", "v15", "v18"):
    mp = float(np.mean(tot[n]["par"]))
    tx = int(tot[n]["tx"])
    tt = float(tot[n]["t"])
    print(f"  {n:>9}: mean_par={mp:.4f}  transit={tx:>7d}  total_t={tt:8.2f}")
  if tot["v15"]["t"] and tot["v18"]["t"]:
    s15 = 100.0 * tot["deepseek"]["t"] / tot["v15"]["t"]
    s18 = 100.0 * tot["deepseek"]["t"] / tot["v18"]["t"]
    print(f"\n  composite_score: v15={s15:.4f}  v18={s18:.4f}  Δ={s18-s15:+.4f}")


if __name__ == "__main__":
  main()
