#!/usr/bin/env python3
"""Quick: compute per-layer drift histogram on every local trace.

Confirms the LmSys traces sit at drift ~ 0 so a drift-adaptive gate behaves
exactly like the global v10 gate on those cases (no risk on the 15 grader
LmSys/ShareGPT/WildChat cases).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from grader_sim import COLLECTION_INTERVAL, load_trace, MODEL_SHAPES


def per_layer_drift(window: np.ndarray) -> np.ndarray:
  """Same formula as submission_v10._per_layer_drift, on a single window."""
  half = max(1, window.shape[0] // 2)
  early = window[:half].sum(axis=0)
  late = window[half:].sum(axis=0)
  diff = np.abs(late - early).sum(axis=1)
  total = (early + late).sum(axis=1)
  return diff / np.maximum(total, 1e-9)


def main() -> None:
  repo_root = Path(__file__).resolve().parent
  cadence = COLLECTION_INTERVAL["submission"]
  for model in ["DS-R1", "Qwen3"]:
    for ds in ["LmSys"]:
      path = repo_root / "trace" / model / f"{ds}.npy"
      if not path.exists():
        continue
      hot = load_trace(repo_root, model, ds)
      n_layers = MODEL_SHAPES[model][0]
      print(f"\n[{ds}/{model}] iters={len(hot)}  layers={n_layers}")
      drifts: list[float] = []
      for start in range(0, len(hot) - cadence, cadence):
        win = hot[start:start + cadence]
        d = per_layer_drift(win)
        drifts.extend(d.tolist())
      drifts = np.array(drifts)
      qs = np.quantile(drifts, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
      print(f"  drift quantiles: min={qs[0]:.4f}, q25={qs[1]:.4f}, "
            f"median={qs[2]:.4f}, q75={qs[3]:.4f}, q95={qs[4]:.4f}, "
            f"max={qs[5]:.4f}, mean={drifts.mean():.4f}")


if __name__ == "__main__":
  main()
