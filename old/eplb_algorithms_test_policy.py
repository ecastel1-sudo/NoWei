#!/usr/bin/env python3
"""Test harness for submission.rebalance against Default and DS-EPLB baselines.

Mirrors dynamic_lb_simulator.py loop semantics:
  - Default     : no rebalance, plain round-robin placement (n_red_expert = 0)
  - DS-EPLB     : DeepSeek EPLB rebalance at cadence 1024 (the sim's hardcoded baseline)
  - Submission  : participant rebalance at a configurable cadence (default 128)

Score = 100 * baseline_modeled_time / candidate_modeled_time
where baseline = DS-EPLB and modeled_time = 60.0 * mean_PAR + transmit * 88_080_384 / 9e11.

Usage:
  python tools/test_policy.py                    # both sample traces, EP32, cadence 128
  python tools/test_policy.py --cadence 64       # finer cadence
  python tools/test_policy.py --model Qwen3      # one trace only
  python tools/test_policy.py --submission submission_v4_anchored.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Make the repo root importable so `eplb_algorithms` resolves regardless of cwd.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Scoring constants (must match dynamic_lb_simulator.py)
BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
BANDWIDTH_BPS = 900_000_000_000

# Trace metadata
MODEL_SHAPES = {
    "DS-R1": (58, 256),
    "Qwen3": (94, 128),
}


def transmission_time(tx: int | float) -> float:
    return tx * EXPERT_BYTES / BANDWIDTH_BPS


def modeled_time(mean_par: float, tx: int | float) -> float:
    return BALANCED_COMPUTE_SECONDS * mean_par + transmission_time(tx)


def init_deploy_table(n_dev: int, n_exp: int, n_red: int, default: bool) -> np.ndarray:
    """Mirrors DynamicAlg.init_deploy_table in the simulator."""
    epd = (n_exp + n_red) // n_dev
    dep = np.zeros((n_dev, epd), dtype=np.int64)
    for d in range(n_dev):
        if default:
            for s in range(epd):
                dep[d, s] = (d * epd + s) % n_exp
        else:
            for s in range(epd - 1):
                dep[d, s] = (d * (epd - 1) + s) % n_exp
            dep[d, -1] = dep[d, -2]
    return dep


def calc_par(hot_layer: np.ndarray, dep_layer: np.ndarray) -> float:
    n_exp = hot_layer.shape[0]
    cut = np.bincount(dep_layer.reshape(-1), minlength=n_exp)
    if np.any(cut == 0):
        raise ValueError("invalid deployment: some logical expert is missing")
    w = hot_layer / cut
    loads = w[dep_layer.reshape(-1)].reshape(dep_layer.shape).sum(-1)
    m = float(loads.mean())
    return float(loads.max() / m) if m > 0 else 1.0


def par_per_iter(hot: np.ndarray, dep: np.ndarray) -> np.ndarray:
    return np.array([calc_par(hot[L], dep[L]) for L in range(hot.shape[0])])


def validate_deployment(dep: np.ndarray, n_layers: int, n_dev: int, exp_per_dev: int, n_exp: int) -> None:
    if dep.shape != (n_layers, n_dev, exp_per_dev):
        raise AssertionError(
            f"bad deployment shape {dep.shape}, expected {(n_layers, n_dev, exp_per_dev)}"
        )
    if dep.dtype != np.int64:
        raise AssertionError(f"deployment dtype must be int64, got {dep.dtype}")
    for L in range(n_layers):
        present = np.bincount(dep[L].reshape(-1), minlength=n_exp)
        if np.any(present == 0):
            missing = int(np.argmax(present == 0))
            raise AssertionError(
                f"layer {L}: logical expert {missing} not present in deployment"
            )


def run_default(hot: np.ndarray, ep: int, n_layers: int, n_exp: int) -> tuple[float, int, float]:
    t0 = time.time()
    cur = np.array([init_deploy_table(ep, n_exp, 0, True) for _ in range(n_layers)])
    s, c = 0.0, 0
    for h in hot:
        p = par_per_iter(h, cur)
        s += float(p.sum())
        c += int(p.size)
    return s / c, 0, time.time() - t0


def run_with_rebalance(
    hot: np.ndarray,
    ep: int,
    n_layers: int,
    n_exp: int,
    cadence: int,
    rebalance_fn: Callable,
    reset_fn: Callable | None = None,
    verbose: bool = False,
) -> tuple[float, int, float, dict]:
    """Closed-loop simulator. Mirrors DynamicAlg.forward()."""
    if reset_fn is not None:
        reset_fn()

    cur = np.array([init_deploy_table(ep, n_exp, ep, False) for _ in range(n_layers)])
    nxt = np.zeros_like(cur)
    redeploy_done_iter = 0
    algo_done_iter = 0
    expert_ready = True
    cur_layers_priority: list[int] = []
    transmit = 0
    par_sum = 0.0
    par_count = 0
    n_calls = 0
    total_algo_time = 0.0
    max_algo_time = 0.0

    t0 = time.time()
    n = len(hot)
    exp_per_dev = (n_exp + ep) // ep
    for i in range(1, n + 1):
        cur_hot = hot[i - 1]
        p = par_per_iter(cur_hot, cur)
        par_sum += float(p.sum())
        par_count += int(p.size)

        if (not expert_ready) and i > algo_done_iter and cur_layers_priority:
            L = cur_layers_priority.pop(0)
            transmit += int(np.sum(cur[L] != nxt[L]))
            cur[L] = nxt[L]

        if len(cur_layers_priority) == 0 and not expert_ready:
            expert_ready = True
            redeploy_done_iter = i

        if i == redeploy_done_iter + cadence + 1 and expert_ready:
            st = time.time()
            win = hot[i - cadence:i]
            change, layers_priority, dep, _ = rebalance_fn(win)
            et = time.time()
            n_calls += 1
            elapsed = et - st
            total_algo_time += elapsed
            max_algo_time = max(max_algo_time, elapsed)

            validate_deployment(np.asarray(dep), n_layers, ep, exp_per_dev, n_exp)
            if change:
                sel = np.asarray(layers_priority, dtype=np.int64)
                cur_layers_priority = sel.tolist()
                expert_ready = False
                nxt[sel] = np.asarray(dep)[sel]
            algo_done_iter = i + math.ceil(elapsed / 0.08) - 1

    runtime = time.time() - t0
    extra = {
        "n_calls": n_calls,
        "total_algo_time": total_algo_time,
        "max_algo_time": max_algo_time,
        "mean_algo_time": total_algo_time / max(n_calls, 1),
    }
    return par_sum / par_count, transmit, runtime, extra


def load_trace(model: str, dataset: str, max_iters: int) -> np.ndarray:
    path = REPO / "trace" / model / f"{dataset}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing trace: {path}")
    hot = np.load(path, mmap_mode="r")[:max_iters]
    hot = np.array(hot, copy=True)
    hot[hot == 0] = 1
    return hot


def load_submission_module(submission_path: Path):
    name = "submission_under_test"
    spec = importlib.util.spec_from_file_location(name, str(submission_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load submission from {submission_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "rebalance"):
        raise AttributeError(f"{submission_path} has no rebalance() function")
    return mod


def run_case(
    model: str,
    dataset: str,
    ep: int,
    max_iters: int,
    cadence: int,
    submission_mod,
) -> list[dict]:
    n_layers, n_exp = MODEL_SHAPES[model]
    hot = load_trace(model, dataset, max_iters)

    # Default
    d_par, d_tx, d_rt = run_default(hot, ep, n_layers, n_exp)

    # DS-EPLB at cadence 1024 (the simulator's hardcoded baseline)
    from eplb_algorithms import rebalance as ds_rebalance_factory
    ds_fn = ds_rebalance_factory(ep, ep, "deepseek")
    ds_par, ds_tx, ds_rt, ds_extra = run_with_rebalance(
        hot, ep, n_layers, n_exp, cadence=1024, rebalance_fn=ds_fn,
    )

    # Submission at user cadence
    reset_fn = getattr(submission_mod, "_reset", None)
    sub_fn = lambda win: submission_mod.rebalance(win, ep, ep)
    s_par, s_tx, s_rt, s_extra = run_with_rebalance(
        hot, ep, n_layers, n_exp, cadence=cadence, rebalance_fn=sub_fn, reset_fn=reset_fn,
    )

    base_t = modeled_time(ds_par, ds_tx)
    rows = [
        ("Default",       d_par, d_tx, d_rt, {}),
        ("DS-EPLB@1024",  ds_par, ds_tx, ds_rt, ds_extra),
        (f"Submission@{cadence}", s_par, s_tx, s_rt, s_extra),
    ]
    return [
        {
            "model": model, "dataset": dataset, "ep": ep, "method": name,
            "mean_par": par, "transmit": tx,
            "transit_s": transmission_time(tx),
            "total_s": modeled_time(par, tx),
            "score": 100.0 * base_t / modeled_time(par, tx),
            "runtime_s": rt,
            **extra,
        }
        for name, par, tx, rt, extra in rows
    ]


def print_table(rows: list[dict]) -> None:
    headers = ["model", "ep", "method", "mean_par", "transmit", "transit_s", "total_s", "score"]
    widths = {h: len(h) for h in headers}
    rendered = []
    for r in rows:
        d = {
            "model": str(r["model"]),
            "ep": str(r["ep"]),
            "method": str(r["method"]),
            "mean_par": f"{r['mean_par']:.4f}",
            "transmit": str(r["transmit"]),
            "transit_s": f"{r['transit_s']:.4f}",
            "total_s": f"{r['total_s']:.4f}",
            "score": f"{r['score']:.2f}",
        }
        rendered.append(d)
        for h in headers:
            widths[h] = max(widths[h], len(d[h]))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for d in rendered:
        print(" | ".join(d[h].ljust(widths[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test submission.rebalance against Default and DS-EPLB.")
    p.add_argument("--submission", default="submission.py",
                   help="path to submission file with a rebalance(hot, n_dev, n_red) function")
    p.add_argument("--model", choices=sorted(MODEL_SHAPES) + ["all"], default="all")
    p.add_argument("--dataset", default="LmSys")
    p.add_argument("--ep", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=1200)
    p.add_argument("--cadence", type=int, default=128,
                   help="collection_interval used for the submission (DS-EPLB stays at 1024)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sub_path = (REPO / args.submission).resolve()
    submission_mod = load_submission_module(sub_path)
    print(f"submission: {sub_path}")
    print(f"cadence: {args.cadence} (DS-EPLB baseline cadence: 1024)")
    print(f"max_iters: {args.max_iters}\n")

    models = sorted(MODEL_SHAPES) if args.model == "all" else [args.model]
    rows: list[dict] = []
    for m in models:
        rows.extend(run_case(m, args.dataset, args.ep, args.max_iters, args.cadence, submission_mod))
    print_table(rows)

    # Per-call algo time summary for the submission
    sub_rows = [r for r in rows if r["method"].startswith("Submission")]
    if sub_rows:
        print()
        for r in sub_rows:
            calls = r.get("n_calls", 0)
            mean_t = r.get("mean_algo_time", 0.0) * 1000.0
            max_t = r.get("max_algo_time", 0.0) * 1000.0
            print(
                f"  [{r['model']}] n_calls={calls:3d}  mean_algo={mean_t:6.1f}ms  "
                f"max_algo={max_t:6.1f}ms  budget=80ms/iter"
            )


if __name__ == "__main__":
    main()
