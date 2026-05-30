#!/usr/bin/env python3
"""Run the focused archive-ready variant against the Qwen3/LmSys trace.

Usage:
    PYTHONPATH=. python tools/run_v2_dual.py
    PYTHONPATH=. python tools/run_v2_dual.py --model DS-R1 --dataset WildChat --ep 64
"""
from pathlib import Path
import argparse
import importlib.util
import numpy as np

REPO = Path(__file__).resolve().parent.parent
NOWEI = REPO / "nowei"
BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
BW = 900_000_000_000
MODEL_SHAPES = {"DS-R1": (58, 256), "Qwen3": (94, 128)}


def transmission_time(tx):
    return tx * EXPERT_BYTES / BW


def modeled_time(par, tx):
    return BALANCED_COMPUTE_SECONDS * par + transmission_time(tx)


def load_trace(model, dataset):
    path = REPO / "trace" / model / f"{dataset}.npy"
    hot = np.load(path, mmap_mode="r")
    hot = np.array(hot, copy=True)
    hot[hot == 0] = 1
    return hot


def load_module_from_path(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_default(hot, ep, nl, ne):
    epd = (ne + ep) // ep
    cur = np.array([
        np.vstack([
            np.array([(d * epd + s) % ne for s in range(epd)], dtype=np.int64)
            for d in range(ep)
        ])
        for _ in range(nl)
    ])
    s = 0.0
    c = 0
    for h in hot:
        for L in range(h.shape[0]):
            dep = cur[L]
            cut = np.bincount(dep.reshape(-1), minlength=h.shape[1])
            w = h[L] / cut
            loads = w[dep.reshape(-1)].reshape(dep.shape).sum(-1)
            s += float(loads.max() / loads.mean())
            c += 1
    return s / c, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_SHAPES, default="Qwen3")
    ap.add_argument("--dataset", default="LmSys")
    ap.add_argument("--ep", type=int, default=32)
    ap.add_argument("--collection-interval", type=int, default=128)
    args = ap.parse_args()

    nl, ne = MODEL_SHAPES[args.model]
    hot = load_trace(args.model, args.dataset)
    print(f"trace {args.model}/{args.dataset} shape={hot.shape} EP{args.ep} ci={args.collection_interval}")

    from eplb_algorithms import rebalance as ds_rebalance
    ds_fn = ds_rebalance(args.ep, args.ep, "deepseek")

    mod = load_module_from_path(REPO / "submission.py")
    if hasattr(mod, "_reset"):
        mod._reset()

    def fn(win, _m=mod):
        return _m.rebalance(win, args.ep, args.ep)

    from nowei.test_variants import run_default as tv_run_default, run_loop, modeled_time
    dpar, dtx = tv_run_default(hot, args.ep, nl, ne)
    bpar, btx = run_loop(hot, args.ep, nl, ne, 1024, ds_fn)
    par, tx = run_loop(hot, args.ep, nl, ne, args.collection_interval, fn, getattr(mod, "_reset", None))
    base_t = modeled_time(bpar, btx)

    print(f"\n{'method':<16}{'mean_PAR':>10}{'transmit':>10}{'transit_s':>11}{'score':>9}")
    print("-" * 56)
    for name, p, t in [("Default", dpar, dtx), ("DS-EPLB", bpar, btx), ("best", par, tx)]:
        transit_s = transmission_time(t)
        score = 100.0 * base_t / modeled_time(p, t)
        print(f"{name:<16}{p:>10.4f}{t:>10d}{transit_s:>11.4f}{score:>9.2f}")


if __name__ == "__main__":
    main()
