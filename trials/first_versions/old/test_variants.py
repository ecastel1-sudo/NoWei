#!/usr/bin/env python3
"""Score every STARE-LB variant against a real (or sample) trace.

USAGE (from the simulator repo root, after `git lfs pull`):

    python test_variants.py --model Qwen3 --dataset LmSys --ep 32
    python test_variants.py --model DS-R1 --dataset WildChat --ep 64 --collection-interval 128

It loads trace/<model>/<dataset>.npy, runs DS-EPLB (official baseline) plus each
submission_v*.py through the SAME closed loop the grader uses, and prints a table
of PAR / transmit / score. Higher score is better; baseline DS-EPLB = 100.

Drop this file + the submission_v*.py files into the repo root.
"""
import argparse, importlib, time, math, os
import numpy as np

BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
BW = 900_000_000_000
MODEL_SHAPES = {"DS-R1": (58, 256), "Qwen3": (94, 128)}


def transmission_time(tx): return tx * EXPERT_BYTES / BW
def modeled_time(par, tx): return BALANCED_COMPUTE_SECONDS * par + transmission_time(tx)


def init_deploy_table(n_dev, n_exp, n_red, default):
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


def calc_par(hot, dep):
    n_exp = hot.shape[0]
    cut = np.bincount(dep.reshape(-1), minlength=n_exp)
    w = hot / cut
    loads = w[dep.reshape(-1)].reshape(dep.shape).sum(-1)
    return float(loads.max() / loads.mean())


def par_per_iter(hot, dep):
    return np.array([calc_par(hot[L], dep[L]) for L in range(hot.shape[0])])


def run_default(hot, ep, nl, ne):
    cur = np.array([init_deploy_table(ep, ne, 0, True) for _ in range(nl)])
    s, c = 0.0, 0
    for h in hot:
        p = par_per_iter(h, cur); s += p.sum(); c += p.size
    return s / c, 0


def run_loop(hot, ep, nl, ne, ci, rebalance_fn, reset_fn=None):
    if reset_fn: reset_fn()
    cur = np.array([init_deploy_table(ep, ne, ep, False) for _ in range(nl)])
    nxt = np.zeros_like(cur)
    rdone, adone = 0, 0
    ready = True
    prio = []
    tx = 0; s = 0.0; c = 0
    n = len(hot)
    for i in range(1, n + 1):
        h = hot[i - 1]
        p = par_per_iter(h, cur); s += p.sum(); c += p.size
        if (not ready) and i > adone and prio:
            L = prio.pop(0)
            tx += int(np.sum(cur[L] != nxt[L])); cur[L] = nxt[L]
        if len(prio) == 0 and not ready:
            ready = True; rdone = i
        if i == rdone + ci + 1 and ready:
            st = time.time()
            win = hot[i - ci:i]
            change, lp, dep, _ = rebalance_fn(win)
            et = time.time()
            if change:
                sel = np.asarray(lp, dtype=np.int64)
                prio = sel.tolist(); ready = False; nxt[sel] = dep[sel]
            adone = i + math.ceil((et - st) / 0.08) - 1
    return s / c, tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_SHAPES, default="Qwen3")
    ap.add_argument("--dataset", default="LmSys")
    ap.add_argument("--ep", type=int, default=32)
    ap.add_argument("--max-iters", type=int, default=0, help="0 = all")
    ap.add_argument("--collection-interval", type=int, default=128)
    args = ap.parse_args()

    nl, ne = MODEL_SHAPES[args.model]
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "trace", args.model, args.dataset + ".npy")
    hot = np.load(path, mmap_mode="r")
    if args.max_iters: hot = hot[:args.max_iters]
    hot = np.array(hot, copy=True); hot[hot == 0] = 1
    print(f"trace {args.model}/{args.dataset} shape={hot.shape} EP{args.ep} ci={args.collection_interval}")

    from eplb_algorithms import rebalance as ds_rebalance
    ds_fn = ds_rebalance(args.ep, args.ep, "deepseek")

    rows = []
    dpar, dtx = run_default(hot, args.ep, nl, ne)
    rows.append(("Default", dpar, dtx))
    # official baseline cadence is 1024
    bpar, btx = run_loop(hot, args.ep, nl, ne, 1024, ds_fn)
    rows.append(("DS-EPLB", bpar, btx))

    for mod in ["submission_v1_base", "submission_v2_dual",
                "submission_v3_reward", "submission_v4_anchored"]:
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError:
            continue
        fn = lambda win, _m=m: _m.rebalance(win, args.ep, args.ep)
        reset = getattr(m, "_reset", None)
        par, tx = run_loop(hot, args.ep, nl, ne, args.collection_interval, fn, reset)
        rows.append((mod.replace("submission_", ""), par, tx))

    base_t = modeled_time(bpar, btx)
    print(f"\n{'method':<14}{'mean_PAR':>10}{'transmit':>10}{'transit_s':>11}{'score':>9}")
    print("-" * 54)
    for name, par, tx in rows:
        transit_s = transmission_time(tx)
        sc = 100.0 * base_t / modeled_time(par, tx)
        print(f"{name:<14}{par:>10.4f}{tx:>10d}{transit_s:>11.4f}{sc:>9.2f}")


if __name__ == "__main__":
    main()
