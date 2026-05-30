#!/usr/bin/env python3
"""Run all submission_*.py and submission*.py under nowei/ and report scores.

Usage: PYTHONPATH=. python tools/run_all_submissions.py
"""
from pathlib import Path
import importlib.util
import numpy as np
import math
import sys

REPO = Path(__file__).resolve().parent.parent
NOWEI = REPO / 'nowei'
TRACE_DIR = REPO / 'trace' / 'Qwen3'
DATASET = 'LmSys'
EP = 32
CI = 128

BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
BW = 900_000_000_000


def transmission_time(tx):
    return tx * EXPERT_BYTES / BW


def modeled_time(par, tx):
    return BALANCED_COMPUTE_SECONDS * par + transmission_time(tx)


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
            st = __import__('time').time()
            win = hot[i - ci:i]
            change, lp, dep, _ = rebalance_fn(win)
            et = __import__('time').time()
            if change:
                sel = np.asarray(lp, dtype=np.int64)
                prio = sel.tolist(); ready = False; nxt[sel] = dep[sel]
            adone = i + math.ceil((et - st) / 0.08) - 1
    return s / c, tx


def load_trace():
    path = TRACE_DIR / (DATASET + '.npy')
    hot = np.load(path, mmap_mode='r')
    hot = np.array(hot, copy=True); hot[hot == 0] = 1
    return hot


def load_module_from_path(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_submission_files():
    files = []
    for p in NOWEI.glob('submission*.py'):
        files.append(p)
    return sorted(files)


def main():
    hot = load_trace()
    nl, ne = hot.shape[1], hot.shape[2]
    # DS-EPLB baseline from repo
    from eplb_algorithms import rebalance as ds_rebalance
    ds_fn = ds_rebalance(EP, EP, 'deepseek')
    rows = []
    dpar, dtx = run_default(hot, EP, nl, ne)
    rows.append(('Default', dpar, dtx))
    bpar, btx = run_loop(hot, EP, nl, ne, 1024, ds_fn)
    rows.append(('DS-EPLB', bpar, btx))

    for p in find_submission_files():
        mod = load_module_from_path(p)
        # try to find rebalance function
        if not hasattr(mod, 'rebalance'):
            continue
        fn = lambda win, _m=mod: _m.rebalance(win, EP, EP)
        reset = getattr(mod, '_reset', None)
        par, tx = run_loop(hot, EP, nl, ne, CI, fn, reset)
        rows.append((p.name, par, tx))

    base_t = modeled_time(rows[1][1], rows[1][2])
    print(f"\n{'method':<24}{'mean_PAR':>10}{'transmit':>10}{'transit_s':>11}{'score':>9}")
    print('-' * 66)
    for name, par, tx in rows:
        transit_s = transmission_time(tx)
        sc = 100.0 * base_t / modeled_time(par, tx)
        print(f"{name:<24}{par:>10.4f}{tx:>10d}{transit_s:>11.4f}{sc:>9.2f}")


if __name__ == '__main__':
    main()
