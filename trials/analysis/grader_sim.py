#!/usr/bin/env python3
"""Local mirror of the competition grader.

Why this file exists:
  We want our local "does it score better?" run to use the SAME conditions
  the official grader uses, so the number we read here is the best possible
  predictor of the score the grader will give us.

What "same conditions" means in practice (decoded from
dynamic_lb_simulator.py lines 341-364 and confirmed against the grader
numbers the user pasted: mean_par=3.02, transmit=197943, total_time=4553
=> ratio 4553/3.02 == 25 == number of cases in the grid):

  - GRID: 4 datasets x 2 models x 4 EPs = 32 cases minus
          (Qwen3 + EP=256) and (Mix + Qwen3) -> 25 cases.
  - PER-CASE config:
      n_red_experts        = ep
      n_devices            = ep
      n_layers / n_experts = MODEL_SHAPES[md]
      collection_interval  = 1024 for deepseek   (line 220)
                             128  for submission (default from line 352)
  - PER-CASE total time    = 60.0 * mean_par + transmit_amount * 88_080_384 / 9e11
  - COMPOSITE aggregation across cases (recovered from your numbers):
      mean_par             = mean over cases     (uniform)
      transmit             = sum over cases
      total_time           = sum over cases
      composite_score      = 100 * sum(deepseek_total) / sum(submission_total)

You only have trace/<model>/<dataset>.npy committed for LmSys on both
models. This script tells you which cases are missing and runs every case
for which the trace file is on disk. Drop more *.npy files into
trace/<model>/<dataset>.npy to extend coverage.

Use --include-default to also score the no-rebalance baseline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from eplb_algorithms import rebalance as build_rebalance


# ---------------------------------------------------------------------------
# Constants that must match dynamic_lb_simulator.py exactly.
# ---------------------------------------------------------------------------
MODEL_SHAPES = {
    "DS-R1": (58, 256),    # (n_layers, n_experts)
    "Qwen3": (94, 128),
}

DATASETS = ["Mix", "ShareGPT", "WildChat", "LmSys"]
MODELS = ["DS-R1", "Qwen3"]
EPS = [32, 64, 128, 256]

# Exclusions copied verbatim from dynamic_lb_simulator.py line 344.
EXCLUDE_RULES = [
    lambda ds, md, ep: md == "Qwen3" and ep == 256,
    lambda ds, md, ep: ds == "Mix" and md == "Qwen3",
]

# Simulator scoring constants (dynamic_lb_simulator.py lines 12-14).
BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
TRANSFER_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000

# ---------------------------------------------------------------------------
# Energy-composite constants (see ENERGY_COMPOSITE_SCORE.pdf).
#
# The simulator's bandwidth constant (900 GB/s) is exactly NVLink 4.0 on
# H100-SXM, so we attach that platform's published power numbers to express
# the existing time score in joules. We expose two per-move energies:
#
#   - E_MOVE_J_TDP  (~0.069 J): TDP-window method. Conservative upper bound.
#                   Mathematically equivalent to the existing time-based
#                   composite score (both terms scale by H100_TDP_W), so any
#                   "TDP energy score" ranks identically to composite_score.
#                   We expose it as a sanity column.
#   - E_MOVE_J_BITS (~0.0011 J): bit-energy method (NVLink 1.3 pJ/bit + HBM3
#                   0.29 pJ/bit). The physically correct datapath cost.
#                   Weighs movement ~63x lighter than the TDP method, so the
#                   resulting energy score genuinely reranks transit-heavy
#                   submissions.
#
# E_PAR_POINT_J is the energy charge for one PAR point per case window:
# the slowest device runs (PAR-1)x extra compute for BALANCED_COMPUTE_SECONDS
# at H100_TDP_W.
# ---------------------------------------------------------------------------
H100_TDP_W = 700.0
NVLINK_PJ_PER_BIT = 1.3
HBM3_PJ_PER_BIT = 0.29
E_PAR_POINT_J = H100_TDP_W * BALANCED_COMPUTE_SECONDS
E_MOVE_J_TDP = H100_TDP_W * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND
E_MOVE_J_BITS = EXPERT_BYTES * 8 * (NVLINK_PJ_PER_BIT + HBM3_PJ_PER_BIT) * 1e-12

# Per-method collection intervals matching dynamic_lb_simulator.py run().
COLLECTION_INTERVAL = {
    "deepseek": 1024,      # line 220
    "submission": 128,     # default from line 352
}


# ---------------------------------------------------------------------------
# Helpers (duplicated from dashboard.py / quickstart.py to keep this file
# self-contained; you can run grader_sim.py without dashboard.py present).
# ---------------------------------------------------------------------------
def init_deploy_table(n_devices: int, n_experts: int, n_red_experts: int,
                      default_layout: bool) -> np.ndarray:
    n_exp_per_dev = (n_experts + n_red_experts) // n_devices
    deploy = np.zeros((n_devices, n_exp_per_dev), dtype=np.int64)
    for d in range(n_devices):
        if default_layout:
            for s in range(n_exp_per_dev):
                deploy[d, s] = (d * n_exp_per_dev + s) % n_experts
        else:
            for s in range(n_exp_per_dev - 1):
                deploy[d, s] = (d * (n_exp_per_dev - 1) + s) % n_experts
            deploy[d, -1] = deploy[d, -2]
    return deploy


def calculate_par(hotness: np.ndarray, deployment: np.ndarray) -> float:
    n_experts = hotness.shape[0]
    cut = np.bincount(deployment.reshape(-1), minlength=n_experts)
    if np.any(cut == 0):
        raise ValueError("deployment must contain every logical expert at least once")
    weights = hotness / cut
    loads = weights[deployment.reshape(-1)].reshape(deployment.shape).sum(-1)
    return float(loads.max() / loads.mean())


def cal_par_per_iter(cur_hotness: np.ndarray, cur_deploy_table: np.ndarray) -> np.ndarray:
    pars = np.zeros((cur_hotness.shape[0],), dtype=np.float64)
    for layer_idx in range(cur_hotness.shape[0]):
        pars[layer_idx] = calculate_par(cur_hotness[layer_idx], cur_deploy_table[layer_idx])
    return pars


def compute_redeploy_cost(old: np.ndarray, new: np.ndarray) -> int:
    return int(np.sum(old != new))


def transmission_time_seconds(transmit_amount: int) -> float:
    return transmit_amount * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND


def modeled_runtime_seconds(mean_par: float, transmit_amount: int) -> float:
    return BALANCED_COMPUTE_SECONDS * mean_par + transmission_time_seconds(transmit_amount)


def composite_energy_joules(mean_par: float, transmit_amount: int, n_cases: int,
                            move_energy_j: float = E_MOVE_J_BITS) -> float:
    """Total joules consumed by a method across n_cases windows.

    Mirrors modeled_runtime_seconds but in energy units. With move_energy_j =
    E_MOVE_J_TDP this collapses to H100_TDP_W * total_time_seconds, so the
    resulting score equals the existing composite_score. With move_energy_j =
    E_MOVE_J_BITS it weighs each expert move at the NVLink+HBM datapath cost
    (~0.0011 J) instead of the TDP-window cost (~0.069 J), which is the
    physically minimal interpretation of expert movement.
    """
    return E_PAR_POINT_J * float(mean_par) * int(n_cases) + move_energy_j * float(transmit_amount)


def load_trace(repo_root: Path, model: str, dataset: str,
               max_iters: Optional[int] = None) -> np.ndarray:
    path = repo_root / "trace" / model / f"{dataset}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing trace: {path}")
    trace = np.load(path, mmap_mode="r")
    if max_iters is not None:
        trace = trace[:max_iters]
    trace = np.array(trace, copy=True)
    trace[trace == 0] = 1
    return trace


# ---------------------------------------------------------------------------
# Per-iteration simulator loop. Mirrors dynamic_lb_simulator.forward() and
# matches dashboard.run_method behavior exactly.
# ---------------------------------------------------------------------------
def run_case_one_method(
    method_name: str,
    hotness: np.ndarray,
    ep: int,
    n_layers: int,
    n_experts: int,
    collection_interval: int,
    rebalance_fn: Optional[Callable],
) -> dict:
    n_iterations = len(hotness)
    per_iter_par = np.zeros(n_iterations, dtype=np.float64)
    total_transmit = 0
    algo_call_seconds: list[float] = []

    use_red = rebalance_fn is not None
    n_red = ep if use_red else 0
    cur_deploy_table = np.array([
        init_deploy_table(ep, n_experts, n_red, default_layout=not use_red)
        for _ in range(n_layers)
    ])
    next_deploy_table = np.zeros_like(cur_deploy_table)

    redeploy_finish_iter = 0
    expert_ready = True
    cur_layers_priority: list[int] = []

    wall_start = time.time()
    for i in range(1, n_iterations + 1):
        cur_hotness = hotness[i - 1]
        pars = cal_par_per_iter(cur_hotness, cur_deploy_table)
        per_iter_par[i - 1] = float(pars.mean())

        if not use_red:
            continue

        if not expert_ready and cur_layers_priority:
            adjust_layer_idx = cur_layers_priority.pop(0)
            total_transmit += compute_redeploy_cost(
                cur_deploy_table[adjust_layer_idx],
                next_deploy_table[adjust_layer_idx],
            )
            cur_deploy_table[adjust_layer_idx] = next_deploy_table[adjust_layer_idx]

        if len(cur_layers_priority) == 0 and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + collection_interval + 1 and expert_ready:
            train_window = hotness[i - collection_interval:i]
            call_start = time.time()
            change, layers_priority, deployment_table, _ = rebalance_fn(train_window)
            algo_call_seconds.append(time.time() - call_start)
            if change:
                selected_layers = np.asarray(layers_priority, dtype=np.int64)
                cur_layers_priority = selected_layers.tolist()
                expert_ready = False
                next_deploy_table[selected_layers] = deployment_table[selected_layers]

    mean_par = float(per_iter_par.mean())
    return {
        "method": method_name,
        "mean_par": mean_par,
        "total_transmit": int(total_transmit),
        "transmission_time_seconds": transmission_time_seconds(total_transmit),
        "total_time_seconds": modeled_runtime_seconds(mean_par, total_transmit),
        "wall_runtime_seconds": time.time() - wall_start,
        "algo_calls": len(algo_call_seconds),
        "algo_max_s": float(max(algo_call_seconds)) if algo_call_seconds else 0.0,
        "algo_mean_s": float(np.mean(algo_call_seconds)) if algo_call_seconds else 0.0,
    }


# ---------------------------------------------------------------------------
# Grid enumeration and case discovery.
# ---------------------------------------------------------------------------
def enumerate_grader_grid() -> list[tuple[str, str, int]]:
    """Yield every (dataset, model, ep) in the grader grid (25 cases)."""
    cases: list[tuple[str, str, int]] = []
    for ds in DATASETS:
        for md in MODELS:
            for ep in EPS:
                if any(rule(ds, md, ep) for rule in EXCLUDE_RULES):
                    continue
                cases.append((ds, md, ep))
    return cases


def split_available_cases(repo_root: Path, cases: list[tuple[str, str, int]]):
    """Partition the grid into (have-trace, missing-trace)."""
    have: list[tuple[str, str, int]] = []
    missing: list[tuple[str, str, int]] = []
    for ds, md, ep in cases:
        if (repo_root / "trace" / md / f"{ds}.npy").exists():
            have.append((ds, md, ep))
        else:
            missing.append((ds, md, ep))
    return have, missing


# ---------------------------------------------------------------------------
# Per-case runner + composite aggregator.
# ---------------------------------------------------------------------------
def run_one_case(repo_root: Path, ds: str, md: str, ep: int,
                 include_default: bool, max_iters: Optional[int]) -> dict:
    n_layers, n_experts = MODEL_SHAPES[md]
    hotness = load_trace(repo_root, md, ds, max_iters=max_iters)
    if len(hotness) <= max(COLLECTION_INTERVAL.values()):
        # The simulator silently no-ops if max_iters <= collection_interval.
        # We warn loudly so the user knows the case did not actually exercise
        # any rebalance call.
        print(f"  [warn] case {ds}/{md}/EP{ep}: trace length {len(hotness)} "
              f"<= max collection_interval {max(COLLECTION_INTERVAL.values())}; "
              f"rebalance() may not be called at all.")

    results: dict[str, dict] = {}

    if include_default:
        results["default"] = run_case_one_method(
            method_name="default", hotness=hotness, ep=ep,
            n_layers=n_layers, n_experts=n_experts,
            collection_interval=0, rebalance_fn=None,
        )

    deepseek_fn = build_rebalance(ep, ep, "deepseek")
    results["deepseek"] = run_case_one_method(
        method_name="deepseek", hotness=hotness, ep=ep,
        n_layers=n_layers, n_experts=n_experts,
        collection_interval=COLLECTION_INTERVAL["deepseek"],
        rebalance_fn=deepseek_fn,
    )

    submission_fn = build_rebalance(ep, ep, "proposed")
    results["submission"] = run_case_one_method(
        method_name="submission", hotness=hotness, ep=ep,
        n_layers=n_layers, n_experts=n_experts,
        collection_interval=COLLECTION_INTERVAL["submission"],
        rebalance_fn=submission_fn,
    )

    return {
        "case": (ds, md, ep),
        "iters": len(hotness),
        "results": results,
    }


def aggregate_composite(case_records: list[dict]) -> dict:
    """Apply the SAME aggregation the grader uses (decoded from the user's
    pasted numbers).

      mean_par     = uniform mean over cases
      transmit     = sum over cases
      total_time   = sum over cases (per-case = 60*PAR + transit_time)
      score        = 100 * sum(deepseek_total) / sum(method_total)
    """
    methods = ("default", "deepseek", "submission")
    agg: dict[str, dict] = {m: {"mean_par": [], "transmit": 0,
                                "total_time": 0.0, "n_cases": 0,
                                "algo_max_s": 0.0, "algo_mean_s": []}
                            for m in methods}
    for rec in case_records:
        for m, r in rec["results"].items():
            a = agg[m]
            a["mean_par"].append(r["mean_par"])
            a["transmit"] += r["total_transmit"]
            a["total_time"] += r["total_time_seconds"]
            a["algo_max_s"] = max(a["algo_max_s"], r["algo_max_s"])
            if r["algo_calls"] > 0:
                a["algo_mean_s"].append(r["algo_mean_s"])
            a["n_cases"] += 1
    out: dict[str, dict] = {}
    deepseek_total = agg["deepseek"]["total_time"] or 1.0
    deepseek_par = float(np.mean(agg["deepseek"]["mean_par"])) if agg["deepseek"]["mean_par"] else 0.0
    deepseek_tx = int(agg["deepseek"]["transmit"])
    deepseek_n = int(agg["deepseek"]["n_cases"]) or 1
    e_baseline_tdp = composite_energy_joules(deepseek_par, deepseek_tx, deepseek_n,
                                             move_energy_j=E_MOVE_J_TDP) or 1.0
    e_baseline_bits = composite_energy_joules(deepseek_par, deepseek_tx, deepseek_n,
                                              move_energy_j=E_MOVE_J_BITS) or 1.0
    for m in methods:
        a = agg[m]
        if a["n_cases"] == 0:
            continue
        mean_par = float(np.mean(a["mean_par"]))
        n_cases = int(a["n_cases"])
        e_tdp = composite_energy_joules(mean_par, a["transmit"], n_cases,
                                        move_energy_j=E_MOVE_J_TDP)
        e_bits = composite_energy_joules(mean_par, a["transmit"], n_cases,
                                         move_energy_j=E_MOVE_J_BITS)
        out[m] = {
            "n_cases": n_cases,
            "mean_par": mean_par,
            "transmit": int(a["transmit"]),
            "transmission_time_seconds": transmission_time_seconds(a["transmit"]),
            "total_time_seconds": float(a["total_time"]),
            "composite_score": 100.0 * deepseek_total / float(a["total_time"]),
            # Energy variants (see ENERGY_COMPOSITE_SCORE.pdf). _tdp ranks
            # identically to composite_score by construction; _bits is the
            # physically minimal version that weighs movement at NVLink+HBM
            # datapath cost (~0.0011 J/move).
            "energy_total_joules_tdp": e_tdp,
            "energy_total_joules_bits": e_bits,
            "energy_score_tdp": 100.0 * e_baseline_tdp / e_tdp,
            "energy_score_bits": 100.0 * e_baseline_bits / e_bits,
            "algo_max_s": a["algo_max_s"],
            "algo_mean_s": float(np.mean(a["algo_mean_s"])) if a["algo_mean_s"] else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def print_per_case_table(case_records: list[dict]) -> None:
    headers = [
        "ds", "md", "ep", "iters",
        "deepseek_par", "submission_par",
        "deepseek_tx", "submission_tx",
        "deepseek_t", "submission_t",
        "score_vs_deepseek",
    ]
    rows: list[dict[str, str]] = []
    for rec in case_records:
        ds, md, ep = rec["case"]
        d = rec["results"].get("deepseek", {})
        s = rec["results"].get("submission", {})
        if not (d and s):
            continue
        score = 100.0 * d["total_time_seconds"] / s["total_time_seconds"]
        rows.append({
            "ds": ds, "md": md, "ep": str(ep), "iters": str(rec["iters"]),
            "deepseek_par": f"{d['mean_par']:.4f}",
            "submission_par": f"{s['mean_par']:.4f}",
            "deepseek_tx": str(d["total_transmit"]),
            "submission_tx": str(s["total_transmit"]),
            "deepseek_t": f"{d['total_time_seconds']:.2f}",
            "submission_t": f"{s['total_time_seconds']:.2f}",
            "score_vs_deepseek": f"{score:.2f}",
        })
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


def print_composite(composite: dict, n_total_grader_cases: int) -> None:
    print()
    print(f"=== COMPOSITE (aggregated over cases we actually ran) ===")
    headers = [
        "method", "n_cases", "mean_par", "transmit",
        "transmission_time_s", "total_time_s",
        "algo_max_s", "algo_mean_s", "composite_score",
        "energy_MJ_bits", "energy_score_bits",
    ]
    rows: list[dict[str, str]] = []
    for m, a in composite.items():
        rows.append({
            "method": m,
            "n_cases": str(a["n_cases"]),
            "mean_par": f"{a['mean_par']:.6f}",
            "transmit": str(a["transmit"]),
            "transmission_time_s": f"{a['transmission_time_seconds']:.4f}",
            "total_time_s": f"{a['total_time_seconds']:.4f}",
            "algo_max_s": f"{a['algo_max_s']:.4f}",
            "algo_mean_s": f"{a['algo_mean_s']:.4f}",
            "composite_score": f"{a['composite_score']:.4f}",
            "energy_MJ_bits": f"{a['energy_total_joules_bits'] / 1e6:.4f}",
            "energy_score_bits": f"{a['energy_score_bits']:.4f}",
        })
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))

    sub = composite.get("submission")
    if sub is not None:
        delta = sub["composite_score"] - 100.0
        sign = "+" if delta >= 0 else ""
        print(f"\n  -> submission composite vs deepseek: "
              f"{sub['composite_score']:.2f}  ({sign}{delta:.2f})")

    if sub and sub["n_cases"] < n_total_grader_cases:
        print(f"\n  [info] aggregated over {sub['n_cases']} local cases. "
              f"The official grader runs {n_total_grader_cases} cases, so the "
              f"score it reports will differ; this number is a lower-noise "
              f"upper bound only over the subset you have locally.")


def print_missing_cases(missing: list[tuple[str, str, int]]) -> None:
    if not missing:
        return
    # Group missing cases by the trace file you would need to add.
    needed: dict[Path, list[tuple[str, str, int]]] = {}
    for ds, md, ep in missing:
        path = Path("trace") / md / f"{ds}.npy"
        needed.setdefault(path, []).append((ds, md, ep))
    print(f"\n[info] {len(missing)} grader case(s) skipped because their "
          f"trace file is not present locally. To match the grader "
          f"exactly, drop these files in place:")
    for path, cases in sorted(needed.items()):
        eps = ", ".join(f"EP{ep}" for _, _, ep in cases)
        print(f"  - {path}   (used by: {eps})")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--include-default", action="store_true",
                   help="Also score the no-rebalance default baseline.")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Truncate every trace to this many iterations. "
                        "Default: use whatever length the trace file has, "
                        "which matches the grader.")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only the listed cases. Format: ds/md/ep, e.g. "
                        "LmSys/DS-R1/32 LmSys/Qwen3/64")
    return p.parse_args()


def parse_only_filter(only: list[str]) -> set[tuple[str, str, int]]:
    chosen: set[tuple[str, str, int]] = set()
    for tok in only:
        try:
            ds, md, ep = tok.split("/")
            chosen.add((ds, md, int(ep)))
        except Exception as exc:
            raise SystemExit(f"--only entry {tok!r} not in ds/md/ep form: {exc}")
    return chosen


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    full_grid = enumerate_grader_grid()
    have, missing = split_available_cases(repo_root, full_grid)
    have_total = len(have)
    if args.only:
        keep = parse_only_filter(args.only)
        have = [c for c in have if c in keep]

    print(f"Grader grid total : {len(full_grid)} cases.")
    print(f"  Have trace locally  : {have_total} case(s).")
    print(f"  Missing trace       : {len(missing)} case(s).")
    if args.only:
        print(f"  Restricted by --only: running {len(have)} case(s).")
    if args.max_iters is not None:
        print(f"  Truncating every trace to {args.max_iters} iters "
              f"(NOT matching the grader, which uses the full trace).")

    case_records: list[dict] = []
    for idx, (ds, md, ep) in enumerate(have, start=1):
        print(f"\n[{idx}/{len(have)}] running case {ds}/{md}/EP{ep} ...")
        t0 = time.time()
        rec = run_one_case(repo_root, ds, md, ep,
                           include_default=args.include_default,
                           max_iters=args.max_iters)
        print(f"   done in {time.time() - t0:.1f}s")
        case_records.append(rec)

    if not case_records:
        print("\nNo cases ran. Add at least one trace file under trace/<model>/<dataset>.npy.")
        return

    print("\n=== Per-case results ===")
    print_per_case_table(case_records)

    composite = aggregate_composite(case_records)
    print_composite(composite, n_total_grader_cases=len(full_grid))

    print_missing_cases(missing)


if __name__ == "__main__":
    main()
