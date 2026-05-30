#!/usr/bin/env python3
"""Sweep DS-EPLB redeployment cadence against PAR and transfer volume.

This script is intentionally separate from dynamic_lb_simulator.py so the
competition simulator remains unchanged. It evaluates DS-R1 traces with
configurable collection intervals and optional per-cycle layer redeployment
budgets, then writes CSV and Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eplb_algorithms import rebalance


DATASETS = ("ShareGPT", "WildChat", "LmSys", "Mix")
EPS = (32, 64, 128, 256)
N_LAYERS = 58
N_EXPERTS = 256


@dataclass(frozen=True)
class Case:
    dataset: str
    ep: int


@dataclass(frozen=True)
class Variant:
    name: str
    interval: int | None
    max_layers_per_cycle: int | None = None


def init_deploy_table(n_devices: int, n_experts: int, n_red_experts: int, default: bool) -> np.ndarray:
    n_exp_per_dev = (n_experts + n_red_experts) // n_devices
    deploy = np.zeros((n_devices, n_exp_per_dev), dtype=np.int64)
    for device in range(n_devices):
        if default:
            for slot in range(n_exp_per_dev):
                deploy[device, slot] = (device * n_exp_per_dev + slot) % n_experts
        else:
            for slot in range(n_exp_per_dev - 1):
                deploy[device, slot] = (device * (n_exp_per_dev - 1) + slot) % n_experts
            deploy[device, -1] = deploy[device, -2]
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


def load_trace(trace_dir: Path, dataset: str) -> np.ndarray:
    trace = np.load(trace_dir / f"{dataset}.npy")
    trace = trace.copy()
    trace[trace == 0] = 1
    return trace


def run_default(hotness: np.ndarray, ep: int) -> tuple[float, int, float]:
    start = time.time()
    cur_deploy_table = np.array(
        [init_deploy_table(ep, N_EXPERTS, 0, default=True) for _ in range(N_LAYERS)]
    )
    pars_sum = 0.0
    count = 0
    for cur_hotness in hotness:
        pars = cal_par_per_iter(cur_hotness, cur_deploy_table)
        pars_sum += float(pars.sum())
        count += pars.size
    return pars_sum / count, 0, time.time() - start


def run_ds_eplb(
    hotness: np.ndarray,
    ep: int,
    interval: int,
    max_layers_per_cycle: int | None,
    rebalance_fn: Callable,
) -> tuple[float, int, float]:
    start = time.time()
    cur_deploy_table = np.array(
        [init_deploy_table(ep, N_EXPERTS, ep, default=False) for _ in range(N_LAYERS)]
    )
    next_deploy_table = np.zeros_like(cur_deploy_table)

    redeploy_finish_iter = 0
    algo_finish_iter = 0
    expert_ready = True
    cur_layers_priority: list[int] = []
    transmit_amount = 0
    pars_sum = 0.0
    count = 0

    for i in range(1, len(hotness) + 1):
        cur_hotness = hotness[i - 1]
        pars = cal_par_per_iter(cur_hotness, cur_deploy_table)
        pars_sum += float(pars.sum())
        count += pars.size

        if not expert_ready and i > algo_finish_iter and cur_layers_priority:
            adjust_layer_idx = cur_layers_priority.pop(0)
            transmit_amount += compute_redeploy_cost(
                cur_deploy_table[adjust_layer_idx], next_deploy_table[adjust_layer_idx]
            )
            cur_deploy_table[adjust_layer_idx] = next_deploy_table[adjust_layer_idx]

        if len(cur_layers_priority) == 0 and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + interval + 1 and expert_ready:
            algo_start = time.time()
            train_window = hotness[i - interval:i]
            change, layers_priority, deployment_table, _ = rebalance_fn(train_window)
            if change:
                selected_layers = np.asarray(layers_priority, dtype=np.int64)
                if max_layers_per_cycle is not None:
                    selected_layers = selected_layers[:max_layers_per_cycle]
                cur_layers_priority = selected_layers.tolist()
                if cur_layers_priority:
                    expert_ready = False
                    next_deploy_table[selected_layers] = deployment_table[selected_layers]
            algo_execute_iters = int(np.ceil((time.time() - algo_start) / 0.08))
            algo_finish_iter = i + algo_execute_iters - 1

    return pars_sum / count, transmit_amount, time.time() - start


def existing_baseline(output_dir: Path, dataset: str, ep: int, algo: str) -> tuple[float, int] | None:
    case_dir = output_dir / "output" / "raw_data" / f"{dataset}_DS-R1_EP{ep}"
    par_path = case_dir / f"{algo}_par.npy"
    transmit_path = case_dir / f"{algo}_transmit_amount.npy"
    if not par_path.exists() or not transmit_path.exists():
        return None
    return float(np.load(par_path).mean()), int(np.load(transmit_path).sum())


def variants(intervals: Iterable[int], max_layers: Iterable[int | None], include_default: bool) -> list[Variant]:
    out: list[Variant] = []
    if include_default:
        out.append(Variant("Default", None, None))
    for interval in intervals:
        for layer_cap in max_layers:
            suffix = "all" if layer_cap is None else str(layer_cap)
            name = f"DS-EPLB@{interval}:layers={suffix}"
            out.append(Variant(name, interval, layer_cap))
    return out


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ds_eplb_sweep.csv"
    fieldnames = [
        "dataset",
        "model",
        "ep",
        "variant",
        "collection_interval",
        "max_layers_per_cycle",
        "mean_par",
        "transmit_amount",
        "runtime_seconds",
        "par_ratio_vs_ds_eplb_1024",
        "transmit_ratio_vs_ds_eplb_1024",
        "par_improvement_vs_default",
        "extra_transmit_per_par_point",
        "matches_existing_ds_eplb_1024",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "ds_eplb_sweep_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# DS-EPLB Transfer/PAR Sweep\n\n")
        f.write("PAR is lower-is-better. `transmit_amount` counts changed expert slots.\n\n")
        f.write("| dataset | EP | variant | mean PAR | transmit | PAR ratio vs DS-EPLB@1024 | transmit ratio |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {dataset} | {ep} | {variant} | {mean_par:.6f} | {transmit_amount} | "
                "{par_ratio_vs_ds_eplb_1024:.6f} | {transmit_ratio_vs_ds_eplb_1024:.6f} |\n".format(
                    **row
                )
            )

    aggregate_rows = aggregate_by_variant(rows)
    aggregate_csv_path = out_dir / "ds_eplb_sweep_aggregate.csv"
    aggregate_fields = [
        "variant",
        "cases",
        "mean_par",
        "total_transmit",
        "mean_transmit",
        "mean_par_ratio_vs_ds_eplb_1024",
        "mean_transmit_ratio_vs_ds_eplb_1024",
        "mean_par_improvement_vs_default",
    ]
    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    aggregate_md_path = out_dir / "ds_eplb_sweep_aggregate.md"
    with aggregate_md_path.open("w", encoding="utf-8") as f:
        f.write("# DS-R1 Cadence Sweep Aggregate\n\n")
        f.write(
            "Averages are unweighted across DS-R1 dataset/EP cases. Lower mean PAR is better; "
            "higher transmit means more expert-slot movement.\n\n"
        )
        f.write(
            "| variant | mean PAR | total transmit | mean transmit ratio vs 1024 | "
            "mean PAR ratio vs 1024 | mean PAR improvement vs Default |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            f.write(
                "| {variant} | {mean_par:.6f} | {total_transmit} | "
                "{mean_transmit_ratio_vs_ds_eplb_1024:.3f} | "
                "{mean_par_ratio_vs_ds_eplb_1024:.3f} | "
                "{mean_par_improvement_vs_default:.3f} |\n".format(**row)
            )


def aggregate_by_variant(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def variant_order(name: str) -> int:
        if name == "Default":
            return -1
        return int(name.split("@", 1)[1].split(":", 1)[0])

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for variant_name in sorted(grouped, key=variant_order):
        variant_rows = grouped[variant_name]
        n = len(variant_rows)
        aggregate_rows.append(
            {
                "variant": variant_name,
                "cases": n,
                "mean_par": sum(float(row["mean_par"]) for row in variant_rows) / n,
                "total_transmit": sum(int(row["transmit_amount"]) for row in variant_rows),
                "mean_transmit": sum(int(row["transmit_amount"]) for row in variant_rows) / n,
                "mean_par_ratio_vs_ds_eplb_1024": sum(
                    float(row["par_ratio_vs_ds_eplb_1024"]) for row in variant_rows
                )
                / n,
                "mean_transmit_ratio_vs_ds_eplb_1024": sum(
                    float(row["transmit_ratio_vs_ds_eplb_1024"]) for row in variant_rows
                )
                / n,
                "mean_par_improvement_vs_default": sum(
                    float(row["par_improvement_vs_default"]) for row in variant_rows
                )
                / n,
            }
        )
    return aggregate_rows


def compute_normalized(rows: list[dict[str, object]]) -> None:
    by_case: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["ep"]))
        by_case.setdefault(key, {})[str(row["variant"])] = row

    for case_rows in by_case.values():
        default = case_rows.get("Default")
        baseline = case_rows.get("DS-EPLB@1024:layers=all")
        if baseline is None:
            continue
        baseline_par = float(baseline["mean_par"])
        baseline_tx = int(baseline["transmit_amount"])
        default_par = float(default["mean_par"]) if default else None

        for row in case_rows.values():
            mean_par = float(row["mean_par"])
            transmit = int(row["transmit_amount"])
            row["par_ratio_vs_ds_eplb_1024"] = baseline_par / mean_par
            row["transmit_ratio_vs_ds_eplb_1024"] = transmit / baseline_tx if baseline_tx else float("inf")
            row["par_improvement_vs_default"] = (
                (default_par - mean_par) / default_par if default_par else 0.0
            )
            if default_par and transmit > 0:
                par_points = default_par - mean_par
                row["extra_transmit_per_par_point"] = transmit / par_points if par_points > 0 else float("inf")
            else:
                row["extra_transmit_per_par_point"] = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("trace/DS-R1"))
    parser.add_argument("--baseline-output-dir", type=Path, default=Path("output"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--pilot", action="store_true", help="Run ShareGPT DS-R1 EP32 pilot only")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--eps", nargs="*", type=int, default=list(EPS))
    parser.add_argument("--intervals", nargs="*", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument(
        "--max-layers",
        nargs="*",
        default=["all"],
        help="Layer caps per redeploy cycle. Use 'all' for uncapped.",
    )
    parser.add_argument("--no-default", action="store_true")
    return parser.parse_args()


def parse_layer_caps(values: list[str]) -> list[int | None]:
    caps: list[int | None] = []
    for value in values:
        if value == "all":
            caps.append(None)
        else:
            caps.append(int(value))
    return caps


def main() -> int:
    args = parse_args()
    cases = [Case(dataset, ep) for dataset in args.datasets for ep in args.eps]
    if args.pilot:
        cases = [Case("ShareGPT", 32)]
        args.intervals = [512, 1024, 2048]

    layer_caps = parse_layer_caps(args.max_layers)
    sweep_variants = variants(args.intervals, layer_caps, include_default=not args.no_default)
    rows: list[dict[str, object]] = []

    for case in cases:
        print(f"loading {case.dataset} DS-R1 EP{case.ep}", flush=True)
        hotness = load_trace(args.trace_dir, case.dataset)
        rebalance_fn = rebalance(case.ep, case.ep, "deepseek")

        for variant in sweep_variants:
            print(f"  running {variant.name}", flush=True)
            if variant.interval is None:
                mean_par, transmit, runtime = run_default(hotness, case.ep)
            else:
                mean_par, transmit, runtime = run_ds_eplb(
                    hotness, case.ep, variant.interval, variant.max_layers_per_cycle, rebalance_fn
                )

            existing = None
            if variant.name == "DS-EPLB@1024:layers=all":
                existing = existing_baseline(args.baseline_output_dir, case.dataset, case.ep, "DS-EPLB")
            matches = ""
            if existing:
                matches = str(abs(existing[0] - mean_par) < 1e-9 and existing[1] == transmit)

            rows.append(
                {
                    "dataset": case.dataset,
                    "model": "DS-R1",
                    "ep": case.ep,
                    "variant": variant.name,
                    "collection_interval": variant.interval or 0,
                    "max_layers_per_cycle": variant.max_layers_per_cycle or "all",
                    "mean_par": mean_par,
                    "transmit_amount": transmit,
                    "runtime_seconds": runtime,
                    "par_ratio_vs_ds_eplb_1024": 0.0,
                    "transmit_ratio_vs_ds_eplb_1024": 0.0,
                    "par_improvement_vs_default": 0.0,
                    "extra_transmit_per_par_point": 0.0,
                    "matches_existing_ds_eplb_1024": matches,
                }
            )

    compute_normalized(rows)
    write_outputs(rows, args.out_dir)
    print(f"wrote {args.out_dir / 'ds_eplb_sweep.csv'}", flush=True)
    print(f"wrote {args.out_dir / 'ds_eplb_sweep_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
