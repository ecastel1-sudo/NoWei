#!/usr/bin/env python3
"""Run a small simulator smoke test using the committed sample traces."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from eplb_algorithms import rebalance


MODEL_SHAPES = {
    "DS-R1": (58, 256),
    "Qwen3": (94, 128),
}

BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
TRANSFER_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000


def init_deploy_table(
    n_devices: int, n_experts: int, n_red_experts: int, default: bool
) -> np.ndarray:
    n_exp_per_dev = (n_experts + n_red_experts) // n_devices
    deploy = np.zeros((n_devices, n_exp_per_dev), dtype=np.int64)
    for device in range(n_devices):
        if default:
            for slot in range(n_exp_per_dev):
                deploy[device, slot] = (device * n_exp_per_dev + slot) % n_experts
        else:
            for slot in range(n_exp_per_dev - 1):
                deploy[device, slot] = (
                    device * (n_exp_per_dev - 1) + slot
                ) % n_experts
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
        pars[layer_idx] = calculate_par(
            cur_hotness[layer_idx], cur_deploy_table[layer_idx]
        )
    return pars


def compute_redeploy_cost(old: np.ndarray, new: np.ndarray) -> int:
    return int(np.sum(old != new))


def transmission_time_seconds(transmit_amount: int | float) -> float:
    return transmit_amount * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND


def modeled_runtime_seconds(mean_par: float, transmit_amount: int | float) -> float:
    return BALANCED_COMPUTE_SECONDS * mean_par + transmission_time_seconds(
        transmit_amount
    )


def load_trace(repo_root: Path, model: str, dataset: str, max_iters: int) -> np.ndarray:
    path = repo_root / "trace" / model / f"{dataset}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing sample trace: {path}")
    trace = np.load(path, mmap_mode="r")[:max_iters]
    trace = np.array(trace, copy=True)
    trace[trace == 0] = 1
    return trace


def run_default(hotness: np.ndarray, ep: int, n_layers: int, n_experts: int) -> tuple[float, int, float]:
    start = time.time()
    cur_deploy_table = np.array(
        [init_deploy_table(ep, n_experts, 0, default=True) for _ in range(n_layers)]
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
    n_layers: int,
    n_experts: int,
    collection_interval: int,
) -> tuple[float, int, float]:
    start = time.time()
    rebalance_fn = rebalance(ep, ep, "deepseek")
    cur_deploy_table = np.array(
        [init_deploy_table(ep, n_experts, ep, default=False) for _ in range(n_layers)]
    )
    next_deploy_table = np.zeros_like(cur_deploy_table)

    redeploy_finish_iter = 0
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

        if not expert_ready and cur_layers_priority:
            adjust_layer_idx = cur_layers_priority.pop(0)
            transmit_amount += compute_redeploy_cost(
                cur_deploy_table[adjust_layer_idx],
                next_deploy_table[adjust_layer_idx],
            )
            cur_deploy_table[adjust_layer_idx] = next_deploy_table[adjust_layer_idx]

        if len(cur_layers_priority) == 0 and not expert_ready:
            expert_ready = True
            redeploy_finish_iter = i

        if i == redeploy_finish_iter + collection_interval + 1 and expert_ready:
            train_window = hotness[i - collection_interval:i]
            change, layers_priority, deployment_table, _ = rebalance_fn(train_window)
            if change:
                selected_layers = np.asarray(layers_priority, dtype=np.int64)
                cur_layers_priority = selected_layers.tolist()
                expert_ready = False
                next_deploy_table[selected_layers] = deployment_table[selected_layers]

    return pars_sum / count, transmit_amount, time.time() - start


def run_case(
    repo_root: Path,
    model: str,
    dataset: str,
    ep: int,
    max_iters: int,
    collection_interval: int,
) -> list[dict[str, object]]:
    n_layers, n_experts = MODEL_SHAPES[model]
    hotness = load_trace(repo_root, model, dataset, max_iters)
    if len(hotness) <= collection_interval:
        raise ValueError(
            f"max_iters must be greater than collection_interval={collection_interval}"
        )

    default_par, default_transmit, default_runtime = run_default(
        hotness, ep, n_layers, n_experts
    )
    ds_par, ds_transmit, ds_runtime = run_ds_eplb(
        hotness, ep, n_layers, n_experts, collection_interval
    )
    rows = [
        {
            "model": model,
            "dataset": dataset,
            "ep": ep,
            "method": "Default",
            "mean_par": default_par,
            "transmit_amount": default_transmit,
            "iterations": len(hotness),
            "runtime_seconds": default_runtime,
        },
        {
            "model": model,
            "dataset": dataset,
            "ep": ep,
            "method": "DS-EPLB",
            "mean_par": ds_par,
            "transmit_amount": ds_transmit,
            "iterations": len(hotness),
            "runtime_seconds": ds_runtime,
        },
    ]
    baseline_total_time = modeled_runtime_seconds(ds_par, ds_transmit)
    for row in rows:
        row["transmission_time_seconds"] = transmission_time_seconds(
            row["transmit_amount"]
        )
        row["total_time_seconds"] = modeled_runtime_seconds(
            row["mean_par"], row["transmit_amount"]
        )
        row["score"] = 100.0 * baseline_total_time / row["total_time_seconds"]
    return rows


def print_rows(rows: list[dict[str, object]]) -> None:
    headers = [
        "model",
        "dataset",
        "ep",
        "method",
        "mean_par",
        "total_transit",
        "transmission_time_seconds",
        "total_time_seconds",
        "score",
    ]
    widths = {header: len(header) for header in headers}
    rendered_rows = []
    for row in rows:
        rendered = {
            "model": str(row["model"]),
            "dataset": str(row["dataset"]),
            "ep": str(row["ep"]),
            "method": str(row["method"]),
            "mean_par": f"{row['mean_par']:.6f}",
            "total_transit": str(row["transmit_amount"]),
            "transmission_time_seconds": f"{row['transmission_time_seconds']:.6f}",
            "total_time_seconds": f"{row['total_time_seconds']:.6f}",
            "score": f"{row['score']:.6f}",
        }
        rendered_rows.append(rendered)
        for header, value in rendered.items():
            widths[header] = max(widths[header], len(value))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rendered_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick simulator check on committed sample traces."
    )
    parser.add_argument("--model", choices=sorted(MODEL_SHAPES), default="Qwen3")
    parser.add_argument("--dataset", default="LmSys")
    parser.add_argument("--ep", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--collection-interval", type=int, default=1024)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run both committed sample traces instead of only --model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    models = sorted(MODEL_SHAPES) if args.all_samples else [args.model]
    rows: list[dict[str, object]] = []
    for model in models:
        rows.extend(
            run_case(
                repo_root=repo_root,
                model=model,
                dataset=args.dataset,
                ep=args.ep,
                max_iters=args.max_iters,
                collection_interval=args.collection_interval,
            )
        )
    print_rows(rows)


if __name__ == "__main__":
    main()
