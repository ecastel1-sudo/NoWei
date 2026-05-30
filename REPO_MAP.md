# Repository Map

This repository contains the standalone simulator code, sample traces, reference submissions, and
experiment outputs for the MoE dynamic load-balancing competition.

## Root Files

- `dynamic_lb_simulator.py`: original simulator entry point. It loads traces, runs placement
  algorithms, computes PAR, counts redeployment transmit amount, and optionally stores raw output.
- `quickstart.py`: fresh-clone smoke runner for the committed `LmSys.npy` sample traces. It prints
  PAR, total transit, estimated transmission time, estimated total time, and score for `Default`
  and `DS-EPLB` without writing generated outputs.
- `README.md`: short project overview, install command, included sample traces, and reference
  submission list.
- `USAGE.md`: local usage notes and trace/submission layout.
- `requirements.txt`: Python dependencies for the simulator.

## Algorithm Code

- `eplb_algorithms/deepseek.py`: DeepSeek EPLB implementation copied into the repo because it is
  not distributed as a PyPI package.
- `eplb_algorithms/__init__.py`: adapter that exposes the simulator-facing `rebalance` function.

## Reference Submissions

- `submissions/smoke/submission.py`: minimal participant API implementation that returns
  `change=False` and does not request redeployment.
- `submissions/hot_expert_baseline/submission.py`: simple participant baseline that uses recent
  hotness to replicate the hottest experts in redundant slots.

These files use the same `rebalance(hotness, n_device, n_red_expert)` API expected by the
Codabench submission bundle.

## Traces

- `trace/DS-R1/LmSys.npy`: committed DS-R1 sample trace tracked through Git LFS.
- `trace/Qwen3/LmSys.npy`: committed Qwen3 sample trace tracked through Git LFS.

Additional private or local traces can be placed under the same `trace/<model>/<dataset>.npy`
layout, but they are ignored by git.

## Experiments And Results

- `experiments/ds_eplb_sweep.py`: script used to sweep DS-EPLB cadence settings.
- `experiments/results/`: committed result summaries and CSV files from previous simulator runs.

Generated raw simulator output under `output/` or `outputs/` is ignored by git.
