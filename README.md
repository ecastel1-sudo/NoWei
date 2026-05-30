# NoWei — MoE Dynamic Load-Balancing Competition

**Team NoWei · LauzHack · Dynamic load balancing for Mixture-of-Experts inference.**

This repository ships **`submission.py`** — our v15 deliverable, an online
EPLB policy that wraps the DeepSeek replicate-then-pack core with five
serving-time ideas (EWMA smoothing, recency-weighted window, tail-only
cost gate, two-level placement anchor, and a TCP-AIMD per-layer adaptive
gate). On the official grader it scores **110.968** (composite),
**1.7184** mean PAR, **531 587** total transmit.

All eighteen versions we built along the way, every grader feedback,
every analysis script, every zip we ever uploaded, and the very first
hand-rolled prototypes are preserved under [`trials/`](trials/) so the
full design history is auditable.

---

## Table of contents

1. [What's in the repo](#1-whats-in-the-repo)
2. [Quickstart](#2-quickstart)
3. [The shipped algorithm at a glance](#3-the-shipped-algorithm-at-a-glance)
4. [Everything we tried (version chain)](#4-everything-we-tried-version-chain)
5. [Score history](#5-score-history-official-grader)
6. [Ideas surveyed but **not** adopted](#6-ideas-surveyed-but-not-adopted)
7. [Reproducing a trial run locally](#7-reproducing-a-trial-run-locally)
8. [Where to look for what](#8-where-to-look-for-what)

---

## 1. What's in the repo

```text
NoWei/
├── submission.py          # ← THE DELIVERABLE (= v15, score 110.968)
├── README.md              # this file
├── CITATIONS.md           # paper-level citations for every algorithmic idea
├── LICENSE
├── requirements.txt
├── eplb_algorithms/       # DeepSeek-EPLB core + simulator adapter
├── dynamic_lb_simulator.py# the official competition simulator
├── quickstart.py          # one-shot smoke runner on the committed sample trace
├── trace/                 # committed Qwen3 / DS-R1 LmSys sample traces (git-lfs)
├── baselines/             # reference participant baselines
│   ├── smoke/             #   no-op submission (returns change=False)
│   └── hot_expert_baseline/ # naive "replicate hottest experts" baseline
├── experiments/           # DS-EPLB cadence sweep + result snapshots
├── output/                # generated figures + per-case CSV summaries
│   ├── figure/            #   per-iter PAR, transmit, EP scaling, score evolution
│   ├── summary/           #   per-case CSV (default / deepseek / submission)
│   └── competitive_ratio/ #   full-run competitive-ratio JSON dumps
└── trials/                # ← every experiment, every dead end, every zip
    ├── README.md          #   index of the trials directory
    ├── submissions/       #   submission_v2 … v18 (the chain that produced v15)
    ├── feedbacks/         #   grader output per version (predictions + scores)
    ├── analysis/          #   compare_*, sweep_*, ablate_*, dashboard, grader_sim
    ├── first_versions/    #   the very first hand-rolled prototypes
    │   ├── old/           #     pre-versioning experiments
    │   └── old-versions/  #     submission-V0 … V5 (numbered prototypes)
    ├── zips/              #   every .zip we ever uploaded to the grader
    └── docs/              #   STARE-LB writeup, energy-composite analysis, etc.
```

## 2. Quickstart

```bash
python -m pip install -r requirements.txt
git lfs pull                              # fetch the sample traces
python quickstart.py                      # score submission.py on Qwen3 / LmSys / EP32
python quickstart.py --model DS-R1        # DS-R1 sample (LmSys, EP64)
python quickstart.py --all-samples        # both committed traces
```

`quickstart.py` runs **Default**, **DeepSeek-EPLB**, and **`submission.py`**
on the same trace and prints PAR, total transit, transmission time,
total time, and competitive score (DeepSeek = 100).

To score a specific trial version, drop its file in alongside
`submission.py` and pass the matching algorithm name; the adapter in
`eplb_algorithms/__init__.py` also auto-resolves `submission_vN` aliases
out of `trials/submissions/`:

```bash
python dynamic_lb_simulator.py --algorithm submission_v15
```

## 3. The shipped algorithm at a glance

`submission.py` (≡ v15) stacks the following blocks, top to bottom in
`rebalance(...)`:

```
hotness window
   │
   ▼
recency-weighted load (v6)        ← linear ramp inside the window
   │  tail-only gate signal (v7-B)   ← last 75 % of the window
   ▼
EWMA across calls (v1)            ← α = 0.7
   │
   ▼
replicate-then-LPT pack (DeepSeek-EPLB core)
   │
   ▼
two-level placement anchor (v4-B)
   │  • device-label row anchor (greedy + 2-opt)
   │  • within-device slot anchor
   ▼
cycle detection over last K placements (v4-A, K = 2)
   │
   ▼
TCP-AIMD per-layer cost gate (v15)   ← the v15 win
   │  • residual-driven safety[L] update
   │  • gain-first commit order (v7-C)
   ▼
deployment + changed-layer priority list
```

Each block is independent and motivated by a peer-reviewed primitive
(Roberts 1959 EWMA, Graham 1969 LPT bound, Lin–Kernighan 1965 2-opt,
Glover 1989 tabu cycle detection, Jacobson 1988 TCP AIMD, …). The
per-block paper citations live in [`CITATIONS.md`](CITATIONS.md).

**Why these and not others?** Every block had to clear three bars:
*(i)* run inside the 80 ms-per-call budget (no per-call ILP), *(ii)*
operate inside the simulator's *replicate-and-place* action space (no
router or tensor-shard tricks), and *(iii)* show a positive, reproducible
delta on the official grader. The chain in §4 logs which ideas cleared
all three and which were reverted.

## 4. Everything we tried (version chain)

The table below summarises every version that produced a grader run.
Files live under [`trials/submissions/submission_vN.py`](trials/submissions/);
the corresponding grader outputs are in
[`trials/feedbacks/scoring_result_vN/`](trials/feedbacks/).

| Ver | Headline idea | Outcome | Status |
|---|---|---|---|
| v0 – v4 (pre-versioning) | First hand-rolled prototypes: EWMA + spike-clip + fixed PAR-gain gate; dual-EWMA drift detector; reward-as-gate; anchored device relabel. | Local-only smoke runs; established the v1 baseline shape. | Archived in [`trials/first_versions/`](trials/first_versions/) |
| **v1** (= old/submission.py) | EWMA-smoothed weights + DeepSeek core + skip-unchanged layers. | Grader baseline ≈ 110.86. | Baseline, kept conceptually |
| v2 | ARC-style cooldown + ghost-list re-weight (T2 boost + cooldown + ghost-hit). | Cut transit −27 % but inflated PAR (1.72 → 1.78). Score 110.86 → **109.37**. | Reverted |
| v2_lowest_trans | v2 retuned for *minimum transit*. | 391 184 moves (record low), but PAR still inflated. | Kept as the transit-floor reference; ideas reused in v18 |
| v3 | EWMA α = 1.0 (no smoothing) + force-first-deploy. | Score **110.366** (regressed from 110.86). Confirmed EWMA is load-bearing. | Reverted |
| v4 | + cycle detection over last K = 2 placements + two-level anchor (device-row + slot). | **110.877** — biggest single jump from a non-DS-EPLB idea (+0.51 over v3). | Kept |
| v5 | Drift-adaptive per-layer recency floor, always on. | **110.344**. Always-on drift hurt stable layers. | Reverted (knob kept off by default in v7) |
| v6 | Intra-window recency-weighted load (linear ramp). | **110.928** (+0.05 over v4). Pure win on non-stationary traces. | Kept |
| v7 | + tail-only gate signal (last 75 % of window) + gain-first commit order. | **110.934** (+0.006). Small but consistent across seeds. | Kept |
| v7_old | Pre-tail-gate snapshot of v7. | Used for ablation (`trials/analysis/ablate_v7.py`). | Reference |
| v8 | Minor refactor of v7 (no logic change). | **110.929**. | Within noise |
| v9 | EWMA history-depth sweep. | **110.910**. | Within noise |
| v10 | α sweep on EWMA (joint with v9). | **110.929**. | Within noise |
| v11 | Half-window split + early/late drift estimate. | **110.910**. | Within noise |
| v12 | Drift-keyed *a-priori* gate threshold. | **110.909**. Drift is a *guess*, not a measurement. | Reverted (motivated v15) |
| v13 | Linear-trend extrapolation on EWMA. | Within v10/v11 noise band. | Reverted |
| v14 | Drift-keyed gate with stronger prior. | **110.91**. Same problem as v12. | Reverted |
| **v15** | **TCP-AIMD per-layer adaptive gate driven by *prediction residual*.** Each layer's `safety[L]` shrinks multiplicatively after a correct PAR prediction and grows after a miss (Jacobson 1988 + adaptive Kalman residual). | **110.968** (+0.034 over v7). **Largest grader-confirmed win after v4's anchor.** | **Shipped** |
| v16 | Trimmed-mean window estimator (10 % symmetric trim, Tukey 1960) + half-window drift. | Local sims even or slightly worse than v15; grader regressed to **99.90** (PAR blew up to 1.906 on volatile cases). | Reverted |
| v17 | Hungarian device-label anchor (`scipy.optimize.linear_sum_assignment`, Kuhn 1955 / Crouse 2016) + queue-delay-aware gate. | Local sims show −3 % transit at equal PAR; grader regressed to **100.14** (likely SciPy fallback path or queue-delay bug). | Reverted |
| v18 | v15 base + scaled-down v2 ARC overlays (cooldown + ghost-hit, clipped to safety ≤ 32). | Aimed at "best PAR (v15) ∧ best transit (v2)"; local sims still trail v15 on Mix/EP256. | Experimental, not graded |

Per-version one-line "why we tried it / what happened" notes are in
[`CITATIONS.md` §13](CITATIONS.md). The two largest grader wins in the
chain (v4 +0.51 and v15 +0.034) are the two ideas worth remembering:

1. **v4 — two-level placement anchor + cycle detection.** Both reduce
   transmit *without touching PAR*: the anchor permutes the new pack to
   maximise overlap with the previous deployment, and the cycle filter
   refuses to commit a layer placement we have already visited recently.
2. **v15 — TCP-AIMD per-layer cost gate.** The pre-v15 chain used a
   fixed `GATE_SAFETY = 16` (refuse a redeploy unless its predicted PAR
   gain is 16× the math break-even). v15 makes the per-layer safety
   *adapt to the prediction residual*: layers whose recent PAR
   predictions held up have their safety driven down toward 1, while
   layers whose predictions missed widely have safety bumped back up to
   16. The mechanism is straight out of TCP-Reno's AIMD and ARC cache
   ghost-list re-tuning — both bake in "evidence, not guesswork".

## 5. Score history (official grader)

Pulled directly from `trials/feedbacks/scoring_result_vN/scores.txt`:

| Version | Score | Mean PAR | Transmit | Note |
|---|---:|---:|---:|---|
| v5 | 110.344 | 1.7308 | 569 908 | Reverted (drift-adapt always-on) |
| v3 | 110.366 | 1.7295 | 572 157 | Reverted (α = 1.0) |
| v4 | 110.877 | 1.7220 | 526 710 | + cycle detection + two-level anchor |
| v11 | 110.910 | 1.7198 | 523 932 | EWMA / history sweep |
| v9 | 110.910 | 1.7198 | 523 932 | EWMA / history sweep |
| v12 | 110.909 | 1.7199 | 524 554 | Drift-keyed gate (rejected) |
| v6 | 110.928 | 1.7194 | 520 951 | + recency-weighted window |
| v8 | 110.929 | 1.7195 | 520 024 | minor refactor of v7 |
| v10 | 110.929 | 1.7194 | 520 024 | EWMA / history sweep |
| v7 | 110.934 | 1.7193 | 524 353 | + tail-gate + gain-first commit |
| **v15** | **110.968** | **1.7184** | **531 587** | **SHIPPED: + TCP-AIMD per-layer gate** |
| v16 | 99.896 | 1.9063 | 1 164 142 | Trimmed-mean + half-window drift (regressed) |
| v17 | 100.137 | 1.9023 | 1 161 471 | Hungarian anchor + queue-delay gate (regressed) |

The Mix and DS-R1 EP128/EP256 cases dominate the grader score, which is
why v16 and v17's PAR blowups on those cases tanked their composites
even though both improved transit-per-PAR on the easier cases.

A handful of pre-versioning grader runs are also preserved (early v1-
shaped variants run before the numbered chain started):

| Folder | Score | Note |
|---|---:|---|
| `scoring_result_+ EWMA + two-level anchor.` | 104.796 | First grader run, baseline shape |
| `scoring_result_109_..._4_gate` | 109.250 | 4-gate tuning, pre-numbering |
| `scoring_result_110_..._16_gate` | 110.855 | 16-gate (= v7 default), pre-numbering |
| `scoring_result_lowest_trans` | 109.369 | v2_lowest_trans's transit-floor run (391 184 moves) |

## 6. Ideas surveyed but **not** adopted

Catalogue of state-of-the-art techniques we read about and ruled out
(detailed citations in [`CITATIONS.md` §12](CITATIONS.md)):

| Algorithm | Why not adopted |
|---|---|
| **SwiftBalancer** (async expert movement, vLLM-Ascend) | Simulator runs the move synchronously inside the cadence call — async pipelining wins us nothing. |
| **NVIDIA Wide-EP / TensorRT-LLM EPLB** | We *are* the online EPLB path; offline EPLB is forbidden by the competition's online-traces model. |
| **HarMoEny** (token-level rebalancing + async prefetch) | Acts at the *router*, not the placement table. Out of action space. |
| **ExFlow** (inter-layer ILP affinity) | Per-call Gurobi ILP busts the 80 ms / call budget; ExFlow itself runs offline. |
| **BIP-based routing** (Sun 2025) | Training-time only; changes gating, not serving placement. |
| **MoEShard** (tensor-shard every expert) | Replaces the replicate-and-place action space entirely. |
| **ReaLB** (per-rank GEMM precision) | Simulator is precision-agnostic. |

## 7. Reproducing a trial run locally

Every script under `trials/analysis/` was used at least once during the
hack; the most useful ones are:

| Script | What it does |
|---|---|
| `trials/analysis/grader_sim.py` | Run the official grader's scoring path on a local trace. |
| `trials/analysis/compare_v1_v15.py` | Side-by-side PAR / transmit trace comparison v1 vs v15. |
| `trials/analysis/compare_v15_v18.py` | Same, v15 vs v18 (the ARC-overlay experiment). |
| `trials/analysis/compare_v6_v7.py` / `compare_v10_v12.py` | Granular mid-chain comparisons. |
| `trials/analysis/sweep_gate.py` | Grid-sweep the fixed `GATE_SAFETY` (pre-v15). |
| `trials/analysis/sweep_tail.py` | Grid-sweep `TAIL_FRACTION` (v7-B). |
| `trials/analysis/ablate_v7.py` | Turn each v7 block on/off individually. |
| `trials/analysis/check_drift.py` | Plot per-layer drift vs grader score. |
| `trials/analysis/check_deepseek_constraint.py` | Verify the DeepSeek runtime constraint isn't violated. |
| `trials/analysis/competitive_ratio.py` | Per-case 4/3-competitive-ratio breakdown (LPT bound). |
| `trials/analysis/dashboard.py` | Streamlit dashboard (interactive). |

All of them read traces from `trace/<model>/<dataset>.npy` and write to
`output/`.

## 8. Where to look for what

| If you want… | …go to |
|---|---|
| The deliverable code | [`submission.py`](submission.py) |
| Why each block of the algorithm is there | [`CITATIONS.md`](CITATIONS.md) |
| The original one-pager design writeup | [`trials/docs/STARE-LB_writeup.md`](trials/docs/STARE-LB_writeup.md) |
| The energy-composite scoring derivation | [`ENERGY_COMPOSITE_SCORE.pdf`](ENERGY_COMPOSITE_SCORE.pdf) |
| A specific trial version's source | `trials/submissions/submission_vN.py` |
| Grader output for a specific version | `trials/feedbacks/scoring_result_vN/scores.txt` |
| Every zip we ever uploaded | `trials/zips/` |
| Pre-versioned hand-rolled prototypes | `trials/first_versions/` |
| Per-case PAR / transmit / EP-scaling figures | `output/figure/` |
| Per-case CSV summary tables | `output/summary/` |

---

*If you're reviewing this submission: start with [`submission.py`](submission.py)
and [`CITATIONS.md`](CITATIONS.md). Everything else is the trail of how we
got there.*
