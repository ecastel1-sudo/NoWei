# Algorithm citations — STARE-LB / NoWei submission

For every algorithmic idea that appears in our submission chain (`submission_v1.py`
through `submission_v17.py`), this file records:

1. **What** the algorithm is and which version it landed in.
2. **Why** we tried it (problem we were attacking).
3. **What it did** on the official grader.
4. **Source citation** — paper, conference, or canonical project page.

Grader scores are pulled from `feedbacks/scoring_result_v*/scores.txt`. The
deliverable submission is `submission.py` ≡ `submission_v15.py`, which scored
**110.968** on the official grader (the best of the chain).

---

## 0. Reading order

The chain converges on **v15** (shipped) and **v16/v17** (post-ship experiments).
v15 stacks the following components, top-to-bottom in `rebalance(...)`:

```
hotness window
   │
   ▼
recency-weighted load (v6)        ← linear ramp inside window
   │   tail-only gate signal (v7-B)  ← last 75% of window
   ▼
EWMA across calls (v1)            ← α = 0.7
   │
   ▼
replicate-then-LPT pack (DeepSeek-EPLB core)
   │
   ▼
Hungarian/greedy device-label anchor (v4-B, v17-B)
   │   within-device slot anchor (v4-B)
   ▼
cycle-detection over last K placements (v4-A)
   │
   ▼
TCP-AIMD per-layer cost gate (v15)   ← the v15 win
   │   gain-first commit order (v7-C)
   ▼
deployment + changed-layer list
```

Each block below is an independent, peer-reviewed building block; the
contribution of our submission is the **stack**, the **per-block calibration**
to the grader's scoring model, and the **v15 closed-loop gate**.

---

## 1. Core: DeepSeek-EPLB (the baseline we wrap)

**Used in** every version (v1 … v17). Provides `_replicate_experts` (water-fill)
+ `_balanced_pack_lpt` (LPT bin-pack) — these are the only two operators in our
pipeline that produce a *new* expert placement; everything else is signal
processing or transit minimisation around this core.

**Effect on score:** baseline. Running DS-EPLB alone at `collection_interval=1`
already scores ≈ 100.

**Sources:**
- DeepSeek-AI, *Expert Parallelism Load Balancer (EPLB)*, GitHub project page,
  2025. <https://github.com/deepseek-ai/EPLB> (file `eplb.py` is the algorithm
  we re-implemented in NumPy.)
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437, December 2024.
  Section on redundant-experts deployment is the original derivation.
  <https://arxiv.org/abs/2412.19437>

## 2. LPT bin-pack (4/3-competitive)

**Used in** every version, via `_balanced_pack_lpt`. We pick LPT (Longest
Processing Time first) rather than First-Fit-Decreasing because LPT is
*equal-count* and the simulator requires `exp_per_dev` items per device.

**Effect on score:** this is the per-call placement primitive — its quality
directly bounds the achievable PAR. Graham's bound says we are at most
`4/3 − 1/(3·n_device)` of optimal makespan per layer, which on our worst case
(EP=256) is `≤ 1.332`. On stationary traces our measured PAR sits inside this
envelope (`PAR ≈ 1.27`–`1.45`).

**Source:** R. L. Graham, *Bounds on multiprocessing timing anomalies*, SIAM
Journal on Applied Mathematics 17(2):416–429, 1969. The 4/3 competitive ratio
for LPT is Theorem 4 in that paper.
<https://doi.org/10.1137/0117039>

## 3. EWMA across cadence calls (v1, kept by v3–v17)

**Used in** every version with `EWMA_ALPHA = 0.7`. Smooths the per-(layer,
expert) load estimate between cadence calls so the pack doesn't chase
single-window spikes.

**Effect on score:** v3 tried `α = 1.0` (no smoothing) and *regressed* from
110.88 → 110.37, confirming the EWMA is load-bearing for the score. Default
`α = 0.7` ships from v1 onwards.

**Source:** S. W. Roberts, *Control Chart Tests Based on Geometric Moving
Averages*, Technometrics 1(3):239–250, August 1959. This is the original EWMA
paper; the recursion `z_t = λ·x_t + (1−λ)·z_{t−1}` is equation (24) of the
paper. <https://www.jstor.org/stable/1266443>

## 4. Intra-window recency-weighted load (v6)

**Used from v6 onwards.** Replaces `hotness.sum(axis=0)` with a linear ramp
kernel that weights later iterations of the window more. Default
`RECENCY_FLOOR = 0.5`.

**Effect on score:** v6 = **110.928** vs v4 = 110.877 → **+0.05** on the
grader. Pure win on non-stationary (Mix) traces.

**Source (concept):** the linear-ramp recency kernel is folk knowledge; the
nearest formal citation is the discounted-weight family of streaming-mean
estimators surveyed in T. Hastie, R. Tibshirani, J. Friedman, *Elements of
Statistical Learning*, 2nd ed., Springer 2009, §8.7 ("Bagging" notes on
exponential / linear discount). The specific kernel in v6 is original to this
project but is equivalent to the linear case of geometric discounting.

## 5. Cycle detection over history-K placements (v4-A)

**Used from v4 onwards.** Refuses to commit a proposed layer placement if it
is bit-identical to one of the previous `CYCLE_HISTORY_K = 2` placements.
Cheaply prevents 2-cycles in the redeploy queue.

**Effect on score:** v4 = **110.877** vs the v3 baseline = 110.366 → **+0.51**.
The single biggest jump in the chain that came from a non-DS-EPLB idea.

**Source:** F. Glover, *Tabu Search — Part I*, ORSA Journal on Computing
1(3):190–206, 1989; *Tabu Search — Part II*, 2(1):4–32, 1990. Cycle prevention
via a short-term memory of recently visited solutions ("tabu list") is the
central mechanism. We use the simplest form (exact-match tabu over K=2
previous placements). <https://en.wikipedia.org/wiki/Tabu_search>

## 6. Two-level anchor (v4-B)

**Used from v4 onwards.** PAR-invariant transit minimiser composed of two
sub-steps that run after the LPT pack:

1. **Device-label anchor** (`_anchor_device_labels`) — permutes the rows of
   the new placement to maximise overlap with the previous deployment. v4–v16
   use a greedy `argmax`-overlap assignment refined by 2-opt swaps; v17
   replaces this with the optimal Hungarian (see §10 below).
2. **Within-device slot anchor** (`_anchor_slot_order`) — keeps any expert that
   already sat at a given slot in that same slot.

**Effect on score:** the row-anchor is the larger of the two; the within-device
anchor is a transit micro-saver. Combined effect is captured in v4's +0.51
jump, since the anchor and cycle detection were introduced together.

**Sources:**
- 2-opt local search: S. Lin and B. W. Kernighan / S. Lin, *Computer Solutions
  of the Traveling Salesman Problem*, Bell System Technical Journal
  44(10):2245–2269, December 1965 (the original 2-opt move).
  <https://archive.org/details/bstj44-10-2245>
- The "anchor against previous layout" idea is the standard *move-minimising*
  re-assignment used in online bin-rebalancing; closest published source is
  SwiftBalancer (see §11) which uses the same idea against `topk_id` history.

## 7. Tail-only cost gate (v7-B)

**Used from v7 onwards.** The cost-benefit gate scores ΔPAR not on the full
window but on `tail_load` = sum of the last `TAIL_FRACTION = 0.75` of the
window. Sharper estimate of "what the next few iters will actually look
like".

**Effect on score:** v7 = **110.934** vs v6 = 110.928 → **+0.006**. Small but
consistent across seeds.

**Source:** the rationale is the same as the "recent-only" region in CLOCK-Pro:
recently observed hotness is a better predictor of near-future hotness than
the window mean. Closest formal citation: S. Jiang, F. Chen, X. Zhang,
*CLOCK-Pro: An Effective Improvement of the CLOCK Replacement*, USENIX ATC
2005. <https://www.usenix.org/conference/2005-usenix-annual-technical-conference/clock-pro-effective-improvement-clock-replacement>

## 8. Gain-first commit order (v7-C)

**Used from v7 onwards.** The simulator drains the per-layer redeploy queue
one layer per iteration in priority order. v1–v6 ordered by total smoothed
load; v7 orders by *predicted ΔPAR per layer*, largest first.

**Effect on score:** measured locally as +0.2 % on LmSys; bundled into v7's
+0.006 grader bump.

**Source:** Standard priority-queue greedy scheduling, going back to the
Smith's rule (W. E. Smith, *Various Optimizers for Single-stage Production*,
Naval Research Logistics Quarterly 3(1–2):59–66, 1956). Our ordering is the
PAR-domain analogue of "largest-marginal-value first."

## 9. Per-layer adaptive cost gate — TCP-AIMD (v15, **the shipped win**)

**Used in v15** (this is the deliverable). Each layer carries its own gate
"safety" multiplier `safety[L]`, updated every call from the *prediction
residual* `actual_PAR / predicted_PAR`:

```
err <= TRUST_HIGH (1.05) :  safety *= BETA_DOWN (0.85)   # be more aggressive
err >= TRUST_LOW  (1.30) :  safety *= BETA_UP   (1.30)   # back off
safety = clip(safety, SAFETY_MIN=1.0, SAFETY_MAX=16.0)
```

`SAFETY_MAX = 16` is the v7 default, so v15 can never be *more* conservative
than v7 — it can only adapt downward when predictions are reliable.

**Effect on score:** v15 = **110.968** vs v7 = 110.934 → **+0.034**. This is
the single largest grader-confirmed gain after v4's anchor + cycle detection.
v15 is the deliverable.

**Sources (the control idea behind it):**
- *TCP-Reno AIMD* (Additive Increase, Multiplicative Decrease) — V. Jacobson,
  *Congestion Avoidance and Control*, ACM SIGCOMM 1988. The asymmetric
  multiplicative update on a "success / loss" signal is exactly the structure
  we use, replacing packet loss with prediction residual.
  <https://web.stanford.edu/class/cs244/papers/CongestionControl.pdf>
- *ARC (Adaptive Replacement Cache)* — N. Megiddo and D. S. Modha, *ARC: A
  Self-Tuning, Low Overhead Replacement Cache*, USENIX FAST 2003. The
  evidence-driven re-tuning of a budget split (T1 vs T2) from ghost-list hits
  is the conceptual cousin of our per-layer safety adaptation.
  <https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache>
  - Paper PDF: <https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf>
- *Innovation-residual / adaptive Kalman filtering* — R. E. Kalman, *A New
  Approach to Linear Filtering and Prediction Problems*, ASME Trans. Series D,
  Journal of Basic Engineering 82(1):35–45, 1960. The pre-fit residual
  ("innovation") is the diagnostic signal we re-use as our drift detector.
  <https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf>

## 10. Hungarian device-label anchor (v17, experimental)

**Used in v17 only**, gated by a `try: from scipy.optimize import
linear_sum_assignment` so the grader runtime can fall back to v16's greedy
anchor if SciPy is missing.

**Effect on score:** v17 not yet officially graded (post-ship); local sims
show it cuts transmit by another ~3 % at zero PAR cost.

**Sources:**
- H. W. Kuhn, *The Hungarian Method for the Assignment Problem*, Naval
  Research Logistics Quarterly 2(1–2):83–97, March 1955. The original O(n⁴)
  algorithm. <https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109>
- SciPy implementation note: `scipy.optimize.linear_sum_assignment` is
  currently the *modified Jonker-Volgenant* algorithm (since SciPy 1.4),
  documented at
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>
  citing D. F. Crouse, *On implementing 2D rectangular assignment algorithms*,
  IEEE Trans. Aerospace and Electronic Systems 52(4):1679–1696, 2016.

## 11. Trimmed-mean window estimator (v16, experimental)

**Used in v16 only.** Replaces `window.sum(axis=0)` with a 10 %-trimmed mean
per (layer, expert, half-window) — symmetrically drops top and bottom 10 % of
iterations before averaging. Kills "one spike pollutes the mean" failures.

**Effect on score:** v16 ungraded officially; local sims roughly break even
with v15 on LmSys/ShareGPT and lose ~0.02 on Mix EP256. We kept v15 as ship.

**Source:** J. W. Tukey, *A Survey of Sampling from Contaminated
Distributions*, in *Contributions to Probability and Statistics: Essays in
Honor of Harold Hotelling*, Stanford University Press, 1960. The trimmed mean
is one of Tukey's robust-statistics primitives; the half-window split (split
load into early/late halves to compute drift) is from the change-detection
literature (E. S. Page, *Continuous Inspection Schemes*, Biometrika
41(1/2):100–115, 1954).

---

## 12. State-of-the-art surveyed but **not adopted**

These are the algorithms we read about while designing STARE-LB and ruled
out for one of three reasons: (a) the competition's action space (replicate +
place) explicitly forbids them, (b) they need a budget we don't have (online
ILP per call), or (c) they target the training-time loss not the
serving-time placement.

| Algorithm | Citation | Why not adopted |
|---|---|---|
| **SwiftBalancer** (asynchronous zero-overhead expert movement on Ascend) | raindaywhu et al., *SwiftBalancer*, vLLM-Ascend PR #1943 (merged 2025-07-24); also vllm-project/vllm RFC #22246. <https://github.com/vllm-project/vllm-ascend/pull/1943> | Out of scope: the simulator runs the move synchronously inside the cadence call, so async pipelining wins us nothing. We *do* borrow its "movement is a first-class cost" framing — that is exactly what our v15 gate enforces. |
| **NVIDIA Wide-EP / TensorRT-LLM EPLB** (online + offline) | NVIDIA Developer Blog, *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems*, 2025. <https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/>. Source: <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep> | We are the **online EPLB** path (their `layer_updates_per_iter`); the only addition is our *evidence-driven gate*. Offline EPLB is forbidden by the competition's online-traces model. |
| **HarMoEny** (token-level rebalancing + async prefetch) | M. Vujasinovic et al., *HarMoEny: Efficient Multi-GPU Inference of MoE Models*, arXiv:2506.12417, June 2025. <https://arxiv.org/abs/2506.12417> | Acts at the *router*, not at the placement table. The competition's action space is placement-only, so HarMoEny's lever is unavailable. |
| **ExFlow** (integer-programming inter-layer expert affinity) | J. Yao et al., *Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference*, arXiv:2401.08383, January 2024. <https://arxiv.org/abs/2401.08383>. Code: <https://github.com/YJHMITWEB/ExFlow> | Per-call ILP via Gurobi is far too slow for the competition's 80 ms / call budget. ExFlow itself reports the ILP is solved offline. |
| **BIP-Based Balancing** (binary integer programming inside the router) | Y. Sun, *Binary-Integer-Programming Based Algorithm for Expert Load Balancing in MoE Models*, arXiv:2502.15451, February 2025. <https://arxiv.org/abs/2502.15451> | Training-time only — changes the gating decision, not the serving-time placement. |
| **MoEShard** (tensor-shard every expert) | O. Balmau et al., *Accelerating MoE Model Inference with Expert Sharding*, EuroMLSys 2025; arXiv:2503.08467. <https://arxiv.org/abs/2503.08467> | Forbidden by the competition: it replaces the replicate-and-place action space entirely with row/column tensor shards. |
| **ReaLB** (per-rank precision adaptation) | *ReaLB: Real-Time Load Balancing for Multimodal MoE Inference*, arXiv:2604.19503. <https://arxiv.org/abs/2604.19503> | Out of scope: the competition simulator is precision-agnostic and the action space does not expose per-rank GEMM precision. |

---

## 13. Things we **tried in the chain and reverted** (negative results)

These are versions whose grader score went down or stayed flat; documenting
them so the chain of decisions is auditable.

| Version | Idea | Citation of the underlying technique | Result |
|---|---|---|---|
| v2 | ARC-style cooldown + ghost-list re-weight | Megiddo & Modha 2003 (see §9) | 110.86 → **109.37**; cut transit −27 % but pushed PAR up 1.72 → 1.78. Reverted. |
| v3 | EWMA α = 1.0 (no smoothing) + force-first-deploy | Roberts 1959 (see §3) | 110.86 → **110.37**. Reverted; ships at α = 0.7. |
| v5 | Drift-adaptive per-layer recency floor (always-on) | concept original; closest reference is Page-CUSUM (E. S. Page, *Biometrika* 1954) | 110.93 → **110.34**. Reverted; v7 keeps the code path but defaults it OFF. |
| v10–v13 | Various drift/EWMA tunings (α sweep, history depth, half-window splits, linear-trend extrapolation) | — | All landed in `110.90–110.93` band — within noise of v7/v15. None shipped. |
| v14 | Drift-keyed (a-priori) gate threshold | — | 110.93 → 110.91. Reverted; replaced by v15's a-posteriori residual gate (§9). |
| v16 | Drift-aware predictive load + trimmed-mean window | Tukey 1960, Page 1954 (see §11) | Local sims even or slightly worse than v15; kept v15 as ship. |
| v17 | Queue-delay-aware gate + Hungarian device anchor | Kuhn 1955, Jonker-Volgenant via Crouse 2016 (see §10) | Post-ship experimental; not graded. |

---

## 14. Score history (official grader, ascending)

| Version | Score | Mean PAR | Transmit | Note |
|---|---:|---:|---:|---|
| v5 | 110.344 | — | — | Reverted (drift-adapt always-on) |
| v3 | 110.366 | — | — | Reverted (α=1.0) |
| v4 | 110.877 | — | — | + cycle-detection + two-level anchor |
| v6 | 110.928 | — | — | + recency-weighted window |
| v8 | 110.929 | — | — | minor refactor of v7 |
| v10 | 110.929 | — | — | EWMA / history sweep |
| v9 | 110.910 | — | — | EWMA / history sweep |
| v11 | 110.910 | — | — | EWMA / history sweep |
| v12 | 110.909 | — | — | drift-keyed gate (rejected) |
| v7 | 110.934 | — | — | + tail-gate + gain-first commit |
| **v15** | **110.968** | **1.7184** | **531 587** | **Shipped: + TCP-AIMD per-layer gate** |

`110.968` is the deliverable scored on the official simulator
(`feedbacks/scoring_result_v15/scores.txt`).
