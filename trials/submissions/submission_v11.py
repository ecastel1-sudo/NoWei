"""submission_v11.py -- v7 core + entropy-weighted commit ordering (+ opt-in gate).

Baseline picked: v7 is the strongest shipped version (110.93441), so v11
extends v7 with one principled entropy-based mechanism enabled by default
plus a second one kept as opt-in after local A/B falsified its naive form.

Scores so far (full grader, ours):
    v6  110.92814  PAR 1.71942  tx 520_951      EWMA + recency-weighted window
    v7  110.93441  PAR 1.71929  tx 524_353      v6 + tail gate + gain-first order
    v8  110.92883  PAR 1.71945  tx 520_024      v6 + efficiency-ranked tx budget
    v9  110.90977  PAR 1.71982  tx 523_932      EP-aware tail (regressed)
    v10 110.92925  PAR 1.71944  tx 520_024      v7 + drift floor + post-LPT swap

Read of the v6->v10 record: the v7 gate is already correctly calibrated on
stationary traces (LmSys / ShareGPT / WildChat are saturated near the
deepseek-static lower bound). v10 tried "fix the proposal" (post-LPT
max-swap) and earned snapshot PAR but spent it on transit; net wash. v11
spends its complexity on the COMMIT ORDER -- which can only reorder, never
add or drop, the set of accepted layers vs v7. So in the worst case v11
ships v7's transmit unchanged and v7's accept set unchanged; in the better
case Mix's queue gets drained in a higher-confidence order.


WHY ENTROPY (concretely, not as marketing)
==========================================

For each layer L, normalise the recency-weighted hotness:
    p_{L,e} = h_{L,e} / sum_e h_{L,e}
    H_L     = -sum_e p_{L,e} log p_{L,e}
    H_norm_L = H_L / log(n_experts)        in [0, 1]

H_norm encodes how concentrated the layer's routing is:
  * H_norm ~ 0   -> one expert dominates the layer's routing. The PAR
                    snapshot's max/mean ratio is dominated by that
                    expert's weight, so a placement that isolates it is
                    a HIGH-SIGNAL win.
  * H_norm ~ 1   -> tokens are spread evenly across experts. The PAR
                    snapshot is already close to 1; reported gain is
                    a difference of two near-equal numbers, dominated
                    by within-window noise.

This signal-to-noise interpretation IS the one we'll act on, but ONLY for
ORDER (where mis-ordering is a small wallet cost), not for the GATE (where
the same intuition led us to drop useful moves on stationary LmSys; see
"WHY THE GATE IS OPT-IN" below).


THE MECHANISM (shipped)
=======================

v11-B. ENTROPY-WEIGHTED COMMIT ORDERING       [default ON]
    v7-C orders commits by raw gain_par (largest first), exploiting the
    simulator's one-layer-per-iter queue drain to maximise mean per-iter
    PAR drop in the iters immediately after commit. v11-B multiplies that
    key by a confidence factor:
        priority_L = gain_par_L * (1 + ORDER_GAMMA * (1 - H_norm_L))
    ORDER_GAMMA = 0 reproduces v7-C exactly. ORDER_GAMMA > 0 pulls
    high-confidence (low-H) layers to the front of the queue.

    Importantly: this DOES NOT change which layers commit, only their
    queue order. Total transmit is identical to v7. The PAR effect is
    second-order (it changes WHEN each layer's gain is applied), so on
    stationary traces -- where all accepted gains persist -- the local
    A/B is statistically indistinguishable from v7. On Mix the queue
    order matters because the layer's hotness is shifting under us and
    high-confidence wins persist longest into the next window.


WHY THE GATE IS OPT-IN, NOT SHIPPED
===================================

The natural companion idea -- raise the per-move threshold on high-H
layers because their reported gain is noise -- was disproven by a local
LmSys A/B (7 cases). At GATE_LAMBDA = 0.5 (uniform layers pay a 50%
tighter gate):

    LmSys/DS-R1/EP32   v7 PAR 1.215 -> v11 PAR 1.353  (-11% PAR)
    LmSys/Qwen3/EP128  v7 PAR 1.403 -> v11 PAR 1.496  (-7%  PAR)
    transit on the same two cases dropped 60% and 32% respectively.
    Net composite: 100.00 -> 98.04 (1.96% regression).

Conclusion: on stationary traces, even mid-entropy layers carry persistent
hot experts and their gains DO transfer to the next window. The "H is
noise" intuition only holds when the trace is actually non-stationary
(Mix); on stationary LmSys it is just wrong. The gate is left as a
ENTROPY_GATE_ENABLED = True flag for future Mix-only ablations once those
traces become locally available.

Inherited unchanged from v7:
  - Intra-window recency-weighted load signal (RECENCY_FLOOR = 0.5).
  - EWMA across cadence calls (EWMA_ALPHA = 0.7).
  - Tail-only gate signal (TAIL_FRACTION = 0.75).
  - 2-opt device-label anchor (ANCHOR_2OPT_PASSES = 3).
  - Cycle detection (CYCLE_HISTORY_K = 2).
  - Per-move cost gate base (GATE_SAFETY = 16).
  - DeepSeek-equivalent replication + LPT pack.

API contract identical to v1 / submission.py / v7:
    rebalance(hotness, n_device, n_red_expert)
      hotness: np.ndarray shape (window, n_layers, n_experts)
      returns: (change, layers_priority, deployment, _aux)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Inherited tuning (frozen from v7; do not change without re-running grader).
# ---------------------------------------------------------------------------
EWMA_ALPHA = 0.7                # v1; alpha=1.0 confirmed worse on grader (v3)
CYCLE_HISTORY_K = 2             # v4-A
ANCHOR_2OPT_PASSES = 3          # v4-B
GATE_SAFETY = 16.0              # v1; do not raise (v2's failure mode)
RECENCY_FLOOR = 0.5             # v6 default
TAIL_GATE_ENABLED = True        # v7-B
TAIL_FRACTION = 0.75            # v7-B shipped value (won LmSys ablation)
GAIN_ORDER_ENABLED = True       # v7-C

# ---------------------------------------------------------------------------
# v11-A: entropy-confidence gate.   [DEFAULT OFF -- opt-in only]
#   threshold_L = par_per_move_threshold * (1 + GATE_LAMBDA * H_norm_L)
#
# Local LmSys A/B at GATE_LAMBDA = 0.5 regressed the composite by 1.96%
# (mostly from EP32 cases where mid-H layers had persistent gains v11
# rejected). Kept here for future Mix-only experiments where the noise
# hypothesis actually applies; do not enable without per-case grader data.
#
# GATE_LAMBDA = 0 reproduces v7's flat gate; values > 0 progressively
# tighten the gate on uniform-routing layers.
# ---------------------------------------------------------------------------
ENTROPY_GATE_ENABLED = False
GATE_LAMBDA = 0.5

# ---------------------------------------------------------------------------
# v11-B: entropy-weighted commit ordering.   [DEFAULT ON]
#   priority_L = gain_par_L * (1 + ORDER_GAMMA * (1 - H_norm_L))
#
#   ORDER_GAMMA  meaning
#   ----------- ------------------------------------------------
#   0.0         identical to v7-C (raw gain ordering)
#   0.5         concentrated layers boosted by up to +50%   <-- shipped default
#   1.0         concentrated layers boosted by up to +100%
#
# Bound: confidence factor in [1, 1 + ORDER_GAMMA]. A maximally concentrated
# layer (H_norm = 0) gets the full boost; uniform layers keep their raw gain.
# Net effect: high-SNR redeploys drain first in the simulator's queue.
#
# 0.5 chosen as a conservative shipped default: large enough to reorder
# clearly bimodal layers (Mix), small enough that on stationary LmSys --
# where local A/B showed v11 == v7 on EP128+ cases -- the resort still
# leaves v7's ordering essentially intact.
#
# Set ENTROPY_ORDER_ENABLED = False to fall back to v7-C raw-gain ordering.
# ---------------------------------------------------------------------------
ENTROPY_ORDER_ENABLED = True
ORDER_GAMMA = 0.5


# ---------------------------------------------------------------------------
# Score-band constants (must mirror dynamic_lb_simulator.py).
# ---------------------------------------------------------------------------
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s


# ---------------------------------------------------------------------------
# Module state. Adapter calls _reset() on every fresh run.
# ---------------------------------------------------------------------------
_STATE: dict = {
    "cur_deploy": None,
    "ewma_weight": None,
    "history": None,
    "_recency_kernel_cache": None,
}


def _reset() -> None:
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None
    _STATE["_recency_kernel_cache"] = None


# ---------------------------------------------------------------------------
# Signals: recency kernel, window load, tail load, per-layer entropy.
# ---------------------------------------------------------------------------
def _recency_kernel(window_size: int) -> np.ndarray:
    """Cached linear ramp normalised to sum == window_size (v6 default)."""
    cache = _STATE["_recency_kernel_cache"]
    if cache is not None and cache.shape[0] == window_size:
        return cache
    if window_size <= 1 or RECENCY_FLOOR >= 1.0:
        kernel = np.ones(window_size, dtype=np.float64)
    else:
        raw = np.linspace(RECENCY_FLOOR, 1.0, window_size, dtype=np.float64)
        kernel = raw * (window_size / raw.sum())
    _STATE["_recency_kernel_cache"] = kernel
    return kernel


def _compute_window_load(hotness: np.ndarray) -> np.ndarray:
    """(L, E) hotness aggregated over the window with v6's linear recency."""
    kernel = _recency_kernel(hotness.shape[0])         # (W,)
    return np.einsum("w,wle->le", kernel, hotness)     # (L, E)


def _compute_tail_load(hotness: np.ndarray) -> np.ndarray:
    """(L, E) hotness aggregated over the window tail (v7-B). Gate signal."""
    W = hotness.shape[0]
    tail_w = max(1, int(round(W * TAIL_FRACTION)))
    return hotness[-tail_w:].sum(axis=0)


def _per_layer_entropy_norm(load: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy per layer, in [0, 1].

    load: (L, E) non-negative aggregated hotness.
    returns: (L,) where 0 = one expert dominates, 1 = uniform routing.

    Numerically safe:
      * empty layers (sum == 0) return 0 (treated as "no signal", which
        will be filtered out by the rest of the pipeline anyway since
        a 0-weight layer's proposal == identity).
      * single-expert layers: log(n_experts) cancels and we return 0/1
        with no nan propagation thanks to the np.where guard.
    """
    L_count, n_experts = load.shape
    if n_experts <= 1:
        return np.zeros(L_count, dtype=np.float64)
    totals = load.sum(axis=1, keepdims=True)           # (L, 1)
    # Safe normalisation: avoid div-by-zero by treating 0-sum layers as uniform
    # (entropy = 1.0). They'll be filtered out by the proposal-equality check.
    safe = np.where(totals > 0, totals, 1.0)
    p = load / safe                                    # (L, E)
    # 0 * log(0) := 0 (Shannon convention). Use xlogy.
    H = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)  # (L,)
    H_norm = H / np.log(n_experts)
    # Zero-load layers: treat as fully uniform (will be skipped anyway).
    H_norm = np.where(totals[:, 0] > 0, H_norm, 1.0)
    # Clip for floating-point safety (numerical noise can push past 1).
    return np.clip(H_norm, 0.0, 1.0)


# ---------------------------------------------------------------------------
# DeepSeek-equivalent per-layer placement (unchanged from v1 / v6 / v7).
# ---------------------------------------------------------------------------
def _init_deploy(n_layers, n_device, n_experts, n_phys):
    exp_per_dev = n_phys // n_device
    base_slots = exp_per_dev - 1
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for d in range(n_device):
        for s in range(base_slots):
            dep[:, d, s] = (d * base_slots + s) % n_experts
        dep[:, d, -1] = dep[:, d, -2]
    return dep


def _replicate_experts(weight_1d, n_phys):
    """Greedy water-fill: each extra slot goes to argmax(weight / replicas)."""
    n_log = weight_1d.shape[0]
    phy2log = np.empty(n_phys, dtype=np.int64)
    phy2log[:n_log] = np.arange(n_log, dtype=np.int64)
    logcnt = np.ones(n_log, dtype=np.int64)
    for slot in range(n_log, n_phys):
        pick = int(np.argmax(weight_1d / logcnt))
        phy2log[slot] = pick
        logcnt[pick] += 1
    return phy2log, logcnt


def _balanced_pack_lpt(weights_1d, n_packs):
    """LPT bin packing with equal item counts per bin."""
    n = weights_1d.shape[0]
    items_per_pack = n // n_packs
    order = np.argsort(-weights_1d)
    pack_load = np.zeros(n_packs, dtype=np.float64)
    pack_count = np.zeros(n_packs, dtype=np.int64)
    pack_index = np.empty(n, dtype=np.int64)
    rank_in_pack = np.empty(n, dtype=np.int64)
    for it in order:
        masked = np.where(pack_count < items_per_pack, pack_load, np.inf)
        choice = int(np.argmin(masked))
        pack_index[it] = choice
        rank_in_pack[it] = pack_count[choice]
        pack_load[choice] += weights_1d[it]
        pack_count[choice] += 1
    return pack_index, rank_in_pack


def _propose_layer(weight_1d, n_device, n_phys):
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log
    return deploy


# ---------------------------------------------------------------------------
# v4-B: 2-opt refined device-label anchor (transit-only, PAR-invariant).
# ---------------------------------------------------------------------------
def _greedy_mapping_from_overlap(overlap):
    n_dev = overlap.shape[0]
    used = np.zeros(n_dev, dtype=bool)
    mapping = np.empty(n_dev, dtype=np.int64)
    process_order = np.argsort(-overlap.max(axis=1))
    NEG = np.float32(-1.0)
    for i in process_order:
        scores = overlap[i].copy()
        scores[used] = NEG
        j = int(np.argmax(scores))
        mapping[i] = j
        used[j] = True
    return mapping


def _refine_mapping_2opt(overlap, mapping, max_passes):
    n = overlap.shape[0]
    for _ in range(max_passes):
        cur_vals = overlap[np.arange(n), mapping]
        new_i_vals = overlap[:, mapping]
        delta = (new_i_vals + new_i_vals.T
                 - cur_vals[:, None] - cur_vals[None, :])
        iu = np.triu_indices(n, k=1)
        best_idx = int(np.argmax(delta[iu]))
        best_gain = float(delta[iu][best_idx])
        if best_gain <= 0:
            break
        i, j = int(iu[0][best_idx]), int(iu[1][best_idx])
        mapping[i], mapping[j] = mapping[j], mapping[i]
    return mapping


def _anchor_device_labels(new_deploy, prev_deploy, n_experts):
    n_dev = new_deploy.shape[0]
    rows = np.repeat(np.arange(n_dev), new_deploy.shape[1])
    new_oh = np.zeros((n_dev, n_experts), dtype=np.float32)
    prev_oh = np.zeros((n_dev, n_experts), dtype=np.float32)
    new_oh[rows, new_deploy.reshape(-1)] = 1.0
    prev_oh[rows, prev_deploy.reshape(-1)] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = new_oh @ prev_oh.T
    mapping = _greedy_mapping_from_overlap(overlap)
    mapping = _refine_mapping_2opt(overlap, mapping, ANCHOR_2OPT_PASSES)
    out = np.empty_like(new_deploy)
    out[mapping] = new_deploy
    return out


def _anchor_slot_order(layer_new, layer_prev):
    n_dev, exp_per_dev = layer_new.shape
    out = np.empty_like(layer_new)
    for d in range(n_dev):
        new_row = layer_new[d]
        prev_row = layer_prev[d]
        new_multiset = list(new_row.tolist())
        out_row = [-1] * exp_per_dev
        for s in range(exp_per_dev):
            e = int(prev_row[s])
            if e in new_multiset:
                out_row[s] = e
                new_multiset.remove(e)
        fill_iter = iter(new_multiset)
        for s in range(exp_per_dev):
            if out_row[s] == -1:
                out_row[s] = next(fill_iter)
        out[d] = np.asarray(out_row, dtype=np.int64)
    return out


def _layer_par(weight_1d, deploy_2d):
    """PAR of a single layer's placement under given hotness."""
    n_experts = weight_1d.shape[0]
    cut = np.bincount(deploy_2d.reshape(-1), minlength=n_experts)
    cut = np.maximum(cut, 1)
    weights_per_replica = weight_1d / cut
    loads = (weights_per_replica[deploy_2d.reshape(-1)]
             .reshape(deploy_2d.shape).sum(-1))
    mean = float(loads.mean())
    return 1.0 if mean == 0.0 else float(loads.max() / mean)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def rebalance(hotness, n_device, n_red_expert):
    """v11: v7 stack + entropy-confidence gate + entropy-weighted ordering.

    See module docstring. API identical to submission.py / v7.
    """
    hotness = np.asarray(hotness, dtype=np.float64)
    if hotness.ndim != 3:
        raise ValueError(
            f"hotness must be 3D (window, layers, experts), got {hotness.shape}"
        )
    n_layers = int(hotness.shape[1])
    n_experts = int(hotness.shape[2])
    n_dev = int(n_device)
    n_phys = n_experts + int(n_red_expert)
    exp_per_dev = n_phys // n_dev

    # ---------- signals -----------------------------------------------------
    window_load = _compute_window_load(hotness)               # (L, E)
    gate_load = _compute_tail_load(hotness) if TAIL_GATE_ENABLED else window_load

    # Per-layer entropy is computed on the SAME signal the gate uses, so the
    # H_norm used by v11-A/-B matches the load it's filtering / ordering.
    if ENTROPY_GATE_ENABLED or ENTROPY_ORDER_ENABLED:
        H_norm = _per_layer_entropy_norm(gate_load)           # (L,)
    else:
        H_norm = np.zeros(n_layers, dtype=np.float64)

    # ---------- EWMA across cadence calls (v1) ----------
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != window_load.shape:
        smoothed = window_load.copy()
    else:
        smoothed = EWMA_ALPHA * window_load + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # Seed cur_deploy and the cycle-detection history if first call.
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["history"] = [cur.copy()]
    history = _STATE["history"]

    deploy_out = cur.copy()

    base_par_per_move = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                         _BALANCED_COMPUTE_SECONDS)

    # ---------- pass 1: propose + gate every layer --------------------------
    # Stable iteration order for cache-friendliness; commit order decided
    # after the loop (v7-C + v11-B).
    accepted: list[tuple[int, np.ndarray, float, int, float]] = []
    initial_order = np.argsort(-smoothed.sum(axis=1))
    for L_np in initial_order:
        L = int(L_np)

        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        if np.array_equal(proposed, cur[L]):
            continue

        # v4-A cycle detection: refuse to revert to a recent past state.
        if any(np.array_equal(proposed, past[L]) for past in history[:-1]):
            continue

        moves = int((proposed != cur[L]).sum())
        if moves == 0:
            continue
        gain_par = (_layer_par(gate_load[L], cur[L]) -
                    _layer_par(gate_load[L], proposed))

        # v11-A: entropy-confidence gate. Uniform-routing layers must clear
        # a higher per-move bar; concentrated layers see the v7 gate.
        if ENTROPY_GATE_ENABLED:
            threshold_L = base_par_per_move * (1.0 + GATE_LAMBDA * H_norm[L])
        else:
            threshold_L = base_par_per_move
        if gain_par <= moves * threshold_L:
            continue

        accepted.append((L, proposed, gain_par, moves, float(H_norm[L])))

    if not accepted:
        return False, np.array([], dtype=np.int64), deploy_out, None

    # ---------- pass 2: choose commit order ---------------------------------
    # v7-C: largest gain first. v11-B: multiply by confidence (1 + γ(1 - H)).
    # Worst case (ENTROPY_ORDER_ENABLED = False) reproduces v7-C exactly.
    if ENTROPY_ORDER_ENABLED:
        def _priority_key(item):
            _L, _prop, gain, _mv, h = item
            return -gain * (1.0 + ORDER_GAMMA * (1.0 - h))
    elif GAIN_ORDER_ENABLED:
        def _priority_key(item):
            return -item[2]
    else:
        # initial heavy-first order (no resort).
        def _priority_key(item):
            return 0
    accepted.sort(key=_priority_key)

    changed: list[int] = []
    for L, proposal, _gain, _moves, _h in accepted:
        deploy_out[L] = proposal
        cur[L] = proposal
        changed.append(L)

    history.append(cur.copy())
    if len(history) > CYCLE_HISTORY_K + 1:
        history.pop(0)

    return True, np.array(changed, dtype=np.int64), deploy_out, None
