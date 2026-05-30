"""submission_v7.py -- v6 + efficiency-ranked transmission budget.

Scoring at the leaderboard band (from per-case grader data):
    total_time ≈ 60 s * mean_PAR * n_cases  +  transmit_amount * 9.787e-5 s
               ≈ compute (~98%)              +  transit (~2%)

v6 already achieves low PAR via the recency-weighted pack, 2-opt anchor,
cycle detection, and cost gate. v7 focuses specifically on TRANSMISSION
TIME while preserving PAR quality.

Problem with v6 on bursty cadences (Mix, cold-start):
  v6 commits every gate-passing layer immediately, in heavy-load order.
  On bursty cadences many layers simultaneously pass the gate, creating
  a per-call transmission spike that is disproportionate to the PAR gain.
  Cutting those spikes is the main lever left for transmission reduction.

v7 solution -- two new mechanisms on top of the v6 stack:

  1. TWO-PASS COLLECT-THEN-COMMIT
     Pass 1: compute proposals for ALL layers, check gate + cycle detection,
     collect every layer that passes as a candidate (layer, proposed, moves,
     gain_par). No commits in pass 1.
     Pass 2: rank candidates by EFFICIENCY = gain_par / moves, then commit
     them in efficiency order subject to a transmission budget.

  2. EFFICIENCY-RANKED TRANSMISSION BUDGET
     Budget = TRANSMIT_BUDGET_FRACTION * n_layers * n_phys total slot
     changes per cadence call.
     Candidates are taken greedily in efficiency (PAR gain per slot change)
     order. If a candidate's moves would exceed the remaining budget it is
     skipped -- not abandoned; a smaller later candidate may still fit
     (0/1 knapsack with greedy efficiency ordering, NOT a simple truncation).

Why this specifically reduces transmission:
  - The least-efficient gate-passing redeployments (barely above the PAR
    threshold) consume disproportionate transmission relative to their PAR
    benefit. These are the ones the budget cuts.
  - On stable traces (few gate-passing layers) the budget never activates;
    behaviour is identical to v6.
  - On bursty traces the budget truncates per-call spikes: at most
    TRANSMIT_BUDGET_FRACTION * n_layers * n_phys slots change per call.

Bonus: layers_priority is returned sorted by PAR gain descending so the
simulator drains the most impactful redeployments first when it processes
the queue one layer per iteration.

Unchanged from v6:
  * Intra-window recency weighting (RECENCY_FLOOR = 0.5).
  * EWMA_ALPHA = 0.7 (confirmed better than 1.0 on full grader, v3 vs v6).
  * 2-opt refined device-label anchor (v4-B, ANCHOR_2OPT_PASSES = 3).
  * Cycle detection (v4-A, CYCLE_HISTORY_K = 2).
  * Per-layer cost gate at GATE_SAFETY = 16.
  * Skip-unchanged-layers guard.
  * Heavy-layers-first layer ordering for the proposal pass.

API contract identical to v1 / submission.py:
  rebalance(hotness, n_device, n_red_expert)
    hotness: np.ndarray shape (window, n_layers, n_experts)
    returns: (change, layers_priority, deployment, _aux)
"""

from __future__ import annotations

import numpy as np

# ---------- tuning constants ------------------------------------------------
EWMA_ALPHA = 0.7  # v1 default; alpha=1.0 confirmed to hurt grader (v3)

# Intra-window linear recency kernel (v6): oldest iter has weight
# RECENCY_FLOOR, newest has weight 1.0. Normalised so sum == window_size.
# RECENCY_FLOOR = 1.0 -> uniform (== v4). 0.5 -> newest 2× oldest (v6 default).
RECENCY_FLOOR = 0.5

CYCLE_HISTORY_K = 2       # v4-A: depth of cycle-detection history
ANCHOR_2OPT_PASSES = 3    # v4-B: max pair-swap passes on device-label anchor

_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s

GATE_SAFETY = 16.0  # v1 tuned; unchanged through v6

# Per-cadence transmission budget as a fraction of total physical slot-count.
#   budget_slots = TRANSMIT_BUDGET_FRACTION * n_layers * n_phys
# At 0.12: at most 12% of all physical slots may change per cadence call.
# Candidates are committed in efficiency (PAR_gain / moves) order until the
# budget is exhausted. On stable traces (few layers pass the gate) this is
# a no-op. On bursty traces it truncates per-call transmission spikes.
# Tune toward 0.20 to prioritise PAR over transit, toward 0.06 to save more
# transit (at mild PAR cost on the most bursty cadences).
TRANSMIT_BUDGET_FRACTION = 0.12


# ---------- module state ----------------------------------------------------
_STATE: dict = {
    "cur_deploy": None,              # (n_layers, n_dev, exp_per_dev) int64
    "ewma_weight": None,             # (n_layers, n_experts) float64
    "history": None,                 # list of past cur_deploy snapshots
    "_recency_kernel_cache": None,   # cached (window_size,) normalised ramp
}


def _reset() -> None:
    """Clear module state. Called by the adapter on every fresh run."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None
    _STATE["_recency_kernel_cache"] = None


def _recency_kernel(window_size: int) -> np.ndarray:
    """Cached linear ramp kernel normalised to sum == window_size."""
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


# ---------- deepseek-equivalent placement (numpy) ---------------------------
def _init_deploy(n_layers: int, n_device: int, n_experts: int,
                 n_phys: int) -> np.ndarray:
    exp_per_dev = n_phys // n_device
    base_slots = exp_per_dev - 1
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for d in range(n_device):
        for s in range(base_slots):
            dep[:, d, s] = (d * base_slots + s) % n_experts
        dep[:, d, -1] = dep[:, d, -2]
    return dep


def _replicate_experts(weight_1d: np.ndarray, n_phys: int):
    n_log = weight_1d.shape[0]
    phy2log = np.empty(n_phys, dtype=np.int64)
    phy2log[:n_log] = np.arange(n_log, dtype=np.int64)
    logcnt = np.ones(n_log, dtype=np.int64)
    for slot in range(n_log, n_phys):
        pick = int(np.argmax(weight_1d / logcnt))
        phy2log[slot] = pick
        logcnt[pick] += 1
    return phy2log, logcnt


def _balanced_pack_lpt(weights_1d: np.ndarray, n_packs: int):
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


def _propose_layer(weight_1d: np.ndarray, n_device: int,
                   n_phys: int) -> np.ndarray:
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log
    return deploy


# ---------- v4-B: 2-opt refined device-label anchor ------------------------
def _greedy_mapping_from_overlap(overlap: np.ndarray) -> np.ndarray:
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


def _refine_mapping_2opt(overlap: np.ndarray, mapping: np.ndarray,
                          max_passes: int) -> np.ndarray:
    """Iterative pair-swap to maximise sum(overlap[i, mapping[i]]). PAR-invariant."""
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


def _anchor_device_labels(new_deploy: np.ndarray, prev_deploy: np.ndarray,
                          n_experts: int) -> np.ndarray:
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


def _anchor_slot_order(layer_new: np.ndarray,
                       layer_prev: np.ndarray) -> np.ndarray:
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


def _layer_par(weight_1d: np.ndarray, deploy_2d: np.ndarray) -> float:
    n_experts = weight_1d.shape[0]
    cut = np.bincount(deploy_2d.reshape(-1), minlength=n_experts)
    cut = np.maximum(cut, 1)
    weights_per_replica = weight_1d / cut
    loads = (weights_per_replica[deploy_2d.reshape(-1)]
             .reshape(deploy_2d.shape).sum(-1))
    mean = float(loads.mean())
    return 1.0 if mean == 0.0 else float(loads.max() / mean)


# ---------- entry point -----------------------------------------------------
def rebalance(hotness, n_device: int, n_red_expert: int):
    """v7: v6 stack + efficiency-ranked transmission budget. See module docstring."""
    hotness = np.asarray(hotness, dtype=np.float64)
    if hotness.ndim != 3:
        raise ValueError(f"hotness must be 3D, got {hotness.shape}")
    window_size = int(hotness.shape[0])
    n_layers = int(hotness.shape[1])
    n_experts = int(hotness.shape[2])
    n_dev = int(n_device)
    n_phys = n_experts + int(n_red_expert)
    exp_per_dev = n_phys // n_dev

    # ---------- v6: intra-window recency-weighted load signal ----------
    kernel = _recency_kernel(window_size)                     # (W,)
    window_load = np.einsum("w,wle->le", kernel, hotness)     # (L, E)

    # ---------- EWMA across cadence calls (v1) ----------
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != window_load.shape:
        smoothed = window_load.copy()
    else:
        smoothed = EWMA_ALPHA * window_load + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # Seed cur_deploy and cycle-detection history on first call / shape change.
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["history"] = [cur.copy()]
    history = _STATE["history"]

    deploy_out = cur.copy()
    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                              _BALANCED_COMPUTE_SECONDS)

    # ---------- PASS 1: collect all gate-passing candidates ----------
    # All proposals are computed against the pre-call cur[] state so that
    # candidates are independent of each other. No commits happen here.
    #
    # Each candidate: (layer_idx, proposed_array, n_moves, par_gain, efficiency)
    # efficiency = par_gain / n_moves  (PAR improvement per slot change)
    candidates: list[tuple[int, np.ndarray, int, float, float]] = []

    layer_order = np.argsort(-smoothed.sum(axis=1))
    for L_np in layer_order:
        L = int(L_np)

        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        if np.array_equal(proposed, cur[L]):
            continue

        # v4-A cycle detection: refuse to revert to a recent committed state.
        # Check against history[:-1] (i.e., all snapshots except current).
        # Using any() with a generator short-circuits on first match.
        if any(np.array_equal(proposed, past[L]) for past in history[:-1]):
            continue

        moves = int((proposed != cur[L]).sum())
        gain_par = (_layer_par(window_load[L], cur[L]) -
                    _layer_par(window_load[L], proposed))

        if gain_par <= moves * par_per_move_threshold:
            continue

        efficiency = gain_par / moves  # always > par_per_move_threshold
        candidates.append((L, proposed, moves, gain_par, efficiency))

    if not candidates:
        return False, np.array([], dtype=np.int64), deploy_out, None

    # ---------- PASS 2: efficiency-ranked commit within budget ----------
    # Sort by efficiency descending: highest PAR gain per slot change first.
    candidates.sort(key=lambda c: -c[4])

    # Hard cap: at most TRANSMIT_BUDGET_FRACTION * n_layers * n_phys total
    # slot changes this cadence call. The budget scales with model size so
    # the fraction is meaningful across different EP / n_layers settings.
    budget = max(1, int(TRANSMIT_BUDGET_FRACTION * n_layers * n_phys))
    remaining = budget
    changed: list[int] = []
    gain_by_layer: dict[int, float] = {}

    for L, proposed, moves, gain_par, efficiency in candidates:
        if moves > remaining:
            # This candidate doesn't fit; try smaller ones (0/1 knapsack).
            continue
        deploy_out[L] = proposed
        cur[L] = proposed
        changed.append(L)
        gain_by_layer[L] = gain_par
        remaining -= moves

    if not changed:
        return False, np.array([], dtype=np.int64), deploy_out, None

    # Return layers_priority sorted by PAR gain descending so the simulator
    # drains the most impactful redeployments first when it processes the queue.
    changed.sort(key=lambda L: -gain_by_layer[L])

    history.append(cur.copy())
    if len(history) > CYCLE_HISTORY_K + 1:
        history.pop(0)

    return True, np.array(changed, dtype=np.int64), deploy_out, None
