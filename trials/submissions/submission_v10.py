"""submission_v10.py -- v6/v7 stack + drift-adaptive recency + post-LPT swap.

Score on grader so far:
  v6:  110.928   (PAR 1.7194, transmit 520_951)  -- best baseline
  v7:  110.934   (PAR 1.7193, transmit 524_353)  -- tiny win on top of v6
  v8:  110.929   (PAR 1.7195, transmit 520_024)  -- wash; transmit budget removed

Where the score is bleeding (from grader EP-scaling plot, v6/v7):

  DS-R1 Mix EP32   -> EP64   -> EP128   -> EP256
       PAR 1.65       2.04        2.69        3.66
  DS-R1 LmSys EP32 -> EP64   -> EP128   -> EP256
       PAR 1.28       1.40        1.63        2.03

The four DS-R1 Mix cases alone contribute ~0.40 to mean_PAR (out of ~1.72)
while every other case sits near the deepseek-static lower bound. Stationary
traces (LmSys / ShareGPT / WildChat) are saturated; Mix is the only big
remaining lever.

Two orthogonal mechanisms on top of v7:

  v10-A. PER-LAYER DRIFT-ADAPTIVE RECENCY FLOOR (quadratic falloff)
         v6/v7 use a single global RECENCY_FLOOR = 0.5 for every layer
         on every dataset. Mix has within-window drift -> we want a
         sharper ramp on those layers; LmSys/ShareGPT/WildChat are
         stationary -> the v6 floor is already tuned.

         Per-layer drift signal (L1 first-half vs second-half):
           drift_L = ||hot_late_L - hot_early_L||_1
                     / (||hot_early_L||_1 + ||hot_late_L||_1)         in [0, 1]

         Floor mapping (QUADRATIC, stationary-safe):
           floor_L = MAX - (MAX - MIN) * drift_L^2

         At drift_L = 0  -> floor = MAX (== v6's 0.5).
         At drift_L = 0.3 -> floor = 0.5 - 0.045 = 0.455 (effectively v6).
         At drift_L = 0.5 -> floor = 0.375.
         At drift_L = 1   -> floor = MIN (0.0; pure ramp, oldest iter ignored).

         Quadratic instead of linear (v7-A's failed shipped attempt) so
         low-drift layers behave indistinguishably from v6 -- the local
         LmSys ablation in v7-A regressed -0.05% precisely because the
         linear ramp nudged stationary layers' floor below 0.5 too fast.

  v10-B. POST-LPT 2-OPT MAX-LOAD SWAP REFINEMENT (PAR-monotonic).
         LPT is only 4/3-approximate for makespan, and with exp_per_dev = 2
         (DS-R1 EP=256 has 2 slots / device) it is essentially trivial:
         every device just gets its 2 highest-weighted available items.
         This is exactly where Mix EP256 PAR explodes (PAR = 3.66).

         The fix: after _balanced_pack_lpt produces an initial deployment,
         run a small loop that picks the heaviest device and tries every
         (slot_in_heavy, slot_in_other_device) swap. If any swap strictly
         reduces max-load, commit it; recompute loads; repeat. Bounded
         iterations + tiny per-call cost (n_dev * exp_per_dev^2 numpy ops).

         PAR-monotonic by construction: commit only if max-load strictly
         decreases. So this can ONLY help PAR; transmit may rise modestly
         because the refined layout differs more from cur[L], but the
         existing v6 cost gate still rejects any layer where the moves
         outweigh the PAR gain. Net: free PAR on the worst-packed layers.

Inherited unchanged from v7:
  - Tail gate (TAIL_FRACTION = 0.75) -- evaluate cost-gate ΔPAR on the
    last 75% of the window so the gate uses fresh signal.
  - Gain-first commit ordering -- the largest-ΔPAR layer is drained
    first by the simulator queue, maximising mean per-iter PAR drop.
  - 2-opt device-label anchor (ANCHOR_2OPT_PASSES = 3) -- transit-only.
  - Cycle detection (CYCLE_HISTORY_K = 2).
  - Per-move cost gate (GATE_SAFETY = 16).
  - EWMA across cadence calls (EWMA_ALPHA = 0.7).

API contract identical to v1 / submission.py:
  rebalance(hotness, n_device, n_red_expert) ->
      (change, layers_priority, deployment, _aux)
  hotness shape = (window, n_layers, n_experts).
"""

from __future__ import annotations

import numpy as np

# ---------- v4/v6 inherited tuning ------------------------------------------
EWMA_ALPHA = 0.7              # v1; alpha=1.0 confirmed worse on grader (v3)
CYCLE_HISTORY_K = 2           # v4-A
ANCHOR_2OPT_PASSES = 3        # v4-B
GATE_SAFETY = 16.0            # v1; do not raise (v2's failure mode)

# ---------- v10-A: drift-adaptive recency floor (quadratic) -----------------
# Layers are scored by within-window drift (L1 distance between first-half
# and second-half hotness sums, normalised by total). Stationary layers sit
# at drift_L ~ 0; Mix layers at drift_L >> 0.
#
# floor_L = RECENCY_FLOOR_MAX
#         - (RECENCY_FLOOR_MAX - RECENCY_FLOOR_MIN) * drift_L^DRIFT_POWER
#
# DRIFT_POWER = 2 keeps stationary layers near v6 (floor ~ 0.5) and only
# sharpens recency on layers that actually drift. v7-A used POWER=1 and
# regressed -0.05% locally on LmSys; quadratic is the conservative fix.
RECENCY_FLOOR_MAX = 0.5       # v6 default; floor at drift = 0
RECENCY_FLOOR_MIN = 0.0       # pure recency ramp; floor at drift = 1
DRIFT_POWER = 2.0             # quadratic -> stationary-safe

# ---------- v10-B: post-LPT swap refinement ---------------------------------
# Greedy max-load swap. SWAP_MAX_PASSES bounds iteration count per layer.
# Each pass is O(n_dev * exp_per_dev^2) vectorised; for DS-R1 EP=256 that
# is 256 * 4 = 1024 ops per pass; for Qwen3 EP=32 it is 32 * 64 = 2048
# ops per pass. Dwarfed by the rest of the proposal cost.
SWAP_MAX_PASSES = 8           # almost always converges in <= 4 passes

# ---------- v7-B: tail gate -------------------------------------------------
# Fraction of the window used as the gate's "what will happen next" signal.
# 0.75 was v7's shipped value (best on local LmSys sweep).
TAIL_GATE_ENABLED = True
TAIL_FRACTION = 0.75

# ---------- v7-C: gain-first commit ordering -------------------------------
GAIN_ORDER_ENABLED = True

# ---------- score-band constants -------------------------------------------
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s


# ---------- module state ----------------------------------------------------
_STATE: dict = {
    "cur_deploy": None,
    "ewma_weight": None,
    "history": None,
}


def _reset() -> None:
    """Clear module state. Adapter calls this on every fresh run."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None


# ---------- v10-A: drift-adaptive per-layer recency kernel -----------------
def _per_layer_drift(hotness: np.ndarray) -> np.ndarray:
    """L1 first-half vs second-half drift per layer, in [0, 1].

    hotness: (W, L, E)
    returns: (L,) -- 0 = stationary, 1 = maximally non-stationary.
    """
    half = max(1, hotness.shape[0] // 2)
    early = hotness[:half].sum(axis=0)             # (L, E)
    late = hotness[half:].sum(axis=0)              # (L, E)
    diff = np.abs(late - early).sum(axis=1)        # (L,)
    total = (early + late).sum(axis=1)             # (L,)
    return diff / np.maximum(total, 1e-9)


def _per_layer_kernel(window_size: int,
                      floor_per_layer: np.ndarray) -> np.ndarray:
    """Per-layer linear ramp kernel of shape (L, W), normalised so the sum
    along W equals window_size for every layer.

    floor_per_layer: (L,) in [0, 1].
    """
    if window_size <= 1:
        return np.ones((floor_per_layer.shape[0], window_size),
                       dtype=np.float64)
    t = np.linspace(0.0, 1.0, window_size, dtype=np.float64)   # (W,)
    f = floor_per_layer[:, None]                               # (L, 1)
    raw = f + (1.0 - f) * t[None, :]                           # (L, W)
    norms = raw.sum(axis=1, keepdims=True)                     # (L, 1)
    return raw * (window_size / np.maximum(norms, 1e-9))


def _compute_window_load(hotness: np.ndarray) -> np.ndarray:
    """(L, E) hotness aggregated over the window with the drift-adaptive
    per-layer kernel applied. Stationary layers see v6's flat-floor kernel;
    drifty layers see a sharper recency ramp.
    """
    _, n_layers, _ = hotness.shape
    drift = _per_layer_drift(hotness)                          # (L,)
    spread = RECENCY_FLOOR_MAX - RECENCY_FLOOR_MIN
    floor = RECENCY_FLOOR_MAX - spread * (drift ** DRIFT_POWER)
    floor = np.clip(floor, RECENCY_FLOOR_MIN, RECENCY_FLOOR_MAX)
    kernel = _per_layer_kernel(hotness.shape[0], floor)        # (L, W)
    # einsum: sum_w kernel[L, w] * hotness[w, L, e] -> (L, E)
    return np.einsum("lw,wle->le", kernel, hotness)


def _compute_tail_load(hotness: np.ndarray) -> np.ndarray:
    """(L, E) hotness aggregated over the window tail only; gate signal."""
    W = hotness.shape[0]
    tail_w = max(1, int(round(W * TAIL_FRACTION)))
    return hotness[-tail_w:].sum(axis=0)


# ---------- deepseek-equivalent placement (numpy) -- unchanged from v1 -----
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


# ---------- v10-B: post-LPT 2-opt max-load swap refinement -----------------
def _refine_packing_max_swap(deploy: np.ndarray,
                              per_replica_weight: np.ndarray) -> np.ndarray:
    """Greedy pair-swap to reduce max-load on a single layer's deployment.

    deploy: (n_dev, exp_per_dev) int64, one logical-expert id per slot.
    per_replica_weight: (n_log,) precomputed weight per logical expert /
        replica count (broadcast via deploy indexing).

    Each iteration:
      1. Find the heaviest device h_max with load = cur_max.
      2. Compute, in a SINGLE vectorised numpy block of shape
         (n_dev, exp_per_dev, exp_per_dev), the new max-load after every
         possible (s_in_h_max, s_in_other_device) swap.
      3. If the global minimum over that block strictly beats cur_max,
         commit that swap. Otherwise stop.

    PAR-monotonic by construction: max-load is non-increasing.

    Replica counts (and therefore per_replica_weight) are invariant under
    swap, so we precompute once outside the loop.

    Performance: each pass is one O(n_dev * exp_per_dev^2) numpy block
    rather than a Python-level loop over devices -- ~10x faster than the
    naive per-device loop on DS-R1 EP=256 (n_dev = 256).
    """
    n_dev, exp_per_dev = deploy.shape
    deploy = deploy.copy()

    for _ in range(SWAP_MAX_PASSES):
        slot_weight = per_replica_weight[deploy]               # (D, S)
        loads = slot_weight.sum(axis=1)                        # (D,)
        h_max = int(loads.argmax())
        cur_max = float(loads[h_max])

        sw_max = slot_weight[h_max]                            # (S,)

        # Vectorised: for every device h and every (s_max, s_h) pair,
        #   delta[h, s_max, s_h] = sw_h[h, s_h] - sw_max[s_max]
        #   new_load_h_max[h, s_max, s_h] = cur_max + delta
        #   new_load_h    [h, s_max, s_h] = loads[h] - delta
        #   new_max       [h, s_max, s_h] = max(new_load_h_max, new_load_h)
        delta = slot_weight[:, None, :] - sw_max[None, :, None]   # (D, S, S)
        new_load_h_max = cur_max + delta
        new_load_h = loads[:, None, None] - delta
        new_max = np.maximum(new_load_h_max, new_load_h)          # (D, S, S)

        # Disqualify swapping with the heavy device against itself.
        new_max[h_max] = np.inf

        flat_argmin = int(new_max.argmin())
        best_new_max = float(new_max.flat[flat_argmin])
        if best_new_max >= cur_max:
            break

        h, s_max_idx, s_h = np.unravel_index(flat_argmin, new_max.shape)
        h = int(h)
        s_max_idx = int(s_max_idx)
        s_h = int(s_h)
        deploy[h_max, s_max_idx], deploy[h, s_h] = (
            deploy[h, s_h], deploy[h_max, s_max_idx]
        )

    return deploy


def _propose_layer(weight_1d: np.ndarray, n_device: int,
                   n_phys: int) -> np.ndarray:
    """LPT pack -> post-LPT max-swap refinement (v10-B)."""
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log

    # v10-B: per-replica weight is constant under swap (cut doesn't change).
    # Pre-bind it so the swap inner loop is purely indexing + arithmetic.
    n_log = weight_1d.shape[0]
    cut = np.bincount(deploy.reshape(-1), minlength=n_log).clip(min=1)
    per_replica = weight_1d / cut
    deploy = _refine_packing_max_swap(deploy, per_replica)
    return deploy


# ---------- v4-B: 2-opt refined device-label anchor (transit-only) ---------
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
    """Iterative pair-swap to maximise sum(overlap[i, mapping[i]]).
    PAR-invariant (only relabels devices)."""
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
    """v10: v6/v7 stack + drift-adaptive recency + post-LPT swap.

    See module docstring. API identical to submission.py.
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

    # ---------- v10-A: drift-adaptive recency-weighted window load ----------
    window_load = _compute_window_load(hotness)              # (L, E)

    # ---------- v7-B: tail-only signal used by the cost gate ----------
    gate_load = _compute_tail_load(hotness) if TAIL_GATE_ENABLED else window_load

    # ---------- EWMA across cadence calls (unchanged from v1) ----------
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
    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                              _BALANCED_COMPUTE_SECONDS)

    # ---------- pass 1: propose + gate every layer, store gains ----------
    initial_order = np.argsort(-smoothed.sum(axis=1))
    accepted: list[tuple[int, np.ndarray, float, int]] = []
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

        # v6 cost gate: the swap-refined proposal can only improve PAR vs
        # the pre-swap LPT pack, so gain_par here is at worst v6's gain --
        # most of the time strictly greater on Mix.
        moves = int((proposed != cur[L]).sum())
        if moves == 0:
            continue
        gain_par = (_layer_par(gate_load[L], cur[L]) -
                    _layer_par(gate_load[L], proposed))
        if gain_par <= moves * par_per_move_threshold:
            continue

        accepted.append((L, proposed, gain_par, moves))

    # ---------- v7-C: pick commit order (largest gain first) ----------
    if GAIN_ORDER_ENABLED:
        accepted.sort(key=lambda x: -x[2])

    changed: list[int] = []
    for L, proposal, _gain, _moves in accepted:
        deploy_out[L] = proposal
        cur[L] = proposal
        changed.append(L)

    if changed:
        history.append(cur.copy())
        if len(history) > CYCLE_HISTORY_K + 1:
            history.pop(0)

    if not changed:
        return False, np.array([], dtype=np.int64), deploy_out, None

    return True, np.array(changed, dtype=np.int64), deploy_out, None
