"""submission_v9.py -- v7 base + EP-aware tail, tail-in-pack, opt-in drift adapt.

Background. The graded v7 (TAIL_FRACTION = 0.25) scored 110.93, exactly
matching v6, with a per-case picture that pointed at exactly one
regression:

    Mix/DS-R1/256   Δpar +0.0133   Δtx +3755   Δt_s +1.17   (BACKFIRED)
    Mix/DS-R1/128   Δpar -0.0095   Δtx  -846   Δt_s -0.65   (helped)
    Mix/DS-R1/64    Δpar -0.0059   Δtx   -12   Δt_s -0.35   (helped)
    Mix/DS-R1/32    Δpar -0.0006   Δtx  +246   Δt_s -0.01   (~wash)
    20 other cases  near-zero deltas                        (~no-op)

Root cause analysis. At EP256/DS-R1 the physical-slot-per-device is
n_phys / n_dev = (256 + 256) / 256 = 2, the smallest in the grid. A
narrow tail (32/128 iters) is noisy at this replication factor, and
the cost gate's PAR comparison flipped sign on edge-case redeployments,
costing both PAR and transit.

v9 keeps the things that proved harmless on the grader (gain ordering,
recency floor 0.5, anchor refinement, cycle detection, GATE_SAFETY=16)
and replaces the brittle bits with three structural changes:

  v9-A. EP-AWARE TAIL WIDTH
        Instead of a single TAIL_FRACTION, the tail width is a function
        of the per-device replication factor (exp_per_dev):
            tail_fraction = clamp(0.5 + 1.0 / exp_per_dev, 0.5, 1.0)
        At EP256/DS-R1 (exp_per_dev=2): tail = 1.0 == full window
          -> the Mix EP256 backfire mode disappears (gate behaviour
             reverts to v6 there).
        At EP128/DS-R1 (exp_per_dev=3): tail ~ 0.83 (still fresh)
          -> the Mix EP128 win stays mostly intact.
        At low EP (exp_per_dev=8+): tail ~ 0.6 (sharper)
          -> sharpest gate where the signal is cleanest.

  v9-B. TAIL SIGNAL IN THE PACK INPUT, not only the gate
        v7 used tail only for the gate's PAR estimate. v9 also blends
        a recency-tail into the pack input that drives _replicate_experts
        and _balanced_pack_lpt:
            pack_input = (1 - β) * window_load + β * tail_load_rescaled
        On stationary traces tail ≈ window so β has no measurable
        effect. On Mix the pack itself sees surging experts, not just
        the gate, so it allocates more replicas to them and packs them
        away from each other. β = TAIL_PACK_BLEND, default 0.30.

  v9-C. DRIFT-ADAPTIVE RECENCY FLOOR (opt-in)
        Same mechanism as v7-A. Disabled by default because local
        ablation cost ~0.05% on stationary; left as a single-flag
        opt-in once we can measure Mix on grader.

Plus unchanged-from-v7:
  v9-D. GAIN-FIRST LAYER ORDERING (default on; free win)

API contract: rebalance(hotness, n_device, n_red_expert) -> (change,
priority, deploy, aux). Identical to every previous version.

Code organisation. Helpers are pure functions that take arrays and
return arrays. Module-level state is one dict whose keys are documented
in STATE_KEYS. Configuration is a flat block of module constants
grouped by purpose, with each constant carrying its empirical or
theoretical justification next to it. No silent defaults.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Scoring constants -- MUST match dynamic_lb_simulator.py / grader_sim.py.
# These drive the cost gate threshold and are NOT tunable.
# ---------------------------------------------------------------------------
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s


# ---------------------------------------------------------------------------
# Tunable configuration.
# Each block is a logically-independent ablation knob.
# Values that have been "burned in" by the grader carry a citation.
# ---------------------------------------------------------------------------

# v1-inherited: smoothing across cadence calls.
#   alpha = 1.0 disables smoothing; alpha < 1.0 carries past calls' weight.
#   v3 grader showed alpha = 1.0 regressed; 0.7 is the stable sweet spot.
EWMA_ALPHA: float = 0.7

# v6-inherited: linear ramp inside the collection window (oldest -> newest).
#   floor = 1.0 -> uniform (== v4); floor = 0.5 -> newest iter weighted 2x oldest.
#   Local ablation showed flat 0.5 outperforms uniform on every trace.
RECENCY_FLOOR: float = 0.5

# v4-inherited: anchor and cycle settings -- both proved PAR-invariant.
CYCLE_HISTORY_K: int = 2          # depth of past placements to refuse reverting to
ANCHOR_2OPT_PASSES: int = 3       # max pair-swap passes on device-label anchor

# v1-inherited: per-move cost gate safety multiplier on PAR/move pricing.
#   1.0 = pure scoring math, 16 = "needs strong PAR proof before redeploying".
#   v1 sweep + grader confirmed 16 is the leaderboard-band optimum.
GATE_SAFETY: float = 16.0

# --- v9-A: EP-aware tail width for the cost gate. ---------------------------
# tail_fraction = clamp(TAIL_BASE_FRACTION + TAIL_INV_SLOPE / exp_per_dev,
#                       TAIL_FRACTION_MIN, TAIL_FRACTION_MAX)
#
# Worked examples on the actual grader grid (n_red_expert = ep):
#   DS-R1 EP256, exp_per_dev = 2 -> 1.00   (== no tail, full window)
#   DS-R1 EP128, exp_per_dev = 3 -> 0.83
#   DS-R1 EP64,  exp_per_dev = 5 -> 0.70
#   DS-R1 EP32,  exp_per_dev = 9 -> 0.61
#   Qwen3 EP128, exp_per_dev = 2 -> 1.00
#   Qwen3 EP64,  exp_per_dev = 3 -> 0.83
#   Qwen3 EP32,  exp_per_dev = 5 -> 0.70
#
# Rationale: variance of the per-device load estimate scales like
# 1/exp_per_dev, so the gate's PAR comparison gets noisier as
# replication shrinks. The wider tail denoises exactly where we saw
# v7 backfire on the grader (Mix EP256, exp_per_dev = 2).
TAIL_GATE_ENABLED: bool = True
TAIL_BASE_FRACTION: float = 0.5
TAIL_INV_SLOPE: float = 1.0
TAIL_FRACTION_MIN: float = 0.5
TAIL_FRACTION_MAX: float = 1.0

# --- v9-B: tail-recency blended into the pack input. -----------------------
# pack_input = (1 - β) * window_load + β * tail_load_rescaled
# β = 0.0 reproduces v7's pack (window_load only). 0.3 is a conservative
# blend (small effect on stationary, meaningful on drifty layers).
#
# Default OFF after local A/B: stacking tail-into-pack on top of the
# recency-weighted window_load double-counts recency and slightly
# regresses stationary PAR on local LmSys. Same shape of trade-off as
# drift-adapt: helps Mix in theory, costs ~0.1% on local stationary.
# Kept as a one-line opt-in for the next Mix-only experiment.
TAIL_PACK_ENABLED: bool = False
TAIL_PACK_BLEND: float = 0.30

# --- v9-C: drift-adaptive per-layer recency floor (opt-in). ----------------
# floor_L = clamp(MAX - sensitivity * drift_L * (MAX - MIN), MIN, MAX)
# MAX = RECENCY_FLOOR ensures drift=0 layers behave exactly like v6.
DRIFT_ADAPT_ENABLED: bool = False
DRIFT_FLOOR_MIN: float = 0.1
DRIFT_SENSITIVITY: float = 1.0

# --- v9-D: gain-first commit ordering. -------------------------------------
GAIN_ORDER_ENABLED: bool = True


# ---------------------------------------------------------------------------
# Module state.
#
# This is the ONLY mutable global. The simulator adapter
# (eplb_algorithms/__init__.py) calls _reset() at the start of every run,
# so values never leak across runs.
#
# STATE_KEYS:
#   cur_deploy   (np.ndarray) -- (n_layers, n_dev, exp_per_dev) int64.
#                                Mirror of the placement the simulator
#                                is currently using (post-redeploy).
#   ewma_weight  (np.ndarray) -- (n_layers, n_experts) float64.
#                                EWMA of pack_input across cadence calls.
#   history      (list[ndarray]) -- past cur_deploy snapshots, oldest first.
#                                Used by cycle detection. Length capped
#                                at CYCLE_HISTORY_K + 1.
# ---------------------------------------------------------------------------
_STATE: dict[str, Any] = {
    "cur_deploy": None,
    "ewma_weight": None,
    "history": None,
}


def _reset() -> None:
    """Clear module state. Called once per simulator run by the adapter."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None


# ---------------------------------------------------------------------------
# Signal computation.
# ---------------------------------------------------------------------------
def _ep_aware_tail_fraction(exp_per_dev: int) -> float:
    """Return the tail fraction (0..1] given the per-device replication factor.

    Smaller exp_per_dev -> wider tail (denoise the gate). See v9-A block
    in the configuration section for the formula and worked examples.
    """
    raw = TAIL_BASE_FRACTION + TAIL_INV_SLOPE / max(1, exp_per_dev)
    return float(np.clip(raw, TAIL_FRACTION_MIN, TAIL_FRACTION_MAX))


def _recency_kernel(window_size: int, floor: float) -> np.ndarray:
    """Linear ramp from `floor` (oldest) to 1.0 (newest), shape (window_size,).

    The kernel is normalised so its values sum to `window_size`; this
    keeps the magnitude of the recency-weighted window comparable to a
    uniform sum, so the EWMA and PAR formulas stay calibrated regardless
    of `floor`.
    """
    if window_size <= 1 or floor >= 1.0:
        return np.ones(window_size, dtype=np.float64)
    raw = np.linspace(floor, 1.0, window_size, dtype=np.float64)
    return raw * (window_size / raw.sum())


def _per_layer_drift(hotness: np.ndarray) -> np.ndarray:
    """L1 first-half vs second-half drift per layer, in [0, 1].

    Input:  hotness  (W, L, E)
    Output: drift_L  (L,)   0 = stationary, 1 = maximally non-stationary.
    """
    half = max(1, hotness.shape[0] // 2)
    early = hotness[:half].sum(axis=0)                              # (L, E)
    late = hotness[half:].sum(axis=0)                               # (L, E)
    diff_l1 = np.abs(late - early).sum(axis=1)                      # (L,)
    total_l1 = (early + late).sum(axis=1)                           # (L,)
    return diff_l1 / np.maximum(total_l1, 1e-9)


def _compute_window_load(hotness: np.ndarray) -> np.ndarray:
    """Recency-weighted window load with optional per-layer drift adaptation.

    Output: (L, E). Drift-adapt uses a per-layer floor; otherwise the
    global RECENCY_FLOOR is used uniformly (equivalent to v6 behaviour).
    """
    window_size = hotness.shape[0]
    n_layers = hotness.shape[1]

    if DRIFT_ADAPT_ENABLED:
        drift = _per_layer_drift(hotness)                           # (L,)
        floor_per_layer = (RECENCY_FLOOR
                           - DRIFT_SENSITIVITY * drift
                             * (RECENCY_FLOOR - DRIFT_FLOOR_MIN))
        floor_per_layer = np.clip(floor_per_layer,
                                  DRIFT_FLOOR_MIN, RECENCY_FLOOR)
        # Build a per-layer kernel of shape (L, W); normalise to sum=W per layer.
        t = np.linspace(0.0, 1.0, window_size, dtype=np.float64)    # (W,)
        f = floor_per_layer[:, None]                                # (L, 1)
        raw = f + (1.0 - f) * t[None, :]                            # (L, W)
        norms = raw.sum(axis=1, keepdims=True)
        kernel = raw * (window_size / np.maximum(norms, 1e-9))      # (L, W)
        return np.einsum("lw,wle->le", kernel, hotness)             # (L, E)

    # Common path: single global kernel, applied to all layers.
    kernel = _recency_kernel(window_size, RECENCY_FLOOR)            # (W,)
    return np.einsum("w,wle->le", kernel, hotness)                  # (L, E)


def _compute_tail_load(hotness: np.ndarray, tail_fraction: float) -> np.ndarray:
    """Sum of hotness over the last `tail_fraction * W` iters. Shape (L, E).

    Rescaled to W * mean_hotness so its magnitude matches window_load and
    blending stays unbiased.
    """
    window_size = hotness.shape[0]
    tail_w = max(1, int(round(window_size * tail_fraction)))
    raw_tail = hotness[-tail_w:].sum(axis=0)                        # (L, E)
    return raw_tail * (window_size / tail_w)


# ---------------------------------------------------------------------------
# DeepSeek-equivalent per-layer placement (unchanged from v1 / v6 / v7).
# ---------------------------------------------------------------------------
def _init_deploy(n_layers: int, n_device: int, n_experts: int,
                 n_phys: int) -> np.ndarray:
    """Return the simulator's default placement for n_red_expert > 0."""
    exp_per_dev = n_phys // n_device
    base_slots = exp_per_dev - 1
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for d in range(n_device):
        for s in range(base_slots):
            dep[:, d, s] = (d * base_slots + s) % n_experts
        dep[:, d, -1] = dep[:, d, -2]
    return dep


def _replicate_experts(weight_1d: np.ndarray, n_phys: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Water-fill replication: each extra slot goes to the expert that
    most reduces max(weight / replica_count). Returns (phy2log, logcnt)."""
    n_log = weight_1d.shape[0]
    phy2log = np.empty(n_phys, dtype=np.int64)
    phy2log[:n_log] = np.arange(n_log, dtype=np.int64)
    logcnt = np.ones(n_log, dtype=np.int64)
    for slot in range(n_log, n_phys):
        pick = int(np.argmax(weight_1d / logcnt))
        phy2log[slot] = pick
        logcnt[pick] += 1
    return phy2log, logcnt


def _balanced_pack_lpt(weights_1d: np.ndarray, n_packs: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Longest-processing-time bin packing with equal item counts per bin."""
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
    """Replicate + LPT pack a single layer. Returns (n_device, exp_per_dev)."""
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log
    return deploy


# ---------------------------------------------------------------------------
# Anchor: device-label and slot-order matching to minimise transit at fixed PAR.
# ---------------------------------------------------------------------------
def _greedy_mapping_from_overlap(overlap: np.ndarray) -> np.ndarray:
    """Greedy 2-approximation: process strongest match first."""
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
    """Iterative best-pair swap; PAR-invariant, strictly higher overlap."""
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
    """Permute device rows of `new_deploy` to maximise overlap with prev."""
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
    """Within each device, keep an expert in its previous slot when present."""
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
    """PAR = max_device_load / mean_device_load. Returns 1.0 for zero load."""
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
def rebalance(hotness: np.ndarray, n_device: int, n_red_expert: int):
    """Per-cadence rebalance call.

    Args:
        hotness: (window, n_layers, n_experts) -- the simulator's last
                 cadence-window of per-iter, per-layer expert hotness.
        n_device: number of physical devices (EP).
        n_red_expert: total redundant experts to allocate across n_device.

    Returns:
        change (bool): True if at least one layer should be redeployed.
        layers_priority (np.ndarray int64): indices of layers to redeploy,
            ordered by descending PAR-gain (largest first).
        deployment (np.ndarray int64): (n_layers, n_device, exp_per_dev)
            full deployment table (unchanged layers keep their entries).
        aux: None (kept for adapter API parity).
    """
    hotness = np.asarray(hotness, dtype=np.float64)
    if hotness.ndim != 3:
        raise ValueError(f"hotness must be 3D (W, L, E), got {hotness.shape}")
    n_layers = int(hotness.shape[1])
    n_experts = int(hotness.shape[2])
    n_dev = int(n_device)
    n_phys = n_experts + int(n_red_expert)
    exp_per_dev = n_phys // n_dev

    # ---------- signals -----------------------------------------------------
    window_load = _compute_window_load(hotness)                     # (L, E)

    # Tail width is EP-aware: at high replication (small exp_per_dev) we
    # widen the tail to denoise, so the gate doesn't flip on noise at
    # high EP -- the v7 grader failure mode at Mix EP256.
    if TAIL_GATE_ENABLED or TAIL_PACK_ENABLED:
        tail_fraction = _ep_aware_tail_fraction(exp_per_dev)
        tail_load = _compute_tail_load(hotness, tail_fraction)      # (L, E)
    else:
        tail_load = window_load

    # Pack input = optional tail blend on top of the recency-weighted window.
    # On stationary traces tail == window so this is a no-op; on Mix the
    # pack itself shifts toward currently-hot experts (not just the gate).
    if TAIL_PACK_ENABLED:
        pack_input = ((1.0 - TAIL_PACK_BLEND) * window_load
                      + TAIL_PACK_BLEND * tail_load)
    else:
        pack_input = window_load

    # Gate signal = tail (or full window when EP-aware tail saturates to 1.0).
    gate_load = tail_load if TAIL_GATE_ENABLED else window_load

    # EWMA across cadence calls smooths the pack input over time.
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != pack_input.shape:
        smoothed = pack_input.copy()
    else:
        smoothed = EWMA_ALPHA * pack_input + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # Initialise placement + cycle history on the first call.
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["history"] = [cur.copy()]
    history = _STATE["history"]

    deploy_out = cur.copy()
    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY
                              / _BALANCED_COMPUTE_SECONDS)

    # ---------- pass 1: propose + gate every layer --------------------------
    # Stable iteration order for cache-friendliness; commit order decided
    # after the loop based on the per-layer gain.
    accepted: list[tuple[int, np.ndarray, float, int]] = []  # (L, proposal, gain, moves)
    initial_order = np.argsort(-smoothed.sum(axis=1))
    for L_np in initial_order:
        L = int(L_np)

        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        if np.array_equal(proposed, cur[L]):
            continue

        # Cycle skip: refuse to revert to any of the last K committed states.
        if any(np.array_equal(proposed, past[L]) for past in history[:-1]):
            continue

        moves = int((proposed != cur[L]).sum())
        if moves == 0:
            continue
        gain_par = (_layer_par(gate_load[L], cur[L])
                    - _layer_par(gate_load[L], proposed))
        if gain_par <= moves * par_per_move_threshold:
            continue

        accepted.append((L, proposed, gain_par, moves))

    # ---------- pass 2: choose commit order and apply -----------------------
    if GAIN_ORDER_ENABLED:
        accepted.sort(key=lambda item: -item[2])  # largest PAR-gain first

    changed: list[int] = []
    for L, proposal, _gain, _moves in accepted:
        deploy_out[L] = proposal
        cur[L] = proposal
        changed.append(L)

    if changed:
        history.append(cur.copy())
        if len(history) > CYCLE_HISTORY_K + 1:
            history.pop(0)
        return True, np.array(changed, dtype=np.int64), deploy_out, None

    return False, np.array([], dtype=np.int64), deploy_out, None
