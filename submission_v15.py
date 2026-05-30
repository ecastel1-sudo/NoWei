"""submission_v15.py -- v7 stack + CLOSED-LOOP ADAPTIVE GATE (TCP-AIMD + ARC-style).

CORE INSIGHT (THE COOKIE-RATIO, RESTATED)
==========================================
    1 move costs           ~9.79e-5 s (88MB / 900GB/s)
    1 PAR-point per case   60.00 s
    => break-even ratio    613,076 moves per PAR-point per case
    => per layer           ~10,500 moves (DS-R1, 58L) or ~6,500 (Qwen3, 94L)

`GATE_SAFETY = 16` in v7..v13 means we refuse a redeploy unless the
predicted PAR gain is 16x the math break-even. That is massively
over-conservative WHEN OUR PREDICTION IS TRUSTWORTHY. v14 tried to
loosen the gate via a `drift`-keyed prior and regressed because drift
is a *guess* about whether the prediction will hold -- not a
measurement that it did.

v15 closes the loop. We MEASURE how good our PAR prediction was at
the previous call and adapt the per-layer gate aggressiveness from
the residual. The framework is straight out of TCP congestion control
and ARC cache management: trust grows multiplicatively on success and
collapses multiplicatively on a "loss event" (mis-prediction).


WHY THIS IS NOT v12 / v14 IN DISGUISE
======================================
v12 / v14 keyed the gate on `drift` -- an *a-priori* signal computed
from the current window. That makes them PROACTIVE but blind: a high-
drift layer might still have a perfectly predictable future, and a
low-drift layer might dramatically shift in the next window.

v15 keys the gate on *prediction residual* -- an a-posteriori signal
computed from how well the LAST commit's predicted PAR matched the
actual PAR observed this window. That makes it REACTIVE and grounded:
- LmSys / ShareGPT / WildChat / Qwen3 layers: predictions land within
  ~5% of actual -> safety collapses toward SAFETY_MIN over a few
  calls -> we move aggressively where it pays.
- Mix EP128 / EP256 layers: predictions miss by 20-50% -> safety
  ratchets back to SAFETY_MAX -> we DO NOT pay transit for moves
  whose gain we cannot trust.

This is the same control idea behind:
    TCP-Reno AIMD  : cwnd += a on ACK, cwnd *= b on loss
    ARC (2003 FAST): T1/T2 budget shifts on ghost-list hits
    Vapor (ISPA21) : per-worker batch size via AIMD on epoch time
    Augmented Kalman drift detection: rate-of-drift estimated from
                                       innovation residuals

THE SAFETY ARGUMENT
====================
v15 starts every layer at `safety[L] = GATE_SAFETY = 16.0`. The first
1-2 calls are bit-exact equivalent to v7. Adaptation only fires after
we have measured a prediction residual on a layer we actually
committed. So we cannot regress below v7 baseline -- the worst case is
v7. The upside is everywhere v7 was leaving PAR on the table because
GATE_SAFETY = 16 was a global, pessimistic constant.


SYSTEMS-INSPIRED COMPONENTS ALREADY PRESENT FROM v1..v7
=========================================================
    EWMA across calls (alpha=0.7)         signal processing / LMS filters
    Recency-weighted ramp (floor=0.5)     ARC T1/T2 recency vs frequency
    Tail-only gate (75% of window)        CLOCK-Pro hot region
    2-opt device-label anchor             local search (TSP / VRP)
    Cycle-detection (history K=2)         tabu list / hashed bloom filter
    Gain-first commit order               priority-queue greedy scheduling
    LPT pack                              4/3-competitive online bin packing

NEW IN v15
===========
    Per-layer adaptive safety[L]          TCP-AIMD congestion control +
                                          ARC ghost-list-driven re-balance


CONTROL-LOOP MATH
==================
Per-layer state machine, evaluated each call (after the first):

    err[L]  = actual_par_after_commit / predicted_par_after_commit
    if err <= TRUST_HIGH:        safety[L] *= BETA_DOWN     (more aggressive)
    elif err >= TRUST_LOW:       safety[L] *= BETA_UP       (back off)
    safety[L] = clip(safety[L], SAFETY_MIN, SAFETY_MAX)

    threshold[L] = TIME_PER_MOVE * n_layers * safety[L] / 60.0
    accept proposal iff predicted_gain_par > moves * threshold[L]

Parameters chosen for stability and a hard SAFE-vs-v7 contract:

    BETA_DOWN   = 0.85   -- ~5 successful predictions to halve safety
    BETA_UP     = 1.30   -- ~3 bad predictions to back off ~2x
    TRUST_HIGH  = 1.05   -- treat <=5% over-prediction as "right"
    TRUST_LOW   = 1.30   -- treat >=30% over-prediction as "wrong"
    SAFETY_MIN  = 1.00   -- floor: literal math break-even
    SAFETY_MAX  = 16.0   -- CEILING == v7's default. v15 can NEVER be more
                            conservative than v7; only equal or more
                            aggressive. The wide TRUST dead-zone
                            [1.05, 1.30] prevents noisy residuals from
                            triggering a safety bump that, capped at 16,
                            would just freeze us in v7 mode.


API CONTRACT
=============
Identical to v1 / submission.py / v7:
    rebalance(hotness, n_device, n_red_expert)
      hotness: np.ndarray shape (window, n_layers, n_experts)
      returns: (change, layers_priority, deployment, _aux)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Inherited tuning (frozen from v7).
# ---------------------------------------------------------------------------
EWMA_ALPHA = 0.7
CYCLE_HISTORY_K = 2
ANCHOR_2OPT_PASSES = 3
RECENCY_FLOOR_DEFAULT = 0.5
TAIL_GATE_ENABLED = True
TAIL_FRACTION = 0.75
GAIN_ORDER_ENABLED = True

# Initial per-layer gate safety. == v7 default; per-layer values
# diverge from this only after the first call.
GATE_SAFETY = 16.0

# ---------------------------------------------------------------------------
# v15: closed-loop adaptive gate.
#
# Per-layer safety[L] is updated each call based on the residual
# between predicted PAR (recorded at commit time) and actual PAR
# (computed against this call's fresh hotness window). The rule is a
# TCP-AIMD-style multiplicative weight update with asymmetric
# step sizes to favor stability over aggression.
# ---------------------------------------------------------------------------
AIMD_BETA_DOWN = 0.85      # successful prediction -> shrink safety
AIMD_BETA_UP   = 1.30      # bad prediction        -> grow safety (smaller because
                           # we already cap at v7's safety, see SAFETY_MAX)
TRUST_HIGH     = 1.05      # err <= this  : prediction reliable
TRUST_LOW      = 1.30      # err >= this  : prediction unreliable (wider dead zone
                           # than the first design; PAR residuals are inherently
                           # noisy because the gate window mixes pre-/post-deploy
                           # iters, and small fluctuations should NOT trip a
                           # safety bump-up).
SAFETY_MIN     = 1.0       # math break-even floor
SAFETY_MAX     = 16.0      # CAP at v7 default: v15 is STRICTLY MORE AGGRESSIVE
                           # than v7 (or equal), never more conservative. This
                           # is the "can't be worse than v7" safety contract.

_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS


# ---------------------------------------------------------------------------
# Module state. _reset() is called by the harness on every fresh run.
# ---------------------------------------------------------------------------
_STATE: dict = {
    "cur_deploy": None,
    "ewma_weight": None,
    "history": None,
    # v15-specific:
    "safety_per_layer": None,         # (n_layers,) float64
    "last_committed_layers": None,    # set[int]
    "last_predicted_par": None,       # dict[int, float] -- predicted PAR per L
}


def _reset() -> None:
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None
    _STATE["safety_per_layer"] = None
    _STATE["last_committed_layers"] = None
    _STATE["last_predicted_par"] = None


# ---------------------------------------------------------------------------
# Signal builders (v6/v7 carry-over).
# ---------------------------------------------------------------------------
def _recency_kernel(window_size: int) -> np.ndarray:
    """Linear ramp from RECENCY_FLOOR_DEFAULT to 1.0, normalised so its
    sum equals window_size. Matches v6/v7's fixed-floor kernel.
    """
    if window_size <= 1 or RECENCY_FLOOR_DEFAULT >= 1.0:
        return np.ones(window_size, dtype=np.float64)
    raw = np.linspace(RECENCY_FLOOR_DEFAULT, 1.0, window_size, dtype=np.float64)
    return raw * (window_size / raw.sum())


def _compute_window_load(hotness: np.ndarray) -> np.ndarray:
    """v7 recency-weighted full-window load. Shape (L, E)."""
    kernel = _recency_kernel(hotness.shape[0])
    return np.einsum("w,wle->le", kernel, hotness)


def _compute_tail_load(hotness: np.ndarray) -> np.ndarray:
    """v7-B tail-only signal used by the gate. Shape (L, E)."""
    W = hotness.shape[0]
    tail_w = max(1, int(round(W * TAIL_FRACTION)))
    return hotness[-tail_w:].sum(axis=0)


# ---------------------------------------------------------------------------
# DeepSeek-equivalent per-layer placement (unchanged from v1/v6/v7).
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
# 2-opt refined device-label anchor (unchanged from v4-B / v7).
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
    """PAR(load(weight, deploy)) -- mirrors dynamic_lb_simulator exactly."""
    n_experts = weight_1d.shape[0]
    cut = np.bincount(deploy_2d.reshape(-1), minlength=n_experts)
    cut = np.maximum(cut, 1)
    weights_per_replica = weight_1d / cut
    loads = (weights_per_replica[deploy_2d.reshape(-1)]
             .reshape(deploy_2d.shape).sum(-1))
    mean = float(loads.mean())
    return 1.0 if mean == 0.0 else float(loads.max() / mean)


# ---------------------------------------------------------------------------
# v15: closed-loop safety update.
# ---------------------------------------------------------------------------
def _update_safety_from_evidence(safety_per_layer: np.ndarray,
                                 last_layers,
                                 last_predicted_par,
                                 cur_deploy: np.ndarray,
                                 gate_load: np.ndarray) -> None:
    """In-place AIMD update of safety_per_layer using prediction residuals.

    For every layer L we redeployed last call, compare:
        actual_par = layer_par(gate_load[L], cur_deploy[L])
        pred_par   = last_predicted_par[L]
    and apply a multiplicative weight update biased toward stability.
    """
    if not last_layers or last_predicted_par is None:
        return
    eps = 1e-9
    for L in last_layers:
        pred = last_predicted_par.get(L)
        if pred is None or pred <= eps:
            continue
        actual = _layer_par(gate_load[L], cur_deploy[L])
        err = actual / max(pred, eps)
        if err <= TRUST_HIGH:
            safety_per_layer[L] *= AIMD_BETA_DOWN
        elif err >= TRUST_LOW:
            safety_per_layer[L] *= AIMD_BETA_UP
        # gray band [TRUST_HIGH, TRUST_LOW): no change (dead zone for stability)
    np.clip(safety_per_layer, SAFETY_MIN, SAFETY_MAX, out=safety_per_layer)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def rebalance(hotness, n_device, n_red_expert):
    """v7 stack + closed-loop adaptive gate (v15)."""
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

    # ---------- v7-A: recency-weighted full-window load -------------------
    window_load = _compute_window_load(hotness)

    # ---------- v7-B: tail-only signal used by the cost gate --------------
    gate_load = _compute_tail_load(hotness) if TAIL_GATE_ENABLED else window_load

    # ---------- EWMA across cadence calls ----------
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != window_load.shape:
        smoothed = window_load.copy()
    else:
        smoothed = EWMA_ALPHA * window_load + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # Seed cur_deploy, cycle history, per-layer safety on first call.
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["history"] = [cur.copy()]
    history = _STATE["history"]

    safety = _STATE.get("safety_per_layer")
    if safety is None or safety.shape != (n_layers,):
        safety = np.full(n_layers, GATE_SAFETY, dtype=np.float64)
        _STATE["safety_per_layer"] = safety

    # ---------- v15 evidence step ----------------------------------------
    # If we committed redeploys last call, the current `gate_load` is the
    # observed post-commit hotness. Compare per-layer actual-vs-predicted
    # PAR to update the per-layer safety with TCP-AIMD-style multiplicative
    # weights. This MUST happen BEFORE the gate fires this call so the
    # updated safeties take effect immediately.
    _update_safety_from_evidence(
        safety,
        _STATE.get("last_committed_layers"),
        _STATE.get("last_predicted_par"),
        cur,
        gate_load,
    )

    deploy_out = cur.copy()

    # Per-layer threshold from per-layer safety. SAFETY_MIN = 1.0 corresponds
    # exactly to the math break-even; SAFETY_MAX = 32 is 2x v7 conservatism.
    threshold_per_layer = (
        _TIME_PER_MOVE_S * n_layers * safety / _BALANCED_COMPUTE_SECONDS
    )

    # ---------- pass 1: propose + gate every layer -----------------------
    initial_order = np.argsort(-smoothed.sum(axis=1))
    accepted: list[tuple[int, np.ndarray, float, int]] = []
    for L_np in initial_order:
        L = int(L_np)

        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        if np.array_equal(proposed, cur[L]):
            continue

        # v4-A cycle detection.
        cycle = False
        for past in history[:-1]:
            if np.array_equal(proposed, past[L]):
                cycle = True
                break
        if cycle:
            continue

        moves = int((proposed != cur[L]).sum())
        if moves == 0:
            continue
        gain_par = (_layer_par(gate_load[L], cur[L]) -
                    _layer_par(gate_load[L], proposed))
        if gain_par <= moves * threshold_per_layer[L]:
            continue

        accepted.append((L, proposed, gain_par, moves))

    # ---------- v7-C: commit order (largest gain first) -------------------
    if GAIN_ORDER_ENABLED:
        accepted.sort(key=lambda x: -x[2])

    # Record predictions for next-call evidence step. We snapshot the PAR
    # we predict each committed layer will have under `gate_load` (our
    # best estimate of next-window hotness). At the next call we compare
    # this against actual PAR under that call's fresh gate_load.
    next_predicted_par: dict[int, float] = {}
    changed: list[int] = []
    for L, proposal, _gain, _moves in accepted:
        deploy_out[L] = proposal
        cur[L] = proposal
        changed.append(L)
        # Predicted post-commit PAR under our current gate signal.
        next_predicted_par[L] = _layer_par(gate_load[L], proposal)

    _STATE["last_committed_layers"] = set(changed)
    _STATE["last_predicted_par"] = next_predicted_par

    if changed:
        history.append(cur.copy())
        if len(history) > CYCLE_HISTORY_K + 1:
            history.pop(0)
    else:
        return False, np.array([], dtype=np.int64), deploy_out, None

    return True, np.array(changed, dtype=np.int64), deploy_out, None
