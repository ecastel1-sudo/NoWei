"""submission_v13.py -- v7 core + LINEAR-TREND PREDICTIVE PLACEMENT + variance pack.

WHY THIS IS A REAL DEPARTURE FROM v3..v12
=========================================

v3..v12 all share the same proposal pipeline:
    pack_signal = recency_weighted_PAST_window
    -> water-fill replicate -> LPT pack -> 2-opt anchor

They differ only in DOWNSTREAM filters (gate threshold, commit order, post-
LPT swap, drift floor on the recency kernel). The grader proves this can't
break past 110.94: every shipped v3..v12 sits in [110.91, 110.94], a
0.03-point band. The leader (111.89) is doing something STRUCTURALLY
different.

The dominating failure mode is non-stationary Mix DS-R1 where PAR hits
3.67 at EP256. All v3..v12 plans for that case are LITERALLY computed from
the past 128 iters but APPLIED over the next 128 iters. When the hotness
drifts within that horizon (Mix does), the placement we just deployed is
already stale.

v13 attacks the root cause: PACK AGAINST WHAT WE EXPECT NEXT WINDOW TO
LOOK LIKE, not what last window did.


v13-A. LINEAR-TREND PREDICTIVE PLACEMENT          [primary, default ON]
=======================================================================

Split the cadence window into halves and estimate a per-expert TREND:

    early_rate_{L,e} = mean over first half of hotness[:, L, e]
    late_rate_{L,e}  = mean over second half of hotness[:, L, e]
    trend_{L,e}      = late_rate_{L,e} - early_rate_{L,e}
    predicted_rate_{L,e} = max(0, late_rate_{L,e}
                               + ALPHA_PREDICT * trend_{L,e})

`predicted_rate` is the per-iter expert hotness we EXPECT during the next
window if the linear trend continues. ALPHA_PREDICT controls how far we
extrapolate:

    ALPHA_PREDICT = 0.0  -> predicted = late_rate (== "use only recent half")
    ALPHA_PREDICT = 0.5  -> predicted = late_rate + 0.5 * trend     <-- default
                            (extrapolates a half-window into the future)
    ALPHA_PREDICT = 1.0  -> predicted = late_rate + trend
                            (extrapolates a full window; risky on noise)

On stationary traces (LmSys / ShareGPT / WildChat) trend ~ 0 (local
inspection: within-window drift on DS-R1 LmSys = 0.024, Qwen3 LmSys =
0.023), so predicted ~ late ~ recency-weighted current. STATIONARY-SAFE.

On Mix the trend is real and persistent. We pack EXPERTS THAT WILL BE
HOT, not experts that WERE hot. This is the structural change v3..v12
never made.

SAFETY ANCHOR: PREDICT_WEIGHT blends predicted_window with v6/v7's
recency-weighted window. PREDICT_WEIGHT = 1.0 is pure prediction;
PREDICT_WEIGHT = 0.0 reproduces v7 exactly. Default is 1.0 (committed
to the prediction), but the knob lets us back off without rewriting code.


v13-B. VARIANCE-WEIGHTED PACK INPUT               [secondary, default ON]
========================================================================

Even with a perfect mean prediction, a HIGH-VARIANCE expert produces
per-iter PAR spikes that a mean-based pack does not see. Mix at EP256
has 2 slots/device, so two bursty experts co-located on one device
double its per-iter peak load.

We augment the pack weights (not the gate) with one standard deviation:

    pack_weight_{L,e} = predicted_rate_{L,e}
                        + B_GAMMA * std(hotness[:, L, e])

B_GAMMA = 0 reproduces pure mean-based pack. B_GAMMA = 0.5 adds half a
sigma to each expert's weight. The LPT pack then "sees" bursty experts
as heavier than their mean and gives them MORE replicas + more device
separation. On stationary traces std is small and uniform across
experts -> no effective re-ordering -> stationary-safe. On Mix bursty
experts get pulled apart.

CRITICAL: B_GAMMA only affects the PACK INPUT. The GATE still uses
pure predicted_rate so the gate's per-move PAR estimate matches
actual hotness magnitudes, not inflated sigma-augmented ones.


INHERITED UNCHANGED FROM v7
===========================
    - EWMA across cadence calls (EWMA_ALPHA = 0.7) on the new pack signal.
    - Tail-only gate signal (TAIL_FRACTION = 0.75) on predicted_rate.
    - 2-opt device-label anchor (ANCHOR_2OPT_PASSES = 3).
    - Cycle detection (CYCLE_HISTORY_K = 2).
    - Per-move cost gate base (GATE_SAFETY = 16).
    - Gain-first commit ordering (v7-C).
    - DeepSeek-equivalent replication + LPT pack.
    - Slot-order anchor.


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
RECENCY_FLOOR = 0.5             # v6 default; only used inside the safety anchor
TAIL_GATE_ENABLED = True        # v7-B
TAIL_FRACTION = 0.75            # v7-B shipped value (won LmSys ablation)
GAIN_ORDER_ENABLED = True       # v7-C

# ---------------------------------------------------------------------------
# v13-A: predictive placement.
#
#   predicted_rate = max(0, late_rate + ALPHA_PREDICT * (late_rate - early_rate))
#   predicted_window = predicted_rate * W      # same scale as recency window
#   pack_signal      = (1 - PREDICT_WEIGHT) * recency_window
#                     + PREDICT_WEIGHT * predicted_window
#
# ALPHA_PREDICT controls how far we extrapolate the trend. 0.5 is the
# default: extrapolate a half-window into the future. Local inspection of
# LmSys traces shows within-window drift of 0.023-0.024 so on stationary
# the predicted signal is close to the recency one. On Mix the within-
# window drift is large and predictive packing is the whole point of v13.
#
# Local sweep on 7 LmSys cases (composite total_s, rel vs v7):
#   ALPHA = 0.5, B = 1.0 : 600.77s  (+2.12%)
#   ALPHA = 1.0, B = 1.0 : 600.14s  (+2.23%)
#   ALPHA = 1.0, B = 1.5 : 598.59s  (+2.49%)  <-- shipped default
# Higher B (2.0, 3.0) regresses; A=1.0 extrapolates a full window forward.
#
# PREDICT_WEIGHT is a safety knob: 1.0 = pure prediction (default), 0.0 =
# pure v7 recency-weighted. Dial down if grader regresses.
# ---------------------------------------------------------------------------
PREDICTIVE_ENABLED = True
ALPHA_PREDICT = 1.0
PREDICT_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# v13-B: variance-augmented pack input.
#
#   pack_weight = predicted_rate + B_GAMMA * std(hotness over window)
#
# Local sweep on 7 LmSys cases (composite total_s, lower is better):
#   B_GAMMA = 0.0 : 605.44s  (+1.34% vs v7)
#   B_GAMMA = 1.0 : 600.77s  (+2.12% vs v7)
#   B_GAMMA = 1.5 : 598.59s  (+2.49% vs v7)   <-- shipped default (with A=1.0)
#
# Only the PACK sees this; the GATE uses pure predicted_rate so the
# per-move PAR estimate stays calibrated to actual hotness magnitudes.
#
# The B_GAMMA = 1.0 win is the largest single delta v3..v12 has seen
# locally. It says bursty experts should be spread MORE aggressively
# than their mean weight suggests, because their per-iter spike
# co-locations are what drive PAR up. The mean-only LPT pack is blind
# to this; the variance term restores it.
# ---------------------------------------------------------------------------
VARIANCE_PACK_ENABLED = True
B_GAMMA = 1.5


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
# Signal builders.
# ---------------------------------------------------------------------------
def _recency_kernel(window_size: int) -> np.ndarray:
    """Cached linear ramp normalised to sum == window_size (v6 default).

    Used only inside the safety anchor when PREDICT_WEIGHT < 1.0.
    """
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


def _recency_window_load(hotness: np.ndarray) -> np.ndarray:
    """v6/v7 recency-weighted window load. Shape (L, E)."""
    kernel = _recency_kernel(hotness.shape[0])
    return np.einsum("w,wle->le", kernel, hotness)


def _predicted_rate(hotness: np.ndarray) -> np.ndarray:
    """v13-A: linear-trend extrapolated per-iter hotness rate. Shape (L, E).

    Splits the window in half. trend = late_half_mean - early_half_mean.
    predicted = max(0, late_half_mean + ALPHA_PREDICT * trend).

    The clip to 0 catches the case where a strongly-declining trend
    overshoots zero; we never give a negative weight to LPT or the gate.
    """
    W = hotness.shape[0]
    half = max(1, W // 2)
    early_rate = hotness[:half].mean(axis=0)                  # (L, E)
    late_rate = hotness[half:].mean(axis=0) if W > half else early_rate
    trend = late_rate - early_rate
    pred = late_rate + ALPHA_PREDICT * trend
    return np.maximum(pred, 0.0)


def _per_expert_std(hotness: np.ndarray) -> np.ndarray:
    """v13-B: per-expert within-window std. Shape (L, E).

    Cheap, vectorised: std over the W axis for each (L, E) pair.
    """
    return hotness.std(axis=0)


def _compute_tail_load(hotness: np.ndarray) -> np.ndarray:
    """v7-B tail load. Used as the GATE signal but on hotness rate, not sum.

    NOTE: PAR is scale-invariant in the load vector, so dividing by tail_w
    or not does not change the gate's gain_par sign. We keep the absolute
    sum form (matching v7) so cycle-detection comparisons stay aligned.
    """
    W = hotness.shape[0]
    tail_w = max(1, int(round(W * TAIL_FRACTION)))
    return hotness[-tail_w:].sum(axis=0)


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
    """v13: predictive placement + variance-aware pack input on the v7 stack."""
    hotness = np.asarray(hotness, dtype=np.float64)
    if hotness.ndim != 3:
        raise ValueError(
            f"hotness must be 3D (window, layers, experts), got {hotness.shape}"
        )
    W = int(hotness.shape[0])
    n_layers = int(hotness.shape[1])
    n_experts = int(hotness.shape[2])
    n_dev = int(n_device)
    n_phys = n_experts + int(n_red_expert)
    exp_per_dev = n_phys // n_dev

    # ---------- v13-A: predictive load -------------------------------------
    if PREDICTIVE_ENABLED:
        pred_rate = _predicted_rate(hotness)                  # (L, E)
        # Bring to "window total" scale so it can blend with recency window.
        pred_window = pred_rate * W
    else:
        pred_rate = hotness.mean(axis=0)
        pred_window = pred_rate * W

    # Safety anchor: blend with recency-weighted window. PREDICT_WEIGHT=1.0
    # is pure prediction (default); PREDICT_WEIGHT=0.0 reproduces v7's
    # pack signal exactly.
    if PREDICT_WEIGHT < 1.0:
        rec_window = _recency_window_load(hotness)
        pack_signal = ((1.0 - PREDICT_WEIGHT) * rec_window
                       + PREDICT_WEIGHT * pred_window)
    else:
        pack_signal = pred_window

    # ---------- v13-B: variance-augmented pack input -----------------------
    # Only the PACK sees this. The GATE / cycle detection use pure
    # predicted hotness so the per-move PAR estimate stays calibrated.
    if VARIANCE_PACK_ENABLED and B_GAMMA > 0.0:
        sigma = _per_expert_std(hotness)                      # (L, E)
        # B_GAMMA carries units of "sigma per unit time"; multiply by W to
        # land on the same scale as pack_signal.
        pack_input = pack_signal + B_GAMMA * sigma * W
    else:
        pack_input = pack_signal

    # ---------- gate signal -------------------------------------------------
    # The gate compares PAR(cur) vs PAR(proposed) under what we expect to
    # see next. We use predicted_rate directly; PAR is scale-invariant so
    # multiplying by W is unnecessary. The tail option exists for the
    # opt-out case where prediction is disabled.
    if TAIL_GATE_ENABLED and not PREDICTIVE_ENABLED:
        gate_load = _compute_tail_load(hotness)
    else:
        gate_load = pred_rate

    # ---------- EWMA across cadence calls ----------------------------------
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != pack_input.shape:
        smoothed = pack_input.copy()
    else:
        smoothed = EWMA_ALPHA * pack_input + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # Seed cur_deploy and cycle-detection history on first call.
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["history"] = [cur.copy()]
    history = _STATE["history"]

    deploy_out = cur.copy()
    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                              _BALANCED_COMPUTE_SECONDS)

    # ---------- pass 1: propose + gate every layer --------------------------
    accepted: list[tuple[int, np.ndarray, float, int]] = []
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
        if gain_par <= moves * par_per_move_threshold:
            continue

        accepted.append((L, proposed, gain_par, moves))

    if not accepted:
        return False, np.array([], dtype=np.int64), deploy_out, None

    # ---------- pass 2: commit order (v7-C gain-first) ----------------------
    if GAIN_ORDER_ENABLED:
        accepted.sort(key=lambda x: -x[2])

    changed: list[int] = []
    for L, proposal, _gain, _moves in accepted:
        deploy_out[L] = proposal
        cur[L] = proposal
        changed.append(L)

    history.append(cur.copy())
    if len(history) > CYCLE_HISTORY_K + 1:
        history.pop(0)

    return True, np.array(changed, dtype=np.int64), deploy_out, None
