"""submission_v17.py -- v16 + queue-delay-aware gate + Hungarian device anchor.

ROOT-CAUSE RECAP (from v15 metrics + v16 design)
================================================
v16 already attacks the *content* of the proposal:
  - Trimmed-mean over the half-window (B): spike-immunity.
  - Drift extrapolation `recent + k * drift` with bin-size-scaled k (A):
    one-step-forward prediction for small-bin / high-EP cases where Mix
    drift is most lethal.

What v16 still does NOT model is the ONE-LAYER-PER-ITER drain queue.
Concretely: even a perfect prediction at time t becomes stale by the
time the layer at queue position p actually deploys, which happens at
time t+p. v16 evaluates every layer's gate against the same predicted
load; that systematically under-credits the layers at the back of the
queue, causing the gate to reject moves that would in fact have helped
when they finally executed.

v17 lands TWO targeted fixes on top of v16. Both are narrow, both have
clean physics, and both keep the v16 stack otherwise intact.

  v17-A. QUEUE-DELAY-AWARE GATE (CORRECTED REWRITE)
         The proposal "shift the temporal evaluation window by queue
         position" had the right physics but inverted slicing — slicing
         hotness[delay:] looks at the more-recent PAST, not the future.
         The real fix uses the drift signal v16 already computes: a
         layer at queue position p will deploy ~p iterations from now,
         so the load it should be evaluated against is

             evaluated_L = recent + k_per_layer * drift * (1 + p/half)

         Two-pass implementation:
           1. Predict + propose for every layer using v16's per-layer
              k (no queue knowledge yet). Compute initial gain on the
              v16 predicted load; rank layers gain-first to assign p.
           2. For each layer at queue position p, build a
              queue-aware evaluated load and re-score ΔPAR for the
              GATE only. Proposals are not re-packed (that would
              double the runtime cost); we only adjust the
              accept/reject decision so the queue-tail layers don't
              get unfairly cut.

         Cost: one extra np call per layer in pass 2. Microseconds.
         Effect: more layer commits accepted on Mix where drift is
         large; no behavioural change on stationary cases (drift ~ 0).

  v17-B. HUNGARIAN DEVICE-LABEL ANCHOR
         v16's `_anchor_device_labels` greedily assigns each device
         to its highest-overlap unused match in max-overlap-first
         order. That is provably suboptimal for the assignment
         problem. scipy.optimize.linear_sum_assignment solves the
         exact (-overlap) cost matching in O(n_dev^3). For our
         worst case n_dev=256, that's a few ms.

         We DROP the proposal's `+0.1` Hamming penalty term — it is
         redundant. The overlap matrix already encodes the Hamming
         distance: maximising overlap == minimising Hamming. Adding
         the penalty would only break ties differently, and the
         break-ties arrived at by `linear_sum_assignment` are
         themselves canonical.

         Robustness: scipy may be missing on the grader runtime.
         We import lazily and fall back to v16's greedy anchor.

  Plus: GAIN-FIRST LAYER ORDERING (v7-C)
         v16 orders layers by total smoothed load. v17 orders by
         predicted PAR drop (largest gain first). This makes the
         queue-position-aware gate (above) physically meaningful:
         "queue position p" actually corresponds to "the p-th most
         valuable redeploy," not just "the p-th heaviest layer".

WHAT v17 DELIBERATELY DOES NOT INCLUDE
======================================
  - Hysteresis-banded AIMD gate: v16 does not run AIMD (v15 did and
    underperformed). Hysteresis is a fix for a problem we don't have;
    adding it would re-introduce v15's chatter risk. See proposal
    component (2).
  - Ghost-list residual regret: the pack ALWAYS places every expert
    (n_phys = n_experts + n_red >= n_experts), so the "unplaced
    experts" set is empty. The intent — boost replicas of surging
    experts — is already covered by v16's drift extrapolation feeding
    the existing water-fill `_replicate_experts`. See proposal
    component (3).

API contract identical to submission.py / submission_v16.py.
"""

from __future__ import annotations

import numpy as np

# Optional dependency: scipy is used for the Hungarian device anchor. If it
# is not present on the grader runtime we transparently fall back to v16's
# greedy anchor (see _anchor_device_labels_hungarian).
try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    _linear_sum_assignment = None

# ---------- tuning constants (one place to change them) ---------------------
# Inherited from v16.
EWMA_ALPHA = 0.7
GATE_SAFETY = 16.0
TRIM_FRAC = 0.10
DRIFT_NOISE_FLOOR = 0.05
BIN_AGG_NULL_BIN = 9
BIN_AGG_K_MAX = 2.0

# ---------- v17-A: queue-delay-aware gate -----------------------------------
# Per-layer gate evaluation uses
#     evaluated_L = recent + k_per_layer * drift * (1 + p/HALF * QUEUE_DELAY_GAIN)
# QUEUE_DELAY_GAIN = 1.0 keeps the natural physical mapping (one drain
# step per drift unit). Set to 0.0 to disable the queue-aware gate
# adjustment (== v16's behaviour). Larger values penalise queue-tail
# layers more: useful if downstream layers' drift is non-linear.
QUEUE_DELAY_GAIN = 1.0

# ---- exact scoring constants from dynamic_lb_simulator.py (DO NOT EDIT) ----
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s

# ---------- module state ----------------------------------------------------
_STATE: dict = {"cur_deploy": None, "ewma_weight": None}


def _reset() -> None:
    """Clear module state. Called by the adapter on every fresh run."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None


# ---------- v16-B: robust trimmed mean over a window axis -------------------
def _trimmed_mean(arr: np.ndarray, trim_frac: float, axis: int = 0) -> np.ndarray:
    """Mean along `axis` after dropping the top and bottom trim_frac of values.

    Uses np.partition (O(n) per slot) rather than a full sort. Cost on the
    worst case (DS-R1, 256 experts, 64-iter half-window): ~45 ms / call.
    """
    n = arr.shape[axis]
    trim = int(np.floor(trim_frac * n))
    if trim_frac <= 0.0 or n - 2 * trim < 2:
        return arr.mean(axis=axis)
    parted = np.partition(arr, [trim, n - trim - 1], axis=axis)
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(trim, n - trim)
    return parted[tuple(sl)].mean(axis=axis)


# ---------- deepseek-equivalent placement (numpy) ---------------------------
def _init_deploy(n_layers: int, n_device: int, n_experts: int,
                 n_phys: int) -> np.ndarray:
    """Default placement matching dynamic_lb_simulator.init_deploy_table."""
    exp_per_dev = n_phys // n_device
    base_slots = exp_per_dev - 1
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for d in range(n_device):
        for s in range(base_slots):
            dep[:, d, s] = (d * base_slots + s) % n_experts
        dep[:, d, -1] = dep[:, d, -2]
    return dep


def _replicate_experts(weight_1d: np.ndarray, n_phys: int):
    """Water-filling replication, equivalent to deepseek's `replicate_experts`."""
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
    """Equal-count LPT bin pack. Equivalent to deepseek's `balanced_packing`."""
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
    """Replicate-then-pack -> placement of shape (n_device, exp_per_dev)."""
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log
    return deploy


# ---------- v17-B: Hungarian device-label anchor ----------------------------
def _build_overlap(new_deploy: np.ndarray, prev_deploy: np.ndarray,
                   n_experts: int) -> np.ndarray:
    """(n_dev, n_dev) BLAS overlap of multi-hot expert sets per device."""
    n_dev = new_deploy.shape[0]
    rows = np.repeat(np.arange(n_dev), new_deploy.shape[1])
    new_oh = np.zeros((n_dev, n_experts), dtype=np.float32)
    prev_oh = np.zeros((n_dev, n_experts), dtype=np.float32)
    new_oh[rows, new_deploy.reshape(-1)] = 1.0
    prev_oh[rows, prev_deploy.reshape(-1)] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return new_oh @ prev_oh.T


def _anchor_device_labels_greedy(new_deploy: np.ndarray, prev_deploy: np.ndarray,
                                 n_experts: int) -> np.ndarray:
    """Original v16 greedy anchor. Used as fallback when scipy is missing."""
    n_dev = new_deploy.shape[0]
    overlap = _build_overlap(new_deploy, prev_deploy, n_experts)
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
    out = np.empty_like(new_deploy)
    out[mapping] = new_deploy
    return out


def _anchor_device_labels_hungarian(new_deploy: np.ndarray, prev_deploy: np.ndarray,
                                    n_experts: int) -> np.ndarray:
    """Optimal device-label assignment via scipy.optimize.linear_sum_assignment.

    Cost matrix is `-overlap`: maximising total overlap is equivalent to
    minimising total cost. PAR is invariant under row permutation so this
    only changes transit; the optimal assignment minimises the number of
    expert moves needed to land on the new layout, strictly better than the
    greedy anchor's solution.
    """
    if not _HAS_SCIPY:
        return _anchor_device_labels_greedy(new_deploy, prev_deploy, n_experts)
    overlap = _build_overlap(new_deploy, prev_deploy, n_experts)
    # row_ind[i] = i (rows are processed in order); col_ind[i] = the prev-device
    # label that new-device i should adopt. Equivalent to mapping[i] = col_ind[i].
    row_ind, col_ind = _linear_sum_assignment(-overlap)
    mapping = np.empty(new_deploy.shape[0], dtype=np.int64)
    mapping[row_ind] = col_ind
    out = np.empty_like(new_deploy)
    out[mapping] = new_deploy
    return out


def _layer_par(weight_1d: np.ndarray, deploy_2d: np.ndarray) -> float:
    """Per-layer PAR. Mirrors dynamic_lb_simulator.calculate_par exactly."""
    n_experts = weight_1d.shape[0]
    cut = np.bincount(deploy_2d.reshape(-1), minlength=n_experts)
    cut = np.maximum(cut, 1)
    weights_per_replica = weight_1d / cut
    loads = (weights_per_replica[deploy_2d.reshape(-1)]
             .reshape(deploy_2d.shape).sum(-1))
    mean = float(loads.mean())
    return 1.0 if mean == 0.0 else float(loads.max() / mean)


def _anchor_slot_order(layer_new: np.ndarray,
                       layer_prev: np.ndarray) -> np.ndarray:
    """Within-device slot anchor (PAR-invariant, transit saver). Identical to v16."""
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


# ---------- v16-A: drift-aware predictive load ------------------------------
def _bin_size_k_base(exp_per_dev: int) -> float:
    """Lookahead aggressiveness: 0 at NULL_BIN, K_MAX at exp_per_dev=2."""
    span = max(1, BIN_AGG_NULL_BIN - 2)
    raw = BIN_AGG_K_MAX * (BIN_AGG_NULL_BIN - exp_per_dev) / span
    return float(max(0.0, min(BIN_AGG_K_MAX, raw)))


def _predict_components(hotness: np.ndarray, exp_per_dev: int):
    """Return the pieces v17 needs everywhere: `recent`, `drift`, and the
    per-layer `k`. Centralising this lets the queue-aware gate reuse the
    same `recent`/`drift` rather than recomputing them.

    Returns (recent, drift, k_per_layer_col).
      recent: (n_layers, n_experts), trimmed mean of the recent half.
      drift:  (n_layers, n_experts), recent - early.
      k_per_layer_col: (n_layers, 1) column vector of per-layer k_base after
                       the confidence guard. Zero on the fast path / on
                       layers below the drift noise floor.
    """
    k_base = _bin_size_k_base(exp_per_dev)
    if k_base == 0.0:
        # Fast path. recent = plain mean, drift = 0 -> queue-aware gate
        # collapses to v16's behaviour.
        recent = hotness.mean(axis=0)
        drift = np.zeros_like(recent)
        n_layers = recent.shape[0]
        return recent, drift, np.zeros((n_layers, 1), dtype=np.float64)

    half = max(1, hotness.shape[0] // 2)
    early = _trimmed_mean(hotness[:half], TRIM_FRAC, axis=0)
    recent = _trimmed_mean(hotness[half:], TRIM_FRAC, axis=0)
    drift = recent - early
    drift_l1 = np.abs(drift).sum(axis=1)
    level_l1 = recent.sum(axis=1) + 1e-12
    confident = drift_l1 / level_l1 > DRIFT_NOISE_FLOOR
    k_per_layer = np.where(confident, k_base, 0.0).astype(np.float64)
    return recent, drift, k_per_layer[:, None]


def _predict_at_offset(recent: np.ndarray, drift: np.ndarray,
                       k_per_layer_col: np.ndarray,
                       offset: float) -> np.ndarray:
    """Predicted hotness shifted `offset` units forward in drift-space.

        out = max(recent + k_per_layer * (1 + offset) * drift, 0)

    `offset = 0` reproduces v16's predicted load. Offsets > 0 add extra
    lookahead — used by v17-A to compensate for the queue's drain delay.
    """
    extra = (1.0 + offset) * k_per_layer_col
    out = recent + extra * drift
    np.maximum(out, 0.0, out=out)
    return out


# ---------- entry point -----------------------------------------------------
def rebalance(hotness, n_device: int, n_red_expert: int):
    """See module docstring."""
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
    half_window = max(1, hotness.shape[0] // 2)

    # ---------- v16-A + v16-B: predict pieces -------------------------------
    recent, drift, k_col = _predict_components(hotness, exp_per_dev)

    # The PROPOSAL load is v16's predicted load: recent + k * drift.
    # We pack on this — that's the bet the algorithm is making about what
    # the next window will look like.
    proposal_load = _predict_at_offset(recent, drift, k_col, offset=0.0)

    # ---------- EWMA across calls (unchanged from v6/v7/v16) ----------------
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != proposal_load.shape:
        smoothed = proposal_load.copy()
    else:
        smoothed = EWMA_ALPHA * proposal_load + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur

    # ---------- pass 1: propose all layers, compute initial gain ------------
    # We need every layer's proposal to (a) rank by gain and (b) measure
    # how many moves it would cost. Pack uses the EWMA-smoothed proposal
    # load (== v16's input to the pack), exactly preserving the cross-call
    # transit-stability v16 has.
    proposed_per_layer = np.empty_like(cur)
    moves_per_layer = np.zeros(n_layers, dtype=np.int64)
    initial_gain_per_layer = np.zeros(n_layers, dtype=np.float64)
    candidate_layers: list[int] = []

    for L in range(n_layers):
        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels_hungarian(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])
        if np.array_equal(proposed, cur[L]):
            continue
        proposed_per_layer[L] = proposed
        moves_per_layer[L] = int((proposed != cur[L]).sum())
        # Initial gain is on the smoothed proposal load (== v16's gate input).
        # This ranking determines queue position for v17-A.
        initial_gain_per_layer[L] = (_layer_par(smoothed[L], cur[L])
                                     - _layer_par(smoothed[L], proposed))
        candidate_layers.append(L)

    # ---------- pass 2: queue-delay-aware gate ------------------------------
    # Sort candidates by initial gain (largest first) -> queue position p.
    # Then re-evaluate ΔPAR using `recent + k * drift * (1 + p/half)`
    # so layers at the back of the queue are scored on the load they
    # will actually face when they finally deploy.
    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                              _BALANCED_COMPUTE_SECONDS)
    deploy_out = cur.copy()
    changed: list[int] = []

    candidate_layers.sort(key=lambda L: -initial_gain_per_layer[L])
    for queue_pos, L in enumerate(candidate_layers):
        # Queue-aware evaluation load: layer L will execute at t + queue_pos.
        # offset shifts predicted drift forward by queue_pos drift-units.
        offset = QUEUE_DELAY_GAIN * (queue_pos / float(half_window))
        if offset > 0.0 and float(k_col[L, 0]) > 0.0:
            evaluated = recent[L] + (1.0 + offset) * float(k_col[L, 0]) * drift[L]
            np.maximum(evaluated, 0.0, out=evaluated)
        else:
            # Fast path: no extra lookahead -> reuse v16's smoothed gate.
            evaluated = smoothed[L]
        moves = moves_per_layer[L]
        gain = (_layer_par(evaluated, cur[L])
                - _layer_par(evaluated, proposed_per_layer[L]))
        if gain <= moves * par_per_move_threshold:
            continue
        deploy_out[L] = proposed_per_layer[L]
        cur[L] = proposed_per_layer[L]
        changed.append(L)

    if not changed:
        return False, np.array([], dtype=np.int64), deploy_out, None

    # `changed` is already gain-sorted (we iterated candidate_layers in that
    # order and only appended successful commits). The simulator drains
    # this list one-per-iter, so gain-first means biggest-PAR-drop sooner.
    return True, np.array(changed, dtype=np.int64), deploy_out, None
