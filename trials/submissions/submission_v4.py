"""submission_v4.py -- v1 (the 110.86 winner) + ARC-pure transit savers.

Where v2 failed:
  v2 added three ARC pieces (T2 weight boost, per-layer cooldown, ghost
  gate boost) and lost score (109.37 < 110.86) because all three could
  HURT PAR -- and PAR is ~98% of total_time at the leaderboard band.

The constraint for v4 is therefore strict:
    "ARC mechanic must be either (a) PAR-invariant on its own, OR (b) a
     pure SKIP that fires only when there is direct evidence the redeploy
     would be reversed."

Two pieces meet that bar:

  v4-A. CYCLE DETECTION  (pure skip, evidence-driven)
     We keep the last K committed placements per layer. If the pack
     proposes a placement that EQUALS any of those past placements, we
     refuse to commit it. The semantics: "I just was in state X, then
     moved to state Y; the new proposal asks me to go back to X. That
     is flapping. Skip until something genuinely new shows up."
     This skip fires ONLY on exact equality, so we never reject a
     PAR-improving redeploy unless it would literally revert us to a
     placement we already paid transit to leave. PAR-loss-free.

  v4-B. 2-OPT REFINEMENT of the device-label anchor (PAR-invariant)
     v1's _anchor_device_labels does a greedy assignment on the
     (n_dev, n_dev) overlap matrix that picks the strongest match
     first. That is a 2-approximation but not optimal. We add a cheap
     vectorised pair-swap refinement: for every pair (i, j), check
     whether swapping mapping[i] and mapping[j] would increase the
     total overlap; if yes, swap. Repeat until no improvement.
     This is a strictly stronger anchor (always greater or equal
     overlap), and PAR is invariant under any device-row permutation,
     so PAR cannot change. Cost: O(n_dev^2) per pass, ~3 passes.

NO changes to:
  * EWMA (alpha=0.7)
  * Cost gate (safety=16)
  * Skip-unchanged-layers
  * The replicate/pack core
  * Slot-order anchor
  * Any weight (T2 boost from v2 stays out -- it cost PAR)

Expected outcome (modeling against v1's 110.86):
  Cycle detection saves transit on traces where hot experts rotate
  through stable orbits (LmSys-like). 2-opt refinement saves a few %
  transit on every cadence call. Combined: PAR unchanged, transit
  modestly down, score ~111-112.

API contract identical to v1 / submission.py.
"""

from __future__ import annotations

import numpy as np

# ---------- tuning constants ------------------------------------------------
# EWMA on per-expert weight across cadence calls. v1 default = 0.7.
EWMA_ALPHA = 0.7

# How many committed placements per layer to remember for cycle detection.
# K=2 catches the 2-cycle (we just left state X, now proposed wants X back).
# K=3 also catches 3-cycles (X -> Y -> Z -> X). Memory is K * (n_layers *
# n_dev * exp_per_dev) int64 -- trivial.
CYCLE_HISTORY_K = 2

# Max number of 2-opt passes on the device-label anchor. Each pass is
# O(n_dev^2). 3 is more than enough in practice; if the first pass finds
# no improvement we early-exit.
ANCHOR_2OPT_PASSES = 3

# ---- scoring constants (DO NOT EDIT) ----
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s

# Cost/benefit gate safety. v1 swept and picked 16. v4 keeps it.
GATE_SAFETY = 16.0


# ---------- module state ----------------------------------------------------
_STATE: dict = {
    "cur_deploy": None,        # (n_layers, n_dev, exp_per_dev) int64
    "ewma_weight": None,       # (n_layers, n_experts) float64
    "history": None,           # list of past committed cur_deploy snapshots
}


def _reset() -> None:
    """Clear module state. Called by the adapter on every fresh run."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None


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
    """v1's greedy: process strongest available match first."""
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
    """Iteratively swap pairs (i, j) of mapping entries to maximise
    sum(overlap[i, mapping[i]]). Vectorised: per pass compute the full
    (n_dev, n_dev) swap-delta matrix in one numpy expression, then take
    the single best positive-delta swap.

    PAR-invariant (permutation only). Strictly better-or-equal anchor
    than greedy.
    """
    n = overlap.shape[0]
    for _ in range(max_passes):
        cur_vals = overlap[np.arange(n), mapping]  # (n,) current per-row val
        # new_i_vals[i, j] = overlap[i, mapping[j]]: what row i would get
        # if it took mapping[j]'s assignment.
        new_i_vals = overlap[:, mapping]  # (n, n)
        # delta[i, j] = gain from swapping mapping[i] <-> mapping[j].
        delta = (new_i_vals + new_i_vals.T
                 - cur_vals[:, None] - cur_vals[None, :])
        # Ignore diagonal (i == j is a no-op) and lower triangle (i > j
        # duplicates of the upper triangle by symmetry).
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
    """v1 anchor + v4-B 2-opt refinement. Still O(n_dev^2 * n_experts)
    overall (the matmul dominates), with a few extra O(n_dev^2) passes
    at the end.
    """
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
    """Same as v1: keep experts in the slot they were in previously."""
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
    """v4: v1 logic + ARC cycle detection + 2-opt refined anchor.
    See module docstring for design rationale.
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

    # ---------- EWMA-smoothed weight (same as v1) ----------
    window_load = hotness.sum(axis=0)
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
        # History starts with one snapshot (the init) so we never reject
        # the first real redeploy as a "cycle".
        _STATE["history"] = [cur.copy()]

    history = _STATE["history"]

    deploy_out = cur.copy()
    changed: list[int] = []

    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                              _BALANCED_COMPUTE_SECONDS)

    layer_order = np.argsort(-smoothed.sum(axis=1))
    for L_np in layer_order:
        L = int(L_np)

        # ----- pack + two-level anchor (v4-B refines the device anchor) -----
        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        # ----- skip unchanged ----- (same as v1)
        if np.array_equal(proposed, cur[L]):
            continue

        # ----- v4-A: cycle detection -----
        # If we'd revert to ANY recent past committed placement, this is
        # flapping; refuse the redeploy. We deliberately do NOT check
        # against the very last snapshot (== cur) since the equality
        # above already handled that.
        cycle = False
        for past in history[:-1]:  # history[-1] is the current state
            if np.array_equal(proposed, past[L]):
                cycle = True
                break
        if cycle:
            continue

        # ----- cost gate (same as v1) -----
        moves = int((proposed != cur[L]).sum())
        gain_par = (_layer_par(window_load[L], cur[L]) -
                    _layer_par(window_load[L], proposed))
        if gain_par <= moves * par_per_move_threshold:
            continue

        deploy_out[L] = proposed
        cur[L] = proposed
        changed.append(L)

    # Push the post-loop cur into history and trim to CYCLE_HISTORY_K + 1
    # snapshots so the cycle check never compares against state from
    # before our remembered window.
    if changed:
        history.append(cur.copy())
        if len(history) > CYCLE_HISTORY_K + 1:
            history.pop(0)

    if not changed:
        return False, np.array([], dtype=np.int64), deploy_out, None

    return True, np.array(changed, dtype=np.int64), deploy_out, None
