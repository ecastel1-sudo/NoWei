"""submission_v2.py -- v1 (deepseek + EWMA + two-level anchor + cost gate)
PLUS an ARC-inspired stability layer on top.

The motivation comes straight from the brief's "Policy Improvements":
  * Smooth hotness over a window to avoid chasing spikes.
  * Distinguish persistent hot experts from transient hot experts.
  * Penalize moves that only slightly improve balance.
  * Preserve placement for cold or stable experts.

v1 covers the first and third via EWMA + cost gate, but it does NOT
distinguish persistent-hot from just-hot experts beyond what EWMA blurs
together, and it has no explicit anti-thrashing memory. v2 adds the three
missing pieces by mapping ARC (Adaptive Replacement Cache) onto our
problem:

  ARC concept                    | MoE/LB equivalent (this file)
  -------------------------------+-------------------------------------------
  Cache capacity c               | n_phys slots per layer (replication budget)
  Item                           | logical expert id
  T1 (recency)                   | experts just promoted to >1 replica
  T2 (frequency, persistent)     | experts replicated >= T2_STREAK_FULL calls
  B1 (ghost recency)             | experts demoted within GHOST_WINDOW calls
                                   that were transient
  B2 (ghost frequency)           | experts demoted within GHOST_WINDOW calls
                                   that used to be T2  -- merged with B1 here
                                   because the action is the same
  Adaptive p (recency/freq dial) | per-layer GATE_SAFETY bump from B-hits
  Eviction policy                | water-filling + LPT pack (deepseek)

What v2 actually does on top of v1, in three small pieces:

  ARC-A. T2 protection (frequency)
     For each (layer, expert) we maintain `t2_streak`, the number of
     consecutive cadence calls the expert sat above HOT_MULT * mean of
     the smoothed weight. We boost the pack input weight by up to
     T2_BOOST_MAX for fully-saturated T2 experts. This stops a one-off
     T1 spike from outranking a persistent T2 in the LPT pack and
     reshuffling the layer for one transient surge.

  ARC-B. Cooldown after redeploy (anti-thrashing on a layer)
     For each layer we remember `last_redeploy_call`. The effective gate
     safety adds COOLDOWN_BOOST_MAX * exp(-age/COOLDOWN_TAU) so a layer
     that just got redeployed faces a much higher bar for the next 1-2
     calls. The exponential decay means the cooldown disappears within a
     few cadences if the workload genuinely changed.

  ARC-C. Ghost-hit gate boost (anti-thrashing on an expert)
     For each (layer, expert) we remember `demoted_at_call`, the call
     when the expert went from "had a replica" to "has only 1 slot". If
     within GHOST_WINDOW calls that same expert is hot again, that is
     literally a B1 hit -- our previous decision to demote it was wrong.
     The gate safety adds GHOST_BOOST_MAX * (ghost_hits / hot_count), so
     when ARC sees a churn pattern it makes the layer much harder to
     touch until the workload settles.

All ARC adjustments are gate-side or weight-side. The deepseek
replicate+pack core and the two-level anchor stay byte-for-byte the
same as v1. So:

  * PAR is at worst v1's PAR plus a small T2-boost-induced drift.
  * Transit is bounded above by v1's transit (the gate is *only ever
    stricter* in v2; never more permissive).

API contract is identical to v1 (rebalance + _reset). Numpy only.
"""

from __future__ import annotations

import numpy as np

# ---------- tuning constants ------------------------------------------------
# (1) EWMA on per-expert weight across cadence calls. Same role as v1.
EWMA_ALPHA = 0.7

# (2) Base gate safety multiplier on top of the exact scoring math.
#     Sweep on LmSys (v1, safety only): 4 -> 16 lifts composite +7 by cutting
#     transit. v2 keeps the base at 16; the ARC layer pushes it higher only
#     where it has evidence (cooldown / ghost hits) instead of globally.
GATE_SAFETY = 16.0

# ---- scoring constants (DO NOT EDIT) ----
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5

# ---- ARC-A: T2 frequency protection ----
# An expert is "hot" this call iff smoothed_weight > HOT_MULT * mean. After
# T2_STREAK_FULL consecutive hot calls the expert is treated as fully T2 and
# its weight is multiplied by (1 + T2_BOOST_MAX) when fed to the pack.
# Below T2_STREAK_FULL the boost is interpolated linearly.
HOT_MULT = 1.5
T2_STREAK_FULL = 3
T2_BOOST_MAX = 0.10  # small: avoid distorting PAR while still preferring T2

# ---- ARC-B: per-layer cooldown after redeploy ----
# Right after the layer gets redeployed (age = 0) we add COOLDOWN_BOOST_MAX
# to the safety multiplier. After COOLDOWN_TAU calls the boost is ~37% of
# max; after 3*tau it is negligible.
COOLDOWN_TAU = 1.5            # cadence calls
COOLDOWN_BOOST_MAX = 16.0     # +safety added at age 0

# ---- ARC-C: ghost-hit gate boost ----
# Experts demoted (replica count went >1 -> 1) within GHOST_WINDOW calls are
# "ghosts". If they are hot this call, that is a B-hit. Safety adds up to
# GHOST_BOOST_MAX scaled by (ghost_hits / current hot count).
GHOST_WINDOW = 3              # cadence calls
GHOST_BOOST_MAX = 8.0


# ---------- module state ----------------------------------------------------
# The adapter at eplb_algorithms/__init__.py calls _reset() at the start of
# every simulator run so no state can leak between cases.
_STATE: dict = {
    "cur_deploy": None,         # (n_layers, n_dev, exp_per_dev) int64
    "ewma_weight": None,        # (n_layers, n_experts) float64
    "t2_streak": None,          # (n_layers, n_experts) int32, ARC-A
    "last_redeploy_call": None, # (n_layers,) int64, ARC-B (call number)
    "demoted_at_call": None,    # (n_layers, n_experts) int64, ARC-C
    "call_count": 0,            # global cadence counter (1-based once running)
}


def _reset() -> None:
    """Clear module state. Called by the adapter on every fresh run."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["t2_streak"] = None
    _STATE["last_redeploy_call"] = None
    _STATE["demoted_at_call"] = None
    _STATE["call_count"] = 0


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
    """v2: same as v1 but with the three ARC adjustments described in the
    module docstring."""
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

    # Bump the global cadence counter first; everything below uses it.
    _STATE["call_count"] += 1
    call_now = _STATE["call_count"]

    # ---------- EWMA (same as v1) ----------
    window_load = hotness.sum(axis=0)
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != window_load.shape:
        smoothed = window_load.copy()
    else:
        smoothed = EWMA_ALPHA * window_load + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # ---------- seed all per-layer ARC state if first call / shape changed --
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["t2_streak"] = np.zeros((n_layers, n_experts), dtype=np.int32)
        _STATE["last_redeploy_call"] = np.zeros(n_layers, dtype=np.int64)
        # Use a sentinel large-negative value so age is huge -> no ghost hits
        # on the very first call.
        _STATE["demoted_at_call"] = np.full((n_layers, n_experts), -10**9,
                                             dtype=np.int64)

    t2_streak = _STATE["t2_streak"]
    last_redeploy_call = _STATE["last_redeploy_call"]
    demoted_at_call = _STATE["demoted_at_call"]

    deploy_out = cur.copy()
    changed: list[int] = []

    # Base gate threshold per move per layer (same formula as v1).
    base_par_per_move = (_TIME_PER_MOVE_S * n_layers /
                          _BALANCED_COMPUTE_SECONDS)

    # ---------- ARC-A: update t2 streaks layer-by-layer ---------------------
    # Vectorised: hot_mask is (n_layers, n_experts). We update streaks once
    # for all layers up-front. Cheap and keeps the inner loop simple.
    layer_mean = smoothed.mean(axis=1, keepdims=True)              # (L, 1)
    hot_mask = smoothed > (HOT_MULT * layer_mean)                  # (L, E) bool
    t2_streak[:] = np.where(hot_mask, t2_streak + 1, 0)

    # Pre-compute per-(layer, expert) T2 weight boost in [1.0, 1+T2_BOOST_MAX].
    t2_factor = (np.minimum(t2_streak, T2_STREAK_FULL).astype(np.float64) /
                 float(T2_STREAK_FULL))
    t2_boost = 1.0 + T2_BOOST_MAX * t2_factor                      # (L, E)
    boosted = smoothed * t2_boost                                  # (L, E)

    # Process layers in order of total smoothed load (heaviest first).
    layer_order = np.argsort(-smoothed.sum(axis=1))
    for L_np in layer_order:
        L = int(L_np)

        # ---- core: deepseek replicate + LPT pack (on T2-boosted weight) ----
        proposed = _propose_layer(boosted[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])
        if np.array_equal(proposed, cur[L]):
            continue

        # ---- ARC-B: cooldown boost ----
        age_b = call_now - int(last_redeploy_call[L])
        cooldown_add = COOLDOWN_BOOST_MAX * float(np.exp(-age_b / COOLDOWN_TAU))

        # ---- ARC-C: ghost hits for THIS layer ----
        # An expert is a "ghost" if we demoted it within GHOST_WINDOW calls.
        ghost_recent = (call_now - demoted_at_call[L]) <= GHOST_WINDOW
        layer_hot = hot_mask[L]
        ghost_active = ghost_recent & layer_hot
        n_hot = int(layer_hot.sum())
        ghost_ratio = (float(ghost_active.sum()) / float(n_hot)) if n_hot else 0.0
        ghost_add = GHOST_BOOST_MAX * ghost_ratio

        effective_safety = GATE_SAFETY + cooldown_add + ghost_add
        threshold = base_par_per_move * effective_safety

        # ---- cost gate (same shape as v1, just with adaptive safety) ----
        moves = int((proposed != cur[L]).sum())
        gain_par = (_layer_par(window_load[L], cur[L]) -
                    _layer_par(window_load[L], proposed))
        if gain_par <= moves * threshold:
            continue

        # ---- commit: also track which experts got newly demoted ----
        # An expert is newly demoted iff it had >1 replica in cur and has
        # exactly 1 in proposed. We use np.bincount on the slot arrays.
        cur_cnt = np.bincount(cur[L].reshape(-1), minlength=n_experts)
        new_cnt = np.bincount(proposed.reshape(-1), minlength=n_experts)
        newly_demoted = (cur_cnt > 1) & (new_cnt == 1)
        # Stamp the demotion time for each newly-demoted expert so future
        # ghost-hit checks can find them.
        demoted_at_call[L] = np.where(newly_demoted, call_now,
                                       demoted_at_call[L])

        deploy_out[L] = proposed
        cur[L] = proposed
        last_redeploy_call[L] = call_now
        changed.append(L)

    if not changed:
        return False, np.array([], dtype=np.int64), deploy_out, None

    return True, np.array(changed, dtype=np.int64), deploy_out, None
