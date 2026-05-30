"""submission_v18.py -- "best of best" = v15 (AIMD adaptive gate) + v2 ARC overlays.

DESIGN OBJECTIVE
================
v15 won composite score (110.97), total modelled time (2629.6 s), and
composite PAR (1.7184) by combining the v7 stack with a TCP-AIMD
adaptive per-layer gate keyed on prediction residual.

v2 (a.k.a. ``submission_v2_lowest_trans.py``) won total transit
(391184) by adding three ARC-inspired overlays on top of v1:
  - ARC-A: T2 frequency boost on the pack input.
  - ARC-B: Per-layer cooldown gate boost right after a redeploy.
  - ARC-C: Per-(layer, expert) ghost-hit gate boost on demoted experts
           that came back hot.

v2's PAR however inflates on Mix/DS-R1 EP128/EP256 (3.14 vs v15's 2.68,
4.53 vs 3.66) because its overlays push the effective gate safety up
toward 16+16+8=40 -- WAY above v7's 16 -- so necessary moves get
refused on volatile workloads. The reason is that v2 stacks ARC on top
of the *static* GATE_SAFETY=16, with no per-layer trust calibration.

v18 = v15's adaptive base safety + v2's three overlays, scaled down so
the overlays merely TIGHTEN the gate on layers/experts that just
moved, instead of overriding the residual-driven trust calibration.
This gives us:
  - v15-quality PAR (AIMD still drives the base toward SAFETY_MIN on
    predictable layers).
  - v2-style anti-thrash on the same layers (cooldown / ghost-hit
    boosts prevent redundant follow-up moves immediately after a
    commit).

SAFETY ARGUMENT
===============
All overlays are STRICTLY ADDITIVE -- they can only RAISE the safety
multiplier, never lower it. We clip to EFFECTIVE_SAFETY_MAX = 32 (= 2x
v7 default) so even in the worst overlay stacking the threshold stays
within an order of magnitude of v7. The clip matters: without it, the
overlay sum can drive the gate so high that the AIMD update on the
NEXT call cannot bring it back into commit-range, and we stall.

API contract identical to v1 / v2 / v7 / v15:
    rebalance(hotness, n_device, n_red_expert)

Layout follows v15 so a per-line diff against v15 highlights only the
ARC overlay additions.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Inherited tuning (frozen from v7 / v15 stack).
# ---------------------------------------------------------------------------
EWMA_ALPHA = 0.7
CYCLE_HISTORY_K = 2
ANCHOR_2OPT_PASSES = 3
RECENCY_FLOOR_DEFAULT = 0.5
TAIL_GATE_ENABLED = True
TAIL_FRACTION = 0.75
GAIN_ORDER_ENABLED = True

# Initial per-layer gate safety. == v7 default; per-layer values diverge
# from this only after the first call when AIMD has residual evidence.
GATE_SAFETY = 16.0

# ---------------------------------------------------------------------------
# v15 closed-loop AIMD parameters (unchanged from v15).
# ---------------------------------------------------------------------------
AIMD_BETA_DOWN = 0.85
AIMD_BETA_UP = 1.30
TRUST_HIGH = 1.05
TRUST_LOW = 1.30
SAFETY_MIN = 1.0
SAFETY_MAX = 16.0

# ---------------------------------------------------------------------------
# v18 ARC overlays -- ported from v2 (lowest_trans) but DELIBERATELY
# SCALED DOWN. AIMD already grows safety[L] toward SAFETY_MAX on layers
# whose past predictions missed badly, so we do not need v2's heavy
# cooldown/ghost boosts on top of that. The scaled-down boosts mainly
# protect the *first 1-2 calls* after a layer commits -- the window in
# which AIMD has not yet collected fresh residual evidence for that
# layer at its new placement.
# ---------------------------------------------------------------------------

# ARC-A: T2 frequency protection on the pack INPUT (not gate). Identical
# tuning to v2 -- this dial is decoupled from the gate so it does not
# need re-scaling. Keeps persistent hot experts from being shuffled by
# transient T1 spikes.
HOT_MULT = 1.5
T2_STREAK_FULL = 3
T2_BOOST_MAX = 0.10

# ARC-B: per-layer cooldown after redeploy. Boost decays as
# exp(-age / COOLDOWN_TAU). v2 used 16 (== full v7 safety on top of the
# base 16, so right after a move the effective safety was 32 there).
# v18 uses 2 because AIMD already pushes safety up on layers with bad
# predictions; the cooldown only needs to bridge the 1-2 call window
# before AIMD has residual evidence at the new placement.
#
# Tuning trace (LmSys subset, 7 cases, comparing to v15):
#   CD=4.0  GH=2.0  T2=0.10 -> composite -0.78 ( -2.3% transit )
#   CD=2.0  GH=1.0  T2=0.10 -> composite -0.75 ( -1.2% transit )
#   CD=1.0  GH=0.5  T2=0.10 -> composite -0.80 ( -4.3% transit )
#   CD=2.0  GH=1.0  T2=0.00 -> composite -1.00 ( -6.5% transit )
# The CD=2 / GH=1 / T2=0.10 row keeps almost all of v15's PAR while
# still freeing the gate to refuse 1-2 follow-up moves per commit.
# On Mix (no local trace) v2's data shows transit drops 50-75% with
# similar overlays in play -- the win is in non-stationary cases.
COOLDOWN_TAU = 1.5
COOLDOWN_BOOST_MAX = 2.0

# ARC-C: ghost-hit gate boost. Same reasoning as ARC-B for the scale-
# down (v2 used 8 -> v18 uses 1). The boost still scales with the
# fraction of CURRENTLY-hot experts that we recently demoted, so on
# truly oscillating layers it can lift the threshold up to GATE_SAFETY +
# COOLDOWN_BOOST_MAX + GHOST_BOOST_MAX = 19, capped at the
# EFFECTIVE_SAFETY_MAX below.
GHOST_WINDOW = 3
GHOST_BOOST_MAX = 1.0

# Hard cap on the post-overlay safety multiplier so the gate cannot
# stall a layer indefinitely. = 2 * v7's default. Above this the
# threshold > 196k moves / PAR-point per case -- meaningless margin.
EFFECTIVE_SAFETY_MAX = 32.0

# ---------------------------------------------------------------------------
# Scoring constants (DO NOT EDIT).
# ---------------------------------------------------------------------------
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS


# ---------------------------------------------------------------------------
# Module state. _reset() is called by the harness on every fresh run.
# ---------------------------------------------------------------------------
_STATE: dict = {
    # v7 / v15 carry-over.
    "cur_deploy": None,
    "ewma_weight": None,
    "history": None,
    "safety_per_layer": None,
    "last_committed_layers": None,
    "last_predicted_par": None,
    # v18 ARC overlay state (ported from v2).
    "t2_streak": None,           # (n_layers, n_experts) int32, ARC-A
    "last_redeploy_call": None,  # (n_layers,) int64, ARC-B
    "demoted_at_call": None,     # (n_layers, n_experts) int64, ARC-C
    "call_count": 0,             # global cadence counter (1-based once running)
}


def _reset() -> None:
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["history"] = None
    _STATE["safety_per_layer"] = None
    _STATE["last_committed_layers"] = None
    _STATE["last_predicted_par"] = None
    _STATE["t2_streak"] = None
    _STATE["last_redeploy_call"] = None
    _STATE["demoted_at_call"] = None
    _STATE["call_count"] = 0


# ---------------------------------------------------------------------------
# Signal builders (v6 / v7 carry-over, unchanged).
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
    window_len = hotness.shape[0]
    tail_w = max(1, int(round(window_len * TAIL_FRACTION)))
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
# 2-opt refined device-label anchor (unchanged from v4-B / v7 / v15).
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
# v15 closed-loop safety update (unchanged).
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
        # dead zone [TRUST_HIGH, TRUST_LOW): no change.
    np.clip(safety_per_layer, SAFETY_MIN, SAFETY_MAX, out=safety_per_layer)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def rebalance(hotness, n_device, n_red_expert):
    """v15 (AIMD adaptive gate + v7 stack) + v2 ARC overlays, scaled down."""
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

    # Bump global cadence counter -- ARC-B / ARC-C state references it.
    _STATE["call_count"] += 1
    call_now = _STATE["call_count"]

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

    # ---------- seed all per-layer state on first call / shape change -----
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur
        _STATE["history"] = [cur.copy()]
        _STATE["t2_streak"] = np.zeros((n_layers, n_experts), dtype=np.int32)
        _STATE["last_redeploy_call"] = np.zeros(n_layers, dtype=np.int64)
        # Sentinel large-negative so age is huge -> no ghost hits on the
        # very first call.
        _STATE["demoted_at_call"] = np.full((n_layers, n_experts), -10**9,
                                             dtype=np.int64)
    history = _STATE["history"]
    t2_streak = _STATE["t2_streak"]
    last_redeploy_call = _STATE["last_redeploy_call"]
    demoted_at_call = _STATE["demoted_at_call"]

    safety = _STATE.get("safety_per_layer")
    if safety is None or safety.shape != (n_layers,):
        safety = np.full(n_layers, GATE_SAFETY, dtype=np.float64)
        _STATE["safety_per_layer"] = safety

    # ---------- v15 evidence step ----------------------------------------
    # If we committed redeploys last call, the current `gate_load` is the
    # observed post-commit hotness. Compare per-layer actual-vs-predicted
    # PAR to update the per-layer safety with TCP-AIMD-style multiplicative
    # weights. MUST happen BEFORE the gate fires so the updated safeties
    # take effect immediately on this call.
    _update_safety_from_evidence(
        safety,
        _STATE.get("last_committed_layers"),
        _STATE.get("last_predicted_par"),
        cur,
        gate_load,
    )

    # ---------- ARC-A (v2): update T2 streaks and build boosted weight ----
    # Vectorised across (layer, expert). Streak counts consecutive calls a
    # given expert sat above HOT_MULT * layer mean of the smoothed signal.
    # Boost is at most (1 + T2_BOOST_MAX) once T2_STREAK_FULL is reached.
    layer_mean = smoothed.mean(axis=1, keepdims=True)              # (L, 1)
    hot_mask = smoothed > (HOT_MULT * layer_mean)                  # (L, E) bool
    t2_streak[:] = np.where(hot_mask, t2_streak + 1, 0)
    t2_factor = (np.minimum(t2_streak, T2_STREAK_FULL).astype(np.float64) /
                 float(T2_STREAK_FULL))
    t2_boost = 1.0 + T2_BOOST_MAX * t2_factor                      # (L, E)
    boosted_smoothed = smoothed * t2_boost                         # (L, E)

    deploy_out = cur.copy()

    # AIMD-driven base threshold per layer (= v15's expression).
    base_threshold_per_layer = (
        _TIME_PER_MOVE_S * n_layers * safety / _BALANCED_COMPUTE_SECONDS
    )
    # Constant factor used by the ARC overlay additions below; lets us add
    # overlay-in-safety-units rather than recompute in PAR units twice.
    par_per_safety_unit = _TIME_PER_MOVE_S * n_layers / _BALANCED_COMPUTE_SECONDS

    # ---------- pass 1: propose + gate every layer -----------------------
    initial_order = np.argsort(-smoothed.sum(axis=1))
    accepted: list[tuple[int, np.ndarray, float, int, np.ndarray]] = []
    for L_np in initial_order:
        L = int(L_np)

        # Pack uses the T2-boosted weight (ARC-A). Anchors / cycle-detect
        # / gate stay on the unboosted signals because PAR / overlap are
        # measured against the unboosted ground truth.
        proposed = _propose_layer(boosted_smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        if np.array_equal(proposed, cur[L]):
            continue

        # v4-A cycle detection (kept from v7 / v15).
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

        # ARC-B: per-layer cooldown gate boost. Decays exponentially with
        # the number of cadence calls since the last commit. age = 0
        # (just committed) -> cooldown_add = COOLDOWN_BOOST_MAX.
        age_b = call_now - int(last_redeploy_call[L])
        cooldown_add = COOLDOWN_BOOST_MAX * float(np.exp(-age_b / COOLDOWN_TAU))

        # ARC-C: ghost-hit gate boost. An expert is a "ghost" on this
        # layer if we demoted it (>1 -> 1 replica) within GHOST_WINDOW
        # calls. The boost scales with the fraction of CURRENTLY-hot
        # experts that are ghosts -- the more our previous demotions
        # are coming back as hot, the more we distrust this layer's
        # proposal.
        ghost_recent = (call_now - demoted_at_call[L]) <= GHOST_WINDOW
        layer_hot = hot_mask[L]
        ghost_active = ghost_recent & layer_hot
        n_hot = int(layer_hot.sum())
        ghost_ratio = (float(ghost_active.sum()) / float(n_hot)) if n_hot else 0.0
        ghost_add = GHOST_BOOST_MAX * ghost_ratio

        # Combine AIMD base safety with ARC overlay additions, then clip
        # at EFFECTIVE_SAFETY_MAX to prevent indefinite stall.
        effective_safety = min(EFFECTIVE_SAFETY_MAX,
                               float(safety[L]) + cooldown_add + ghost_add)
        threshold_eff = par_per_safety_unit * effective_safety

        # Sanity: if AIMD already pushed base above EFFECTIVE_SAFETY_MAX
        # (cannot happen because SAFETY_MAX=16 < EFFECTIVE_SAFETY_MAX=32,
        # but kept for future safety re-tuning), fall back to the AIMD
        # threshold so we never get more permissive than v15.
        if threshold_eff < base_threshold_per_layer[L]:
            threshold_eff = base_threshold_per_layer[L]

        gain_par = (_layer_par(gate_load[L], cur[L]) -
                    _layer_par(gate_load[L], proposed))
        if gain_par <= moves * threshold_eff:
            continue

        # Compute newly-demoted experts for THIS proposal. ARC-C needs
        # them stamped at commit time so future calls can find their
        # ghosts. We compute now (before gain ordering) so the proposal
        # objects can be passed through unchanged.
        cur_cnt = np.bincount(cur[L].reshape(-1), minlength=n_experts)
        new_cnt = np.bincount(proposed.reshape(-1), minlength=n_experts)
        newly_demoted = (cur_cnt > 1) & (new_cnt == 1)

        accepted.append((L, proposed, gain_par, moves, newly_demoted))

    # ---------- v7-C: commit order (largest gain first) -------------------
    if GAIN_ORDER_ENABLED:
        accepted.sort(key=lambda x: -x[2])

    # Record predictions for next-call evidence step. We snapshot the PAR
    # we predict each committed layer will have under `gate_load` (our
    # best estimate of next-window hotness). At the next call we compare
    # this against actual PAR under that call's fresh gate_load.
    next_predicted_par: dict[int, float] = {}
    changed: list[int] = []
    for L, proposal, _gain, _moves, newly_demoted in accepted:
        deploy_out[L] = proposal
        cur[L] = proposal
        changed.append(L)
        next_predicted_par[L] = _layer_par(gate_load[L], proposal)

        # ARC-B / ARC-C state writes happen ONLY on actually-committed
        # layers. Mirrors v2's bookkeeping.
        last_redeploy_call[L] = call_now
        demoted_at_call[L] = np.where(newly_demoted, call_now,
                                       demoted_at_call[L])

    _STATE["last_committed_layers"] = set(changed)
    _STATE["last_predicted_par"] = next_predicted_par

    if changed:
        history.append(cur.copy())
        if len(history) > CYCLE_HISTORY_K + 1:
            history.pop(0)
    else:
        return False, np.array([], dtype=np.int64), deploy_out, None

    return True, np.array(changed, dtype=np.int64), deploy_out, None
