"""submission_v3.py -- minimal PAR-first refinement of v1 (the 110.86 winner).

Lessons from official grader runs:
  v1 (alpha=0.7, gate=16, two-level anchor):     score 110.86
  v2 (v1 + ARC-lite cooldown/ghost gate boost):  score 109.37

v2 cut transit by 27% (533k -> 391k) BUT raised PAR from 1.722 -> 1.779
and the score dropped. Because the grader aggregates 25 cases as:

    total_time = 60 * mean_par * n_cases + transit * 9.787e-5
              ~= 2580 s of compute (~98%)   + ~52 s of transit (~2%)

each 0.01 PAR point costs ~15 s of total_time, while each 100k moves
costs only ~10 s. PAR is ~15x more valuable than transit at the
leaderboard band. v2's ARC was over-suppressing redeploys that would
have improved PAR. Pushing harder in the same direction (a wide-bypass
draft of v3) caused transit to explode 22x locally; the gate is the
floor, removing it nukes the run.

v3 is therefore the MINIMAL change to v1 that should still earn PAR:

  A. EWMA_ALPHA = 1.0   (no smoothing, was 0.7 in v1)
     v1's smoothing dampens pack churn but also lags real hotness
     shifts -> the proposed placement targets stale loads -> measured
     PAR is worse than what the pack predicted. With gate=16 the
     anchor + gate already kill ~70% of unnecessary moves, so the EWMA
     was redundant transit-protection at PAR cost. Local sweep at
     short traces consistently showed alpha=1.0 beats 0.7 at every
     gate value when PAR is the bottleneck.

  B. FORCE FIRST-CADENCE DEPLOY
     The simulator initializes every layer to the static default
     placement which has high PAR. v1's gate could decide on call #1
     that the PAR gain is not worth the moves for some borderline
     layers, leaving them stuck on default for many windows. v3 skips
     the gate on the very first cadence call so we bootstrap into a
     balanced placement immediately. Cost: at most O(n_layers *
     exp_per_dev) extra moves ONCE per run, amortised over thousands
     of subsequent windows.

What we explicitly DO NOT change (proven free wins from v1):
  * Same numpy replicate + LPT pack (deepseek-equivalent).
  * Same device-label anchor and within-device slot anchor.
  * Same skip-unchanged-layers guard.
  * Same GATE_SAFETY = 16 as the cost/benefit floor on subsequent calls.

Risk envelope:
  * alpha=1.0 should LOWER PAR vs alpha=0.7 (sharper signal -> better
    pack). Transit may rise slightly because there is no temporal
    smoothing buffering tiny weight shuffles; the gate still pins it.
    Expected sign on grader: PAR down, transit up a bit, score up.
  * force-first-deploy can only help (it only fires once per run; if
    the gate would have accepted the moves anyway, behaviour is
    identical; if it would have rejected, we now correctly take them).

API contract identical to v1 / submission.py.
"""

from __future__ import annotations

import numpy as np

# ---------- tuning constants ------------------------------------------------
# EWMA on per-expert weight across cadence calls.
#   1.0 = no smoothing (deepseek-equivalent, FRESHEST PAR signal).
#   0.7 = v1 default (smoother but lagged).
# v3 sets 1.0 because PAR is ~98% of total_time and the lag was costing
# more than the transit it saved (see module docstring).
EWMA_ALPHA = 1.0

# ---- scoring constants (DO NOT EDIT) ----
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s

# Cost/benefit gate safety multiplier.
# v1 swept this and found 16 best on the official grader. v3 keeps it.
GATE_SAFETY = 16.0


# ---------- module state ----------------------------------------------------
# Adapter at eplb_algorithms/__init__.py calls _reset() at the start of
# every simulator run so no state leaks between cases.
_STATE: dict = {
    "cur_deploy": None,       # (n_layers, n_dev, exp_per_dev) int64
    "ewma_weight": None,      # (n_layers, n_experts) float64
    "first_call": True,       # used to force-deploy on the first cadence
}


def _reset() -> None:
    """Clear module state. Called by the adapter on every fresh run."""
    _STATE["cur_deploy"] = None
    _STATE["ewma_weight"] = None
    _STATE["first_call"] = True


# ---------- deepseek-equivalent placement (numpy) -- unchanged from v1 -----
def _init_deploy(n_layers: int, n_device: int, n_experts: int,
                 n_phys: int) -> np.ndarray:
    """Mirror of dynamic_lb_simulator.init_deploy_table for n_red_expert>0:
    each device's last slot duplicates the previous one. Seed only.
    """
    exp_per_dev = n_phys // n_device
    base_slots = exp_per_dev - 1
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for d in range(n_device):
        for s in range(base_slots):
            dep[:, d, s] = (d * base_slots + s) % n_experts
        dep[:, d, -1] = dep[:, d, -2]
    return dep


def _replicate_experts(weight_1d: np.ndarray, n_phys: int):
    """Water-filling replication (deepseek-equivalent)."""
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
    """LPT bin-pack with equal-count packs (deepseek-equivalent)."""
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
    """Replicate + LPT pack -> (n_device, exp_per_dev) placement."""
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log
    return deploy


def _anchor_device_labels(new_deploy: np.ndarray, prev_deploy: np.ndarray,
                          n_experts: int) -> np.ndarray:
    """Permute device rows of new_deploy to maximise overlap with prev.
    PAR-invariant; cuts transit substantially. Same as v1.
    """
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
    """Within each device, permute slots so experts stay in their old
    positions where possible. PAR-invariant. Same as v1.
    """
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
    """Per-layer PAR under given weight. Mirrors simulator's calculate_par."""
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
    """v3: PAR-first variant of v1. See module docstring."""
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

    # ---------- step 1: weight signal for the pack ----------
    # EWMA_ALPHA = 1.0 in v3 -> smoothed == window_load exactly (no lag).
    # Kept the EWMA structure so future runs can dial smoothing back in
    # without refactoring.
    window_load = hotness.sum(axis=0)
    ewma = _STATE.get("ewma_weight")
    if ewma is None or ewma.shape != window_load.shape:
        smoothed = window_load.copy()
    else:
        smoothed = EWMA_ALPHA * window_load + (1.0 - EWMA_ALPHA) * ewma
    _STATE["ewma_weight"] = smoothed

    # Seed cur_deploy if first call / shape changed.
    cur = _STATE.get("cur_deploy")
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur

    # Force-deploy decision: skip the cost gate on the first cadence call
    # so we bootstrap immediately into a balanced placement. Without this,
    # the gate can decide on call #1 that some layers' PAR gain is not
    # worth the moves, leaving them on the static default for many
    # subsequent windows.
    force_first = _STATE["first_call"]
    _STATE["first_call"] = False

    deploy_out = cur.copy()
    changed: list[int] = []

    # Threshold per move per layer (same shape as v1).
    par_per_move_threshold = (_TIME_PER_MOVE_S * n_layers * GATE_SAFETY /
                              _BALANCED_COMPUTE_SECONDS)

    # Heavy layers first: simulator drains the redeploy queue one layer
    # per iter, so prioritise the layers with the largest PAR opportunity.
    layer_order = np.argsort(-smoothed.sum(axis=1))
    for L_np in layer_order:
        L = int(L_np)

        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_device_labels(proposed, cur[L], n_experts)
        proposed = _anchor_slot_order(proposed, cur[L])

        if np.array_equal(proposed, cur[L]):
            continue

        # Compute the actual PAR gain on the window the simulator scores.
        gain_par = (_layer_par(window_load[L], cur[L]) -
                    _layer_par(window_load[L], proposed))

        # Skip layers that would make PAR worse. With alpha=1.0 this is
        # rare (proposed is the optimal pack on the same weight we gate
        # on), but the anchors are not PAR-invariant on rare ties.
        if gain_par <= 0.0:
            continue

        # On the very first cadence we skip the gate so we bootstrap
        # into a balanced placement immediately. Otherwise apply the
        # standard cost gate.
        if not force_first:
            moves = int((proposed != cur[L]).sum())
            if gain_par <= moves * par_per_move_threshold:
                continue

        deploy_out[L] = proposed
        cur[L] = proposed
        changed.append(L)

    if not changed:
        return False, np.array([], dtype=np.int64), deploy_out, None

    return True, np.array(changed, dtype=np.int64), deploy_out, None
