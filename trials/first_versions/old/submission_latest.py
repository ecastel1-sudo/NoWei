"""MoE dynamic load-balancing policy -- numpy only.

What this policy does, in plain English:

1. Smooth the hotness with EWMA + a per-layer p99 clip so a single batch of
   weird routing cannot whipsaw our placement.
2. For each MoE layer, propose a new device-level placement: replicate the
   hottest experts onto the redundant slots (water-filling on load/replica),
   then bin-pack the physical replicas onto devices with LPT.
3. Anchor the new placement to the current one: PAR only depends on the
   multiset of per-device loads, so we permute the device labels of the
   new placement to maximise overlap with the current one. Same PAR, fewer
   actual expert moves on the wire. We use a greedy match (no scipy).
4. Cost / benefit gate using the simulator's own scoring constants:
       seconds_saved  = (cur_par - new_par) * 60.0          # PAR -> seconds
       seconds_spent  = moves * 88_080_384 / 9e11 * SAFETY  # transit
   Only commit a layer if seconds_saved > seconds_spent.
5. Apply the surviving layers in priority order (highest net seconds saved
   first) so the simulator -- which redeploys one layer per iteration --
   gets the biggest PAR drop sooner.

Constraints respected:
- numpy only (no scipy, no torch dependency at runtime)
- algorithm time budget: O(n_phys^2) replicate + O(n_phys log n_phys) pack +
  O(n_dev^2) greedy anchor per layer, all in numpy.
- model-agnostic: nothing depends on n_layers / n_experts / n_device specifically.

Participant API (matches the simulator):
    rebalance(hotness, n_device, n_red_expert)
        hotness: np.ndarray (window, n_layers, n_experts)
        returns: (change, layers_priority, deployment, _)
"""

from __future__ import annotations

import numpy as np

# ---------- scoring constants (mirror dynamic_lb_simulator.py) --------------
EXPERT_BYTES = 88_080_384
BANDWIDTH_BPS = 900_000_000_000
SEC_PER_MOVE = EXPERT_BYTES / BANDWIDTH_BPS  # ~9.787e-5 s
SEC_PER_PAR = 60.0                            # balanced_compute_seconds

# ---------- tuning constants -------------------------------------------------
EWMA_ALPHA = 0.3              # smaller = longer memory; 0.3 -> half-life ~2 windows
SPIKE_CLIP_PCTL = 99.0        # per-layer percentile clip on raw window load
GATE_SAFETY = 20.0            # multiplier on the move-cost half of the gate
GATE_MIN_CUR_PAR = 1.02       # skip layers already near optimal
GATE_MIN_GAIN = 0.0           # require strictly positive PAR gain

# ---------- module state -----------------------------------------------------
_STATE: dict = {"ewma": None, "cur_deploy": None}


def _reset() -> None:
    _STATE["ewma"] = None
    _STATE["cur_deploy"] = None


# ---------- helpers ----------------------------------------------------------
def _init_deploy(n_layers: int, n_device: int, n_experts: int, n_phys: int) -> np.ndarray:
    """Default placement matching the simulator's `init_deploy_table`."""
    exp_per_dev = n_phys // n_device
    base_slots = exp_per_dev - 1
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for d in range(n_device):
        for s in range(base_slots):
            dep[:, d, s] = (d * base_slots + s) % n_experts
        dep[:, d, -1] = dep[:, d, -2]
    return dep


def _replicate_experts(weight_1d: np.ndarray, n_phys: int):
    """Water-filling replication: each extra slot goes to argmax(load/replicas).

    Returns:
        phy2log: (n_phys,) -- logical expert id of each physical replica
        logcnt:  (n_log,)  -- replica count per logical expert
    """
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
    """LPT (longest-processing-time first) bin packing with equal-count packs.

    Item count must divide n_packs evenly. Each pack ends up with the same
    number of items but loads are kept as balanced as the greedy allows.
    """
    n = weights_1d.shape[0]
    items_per_pack = n // n_packs
    order = np.argsort(-weights_1d)
    pack_load = np.zeros(n_packs, dtype=np.float64)
    pack_count = np.zeros(n_packs, dtype=np.int64)
    pack_index = np.empty(n, dtype=np.int64)
    rank_in_pack = np.empty(n, dtype=np.int64)
    for it in order:
        # Pick the lightest pack that still has room.
        masked = np.where(pack_count < items_per_pack, pack_load, np.inf)
        choice = int(np.argmin(masked))
        pack_index[it] = choice
        rank_in_pack[it] = pack_count[choice]
        pack_load[choice] += weights_1d[it]
        pack_count[choice] += 1
    return pack_index, rank_in_pack


def _propose_layer(weight_1d: np.ndarray, n_device: int, n_phys: int) -> np.ndarray:
    """Replicate-then-pack -> placement of shape (n_device, exp_per_dev)."""
    phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
    tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
    pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
    deploy[pack_idx, rank] = phy2log
    return deploy


def _par_of(weight_1d: np.ndarray, deploy_layer: np.ndarray) -> float:
    """PAR = max(device_load) / mean(device_load) for one layer."""
    n_exp = weight_1d.shape[0]
    cnt = np.bincount(deploy_layer.reshape(-1), minlength=n_exp)
    cnt = np.maximum(cnt, 1)
    w = weight_1d / cnt
    loads = w[deploy_layer.reshape(-1)].reshape(deploy_layer.shape).sum(-1)
    m = float(loads.mean())
    return float(loads.max() / m) if m > 0 else 1.0


def _spike_clip(window_load: np.ndarray, pctl: float) -> np.ndarray:
    """Cap each layer's expert loads at its own p99 to kill in-window spikes."""
    cap = np.percentile(window_load, pctl, axis=1, keepdims=True)
    return np.minimum(window_load, cap)


def _anchor_to_prev(
    new_deploy: np.ndarray, prev_deploy: np.ndarray, n_experts: int
) -> np.ndarray:
    """Permute device labels of `new_deploy` to maximise overlap with `prev_deploy`.

    PAR is unchanged because it only depends on per-device loads, but the
    physical transit cost (sum of slot-wise diffs) drops a lot.

    Implementation: build two (n_dev, n_experts) multi-hot matrices and let
    BLAS compute the n_dev x n_dev overlap matrix as a float32 matmul. Then
    greedy assignment: for each new-device in order of strongest available
    match, lock in the still-free prev-device that shares the most experts.
    O(n_dev^2 * n_experts) for the matmul (BLAS) + O(n_dev^2) for the match.
    """
    n_dev, _ = new_deploy.shape
    rows = np.repeat(np.arange(n_dev), new_deploy.shape[1])
    new_oh = np.zeros((n_dev, n_experts), dtype=np.float32)
    prev_oh = np.zeros((n_dev, n_experts), dtype=np.float32)
    new_oh[rows, new_deploy.reshape(-1)] = 1.0
    prev_oh[rows, prev_deploy.reshape(-1)] = 1.0
    overlap = new_oh @ prev_oh.T  # (n_dev, n_dev), float32, BLAS

    # Greedy assignment, process strongest matches first.
    used = np.zeros(n_dev, dtype=bool)
    mapping = np.empty(n_dev, dtype=np.int64)
    process_order = np.argsort(-overlap.max(axis=1))
    NEG_INF = np.float32(-1.0)
    for i in process_order:
        scores = overlap[i].copy()
        scores[used] = NEG_INF
        j = int(np.argmax(scores))
        mapping[i] = j
        used[j] = True

    out = np.empty_like(new_deploy)
    out[mapping] = new_deploy
    return out


# ---------- main entry -------------------------------------------------------
def rebalance(hotness, n_device: int, n_red_expert: int):
    hotness = np.asarray(hotness, dtype=np.float64)
    if hotness.ndim != 3:
        raise ValueError(
            f"hotness must be 3D (window, layers, experts), got shape {hotness.shape}"
        )
    n_layers = int(hotness.shape[1])
    n_experts = int(hotness.shape[2])
    n_dev = int(n_device)
    n_phys = n_experts + int(n_red_expert)
    exp_per_dev = n_phys // n_dev

    # Step 1: smoothed hotness signal.
    window_load = _spike_clip(hotness.sum(axis=0), SPIKE_CLIP_PCTL)
    ewma = _STATE.get("ewma")
    if ewma is None or ewma.shape != (n_layers, n_experts):
        _STATE["ewma"] = window_load.copy()
        _STATE["cur_deploy"] = _init_deploy(n_layers, n_dev, n_experts, n_phys)
    else:
        _STATE["ewma"] = EWMA_ALPHA * window_load + (1.0 - EWMA_ALPHA) * ewma

    cur = _STATE["cur_deploy"]
    if cur is None or cur.shape != (n_layers, n_dev, exp_per_dev):
        cur = _init_deploy(n_layers, n_dev, n_experts, n_phys)
        _STATE["cur_deploy"] = cur

    smoothed = _STATE["ewma"]
    deploy_out = cur.copy()

    # Step 2-4: per-layer propose, anchor, gate.
    candidates: list[tuple[float, int, np.ndarray]] = []
    for L in range(n_layers):
        cur_par = _par_of(smoothed[L], cur[L])
        if cur_par < GATE_MIN_CUR_PAR:
            continue
        proposed = _propose_layer(smoothed[L], n_dev, n_phys)
        proposed = _anchor_to_prev(proposed, cur[L], n_experts)
        new_par = _par_of(smoothed[L], proposed)
        gain = cur_par - new_par
        if gain <= GATE_MIN_GAIN:
            continue
        cost = int(np.sum(cur[L] != proposed))
        if cost == 0:
            continue
        net = gain * SEC_PER_PAR - cost * SEC_PER_MOVE * GATE_SAFETY
        if net > 0.0:
            candidates.append((net, L, proposed))

    if not candidates:
        return False, np.array([], dtype=np.int64), deploy_out, None

    # Step 5: highest net seconds saved first.
    candidates.sort(reverse=True, key=lambda x: x[0])
    chosen = np.empty(len(candidates), dtype=np.int64)
    for k, (_net, L, proposed) in enumerate(candidates):
        deploy_out[L] = proposed
        cur[L] = proposed
        chosen[k] = L

    return True, chosen, deploy_out, None
