"""STARE-LB v4 (REWARD + ANCHORED) - standalone safe default

Standalone copy with conservative SAFETY default to reduce churn. Drop as
`submission.py` for upload if desired.
"""
import numpy as np
import time
import torch

EXPERT_BYTES = 88_080_384
BW = 900_000_000_000
SEC_PER_MOVE = EXPERT_BYTES / BW
SEC_PER_PAR = 60.0


def balanced_packing(weight, num_packs):
    num_layers, num_groups = weight.shape
    groups_per_pack = num_groups // num_packs
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64).expand(weight.shape)
        return pack_index, torch.zeros_like(weight, dtype=torch.int64)
    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(pack_index, -1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                       key=pack_weights.__getitem__)
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight, num_phy):
    n, num_log = weight.shape
    phy2log = torch.arange(num_phy, dtype=torch.int64).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64)
    logcnt = torch.ones(n, num_log, dtype=torch.int64)
    arangen = torch.arange(n, dtype=torch.int64)
    for i in range(num_log, num_phy):
        idx = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = idx
        rank[:, i] = logcnt[arangen, idx]
        logcnt[arangen, idx] += 1
    return phy2log, rank, logcnt


def placement_for_weight(weight_1d, n_device, n_phys):
    n_experts = weight_1d.shape[0]
    w = torch.from_numpy(weight_1d.reshape(1, n_experts).astype(np.float32))
    phy2log, rank, logcnt = replicate_experts(w, n_phys)
    tokens_per_phy = (w / logcnt).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, n_device)
    exp_per_dev = n_phys // n_device
    phy2dev = pack_index * exp_per_dev + rank_in_pack
    inv = torch.empty_like(phy2dev)
    inv.scatter_(1, phy2dev, torch.arange(n_phys, dtype=torch.int64).expand(phy2dev.shape))
    log_of_phy = phy2log.gather(-1, inv)
    return log_of_phy.numpy().reshape(n_device, exp_per_dev)


def par_of(weight_1d, deployment):
    n_experts = weight_1d.shape[0]
    cut = np.maximum(np.bincount(deployment.reshape(-1), minlength=n_experts), 1)
    w = weight_1d / cut
    loads = w[deployment.reshape(-1)].reshape(deployment.shape).sum(-1)
    m = loads.mean()
    return float(loads.max() / m) if m > 0 else 1.0


def init_deploy(n_layers, n_device, n_experts, n_phys):
    exp_per_dev = n_phys // n_device
    dep = np.zeros((n_layers, n_device, exp_per_dev), dtype=np.int64)
    for L in range(n_layers):
        for d in range(n_device):
            for s in range(exp_per_dev - 1):
                dep[L, d, s] = (d * (exp_per_dev - 1) + s) % n_experts
            dep[L, d, -1] = dep[L, d, -2]
    return dep


def anchored_placement(weight_1d, n_device, n_phys, prev_deploy):
    new = placement_for_weight(weight_1d, n_device, n_phys)
    if prev_deploy is None:
        return new
    n_dev = new.shape[0]
    prev_sets = [set(prev_deploy[d].tolist()) for d in range(n_dev)]
    new_sets = [set(new[d].tolist()) for d in range(n_dev)]
    used_prev = set()
    mapping = {}
    order = sorted(range(n_dev), key=lambda i: -max(len(new_sets[i] & prev_sets[j]) for j in range(n_dev)))
    for i in order:
        best_j, best_ov = None, -1
        for j in range(n_dev):
            if j in used_prev:
                continue
            ov = len(new_sets[i] & prev_sets[j])
            if ov > best_ov:
                best_ov, best_j = ov, j
        mapping[i] = best_j
        used_prev.add(best_j)
    out = np.zeros_like(new)
    for i in range(n_dev):
        out[mapping[i]] = new[i]
    return out

_STATE = {"ewma": None, "fast": None, "slow": None, "cur": None}


def _reset():
    _STATE["ewma"] = None; _STATE["fast"] = None
    _STATE["slow"] = None; _STATE["cur"] = None

ALPHA = 0.2
CLIP_PCTL = 99.0
MIN_PAR = 1.02
# Conservative default safety
SAFETY = 200.0


def rebalance(hotness, n_device, n_red_expert):
    hotness = np.asarray(hotness)
    n_layers, n_experts = hotness.shape[1], hotness.shape[2]
    n_phys = n_experts + n_red_expert
    w = hotness.sum(0).astype(np.float64)
    exp_per_dev = n_phys // n_device
    expected_cur_shape = (n_layers, n_device, exp_per_dev)
    expected_ewma_shape = (n_layers, n_experts)
    if _STATE.get("ewma") is None or _STATE.get("ewma").shape != expected_ewma_shape:
        _STATE["ewma"] = w.copy()
        _STATE["cur"] = init_deploy(n_layers, n_device, n_experts, n_phys)
    else:
        cur = _STATE.get("cur")
        if cur is None or cur.shape != expected_cur_shape:
            _STATE["cur"] = init_deploy(n_layers, n_device, n_experts, n_phys)
    cap = np.percentile(w, CLIP_PCTL, axis=1, keepdims=True)
    w = np.minimum(w, cap)
    if _STATE["ewma"] is None:
        _STATE["ewma"] = w.copy()
    else:
        _STATE["ewma"] = ALPHA * w + (1 - ALPHA) * _STATE["ewma"]
    ewma = _STATE["ewma"]; cur = _STATE["cur"]; dep = cur.copy()
    # Keep the search bounded even on EP32; DS-R1-sized traces need the same
    # cap or the controller spends too long evaluating every layer.
    if n_experts >= 256:
        K = 2 if n_device <= 32 else 3
    elif n_device <= 32:
        K = 4
    else:
        K = 3 if n_device <= 64 else 2
    K = min(K, n_layers)
    layer_totals = w.sum(axis=1)
    cand_idx = np.argpartition(-layer_totals, K - 1)[:K]
    cand = []
    for L in cand_idx:
        prop = anchored_placement(ewma[L], n_device, n_phys, cur[L])
        if _STATE.get("cur") is None or _STATE.get("cur").shape[1:] != prop.shape:
            _STATE["cur"] = init_deploy(n_layers, n_device, n_experts, n_phys)
        cur = _STATE["cur"]
        cur_par = par_of(ewma[L], cur[L]); new_par = par_of(ewma[L], prop)
        gain = cur_par - new_par; cost = int(np.sum(cur[L] != prop))
        if cost == 0 or cur_par < MIN_PAR:
            continue
        net = gain * SEC_PER_PAR - cost * SEC_PER_MOVE * SAFETY
        if net > 0:
            cand.append((net, L, prop))
    if not cand:
        return False, np.array([], dtype=np.int64), dep, None
    cand.sort(reverse=True, key=lambda x: x[0])
    chosen = []
    for _, L, prop in cand:
        dep[L] = prop; cur[L] = prop; chosen.append(L)
    return True, np.array(chosen, dtype=np.int64), dep, None
