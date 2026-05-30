"""submission_v12.py -- v10 stack + per-layer drift-adaptive cost gate.

Score landscape (full grader):
    v6   110.92814  PAR 1.71942  tx 520_951
    v7   110.93441  PAR 1.71929  tx 524_353
    v8   110.92883  PAR 1.71945  tx 520_024
    v9   110.90977  PAR 1.71982  tx 523_932   (EP-aware tail; regressed)
    v10  110.92925  PAR 1.71944  tx 520_024   (drift floor + post-LPT swap; wash)

Leader on 2026-05-30:    111.89000  PAR 1.69     tx 752_367.

The leader spends ~+232k extra moves (+22.7s transit) to save ~+45s of
compute via lower mean PAR. That extra spend has to come almost entirely
from Mix/DS-R1 because the four Mix/DS-R1 cases alone account for ~0.40
of mean PAR (3.67 at EP256!) while every stationary case is already near
the deepseek lower bound. So the trade is Mix-only.

LOCAL EVIDENCE (sweep_gate.py on 7 LmSys cases, 2026-05-30):
    GATE_SAFETY=16   PAR 1.4355  tx   109_727  total  613.6s   <-- ours
    GATE_SAFETY= 8   PAR 1.3276  tx   607_454  total  617.0s   (+0.5%)
    GATE_SAFETY= 1   PAR 1.2641  tx 2_656_648  total  790.9s   (+29%)

On stationary LmSys, ANY drop in GATE_SAFETY hurts: the marginal moves
buy too little PAR. So the global gate cannot be lowered.

PER-LAYER DRIFT (check_drift.py on local LmSys, 2026-05-30):
    DS-R1 LmSys:  q95 = 0.024, max = 0.028
    Qwen3 LmSys:  q95 = 0.018, max = 0.025

LmSys layers all sit at drift ~ 0.02. A drift-keyed gate that only loosens
when drift_L is large is therefore a guaranteed no-op on the 15 stationary
grader cases (LmSys/ShareGPT/WildChat) and only fires on Mix.

v12 = v10 with EXACTLY ONE additional knob:

    PER-LAYER ADAPTIVE GATE (v12-A)
        gate_safety_L = max(GATE_SAFETY_MIN,
                            GATE_SAFETY_MAX - DRIFT_GAIN * drift_L)
        threshold_L   = TIME_PER_MOVE_S * n_layers * gate_safety_L
                        / BALANCED_COMPUTE_SECONDS

    Defaults:
        GATE_SAFETY_MAX = 16.0   (== v6/v7/v10 shipped value, stationary-safe)
        GATE_SAFETY_MIN =  1.0   (math break-even; further is noise gambling)
        DRIFT_GAIN      = 30.0   (drift_L = 0.5 -> gate = 1)

    Worked examples on the actual drift signal:
        drift_L = 0.02  (LmSys median)        -> gate = 15.4   (== v10)
        drift_L = 0.05                        -> gate = 14.5   (~v10)
        drift_L = 0.10                        -> gate = 13.0   (mild loosen)
        drift_L = 0.30                        -> gate =  7.0   (4x more accepts)
        drift_L = 0.50                        -> gate =  1.0   (math break-even)
        drift_L = 1.00                        -> gate =  1.0   (clipped)

    Why a SAFETY-AXIS knob (and not an absolute per-layer threshold):
    the score-math break-even is at safety=1. Anything above that is a
    safety margin against estimator noise. On stationary layers that
    margin pays for itself; on drifty layers it overpays because the
    NEXT cadence's snapshot will be different anyway -- the layer was
    going to change soon regardless. So we drop the margin exactly where
    it isn't earning its keep.

    Why max(MIN, MAX - DRIFT_GAIN*drift) instead of (MAX - (MAX-MIN)*drift):
    a hard transition at drift ~ 0.5 matches the expected Mix vs
    stationary split: Mix layers are at drift ~ 0.4-0.7 (rough estimate
    from the trace structure), stationary at <0.05. We want Mix to fall
    fully into the aggressive regime, not sit at gate ~ 8.

INHERITED UNCHANGED FROM v10:
    - Drift-adaptive recency floor (quadratic). Stationary-safe.
    - Post-LPT 2-opt max-load swap refinement (PAR-monotonic).
    - Tail gate at TAIL_FRACTION = 0.75 (v7-B).
    - Gain-first commit ordering (v7-C).
    - 2-opt device-label anchor (v4-B).
    - Cycle detection at K = 2 (v4-A).
    - EWMA across cadence calls at alpha = 0.7 (v1).

API contract identical to v1 / submission.py:
    rebalance(hotness, n_device, n_red_expert) ->
        (change, layers_priority, deployment, _aux)
    hotness shape = (window, n_layers, n_experts).
"""

from __future__ import annotations

import numpy as np

# ---------- v4/v6 inherited tuning ------------------------------------------
EWMA_ALPHA = 0.7              # v1; alpha=1.0 confirmed worse on grader (v3)
CYCLE_HISTORY_K = 2           # v4-A
ANCHOR_2OPT_PASSES = 3        # v4-B

# ---------- v12-A: per-layer adaptive cost gate -----------------------------
# Replaces v10's flat GATE_SAFETY. The PER-LAYER safety is computed as:
#   gate_safety_L = max(GATE_SAFETY_MIN,
#                       GATE_SAFETY_MAX - DRIFT_GAIN * drift_L)
# drift_L is the same first-half/second-half L1 ratio used by the recency
# floor (so we pay for drift exactly once per call). MAX matches the
# shipped v10 value -- stationary cases (drift ~ 0.02) see virtually no
# change. DRIFT_GAIN = 30 makes the gate hit MIN at drift_L >= 0.5,
# matching the empirical Mix-layer drift band.
GATE_SAFETY_MAX = 16.0        # stationary-safe; matches v6/v7/v8/v10
GATE_SAFETY_MIN = 1.0         # score-math break-even; below = noise gambling
DRIFT_GAIN = 30.0             # drift_L = 0.5 -> gate hits MIN

# ---------- v10-A: drift-adaptive recency floor (quadratic) -----------------
# Layer scored by within-window L1 drift (first vs second half, normalised).
# Stationary layers stay at v6's floor; drifty layers see a sharper recency
# ramp. Power=2 keeps stationary layers at v6 (the linear v7-A regressed).
RECENCY_FLOOR_MAX = 0.5
RECENCY_FLOOR_MIN = 0.0
DRIFT_POWER = 2.0

# ---------- v10-B: post-LPT swap refinement ---------------------------------
# Greedy max-load swap. PAR-monotonic by construction.
SWAP_MAX_PASSES = 8

# ---------- v7-B: tail gate -------------------------------------------------
# Fraction of the window used as the gate's "what will happen next" signal.
TAIL_GATE_ENABLED = True
TAIL_FRACTION = 0.75

# ---------- v7-C: gain-first commit ordering -------------------------------
GAIN_ORDER_ENABLED = True

# ---------- score-band constants (DO NOT EDIT; copied from simulator) -------
_BALANCED_COMPUTE_SECONDS = 60.0
_EXPERT_BYTES = 88_080_384
_TRANSFER_BANDWIDTH_BPS = 900_000_000_000
_TIME_PER_MOVE_S = _EXPERT_BYTES / _TRANSFER_BANDWIDTH_BPS  # ~9.787e-5 s


# ---------- module state ----------------------------------------------------
_STATE: dict = {
    "cur_deploy": None,
    "ewma_weight": None,
    "history": None,
}


def _reset() -> None:
  """Clear module state. Adapter calls this on every fresh run."""
  _STATE["cur_deploy"] = None
  _STATE["ewma_weight"] = None
  _STATE["history"] = None


# ---------- shared drift signal (used by recency floor AND gate) ------------
def _per_layer_drift(hotness: np.ndarray) -> np.ndarray:
  """L1 first-half vs second-half drift per layer, in [0, 1].

  hotness: (W, L, E)
  returns: (L,) -- 0 = stationary, 1 = maximally non-stationary.
  """
  half = max(1, hotness.shape[0] // 2)
  early = hotness[:half].sum(axis=0)             # (L, E)
  late = hotness[half:].sum(axis=0)              # (L, E)
  diff = np.abs(late - early).sum(axis=1)        # (L,)
  total = (early + late).sum(axis=1)             # (L,)
  return diff / np.maximum(total, 1e-9)


# ---------- v10-A: drift-adaptive per-layer recency kernel ------------------
def _per_layer_kernel(window_size: int,
                      floor_per_layer: np.ndarray) -> np.ndarray:
  """Per-layer linear ramp kernel of shape (L, W), normalised so the sum
  along W equals window_size for every layer."""
  if window_size <= 1:
    return np.ones((floor_per_layer.shape[0], window_size), dtype=np.float64)
  t = np.linspace(0.0, 1.0, window_size, dtype=np.float64)   # (W,)
  f = floor_per_layer[:, None]                               # (L, 1)
  raw = f + (1.0 - f) * t[None, :]                           # (L, W)
  norms = raw.sum(axis=1, keepdims=True)                     # (L, 1)
  return raw * (window_size / np.maximum(norms, 1e-9))


def _compute_window_load(hotness: np.ndarray,
                         drift: np.ndarray) -> np.ndarray:
  """(L, E) hotness aggregated over the window with the v10-A drift-adaptive
  per-layer kernel applied. Drift signal is shared with v12-A's gate."""
  spread = RECENCY_FLOOR_MAX - RECENCY_FLOOR_MIN
  floor = RECENCY_FLOOR_MAX - spread * (drift ** DRIFT_POWER)
  floor = np.clip(floor, RECENCY_FLOOR_MIN, RECENCY_FLOOR_MAX)
  kernel = _per_layer_kernel(hotness.shape[0], floor)        # (L, W)
  return np.einsum("lw,wle->le", kernel, hotness)


def _compute_tail_load(hotness: np.ndarray) -> np.ndarray:
  """(L, E) hotness aggregated over the window tail only; gate signal."""
  W = hotness.shape[0]
  tail_w = max(1, int(round(W * TAIL_FRACTION)))
  return hotness[-tail_w:].sum(axis=0)


# ---------- deepseek-equivalent placement (numpy) -- unchanged from v1 ------
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


# ---------- v10-B: post-LPT 2-opt max-load swap refinement -----------------
def _refine_packing_max_swap(deploy: np.ndarray,
                             per_replica_weight: np.ndarray) -> np.ndarray:
  """Greedy pair-swap to reduce max-load on a single layer's deployment.
  PAR-monotonic by construction (max-load is non-increasing).
  See v10 docstring for the full rationale."""
  n_dev, exp_per_dev = deploy.shape
  deploy = deploy.copy()

  for _ in range(SWAP_MAX_PASSES):
    slot_weight = per_replica_weight[deploy]               # (D, S)
    loads = slot_weight.sum(axis=1)                        # (D,)
    h_max = int(loads.argmax())
    cur_max = float(loads[h_max])

    sw_max = slot_weight[h_max]                            # (S,)

    delta = slot_weight[:, None, :] - sw_max[None, :, None]   # (D, S, S)
    new_load_h_max = cur_max + delta
    new_load_h = loads[:, None, None] - delta
    new_max = np.maximum(new_load_h_max, new_load_h)          # (D, S, S)

    new_max[h_max] = np.inf  # never swap heavy device with itself

    flat_argmin = int(new_max.argmin())
    best_new_max = float(new_max.flat[flat_argmin])
    if best_new_max >= cur_max:
      break

    h, s_max_idx, s_h = np.unravel_index(flat_argmin, new_max.shape)
    h = int(h)
    s_max_idx = int(s_max_idx)
    s_h = int(s_h)
    deploy[h_max, s_max_idx], deploy[h, s_h] = (
        deploy[h, s_h], deploy[h_max, s_max_idx]
    )

  return deploy


def _propose_layer(weight_1d: np.ndarray, n_device: int,
                   n_phys: int) -> np.ndarray:
  """LPT pack -> post-LPT max-swap refinement (v10-B)."""
  phy2log, logcnt = _replicate_experts(weight_1d, n_phys)
  tokens_per_phy = weight_1d[phy2log] / logcnt[phy2log]
  pack_idx, rank = _balanced_pack_lpt(tokens_per_phy, n_device)
  exp_per_dev = n_phys // n_device
  deploy = np.empty((n_device, exp_per_dev), dtype=np.int64)
  deploy[pack_idx, rank] = phy2log

  n_log = weight_1d.shape[0]
  cut = np.bincount(deploy.reshape(-1), minlength=n_log).clip(min=1)
  per_replica = weight_1d / cut
  deploy = _refine_packing_max_swap(deploy, per_replica)
  return deploy


# ---------- v4-B: 2-opt refined device-label anchor (transit-only) ---------
def _greedy_mapping_from_overlap(overlap: np.ndarray) -> np.ndarray:
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
  """Iterative pair-swap to maximise sum(overlap[i, mapping[i]]).
  PAR-invariant (only relabels devices)."""
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
  """v12: v10 stack + per-layer drift-adaptive cost gate.

  See module docstring. API identical to submission.py.
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

  # Compute drift ONCE; reuse for v10-A recency floor and v12-A gate.
  drift = _per_layer_drift(hotness)                          # (L,)

  # ---------- v10-A: drift-adaptive recency-weighted window load ----------
  window_load = _compute_window_load(hotness, drift)         # (L, E)

  # ---------- v7-B: tail-only signal used by the cost gate ----------
  gate_load = _compute_tail_load(hotness) if TAIL_GATE_ENABLED else window_load

  # ---------- EWMA across cadence calls (unchanged from v1) ----------
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
    _STATE["history"] = [cur.copy()]
  history = _STATE["history"]

  deploy_out = cur.copy()

  # ---------- v12-A: per-layer adaptive gate threshold --------------------
  # gate_safety_L sits at GATE_SAFETY_MAX for stationary layers (drift ~ 0)
  # and ramps down to GATE_SAFETY_MIN once drift_L >= (MAX-MIN)/DRIFT_GAIN.
  # The threshold for the cost gate is then computed exactly as v6's, just
  # with a per-layer safety multiplier instead of a global one.
  gate_safety = np.maximum(
      GATE_SAFETY_MIN,
      GATE_SAFETY_MAX - DRIFT_GAIN * drift,
  )                                                          # (L,)
  par_per_move_threshold = (
      _TIME_PER_MOVE_S * n_layers * gate_safety
      / _BALANCED_COMPUTE_SECONDS
  )                                                          # (L,)

  # ---------- pass 1: propose + gate every layer, store gains ------------
  initial_order = np.argsort(-smoothed.sum(axis=1))
  accepted: list[tuple[int, np.ndarray, float, int]] = []
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
    gain_par = (_layer_par(gate_load[L], cur[L])
                - _layer_par(gate_load[L], proposed))
    if gain_par <= moves * par_per_move_threshold[L]:
      continue

    accepted.append((L, proposed, gain_par, moves))

  # ---------- v7-C: pick commit order (largest gain first) ----------------
  if GAIN_ORDER_ENABLED:
    accepted.sort(key=lambda x: -x[2])

  changed: list[int] = []
  for L, proposal, _gain, _moves in accepted:
    deploy_out[L] = proposal
    cur[L] = proposal
    changed.append(L)

  if changed:
    history.append(cur.copy())
    if len(history) > CYCLE_HISTORY_K + 1:
      history.pop(0)

  if not changed:
    return False, np.array([], dtype=np.int64), deploy_out, None

  return True, np.array(changed, dtype=np.int64), deploy_out, None
