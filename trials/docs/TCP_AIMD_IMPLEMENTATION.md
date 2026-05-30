# TCP-AIMD-based gate (v15, shipped)

How we mapped **TCP-Reno's AIMD congestion-control loop** (Jacobson,
SIGCOMM 1988) onto the per-layer redeploy gate in our MoE
expert-placement problem.

- **Code:** [`submission_v15.py`](../submissions/submission_v15.py) (shipped as `submission.py`)
- **Sister doc:** [`ARC_IMPLEMENTATION.md`](./ARC_IMPLEMENTATION.md) — the earlier evidence-driven attempt this replaces.

> v15 keeps every v7 component byte-for-byte and only **wraps the
> commit gate in a closed-loop AIMD controller**. The first call is
> bit-exact equivalent to v7; adaptation only kicks in after we have
> observed a prediction residual on a layer we actually committed.

---

## TL;DR

| Metric    | v7 baseline | v15 (AIMD gate) | Δ              |
| --------- | ----------: | --------------: | -------------- |
| Score     |     110.934 |     **110.968** | **+0.034**     |
| Mean PAR  |      1.7193 |      **1.7184** | better         |
| Transit   |     524 k   |     531 k       | +1.3 % (worse) |

Largest grader-confirmed gain after v4's anchor. `safety[L]` drifts
*down* on stationary traces (LmSys / ShareGPT / WildChat / Qwen3)
where predictions land within ~5 %, and ratchets back *up* on Mix
EP128 / EP256 where predictions miss by 20–50 %.

---

## The idea

ARC (see [`ARC_IMPLEMENTATION.md`](./ARC_IMPLEMENTATION.md)) gave us
the right philosophy — *adapt the policy to evidence* — but the wrong
mechanism: stacked additive bumps on a static safety of 16 froze
moves on volatile workloads.

TCP-Reno solves the same problem in networking: how do you decide
how aggressive to be when you don't know the channel? Answer: pick a
default, watch for a loss signal, react multiplicatively.

We do the same thing per layer. Each layer gets a **trust dial**
(`safety[L]`). We promise a PAR after every commit, check it next
round, and turn the dial down on success or up on failure.

---

## Intuition (the trust-dial version)

Each layer has its own **trust dial**. We promise a result, check
it next round, and turn the dial.

- **Promise.** On every commit we write down: *"this layer's PAR
  will be X."*
- **Check.** Next call we measure: *was it X?* → `err = actual / predicted`.
- **Kept promise → braver.** `err ≤ 1.05` → dial × **0.85**. Move
  more freely next time.
- **Broken promise → shyer.** `err ≥ 1.30` → dial × **1.30**.
  Demand more proof next time.
- **Meh promise → ignore.** Inside `[1.05, 1.30]` we don't touch
  the dial. Noise shouldn't move it.
- **Floor & ceiling.** Dial clipped to `[1, 16]`. The ceiling is
  v7's old fixed value, so **v15 can never be more scared than v7**.
- **Day one is v7.** First call has nothing to check, so the dial
  starts at 16 everywhere — bit-exact v7 behaviour.

**One-liner.** v7 picks *where* experts go. v15 adds a per-layer
"how much proof do I need?" knob that learns from broken promises —
TCP-style: slow down after a jam, speed up when the road is clear,
but never slower than the old speed limit.

---

## Conceptual mapping

| TCP-Reno concept                  | v15 equivalent                                                  |
| --------------------------------- | --------------------------------------------------------------- |
| Congestion window `cwnd`          | Per-layer gate aggressiveness `safety_per_layer[L]`             |
| ACK on a sent segment             | Layer's last-call PAR prediction matched reality (`err ≤ 1.05`) |
| Loss event (timeout / 3 dup ACKs) | Layer's last-call PAR prediction missed badly (`err ≥ 1.30`)    |
| Additive increase                 | *Disabled* — see "Why asymmetric" below                         |
| Multiplicative decrease           | `safety[L] *= BETA_DOWN` on success (gate gets more aggressive) |
| Multiplicative back-off on loss   | `safety[L] *= BETA_UP` on miss (gate gets more conservative)    |
| `ssthresh` / cwnd ceiling         | `SAFETY_MAX = 16.0` (== v7's static `GATE_SAFETY`)              |
| Minimum cwnd                      | `SAFETY_MIN = 1.0` (literal math break-even)                    |

The placement algorithm (DeepSeek replicate + LPT pack), the EWMA
smoother, the 2-opt device anchor, and the cycle-detection history
are all **unchanged from v7**. AIMD only modulates the **commit
threshold** per layer.

---

## The control loop

### Signal — prediction residual

After a commit, we record what PAR we *expected* the redeployed
layer to have under the current gate signal. On the next call, we
evaluate the same layer against the new `gate_load`:

```
err[L] = actual_par_after_commit / predicted_par_after_commit
```

`err` is *a posteriori* evidence: how well last call's plan held up.

### Update rule (asymmetric AIMD)

```
if err <= TRUST_HIGH:    safety[L] *= BETA_DOWN     # prediction was reliable
elif err >= TRUST_LOW:   safety[L] *= BETA_UP       # prediction was unreliable
else:                    pass                       # dead zone — stability
safety[L] = clip(safety[L], SAFETY_MIN, SAFETY_MAX)
```

| Constant      | Value | Role                                                |
| ------------- | ----: | --------------------------------------------------- |
| `BETA_DOWN`   |  0.85 | ~5 good calls to halve safety                       |
| `BETA_UP`     |  1.30 | ~3 bad calls to roughly double back off             |
| `TRUST_HIGH`  |  1.05 | Treat ≤ 5 % over-prediction as "right"              |
| `TRUST_LOW`   |  1.30 | Treat ≥ 30 % over-prediction as "wrong"             |
| `SAFETY_MIN`  |   1.0 | Math break-even floor                               |
| `SAFETY_MAX`  |  16.0 | Hard cap == v7 default. v15 ≥ v7 in aggressiveness. |

### Gate

The per-layer commit threshold is rebuilt every call from the live
`safety[L]`:

```
threshold[L] = TIME_PER_MOVE * n_layers * safety[L] / 60.0
commit iff gain_par > moves * threshold[L]
```

`SAFETY_MAX = 16.0` enforces a strict contract: **v15 can never be
more conservative than v7** — only equal or more aggressive.

---

## Per-call flow (v15 `rebalance`)

```
1.  smoothed = EWMA(α=0.7) over window_load                # v7 carry-over
2.  gate_load = tail-only signal (last 75% of window)      # v7 carry-over
3.  Seed cur_deploy / history / safety_per_layer on first call
    safety_per_layer[L] = GATE_SAFETY = 16.0  (== v7 default)

4.  AIMD evidence step (skipped on first call):
      for L in last_committed_layers:
        actual = layer_par(gate_load[L], cur[L])
        pred   = last_predicted_par[L]
        err    = actual / pred
        update safety_per_layer[L] per the rule above
        clip into [SAFETY_MIN, SAFETY_MAX]

5.  Per-layer threshold from per-layer safety:
      threshold_per_layer = TIME_PER_MOVE * n_layers * safety / 60

6.  For each layer L, heaviest-first:
      proposed = _propose_layer(smoothed[L], ...)           # v7 placement
      proposed = anchor_device_labels(proposed, cur[L])     # v7 anchor
      proposed = anchor_slot_order(proposed, cur[L])
      if proposed == cur[L]: continue
      if cycle: continue                                    # v4-A cycle filter
      gain_par = layer_par(gate_load, cur) - layer_par(gate_load, proposed)
      if gain_par <= moves * threshold_per_layer[L]: continue   # AIMD gate

      accepted.append((L, proposed, gain_par, moves))

7.  Commit largest-gain-first (v7-C). For each commit:
      next_predicted_par[L] = layer_par(gate_load[L], proposal)

8.  Persist for next-call evidence step:
      last_committed_layers = set(committed L)
      last_predicted_par    = next_predicted_par
```

---

## Why asymmetric (and why it's not literal TCP)

Stock TCP-Reno is *additive* increase, *multiplicative* decrease.
v15 is **MIMD-with-a-cap**: both directions are multiplicative, and
the upward direction is capped at v7's static safety value. Two
reasons:

1. **Hard "no-regression vs v7" contract.** Capping at
   `SAFETY_MAX = 16` guarantees the gate cannot become stricter than
   v7's. Any miss just walks back to v7 behaviour, never past it.
2. **Wide dead zone `[1.05, 1.30]`.** PAR residuals are inherently
   noisy because one cadence window mixes pre- and post-deploy
   iterations. A wide neutral band prevents jitter from triggering a
   `BETA_UP` that would just freeze us at the cap.

We **borrow the control philosophy** from TCP congestion control
(Jacobson, SIGCOMM '88): react to what actually happened, grow
trust on confirmation, shrink it on failure, keep a bounded dial. We
do **not** implement TCP-Reno (no packets, no RTT, no additive
`cwnd += 1`). v15 is a **MIMD-style** per-layer gate on PAR
prediction residuals — inverted semantics (smaller `safety` =
braver), plus a dead zone. *"TCP-AIMD" here means inspired by, not
is.*

---

## Where to look in the code

All locations refer to [`submission_v15.py`](../submissions/submission_v15.py):

| Concern                                 | Location                                       |
| --------------------------------------- | ---------------------------------------------- |
| AIMD constants (`BETA_*`, `TRUST_*`, …) | lines ~137–148                                 |
| Per-layer state (`safety_per_layer`, …) | `_STATE`, lines ~160–168                       |
| Update rule                             | `_update_safety_from_evidence`, lines ~347–373 |
| Evidence step (called before the gate)  | `rebalance`, lines ~419–431                    |
| Gate using per-layer threshold          | `rebalance`, lines ~437–469                    |
| Recording predictions for next call     | `rebalance`, lines ~477–491                    |

The state seeded at `safety_per_layer[L] = GATE_SAFETY` plus the
"skip update on first call" guard in `_update_safety_from_evidence`
together give the bit-exact-as-v7-on-call-1 property.

---

## Citations / prior art

- **TCP-Reno AIMD** — V. Jacobson, *Congestion Avoidance and
  Control*, SIGCOMM 1988. The control philosophy.
- **ARC adaptive `p`** — Megiddo & Modha, FAST 2003. Same family of
  *evidence-driven retuning* but a different mechanism. See
  [`ARC_IMPLEMENTATION.md`](./ARC_IMPLEMENTATION.md).
- **Vapor (ISPA21)** — per-worker batch-size AIMD on observed epoch
  time; same closed-loop spirit applied to a different decision.

For the full citation list see [`CITATIONS.md`](../../CITATIONS.md).
