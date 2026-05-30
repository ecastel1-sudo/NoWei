# ARC-based attempt (v2)

Short writeup of how we tried to use **ARC** (Adaptive Replacement
Cache, Megiddo & Modha, FAST 2003) to control redeploys in our MoE
expert-placement problem.

- **Code:** [`submission_v2.py`](../submissions/submission_v2.py)
- **Transit-tuned retune:** [`submission_v2_lowest_trans.py`](../submissions/submission_v2_lowest_trans.py)
- **Later hybrid on top of v15:** [`submission_v18.py`](../submissions/submission_v18.py)
- **Sister doc:** [`TCP_AIMD_IMPLEMENTATION.md`](./TCP_AIMD_IMPLEMENTATION.md) — the closed-loop gate that eventually replaced this.

> The shipped submission (v15) does **not** use this. v2 lost grader
> score and we moved on. But it reshaped every later version, so it's
> worth keeping the notes.

---

## TL;DR

| Metric    | v1 baseline | v2 (ARC overlays) | Δ              |
| --------- | ----------: | ----------------: | -------------- |
| Score     |      110.86 |        **109.37** | **−1.49**      |
| Mean PAR  |       1.722 |             1.779 | +0.057 (worse) |
| Transit   |      533 k  |        **391 k**  | **−27 %**      |

**Lowest transit any version of the chain ever hit.** But the grader
weights PAR ~15× more than transit at the leaderboard band, so the PAR
drift swallowed the win. Right idea, wrong cost function.

---

## The idea

ARC is a CPU/disk cache that decides what to evict by keeping two
small memories of *recently evicted* items: things that turned out to
be one-off, and things that turned out to be persistently hot. Every
time something it kicked out comes back, ARC takes that as feedback
and shifts its policy a bit. It's constantly retuning itself based on
what it just got wrong.

The analogy to our problem is direct. We're not caching pages, we're
deciding which experts get more than one replica on the GPUs, but the
question is the same: *am I keeping the right things around, and am I
throwing out the right things?* An expert we demoted from two replicas
to one, and then immediately had to redeploy, is basically a ghost
hit. An expert that's been hot for three cadence calls in a row is a
T2-style frequent item that shouldn't be displaced by a one-call
spike.

v2 was an attempt to take that vocabulary literally and bolt it onto
v1.

---

## Intuition (the toy-room version)

Think of experts as **toys** and GPUs as **shelves**. Popular toys
get a spare copy. Rearranging the room is annoying (that's transit).
ARC told us three things to watch for:

- **"This toy is always popular" (ARC-A).** Not "everyone wanted it
  *once* this morning." If a toy has been hot for **3 checks in a
  row**, we whisper to the packer: *"this one counts a little extra."*
  So we don't shuffle the room because of a one-minute fad.
- **"We just cleaned this corner" (ARC-B).** If we just moved toys on
  shelf L, we say: *"don't touch shelf L again right away."* Like
  tidying the Lego corner — don't dump it all out again 2 minutes
  later. The rule fades after a couple of turns.
- **"Oops, we put that toy away and they want it again" (ARC-C —
  ghost).** We removed a spare copy of a toy (demoted 2 → 1). Next
  turn it's hot again. ARC calls that a **ghost**: *"we were wrong
  to put it away."* So we make the whole shelf harder to rearrange
  until things calm down.

---

## The three mechanisms in detail

v1 was already running EWMA on the per-expert load, feeding that into
the DeepSeek replicate-and-pack core, anchoring the result against
the previous placement, then deciding whether to commit using a cost
gate. **v2 leaves all of that alone.** It adds three things on top.

### ARC-A — protect persistently-hot experts (input side)

A small bias on the input to the packer. We keep a per-expert counter
that goes up every time the expert sits above 1.5× the layer's mean
load, and resets when it doesn't. Once that counter reaches **3 calls
in a row**, we multiply the expert's weight by **1.10** before handing
it to the packer (ramped linearly below 3). The packer doesn't know
anything has changed — it just sees a slightly nudged input where
persistent hot experts get a small advantage over fresh spikes.

### ARC-B — cooldown right after a redeploy (gate side)

Makes the cost gate stricter right after we move a layer. v1 used a
fixed safety multiplier of **16**, meaning a proposed redeploy had to
show a PAR gain worth at least 16× the break-even cost of its moves.
v2 adds an exponentially decaying bump on top:

```
cooldown_bump[L] = 16 * exp(-(call − last_redeploy[L]) / 1.5)
```

At `age = 0` the layer is **32× harder** to convince than v1 was. The
bump is gone within ~3 calls. Point: stop the algorithm from chasing
the workload back and forth across a layer it just paid to rebalance.

### ARC-C — ghost-hit boost (gate side, most literally ARC)

When we commit a redeploy, we mark every expert whose replica count
went from "more than one" down to "exactly one" — the **demotions**.
If on a later call one of those recently-demoted experts is hot again,
that's a **ghost hit**. We use the fraction of currently-hot experts
on the layer that are ghosts as another additive bump on the gate's
safety multiplier, capped at **+8**:

```
ghost_bump[L] = 8 * (ghost_active[L].sum() / hot_mask[L].sum())
```

If our previous demotions are coming back to bite us, distrust the
next proposal until things settle.

### The combined gate

```
effective_safety[L] = 16 + cooldown_bump[L] + ghost_bump[L]
threshold[L]        = base * effective_safety[L]
commit iff gain_par > moves * threshold[L]
```

All three bumps are **additive on top of 16**, so v2's gate is always
at least as strict as v1's, never more permissive. Transit can only
go down. PAR is allowed to drift, but only by the small amount driven
by the T2 boost on the input side.

---

## Per-call flow (v2 `rebalance`)

```
1.  call_count += 1
2.  smoothed = EWMA(α=0.7) over window_load                 # v1, unchanged
3.  Seed per-layer ARC state on first call

4.  hot_mask  = smoothed > 1.5 * layer_mean
    t2_streak = (hot_mask) ? t2_streak + 1 : 0              # ARC-A counter
    boosted   = smoothed * (1 + 0.10 * t2_factor)           # ARC-A boost

5.  For each layer L, heaviest-first:
      proposed = deepseek_replicate_and_pack(boosted[L])
      proposed = anchor_device_labels(proposed, cur[L])
      proposed = anchor_slot_order(proposed, cur[L])
      if proposed == cur[L]: continue

      cooldown = 16 * exp(-(call − last_redeploy[L]) / 1.5)  # ARC-B
      ghosts   = recently_demoted[L] & hot_mask[L]
      ghost_add = 8 * (ghosts.sum() / hot_mask[L].sum())     # ARC-C

      threshold = base * (16 + cooldown + ghost_add)
      if gain_par <= moves * threshold: continue             # gate

      # commit
      newly_demoted = (cur_count > 1) & (new_count == 1)
      demoted_at_call[L, newly_demoted] = call_now           # ARC-C state
      last_redeploy[L] = call_now                            # ARC-B state
      cur[L] = proposed
```

No new placement algorithm, no re-ordering of the existing pipeline,
no router-level change. ARC shows up only as a reshape of the packer's
input and three extra terms in the gate's threshold.

---

## Why we dropped it

The grader's composite score is roughly `60s × mean_PAR` of compute
per case plus a small term for the actual data movement. On the 25-
case mix, compute is **~98 %** of total time, so **PAR matters about
15× more than transit** at the leaderboard band.

v2's gate was strict enough that it occasionally refused redeploys
that *would* have improved PAR — PAR drifted 1.722 → 1.779. The
transit savings (−27 %) just couldn't pay for that drift.

That single result reshaped the next several versions:

- **v3** tried the opposite extreme (no EWMA smoothing) and also
  lost, confirming the diagnosis.
- **v4** codified the lesson into a rule: keep only the ARC pieces
  that *cannot hurt PAR* even in principle. Two qualified —
  **cycle detection** (refuse a proposal that would send a layer
  back to a placement it just left) and a **2-opt anchor refinement**
  (a pure permutation, PAR-invariant). Both shipped. +0.51 on the
  grader — biggest single win of the chain after v15.
- **v15** went back to ARC's underlying idea — *adapt the policy to
  evidence* — but expressed it as a **TCP-AIMD per-layer trust dial**
  driven by prediction residuals, instead of stacking cooldown and
  ghost bumps on a static 16. See
  [`TCP_AIMD_IMPLEMENTATION.md`](./TCP_AIMD_IMPLEMENTATION.md).
- **v18** tried v15's adaptive base + scaled-down v2 overlays. Local
  sims didn't beat v15 cleanly on Mix/EP256; never went to the grader.

---

## Where to look in the code

| Concern                                                 | File                                                          |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| Full v2 implementation, sections labelled `ARC-A/B/C`   | [`submission_v2.py`](../submissions/submission_v2.py)         |
| Same overlays retuned for minimum transit               | [`submission_v2_lowest_trans.py`](../submissions/submission_v2_lowest_trans.py) |
| Hybrid: v15 AIMD base + scaled-down v2 overlays         | [`submission_v18.py`](../submissions/submission_v18.py)       |
| Cited references for ARC and the AIMD analogy           | [`CITATIONS.md`](../../CITATIONS.md) §9                       |

---

## Citations / prior art

- **ARC (Adaptive Replacement Cache)** — N. Megiddo & D. S. Modha,
  *ARC: A Self-Tuning, Low-Overhead Replacement Cache*, USENIX FAST
  2003. The T1/T2/B1/B2 vocabulary and the ghost-list-driven
  retuning are taken directly from here.
- **TCP-Reno AIMD** — V. Jacobson, *Congestion Avoidance and
  Control*, SIGCOMM 1988. Family of *evidence-driven retuning*; the
  mechanism v15 actually uses, see sister doc.

For the full citation list see [`CITATIONS.md`](../../CITATIONS.md).
