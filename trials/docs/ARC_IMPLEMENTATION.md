# ARC-based attempt

Short writeup of how we tried to use ARC (Adaptive Replacement Cache,
Megiddo & Modha, 2003) for the MoE placement problem, what worked, and
what didn't. The code is in
[`submission_v2.py`](../submissions/submission_v2.py); a later mix with
v15 is in [`submission_v18.py`](../submissions/submission_v18.py).

The shipped submission (v15) doesn't actually use this. v2 lost score
on the grader and we moved on. But the line of thinking it started is
the reason v4 and v15 look the way they do, so it's worth keeping the
notes.

## The idea

ARC is a CPU/disk cache that decides what to evict by keeping two
small memories of *recently evicted* items: things that turned out to
be one-off, and things that turned out to be persistently hot. Every
time something it kicked out comes back, ARC takes that as feedback
and shifts its policy a bit. So the cache is constantly retuning
itself based on what it just got wrong.

The analogy to our problem is pretty direct. We're not caching pages,
we're picking which experts get more than one replica on the GPUs, but
the question is the same: "am I keeping the right things around, and
am I throwing out the right things?" An expert we demoted from two
replicas down to one and then immediately had to redeploy is basically
a ghost hit. An expert that's been hot for three cadence calls in a
row is a T2-style frequent item that shouldn't be displaced by a
one-call spike.

So v2 was an attempt to take that vocabulary literally and bolt it
onto v1.

## What v2 actually does

v1 was already running EWMA on the per-expert load, then feeding that
into the DeepSeek replicate-and-pack core, then anchoring the result
against the previous placement, then deciding whether to commit using
a cost gate. v2 leaves all of that alone. It adds three things on top.

The first one, which we called ARC-A, is a small bias on the input to
the packer. We keep a per-expert counter that goes up every time the
expert sits above 1.5× the layer's mean load, and resets when it
doesn't. Once that counter reaches three calls in a row, we multiply
the expert's weight by 1.10 before handing it to the packer. The
packer doesn't know anything has changed; it just sees a slightly
nudged input where persistent hot experts get a small advantage over
fresh spikes.

The second piece, ARC-B, makes the cost gate stricter right after we
move a layer. v1 used a fixed safety multiplier of 16 — meaning a
proposed redeploy had to show a PAR gain worth at least 16 times the
break-even cost of the moves it required. v2 adds an exponentially
decaying bump on top of that. Right after a commit the bump is +16
(so the layer is now 32× harder to convince than v1 was), and it
fades out over a couple of cadence calls. The point is to stop the
algorithm from chasing the workload back and forth across a layer it
just paid to rebalance.

The third piece, ARC-C, is the one that's most literally ARC. When we
commit a redeploy we mark every expert whose replica count went from
"more than one" down to "exactly one" — those are the demotions. If
on a later call one of those recently-demoted experts is hot again,
that's a ghost hit, and we use the fraction of currently-hot experts
on the layer that are ghosts as another additive bump on the gate's
safety multiplier (capped at +8). The intuition is the same as ARC's
own: if our previous decisions are coming back to bite us, we should
distrust the next proposal until things settle.

All three bumps are additive on top of the base safety of 16, so the
gate in v2 is always at least as strict as in v1, and never more
permissive. That matters for reasoning about the result: transit can
only go down. PAR is allowed to drift, but only by a small amount
driven by the T2 boost on the input side.

## How a single call looks

Roughly, one `rebalance(...)` in v2 goes like this. The EWMA updates
the smoothed weight as before. The hot mask gets recomputed, the T2
streak counter goes up or resets, and we build the boosted weight
that the packer will actually see. Then we walk the layers from
heaviest to lightest. For each one we run the regular DeepSeek
replicate-and-pack, then the device-label anchor, then the slot
anchor — all unchanged from v1. If the proposal equals the current
placement we skip. Otherwise we compute the cooldown bump from how
many calls ago we last touched this layer, the ghost-hit ratio from
how many of the layer's currently-hot experts are recent demotions,
add both to the base safety of 16, and compare the PAR gain against
that threshold. If we commit, we figure out which experts are newly
demoted by this proposal and stamp the demotion time so future calls
can recognize them as ghosts.

That's the whole loop. There is no new placement algorithm, no
re-ordering of the existing pipeline, no router-level change. ARC
shows up only as a small reshaping of the packer's input and three
extra terms in the gate's threshold.

## Why we dropped it

On the grader v2 cut total transit by 27 %, from about 533 k moves
down to 391 k — a clean win on the metric we thought mattered. But
the score dropped from 110.86 to 109.37.

The reason came out of the grader's own formula. The composite score
is roughly 60 seconds of compute per case times mean PAR, plus a
small term for the actual data movement. On a 25-case mix, compute
ends up being something like 98 % of the total time, so PAR matters
about fifteen times more than transit at the leaderboard band. v2's
gate was strict enough that it was occasionally refusing redeploys
that would have improved PAR, and PAR drifted from 1.722 to 1.779.
The transit savings just couldn't pay for that.

That result is what shaped the next several versions. v3 tried the
opposite extreme (no EWMA smoothing) and also lost, which confirmed
the diagnosis. v4 codified the lesson into a rule: keep only the
pieces of the ARC line of thinking that cannot hurt PAR even in
principle. Two pieces qualified — cycle detection, which refuses to
commit a proposal that would send a layer back to a placement it
just left, and a 2-opt refinement of the device-label anchor, which
is a permutation and so cannot change PAR at all. Both shipped, and
that version jumped +0.51 on the grader, the biggest single win of
the chain after the eventual v15.

v15 itself went back to ARC's underlying idea — adapt the policy to
evidence — but expressed it differently. Instead of stacking cooldown
and ghost-hit bumps on a static safety of 16, each layer's safety is
allowed to drift on its own, multiplicatively, based on how well its
own PAR predictions have been holding up. That's TCP-Reno's AIMD
shape applied to a placement gate. It feels related to ARC and we
cite ARC for it, but the mechanism is closed-loop control, not a
ghost-list bookkeeping.

v18 was a final attempt to have both — v15's adaptive base plus v2's
overlays, but with the overlay magnitudes scaled way down so they
just nudge the first call or two after a commit. Local sims didn't
beat v15 cleanly on the harder Mix/EP256 cases, so it never went to
the grader.

## Where to look

The v2 code reads in the same order as the sections above, with the
ARC-A / B / C labels in the comments:
[`trials/submissions/submission_v2.py`](../submissions/submission_v2.py).

The v18 hybrid is useful if you want to see how the same overlays
look on top of an AIMD base instead of a fixed one:
[`trials/submissions/submission_v18.py`](../submissions/submission_v18.py).

The paper-level citation for ARC, and the AIMD analogy used in v15,
are in [`CITATIONS.md`](../../CITATIONS.md) §9.
