# STARE-LB variants — test bundle

Four submission variants to score against your real trace, plus a one-command
runner. Each `.py` is self-contained (only numpy + torch). Each `.zip` has it
renamed to `submission.py` at the root, ready to upload to Codabench as-is.

## The four variants

| File | Idea | What it optimises for |
|------|------|----------------------|
| `submission_v1_base.py` | EWMA + spike-clip + fixed PAR-gain gate | the shipped baseline; low transmit |
| `submission_v2_dual.py` | dual-EWMA drift detector (your MLFQ idea) | adapts faster where the workload is drifting |
| `submission_v3_reward.py` | gate = competition reward directly (your RL reward, greedy, no training) | provably score-aligned; `SAFETY` knob controls movement |
| `submission_v4_anchored.py` | v3 + anchored device relabel | same PAR as v3 with the fewest expert moves |

All share the DeepSeek replicate+pack core (inlined). They differ only in *what
load they feed it* (raw vs EWMA vs dual-EWMA) and *when they decide a move is
worth it* (fixed PAR floor vs reward arithmetic).

## How to test on your trace

Put `test_variants.py` and the four `submission_v*.py` in the simulator repo
root (next to `dynamic_lb_simulator.py`), then:

```bash
git lfs pull                       # get the real traces first
python test_variants.py --model Qwen3 --dataset LmSys --ep 32
python test_variants.py --model DS-R1 --dataset WildChat --ep 64 --collection-interval 128
```

It prints PAR / transmit / score for Default, DS-EPLB (official baseline = 100),
and each variant through the exact grader closed-loop. Higher score wins.

`--collection-interval` is the cadence for the variants (DS-EPLB uses the
official 1024). Sweep it (64 / 128 / 256) — cadence matters a lot.

## Tuning knobs (top of each file)

- `ALPHA` — EWMA memory. Smaller = longer memory = more spike resistance, slower
  drift tracking. Start 0.2; try 0.1–0.4.
- `PAR_GAIN_THRESHOLD` (v1) / `BASE_GAIN` (v2) — minimum PAR drop to bother
  moving a layer. Higher = less transmit.
- `SAFETY` (v3/v4) — how many times more expensive to treat each expert move
  than the first-order model says. Default 200 (movement-frugal, score-neutral
  on the literal model). Drop to 1.0 to chase raw score if the grader's transmit
  cost really is that cheap.
- `ALPHA_FAST` / `ALPHA_SLOW` / `DRIFT_SCALE` (v2) — drift detector sensitivity.

## What we learned on synthetic data (expect different magnitudes on real traces)

- The score model is heavily PAR-dominated: ~6,000 expert moves ≈ 0.01 PAR. So a
  pure reward gate (v3, SAFETY=1) churns a lot for almost no score change. Hence
  the high default SAFETY — it keeps score and slashes transmit, which protects
  you if the real grader penalises movement harder than the first-order model.
- v4-anchored consistently had the lowest transmit at equal PAR.
- v1/v2 are naturally transmit-frugal because of the explicit PAR-gain floor.

Pick the winner on YOUR trace — that's the whole point of shipping all four.
