# `trials/` — full design history

Everything we built that *didn't* end up as the deliverable. Kept so the
chain of decisions from the very first prototype to the shipped v15 is
fully auditable.

## Layout

```text
trials/
├── README.md            # this file
├── submissions/         # submission_v2 … v18 (the official chain)
├── feedbacks/           # grader output for every version (predictions + scores)
├── zips/                # every .zip we ever uploaded to the grader
├── analysis/            # comparison / sweep / ablation / dashboard scripts
├── first_versions/      # pre-versioning prototypes
│   ├── old/             #   the very first hand-rolled experiments
│   └── old-versions/    #   submission-V0 … V5 (numbered prototypes)
└── docs/                # design writeups + supporting analysis
```

## Where to look for what

| If you want… | …go to |
|---|---|
| The source of a specific trial version | `submissions/submission_vN.py` |
| The official grader output for that version | `feedbacks/scoring_result_vN/scores.txt` |
| Per-layer per-iter prediction logs for that version | `feedbacks/prediction_result_vN/` |
| A zip we actually uploaded | `zips/submission_vN.zip` |
| The exploded contents of the v18 upload | (was rebuilt from `zips/submission_v18.zip` — extract it again if needed) |
| A side-by-side comparison of two versions | `analysis/compare_<a>_<b>.py` |
| The original STARE-LB design writeup | `docs/STARE-LB_writeup.md` |
| The energy-composite scoring derivation | `docs/ENERGY_COMPOSITE_SCORE.pdf` |
| The original `USAGE.md` and `REPO_MAP.md` snapshots | `docs/` |

## Naming conventions

* `submissions/submission_vN.py` — Nth iteration in the official chain.
  Loaded by `eplb_algorithms/__init__.py` when the simulator is invoked
  with `--algorithm submission_vN`.
* `feedbacks/scoring_result_vN/` — what the grader returned: `scores.txt`
  (the headline numbers) and `scores.json` (the same in JSON).
* `feedbacks/prediction_result_vN/` — per-layer / per-iter prediction
  dumps used to debug AIMD residual behaviour.
* `feedbacks/*_+ EWMA + two-level anchor*` — early runs from before we
  switched to numeric versions; correspond to the v1-baseline shape.
* `zips/submission_vN.zip` — the exact archive we uploaded for version N.
  Filenames with extra suffixes (`_old`, `_best`, `_arc`, `_16gate`,
  `_v3`, `_86`) are alternative builds of the same version, kept so the
  upload trail matches Codabench history.

## How to score a trial version locally

```bash
python dynamic_lb_simulator.py --algorithm submission_v7
```

The adapter in `eplb_algorithms/__init__.py` auto-resolves
`submission_vN` to `trials/submissions/submission_vN.py` (root fallback
also works for ad-hoc copies).

## Pre-versioning prototypes

`first_versions/old/` and `first_versions/old-versions/` contain the
hand-rolled scripts from before we started numbering versions. They are
*not* directly comparable to the v2 – v18 chain (different API shapes,
different cadence assumptions) but they document the design steps that
led to v1.

| Folder | What's in it |
|---|---|
| `first_versions/old/` | Early simulator wrappers, run scripts, and the first stand-alone `submission.py`. |
| `first_versions/old-versions/` | Numbered prototypes `submission-V0.py` … `submission-5.py`, `submission_v2_dual.py`, `submission_v3_reward.py`, `submission_v4_anchored.py`. These are the conceptual ancestors of v1 – v4. |
