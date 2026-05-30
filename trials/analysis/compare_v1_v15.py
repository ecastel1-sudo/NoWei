#!/usr/bin/env python3
"""Per-case grader-feedback diff between the very first submission ('v1' =
'+ EWMA + two-level anchor') and the latest one (v15).

Reuses dashboard.py's ingestion + plotting, but slices the timeline down
to [first, last] so the existing latest-vs-previous helper produces a
v1-vs-v15 diff instead of a v12-vs-v15 diff. Result is written to a
distinct file so the canonical Grader_Latest_vs_Previous.png stays
intact.
"""
from __future__ import annotations

from pathlib import Path

from dashboard import (
  _save_grader_latest_vs_previous,
  discover_grader_versions,
)


# Canonical key (after _canonicalize_grader_suffix) of the very first
# submission. Centralised so we don't sprinkle the literal everywhere.
V1_LABEL = "+ EWMA + two-level anchor"
V_LATEST_LABEL = "v15"
OUT_NAME = "Grader_v1_vs_v15.png"


def _find(versions, label: str):
  for v in versions:
    if v.label == label:
      return v
  raise SystemExit(
      f"Could not find grader version with label={label!r}. "
      f"Available: {[v.label for v in versions]}"
  )


def main() -> None:
  repo_root = Path(__file__).resolve().parent
  versions = discover_grader_versions(repo_root)
  if len(versions) < 2:
    raise SystemExit("Need at least 2 grader versions in feedbacks/")

  v1 = _find(versions, V1_LABEL)
  v_latest = _find(versions, V_LATEST_LABEL)

  # The plotter treats versions[-1] as 'latest' and versions[-2] as
  # 'previous'; pass them in that order.
  out_path = repo_root / "output" / "figure" / OUT_NAME
  out_path.parent.mkdir(parents=True, exist_ok=True)
  _save_grader_latest_vs_previous([v1, v_latest], out_path)
  print(f"  -> wrote {out_path}")
  print(f"     {V1_LABEL!r} (score={v1.composite_score:.3f}) "
        f"vs {V_LATEST_LABEL!r} (score={v_latest.composite_score:.3f}), "
        f"Δ={v_latest.composite_score - v1.composite_score:+.3f}")


if __name__ == "__main__":
  main()
