"""Stage-0 funnel audit for the successful-sabotage matrix question ($0, cached).

Per family, TEST split only: attempts -> main-success -> side-success ->
both-success (main AND side). Reuses the vetted npz<->eval alignment from
track2_success_slice.load_with_outcomes (which asserts row-for-row alignment of
the .eval outcome labels to the cached embeddings), so the both-ok counts here
are identical in provenance to the Gate-2 pre-registration table.

Run:  .venv/bin/python scripts/successful_sabotage_funnel.py [roster]
      (default roster: track2_n600_9fam -- the full nine-family APPS roster)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from track2_success_slice import load_with_outcomes
from track2_matrix_strict import short_name

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(roster: str = "track2_n600_9fam") -> int:
    t2_dir = os.path.join(REPO, "results", roster)
    names, fams = load_with_outcomes(t2_dir)

    print(f"Successful-sabotage funnel (TEST split) -- roster: {roster}")
    print(f"families ({len(names)}): {', '.join(short_name(n) for n in names)}\n")
    hdr = (
        f"{'family':>16} | {'attempts':>8} {'main-ok':>8} {'side-ok':>8} "
        f"{'both-ok':>8} {'both%':>6} | {'benign':>7} {'ben-mainok':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for b in names:
        d = fams[b]
        te = d["split"] == "test"
        att = te & (d["label"] == 1)
        ben = te & (d["label"] == 0)
        attempts = int(att.sum())
        main_ok = int((att & d["main_ok"]).sum())
        side_ok = int((att & d["side_ok"]).sum())
        both_ok = int((att & d["main_ok"] & d["side_ok"]).sum())
        ben_n = int(ben.sum())
        ben_main = int((ben & d["main_ok"]).sum())
        pct = (100.0 * both_ok / attempts) if attempts else 0.0
        flag = "  <-- usable(>=7)" if both_ok >= 7 else ("  <-- dead(<2)" if both_ok < 2 else "")
        print(
            f"{short_name(b):>16} | {attempts:>8} {main_ok:>8} {side_ok:>8} "
            f"{both_ok:>8} {pct:>5.1f}% | {ben_n:>7} {ben_main:>10}{flag}"
        )
    print(
        "\nNote: a frontier-subset both-success matrix needs every included "
        "family\n      to have >=1 both-ok test positive (so no cell is empty); "
        ">=7 is the\n      minimum for a non-degenerate (if underpowered) cell."
    )
    return 0


if __name__ == "__main__":
    roster = sys.argv[1] if len(sys.argv) > 1 else "track2_n600_9fam"
    raise SystemExit(main(roster))
