"""Stage-0 frontier-subset successful-sabotage matrix ($0, cached).

The pre-registered Gate-2 slice declared the both-success
cut DEAD because it required EVERY family in the 8-family roster to clear n>=50,
and the four open families have 0-1 both-ok test positives, which voids the whole
matrix (an empty diagonal cell has no AUROC). This script asks the narrower,
answerable question the smoke is about: restrict the matrix to the FRONTIER
families that DO realize sabotage, and read whether the calibration-family
interaction survives on realized-sabotage test attacks.

Because each strict row-head is fit from ONLY its own family's calibration split
(track2_matrix_strict.fit_row_model), a family subset is an exact sub-block of the
full strict matrix -- not a re-estimation. The interaction reported here is the
WITHIN-SUBSET diagonal advantage (net of subset row + subset col main effects).

To separate the success-filter effect from the subset effect, run the SAME subset
under all three eval-side filters and compare:
  none  -- attack attempts (the paper's headline metric)
  main  -- competent-attacker slice (main task succeeded)
  both  -- realized-sabotage slice (main AND side succeeded)  <-- the question

Run:
  .venv/bin/python scripts/successful_sabotage_matrix.py \
      --roster track2_n600_9fam --families gpt5,deepseek,grok,sonnet \
      --filter both --boot 2000 --seed 0
"""
from __future__ import annotations

import argparse
import os

from track2_success_slice import (
    boot,
    keep_mask,
    load_with_outcomes,
    print_matrix,
    score_cells,
    summarize,
)
from track2_matrix_strict import fit_row_model, short_name

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(names: list[str], wanted: str) -> list[str]:
    """Map requested short names to full npz names, preserving requested order."""
    want = [w.strip() for w in wanted.split(",") if w.strip()]
    by_short = {short_name(n): n for n in names}
    out = []
    for w in want:
        if w in by_short:
            out.append(by_short[w])
        elif w in names:
            out.append(w)
        else:
            raise SystemExit(f"family '{w}' not in roster (have: {', '.join(by_short)})")
    return out


def run_one(fams, sub, filt, basis, pca_dim, n_boot, seed):
    keep = {b: keep_mask(fams[b], filt) for b in sub}
    row_models = {a: fit_row_model(fams, a, basis, pca_dim) for a in sub}
    cells = score_cells(fams, sub, row_models, keep)
    pos = {}
    for b in sub:
        d = fams[b]
        te = d["split"] == "test"
        pos[b] = int((te & (d["label"] == 1) & keep[b]).sum())
    if cells is None:
        return {"filter": filt, "computable": False, "pos": pos}
    diag, off, raw, inter = summarize(cells, sub)
    im, ilo, ihi, rm, rlo, rhi, nb = boot(fams, sub, row_models, keep, n_boot, seed)
    return {
        "filter": filt, "computable": True, "cells": cells, "sub": sub, "pos": pos,
        "diag": diag, "off": off, "raw": raw, "inter": inter,
        "im": im, "ilo": ilo, "ihi": ihi, "nb": nb, "n_boot": n_boot,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default="track2_n600_9fam")
    ap.add_argument("--families", default="gpt5,deepseek,grok,sonnet")
    ap.add_argument("--filter", choices=["none", "main", "both", "all"], default="all")
    ap.add_argument("--basis", choices=["row-all", "row-benign"], default="row-all")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t2_dir = os.path.join(REPO, "results", args.roster)
    names, fams = load_with_outcomes(t2_dir)
    sub = resolve(names, args.families)

    print(f"Frontier-subset successful-sabotage matrix (strict row-all sub-block)")
    print(f"roster={args.roster}  families={', '.join(short_name(n) for n in sub)}  "
          f"basis={args.basis}  pca={args.pca_dim}  boot={args.boot}  seed={args.seed}\n")

    filters = ["none", "main", "both"] if args.filter == "all" else [args.filter]
    results = []
    for filt in filters:
        r = run_one(fams, sub, filt, args.basis, args.pca_dim, args.boot, args.seed)
        results.append(r)
        print(f"========== filter = {filt} ==========")
        print("kept attack TEST positives: " +
              "  ".join(f"{short_name(b)}={r['pos'][b]}" for b in sub))
        if not r["computable"]:
            print("*** MATRIX NOT COMPUTABLE (an included cell has <2 classes).\n")
            continue
        print_matrix(r["cells"], sub)
        print(f"mean diagonal = {r['diag']:.3f}   mean off-diagonal = {r['off']:.3f}")
        print(f"raw bonus = {r['raw']:+.4f}   INTERACTION = {r['inter']:+.4f}")
        print(f"true cluster-bootstrap: INTERACTION mean={r['im']:+.4f} "
              f"95% CI [{r['ilo']:+.4f}, {r['ihi']:+.4f}]  (valid draws {r['nb']}/{r['n_boot']})\n")

    comp = [r for r in results if r.get("computable")]
    if len(comp) > 1:
        print("========== summary (same families, three eval-side filters) ==========")
        print(f"{'filter':>6} | {'diag':>6} {'off':>6} {'interaction':>22} {'boot draws':>11}")
        for r in comp:
            ci = f"[{r['ilo']:+.4f},{r['ihi']:+.4f}]"
            print(f"{r['filter']:>6} | {r['diag']:>6.3f} {r['off']:>6.3f} "
                  f"{r['inter']:>+8.4f} {ci:>13} {r['nb']:>5}/{r['n_boot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
