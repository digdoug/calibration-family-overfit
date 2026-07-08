"""Pseudo-family placebo — the pipeline negative control.

Control 17 (track2_permtest.py) tests the headline interaction for significance
GIVEN the AUROC matrix. It cannot test whether the head-fitting PIPELINE itself
manufactures a diagonal bonus when a head is calibrated on its own subsample and
evaluated on its own held-out tasks, with no true family difference. This module
runs that negative control.

For each real family F we randomly partition F's TASK IDs into two pseudo-families
F_a, F_b (same lineage, same model, same style; the only thing separating them is
a random task split). We then run the IDENTICAL strict pipeline used for the
headline -- imported from track2_matrix_strict so the head-fitting is byte-faithful
-- and compute the 2x2 double-centered interaction. Both axes are the same lineage,
so the interaction should be ~0. We repeat over K random partitions to trace the
null distribution.

Pre-registered (committed before this ran).

Run:
  .venv-controlarena/bin/python scripts/track2_pseudofamily_placebo.py track2_n600 --basis row-all
  .venv-controlarena/bin/python scripts/track2_pseudofamily_placebo.py track2_n600_actiononly --basis row-benign
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

# Byte-identical head-fitting / scoring / statistic as the headline strict matrix.
from track2_matrix_strict import (
    diag_interaction,
    fit_row_model,
    load_families,
    score_cells,
    short_name,
    split_of,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pseudo_half(task: str, part_seed: int) -> str:
    """Assign a task to pseudo-family 'a' or 'b' by a hash independent of the
    cal/test split (which uses split_of(task, seed=0))."""
    h = int(hashlib.md5(f"pf{part_seed}:{task}".encode()).hexdigest(), 16)
    return "a" if (h % 2 == 0) else "b"


def make_pseudo_fams(
    d: dict[str, np.ndarray], part_seed: int
) -> dict[str, dict[str, np.ndarray]]:
    """Split one real family's record dict into two pseudo-family record dicts,
    preserving the exact emb/label/task/split schema fit_row_model expects."""
    halves = np.array([pseudo_half(str(t), part_seed) for t in d["task"]], dtype=object)
    out: dict[str, dict[str, np.ndarray]] = {}
    for h in ("a", "b"):
        m = halves == h
        out[f"pf_{h}"] = {
            "emb": d["emb"][m],
            "label": d["label"][m],
            "task": d["task"][m],
            "split": d["split"][m],
        }
    return out


def placebo_interaction(
    d: dict[str, np.ndarray], part_seed: int, basis: str, pca_dim: int
) -> tuple[float, float, float] | None:
    """One placebo 2x2: returns (interaction, mean_diag, mean_off) or None if a
    pseudo-family cal/test split degenerates to one class (counted, not hidden)."""
    sub = make_pseudo_fams(d, part_seed)
    names = ["pf_a", "pf_b"]
    try:
        row_models = {a: fit_row_model(sub, a, basis, pca_dim) for a in names}
    except ValueError:
        return None
    cells = score_cells(sub, names, row_models)
    if cells is None:
        return None
    diag = float(np.mean([cells[(x, x)] for x in names]))
    off = float(np.mean([cells[(a, b)] for a in names for b in names if a != b]))
    return diag_interaction(cells, names), diag, off


def band(x: np.ndarray) -> tuple[float, float, float]:
    return float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("t2", nargs="?", default="track2_n600")
    ap.add_argument("--basis", choices=["row-all", "row-benign"], default="row-all")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--k", type=int, default=300, help="random task partitions per family")
    ap.add_argument("--real", type=float, default=0.1717,
                    help="real strict headline interaction for the one-sided tail")
    args = ap.parse_args()

    t2_dir = os.path.join(REPO, "results", args.t2)
    names, fams = load_families(t2_dir)
    print(f"pseudo-family placebo: {t2_dir}")
    print(f"families: {', '.join(short_name(n) for n in names)}")
    print(f"basis: {args.basis} | pca_dim: {args.pca_dim} | partitions/family: {args.k}")
    print(f"real strict headline interaction (tail reference): {args.real:+.4f}\n")

    pooled_int: list[float] = []
    pooled_diag: list[float] = []
    pooled_off: list[float] = []
    skipped_total = 0
    print(f"{'family':>12}  {'n_ok':>5}  {'skip':>4}  {'mean_int':>9}  {'2.5%':>8}  {'97.5%':>8}  "
          f"{'diag':>6}  {'off':>6}")
    for fam in names:
        d = fams[fam]
        ints: list[float] = []
        diags: list[float] = []
        offs: list[float] = []
        skipped = 0
        for s in range(args.k):
            r = placebo_interaction(d, s, args.basis, args.pca_dim)
            if r is None:
                skipped += 1
                continue
            ints.append(r[0]); diags.append(r[1]); offs.append(r[2])
        skipped_total += skipped
        ia = np.array(ints)
        m, lo, hi = band(ia)
        pooled_int += ints; pooled_diag += diags; pooled_off += offs
        print(f"{short_name(fam):>12}  {len(ints):>5}  {skipped:>4}  {m:>+9.4f}  {lo:>+8.4f}  "
              f"{hi:>+8.4f}  {np.mean(diags):>6.3f}  {np.mean(offs):>6.3f}")

    pa = np.array(pooled_int)
    m, lo, hi = band(pa)
    tail = float(np.mean(pa >= args.real))
    print("\n===== POOLED placebo null distribution =====")
    print(f"n partitions (ok)        = {len(pa)}  (skipped one-class: {skipped_total})")
    print(f"mean placebo diagonal    = {np.mean(pooled_diag):.3f}")
    print(f"mean placebo off-diag    = {np.mean(pooled_off):.3f}")
    print(f"placebo INTERACTION mean = {m:+.4f}  95% band [{lo:+.4f}, {hi:+.4f}]")
    print(f"one-sided tail frac >= real {args.real:+.4f} = {tail:.4f}")

    print("\n----- live/die (pre-registered) -----")
    quarter = 0.043
    if abs(m) < quarter and lo <= 0.0 <= hi and tail < 0.01:
        verdict = "PASS (negative control holds: pipeline does not manufacture a diagonal bonus)"
    elif abs(m) < quarter and tail < 0.01:
        verdict = "PARTIAL (small nonzero null floor; report headline net of floor)"
    else:
        verdict = "FAIL (material same-source advantage; qualifies the headline)"
    print(f"  pooled mean={m:+.4f}  band=[{lo:+.4f},{hi:+.4f}]  tail={tail:.4f}  ->  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
