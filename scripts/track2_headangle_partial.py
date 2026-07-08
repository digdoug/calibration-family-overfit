"""Head-angle mechanism, NET of row+column main effects (partial correlation).

Extends scripts/track2_headangle.py. That result reports the RAW Spearman rho
between cross-family head cosine cos(w_A,w_B) and off-diagonal transfer AUROC(A->B).
A reviewer asks whether that raw correlation is just an obviousness/capability proxy
(some eval families are easy for everyone; some calibration heads are broadly sharp).
The paper's thesis is the INTERACTION: the diagonal advantage net of the per-row
(calibration family) and per-column (evaluation family) main effects. This script
tests the mechanism at the same altitude:

  resid_AUROC = AUROC(A,B) - (additive two-way fit over off-diagonal cells)
  resid_cos   = cos(w_A,w_B) - (same two-way fit)
  headline    = Spearman(resid_cos, resid_AUROC)            (full partial)
  secondary   = Spearman(cos,       resid_AUROC)            (interaction-tie)

If cosine still predicts transfer after both main effects are removed, head geometry
explains the interaction-level structure, not merely attacker obviousness.

Cluster bootstrap by test task (cos side is a fixed head-geometry constant; only the
AUROC side resamples). Reuses track2_headangle for w_A and per-cell AUROC so the
matrix is byte-identical to the headline strict matrix.

Run:
  .venv/bin/python scripts/track2_headangle_partial.py track2_n600
  .venv/bin/python scripts/track2_headangle_partial.py track2_n600_9fam
  .venv/bin/python scripts/track2_headangle_partial.py track2_n600_actiononly --basis row-benign
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.stats import spearmanr

import track2_matrix_strict as tms
import track2_headangle as tha

REPO = tms.REPO


def twoway_design(a_idx, b_idx, k):
    """Additive intercept + (k-1) row(A) dummies + (k-1) col(B) dummies, reference
    coding (drop level 0). Rows = off-diagonal cells in the given order."""
    n = len(a_idx)
    cols = [np.ones(n)]
    for lvl in range(1, k):
        cols.append((np.asarray(a_idx) == lvl).astype(float))
    for lvl in range(1, k):
        cols.append((np.asarray(b_idx) == lvl).astype(float))
    return np.vstack(cols).T  # (n, 1 + 2(k-1))


def residualize(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("t2", nargs="?", default="track2_n600")
    ap.add_argument("--basis", choices=["row-all", "row-benign"], default="row-all")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t2_dir = os.path.join(REPO, "results", args.t2)
    names, fams = tms.load_families(t2_dir)
    k = len(names)
    name_idx = {n: i for i, n in enumerate(names)}
    print(f"head-angle PARTIAL mechanism (net of row+col main effects): {t2_dir}")
    print(f"families ({k}): {', '.join(tms.short_name(n) for n in names)}")
    print(f"basis: {args.basis} | pca_dim: {args.pca_dim}")

    row_models = {a: tms.fit_row_model(fams, a, args.basis, args.pca_dim) for a in names}
    w = {a: tha.head_direction(row_models[a]) for a in names}

    test_idx = {b: np.flatnonzero(fams[b]["split"] == "test") for b in names}

    # off-diagonal cells in a fixed order
    cells = [(a, b) for a in names for b in names if a != b]
    a_idx = [name_idx[a] for a, b in cells]
    b_idx = [name_idx[b] for a, b in cells]
    cos_arr = np.array([tha.cosine(w[a], w[b]) for a, b in cells])

    X = twoway_design(a_idx, b_idx, k)
    n_off, n_par = X.shape
    print(f"off-diagonal cells: {n_off}  |  two-way model params: {n_par}  |  resid dof: {n_off - n_par}")
    resid_cos = residualize(cos_arr, X)  # fixed across bootstrap (cos does not depend on tasks)

    def stats_for(au_arr):
        resid_au = residualize(au_arr, X)
        raw = spearmanr(cos_arr, au_arr).statistic
        partial = spearmanr(resid_cos, resid_au).statistic            # full partial (headline)
        semi = spearmanr(cos_arr, resid_au).statistic                 # cos vs interaction residual
        return raw, partial, semi

    au_point = np.array([tha.cell_auroc(fams, b, row_models[a], test_idx[b]) for a, b in cells])
    raw_pt, partial_pt, semi_pt = stats_for(au_point)

    print("\n===== point estimates (full test split) =====")
    print(f"raw     Spearman rho(cos, AUROC)          = {raw_pt:+.3f}")
    print(f"PARTIAL Spearman rho(resid_cos, resid_AU) = {partial_pt:+.3f}   <- HEADLINE (net of row+col)")
    print(f"semi    Spearman rho(cos, resid_AUROC)    = {semi_pt:+.3f}   (cos vs interaction residual)")

    # cluster bootstrap by test task: resample tasks, recompute AUROC side + both rho.
    rng = np.random.default_rng(args.seed)
    all_test_tasks = sorted(
        {str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"}
    )
    raws, partials, semis = [], [], []
    for _ in range(args.boot):
        draw = rng.choice(all_test_tasks, size=len(all_test_tasks), replace=True).tolist()
        idx_b, ok = {}, True
        for b in names:
            d = fams[b]
            chunks = [np.flatnonzero((d["split"] == "test") & (d["task"] == t)) for t in draw]
            ix = np.concatenate([c for c in chunks if len(c)]) if chunks else np.array([], dtype=int)
            if len(ix) == 0 or len(np.unique(d["label"][ix])) < 2:
                ok = False
                break
            idx_b[b] = ix
        if not ok:
            continue
        au_b = np.array([tha.cell_auroc(fams, b, row_models[a], idx_b[b]) for a, b in cells])
        r, p, s = stats_for(au_b)
        if not (np.isnan(r) or np.isnan(p) or np.isnan(s)):
            raws.append(r); partials.append(p); semis.append(s)

    def ci(arr):
        arr = np.array(arr)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return arr.mean(), lo, hi, len(arr)

    print(f"\ncluster-bootstrap (n={len(partials)} draws):")
    for lab, arr in [("raw    rho(cos,AUROC)     ", raws),
                     ("PARTIAL rho(rc,rAU) HEAD  ", partials),
                     ("semi   rho(cos,rAU)       ", semis)]:
        m, lo, hi, _ = ci(arr)
        flag = "LIVE" if lo > 0 else ("DIE " if hi < 0 else "span")
        print(f"  {lab}: mean={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  [{flag}]")

    print("\nLIVE if PARTIAL CI clear of 0 and positive: head geometry explains the")
    print("interaction-level transfer structure net of the same main effects the headline removes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
