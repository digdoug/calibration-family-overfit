"""Out-of-sample validation of the geometric mechanism (LOFO / LOO cross-validation).

The head-angle result (track2_headangle.py) reports an IN-SAMPLE Spearman rho
between cos(w_A,w_B) and off-diagonal transfer AUROC. This script asks the
strongest-reviewer question: is that relationship PREDICTIVE out-of-sample? It fits a
simple OLS line AUROC ~ a + b*cos on cells that EXCLUDE a held-out lineage, predicts
the held-out lineage's transfer cells, and measures pooled out-of-sample rank
correlation + an MAE skill score vs a predict-the-mean baseline. Cluster-bootstrapped
by test task. Pre-registered before running.

Reuses track2_headangle (hence track2_matrix_strict), so head directions, the AUROC
matrix, and the off-diagonal cell set are byte-identical to the headline strict matrix.

Run:
  .venv/bin/python scripts/track2_headangle_cv.py track2_n600
  .venv/bin/python scripts/track2_headangle_cv.py track2_n600_9fam
  .venv/bin/python scripts/track2_headangle_cv.py track2_n600 --basis row-benign   # action-only
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.stats import spearmanr

import track2_headangle as th
import track2_matrix_strict as tms

REPO = tms.REPO


def ols_fit(x: np.ndarray, y: np.ndarray):
    """OLS y ~ a + b*x -> (slope, intercept)."""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def lofo(names, cos_of, au_of):
    """Leave-one-family-out: predict cells touching F from a map fit without F.

    Returns pooled (pred, actual, base) arrays over all held-out cells.
    """
    preds, actuals, bases = [], [], []
    for f in names:
        train = [(a, b) for a in names for b in names
                 if a != b and a != f and b != f]
        test = [(a, b) for a in names for b in names
                if a != b and (a == f or b == f)]
        if len(train) < 3 or not test:
            continue
        xc = np.array([cos_of[c] for c in train])
        yc = np.array([au_of[c] for c in train])
        slope, intercept = ols_fit(xc, yc)
        base = float(yc.mean())
        for c in test:
            preds.append(slope * cos_of[c] + intercept)
            actuals.append(au_of[c])
            bases.append(base)
    return np.array(preds), np.array(actuals), np.array(bases)


def loo(names, cos_of, au_of):
    """Leave-one-cell-out: predict each off-diag cell from a map fit on all others."""
    cells = [(a, b) for a in names for b in names if a != b]
    preds, actuals, bases = [], [], []
    for held in cells:
        train = [c for c in cells if c != held]
        xc = np.array([cos_of[c] for c in train])
        yc = np.array([au_of[c] for c in train])
        slope, intercept = ols_fit(xc, yc)
        preds.append(slope * cos_of[held] + intercept)
        actuals.append(au_of[held])
        bases.append(float(yc.mean()))
    return np.array(preds), np.array(actuals), np.array(bases)


def metrics(pred, actual, base):
    """Pooled OOS Spearman + MAE skill (1 - MAE_geom/MAE_base)."""
    rho = spearmanr(pred, actual).statistic
    mae_geom = float(np.abs(pred - actual).mean())
    mae_base = float(np.abs(base - actual).mean())
    skill = 1 - mae_geom / mae_base if mae_base > 0 else float("nan")
    return rho, skill, mae_geom, mae_base


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
    print(f"head-angle CV: {t2_dir}")
    print(f"families ({len(names)}): {', '.join(tms.short_name(n) for n in names)}")
    print(f"basis: {args.basis} | pca_dim: {args.pca_dim}")

    row_models = {a: tms.fit_row_model(fams, a, args.basis, args.pca_dim) for a in names}
    w = {a: th.head_direction(row_models[a]) for a in names}

    test_idx = {b: np.flatnonzero(fams[b]["split"] == "test") for b in names}
    cos_of = {(a, b): th.cosine(w[a], w[b]) for a in names for b in names if a != b}

    def au_map(idx_by_b):
        return {(a, b): th.cell_auroc(fams, b, row_models[a], idx_by_b[b])
                for a in names for b in names if a != b}

    au_of = au_map(test_idx)
    n_off = len(cos_of)

    # in-sample rho for reference (matches track2_headangle point estimate)
    insamp_rho = spearmanr(np.array([cos_of[c] for c in cos_of]),
                           np.array([au_of[c] for c in cos_of])).statistic

    lofo_pred, lofo_act, lofo_base = lofo(names, cos_of, au_of)
    loo_pred, loo_act, loo_base = loo(names, cos_of, au_of)
    lofo_rho, lofo_skill, lofo_mg, lofo_mb = metrics(lofo_pred, lofo_act, lofo_base)
    loo_rho, loo_skill, loo_mg, loo_mb = metrics(loo_pred, loo_act, loo_base)

    print(f"\noff-diagonal cells: {n_off} | in-sample rho(cos,AUROC) = {insamp_rho:+.3f}")
    print(f"LOFO (leave-one-FAMILY-out, primary):  held-out cells={len(lofo_act)}")
    print(f"   pooled OOS Spearman rho = {lofo_rho:+.3f}")
    print(f"   MAE skill = {lofo_skill:+.3f}  (geom {lofo_mg:.4f} vs mean-baseline {lofo_mb:.4f})")
    print(f"LOO  (leave-one-CELL-out, secondary):  held-out cells={len(loo_act)}")
    print(f"   pooled OOS Spearman rho = {loo_rho:+.3f}")
    print(f"   MAE skill = {loo_skill:+.3f}  (geom {loo_mg:.4f} vs mean-baseline {loo_mb:.4f})")

    # cluster bootstrap by test task: resample tasks, recompute AUROC side, redo CV.
    rng = np.random.default_rng(args.seed)
    all_test_tasks = sorted(
        {str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"}
    )
    lofo_rhos, lofo_skills, loo_rhos, loo_skills = [], [], [], []
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
        au_b = au_map(idx_b)
        lp, la, lb = lofo(names, cos_of, au_b)
        op, oa, ob = loo(names, cos_of, au_b)
        lr, ls, _, _ = metrics(lp, la, lb)
        orr, os_, _, _ = metrics(op, oa, ob)
        if not np.isnan(lr):
            lofo_rhos.append(lr); lofo_skills.append(ls)
        if not np.isnan(orr):
            loo_rhos.append(orr); loo_skills.append(os_)

    def ci(arr):
        a = np.array(arr)
        return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)

    lr_m, lr_lo, lr_hi = ci(lofo_rhos)
    ls_m, ls_lo, ls_hi = ci(lofo_skills)
    or_m, or_lo, or_hi = ci(loo_rhos)
    os_m, os_lo, os_hi = ci(loo_skills)
    print(f"\n=== cluster bootstrap by test task (LOFO n={len(lofo_rhos)}, LOO n={len(loo_rhos)}) ===")
    print(f"LOFO OOS Spearman: mean={lr_m:+.3f}  95% CI [{lr_lo:+.3f}, {lr_hi:+.3f}]")
    print(f"LOFO MAE skill:    mean={ls_m:+.3f}  95% CI [{ls_lo:+.3f}, {ls_hi:+.3f}]")
    print(f"LOO  OOS Spearman: mean={or_m:+.3f}  95% CI [{or_lo:+.3f}, {or_hi:+.3f}]")
    print(f"LOO  MAE skill:    mean={os_m:+.3f}  95% CI [{os_lo:+.3f}, {os_hi:+.3f}]")
    print("\nLIVE (predictive) if LOFO OOS Spearman CI clear of 0 & positive AND MAE skill > 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
