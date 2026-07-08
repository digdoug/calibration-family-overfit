"""Geometric mechanism of the cross-family transfer gap (head-direction misalignment).

Reuses track2_matrix_strict's exact strict (non-transductive) per-row preprocessing
and logistic head, so the AUROC matrix here is identical to the headline strict
matrix. For each calibration family A it reconstructs the sabotage-discriminative
direction in RAW embedding space:

    w_A = (pca_A.components_.T @ head_A.coef_[0]) / scaler_A.scale_     (elementwise)
    w_A = head_A.coef_[0] / scaler_A.scale_                            (no PCA)

i.e. the gradient of A's strict decision_function w.r.t. the raw embedding. It then
tests whether the OFF-DIAGONAL transfer is governed by the angle between families'
directions: cos(w_A, w_B) should predict AUROC(A scores B), and (1 - cos) the drop.

This is a post-hoc mechanistic explanation of the "concept shift" claim, NOT a
deployable benign-only predictor (cos needs labeled sabotage from BOTH families).

Run:
  .venv/bin/python scripts/track2_headangle.py track2_n600
  .venv/bin/python scripts/track2_headangle.py track2_n600_9fam --basis row-benign
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import track2_matrix_strict as tms

REPO = tms.REPO


def head_direction(model: tms.RowModel) -> np.ndarray:
    """Direction of A's decision_function in raw embedding space (unnormalized)."""
    coef = model.head.coef_[0]
    if model.pca is not None:
        w_scaled = model.pca.components_.T @ coef  # back out of PCA -> scaled-emb space
    else:
        w_scaled = coef
    return w_scaled / model.scaler.scale_  # undo standardization -> raw-emb space


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def cell_auroc(fams, b, model, idx) -> float:
    d = fams[b]
    scores = tms.transform_score(model, d["emb"][idx])
    return roc_auc_score(d["label"][idx], scores)


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
    print(f"head-angle mechanism: {t2_dir}")
    print(f"families: {', '.join(tms.short_name(n) for n in names)}")
    print(f"basis: {args.basis} | pca_dim: {args.pca_dim}")

    row_models = {a: tms.fit_row_model(fams, a, args.basis, args.pca_dim) for a in names}
    w = {a: head_direction(row_models[a]) for a in names}

    # full-test index per family (matches headline strict matrix point estimate)
    test_idx = {b: np.flatnonzero(fams[b]["split"] == "test") for b in names}
    auroc = {(a, b): cell_auroc(fams, b, row_models[a], test_idx[b]) for a in names for b in names}
    diag = {b: auroc[(b, b)] for b in names}

    rows = []  # (A, B, cos, auroc_AB, drop)
    for a in names:
        for b in names:
            if a == b:
                continue
            c = cosine(w[a], w[b])
            rows.append((a, b, c, auroc[(a, b)], diag[b] - auroc[(a, b)]))

    cos_arr = np.array([r[2] for r in rows])
    au_arr = np.array([r[3] for r in rows])
    drop_arr = np.array([r[4] for r in rows])

    print("\n===== off-diagonal cells: cos(w_A,w_B) vs transfer =====")
    print(f"{'A->B':>22}  {'cos':>7}  {'AUROC':>7}  {'drop':>7}")
    for a, b, c, au, dr in sorted(rows, key=lambda r: r[2]):
        print(f"{tms.short_name(a)+'->'+tms.short_name(b):>22}  {c:>7.3f}  {au:>7.3f}  {dr:>7.3f}")

    rho_au = spearmanr(cos_arr, au_arr).statistic
    rho_drop = spearmanr(1 - cos_arr, drop_arr).statistic
    # OLS drop ~ (1 - cos)
    x = 1 - cos_arr
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, drop_arr, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = float(((drop_arr - pred) ** 2).sum())
    ss_tot = float(((drop_arr - drop_arr.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nmean off-diagonal cos      = {cos_arr.mean():+.3f}")
    print(f"Spearman rho(cos, AUROC)   = {rho_au:+.3f}   (expect > 0)")
    print(f"Spearman rho(1-cos, drop)  = {rho_drop:+.3f}   (expect > 0; equals above by monotonicity)")
    print(f"OLS drop ~ (1-cos): slope  = {slope:+.3f}  intercept = {intercept:+.3f}  R^2 = {r2:.3f}")

    # cluster bootstrap by test task: resample tasks, recompute AUROC side + rho.
    # w (head geometry) is a fixed constant; only the AUROC/drop side is resampled.
    rng = np.random.default_rng(args.seed)
    all_test_tasks = sorted(
        {str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"}
    )
    rhos = []
    for _ in range(args.boot):
        draw = rng.choice(all_test_tasks, size=len(all_test_tasks), replace=True).tolist()
        idx_b = {}
        ok = True
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
        au_b = {(a, bb): cell_auroc(fams, bb, row_models[a], idx_b[bb]) for a in names for bb in names}
        cvals, avals = [], []
        for a in names:
            for bb in names:
                if a == bb:
                    continue
                cvals.append(cosine(w[a], w[bb]))
                avals.append(au_b[(a, bb)])
        rr = spearmanr(np.array(cvals), np.array(avals)).statistic
        if not np.isnan(rr):
            rhos.append(rr)
    rhos = np.array(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    print(f"\ncluster-bootstrap rho(cos,AUROC) (n={len(rhos)}): mean={rhos.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("\nLIVE if CI clear of 0 and positive: concept shift has a measured geometric form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
