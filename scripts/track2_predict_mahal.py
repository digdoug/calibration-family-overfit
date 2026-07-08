"""Control 18b — does the SHIPPED Step-2 detector (Mahalanobis d_A) predict the drop?

Swaps the deployable off-lineage drift detector (squared Mahalanobis distance to
family A's cal-benign Gaussian, Ledoit-Wolf shrinkage -- the EXACT Control 19A /
track2_ood_gate statistic) into Control 18's transfer-risk-prediction harness,
holding the target Y(A,B)=AUROC[A,A]-AUROC[A,B] and the inference (Spearman +
exact Mantel + LOFO) byte-identical to track2_predict. The only thing that changes
vs Control 18 is the predictor, so this isolates "is Control 18's must-measure
verdict an artifact of a weak benign statistic, or does the detector we actually
ship also fail to predict the loss?"

Predictors (CAL-split benign only, disjoint from the TEST-split Y, same pooled
PCA-50 space as the headline matrix):
  mahal_auroc   : AUROC of d_A separating A-cal-benign vs B-cal-benign (the
                  cal-split analog of Control 19A's 0.836 detector number). PRIMARY.
  mahal_logratio: log(mean d_A over B-cal-benign / mean d_A over A-cal-benign),
                  the scale-comparable raw drift magnitude.

Run: .venv-controlarena/bin/python scripts/track2_predict_mahal.py [<dir>...]
$0, local CPU on cached embeddings. Pre-registered before running.
"""
from __future__ import annotations
import os, sys
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# reuse Control 18's harness verbatim so Y and the inference are byte-identical
from track2_predict import (REPO, load_dir, build_matrix, benign_cal,
                            mantel_p, lofo_generalize, OPEN4)


def fit_da(p_cal_benign):
    """Step-2 detector d_A: Gaussian on A's CAL-split benign, Ledoit-Wolf precision.
    Exactly track2_ood_gate.build's ood[A] + maha()."""
    lw = LedoitWolf().fit(p_cal_benign)
    return p_cal_benign.mean(0), lw.precision_


def maha(p, model):
    mu, prec = model
    z = p - mu
    return np.einsum("ij,jk,ik->i", z, prec, z)  # squared Mahalanobis


def analyze(t2, keep=None):
    fams = load_dir(t2, keep=keep)
    names, cells = build_matrix(fams)          # populates d["p"] (pooled PCA-50), exact headline cells
    n = len(names)
    benign = {nm: benign_cal(fams, nm) for nm in names}   # CAL-split benign in p-space
    da = {nm: fit_da(benign[nm]) for nm in names}         # one detector per calibration family

    Y = np.zeros((n, n))
    X = {"mahal_auroc": np.zeros((n, n)), "mahal_logratio": np.zeros((n, n))}
    base = {nm: float(np.mean(maha(benign[nm], da[nm]))) for nm in names}  # A's own benign d_A baseline
    for i, A in enumerate(names):
        for j, B in enumerate(names):
            Y[i, j] = cells[(A, A)] - cells[(A, B)]
            if i == j:
                continue
            dB = maha(benign[B], da[A])         # d_A evaluated on B's cal-benign
            dA = maha(benign[A], da[A])          # d_A on A's own cal-benign
            lab = np.r_[np.zeros(len(dA)), np.ones(len(dB))]
            sc = np.r_[dA, dB]
            X["mahal_auroc"][i, j] = roc_auc_score(lab, sc)
            X["mahal_logratio"][i, j] = float(np.log(np.mean(dB) / (base[A] + 1e-12) + 1e-12))

    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    ys = [Y[i, j] for (i, j) in off]
    tag = os.path.basename(t2.rstrip("/")) + (f"  [keep={sorted(keep)}]" if keep else "")
    print(f"\n=== {tag}  ({n} families, {len(off)} off-diag pairs) ===")
    print("families:", names)
    print(f"mean transfer drop Y = {np.mean(ys):+.3f}")
    print(f"validity: mahal_auroc off-diag mean = "
          f"{np.mean([X['mahal_auroc'][i,j] for (i,j) in off]):.3f}  "
          f"(cf Control 19A pop OOD-detect 0.836 full / 0.681 action)")
    for k in ("mahal_auroc", "mahal_logratio"):
        xs = [X[k][i, j] for (i, j) in off]
        rho = spearmanr(xs, ys).statistic
        p, tot = mantel_p(names, X[k], Y, rho)
        lofo_pool, lofo_within = lofo_generalize(names, X[k], Y)
        verdict = ("LOFO-NULL" if (lofo_pool <= 0.10 and p > 0.05) else "LOFO-SIGNAL?")
        print(f"  {k:14s}  X in[{min(xs):.3f},{max(xs):.3f}]  in-sample rho={rho:+.3f}"
              f"  Mantel p={p:.4f} ({tot} perms)   LOFO: pooled={lofo_pool:+.3f}"
              f" within-fam={lofo_within:+.3f}   [{verdict}]")
    return names, X, Y


def main():
    dirs = sys.argv[1:] or [
        os.path.join(REPO, "results", "track2_n600_8fam"),
        os.path.join(REPO, "results", "track2_n600_8fam_actiononly"),
        os.path.join(REPO, "results", "track2_n600_7fam"),
        os.path.join(REPO, "results", "track2_n600_7fam_actiononly"),
    ]
    for t2 in dirs:
        if not os.path.isdir(t2):
            print(f"(skip missing {t2})"); continue
        analyze(t2)                    # full roster
        analyze(t2, keep=OPEN4)        # 4-open primary subset


if __name__ == "__main__":
    main()
