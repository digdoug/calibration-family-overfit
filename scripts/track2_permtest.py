"""Control 17 — EXACT permutation test of the diagonal-matching interaction.

The headline interaction (track2_matrix.py) is reported with a cluster-bootstrap
CI: that quantifies how STABLE the estimate is under task resampling. It does NOT
test the estimate against a null of "no family-specific structure." A best-paper
reviewer's sharpest remaining statistical objection on the headline is exactly:
"could a diagonal bonus this large arise by chance if the calibration<->eval family
correspondence were arbitrary, given the same row/col main effects?"

This module answers that with the textbook non-parametric interaction test.

Statistic: build the N x N Track-2 AUROC matrix with the SAME head-fitting as
track2_matrix.py (pooled-cal PCA-50, per-calibration-family logistic head), so the
cells match the reported numbers exactly. Double-center it:
    R[i,j] = AUROC[i,j] - row_i - col_j + grand
The interaction is mean_i R[i,i] (mean of the diagonal of the double-centered
matrix) -- identical to track2_matrix.diag_interaction.

Null: the diagonal arrangement is arbitrary. Under H0 the observed diagonal is
exchangeable with any other system of distinct representatives (one cell per row,
one per column). We enumerate ALL N! column permutations pi and compute
    stat(pi) = mean_i R[i, pi(i)] .
Because R is FIXED (double-centering is a property of the matrix, invariant to which
diagonal we then select), this holds ALL main effects exactly constant -- it isolates
the interaction. The observed diagonal is one of the N! permutations.

    exact p = ( #{pi : stat(pi) >= observed} ) / N!

Exact enumeration is tractable for N<=8 (8! = 40320). For N=4 the p-floor is 1/24 =
0.0417 (rank-1 still clears 0.05); for N=7 it is 1/5040 ~ 2e-4; at the 8-family fold
1/40320 ~ 2.5e-5. The test therefore SHARPENS as families are added -- the opposite
of an underpowered headline.

This is complementary to, not a replacement for, the bootstrap CI: bootstrap = "the
estimate is stable"; permutation = "the estimate is significant against the
no-interaction null." Reviewers want both. $0, local, reuses existing npz.

Run:  .venv-controlarena/bin/python scripts/track2_permtest.py [npz_dir]
      (default track2_n600 = the 4-family open registered headline)
"""
from __future__ import annotations
import glob, os, sys, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# reuse the EXACT split + interaction definition from the headline pipeline
from track2_matrix import split_of, diag_interaction  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_matrix(npz_dir):
    """Replicate track2_matrix head-fitting exactly; return (names, AUROC dict)."""
    fams = {}
    for f in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])

    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))

    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])

    M = {}
    for A in names:
        for B in names:
            d = fams[B]; te = d["split"] == "test"
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
    return names, M


def exact_perm_test(names, M):
    """Exact permutation test on the double-centered matrix."""
    n = len(names)
    arr = np.array([[M[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    R = arr - row[:, None] - col[None, :] + grand   # double-centered residual matrix
    observed = float(np.mean([R[i, i] for i in range(n)]))

    stats = []
    for perm in itertools.permutations(range(n)):
        stats.append(np.mean([R[i, perm[i]] for i in range(n)]))
    stats = np.array(stats)
    nperm = len(stats)                                # = n!
    ge = int(np.sum(stats >= observed - 1e-12))
    p = ge / nperm
    rank = int(np.sum(stats > observed + 1e-12)) + 1  # 1 = most extreme
    return observed, p, rank, nperm, stats


def main():
    npz_dir = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 else "track2_n600")
    label = os.path.basename(npz_dir)
    names, M = build_matrix(npz_dir)
    inter = diag_interaction(M, names)
    observed, p, rank, nperm, stats = exact_perm_test(names, M)

    print(f"\n===== Control 17: exact permutation test  [{label}] =====")
    print(f"families (N={len(names)}): {', '.join(n.replace('ow_','').replace('_n600','').replace('_n200','') for n in names)}")
    # sanity: the perm-test 'observed' must equal diag_interaction to the bit
    assert abs(observed - inter) < 1e-9, f"observed {observed} != diag_interaction {inter}"
    print(f"observed interaction        = {observed:+.4f}  (== track2_matrix headline)")
    print(f"# permutations (N!)         = {nperm}")
    print(f"observed rank               = {rank} / {nperm}   (1 = most extreme matching)")
    print(f"EXACT permutation p         = {p:.6f}   (p-floor = 1/N! = {1/nperm:.6f})")
    print(f"null mean / sd              = {stats.mean():+.4f} / {stats.std():.4f}")
    if p < 0.05:
        print("VERDICT: diagonal matching is significant net of main effects (p<0.05).")
    else:
        print("VERDICT: NOT significant -- the diagonal bonus is consistent with arbitrary matching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
