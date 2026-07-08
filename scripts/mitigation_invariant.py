"""Constructive-mitigation attempt #2 (free reanalysis, 5-family n=600).

Routing/ensembling failed at 5-family (Control 9 redo). New method: FAMILY-INVARIANT POOLED head.
For a held-out family B (benign data available, attack labels not), (1) fit the family-ID subspace
on the pooled BENIGN cal embeddings, (2) project out its top-r directions, (3) train ONE attack head
on the POOLED other families' cal (honest+attack) in that residual space, (4) score B. The hope:
removing family identity lets a pooled head learn a family-agnostic attack direction that transfers.

Baselines to beat: single-foreign mean 0.588; naive pooling (r=0 here) ~0.45-0.59; in-distribution
ceiling (B's own head) mean diag 0.75. Run with .venv.
"""
import os, glob, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from track2_matrix import split_of  # noqa: E402
SHORT = lambda n: n.replace("ow_", "").replace("_n600", "")


def load():
    fams = {}
    for f in sorted(glob.glob(os.path.join(REPO, "results", "track2_n600_5fam", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        fams[SHORT(os.path.basename(f)[:-4])] = {
            "emb": d["emb"], "label": d["label"].astype(int),
            "split": np.array([split_of(str(t)) for t in d["task_id"]])}
    return fams, sorted(fams)


def fid_subspace(P, fam_idx, r):
    """Top-r family-identity directions from a multiclass logistic fit on benign embeddings."""
    if r == 0:
        return np.zeros((P.shape[1], 0))
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(P, fam_idx)
    U, S, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
    return Vt[:r].T  # (d, r) orthonormal-ish basis of the family-ID subspace


def project_out(P, V):
    return P - (P @ V) @ V.T if V.shape[1] else P


def main():
    fams, names = load()
    # shared scaler + PCA-50 on pooled cal (same as track2_matrix)
    cal_emb = np.vstack([fams[a]["emb"][fams[a]["split"] == "cal"] for a in names])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for a in names:
        fams[a]["p"] = pca.transform(scaler.transform(fams[a]["emb"]))

    # in-distribution ceiling (own head) for reference
    ceil = {}
    for B in names:
        d = fams[B]; cal = d["split"] == "cal"; te = d["split"] == "test"
        h = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
        ceil[B] = roc_auc_score(d["label"][te], h.decision_function(d["p"][te]))

    print(f"{'r':>3} | " + " | ".join(f"{SHORT(b):>7}" for b in names) + " |   MEAN  (vs foreign1 0.588, ceil %.3f)" % np.mean(list(ceil.values())))
    for r in (0, 1, 2, 3, 5):
        per = {}
        for B in names:
            others = [a for a in names if a != B]
            # family-ID subspace fit on benign cal of ALL families (B's benign is available)
            ben = [(fams[a]["p"][(fams[a]["split"] == "cal") & (fams[a]["label"] == 0)], i)
                   for i, a in enumerate(names)]
            Pb = np.vstack([p for p, _ in ben])
            yb = np.concatenate([[i] * len(p) for p, i in ben])
            V = fid_subspace(Pb, yb, r)
            # pooled FOREIGN cal (honest+attack) in residual space -> one invariant head
            Xtr = np.vstack([project_out(fams[a]["p"][fams[a]["split"] == "cal"], V) for a in others])
            ytr = np.concatenate([fams[a]["label"][fams[a]["split"] == "cal"] for a in others])
            head = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr)
            te = fams[B]["split"] == "test"
            XB = project_out(fams[B]["p"][te], V)
            per[B] = roc_auc_score(fams[B]["label"][te], head.decision_function(XB))
        mean = np.mean(list(per.values()))
        tag = "  <- naive pooling" if r == 0 else ""
        print(f"{r:>3} | " + " | ".join(f"{per[b]:7.3f}" for b in names) + f" |  {mean:.3f}{tag}")


if __name__ == "__main__":
    main()
