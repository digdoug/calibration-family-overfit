"""Control 18 — a BENIGN-ONLY transfer-risk predictor.

Question: before deploying a family-A-calibrated monitor against family B, can a
deployer ESTIMATE the detection drop using only BENIGN transcripts from A and B
(no sabotage data from B)?

Target  Y(A,B) = AUROC[A,A] - AUROC[A,B]  (transfer drop; exact track2_matrix cells,
                 pooled-cal PCA-50 per-family logistic head, TEST-split AUROC).
Predictor X(A,B), on CAL-split benign (label==0) only -- disjoint task split from Y:
  distinguish  : 5-fold CV logistic 2-sample AUROC, A-benign vs B-benign (PCA-50).
  centroid_cos : cosine distance of benign means.
  mmd_rbf      : RBF-MMD^2 (median-heuristic bandwidth).
Spearman rho(X,Y) over off-diagonal ordered pairs; exact Mantel permutation p
(permute family identity jointly, all N! perms). LOFO held-out rank generalization.

Reuses track2_matrix's split_of + pooled PCA so Y == the headline matrix exactly.

Run: .venv-controlarena/bin/python scripts/track2_predict.py <dir> [<dir>...]
"""
from __future__ import annotations
import glob, os, sys, hashlib, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_FRAC = 60  # identical to track2_matrix


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def short(name):
    return (name.replace("frontier_", "").replace("ow_", "").replace("_n600", "")
            .replace("gemini_flash", "gemini"))


def load_dir(t2, keep=None):
    fams = {}
    for f in sorted(glob.glob(os.path.join(t2, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        name = short(os.path.basename(f)[:-4])
        if keep is not None and name not in keep:
            continue
        sp = np.array([split_of(t) for t in d["task_id"]])
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"], "split": sp}
    return fams


def build_matrix(fams):
    """Exact track2_matrix cells (pooled-cal PCA-50, per-family head, test AUROC)."""
    names = list(fams)
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = fams[A]; m = d["split"] == "cal"
        heads[A] = LogisticRegression(max_iter=2000, C=1.0).fit(d["p"][m], d["label"][m])
    cells = {}
    for A in names:
        for B in names:
            d = fams[B]; m = d["split"] == "test"
            cells[(A, B)] = roc_auc_score(d["label"][m], heads[A].decision_function(d["p"][m]))
    return names, cells


def benign_cal(fams, name):
    """CAL-split benign (label==0) embeddings in the pooled PCA space."""
    d = fams[name]; m = (d["split"] == "cal") & (d["label"] == 0)
    return d["p"][m]


def distinguish(Xa, Xb):
    """5-fold CV logistic 2-sample AUROC: how separable are A-benign vs B-benign."""
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    s = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cv,
                          method="decision_function")
    return roc_auc_score(y, s)


def centroid_cos(Xa, Xb):
    ma, mb = Xa.mean(0), Xb.mean(0)
    return 1.0 - float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))


def mmd_rbf(Xa, Xb):
    Z = np.vstack([Xa, Xb])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    med = np.median(d2[d2 > 0]); gamma = 1.0 / (med + 1e-12)
    K = np.exp(-gamma * d2)
    na = len(Xa)
    Kaa, Kbb, Kab = K[:na, :na], K[na:, na:], K[:na, na:]
    return float(Kaa.mean() + Kbb.mean() - 2 * Kab.mean())


def mantel_p(names, Xmat, Ymat, observed_rho):
    """Exact Mantel: permute family identity jointly, recompute off-diag Spearman."""
    n = len(names); idx = list(range(n))
    perms = list(itertools.permutations(idx))
    ge = 0; tot = 0
    offmask = [(i, j) for i in range(n) for j in range(n) if i != j]
    for perm in perms:
        xs = [Xmat[perm[i], perm[j]] for (i, j) in offmask]
        ys = [Ymat[i, j] for (i, j) in offmask]
        r = spearmanr(xs, ys).statistic
        if np.isnan(r):
            continue
        tot += 1
        if r >= observed_rho - 1e-12:
            ge += 1
    return ge / tot, tot


def lofo_generalize(names, Xmat, Ymat):
    """Leave-one-family-out, GENUINE out-of-sample: for each held-out family F,
    fit Y~X (linear) on pairs NOT involving F, predict Y for pairs involving F,
    pool the held-out predictions, report Spearman(pred, actual). The X->Y map is
    fit on disjoint families, so a NEW family whose pairs break the pattern lowers
    it -- this tests whether a deployer with only a new family's benign data can
    predict its transfer risk."""
    n = len(names)
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    pred, actual, within = [], [], []
    for f in range(n):
        tr = [(i, j) for (i, j) in off if i != f and j != f]
        te = [(i, j) for (i, j) in off if (i == f or j == f)]
        if len(tr) < 3:
            continue
        Xtr = np.array([[Xmat[i, j]] for (i, j) in tr]); Ytr = np.array([Ymat[i, j] for (i, j) in tr])
        reg = LinearRegression().fit(Xtr, Ytr)
        for (i, j) in te:
            pred.append(float(reg.predict([[Xmat[i, j]]])[0])); actual.append(Ymat[i, j])
        # interpretable variant: within held-out family, does X rank its drops?
        wr = spearmanr([Xmat[i, j] for (i, j) in te], [Ymat[i, j] for (i, j) in te]).statistic
        if not np.isnan(wr):
            within.append(wr)
    pooled = spearmanr(pred, actual).statistic
    return pooled, float(np.mean(within)) if within else float("nan")


def analyze(t2, keep=None):
    fams = load_dir(t2, keep=keep)
    names, cells = build_matrix(fams)
    n = len(names)
    Y = np.zeros((n, n)); X = {k: np.zeros((n, n)) for k in ("distinguish", "centroid_cos", "mmd_rbf")}
    benign = {nm: benign_cal(fams, nm) for nm in names}
    for i, A in enumerate(names):
        for j, B in enumerate(names):
            Y[i, j] = cells[(A, A)] - cells[(A, B)]
            if i == j:
                continue
            X["distinguish"][i, j] = distinguish(benign[A], benign[B])
            X["centroid_cos"][i, j] = centroid_cos(benign[A], benign[B])
            X["mmd_rbf"][i, j] = mmd_rbf(benign[A], benign[B])
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    ys = [Y[i, j] for (i, j) in off]
    print(f"\n=== {os.path.basename(t2.rstrip('/'))}  ({n} families, {len(off)} off-diag pairs) ===")
    print("families:", names)
    print(f"mean transfer drop Y = {np.mean(ys):+.3f}  (diag-minus-offdiag headline quantity)")
    for k in ("distinguish", "centroid_cos", "mmd_rbf"):
        xs = [X[k][i, j] for (i, j) in off]
        rho = spearmanr(xs, ys).statistic
        p, tot = mantel_p(names, X[k], Y, rho)
        lofo_pool, lofo_within = lofo_generalize(names, X[k], Y)
        print(f"  {k:13s}  X in[{min(xs):.3f},{max(xs):.3f}]  in-sample rho={rho:+.3f}"
              f"  Mantel p={p:.4f} ({tot} perms)   LOFO: pooled={lofo_pool:+.3f} within-fam={lofo_within:+.3f}")
    return names, X, Y


OPEN4 = {"gemma", "llama", "mistral", "qwen"}


def main():
    dirs = sys.argv[1:] or [
        os.path.join(REPO, "results", "track2_n600_7fam"),
        os.path.join(REPO, "results", "track2_n600_7fam_actiononly"),
    ]
    for t2 in dirs:
        analyze(t2)                       # full roster
        analyze(t2, keep=OPEN4)           # 4-family open headline subset


if __name__ == "__main__":
    main()
