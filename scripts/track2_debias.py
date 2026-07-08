"""Control 19 — family-subspace debiasing: mechanism + roster-independent defense.

Identify the family-discriminative subspace S from BENIGN (honest) cal-split
embeddings only (non-circular w.r.t. the sabotage label), project S OUT of every
embedding, and re-run the exact Track-2 matrix pipeline. Compare:

  before                : the published interaction
  family-deflate        : remove the rank-(k-1) between-family mean subspace
  random-deflate (R=20) : remove a random orthonormal subspace of the SAME rank

The random placebo is essential: projection removes capacity, which mechanically
shrinks any AUROC gap, so a family-specific mechanism requires the family deflation to
shrink the interaction MORE than equal-rank random deflation. Pre-registered
before running, with live/die criteria.

Run: .venv/bin/python scripts/track2_debias.py [track2_n600]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track2_matrix import split_of, diag_interaction, cluster_boot  # reuse exact infra

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_RANDOM = 20


def load(dirname):
    T2 = os.path.join(REPO, "results", dirname)
    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])
    return fams, list(fams)


def family_subspace(fams, names):
    """Orthonormal basis (1536 x k-1) of the between-family mean subspace, learned
    from cal-split HONEST embeddings only (label==0). This is the rank-(k-1) linear
    subspace that encodes family identity (textbook between-class scatter)."""
    means = []
    for A in names:
        d = fams[A]
        m = (d["split"] == "cal") & (d["label"] == 0)
        means.append(d["emb"][m].mean(axis=0))
    M = np.array(means)            # k x 1536
    C = M - M.mean(axis=0)         # center -> rank k-1
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[: len(names) - 1].T  # 1536 x (k-1), orthonormal columns


def family_inlp_basis(fams, names, rank):
    """Iterative nullspace-projection family-erasure basis (dim x rank), learned from
    cal-honest embeddings only. Refits a family-ID logistic on the deflated data each
    round so it keeps finding residual (covariance-carried) family signal -> a genuine
    high-rank linear family-erasure, not just the k-1 mean subspace. Answers the
    'you only removed 3 dims' objection."""
    X, y = [], []
    for i, A in enumerate(names):
        d = fams[A]
        m = (d["split"] == "cal") & (d["label"] == 0)
        X.append(d["emb"][m]); y += [i] * int(m.sum())
    X = np.vstack(X); y = np.array(y)
    basis = []
    while len(basis) < rank:
        B = np.array(basis).T if basis else None
        Xc = X - (X @ B) @ B.T if B is not None else X
        clf = LogisticRegression(C=1.0, max_iter=1500).fit(Xc, y)
        for w in clf.coef_:                       # n_fam rows (or 1 if binary)
            v = w.copy()
            for b in basis:
                v = v - (v @ b) * b
            n = np.linalg.norm(v)
            if n > 1e-8:
                basis.append(v / n)
                if len(basis) >= rank:
                    break
    return np.array(basis[:rank]).T


def random_subspace(dim, r, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, r)))
    return Q[:, :r]


def deflate(emb, B):
    """Project OUT the column space of orthonormal B: x - (x@B)@B.T."""
    if B is None:
        return emb
    return emb - (emb @ B) @ B.T


def run_pipeline(fams, names, B, do_boot=True):
    """The identical track2_matrix pipeline on deflated embeddings.
    Returns (diag, off, interaction, boot tuple or None)."""
    work = {}
    for A, d in fams.items():
        work[A] = {"label": d["label"], "task": d["task"], "split": d["split"],
                   "emb": deflate(d["emb"], B)}
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in work.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in work.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = work[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    M = {}
    for A in names:
        for B2 in names:
            d = work[B2]; te = d["split"] == "test"
            M[(A, B2)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
    diag = np.mean([M[(x, x)] for x in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    inter = diag_interaction(M, names)
    boot = cluster_boot(work, names, heads) if do_boot else None
    return diag, off, inter, boot


def main():
    dirname = sys.argv[1] if len(sys.argv) > 1 else "track2_n600"
    fams, names = load(dirname)
    k = len(names)
    print(f"\n##### Control 19 family-subspace debiasing — {dirname} ({k} families) #####")
    print(f"families: {', '.join(names)}")
    print(f"subspace rank removed = k-1 = {k-1} (of 1536 dims)\n")

    # ---- before ----
    d0, o0, i0, b0 = run_pipeline(fams, names, None)
    print(f"{'condition':<22}{'diag':>7}{'offdiag':>9}{'interaction':>13}{'  interaction 95% CI':>24}")
    print(f"{'BEFORE (published)':<22}{d0:>7.3f}{o0:>9.3f}{i0:>+13.4f}   [{b0[4]:+.4f}, {b0[5]:+.4f}]")

    # ---- family deflation ----
    Bfam = family_subspace(fams, names)
    df, of, ifam, bf = run_pipeline(fams, names, Bfam)
    print(f"{'FAMILY-deflate':<22}{df:>7.3f}{of:>9.3f}{ifam:>+13.4f}   [{bf[4]:+.4f}, {bf[5]:+.4f}]")

    # ---- random deflation placebo (R seeds) ----
    rd, ro, ri = [], [], []
    for s in range(R_RANDOM):
        Br = random_subspace(fams[names[0]]["emb"].shape[1], k - 1, seed=1000 + s)
        dr, orr, ir, _ = run_pipeline(fams, names, Br)
        rd.append(dr); ro.append(orr); ri.append(ir)
    rd, ro, ri = np.array(rd), np.array(ro), np.array(ri)
    print(f"{'RANDOM-deflate(R=%d)' % R_RANDOM:<22}{rd.mean():>7.3f}{ro.mean():>9.3f}{ri.mean():>+13.4f}"
          f"   sd={ri.std():.4f} [{np.percentile(ri,2.5):+.4f}, {np.percentile(ri,97.5):+.4f}]")

    # ---- INLP aggressive-erasure rank sweep (answers 'you only removed 3 dims') ----
    dim = fams[names[0]]["emb"].shape[1]
    ranks = [r for r in (3, 10, 25, 50) if r < dim]
    print(f"\n--- INLP family-erasure rank sweep (family vs matched-rank random placebo) ---")
    print(f"{'rank':>5}{'  fam_int':>10}{'  fam_off':>9}{'  fam_diag':>10}"
          f"{'   rnd_int(mean)':>16}{'  rnd_off':>9}{'   z(fam vs rnd)':>16}")
    for r in ranks:
        Binlp = family_inlp_basis(fams, names, r)
        dfi, ofi, ii, _ = run_pipeline(fams, names, Binlp, do_boot=False)
        rri = []; rro = []
        for s in range(R_RANDOM):
            Br = random_subspace(dim, r, seed=2000 + s)
            _, orr2, ir2, _ = run_pipeline(fams, names, Br, do_boot=False)
            rri.append(ir2); rro.append(orr2)
        rri = np.array(rri); rro = np.array(rro)
        zr = (ii - rri.mean()) / (rri.std() + 1e-9)
        print(f"{r:>5}{ii:>+10.4f}{ofi:>+9.4f}{dfi:>+10.3f}"
              f"{rri.mean():>+16.4f}{rro.mean():>+9.4f}{zr:>+16.2f}")

    # ---- verdict ----
    print("\n--- interpretation ---")
    drop_fam = i0 - ifam
    drop_rand = i0 - ri.mean()
    z = (ifam - ri.mean()) / (ri.std() + 1e-9)
    print(f"interaction shrinkage: family={drop_fam:+.4f}  random(mean)={drop_rand:+.4f}")
    print(f"family interaction vs random-deflate null: z = {z:+.2f} "
          f"(family below random null => family-specific removal)")
    print(f"off-diagonal change: family {of-o0:+.4f}  random {ro.mean()-o0:+.4f}   "
          f"(positive = better cross-family detection = the defense metric)")
    print(f"diagonal change: family {df-d0:+.4f}  (detection capacity retained?)")
    if bf[4] > 0 and z > -2:
        print("=> IRREDUCIBLE branch: interaction survives family-deflation (CI>0) and is "
              "not shrunk beyond the random-rank placebo -> family signal entangled with "
              "the sabotage signal; cannot be linearly debiased.")
    elif z < -2 and (of - o0) > 0:
        print("=> MECHANISM+DEFENSE branch: family-deflation shrinks the interaction beyond "
              "the random placebo AND raises off-diagonal -> removable family subspace; "
              "project-out debiasing is a roster-independent defense.")
    else:
        print("=> MIXED: see numbers; report descriptively.")


if __name__ == "__main__":
    raise SystemExit(main())
