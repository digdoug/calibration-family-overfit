"""Control 22 — cross-ARCHITECTURE cell-level concordance of the transfer matrix.

Controls 10/11 show the *aggregate* interaction is positive for both the
frozen-embedding+logistic head AND the trainable Phi-3.5 LoRA reader. They do NOT
test whether the two monitors agree on WHICH family pairs fail to transfer. If two
architecturally-disjoint monitors localize the failure to the SAME lineage pairs,
the cross-family gap is a property of the lineage PAIR (the sabotage data), not of
any one monitor implementation — a sharper, mechanistic form of "monitor-robust".

We compute each monitor's canonical 4x4 AUROC matrix on the SAME 4 open families
and the SAME md5-seed-0 60/40 task split, then measure cell-level concordance:
  (A) all 16 cells, (B) the 12 OFF-diagonal (transfer) cells [the sharp claim:
      the diagonal is excluded so concordance is not driven by "own family easy"],
  (C) double-centered interaction residuals [agreement net of row/col main effects].
Significance: a Mantel permutation test that jointly relabels families (24 perms),
holding each matrix's marginal structure fixed.

The two matrices are INDEPENDENT generations (LoRA n~150/cell vs embedding n=600);
the n gap only adds noise to the LoRA matrix, so any positive concordance is a
conservative LOWER bound. This is matrix-level concordance by design (sample-level
pairing across two separate generations is not valid).

Run: .venv/bin/python scripts/track2_concordance.py
"""
from __future__ import annotations
import glob, os, hashlib, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMS = ["gemma", "llama", "mistral", "qwen"]
CAL_FRAC = 60


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def embedding_matrix(emb_dir="results/track2_n600"):
    """Canonical track2_matrix logic: PCA-50 pooled cal, per-family logistic head,
    cell (A,B) = AUROC of head A on family B's test split. Returns 4x4 in FAMS order."""
    fams = {}
    for f in sorted(glob.glob(os.path.join(REPO, emb_dir, "*.npz"))):
        name = os.path.basename(f)[:-4].replace("ow_", "").replace("_n600", "")
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in FAMS:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    M = np.zeros((4, 4))
    for i, A in enumerate(FAMS):
        for j, B in enumerate(FAMS):
            d = fams[B]; te = d["split"] == "test"
            M[i, j] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
    return M


def lora_matrix(npz="results/track2_lora/scores_lora.npz"):
    d = np.load(os.path.join(REPO, npz), allow_pickle=True)
    score, label, calib, ev = d["score"], d["label"], d["calib"], d["eval"]
    M = np.zeros((4, 4))
    for i, A in enumerate(FAMS):
        for j, B in enumerate(FAMS):
            m = (calib == A) & (ev == B)
            y = label[m]
            M[i, j] = roc_auc_score(y, score[m]) if len(np.unique(y)) == 2 else np.nan
    return M


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def double_center(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def mantel(M1, M2, cells="off", stat="spearman"):
    """Joint-relabel Mantel test. Permute family order pi applied to BOTH rows and
    cols of M2; recompute correlation on the chosen cell set. Exact over all 4!=24
    perms. Returns observed corr, exact p (one-sided, fraction of perms >= observed)."""
    idx = np.arange(4)
    if cells == "off":
        mask = ~np.eye(4, dtype=bool)
    elif cells == "all":
        mask = np.ones((4, 4), dtype=bool)
    elif cells == "resid":
        M1, M2 = double_center(M1), double_center(M2)
        mask = ~np.eye(4, dtype=bool)
    corr = spearman if stat == "spearman" else pearson
    a = M1[mask]
    obs = corr(a, M2[mask])
    ge = 0; perms = list(itertools.permutations(idx))
    for p in perms:
        p = np.array(p)
        M2p = M2[np.ix_(p, p)]
        if corr(a, M2p[mask]) >= obs - 1e-12:
            ge += 1
    return obs, ge / len(perms)


def report(name, Mlora):
    Memb = embedding_matrix()
    print(f"\n================ {name} ================")
    print("embedding-head matrix (rows=calib, cols=eval):")
    for i, A in enumerate(FAMS):
        print(f"  {A:8s} " + " ".join(f"{Memb[i,j]:.3f}" for j in range(4)))
    print("LoRA matrix:")
    for i, A in enumerate(FAMS):
        print(f"  {A:8s} " + " ".join(f"{Mlora[i,j]:.3f}" for j in range(4)))

    off = ~np.eye(4, dtype=bool)
    print("\n-- concordance (embedding-head vs LoRA) --")
    for label, cells in [("all 16 cells", "all"), ("12 OFF-diagonal", "off"),
                          ("double-centered resid (off-diag)", "resid")]:
        rho, p = mantel(Memb, Mlora, cells=cells, stat="spearman")
        rr, _ = mantel(Memb, Mlora, cells=cells if cells != "resid" else "off", stat="pearson") \
            if cells != "resid" else (pearson(double_center(Memb)[off], double_center(Mlora)[off]), None)
        print(f"  {label:34s}: Spearman rho={rho:+.3f} (Mantel p={p:.3f}, exact/24)  Pearson r={rr:+.3f}")
    # diagonal vs off agreement, plainly
    de, do = np.diag(Memb).mean(), Memb[off].mean()
    dl, dol = np.diag(Mlora).mean(), Mlora[off].mean()
    print(f"  embedding diag {de:.3f} vs off {do:.3f}; LoRA diag {dl:.3f} vs off {dol:.3f}")
    return Memb


def main():
    Memb = report("Phi-3.5 LoRA (seed 0, headline)", lora_matrix())
    for tag, npz in [("Phi-3.5 seed1", "results/track2_lora/scores_lora_seed1.npz"),
                     ("Phi-3.5 seed2", "results/track2_lora/scores_lora_seed2.npz"),
                     ("Phi-4-mini seed1", "results/track2_lora/scores_lora_phi4mini_seed1.npz")]:
        if os.path.exists(os.path.join(REPO, npz)):
            try:
                report(tag, lora_matrix(npz))
            except Exception as e:
                print(f"\n[{tag}] skipped: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
