"""Track-2 step 2: the family-CONDITIONED AUROC transfer matrix.

For each calibration family A, train a logistic head on A's calibration-split
honest-vs-attack transcript embeddings; score every eval family B's TEST-split
transcripts. Cell (A,B) = AUROC of A's head on B's held-out data.

Diagonal (A's head on A's held-out tasks) vs off-diagonal (A's head on B) tests
RANKING overfit: if the head overfits family A's stylistic signature, the diagonal
beats the off-diagonal NET of main effects -> the thesis. Null/negative -> pivot.

Split is task-disjoint and shared across families (no task in both cal & test).
PCA-50 on pooled cal embeddings keeps the per-family heads stable (n_cal~200, 1536-d).

Run: .venv/bin/python scripts/track2_matrix.py
"""
from __future__ import annotations
import glob, os, hashlib, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _clusterboot import task_rows, draw_index  # multiplicity-preserving cluster bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 else "track2")
CAL_FRAC = 60  # % of tasks to calibration (rest to test)


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells, names):
    """Mean diagonal INTERACTION net of row+col main effects (the real headline).
    interaction[A,B] = AUROC[A,B] - row_A - col_B + grand ; report mean over diagonal."""
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def cluster_boot(fams, names, heads, n_boot=1000, seed=0):
    """Bootstrap BOTH the raw diagonal bonus and the main-effects-removed interaction,
    resampling TEST tasks (cluster bootstrap by task)."""
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    rows = {B: task_rows(fams[B]["task"], fams[B]["split"] == "test", test_tasks) for B in names}
    raws, ints = [], []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        idxs = {B: draw_index(rows[B], draw) for B in names}
        cells, ok = {}, True
        for A in names:
            for B in names:
                d = fams[B]; idx = idxs[B]
                y = d["label"][idx]
                if len(np.unique(y)) < 2:
                    ok = False; break
                cells[(A, B)] = roc_auc_score(y, heads[A].decision_function(d["p"][idx]))
            if not ok:
                break
        if not ok:
            continue
        diag = np.mean([cells[(x, x)] for x in names])
        off = np.mean([cells[(a, b)] for a in names for b in names if a != b])
        raws.append(diag - off)
        ints.append(diag_interaction(cells, names))
    r, it = np.array(raws), np.array(ints)
    return (r.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5),
            it.mean(), np.percentile(it, 2.5), np.percentile(it, 97.5), len(it))


def main():
    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])

    # scaler + PCA on POOLED calibration embeddings (shared subspace)
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))

    # per-calibration-family logistic head
    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])

    # AUROC matrix: head A scored on family B's test split
    print("\n===== Track-2 AUROC matrix (rows=calib head, cols=eval family) =====")
    print("            " + "  ".join(f"{n.replace('ow_','').replace('_n200',''):>9}" for n in names))
    M = {}
    for A in names:
        row = []
        for B in names:
            d = fams[B]; te = d["split"] == "test"
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
            row.append(f"{M[(A,B)]:.3f}")
        print(f"{A.replace('ow_','').replace('_n200',''):>10}  " + "  ".join(f"{x:>9}" for x in row))

    diag = np.mean([M[(x, x)] for x in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    inter = diag_interaction(M, names)
    print(f"\nmean diagonal AUROC      = {diag:.3f}")
    print(f"mean off-diagonal AUROC  = {off:.3f}")
    print(f"raw diagonal bonus       = {diag-off:+.4f}  (confounded by row/col main effects)")
    print(f"INTERACTION (net of main effects) = {inter:+.4f}   <-- THE HEADLINE (s1b)")
    rm, rlo, rhi, im, ilo, ihi, nb = cluster_boot(fams, names, heads)
    print(f"  cluster-bootstrap (resample test tasks, n={nb}):")
    print(f"    raw bonus   : mean={rm:+.4f}  95% CI [{rlo:+.4f}, {rhi:+.4f}]")
    print(f"    INTERACTION : mean={im:+.4f}  95% CI [{ilo:+.4f}, {ihi:+.4f}]")
    print("\nInterpretation: interaction positive & CI>0 -> RANKING OVERFIT (thesis holds).")
    print("                null/negative -> pivot to obviousness+drift framing.")


if __name__ == "__main__":
    raise SystemExit(main())
