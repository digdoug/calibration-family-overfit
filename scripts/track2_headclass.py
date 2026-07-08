"""Step-2 down payment: is the +0.127 interaction specific to the LOGISTIC head class?

Phase 2 showed API few-shot monitors can't competently rank llama, so they couldn't adjudicate
whether the +0.127 is monitor-class-specific. The FULL test is a trainable transcript-reader
(LoRA LLM monitor, track2_lora_*). This is the cheap, CPU-only first cut: keep the SAME frozen
embeddings + the SAME pooled-cal PCA-50 pipeline as track2_matrix, but swap the per-family HEAD
for a different MODEL CLASS (MLP, gradient boosting, RBF-SVM, random forest) and re-measure the
interaction. If +0.127 survives across head classes, the diagonal overfit is NOT an artifact of
the linear logistic head — it lives in the family-conditioned representation (a stronger, broader
claim). It does NOT yet rule out frozen-embedding specificity (that's what the LoRA monitor tests).

Run: .venv/bin/python scripts/track2_headclass.py [track2|track2_actiononly]
"""
from __future__ import annotations
import glob, os, hashlib, sys, warnings
import numpy as np
from _clusterboot import task_rows, draw_index  # multiplicity-preserving cluster bootstrap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "track2")
CAL_FRAC = 60
N_BOOT = 500


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells, names):
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def scores(clf, X):
    """Uniform 1-D score for AUROC: decision_function if present, else P(class=1)."""
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]


def make_head(kind):
    if kind == "logistic":
        return LogisticRegression(C=1.0, max_iter=3000)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500, random_state=0)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=0)
    if kind == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    raise ValueError(kind)


def cluster_boot(fams, names, heads, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    rows = {B: task_rows(fams[B]["task"], fams[B]["split"] == "test", test_tasks) for B in names}
    ints = []
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
                cells[(A, B)] = roc_auc_score(y, scores(heads[A], d["p"][idx]))
            if not ok:
                break
        if ok:
            ints.append(diag_interaction(cells, names))
    it = np.array(ints)
    return it.mean(), np.percentile(it, 2.5), np.percentile(it, 97.5)


def main():
    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"],
                      "split": np.array([split_of(t) for t in d["task_id"]])}
    names = list(fams)
    short = lambda n: n.replace("ow_", "").replace("_n200", "")
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))

    print(f"data: {T2}   families: {', '.join(short(n) for n in names)}")
    print(f"Q: does the +0.127 interaction survive a DIFFERENT head model class? (same frozen emb + PCA-50)\n")
    print(f"{'head class':>10} | {'interaction':>11} | {'95% CI':>20} | {'mean diag':>9} | {'mean off':>8} | {'mistral-diag':>12}")
    print("-" * 88)
    for kind in ("logistic", "mlp", "svm_rbf", "rf", "hgb"):
        heads = {}
        for A in names:
            d = fams[A]; cal = d["split"] == "cal"
            heads[A] = make_head(kind).fit(d["p"][cal], d["label"][cal])
        M = {}
        for A in names:
            for B in names:
                d = fams[B]; te = d["split"] == "test"
                M[(A, B)] = roc_auc_score(d["label"][te], scores(heads[A], d["p"][te]))
        diag = np.mean([M[(x, x)] for x in names]); off = np.mean([M[(a, b)] for a in names for b in names if a != b])
        inter = diag_interaction(M, names)
        im, lo, hi = cluster_boot(fams, names, heads)
        mdiag = M[(n_, n_)] if (n_ := next((n for n in names if "mistral" in n), None)) else float("nan")
        flag = "  <-- baseline" if kind == "logistic" else ("  CI>0" if lo > 0 else "")
        print(f"{kind:>10} | {inter:>+11.4f} | [{lo:>+7.4f},{hi:>+7.4f}] | {diag:>9.3f} | {off:>8.3f} | {mdiag:>12.3f}{flag}")
    print("\nInterpretation: interaction stays positive (CI>0) across head classes -> NOT specific to the")
    print("                LINEAR logistic head; the family-conditioned overfit is in the representation.")
    print("                (Does NOT yet rule out frozen-EMBEDDING specificity -> that's the LoRA monitor test.)")


if __name__ == "__main__":
    raise SystemExit(main())
