"""Strict (leakage-free) head-class robustness — Control 10 on the non-transductive basis.

track2_headclass.py asks whether the diagonal-bonus interaction is specific to the
LINEAR logistic head by swapping the per-family head for MLP / SVM-RBF / random
forest / HistGradientBoosting, all on the LEAKY transductive pipeline (one shared
scaler+PCA-50 fit on pooled cal embeddings, target family in the basis). The
headline matrix has moved to the strict non-transductive basis
(track2_matrix_strict: scaler/PCA fit per calibration row A on A's cal split only),
so this robustness check must report on the SAME basis for a consistent paper.

This re-runs the IDENTICAL head-class sweep (same split, same PCA-50, same uniform
AUROC score = decision_function or P(class=1), same diagonal-interaction statistic,
true multiplicity task-cluster bootstrap) but with per-row-A preprocessing. Only the
basis + bootstrap differ from track2_headclass.

Run: .venv/bin/python scripts/track2_headclass_strict.py [--t2 track2] [--basis row-all]
"""
from __future__ import annotations
import glob, os, sys, warnings
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402  (REPO, split_of, short_name, diag_interaction)
import track2_blatancy_strict as BS  # noqa: E402  (matrix_keep, boot_keep)

PCA_DIM = 50


def make_head(kind):
    if kind == "logistic":
        return LogisticRegression(C=1.0, max_iter=3000)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500, random_state=0)
    if kind == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=0)
    raise ValueError(kind)


@dataclass
class HeadModel:
    scaler: StandardScaler
    pca: PCA | None
    head: object


def _uniform_score(head, X):
    """Uniform 1-D score for AUROC: decision_function if present else P(class=1)."""
    if hasattr(head, "decision_function"):
        return head.decision_function(X)
    return head.predict_proba(X)[:, 1]


def fit_row_head(fams, family, basis, kind, pca_dim=PCA_DIM):
    """Per-row-A scaler+PCA(pca_dim) + a head of class `kind`, trained on A's cal split only.

    Mirrors track2_matrix_strict.fit_row_model but swaps the head class. The basis
    (data used to fit scaler/PCA) excludes ALL target-family data by construction.
    """
    d = fams[family]
    cal = d["split"] == "cal"
    basis_mask = cal if basis == "row-all" else (cal & (d["label"] == 0))
    x_basis = d["emb"][basis_mask]
    x_train = d["emb"][cal]
    y_train = d["label"][cal]
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{family}: cal split one class")
    scaler = StandardScaler().fit(x_basis)
    x_basis_s = scaler.transform(x_basis)
    n_comp = min(pca_dim, x_basis_s.shape[0] - 1, x_basis_s.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, random_state=0).fit(x_basis_s)
        x_train_f = pca.transform(scaler.transform(x_train))
    else:
        pca = None
        x_train_f = scaler.transform(x_train)
    head = make_head(kind).fit(x_train_f, y_train)
    return HeadModel(scaler=scaler, pca=pca, head=head)


def _transform_score(model, emb):
    x = model.scaler.transform(emb)
    if model.pca is not None:
        x = model.pca.transform(x)
    return _uniform_score(model.head, x)


def precompute_scores_head(fams, names, basis, kind):
    models = {a: fit_row_head(fams, a, basis, kind) for a in names}
    return {a: {b: _transform_score(models[a], fams[b]["emb"]) for b in names} for a in names}


def main():
    args = sys.argv[1:]
    t2 = "track2"
    if "--t2" in args:
        i = args.index("--t2"); t2 = args[i + 1]; del args[i:i + 2]
    basis = "row-all"
    if "--basis" in args:
        i = args.index("--basis"); basis = args[i + 1]; del args[i:i + 2]
    T2 = os.path.join(M.REPO, "results", t2)

    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"].astype(np.float32), "label": d["label"].astype(int),
                      "task": d["task_id"].astype(object)}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([M.split_of(str(t)) for t in d["task"]], dtype=object)
        d["keep"] = np.ones(len(d["task"]), dtype=bool)

    print(f"STRICT head-class robustness | t2={t2} basis={basis} | families={[M.short_name(n) for n in names]}")
    print(f"Q: does the strict interaction survive a DIFFERENT head model class? (per-row-A PCA-{PCA_DIM})\n")

    for kind in ("logistic", "mlp", "svm_rbf", "rf", "hgb"):
        scores = precompute_scores_head(fams, names, basis, kind)
        tag = f"{kind:>9}" + ("  <-- baseline" if kind == "logistic" else "")
        BS.matrix_keep(fams, names, scores, tag)

    print("\nInterpretation: interaction stays positive (CI>0) across head classes on the strict basis ->")
    print("                NOT specific to the linear logistic head; the family-conditioned overfit is in")
    print("                the representation. (Frozen-EMBEDDING specificity is the LoRA-monitor test.)")


if __name__ == "__main__":
    raise SystemExit(main())
