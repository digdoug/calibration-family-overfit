"""Strict (leakage-free) head-overfit ablation — Control 6 on the non-transductive basis.

track2_ablation.py runs the L2-C x PCA-dim ablation on the LEAKY transductive
pipeline (one shared scaler+PCA fit on pooled cal embeddings from all families,
so the target family leaks into the feature basis). The headline matrix has since
moved to the strict non-transductive basis (track2_matrix_strict: scaler/PCA fit
per calibration row A on A's cal split only). For a consistent presentation the
robustness ablation must report on the SAME strict basis as the headline.

This re-runs the IDENTICAL ablation design (same split, same L2-C x PCA-dim grid,
same diagonal-interaction statistic, true multiplicity task-cluster bootstrap) but
with per-row-A preprocessing. Only the basis + bootstrap differ from track2_ablation.

Run: .venv/bin/python scripts/track2_ablation_strict.py [--t2 track2] [--basis row-all]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402  (REPO, split_of, short_name, diag_interaction)
import track2_blatancy_strict as BS  # noqa: E402  (matrix_keep, boot_keep)

C_GRID = [0.01, 0.1, 1.0, 10.0]
DIM_GRID = [10, 25, 50]


def fit_row_model_cd(fams, family, basis, c, pca_dim):
    """Per-row-A scaler+PCA(pca_dim) + LogisticRegression(C=c), trained on A's cal split only.

    Mirrors track2_matrix_strict.fit_row_model but exposes C and pca_dim (which it
    hardcodes to C=1.0). The basis (data used to fit scaler/PCA) excludes ALL
    target-family data by construction.
    """
    d = fams[family]
    cal = d["split"] == "cal"
    if basis == "row-benign":
        basis_mask = cal & (d["label"] == 0)
    elif basis == "row-all":
        basis_mask = cal
    else:
        raise ValueError(basis)
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
    head = LogisticRegression(C=c, max_iter=5000).fit(x_train_f, y_train)
    return M.RowModel(scaler=scaler, pca=pca, head=head)


def precompute_scores_cd(fams, names, basis, c, pca_dim):
    models = {a: fit_row_model_cd(fams, a, basis, c, pca_dim) for a in names}
    return {a: {b: M.transform_score(models[a], fams[b]["emb"]) for b in names} for a in names}


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

    print(f"STRICT head-overfit ablation | t2={t2} basis={basis} | families={[M.short_name(n) for n in names]}")
    print(f"grid: L2 C in {C_GRID} x PCA dims in {DIM_GRID}  (headline = C=1.0, dim=50)")
    print("Q: does the strict interaction survive strong L2 / low PCA dim? (per-row-A preprocessing)\n")

    summary = []
    for dim in DIM_GRID:
        for c in C_GRID:
            scores = precompute_scores_cd(fams, names, basis, c, dim)
            tag = f"PCA={dim:>2} C={c:<5}" + ("  <-- headline" if (c == 1.0 and dim == 50) else "")
            inter, (im, ilo, ihi) = BS.matrix_keep(fams, names, scores, tag)
            summary.append((dim, c, inter, ilo, ihi))

    pos = [s for s in summary if s[3] > 0]
    print(f"\n{len(pos)}/{len(summary)} grid cells have interaction CI strictly > 0 (strict basis).")
    worst = min(summary, key=lambda s: s[2])
    print(f"weakest cell: PCA={worst[0]} C={worst[1]} -> interaction {worst[2]:+.4f} [{worst[3]:+.4f},{worst[4]:+.4f}]")


if __name__ == "__main__":
    raise SystemExit(main())
