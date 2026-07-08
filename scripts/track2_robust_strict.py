"""Strict (leak-free) counterparts of Control 6 (head-overfit ablation) and
Control 10 (head-class robustness).

Both controls defend the HEADLINE interaction against "it's just small-n logistic-
head overfit" (Ctrl 6) and "it's an artifact of the linear logistic head" (Ctrl 10).
They were originally computed on the LEAKY transductive basis (one shared scaler/PCA
fit on the pooled four-family calibration embeddings), while the headline matrix has
since moved to the STRICT non-transductive basis (track2_matrix_strict.py: per-
calibration-family scaler/PCA, target family absent from the whole pipeline; leaky
+0.1098 -> strict +0.1717 on 4-open n600). A reviewer would flag the basis mismatch:
the robustness defenses must sit on the same pipeline as the number they defend.

This re-runs both on the SAME strict basis as the headline, with the multiplicity-
preserving task-cluster bootstrap. Same grids, same metric (interaction net of
row+col main effects), only the (already-vetted, already-headline) feature basis
changes -> a robustness re-computation, not a new pre-registered experiment.

Run:
  .venv-controlarena/bin/python scripts/track2_robust_strict.py ablation  [t2] [--basis row-all] [--boot 1000]
  .venv-controlarena/bin/python scripts/track2_robust_strict.py headclass [t2] [--basis row-all] [--boot 500]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402  (load_families, split_of, diag_interaction, short_name)

warnings.filterwarnings("ignore")

C_GRID = [0.01, 0.1, 1.0, 10.0]
DIM_GRID = [10, 25, 50]
HEAD_KINDS = ["logistic", "mlp", "svm_rbf", "rf", "hgb"]


def make_head(kind, C=1.0):
    if kind == "logistic":
        return LogisticRegression(C=C, max_iter=5000)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500, random_state=0)
    if kind == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=0)
    raise ValueError(kind)


def fit_row(d, basis, pca_dim, head_factory):
    """Per-calibration-family preprocessing fit on A's own basis only (strict)."""
    cal = d["split"] == "cal"
    basis_mask = (cal & (d["label"] == 0)) if basis == "row-benign" else cal
    x_basis = d["emb"][basis_mask]
    x_train = d["emb"][cal]
    y_train = d["label"][cal]
    if len(np.unique(y_train)) < 2:
        raise ValueError("calibration split has one class only")
    scaler = StandardScaler().fit(x_basis)
    xb = scaler.transform(x_basis)
    xt = scaler.transform(x_train)
    n_comp = min(pca_dim, xb.shape[0] - 1, xb.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(xb) if n_comp >= 2 else None
    xtf = pca.transform(xt) if pca is not None else xt
    head = head_factory().fit(xtf, y_train)
    return (scaler, pca, head)


def row_score(model, emb):
    scaler, pca, head = model
    x = scaler.transform(emb)
    if pca is not None:
        x = pca.transform(x)
    if hasattr(head, "decision_function"):
        s = head.decision_function(x)
    else:
        s = head.predict_proba(x)[:, 1]
    return np.asarray(s).ravel()


def cells_on(fams, names, models, idx_of):
    cells = {}
    for a in names:
        for b in names:
            d = fams[b]
            idx = idx_of[b]
            if len(idx) == 0 or len(np.unique(d["label"][idx])) < 2:
                return None
            cells[(a, b)] = roc_auc_score(d["label"][idx], row_score(models[a], d["emb"][idx]))
    return cells


def full_idx(fams, names):
    return {b: np.flatnonzero(fams[b]["split"] == "test") for b in names}


def summarize(cells, names):
    diag = float(np.mean([cells[(x, x)] for x in names]))
    off = float(np.mean([cells[(a, b)] for a in names for b in names if a != b]))
    return diag, off, M.diag_interaction(cells, names)


def cluster_boot(fams, names, models, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    # precompute test-eligible rows per (family, task) for fast multiplicity expansion
    test_pos = {b: {} for b in names}
    for b in names:
        d = fams[b]
        te = d["split"] == "test"
        for tk in test_tasks:
            rows = np.flatnonzero(te & (d["task"] == tk))
            if len(rows):
                test_pos[b][tk] = rows
    ints = []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        idx_of = {}
        for b in names:
            chunks = [test_pos[b][tk] for tk in draw if tk in test_pos[b]]
            idx_of[b] = np.concatenate(chunks) if chunks else np.array([], dtype=int)
        cells = cells_on(fams, names, models, idx_of)
        if cells is None:
            continue
        ints.append(M.diag_interaction(cells, names))
    it = np.array(ints)
    return float(it.mean()), float(np.percentile(it, 2.5)), float(np.percentile(it, 97.5)), len(it)


def run_ablation(fams, names, basis, n_boot):
    print(f"\n===== STRICT Control 6 — head-overfit ablation (L2-C x PCA-dim), basis={basis} =====")
    print(f"grid: L2 C in {C_GRID} x PCA dims in {DIM_GRID}  (headline = C=1.0, dim=50)\n")
    print(f"{'PCA':>4} {'C':>6} | {'interaction':>11}  {'95% CI':>20}  {'mean_diag':>9}  {'mean_off':>8}")
    print("-" * 72)
    summary = []
    for dim in DIM_GRID:
        for C in C_GRID:
            models = {a: fit_row(fams[a], basis, dim, lambda C=C: make_head("logistic", C)) for a in names}
            cells = cells_on(fams, names, models, full_idx(fams, names))
            diag, off, inter = summarize(cells, names)
            im, lo, hi, nb = cluster_boot(fams, names, models, n_boot)
            flag = "  <-- headline" if (C == 1.0 and dim == 50) else ("  CI>0" if lo > 0 else "")
            print(f"{dim:>4} {C:>6} | {inter:>+11.4f}  [{lo:>+7.4f},{hi:>+7.4f}]  {diag:>9.3f}  {off:>8.3f}{flag}")
            summary.append((dim, C, inter, lo, hi))
    pos = [s for s in summary if s[3] > 0]
    print(f"\n{len(pos)}/{len(summary)} grid cells have interaction CI strictly > 0.")
    worst = min(summary, key=lambda s: s[2])
    print(f"weakest cell: PCA={worst[0]} C={worst[1]} -> interaction {worst[2]:+.4f} [{worst[3]:+.4f},{worst[4]:+.4f}]")


def run_headclass(fams, names, basis, n_boot):
    print(f"\n===== STRICT Control 10 — head-class robustness (PCA-50), basis={basis} =====")
    print(f"{'head class':>10} | {'interaction':>11} | {'95% CI':>20} | {'mean diag':>9} | {'mean off':>8} | {'mistral-diag':>12}")
    print("-" * 90)
    for kind in HEAD_KINDS:
        models = {a: fit_row(fams[a], basis, 50, lambda kind=kind: make_head(kind)) for a in names}
        cells = cells_on(fams, names, models, full_idx(fams, names))
        diag, off, inter = summarize(cells, names)
        im, lo, hi, nb = cluster_boot(fams, names, models, n_boot)
        mname = next((n for n in names if "mistral" in n), None)
        mdiag = cells[(mname, mname)] if mname else float("nan")
        flag = "  <-- baseline" if kind == "logistic" else ("  CI>0" if lo > 0 else "")
        print(f"{kind:>10} | {inter:>+11.4f} | [{lo:>+7.4f},{hi:>+7.4f}] | {diag:>9.3f} | {off:>8.3f} | {mdiag:>12.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ablation", "headclass", "both"])
    ap.add_argument("t2", nargs="?", default="track2_n600")
    ap.add_argument("--basis", choices=["row-all", "row-benign"], default="row-all")
    ap.add_argument("--boot", type=int, default=None)
    args = ap.parse_args()

    t2_dir = os.path.join(M.REPO, "results", args.t2)
    names, fams = M.load_families(t2_dir)
    print(f"strict robustness controls: {t2_dir}")
    print(f"families: {', '.join(M.short_name(n) for n in names)} | basis={args.basis}")

    if args.mode in ("ablation", "both"):
        run_ablation(fams, names, args.basis, args.boot or 1000)
    if args.mode in ("headclass", "both"):
        run_headclass(fams, names, args.basis, args.boot or 500)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
