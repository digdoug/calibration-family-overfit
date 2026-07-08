"""Strict (leakage-free) counterpart of Control 14c (track2_mixedeffects.py).

Control 14c is the single conditional placement-value OLS that adjusts the diagonal detection
bonus for the CONTINUOUS Haiku obviousness grade and clusters by task, but it uses the LEAKY
transductive pipeline (one shared scaler/PCA on the pooled four-family calibration embeddings).
This re-runs the IDENTICAL model on the strict NON-transductive basis (track2_matrix_strict:
per-calibration-family scaler/PCA, target family absent from the whole pipeline), so the
conditional confirmation is on the same leak-free pipeline as the headline matrix.

Only the scoring function changes: head-A scores on family B come from B's own per-family strict
RowModel (M.transform_score) instead of the shared pooled head. Everything else (placement value,
design matrix, validity check, task-cluster bootstrap) is reused from track2_mixedeffects.

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_mixedeffects_strict.py \
        [--basis row-all|row-benign] [--boot N]
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402
from _clusterboot import task_rows, draw_index  # noqa: E402
import track2_mixedeffects as ME  # noqa: E402  (load, placement, ols, diag_interaction)
from track2_blatancy import diag_interaction  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PCA_DIM = 50


def strict_scores(fams, names, basis):
    """S[(A,B)] = strict head-A decision scores on ALL rows of family B (per-family A basis)."""
    models = {a: M.fit_row_model(fams, a, basis, PCA_DIM) for a in names}
    return {(a, b): M.transform_score(models[a], fams[b]["emb"]) for a in names for b in names}


def build_rows(fams, names, S, att_idx, ben_idx, grade_center):
    """Long design from per-family attack/benign row indices, using strict scores S."""
    ben_scores = {(A, B): S[(A, B)][ben_idx[B]] for A in names for B in names}
    pol_levels = names[1:]
    mon_levels = names[1:]
    rows_y, rows_X = [], []
    for B in names:
        d = fams[B]
        ai = att_idx[B]
        if len(ai) == 0:
            continue
        g = d["blat2"][ai]; m = ~np.isnan(g)
        ai = ai[m]; g = g[m]
        if len(ai) == 0:
            continue
        gc = g - grade_center
        for A in names:
            v = ME.placement(S[(A, B)][ai], ben_scores[(A, B)])
            n = len(v)
            X = np.zeros((n, 1 + len(pol_levels) + len(mon_levels) + 2))
            X[:, 0] = 1.0
            for k, lv in enumerate(pol_levels):
                if B == lv:
                    X[:, 1 + k] = 1.0
            for k, lv in enumerate(mon_levels):
                if A == lv:
                    X[:, 1 + len(pol_levels) + k] = 1.0
            X[:, 1 + len(pol_levels) + len(mon_levels)] = 1.0 if A == B else 0.0
            X[:, -1] = gc
            rows_y.append(v); rows_X.append(X)
    y = np.concatenate(rows_y); X = np.vstack(rows_X)
    cols = (["intercept"] + [f"pol[{l}]" for l in pol_levels]
            + [f"mon[{l}]" for l in mon_levels] + ["diagonal", "blatancy_c"])
    return X, y, cols


def matrix_interaction(fams, names, S):
    Mx = {}
    for A in names:
        for B in names:
            d = fams[B]; te = (d["split"] == "test")
            att = te & (d["label"] == 1) & ~np.isnan(d["blat2"])
            ben = te & (d["label"] == 0)
            idx = np.flatnonzero(att | ben)
            Mx[(A, B)] = roc_auc_score(d["label"][idx], S[(A, B)][idx])
    diag = np.mean([Mx[(x, x)] for x in names])
    off = np.mean([Mx[(a, b)] for a in names for b in names if a != b])
    return diag_interaction(Mx, names), diag - off, Mx


def point_estimates(fams, names, S, grade_center):
    test = {B: (fams[B]["split"] == "test") for B in names}
    att_idx = {B: np.flatnonzero(test[B] & (fams[B]["label"] == 1)) for B in names}
    ben_idx = {B: np.flatnonzero(test[B] & (fams[B]["label"] == 0)) for B in names}
    X, y, cols = build_rows(fams, names, S, att_idx, ben_idx, grade_center)
    di = cols.index("diagonal"); bi = cols.index("blatancy_c")
    beta_full = ME.ols(X, y)
    beta_nb = ME.ols(np.delete(X, bi, axis=1), y)
    Xint = np.hstack([X, (X[:, di] * X[:, bi])[:, None]])
    beta_int = ME.ols(Xint, y)
    return {"n_rows": len(y), "diag_with_blat": float(beta_full[di]), "blat_coef": float(beta_full[bi]),
            "diag_no_blat": float(beta_nb[di]), "diag_in_intmodel": float(beta_int[di]),
            "diagXblat": float(beta_int[-1])}


def cluster_boot(fams, names, S, grade_center, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values()
                         for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    att_rows = {B: task_rows(fams[B]["task"], (fams[B]["split"] == "test")
                & (fams[B]["label"] == 1) & ~np.isnan(fams[B]["blat2"]), test_tasks) for B in names}
    ben_rows = {B: task_rows(fams[B]["task"], (fams[B]["split"] == "test")
                & (fams[B]["label"] == 0), test_tasks) for B in names}
    dwb, dnb, bc = [], [], []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        att_idx = {B: draw_index(att_rows[B], draw) for B in names}
        ben_idx = {B: draw_index(ben_rows[B], draw) for B in names}
        if any(len(att_idx[B]) == 0 or len(ben_idx[B]) == 0 for B in names):
            continue
        X, y, cols = build_rows(fams, names, S, att_idx, ben_idx, grade_center)
        di = cols.index("diagonal"); bi = cols.index("blatancy_c")
        b = ME.ols(X, y)
        dwb.append(b[di]); bc.append(b[bi])
        dnb.append(ME.ols(np.delete(X, bi, axis=1), y)[di])
    f = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"n_boot": len(dwb), "diag_with_blat": f(dwb), "diag_no_blat": f(dnb), "blat_coef": f(bc)}


def main():
    args = sys.argv[1:]
    n_boot = 1000
    if "--boot" in args:
        i = args.index("--boot"); n_boot = int(args[i + 1]); del args[i:i + 2]
    basis = "row-all"
    if "--basis" in args:
        i = args.index("--basis"); basis = args[i + 1]; del args[i:i + 2]

    fams = ME.load(); names = list(fams)
    print(f"STRICT Control 14c | basis={basis} | families={names}")
    S = strict_scores(fams, names, basis)

    pooled_g = np.concatenate([fams[B]["blat2"][(fams[B]["split"] == "test")
                               & (fams[B]["label"] == 1) & ~np.isnan(fams[B]["blat2"])] for B in names])
    gc = float(np.mean(pooled_g))
    print(f"graded test attacks pooled n={len(pooled_g)}  grade center={gc:.3f} sd={pooled_g.std():.3f}")

    mi, raw, _ = matrix_interaction(fams, names, S)
    print(f"\nVALIDITY TARGET (cell-AUROC strict, graded test attacks): "
          f"diag_interaction={mi:+.4f}  raw diag-off={raw:+.4f}")

    N = len(names); sym = (N - 1) / N
    pe = point_estimates(fams, names, S, gc)
    print(f"\nPlacement-value OLS point estimates (strict; n_rows={pe['n_rows']}):")
    print(f"  diagonal coef, NO blatancy   = {pe['diag_no_blat']:+.4f}   "
          f"(sym-equiv = {pe['diag_no_blat']*sym:+.4f} vs diag_interaction {mi:+.4f})")
    print(f"  diagonal coef, WITH blatancy = {pe['diag_with_blat']:+.4f}   "
          f"(sym-equiv {pe['diag_with_blat']*sym:+.4f})")
    print(f"  blatancy_c coef              = {pe['blat_coef']:+.5f}  (per +1 grade pt)")
    print(f"  diagonal (in diag:blat model)= {pe['diag_in_intmodel']:+.4f}   "
          f"diagonal:blatancy = {pe['diagXblat']:+.5f}")
    vc = abs(pe["diag_no_blat"] * sym - mi)
    print(f"\nVALIDITY: |no-blat diagonal x{(N-1)}/{N} - diag_interaction| = {vc:.4f}  "
          f"({'PASS' if vc < 0.02 else 'CHECK'} <0.02)")

    print(f"\nCluster bootstrap by task (target {n_boot}) ...")
    cb = cluster_boot(fams, names, S, gc, n_boot)
    print(f"  (n_boot kept = {cb['n_boot']})")
    def show(k, lab):
        m, lo, hi = cb[k]
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI INCLUDES 0"
        print(f"  {lab:<28} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   {sig}")
    show("diag_no_blat", "diagonal, no blatancy")
    show("diag_with_blat", "diagonal, adj. blatancy")
    show("blat_coef", "blatancy_c slope")

    print("\nInterpretation: the diagonal coefficient adj. the continuous obviousness grade, with a")
    print("  task-clustered CI excluding 0 on the leak-free per-family basis, confirms the diagonal")
    print("  bonus is not an artifact of obviousness, task clustering, OR the transductive basis.")


if __name__ == "__main__":
    raise SystemExit(main())
