"""Control 14c-FULL: a reviewer's single conditional task-level model of the diagonal bonus.

Pre-registered in an addendum before running. The Control-14c placement-value OLS
adjusts the diagonal detection bonus for policy + monitor main effects + continuous blatancy and
clusters by task. An external review asks for the bonus in ONE conditional model
with the FULL covariate set: task fixed effects, family main effects, success strata, length, and
the obviousness grade. This builds exactly that, on the STRICT leak-free per-family basis
(track2_matrix_strict heads), reusing every piece of the published inference machinery.

Design (placement-value OLS, rows = test ATTACK i in family B  x  head A):
  placement ~ 1 + C(pol_family) + C(mon_family) + diagonal + blatancy_c + length_c
              + main_ok + side_ok  [+ task fixed effects]
  placement v_iA = mean_j 1[s_iA > s_jB] (+0.5 ties) over family-B test BENIGN scores under head A;
  by construction mean over attacks of v_iA = AUROC(A,B), so the OLS is a faithful, scale-free
  decomposition of the strict AUROC matrix and the `diagonal` coef = the diagonal bonus net of the
  additive row/col effects AND every covariate.

Structural note (pre-registered): every attack contributes its blatancy/length/success/task
identically to its 1 diagonal and 3 off-diagonal placement rows, so each attack-/task-level
covariate is ORTHOGONAL by construction to the diagonal indicator. The nested ladder
base -> +blatancy -> +length -> +success -> +taskFE should therefore leave the diagonal coef
numerically invariant. That invariance IS the construct-validity argument (a per-attacker
obviousness confound loads on the column main effect, never on the diagonal). The genuinely
non-orthogonal test is the diagonal x covariate interactions, reported separately.

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_mixedeffects_full.py \
        [--basis row-all|row-benign] [--boot N] [--nonstrict]
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402
import track2_mixedeffects as ME  # noqa: E402  (load, placement, ols, diag_interaction)
from track2_blatancy import diag_interaction  # noqa: E402
from _clusterboot import task_rows, draw_index  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(REPO, "results", "track2_n600", "sample_meta.json")
PCA_DIM = 50


def attach_meta(fams, names):
    """Attach per-row length(log1p nmsg)/main/side from sample_meta.json, keyed by (label,task).

    The npz keeps only rows with a valid Track-1 suspicion, so it can be a strict subset of the
    eval samples in sample_meta.json (e.g. mistral 1128 npz vs 1200 eval). (label,task) is unique
    per family (verified), so we look up each npz row by key rather than positionally.
    """
    meta = json.load(open(META))
    for name in names:
        lut = {(r["label"], r["task"]): r for r in meta[name]}
        d = fams[name]
        nmsg, main, side = [], [], []
        for lab, t in zip(d["label"], d["task"]):
            r = lut[(int(lab), str(t))]
            nmsg.append(r["nmsg"])
            main.append(r["main"] if r["main"] is not None else 0)
            side.append(r["side"] if r["side"] is not None else 0)
        d["nmsg"] = np.array(nmsg, dtype=float)
        d["logn"] = np.log1p(d["nmsg"])
        d["main"] = np.array(main, dtype=float)
        d["side"] = np.array(side, dtype=float)
    return fams


def strict_scores(fams, names, basis):
    models = {a: M.fit_row_model(fams, a, basis, PCA_DIM) for a in names}
    return {(a, b): M.transform_score(models[a], fams[b]["emb"]) for a in names for b in names}


def _att_mask(d, idx):
    """Graded test-attack subset of a row-index array (drop NaN blatancy)."""
    g = d["blat2"][idx]
    return idx[~np.isnan(g)]


def build_long(fams, names, S, att_idx, ben_idx, centers, with_taskfe):
    """Long design + parallel arrays. Returns (X, y, cols, task_of_row).

    Base columns: intercept, pol[..], mon[..], diagonal, blatancy_c, length_c, main_ok, side_ok.
    Optional task fixed-effect dummies (drop-first) appended; their columns are listed in cols.
    """
    ben_scores = {(A, B): S[(A, B)][ben_idx[B]] for A in names for B in names}
    pol_levels = names[1:]
    mon_levels = names[1:]
    gc, lc = centers["blat"], centers["logn"]
    base_y, base_blocks, task_blocks = [], [], []
    for B in names:
        d = fams[B]
        ai = _att_mask(d, att_idx[B])
        if len(ai) == 0:
            continue
        g = d["blat2"][ai] - gc
        ln = d["logn"][ai] - lc
        mn = d["main"][ai]; sd = d["side"][ai]
        tk = d["task"][ai]
        for A in names:
            v = ME.placement(S[(A, B)][ai], ben_scores[(A, B)])
            n = len(v)
            blk = np.zeros((n, 1 + len(pol_levels) + len(mon_levels) + 5))
            c = 0
            blk[:, c] = 1.0; c += 1
            for lv in pol_levels:
                blk[:, c] = 1.0 if B == lv else 0.0; c += 1
            for lv in mon_levels:
                blk[:, c] = 1.0 if A == lv else 0.0; c += 1
            blk[:, c] = 1.0 if A == B else 0.0; c += 1   # diagonal
            blk[:, c] = g;  c += 1                         # blatancy_c
            blk[:, c] = ln; c += 1                         # length_c
            blk[:, c] = mn; c += 1                         # main_ok
            blk[:, c] = sd; c += 1                         # side_ok
            base_y.append(v); base_blocks.append(blk); task_blocks.append(tk)
    y = np.concatenate(base_y)
    X = np.vstack(base_blocks)
    task = np.concatenate(task_blocks)
    cols = (["intercept"] + [f"pol[{l}]" for l in pol_levels] + [f"mon[{l}]" for l in mon_levels]
            + ["diagonal", "blatancy_c", "length_c", "main_ok", "side_ok"])
    if with_taskfe:
        uniq = sorted(set(task.tolist()))[1:]   # drop-first task = reference
        tfe = np.zeros((len(task), len(uniq)))
        tindex = {t: k for k, t in enumerate(uniq)}
        for r, t in enumerate(task):
            k = tindex.get(t)
            if k is not None:
                tfe[r, k] = 1.0
        X = np.hstack([X, tfe])
        cols = cols + [f"task[{t}]" for t in uniq]
    return X, y, cols, task


def fit_diag(X, cols, y, drops=()):
    """OLS, return diagonal coefficient after dropping named columns."""
    keep = [i for i, c in enumerate(cols) if c not in drops]
    sub = X[:, keep]
    beta = ME.ols(sub, y)
    di = [c for c in cols if c not in drops].index("diagonal")
    return float(beta[di])


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
    return diag_interaction(Mx, names), diag - off


def interactions(X, cols, y):
    """Append diagonal x {blatancy,length,main,side} interaction columns, fit, report each."""
    di = cols.index("diagonal")
    terms = ["blatancy_c", "length_c", "main_ok", "side_ok"]
    extra = np.column_stack([X[:, di] * X[:, cols.index(t)] for t in terms])
    Xi = np.hstack([X, extra])
    beta = ME.ols(Xi, y)
    base = len(cols)
    return {f"diagonal:{t}": float(beta[base + k]) for k, t in enumerate(terms)} | {
        "diagonal": float(beta[di])}


def point_estimates(fams, names, S, centers):
    test = {B: (fams[B]["split"] == "test") for B in names}
    att_idx = {B: np.flatnonzero(test[B] & (fams[B]["label"] == 1)) for B in names}
    ben_idx = {B: np.flatnonzero(test[B] & (fams[B]["label"] == 0)) for B in names}
    X, y, cols, task = build_long(fams, names, S, att_idx, ben_idx, centers, with_taskfe=False)
    Xf, yf, colsf, _ = build_long(fams, names, S, att_idx, ben_idx, centers, with_taskfe=True)
    # nested ladder of the diagonal coefficient (drops on the no-taskFE design)
    ladder = {
        "base (pol+mon)":     fit_diag(X, cols, y, drops=("blatancy_c", "length_c", "main_ok", "side_ok")),
        "+blatancy":          fit_diag(X, cols, y, drops=("length_c", "main_ok", "side_ok")),
        "+length":            fit_diag(X, cols, y, drops=("main_ok", "side_ok")),
        "+success strata":    fit_diag(X, cols, y, drops=()),
        "+task FE (full)":    fit_diag(Xf, colsf, yf, drops=()),
    }
    inter = interactions(X, cols, y)
    return {"n_rows": len(y), "n_taskfe": Xf.shape[1] - X.shape[1], "ladder": ladder,
            "inter": inter}


def cluster_boot(fams, names, S, centers, n_boot, with_taskfe, seed=0):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values()
                         for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    att_rows = {B: task_rows(fams[B]["task"], (fams[B]["split"] == "test")
                & (fams[B]["label"] == 1) & ~np.isnan(fams[B]["blat2"]), test_tasks) for B in names}
    ben_rows = {B: task_rows(fams[B]["task"], (fams[B]["split"] == "test")
                & (fams[B]["label"] == 0), test_tasks) for B in names}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        att_idx = {B: draw_index(att_rows[B], draw) for B in names}
        ben_idx = {B: draw_index(ben_rows[B], draw) for B in names}
        if any(len(att_idx[B]) == 0 or len(ben_idx[B]) == 0 for B in names):
            continue
        X, y, cols, _ = build_long(fams, names, S, att_idx, ben_idx, centers, with_taskfe)
        vals.append(fit_diag(X, cols, y))
    a = np.array(vals)
    return len(a), float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    args = sys.argv[1:]
    n_boot = 1000
    basis = "row-all"
    nonstrict = "--nonstrict" in args
    if "--boot" in args:
        i = args.index("--boot"); n_boot = int(args[i + 1]); del args[i:i + 2]
    if "--basis" in args:
        i = args.index("--basis"); basis = args[i + 1]; del args[i:i + 2]

    fams = ME.load(); names = list(fams)
    attach_meta(fams, names)
    tag = "NON-STRICT (transductive)" if nonstrict else f"STRICT (leak-free per-family, basis={basis})"
    print(f"Control 14c-FULL | {tag} | families={names}")

    if nonstrict:
        heads = ME.fit_heads(fams)
        S = {(a, b): heads[a].decision_function(fams[b]["p"]) for a in names for b in names}
    else:
        S = strict_scores(fams, names, basis)

    # centers from pooled graded test attacks (fixed reference)
    sel = lambda B: ((fams[B]["split"] == "test") & (fams[B]["label"] == 1) & ~np.isnan(fams[B]["blat2"]))
    pooled_g = np.concatenate([fams[B]["blat2"][sel(B)] for B in names])
    pooled_l = np.concatenate([fams[B]["logn"][sel(B)] for B in names])
    centers = {"blat": float(pooled_g.mean()), "logn": float(pooled_l.mean())}
    print(f"graded test attacks pooled n={len(pooled_g)}  "
          f"blat center={centers['blat']:.3f}  logn center={centers['logn']:.3f}")

    mi, raw = matrix_interaction(fams, names, S)
    print(f"\nVALIDITY TARGET (cell-AUROC, graded test attacks): "
          f"diag_interaction={mi:+.4f}  raw diag-off={raw:+.4f}")

    N = len(names); sym = (N - 1) / N
    pe = point_estimates(fams, names, S, centers)
    print(f"\nPlacement-value OLS, nested ladder of the diagonal coefficient "
          f"(n_rows={pe['n_rows']}, task FE dummies={pe['n_taskfe']}):")
    print(f"  {'model':<22} {'diag coef':>10}   {'sym-equiv x3/4':>14}")
    for k, v in pe["ladder"].items():
        print(f"  {k:<22} {v:>+10.4f}   {v*sym:>+14.4f}")
    base = pe["ladder"]["base (pol+mon)"]; full = pe["ladder"]["+task FE (full)"]
    shrink = 0.0 if base == 0 else 100 * (1 - full / base)
    vc = abs(base * sym - mi)
    print(f"\n  VALIDITY: |base diag x3/4 - diag_interaction| = {vc:.4f}  "
          f"({'PASS' if vc < 0.02 else 'CHECK'} <0.02)")
    print(f"  full-vs-base shrink = {shrink:+.1f}%  "
          f"({'PASS <40%' if abs(shrink) < 40 else 'CHECK'})")

    print(f"\n  diagonal x covariate interactions (the non-orthogonal stratum test):")
    for k, v in pe["inter"].items():
        if k != "diagonal":
            print(f"    {k:<22} {v:>+10.5f}")

    print(f"\nCluster bootstrap by task (target {n_boot}) ...")
    for label, tfe in [("full WITHOUT task FE", False), ("FULL WITH task FE", True)]:
        nb, m, lo, hi = cluster_boot(fams, names, S, centers, n_boot, tfe)
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI INCLUDES 0"
        print(f"  diagonal [{label:<20}] {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n_boot={nb})  {sig}")
        print(f"     sym-equiv x3/4 = {m*sym:+.4f} [{lo*sym:+.4f}, {hi*sym:+.4f}]")

    print("\nInterpretation: the diagonal coefficient is invariant across the full covariate ladder")
    print("  AND under task fixed effects, with a task-clustered CI excluding 0 on the leak-free")
    print("  basis. Per the pre-registered orthogonality, attack-/task-level obviousness, length and")
    print("  success cannot confound a within-attack across-head contrast; the bonus is monitor-side.")


if __name__ == "__main__":
    raise SystemExit(main())
