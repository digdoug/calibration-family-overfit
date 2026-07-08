"""Control 14c: conditional (mixed-effects-style) confirmatory model of the diagonal bonus.

Pre-registered before running. The headline interaction is established by (i) the two-way
ANOVA-style decomposition of the AUROC matrix and (ii) task-clustered bootstrap CIs. This adds
the reviewer-proof SINGLE conditional model: a placement-value OLS that adjusts for a CONTINUOUS
obviousness covariate (the vendor-disjoint Control-14b Haiku grade) and clusters by task inside
the estimator, then bootstraps by task.

Placement value v_iA = mean_j 1[s_iA > s_jB] (+0.5 ties) over family-B test BENIGN scores under
head A. By construction mean over attacks of v_iA = AUROC(A,B), so OLS on placement with
policy_family + monitor_family dummies is a faithful, scale-free decomposition of the AUROC
matrix; the `diagonal` coefficient = the diagonal detection bonus net of additive row/col main
effects. Adding blatancy_centered conditions that bonus on the independent obviousness grade.

Zero new deps: numpy + the project's own _clusterboot. No statsmodels (frozen venv).

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_mixedeffects.py [--boot N]
     (requires results/blatancy2/grades.json complete for the four open families)
"""
from __future__ import annotations
import os, json, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _clusterboot import task_rows, draw_index
from track2_blatancy import split_of, diag_interaction, extract_suspicion

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRADES = os.path.join(REPO, "results", "blatancy2", "grades.json")
FAMILIES = ["ow_qwen_n600", "ow_mistral_n600", "ow_gemma_n600", "ow_llama_n600"]


def load():
    """Load the four open families with embeddings, labels, tasks, suspicion, Haiku grade."""
    T2 = os.path.join(REPO, "results", "track2_n600")
    grades = json.load(open(GRADES))
    fams = {}
    for name in FAMILIES:
        d = np.load(os.path.join(T2, name + ".npz"), allow_pickle=True)
        susp, labs, tasks = extract_suspicion(os.path.join(REPO, "results", f"W2_{name}"))
        assert len(susp) == len(d["task_id"]) and np.all(tasks == d["task_id"]) \
            and np.all(labs == d["label"]), f"{name}: suspicion not aligned to npz rows"
        gmap = grades.get(name, {})
        blat2 = np.array([gmap.get(str(t), np.nan) for t in d["task_id"]], dtype=float)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"],
                      "susp": susp, "blat2": blat2,
                      "split": np.array([split_of(t) for t in d["task_id"]])}
    return fams


def fit_heads(fams):
    names = list(fams)
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    return heads


def placement(att_scores, ben_scores):
    """v_i = mean_j 1[att_i > ben_j] + 0.5 1[==], vectorised. mean_i v_i == AUROC."""
    if len(ben_scores) == 0:
        return np.full(len(att_scores), np.nan)
    a = att_scores[:, None]; b = ben_scores[None, :]
    return (np.sum(a > b, axis=1) + 0.5 * np.sum(a == b, axis=1)) / len(ben_scores)


def build_rows(fams, names, heads, att_idx, ben_idx, grade_center):
    """Assemble the long design from given per-family attack/benign row-index arrays.

    Returns X (design), y (placement), col_names, task_ids(for attacks, tiled over heads).
    Columns: intercept, policy_family dummies(drop first), monitor_family dummies(drop first),
             diagonal, blatancy_centered. Rows = (attack i in B) x (head A).
    """
    # per (B, A) benign scores depend only on head A and family B's benign rows
    ben_scores = {(A, B): heads[A].decision_function(fams[B]["p"][ben_idx[B]])
                  for A in names for B in names}
    pol_levels = names[1:]   # drop-first dummy reference = names[0]
    mon_levels = names[1:]
    rows_y, rows_X, rows_task = [], [], []
    for B in names:
        d = fams[B]
        ai = att_idx[B]
        if len(ai) == 0:
            continue
        g = d["blat2"][ai]
        m = ~np.isnan(g)
        ai = ai[m]; g = g[m]
        if len(ai) == 0:
            continue
        att_p = d["p"][ai]
        gc = g - grade_center
        tasks_B = d["task"][ai]
        for A in names:
            v = placement(heads[A].decision_function(att_p), ben_scores[(A, B)])
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
            rows_y.append(v); rows_X.append(X); rows_task.append(tasks_B)
    y = np.concatenate(rows_y); X = np.vstack(rows_X); task = np.concatenate(rows_task)
    cols = (["intercept"] + [f"pol[{l}]" for l in pol_levels]
            + [f"mon[{l}]" for l in mon_levels] + ["diagonal", "blatancy_c"])
    return X, y, cols, task


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def point_estimates(fams, names, heads, grade_center):
    """Full-sample OLS coefficients: with blatancy, without blatancy, with diag:blat term."""
    test = {B: (fams[B]["split"] == "test") for B in names}
    att_idx = {B: np.flatnonzero(test[B] & (fams[B]["label"] == 1)) for B in names}
    ben_idx = {B: np.flatnonzero(test[B] & (fams[B]["label"] == 0)) for B in names}
    X, y, cols, _ = build_rows(fams, names, heads, att_idx, ben_idx, grade_center)
    di = cols.index("diagonal"); bi = cols.index("blatancy_c")
    beta_full = ols(X, y)
    # no-blatancy: drop the blatancy column
    Xnb = np.delete(X, bi, axis=1)
    beta_nb = ols(Xnb, y)
    # with diagonal:blatancy interaction term appended
    Xint = np.hstack([X, (X[:, di] * X[:, bi])[:, None]])
    beta_int = ols(Xint, y)
    return {
        "cols": cols, "n_rows": len(y),
        "diag_with_blat": float(beta_full[di]),
        "blat_coef": float(beta_full[bi]),
        "diag_no_blat": float(beta_nb[di]),  # validity check vs matrix interaction
        "diag_in_intmodel": float(beta_int[di]),
        "diagXblat": float(beta_int[-1]),
    }


def matrix_interaction(fams, names, heads):
    """The published-style cell-AUROC interaction on graded test attacks (validity target)."""
    M = {}
    for A in names:
        for B in names:
            d = fams[B]
            te = (d["split"] == "test")
            att = te & (d["label"] == 1) & ~np.isnan(d["blat2"])
            ben = te & (d["label"] == 0)
            idx = np.flatnonzero(att | ben)
            y = d["label"][idx]
            M[(A, B)] = roc_auc_score(y, heads[A].decision_function(d["p"][idx]))
    diag = np.mean([M[(x, x)] for x in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    return diag_interaction(M, names), diag - off, M


def cluster_boot(fams, names, heads, grade_center, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values()
                         for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    att_rows = {B: task_rows(fams[B]["task"],
                             (fams[B]["split"] == "test") & (fams[B]["label"] == 1)
                             & ~np.isnan(fams[B]["blat2"]), test_tasks) for B in names}
    ben_rows = {B: task_rows(fams[B]["task"],
                             (fams[B]["split"] == "test") & (fams[B]["label"] == 0),
                             test_tasks) for B in names}
    dwb, dnb, bc = [], [], []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        att_idx = {B: draw_index(att_rows[B], draw) for B in names}
        ben_idx = {B: draw_index(ben_rows[B], draw) for B in names}
        if any(len(att_idx[B]) == 0 or len(ben_idx[B]) == 0 for B in names):
            continue
        X, y, cols, _ = build_rows(fams, names, heads, att_idx, ben_idx, grade_center)
        di = cols.index("diagonal"); bi = cols.index("blatancy_c")
        b = ols(X, y)
        dwb.append(b[di]); bc.append(b[bi])
        dnb.append(ols(np.delete(X, bi, axis=1), y)[di])
    f = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"n_boot": len(dwb), "diag_with_blat": f(dwb), "diag_no_blat": f(dnb),
            "blat_coef": f(bc)}


def main():
    args = sys.argv[1:]
    n_boot = 1000
    if "--boot" in args:
        i = args.index("--boot"); n_boot = int(args[i + 1]); del args[i:i + 2]
    fams = load(); names = list(fams)
    heads = fit_heads(fams)

    # center blatancy on the pooled graded test-attack mean (fixed reference)
    pooled_g = np.concatenate([fams[B]["blat2"][(fams[B]["split"] == "test")
                               & (fams[B]["label"] == 1) & ~np.isnan(fams[B]["blat2"])]
                               for B in names])
    gc = float(np.mean(pooled_g))
    print(f"families={names}  graded test attacks pooled n={len(pooled_g)}  "
          f"grade mean(center)={gc:.3f} sd={pooled_g.std():.3f}")

    mi, raw, M = matrix_interaction(fams, names, heads)
    print(f"\nVALIDITY TARGET (cell-AUROC, graded test attacks): "
          f"diag_interaction={mi:+.4f}  raw diag-off={raw:+.4f}")
    short = lambda n: n.replace("ow_", "").replace("_n600", "")
    print("AUROC matrix (graded test attacks):")
    print("            " + "  ".join(f"{short(n):>8}" for n in names))
    for A in names:
        print(f"{short(A):>10}  " + "  ".join(f"{M[(A,B)]:>8.3f}" for B in names))

    # A single diagonal-indicator coefficient in a balanced N-family additive model equals
    # (N/(N-1)) x the symmetric two-way interaction; report both so the number is directly
    # comparable to the paper's symmetric +0.110 headline.
    N = len(names); sym = (N - 1) / N
    pe = point_estimates(fams, names, heads, gc)
    print(f"\nPlacement-value OLS point estimates (n_rows={pe['n_rows']}):")
    print(f"  diagonal coef, NO blatancy   = {pe['diag_no_blat']:+.4f}   "
          f"(sym-equiv x{(N-1)}/{N} = {pe['diag_no_blat']*sym:+.4f} vs diag_interaction {mi:+.4f})")
    print(f"  diagonal coef, WITH blatancy = {pe['diag_with_blat']:+.4f}   "
          f"(sym-equiv {pe['diag_with_blat']*sym:+.4f})")
    print(f"  blatancy_c coef              = {pe['blat_coef']:+.5f}  "
          f"(per +1 grade pt; rows are attacks, placement in [0,1])")
    print(f"  diagonal (in diag:blat model)= {pe['diag_in_intmodel']:+.4f}   "
          f"diagonal:blatancy = {pe['diagXblat']:+.5f}")

    vc = abs(pe["diag_no_blat"] * sym - mi)
    print(f"\nVALIDITY: |no-blat diagonal x{(N-1)}/{N} - diag_interaction| = {vc:.4f}  "
          f"({'PASS' if vc < 0.02 else 'CHECK'} <0.02)")

    print(f"\nCluster bootstrap by task (n_boot target {n_boot}) ...")
    cb = cluster_boot(fams, names, heads, gc, n_boot)
    def show(k, lab):
        m, lo, hi = cb[k]
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI INCLUDES 0"
        print(f"  {lab:<28} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   {sig}")
    print(f"  (n_boot kept = {cb['n_boot']})")
    show("diag_no_blat",   "diagonal, no blatancy")
    show("diag_with_blat", "diagonal, adj. blatancy")
    show("blat_coef",      "blatancy_c slope")

    print("\nInterpretation: the diagonal coefficient adj. blatancy with a task-clustered 95% CI")
    print("  excluding 0, ~unchanged vs no-blatancy, = the diagonal detection bonus survives a")
    print("  single conditional model adjusting for the independent vendor-disjoint obviousness")
    print("  grade and clustering by task. Confirms Control 14b in regression form.")


if __name__ == "__main__":
    raise SystemExit(main())
