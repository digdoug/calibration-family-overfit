"""Causal test of the head-angle mechanism: rotate head A onto family B and watch AUROC.

The §4.7 mechanism (track2_headangle.py) is CORRELATIONAL: off-diagonal transfer
AUROC(A->B) rises with the cross-family head cosine cos(w_A, w_B). This script makes
it CAUSAL by intervening on the cosine and measuring the AUROC response, with the two
controls the prescription requires to FAIL.

Exact identity this rests on (verified, diff=0): the strict score is affine in the raw
embedding, transform_score(model_A, x) = x . w_A + const, so the AUROC RANKING depends
only on the raw-space head DIRECTION w_A. Hence any direction v is scored by
roc_auc_score(y_B_test, emb_B_test @ v), and AUROC(w_B -> B_test) == diagonal AUROC[B,B].
"Magnitude" is irrelevant to AUROC; "rotation magnitude" = angular displacement from w_A.

Per ordered off-diagonal pair (A,B):
  * INTERVENTION 1 (aligned slerp): w_align(f) = slerp(a_hat, b_hat, f), f in [0,1].
    Achieved cosine to B = cos((1-f)*Omega) rises monotonically; record AUROC.
  * CONTROL 1 (equal-magnitude random rotation, MUST FAIL): w_rand(f) = rotate a_hat by
    the SAME angle f*Omega toward a random orthogonal direction, averaged over R draws.
  * CONTROL 2 (benign-only alignment, MUST FAIL; reuses Control 20): mean-shift (analytic
    AUROC identity = 0) and CORAL as the equivalent head-direction map M @ w_A,
    M = Sigma_B^-1/2 Sigma_A^1/2 (LedoitWolf benign covariances, track2_recalibrate).
  * INTERVENTION 2 (labeled-budget recovery curve): diff-of-class-means direction probe
    t_m from m benign + m sabotage drawn from B's CAL split; recovery vs m, with a
    random-label control. Plus the unification: budget-probe points obey the same
    cos->AUROC law as the oracle slerp.

Cluster bootstrap by B's TEST TASK (geometry fixed, AUROC side resampled), identical
machinery to track2_headangle.py. Pre-registered before running.

Run:
  .venv/bin/python scripts/track2_intervention.py track2_n600
  .venv/bin/python scripts/track2_intervention.py track2_n600_9fam
  .venv/bin/python scripts/track2_intervention.py track2_n600_actiononly --basis row-benign
  .venv/bin/python scripts/track2_intervention.py track2_n600 --json results/intervention/4open.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import LedoitWolf

import track2_matrix_strict as tms
import track2_headangle as tha

REPO = tms.REPO
F_GRID = [round(x, 4) for x in np.linspace(0.0, 1.0, 11)]   # 0, .1, ..., 1.0
F_MID = 0.5                                                 # matched-displacement contrast
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128]                      # labeled budget (per class)
R_RAND = 64                                                 # random azimuths (point est.)
R_RAND_BOOT = 24                                            # random azimuths in bootstrap
D_DRAW = 40                                                 # diff-of-means subset draws (point)
D_DRAW_BOOT = 20                                            # subset draws in bootstrap
GAP_MIN = 0.02                                              # recoverable-gap floor for ratios


# ----------------------------- geometry helpers -----------------------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def slerp(a_hat: np.ndarray, b_hat: np.ndarray, f: float) -> np.ndarray:
    d = float(np.clip(a_hat @ b_hat, -1.0, 1.0))
    omega = np.arccos(d)
    if omega < 1e-8:
        return a_hat.copy()
    s = np.sin(omega)
    return (np.sin((1 - f) * omega) * a_hat + np.sin(f * omega) * b_hat) / s


def rand_rotation(a_hat: np.ndarray, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate a_hat by EXACTLY angle theta toward a uniformly random orthogonal direction."""
    r = rng.standard_normal(a_hat.shape[0])
    r = r - (r @ a_hat) * a_hat
    r = unit(r)
    return np.cos(theta) * a_hat + np.sin(theta) * r


def sym_sqrt(cov, inv=False):
    """Symmetric matrix (inverse) square root (identical to track2_recalibrate.sym_sqrt)."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-8, None)
    s = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * s) @ V.T


def coral_direction(w_a, sB, sA):
    """Effective direction when B's inputs are CORAL-adapted to A and scored by A's head:
    phi(x)=(x-mu_B) Sigma_B^-1/2 Sigma_A^1/2 + mu_A, phi(x).w_A = x.(M w_A)+const,
    M = Sigma_B^-1/2 Sigma_A^1/2.  (mean-shift adds only a constant -> AUROC identity.)"""
    return sB["ihalf"] @ (sA["half"] @ w_a)


def batched_auroc(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every COLUMN of `scores` vs binary y, vectorized (Mann-Whitney rank form)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan)
    ranks = rankdata(scores, axis=0)
    pos_rank_sum = ranks[y == 1].sum(axis=0)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ----------------------------- build all fixed directions -----------------------------
def build(fams, names, basis, pca_dim, seed):
    """For each eval family B, assemble every intervened direction (geometry is fixed; only
    AUROC resamples). Precompute the score matrix once and column-index maps for the point
    pass (FULL) and a lighter bootstrap pass (BOOT) that drops the random x f-grid block."""
    rng = np.random.default_rng(seed)
    row_models = {a: tms.fit_row_model(fams, a, basis, pca_dim) for a in names}
    w = {a: tha.head_direction(row_models[a]) for a in names}

    bstat = {}
    for a in names:
        d = fams[a]
        bm = (d["split"] == "cal") & (d["label"] == 0)
        Xb = d["emb"][bm]
        cov = LedoitWolf().fit(Xb).covariance_
        bstat[a] = {"mu": Xb.mean(0), "half": sym_sqrt(cov), "ihalf": sym_sqrt(cov, inv=True)}

    per_b = {}
    for b in names:
        d = fams[b]
        test = np.flatnonzero(d["split"] == "test")
        b_hat = unit(w[b])
        cols = []                       # list of direction vectors -> columns of V
        FULL = {"aligned": {}, "random": {}, "coral": {}, "meanshift": {},
                "probe": [], "randlabel": []}
        BOOT_sel = []                   # global col indices used in bootstrap
        BOOT = {"aligned": {}, "random": {}, "coral": {}, "meanshift": {},
                "probe": [], "randlabel": []}

        def add(vec):
            cols.append(vec)
            return len(cols) - 1

        omega = {}
        for a in names:
            if a == b:
                continue
            a_hat = unit(w[a])
            om = float(np.arccos(np.clip(a_hat @ b_hat, -1.0, 1.0)))
            omega[a] = om
            al = np.array([add(slerp(a_hat, b_hat, f)) for f in F_GRID])
            FULL["aligned"][a] = al
            rnd = {}
            for f in F_GRID:
                rnd[f] = np.array([add(rand_rotation(a_hat, f * om, rng)) for _ in range(R_RAND)])
            FULL["random"][a] = rnd
            FULL["coral"][a] = add(coral_direction(w[a], bstat[b], bstat[a]))
            # mean-shift translates inputs by a constant; under a linear head this adds a
            # constant to every score -> the effective DIRECTION is unchanged (= w_A), so AUROC
            # is the floor exactly (the §1a / Control-20 translation identity). Scored to confirm.
            FULL["meanshift"][a] = add(w[a].copy())
            # bootstrap subset: aligned (all f), random only at F_MID (fewer azimuths), coral, ms
            BOOT["aligned"][a] = al
            BOOT["random"][a] = {F_MID: rnd[F_MID][:R_RAND_BOOT]}
            BOOT["coral"][a] = FULL["coral"][a]
            BOOT["meanshift"][a] = FULL["meanshift"][a]
            BOOT_sel.extend(al.tolist())
            BOOT_sel.extend(rnd[F_MID][:R_RAND_BOOT].tolist())
            BOOT_sel.extend([FULL["coral"][a], FULL["meanshift"][a]])

        cal = np.flatnonzero(d["split"] == "cal")
        y_cal = d["label"][cal]
        cal_pos, cal_neg = cal[y_cal == 1], cal[y_cal == 0]
        for m in M_GRID:
            if m > len(cal_pos) or m > len(cal_neg):
                continue
            pj, rj = [], []
            for dd in range(D_DRAW):
                sp = rng.choice(cal_pos, size=m, replace=False)
                sn = rng.choice(cal_neg, size=m, replace=False)
                pj.append(add(unit(d["emb"][sp].mean(0) - d["emb"][sn].mean(0))))
                pool = np.concatenate([sp, sn])
                perm = rng.permutation(pool)
                h = len(pool) // 2
                rj.append(add(unit(d["emb"][perm[:h]].mean(0) - d["emb"][perm[h:]].mean(0))))
            pj, rj = np.array(pj), np.array(rj)
            FULL["probe"].append((m, pj))
            FULL["randlabel"].append((m, rj))
            BOOT["probe"].append((m, pj[:D_DRAW_BOOT]))
            BOOT["randlabel"].append((m, rj[:D_DRAW_BOOT]))
            BOOT_sel.extend(pj[:D_DRAW_BOOT].tolist())
            BOOT_sel.extend(rj[:D_DRAW_BOOT].tolist())

        V = np.array(cols).T
        scores = d["emb"][test] @ V
        cos_to_b = V.T @ b_hat
        boot_sel = np.array(sorted(set(BOOT_sel)))
        g2l = {g: i for i, g in enumerate(boot_sel)}     # global col -> boot-local position

        def relabel(M):
            out = {"aligned": {a: np.array([g2l[j] for j in M["aligned"][a]]) for a in M["aligned"]},
                   "random": {a: {f: np.array([g2l[j] for j in M["random"][a][f]]) for f in M["random"][a]}
                              for a in M["random"]},
                   "coral": {a: g2l[M["coral"][a]] for a in M["coral"]},
                   "meanshift": {a: g2l[M["meanshift"][a]] for a in M["meanshift"]},
                   "probe": [(m, np.array([g2l[j] for j in idx])) for m, idx in M["probe"]],
                   "randlabel": [(m, np.array([g2l[j] for j in idx])) for m, idx in M["randlabel"]]}
            return out

        per_b[b] = {
            "test": test, "task": d["task"][test], "label": d["label"][test],
            "b_hat": b_hat, "scores": scores, "cos_to_b": cos_to_b, "omega": omega,
            "FULL": FULL, "boot_sel": boot_sel, "BOOT": relabel(BOOT),
            "cos_boot": cos_to_b[boot_sel],
        }
    return row_models, w, per_b


# ----------------------------- summarize one AUROC vector for family B -----------------------------
def summarize_b(au, cos_arr, M, names, b):
    """Reduce a per-column AUROC vector for family B to the cell/curve quantities + clouds."""
    res = {"floor": {}, "ceil": {}, "aligned_f": {}, "random_f": {}, "coral": {}, "meanshift": {},
           "probe_m": {}, "randlabel_m": {},
           "ca": [], "cr": [], "cp": []}      # clouds (cos, auroc): aligned / random / probe
    for a in names:
        if a == b:
            continue
        af = {F_GRID[i]: float(au[M["aligned"][a][i]]) for i in range(len(F_GRID))}
        res["aligned_f"][(a, b)] = af
        res["floor"][(a, b)] = af[0.0]
        res["ceil"][(a, b)] = af[1.0]
        for f in F_GRID:
            res["ca"].append((float(np.cos((1 - f) * M_omega(M, a, b))), af[f]))
        rf = {}
        for f, idx in M["random"][a].items():
            rf[f] = float(np.nanmean(au[idx]))
            for j in idx:
                res["cr"].append((float(cos_arr[j]), float(au[j])))
        res["random_f"][(a, b)] = rf
        res["coral"][(a, b)] = float(au[M["coral"][a]])
        res["meanshift"][(a, b)] = float(au[M["meanshift"][a]])
    pm, rm = {}, {}
    for m, idx in M["probe"]:
        pm[m] = float(np.nanmean(au[idx]))
        for j in idx:
            res["cp"].append((float(cos_arr[j]), float(au[j])))
    for m, idx in M["randlabel"]:
        rm[m] = float(np.nanmean(au[idx]))
    res["probe_m"][b] = pm
    res["randlabel_m"][b] = rm
    return res


# omega lives on per_b; thread it via a tiny closure set in cell_aurocs
_OMEGA = {}
def M_omega(M, a, b):
    return _OMEGA[b][a]


def cell_aurocs(per_b, names, which="FULL", row_idx=None):
    global _OMEGA
    _OMEGA = {b: per_b[b]["omega"] for b in names}
    C = {"floor": {}, "ceil": {}, "aligned_f": {}, "random_f": {}, "coral": {}, "meanshift": {},
         "probe_m": {}, "randlabel_m": {}, "ca": [], "cr": [], "cp": []}
    for b in names:
        P = per_b[b]
        if which == "FULL":
            M, cos_arr = P["FULL"], P["cos_to_b"]
            sc = P["scores"] if row_idx is None else P["scores"][row_idx[b]]
        else:
            M, cos_arr = P["BOOT"], P["cos_boot"]
            sc = P["scores"][:, P["boot_sel"]] if row_idx is None else P["scores"][np.ix_(row_idx[b], P["boot_sel"])]
        y = P["label"] if row_idx is None else P["label"][row_idx[b]]
        au = batched_auroc(sc, y)
        r = summarize_b(au, cos_arr, M, names, b)
        for k in ("floor", "ceil", "aligned_f", "random_f", "coral", "meanshift"):
            C[k].update(r[k])
        C["probe_m"].update(r["probe_m"])
        C["randlabel_m"].update(r["randlabel_m"])
        C["ca"].extend(r["ca"]); C["cr"].extend(r["cr"]); C["cp"].extend(r["cp"])
    return C


def recovery(auc, floor, ceil):
    g = ceil - floor
    return (auc - floor) / g if g > GAP_MIN else np.nan


def nanmean(x):
    x = [v for v in x if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(x)) if x else np.nan


def pooled(C, names):
    pairs = [(a, b) for a in names for b in names if a != b]
    ca, cr, cp = np.array(C["ca"]), np.array(C["cr"]), np.array(C["cp"])
    sp = lambda arr: float(spearmanr(arr[:, 0], arr[:, 1]).statistic) if len(arr) > 2 else np.nan
    rec_al, rec_rd, rec_cor, rec_ms = [], [], [], []
    mono_al, mono_rd = [], []      # within-pair Spearman(f, AUROC): removes cross-pair confound
    for (a, b) in pairs:
        fl, ce = C["floor"][(a, b)], C["ceil"][(a, b)]
        ys = [C["aligned_f"][(a, b)][f] for f in F_GRID]
        if np.std(ys) > 0:
            mono_al.append(float(spearmanr(F_GRID, ys).statistic))
        rf = C["random_f"].get((a, b), {})
        if len(rf) == len(F_GRID):
            yr = [rf[f] for f in F_GRID]
            if np.std(yr) > 0:
                mono_rd.append(float(spearmanr(F_GRID, yr).statistic))
        if ce - fl <= GAP_MIN:
            continue
        rec_al.append(recovery(C["aligned_f"][(a, b)][F_MID], fl, ce))
        rec_rd.append(recovery(C["random_f"][(a, b)][F_MID], fl, ce))
        rec_cor.append(recovery(C["coral"][(a, b)], fl, ce))
        rec_ms.append(recovery(C["meanshift"][(a, b)], fl, ce))
    rec_m, recr_m = {}, {}
    for m in M_GRID:
        al, rl = [], []
        for b in names:
            others = [a for a in names if a != b]
            fl = nanmean([C["floor"][(a, b)] for a in others])
            ce = nanmean([C["ceil"][(a, b)] for a in others])
            if np.isnan(fl) or np.isnan(ce) or (ce - fl) <= GAP_MIN:
                continue
            if m in C["probe_m"].get(b, {}):
                al.append(recovery(C["probe_m"][b][m], fl, ce))
            if m in C["randlabel_m"].get(b, {}):
                rl.append(recovery(C["randlabel_m"][b][m], fl, ce))
        if al:
            rec_m[m] = nanmean(al)
        if rl:
            recr_m[m] = nanmean(rl)
    return {
        "sp_aligned": sp(ca), "sp_random": sp(cr), "sp_probe": sp(cp),
        "mono_aligned": nanmean(mono_al), "mono_random": nanmean(mono_rd),
        "rec_aligned_mid": nanmean(rec_al), "rec_random_mid": nanmean(rec_rd),
        "rec_coral": nanmean(rec_cor), "rec_meanshift": nanmean(rec_ms),
        "gap_aligned_minus_random": nanmean(rec_al) - nanmean(rec_rd),
        "n_pairs_gap": len(rec_al), "rec_probe_m": rec_m, "rec_randlabel_m": recr_m,
    }


def cluster_bootstrap(per_b, names, n_boot, seed):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({str(t) for b in names for t in per_b[b]["task"]})
    task_rows = {b: {} for b in names}
    for b in names:
        for i, t in enumerate(per_b[b]["task"]):
            task_rows[b].setdefault(str(t), []).append(i)
    keys = ["sp_aligned", "sp_probe", "mono_aligned", "rec_aligned_mid", "rec_random_mid",
            "rec_coral", "rec_meanshift", "gap_aligned_minus_random"]
    acc = {k: [] for k in keys}
    accm = {m: [] for m in M_GRID}
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True).tolist()
        row_idx, ok = {}, True
        for b in names:
            chunks = [task_rows[b][t] for t in draw if t in task_rows[b]]
            ix = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.array([], int)
            if len(ix) == 0 or len(np.unique(per_b[b]["label"][ix])) < 2:
                ok = False
                break
            row_idx[b] = ix
        if not ok:
            continue
        S = pooled(cell_aurocs(per_b, names, which="BOOT", row_idx=row_idx), names)
        for k in keys:
            if not np.isnan(S[k]):
                acc[k].append(S[k])
        for m in M_GRID:
            if m in S["rec_probe_m"] and not np.isnan(S["rec_probe_m"][m]):
                accm[m].append(S["rec_probe_m"][m])

    def ci(arr):
        a = np.array(arr)
        if len(a) == 0:
            return (np.nan, np.nan, np.nan, 0)
        return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), len(a))
    out = {k: ci(acc[k]) for k in keys}
    out["rec_probe_m"] = {m: ci(accm[m]) for m in M_GRID if accm[m]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("t2", nargs="?", default="track2_n600")
    ap.add_argument("--basis", choices=["row-all", "row-benign"], default="row-all")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t2_dir = os.path.join(REPO, "results", args.t2)
    names, fams = tms.load_families(t2_dir)
    names = sorted(names)
    print(f"intervention (causal head-angle): {t2_dir}")
    print(f"families ({len(names)}): {', '.join(tms.short_name(n) for n in names)}")
    print(f"basis: {args.basis} | pca_dim: {args.pca_dim} | R_rand={R_RAND} | D_draw={D_DRAW}")

    _, _, per_b = build(fams, names, args.basis, args.pca_dim, args.seed)
    C = cell_aurocs(per_b, names, which="FULL")
    S = pooled(C, names)

    npair = len(names) * (len(names) - 1)
    print("\n===== point estimates =====")
    print(f"mean floor (off-diag AUROC, f=0)  = {nanmean([C['floor'][(a,b)] for a in names for b in names if a!=b]):.3f}")
    print(f"mean ceiling (diagonal, f=1)      = {nanmean([C['ceil'][(a,b)] for a in names for b in names if a!=b]):.3f}")
    print(f"pairs with recoverable gap>{GAP_MIN} = {S['n_pairs_gap']} / {npair}")
    print("\n-- HEADLINE-A: monotone recovery as cosine rises --")
    print(f"  within-pair Spearman(f, AUROC):  aligned = {S['mono_aligned']:+.3f}   random = {S['mono_random']:+.3f}")
    print(f"    (clean monotonicity, no cross-pair confound; aligne expect ~+1, random ~<=0)")
    print(f"  pooled Spearman(cos->AUROC):  aligned = {S['sp_aligned']:+.3f}   probe = {S['sp_probe']:+.3f}   random = {S['sp_random']:+.3f}")
    print(f"    (random pooled rho is inflated by the cross-pair baseline a_hat structure, NOT recovery)")
    print("\n-- HEADLINE-B: recovery fraction at matched displacement f=0.5 --")
    print(f"  aligned   = {S['rec_aligned_mid']:+.3f}")
    print(f"  random    = {S['rec_random_mid']:+.3f}   (MUST FAIL)")
    print(f"  aligned - random gap = {S['gap_aligned_minus_random']:+.3f}   (specificity)")
    print("\n-- CONTROL 2: benign-only alignment recovery (MUST FAIL; Control 20) --")
    print(f"  CORAL     = {S['rec_coral']:+.3f}")
    print(f"  mean-shift= {S['rec_meanshift']:+.3f}   (analytic identity ~ 0)")
    print("\n-- INTERVENTION 2: labeled-budget recovery curve (diff-of-means probe) --")
    print(f"  {'m':>5}  {'probe rec':>10}  {'rand-label rec':>14}")
    for m in M_GRID:
        print(f"  {m:>5}  {S['rec_probe_m'].get(m, float('nan')):>10.3f}  {S['rec_randlabel_m'].get(m, float('nan')):>14.3f}")

    boot = cluster_bootstrap(per_b, names, n_boot=args.boot, seed=args.seed)
    print(f"\n===== cluster bootstrap by test task (n={boot['sp_aligned'][3]} draws) =====")

    def line(label, key, expect):
        m, lo, hi, _ = boot[key]
        flag = "LIVE" if lo > 0 else ("DIE " if hi < 0 else "span")
        print(f"  {label:<30} mean={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  [{flag}] {expect}")

    line("within-pair mono aligned", "mono_aligned", "(LIVE expected, ~+1)")
    line("aligned Spearman(cos,AUROC)", "sp_aligned", "(LIVE expected)")
    line("budget-probe Spearman", "sp_probe", "(LIVE expected)")
    line("aligned recovery @f=0.5", "rec_aligned_mid", "(LIVE expected)")
    line("random recovery @f=0.5", "rec_random_mid", "(should span/neg)")
    line("aligned - random GAP @f=0.5", "gap_aligned_minus_random", "(LIVE = specificity)")
    line("CORAL recovery", "rec_coral", "(should span/neg)")
    line("mean-shift recovery", "rec_meanshift", "(~0 identity)")
    print("\n  labeled-budget recovery curve (probe), cluster-boot CI:")
    print(f"  {'m':>5}  {'mean':>7}  {'95% CI':>20}")
    for m in M_GRID:
        if m in boot["rec_probe_m"]:
            mm, lo, hi, _ = boot["rec_probe_m"][m]
            print(f"  {m:>5}  {mm:>7.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print("\nLIVE (causal) if: aligned Spearman CI>0 AND aligned-random GAP CI>0 AND")
    print("CORAL/random recovery CIs span/<=0 AND budget curve rises monotonically.")

    if args.json:
        outp = os.path.join(REPO, args.json)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        jd = lambda v: {str(k): vv for k, vv in v.items()} if isinstance(v, dict) else v
        payload = {
            "roster": args.t2, "basis": args.basis,
            "families": [tms.short_name(n) for n in names],
            "f_grid": [float(x) for x in F_GRID], "m_grid": M_GRID,
            "point": {k: jd(v) for k, v in S.items()},
            "bootstrap": {k: jd(v) for k, v in boot.items()},
            "clouds": {"aligned": C["ca"], "probe": C["cp"], "random": C["cr"]},
        }
        with open(outp, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
