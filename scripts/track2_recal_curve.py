"""Control 21 — the recalibration learning curve.

The constructive capstone of the Control 18-19-20 deployment quartet. Control 20
closed the quartet on "only labeled sabotage from the faced lineage fixes the
cross-family gap." This asks the operational follow-up: HOW MUCH labeled B
sabotage do you need, and is the off-lineage A-monitor a useful WARM START?

For each ordered off-diagonal pair (A,B): the A-head is the off-lineage monitor.
The deployer elicits + grades k attack + k benign B transcripts (drawn from B's
CAL split) and recalibrates three ways, then scores B's TEST split:
  scratch(k)      = fresh head on the 2k B examples only
  augment(k)      = head on A's full cal + 2k B examples (unweighted)
  augment_bal(k)  = same pool, B upweighted to equal A's total weight (warm start)
Asymptotes from the headline matrix: floor=AUROC[A,B] (k=0), ceiling=AUROC[B,B].

Pooled over off-diagonal cells, seed-averaged over R draws of the k-subset,
cluster-bootstrap CI by task. Pre-registered before running.

Run: .venv-controlarena/bin/python scripts/track2_recal_curve.py [<dir> ...]
"""
from __future__ import annotations
import glob, os, sys, hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_FRAC = 60
OPEN4 = {"gemma", "llama", "mistral", "qwen"}
KS = [0, 5, 10, 20, 40, 80, 150, 300]
NDRAW = 10          # seed-averaged draws of the k-subset
NBOOT = 1000
C = 1.0             # same solver/C as the matrix head


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def short(name):
    return (name.replace("frontier_", "").replace("ow_", "").replace("_n600", "")
            .replace("gemini_flash", "gemini"))


def load_dir(t2, keep=None):
    fams = {}
    for f in sorted(glob.glob(os.path.join(t2, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        name = short(os.path.basename(f)[:-4])
        if keep is not None and name not in keep:
            continue
        sp = np.array([split_of(t) for t in d["task_id"]])
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"], "split": sp}
    return fams


def build(fams):
    """Pooled-cal PCA-50 identical to track2_matrix; per-family frozen heads.
    Returns names, the projected per-family arrays, and the diagonal heads."""
    names = list(fams)
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = fams[A]
        cal = d["split"] == "cal"
        heads[A] = LogisticRegression(max_iter=2000, C=C).fit(d["p"][cal], d["label"][cal])
    return names, heads


def fit_head(X, y, w=None):
    return LogisticRegression(max_iter=2000, C=C).fit(X, y, sample_weight=w)


def analyze(t2, keep=None):
    fams = load_dir(t2, keep=keep)
    names, heads = build(fams)
    n = len(names)
    print(f"\n=== {os.path.basename(t2.rstrip('/'))}  ({n} families) ===")
    print("families:", names)

    # per-family cal pools (by label) and test rows
    cal = {}
    test = {}
    for fam in names:
        d = fams[fam]
        c = d["split"] == "cal"; m = d["split"] == "test"
        cal[fam] = {"atk": d["p"][c & (d["label"] == 1)],
                    "ben": d["p"][c & (d["label"] == 0)],
                    "Xall": d["p"][c], "yall": d["label"][c]}
        test[fam] = {"p": d["p"][m], "lab": d["label"][m], "task": d["task"][m]}

    # asymptotes from the headline matrix
    diag = {A: roc_auc_score(test[A]["lab"], heads[A].decision_function(test[A]["p"]))
            for A in names}
    pairs = [(A, B) for A in names for B in names if A != B]
    floor_cell = {(A, B): roc_auc_score(test[B]["lab"],
                                        heads[A].decision_function(test[B]["p"]))
                  for (A, B) in pairs}
    mean_diag = float(np.mean([diag[A] for A in names]))
    mean_floor = float(np.mean([floor_cell[ab] for ab in pairs]))
    print(f"ceiling (mean diagonal)      = {mean_diag:.3f}")
    print(f"floor   (mean off-diag, k=0) = {mean_floor:.3f}   gap = {mean_diag - mean_floor:+.3f}")

    # ----- the learning curve: per-strategy per-k pooled AUROC, seed-averaged
    # cache per (strategy,k,draw) the per-cell test scores so the bootstrap can
    # reuse them. scores[strat][k][(A,B)] = list over draws of decision_function
    # arrays aligned to test[B] rows.
    strategies = ["scratch", "augment", "augment_bal"]
    scores = {s: {k: {} for k in KS} for s in strategies}

    for (A, B) in pairs:
        Xt = test[B]["p"]
        Aall_X, Aall_y = cal[A]["Xall"], cal[A]["yall"]
        wA = float(len(Aall_y))
        for k in KS:
            for s in strategies:
                scores[s][k][(A, B)] = []
            if k == 0:
                # floor: A-head on B (single deterministic "draw")
                df = heads[A].decision_function(Xt)
                for s in strategies:
                    scores[s][0][(A, B)] = [df]
                continue
            for draw in range(NDRAW):
                rng = np.random.default_rng(1000 * draw + 7)
                na, nb = len(cal[B]["atk"]), len(cal[B]["ben"])
                ia = rng.choice(na, size=min(k, na), replace=False)
                ib = rng.choice(nb, size=min(k, nb), replace=False)
                Xb = np.vstack([cal[B]["atk"][ia], cal[B]["ben"][ib]])
                yb = np.concatenate([np.ones(len(ia)), np.zeros(len(ib))])
                # scratch
                try:
                    h = fit_head(Xb, yb)
                    scores["scratch"][k][(A, B)].append(h.decision_function(Xt))
                except ValueError:
                    pass
                # augment (unweighted pool A + B)
                Xp = np.vstack([Aall_X, Xb]); yp = np.concatenate([Aall_y, yb])
                h = fit_head(Xp, yp)
                scores["augment"][k][(A, B)].append(h.decision_function(Xt))
                # augment_bal: B upweighted so sum(wB) == sum(wA)
                wb = wA / len(yb)
                w = np.concatenate([np.ones(len(Aall_y)), np.full(len(yb), wb)])
                h = fit_head(Xp, yp, w=w)
                scores["augment_bal"][k][(A, B)].append(h.decision_function(Xt))

    def pooled_auc(strat, k, draw_reduce=True):
        """Mean over off-diag cells of the seed-averaged cell AUROC."""
        vals = []
        for ab in pairs:
            y = test[ab[1]]["lab"]
            aucs = [roc_auc_score(y, df) for df in scores[strat][k][ab]]
            if aucs:
                vals.append(float(np.mean(aucs)))
        return float(np.mean(vals)) if vals else np.nan

    gap = mean_diag - mean_floor
    print(f"\nlearning curve  (pooled off-diag AUROC; gap closure = (auc-floor)/(ceiling-floor))")
    print(f"  {'k':>4} | " + " ".join(f"{s:>22}" for s in strategies))
    closure = {s: {} for s in strategies}
    for k in KS:
        row = []
        for s in strategies:
            a = pooled_auc(s, k)
            gc = (a - mean_floor) / gap if gap > 1e-9 else float("nan")
            closure[s][k] = gc
            row.append(f"{a:.3f} ({100*gc:5.1f}%)")
        print(f"  {k:>4} | " + " ".join(f"{r:>22}" for r in row))

    def k90(s):
        for k in KS:
            if k > 0 and closure[s][k] >= 0.90:
                return k
        return None
    print("\nk_90 (smallest k reaching >=90% gap closure):")
    for s in strategies:
        print(f"  {s:>14}: {k90(s)}")

    # ----- warm-start lift at small k: augment_bal - scratch, cluster-boot by task
    universe = sorted(set.intersection(*[set(test[B]["task"]) for (_, B) in pairs]))
    # for each cell, map task -> test-row indices (within test[B])
    cell_task_idx = {}
    for (A, B) in pairs:
        tk = test[B]["task"]
        cell_task_idx[(A, B)] = {t: np.where(tk == t)[0] for t in universe}

    def boot_lift(k):
        """Paired cluster-boot over shared task universe of
        pooled(augment_bal) - pooled(scratch) at k. Uses the SAME estimator as
        the learning-curve table: per cell, average the per-DRAW AUROCs (the
        realistic single-recalibration number, NOT a score-ensemble across draws
        which would denoise the high-variance scratch head and bias the contrast)."""
        U = np.array(universe)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(NBOOT):
            drawn = rng.choice(U, size=len(U), replace=True)
            psc, pab = [], []
            for ab in pairs:
                y = test[ab[1]]["lab"]; cti = cell_task_idx[ab]
                idx = np.concatenate([cti[t] for t in drawn])
                yy = y[idx]
                if len(np.unique(yy)) < 2:
                    continue
                sc = [roc_auc_score(yy, df[idx]) for df in scores["scratch"][k][ab]]
                ab_ = [roc_auc_score(yy, df[idx]) for df in scores["augment_bal"][k][ab]]
                if sc:
                    psc.append(float(np.mean(sc)))
                if ab_:
                    pab.append(float(np.mean(ab_)))
            if psc and pab:
                diffs.append(np.mean(pab) - np.mean(psc))
        return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    print("\nwarm-start lift  (augment_bal - scratch), cluster-boot CI by task:")
    print(f"  {'k':>4} {'lift':>9} {'95% CI':>22}   (WARM if CI>0)")
    for k in [5, 10, 20]:
        m, lo, hi = boot_lift(k)
        flag = "WARM" if lo > 0 else ("scratch" if hi < 0 else "ns")
        print(f"  {k:>4} {m:>+9.4f} [{lo:>+7.4f},{hi:>+7.4f}]   {flag}")

    # ----- per-cell floor/ceiling table
    print("\nper-cell floor (k=0) and ceiling (diagonal of A->B is diag[B]):")
    print(f"  {'A->B':>20} {'floor':>6} {'ceil(B)':>8} {'gap':>6}")
    for (A, B) in pairs:
        print(f"  {short(A)+'->'+short(B):>20} {floor_cell[(A,B)]:>6.3f} "
              f"{diag[B]:>8.3f} {diag[B]-floor_cell[(A,B)]:>+6.3f}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or [os.path.join(REPO, "results", "track2_n600")]
    for d in dirs:
        keep = OPEN4 if os.path.basename(d.rstrip("/")) in ("track2_n600", "track2_n600_actiononly") else None
        analyze(d, keep=keep)
