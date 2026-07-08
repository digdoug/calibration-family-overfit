"""Control 23 -- the MULTI-FAMILY CALIBRATION CURVE (s6 constructive defense).

Pre-registered (written before running).

Control 3 tested LOFO calibration at ONE point (pool all N-1=3 others) at 4
families and found a null. At 8 families we sweep K = 1..N-1 pooled calibration
families and draw the curve: how much held-out-family detection do you recover by
calibrating a monitor on a PANEL of K families it never faces at deploy time?

For each held-out target family B:
  diag_B    = B's own-cal head on B's test           (in-distribution ceiling)
  pooledK_B = mean over random size-K subsets S of the other families of
              (head trained on S's pooled cal) scored on B's test
  K=1 averaged over all singletons == Control 3's single-foreign floor.

Two embedding spaces (full + action-only via the dir arg) and two PCA bases:
  primary (matrix-comparable): scaler+PCA on ALL families' pooled cal (Ctrl 3 caveat
           kept: subspace sees B's benign cal, never B's attack labels). Bootstrapped.
  strict  (held-out): scaler+PCA on each training subset's cal only, B fully absent.
           Point estimates only (per-cell PCA refit makes a bootstrap impractical);
           a SHAPE-robustness check on the primary curve.

Cluster bootstrap by task_id on each held-out family's test split, B-stratified.

Run: .venv/bin/python scripts/track2_multifam_cal.py [track2dir]   (default track2_n600_8fam)
"""
from __future__ import annotations
import glob, os, hashlib, sys, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 else "track2_n600_8fam")
CAL_FRAC = 60
PCA_K = 50
MAX_SUBSETS = 24      # cap random size-K subsets per held-out B
N_BOOT = 1000
SHORT = lambda s: (s.replace("frontier_", "").replace("ow_", "").replace("_n600", "")
                   .replace("_n200", "").replace("gemini_flash", "gemini").replace("gpt5", "gpt-5"))


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def load():
    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = SHORT(os.path.basename(f)[:-4])
        d = np.load(f, allow_pickle=True)
        sp = np.array([split_of(t) for t in d["task_id"]])
        fams[name] = {"emb": d["emb"], "label": np.asarray(d["label"]),
                      "task": np.asarray(d["task_id"]), "split": sp}
    return fams


def global_space(fams):
    cal = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    sc = StandardScaler().fit(cal)
    pca = PCA(n_components=PCA_K, random_state=0).fit(sc.transform(cal))
    P = {n: pca.transform(sc.transform(d["emb"])) for n, d in fams.items()}
    return P


def fit_space_on(emb):
    sc = StandardScaler().fit(emb)
    pca = PCA(n_components=PCA_K, random_state=0).fit(sc.transform(emb))
    return lambda e: pca.transform(sc.transform(e))


def head(X, y):
    return LogisticRegression(C=1.0, max_iter=3000).fit(X, y)


def subsets_of(others, K, rng):
    allc = list(itertools.combinations(others, K))
    if len(allc) <= MAX_SUBSETS:
        return allc
    idx = rng.choice(len(allc), size=MAX_SUBSETS, replace=False)
    return [allc[i] for i in idx]


def curve_primary(fams, P):
    """Returns per-B, per-K score vectors on B's test (for bootstrap) + point AUROCs."""
    names = list(fams)
    Ks = list(range(1, len(names)))
    # per-B: test mask, labels, tasks; diag scores; pooledK -> list of subset score-vectors
    store = {}
    for B in names:
        d = fams[B]; te = d["split"] == "test"
        yte = d["label"][te]; tte = d["task"][te]
        cal = d["split"] == "cal"
        diag = head(P[B][cal], d["label"][cal]).decision_function(P[B][te])
        others = [a for a in names if a != B]
        pk = {}
        rng = np.random.default_rng(abs(hash(B)) % (2**32))
        for K in Ks:
            vecs = []
            for S in subsets_of(others, K, rng):
                ox = np.vstack([P[a][fams[a]["split"] == "cal"] for a in S])
                oy = np.concatenate([fams[a]["label"][fams[a]["split"] == "cal"] for a in S])
                vecs.append(head(ox, oy).decision_function(P[B][te]))
            pk[K] = np.array(vecs)   # (n_subsets, n_test)
        store[B] = {"y": yte, "task": tte, "diag": diag, "pk": pk}
    return names, Ks, store


def point_curve(names, Ks, store):
    diag = float(np.mean([roc_auc_score(store[B]["y"], store[B]["diag"]) for B in names]))
    curve = {}
    for K in Ks:
        per_B = []
        for B in names:
            aurocs = [roc_auc_score(store[B]["y"], v) for v in store[B]["pk"][K]]
            per_B.append(np.mean(aurocs))
        curve[K] = float(np.mean(per_B))
    return diag, curve


def boot_curve(names, Ks, store, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    # precompute per-B task clusters
    clusters = {B: [np.where(store[B]["task"] == t)[0] for t in np.unique(store[B]["task"])]
                for B in names}
    diag_b, curve_b = [], {K: [] for K in Ks}
    diff_b = []   # pooled[max] - pooled[1]
    for _ in range(n_boot):
        d_acc, c_acc = [], {K: [] for K in Ks}
        for B in names:
            cl = clusters[B]
            pick = rng.integers(0, len(cl), size=len(cl))
            idx = np.concatenate([cl[i] for i in pick])
            y = store[B]["y"][idx]
            if len(np.unique(y)) < 2:
                continue
            d_acc.append(roc_auc_score(y, store[B]["diag"][idx]))
            for K in Ks:
                c_acc[K].append(np.mean([roc_auc_score(y, v[idx]) for v in store[B]["pk"][K]]))
        if not d_acc:
            continue
        diag_b.append(np.mean(d_acc))
        for K in Ks:
            curve_b[K].append(np.mean(c_acc[K]))
        diff_b.append(np.mean(c_acc[Ks[-1]]) - np.mean(c_acc[Ks[0]]))
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"diag": ci(diag_b),
            "curve": {K: ci(curve_b[K]) for K in Ks},
            "diff": (float(np.mean(diff_b)), *ci(diff_b))}


def curve_strict(fams):
    """Held-out PCA: refit scaler+PCA on each training subset; point estimates only."""
    names = list(fams)
    Ks = list(range(1, len(names)))
    diag_acc = []
    for B in names:
        d = fams[B]; te = d["split"] == "test"; cal = d["split"] == "cal"
        tf = fit_space_on(d["emb"][cal])
        h = head(tf(d["emb"][cal]), d["label"][cal])
        diag_acc.append(roc_auc_score(d["label"][te], h.decision_function(tf(d["emb"][te]))))
    diag = float(np.mean(diag_acc))
    curve = {}
    for K in Ks:
        per_B = []
        for B in names:
            d = fams[B]; te = d["split"] == "test"
            others = [a for a in names if a != B]
            rng = np.random.default_rng((abs(hash(B)) + K) % (2**32))
            au = []
            for S in subsets_of(others, K, rng):
                cale = np.vstack([fams[a]["emb"][fams[a]["split"] == "cal"] for a in S])
                caly = np.concatenate([fams[a]["label"][fams[a]["split"] == "cal"] for a in S])
                tf = fit_space_on(cale)
                h = head(tf(cale), caly)
                au.append(roc_auc_score(d["label"][te], h.decision_function(tf(d["emb"][te]))))
            per_B.append(np.mean(au))
        curve[K] = float(np.mean(per_B))
    return diag, curve


def main():
    print(f"data: {T2}")
    fams = load()
    print("families (" + str(len(fams)) + "):", ", ".join(fams))
    P = global_space(fams)
    names, Ks, store = curve_primary(fams, P)
    diag, curve = point_curve(names, Ks, store)
    boot = boot_curve(names, Ks, store)

    print("\n===== PRIMARY (matrix-comparable PCA, B-stratified task-cluster bootstrap) =====")
    print(f"  diagonal ceiling (own-family head)      = {diag:.3f}  CI{boot['diag']}")
    print(f"  {'K (pooled calib families)':>26}  meanAUROC      95% CI        recovery-of-gap")
    floor = curve[Ks[0]]
    for K in Ks:
        rec = 100 * (curve[K] - floor) / max(diag - floor, 1e-9)
        lo, hi = boot["curve"][K]
        print(f"  {K:>26}  {curve[K]:.3f}        [{lo:.3f},{hi:.3f}]    {rec:+5.0f}%")
    dm, dlo, dhi = boot["diff"]
    print(f"\n  pooled[K={Ks[-1]}] - pooled[K=1]  = {dm:+.3f}  CI[{dlo:+.3f},{dhi:+.3f}]"
          f"   {'(CI>0: pooling MORE helps)' if dlo > 0 else '(CI spans 0: no benefit from more families)'}")
    full_rec = 100 * (curve[Ks[-1]] - floor) / max(diag - floor, 1e-9)
    print(f"  pooling all {Ks[-1]} foreign families recovers {full_rec:.0f}% of the diagonal gap "
          f"(floor {floor:.3f} -> {curve[Ks[-1]]:.3f}, ceiling {diag:.3f}).")

    sdiag, scurve = curve_strict(fams)
    print("\n===== ROBUSTNESS (strict held-out PCA, B absent from whole pipeline; points only) =====")
    print(f"  diagonal ceiling = {sdiag:.3f}")
    sfloor = scurve[Ks[0]]
    print("  K:  " + "  ".join(f"{K}={scurve[K]:.3f}" for K in Ks))
    print(f"  pooled[{Ks[-1]}]-pooled[1] = {scurve[Ks[-1]]-sfloor:+.3f};  "
          f"recovers {100*(scurve[Ks[-1]]-sfloor)/max(sdiag-sfloor,1e-9):.0f}% of the gap "
          f"-> curve shape {'CONFIRMS' if (scurve[Ks[-1]]-sfloor)*(curve[Ks[-1]]-floor) >= 0 else 'DIFFERS from'} primary.")


if __name__ == "__main__":
    raise SystemExit(main())
