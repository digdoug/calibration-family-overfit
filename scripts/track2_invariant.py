"""Phase 2.5: family-INVARIANT scoring — the mitigation beat (no API, no generation).

The LOFO control (Control 3) showed naive multi-family pooling gives NO defensive
payoff: a held-out family B is scored at ~0.54 AUROC by the mean single foreign head
(foreign1) vs 0.71 by its own head (diag), and pooled-3 is even worse (0.45, can
anti-transfer). That left "so what do we DO?" open. This script answers it on the
EXISTING embeddings — three family-adaptive scoring strategies, each measured as
cross-family transfer (how well we rank a HELD-OUT family's attacks using only the
OTHER families + B's *benign* data) vs the single-foreign baseline:

  (a) NEAREST-FAMILY ROUTING — pick the closest calibration family A* to B by benign-
      embedding similarity (centroid cosine), score B with A*'s head. Benign data for B
      is realistic (honest deployment traffic); attack labels for B are NOT used.
  (b) HEAD ENSEMBLE — aggregate the foreign heads' decision scores (each z-scored by its
      own benign-cal baseline), mean and max, score B.
  (c) FAMILY-SIGNATURE-SUBSPACE REMOVAL — fit the family-ID classifier (Control 2) on
      benign cal embeddings, take its top-r discriminative directions, project them OUT
      of the embedding, retrain the per-family attack heads on the residual, re-measure
      the interaction AND off-diagonal transfer. If the interaction SHRINKS, the overfit
      lives in a removable family-identity subspace -> a principled mitigation. If it
      doesn't, attack signal is entangled with family identity -> also publishable.

Same pipeline as track2_matrix / track2_controls (task-disjoint split, scaler+PCA-50 on
pooled benign+attack cal) so numbers are directly comparable.

Run: .venv/bin/python scripts/track2_invariant.py [track2|track2_actiononly]
"""
from __future__ import annotations
import glob, os, hashlib, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from _clusterboot import task_rows, draw_index  # multiplicity-preserving cluster bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "track2")
CAL_FRAC = 60
N_BOOT = 1000


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells, names):
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def load():
    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4].replace("ow_", "").replace("_n200", "")
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"],
                      "split": np.array([split_of(t) for t in d["task_id"]])}
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    return fams


def fit_heads(fams, names, key="p"):
    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d[key][cal], d["label"][cal])
    return heads


def matrix(fams, names, heads, key="p"):
    M = {}
    for A in names:
        for B in names:
            d = fams[B]; te = d["split"] == "test"
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d[key][te]))
    diag = float(np.mean([M[(x, x)] for x in names]))
    off = float(np.mean([M[(a, b)] for a in names for b in names if a != b]))
    return M, diag, off, diag_interaction(M, names)


def boot_interaction(fams, names, heads, key="p", n_boot=N_BOOT, seed=0):
    """Cluster-bootstrap the interaction in a given representation (VERBATIM track2_matrix logic)."""
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
                cells[(A, B)] = roc_auc_score(y, heads[A].decision_function(d[key][idx]))
            if not ok:
                break
        if ok:
            ints.append(diag_interaction(cells, names))
    it = np.array(ints)
    return it.mean(), np.percentile(it, 2.5), np.percentile(it, 97.5)


# ---------- (a) nearest-family routing ----------
def benign_centroid(d, key="p"):
    m = (d["split"] == "cal") & (d["label"] == 0)
    return d[key][m].mean(axis=0)


def route(fams, names, key="p"):
    cents = {A: benign_centroid(fams[A], key) for A in names}
    pick = {}
    for B in names:
        cb = cents[B]
        sims = {A: float(np.dot(cents[A], cb) / (np.linalg.norm(cents[A]) * np.linalg.norm(cb) + 1e-12))
                for A in names if A != B}
        pick[B] = max(sims, key=sims.get)
    return pick


# ---------- (b) head ensemble ----------
def ensemble_auroc(fams, names, heads, B, key="p", agg="mean"):
    d = fams[B]; te = d["split"] == "test"; y = d["label"][te]
    zs = []
    for A in names:
        if A == B:
            continue
        cal_ben = (fams[A]["split"] == "cal") & (fams[A]["label"] == 0)
        ref = heads[A].decision_function(fams[A]["p"][cal_ben])  # this head's honest baseline
        mu, sd = float(ref.mean()), float(ref.std()) + 1e-9
        zs.append((heads[A].decision_function(d[key][te]) - mu) / sd)
    Z = np.array(zs)
    s = Z.mean(axis=0) if agg == "mean" else Z.max(axis=0)
    return roc_auc_score(y, s)


# ---------- (c) subspace removal ----------
def family_id_dirs(fams, names, key="p"):
    """Fit 4-way family-ID on benign cal; return its coef row-space basis (V) + the classifier."""
    Xtr, ytr = [], []
    for i, A in enumerate(names):
        d = fams[A]; m = (d["split"] == "cal") & (d["label"] == 0)
        Xtr.append(d[key][m]); ytr += [i] * int(m.sum())
    Xtr = np.vstack(Xtr); ytr = np.array(ytr)
    fid = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr)
    # orthonormal basis of the discriminative (coefficient) row space
    _, _, Vt = np.linalg.svd(fid.coef_, full_matrices=False)  # rows of Vt span it
    return Vt, fid


def fid_accuracy(fams, names, key):
    """Held-out 4-way family-ID accuracy on benign test in the given representation."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for i, A in enumerate(names):
        d = fams[A]; ben = d["label"] == 0
        tr = ben & (d["split"] == "cal"); te = ben & (d["split"] == "test")
        Xtr.append(d[key][tr]); ytr += [i] * int(tr.sum())
        Xte.append(d[key][te]); yte += [i] * int(te.sum())
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack(Xtr), np.array(ytr))
    return accuracy_score(np.array(yte), clf.predict(np.vstack(Xte)))


# ---------- cluster bootstrap for a held-out-B strategy mean AUROC ----------
def boot_strategy(fams, names, fn, n_boot=N_BOOT, seed=0):
    """fn(B, idx) -> AUROC for held-out B on the resampled test rows (idx is an
    integer row-index array, multiplicity preserved). Returns (mean, lo, hi) of the
    across-B mean AUROC under cluster bootstrap by task."""
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    # Precompute test-eligible rows per family; the bootstrap draws tasks with
    # replacement and EXPANDS to row indices preserving multiplicity (true cluster
    # bootstrap). The old code collapsed the draw with set() + a boolean mask,
    # keeping each drawn task once -> CIs ~1/3 too narrow.
    test_rows = {B: task_rows(fams[B]["task"], fams[B]["split"] == "test", test_tasks) for B in names}
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        per_b, ok = [], True
        for B in names:
            idx = draw_index(test_rows[B], draw)
            try:
                a = fn(B, idx)
            except Exception:
                ok = False; break
            if a is None:
                ok = False; break
            per_b.append(a)
        if ok:
            vals.append(float(np.mean(per_b)))
    v = np.array(vals)
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    short = lambda n: n
    fams = load()
    names = list(fams)
    print(f"data: {T2}   families: {', '.join(names)}")
    heads = fit_heads(fams, names)
    M, diag, off, inter = matrix(fams, names, heads)
    foreign1 = {B: float(np.mean([M[(A, B)] for A in names if A != B])) for B in names}
    print(f"\nbaseline (original PCA-50 space): mean diag {diag:.3f} | mean off-diag {off:.3f} | "
          f"interaction {inter:+.4f}")
    print(f"  single-foreign baseline (mean foreign head per held-out B): "
          f"{np.mean(list(foreign1.values())):.3f}")

    # ---- (a) nearest-family routing ----
    print("\n===== (a) NEAREST-FAMILY ROUTING (benign-centroid cosine) =====")
    pick = route(fams, names)
    print(f"  routing: " + ", ".join(f"{B}->{pick[B]}" for B in names))
    routed = {B: M[(pick[B], B)] for B in names}
    print(f"  {'held-out B':>9} | {'diag(ceil)':>10} | {'foreign1':>9} | {'routed A*':>9} | "
          f"{'routed AUROC':>12} | gain vs foreign1")
    for B in names:
        print(f"  {B:>9} | {M[(B,B)]:>10.3f} | {foreign1[B]:>9.3f} | {pick[B]:>9} | "
              f"{routed[B]:>12.3f} | {routed[B]-foreign1[B]:+.3f}")
    mr = float(np.mean(list(routed.values()))); mf = float(np.mean(list(foreign1.values())))
    print(f"  {'MEAN':>9} | {diag:>10.3f} | {mf:>9.3f} | {'':>9} | {mr:>12.3f} | {mr-mf:+.3f}")

    def routed_fn(B, idx):
        d = fams[B]
        if len(idx) == 0 or len(np.unique(d["label"][idx])) < 2:
            return None
        return roc_auc_score(d["label"][idx], heads[pick[B]].decision_function(d["p"][idx]))

    def foreign_fn(B, idx):
        d = fams[B]
        if len(idx) == 0 or len(np.unique(d["label"][idx])) < 2:
            return None
        return float(np.mean([roc_auc_score(d["label"][idx], heads[A].decision_function(d["p"][idx]))
                              for A in names if A != B]))

    rm, rlo, rhi = boot_strategy(fams, names, routed_fn)
    fm, flo, fhi = boot_strategy(fams, names, foreign_fn)
    print(f"  cluster-boot: routed {rm:.3f} [{rlo:.3f},{rhi:.3f}]  vs  foreign1 {fm:.3f} [{flo:.3f},{fhi:.3f}]")

    # ---- (b) head ensemble ----
    print("\n===== (b) HEAD ENSEMBLE over foreign heads (z-scored by each head's benign baseline) =====")
    emean = {B: ensemble_auroc(fams, names, heads, B, agg="mean") for B in names}
    emax = {B: ensemble_auroc(fams, names, heads, B, agg="max") for B in names}
    print(f"  {'held-out B':>9} | {'foreign1':>9} | {'ens-mean':>9} | {'ens-max':>9}")
    for B in names:
        print(f"  {B:>9} | {foreign1[B]:>9.3f} | {emean[B]:>9.3f} | {emax[B]:>9.3f}")
    print(f"  {'MEAN':>9} | {mf:>9.3f} | {np.mean(list(emean.values())):>9.3f} | {np.mean(list(emax.values())):>9.3f}")

    # ---- (c) family-signature-subspace removal ----
    print("\n===== (c) FAMILY-SIGNATURE-SUBSPACE REMOVAL (project out family-ID directions) =====")
    Vt, _ = family_id_dirs(fams, names)
    print(f"  family-ID held-out acc in ORIGINAL space: {fid_accuracy(fams, names, 'p'):.3f} (chance {1/len(names):.2f})")
    im0, ilo0, ihi0 = boot_interaction(fams, names, heads, "p")
    print(f"  {'rank r':>6} | {'fid acc':>7} | {'interaction':>11} | {'95% CI':>18} | {'diag':>6} | {'off':>6} | {'off vs orig':>11}")
    print(f"  {'0(orig)':>6} | {fid_accuracy(fams,names,'p'):>7.3f} | {inter:>+11.4f} | "
          f"[{ilo0:>+6.4f},{ihi0:>+6.4f}] | {diag:>6.3f} | {off:>6.3f} | {'--':>11}")
    for r in (1, 2, 3, 5, 10):
        Q = Vt[:r].T  # (50, r)
        for d in fams.values():
            d["pr"] = d["p"] - (d["p"] @ Q) @ Q.T
        heads_r = fit_heads(fams, names, key="pr")
        Mr, diagr, offr, interr = matrix(fams, names, heads_r, key="pr")
        imr, ilor, ihir = boot_interaction(fams, names, heads_r, "pr")
        acc_r = fid_accuracy(fams, names, "pr")
        print(f"  {r:>6} | {acc_r:>7.3f} | {interr:>+11.4f} | [{ilor:>+6.4f},{ihir:>+6.4f}] | "
              f"{diagr:>6.3f} | {offr:>6.3f} | {offr-off:>+11.4f}")

    print("\nInterpretation:")
    print("  (a)/(b) recover cross-family AUROC above the single-foreign baseline => family-ADAPTIVE")
    print("          scoring is a real (if partial) defense for a held-out attacker family.")
    print("  (c) interaction SHRINKS as family-ID directions are removed => the diagonal overfit")
    print("      lives in a removable family-identity subspace (a principled mitigation). If the")
    print("      interaction is stable while off-diagonal rises, family identity is the transfer")
    print("      barrier but attack signal is partly entangled with it — both are publishable.")


if __name__ == "__main__":
    raise SystemExit(main())
