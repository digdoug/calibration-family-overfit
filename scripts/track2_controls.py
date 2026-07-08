"""Track-2 confound controls on the EXISTING per-family embeddings (no generation).

Two cheap controls, both on results/<track2dir>/*.npz (default track2):

  (1) FAMILY-ID classifier  -- train a 4-way multiclass head on BENIGN (honest)
      embeddings only; held-out accuracy vs 25% chance. High accuracy => the
      embedding carries a strong family signature even in benign behavior, which
      is *why* a per-family attack head transfers poorly (it keys on signature).
      This contextualizes the Track-2 interaction; it does not by itself prove it.

  (2) LOFO calibration (the s6 DEFENSIVE-PAYOFF test) -- for each held-out family
      B, compare three monitors scored on B's TEST split:
         diag_B      : head trained on B's own cal split        (in-distribution ceiling)
         pooled3_B   : head trained on the POOLED cal of the OTHER 3 families
         foreign1_B  : mean over A!=B of (head trained on A alone) on B  (single foreign)
      If pooled3_B > foreign1_B, calibrating on MANY families helps a held-out
      attacker -> "calibrate broadly" is a real defense. If pooled3_B ~ diag_B,
      pooling nearly recovers in-distribution performance.

Shares the matrix's pipeline (task-disjoint split, scaler+PCA-50 on pooled cal)
so numbers are directly comparable to track2_matrix.py.

Run: .venv/bin/python scripts/track2_controls.py [track2dir]
"""
from __future__ import annotations
import glob, os, hashlib, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 else "track2")
CAL_FRAC = 60


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def load():
    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4].replace("ow_", "").replace("_n200", "")
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
        fams[name]["split"] = np.array([split_of(t) for t in d["task_id"]])
    # shared scaler + PCA-50 on POOLED cal (identical to track2_matrix)
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    return fams


def family_id(fams):
    print("\n===== (1) FAMILY-ID classifier on BENIGN transcripts =====")
    names = list(fams)
    Xtr, ytr, Xte, yte = [], [], [], []
    for i, (nm, d) in enumerate(fams.items()):
        ben = d["label"] == 0
        tr = ben & (d["split"] == "cal"); te = ben & (d["split"] == "test")
        Xtr.append(d["p"][tr]); ytr += [i] * int(tr.sum())
        Xte.append(d["p"][te]); yte += [i] * int(te.sum())
    Xtr = np.vstack(Xtr); Xte = np.vstack(Xte); ytr = np.array(ytr); yte = np.array(yte)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr)  # multinomial by default in modern sklearn
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred); chance = 1.0 / len(names)
    print(f"  4-way held-out accuracy = {acc:.3f}   (chance = {chance:.3f}, n_test={len(yte)})")
    for i, nm in enumerate(names):
        m = yte == i
        print(f"    {nm:>9}: recall = {accuracy_score(yte[m], pred[m]):.3f}  (n={int(m.sum())})")
    print(f"  => benign transcripts {'ARE' if acc > 2*chance else 'are NOT strongly'} family-separable "
          f"({acc/chance:.1f}x chance).")
    return float(acc)


def lofo(fams):
    print("\n===== (2) LOFO calibration -- the defensive-payoff test (s6) =====")
    names = list(fams)
    heads = {}  # single-family heads (same as the matrix)
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    print(f"  {'held-out B':>9}   diag_B   pooled3_B   foreign1_B   pooled-minus-foreign")
    rows = []
    for B in names:
        d = fams[B]; te = d["split"] == "test"; yte = d["label"][te]
        diagB = roc_auc_score(yte, heads[B].decision_function(d["p"][te]))
        # pooled head on the OTHER 3 families' cal split
        ox = np.vstack([fams[A]["p"][fams[A]["split"] == "cal"] for A in names if A != B])
        oy = np.concatenate([fams[A]["label"][fams[A]["split"] == "cal"] for A in names if A != B])
        pooled = LogisticRegression(C=1.0, max_iter=3000).fit(ox, oy)
        pooled3B = roc_auc_score(yte, pooled.decision_function(d["p"][te]))
        foreign1B = float(np.mean([roc_auc_score(yte, heads[A].decision_function(d["p"][te]))
                                   for A in names if A != B]))
        rows.append((diagB, pooled3B, foreign1B))
        print(f"  {B:>9}   {diagB:.3f}    {pooled3B:.3f}      {foreign1B:.3f}        {pooled3B-foreign1B:+.3f}")
    diag, pooled3, foreign1 = (float(np.mean([r[i] for r in rows])) for i in range(3))
    print(f"  {'MEAN':>9}   {diag:.3f}    {pooled3:.3f}      {foreign1:.3f}        {pooled3-foreign1:+.3f}")
    print(f"  => pooling 3 foreign families {'BEATS' if pooled3 > foreign1 else 'does NOT beat'} a single "
          f"foreign family by {pooled3-foreign1:+.3f} AUROC")
    print(f"     and recovers {100*(pooled3-foreign1)/max(diag-foreign1,1e-9):.0f}% of the "
          f"in-distribution gap (diag {diag:.3f} - foreign {foreign1:.3f}).")
    return {"diag": diag, "pooled3": pooled3, "foreign1": foreign1}


def main():
    print(f"data: {T2}")
    fams = load()
    print("families:", ", ".join(f"{n}(n={len(d['label'])})" for n, d in fams.items()))
    family_id(fams)
    lofo(fams)


if __name__ == "__main__":
    raise SystemExit(main())
