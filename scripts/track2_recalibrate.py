"""Control 20 — benign-only recalibration (covariate-shift vs concept-shift).

The constructive completion of the Control 18+19 measurement-validity arc.
Controls 18/19: a deployer running a family-A monitor on an off-lineage stream
can DETECT it is off-lineage (19A) but can neither PREDICT the detection drop
ex-ante (18) nor recover it with a per-input OOD-deferral gate (19B). The
prescription was "detect, then RECALIBRATE on the faced family." This asks the
operational follow-up: can the CHEAPEST recalibration -- benign-only,
unsupervised domain adaptation of the monitor's inputs to family B, with NO B
sabotage labels and the A-head frozen -- recover the cross-family AUROC drop?

This is the textbook covariate-shift vs concept-shift test. If the cross-family
failure is a covariate shift (B lives in a shifted region but the benign-vs-
sabotage boundary is unchanged), aligning the benign clouds fixes it for free.
If it is a concept shift (B's sabotage signature itself differs), benign-only
alignment cannot help and recalibration requires labeled sabotage from B.

Information set (non-circular, the deployer's): head A frozen (track2_matrix
head, trained on A cal only); adaptation fit on B's CAL-split BENIGN ONLY
(label==0, no attacks, no labels); evaluated on B's TEST split. Pooled-cal
PCA-50 identical to track2_matrix/track2_ood_gate so cells == the headline.

Pre-registered (both LIVE/DIE branches registered BEFORE running).

Run: .venv-controlarena/bin/python scripts/track2_recalibrate.py [<dir> ...]
"""
from __future__ import annotations
import glob, os, sys, hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_FRAC = 60
OPEN4 = {"gemma", "llama", "mistral", "qwen"}
NBOOT = 1000


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


def sym_sqrt(cov, inv=False):
    """Symmetric matrix (inverse) square root via eigendecomposition."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-8, None)
    s = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * s) @ V.T


def build(fams):
    """Pooled-cal PCA-50 + per-family frozen heads + per-family benign-cloud
    stats (mean, LedoitWolf cov, and CORAL maps) on CAL-split benign only.
    Exactly track2_matrix's head/PCA."""
    names = list(fams)
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads, bstat = {}, {}
    for A in names:
        d = fams[A]
        cal = d["split"] == "cal"
        heads[A] = LogisticRegression(max_iter=2000, C=1.0).fit(d["p"][cal], d["label"][cal])
        # benign-cloud stats on A's CAL-split benign only (deployer's info set)
        bm = cal & (d["label"] == 0)
        Xb = d["p"][bm]
        cov = LedoitWolf().fit(Xb).covariance_
        bstat[A] = {"mu": Xb.mean(0), "cov": cov,
                    "half": sym_sqrt(cov), "ihalf": sym_sqrt(cov, inv=True)}
    return names, heads, bstat


def adapt_meanshift(x, sB, sA):
    """Translate B's points so its benign cloud centers where A's does."""
    return x + (sA["mu"] - sB["mu"])


def adapt_coral(x, sB, sA):
    """Whiten by B's benign cov, recolor to A's benign cov (CORAL), re-center to A."""
    return (x - sB["mu"]) @ sB["ihalf"] @ sA["half"] + sA["mu"]


METHODS = {"meanshift": adapt_meanshift, "coral": adapt_coral}


def analyze(t2, keep=None):
    fams = load_dir(t2, keep=keep)
    names, heads, bstat = build(fams)
    n = len(names)
    print(f"\n=== {os.path.basename(t2.rstrip('/'))}  ({n} families) ===")
    print("families:", names)

    # per-family test rows + scores under each scheme
    test = {}
    for fam in names:
        d = fams[fam]
        m = d["split"] == "test"
        test[fam] = {"p": d["p"][m], "lab": d["label"][m], "task": d["task"][m]}

    # diagonal AUROC (head A on A's own test) -- no adaptation
    diag = {A: roc_auc_score(test[A]["lab"], heads[A].decision_function(test[A]["p"]))
            for A in names}

    # off-diagonal cells: base + each adaptation method.  scores cached per cell.
    cells = {}  # (A,B) -> {task, lab, base_score, meanshift_score, coral_score}
    for A in names:
        for B in names:
            if A == B:
                continue
            tB = test[B]
            sc = {"base": heads[A].decision_function(tB["p"])}
            for mname, fn in METHODS.items():
                sc[mname] = heads[A].decision_function(fn(tB["p"], bstat[B], bstat[A]))
            cells[(A, B)] = {"task": tB["task"], "lab": tB["lab"], **{f"s_{k}": v for k, v in sc.items()}}

    def cell_auc(c, key, idx):
        y, s = c["lab"][idx], c[f"s_{key}"][idx]
        if len(np.unique(y)) < 2:
            return None
        return roc_auc_score(y, s)

    # point estimates: pooled over off-diag cells
    def pooled(key, sub_by_cell):
        vals = [cell_auc(c, key, sub_by_cell[ab]) for ab, c in cells.items()]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else np.nan

    full_sub = {ab: np.arange(len(c["task"])) for ab, c in cells.items()}
    mean_diag = float(np.mean([diag[A] for A in names]))
    base = pooled("base", full_sub)
    print(f"mean diagonal AUROC        = {mean_diag:.3f}")
    print(f"mean off-diag AUROC (base) = {base:.3f}   gap = {mean_diag - base:+.3f}")

    # shared test-task universe for the paired cluster-bootstrap
    universe = sorted(set.intersection(*[set(c["task"]) for c in cells.values()]))
    by_cell_task = {ab: {t: np.where(c["task"] == t)[0] for t in universe}
                    for ab, c in cells.items()}

    def boot_lift(key):
        """Paired cluster-boot over shared task universe: pooled(adapt)-pooled(base)."""
        rng = np.random.default_rng(0)
        U = np.array(universe)
        diffs, adapts = [], []
        for _ in range(NBOOT):
            drawn = rng.choice(U, size=len(U), replace=True)
            sub = {ab: np.concatenate([by_cell_task[ab][t] for t in drawn]) for ab in cells}
            a = pooled(key, sub); b = pooled("base", sub)
            if not (np.isnan(a) or np.isnan(b)):
                diffs.append(a - b); adapts.append(a)
        return (np.mean(adapts), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5),
                np.mean(diffs))

    gap = mean_diag - base
    print(f"\n{'method':>10} {'off-diag':>9} {'lift':>8} {'95% CI':>20} {'recovery%':>10}  (LIVE if lift CI>0 & recov>=25%)")
    for mname in METHODS:
        adapt_mean, lo, hi, mlift = boot_lift(mname)
        recov = 100.0 * mlift / gap if gap > 1e-9 else float("nan")
        flag = "LIVE" if lo > 0 and recov >= 25 else "die"
        print(f"{mname:>10} {pooled(mname, full_sub):>9.3f} {mlift:>+8.4f} "
              f"[{lo:>+7.4f},{hi:>+7.4f}] {recov:>9.1f}  {flag}")

    # per-cell detail (point estimates)
    print(f"\nper-cell off-diagonal AUROC (A=head, B=stream):")
    print(f"  {'A->B':>20} {'base':>6} {'mshift':>7} {'coral':>6} {'gapclosed_coral%':>16}")
    for (A, B), c in sorted(cells.items()):
        b = cell_auc(c, "base", full_sub[(A, B)])
        ms = cell_auc(c, "meanshift", full_sub[(A, B)])
        cr = cell_auc(c, "coral", full_sub[(A, B)])
        g = diag[A] - b
        gc = 100.0 * (cr - b) / g if g > 1e-9 else float("nan")
        print(f"  {short(A)+'->'+short(B):>20} {b:>6.3f} {ms:>7.3f} {cr:>6.3f} {gc:>16.1f}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or [os.path.join(REPO, "results", "track2_n600")]
    for d in dirs:
        keep = OPEN4 if os.path.basename(d.rstrip("/")) in ("track2_n600", "track2_n600_actiononly") else None
        analyze(d, keep=keep)
