"""Confound control: difficulty-matching the Track-2 interaction (the design doc s4, s1b#1).

Rules out "the +0.127 interaction is just capability/difficulty spread." The four
families differ in attack main-task-success (gemma 32%, qwen 27%, mistral 40%,
llama 41% -- a ~14pt spread). If the diagonal-bonus interaction were an artifact of
that spread, equalizing it across families should kill the interaction.

Method: subsample each EVAL family's TEST-split ATTACK samples so every family has
an IDENTICAL main-task-success composition (same #main=T and #main=F = the per-stratum
min across families), then re-run the EXACT track2_matrix interaction + cluster bootstrap
on the matched test set. Honest (negative) test samples are kept in full; the difficulty
confound is about the ATTACK distribution. Heads are trained on the SAME full cal splits
as the headline (we re-score the same monitor on a difficulty-matched test set).

NB the interaction is ALREADY net of additive row (monitor capability) + column (attacker
obviousness) MAIN effects -- so this tests whether RESIDUAL diagonal structure survives
difficulty matching. Success-filter trap: both-success (main AND side) is ~0% at n=200,
so we match on main_task_success only (a capability proxy), never on backdoor success.

Run: .venv/bin/python scripts/track2_difficulty.py [--seed N] [--t2 track2|track2_actiononly]
"""
from __future__ import annotations
import glob, os, hashlib, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _clusterboot import task_rows, draw_index  # multiplicity-preserving cluster bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_FRAC = 60


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells, names):
    """VERBATIM from track2_matrix.py."""
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def cluster_boot(fams, names, heads, n_boot=1000, seed=0):
    """Mirrors track2_matrix.cluster_boot, but the test mask honours d['keep'] (the
    difficulty-matched attack subset). Honest test rows always kept; attack rows kept
    iff selected into the matched subset."""
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values()
                         for t, sp, k in zip(d["task"], d["split"], d["keep"]) if sp == "test" and k})
    rows = {B: task_rows(fams[B]["task"], (fams[B]["split"] == "test") & fams[B]["keep"], test_tasks)
            for B in names}
    raws, ints = [], []
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
                cells[(A, B)] = roc_auc_score(y, heads[A].decision_function(d["p"][idx]))
            if not ok:
                break
        if not ok:
            continue
        diag = np.mean([cells[(x, x)] for x in names])
        off = np.mean([cells[(a, b)] for a in names for b in names if a != b])
        raws.append(diag - off)
        ints.append(diag_interaction(cells, names))
    r, it = np.array(raws), np.array(ints)
    return (r.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5),
            it.mean(), np.percentile(it, 2.5), np.percentile(it, 97.5), len(it))


def attack_main_success():
    """task_id -> main_task_success (bool) for ATTACK samples, per family name."""
    out = {}
    for d in sorted(glob.glob(os.path.join(REPO, "results", "W2_ow_*_n200"))):
        fam = os.path.basename(d).replace("W2_", "")
        df = pd.read_csv(os.path.join(d, "samples.csv"))
        a = df[df["mode"] == "attack"].drop_duplicates("task_id", keep="first")
        out[fam] = {str(t): bool(s) for t, s in zip(a["task_id"], a["main_task_success"])}
    return out


def build_matched_keep(fams, names, succ, seed):
    """Per family, split TEST attack rows by main_task_success, take the per-stratum min
    count across families (seeded subsample). Honest rows always kept. Sets d['keep']."""
    # available test-attack task_ids per family, by success stratum
    avail = {}
    for B in names:
        d = fams[B]
        te_att = (d["split"] == "test") & (d["label"] == 1)
        tasks = d["task"][te_att]
        s = succ[B]
        T = [t for t in tasks if s.get(t, False)]
        F = [t for t in tasks if not s.get(t, False)]
        avail[B] = (T, F)
    tgt_T = min(len(T) for T, F in avail.values())
    tgt_F = min(len(F) for T, F in avail.values())
    print(f"matched attack composition per family: main=T {tgt_T}, main=F {tgt_F} "
          f"(rate {tgt_T/(tgt_T+tgt_F)*100:.1f}%); n_pos/family={tgt_T+tgt_F}")
    rng = np.random.default_rng(seed)
    chosen = {}
    for B in names:
        T, F = avail[B]
        selT = {T[i] for i in rng.choice(len(T), size=tgt_T, replace=False)}
        selF = {F[i] for i in rng.choice(len(F), size=tgt_F, replace=False)}
        chosen[B] = selT | selF
    for B in names:
        d = fams[B]
        keep = np.ones(len(d["task"]), dtype=bool)
        for i, (t, lab, sp) in enumerate(zip(d["task"], d["label"], d["split"])):
            if sp == "test" and lab == 1 and t not in chosen[B]:
                keep[i] = False
        d["keep"] = keep
        # sanity: matched test-attack success rate
        ta = (d["split"] == "test") & (d["label"] == 1) & d["keep"]
        rate = np.mean([succ[B].get(t, False) for t in d["task"][ta]]) if ta.sum() else float("nan")
        print(f"  {B}: kept {int(ta.sum())} test attacks, main-success {rate*100:.1f}%")


def main():
    args = sys.argv[1:]
    seed = 0
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    t2 = "track2"
    if "--t2" in args:
        i = args.index("--t2"); t2 = args[i + 1]; del args[i:i + 2]
    T2 = os.path.join(REPO, "results", t2)

    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])

    succ = attack_main_success()
    build_matched_keep(fams, names, succ, seed)

    # SAME pipeline as track2_matrix: pooled-cal scaler + PCA-50, per-family logistic head
    # on the FULL cal split (heads unchanged; only the test positives are difficulty-matched).
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])

    short = lambda n: n.replace("ow_", "").replace("_n200", "")
    print(f"\n===== Difficulty-matched Track-2 AUROC matrix ({t2}; rows=calib head, cols=eval) =====")
    print("            " + "  ".join(f"{short(n):>9}" for n in names))
    M = {}
    for A in names:
        row = []
        for B in names:
            d = fams[B]; te = (d["split"] == "test") & d["keep"]
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
            row.append(f"{M[(A,B)]:.3f}")
        print(f"{short(A):>10}  " + "  ".join(f"{x:>9}" for x in row))

    diag = np.mean([M[(x, x)] for x in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    inter = diag_interaction(M, names)
    print(f"\nmean diagonal AUROC      = {diag:.3f}")
    print(f"mean off-diagonal AUROC  = {off:.3f}")
    print(f"raw diagonal bonus       = {diag-off:+.4f}")
    print(f"INTERACTION (net of main effects, difficulty-matched) = {inter:+.4f}")
    rm, rlo, rhi, im, ilo, ihi, nb = cluster_boot(fams, names, heads, seed=seed)
    print(f"  cluster-bootstrap (resample matched test tasks, n={nb}):")
    print(f"    raw bonus   : mean={rm:+.4f}  95% CI [{rlo:+.4f}, {rhi:+.4f}]")
    print(f"    INTERACTION : mean={im:+.4f}  95% CI [{ilo:+.4f}, {ihi:+.4f}]")
    print("\nInterpretation: interaction survives difficulty-matching & CI>0 -> NOT a capability/")
    print("                difficulty artifact (residual diagonal structure is real). Vanishes -> ")
    print("                the diagonal bonus was difficulty spread, not ranking overfit.")


if __name__ == "__main__":
    raise SystemExit(main())
