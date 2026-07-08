"""Control: blatancy-stratified replication of the Track-2 interaction (the design doc s4, s1b#1).

Rules out "the interaction is just families differing in attack BLATANCY / incrimination
level" -- the Async Control / Auditing-Sabotage-Bench finding that incrimination level
dominates detection (the design doc s1b explanation #1, the #1 alternative explanation). The
paper's abstract claims a monitor ranks its own family's sabotage above "an EQUALLY BLATANT
attack from a different lineage"; this control is what backs the word "equally blatant".

Control 4 matched CAPABILITY (main_task_success); this matches BLATANCY -- an independent and
more direct obviousness axis. The blatancy proxy is the Track-1 FIXED-PROMPT monitor's
per-transcript suspicion (max_suspicion_scorer, gpt-4.1-mini, 0-1) -- a DIFFERENT monitor from
the Track-2 embedding+logistic head being evaluated, so stratifying test attacks by it and then
measuring the Track-2 diagonal bonus is NON-CIRCULAR. Suspicion is re-extracted from the eval
logs in the IDENTICAL iteration order track2_embed.load_family used, then asserted 1:1 aligned
to the npz rows (same task_id + label, elementwise).

Two designs, both re-score the SAME per-family heads (trained on the FULL cal split, unchanged --
exactly as Control 4) on a blatancy-controlled TEST attack set, honest negatives kept in full:
  (B) blatancy-MATCHED: subsample each eval family's TEST attacks to an IDENTICAL suspicion-band
      composition (per-band min count across families) -> one interaction. Neutralizes the
      blatancy<->family confound by construction (direct analog of Control 4's difficulty match).
  (A) within-stratum: restrict TEST attacks to a single blatancy band -> interaction per band.
      The literal s4 ask "show the gap persists within a blatancy stratum".

The interaction is ALREADY net of additive row (monitor capability) + column (attacker
obviousness) MAIN effects, so a per-family blatancy MEAN difference is already subtracted; this
control attacks the subtler confound that blatancy varies WITHIN family in a family-correlated way
the linear column main-effect cannot absorb.

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_blatancy.py \
        [--t2 track2_n600|track2_n600_actiononly] [--seed N]
"""
from __future__ import annotations
import glob, os, hashlib, sys
import numpy as np
from inspect_ai.log import read_eval_log
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _clusterboot import task_rows, draw_index  # multiplicity-preserving cluster bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_FRAC = 60
# grid-clean bands on the 0-1 suspicion scale (values are discretised to 0.1 steps).
# mid (0.3-0.6) is thin for every family (~14-29 test attacks); LOW vs HIGH are the
# well-powered, well-separated strata used for the within-stratum test (design A). All
# three feed the matched design (B), which uses the per-band MIN so thinness is shared.
BANDS = {"low": lambda s: s <= 0.2, "mid": lambda s: (s >= 0.3) & (s <= 0.6), "high": lambda s: s >= 0.7}


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells, names):
    """VERBATIM from track2_matrix.py / track2_difficulty.py."""
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def cluster_boot(fams, names, heads, n_boot=1000, seed=0):
    """Mirrors track2_difficulty.cluster_boot: resample TEST tasks (cluster), honour d['keep']
    (honest test rows always kept; attack test rows kept iff in the blatancy-controlled subset)."""
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


def extract_suspicion(fam_dir):
    """Per-row Track-1 suspicion (max_suspicion_scorer), in track2_embed.load_family order."""
    susp, labs, tasks = [], [], []
    for f in sorted(glob.glob(os.path.join(fam_dir, "*.eval"))):
        label = 1 if "attack" in os.path.basename(f) else 0
        for s in (read_eval_log(f).samples or []):
            sc = s.scores.get("max_suspicion_scorer") if s.scores else None
            if not (sc and str(sc.value) not in ("N", "None")):
                continue
            susp.append(float(sc.value)); labs.append(label); tasks.append(str(s.id))
    return np.array(susp), np.array(labs), np.array(tasks, dtype=object)


def matrix(fams, names, heads, tag):
    M = {}
    for A in names:
        for B in names:
            d = fams[B]; te = (d["split"] == "test") & d["keep"]
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
    diag = np.mean([M[(x, x)] for x in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    short = lambda n: n.replace("ow_", "").replace("_n600", "").replace("_n200", "")
    print(f"\n----- {tag} -----")
    print("            " + "  ".join(f"{short(n):>9}" for n in names))
    for A in names:
        print(f"{short(A):>10}  " + "  ".join(f"{M[(A,B)]:>9.3f}" for B in names))
    inter = diag_interaction(M, names)
    print(f"  diag {diag:.3f} | off {off:.3f} | raw bonus {diag-off:+.4f} | INTERACTION {inter:+.4f}")
    rm, rlo, rhi, im, ilo, ihi, nb = cluster_boot(fams, names, heads, seed=0)
    print(f"  cluster-boot (n={nb}): raw {rm:+.4f} [{rlo:+.4f},{rhi:+.4f}]  "
          f"INTERACTION {im:+.4f} [{ilo:+.4f},{ihi:+.4f}]")
    return inter, (im, ilo, ihi)


def set_keep_band(fams, names, band_fn):
    """keep = all honest test rows + attack test rows whose suspicion is in the band."""
    for B in names:
        d = fams[B]
        keep = np.zeros(len(d["task"]), dtype=bool)
        for i, (lab, sp, su) in enumerate(zip(d["label"], d["split"], d["susp"])):
            if sp != "test":
                continue
            keep[i] = (lab == 0) or (lab == 1 and bool(band_fn(su)))
        d["keep"] = keep


def set_keep_matched(fams, names, seed):
    """Subsample each family's TEST attacks to an identical per-band composition (per-band min)."""
    avail = {B: {} for B in names}
    for B in names:
        d = fams[B]
        te_att = (d["split"] == "test") & (d["label"] == 1)
        for bn, fn in BANDS.items():
            avail[B][bn] = d["task"][te_att & np.array([bool(fn(s)) for s in d["susp"]])]
    tgt = {bn: min(len(avail[B][bn]) for B in names) for bn in BANDS}
    print(f"matched per-family attack composition: " +
          " ".join(f"{bn}={tgt[bn]}" for bn in BANDS) + f"  (n_pos/family={sum(tgt.values())})")
    rng = np.random.default_rng(seed)
    chosen = {}
    for B in names:
        sel = set()
        for bn in BANDS:
            pool = avail[B][bn]
            sel |= {pool[i] for i in rng.choice(len(pool), size=tgt[bn], replace=False)}
        chosen[B] = sel
    for B in names:
        d = fams[B]
        keep = np.ones(len(d["task"]), dtype=bool)
        for i, (t, lab, sp) in enumerate(zip(d["task"], d["label"], d["split"])):
            if sp == "test" and lab == 1 and t not in chosen[B]:
                keep[i] = False
        d["keep"] = keep
        ta = (d["split"] == "test") & (d["label"] == 1) & d["keep"]
        msusp = d["susp"][ta]
        print(f"  {B}: kept {int(ta.sum())} test attacks, mean suspicion {msusp.mean():.3f}")


def main():
    args = sys.argv[1:]
    seed = 0
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    t2 = "track2_n600"
    if "--t2" in args:
        i = args.index("--t2"); t2 = args[i + 1]; del args[i:i + 2]
    T2 = os.path.join(REPO, "results", t2)

    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        susp, labs, tasks = extract_suspicion(os.path.join(REPO, "results", f"W2_{name}"))
        assert len(susp) == len(d["task_id"]) and np.all(tasks == d["task_id"]) and np.all(labs == d["label"]), \
            f"{name}: suspicion extraction not aligned to npz rows"
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"], "susp": susp}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])

    # band census (test attacks)
    print(f"t2={t2}  families={names}")
    print("TEST-attack counts by blatancy band (suspicion = Track-1 fixed-prompt monitor):")
    for B in names:
        d = fams[B]; te = (d["split"] == "test") & (d["label"] == 1); s = d["susp"][te]
        print(f"  {B}: " + " ".join(f"{bn}={int(fn(s).sum())}" for bn, fn in BANDS.items()) + f"  (total {int(te.sum())})")

    # SAME pipeline as track2_matrix/difficulty: pooled-cal scaler + PCA-50, per-family logistic head
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])

    print("\n================ DESIGN B: blatancy-MATCHED interaction ================")
    set_keep_matched(fams, names, seed)
    matrix(fams, names, heads, "blatancy-matched (identical suspicion composition across families)")

    print("\n================ DESIGN A: within-stratum interactions ================")
    for bn in ("low", "high"):
        set_keep_band(fams, names, BANDS[bn])
        matrix(fams, names, heads, f"within-stratum: {bn}-blatancy test attacks only")

    print("\nInterpretation: interaction stays CI>0 under blatancy-matching AND within each stratum")
    print("  -> the diagonal bonus is NOT families differing in attack blatancy/incrimination;")
    print("  a family-conditioned monitor preferentially ranks its OWN family's EQUALLY-blatant")
    print("  sabotage. Vanishes -> the apparent transfer gap was blatancy spread (pivot to s1b#1).")


if __name__ == "__main__":
    raise SystemExit(main())
