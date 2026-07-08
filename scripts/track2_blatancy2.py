"""Control 14b: blatancy-match the Track-2 interaction using an INDEPENDENT proxy.

Identical machinery to scripts/track2_blatancy.py (same per-family heads, PCA-50 pipeline,
multiplicity-preserving cluster bootstrap, matched + within-stratum designs) but the blatancy
banding variable is proxy #2 = the vendor-disjoint Claude-Haiku obviousness grade (0-10) from
scripts/blatancy_grader.py (results/blatancy2/grades.json), NOT the gpt-4.1-mini suspicion.
Pre-registered before running. Honest negatives kept in full; only TEST attacks are banded.

Bands = pooled TERTILES of the grade across all four families' test attacks (pre-specified,
well-powered). Reports the validity gate inline: corr(grade, monitor suspicion) pooled + per
family (must be positive => valid obviousness axis; not ~1.0 => not a redundant copy of proxy #1).

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_blatancy2.py [--seed N]
     (requires results/blatancy2/grades.json complete for the four open families)
"""
from __future__ import annotations
import glob, os, json, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _clusterboot import task_rows, draw_index
from track2_blatancy import (split_of, diag_interaction, extract_suspicion,
                             cluster_boot, matrix)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRADES = os.path.join(REPO, "results", "blatancy2", "grades.json")
FAMILIES = ["ow_qwen_n600", "ow_mistral_n600", "ow_gemma_n600", "ow_llama_n600"]


def set_keep_band(fams, names, lo, hi):
    """keep = all honest test rows + attack test rows whose grade is in [lo, hi]."""
    for B in names:
        d = fams[B]
        keep = np.zeros(len(d["task"]), dtype=bool)
        for i, (lab, sp, g) in enumerate(zip(d["label"], d["split"], d["blat2"])):
            if sp != "test":
                continue
            keep[i] = (lab == 0) or (lab == 1 and not np.isnan(g) and lo <= g <= hi)
        d["keep"] = keep


def set_keep_matched(fams, names, bands, seed):
    """Subsample each family's TEST attacks to identical per-band composition (per-band min)."""
    avail = {B: {} for B in names}
    for B in names:
        d = fams[B]
        te_att = (d["split"] == "test") & (d["label"] == 1) & ~np.isnan(d["blat2"])
        for bn, (lo, hi) in bands.items():
            sel = te_att & (d["blat2"] >= lo) & (d["blat2"] <= hi)
            avail[B][bn] = d["task"][sel]
    tgt = {bn: min(len(avail[B][bn]) for B in names) for bn in bands}
    print(f"matched per-family attack composition: " +
          " ".join(f"{bn}={tgt[bn]}" for bn in bands) + f"  (n_pos/family={sum(tgt.values())})")
    rng = np.random.default_rng(seed)
    chosen = {}
    for B in names:
        s = set()
        for bn in bands:
            pool = avail[B][bn]
            s |= {pool[i] for i in rng.choice(len(pool), size=tgt[bn], replace=False)}
        chosen[B] = s
    for B in names:
        d = fams[B]
        keep = np.ones(len(d["task"]), dtype=bool)
        for i, (t, lab, sp) in enumerate(zip(d["task"], d["label"], d["split"])):
            if sp == "test" and lab == 1 and t not in chosen[B]:
                keep[i] = False
        d["keep"] = keep
        ta = (d["split"] == "test") & (d["label"] == 1) & d["keep"]
        print(f"  {B}: kept {int(ta.sum())} test attacks, mean grade {np.nanmean(d['blat2'][ta]):.2f}")


def main():
    args = sys.argv[1:]
    seed = 0
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    T2 = os.path.join(REPO, "results", "track2_n600")
    grades = json.load(open(GRADES))

    fams = {}
    for name in FAMILIES:
        f = os.path.join(T2, name + ".npz")
        d = np.load(f, allow_pickle=True)
        susp, labs, tasks = extract_suspicion(os.path.join(REPO, "results", f"W2_{name}"))
        assert len(susp) == len(d["task_id"]) and np.all(tasks == d["task_id"]) and np.all(labs == d["label"]), \
            f"{name}: suspicion extraction not aligned to npz rows"
        gmap = grades.get(name, {})
        blat2 = np.array([gmap.get(str(t), np.nan) for t in d["task_id"]], dtype=float)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"],
                      "susp": susp, "blat2": blat2}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])

    # coverage + validity gate (test attacks)
    print(f"families={names}")
    pooled_g, pooled_s = [], []
    print("\nVALIDITY GATE — grade coverage + corr(grade, monitor suspicion) on TEST attacks:")
    for B in names:
        d = fams[B]; te = (d["split"] == "test") & (d["label"] == 1)
        g = d["blat2"][te]; s = d["susp"][te]
        m = ~np.isnan(g)
        c = np.corrcoef(g[m], s[m])[0, 1] if m.sum() > 5 else float("nan")
        pooled_g += list(g[m]); pooled_s += list(s[m])
        print(f"  {B}: graded {int(m.sum())}/{int(te.sum())}  grade mean {np.nanmean(g):.2f} "
              f"sd {np.nanstd(g):.2f}  corr(grade,susp)={c:+.3f}")
    pg, ps = np.array(pooled_g), np.array(pooled_s)
    print(f"  POOLED: n={len(pg)}  corr(grade,susp)={np.corrcoef(pg, ps)[0,1]:+.3f}  "
          f"(valid: >0 ; independent: <1)")

    # pooled tertile cut points
    q1, q2 = np.percentile(pg, [33.333, 66.667])
    bands = {"low": (-1e9, q1), "mid": (np.nextafter(q1, 1e9), q2), "high": (np.nextafter(q2, 1e9), 1e9)}
    print(f"\npooled grade tertiles: low<= {q1:.2f} < mid <= {q2:.2f} < high")
    print("TEST-attack counts by grade tertile:")
    for B in names:
        d = fams[B]; te = (d["split"] == "test") & (d["label"] == 1)
        g = d["blat2"][te]
        print(f"  {B}: " + " ".join(f"{bn}={int(((g>=lo)&(g<=hi)).sum())}" for bn, (lo, hi) in bands.items())
              + f"  (graded {int((~np.isnan(g)).sum())})")

    # SAME pipeline as track2_matrix/blatancy
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])

    print("\n================ DESIGN B: blatancy-MATCHED (independent grade) ================")
    set_keep_matched(fams, names, bands, seed)
    matrix(fams, names, heads, "blatancy-matched on INDEPENDENT Haiku obviousness grade")

    print("\n================ DESIGN A: within grade-tertile interactions ================")
    for bn in ("low", "high"):
        lo, hi = bands[bn]
        set_keep_band(fams, names, lo, hi)
        matrix(fams, names, heads, f"within-stratum: {bn} grade-tertile test attacks only")

    print("\nInterpretation: interaction stays CI>0 under matching on the INDEPENDENT,")
    print("  vendor-disjoint obviousness grader -> the equal-blatancy residual is NOT an")
    print("  artifact of the gpt-4.1-mini suspicion axis (Control 14, proxy #1). Vanishes ->")
    print("  the residual is proxy-specific; soften 'equally blatant' to the monitor's own axis.")


if __name__ == "__main__":
    raise SystemExit(main())
