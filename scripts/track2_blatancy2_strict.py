"""Strict (leakage-free) counterpart of Control 14b (track2_blatancy2.py).

Control 14b matches the Track-2 interaction on the INDEPENDENT vendor-disjoint Claude-Haiku
obviousness grade, but using the LEAKY transductive pipeline (one shared scaler/PCA fit on the
pooled four-family calibration embeddings, so the target family's distribution leaks into the
feature basis). The headline matrix was re-piped to the strict NON-transductive basis
(track2_matrix_strict: per-calibration-family scaler/PCA, target family absent from the whole
pipeline). This script re-runs the SAME independent-grade matching on that strict basis + a true
multiplicity task-cluster bootstrap, so the most-important confound class is reported on the same
leak-free pipeline as the headline.

Only the preprocessing/bootstrap change vs track2_blatancy2: the grade-banding (set_keep_band /
set_keep_matched on the Haiku grade) and the matrix/boot machinery are reused verbatim.

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_blatancy2_strict.py \
        [--basis row-all|row-benign] [--seed N]
"""
from __future__ import annotations
import glob, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402  (fit_row_model, transform_score, diag_interaction, short_name)
import track2_blatancy_strict as BS  # noqa: E402  (precompute_scores, matrix_keep, boot_keep)
import track2_blatancy2 as B2  # noqa: E402  (set_keep_band, set_keep_matched on the grade)
from track2_blatancy import split_of, extract_suspicion  # noqa: E402

REPO = M.REPO
GRADES = os.path.join(REPO, "results", "blatancy2", "grades.json")
FAMILIES = B2.FAMILIES


def main():
    args = sys.argv[1:]
    seed = 0
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    basis = "row-all"
    if "--basis" in args:
        i = args.index("--basis"); basis = args[i + 1]; del args[i:i + 2]
    T2 = os.path.join(REPO, "results", "track2_n600")
    grades = json.load(open(GRADES))

    fams = {}
    for name in FAMILIES:
        d = np.load(os.path.join(T2, name + ".npz"), allow_pickle=True)
        susp, labs, tasks = extract_suspicion(os.path.join(REPO, "results", f"W2_{name}"))
        assert len(susp) == len(d["task_id"]) and np.all(tasks == d["task_id"]) and np.all(labs == d["label"]), \
            f"{name}: suspicion extraction not aligned to npz rows"
        gmap = grades.get(name, {})
        blat2 = np.array([gmap.get(str(t), np.nan) for t in d["task_id"]], dtype=float)
        fams[name] = {"emb": d["emb"].astype(np.float32), "label": d["label"].astype(int),
                      "task": d["task_id"].astype(object), "susp": susp, "blat2": blat2}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([split_of(str(t)) for t in d["task"]], dtype=object)
        d["keep"] = np.ones(len(d["task"]), dtype=bool)

    # validity gate + pooled tertiles (identical to track2_blatancy2)
    print(f"STRICT Control 14b | basis={basis} | families={[M.short_name(n) for n in names]}")
    pooled_g, pooled_s = [], []
    print("\nVALIDITY GATE — corr(grade, monitor suspicion) on TEST attacks:")
    for B in names:
        d = fams[B]; te = (d["split"] == "test") & (d["label"] == 1)
        g = d["blat2"][te]; s = d["susp"][te]; m = ~np.isnan(g)
        c = np.corrcoef(g[m], s[m])[0, 1] if m.sum() > 5 else float("nan")
        pooled_g += list(g[m]); pooled_s += list(s[m])
        print(f"  {B}: graded {int(m.sum())}/{int(te.sum())}  grade mean {np.nanmean(g):.2f}  corr={c:+.3f}")
    pg, ps = np.array(pooled_g), np.array(pooled_s)
    print(f"  POOLED: n={len(pg)}  corr(grade,susp)={np.corrcoef(pg, ps)[0,1]:+.3f}")
    q1, q2 = np.percentile(pg, [33.333, 66.667])
    bands = {"low": (-1e9, q1), "mid": (np.nextafter(q1, 1e9), q2), "high": (np.nextafter(q2, 1e9), 1e9)}
    print(f"pooled grade tertiles: low<= {q1:.2f} < mid <= {q2:.2f} < high")

    scores = BS.precompute_scores(fams, names, basis)

    print("\n== DESIGN B: blatancy-MATCHED on INDEPENDENT Haiku grade (strict basis) ==")
    B2.set_keep_matched(fams, names, bands, seed)
    BS.matrix_keep(fams, names, scores, "blatancy2-matched (strict)")

    print("\n== DESIGN A: within grade-tertile (strict basis) ==")
    for bn in ("low", "high"):
        lo, hi = bands[bn]
        B2.set_keep_band(fams, names, lo, hi)
        BS.matrix_keep(fams, names, scores, f"within-stratum {bn} (strict)")

    print("\nInterpretation: interaction CI>0 under matching on the INDEPENDENT vendor-disjoint")
    print("  grade AND the leak-free per-family basis -> the equal-blatancy residual is not an")
    print("  artifact of either the gpt-4.1-mini suspicion axis OR the transductive feature basis.")


if __name__ == "__main__":
    raise SystemExit(main())
