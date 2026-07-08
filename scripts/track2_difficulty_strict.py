"""Strict (leakage-free) difficulty control (main-task-success matched).

Same capability-matching as track2_difficulty.py (subsample TEST attacks to an
identical main_task_success composition across families), but per-calibration-family
preprocessing + true multiplicity bootstrap. Reuses track2_difficulty's matching
and track2_blatancy_strict's strict scoring harness.

Run: .venv-controlarena/bin/python scripts/track2_difficulty_strict.py [--t2 track2|track2_actiononly] [--seed N]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402
import track2_difficulty as D  # noqa: E402
import track2_blatancy_strict as BS  # noqa: E402


def main():
    args = sys.argv[1:]
    seed = 0
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    t2 = "track2"  # n200, where npz family names match the W2_*_n200 success dirs natively
    if "--t2" in args:
        i = args.index("--t2"); t2 = args[i + 1]; del args[i:i + 2]
    T2 = os.path.join(M.REPO, "results", t2)

    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"].astype(np.float32), "label": d["label"].astype(int),
                      "task": d["task_id"].astype(object)}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([M.split_of(str(t)) for t in d["task"]], dtype=object)
        d["keep"] = np.ones(len(d["task"]), dtype=bool)

    succ = D.attack_main_success()
    print(f"STRICT difficulty | t2={t2} | families={[M.short_name(n) for n in names]}")
    scores = BS.precompute_scores(fams, names, "row-all")
    D.build_matched_keep(fams, names, succ, seed)
    BS.matrix_keep(fams, names, scores, "difficulty-matched (main-success matched)")


if __name__ == "__main__":
    raise SystemExit(main())
