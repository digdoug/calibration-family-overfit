"""Strict (leakage-free) blatancy control.

Same blatancy-matching as track2_blatancy.py (Track-1 fixed-prompt suspicion bands,
designs B=matched and A=within-stratum), but LEAKAGE-FREE: per-calibration-family
preprocessing (track2_matrix_strict.fit_row_model, scaler/PCA on family A's cal only)
and a TRUE multiplicity task-cluster bootstrap. Reuses track2_blatancy's matching
helpers verbatim so only the preprocessing + bootstrap change.

Run: source ~/.zshrc && .venv-controlarena/bin/python scripts/track2_blatancy_strict.py \
        [--t2 track2_n600|track2_n600_actiononly] [--basis row-all] [--seed N]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track2_matrix_strict as M  # noqa: E402
import track2_blatancy as B  # noqa: E402  (extract_suspicion, BANDS, set_keep_matched, set_keep_band)


def precompute_scores(fams, names, basis, pca_dim=50):
    models = {a: M.fit_row_model(fams, a, basis, pca_dim) for a in names}
    # scores[A][B] = head A (with A-only preprocessing) applied to ALL of family B's rows
    return {a: {b: M.transform_score(models[a], fams[b]["emb"]) for b in names} for a in names}


def matrix_keep(fams, names, scores, tag):
    cells = {}
    for a in names:
        for b in names:
            d = fams[b]
            te = (d["split"] == "test") & d["keep"]
            cells[(a, b)] = roc_auc_score(d["label"][te], scores[a][b][te])
    diag = np.mean([cells[(x, x)] for x in names])
    off = np.mean([cells[(a, b)] for a in names for b in names if a != b])
    inter = M.diag_interaction(cells, names)
    im, ilo, ihi, nb = boot_keep(fams, names, scores)
    print(f"  {tag}: diag {diag:.3f} | off {off:.3f} | raw {diag-off:+.4f} | "
          f"INTERACTION {inter:+.4f}  boot[{ilo:+.4f},{ihi:+.4f}] (n={nb})")
    return inter, (im, ilo, ihi)


def boot_keep(fams, names, scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    # per-(B) task -> kept test-row indices, for true multiplicity resampling
    idxmap = {b: {} for b in names}
    for b in names:
        d = fams[b]
        for i, (t, sp, k) in enumerate(zip(d["task"], d["split"], d["keep"])):
            if sp == "test" and k:
                idxmap[b].setdefault(str(t), []).append(i)
        idxmap[b] = {kk: np.array(v) for kk, v in idxmap[b].items()}
    test_tasks = sorted({t for b in names for t in idxmap[b]})
    ints = []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        cells, ok = {}, True
        for b in names:
            ii = [idxmap[b][t] for t in draw if t in idxmap[b]]
            if not ii:
                ok = False
                break
            idx = np.concatenate(ii)
            y = fams[b]["label"][idx]
            if len(np.unique(y)) < 2:
                ok = False
                break
            for a in names:
                cells[(a, b)] = roc_auc_score(y, scores[a][b][idx])
        if not ok:
            continue
        ints.append(M.diag_interaction(cells, names))
    it = np.array(ints)
    return float(it.mean()), float(np.percentile(it, 2.5)), float(np.percentile(it, 97.5)), len(it)


def main():
    args = sys.argv[1:]
    seed = 0
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    t2 = "track2_n600"
    if "--t2" in args:
        i = args.index("--t2"); t2 = args[i + 1]; del args[i:i + 2]
    basis = "row-all"
    if "--basis" in args:
        i = args.index("--basis"); basis = args[i + 1]; del args[i:i + 2]
    T2 = os.path.join(M.REPO, "results", t2)

    fams = {}
    for f in sorted(glob.glob(os.path.join(T2, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        susp, labs, tasks = B.extract_suspicion(os.path.join(M.REPO, "results", f"W2_{name}"))
        assert len(susp) == len(d["task_id"]) and np.all(tasks == d["task_id"]) and np.all(labs == d["label"]), \
            f"{name}: suspicion extraction not aligned to npz rows"
        fams[name] = {"emb": d["emb"].astype(np.float32), "label": d["label"].astype(int),
                      "task": d["task_id"].astype(object), "susp": susp}
    names = list(fams)
    for d in fams.values():
        d["split"] = np.array([B.split_of(str(t)) for t in d["task"]], dtype=object)
        d["keep"] = np.ones(len(d["task"]), dtype=bool)

    print(f"STRICT blatancy control | t2={t2} basis={basis} | families={[M.short_name(n) for n in names]}")
    scores = precompute_scores(fams, names, basis)

    print("\n== DESIGN B: blatancy-MATCHED (identical suspicion composition across families) ==")
    B.set_keep_matched(fams, names, seed)
    matrix_keep(fams, names, scores, "blatancy-matched")

    print("\n== DESIGN A: within-stratum ==")
    for bn in ("low", "high"):
        B.set_keep_band(fams, names, B.BANDS[bn])
        matrix_keep(fams, names, scores, f"within-stratum {bn}")


if __name__ == "__main__":
    raise SystemExit(main())
