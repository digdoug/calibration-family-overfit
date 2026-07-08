#!/usr/bin/env python3
"""Minimum-detectable-effect for the SHADE primary-cut interaction.

Reuses the EXACT multiplicity-preserving cluster bootstrap from
``track2_matrix`` (same loading, same head fit, same resample) so the
reported SE is byte-faithful to the grounded SHADE CI. Reports the
bootstrap SE of the interaction and the two-sided alpha=0.05, 80%-power
minimum detectable effect (MDE = (z_.975 + z_.80) * SE), to substantiate
the "underpowered to resolve an effect this small" framing with a number.

$0, deterministic (seed 0), cached embeddings only. Aggregate output.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _clusterboot import task_rows, draw_index  # noqa: E402
from track2_matrix import T2, split_of, diag_interaction  # noqa: E402


def load(sub):
    fams = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(T2), sub, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in fams.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    names = list(fams)
    for A in names:
        d = fams[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    return fams, names, heads


def boot_interactions(fams, names, heads, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    test_tasks = sorted({t for d in fams.values()
                         for t, sp in zip(d["task"], d["split"]) if sp == "test"})
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
                cells[(A, B)] = roc_auc_score(y, heads[A].decision_function(d["p"][idx]))
            if not ok:
                break
        if not ok:
            continue
        ints.append(diag_interaction(cells, names))
    return np.array(ints), len(test_tasks)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "track2_shade"
    fams, names, heads = load(sub)
    ints, n_test = boot_interactions(fams, names, heads)
    se = float(ints.std(ddof=1))
    point = float(np.mean(ints))
    z975, z80 = 1.95996, 0.84162
    mde = (z975 + z80) * se
    print(f"SHADE {sub}: {n_test} test tasks, {len(ints)} valid resamples")
    print(f"  interaction boot mean   = {point:+.4f}")
    print(f"  bootstrap SE            = {se:.4f}")
    print(f"  95% CI (percentile)     = [{np.percentile(ints,2.5):+.4f}, {np.percentile(ints,97.5):+.4f}]")
    print(f"  MDE (two-sided a=.05, 80% power) = {mde:+.4f}")
    print(f"  observed +0.0302 is {'BELOW' if 0.0302 < mde else 'above'} the MDE "
          f"-> {'underpowered to resolve an effect this small' if 0.0302 < mde else 'resolvable'}")


if __name__ == "__main__":
    raise SystemExit(main())
