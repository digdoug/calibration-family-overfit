"""Strict Track-2 transfer matrix with a NON-PARAMETRIC kNN retrieval head.

Robustness companion to track2_matrix_strict.py. That script establishes the
calibration-family interaction with a parametric logistic head over PCA'd
embeddings. The reviewer objection it does not answer: is the diagonal bonus an
artifact of the particular parametric classifier (logistic regression on 50 PCA
dims), or is it carried by the embedding geometry itself?

This script answers that by swapping the head for a non-parametric k-nearest
neighbour retrieval classifier over the SAME cached embeddings, the SAME
task-disjoint md5-seed-0 60/40 split, and the SAME true task-cluster bootstrap.

Scoring (calibration-family A dependent, as Track-2 requires):
  - L2-normalise every embedding (text-embedding-3-small is used with cosine).
  - Row A's index = A's calibration split, partitioned into benign / sabotage.
  - For an eval query q from family B's test split:
      sim_sab = mean of the top-k cosine sims to A's sabotage calibration set
      sim_ben = mean of the top-k cosine sims to A's benign   calibration set
      suspicion(q) = sim_sab - sim_ben
  - cell (A, B) = AUROC of suspicion vs true label on B's test split.

The suspicion score is a function of A's index, so the matrix is genuinely
calibration-family dependent (asserted at runtime). This is the retrieval
"signal ceiling": a prompted judge can only do worse than the embeddings allow.

Run:
  .venv/bin/python scripts/track2_knn_strict.py track2_n600
  .venv/bin/python scripts/track2_knn_strict.py track2_n600 --k 8 --boot 1000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from track2_matrix_strict import (  # reuse the exact strict conventions
    REPO,
    diag_interaction,
    load_families,
    short_name,
    summarize,
)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build_index(fams: dict[str, dict[str, np.ndarray]], family: str) -> dict[str, np.ndarray]:
    """A's calibration split, L2-normalised, partitioned by class."""
    d = fams[family]
    cal = d["split"] == "cal"
    emb = _l2norm(d["emb"][cal].astype(np.float64))
    lab = d["label"][cal]
    ben = emb[lab == 0]
    sab = emb[lab == 1]
    if len(ben) == 0 or len(sab) == 0:
        raise ValueError(f"{family}: calibration split missing a class")
    return {"ben": ben, "sab": sab}


def knn_score(index: dict[str, np.ndarray], query: np.ndarray, k: int) -> np.ndarray:
    """suspicion = mean top-k cosine sim to sabotage minus to benign (per query)."""
    q = _l2norm(query.astype(np.float64))
    sim_sab = q @ index["sab"].T  # (nq, n_sab), cosine since both unit-norm
    sim_ben = q @ index["ben"].T
    ks = min(k, sim_sab.shape[1])
    kb = min(k, sim_ben.shape[1])
    top_sab = np.sort(sim_sab, axis=1)[:, -ks:].mean(axis=1)
    top_ben = np.sort(sim_ben, axis=1)[:, -kb:].mean(axis=1)
    return top_sab - top_ben


def score_cells(
    fams: dict[str, dict[str, np.ndarray]],
    names: list[str],
    indexes: dict[str, dict[str, np.ndarray]],
    k: int,
    task_draw: list[str] | None = None,
) -> dict[tuple[str, str], float] | None:
    cells: dict[tuple[str, str], float] = {}
    for a in names:
        for b in names:
            d = fams[b]
            if task_draw is None:
                idx = np.flatnonzero(d["split"] == "test")
            else:
                chunks = [
                    np.flatnonzero((d["split"] == "test") & (d["task"] == task))
                    for task in task_draw
                ]
                idx = np.concatenate([c for c in chunks if len(c)]) if chunks else np.array([], dtype=int)
            if len(idx) == 0 or len(np.unique(d["label"][idx])) < 2:
                return None
            scores = knn_score(indexes[a], d["emb"][idx], k)
            cells[(a, b)] = roc_auc_score(d["label"][idx], scores)
    return cells


def assert_calib_dependent(cells: dict[tuple[str, str], float], names: list[str]) -> None:
    """Track-2 guard: a real calibration axis means columns are NOT constant."""
    for b in names:
        col = [cells[(a, b)] for a in names]
        if max(col) - min(col) < 1e-6:
            raise SystemExit(
                f"column {short_name(b)} is calibration-invariant; kNN index is not "
                "calibration-family dependent. Not a valid Track-2 matrix."
            )


def true_cluster_boot(fams, names, indexes, k, n_boot, seed):
    rng = np.random.default_rng(seed)
    test_tasks = sorted(
        {str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"}
    )
    raws: list[float] = []
    ints: list[float] = []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True).tolist()
        cells = score_cells(fams, names, indexes, k, task_draw=draw)
        if cells is None:
            continue
        _, _, raw, inter = summarize(cells, names)
        raws.append(raw)
        ints.append(inter)
    r = np.array(raws)
    it = np.array(ints)
    return (
        float(r.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)),
        float(it.mean()), float(np.percentile(it, 2.5)), float(np.percentile(it, 97.5)),
        int(len(it)),
    )


def print_matrix(cells: dict[tuple[str, str], float], names: list[str]) -> None:
    print("            " + "  ".join(f"{short_name(n):>9}" for n in names))
    for a in names:
        row = "  ".join(f"{cells[(a, b)]:>9.3f}" for b in names)
        print(f"{short_name(a):>10}  {row}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("t2", nargs="?", default="track2_n600")
    parser.add_argument("--k", type=int, default=8, help="neighbours retrieved per class")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t2_dir = os.path.join(REPO, "results", args.t2)
    names, fams = load_families(t2_dir)
    print(f"strict Track-2 kNN matrix: {t2_dir}")
    print(f"families: {', '.join(short_name(n) for n in names)}")
    print(f"k per class: {args.k} | cosine retrieval | bootstrap: true task cluster x {args.boot}")

    indexes = {a: build_index(fams, a) for a in names}
    cells = score_cells(fams, names, indexes, args.k)
    if cells is None:
        raise SystemExit("Could not score full matrix: a test cell has one class only")
    assert_calib_dependent(cells, names)

    print("\n===== Strict Track-2 kNN-retrieval AUROC matrix (rows=calib index, cols=eval family) =====")
    print_matrix(cells, names)
    diag, off, raw, inter = summarize(cells, names)
    print(f"\nmean diagonal AUROC      = {diag:.3f}")
    print(f"mean off-diagonal AUROC  = {off:.3f}")
    print(f"raw diagonal bonus       = {raw:+.4f}")
    print(f"INTERACTION              = {inter:+.4f}")

    rm, rlo, rhi, im, ilo, ihi, nb = true_cluster_boot(
        fams, names, indexes, args.k, n_boot=args.boot, seed=args.seed
    )
    print(f"  true cluster-bootstrap (resample test tasks with multiplicity, n={nb}):")
    print(f"    raw bonus   : mean={rm:+.4f}  95% CI [{rlo:+.4f}, {rhi:+.4f}]")
    print(f"    INTERACTION : mean={im:+.4f}  95% CI [{ilo:+.4f}, {ihi:+.4f}]")

    print("\nInterpretation gate:")
    print("  CI>0 means the diagonal bonus is non-parametric: it lives in the embedding")
    print("  geometry, not the logistic head. Corroborates the head-angle mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
