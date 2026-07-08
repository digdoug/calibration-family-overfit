"""Strict (non-transductive) Track-2 transfer matrix — the leak-free headline basis.

This is the non-transductive counterpart to track2_matrix.py. The original
headline script fits one shared scaler/PCA on pooled calibration embeddings from
all families, which is fine for a descriptive representation analysis but leaks
the target family into the feature basis for a deployer-facing calibrate-on-A /
test-on-B claim.

This script fits preprocessing separately for each calibration row A, using only
A's calibration split by default. It then trains A's head on A-cal and scores
every evaluation family B with that A-only preprocessing/head. It also uses a
true task-cluster bootstrap with multiplicity, rather than converting sampled
task IDs to a set.

Run:
  .venv/bin/python scripts/track2_matrix_strict.py track2_n600
  .venv/bin/python scripts/track2_matrix_strict.py track2_n600_actiononly --basis row-benign
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_FRAC = 60


def split_of(task: str, seed: int = 0) -> str:
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells: dict[tuple[str, str], float], names: list[str]) -> float:
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean()
    row = arr.mean(axis=1)
    col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def short_name(name: str) -> str:
    return (
        name.replace("ow_", "")
        .replace("_n200", "")
        .replace("_n600", "")
        .replace("frontier_", "")
    )


@dataclass
class RowModel:
    scaler: StandardScaler
    pca: PCA | None
    head: LogisticRegression


def load_families(t2_dir: str) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    fams: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(os.path.join(t2_dir, "*.npz"))):
        name = os.path.basename(path)[:-4]
        d = np.load(path, allow_pickle=True)
        fams[name] = {
            "emb": d["emb"].astype(np.float32),
            "label": d["label"].astype(int),
            "task": d["task_id"].astype(object),
        }
        fams[name]["split"] = np.array([split_of(str(t)) for t in fams[name]["task"]], dtype=object)
    names = list(fams)
    if not names:
        raise SystemExit(f"No .npz files found in {t2_dir}")
    return names, fams


def fit_row_model(
    fams: dict[str, dict[str, np.ndarray]],
    family: str,
    basis: str,
    pca_dim: int,
) -> RowModel:
    d = fams[family]
    cal = d["split"] == "cal"
    if basis == "row-benign":
        basis_mask = cal & (d["label"] == 0)
    elif basis == "row-all":
        basis_mask = cal
    else:
        raise ValueError(f"unknown basis: {basis}")

    x_basis = d["emb"][basis_mask]
    x_train = d["emb"][cal]
    y_train = d["label"][cal]
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{family}: calibration split has one class only")

    scaler = StandardScaler().fit(x_basis)
    x_basis_s = scaler.transform(x_basis)
    x_train_s = scaler.transform(x_train)
    n_components = min(pca_dim, x_basis_s.shape[0] - 1, x_basis_s.shape[1])
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=0).fit(x_basis_s)
        x_train_f = pca.transform(x_train_s)
    else:
        pca = None
        x_train_f = x_train_s

    head = LogisticRegression(C=1.0, max_iter=3000).fit(x_train_f, y_train)
    return RowModel(scaler=scaler, pca=pca, head=head)


def transform_score(model: RowModel, emb: np.ndarray) -> np.ndarray:
    x = model.scaler.transform(emb)
    if model.pca is not None:
        x = model.pca.transform(x)
    return model.head.decision_function(x)


def score_cells(
    fams: dict[str, dict[str, np.ndarray]],
    names: list[str],
    row_models: dict[str, RowModel],
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
            scores = transform_score(row_models[a], d["emb"][idx])
            cells[(a, b)] = roc_auc_score(d["label"][idx], scores)
    return cells


def summarize(cells: dict[tuple[str, str], float], names: list[str]) -> tuple[float, float, float, float]:
    diag = float(np.mean([cells[(x, x)] for x in names]))
    off = float(np.mean([cells[(a, b)] for a in names for b in names if a != b]))
    return diag, off, diag - off, diag_interaction(cells, names)


def true_cluster_boot(
    fams: dict[str, dict[str, np.ndarray]],
    names: list[str],
    row_models: dict[str, RowModel],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float, float, float, int]:
    rng = np.random.default_rng(seed)
    test_tasks = sorted(
        {str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"}
    )
    raws: list[float] = []
    ints: list[float] = []
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True).tolist()
        cells = score_cells(fams, names, row_models, task_draw=draw)
        if cells is None:
            continue
        _, _, raw, inter = summarize(cells, names)
        raws.append(raw)
        ints.append(inter)
    r = np.array(raws)
    it = np.array(ints)
    return (
        float(r.mean()),
        float(np.percentile(r, 2.5)),
        float(np.percentile(r, 97.5)),
        float(it.mean()),
        float(np.percentile(it, 2.5)),
        float(np.percentile(it, 97.5)),
        int(len(it)),
    )


def print_matrix(cells: dict[tuple[str, str], float], names: list[str]) -> None:
    print("            " + "  ".join(f"{short_name(n):>9}" for n in names))
    for a in names:
        row = "  ".join(f"{cells[(a, b)]:>9.3f}" for b in names)
        print(f"{short_name(a):>10}  {row}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("t2", nargs="?", default="track2_n600", help="results/<dir> containing family .npz files")
    parser.add_argument(
        "--basis",
        choices=["row-all", "row-benign"],
        default="row-all",
        help="data used to fit each row's scaler/PCA; both exclude all target-family data",
    )
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t2_dir = os.path.join(REPO, "results", args.t2)
    names, fams = load_families(t2_dir)
    print(f"strict Track-2 matrix: {t2_dir}")
    print(f"families: {', '.join(short_name(n) for n in names)}")
    print(f"basis: {args.basis} | pca_dim: {args.pca_dim} | bootstrap: true task cluster x {args.boot}")

    row_models = {a: fit_row_model(fams, a, args.basis, args.pca_dim) for a in names}
    cells = score_cells(fams, names, row_models)
    if cells is None:
        raise SystemExit("Could not score full matrix: at least one test cell has one class only")

    print("\n===== Strict Track-2 AUROC matrix (rows=calib-only head, cols=eval family) =====")
    print_matrix(cells, names)
    diag, off, raw, inter = summarize(cells, names)
    print(f"\nmean diagonal AUROC      = {diag:.3f}")
    print(f"mean off-diagonal AUROC  = {off:.3f}")
    print(f"raw diagonal bonus       = {raw:+.4f}")
    print(f"INTERACTION              = {inter:+.4f}")

    rm, rlo, rhi, im, ilo, ihi, nb = true_cluster_boot(
        fams, names, row_models, n_boot=args.boot, seed=args.seed
    )
    print(f"  true cluster-bootstrap (resample test tasks with multiplicity, n={nb}):")
    print(f"    raw bonus   : mean={rm:+.4f}  95% CI [{rlo:+.4f}, {rhi:+.4f}]")
    print(f"    INTERACTION : mean={im:+.4f}  95% CI [{ilo:+.4f}, {ihi:+.4f}]")

    print("\nInterpretation gate:")
    print("  CI>0 here closes the transductive-feature-basis objection.")
    print("  Collapse/null means the current headline is a shared-representation artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
