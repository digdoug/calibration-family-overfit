"""Text-free per-transcript score tables for the supplementary code appendix.

One CSV per roster directory. Per transcript: family, row index, task id, split,
label, and each calibration-family head's suspicion score on EVERY reported basis:
strict row-all (strict_*: per-row scaler/PCA/head fit on the calibration family's
full calibration split, track2_matrix_strict.py default, the headline basis),
strict row-benign (strictb_*: scaler/PCA fit on the calibration family's BENIGN
calibration rows only; the pre-registered basis for the bigcodebench action-only
number, pre-registered), and transductive (trans_*: pooled
scaler/PCA, track2_matrix.py). These columns are sufficient to reproduce every
embedding-head matrix, interaction, cluster-bootstrap CI, and catch rate in the
paper with zero transcript text, zero task text, and zero model output.

The LoRA monitor's cross-score npz files are already per-transcript long-format
(score/label/task/calib/eval); they are exported to CSV as-is.

SHADE is EXCLUDED: its data agreement allows
aggregate numbers only, so no per-transcript rows leave the repo.

Run: .venv/bin/python scripts/dump_score_tables.py
Output: results/score_tables/*.csv (+ MANIFEST.md)
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track2_matrix_strict import fit_row_model, load_families, short_name, transform_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "score_tables")

# Every embedding-head roster the paper reports (the paper's appendix registry).
# SHADE dirs are deliberately absent (aggregate-only data agreement).
ROSTER_DIRS = [
    # APPS, four open families (frozen primary) + action-only
    "track2_n600", "track2_n600_actiononly",
    # APPS frontier expansions 5..9 families + action-only
    "track2_n600_5fam", "track2_n600_5fam_actiononly",
    "track2_n600_6fam", "track2_n600_6fam_actiononly",
    "track2_n600_7fam", "track2_n600_7fam_actiononly",
    "track2_n600_8fam", "track2_n600_8fam_actiononly",
    "track2_n600_9fam", "track2_n600_9fam_actiononly",
    # frontier-only sub-matrices
    "track2_frontieronly3", "track2_frontieronly3_actiononly",
    "track2_frontieronly4", "track2_frontieronly4_actiononly",
    "track2_frontieronly5", "track2_frontieronly5_actiononly",
    # bigcodebench second substrate (3-family box rerun + earlier 2-family replication)
    "track2_bcb_box", "track2_bcb_box_actiononly", "track2_bcb",
    # embedder-vendor controls
    "track2_3large", "track2_ada002", "track2_n600_nomic", "track2_n600_nomic_actiononly",
]

LORA_NPZ = sorted(
    glob.glob(os.path.join(REPO, "results", "track2_lora", "*.npz"))
    + glob.glob(os.path.join(REPO, "results", "track2_lora_staircase", "*.npz"))
)


def transductive_models(fams: dict, names: list[str]):
    """Pooled scaler/PCA-50 + per-family heads, exactly as track2_matrix.py."""
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in fams.values()])
    scaler = StandardScaler().fit(cal_emb)
    n_comp = min(50, cal_emb.shape[0] - 1, cal_emb.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(scaler.transform(cal_emb))
    proj = {n: pca.transform(scaler.transform(fams[n]["emb"])) for n in names}
    heads = {}
    for a in names:
        cal = fams[a]["split"] == "cal"
        heads[a] = LogisticRegression(C=1.0, max_iter=3000).fit(proj[a][cal], fams[a]["label"][cal])
    return proj, heads


def success_meta(t2: str, fams: dict, names: list[str]):
    """Main/side-task success flags (from the generation grader) where the meta file
    exists and joins exactly: (task, label) must be a unique key on both sides and
    every npz row must have a meta row (the meta may be a superset; the embedding
    step drops empty-output transcripts). Returns per-family dicts keyed by
    (task, label)."""
    import json

    meta_path = os.path.join(REPO, "results", t2, "sample_meta.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path))
    out = {}
    for n in names:
        rows = meta.get(n)
        if rows is None:
            return None
        keyed = {(str(r["task"]), int(r["label"])): r for r in rows}
        npz_keys = {(str(t), int(y)) for t, y in zip(fams[n]["task"], fams[n]["label"])}
        if len(keyed) != len(rows) or len(npz_keys) != len(fams[n]["label"]) \
                or not npz_keys <= set(keyed):
            return None
        out[n] = keyed
    return out


def dump_roster(t2: str) -> tuple[str, int, list[str]]:
    names, fams = load_families(os.path.join(REPO, "results", t2))
    strict = {a: fit_row_model(fams, a, "row-all", 50) for a in names}
    strictb = {a: fit_row_model(fams, a, "row-benign", 50) for a in names}
    proj, trans = transductive_models(fams, names)
    meta = success_meta(t2, fams, names)
    shorts = [short_name(n) for n in names]
    cols = (["family", "row", "task_id", "split", "label"]
            + [f"strict_{s}" for s in shorts] + [f"strictb_{s}" for s in shorts]
            + [f"trans_{s}" for s in shorts]
            + (["main_ok", "side_ok"] if meta else []))
    path = os.path.join(OUT, f"{t2}.csv")
    n_rows = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for b in names:
            d = fams[b]
            s_scores = {a: transform_score(strict[a], d["emb"]) for a in names}
            sb_scores = {a: transform_score(strictb[a], d["emb"]) for a in names}
            t_scores = {a: trans[a].decision_function(proj[b]) for a in names}
            for i in range(len(d["label"])):
                row = [short_name(b), i, str(d["task"][i]), d["split"][i], int(d["label"][i])]
                row += [f"{s_scores[a][i]:.6f}" for a in names]
                row += [f"{sb_scores[a][i]:.6f}" for a in names]
                row += [f"{t_scores[a][i]:.6f}" for a in names]
                if meta:
                    m = meta[b][(str(d["task"][i]), int(d["label"][i]))]
                    row += [int(m["main"]), int(m["side"])]
                w.writerow(row)
                n_rows += 1
    return path, n_rows, shorts


def dump_lora(npz_path: str) -> tuple[str, int]:
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.keys())
    assert {"score", "label", "task", "calib", "eval"} <= keys, \
        f"{npz_path}: unexpected LoRA npz keys {sorted(keys)}"
    stem = os.path.basename(npz_path)[:-4]
    parent = os.path.basename(os.path.dirname(npz_path))
    path = os.path.join(OUT, f"lora_{parent.replace('track2_lora', 'lora').strip('_')}_{stem}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["calib_family", "eval_family", "task_id", "label", "score"])
        for i in range(len(d["score"])):
            w.writerow([str(d["calib"][i]), str(d["eval"][i]), str(d["task"][i]),
                        int(d["label"][i]), f"{float(d['score'][i]):.6f}"])
    return path, len(d["score"])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    manifest = ["# Score-table manifest (text-free per-transcript scores)", "",
                "Produced by scripts/dump_score_tables.py. SHADE excluded (aggregate-only",
                "data agreement).", "",
                "Score bases: strict_* = row-all (headline strict basis); strictb_* =",
                "row-benign (the pre-registered basis for the bigcodebench action-only",
                "number, pre-registered); trans_* = transductive",
                "pooled basis. Matrices/interactions are computed on split == test rows.", ""]
    for t2 in ROSTER_DIRS:
        path, n, shorts = dump_roster(t2)
        line = f"- {os.path.basename(path)}: {n} rows, families {', '.join(shorts)}"
        print(line)
        manifest.append(line)
    for npz in LORA_NPZ:
        path, n = dump_lora(npz)
        line = f"- {os.path.basename(path)}: {n} rows (LoRA cross-score, long format)"
        print(line)
        manifest.append(line)
    open(os.path.join(OUT, "MANIFEST.md"), "w").write("\n".join(manifest) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
