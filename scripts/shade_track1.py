"""SHADE Track-1 (fixed monitor, operational calibration transfer) -- FREE slice.

The bundled SHADE transcripts ship each trajectory scored 0-100 by several
monitor models (the original experiments' monitoring_response, parsed at export).
For each monitor M with full coverage of the pre-registered families/tasks:

  - per eval family B: honest-vs-attack AUROC of M's scores (column structure
    only -- a FIXED monitor has no calibration-family AUROC axis, the design doc s1a;
    we report this as capability context, NOT as a transfer matrix)
  - Track-1 tau-transfer: tau_A(q) = (1-q) quantile of M's scores on A's
    CALIBRATION-split benign; TPR/FPR of tau_A applied to B's TEST split
    (rows = calibration family, cols = eval family). This IS calibration
    transfer (the threshold moves), mirroring the code substrate's Track-1.

Same md5-seed-0 60/40 task split, same common-task set as the Track-2 matrix
(read from the track2_shade npz task lists so both tracks use identical data).
Cluster bootstrap by test task for the tau-transfer diagonal bonus.

Output: results/shade_track1.json (aggregate only) + stdout report.

Run: .venv-controlarena/bin/python scripts/shade_track1.py [--q 0.01] [--hint]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USE_HINT = "--hint" in sys.argv
EXPORT = os.path.join(REPO, "results", "shade_export" + ("_hint" if USE_HINT else ""))
T2DIR = os.path.join(REPO, "results", "track2_shade" + ("_hint" if USE_HINT else ""))
Q = float(sys.argv[sys.argv.index("--q") + 1]) if "--q" in sys.argv else 0.01
FAMILIES = ["shade_claude35s", "shade_gpt41", "shade_gemini25f"]
CAL_FRAC = 60


def split_of(task, seed=0):
    h = int(hashlib.md5(f"{seed}:{task}".encode()).hexdigest(), 16)
    return "cal" if (h % 100) < CAL_FRAC else "test"


def diag_interaction(cells, names):
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


def main() -> int:
    # common task set = exactly what the Track-2 matrix used
    common = set()
    for f in glob.glob(os.path.join(T2DIR, "*.npz")):
        common |= set(np.load(f, allow_pickle=True)["task_id"].tolist())
    if not common:
        print(f"no npz in {T2DIR} yet -- run shade_embed.py first"); return 1
    print(f"common tasks: {len(common)}")

    # records: fam -> list of (task, label, split, {monitor: score})
    recs = {}
    for fam in FAMILIES:
        rows = []
        for mode in ("benign", "attack"):
            for line in open(os.path.join(EXPORT, mode, f"{fam}.jsonl")):
                r = json.loads(line)
                if r["task_id"] in common:
                    rows.append((r["task_id"], r["label"], split_of(r["task_id"]),
                                 r["monitor_scores"]))
        recs[fam] = rows

    # monitors with full coverage: scored >=80% of rows in every family
    cov = {}
    for fam, rows in recs.items():
        for _, _, _, ms in rows:
            for m in ms:
                cov.setdefault(m, {f: 0 for f in FAMILIES})[fam] += 1
    monitors = [m for m, c in cov.items()
                if all(c[f] >= 0.8 * len(recs[f]) for f in FAMILIES)]
    print("monitor coverage (rows scored / rows):")
    for m, c in sorted(cov.items()):
        note = " <- USED" if m in monitors else ""
        print(f"  {m:34s} " + " ".join(
            f"{c[f]}/{len(recs[f])}" for f in FAMILIES) + note)

    out = {"q": Q, "families": FAMILIES, "monitors": {}}
    rng = np.random.default_rng(0)
    for M in monitors:
        # arrays per family (rows where M scored)
        fam_d = {}
        for fam, rows in recs.items():
            sel = [(t, l, sp, ms[M]) for t, l, sp, ms in rows if M in ms]
            fam_d[fam] = {
                "task": np.array([x[0] for x in sel], dtype=object),
                "label": np.array([x[1] for x in sel]),
                "split": np.array([x[2] for x in sel]),
                "score": np.array([x[3] for x in sel], dtype=float),
            }
        # per-eval-family AUROC (test split; column view, no calibration axis)
        auroc = {}
        for fam, d in fam_d.items():
            te = d["split"] == "test"
            if len(np.unique(d["label"][te])) == 2:
                auroc[fam] = float(roc_auc_score(d["label"][te], d["score"][te]))
        # tau-transfer TPR/FPR
        tau = {A: float(np.quantile(
            fam_d[A]["score"][(fam_d[A]["split"] == "cal") & (fam_d[A]["label"] == 0)],
            1 - Q)) for A in FAMILIES}
        TPR, FPR = {}, {}
        for A in FAMILIES:
            for B in FAMILIES:
                d = fam_d[B]; te = d["split"] == "test"
                att = te & (d["label"] == 1); ben = te & (d["label"] == 0)
                TPR[(A, B)] = float(np.mean(d["score"][att] >= tau[A]))
                FPR[(A, B)] = float(np.mean(d["score"][ben] >= tau[A]))
        diag = float(np.mean([TPR[(x, x)] for x in FAMILIES]))
        off = float(np.mean([TPR[(a, b)] for a in FAMILIES for b in FAMILIES if a != b]))
        inter = diag_interaction(TPR, FAMILIES)
        # cluster bootstrap by test task (raw bonus + interaction)
        test_tasks = sorted({t for d in fam_d.values()
                             for t, sp in zip(d["task"], d["split"]) if sp == "test"})
        boots, ints = [], []
        for _ in range(1000):
            keep = set(rng.choice(test_tasks, size=len(test_tasks), replace=True).tolist())
            cells, ok = {}, True
            for A in FAMILIES:
                for B in FAMILIES:
                    d = fam_d[B]
                    m = (d["split"] == "test") & np.array([t in keep for t in d["task"]])
                    att = m & (d["label"] == 1); ben = m & (d["label"] == 0)
                    if att.sum() == 0 or ben.sum() == 0:
                        ok = False; break
                    cells[(A, B)] = float(np.mean(d["score"][att] >= tau[A]))
                if not ok:
                    break
            if not ok:
                continue
            bd = np.mean([cells[(x, x)] for x in FAMILIES])
            bo = np.mean([cells[(a, b)] for a in FAMILIES for b in FAMILIES if a != b])
            boots.append(bd - bo); ints.append(diag_interaction(cells, FAMILIES))
        blo, bhi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        ilo, ihi = (np.percentile(ints, [2.5, 97.5]) if ints else (np.nan, np.nan))

        print(f"\n=== monitor {M} (q={Q*100:g}%) ===")
        print("  per-eval-family AUROC (fixed monitor; column view): " +
              ", ".join(f"{f.replace('shade_','')}={a:.3f}" for f, a in auroc.items()))
        print("  tau-transfer TPR matrix (rows=tau from A, cols=eval B):")
        for A in FAMILIES:
            print(f"    {A.replace('shade_',''):12s} " + "  ".join(
                f"{TPR[(A,B)]*100:5.1f}" for B in FAMILIES))
        print(f"  diag {diag*100:.1f}% vs off {off*100:.1f}% -> raw bonus "
              f"{(diag-off)*100:+.1f} pts [boot {blo*100:+.1f}, {bhi*100:+.1f}] | "
              f"interaction {inter*100:+.1f} pts [{ilo*100:+.1f}, {ihi*100:+.1f}] "
              f"(n_boot={len(boots)})")
        out["monitors"][M] = {
            "auroc_by_eval_family": auroc,
            "TPR": {f"{a}->{b}": TPR[(a, b)] for a in FAMILIES for b in FAMILIES},
            "FPR": {f"{a}->{b}": FPR[(a, b)] for a in FAMILIES for b in FAMILIES},
            "diag": diag, "off": off, "raw_bonus": diag - off,
            "raw_bonus_ci": [float(blo), float(bhi)],
            "interaction": inter, "interaction_ci": [float(ilo), float(ihi)],
            "n_boot": len(boots),
        }

    path = os.path.join(REPO, "results",
                        "shade_track1" + ("_hint" if USE_HINT else "") + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
