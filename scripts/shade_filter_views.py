"""Materialize SHADE filtered sensitivity views from the master npz (no re-embed).

From results/track2_shade*/{fam}.npz (which carry main_ok/keep_side/keep_strict
masks from shade_export), write results/<dir>_filtered/{fam}.npz keeping only
the SHADE-native success-filtered rows (benign: main-task success; attack:
side-task success) so track2_matrix.py / track2_safety.py run on them unchanged.

Run: python3 scripts/shade_filter_views.py [track2_shade track2_shade_actiononly ...]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = [a for a in sys.argv[1:] if not a.startswith("--")] or [
    "track2_shade", "track2_shade_actiononly"]

for d in DIRS:
    src = os.path.join(REPO, "results", d)
    dst = os.path.join(REPO, "results", d + "_filtered")
    os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        keep = z["keep_side"].astype(bool)
        np.savez(os.path.join(dst, os.path.basename(f)),
                 emb=z["emb"][keep], label=z["label"][keep],
                 task_id=z["task_id"][keep])
        lab = z["label"][keep]
        print(f"{d}/{os.path.basename(f)}: kept {keep.sum()}/{len(keep)} "
              f"({int(lab.sum())} attack / {int((lab == 0).sum())} benign, "
              f"{len(set(z['task_id'][keep].tolist()))} tasks)")
