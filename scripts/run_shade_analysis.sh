#!/usr/bin/env bash
# SHADE analysis battery: matrices (full/action-only, unfiltered/filtered),
# safety translation, Track-1 bonus. All outputs aggregate-only; logs+json in
# gitignored results/. Run after shade_embed.py has written both npz dirs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-controlarena/bin/python

echo "== filter views =="
python3 scripts/shade_filter_views.py track2_shade track2_shade_actiononly \
  | tee results/shade_filter_views.log

for d in track2_shade track2_shade_actiononly track2_shade_filtered track2_shade_actiononly_filtered; do
  echo "== matrix: $d =="
  $PY scripts/track2_matrix.py "$d" 2>&1 | tee "results/shade_matrix_${d}.log"
done

echo "== safety translation (budget-enforced etc.) =="
$PY scripts/track2_safety.py track2_shade --qs 0.01,0.02,0.05 2>&1 \
  | tee results/shade_safety.log
$PY scripts/track2_safety.py track2_shade_actiononly --qs 0.01,0.02,0.05 2>&1 \
  | tee results/shade_safety_actiononly.log

echo "== track-1 (bundled monitor scores) =="
$PY scripts/shade_track1.py 2>&1 | tee results/shade_track1.log

echo "ALL DONE"
