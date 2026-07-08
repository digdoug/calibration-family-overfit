#!/bin/bash
# Control 15 fold — TURNKEY. Harvest every staircase LoRA checkpoint from the
# RunPod box, then print the monitor-capability staircase table (interaction +
# task-clustered 95% CI per rung, with the competence gate). Local-only after
# harvest: no GPU, no API. Mirrors analysis_deepseek_fold.sh for the frontier fold.
#
# Pod (Control 15 staircase):
#   id YOUR_POD_ID  name smt-staircase  1xA100-80GB PCIe  $1.39/hr
#   SSH YOUR_POD_IP:30002 root  /  log /root/staircase.log  /  cwd /root/smt
#   rungs: phi4_seed0, phi4_seed1, yi34b_seed0, yi34b_seed1 -> results/track2_lora/<tag>.npz
#
# Run this whenever you want the current staircase result (partial rungs show
# "PARTIAL n/4 fam"). When `staircase ALL DONE` is in the log AND all 4 npz are
# harvested -> this is the final Control 15 result; then STOP THE POD:
#   runpodctl stop pod YOUR_POD_ID      # results are on ephemeral disk — harvest FIRST (this script does)
set -uo pipefail
cd "$(dirname "$0")/.."

POD_HOST=YOUR_POD_IP
POD_PORT=30002
SRC=/root/smt/results/track2_lora
DST=results/track2_lora_staircase
mkdir -p "$DST"

echo "=== harvest staircase npz from pod (idempotent) ==="
scp -o StrictHostKeyChecking=no -o ConnectTimeout=20 -P "$POD_PORT" \
    "root@${POD_HOST}:${SRC}/phi4_seed0.npz" \
    "root@${POD_HOST}:${SRC}/phi4_seed1.npz" \
    "root@${POD_HOST}:${SRC}/yi34b_seed0.npz" \
    "root@${POD_HOST}:${SRC}/yi34b_seed1.npz" \
    "$DST/" 2>&1 | grep -vE "No such file|not a regular file" || true
echo "harvested:"; ls -la "$DST"/*.npz 2>/dev/null || echo "  (none yet)"

echo
.venv/bin/python scripts/staircase_summary.py \
  "Phi-3.5-mini:3.8B:results/track2_lora/scores_lora.npz" \
  "Phi-4:14B:${DST}/phi4_seed0.npz" \
  "Phi-4_s1:14B:${DST}/phi4_seed1.npz" \
  "Yi-1.5-Chat:34B:${DST}/yi34b_seed0.npz" \
  "Yi-1.5_s1:34B:${DST}/yi34b_seed1.npz"

# At the REAL fold (all 4 rungs harvested + `staircase ALL DONE` in the pod log),
# regenerate the staircase figure (fig6_staircase, auto-fires now that the dir
# exists). Skipped at partial harvests to avoid the heavy heatmap regen on every
# liveness check. After this: fold the table into CONTROLS_RESULTS.md Control 15 +
# PAPER §4.2 (weak-monitor-artifact rebuttal), add fig6 to the build, rebuild PDFs,
# then STOP the pod (ephemeral disk — harvest is done above):
#   runpodctl stop pod YOUR_POD_ID
N_NPZ=$(ls "$DST"/{phi4_seed0,phi4_seed1,yi34b_seed0,yi34b_seed1}.npz 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_NPZ" = "4" ]; then
  echo; echo "=== all 4 rungs harvested -> regenerating figures (fig6_staircase) ==="
  .venv/bin/python scripts/make_figures.py 2>&1 | grep -i staircase || true
else
  echo; echo "(partial: $N_NPZ/4 rungs harvested -> skipping figure regen; run make_figures.py at the full fold)"
fi
