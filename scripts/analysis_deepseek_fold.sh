#!/bin/bash
# DeepSeek 9th-family fold: run when W2_frontier_deepseek_n600 finishes on the droplet
# (ETA ~04:00-06:00 UTC Jun 14; gates nothing). DeepSeek is a 5th INDEPENDENT frontier
# vendor (DeepSeek lineage) on top of xAI/Anthropic/Google/OpenAI -> a 9-family headline
# matrix and a frontier-only-FIVE sub-matrix (permutation p-floor drops 1/24 -> 1/120).
#
# Mirrors analysis_frontier_fold.sh exactly; that script stays FROZEN as the canonical
# 8-family fold (reproducibility). This one extends the local 8-family npz with DeepSeek.
# DeepSeek pre-fold de-risk (2026-06-13): Track-1 viability AUROC 0.840
# at partial n (honest mean susp 0.032 vs attack 0.420) = viable attacker, stock prompt
# transfers, no ~0.50 wiring trap; ingests cleanly (0 empty-text, full median 10538 chars
# >> action-only 1191) in both metric modes. So the fold can embed DeepSeek unattended.
#
#   bash scripts/analysis_deepseek_fold.sh              # pull + fold deepseek
#   SKIP_PULL=1 bash scripts/analysis_deepseek_fold.sh  # deepseek already local
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source ~/.zshrc 2>/dev/null || true
[ -n "${OPENAI_API_KEY:-}" ] || { echo "NO_OPENAI_KEY"; exit 2; }
DROPLET=${DROPLET:-YOUR_DROPLET_IP}

DS_DIR=results/W2_frontier_deepseek_n600
if [ -z "${SKIP_PULL:-}" ]; then
  echo "=== PULL deepseek from droplet $(date +%H:%M:%S) ==="
  mkdir -p "$DS_DIR"
  rsync -az "root@${DROPLET}:calibration-family-overfit/${DS_DIR}/" "$DS_DIR/" || { echo PULL_FAIL; exit 1; }
fi
ls "$DS_DIR"/*.eval >/dev/null 2>&1 || { echo "NO_DEEPSEEK_EVAL — did the run finish?"; exit 1; }

# 9-family dirs = the canonical 8-family (from analysis_frontier_fold.sh) + deepseek embedded fresh.
for mode in "" _actiononly; do
  SRC=results/track2_n600_8fam${mode}
  DST=results/track2_n600_9fam${mode}
  [ -d "$SRC" ] || { echo "MISSING $SRC — run analysis_frontier_fold.sh (8-family fold) first"; exit 1; }
  mkdir -p "$DST"; cp "$SRC"/*.npz "$DST"/
  flag=""; [ -n "$mode" ] && flag="--action-only"
  echo "=== EMBED deepseek ${mode:-full} $(date +%H:%M:%S) ==="
  .venv-controlarena/bin/python scripts/track2_embed.py $flag --out track2_n600_9fam${mode} "$DS_DIR" || { echo EMBED_FAIL; exit 1; }
done

echo "=== MATRIX FULL $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_9fam
echo "=== MATRIX ACTION-ONLY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_9fam_actiononly
echo "=== PERM-TEST (Control 17) FULL + ACTION-ONLY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_permtest.py track2_n600_9fam            # 9! = 362880 perms, p-floor 2.8e-6
.venv/bin/python scripts/track2_permtest.py track2_n600_9fam_actiononly
echo "=== OOD-GATE (Control 19) FULL + ACTION-ONLY, full 9-family roster $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_ood_gate.py results/track2_n600_9fam results/track2_n600_9fam_actiononly
echo "=== RECALIBRATE (Control 20) FULL + ACTION-ONLY, full 9-family roster $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_recalibrate.py results/track2_n600_9fam results/track2_n600_9fam_actiononly
echo "=== RECAL CURVE (Control 21) FULL + ACTION-ONLY, full 9-family roster $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_recal_curve.py results/track2_n600_9fam results/track2_n600_9fam_actiononly
echo "=== SAFETY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_safety.py track2_n600_9fam --xfit

# Frontier-only sub-matrix = 5 vendors (xAI grok, Anthropic sonnet, Google gemini, OpenAI gpt-5, DeepSeek)
for mode in "" _actiononly; do
  FO=results/track2_frontieronly5${mode}; mkdir -p "$FO"
  for f in frontier_grok_n600 frontier_sonnet_n600 gemini_flash_n600 frontier_gpt5_n600 frontier_deepseek_n600; do
    cp results/track2_n600_9fam${mode}/$f.npz "$FO"/ 2>/dev/null
  done
  echo "=== FRONTIER-ONLY (5 vendors) ${mode:-full} $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/track2_matrix.py track2_frontieronly5${mode}
  .venv/bin/python scripts/track2_permtest.py track2_frontieronly5${mode}  # N=5, 5!=120 perms, p-floor 1/120=0.0083 (vs N=4's 1/24=0.042)
done
echo "=== FIGURES (9fam + frontier-5 heatmaps auto-fire now the dirs exist) $(date +%H:%M:%S) ==="
.venv/bin/python scripts/make_figures.py   # emits fig1f_matrix_9fam + fig1g_matrix_frontier5; fig5 -> 9fam
echo "=== DEEPSEEK_FOLD_DONE $(date +%H:%M:%S) ==="
echo "NEXT (manual, post-fold): judge diagonals at balanced n; add the 9-fam + frontier-5 forest rows"
echo "  to fig_forest() with their bootstrap CIs (the ONE thing not auto-wired -- CI comes from the"
echo "  track2_matrix bootstrap above); fold §4.5 prose + Table rows; rebuild PDFs."
