#!/bin/bash
# Frontier fold: run when the gpt-5 n=600 generation finishes on the droplet.
# Reuses the 7-family embeddings already staged (6 canon + grok); embeds ONLY
# the new gpt-5 family, then rebuilds the matrix / safety / frontier-only views.
# deepseek (optional 5th) folds the same way if it lands: add its W2 dir below.
#
#   bash scripts/analysis_frontier_fold.sh            # pull + fold gpt-5
#   SKIP_PULL=1 bash scripts/analysis_frontier_fold.sh  # gpt-5 already local
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source ~/.zshrc 2>/dev/null || true
[ -n "${OPENAI_API_KEY:-}" ] || { echo "NO_OPENAI_KEY"; exit 2; }
DROPLET=${DROPLET:-YOUR_DROPLET_IP}

GPT5_DIR=results/W2_frontier_gpt5_n600
if [ -z "${SKIP_PULL:-}" ]; then
  echo "=== PULL gpt-5 from droplet $(date +%H:%M:%S) ==="
  mkdir -p "$GPT5_DIR"
  rsync -az "root@${DROPLET}:calibration-family-overfit/${GPT5_DIR}/" "$GPT5_DIR/" || { echo PULL_FAIL; exit 1; }
fi
ls "$GPT5_DIR"/*.eval >/dev/null 2>&1 || { echo "NO_GPT5_EVAL — did the run finish?"; exit 1; }

# 8-family dirs = the staged 7-family (6 canon + grok) + gpt-5 embedded fresh.
for mode in "" _actiononly; do
  SRC=results/track2_n600_7fam${mode}
  DST=results/track2_n600_8fam${mode}
  [ -d "$SRC" ] || { echo "MISSING $SRC — run the grok pre-stage first"; exit 1; }
  mkdir -p "$DST"; cp "$SRC"/*.npz "$DST"/
  flag=""; [ -n "$mode" ] && flag="--action-only"
  echo "=== EMBED gpt-5 ${mode:-full} $(date +%H:%M:%S) ==="
  .venv-controlarena/bin/python scripts/track2_embed.py $flag --out track2_n600_8fam${mode} "$GPT5_DIR" || { echo EMBED_FAIL; exit 1; }
done

echo "=== MATRIX FULL $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_8fam
echo "=== MATRIX ACTION-ONLY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_8fam_actiononly
echo "=== PERM-TEST (Control 17) FULL + ACTION-ONLY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_permtest.py track2_n600_8fam            # 8! = 40320 perms, p-floor 2.5e-5
.venv/bin/python scripts/track2_permtest.py track2_n600_8fam_actiononly
echo "=== OOD-GATE (Control 19) FULL + ACTION-ONLY, full 8-family roster $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_ood_gate.py results/track2_n600_8fam results/track2_n600_8fam_actiononly
echo "=== RECALIBRATE (Control 20) FULL + ACTION-ONLY, full 8-family roster $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_recalibrate.py results/track2_n600_8fam results/track2_n600_8fam_actiononly
echo "=== RECAL CURVE (Control 21) FULL + ACTION-ONLY, full 8-family roster $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_recal_curve.py results/track2_n600_8fam results/track2_n600_8fam_actiononly
echo "=== SAFETY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_safety.py track2_n600_8fam --xfit

# Frontier-only sub-matrix = 4 vendors (xAI grok, Anthropic sonnet, Google gemini, OpenAI gpt-5)
for mode in "" _actiononly; do
  FO=results/track2_frontieronly4${mode}; mkdir -p "$FO"
  for f in frontier_grok_n600 frontier_sonnet_n600 gemini_flash_n600 frontier_gpt5_n600; do
    cp results/track2_n600_8fam${mode}/$f.npz "$FO"/ 2>/dev/null
  done
  echo "=== FRONTIER-ONLY (4 vendors) ${mode:-full} $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/track2_matrix.py track2_frontieronly4${mode}
  .venv/bin/python scripts/track2_permtest.py track2_frontieronly4${mode}  # N=4, p-floor 1/24 (vs N=3's 1/6)
done
echo "=== FRONTIER_FOLD_DONE $(date +%H:%M:%S) ==="
echo "NEXT: judge diagonals at balanced n, fold forest row + §4.x, make_figures.py, rebuild PDFs."
