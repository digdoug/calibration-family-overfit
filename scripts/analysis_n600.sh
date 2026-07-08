#!/bin/bash
# One-shot n=600 analysis chain (scale-n result, 2026-06-11). Detach + log.
cd "$(dirname "$0")/.." || exit 1
source ~/.zshrc 2>/dev/null || true
[ -n "${OPENAI_API_KEY:-}" ] || { echo "NO_OPENAI_KEY"; exit 2; }
DIRS="results/W2_ow_mistral_n600 results/W2_ow_llama_n600 results/W2_ow_qwen_n600 results/W2_ow_gemma_n600"

echo "=== EMBED FULL START $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_embed.py --out track2_n600 $DIRS || { echo EMBED_FULL_FAIL; exit 1; }
echo "=== EMBED ACTION START $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_embed.py --action-only --out track2_n600_actiononly $DIRS || { echo EMBED_ACT_FAIL; exit 1; }
echo "=== MATRIX FULL $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600
echo "=== MATRIX ACTIONONLY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_actiononly
echo "=== SAFETY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_safety.py track2_n600 --xfit
echo "=== W2 FIXED-PROMPT MATRIX $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/w2_run_matrix.py --families ow_mistral_n600 ow_llama_n600 ow_qwen_n600 ow_gemma_n600 --no-success-filter --fpr 0.01
echo "=== ANALYSIS_ALL_DONE $(date +%H:%M:%S) ==="
