#!/bin/bash
# 5-family fold: the 4 open-weight n=600 families + the Claude Sonnet frontier family.
cd "$(dirname "$0")/.." || exit 1
source ~/.zshrc 2>/dev/null || true
[ -n "${OPENAI_API_KEY:-}" ] || { echo "NO_OPENAI_KEY"; exit 2; }
D5="results/W2_ow_mistral_n600 results/W2_ow_llama_n600 results/W2_ow_qwen_n600 results/W2_ow_gemma_n600 results/W2_frontier_sonnet_n600"

echo "=== EMBED FULL $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_embed.py --out track2_n600_5fam $D5 || { echo EMBED_FULL_FAIL; exit 1; }
echo "=== EMBED ACTIONONLY $(date +%H:%M:%S) ==="
.venv-controlarena/bin/python scripts/track2_embed.py --action-only --out track2_n600_5fam_actiononly $D5 || { echo EMBED_ACT_FAIL; exit 1; }
echo "=== MATRIX FULL $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_5fam
echo "=== MATRIX ACTIONONLY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_5fam_actiononly
echo "=== SAFETY $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_safety.py track2_n600_5fam --xfit
echo "=== ANALYSIS_5FAM_DONE $(date +%H:%M:%S) ==="
