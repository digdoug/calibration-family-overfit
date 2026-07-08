#!/bin/bash
# Monitor-VENDOR robustness: re-embed the 4-family n=600 core with a non-OpenAI open embedder
# (nomic-embed-text-v1.5, local) and re-run the interaction. Extends Control 7 across vendors.
cd "$(dirname "$0")/.." || exit 1
source ~/.zshrc 2>/dev/null || true
D4="results/W2_ow_mistral_n600 results/W2_ow_llama_n600 results/W2_ow_qwen_n600 results/W2_ow_gemma_n600"
EMB="local:nomic-ai/nomic-embed-text-v1.5"
echo "=== EMBED FULL (nomic) $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_embed.py --emb-model "$EMB" --out track2_n600_nomic $D4 || { echo EMBED_FULL_FAIL; exit 1; }
echo "=== EMBED ACTIONONLY (nomic) $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_embed.py --action-only --emb-model "$EMB" --out track2_n600_nomic_actiononly $D4 || { echo EMBED_ACT_FAIL; exit 1; }
echo "=== MATRIX FULL (nomic) $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_nomic
echo "=== MATRIX ACTIONONLY (nomic) $(date +%H:%M:%S) ==="
.venv/bin/python scripts/track2_matrix.py track2_n600_nomic_actiononly
echo "=== ANALYSIS_NOMIC_DONE $(date +%H:%M:%S) ==="
