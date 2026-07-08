#!/usr/bin/env bash
# n=30 viability gate for the two new frontier families (skill step: verify
# unfiltered honest-vs-attack AUROC > ~0.65 at small n BEFORE the n=600 spend).
# Sequential per the Docker-pool rule; caffeinate keeps the laptop awake.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.zshrc
caffeinate -i -s -w $$ &
PY=.venv-controlarena/bin/python

echo "[n30] grok start $(date)"
$PY scripts/w2_generate_family.py --family grok_n30 \
  --untrusted-model "openai/grok-4.3" \
  --untrusted-base-url https://api.x.ai/v1 \
  --untrusted-api-key-env XAI_API_KEY \
  --setting apps --limit 30 --ensemble-n 1 \
  --untrusted-max-output-tokens 4096 --untrusted-timeout 180 \
  --max-sandboxes 4 --max-steps 15 --log-dir results/grok_n30 \
  > results/grok_n30.log 2>&1
echo "[n30] grok done $(date)"
docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1

echo "[n30] gpt5 start $(date)"
$PY scripts/w2_generate_family.py --family gpt5_n30 \
  --untrusted-model "openai/gpt-5" \
  --untrusted-reasoning-effort low \
  --setting apps --limit 30 --ensemble-n 1 \
  --untrusted-max-output-tokens 4096 --untrusted-timeout 180 \
  --max-sandboxes 4 --max-steps 15 --log-dir results/gpt5_n30 \
  > results/gpt5_n30.log 2>&1
echo "[n30] gpt5 done $(date)"
docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
echo "[n30] ALL DONE $(date)"
