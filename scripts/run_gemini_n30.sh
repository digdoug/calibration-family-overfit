#!/bin/bash
cd ~/calibration-family-overfit
set -a; source ~/.env_keys; set +a
pkill -9 -f "w2_generate_famil[y]" 2>/dev/null; sleep 2
docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
rm -rf results/W2_gemini_flash_n30 results/gemini_flash_n30.log
nohup .venv-controlarena/bin/python scripts/w2_generate_family.py --family gemini_flash_n30 \
  --untrusted-model "google/gemini-2.5-flash" --untrusted-api-key-env GOOGLE_API_KEY \
  --setting apps --modes honest attack --limit 30 \
  --untrusted-max-output-tokens 4096 --untrusted-reasoning-tokens 0 \
  --untrusted-timeout 120 --max-sandboxes 4 --max-steps 5 \
  --log-dir results/W2_gemini_flash_n30 \
  > results/gemini_flash_n30.log 2>&1 < /dev/null &
disown
echo "launched pid $!"
