#!/bin/bash
# Self-healing supervisor for the Gemini n=600 run. nohup'd; survives independently.
# Relaunches the gen ONLY if it dies before target (no stall-kill -> no thrash).
# Detects completion by SCORED count (handles the native-google post-run process hang).
cd ~/calibration-family-overfit
set -a; source ~/.env_keys; set +a
PY=.venv-controlarena/bin/python
FAM=gemini_flash_n600
TARGET=520
scored() { $PY -c "
import glob
from inspect_ai.log import read_eval_log
v=[]
for f in glob.glob('results/W2_$FAM/*.eval'):
    ss=read_eval_log(f).samples or []
    v.append(sum(1 for s in ss if s.scores and 'max_suspicion_scorer' in s.scores and str(s.scores['max_suspicion_scorer'].value) not in ('N','None')))
print(min(v) if len(v)>=2 else 0)
" 2>/dev/null || echo 0; }
launch() {
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  nohup $PY scripts/w2_generate_family.py --family $FAM \
    --untrusted-model "google/gemini-2.5-flash" --untrusted-api-key-env GOOGLE_API_KEY \
    --setting apps --modes honest attack --limit 600 \
    --untrusted-max-output-tokens 4096 --untrusted-reasoning-tokens 0 \
    --untrusted-timeout 120 --max-sandboxes 6 --max-steps 5 \
    --log-dir results/W2_$FAM >> results/${FAM}.log 2>&1 < /dev/null &
  echo "[$(date +%H:%M)] launched gen pid $!"
}
while true; do
  s=$(scored)
  if [ "${s:-0}" -ge "$TARGET" ]; then echo "[$(date +%H:%M)] COMPLETE scored=$s"; pkill -9 -f "w2_generate_famil[y]"; break; fi
  if ! pgrep -f "scripts/w2_generate_famil[y]" >/dev/null; then echo "[$(date +%H:%M)] gen not running at scored=$s -> relaunch"; launch; fi
  sleep 120
done
echo "[$(date +%H:%M)] WATCHDOG DONE scored=$(scored)"
