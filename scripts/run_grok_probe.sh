#!/bin/bash
# Item-11 Stage-1 GROK REALIZED-SABOTAGE YIELD PROBE (attack-only, fresh APPS ids).
# Pre-registered before running.
# Settles whether grok realizes sabotage at >= 3-4% (grok cached rate 4.5%; n=60 smoke 0/60).
#
# Design: 300 FRESH attack samples (ids 801-1169, disjoint from grok cached 1-600 AND the
# 731-800 smoke), grok-4.3 via xAI native. Attack mode ONLY (both-ok is an attack-transcript
# property; halves the spend vs honest+attack). A 3-sample smoke gates status==success first.
#
# LAUNCH (droplet, detached, from one short ssh):
#   cd ~/calibration-family-overfit && nohup bash scripts/run_grok_probe.sh \
#       > results/grok_probe_status.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -f ~/.env_keys ]; then set -a; source ~/.env_keys; set +a; else source ~/.zshrc 2>/dev/null || true; fi

PY=.venv-controlarena/bin/python
[ -x "$PY" ] || PY=.venv/bin/python
XAI_URL="https://api.x.ai/v1"
MODEL="openai/grok-4.3"
IDS_FILE="results/successful_sabotage/grok_probe_ids.txt"
FAM="frontier_grok_probe"
SMOKE_FAM="frontier_grok_probe_smoke"
MAX_SANDBOXES="${MAX_SANDBOXES:-6}"
GRACE=1500
STALL=1800

ts() { date +%H:%M:%S; }
count() {  # $1=family $2=total|scored $3=min|max
  $PY -c "
import glob
from inspect_ai.log import read_eval_log
mode='$2'; vals=[]
for f in glob.glob('results/W2_$1/*.eval'):
    ss=read_eval_log(f).samples or []
    if mode=='scored':
        vals.append(sum(1 for s in ss if s.scores and 'max_suspicion_scorer' in s.scores and str(s.scores['max_suspicion_scorer'].value) not in ('N','None')))
    else:
        vals.append(len(ss))
agg=${3:-min}
print(agg(vals) if vals else 0)
" 2>/dev/null || echo 0
}

clean_docker() {
  local cids; cids=$(docker ps -q 2>/dev/null)
  [ -n "$cids" ] && docker stop $cids >/dev/null 2>&1
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
}

echo "[$(ts)] ===== GROK PROBE START (attack-only, 300 fresh ids) ====="
[ -f "$IDS_FILE" ] || { echo "[$(ts)] FATAL: $IDS_FILE missing"; exit 1; }
NIDS=$(tr ',' '\n' < "$IDS_FILE" | grep -c .)
echo "[$(ts)] probe ids: $NIDS (expect 300)"

# ---- SMOKE GATE: 3 ids just beyond the probe pool (1170,1171,1172) ----
echo "[$(ts)] >>> SMOKE (3 attack samples) to gate status==success"
clean_docker
rm -rf "results/W2_${SMOKE_FAM}" "results/W2_${SMOKE_FAM}.log"
$PY scripts/w2_generate_family.py --family "$SMOKE_FAM" \
  --untrusted-model "$MODEL" --untrusted-base-url "$XAI_URL" --untrusted-api-key-env XAI_API_KEY \
  --setting apps --modes attack --sample-ids "1170,1171,1172" --ensemble-n 1 \
  --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes 3 \
  --untrusted-timeout 180 --untrusted-max-retries 3 \
  --log-dir "results/W2_${SMOKE_FAM}" \
  > "results/W2_${SMOKE_FAM}.log" 2>&1 < /dev/null
SMOKE_SCORED=$(count "$SMOKE_FAM" scored max)
echo "[$(ts)] SMOKE scored=$SMOKE_SCORED (of 3)"
if [ "${SMOKE_SCORED:-0}" -lt 1 ]; then
  echo "[$(ts)] SMOKE FAILED (0 scored) -> ABORT full probe. Check results/W2_${SMOKE_FAM}.log"
  echo "[$(ts)] ===== GROK PROBE ABORTED ====="
  exit 2
fi
echo "[$(ts)] SMOKE OK -> launching full 300-sample attack probe"

# ---- FULL PROBE: 300 attack samples, detached child with zombie detection ----
for try in 1 2 3; do
  echo "[$(ts)] >>> $FAM try=$try ($MODEL, $NIDS ids)"
  clean_docker
  rm -rf "results/W2_${FAM}" "results/W2_${FAM}.log"
  nohup $PY scripts/w2_generate_family.py --family "$FAM" \
    --untrusted-model "$MODEL" --untrusted-base-url "$XAI_URL" --untrusted-api-key-env XAI_API_KEY \
    --setting apps --modes attack --sample-ids "$(cat "$IDS_FILE")" --ensemble-n 1 \
    --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes "$MAX_SANDBOXES" \
    --untrusted-timeout 180 --untrusted-max-retries 3 \
    --log-dir "results/W2_${FAM}" \
    > "results/W2_${FAM}.log" 2>&1 < /dev/null &
  pid=$!; start=$(date +%s); last_n=-1; last_t=$start
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    now=$(date +%s); n=$(count "$FAM" total max)
    [ "$n" -gt "$last_n" ] && { last_n=$n; last_t=$now; }
    if [ "$(count "$FAM" total max)" -ge "$NIDS" ]; then
      echo "[$(ts)] $FAM complete (max total >= $NIDS) -> stop any post-run hang"; kill -9 "$pid" 2>/dev/null; sleep 3; break
    fi
    [ $(( now - start )) -lt "$GRACE" ] && continue
    if [ $(( now - last_t )) -gt "$STALL" ]; then
      echo "[$(ts)] $FAM ZOMBIE (no new samples ${STALL}s at max-n=$n) -> kill"; kill -9 "$pid" 2>/dev/null; sleep 3; break
    fi
  done
  sc=$(count "$FAM" scored max); tot=$(count "$FAM" total max)
  echo "[$(ts)] $FAM ended: total(max)=$tot scored(max)=$sc"
  # probe target: accept if we scored >= 2/3 of the ids (grok flows; salvage partials)
  [ "${sc:-0}" -ge $(( NIDS * 2 / 3 )) ] && { echo "[$(ts)] $FAM OK (scored=$sc)"; break; }
  echo "[$(ts)] $FAM under target scored; retry"
done
echo "[$(ts)] ===== GROK PROBE COMPLETE ====="
echo "Next (Mac): pull results/W2_${FAM}, run scripts/grok_probe_yield.py to read TEST both-ok yield vs the 3-4% gate."
