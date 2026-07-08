#!/bin/bash
# Single-family generation for the FRONTIER attacker (frontier-attacker extension, started early
# 2026-06-11 — the pre-registered smoke cleared at AUROC 0.754). One family,
# Claude Sonnet 4.6 via the NATIVE inspect_ai anthropic provider (no base-url; ANTHROPIC_API_KEY).
#
# Reuses the hardened orchestration from run_apps_resume.sh: detached gen, MAX-across-modes
# liveness (the min-counter false-kill bug fix), startup grace, SCORED-count validation, salvage.
# ADDS an early-complete check (min total >= LIMIT in both modes => done) so the benign post-run
# inspect hang seen in the n=30 smoke doesn't burn a full STALL window.
#
# LAUNCH (detached):
#   cd ~/calibration-family-overfit && nohup bash scripts/run_frontier_family.sh \
#       > results/frontier_n600_status.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -f ~/.env_keys ]; then set -a; source ~/.env_keys; set +a; else source ~/.zshrc 2>/dev/null || true; fi
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -s -w $$ &

PY=.venv-controlarena/bin/python
LIMIT="${LIMIT:-600}"
MAX_SANDBOXES="${MAX_SANDBOXES:-4}"     # Anthropic API concurrency — conservative; smoke ran at 4
FAM="frontier_sonnet_n${LIMIT}"
MODEL="anthropic/claude-sonnet-4-6"
GRACE=1200
STALL=1800
MIN_OK=$(( LIMIT / 2 ))

ts() { date +%H:%M:%S; }
count() {  # $1=total|scored  $2=min|max  (family fixed = $FAM)
  $PY -c "
import glob
from inspect_ai.log import read_eval_log
mode='$1'; vals=[]
for f in glob.glob('results/W2_${FAM}/*.eval'):
    ss=read_eval_log(f).samples or []
    if mode=='scored':
        vals.append(sum(1 for s in ss if s.scores and 'max_suspicion_scorer' in s.scores and str(s.scores['max_suspicion_scorer'].value) not in ('N','None')))
    else:
        vals.append(len(ss))
agg=${2:-min}
print(agg(vals) if vals else 0)
" 2>/dev/null || echo 0
}

for try in 1 2 3; do
  echo "[$(ts)] >>> $FAM limit=$LIMIT try=$try ($MODEL)"
  cids=$(docker ps -q 2>/dev/null); [ -n "$cids" ] && docker stop $cids >/dev/null 2>&1
  docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
  rm -rf "results/W2_${FAM}" "results/W2_${FAM}.log"
  nohup $PY scripts/w2_generate_family.py --family "$FAM" \
    --untrusted-model "$MODEL" --untrusted-api-key-env ANTHROPIC_API_KEY \
    --setting apps --modes honest attack --limit "$LIMIT" --ensemble-n 1 \
    --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes "$MAX_SANDBOXES" \
    --untrusted-timeout 120 --untrusted-max-retries 3 \
    --log-dir "results/W2_${FAM}" \
    > "results/W2_${FAM}.log" 2>&1 < /dev/null &
  pid=$!; start=$(date +%s); last_n=-1; last_t=$start
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    now=$(date +%s); n=$(count total max)
    [ "$n" -gt "$last_n" ] && { last_n=$n; last_t=$now; }
    # early-complete: both modes reached LIMIT but inspect hasn't exited (post-run hang)
    if [ "$(count total min)" -ge "$LIMIT" ]; then
      echo "[$(ts)] $FAM complete (min total >= $LIMIT) -> stop hung proc"; kill -9 "$pid" 2>/dev/null; sleep 3; break
    fi
    [ $(( now - start )) -lt "$GRACE" ] && continue
    if [ $(( now - last_t )) -gt "$STALL" ]; then
      echo "[$(ts)] $FAM ZOMBIE (no new samples any mode ${STALL}s at max-n=$n) -> kill"; kill -9 "$pid" 2>/dev/null; sleep 3; break
    fi
  done
  sc=$(count scored min)
  echo "[$(ts)] $FAM ended: total(min)=$(count total min) scored(min)=$sc"
  [ "${sc:-0}" -ge "$MIN_OK" ] && { echo "[$(ts)] $FAM OK (scored=$sc>=$MIN_OK)"; break; }
  echo "[$(ts)] $FAM under $MIN_OK scored; retry"
done
echo "[$(ts)] ===== FRONTIER FAMILY DONE ====="
echo "Next (Mac): pull results/W2_${FAM} then 5-family track2_embed/matrix/safety incl. this family."
