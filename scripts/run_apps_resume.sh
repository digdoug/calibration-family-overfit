#!/bin/bash
# One-off resume orchestrator for the APPS n=600 scale-up run (2026-06-10).
#
# Context: run_family_pipeline.sh's original zombie detector stall-killed a
# healthy mistral run (it watched the MIN sample count across modes, which
# freezes >15 min when 3 sandboxes are shared by 2 concurrent tasks + inspect's
# buffered .eval flush). The orchestrator was killed; the mistral try-2 GEN
# process survived. This script ADOPTS that running gen process, supervises it
# with the fixed liveness logic (MAX across modes, STALL=1800), and then chains
# the remaining families exactly like the fixed pipeline.
#
# Launch (detached, APPEND to the existing status log):
#   cd ~/calibration-family-overfit && nohup bash scripts/run_apps_resume.sh \
#       >> results/apps500_status.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -f ~/.env_keys ]; then set -a; source ~/.env_keys; set +a; else source ~/.zshrc 2>/dev/null || true; fi
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -s -w $$ &

PY=.venv-controlarena/bin/python
DI_URL="https://api.deepinfra.com/v1/openai"
LIMIT="${LIMIT:-600}"
MAX_SANDBOXES="${MAX_SANDBOXES:-3}"
SUF="n${LIMIT}"
GRACE=1500
STALL=1800     # liveness on MAX across modes; 30 min of total silence = zombie
MIN_OK=$(( LIMIT / 2 ))

$PY scripts/patch_force_tool_choice.py || true

ts() { date +%H:%M:%S; }
count() {  # $1=fam $2=total|scored $3=min|max
  $PY -c "
import glob
from inspect_ai.log import read_eval_log
mode='$2'
vals=[]
for f in glob.glob('results/W2_$1/*.eval'):
    ss=read_eval_log(f).samples or []
    if mode=='scored':
        vals.append(sum(1 for s in ss if s.scores and 'max_suspicion_scorer' in s.scores and str(s.scores['max_suspicion_scorer'].value) not in ('N','None')))
    else:
        vals.append(len(ss))
agg = ${3:-min}
print(agg(vals) if vals else 0)
" 2>/dev/null || echo 0
}

watch_pid() {  # $1=pid $2=fam ; returns when pid exits or is zombie-killed
  local pid=$1 fam=$2 start last_n last_t now n
  start=$(date +%s); last_n=$(count "$fam" total max); last_t=$start
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    now=$(date +%s); n=$(count "$fam" total max)
    [ "$n" -gt "$last_n" ] && { last_n=$n; last_t=$now; }
    [ $(( now - start )) -lt "$GRACE" ] && continue
    if [ $(( now - last_t )) -gt "$STALL" ]; then
      echo "[$(ts)] $fam ZOMBIE (no new samples in ANY mode for $(( now - last_t ))s at max-n=$n) -> kill"
      kill -9 "$pid" 2>/dev/null; sleep 3; break
    fi
  done
}

run_family() {  # name model extra_flags...
  local fam=$1 model=$2; shift 2; local extra="$*"
  local log="results/W2_${fam}.log"
  for try in 1 2 3; do
    echo "[$(ts)] >>> $fam limit=$LIMIT try=$try  ($model $extra)"
    local cids; cids=$(docker ps -q 2>/dev/null)
    [ -n "$cids" ] && docker stop $cids >/dev/null 2>&1
    docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
    rm -rf "results/W2_${fam}" "$log"
    nohup $PY scripts/w2_generate_family.py --family "$fam" \
      --untrusted-model "$model" --untrusted-base-url "$DI_URL" --untrusted-api-key-env DEEPINFRA_API_KEY \
      --setting apps --modes honest attack --limit "$LIMIT" --ensemble-n 1 \
      --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes "$MAX_SANDBOXES" \
      --untrusted-timeout 120 --untrusted-max-retries 3 $extra \
      > "$log" 2>&1 < /dev/null &
    watch_pid $! "$fam"
    local sc; sc=$(count "$fam" scored min)
    echo "[$(ts)] $fam ended: total(min)=$(count "$fam" total min) scored(min)=$sc"
    [ "${sc:-0}" -ge "$MIN_OK" ] && { echo "[$(ts)] $fam OK (scored=$sc>=$MIN_OK)"; return 0; }
    echo "[$(ts)] $fam under $MIN_OK scored; retry"
  done
  echo "[$(ts)] $fam FAILED after 3 tries (continuing to next family)"; return 1
}

echo "[$(ts)] ===== RESUME ORCHESTRATOR START (limit=$LIMIT sandboxes=$MAX_SANDBOXES stall=$STALL) ====="

# --- Phase 1: adopt the already-running mistral gen, if it's still alive ---
FAM="ow_mistral_${SUF}"
PID=$(pgrep -f "w2_generate_family.py --family ${FAM}" | head -1)
if [ -n "${PID:-}" ]; then
  echo "[$(ts)] adopting running $FAM gen (pid=$PID)"
  watch_pid "$PID" "$FAM"
  SC=$(count "$FAM" scored min)
  echo "[$(ts)] $FAM (adopted) ended: total(min)=$(count "$FAM" total min) scored(min)=$SC"
  if [ "${SC:-0}" -ge "$MIN_OK" ]; then
    echo "[$(ts)] $FAM OK (scored=$SC>=$MIN_OK)"
  else
    echo "[$(ts)] $FAM (adopted) under $MIN_OK; full retries"
    run_family "$FAM" "openai/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  fi
else
  echo "[$(ts)] no running $FAM gen found; launching fresh"
  run_family "$FAM" "openai/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
fi

# --- Phase 2: remaining families ---
run_family "ow_llama_${SUF}"  "openai/meta-llama/Llama-3.3-70B-Instruct"
run_family "ow_qwen_${SUF}"   "openai/Qwen/Qwen3-14B" --untrusted-disable-thinking
[ "${SKIP_GEMMA:-0}" = "1" ] || run_family "ow_gemma_${SUF}" "openai/google/gemma-3-27b-it" --untrusted-force-tool-choice
echo "[$(ts)] ===== APPS RESUME PIPELINE COMPLETE ====="
