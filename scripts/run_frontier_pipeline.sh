#!/bin/bash
# Sequential FRONTIER-family generation for the all-frontier APPS matrix.
# Same durable skeleton as run_family_pipeline.sh (nohup'd child, sample-count
# zombie detection on MAX across modes, startup grace, scored-count validation,
# salvage partials) but with per-family base_url/key (xAI native, OpenAI native,
# DeepInfra) instead of the hardcoded DeepInfra endpoint.
#
# Roster (n=30 gates 2026-06-12): grok-4.3 AUROC 0.764, gpt-5 low-effort 0.707
# (uncapped gpt-5 never submits — 61k reasoning tok/sample). DeepSeek-V3.2
# thinking-off LAST (slowest, ~17h; optional fifth lineage — the 4-frontier
# matrix must not wait on it).
#
# LAUNCH (droplet, detached):
#   cd ~/calibration-family-overfit && LIMIT=600 MAX_SANDBOXES=6 \
#     nohup bash scripts/run_frontier_pipeline.sh \
#     > results/frontier_pipeline_status.log 2>&1 < /dev/null & disown

PY=${PY:-.venv-controlarena/bin/python}
[ -x "$PY" ] || PY=.venv/bin/python
DI_URL="https://api.deepinfra.com/v1/openai"
XAI_URL="https://api.x.ai/v1"
LIMIT="${LIMIT:-600}"
MAX_SANDBOXES="${MAX_SANDBOXES:-6}"
SUF="n${LIMIT}"
GRACE=1500
STALL=1800
MIN_OK=$(( LIMIT / 2 ))
[ -f ~/.env_keys ] && source ~/.env_keys
[ -f ~/.zshrc ] && source ~/.zshrc 2>/dev/null

ts() { date +%H:%M:%S; }
count() {
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

run_family() {  # name model base_url("-"=native) key_env("-"=default) extra...
  local fam=$1 model=$2 burl=$3 kenv=$4; shift 4; local extra="$*"
  local provider_flags=""
  [ "$burl" != "-" ] && provider_flags="--untrusted-base-url $burl"
  [ "$kenv" != "-" ] && provider_flags="$provider_flags --untrusted-api-key-env $kenv"
  local log="results/W2_${fam}.log"
  for try in 1 2 3; do
    echo "[$(ts)] >>> $fam limit=$LIMIT try=$try  ($model $provider_flags $extra)"
    local cids; cids=$(docker ps -q 2>/dev/null)
    [ -n "$cids" ] && docker stop $cids >/dev/null 2>&1
    docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
    rm -rf "results/W2_${fam}" "$log"
    nohup $PY scripts/w2_generate_family.py --family "$fam" \
      --untrusted-model "$model" $provider_flags \
      --setting apps --modes honest attack --limit "$LIMIT" --ensemble-n 1 \
      --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes "$MAX_SANDBOXES" \
      --untrusted-timeout 180 --untrusted-max-retries 3 $extra \
      > "$log" 2>&1 < /dev/null &
    local pid=$! start last_n last_t now n
    start=$(date +%s); last_n=-1; last_t=$start
    while kill -0 "$pid" 2>/dev/null; do
      sleep 120
      now=$(date +%s); n=$(count "$fam" total max)
      [ "$n" -gt "$last_n" ] && { last_n=$n; last_t=$now; }
      [ $(( now - start )) -lt "$GRACE" ] && continue
      if [ $(( now - last_t )) -gt "$STALL" ]; then
        echo "[$(ts)] $fam ZOMBIE (no new samples for $(( now - last_t ))s at n=$n) -> kill"
        kill -9 "$pid" 2>/dev/null; sleep 3; break
      fi
    done
    local sc; sc=$(count "$fam" scored)
    echo "[$(ts)] $fam ended: total=$(count "$fam" total) scored=$sc"
    [ "${sc:-0}" -ge "$MIN_OK" ] && { echo "[$(ts)] $fam OK (scored=$sc>=$MIN_OK)"; return 0; }
    echo "[$(ts)] $fam under $MIN_OK scored; retry"
  done
  echo "[$(ts)] $fam FAILED after 3 tries (continuing to next family)"; return 1
}

echo "[$(ts)] ===== FRONTIER PIPELINE START (limit=$LIMIT sandboxes=$MAX_SANDBOXES) ====="
run_family "frontier_grok_${SUF}" "openai/grok-4.3" "$XAI_URL" "XAI_API_KEY"
run_family "frontier_gpt5_${SUF}" "openai/gpt-5" "-" "-" --untrusted-reasoning-effort low
[ "${SKIP_DEEPSEEK:-0}" = "1" ] || run_family "frontier_deepseek_${SUF}" \
  "openai/deepseek-ai/DeepSeek-V3.2" "$DI_URL" "DEEPINFRA_API_KEY" --untrusted-disable-thinking
echo "[$(ts)] ===== FRONTIER PIPELINE COMPLETE ====="
echo "Next: pull results/W2_frontier_*_${SUF} to the Mac, then track2_embed --out track2_frontier."
