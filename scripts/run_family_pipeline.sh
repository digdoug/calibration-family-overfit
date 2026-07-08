#!/bin/bash
# Robust sequential open-weight family generation for the APPS transfer matrix.
# Portable: macOS laptop or a Linux/amd64 box (DigitalOcean droplet). The scale-n
# (paper-n) regeneration runs on the droplet.
#
# Fixes the failure modes we hit on the W2 runs:
#  - watcher dies -> child gen dies: launch gen detached (nohup), monitor separately.
#  - PID-based monitor hangs on a ZOMBIE: monitor SAMPLE-COUNT progress, not just PID.
#  - false-kill during startup: a GRACE period before any stall-check.
#  - garbage run (samples but no submissions) false-passes: validate on SCORED count.
#
# LAUNCH ONCE, detached (macOS has no setsid; nohup+disown is what survives):
#   cd <repo> && LIMIT=600 MAX_SANDBOXES=3 nohup bash scripts/run_family_pipeline.sh \
#       > results/apps_pipeline_status.log 2>&1 < /dev/null & disown
#   tail -f results/apps_pipeline_status.log
#
# Env knobs: LIMIT (default 200; family dirs are named W2_ow_<fam>_n$LIMIT),
#   MAX_SANDBOXES (default 8 for a Mac; use 3 on a 4-core/8GB droplet — proven by
#   the bigcodebench run), SKIP_GEMMA=1 to drop the fragile 4th family.
set -u
cd "$(dirname "$0")/.." || exit 1
# Keys: Linux box keeps them in ~/.env_keys (chmod 600); Mac in ~/.zshrc.
if [ -f ~/.env_keys ]; then set -a; source ~/.env_keys; set +a; else source ~/.zshrc 2>/dev/null || true; fi
# macOS-only sleep guard; no-op on a Linux server.
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -s -w $$ &

PY=.venv-controlarena/bin/python
DI_URL="https://api.deepinfra.com/v1/openai"
LIMIT="${LIMIT:-200}"
MAX_SANDBOXES="${MAX_SANDBOXES:-8}"
SUF="n${LIMIT}"
GRACE=1500     # 25-min startup grace (first docker pull + slow first samples)
STALL=1800     # 30 min with NO new samples IN ANY MODE => zombie -> kill+retry.
               # Must be generous + measured on the MAX across modes: with few
               # sandboxes shared by 2 concurrent tasks, one mode legitimately
               # starves >15 min, and inspect's buffered .eval flush freezes the
               # on-disk count meanwhile (false-killed mistral n=600 try-1 at
               # "no new samples for 982s at n=20" while attack was advancing).
MIN_OK=$(( LIMIT / 2 ))  # min SCORED samples/mode to accept a run (else retry)

# Gemma-3 needs tool_choice='any' to submit (CA_FORCE_TOOL_CHOICE); idempotent.
$PY scripts/patch_force_tool_choice.py || true

ts() { date +%H:%M:%S; }
count() {  # $1=fam $2=total|scored $3=min|max  -> aggregate across modes.
  # min = VALIDATION (both modes must be complete); max = LIVENESS (any
  # progress anywhere means the process is alive — never stall-kill on min).
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

run_family() {  # name model extra_flags...
  local fam=$1 model=$2; shift 2; local extra="$*"
  local log="results/W2_${fam}.log"
  for try in 1 2 3; do
    echo "[$(ts)] >>> $fam limit=$LIMIT try=$try  ($model $extra)"
    # HARD docker clean: stop lingering containers FIRST (else their networks aren't
    # pruned and the address pool exhausts ~25 -> every sample errors instantly).
    local cids; cids=$(docker ps -q 2>/dev/null)
    [ -n "$cids" ] && docker stop $cids >/dev/null 2>&1
    docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
    echo "[$(ts)] docker networks now: $(docker network ls 2>/dev/null | wc -l | tr -d ' ')"
    rm -rf "results/W2_${fam}" "$log"
    nohup $PY scripts/w2_generate_family.py --family "$fam" \
      --untrusted-model "$model" --untrusted-base-url "$DI_URL" --untrusted-api-key-env DEEPINFRA_API_KEY \
      --setting apps --modes honest attack --limit "$LIMIT" --ensemble-n 1 \
      --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes "$MAX_SANDBOXES" \
      --untrusted-timeout 120 --untrusted-max-retries 3 $extra \
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

echo "[$(ts)] ===== APPS PIPELINE START (limit=$LIMIT sandboxes=$MAX_SANDBOXES) ====="
run_family "ow_mistral_${SUF}" "openai/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
run_family "ow_llama_${SUF}"   "openai/meta-llama/Llama-3.3-70B-Instruct"
run_family "ow_qwen_${SUF}"    "openai/Qwen/Qwen3-14B" --untrusted-disable-thinking
# Gemma WORKS on APPS with force-tool-choice (n=200 run 2026-06-08: 200/200 scored;
# the ~97% NOANSWER failure was bigcodebench-specific). Last so it can't block others.
[ "${SKIP_GEMMA:-0}" = "1" ] || run_family "ow_gemma_${SUF}" "openai/google/gemma-3-27b-it" --untrusted-force-tool-choice
echo "[$(ts)] ===== APPS PIPELINE COMPLETE ====="
echo "Next (no Docker needed — embeddings call OpenAI; run on the Mac after pulling .eval):"
echo "  .venv-controlarena/bin/python scripts/track2_embed.py --out track2_${SUF} results/W2_ow_*_${SUF}"
echo "  .venv/bin/python scripts/track2_matrix.py track2_${SUF}"
