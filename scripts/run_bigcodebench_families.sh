#!/bin/bash
# 2nd-substrate generation: the SAME open-weight families on BIGCODEBENCH (Phase 4 step 1).
# Portable — meant to run on a Linux/amd64 box (the bigcodebench eval image is amd64-only and
# 24.8GB; native amd64 avoids the ~10x QEMU-emulation slowdown you'd get on Apple Silicon).
#
# Reuses the robust orchestration from run_family_pipeline.sh: detached gen (survives watcher
# death), SAMPLE-COUNT zombie detection (not PID), a startup GRACE, SCORED-count validation,
# Docker hard-clean between families. Families run SEQUENTIALLY (two at once blow the Docker pool).
#
# LAUNCH ONCE, detached:
#   cd <repo> && nohup bash scripts/run_bigcodebench_families.sh \
#       > results/bcb_pipeline_status.log 2>&1 < /dev/null & disown
#   tail -f results/bcb_pipeline_status.log
#
# Env knobs: LIMIT (default 200), MAX_SANDBOXES (default 4; raise on a big-RAM box — each
#   container is mem_limit 4gb), RUN_GEMMA=1 to also attempt Gemma (fragile, see note).
set -u
cd "$(dirname "$0")/.." || exit 1
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
# macOS-only sleep guard; on a Linux server this is a no-op (servers don't idle-sleep).
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -s -w $$ &

PY=.venv-controlarena/bin/python
DI_URL="https://api.deepinfra.com/v1/openai"
LIMIT="${LIMIT:-200}"
MAX_SANDBOXES="${MAX_SANDBOXES:-4}"
GRACE=1500     # bigcodebench first-container start can be slow; 25-min startup grace
STALL=900      # 15 min with NO new samples (after grace) => zombie -> kill+retry
MIN_OK=100     # min SCORED samples/mode to accept an n=200 run

# Ensure the Apple-Silicon platform patch is applied (NO-OP on amd64 where the image is native;
# needed only if this is ever run on arm64). Safe to run every time.
$PY scripts/patch_bigcodebench_platform.py 2>/dev/null || true

ts() { date +%H:%M:%S; }
count() {  # $1=fam  $2=total|scored  -> min across modes
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
print(min(vals) if vals else 0)
" 2>/dev/null || echo 0
}

run_family() {  # name model extra_flags...
  local fam=$1 model=$2; shift 2; local extra="$*"
  local log="results/W2_${fam}.log"
  for try in 1 2 3; do
    echo "[$(ts)] >>> $fam try=$try ($model $extra)"
    local cids; cids=$(docker ps -q 2>/dev/null)
    [ -n "$cids" ] && docker stop $cids >/dev/null 2>&1
    docker container prune -f >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1
    rm -rf "results/W2_${fam}" "$log"
    nohup $PY scripts/w2_generate_family.py --family "$fam" \
      --untrusted-model "$model" --untrusted-base-url "$DI_URL" --untrusted-api-key-env DEEPINFRA_API_KEY \
      --setting bigcodebench --modes honest attack --limit "$LIMIT" --ensemble-n 1 \
      --untrusted-max-output-tokens 4096 --max-steps 15 --max-sandboxes "$MAX_SANDBOXES" \
      --untrusted-timeout 120 --untrusted-max-retries 3 $extra \
      > "$log" 2>&1 < /dev/null &
    local pid=$! start last_n last_t now n
    start=$(date +%s); last_n=-1; last_t=$start
    while kill -0 "$pid" 2>/dev/null; do
      sleep 120
      now=$(date +%s); n=$(count "$fam" total)
      [ "$n" -gt "$last_n" ] && { last_n=$n; last_t=$now; }
      [ $(( now - start )) -lt "$GRACE" ] && continue
      if [ $(( now - last_t )) -gt "$STALL" ]; then
        echo "[$(ts)] $fam ZOMBIE (no new samples ${STALL}s at n=$n) -> kill"; kill -9 "$pid" 2>/dev/null; sleep 3; break
      fi
    done
    local sc; sc=$(count "$fam" scored)
    echo "[$(ts)] $fam ended: total=$(count "$fam" total) scored=$sc"
    [ "${sc:-0}" -ge "$MIN_OK" ] && { echo "[$(ts)] $fam OK (scored=$sc)"; return 0; }
    echo "[$(ts)] $fam under $MIN_OK scored; retry"
  done
  echo "[$(ts)] $fam FAILED after 3 tries (continuing to next family)"; return 1
}

echo "[$(ts)] ===== BIGCODEBENCH PIPELINE START (limit=$LIMIT sandboxes=$MAX_SANDBOXES) ====="
run_family bcb_mistral_n200 "openai/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
run_family bcb_llama_n200   "openai/meta-llama/Llama-3.3-70B-Instruct"
run_family bcb_qwen_n200    "openai/Qwen/Qwen3-14B" --untrusted-disable-thinking
# Gemma is FRAGILE on the multi-turn eval (--untrusted-force-tool-choice gives ~97% NOANSWER:
# with multiple tools, "force any tool" doesn't reliably pick submit — see runbook). Off by
# default so it doesn't burn ~1-2h retrying. RUN_GEMMA=1 to attempt it (or substitute a
# closed-API 4th family). A 3-family bcb matrix already replicates the apps transfer structure.
[ "${RUN_GEMMA:-0}" = "1" ] && run_family bcb_gemma_n200 "openai/google/gemma-3-27b-it" --untrusted-force-tool-choice
echo "[$(ts)] ===== BIGCODEBENCH PIPELINE COMPLETE ====="
echo "Next (cheap, no Docker — run anywhere with the OpenAI key):"
echo "  .venv-controlarena/bin/python scripts/track2_embed.py --out track2_bcb results/W2_bcb_*_n200"
echo "  .venv/bin/python scripts/track2_matrix.py track2_bcb     # interaction vs the APPS +0.127"
