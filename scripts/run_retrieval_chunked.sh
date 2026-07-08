#!/bin/bash
# Robust chunked runner for track2_retrieval.py scoring (mirrors run_fewshot_chunked.sh).
#
# WHY: this environment stalls after a burst of ~10-40 OpenAI requests, httpx timeout never fires.
# Workaround: small resumable chunks in FRESH processes; kill a stalled chunk after a wall-clock
# timeout and re-launch. The per-call checkpoint makes every chunk resume exactly where it stopped.
#
# Usage: scripts/run_retrieval_chunked.sh <model> <extra-args...>
#   scripts/run_retrieval_chunked.sh gpt-4.1-mini --probe --probe-n 40 --families mistral,llama
#   scripts/run_retrieval_chunked.sh gpt-4.1-mini --test-n 100
# Env: CHUNK (calls/chunk, default 12), CHUNK_TIMEOUT (s before kill, default 70), MAXATT (default 600),
#      MAXUSD (global $ cap across chunks, default 25).
set -u
MODEL="${1:?usage: run_retrieval_chunked.sh <model> <extra-args...>}"; shift || true
EXTRA="$*"
CHUNK="${CHUNK:-12}"
CHUNK_TIMEOUT="${CHUNK_TIMEOUT:-70}"
MAXATT="${MAXATT:-600}"
MAXUSD="${MAXUSD:-25}"
PY=.venv-controlarena/bin/python
TAG="${MODEL//[^a-zA-Z0-9]/}"
LOG="results/track2_retrieval/retrieval_chunked_${TAG}.log"
mkdir -p results/track2_retrieval
echo "=== chunked run model=$MODEL extra='$EXTRA' CHUNK=$CHUNK timeout=${CHUNK_TIMEOUT}s MAXUSD=$MAXUSD $(date '+%H:%M:%S') ===" | tee -a "$LOG"
stalls=0
for att in $(seq 1 "$MAXATT"); do
  tmp="$(mktemp)"
  $PY scripts/track2_retrieval.py --model "$MODEL" $EXTRA --limit "$CHUNK" >"$tmp" 2>&1 &
  pid=$!
  alive=1
  for s in $(seq 1 "$CHUNK_TIMEOUT"); do
    kill -0 "$pid" 2>/dev/null || { alive=0; break; }
    sleep 1
  done
  if [ "$alive" -eq 1 ]; then
    kill "$pid" 2>/dev/null; sleep 1; kill -9 "$pid" 2>/dev/null
    stalls=$((stalls+1))
    echo "[wrapper att=$att] STALL-KILL after ${CHUNK_TIMEOUT}s (total stalls=$stalls)" | tee -a "$LOG"
  fi
  grep -hE '\+[0-9]+/|DONE|STOPPED|TIME-BOXED|ERR|nothing to do' "$tmp" >>"$LOG" 2>/dev/null
  if grep -q 'nothing to do' "$tmp" 2>/dev/null; then
    echo "[wrapper] COMPLETE at att=$att ($(date '+%H:%M:%S')), stalls=$stalls" | tee -a "$LOG"
    rm -f "$tmp"; exit 0
  fi
  rm -f "$tmp"
  cur=$($PY scripts/track2_retrieval.py --model "$MODEL" --cost 2>/dev/null | grep -oE '\$[0-9.]+' | head -1 | tr -d '$')
  if [ -n "${cur:-}" ] && awk "BEGIN{exit !(${cur:-0} > $MAXUSD)}"; then
    echo "[wrapper] GLOBAL BUDGET \$$cur > \$$MAXUSD — STOPPING ($(date '+%H:%M:%S'))." | tee -a "$LOG"
    exit 2
  fi
  [ $((att % 10)) -eq 0 ] && echo "[wrapper att=$att] cumulative spend \$${cur:-?} ($(date '+%H:%M:%S'))" | tee -a "$LOG"
done
echo "[wrapper] hit MAXATT=$MAXATT without completion ($(date '+%H:%M:%S'))" | tee -a "$LOG"
exit 1
