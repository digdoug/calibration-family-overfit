#!/bin/bash
# Robust chunked runner for track2_fewshot_v2.py scoring.
#
# WHY: this environment stalls after a burst of ~10-40 OpenAI requests, and the httpx read/connect
# timeout never fires (the request hangs forever, 0 CPU). No client tuning fixed it (concurrent or
# sequential, keepalive or not). The mechanical workaround: run SMALL resumable chunks in FRESH
# processes (a fresh process reliably does a chunk before any stall), and if a chunk stalls, KILL it
# after a hard wall-clock timeout and re-launch — the checkpoint (written after every call) makes
# every chunk resume exactly where it left off. Net effect: progress is guaranteed; a stall costs at
# most one chunk-timeout of wasted wall-clock, not a dead run.
#
# Usage: scripts/run_fewshot_chunked.sh <model> <extra-args...>
#   scripts/run_fewshot_chunked.sh gpt-4.1-mini --probe --probe-n 40
#   scripts/run_fewshot_chunked.sh gpt-4.1-mini --test-n 100
# Env: CHUNK (calls/chunk, default 12), CHUNK_TIMEOUT (s before kill, default 70), MAXATT (default 400),
#      MAX_USD passed through via extra-args.
set -u
MODEL="${1:?usage: run_fewshot_chunked.sh <model> <extra-args...>}"; shift || true
EXTRA="$*"
CHUNK="${CHUNK:-12}"
CHUNK_TIMEOUT="${CHUNK_TIMEOUT:-70}"
MAXATT="${MAXATT:-400}"
MAXUSD="${MAXUSD:-18}"          # global spend cap across chunks (real-money guardrail)
PY=.venv-controlarena/bin/python
TAG="${MODEL//[^a-zA-Z0-9]/}"
LOG="results/fewshot_chunked_${TAG}.log"
echo "=== chunked run model=$MODEL extra='$EXTRA' CHUNK=$CHUNK timeout=${CHUNK_TIMEOUT}s $(date '+%H:%M:%S') ===" | tee -a "$LOG"
stalls=0
for att in $(seq 1 "$MAXATT"); do
  tmp="$(mktemp)"
  $PY scripts/track2_fewshot_v2.py --model "$MODEL" $EXTRA --limit "$CHUNK" >"$tmp" 2>&1 &
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
  # surface only the meaningful lines from each chunk
  grep -hE '\+[0-9]+/|DONE|STOPPED|TIME-BOXED|ERR|nothing to do|STOPPED at' "$tmp" >>"$LOG" 2>/dev/null
  if grep -q 'nothing to do' "$tmp" 2>/dev/null; then
    echo "[wrapper] COMPLETE at att=$att ($(date '+%H:%M:%S')), stalls=$stalls" | tee -a "$LOG"
    rm -f "$tmp"; exit 0
  fi
  rm -f "$tmp"
  # GLOBAL budget guardrail across chunks (per-chunk --max-usd only sees one chunk's spend)
  cur=$($PY scripts/track2_fewshot_v2.py --model "$MODEL" --cost 2>/dev/null | grep -oE '\$[0-9.]+' | head -1 | tr -d '$')
  if [ -n "${cur:-}" ] && awk "BEGIN{exit !(${cur:-0} > $MAXUSD)}"; then
    echo "[wrapper] GLOBAL BUDGET \$$cur > \$$MAXUSD — STOPPING ($(date '+%H:%M:%S'))." | tee -a "$LOG"
    exit 2
  fi
  [ $((att % 10)) -eq 0 ] && echo "[wrapper att=$att] cumulative spend \$${cur:-?} ($(date '+%H:%M:%S'))" | tee -a "$LOG"
done
echo "[wrapper] hit MAXATT=$MAXATT without completion ($(date '+%H:%M:%S'))" | tee -a "$LOG"
exit 1
