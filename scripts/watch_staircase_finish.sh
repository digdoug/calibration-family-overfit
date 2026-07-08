#!/bin/bash
# Control 15 — detached watcher to DECOUPLE GPU billing from session timing.
# Polls the RunPod staircase pod until the final rung (Yi-34B seed1) lands, then:
#   (1) harvests all 4 npz + regens fig6 via analysis_staircase_fold.sh,
#   (2) saves the pod's staircase.log tail locally (it dies with the ephemeral pod),
#   (3) STOPS the pod so the $1.39/hr A100 stops billing immediately.
# It does NOT touch the paper — the local-only fold (one table row + one sentence
# in §4.2) runs separately from the harvested npz.
#
# Launched detached (nohup + caffeinate) so the GPU stops promptly regardless of
# session timing. Self-managing; safe to run once.
#
# Completion signal (BOTH required, robust against mid-write):
#   - "staircase ALL DONE" present in /root/staircase.log (authoritative), OR
#   - yi34b_seed1.npz size == 60114 bytes (matches the 3 completed rungs exactly).
# Safety: bails WITHOUT stopping the pod if it has polled >4h (lets a human decide).
set -uo pipefail
cd "$(dirname "$0")/.."

POD_ID=YOUR_POD_ID
POD_HOST=YOUR_POD_IP
POD_PORT=30002
SEED1_BYTES=60114            # exact size of a completed 4-family rung npz
MARKER=results/track2_lora_staircase/SEED1_DONE.marker
LOGSAVE=results/track2_lora_staircase/staircase_pod.log
mkdir -p results/track2_lora_staircase

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $POD_PORT root@$POD_HOST"

echo "[watch] start $(date -u +%H:%M:%S)UTC  polling pod $POD_ID for Yi-34B seed1"
DEADLINE=$(( $(date +%s) + 4*3600 ))
while :; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then
    echo "[watch] $(date -u +%H:%M:%S)UTC TIMEOUT (>4h) — NOT stopping pod; leaving for a human."
    exit 2
  fi
  # one round-trip: grab log-done flag + seed1 size
  STATUS=$($SSH 'grep -q "staircase ALL DONE" /root/staircase.log 2>/dev/null && echo LOGDONE; stat -c %s /root/smt/results/track2_lora/yi34b_seed1.npz 2>/dev/null || echo 0' 2>/dev/null)
  LOGDONE=$(echo "$STATUS" | grep -c LOGDONE)
  SZ=$(echo "$STATUS" | grep -vE LOGDONE | tail -1)
  SZ=${SZ:-0}
  echo "[watch] $(date -u +%H:%M:%S)UTC  logdone=$LOGDONE  seed1_bytes=$SZ"
  if [ "$LOGDONE" -ge 1 ] || [ "$SZ" = "$SEED1_BYTES" ]; then
    echo "[watch] completion detected -> harvesting + saving log + stopping pod"
    $SSH 'tail -60 /root/staircase.log' > "$LOGSAVE" 2>/dev/null || true
    bash scripts/analysis_staircase_fold.sh > results/track2_lora_staircase/fold_at_finish.log 2>&1 || true
    NPZ=$(ls results/track2_lora_staircase/{phi4_seed0,phi4_seed1,yi34b_seed0,yi34b_seed1}.npz 2>/dev/null | wc -l | tr -d ' ')
    echo "[watch] harvested $NPZ/4 npz"
    if [ "$NPZ" = "4" ]; then
      runpodctl stop pod "$POD_ID" 2>&1 | sed 's/^/[watch] runpodctl: /' || true
      echo "STOPPED $(date -u +%H:%M:%S)UTC  npz=4/4  pod=$POD_ID" > "$MARKER"
      echo "[watch] DONE — pod stopped, marker written: $MARKER"
    else
      echo "[watch] harvest incomplete ($NPZ/4) — NOT stopping pod; leaving for a human."
      echo "INCOMPLETE $(date -u +%H:%M:%S)UTC npz=$NPZ/4" > "$MARKER"
    fi
    exit 0
  fi
  sleep 120
done
