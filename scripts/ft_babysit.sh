#!/bin/bash
# item12b Vertex FT babysitter: poll tuning-job heartbeats every 30 min until all
# non-mistral jobs reach a terminal state. Pre-registered patience protocol:
# ~5h RUNNING at 0 TensorBoard steps is normal provisioning; done =
# SUCCEEDED + servable endpoint; never cancel on status alone.
# Launch: nohup caffeinate -i -s bash scripts/ft_babysit.sh > /dev/null 2>&1 & disown
set -u
cd "$(dirname "$0")/.."
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcp-sa.json" GCP_LOCATION=europe-west4
LOG=results/track2_ftmon_vertex/babysit_20260707.log
JOBS=results/track2_ftmon_vertex/jobs.json
echo "BABYSIT START $(date -u +%FT%TZ) pid=$$" >> "$LOG"
while true; do
  ts=$(date -u +%FT%TZ)
  hb=$(.venv-controlarena/bin/python scripts/ft_monitor_vertex.py heartbeat 2>&1) || true
  echo "POLL $ts" >> "$LOG"
  echo "$hb" | grep -v ExperimentalWarning >> "$LOG"
  states=$(python3 -c "
import json
j = json.load(open('$JOBS'))
print(' '.join(f\"{s}:{v['state'].split('.')[-1]}\" for s, v in j.items() if s != 'mistral'))" 2>>"$LOG") || true
  echo "STATES $ts $states" >> "$LOG"
  for kv in $states; do
    s=${kv%%:*}; st=${kv##*:}
    case $st in
      *SUCCEEDED*|*FAILED*|*CANCELLED*|*EXPIRED*)
        grep -q "TERMINAL $s " "$LOG" || echo "TERMINAL $s $st $ts" >> "$LOG";;
    esac
  done
  n=$(grep -c "^TERMINAL " "$LOG" || true)
  if [ "${n:-0}" -ge 3 ]; then echo "ALLDONE $ts" >> "$LOG"; break; fi
  sleep 1800
done
