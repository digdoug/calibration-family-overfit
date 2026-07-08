#!/bin/bash
# 34B rung re-run (the lost pod's missing rung). Yi-1.5-34B-Chat, 2 seeds.
# Results pulled incrementally by the launcher after each seed's npz is written,
# so a pod death cannot lose them (the first run's lesson: volumeSize 0 wipes on stop).
set -uo pipefail
cd /root/smt
LOG=/root/staircase34b.log
echo "[$(date)] 34b start" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
if [ ! -x .venv-gpu/bin/python ]; then
  python3 -m venv .venv-gpu --system-site-packages
  .venv-gpu/bin/pip install -q transformers peft accelerate sentencepiece scikit-learn numpy 2>&1 | tail -1 | tee -a "$LOG"
fi
PY=.venv-gpu/bin/python
mkdir -p results/track2_lora
run() { local seed=$1
  echo "[$(date)] === yi34b_seed$seed ===" | tee -a "$LOG"
  CKPT_PATH=results/track2_lora/yi34b_seed$seed.npz TRAIN_SEED=$seed BASE_MODEL=01-ai/Yi-1.5-34B-Chat \
    MAX_LEN=1536 BATCH=1 EPOCHS=4 $PY scripts/track2_lora_monitor.py >>"$LOG" 2>&1 \
    && echo "[$(date)] yi34b_seed$seed OK" | tee -a "$LOG" \
    || echo "[$(date)] yi34b_seed$seed FAILED rc=$?" | tee -a "$LOG"
}
run 0
run 1
echo "[$(date)] 34b ALL DONE" | tee -a "$LOG"
