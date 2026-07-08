#!/bin/bash
# Monitor-capability staircase (Control 15): train LoRA monitors at increasing
# base-model scale, holding split/data/epochs/rank fixed, to show the
# cross-family interaction is NOT a weak-monitor artifact. Runs DETACHED on a
# RunPod A100-80GB. Existing rung: Phi-3.5-mini 3.8B = +0.179 (Control 11).
# New rungs here: Phi-4 14B, Yi-1.5-34B-Chat 34B. All vendor-disjoint from every
# policy family (the design doc s3). Distinct CKPT_PATH per run (the script's default
# is seed-suffixed only, so different bases would collide).
set -uo pipefail
cd /root/smt
LOG=/root/staircase.log
echo "[$(date)] staircase start" | tee -a "$LOG"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$LOG"

# env once (base image ships torch+cuda; add the rest, inherit torch via system-site-packages)
if [ ! -x .venv-gpu/bin/python ]; then
  python3 -m venv .venv-gpu --system-site-packages
  .venv-gpu/bin/pip install -q transformers peft accelerate sentencepiece scikit-learn numpy 2>&1 | tail -2 | tee -a "$LOG"
fi
PY=.venv-gpu/bin/python
$PY -c "import torch;print('torch',torch.__version__,'cuda_ok',torch.cuda.is_available())" | tee -a "$LOG"
mkdir -p results/track2_lora

run() {  # base seed maxlen batch tag
  local base=$1 seed=$2 maxlen=$3 batch=$4 tag=$5
  echo "[$(date)] === $tag (base=$base seed=$seed maxlen=$maxlen batch=$batch) ===" | tee -a "$LOG"
  CKPT_PATH=results/track2_lora/${tag}.npz TRAIN_SEED=$seed BASE_MODEL=$base \
    MAX_LEN=$maxlen BATCH=$batch EPOCHS=4 \
    $PY scripts/track2_lora_monitor.py >>"$LOG" 2>&1 \
    && echo "[$(date)] $tag OK" | tee -a "$LOG" \
    || echo "[$(date)] $tag FAILED rc=$?" | tee -a "$LOG"
}

# Rung 1: Phi-4 14B (clean 80GB fit) — 2 seeds. Doubles as the smoke gate:
# if mistral & llama diagonal AUROC < 0.70 here, larger rungs are unlikely to help.
run microsoft/Phi-4 0 2048 2 phi4_seed0
run microsoft/Phi-4 1 2048 2 phi4_seed1
# Rung 2 (headline strong monitor): Yi-1.5-34B-Chat — 2 seeds, conservative for 80GB bf16.
run 01-ai/Yi-1.5-34B-Chat 0 1536 1 yi34b_seed0
run 01-ai/Yi-1.5-34B-Chat 1 1536 1 yi34b_seed1
echo "[$(date)] staircase ALL DONE" | tee -a "$LOG"
