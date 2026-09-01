#!/usr/bin/env bash
# Run every stage end to end, then score all four checkpoints on the same prompts.
#
#   ./scripts/run_pipeline.sh                    # configs/toy.yaml  (~25 min)
#   ./scripts/run_pipeline.sh configs/smoke.yaml # ~1 min wiring check

set -euo pipefail

CONFIG="${1:-configs/toy.yaml}"
PY="${PYTHON:-python3}"
OUT="$($PY -c "from nanoseedance.config import Config; print(Config.load('$CONFIG').train.out_dir)")"

echo "=============================================================="
echo " nano-seedance pipeline    config=$CONFIG    out=$OUT"
echo "=============================================================="

echo; echo "--- stage 1/5: autoencoders ---------------------------------"
$PY -m nanoseedance.train_vae --config "$CONFIG"

echo; echo "--- stage 2/5: DiT pretraining ------------------------------"
$PY -m nanoseedance.train_dit --config "$CONFIG"

echo; echo "--- stage 3/5: supervised fine-tuning -----------------------"
$PY -m nanoseedance.sft --config "$CONFIG"

echo; echo "--- stage 4/5: RL, Flow-GRPO --------------------------------"
$PY -m nanoseedance.rl.grpo --config "$CONFIG"

echo; echo "--- stage 5/5: RL, Diffusion-DPO ----------------------------"
$PY -m nanoseedance.rl.dpo --config "$CONFIG"

echo; echo "--- evaluation ----------------------------------------------"
$PY -m nanoseedance.evaluate --config "$CONFIG" \
    --ckpt "$OUT/dit/ckpt_last.pt" \
           "$OUT/sft/ckpt_last.pt" \
           "$OUT/grpo/ckpt_last.pt" \
           "$OUT/dpo/ckpt_last.pt"

echo; echo "--- samples --------------------------------------------------"
$PY -m nanoseedance.sample --config "$CONFIG" --ckpt "$OUT/sft/ckpt_last.pt"

echo; echo "done. artifacts under $OUT/"
