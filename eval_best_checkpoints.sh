#!/usr/bin/env bash
set -e

OUTPUT_DIR="./ddpm_diffusers_results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
INFERENCE_STEPS=50  # Use 200 for faster eval, 1000 for higher quality
NUM_SAMPLES=1000
EVAL_SPLIT="val"

CHECKPOINTS=(
  "checkpoint_epoch_68.pt"
  "checkpoint_epoch_106.pt"
  "checkpoint_epoch_129.pt"
)

echo "=============================================="
echo "Evaluating best DDPM checkpoints"
echo "Inference steps: ${INFERENCE_STEPS}"
echo "Samples: ${NUM_SAMPLES}"
echo "Split: ${EVAL_SPLIT}"
echo "=============================================="

for CKPT in "${CHECKPOINTS[@]}"; do
  CKPT_PATH="${CHECKPOINT_DIR}/${CKPT}"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "❌ Checkpoint not found: ${CKPT_PATH}"
    continue
  fi

  echo ""
  echo "----------------------------------------------"
  echo "▶ Evaluating ${CKPT}"
  echo "----------------------------------------------"

  python3 train.py \
    --eval_only \
    --calculate_fid \
    --checkpoint "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_eval_samples "${NUM_SAMPLES}" \
    --inference_steps "${INFERENCE_STEPS}"

done

echo ""
echo "✅ Evaluation completed for all checkpoints"
