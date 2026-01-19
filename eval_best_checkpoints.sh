#!/usr/bin/env bash
set -e

OUTPUT_DIR="./ddpm_diffusers_results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
INFERENCE_STEPS=50  # Use 200 for faster eval, 1000 for higher quality
NUM_SAMPLES=1000
EVAL_SPLIT="test"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/eval_log_${TIMESTAMP}.txt"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Function to log messages to both stdout and log file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

CHECKPOINTS=(
  # Early–mid (diversity / structure)
  "checkpoint_epoch_33.pt"
  "checkpoint_epoch_42.pt"
  "checkpoint_epoch_44.pt"
  "checkpoint_epoch_55.pt"
  "checkpoint_epoch_60.pt"
  "checkpoint_epoch_62.pt"
  "checkpoint_epoch_63.pt"
  "checkpoint_epoch_65.pt"
  "checkpoint_epoch_68.pt"
  "checkpoint_epoch_70.pt"

  # Mid (best balance – often best FID)
  "checkpoint_epoch_75.pt"
  "checkpoint_epoch_76.pt"
  "checkpoint_epoch_82.pt"
  "checkpoint_epoch_93.pt"
  "checkpoint_epoch_94.pt"
  "checkpoint_epoch_96.pt"
  "checkpoint_epoch_98.pt"
  "checkpoint_epoch_100.pt"
  "checkpoint_epoch_106.pt"
  "checkpoint_epoch_108.pt"

  # Best overall (lowest loss – late stage)
  "checkpoint_epoch_124.pt"
  "checkpoint_epoch_125.pt"
  "checkpoint_epoch_129.pt"
  "checkpoint_epoch_132.pt"
)

log "=============================================="
log "Evaluating DDPM checkpoints (Top + Best Overall)"
log "Inference steps: ${INFERENCE_STEPS}"
log "Samples: ${NUM_SAMPLES}"
log "Split: ${EVAL_SPLIT}"
log "Log file: ${LOG_FILE}"
log "=============================================="

for CKPT in "${CHECKPOINTS[@]}"; do
  CKPT_PATH="${CHECKPOINT_DIR}/${CKPT}"

  if [[ ! -f "${CKPT_PATH}" ]]; then
    log "❌ Checkpoint not found: ${CKPT_PATH}"
    continue
  fi

  log ""
  log "----------------------------------------------"
  log "▶ Evaluating ${CKPT}"
  log "----------------------------------------------"

  # Run evaluation and log both stdout and stderr
  python3 train.py \
    --eval_only \
    --calculate_fid \
    --checkpoint "${CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_eval_samples "${NUM_SAMPLES}" \
    --inference_steps "${INFERENCE_STEPS}" \
    2>&1 | tee -a "${LOG_FILE}"

done

log ""
log "✅ Evaluation completed for all checkpoints"
log "📄 Full log saved to: ${LOG_FILE}"
