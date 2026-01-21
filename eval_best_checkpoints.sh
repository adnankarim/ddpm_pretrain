#!/usr/bin/env bash
set -e

# Configuration
NUM_SAMPLES=1000
EVAL_SPLIT="test"
INFERENCE_STEPS_LIST=(50 1000)  # Evaluate with both 50 and 1000 steps

# Checkpoints from ppo.py
THETA_CHECKPOINT="./ddpm_diffusers_results/checkpoints/checkpoint_epoch_60.pt"
PHI_CHECKPOINT="./results_phi_phi/checkpoints/checkpoint_epoch_100.pt"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./eval_log_${TIMESTAMP}.txt"

# Ensure output directory exists
mkdir -p "./eval_results"

# Function to log messages to both stdout and log file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

log "=============================================="
log "Evaluating Theta and Phi Models"
log "Samples: ${NUM_SAMPLES}"
log "Split: ${EVAL_SPLIT}"
log "Inference steps: ${INFERENCE_STEPS_LIST[*]}"
log "Log file: ${LOG_FILE}"
log "=============================================="

# Models to evaluate: (direction, checkpoint_path, output_dir)
MODELS=(
  "theta|${THETA_CHECKPOINT}|./ddpm_diffusers_results"
  "phi|${PHI_CHECKPOINT}|./results_phi_phi"
)

for MODEL_SPEC in "${MODELS[@]}"; do
  IFS='|' read -r DIRECTION CKPT_PATH OUTPUT_DIR <<< "${MODEL_SPEC}"
  
  if [[ ! -f "${CKPT_PATH}" ]]; then
    log ""
    log "❌ Checkpoint not found: ${CKPT_PATH}"
    log "   Skipping ${DIRECTION} model evaluation"
    continue
  fi

  log ""
  log "=============================================="
  log "Evaluating ${DIRECTION^^} Model"
  log "Checkpoint: ${CKPT_PATH}"
  log "Output dir: ${OUTPUT_DIR}"
  log "=============================================="

  for INFERENCE_STEPS in "${INFERENCE_STEPS_LIST[@]}"; do
    log ""
    log "----------------------------------------------"
    log "▶ ${DIRECTION^^} Model - ${INFERENCE_STEPS} inference steps"
    log "----------------------------------------------"

    # Run evaluation and log both stdout and stderr
    python3 train2.py \
      --eval_only \
      --calculate_fid \
      --checkpoint "${CKPT_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --eval_split "${EVAL_SPLIT}" \
      --num_eval_samples "${NUM_SAMPLES}" \
      --inference_steps "${INFERENCE_STEPS}" \
      --direction "${DIRECTION}" \
      2>&1 | tee -a "${LOG_FILE}"

    log ""
    log "✓ Completed ${DIRECTION} evaluation with ${INFERENCE_STEPS} steps"
  done
done

log ""
log "=============================================="
log "✅ Evaluation completed for all models"
log "📄 Full log saved to: ${LOG_FILE}"
log "=============================================="