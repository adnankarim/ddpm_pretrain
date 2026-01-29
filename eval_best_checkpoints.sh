#!/usr/bin/env bash
set -e

# Configuration
NUM_SAMPLES=5000
EVAL_SPLIT="test"
INFERENCE_STEPS_LIST=(1000)  # Evaluate with both 50 and 1000 steps

# Checkpoints from ppo.py
PHI_CHECKPOINT="./results_phi_phi/checkpoints/checkpoint_epoch_100.pt"

# Checkpoints from sdlora.py
SDLORA_CHECKPOINTS=(
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_1.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_2.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_3.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_4.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_5.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_6.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_7.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_8.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_9.pt"
  "./controlnet_lora_results/checkpoints/checkpoint_epoch_10.pt"
)

# Checkpoints from flux.py
FLUX_CHECKPOINT_DIR="./out_flux_bbbc021/checkpoint-4000"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./eval_log_${TIMESTAMP}.txt"

# Ensure output directory exists
mkdir -p "./eval_results"
mkdir -p "./outputs/evaluation"

# Function to log messages to both stdout and log file
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

log "=============================================="
log "Evaluating All Models (PPO, SDLoRA, FLUX)"
log "Samples: ${NUM_SAMPLES}"
log "Split: ${EVAL_SPLIT}"
log "Inference steps: ${INFERENCE_STEPS_LIST[*]}"
log "Log file: ${LOG_FILE}"
log "=============================================="

# ============================================================================
# 1. Evaluate PPO Models (train2.py)
# ============================================================================
log ""
log "=============================================="
log "SECTION 1: PPO Models (train2.py)"
log "=============================================="

# Models to evaluate: (direction, checkpoint_path, output_dir)
PPO_MODELS=(
  "theta|${THETA_CHECKPOINT}|./ddpm_diffusers_results"
  "phi|${PHI_CHECKPOINT}|./results_phi_phi"
)

for MODEL_SPEC in "${PPO_MODELS[@]}"; do
  IFS='|' read -r DIRECTION CKPT_PATH OUTPUT_DIR <<< "${MODEL_SPEC}"
  
  if [[ ! -f "${CKPT_PATH}" ]]; then
    log ""
    log "❌ Checkpoint not found: ${CKPT_PATH}"
    log "   Skipping ${DIRECTION} model evaluation"
    continue
  fi

  log ""
  log "=============================================="
  log "Evaluating ${DIRECTION^^} Model (PPO)"
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

# ============================================================================
# 2. Evaluate SDLoRA Models (sdlora.py)
# ============================================================================
log ""
log "=============================================="
log "SECTION 2: SDLoRA Models (sdlora.py)"
log "=============================================="

for CKPT_PATH in "${SDLORA_CHECKPOINTS[@]}"; do
  if [[ ! -f "${CKPT_PATH}" ]]; then
    log ""
    log "❌ Checkpoint not found: ${CKPT_PATH}"
    log "   Skipping this checkpoint"
    continue
  fi

  # Extract epoch number from checkpoint path
  EPOCH_NUM=$(basename "${CKPT_PATH}" | sed 's/checkpoint_epoch_\([0-9]*\)\.pt/\1/')
  
  log ""
  log "=============================================="
  log "Evaluating SDLoRA Model - Epoch ${EPOCH_NUM}"
  log "Checkpoint: ${CKPT_PATH}"
  log "=============================================="

  for INFERENCE_STEPS in "${INFERENCE_STEPS_LIST[@]}"; do
    log ""
    log "----------------------------------------------"
    log "▶ SDLoRA Epoch ${EPOCH_NUM} - ${INFERENCE_STEPS} inference steps"
    log "----------------------------------------------"

    # Run evaluation and log both stdout and stderr
    python3 sdlora.py \
      --eval_only \
      --checkpoint "${CKPT_PATH}" \
      --eval_split "${EVAL_SPLIT}" \
      --num_samples "${NUM_SAMPLES}" \
      --inference_steps "${INFERENCE_STEPS}" \
      --output_dir "./controlnet_lora_results" \
      2>&1 | tee -a "${LOG_FILE}"

    log ""
    log "✓ Completed SDLoRA Epoch ${EPOCH_NUM} evaluation with ${INFERENCE_STEPS} steps"
  done
done

# ============================================================================
# 3. Evaluate FLUX Model (flux.py)
# ============================================================================
log ""
log "=============================================="
log "SECTION 3: FLUX Model (flux.py)"
log "=============================================="

if [[ ! -d "${FLUX_CHECKPOINT_DIR}" ]]; then
  log ""
  log "❌ Checkpoint directory not found: ${FLUX_CHECKPOINT_DIR}"
  log "   Skipping FLUX model evaluation"
else
  log ""
  log "=============================================="
  log "Evaluating FLUX Model"
  log "Checkpoint dir: ${FLUX_CHECKPOINT_DIR}"
  log "=============================================="

  for INFERENCE_STEPS in "${INFERENCE_STEPS_LIST[@]}"; do
    log ""
    log "----------------------------------------------"
    log "▶ FLUX Model - ${INFERENCE_STEPS} inference steps"
    log "----------------------------------------------"

    # Run evaluation and log both stdout and stderr
    # flux.py uses Accelerator, can run directly (handles single-GPU automatically)
    python3 flux.py \
      --eval_only \
      --checkpoint_dir "${FLUX_CHECKPOINT_DIR}" \
      --eval_split "${EVAL_SPLIT}" \
      --num_samples "${NUM_SAMPLES}" \
      --inference_steps "${INFERENCE_STEPS}" \
      --output_dir "./out_flux_bbbc021" \
      --data_dir "./data/bbbc021_all" \
      --metadata_file "./metadata/bbbc021_df_all.csv" \
      --paths_csv "./data/bbbc021_all/paths.csv" \
      --resolution 96 \
      --pretrained_model "black-forest-labs/FLUX.1-dev" \
      2>&1 | tee -a "${LOG_FILE}"

    log ""
    log "✓ Completed FLUX evaluation with ${INFERENCE_STEPS} steps"
  done
fi

log ""
log "=============================================="
log "✅ Evaluation completed for all models"
log "📄 Full log saved to: ${LOG_FILE}"
log "📊 Results saved to: ./outputs/evaluation/"
log "=============================================="