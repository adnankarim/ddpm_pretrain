#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Conditional DDPM training runner
# ==============================
# Assumes this file lives in: ~/ddpm_pretrain/
# and train.py is in the same folder.
#
# Default inits:
#   theta init  -> ddpm_uncond_all/theta_ref_ema_best.pt
#   phi init    -> ddpm_uncond_ctrl/theta_ctrl_ema_best.pt
#
# Examples:
#   ./train_conditional.sh --direction theta
#   ./train_conditional.sh --direction phi
#   ./train_conditional.sh --direction both
#   ./train_conditional.sh --direction theta --overall_init ddpm_uncond_all/checkpoints/checkpoint_epoch_200.pt
#   ./train_conditional.sh --direction both --epochs_each 100 --output_base my_runs
#   ./train_conditional.sh --direction theta --resume
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train.py"

# ---- Defaults ----
DIRECTION="theta"                # theta | phi | both
EPOCHS_EACH=50
OUTPUT_BASE="ddpm_diffusers_results"
INFERENCE_STEPS=50
NUM_EVAL_SAMPLES=1000

CALCULATE_FID=0
FID_ONLY=0
RESUME=0
TRAIN_CONV_IN=0
PATHS_CSV=""

OVERALL_INIT="ddpm_uncond_all/theta_ref_ema_best.pt"
CTRL_INIT="ddpm_uncond_ctrl/theta_ctrl_ema_best.pt"

# ---- Arg parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --direction)        DIRECTION="${2:-}"; shift 2 ;;
    --epochs_each)      EPOCHS_EACH="${2:-}"; shift 2 ;;
    --output_base)      OUTPUT_BASE="${2:-}"; shift 2 ;;
    --overall_init)     OVERALL_INIT="${2:-}"; shift 2 ;;
    --ctrl_init)        CTRL_INIT="${2:-}"; shift 2 ;;
    --paths_csv)        PATHS_CSV="${2:-}"; shift 2 ;;
    --inference_steps)  INFERENCE_STEPS="${2:-}"; shift 2 ;;
    --num_eval_samples) NUM_EVAL_SAMPLES="${2:-}"; shift 2 ;;
    --calculate_fid)    CALCULATE_FID=1; shift 1 ;;
    --fid_only)         FID_ONLY=1; shift 1 ;;
    --resume)           RESUME=1; shift 1 ;;
    --train_conv_in)    TRAIN_CONV_IN=1; shift 1 ;;
    -h|--help)
      cat <<EOF
Usage:
  ./train_conditional.sh [options]

Options:
  --direction theta|phi|both        (default: theta)
  --epochs_each N                   (default: 50)
  --output_base NAME                (default: ddpm_diffusers_results)
  --overall_init PATH               (default: ddpm_uncond_all/theta_ref_ema_best.pt)
  --ctrl_init PATH                  (default: ddpm_uncond_ctrl/theta_ctrl_ema_best.pt)
  --paths_csv PATH                  (optional)
  --inference_steps N               (default: 200)
  --num_eval_samples N              (default: 1000)
  --calculate_fid                   enable FID/KID during eval
  --fid_only                        skip KL/MSE/PSNR/SSIM in eval
  --resume                          resume from latest.pt in phase output dir
  --train_conv_in                   also train conv_in (optional)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# ---- Build common args ----
COMMON_ARGS=( "--epochs_each" "${EPOCHS_EACH}" "--inference_steps" "${INFERENCE_STEPS}" "--num_eval_samples" "${NUM_EVAL_SAMPLES}" )
if [[ "${CALCULATE_FID}" -eq 1 ]]; then COMMON_ARGS+=( "--calculate_fid" ); fi
if [[ "${FID_ONLY}" -eq 1 ]]; then COMMON_ARGS+=( "--fid_only" ); fi
if [[ "${RESUME}" -eq 1 ]]; then COMMON_ARGS+=( "--resume" ); fi
if [[ "${TRAIN_CONV_IN}" -eq 1 ]]; then COMMON_ARGS+=( "--train_conv_in" ); fi
if [[ -n "${PATHS_CSV}" ]]; then COMMON_ARGS+=( "--paths_csv" "${PATHS_CSV}" ); fi

run_theta() {
  local outdir="${OUTPUT_BASE}_theta"
  mkdir -p "${SCRIPT_DIR}/${outdir}/logs"
  echo "=== Training THETA ==="
  echo "  overall_init: ${OVERALL_INIT}"
  echo "  outdir:       ${outdir}"
  python3 "${TRAIN_PY}" \
    --direction theta \
    --overall_init "${OVERALL_INIT}" \
    --output_dir "${outdir}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/${outdir}/logs/train_theta_$(date +%Y%m%d_%H%M%S).log"
}

run_phi() {
  local outdir="${OUTPUT_BASE}_phi"
  mkdir -p "${SCRIPT_DIR}/${outdir}/logs"
  echo "=== Training PHI ==="
  echo "  ctrl_init: ${CTRL_INIT}"
  echo "  outdir:    ${outdir}"
  python3 "${TRAIN_PY}" \
    --direction phi \
    --ctrl_init "${CTRL_INIT}" \
    --output_dir "${outdir}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/${outdir}/logs/train_phi_$(date +%Y%m%d_%H%M%S).log"
}

# ---- Go ----
cd "${SCRIPT_DIR}"

case "${DIRECTION}" in
  theta) run_theta ;;
  phi)   run_phi ;;
  both)
    run_theta
    run_phi
    ;;
  *)
    echo "Invalid --direction: ${DIRECTION} (use theta|phi|both)"
    exit 1
    ;;
esac

echo "✅ Done."
