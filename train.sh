#!/usr/bin/env bash
set -e

DIRECTION="both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --direction) DIRECTION="$2"; shift 2;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

COMMON_ARGS=(
  --calculate_fid
  --inference_steps 50
  --num_eval_samples 5000
  --eval_freq 5
)

if [[ "$DIRECTION" == "theta" || "$DIRECTION" == "both" ]]; then
  echo "=== Training THETA ==="
  python3 traincond3.py \
    --direction theta \
    --overall_init ddpm_uncond_all/theta_ref_ema_best.pt \
    --output_dir ddpm_diffusers_results_theta_theta \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
fi

if [[ "$DIRECTION" == "phi" || "$DIRECTION" == "both" ]]; then
  echo "=== Training PHI ==="
  python3 traincond3.py \
    --direction phi \
    --ctrl_init ddpm_uncond_ctrl/theta_ctrl_ema_best.pt \
    --output_dir ddpm_diffusers_results_phi_phi \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
fi
