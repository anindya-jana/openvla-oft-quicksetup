#!/usr/bin/env bash
# Smoke-test evaluation for OpenVLA-OFT on LIBERO
# - Uses combined checkpoint: moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
# - Runs libero_spatial suite with 2 trials per task
# - Keeps HF cache inside this repo at ./hf-cache (as requested)
#
# Usage:
#   bash scripts/run_libero_eval_smoke.sh
#   bash scripts/run_libero_eval_smoke.sh <ENV_NAME>             # default: libero_openvla
#   bash scripts/run_libero_eval_smoke.sh <ENV_NAME> --extra args # pass extra flags to eval script
#
# Examples:
#   bash scripts/run_libero_eval_smoke.sh
#   bash scripts/run_libero_eval_smoke.sh libero_openvla --load_in_8bit True
#
# Entrypoint calls: python eval_libero() in experiments/robot/libero/run_libero_eval.py

set -Eeuo pipefail

#-------------------------------------------
# Helpers
#-------------------------------------------
log()  { printf "\033[1;34m[libero-eval]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[libero-eval][warn]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[libero-eval][error]\033[0m %s\n" "$*" 1>&2; }

#-------------------------------------------
# Params
#-------------------------------------------
ENV_NAME="${1:-libero_openvla}"
shift || true  # shift once if provided; safe if not

# Extra args for eval script (optional)
EXTRA_ARGS=("$@")

#-------------------------------------------
# Repo paths
#-------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

# In-repo HF cache
HF_CACHE="$REPO_ROOT/hf-cache"
mkdir -p "$HF_CACHE"

#-------------------------------------------
# Headless rendering + caches
#-------------------------------------------
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

log "Repo root:                $REPO_ROOT"
log "Conda env name:           $ENV_NAME"
log "HF cache:                 $HF_CACHE"
log "MUJOCO_GL:                ${MUJOCO_GL}"
log "EGL_DEVICE_ID:            ${EGL_DEVICE_ID}"

#-------------------------------------------
# Python launcher (prefer conda-run)
#-------------------------------------------
PYTHON_LAUNCH=("python")
if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -xq "$ENV_NAME"; then
    PYTHON_LAUNCH=("conda" "run" "-n" "$ENV_NAME" "python")
  else
    warn "Conda env '$ENV_NAME' not found. Falling back to system python."
    warn "If you intended to use the env, run: conda activate $ENV_NAME"
  fi
else
  warn "conda not found on PATH; using system python."
fi

#-------------------------------------------
# Validate LIBERO install
#-------------------------------------------
if [ ! -d "./LIBERO" ]; then
  warn "LIBERO repo not found at ./LIBERO. You must clone and install it:"
  warn "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git"
  warn "  pip install -e ./LIBERO        # or conda run -n $ENV_NAME pip install -e ./LIBERO"
fi

#-------------------------------------------
# Build command
#-------------------------------------------
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
TASK_SUITE="libero_spatial"
TRIALS=2

CMD=(
  "${PYTHON_LAUNCH[@]}"
  "experiments/robot/libero/run_libero_eval.py"
  --pretrained_checkpoint "$CHECKPOINT"
  --task_suite_name "$TASK_SUITE"
  --num_trials_per_task "$TRIALS"
  --use_wandb False
  --center_crop True
)

# Append any extra args passed to this script
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

log "Running smoke-test evaluation..."
log "Command:"
printf "  %q " "${CMD[@]}"; printf "\n"

#-------------------------------------------
# Execute
#-------------------------------------------
"${CMD[@]}"

#-------------------------------------------
# Post info
#-------------------------------------------
log "If successful, check outputs:"
log "  Logs:     ./experiments/logs/EVAL-*.txt"
log "  Rollouts: ./rollouts/<DATE>/*.mp4"

log "Tip: To reduce VRAM if needed, re-run and append one of:"
log "  --load_in_8bit True    or    --load_in_4bit True"