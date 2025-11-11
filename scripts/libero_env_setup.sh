#!/usr/bin/env bash
# Setup script for no-training LIBERO evaluation with OpenVLA-OFT
# - Creates conda env: libero_openvla (Python 3.10.14)
# - Installs: PyTorch 2.2.0 + cu121, this repo (editable), LIBERO (editable), LIBERO extras
# - Prepares in-repo HF cache at ./hf-cache
# Usage:
#   bash scripts/libero_env_setup.sh            # uses default env name: libero_openvla
#   bash scripts/libero_env_setup.sh <ENV>      # custom env name
#
# After running, either:
#   - conda activate <ENV>
#   - or use: conda run -n <ENV> python ...
set -Eeuo pipefail

#-------------------------------------------
# Helpers
#-------------------------------------------
log() { printf "\033[1;34m[libero-setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[libero-setup][warn]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[libero-setup][error]\033[0m %s\n" "$*" 1>&2; }
die() { err "$*"; exit 1; }

#-------------------------------------------
# Pre-flight checks
#-------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  die "conda not found. Please install Miniconda/Anaconda and ensure 'conda' is on PATH."
fi

# Determine repo root (script is under ./scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${1:-libero_openvla}"
PY_VER="3.10.14"

log "Repository root: $REPO_ROOT"
log "Target conda env: $ENV_NAME (Python $PY_VER)"
log "This will install PyTorch 2.2.0 + cu121 and project dependencies."

#-------------------------------------------
# Create env if missing
#-------------------------------------------
if conda env list | awk '{print $1}' | grep -xq "$ENV_NAME"; then
  log "Conda env '$ENV_NAME' already exists. Skipping creation."
else
  log "Creating conda env '$ENV_NAME' with Python $PY_VER..."
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

#-------------------------------------------
# Install PyTorch 2.2.0 + cu121
#-------------------------------------------
log "Installing PyTorch 2.2.0 + cu121 (torch==2.2.0, torchvision==0.17.0, torchaudio==2.2.0)..."
conda run -n "$ENV_NAME" python -c "import sys; print(sys.version)"
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

#-------------------------------------------
# Install this repo (editable)
#-------------------------------------------
log "Installing this repo in editable mode (pip install -e .)..."
conda run -n "$ENV_NAME" pip install -e .

#-------------------------------------------
# Clone and install LIBERO
#-------------------------------------------
if [ -d "$REPO_ROOT/LIBERO" ]; then
  log "LIBERO repo already exists at ./LIBERO. Skipping clone."
else
  log "Cloning LIBERO into ./LIBERO..."
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi

log "Installing LIBERO (editable)..."
conda run -n "$ENV_NAME" pip install -e ./LIBERO

#-------------------------------------------
# Install LIBERO-specific requirements from openvla-oft
#-------------------------------------------
REQ_FILE="experiments/robot/libero/libero_requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
  die "Requirements file not found: $REQ_FILE"
fi
log "Installing LIBERO-specific requirements from $REQ_FILE ..."
conda run -n "$ENV_NAME" pip install -r "$REQ_FILE"

#-------------------------------------------
# Prepare in-repo HF cache
#-------------------------------------------
HF_CACHE="$REPO_ROOT/hf-cache"
mkdir -p "$HF_CACHE"
log "Prepared HF cache directory at: $HF_CACHE"

#-------------------------------------------
# Rendering / headless tips
#-------------------------------------------
cat <<'EOS'
Notes:
- For headless rendering on NVIDIA, set:
    export MUJOCO_GL=egl
    export EGL_DEVICE_ID=0
- If EGL is unavailable, try:
    export MUJOCO_GL=osmesa   # Requires libosmesa6
- Some systems need system GL libs installed (e.g., libgl1, libegl1).

EOS

#-------------------------------------------
# Quick validations
#-------------------------------------------
log "Validating PyTorch install and CUDA availability..."
conda run -n "$ENV_NAME" python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
PY

# Print next steps
cat <<EOS
----------------------------------------------------------------------------
Setup complete.

Next steps (choose one):

A) Activate the environment in your shell:
   conda activate $ENV_NAME

   Then set caches and run smoke test:
   export HF_HOME="$REPO_ROOT/hf-cache"
   export TRANSFORMERS_CACHE="$REPO_ROOT/hf-cache"
   export MUJOCO_GL=egl
   export EGL_DEVICE_ID=0
   bash scripts/run_libero_eval_smoke.sh

B) Without activating, run via conda-run:
   HF_HOME="$REPO_ROOT/hf-cache" TRANSFORMERS_CACHE="$REPO_ROOT/hf-cache" \
   MUJOCO_GL=egl EGL_DEVICE_ID=0 \
   conda run -n $ENV_NAME \
   python experiments/robot/libero/run_libero_eval.py \
     --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
     --task_suite_name libero_spatial \
     --num_trials_per_task 2 \
     --use_wandb False \
     --center_crop True

Outputs:
- Logs:        ./experiments/logs/EVAL-*.txt
- Rollouts:    ./rollouts/<DATE>/...mp4

If you encounter OOM or need reduced VRAM, rerun with:
  --load_in_8bit True    OR    --load_in_4bit True
----------------------------------------------------------------------------
EOS

log "Done."