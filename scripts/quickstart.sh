#!/usr/bin/env bash
# LensTrainer-LoboForge — clone, venv, deps. Run from repo root or via curl | bash.
#
# RunPod GPU pod (fresh PyTorch template): use scripts/setup_runpod.sh instead —
# it installs apt deps, checks nvidia-smi, smoke-tests imports, and writes runpod_env.sh.
set -euo pipefail

REPO_URL="${LOBFORGE_TRAINER_REPO:-https://github.com/LoboForge/LoboForge-LensTrainer.git}"
INSTALL_DIR="${LOBFORGE_TRAINER_DIR:-${HOME}/LoboForge-LensTrainer}"
PYTHON="${PYTHON:-python3}"

run_from_repo_root() {
  if [[ -f "train.py" && -f "requirements.txt" ]]; then
    pwd
    return 0
  fi
  return 1
}

if run_from_repo_root >/dev/null; then
  ROOT="$(pwd)"
else
  echo "==> Cloning LensTrainer-LoboForge into ${INSTALL_DIR}"
  if [[ -d "${INSTALL_DIR}/.git" ]]; then
    echo "    Repo already exists — pulling latest"
    git -C "${INSTALL_DIR}" pull --ff-only
  else
    git clone "${REPO_URL}" "${INSTALL_DIR}"
  fi
  ROOT="${INSTALL_DIR}"
  cd "${ROOT}"
fi

echo "==> Using repo: ${ROOT}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "error: ${PYTHON} not found. Install Python 3.11+ and retry." >&2
  exit 1
fi

echo "==> Creating venv (.venv)"
"${PYTHON}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies (this may take a few minutes)"
pip install -U pip wheel
pip install -r requirements.txt

# shellcheck disable=SC1091
source "${ROOT}/scripts/install_microsoft_lens.sh"
install_microsoft_lens "${ROOT}"

cat <<EOF

================================================================================
Setup complete.

Before first train:
  1. huggingface-cli login
  2. Accept the license at https://huggingface.co/microsoft/Lens-Base

Put images + sidecar .txt captions in a folder (see README "Dataset layout").

Train (16GB+ GPU, default preset):
  cd ${ROOT}
  source .venv/bin/activate
  export PYTHONPATH="${ROOT}/vendor/Lens:\${PYTHONPATH}"
  python train.py configs/train_lora_lens_base_24gb.yaml \\
    --set dataset.folder_path=/path/to/your/dataset \\
    --set sample.trigger_word=your_trigger \\
    --set job.output_dir=./output/my-lora

Local assembled weights (optional):
  python train.py configs/train_lora_lens_base_24gb.yaml \\
    --set model.repo_id=./models/Lens-Base \\
    --set dataset.folder_path=/path/to/your/dataset \\
    --set sample.trigger_word=your_trigger

Output: job.output_dir/lora_final.safetensors (ComfyUI-compatible)

VRAM: minimum 16GB GPU + 32GB system RAM (see README).
================================================================================
EOF

if [[ -n "${DATASET_PATH:-}" ]]; then
  TRIGGER="${TRIGGER_WORD:-mytrigger}"
  OUTPUT="${OUTPUT_DIR:-./output/lens-lora-run}"
  MODEL="${MODEL_REPO:-microsoft/Lens-Base}"
  echo "==> DATASET_PATH set — starting training"
  python train.py configs/train_lora_lens_base_24gb.yaml \
    --set "model.repo_id=${MODEL}" \
    --set "dataset.folder_path=${DATASET_PATH}" \
    --set "sample.trigger_word=${TRIGGER}" \
    --set "job.output_dir=${OUTPUT}"
fi
