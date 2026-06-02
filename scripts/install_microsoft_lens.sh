#!/usr/bin/env bash
# Clone/update microsoft/Lens into vendor/Lens (not pip-installable — no pyproject.toml).
# Usage:  source scripts/install_microsoft_lens.sh && install_microsoft_lens /path/to/LoboForge-LensTrainer
set -euo pipefail

LENS_REPO="${LENS_REPO:-https://github.com/microsoft/Lens.git}"
LENS_GIT_REF="${LENS_GIT_REF:-main}"

install_microsoft_lens() {
  local root="${1:?install root required}"
  local lens_dir="${root}/vendor/Lens"

  if [[ -d "${lens_dir}/lens" && -f "${lens_dir}/inference.py" ]]; then
    if [[ -d "${lens_dir}/.git" ]]; then
      printf '==> Updating microsoft/Lens in vendor/Lens\n'
      git -C "${lens_dir}" fetch origin "${LENS_GIT_REF}" --depth 1 2>/dev/null || true
      git -C "${lens_dir}" checkout "${LENS_GIT_REF}" 2>/dev/null || git -C "${lens_dir}" pull --ff-only || true
    fi
    return 0
  fi

  printf '==> Cloning microsoft/Lens → %s\n' "${lens_dir}"
  mkdir -p "$(dirname "${lens_dir}")"
  if [[ -d "${lens_dir}/.git" ]]; then
    rm -rf "${lens_dir}"
  fi
  git clone --depth 1 --branch "${LENS_GIT_REF}" "${LENS_REPO}" "${lens_dir}" \
    || git clone --depth 1 "${LENS_REPO}" "${lens_dir}"

  if [[ ! -d "${lens_dir}/lens" ]]; then
    printf '==> [error] Lens clone missing lens/ package under %s\n' "${lens_dir}" >&2
    return 1
  fi
}

lens_pythonpath() {
  local root="${1:?}"
  printf '%s/vendor/Lens' "${root}"
}
