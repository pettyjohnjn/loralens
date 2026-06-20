#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

SLOT_MANIFEST="${1:-}"
SLOT_MANIFEST="${SLOT_MANIFEST:?missing SLOT_MANIFEST}"

if [[ ! -f "${SLOT_MANIFEST}" ]]; then
  echo "Slot manifest not found: ${SLOT_MANIFEST}"
  exit 1
fi

if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
  loralens_bootstrap_runtime
  loralens_install_runtime_deps
fi

export SKIP_ENV_SETUP=1

mapfile -t RUN_ENV_FILES < "${SLOT_MANIFEST}"
for env_file in "${RUN_ENV_FILES[@]}"; do
  [[ -n "${env_file}" ]] || continue
  echo "[worker $(hostname)] starting $(basename "${env_file}" .env)"
  "${SCRIPT_DIR}/run_env_task.sh" "${env_file}"
  echo "[worker $(hostname)] completed $(basename "${env_file}" .env)"
done
