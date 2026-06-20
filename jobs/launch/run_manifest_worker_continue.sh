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

log_gpu_state() {
  local label="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[worker $(hostname)] gpu_state ${label}"
    nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader || true
  fi
}

failed=0
mapfile -t RUN_ENV_FILES < "${SLOT_MANIFEST}"
for env_file in "${RUN_ENV_FILES[@]}"; do
  [[ -n "${env_file}" ]] || continue
  run_name="$(basename "${env_file}" .env)"
  log_gpu_state "before_${run_name}"
  echo "[worker $(hostname)] starting ${run_name}"
  if "${SCRIPT_DIR}/run_env_task.sh" "${env_file}"; then
    echo "[worker $(hostname)] completed ${run_name}"
  else
    rc="$?"
    echo "[worker $(hostname)] failed ${run_name} rc=${rc}"
    failed=1
  fi
  log_gpu_state "after_${run_name}"
  sleep "${RUN_CLEANUP_SLEEP_SEC:-10}"
done

exit "${failed}"
