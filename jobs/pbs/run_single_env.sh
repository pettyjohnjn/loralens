#!/bin/bash
set -euo pipefail

resolve_repo_dir() {
  local candidate=""
  local parent=""

  for candidate in "${LORALENS_REPO_DIR:-}" "${REPO_DIR:-}" "${PBS_O_WORKDIR:-}"; do
    [[ -n "${candidate}" ]] || continue
    parent="${candidate}"
    while [[ -n "${parent}" ]]; do
      if [[ -f "${parent}/jobs/lib/common.sh" ]]; then
        printf '%s\n' "${parent}"
        return 0
      fi
      [[ "${parent}" != "/" ]] || break
      parent="$(dirname "${parent}")"
    done
  done

  return 1
}

REPO_DIR="${REPO_DIR:-$(resolve_repo_dir || true)}"
if [[ -z "${REPO_DIR}" ]]; then
  echo "Unable to locate repo root; set REPO_DIR or LORALENS_REPO_DIR before launching PBS."
  exit 1
fi
LORALENS_REPO_DIR="${LORALENS_REPO_DIR:-${REPO_DIR}}"
TASK_SCRIPT="${TASK_SCRIPT:-${REPO_DIR}/jobs/launch/run_env_task.sh}"
RUN_ENV_FILE="${RUN_ENV_FILE:?missing RUN_ENV_FILE}"

cd "${REPO_DIR}"

if [[ ! -f "${RUN_ENV_FILE}" ]]; then
  echo "Run env file not found: ${RUN_ENV_FILE}"
  exit 1
fi

echo "Using run env: ${RUN_ENV_FILE}"
exec "${TASK_SCRIPT}" "${RUN_ENV_FILE}"
