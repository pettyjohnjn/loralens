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
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

WORKER_SCRIPT="${WORKER_SCRIPT:-${LORALENS_REPO_DIR}/jobs/launch/run_manifest_worker.sh}"
SWEEP_MANIFEST="${SWEEP_MANIFEST:?missing SWEEP_MANIFEST}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-999999}"

cd "${LORALENS_REPO_DIR}"

if [[ ! -f "${SWEEP_MANIFEST}" ]]; then
  echo "Manifest not found: ${SWEEP_MANIFEST}"
  exit 1
fi

if [[ ! -x "${WORKER_SCRIPT}" ]]; then
  echo "Worker script is not executable: ${WORKER_SCRIPT}"
  exit 1
fi

mapfile -t NODES < <(loralens_unique_pbs_nodes)
if [[ "${#NODES[@]}" -eq 0 ]]; then
  echo "No allocated nodes found."
  exit 1
fi

SLOTS="${#NODES[@]}"
if (( MAX_CONCURRENT_RUNS < SLOTS )); then
  SLOTS="${MAX_CONCURRENT_RUNS}"
fi

mapfile -t RUN_ENV_FILES < "${SWEEP_MANIFEST}"
if [[ "${#RUN_ENV_FILES[@]}" -eq 0 ]]; then
  echo "Manifest is empty."
  exit 1
fi

echo "Manifest: ${SWEEP_MANIFEST}"
echo "Allocated nodes (${#NODES[@]}): ${NODES[*]}"
echo "Using ${SLOTS} concurrent slots"

launch_slot() {
  local slot="$1"
  local slot_manifest="$2"
  local remote_cmd

  echo "[slot ${slot}] launching worker on ${NODES[$slot]} with $(wc -l < "${slot_manifest}") runs"
  if [[ "${NODES[$slot]}" == "${HOSTNAME}" ]]; then
    "${WORKER_SCRIPT}" "${slot_manifest}" &
  else
    printf -v remote_cmd 'cd %q && %q %q' \
      "${LORALENS_REPO_DIR}" \
      "${WORKER_SCRIPT}" \
      "${slot_manifest}"
    ssh "${NODES[$slot]}" "${remote_cmd}" &
  fi
  PIDS[$slot]="$!"
  ACTIVE_ENVS[$slot]="${slot_manifest}"
}

declare -a PIDS=()
declare -a ACTIVE_ENVS=()
declare -a SLOT_MANIFESTS=()
failed=0

for ((slot=0; slot<SLOTS; slot++)); do
  SLOT_MANIFESTS[$slot]="${SWEEP_MANIFEST}.slot${slot}"
  : > "${SLOT_MANIFESTS[$slot]}"
done

for idx in "${!RUN_ENV_FILES[@]}"; do
  slot=$(( idx % SLOTS ))
  printf '%s\n' "${RUN_ENV_FILES[$idx]}" >> "${SLOT_MANIFESTS[$slot]}"
done

for ((slot=0; slot<SLOTS; slot++)); do
  if [[ -s "${SLOT_MANIFESTS[$slot]}" ]]; then
    launch_slot "${slot}" "${SLOT_MANIFESTS[$slot]}"
  fi
done

for ((slot=0; slot<SLOTS; slot++)); do
  pid="${PIDS[$slot]:-}"
  if [[ -z "${pid}" ]]; then
    continue
  fi

  if wait "${pid}"; then
    echo "[slot ${slot}] completed $(basename "${ACTIVE_ENVS[$slot]}")"
  else
    echo "[slot ${slot}] failed $(basename "${ACTIVE_ENVS[$slot]}")"
    failed=1
  fi
done

exit "${failed}"
