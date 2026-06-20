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

TASK_SCRIPT="${TASK_SCRIPT:-${LORALENS_REPO_DIR}/jobs/launch/run_env_task.sh}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
REQUIRED_NNODES="${REQUIRED_NNODES:-0}"
INPUT_RUN_ENV_FILE="${RUN_ENV_FILE:-}"

cd "${LORALENS_REPO_DIR}"

if [[ -n "${INPUT_RUN_ENV_FILE}" ]]; then
  if [[ ! -f "${INPUT_RUN_ENV_FILE}" ]]; then
    echo "Run env file not found: ${INPUT_RUN_ENV_FILE}"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${INPUT_RUN_ENV_FILE}"
fi

mapfile -t NODES < <(loralens_unique_pbs_nodes)
NNODES="${#NODES[@]}"

if (( REQUIRED_NNODES > 0 && NNODES != REQUIRED_NNODES )); then
  echo "Expected ${REQUIRED_NNODES} unique nodes, got ${NNODES}"
  exit 1
fi

MASTER_ADDR="${NODES[0]}"
LOCAL_HOST="$(hostname)"
LOCAL_HOST_SHORT="${LOCAL_HOST%%.*}"

if [[ -n "${INPUT_RUN_ENV_FILE}" ]]; then
  RUN_ENV_FILE="${INPUT_RUN_ENV_FILE}"
else
  RUN_ENV_ROOT="${LORALENS_REPO_DIR}/tmp/pbs_single_run_env"
  mkdir -p "${RUN_ENV_ROOT}"
  RUN_ENV_FILE="${RUN_ENV_ROOT}/$(date -u +%Y%m%dT%H%M%SZ).env"
  loralens_write_current_env_file "${RUN_ENV_FILE}"
fi

echo "[pbs] task_script=${TASK_SCRIPT}"
echo "[pbs] nnodes=${NNODES}"
echo "[pbs] nproc_per_node=${NPROC_PER_NODE}"
echo "[pbs] master_addr=${MASTER_ADDR}"
echo "[pbs] master_port=${MASTER_PORT}"
echo "[pbs] run_env_file=${RUN_ENV_FILE}"
printf '[pbs] nodes=%s\n' "${NODES[*]}"

PIDS=()
FAIL=0

for NODE_RANK in "${!NODES[@]}"; do
  NODE="${NODES[${NODE_RANK}]}"
  NODE_SHORT="${NODE%%.*}"
  printf -v REMOTE_CMD 'cd %q && export RUN_ENV_FILE=%q NNODES=%q NPROC_PER_NODE=%q NODE_RANK=%q MASTER_ADDR=%q MASTER_PORT=%q; %q %q' \
    "${LORALENS_REPO_DIR}" \
    "${RUN_ENV_FILE}" \
    "${NNODES}" \
    "${NPROC_PER_NODE}" \
    "${NODE_RANK}" \
    "${MASTER_ADDR}" \
    "${MASTER_PORT}" \
    "${TASK_SCRIPT}" \
    "${RUN_ENV_FILE}"

  echo "[pbs] launching node_rank=${NODE_RANK} on ${NODE}"
  if [[ "${NODE}" == "${LOCAL_HOST}" || "${NODE_SHORT}" == "${LOCAL_HOST_SHORT}" ]]; then
    bash -lc "${REMOTE_CMD}" &
  else
    ssh "${NODE}" "${REMOTE_CMD}" &
  fi
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

exit "${FAIL}"
