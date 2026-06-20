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
SWEEP_MANIFEST="${SWEEP_MANIFEST:?missing SWEEP_MANIFEST}"
RUNS_PER_GROUP="${RUNS_PER_GROUP:-1}"
NODES_PER_RUN="${NODES_PER_RUN:-2}"

cd "${LORALENS_REPO_DIR}"

if [[ ! -f "${SWEEP_MANIFEST}" ]]; then
  echo "Manifest not found: ${SWEEP_MANIFEST}"
  exit 1
fi

mapfile -t NODES < <(loralens_unique_pbs_nodes)
mapfile -t RUN_ENV_FILES < "${SWEEP_MANIFEST}"

required_nodes=$(( ${#RUN_ENV_FILES[@]} * NODES_PER_RUN ))
if [[ "${#NODES[@]}" -lt "${required_nodes}" ]]; then
  echo "Expected at least ${required_nodes} allocated nodes, found ${#NODES[@]}"
  printf 'Nodes: %s\n' "${NODES[@]}"
  exit 1
fi

launch_rank() {
  local node="$1"
  local env_file="$2"
  local node_rank="$3"
  local master_addr="$4"
  local master_port="$5"
  local remote_cmd

  printf -v remote_cmd 'cd %q && export RUN_ENV_FILE=%q NNODES=%q NODE_RANK=%q MASTER_ADDR=%q MASTER_PORT=%q LOG_SUFFIX=%q; %q %q' \
    "${LORALENS_REPO_DIR}" \
    "${env_file}" \
    "${NODES_PER_RUN}" \
    "${node_rank}" \
    "${master_addr}" \
    "${master_port}" \
    "node${node_rank}" \
    "${TASK_SCRIPT}" \
    "${env_file}"

  if [[ "${node}" == "${HOSTNAME}" ]]; then
    bash -lc "${remote_cmd}" &
  else
    ssh "${node}" "${remote_cmd}" &
  fi
  echo "$!"
}

declare -a RUN_PIDS=()
declare -a RUN_NAMES=()
failed=0

for idx in "${!RUN_ENV_FILES[@]}"; do
  env_file="${RUN_ENV_FILES[$idx]}"
  start_node=$(( idx * NODES_PER_RUN ))
  master_node="${NODES[$start_node]}"
  worker_node="${NODES[$((start_node + 1))]}"
  master_port="$((29500 + idx))"
  run_name="$(basename "${env_file}" .env)"

  echo "[run ${idx}] ${run_name}"
  echo "[run ${idx}] nodes: ${master_node} ${worker_node}"
  echo "[run ${idx}] master: ${master_node}:${master_port}"

  pid0="$(launch_rank "${master_node}" "${env_file}" 0 "${master_node}" "${master_port}")"
  pid1="$(launch_rank "${worker_node}" "${env_file}" 1 "${master_node}" "${master_port}")"
  RUN_PIDS+=("${pid0} ${pid1}")
  RUN_NAMES+=("${run_name}")
done

for idx in "${!RUN_PIDS[@]}"; do
  IFS=' ' read -r pid0 pid1 <<< "${RUN_PIDS[$idx]}"
  run_failed=0
  if ! wait "${pid0}"; then
    run_failed=1
  fi
  if ! wait "${pid1}"; then
    run_failed=1
  fi

  if (( run_failed )); then
    echo "[${RUN_NAMES[$idx]}] failed"
    failed=1
  else
    echo "[${RUN_NAMES[$idx]}] completed successfully"
  fi
done

exit "${failed}"
