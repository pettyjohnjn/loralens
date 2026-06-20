#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

RUN_NAME="${RUN_NAME:?missing RUN_NAME}"
RUN_LOG="${RUN_LOG:?missing RUN_LOG}"
TRAIN_COMMAND="${TRAIN_COMMAND:?missing TRAIN_COMMAND}"

loralens_bootstrap_runtime
loralens_install_runtime_deps

cd "${LORALENS_SRC_DIR}"
mkdir -p "$(dirname "${RUN_LOG}")"
exec > >(tee "${RUN_LOG}") 2>&1

echo "[run] name=${RUN_NAME}"
loralens_print_gpu_probe
loralens_python_probe

eval "${TRAIN_COMMAND}"
