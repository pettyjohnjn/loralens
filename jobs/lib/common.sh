#!/bin/bash
set -euo pipefail

loralens_repo_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

LORALENS_REPO_DIR="${LORALENS_REPO_DIR:-$(loralens_repo_dir)}"
LORALENS_SRC_DIR="${LORALENS_SRC_DIR:-${LORALENS_REPO_DIR}/src}"

loralens_format_walltime() {
  local total_minutes="$1"
  printf '%02d:%02d:00' "$((total_minutes / 60))" "$((total_minutes % 60))"
}

loralens_resolve_lora_alpha() {
  local rank="${1:-}"
  local alpha="${2:-${LORA_ALPHA-}}"

  if [[ -n "${alpha}" ]]; then
    printf '%s\n' "${alpha}"
    return 0
  fi

  if [[ -n "${rank}" ]]; then
    printf '%s\n' "${rank}"
    return 0
  fi

  return 1
}

loralens_load_modules() {
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile ]]; then
      # PBS-launched non-login shells often need site init first.
      source /etc/profile >/dev/null 2>&1 || true
    fi
  fi

  if ! command -v module >/dev/null 2>&1; then
    local module_init
    for module_init in \
      /etc/profile.d/modules.sh \
      /usr/share/lmod/lmod/init/bash \
      /opt/cray/pe/lmod/lmod/init/bash
    do
      if [[ -f "${module_init}" ]]; then
        # shellcheck disable=SC1090
        source "${module_init}"
        break
      fi
    done
  fi

  if ! command -v module >/dev/null 2>&1; then
    echo "module command unavailable on $(hostname)"
    return 127
  fi

  module use /soft/modulefiles
  module load conda
}

loralens_bootstrap_runtime() {
  local conda_env="${1:-${CONDA_ENV_NAME:-LoRA_Lens}}"

  loralens_load_modules
  conda activate "${conda_env}"

  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

  if [[ -f "${HOME}/.hf_env" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.hf_env"
  fi
}

loralens_install_runtime_deps() {
  if [[ "${INSTALL_RUNTIME_DEPS:-0}" != "1" ]]; then
    return 0
  fi

  local hookbox_source="${HOOKBOX_INSTALL_SOURCE:-git+https://github.com/pettyjohnjn/hookbox.git}"
  local subset_source="${SUBSET_KL_INSTALL_SOURCE:-git+https://github.com/pettyjohnjn/subset-kl.git}"
  local indexed_source="${INDEXED_LOGITS_INSTALL_SOURCE:-git+https://github.com/pettyjohnjn/indexed_logits.git}"

  pip install accelerate --quiet 2>/dev/null || true
  pip install "${hookbox_source}" --quiet 2>/dev/null || true
  pip install "${subset_source}" --quiet 2>/dev/null || true
  pip install "${indexed_source}" --quiet 2>/dev/null || true
}

loralens_python_probe() {
  python - <<'PY'
import os
import sys
import torch

print("[setup] sys.executable =", sys.executable)
print("[setup] cwd =", os.getcwd())
print("[setup] torch.cuda.is_available =", torch.cuda.is_available())
print("[setup] torch.cuda.device_count =", torch.cuda.device_count())

for module_name in ("subset_kl", "loralens"):
    try:
        module = __import__(module_name)
        print(f"[setup] {module_name} =", module.__file__)
    except Exception as exc:
        print(f"[setup] {module_name} import failed =", exc)
PY
}

loralens_print_gpu_probe() {
  echo "[setup] host=$(hostname)"
  echo "[setup] python=$(which python)"
  echo "[setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
}

loralens_init_staging_dir() {
  local staging_root="$1"
  mkdir -p "${staging_root}"
  STAGING_DIR="${staging_root}/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${STAGING_DIR}"
  RUN_ENV_FILES=()
}

loralens_write_env_file() {
  local env_file="$1"
  shift
  local key
  local value

  {
    while (($#)); do
      key="${1%%=*}"
      value="${1#*=}"
      printf '%s=%q\n' "${key}" "${value}"
      shift
    done
  } > "${env_file}"
}

loralens_write_current_env_file() {
  local env_file="$1"
  local name

  {
    while IFS= read -r name; do
      printf '%s=%q\n' "${name}" "${!name}"
    done < <(loralens_exported_env_names)
  } > "${env_file}"
}

loralens_add_run() {
  local run_name="$1"
  local output_dir="$2"
  shift 2

  local env_file="${STAGING_DIR}/${run_name}.env"
  loralens_write_env_file \
    "${env_file}" \
    "RUN_NAME=${run_name}" \
    "OUTPUT_DIR=${output_dir}" \
    "$@"
  RUN_ENV_FILES+=("${env_file}")
}

loralens_write_manifest() {
  local manifest_file="$1"
  printf '%s\n' "${RUN_ENV_FILES[@]}" > "${manifest_file}"
}

loralens_unique_pbs_nodes() {
  if [[ -z "${PBS_NODEFILE:-}" ]]; then
    printf '%s\n' "${HOSTNAME}"
    return 0
  fi

  awk '!seen[$0]++' "${PBS_NODEFILE}"
}

loralens_exported_env_names() {
  compgen -e | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' | sort
}
