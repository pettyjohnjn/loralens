#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug-scaling}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-1:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-llama3_70b_feas_proxy}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Use the 8B model as a single-node scaling proxy. Tiny values keep this
# profiling job focused on memory terms rather than useful training.
MODEL_NAME="${MODEL_NAME:-/grand/SuperBERT/aswathy/models/models--meta-llama--Meta-Llama-3-8B-Instruct}"
BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/llama3_70b_feasibility_proxy}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/llama3_70b_feasibility_proxy}"
LOG_SUBDIR="${LOG_SUBDIR:-llama3_70b_feasibility_proxy}"

DATA_SOURCE="${DATA_SOURCE:-pile}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-3}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-1e-3}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
SAVE_EVERY="${SAVE_EVERY:-${NUM_STEPS}}"
LOG_EVERY="${LOG_EVERY:-1}"
SEED="${SEED:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-64}"
SUBSET_KL_K="${SUBSET_KL_K:-256}"
SUBSET_KL_MODE="${SUBSET_KL_MODE:-topk}"
SUBSET_KL_K_TAIL="${SUBSET_KL_K_TAIL:-0}"
ACTIVATION_SITE_PRESET="${ACTIVATION_SITE_PRESET:-residual}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-0}"

COMMON_RUN_KV=(
  "MODEL_NAME=${MODEL_NAME}"
  "DATA_SOURCE=${DATA_SOURCE}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "NUM_STEPS=${NUM_STEPS}"
  "WARMUP_STEPS=${WARMUP_STEPS}"
  "LR=${LR}"
  "TOKENS_PER_STEP=${TOKENS_PER_STEP}"
  "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
  "SAVE_EVERY=${SAVE_EVERY}"
  "LOG_EVERY=${LOG_EVERY}"
  "SEED=${SEED}"
  "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
  "ACTIVATION_SITE_PRESET=${ACTIVATION_SITE_PRESET}"
  "MODEL_DTYPE=${MODEL_DTYPE}"
  "AMP_DTYPE=${AMP_DTYPE}"
  "INSTALL_RUNTIME_DEPS=${INSTALL_RUNTIME_DEPS}"
  "LOG_SUBDIR=${LOG_SUBDIR}"
)

add_run() {
  local run_name="$1"
  local output_dir="${BASE_OUT}/${run_name}"
  shift
  mkdir -p "${output_dir}"
  loralens_add_run "${run_name}" "${output_dir}" "${COMMON_RUN_KV[@]}" "$@"
}

submit_manifest() {
  local manifest_file="$1"
  local run_count="$2"

  echo "Submitting ${PBS_JOBNAME}"
  echo "Manifest: ${manifest_file}"
  echo "Runs: ${run_count}"
  echo "Queue: ${PBS_QUEUE}"
  echo "Select: ${PBS_SELECT}"
  echo "Walltime: ${PBS_WALLTIME}"
  echo "Output root: ${BASE_OUT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: qsub -q ${PBS_QUEUE} -A ${PBS_ACCOUNT} -N ${PBS_JOBNAME} -l select=${PBS_SELECT} -l walltime=${PBS_WALLTIME} -l filesystems=${PBS_FILESYSTEMS} -v SWEEP_MANIFEST=${manifest_file},MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR} ${PBS_SCRIPT}"
    return 0
  fi

  qsub \
    -q "${PBS_QUEUE}" \
    -A "${PBS_ACCOUNT}" \
    -N "${PBS_JOBNAME}" \
    -l "select=${PBS_SELECT}" \
    -l "walltime=${PBS_WALLTIME}" \
    -l "filesystems=${PBS_FILESYSTEMS}" \
    -v "SWEEP_MANIFEST=${manifest_file},MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR}" \
    "${PBS_SCRIPT}"
}

loralens_init_staging_dir "${STAGING_ROOT}"
mkdir -p "${BASE_OUT}"

add_run "fullrank_fullkl" \
  "LENS_TYPE=tuned" \
  "LOSS_TYPE=kl"

add_run "lora_r64_fullkl" \
  "LENS_TYPE=lora" \
  "LOSS_TYPE=kl" \
  "LORA_RANK=64"

add_run "fullrank_subsetkl_k256" \
  "LENS_TYPE=tuned" \
  "LOSS_TYPE=subset_kl" \
  "SUBSET_KL_MODE=${SUBSET_KL_MODE}" \
  "SUBSET_KL_K=${SUBSET_KL_K}" \
  "SUBSET_KL_K_TAIL=${SUBSET_KL_K_TAIL}"

add_run "lora_r64_subsetkl_k256" \
  "LENS_TYPE=lora" \
  "LOSS_TYPE=subset_kl" \
  "LORA_RANK=64" \
  "SUBSET_KL_MODE=${SUBSET_KL_MODE}" \
  "SUBSET_KL_K=${SUBSET_KL_K}" \
  "SUBSET_KL_K_TAIL=${SUBSET_KL_K_TAIL}"

MANIFEST_FILE="${STAGING_DIR}/manifest.llama3_70b_feasibility_proxy.txt"
loralens_write_manifest "${MANIFEST_FILE}"

submit_manifest "${MANIFEST_FILE}" "${#RUN_ENV_FILES[@]}"
