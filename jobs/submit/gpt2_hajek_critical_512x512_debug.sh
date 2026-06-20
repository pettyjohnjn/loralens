#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${REPO_DIR}/jobs/launch/run_manifest_worker_continue.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-2:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_hajek_crit}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-2}"
DRY_RUN="${DRY_RUN:-0}"

BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2/gpt2_sweep_1000}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_sweep_1000/hajek_critical_512x512_debug}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_sweep_1000}"

MODEL_NAME="${MODEL_NAME:-gpt2}"
DATA_SOURCE="${DATA_SOURCE:-pile}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-1e-3}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-262144}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOKEN_SHIFT="${TOKEN_SHIFT:-0}"
SAVE_EVERY="${SAVE_EVERY:-125}"
SEED="${SEED:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
SUBSET_KL_TAIL_CLIP="${SUBSET_KL_TAIL_CLIP:-50.0}"
SUBSET_KL_TAIL_OVERSAMPLE="${SUBSET_KL_TAIL_OVERSAMPLE:-4}"

COMMON_RUN_KV=(
  "MODEL_NAME=${MODEL_NAME}"
  "DATA_SOURCE=${DATA_SOURCE}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "NUM_STEPS=${NUM_STEPS}"
  "WARMUP_STEPS=${WARMUP_STEPS}"
  "LR=${LR}"
  "TOKENS_PER_STEP=${TOKENS_PER_STEP}"
  "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
  "TOKEN_SHIFT=${TOKEN_SHIFT}"
  "SAVE_EVERY=${SAVE_EVERY}"
  "SEED=${SEED}"
  "LOG_SUBDIR=${LOG_SUBDIR}"
  "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
  "SUBSET_KL_TAIL_CLIP=${SUBSET_KL_TAIL_CLIP}"
  "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE}"
)

latest_checkpoint_path() {
  local output_dir="$1"
  find -L "${output_dir}" -maxdepth 1 -type f -name 'lens_step_*.pt' -printf '%T@\t%p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -f2-
}

checkpoint_step() {
  local checkpoint_path="$1"
  local stem
  stem="$(basename "${checkpoint_path}")"
  stem="${stem%.pt}"
  printf '%s\n' "${stem##*_}"
}

latest_metric_step() {
  local metrics_file="$1"
  if [[ ! -f "${metrics_file}" ]]; then
    printf '0\n'
    return 0
  fi
  awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ { step=$1 } END { print step + 0 }' "${metrics_file}"
}

add_or_skip_hajek_512x512() {
  local rank="$1"
  local run_name="lora_subset_hajek_r${rank}_k512_tail512"
  local output_dir="${BASE_OUT}/${run_name}"
  local current_checkpoint=""
  local checkpoint_step_value=0
  local metric_step_value=0
  local current_step=0
  local run_kv=("${COMMON_RUN_KV[@]}")

  mkdir -p "${output_dir}"
  current_checkpoint="$(latest_checkpoint_path "${output_dir}" || true)"
  if [[ -n "${current_checkpoint}" ]]; then
    checkpoint_step_value="$(checkpoint_step "${current_checkpoint}")"
  fi
  metric_step_value="$(latest_metric_step "${output_dir}/metrics.csv")"
  current_step="${metric_step_value}"
  if (( checkpoint_step_value > current_step )); then
    current_step="${checkpoint_step_value}"
  fi

  if (( current_step >= NUM_STEPS )); then
    echo "[skip] ${run_name} already reached step ${current_step}"
    return 0
  fi

  if [[ -n "${current_checkpoint}" ]]; then
    echo "[resume] ${run_name} from checkpoint step ${checkpoint_step_value}; metrics step ${metric_step_value}"
    run_kv+=("RESUME_CHECKPOINT=${current_checkpoint}")
  else
    echo "[new] ${run_name}"
  fi

  loralens_add_run \
    "${run_name}" \
    "${output_dir}" \
    "${run_kv[@]}" \
    "LENS_TYPE=lora" \
    "LOSS_TYPE=subset_kl" \
    "LORA_RANK=${rank}" \
    "SUBSET_KL_MODE=hajek" \
    "SUBSET_KL_K=512" \
    "SUBSET_KL_K_TAIL=512"
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
  echo "Max concurrent runs: ${MAX_CONCURRENT_RUNS}"
  echo "Output root: ${BASE_OUT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: qsub -q ${PBS_QUEUE} -A ${PBS_ACCOUNT} -N ${PBS_JOBNAME} -l select=${PBS_SELECT} -l walltime=${PBS_WALLTIME} -l filesystems=${PBS_FILESYSTEMS} -v SWEEP_MANIFEST=${manifest_file},MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS},WORKER_SCRIPT=${WORKER_SCRIPT},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR} ${PBS_SCRIPT}"
    return 0
  fi

  qsub \
    -q "${PBS_QUEUE}" \
    -A "${PBS_ACCOUNT}" \
    -N "${PBS_JOBNAME}" \
    -l "select=${PBS_SELECT}" \
    -l "walltime=${PBS_WALLTIME}" \
    -l "filesystems=${PBS_FILESYSTEMS}" \
    -v "SWEEP_MANIFEST=${manifest_file},MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS},WORKER_SCRIPT=${WORKER_SCRIPT},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR}" \
    "${PBS_SCRIPT}"
}

loralens_init_staging_dir "${STAGING_ROOT}"
mkdir -p "${BASE_OUT}"

add_or_skip_hajek_512x512 256
add_or_skip_hajek_512x512 384

TOTAL_RUNS="${#RUN_ENV_FILES[@]}"
if (( TOTAL_RUNS == 0 )); then
  echo "No critical Hájek 512+512 runs need to be submitted."
  exit 0
fi

MANIFEST_FILE="${STAGING_DIR}/manifest.gpt2_hajek_critical_512x512_debug.txt"
loralens_write_manifest "${MANIFEST_FILE}"

submit_manifest "${MANIFEST_FILE}" "${TOTAL_RUNS}"
