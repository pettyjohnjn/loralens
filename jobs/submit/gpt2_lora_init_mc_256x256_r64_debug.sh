#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug-scaling}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-2:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_lora_init_mc256}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-2}"
DRY_RUN="${DRY_RUN:-0}"

BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_lora_init_mc_256x256_r64_debug}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_lora_init_mc_256x256_r64_debug}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_lora_init_mc_256x256_r64_debug}"

MODEL_NAME="${MODEL_NAME:-gpt2}"
DATA_SOURCE="${DATA_SOURCE:-pile}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-250}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-1.5e-3}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-262144}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOKEN_SHIFT="${TOKEN_SHIFT:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
SAVE_EVERY="${SAVE_EVERY:-50}"
LOG_EVERY="${LOG_EVERY:-1}"
SEED="${SEED:-0}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
HEAD_K="${HEAD_K:-256}"
TAIL_K="${TAIL_K:-256}"
SUBSET_KL_TAIL_CLIP="${SUBSET_KL_TAIL_CLIP:-50.0}"
SUBSET_KL_TAIL_OVERSAMPLE="${SUBSET_KL_TAIL_OVERSAMPLE:-4}"

LORA_INIT_CALIBRATION_TOKENS="${LORA_INIT_CALIBRATION_TOKENS:-50000}"
LORA_INIT_RIDGE_LAMBDA="${LORA_INIT_RIDGE_LAMBDA:-1e-3}"
LORA_INIT_RIDGE_LAMBDA_SCALE="${LORA_INIT_RIDGE_LAMBDA_SCALE:-trace_xxt_over_d}"
LORA_INIT_STATS_DTYPE="${LORA_INIT_STATS_DTYPE:-float32}"
LORA_INIT_JITTER="${LORA_INIT_JITTER:-1e-6}"
LORA_INIT_SVD_METRIC="${LORA_INIT_SVD_METRIC:-residual}"
LORA_INIT_NORMALIZATION="${LORA_INIT_NORMALIZATION:-none}"

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
  "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
  "SAVE_EVERY=${SAVE_EVERY}"
  "LOG_EVERY=${LOG_EVERY}"
  "SEED=${SEED}"
  "LOG_SUBDIR=${LOG_SUBDIR}"
  "LENS_TYPE=lora"
  "LOSS_TYPE=subset_kl"
  "ACTIVATION_SITE_PRESET=residual"
  "LORA_RANK=${LORA_RANK}"
  "LORA_ALPHA=${LORA_ALPHA}"
  "SUBSET_KL_MODE=mc"
  "SUBSET_KL_K=${HEAD_K}"
  "SUBSET_KL_K_TAIL=${TAIL_K}"
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

add_or_skip_run() {
  local run_name="$1"
  local output_dir="$2"
  shift 2

  local current_checkpoint=""
  local current_step=0
  local run_kv=("$@")

  mkdir -p "${output_dir}"
  current_checkpoint="$(latest_checkpoint_path "${output_dir}" || true)"
  if [[ -n "${current_checkpoint}" ]]; then
    current_step="$(checkpoint_step "${current_checkpoint}")"
  fi

  if (( current_step >= NUM_STEPS )); then
    echo "[skip] ${run_name} already reached step ${current_step}"
    return 0
  fi

  if [[ -n "${current_checkpoint}" ]]; then
    echo "[resume] ${run_name} from step ${current_step}"
    run_kv+=("RESUME_CHECKPOINT=${current_checkpoint}")
  else
    echo "[new] ${run_name}"
  fi

  loralens_add_run "${run_name}" "${output_dir}" "${run_kv[@]}"
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

echo "Preparing GPT-2 LoRA r=${LORA_RANK} MC 256+256 init comparison"
echo "Output root: ${BASE_OUT}"
echo "Estimator: subset_kl mc, head=${HEAD_K}, tail=${TAIL_K}"
echo "Steps: ${NUM_STEPS}; save_every=${SAVE_EVERY}; lr=${LR}"

add_or_skip_run \
  "lora_mc256x256_r${LORA_RANK}_default_lora" \
  "${BASE_OUT}/lora_mc256x256_r${LORA_RANK}_default_lora" \
  "${COMMON_RUN_KV[@]}" \
  "LORA_INIT=default_lora"

add_or_skip_run \
  "lora_mc256x256_r${LORA_RANK}_ridge_svd" \
  "${BASE_OUT}/lora_mc256x256_r${LORA_RANK}_ridge_svd" \
  "${COMMON_RUN_KV[@]}" \
  "LORA_INIT=ridge_svd" \
  "LORA_INIT_CALIBRATION_TOKENS=${LORA_INIT_CALIBRATION_TOKENS}" \
  "LORA_INIT_RIDGE_LAMBDA=${LORA_INIT_RIDGE_LAMBDA}" \
  "LORA_INIT_RIDGE_LAMBDA_SCALE=${LORA_INIT_RIDGE_LAMBDA_SCALE}" \
  "LORA_INIT_STATS_DTYPE=${LORA_INIT_STATS_DTYPE}" \
  "LORA_INIT_JITTER=${LORA_INIT_JITTER}" \
  "LORA_INIT_SVD_METRIC=${LORA_INIT_SVD_METRIC}" \
  "LORA_INIT_NORMALIZATION=${LORA_INIT_NORMALIZATION}"

MANIFEST_FILE="${STAGING_DIR}/manifest.txt"
loralens_write_manifest "${MANIFEST_FILE}"
RUN_COUNT="${#RUN_ENV_FILES[@]}"

if (( RUN_COUNT == 0 )); then
  echo "No runs to submit."
  exit 0
fi

submit_manifest "${MANIFEST_FILE}" "${RUN_COUNT}"
