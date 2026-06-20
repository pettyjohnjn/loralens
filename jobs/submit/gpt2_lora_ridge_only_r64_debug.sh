#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-1:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-00:30:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_ridge_only_r64}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-1}"
DRY_RUN="${DRY_RUN:-0}"

BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_lora_init_fullkl_r64_debug}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_lora_ridge_only_r64_debug}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_lora_ridge_only_r64_debug}"

MODEL_NAME="${MODEL_NAME:-gpt2}"
DATA_SOURCE="${DATA_SOURCE:-pile}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOKEN_SHIFT="${TOKEN_SHIFT:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
SEED="${SEED:-0}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
LORA_INIT_CALIBRATION_TOKENS="${LORA_INIT_CALIBRATION_TOKENS:-50000}"
LORA_INIT_RIDGE_LAMBDA="${LORA_INIT_RIDGE_LAMBDA:-1e-3}"
LORA_INIT_RIDGE_LAMBDA_SCALE="${LORA_INIT_RIDGE_LAMBDA_SCALE:-trace_xxt_over_d}"
LORA_INIT_STATS_DTYPE="${LORA_INIT_STATS_DTYPE:-float32}"
LORA_INIT_JITTER="${LORA_INIT_JITTER:-1e-6}"

RUN_NAME="lora_fullkl_r${LORA_RANK}_ridge_svd_step0"
OUTPUT_DIR="${BASE_OUT}/${RUN_NAME}"

submit_manifest() {
  local manifest_file="$1"

  echo "Submitting ${PBS_JOBNAME}"
  echo "Manifest: ${manifest_file}"
  echo "Queue: ${PBS_QUEUE}"
  echo "Select: ${PBS_SELECT}"
  echo "Walltime: ${PBS_WALLTIME}"

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
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/lens_step_0.pt" ]]; then
  echo "[skip] ${RUN_NAME} already has ${OUTPUT_DIR}/lens_step_0.pt"
  exit 0
fi

echo "Preparing GPT-2 rank-${LORA_RANK} ridge-only LoRA lens"
echo "Output dir: ${OUTPUT_DIR}"
echo "Calibration tokens: ${LORA_INIT_CALIBRATION_TOKENS}"

loralens_add_run \
  "${RUN_NAME}" \
  "${OUTPUT_DIR}" \
  "MODEL_NAME=${MODEL_NAME}" \
  "DATA_SOURCE=${DATA_SOURCE}" \
  "BATCH_SIZE=${BATCH_SIZE}" \
  "NUM_STEPS=0" \
  "WARMUP_STEPS=0" \
  "LR=1.5e-3" \
  "TOKENS_PER_STEP=262144" \
  "MAX_SEQ_LEN=${MAX_SEQ_LEN}" \
  "TOKEN_SHIFT=${TOKEN_SHIFT}" \
  "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}" \
  "SAVE_EVERY=1" \
  "SAVE_INITIAL_CHECKPOINT=1" \
  "LOG_EVERY=1" \
  "SEED=${SEED}" \
  "LOG_SUBDIR=${LOG_SUBDIR}" \
  "LENS_TYPE=lora" \
  "LOSS_TYPE=kl" \
  "ACTIVATION_SITE_PRESET=residual" \
  "LORA_RANK=${LORA_RANK}" \
  "LORA_ALPHA=${LORA_ALPHA}" \
  "LORA_INIT=ridge_svd" \
  "LORA_INIT_CALIBRATION_TOKENS=${LORA_INIT_CALIBRATION_TOKENS}" \
  "LORA_INIT_RIDGE_LAMBDA=${LORA_INIT_RIDGE_LAMBDA}" \
  "LORA_INIT_RIDGE_LAMBDA_SCALE=${LORA_INIT_RIDGE_LAMBDA_SCALE}" \
  "LORA_INIT_STATS_DTYPE=${LORA_INIT_STATS_DTYPE}" \
  "LORA_INIT_JITTER=${LORA_INIT_JITTER}"

MANIFEST_FILE="${STAGING_DIR}/manifest.txt"
loralens_write_manifest "${MANIFEST_FILE}"
submit_manifest "${MANIFEST_FILE}"
