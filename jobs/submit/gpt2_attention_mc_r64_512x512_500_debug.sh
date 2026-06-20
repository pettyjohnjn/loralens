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
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_attn_mc_r64}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-1}"
DRY_RUN="${DRY_RUN:-0}"

RUN_NAME="${RUN_NAME:-lora_subset_mc_attn_r64_k512_tail512_500}"
BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_attention_mc_debug_500}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_attention_mc_debug_500}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_attention_mc_debug_500}"

loralens_init_staging_dir "${STAGING_ROOT}"
mkdir -p "${BASE_OUT}/${RUN_NAME}"

loralens_add_run \
  "${RUN_NAME}" \
  "${BASE_OUT}/${RUN_NAME}" \
  "MODEL_NAME=${MODEL_NAME:-gpt2}" \
  "DATA_SOURCE=${DATA_SOURCE:-pile}" \
  "BATCH_SIZE=${BATCH_SIZE:-32}" \
  "NUM_STEPS=${NUM_STEPS:-500}" \
  "WARMUP_STEPS=${WARMUP_STEPS:-0}" \
  "LR=${LR:-1e-3}" \
  "TOKENS_PER_STEP=${TOKENS_PER_STEP:-262144}" \
  "MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}" \
  "TOKEN_SHIFT=${TOKEN_SHIFT:-0}" \
  "SAVE_EVERY=${SAVE_EVERY:-250}" \
  "SEED=${SEED:-0}" \
  "LOG_SUBDIR=${LOG_SUBDIR}" \
  "ACTIVATION_SITE_PRESET=gpt2_attention" \
  "LENS_TYPE=lora" \
  "LOSS_TYPE=subset_kl" \
  "LORA_RANK=${LORA_RANK:-64}" \
  "SUBSET_KL_MODE=mc" \
  "SUBSET_KL_K=${SUBSET_KL_K:-512}" \
  "SUBSET_KL_K_TAIL=${SUBSET_KL_K_TAIL:-512}" \
  "SUBSET_KL_TAIL_CLIP=${SUBSET_KL_TAIL_CLIP:-50.0}" \
  "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE:-4}" \
  "KL_CHUNK_SIZE=${KL_CHUNK_SIZE:-128}"

MANIFEST_FILE="${STAGING_DIR}/manifest.gpt2_attention_mc_debug.txt"
loralens_write_manifest "${MANIFEST_FILE}"

echo "Submitting ${PBS_JOBNAME}"
echo "Manifest: ${MANIFEST_FILE}"
echo "Run env: ${RUN_ENV_FILES[0]}"
echo "Output root: ${BASE_OUT}"
echo "Queue: ${PBS_QUEUE}"
echo "Select: ${PBS_SELECT}"
echo "Walltime: ${PBS_WALLTIME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: qsub -q ${PBS_QUEUE} -A ${PBS_ACCOUNT} -N ${PBS_JOBNAME} -l select=${PBS_SELECT} -l walltime=${PBS_WALLTIME} -l filesystems=${PBS_FILESYSTEMS} -v SWEEP_MANIFEST=${MANIFEST_FILE},MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR} ${PBS_SCRIPT}"
  exit 0
fi

qsub \
  -q "${PBS_QUEUE}" \
  -A "${PBS_ACCOUNT}" \
  -N "${PBS_JOBNAME}" \
  -l "select=${PBS_SELECT}" \
  -l "walltime=${PBS_WALLTIME}" \
  -l "filesystems=${PBS_FILESYSTEMS}" \
  -v "SWEEP_MANIFEST=${MANIFEST_FILE},MAX_CONCURRENT_RUNS=${MAX_CONCURRENT_RUNS},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR}" \
  "${PBS_SCRIPT}"
