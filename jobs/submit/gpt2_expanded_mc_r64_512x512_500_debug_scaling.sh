#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_all_nodes_single_run.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug-scaling}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-10:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_expand_mc_r64}"
DRY_RUN="${DRY_RUN:-0}"

RUN_NAME="${RUN_NAME:-lora_subset_mc_gpt2_expanded_r64_k512_tail512_500}"
BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_expanded_mc_debug_500}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_expanded_mc_debug_500}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_expanded_mc_debug_500}"
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-/grand/SuperBERT/pettyjohnjn/cache}"

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
  "SAVE_EVERY=${SAVE_EVERY:-50}" \
  "SEED=${SEED:-0}" \
  "LOG_SUBDIR=${LOG_SUBDIR}" \
  "RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}" \
  "HF_HOME=${HF_HOME:-${HF_HOME_DEFAULT}}" \
  "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME_DEFAULT}/datasets}" \
  "HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME_DEFAULT}/hub}" \
  "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}" \
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-}" \
  "ACTIVATION_SITE_PRESET=gpt2_expanded" \
  "LENS_TYPE=lora" \
  "LOSS_TYPE=subset_kl" \
  "LORA_RANK=${LORA_RANK:-64}" \
  "SUBSET_KL_MODE=mc" \
  "SUBSET_KL_K=${SUBSET_KL_K:-512}" \
  "SUBSET_KL_K_TAIL=${SUBSET_KL_K_TAIL:-512}" \
  "SUBSET_KL_TAIL_CLIP=${SUBSET_KL_TAIL_CLIP:-50.0}" \
  "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE:-4}" \
  "KL_CHUNK_SIZE=${KL_CHUNK_SIZE:-128}"

RUN_ENV_FILE="${RUN_ENV_FILES[0]}"

echo "Submitting ${PBS_JOBNAME}"
echo "Run env: ${RUN_ENV_FILE}"
echo "Output root: ${BASE_OUT}"
echo "Queue: ${PBS_QUEUE}"
echo "Select: ${PBS_SELECT}"
echo "Walltime: ${PBS_WALLTIME}"
echo "PBS script: ${PBS_SCRIPT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: qsub -q ${PBS_QUEUE} -A ${PBS_ACCOUNT} -N ${PBS_JOBNAME} -l select=${PBS_SELECT} -l walltime=${PBS_WALLTIME} -l filesystems=${PBS_FILESYSTEMS} -v RUN_ENV_FILE=${RUN_ENV_FILE},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR} ${PBS_SCRIPT}"
  exit 0
fi

qsub \
  -q "${PBS_QUEUE}" \
  -A "${PBS_ACCOUNT}" \
  -N "${PBS_JOBNAME}" \
  -l "select=${PBS_SELECT}" \
  -l "walltime=${PBS_WALLTIME}" \
  -l "filesystems=${PBS_FILESYSTEMS}" \
  -v "RUN_ENV_FILE=${RUN_ENV_FILE},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR}" \
  "${PBS_SCRIPT}"
