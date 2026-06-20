#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
PBS_QUEUE="${PBS_QUEUE:-prod}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-24:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-03:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_mc_combined_prod}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-24}"
DRY_RUN="${DRY_RUN:-0}"

LOWRANK_BASE_OUT="${LOWRANK_BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_preemptable_lowrank_mc_sweep_1000}"
STANDARD_BASE_OUT="${STANDARD_BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_preemptable_mc_sweep_1000}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_combined_mc_sweep_1000}"

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

LOWRANK_RANKS_CSV="${LOWRANK_RANKS_CSV:-1,4,8,16,32}"
STANDARD_RANKS_CSV="${STANDARD_RANKS_CSV:-64,128,256,384}"
IFS=',' read -r -a LOWRANK_RANKS <<< "${LOWRANK_RANKS_CSV}"
IFS=',' read -r -a STANDARD_RANKS <<< "${STANDARD_RANKS_CSV}"
HEAD_TAIL_PAIRS=(
  "256 256"
  "512 512"
  "768 768"
  "1024 1024"
  "512 1024"
  "1024 512"
)

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
  local log_subdir="$3"
  shift 3

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

  loralens_add_run \
    "${run_name}" \
    "${output_dir}" \
    "${COMMON_RUN_KV[@]}" \
    "LOG_SUBDIR=${log_subdir}" \
    "${run_kv[@]}"
}

add_rank_family() {
  local base_out="$1"
  local log_subdir="$2"
  shift 2
  local ranks=("$@")
  local rank
  local spec
  local k
  local tail

  for rank in "${ranks[@]}"; do
    for spec in "${HEAD_TAIL_PAIRS[@]}"; do
      read -r k tail <<< "${spec}"
      add_or_skip_run \
        "lora_subset_mc_r${rank}_k${k}_tail${tail}" \
        "${base_out}/lora_subset_mc_r${rank}_k${k}_tail${tail}" \
        "${log_subdir}" \
        "LENS_TYPE=lora" \
        "LOSS_TYPE=subset_kl" \
        "LORA_RANK=${rank}" \
        "SUBSET_KL_MODE=mc" \
        "SUBSET_KL_K=${k}" \
        "SUBSET_KL_K_TAIL=${tail}" \
        "SUBSET_KL_TAIL_CLIP=${SUBSET_KL_TAIL_CLIP}" \
        "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE}" \
        "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
    done
  done
}

loralens_init_staging_dir "${STAGING_ROOT}"
mkdir -p "${LOWRANK_BASE_OUT}" "${STANDARD_BASE_OUT}"

echo "Preparing combined GPT-2 MC head/tail continuation"
echo "Low-rank root: ${LOWRANK_BASE_OUT}"
echo "Standard root: ${STANDARD_BASE_OUT}"
echo "Staging dir: ${STAGING_DIR}"

add_rank_family "${LOWRANK_BASE_OUT}" "gpt2_preemptable_lowrank_mc_sweep_1000" "${LOWRANK_RANKS[@]}"
add_rank_family "${STANDARD_BASE_OUT}" "gpt2_preemptable_mc_sweep_1000" "${STANDARD_RANKS[@]}"

TOTAL_RUNS="${#RUN_ENV_FILES[@]}"
if (( TOTAL_RUNS == 0 )); then
  echo "No runnable combined MC sweep runs were generated."
  exit 0
fi

MANIFEST_FILE="${STAGING_DIR}/manifest.gpt2_combined_mc_sweep.txt"
loralens_write_manifest "${MANIFEST_FILE}"

echo "Generated ${TOTAL_RUNS} runnable runs"
echo "Submitting ${PBS_JOBNAME}"
echo "Manifest: ${MANIFEST_FILE}"
echo "Queue: ${PBS_QUEUE}"
echo "Select: ${PBS_SELECT}"
echo "Walltime: ${PBS_WALLTIME}"
echo "Max concurrent runs: ${MAX_CONCURRENT_RUNS}"

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
