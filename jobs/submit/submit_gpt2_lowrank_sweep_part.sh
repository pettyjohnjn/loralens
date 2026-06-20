#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

PART_INDEX="${PART_INDEX:-${1:-}}"
if [[ -z "${PART_INDEX}" ]]; then
  echo "Usage: PART_INDEX=1 $0"
  echo "       $0 1"
  exit 2
fi

PART_COUNT="${PART_COUNT:-4}"
if (( PART_INDEX < 1 || PART_INDEX > PART_COUNT )); then
  echo "PART_INDEX must be between 1 and ${PART_COUNT}; got ${PART_INDEX}"
  exit 2
fi

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug-scaling}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-10:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2lowrank_${PART_INDEX}of${PART_COUNT}}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-10}"
DRY_RUN="${DRY_RUN:-0}"

BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_debug_scaling_lowrank_sweep_1000}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_debug_scaling_lowrank_sweep_1000/part_${PART_INDEX}of${PART_COUNT}}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_debug_scaling_lowrank_sweep_1000}"

MODEL_NAME="${MODEL_NAME:-gpt2}"
DATA_SOURCE="${DATA_SOURCE:-pile}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-1e-3}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-262144}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOKEN_SHIFT="${TOKEN_SHIFT:-0}"
SAVE_EVERY="${SAVE_EVERY:-250}"
SEED="${SEED:-0}"

KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
SUBSET_KL_TAIL_CLIP="${SUBSET_KL_TAIL_CLIP:-50.0}"
SUBSET_KL_TAIL_OVERSAMPLE="${SUBSET_KL_TAIL_OVERSAMPLE:-4}"
HEAD_TAIL_MODE="${HEAD_TAIL_MODE:-hajek}"
HEAD_TAIL_RUN_FAMILY="${HEAD_TAIL_RUN_FAMILY:-${HEAD_TAIL_MODE}}"
SWEEP_LABEL="${SWEEP_LABEL:-GPT-2 low-rank sweep}"

LORA_RANKS_CSV="${LORA_RANKS_CSV:-1,4,8,16,32}"
IFS=',' read -r -a LORA_RANKS <<< "${LORA_RANKS_CSV}"
TOPK_HEAD_SIZES=(256 512 768 1024)
HAJEK_PAIRS=(
  "256 256"
  "512 512"
  "768 768"
  "1024 1024"
  "512 1024"
  "1024 512"
)

INCLUDE_LORA_FULL_KL="${INCLUDE_LORA_FULL_KL:-1}"
INCLUDE_LORA_TOPK="${INCLUDE_LORA_TOPK:-1}"
INCLUDE_LORA_HAJEK="${INCLUDE_LORA_HAJEK:-1}"

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

add_full_kl_runs() {
  local rank
  if [[ "${INCLUDE_LORA_FULL_KL}" != "1" ]]; then
    return 0
  fi

  for rank in "${LORA_RANKS[@]}"; do
    add_or_skip_run \
      "lora_kl_r${rank}" \
      "${BASE_OUT}/lora_kl_r${rank}" \
      "${COMMON_RUN_KV[@]}" \
      "LENS_TYPE=lora" \
      "LOSS_TYPE=kl" \
      "LORA_RANK=${rank}" \
      "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
  done
}

add_topk_runs() {
  local k
  local rank
  if [[ "${INCLUDE_LORA_TOPK}" != "1" ]]; then
    return 0
  fi

  for rank in "${LORA_RANKS[@]}"; do
    for k in "${TOPK_HEAD_SIZES[@]}"; do
      add_or_skip_run \
        "lora_subset_topk_r${rank}_k${k}" \
        "${BASE_OUT}/lora_subset_topk_r${rank}_k${k}" \
        "${COMMON_RUN_KV[@]}" \
        "LENS_TYPE=lora" \
        "LOSS_TYPE=subset_kl" \
        "LORA_RANK=${rank}" \
        "SUBSET_KL_MODE=topk" \
        "SUBSET_KL_K=${k}" \
        "SUBSET_KL_K_TAIL=0" \
        "SUBSET_KL_TAIL_CLIP=${SUBSET_KL_TAIL_CLIP}" \
        "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE}" \
        "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
    done
  done
}

add_hajek_runs() {
  local spec
  local rank
  local k
  local tail
  if [[ "${INCLUDE_LORA_HAJEK}" != "1" ]]; then
    return 0
  fi

  for rank in "${LORA_RANKS[@]}"; do
    for spec in "${HAJEK_PAIRS[@]}"; do
      read -r k tail <<< "${spec}"
      add_or_skip_run \
        "lora_subset_${HEAD_TAIL_RUN_FAMILY}_r${rank}_k${k}_tail${tail}" \
        "${BASE_OUT}/lora_subset_${HEAD_TAIL_RUN_FAMILY}_r${rank}_k${k}_tail${tail}" \
        "${COMMON_RUN_KV[@]}" \
        "LENS_TYPE=lora" \
        "LOSS_TYPE=subset_kl" \
        "LORA_RANK=${rank}" \
        "SUBSET_KL_MODE=${HEAD_TAIL_MODE}" \
        "SUBSET_KL_K=${k}" \
        "SUBSET_KL_K_TAIL=${tail}" \
        "SUBSET_KL_TAIL_CLIP=${SUBSET_KL_TAIL_CLIP}" \
        "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE}" \
        "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
    done
  done
}

submit_manifest() {
  local manifest_file="$1"
  local run_count="$2"

  echo "Submitting ${PBS_JOBNAME}"
  echo "Manifest: ${manifest_file}"
  echo "Runs in part: ${run_count}"
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

echo "Preparing ${SWEEP_LABEL} part ${PART_INDEX}/${PART_COUNT}"
echo "Output root: ${BASE_OUT}"
echo "Staging dir: ${STAGING_DIR}"
echo "LoRA ranks: ${LORA_RANKS_CSV}"
echo "Head/tail run family: ${HEAD_TAIL_RUN_FAMILY}"
echo "Head/tail subset_kl mode: ${HEAD_TAIL_MODE}"

add_full_kl_runs
add_topk_runs
add_hajek_runs

TOTAL_RUNS="${#RUN_ENV_FILES[@]}"
if (( TOTAL_RUNS == 0 )); then
  echo "No runnable low-rank runs were generated."
  exit 0
fi

RUNS_PER_PART=$(( (TOTAL_RUNS + PART_COUNT - 1) / PART_COUNT ))
START=$(( (PART_INDEX - 1) * RUNS_PER_PART ))
REMAINING=$(( TOTAL_RUNS - START ))

if (( REMAINING <= 0 )); then
  echo "Part ${PART_INDEX}/${PART_COUNT} has no remaining runs."
  exit 0
fi

CHUNK_RUNS="${RUNS_PER_PART}"
if (( REMAINING < CHUNK_RUNS )); then
  CHUNK_RUNS="${REMAINING}"
fi

MANIFEST_FILE="${STAGING_DIR}/manifest.lowrank.${PART_INDEX}of${PART_COUNT}.txt"
printf '%s\n' "${RUN_ENV_FILES[@]:START:CHUNK_RUNS}" > "${MANIFEST_FILE}"

echo "Generated ${TOTAL_RUNS} runnable runs across all parts"
echo "Part ${PART_INDEX} starts at run offset ${START} and contains ${CHUNK_RUNS} runs"

submit_manifest "${MANIFEST_FILE}" "${CHUNK_RUNS}"
