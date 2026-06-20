#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

CLIP_VALUES_CSV="${CLIP_VALUES_CSV:-5,10,20,50,100,200,inf}"
IFS=',' read -r -a CLIP_VALUES <<< "${CLIP_VALUES_CSV}"
NODE_COUNT="${#CLIP_VALUES[@]}"

PBS_SCRIPT="${PBS_SCRIPT:-${REPO_DIR}/jobs/pbs/run_sweep_manifest.sh}"
PBS_QUEUE="${PBS_QUEUE:-debug-scaling}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-${NODE_COUNT}:ngpus=4}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-gpt2_hajek_clip}"
MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-${NODE_COUNT}}"
DRY_RUN="${DRY_RUN:-0}"

BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_hajek_clip_sweep_250_debug_scaling}"
STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_hajek_clip_sweep_250_debug_scaling}"
LOG_SUBDIR="${LOG_SUBDIR:-gpt2_hajek_clip_sweep_250_debug_scaling}"

MODEL_NAME="${MODEL_NAME:-gpt2}"
DATA_SOURCE="${DATA_SOURCE:-pile}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-250}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-1e-3}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-262144}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOKEN_SHIFT="${TOKEN_SHIFT:-0}"
SAVE_EVERY="${SAVE_EVERY:-50}"
SEED="${SEED:-0}"

KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
LORA_RANK="${LORA_RANK:-64}"
HAJEK_K="${HAJEK_K:-256}"
HAJEK_TAIL="${HAJEK_TAIL:-256}"
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
)

clip_slug() {
  local value="$1"
  printf '%s\n' "${value}" | sed -e 's/+//g' -e 's/-/m/g' -e 's/\./p/g'
}

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

echo "Preparing GPT-2 Hajek clip sweep"
echo "Output root: ${BASE_OUT}"
echo "Staging dir: ${STAGING_DIR}"
echo "Clip values: ${CLIP_VALUES[*]}"
echo "Budget: ${HAJEK_K}+${HAJEK_TAIL}, oversample=${SUBSET_KL_TAIL_OVERSAMPLE}"

for clip in "${CLIP_VALUES[@]}"; do
  slug="$(clip_slug "${clip}")"
  add_or_skip_run \
    "lora_subset_hajek_r${LORA_RANK}_k${HAJEK_K}_tail${HAJEK_TAIL}_clip${slug}" \
    "${BASE_OUT}/lora_subset_hajek_r${LORA_RANK}_k${HAJEK_K}_tail${HAJEK_TAIL}_clip${slug}" \
    "${COMMON_RUN_KV[@]}" \
    "LENS_TYPE=lora" \
    "LOSS_TYPE=subset_kl" \
    "LORA_RANK=${LORA_RANK}" \
    "SUBSET_KL_MODE=hajek" \
    "SUBSET_KL_K=${HAJEK_K}" \
    "SUBSET_KL_K_TAIL=${HAJEK_TAIL}" \
    "SUBSET_KL_TAIL_CLIP=${clip}" \
    "SUBSET_KL_TAIL_OVERSAMPLE=${SUBSET_KL_TAIL_OVERSAMPLE}" \
    "KL_CHUNK_SIZE=${KL_CHUNK_SIZE}"
done

TOTAL_RUNS="${#RUN_ENV_FILES[@]}"
if (( TOTAL_RUNS == 0 )); then
  echo "No runnable Hajek clip-sweep runs were generated."
  exit 0
fi

MANIFEST_FILE="${STAGING_DIR}/manifest.gpt2_hajek_clip_sweep_250_debug_scaling.txt"
printf '%s\n' "${RUN_ENV_FILES[@]}" > "${MANIFEST_FILE}"

echo "Generated ${TOTAL_RUNS} runnable runs"
submit_manifest "${MANIFEST_FILE}" "${TOTAL_RUNS}"
