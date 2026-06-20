#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/lib/common.sh"

usage() {
  cat <<'EOF'
Usage:
  jobs/submit/requeue_until_complete.sh RUN_ENV_FILE [PBS_SCRIPT]

Stages one env file, submits exactly one PBS job, and if OUTPUT_DIR already
contains checkpoints it resumes from the newest one. Re-run this script to
launch the next segment.

Arguments:
  RUN_ENV_FILE   Env file consumed by jobs/launch/run_env_task.sh
  PBS_SCRIPT     Optional PBS entrypoint. Defaults to jobs/pbs/run_single_env.sh

Environment overrides:
  PBS_QUEUE      Queue passed to qsub
  PBS_ACCOUNT    Account passed to qsub
  PBS_SELECT     Resource select string, e.g. 1 or 1:ngpus=4
  PBS_WALLTIME   Walltime passed to qsub
  PBS_FILESYSTEMS Filesystems resource passed to qsub
  PBS_JOBNAME    Base PBS job name. Defaults to RUN_NAME

Notes:
  - If no checkpoint exists yet, the job starts from scratch.
  - If checkpoints already exist, the next submission resumes from whichever
    `lens_step_*.pt` file was written most recently.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_ENV_INPUT="${1:-}"
if [[ -z "${RUN_ENV_INPUT}" ]]; then
  usage >&2
  exit 2
fi

PBS_SCRIPT_INPUT="${2:-${REPO_DIR}/jobs/pbs/run_single_env.sh}"

RUN_ENV_FILE="$(cd "$(dirname "${RUN_ENV_INPUT}")" && pwd)/$(basename "${RUN_ENV_INPUT}")"
PBS_SCRIPT="$(cd "$(dirname "${PBS_SCRIPT_INPUT}")" && pwd)/$(basename "${PBS_SCRIPT_INPUT}")"

if [[ ! -f "${RUN_ENV_FILE}" ]]; then
  echo "Run env file not found: ${RUN_ENV_FILE}"
  exit 1
fi

if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "PBS script not found: ${PBS_SCRIPT}"
  exit 1
fi

# shellcheck disable=SC1090
source "${RUN_ENV_FILE}"

RUN_NAME="${RUN_NAME:?missing RUN_NAME in ${RUN_ENV_FILE}}"
OUTPUT_DIR="${OUTPUT_DIR:?missing OUTPUT_DIR in ${RUN_ENV_FILE}}"
NUM_STEPS="${NUM_STEPS:?missing NUM_STEPS in ${RUN_ENV_FILE}}"

PBS_QUEUE="${PBS_QUEUE:-debug}"
PBS_ACCOUNT="${PBS_ACCOUNT:-ModCon}"
PBS_SELECT="${PBS_SELECT:-1}"
PBS_WALLTIME="${PBS_WALLTIME:-01:00:00}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:grand:eagle}"
PBS_JOBNAME="${PBS_JOBNAME:-${RUN_NAME}}"

STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/single_run_submit/${RUN_NAME}}"
loralens_init_staging_dir "${STAGING_ROOT}"

latest_checkpoint_path() {
  local output_dir="$1"
  find "${output_dir}" -maxdepth 1 -type f -name 'lens_step_*.pt' -printf '%T@\t%p\n' \
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

mkdir -p "${OUTPUT_DIR}"
current_checkpoint="$(latest_checkpoint_path "${OUTPUT_DIR}" || true)"
current_step=0
if [[ -n "${current_checkpoint}" ]]; then
  current_step="$(checkpoint_step "${current_checkpoint}")"
fi

echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Target num_steps: ${NUM_STEPS}"
echo "Queue: ${PBS_QUEUE}"
echo "Account: ${PBS_ACCOUNT}"
echo "Select: ${PBS_SELECT}"
echo "Walltime: ${PBS_WALLTIME}"
echo "Filesystems: ${PBS_FILESYSTEMS}"
echo "PBS script: ${PBS_SCRIPT}"
if [[ -n "${current_checkpoint}" ]]; then
  echo "Resuming from existing checkpoint: ${current_checkpoint} (step ${current_step})"
else
  echo "No checkpoint found. Starting from scratch."
fi

if (( current_step >= NUM_STEPS )); then
  echo "Run already reached target step ${NUM_STEPS}."
  echo "Latest checkpoint: ${current_checkpoint}"
  exit 0
fi

STAGED_ENV_FILE="${STAGING_DIR}/$(basename "${RUN_ENV_FILE}")"
awk '
  /^RESUME_CHECKPOINT=/ { next }
  /^RESUME_DIR=/ { next }
  { print }
' "${RUN_ENV_FILE}" > "${STAGED_ENV_FILE}"

if [[ -n "${current_checkpoint}" ]]; then
  printf 'RESUME_CHECKPOINT=%q\n' "${current_checkpoint}" >> "${STAGED_ENV_FILE}"
fi

echo "Staged env: ${STAGED_ENV_FILE}"

JOB_ID="$(
  qsub \
    -q "${PBS_QUEUE}" \
    -A "${PBS_ACCOUNT}" \
    -N "${PBS_JOBNAME}" \
    -l "select=${PBS_SELECT}" \
    -l "walltime=${PBS_WALLTIME}" \
    -l "filesystems=${PBS_FILESYSTEMS}" \
    -v "RUN_ENV_FILE=${STAGED_ENV_FILE},REPO_DIR=${REPO_DIR},LORALENS_REPO_DIR=${REPO_DIR}" \
    "${PBS_SCRIPT}"
)"

echo "Submitted job: ${JOB_ID}"
