#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Re-run only the former Hajek head/tail grid for the standard ranks, using
# the subset_kl Monte Carlo estimator instead of Hajek.
exec env \
  SWEEP_LABEL="GPT-2 standard-rank head/tail MC sweep" \
  PART_INDEX=1 \
  PART_COUNT=1 \
  LORA_RANKS_CSV="${LORA_RANKS_CSV:-64,128,256,384}" \
  INCLUDE_LORA_FULL_KL=0 \
  INCLUDE_LORA_TOPK=0 \
  INCLUDE_LORA_HAJEK=1 \
  HEAD_TAIL_MODE=mc \
  HEAD_TAIL_RUN_FAMILY=mc \
  BASE_OUT="${BASE_OUT:-${REPO_DIR}/src/checkpoints/gpt2_preemptable_mc_sweep_1000}" \
  STAGING_ROOT="${STAGING_ROOT:-${REPO_DIR}/tmp/gpt2_preemptable_mc_sweep_1000}" \
  LOG_SUBDIR="${LOG_SUBDIR:-gpt2_preemptable_mc_sweep_1000}" \
  SAVE_EVERY="${SAVE_EVERY:-125}" \
  MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-10}" \
  PBS_QUEUE="${PBS_QUEUE:-prod}" \
  PBS_SELECT="${PBS_SELECT:-10:ngpus=4}" \
  PBS_WALLTIME="${PBS_WALLTIME:-03:00:00}" \
  PBS_JOBNAME="${PBS_JOBNAME:-gpt2standard_mc_prod}" \
  "${SCRIPT_DIR}/submit_gpt2_lowrank_sweep_part.sh"
