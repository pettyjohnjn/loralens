#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build one manifest containing every incomplete run. The shared submitter
# skips configs already at NUM_STEPS and resumes from the latest checkpoint.
exec env \
  PART_INDEX=1 \
  PART_COUNT=1 \
  SAVE_EVERY=125 \
  MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-10}" \
  PBS_QUEUE="${PBS_QUEUE:-preemptable}" \
  PBS_SELECT="${PBS_SELECT:-10:ngpus=4}" \
  PBS_WALLTIME="${PBS_WALLTIME:-10:00:00}" \
  PBS_JOBNAME="${PBS_JOBNAME:-gpt2lowrank_resume_preemptable}" \
  "${SCRIPT_DIR}/submit_gpt2_lowrank_sweep_part.sh"
