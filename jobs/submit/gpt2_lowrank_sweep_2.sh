#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env PART_INDEX=2 PART_COUNT="${PART_COUNT:-4}" \
  "${SCRIPT_DIR}/submit_gpt2_lowrank_sweep_part.sh"
