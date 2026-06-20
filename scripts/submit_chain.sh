#!/usr/bin/env bash
# submit_chain.sh — submit a sequence of PBS jobs where each waits for the previous.
#
# Usage:
#   # Chain mode: jobs run one after another
#   ./scripts/submit_chain.sh job1.pbs job2.pbs job3.pbs
#
#   # Fan-out mode: all jobs run in parallel (no dependencies)
#   ./scripts/submit_chain.sh --parallel job1.pbs job2.pbs job3.pbs
#
#   # Wait mode: jobs fan-out but a final job waits for ALL of them
#   ./scripts/submit_chain.sh --parallel --then finalize.pbs job1.pbs job2.pbs
#
#   # Dry-run: print the qsub commands without submitting
#   ./scripts/submit_chain.sh --dry-run job1.pbs job2.pbs job3.pbs
#
# Dependency semantics:
#   afterok  — next job starts only if the previous exited 0    (default)
#   afterany — next job starts regardless of exit code
#
# Override with:
#   DEPEND_TYPE=afterany ./scripts/submit_chain.sh job1.pbs job2.pbs
#
set -euo pipefail

DEPEND_TYPE="${DEPEND_TYPE:-afterok}"
PARALLEL=false
FINAL_PBS=""
DRY_RUN=false
PBS_SCRIPTS=()

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)   PARALLEL=true; shift ;;
        --then)       FINAL_PBS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --depend)     DEPEND_TYPE="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            PBS_SCRIPTS+=("$1"); shift ;;
    esac
done

if [[ ${#PBS_SCRIPTS[@]} -eq 0 ]]; then
    echo "Error: no PBS scripts provided." >&2
    exit 1
fi

# ── Submit helper ────────────────────────────────────────────────────────────
submit() {
    local script="$1"
    local dep_flag="${2:-}"

    if [[ ! -f "$script" ]]; then
        echo "Error: PBS script not found: $script" >&2
        exit 1
    fi

    local cmd="qsub"
    [[ -n "$dep_flag" ]] && cmd="$cmd -W depend=${dep_flag}"
    cmd="$cmd $script"

    if $DRY_RUN; then
        echo "[dry-run] $cmd"
        echo "DRY_RUN_JOBID_$$_${RANDOM}"
        return
    fi

    local jobid
    jobid=$(eval "$cmd")
    echo "$jobid"
}

# ── Chain mode ───────────────────────────────────────────────────────────────
if ! $PARALLEL; then
    prev_id=""
    declare -a submitted_ids=()

    echo "Submitting chain (${#PBS_SCRIPTS[@]} jobs, depend=${DEPEND_TYPE}):"
    for script in "${PBS_SCRIPTS[@]}"; do
        dep_flag=""
        [[ -n "$prev_id" ]] && dep_flag="${DEPEND_TYPE}:${prev_id}"

        jobid=$(submit "$script" "$dep_flag")
        submitted_ids+=("$jobid")
        prev_id="$jobid"

        echo "  $(basename "$script")  →  $jobid"
        [[ -n "$dep_flag" ]] && echo "      (waits for ${dep_flag})"
    done

    if [[ -n "$FINAL_PBS" ]]; then
        dep_flag="${DEPEND_TYPE}:${prev_id}"
        jobid=$(submit "$FINAL_PBS" "$dep_flag")
        echo "  [final] $(basename "$FINAL_PBS")  →  $jobid"
    fi

    echo ""
    echo "Chain submitted: ${submitted_ids[*]}"

# ── Parallel (fan-out) mode ──────────────────────────────────────────────────
else
    declare -a submitted_ids=()

    echo "Submitting parallel fan-out (${#PBS_SCRIPTS[@]} jobs):"
    for script in "${PBS_SCRIPTS[@]}"; do
        jobid=$(submit "$script" "")
        submitted_ids+=("$jobid")
        echo "  $(basename "$script")  →  $jobid"
    done

    if [[ -n "$FINAL_PBS" ]]; then
        # Final job waits for ALL parallel jobs
        all_ids=$(IFS=":"; echo "${submitted_ids[*]}")
        dep_flag="${DEPEND_TYPE}:${all_ids}"
        jobid=$(submit "$FINAL_PBS" "$dep_flag")
        echo "  [final] $(basename "$FINAL_PBS")  →  $jobid  (waits for all)"
    fi

    echo ""
    echo "Submitted: ${submitted_ids[*]}"
fi
