#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

RUN_ENV_FILE="${RUN_ENV_FILE:-${1:-}}"
if [[ -n "${RUN_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_ENV_FILE}"
fi

for env_name in \
  HF_HOME \
  HF_DATASETS_CACHE \
  HF_HUB_CACHE \
  HF_DATASETS_OFFLINE \
  HF_HUB_OFFLINE \
  TRANSFORMERS_OFFLINE
do
  if [[ -n "${!env_name:-}" ]]; then
    export "${env_name}"
  fi
done

RUN_NAME="${RUN_NAME:?missing RUN_NAME}"
MODEL_NAME="${MODEL_NAME:?missing MODEL_NAME}"
DATA_SOURCE="${DATA_SOURCE:-pile}"
LENS_TYPE="${LENS_TYPE:?missing LENS_TYPE}"
LOSS_TYPE="${LOSS_TYPE:?missing LOSS_TYPE}"
OUTPUT_DIR="${OUTPUT_DIR:?missing OUTPUT_DIR}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_DIR="${RESUME_DIR:-}"
if [[ -n "${RESUME_CHECKPOINT}" && -n "${RESUME_DIR}" ]]; then
  echo "Set only one of RESUME_CHECKPOINT or RESUME_DIR"
  exit 1
fi

if [[ -z "${RESUME_CHECKPOINT}" && -n "${RESUME_DIR}" ]]; then
  RESUME_CHECKPOINT="${RESUME_DIR}"
fi

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-225}"
WARMUP_STEPS="${WARMUP_STEPS:-25}"
LR="${LR:-1e-3}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-262144}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOKEN_SHIFT="${TOKEN_SHIFT:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-128}"
SAVE_EVERY="${SAVE_EVERY:-${NUM_STEPS}}"
LOG_EVERY="${LOG_EVERY:-1}"
SEED="${SEED:-0}"

SUBSET_KL_MODE="${SUBSET_KL_MODE:-topk}"
SUBSET_KL_K="${SUBSET_KL_K:-128}"
SUBSET_KL_K_TAIL="${SUBSET_KL_K_TAIL:-0}"
SUBSET_KL_TAIL_CLIP="${SUBSET_KL_TAIL_CLIP:-50.0}"
SUBSET_KL_TAIL_OVERSAMPLE="${SUBSET_KL_TAIL_OVERSAMPLE:-4}"
SUBSET_KL_TAIL_PROPOSAL="${SUBSET_KL_TAIL_PROPOSAL:-target}"
SUBSET_KL_TAIL_PROPOSAL_ALPHA="${SUBSET_KL_TAIL_PROPOSAL_ALPHA:-0.8}"
SUBSET_KL_TAIL_PROPOSAL_TAU="${SUBSET_KL_TAIL_PROPOSAL_TAU:-0.7}"

MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
ACTIVATION_SITE_PRESET="${ACTIVATION_SITE_PRESET:-residual}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
MODEL_PARALLEL="${MODEL_PARALLEL:-0}"
MULTINODE_MODEL_PARALLEL="${MULTINODE_MODEL_PARALLEL:-0}"
FSDP_OFFLOAD_TO_CPU="${FSDP_OFFLOAD_TO_CPU:-0}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
PER_GPU_MAX_MEMORY="${PER_GPU_MAX_MEMORY:-}"
FORCE_CPU="${FORCE_CPU:-0}"
DDP="${DDP:-1}"
MEASURE_RSS="${MEASURE_RSS:-0}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
LOG_SUFFIX="${LOG_SUFFIX:-}"

LOG_SUBDIR="${LOG_SUBDIR:-runs}"
LOG_ROOT="${LOG_ROOT:-${LORALENS_REPO_DIR}/logs/${LOG_SUBDIR}}"
SUBSET_KL_SRC_DIR="${SUBSET_KL_SRC_DIR:-${LORALENS_REPO_DIR}/../subset-kl/src}"

if [[ "${FORCE_CPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  MODEL_PARALLEL=0
  MULTINODE_MODEL_PARALLEL=0
  FSDP_OFFLOAD_TO_CPU=0
  CPU_OFFLOAD_GB=0
  PER_GPU_MAX_MEMORY=""
  NPROC_PER_NODE=1
  DDP=0
fi

if [[ -z "${LOG_SUFFIX}" && "${NNODES}" != "1" ]]; then
  LOG_SUFFIX="node${NODE_RANK}"
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_DIR}"

if [[ -n "${LOG_SUFFIX}" ]]; then
  LOG_FILE="${LOG_ROOT}/${RUN_NAME}.${LOG_SUFFIX}.log"
else
  LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"
fi

cd "${LORALENS_REPO_DIR}"

if [[ "${SKIP_ENV_SETUP:-0}" != "1" ]]; then
  loralens_bootstrap_runtime
  loralens_install_runtime_deps
fi

if [[ -d "${SUBSET_KL_SRC_DIR}" ]]; then
  export PYTHONPATH="${SUBSET_KL_SRC_DIR}:${PYTHONPATH:-}"
fi
export PYTHONPATH="${LORALENS_SRC_DIR}:${PYTHONPATH:-}"

cd "${LORALENS_SRC_DIR}"
exec >> "${LOG_FILE}" 2>&1

echo
echo "===== run_env_task start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
echo "[setup] run_name=${RUN_NAME}"
echo "[setup] env_file=${RUN_ENV_FILE:-<inline>}"
echo "[setup] output_dir=${OUTPUT_DIR}"
echo "[setup] log_file=${LOG_FILE}"
echo "[setup] nnodes=${NNODES}"
echo "[setup] nproc_per_node=${NPROC_PER_NODE}"
echo "[setup] node_rank=${NODE_RANK}"
echo "[setup] master_addr=${MASTER_ADDR}"
echo "[setup] master_port=${MASTER_PORT}"
echo "[setup] model_name=${MODEL_NAME}"
echo "[setup] lens_type=${LENS_TYPE}"
echo "[setup] loss_type=${LOSS_TYPE}"
echo "[setup] force_cpu=${FORCE_CPU}"
echo "[setup] ddp=${DDP}"
echo "[setup] measure_rss=${MEASURE_RSS}"
loralens_print_gpu_probe
loralens_python_probe

CMD=(torchrun)
if [[ "${NNODES}" == "1" ]]; then
  CMD+=(--standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}")
else
  CMD+=(
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
fi

CMD+=(
  -m loralens.cli.main train
  --model_name "${MODEL_NAME}"
  --model_dtype "${MODEL_DTYPE}"
  --data_source "${DATA_SOURCE}"
  --lens_type "${LENS_TYPE}"
  --activation_site_preset "${ACTIVATION_SITE_PRESET}"
  --loss_type "${LOSS_TYPE}"
  --token_shift "${TOKEN_SHIFT}"
  --kl_chunk_size "${KL_CHUNK_SIZE}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --batch_size "${BATCH_SIZE}"
  --warmup_steps "${WARMUP_STEPS}"
  --lr_schedule "${LR_SCHEDULE:-constant}"
  --min_lr_ratio "${MIN_LR_RATIO:-0.0}"
  --num_steps "${NUM_STEPS}"
  --lr "${LR}"
  --tokens_per_step "${TOKENS_PER_STEP}"
  --amp
  --amp_dtype "${AMP_DTYPE}"
  --log_every "${LOG_EVERY}"
  --save_every "${SAVE_EVERY}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
)

if [[ "${SAVE_INITIAL_CHECKPOINT:-0}" == "1" ]]; then
  CMD+=(--save_initial_checkpoint)
fi

if [[ "${DDP}" == "0" ]]; then
  CMD+=(--no_ddp)
fi

if [[ "${LENS_TYPE}" == "lora" ]]; then
  LORA_RANK="${LORA_RANK:?missing LORA_RANK for LoRA run}"
  LORA_ALPHA="$(loralens_resolve_lora_alpha "${LORA_RANK}" "${LORA_ALPHA-}")"
  CMD+=(--lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}")

  LORA_INIT="${LORA_INIT:-default_lora}"
  LORA_INIT_CALIBRATION_TOKENS="${LORA_INIT_CALIBRATION_TOKENS:-50000}"
  LORA_INIT_RIDGE_LAMBDA="${LORA_INIT_RIDGE_LAMBDA:-1e-3}"
  LORA_INIT_RIDGE_LAMBDA_SCALE="${LORA_INIT_RIDGE_LAMBDA_SCALE:-trace_xxt_over_d}"
  LORA_INIT_STATS_DTYPE="${LORA_INIT_STATS_DTYPE:-float32}"
  LORA_INIT_JITTER="${LORA_INIT_JITTER:-1e-6}"
  LORA_INIT_SVD_METRIC="${LORA_INIT_SVD_METRIC:-residual}"
  LORA_INIT_NORMALIZATION="${LORA_INIT_NORMALIZATION:-none}"
  CMD+=(
    --lora_init "${LORA_INIT}"
    --lora_init_calibration_tokens "${LORA_INIT_CALIBRATION_TOKENS}"
    --lora_init_ridge_lambda "${LORA_INIT_RIDGE_LAMBDA}"
    --lora_init_ridge_lambda_scale "${LORA_INIT_RIDGE_LAMBDA_SCALE}"
    --lora_init_stats_dtype "${LORA_INIT_STATS_DTYPE}"
    --lora_init_jitter "${LORA_INIT_JITTER}"
    --lora_init_svd_metric "${LORA_INIT_SVD_METRIC}"
    --lora_init_normalization "${LORA_INIT_NORMALIZATION}"
  )
fi

if [[ "${LOSS_TYPE}" == "subset_kl" ]]; then
  CMD+=(
    --subset_kl_k "${SUBSET_KL_K}"
    --subset_kl_mode "${SUBSET_KL_MODE}"
    --subset_kl_k_tail "${SUBSET_KL_K_TAIL}"
    --subset_kl_tail_clip "${SUBSET_KL_TAIL_CLIP}"
    --subset_kl_tail_oversample "${SUBSET_KL_TAIL_OVERSAMPLE}"
    --subset_kl_tail_proposal "${SUBSET_KL_TAIL_PROPOSAL}"
    --subset_kl_tail_proposal_alpha "${SUBSET_KL_TAIL_PROPOSAL_ALPHA}"
    --subset_kl_tail_proposal_tau "${SUBSET_KL_TAIL_PROPOSAL_TAU}"
  )
fi

if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  CMD+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi

if [[ "${MODEL_PARALLEL}" == "1" ]]; then
  CMD+=(--model_parallel)
fi

if [[ -n "${PER_GPU_MAX_MEMORY}" ]]; then
  CMD+=(--per_gpu_max_memory "${PER_GPU_MAX_MEMORY}")
fi

if [[ "${CPU_OFFLOAD_GB}" != "0" ]]; then
  CMD+=(--cpu_offload_gb "${CPU_OFFLOAD_GB}")
fi

if [[ "${MULTINODE_MODEL_PARALLEL}" == "1" ]]; then
  CMD+=(--multinode_model_parallel)
fi

if [[ "${FSDP_OFFLOAD_TO_CPU}" == "1" ]]; then
  CMD+=(--fsdp_offload_to_cpu)
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

printf 'COMMAND: %q ' "${CMD[@]}"
printf '\n'

RUN_START_EPOCH="$(date +%s)"
RUN_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[timing] start_utc=${RUN_START_ISO}"

if [[ "${MEASURE_RSS}" == "1" ]]; then
  python "${LORALENS_REPO_DIR}/scripts/run_with_rss_sampler.py" -- "${CMD[@]}"
else
  "${CMD[@]}"
fi

RUN_END_EPOCH="$(date +%s)"
RUN_END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ELAPSED_SEC="$((RUN_END_EPOCH - RUN_START_EPOCH))"
printf '[timing] end_utc=%s | elapsed_sec=%s | elapsed_min=%.2f\n' \
  "${RUN_END_ISO}" \
  "${RUN_ELAPSED_SEC}" \
  "$(awk "BEGIN { print ${RUN_ELAPSED_SEC} / 60.0 }")"
