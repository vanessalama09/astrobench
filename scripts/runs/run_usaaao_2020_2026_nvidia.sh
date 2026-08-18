#!/usr/bin/env bash
# Run the already-tested local models on the 2020--2026 USAAAO data.
# Launch from the repository root, preferably inside tmux:
#   tmux new -s usaaao-2020-2026
#   bash scripts/runs/run_usaaao_2020_2026_nvidia.sh
#
# Set GPU_IDS to three genuinely free NVIDIA GPUs before starting, e.g.
#   GPU_IDS=1,2,3 bash scripts/runs/run_usaaao_2020_2026_nvidia.sh

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Use the CUDA-compatible environment unless the caller explicitly disables
# automatic activation (for example, inside a site-specific batch wrapper).
ENV_NAME="${ENV_NAME:-astro-eval-clean}"
if [[ "${AUTO_ACTIVATE_ENV:-1}" == "1" ]] && command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "$ENV_NAME"
fi

if [[ -z "${GPU_IDS:-}" ]]; then
  echo "Set GPU_IDS to three verified-free GPUs, e.g. GPU_IDS=1,2,3" >&2
  exit 2
fi
GPU_IDS="$GPU_IDS"
IFS=',' read -r GPU_A GPU_B GPU_C <<< "$GPU_IDS"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/usaaao_qa_local}"
YEARS=(2020 2021 2022 2023 2024 2025 2026)
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_STEM="${RUN_STEM:-usaaao_2020_2026}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/benchmark_results/$RUN_STEM}"
MASTER_LOG="$RESULT_ROOT/${RUN_STEM}.log"
mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$MASTER_LOG") 2>&1

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [[ ! -d "$DATASET_ROOT/images" || ! -f "$DATASET_ROOT/2020.jsonl" ]]; then
  log "ERROR: expected dataset root with root-level year JSONL files and images/: $DATASET_ROOT"
  exit 1
fi

log "Run stem: $RUN_STEM"
log "Dataset: $DATASET_ROOT"
log "Years: ${YEARS[*]}"
log "GPU IDs: $GPU_IDS"
log "Results: $RESULT_ROOT"
log "Python: $PYTHON_BIN ($($PYTHON_BIN -c 'import sys; print(sys.executable)'))"

$PYTHON_BIN - <<'PY'
import sys
try:
    import torch
except ImportError as exc:
    raise SystemExit(f"PyTorch is unavailable in {sys.executable}: {exc}")
if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA is unavailable in {sys.executable}; use the CUDA-compatible environment."
    )
print("PyTorch:", torch.__version__)
print("Compiled CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())
PY

run_standard() {
  local slug="$1" model="$2" gpu="$3" mode="$4"
  local out="$RESULT_ROOT/$slug" log_file="$RESULT_ROOT/$slug/$slug.log"
  mkdir -p "$out"
  {
    echo "[$(date)] model=$model gpu=$gpu mode=$mode"
    export CUDA_VISIBLE_DEVICES="$gpu"
    cmd=("$PYTHON_BIN" scripts/evaluate/evaluate_usaaao_qa.py
      --model "$model" --dataset-root "$DATASET_ROOT" --years "${YEARS[@]}"
      --output-dir "$out" --judge --judge-model "$JUDGE_MODEL"
      --run-stem "${RUN_STEM}_${slug}" --resume)
    [[ "$mode" == text-only ]] && cmd+=(--text-only)
    printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    if ! "${cmd[@]}"; then
      log "FAILED: $slug (continuing with remaining models)"
      return 1
    fi
  } > >(tee -a "$log_file") 2>&1
}

run_large() {
  local slug="$1" model="$2"
  local out="$RESULT_ROOT/$slug" log_file="$RESULT_ROOT/$slug/$slug.log"
  mkdir -p "$out"
  {
    echo "[$(date)] large model=$model gpus=$GPU_IDS mode=text-only"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    cmd=("$PYTHON_BIN" scripts/evaluate/evaluate_usaaao_qa_large.py
      --model "$model" --dataset-root "$DATASET_ROOT" --years "${YEARS[@]}"
      --output-dir "$out" --text-only --judge --judge-model "$JUDGE_MODEL"
      --run-stem "${RUN_STEM}_${slug}" --resume)
    printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    if ! "${cmd[@]}"; then
      log "FAILED: $slug (continuing with remaining models)"
      return 1
    fi
  } > >(tee -a "$log_file") 2>&1
}

log "Starting multimodal models with available one-GPU slots."
run_standard gemma-3-4b-it google/gemma-3-4b-it "$GPU_A" multimodal &
pid1=$!
run_standard gemma-4-26b-a4b google/gemma-4-26B-A4B-it "$GPU_B" multimodal &
pid2=$!
wait "$pid1" "$pid2" || true
run_standard gemma-4-31b google/gemma-4-31B-it "$GPU_A" multimodal || true

log "Starting small text-only models two at a time."
run_standard llama-3.2-1b-instruct meta-llama/Llama-3.2-1B-Instruct "$GPU_A" text-only &
pid1=$!
run_standard llama-3.1-8b meta-llama/Llama-3.1-8B "$GPU_B" text-only &
pid2=$!
wait "$pid1" "$pid2" || true
run_standard llama-3-8b-instruct meta-llama/Meta-Llama-3-8B-Instruct "$GPU_A" text-only &
pid1=$!
run_standard astrollama-3-8b-aic AstroMLab/astrollama-3-8b-chat_aic "$GPU_B" text-only &
pid2=$!
wait "$pid1" "$pid2" || true
run_standard astrosage-8b AstroMLab/AstroSage-8B "$GPU_A" text-only &
pid1=$!
run_standard gpt-oss-20b openai/gpt-oss-20b "$GPU_B" text-only &
pid2=$!
wait "$pid1" "$pid2" || true

log "Starting 70B models with all requested GPUs (exclusive)."
run_large llama-3.1-70b-instruct meta-llama/Meta-Llama-3.1-70B-Instruct || true
run_large astrosage-70b-20251009 AstroMLab/AstroSage-70B-20251009 || true

log "All requested local-model evaluations completed."
log "Master log: $MASTER_LOG"
