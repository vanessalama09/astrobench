#!/usr/bin/env bash
set -euo pipefail

# Rejudge existing GPT-OSS 120B predictions with a Claude/i2 judge.
# This is a judge-only run: it does not call the GPT-OSS 120B generation endpoint.
#
# Required environment:
#   export API_KEY="..."
# Optional overrides:
#   export SECOND_JUDGE_MODEL="claude-sonnet-4-6"
#   export JUDGE_OPENAI_BASE_URL="..."
#   export GPTOSS120B_CSV="path/to/gpt-oss-120b.csv"
#   export JUDGE_SENSITIVITY_OUT_BASE="benchmark_results/judge_sensitivity_claude_jsonfix"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SECOND_JUDGE_MODEL="${SECOND_JUDGE_MODEL:-claude-sonnet-4-6}"
JUDGE_OPENAI_BASE_URL="${JUDGE_OPENAI_BASE_URL:-<url...>}"

if [[ -n "${API_KEY:-}" ]]; then
  export JUDGE_OPENAI_API_KEY="$API_KEY"
  export OPENAI_API_KEY="$API_KEY"
elif [[ -z "${JUDGE_OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: Set API_KEY or JUDGE_OPENAI_API_KEY before running." >&2
  exit 1
fi

unset AZURE_OPENAI_ENDPOINT
unset OPENAI_AZURE_ENDPOINT
unset JUDGE_AZURE_OPENAI_ENDPOINT
unset JUDGE_OPENAI_AZURE_ENDPOINT
unset OPENAI_USE_AZURE

export JUDGE_OPENAI_BASE_URL
export OPENAI_BASE_URL="$JUDGE_OPENAI_BASE_URL"
export JUDGE_OPENAI_USE_AZURE=false

first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

INPUT_CSV="$(first_existing \
  "${GPTOSS120B_CSV:-}" \
  "benchmark_results/usaaao_2017_2026_merged/models/gpt-oss-120b/usaaao_2017_2026_gpt-oss-120b.csv" \
  "benchmark_results/usaaao_2017_2026/gpt-oss-120b-i2/usaaao_gpt-oss-120b_eval_20260805-113446.csv" \
  || true)"

if [[ -z "$INPUT_CSV" ]]; then
  echo "ERROR: Could not locate the GPT-OSS 120B input CSV." >&2
  echo "Look for it with:" >&2
  echo "  find benchmark_results -iname '*gpt*oss*120b*.csv' -o -iname '*gpt-oss-120b*.csv' | sort" >&2
  echo "Then rerun with:" >&2
  echo "  export GPTOSS120B_CSV='path/to/gpt-oss-120b.csv'" >&2
  echo "  bash scripts/runs/run_gptoss120b_claude_rejudge.sh" >&2
  exit 1
fi

OUT_BASE="${JUDGE_SENSITIVITY_OUT_BASE:-benchmark_results/judge_sensitivity_claude_jsonfix}"
OUT_DIR="$OUT_BASE/gpt-oss-120b_2017_2026"

echo "Judge model: $SECOND_JUDGE_MODEL"
echo "Judge base URL: $JUDGE_OPENAI_BASE_URL"
echo "Input CSV: $INPUT_CSV"
python3 scripts/evaluate/rejudge_usaaao_csv.py \
  --input-csv "$INPUT_CSV" \
  --output-dir "$OUT_DIR" \
  --judge-model "$SECOND_JUDGE_MODEL" \
  --run-stem "usaaao_2017_2026_gpt-oss-120b_judged_by_claude-sonnet-4-6" \
  --text-only \
  --resume

echo "Done."
