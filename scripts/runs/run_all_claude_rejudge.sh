#!/usr/bin/env bash
set -euo pipefail

# Sequentially rejudge the selected top/representative model predictions with
# a Claude/i2 judge. This script is judge-only: it does not regenerate model
# answers and does not require GPUs.
#
# Required environment:
#   export API_KEY="..."
#
# Optional overrides:
#   export SECOND_JUDGE_MODEL="claude-sonnet-4-6"
#   export JUDGE_OPENAI_BASE_URL="..."
#   export JUDGE_SENSITIVITY_OUT_BASE="benchmark_results/judge_sensitivity_claude_jsonfix"
#
# Optional CSV path overrides, if a cluster has different result locations:
#   export GPT55_CSV="path/to/gpt-5.5.csv"
#   export GPTOSS120B_CSV="path/to/gpt-oss-120b.csv"
#   export GPTOSS20B_MERGED_CSV="path/to/gpt-oss-20b-merged.csv"
#   export GPTOSS20B_2017_2019_CSV="path/to/gpt-oss-20b-2017-2019.csv"
#   export GPTOSS20B_2020_2026_CSV="path/to/gpt-oss-20b-2020-2026.csv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SECOND_JUDGE_MODEL="${SECOND_JUDGE_MODEL:-claude-sonnet-4-6}"
JUDGE_OPENAI_BASE_URL="${JUDGE_OPENAI_BASE_URL:-<url...>}"
JUDGE_SENSITIVITY_OUT_BASE="${JUDGE_SENSITIVITY_OUT_BASE:-benchmark_results/judge_sensitivity_claude_jsonfix}"

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

export SECOND_JUDGE_MODEL
export JUDGE_OPENAI_BASE_URL
export OPENAI_BASE_URL="$JUDGE_OPENAI_BASE_URL"
export JUDGE_OPENAI_USE_AZURE=false
export JUDGE_SENSITIVITY_OUT_BASE

echo "Judge model: $SECOND_JUDGE_MODEL"
echo "Judge base URL: $JUDGE_OPENAI_BASE_URL"
echo "Output base: $JUDGE_SENSITIVITY_OUT_BASE"

echo "Checking JSON parser handles Claude fenced JSON..."
python3 - <<'PY'
import sys

sys.path.insert(0, "scripts/evaluate")
from evaluate_usaaao_qa import OpenAIJudge

payload = '```json\n{"verdict":"correct","score":1.0,"rationale":"json parser smoke test"}\n```'
parsed = OpenAIJudge._parse_json_payload(payload)
assert parsed["score"] == 1.0, parsed
print("JSON parser check passed.")
PY

run_one() {
  local label="$1"
  local script="$2"
  echo
  echo "============================================================"
  echo "Running Claude rejudge for: $label"
  echo "Script: $script"
  echo "Started: $(date)"
  echo "============================================================"
  bash "$script"
  echo "Finished: $(date)"
}

run_one "AstroSage 70B" "scripts/runs/run_astrosage70b_claude_rejudge.sh"
run_one "Gemma 4 31B" "scripts/runs/run_gemma4_31b_claude_rejudge.sh"
run_one "GPT-OSS 20B" "scripts/runs/run_gptoss20b_claude_rejudge.sh"
run_one "GPT-5.5" "scripts/runs/run_gpt55_claude_rejudge.sh"
run_one "GPT-OSS 120B" "scripts/runs/run_gptoss120b_claude_rejudge.sh"

echo
echo "All Claude rejudge jobs completed."
echo "Outputs are under: $JUDGE_SENSITIVITY_OUT_BASE"
