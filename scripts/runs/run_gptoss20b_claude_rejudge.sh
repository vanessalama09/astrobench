#!/usr/bin/env bash
set -euo pipefail

# Rejudge existing GPT-OSS 20B predictions with a Claude/i2 judge.
# This is a judge-only run: it does not load the model and does not use GPUs.
#
# Required environment:
#   export API_KEY="..."
# Optional overrides:
#   export SECOND_JUDGE_MODEL="claude-sonnet-4-6"
#   export JUDGE_OPENAI_BASE_URL="..."
#   export GPTOSS20B_MERGED_CSV="path/to/merged.csv"
#   export GPTOSS20B_2017_2019_CSV="path/to/2017_2019.csv"
#   export GPTOSS20B_2020_2026_CSV="path/to/2020_2026.csv"

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

MERGED_IN="$(first_existing \
  "${GPTOSS20B_MERGED_CSV:-}" \
  "benchmark_results/usaaao_2017_2026_merged/models/gpt-oss-20b/usaaao_2017_2026_gpt-oss-20b.csv" \
  "benchmark_results/usaaao_2017_2026/gpt-oss-20b/usaaao_2017_2026_gpt-oss-20b.csv" \
  "benchmark_results/gpt-oss-20b/usaaao_2017_2026_gpt-oss-20b.csv" \
  || true)"

IN_2017_2019="$(first_existing \
  "${GPTOSS20B_2017_2019_CSV:-}" \
  "benchmark_results/usaaao_gpt-oss-20b_eval.csv" \
  "benchmark_results/2017-2019benchmark_results/usaaao_gpt-oss-20b_eval.csv" \
  "benchmark_results/usaaao_2017_2019/gpt-oss-20b/usaaao_2017_2019_gpt-oss-20b.csv" \
  "benchmark_results/usaaao_2017_2019/gpt-oss-20b/usaaao_gpt-oss-20b_eval.csv" \
  || true)"

IN_2020_2026="$(first_existing \
  "${GPTOSS20B_2020_2026_CSV:-}" \
  "benchmark_results/usaaao_2020_2026/gpt-oss-20b/usaaao_2020_2026_gpt-oss-20b.csv" \
  || true)"

OUT_BASE="${JUDGE_SENSITIVITY_OUT_BASE:-benchmark_results/judge_sensitivity_claude_jsonfix}"
OUT_ALL="$OUT_BASE/gpt-oss-20b_2017_2026"
OUT_2017_2019="$OUT_BASE/gpt-oss-20b_2017_2019"
OUT_2020_2026="$OUT_BASE/gpt-oss-20b_2020_2026"

echo "Judge model: $SECOND_JUDGE_MODEL"
echo "Judge base URL: $JUDGE_OPENAI_BASE_URL"

if [[ -n "$MERGED_IN" && -f "$MERGED_IN" ]]; then
  echo "Found merged 2017--2026 GPT-OSS 20B CSV."
  echo "Input CSV: $MERGED_IN"
  python3 scripts/evaluate/rejudge_usaaao_csv.py \
    --input-csv "$MERGED_IN" \
    --output-dir "$OUT_ALL" \
    --judge-model "$SECOND_JUDGE_MODEL" \
    --run-stem "usaaao_2017_2026_gpt-oss-20b_judged_by_claude-sonnet-4-6" \
    --text-only \
    --resume
  echo "Done."
  exit 0
fi

if [[ -z "$IN_2017_2019" || -z "$IN_2020_2026" ]]; then
  echo "ERROR: Could not locate the GPT-OSS 20B input CSVs." >&2
  echo "Look for them on this machine with:" >&2
  echo "  find benchmark_results -name '*gpt*oss*20b*.csv' -o -name '*gpt-oss-20b*.csv' | sort" >&2
  echo "Then rerun with explicit paths, for example:" >&2
  echo "  export GPTOSS20B_2017_2019_CSV='path/to/2017_2019.csv'" >&2
  echo "  export GPTOSS20B_2020_2026_CSV='path/to/2020_2026.csv'" >&2
  echo "  bash scripts/runs/run_gptoss20b_claude_rejudge.sh" >&2
  exit 1
fi

echo "Running 2017--2019 GPT-OSS 20B Claude rejudge..."
python3 scripts/evaluate/rejudge_usaaao_csv.py \
  --input-csv "$IN_2017_2019" \
  --output-dir "$OUT_2017_2019" \
  --judge-model "$SECOND_JUDGE_MODEL" \
  --run-stem "usaaao_2017_2019_gpt-oss-20b_judged_by_claude-sonnet-4-6" \
  --text-only \
  --resume

echo "Running 2020--2026 GPT-OSS 20B Claude rejudge..."
python3 scripts/evaluate/rejudge_usaaao_csv.py \
  --input-csv "$IN_2020_2026" \
  --output-dir "$OUT_2020_2026" \
  --judge-model "$SECOND_JUDGE_MODEL" \
  --run-stem "usaaao_2020_2026_gpt-oss-20b_judged_by_claude-sonnet-4-6" \
  --text-only \
  --resume

echo "Combining 2017--2026 Claude-judged outputs..."
python3 - <<'PY'
import csv
import json
from pathlib import Path
from statistics import mean

import os

out_base = Path(os.environ.get("JUDGE_SENSITIVITY_OUT_BASE", "benchmark_results/judge_sensitivity_claude_jsonfix"))
parts = [
    out_base / "gpt-oss-20b_2017_2019" / "usaaao_2017_2019_gpt-oss-20b_judged_by_claude-sonnet-4-6.csv",
    out_base / "gpt-oss-20b_2020_2026" / "usaaao_2020_2026_gpt-oss-20b_judged_by_claude-sonnet-4-6.csv",
]
out_dir = out_base / "gpt-oss-20b_2017_2026"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "usaaao_2017_2026_gpt-oss-20b_judged_by_claude-sonnet-4-6.csv"
out_json = out_dir / "usaaao_2017_2026_gpt-oss-20b_judged_by_claude-sonnet-4-6_summary.json"

rows = []
fieldnames = None
for path in parts:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)

rows.sort(key=lambda r: (int(r.get("year", 0)), r.get("id", "")))
with out_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

scores = []
for row in rows:
    try:
        scores.append(float(row["judge_score"]))
    except Exception:
        pass

summary = {
    "model_slug": "gpt-oss-20b",
    "judge_model": rows[0].get("judge_model") if rows else None,
    "num_examples": len(rows),
    "judge_score": mean(scores) if scores else None,
    "years": sorted({int(r["year"]) for r in rows if r.get("year")}),
    "parts": [str(p) for p in parts],
    "output_csv": str(out_csv),
}
with out_json.open("w") as handle:
    json.dump(summary, handle, indent=2)

print(f"Wrote combined CSV: {out_csv}")
print(f"Wrote combined summary: {out_json}")
print(f"Combined judge score: {summary['judge_score']}")
PY

echo "Done."
