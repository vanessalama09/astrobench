#!/usr/bin/env python3
"""Re-score saved USAAAO benchmark predictions with a different LLM judge.

This is intended for judge-sensitivity checks. It reads an existing per-example
CSV containing question, reference_answer, and prediction columns, calls the
configured judge, and writes a new CSV with the same rows but updated
judge_verdict, judge_score, judge_rationale, and judge_model fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluate_usaaao_qa import OpenAIJudge, summarize


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            row["id"]
            for row in csv.DictReader(handle)
            if row.get("id") and row.get("judge_score") not in {None, ""}
        }


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--run-stem", required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    if args.text_only:
        rows = [row for row in rows if not parse_bool(row.get("has_image", ""))]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_csv = args.output_dir / f"{args.run_stem}.csv"
    output_json = args.output_dir / f"{args.run_stem}_summary.json"
    completed_ids = load_completed_ids(output_csv) if args.resume else set()

    fieldnames = list(rows[0].keys()) if rows else []
    for field in ("judge_model", "source_csv"):
        if field not in fieldnames:
            fieldnames.insert(0, field)

    judge = OpenAIJudge(args.judge_model)
    judged_rows: List[Dict[str, Any]] = []

    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Judge model: {args.judge_model}")
    print(f"Rows to consider: {len(rows)}")
    print(f"Resume: {args.resume} ({len(completed_ids)} completed)")

    for idx, row in enumerate(rows, start=1):
        row_id = row.get("id", f"row{idx}")
        if row_id in completed_ids:
            print(f"[{idx}/{len(rows)}] {row_id} (already completed, skipping)")
            continue
        print(f"[{idx}/{len(rows)}] {row_id}")
        judged = judge.judge(
            row.get("question", ""),
            row.get("reference_answer", ""),
            row.get("prediction", ""),
        )
        out = dict(row)
        out["judge_model"] = args.judge_model
        out["source_csv"] = str(args.input_csv)
        out["judge_verdict"] = judged.get("verdict")
        out["judge_score"] = judged.get("score")
        out["judge_rationale"] = judged.get("rationale")
        write_rows(output_csv, fieldnames, [out])
        judged_rows.append(out)

    final_rows = read_rows(output_csv)
    summary = summarize(final_rows)
    summary["judge_model"] = args.judge_model
    summary["source_csv"] = str(args.input_csv)
    summary["output_csv"] = str(output_csv)
    with output_json.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote rejudged CSV to: {output_csv}")
    print(f"Wrote summary to: {output_json}")
    print(f"Judge score: {summary.get('judge_score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
