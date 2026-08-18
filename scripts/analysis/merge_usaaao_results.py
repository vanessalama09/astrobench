#!/usr/bin/env python3
"""Merge USAAAO benchmark result shards into analysis-ready tables.

The raw benchmark outputs are intentionally kept per run/model. This script
builds a canonical long-form table across year ranges, plus per-model merged
CSV/summary JSON files that can drive paper tables and exploratory analysis.

Default inputs:
  - benchmark_results/2017-2019benchmark_results/
  - benchmark_results/usaaao_2020_2026/
  - benchmark_results/usaaao_2017_2026/

Default outputs:
  - benchmark_results/usaaao_2017_2026_merged/all_models_per_example.csv
  - benchmark_results/usaaao_2017_2026_merged/model_summaries.csv
  - benchmark_results/usaaao_2017_2026_merged/by_model_year.csv
  - benchmark_results/usaaao_2017_2026_merged/by_model_modality.csv
  - benchmark_results/usaaao_2017_2026_merged/by_model_question_length.csv
  - benchmark_results/usaaao_2017_2026_merged/models/<model_slug>/
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EVALUATE_DIR = REPO_ROOT / "scripts" / "evaluate"
sys.path.insert(0, str(EVALUATE_DIR))

from evaluate_usaaao_qa import CSV_FIELDNAMES, OUTPUT_DIR, model_slug


METRIC_KEYS = [
    "bleu",
    "rouge_l",
    "bertscore_f1",
    "cosine_similarity",
    "judge_score",
]

GROUP_METRIC_KEYS = [
    "judge_score",
    "bleu",
    "rouge_l",
    "bertscore_f1",
    "cosine_similarity",
]

MODEL_META_FIELDS = [
    "model_slug",
    "generation_model",
    "judge_model",
    "source_span",
    "source_csv",
]

MERGED_FIELDNAMES = MODEL_META_FIELDS + CSV_FIELDNAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=OUTPUT_DIR,
        help="Root benchmark_results directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Input result directory to scan. Can be repeated. If omitted, scans "
            "the known 2017-2019 and 2020-2026 result locations."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "usaaao_2017_2026_merged",
        help="Directory for merged outputs.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also include CSV files without summary JSON metadata.",
    )
    return parser.parse_args()


def coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def metric_average(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = [value for row in rows if (value := coerce_float(row.get(key))) is not None]
    if not values:
        return None
    return float(statistics.mean(values))


def slug_from_summary_or_path(summary: Dict[str, Any], csv_path: Path) -> str:
    generation_model = summary.get("generation_model")
    if isinstance(generation_model, str) and generation_model.strip():
        return model_slug(generation_model)
    name = csv_path.stem
    for prefix in ("usaaao_", "usaaao_2020_2026_", "usaaao_2017_2026_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ("_eval",):
        if suffix in name:
            name = name.split(suffix, 1)[0]
    return model_slug(name)


def source_span_for_path(path: Path) -> str:
    parts = set(path.parts)
    text = str(path)
    if "2017-2019benchmark_results" in parts:
        return "2017-2019"
    if "usaaao_2020_2026" in parts:
        return "2020-2026"
    if "usaaao_2017_2026" in parts:
        return "2017-2026"
    if "2017" in text and "2019" in text:
        return "2017-2019"
    if "2020" in text and "2026" in text:
        return "2020-2026"
    return "unknown"


def paired_csv(summary_path: Path) -> Optional[Path]:
    name = summary_path.name
    if not name.endswith("_summary.json"):
        return None
    candidate = summary_path.with_name(name[: -len("_summary.json")] + ".csv")
    return candidate if candidate.exists() else None


def discover_runs(input_dirs: List[Path], include_incomplete: bool) -> List[Tuple[Path, Optional[Path]]]:
    runs: List[Tuple[Path, Optional[Path]]] = []
    seen_csvs = set()
    skip_markers = {"failed_or_abandoned", "failed", "abandoned"}

    def should_skip(path: Path) -> bool:
        parts = set(path.parts)
        if parts & skip_markers:
            return True
        return any(part.endswith("-pilot") or part.endswith("_pilot") for part in path.parts)

    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"WARNING: missing input directory: {input_dir}", file=sys.stderr)
            continue
        for summary_path in sorted(input_dir.rglob("*_summary.json")):
            if should_skip(summary_path):
                continue
            csv_path = paired_csv(summary_path)
            if csv_path is None:
                print(f"WARNING: no paired CSV for summary: {summary_path}", file=sys.stderr)
                continue
            runs.append((csv_path, summary_path))
            seen_csvs.add(csv_path.resolve())

        if include_incomplete:
            for csv_path in sorted(input_dir.rglob("*.csv")):
                if should_skip(csv_path):
                    continue
                if csv_path.resolve() not in seen_csvs:
                    runs.append((csv_path, None))
                    seen_csvs.add(csv_path.resolve())

    return runs


def load_run(csv_path: Path, summary_path: Optional[Path]) -> List[Dict[str, Any]]:
    summary: Dict[str, Any] = {}
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())

    slug = slug_from_summary_or_path(summary, csv_path)
    generation_model = summary.get("generation_model") or slug
    judge_model = summary.get("judge_model")
    source_span = source_span_for_path(csv_path)

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {field: raw.get(field, "") for field in CSV_FIELDNAMES}
            year = coerce_int(row.get("year"))
            if year is not None:
                row["year"] = year
            row["has_image"] = coerce_bool(row.get("has_image"))
            for key in METRIC_KEYS:
                value = coerce_float(row.get(key))
                row[key] = value if value is not None else ""
            row.update(
                {
                    "model_slug": slug,
                    "generation_model": generation_model,
                    "judge_model": judge_model or "",
                    "source_span": source_span,
                    "source_csv": str(csv_path),
                }
            )
            rows.append(row)
    return rows


def dedupe_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Deduplicate rows by model/example id, preferring later/full source spans."""

    priority = {"unknown": 0, "2017-2019": 1, "2020-2026": 1, "2017-2026": 2}
    chosen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = (str(row.get("model_slug")), str(row.get("id")))
        old = chosen.get(key)
        if old is None:
            chosen[key] = row
            continue
        duplicates += 1
        old_priority = priority.get(str(old.get("source_span")), 0)
        new_priority = priority.get(str(row.get("source_span")), 0)
        if new_priority >= old_priority:
            chosen[key] = row

    merged = list(chosen.values())
    merged.sort(key=lambda row: (str(row.get("model_slug")), int(row.get("year") or 0), str(row.get("id"))))
    return merged, duplicates


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_examples": len(rows),
        "years": sorted({int(row["year"]) for row in rows if row.get("year") != ""}),
    }
    for key in METRIC_KEYS:
        summary[key] = metric_average(rows, key)

    def grouped(label_key: str):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(label_key, ""))].append(row)
        return {
            label: summarize_group(group)
            for label, group in sorted(groups.items(), key=lambda item: item[0])
        }

    by_modality: Dict[str, List[Dict[str, Any]]] = {"text_only": [], "multimodal": []}
    for row in rows:
        by_modality["multimodal" if coerce_bool(row.get("has_image")) else "text_only"].append(row)

    summary["by_year"] = grouped("year")
    summary["by_question_length"] = grouped("question_length")
    summary["by_modality"] = {
        label: summarize_group(group)
        for label, group in by_modality.items()
        if group
    }
    return summary


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"count": len(rows)}
    for key in GROUP_METRIC_KEYS:
        payload[key] = metric_average(rows, key)
    return payload


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_model_summary(slug: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    generation_models = sorted({str(row.get("generation_model", "")) for row in rows if row.get("generation_model")})
    judge_models = sorted({str(row.get("judge_model", "")) for row in rows if row.get("judge_model")})
    source_spans = sorted({str(row.get("source_span", "")) for row in rows if row.get("source_span")})
    payload: Dict[str, Any] = {
        "model_slug": slug,
        "generation_model": "; ".join(generation_models),
        "judge_model": "; ".join(judge_models),
        "source_spans": "; ".join(source_spans),
        "num_examples": summary["num_examples"],
        "years": " ".join(str(year) for year in summary["years"]),
    }
    for key in METRIC_KEYS:
        payload[key] = summary.get(key)
    return payload


def grouped_rows(
    rows_by_model: Dict[str, List[Dict[str, Any]]],
    group_key: str,
    output_label: str,
) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for slug, rows in sorted(rows_by_model.items()):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if group_key == "modality":
                label = "multimodal" if coerce_bool(row.get("has_image")) else "text_only"
            else:
                label = str(row.get(group_key, ""))
            groups[label].append(row)
        for label, group in sorted(groups.items(), key=lambda item: item[0]):
            payload: Dict[str, Any] = {"model_slug": slug, output_label: label}
            payload.update(summarize_group(group))
            flat.append(payload)
    return flat


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    input_dirs = args.input_dir or [
        results_root / "2017-2019benchmark_results",
        results_root / "usaaao_2020_2026",
        results_root / "usaaao_2017_2026",
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(input_dirs, args.include_incomplete)
    if not runs:
        print("No result runs found.", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    manifest_runs: List[Dict[str, Any]] = []
    for csv_path, summary_path in runs:
        rows = load_run(csv_path, summary_path)
        all_rows.extend(rows)
        manifest_runs.append(
            {
                "csv": str(csv_path),
                "summary": str(summary_path) if summary_path else None,
                "rows": len(rows),
                "source_span": source_span_for_path(csv_path),
            }
        )

    merged_rows, duplicate_count = dedupe_rows(all_rows)
    rows_by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        rows_by_model[str(row["model_slug"])].append(row)

    write_csv(output_dir / "all_models_per_example.csv", merged_rows, MERGED_FIELDNAMES)

    model_summary_rows: List[Dict[str, Any]] = []
    for slug, rows in sorted(rows_by_model.items()):
        model_dir = output_dir / "models" / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda row: (int(row.get("year") or 0), str(row.get("id"))))
        summary = summarize_rows(rows)
        summary["model_slug"] = slug
        summary["generation_model"] = sorted(
            {str(row.get("generation_model", "")) for row in rows if row.get("generation_model")}
        )
        summary["judge_model"] = sorted(
            {str(row.get("judge_model", "")) for row in rows if row.get("judge_model")}
        )
        summary["source_spans"] = sorted(
            {str(row.get("source_span", "")) for row in rows if row.get("source_span")}
        )
        write_csv(model_dir / f"usaaao_2017_2026_{slug}.csv", rows, MERGED_FIELDNAMES)
        (model_dir / f"usaaao_2017_2026_{slug}_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        model_summary_rows.append(flatten_model_summary(slug, rows, summary))

    summary_fields = [
        "model_slug",
        "generation_model",
        "judge_model",
        "source_spans",
        "num_examples",
        "years",
        *METRIC_KEYS,
    ]
    write_csv(output_dir / "model_summaries.csv", model_summary_rows, summary_fields)

    group_fields = ["model_slug", "group", "count", *GROUP_METRIC_KEYS]
    by_year = grouped_rows(rows_by_model, "year", "group")
    by_modality = grouped_rows(rows_by_model, "modality", "group")
    by_length = grouped_rows(rows_by_model, "question_length", "group")
    write_csv(output_dir / "by_model_year.csv", by_year, group_fields)
    write_csv(output_dir / "by_model_modality.csv", by_modality, group_fields)
    write_csv(output_dir / "by_model_question_length.csv", by_length, group_fields)

    manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "runs": manifest_runs,
        "num_input_runs": len(runs),
        "num_input_rows": len(all_rows),
        "num_duplicate_model_example_rows": duplicate_count,
        "num_merged_rows": len(merged_rows),
        "num_models": len(rows_by_model),
        "models": sorted(rows_by_model),
        "outputs": {
            "all_models_per_example": str(output_dir / "all_models_per_example.csv"),
            "model_summaries": str(output_dir / "model_summaries.csv"),
            "by_model_year": str(output_dir / "by_model_year.csv"),
            "by_model_modality": str(output_dir / "by_model_modality.csv"),
            "by_model_question_length": str(output_dir / "by_model_question_length.csv"),
            "per_model_dir": str(output_dir / "models"),
        },
    }
    (output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Merged {len(runs)} runs into {len(rows_by_model)} models.")
    print(f"Input rows: {len(all_rows)}; duplicates removed: {duplicate_count}; merged rows: {len(merged_rows)}")
    print(f"Wrote merged outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
