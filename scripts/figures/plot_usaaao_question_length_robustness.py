#!/usr/bin/env python3
"""Create a question-length robustness plot for USAAAO_QA.

The figure aggregates per-example judge scores by question-length category on
the fair text-only subset. It is designed to show whether model performance
degrades as problems become longer/more context-heavy.

Default outputs:

  figures/usaaao_question_length_robustness.pdf
  figures/usaaao_question_length_robustness.svg
  figures/usaaao_question_length_robustness.png
  figures/usaaao_question_length_robustness_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MERGED_DIR = REPO_ROOT / "benchmark_results" / "usaaao_2017_2026_merged"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
LENGTH_ORDER = ["short", "medium", "long"]
LENGTH_LABELS = {"short": "Short", "medium": "Medium", "long": "Long"}

DEFAULT_MODELS = [
    "gpt-5.5",
    "claude-sonnet-4-6",
    "gpt-oss-120b",
    "gemma-4-31b",
    "astrosage-70b-20251009",
    "meta-llama-3.1-70b",
]

MODEL_LABELS = {
    "astrosage-70b-20251009": "AstroSage 70B",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemma-4-31b": "Gemma 4 31B",
    "gpt-5.5": "GPT-5.5",
    "gpt-oss-120b": "GPT-OSS 120B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "meta-llama-3.1-70b": "Llama 3.1 70B",
}

MODEL_COLORS = {
    "gpt-5.5": "#9C6ADE",
    "claude-sonnet-4-6": "#C17B33",
    "gpt-oss-120b": "#6F4E7C",
    "gemma-4-31b": "#F58518",
    "astrosage-70b-20251009": "#54A24B",
    "meta-llama-3.1-70b": "#4C78A8",
    "gpt-oss-20b": "#B279A2",
}

WIDTH = 620
HEIGHT = 348
LEFT = 58
TOP = 88
CHART_W = 486
CHART_H = 198
Y_MAX = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=DEFAULT_MERGED_DIR,
        help="Directory containing all_models_per_example.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figure files.",
    )
    parser.add_argument(
        "--stem",
        default="usaaao_question_length_robustness",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model slugs to include, in plotted/legend order.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["svg", "pdf", "png"],
        default=["svg", "pdf", "png"],
        help="Output formats.",
    )
    return parser.parse_args()


def is_false(value: str) -> bool:
    return value.strip().lower() in {"false", "0", "no", ""}


def load_length_scores(path: Path, selected_models: Iterable[str]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required per-example CSV: {path}")

    selected = list(selected_models)
    selected_set = set(selected)
    scores_by_key: Dict[tuple[str, str], List[float]] = defaultdict(list)
    bucket_ids: Dict[str, set[str]] = {name: set() for name in LENGTH_ORDER}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model_slug", "id", "question_length", "has_image", "judge_score"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            length = row["question_length"].strip().lower()
            if length not in LENGTH_ORDER or not is_false(row["has_image"]):
                continue
            bucket_ids[length].add(row["id"])
            slug = row["model_slug"]
            if slug in selected_set:
                scores_by_key[(slug, length)].append(float(row["judge_score"]))

    bucket_counts = {length: len(bucket_ids[length]) for length in LENGTH_ORDER}
    data: List[Dict[str, Any]] = []
    for slug in selected:
        length_items: List[Dict[str, float | int | str | None]] = []
        means: List[float] = []
        for length in LENGTH_ORDER:
            scores = scores_by_key.get((slug, length), [])
            if scores:
                mean = sum(scores) / len(scores)
                means.append(mean)
                length_items.append(
                    {
                        "length": length,
                        "count": len(scores),
                        "judge_score": mean,
                    }
                )
            else:
                length_items.append({"length": length, "count": 0, "judge_score": None})
        if not means:
            raise ValueError(f"No text-only question-length data found for selected model: {slug}")
        data.append(
            {
                "model_slug": slug,
                "model": MODEL_LABELS.get(slug, slug),
                "overall_length_mean": sum(means) / len(means),
                "length_sd": statistics.pstdev(means) if len(means) > 1 else 0.0,
                "short_to_long_drop": (
                    float(length_items[0]["judge_score"]) - float(length_items[-1]["judge_score"])
                    if length_items[0]["judge_score"] is not None and length_items[-1]["judge_score"] is not None
                    else None
                ),
                "by_length": length_items,
            }
        )
    return data, bucket_counts


def escape_svg(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 10,
    weight: str = "400",
    anchor: str = "start",
    fill: str = "#222222",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{escape_svg(text)}</text>"
    )


def x_for_length(length: str) -> float:
    idx = LENGTH_ORDER.index(length)
    return LEFT + CHART_W * idx / (len(LENGTH_ORDER) - 1)


def y_for_score(score: float) -> float:
    return TOP + CHART_H - CHART_H * score / Y_MAX


def polyline_points(item: Dict[str, Any]) -> str:
    points = []
    for entry in item["by_length"]:
        score = entry["judge_score"]
        if score is None:
            continue
        points.append(f"{x_for_length(str(entry['length'])):.1f},{y_for_score(float(score)):.1f}")
    return " ".join(points)


def write_svg(data: List[Dict[str, Any]], bucket_counts: Dict[str, int], path: Path) -> None:
    muted = "#666666"
    grid = "#D8D8D8"
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(WIDTH / 2, 22, "Question-length robustness", size=15, weight="700", anchor="middle"),
        svg_text(WIDTH / 2, 39, "Fair text-only judge score by length category", size=10, anchor="middle", fill=muted),
    ]

    legend_x = LEFT + 8
    legend_y = 58
    cursor = legend_x
    row_y = legend_y
    for idx, item in enumerate(data):
        if idx == 3:
            cursor = legend_x
            row_y += 15
        color = MODEL_COLORS.get(item["model_slug"], "#999999")
        parts.append(f'<line x1="{cursor:.1f}" y1="{row_y-3:.1f}" x2="{cursor+14:.1f}" y2="{row_y-3:.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{cursor+7:.1f}" cy="{row_y-3:.1f}" r="2.8" fill="{color}" stroke="white" stroke-width="0.6"/>')
        parts.append(svg_text(cursor + 19, row_y, item["model"], size=8, fill=muted))
        cursor += 150 if len(item["model"]) > 13 else 120

    for tick in [0.0, 0.3, 0.6, 0.9]:
        y = y_for_score(tick)
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT+CHART_W}" y2="{y:.1f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(svg_text(LEFT - 8, y + 4, f"{tick:.1f}", size=9, anchor="end", fill=muted))

    for length in LENGTH_ORDER:
        x = x_for_length(length)
        parts.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP+CHART_H}" stroke="#EEEEEE" stroke-width="0.9"/>')
        parts.append(svg_text(x, TOP + CHART_H + 18, f"{LENGTH_LABELS[length]}", size=10, anchor="middle", fill="#333333"))
        parts.append(svg_text(x, TOP + CHART_H + 32, f"n={bucket_counts[length]}", size=8, anchor="middle", fill=muted))

    for item in data:
        color = MODEL_COLORS.get(item["model_slug"], "#999999")
        parts.append(
            f'<polyline points="{polyline_points(item)}" fill="none" stroke="{color}" '
            f'stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for entry in item["by_length"]:
            score = entry["judge_score"]
            if score is None:
                continue
            parts.append(
                f'<circle cx="{x_for_length(str(entry["length"])):.1f}" cy="{y_for_score(float(score)):.1f}" '
                f'r="3.8" fill="{color}" stroke="white" stroke-width="0.8"/>'
            )

    parts.append(f'<line x1="{LEFT}" y1="{TOP+CHART_H}" x2="{LEFT+CHART_W}" y2="{TOP+CHART_H}" stroke="#999999" stroke-width="1"/>')
    parts.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP+CHART_H}" stroke="#999999" stroke-width="1"/>')
    parts.append(svg_text(LEFT + CHART_W / 2, HEIGHT - 13, "Question-length category", size=10, anchor="middle", fill=muted))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"Wrote {path}")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb(hex_color: str, *, stroke: bool = False) -> str:
    hex_color = hex_color.lstrip("#")
    vals = " ".join(f"{int(hex_color[i:i+2], 16) / 255:.4f}" for i in (0, 2, 4))
    return vals + (" RG" if stroke else " rg")


def write_pdf(data: List[Dict[str, Any]], bucket_counts: Dict[str, int], path: Path) -> None:
    commands: List[str] = []

    def y_coord(sy: float) -> float:
        return HEIGHT - sy

    def text(
        x: float,
        sy: float,
        value: str,
        *,
        size: int = 10,
        center: bool = False,
        right: bool = False,
        fill: str = "#222222",
    ) -> None:
        width = len(value) * size * 0.52
        if center:
            x -= width / 2
        if right:
            x -= width
        commands.append(f"BT {rgb(fill)} /F1 {size} Tf {x:.1f} {y_coord(sy):.1f} Td ({pdf_escape(value)}) Tj ET")

    def line(x1: float, sy1: float, x2: float, sy2: float, *, stroke: str = "#D8D8D8", lw: float = 1.0) -> None:
        commands.append(f"{rgb(stroke, stroke=True)} {lw} w {x1:.1f} {y_coord(sy1):.1f} m {x2:.1f} {y_coord(sy2):.1f} l S")

    def rect(x: float, sy: float, w: float, h: float, fill: str) -> None:
        commands.append(f"{rgb(fill)} {x:.1f} {y_coord(sy+h):.1f} {w:.1f} {h:.1f} re f")

    def circle(cx: float, sy: float, r: float, fill: str) -> None:
        k = 0.5522847498
        c = r * k
        cy = y_coord(sy)
        commands.append(
            f"{rgb(fill)} {cx:.1f} {cy+r:.1f} m "
            f"{cx+c:.1f} {cy+r:.1f} {cx+r:.1f} {cy+c:.1f} {cx+r:.1f} {cy:.1f} c "
            f"{cx+r:.1f} {cy-c:.1f} {cx+c:.1f} {cy-r:.1f} {cx:.1f} {cy-r:.1f} c "
            f"{cx-c:.1f} {cy-r:.1f} {cx-r:.1f} {cy-c:.1f} {cx-r:.1f} {cy:.1f} c "
            f"{cx-r:.1f} {cy+c:.1f} {cx-c:.1f} {cy+r:.1f} {cx:.1f} {cy+r:.1f} c f"
        )

    def polyline(item: Dict[str, Any], color: str) -> None:
        entries = [entry for entry in item["by_length"] if entry["judge_score"] is not None]
        if not entries:
            return
        first = entries[0]
        path_bits = [f"{x_for_length(str(first['length'])):.1f} {y_coord(y_for_score(float(first['judge_score']))):.1f} m"]
        for entry in entries[1:]:
            path_bits.append(f"{x_for_length(str(entry['length'])):.1f} {y_coord(y_for_score(float(entry['judge_score']))):.1f} l")
        commands.append(f"{rgb(color, stroke=True)} 2.1 w {' '.join(path_bits)} S")

    rect(0, 0, WIDTH, HEIGHT, "#FFFFFF")
    text(WIDTH / 2, 22, "Question-length robustness", size=15, center=True)
    text(WIDTH / 2, 39, "Fair text-only judge score by length category", size=10, center=True, fill="#666666")

    legend_x = LEFT + 8
    legend_y = 58
    cursor = legend_x
    row_y = legend_y
    for idx, item in enumerate(data):
        if idx == 3:
            cursor = legend_x
            row_y += 15
        color = MODEL_COLORS.get(item["model_slug"], "#999999")
        line(cursor, row_y - 3, cursor + 14, row_y - 3, stroke=color, lw=2.0)
        circle(cursor + 7, row_y - 3, 2.8, color)
        text(cursor + 19, row_y, item["model"], size=8, fill="#666666")
        cursor += 150 if len(item["model"]) > 13 else 120

    for tick in [0.0, 0.3, 0.6, 0.9]:
        sy = y_for_score(tick)
        line(LEFT, sy, LEFT + CHART_W, sy)
        text(LEFT - 8, sy + 4, f"{tick:.1f}", size=9, right=True, fill="#666666")

    for length in LENGTH_ORDER:
        x = x_for_length(length)
        line(x, TOP, x, TOP + CHART_H, stroke="#EEEEEE", lw=0.9)
        text(x, TOP + CHART_H + 18, LENGTH_LABELS[length], size=10, center=True, fill="#333333")
        text(x, TOP + CHART_H + 32, f"n={bucket_counts[length]}", size=8, center=True, fill="#666666")

    for item in data:
        color = MODEL_COLORS.get(item["model_slug"], "#999999")
        polyline(item, color)
        for entry in item["by_length"]:
            score = entry["judge_score"]
            if score is not None:
                circle(x_for_length(str(entry["length"])), y_for_score(float(score)), 3.8, color)

    line(LEFT, TOP + CHART_H, LEFT + CHART_W, TOP + CHART_H, stroke="#999999")
    line(LEFT, TOP, LEFT, TOP + CHART_H, stroke="#999999")
    text(LEFT + CHART_W / 2, HEIGHT - 13, "Question-length category", size=10, center=True, fill="#666666")

    content = "\n".join(commands).encode("latin-1")
    objects: List[bytes] = []

    def add(obj: bytes | str) -> int:
        if isinstance(obj, str):
            obj = obj.encode("latin-1")
        objects.append(obj)
        return len(objects)

    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    stream = add(f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream")
    page = add(
        f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] "
        f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>"
    )
    pages = add(f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>")
    objects[page - 1] = (
        f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] "
        f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>"
    ).encode("latin-1")
    catalog = add(f"<< /Type /Catalog /Pages {pages} 0 R >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    out.extend(f"trailer << /Size {len(objects)+1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))
    path.write_bytes(out)
    print(f"Wrote {path}")


def write_png(path: Path, pdf_path: Path) -> None:
    if shutil.which("sips") is None:
        print("Skipping PNG: macOS `sips` is not available and matplotlib is not installed.")
        return
    subprocess.run(["sips", "-s", "format", "png", str(pdf_path), "--out", str(path)], check=True)
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    source = args.merged_dir / "all_models_per_example.csv"
    data, bucket_counts = load_length_scores(source, args.models)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "source": str(source),
        "subset": "text-only",
        "metric": "judge_score",
        "length_order": LENGTH_ORDER,
        "bucket_counts": bucket_counts,
        "models": args.models,
        "items": data,
    }
    stats_path = args.output_dir / f"{args.stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")

    pdf_path = args.output_dir / f"{args.stem}.pdf"
    if "svg" in args.formats:
        write_svg(data, bucket_counts, args.output_dir / f"{args.stem}.svg")
    if "pdf" in args.formats or "png" in args.formats:
        write_pdf(data, bucket_counts, pdf_path)
    if "png" in args.formats:
        write_png(args.output_dir / f"{args.stem}.png", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
