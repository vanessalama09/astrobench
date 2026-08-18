#!/usr/bin/env python3
"""Create data-overview figures for the USAAAO_QA paper.

The script reads the local JSONL corpus and generates a compact figure for the
paper's data section:

  1. current benchmark-format coverage, including first-round MCQ counts;
  2. yearly open-ended question counts, split by text-only vs image-linked;
  3. open-ended question-length distribution.

The local JSONL corpus currently contains the open-ended/NAC-style round. The
first-round multiple-choice category is represented by an aggregate count. For
2017--2026, this is 301 questions: 31 in 2017 and 30 in each year from
2018--2026.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "usaaao_qa_local"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing root-level year JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--stem",
        default="usaaao_data_overview",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--mcq-count",
        type=int,
        default=301,
        help=(
            "Multiple-choice/first-round example count to include in the "
            "format pie chart. Default: 301 for 2017--2026."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg"],
        choices=["pdf", "png", "svg"],
        help=(
            "Figure formats to write. SVG uses only the Python standard library; "
            "PDF/PNG require matplotlib."
        ),
    )
    return parser.parse_args()


def load_jsonl_records(dataset_root: Path) -> List[Dict[str, Any]]:
    paths = sorted(dataset_root.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found in {dataset_root}")

    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def question_type(record: Dict[str, Any]) -> str:
    return str(record.get("question_type") or "open-ended").strip().lower()


def has_image(record: Dict[str, Any]) -> bool:
    return bool(record.get("image"))


def summarize_records(records: Iterable[Dict[str, Any]], mcq_count: int | None) -> Dict[str, Any]:
    records = list(records)
    open_ended = [record for record in records if question_type(record) == "open-ended"]
    years = sorted({int(record["year"]) for record in open_ended})
    by_year = {
        year: {
            "total": 0,
            "text_only": 0,
            "image_linked": 0,
            "short": 0,
            "medium": 0,
            "long": 0,
        }
        for year in years
    }

    length_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter(question_type(record) for record in records)
    parent_count = 0
    subpart_count = 0
    image_count = 0

    for record in open_ended:
        year = int(record["year"])
        length = str(record.get("question_length") or "unknown").lower()
        image = has_image(record)

        by_year[year]["total"] += 1
        by_year[year]["image_linked" if image else "text_only"] += 1
        if length in {"short", "medium", "long"}:
            by_year[year][length] += 1
        length_counter[length] += 1
        image_count += int(image)
        if record.get("parent_id") is None:
            parent_count += 1
        else:
            subpart_count += 1

    return {
        "dataset_root": str(DEFAULT_DATASET_ROOT),
        "num_records": len(records),
        "num_open_ended": len(open_ended),
        "num_mcq": mcq_count,
        "num_total_with_mcq": len(open_ended) + (mcq_count or 0),
        "question_type_counts": dict(type_counter),
        "years": years,
        "num_years": len(years),
        "by_year": by_year,
        "question_length_counts": dict(length_counter),
        "num_text_only": len(open_ended) - image_count,
        "num_image_linked": image_count,
        "num_parent_questions": parent_count,
        "num_subpart_records": subpart_count,
    }


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate the data overview figure. "
            "Install it with `pip install matplotlib`."
        ) from exc
    return plt, Patch


def annotate_bar_values(ax, bars, *, dy: float = 0.8, fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + height + dy,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def make_overview_figure(summary: Dict[str, Any], output_dir: Path, stem: str, formats: List[str]) -> None:
    plt, Patch = require_matplotlib()

    years = summary["years"]
    text_counts = [summary["by_year"][year]["text_only"] for year in years]
    image_counts = [summary["by_year"][year]["image_linked"] for year in years]
    length_order = ["short", "medium", "long"]
    length_counts = [summary["question_length_counts"].get(label, 0) for label in length_order]

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(12, 3.15), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.85, 1.0])
    ax_format = fig.add_subplot(grid[0, 0])
    ax_year = fig.add_subplot(grid[0, 1])
    ax_length = fig.add_subplot(grid[0, 2])

    # Panel A: benchmark-format coverage.
    open_count = summary["num_open_ended"]
    mcq_count = summary["num_mcq"]
    if mcq_count is not None and mcq_count > 0:
        wedges, _ = ax_format.pie(
            [open_count, mcq_count],
            startangle=90,
            counterclock=False,
            radius=0.82,
            colors=["#4C78A8", "#F58518"],
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        ax_format.legend(
            wedges,
            [f"Open-ended: {open_count}", f"MCQ: {mcq_count}"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.10),
            frameon=False,
        )
        ax_format.text(
            0.5,
            0.91,
            f"{open_count + mcq_count} catalogued questions",
            transform=ax_format.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
        )
    else:
        ax_format.pie(
            [open_count],
            startangle=90,
            colors=["#4C78A8"],
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        ax_format.text(
            0,
            0,
            f"{open_count}\nopen-ended",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )
        ax_format.text(
            0,
            -1.33,
            "MCQ / first-round questions:\nplanned conversion",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax_format.set_title("(a) Question formats")

    # Panel B: yearly counts.
    x = list(range(len(years)))
    width = 0.68
    ax_year.bar(
        x,
        text_counts,
        width=width,
        label="Text-only",
        color="#72B7B2",
    )
    bars_image = ax_year.bar(
        x,
        image_counts,
        bottom=text_counts,
        width=width,
        label="Image-linked",
        color="#E45756",
    )
    totals = [text + image for text, image in zip(text_counts, image_counts)]
    for xpos, total in zip(x, totals):
        ax_year.text(xpos, total + 0.8, str(total), ha="center", va="bottom", fontsize=8)
    ax_year.set_title("(b) Open-ended examples by year")
    ax_year.set_ylabel("Examples")
    ax_year.set_xticks(x)
    ax_year.set_xticklabels(years, rotation=0)
    ax_year.set_ylim(0, max(totals) + 8)
    ax_year.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax_year.legend(frameon=False, ncol=2, loc="upper left")

    # Panel C: question-length distribution.
    colors = ["#54A24B", "#EECA3B", "#B279A2"]
    bars = ax_length.bar(length_order, length_counts, color=colors, width=0.65)
    annotate_bar_values(ax_length, bars, dy=2)
    ax_length.set_title("(c) Question length")
    ax_length.set_ylabel("Examples")
    ax_length.set_ylim(0, max(length_counts) + 22)
    ax_length.grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.suptitle("USAAAO_QA data overview", fontsize=11, weight="bold")

    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"Wrote {path}")

    plt.close(fig)


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "400",
    anchor: str = "start",
    fill: str = "#222222",
) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escaped}</text>'
    )


def make_overview_svg(summary: Dict[str, Any], output_dir: Path, stem: str) -> None:
    years = summary["years"]
    text_counts = [summary["by_year"][year]["text_only"] for year in years]
    image_counts = [summary["by_year"][year]["image_linked"] for year in years]
    totals = [text + image for text, image in zip(text_counts, image_counts)]
    length_order = ["short", "medium", "long"]
    length_counts = [summary["question_length_counts"].get(label, 0) for label in length_order]

    width, height = 1200, 480
    margin = 52
    colors = {
        "blue": "#4C78A8",
        "teal": "#72B7B2",
        "red": "#E45756",
        "green": "#54A24B",
        "yellow": "#EECA3B",
        "purple": "#B279A2",
        "grid": "#D8D8D8",
        "text": "#222222",
        "muted": "#666666",
    }

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "USAAAO_QA data overview", size=20, weight="700", anchor="middle"),
    ]

    # Panel A: format coverage, rendered as a simple donut.
    cx, cy, r = 155, 188, 88
    parts.append(svg_text(60, 66, "(a) Question formats", size=15, weight="700"))
    mcq_count = summary["num_mcq"] or 0
    open_count = summary["num_open_ended"]
    if mcq_count > 0:
        # Donut slices for open-ended and MCQ. The path math is deliberately
        # simple because this fallback SVG is for environments without matplotlib.
        import math

        total_formats = open_count + mcq_count
        open_fraction = open_count / total_formats
        start = -90
        end_open = start + 360 * open_fraction

        def polar(angle_deg: float) -> tuple[float, float]:
            angle = math.radians(angle_deg)
            return cx + r * math.cos(angle), cy + r * math.sin(angle)

        def wedge(start_deg: float, end_deg: float, color: str) -> None:
            x1, y1 = polar(start_deg)
            x2, y2 = polar(end_deg)
            large_arc = 1 if end_deg - start_deg > 180 else 0
            parts.append(
                f'<path d="M {cx} {cy} L {x1:.1f} {y1:.1f} A {r} {r} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z" fill="{color}"/>'
            )

        wedge(start, end_open, colors["blue"])
        wedge(end_open, start + 360, "#F58518")
    else:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{colors["blue"]}"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="50" fill="white"/>')
    parts.append(svg_text(cx, cy - 6, str(open_count + mcq_count), size=26, weight="700", anchor="middle"))
    parts.append(svg_text(cx, cy + 18, "total questions", size=13, anchor="middle", fill=colors["muted"]))
    parts.append(f'<rect x="72" y="318" width="12" height="12" fill="{colors["blue"]}"/>')
    parts.append(svg_text(90, 329, f"Open-ended: {open_count}", size=12, fill=colors["muted"]))
    parts.append('<rect x="72" y="340" width="12" height="12" fill="#F58518"/>')
    parts.append(svg_text(90, 351, f"MCQ: {mcq_count}", size=12, fill=colors["muted"]))

    # Panel B: yearly stacked bars.
    x0, y0, chart_w, chart_h = 350, 100, 515, 235
    max_total = max(totals)
    parts.append(svg_text(x0, 82, "(b) Open-ended examples by year", size=15, weight="700"))
    for i in range(5):
        value = round(max_total * i / 4)
        y = y0 + chart_h - chart_h * value / max_total
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+chart_w}" y2="{y:.1f}" stroke="{colors["grid"]}" stroke-width="1"/>')
        parts.append(svg_text(x0 - 12, y + 4, str(value), size=10, anchor="end", fill=colors["muted"]))
    bar_gap = 10
    bar_w = (chart_w - bar_gap * (len(years) - 1)) / len(years)
    for idx, year in enumerate(years):
        x = x0 + idx * (bar_w + bar_gap)
        text_h = chart_h * text_counts[idx] / max_total
        image_h = chart_h * image_counts[idx] / max_total
        base = y0 + chart_h
        parts.append(f'<rect x="{x:.1f}" y="{base-text_h:.1f}" width="{bar_w:.1f}" height="{text_h:.1f}" fill="{colors["teal"]}"/>')
        parts.append(f'<rect x="{x:.1f}" y="{base-text_h-image_h:.1f}" width="{bar_w:.1f}" height="{image_h:.1f}" fill="{colors["red"]}"/>')
        parts.append(svg_text(x + bar_w / 2, base - text_h - image_h - 5, str(totals[idx]), size=10, anchor="middle"))
        parts.append(svg_text(x + bar_w / 2, base + 18, str(year), size=10, anchor="middle", fill=colors["muted"]))
    legend_y = y0 + chart_h + 42
    parts.append(f'<rect x="{x0}" y="{legend_y-10}" width="12" height="12" fill="{colors["teal"]}"/>')
    parts.append(svg_text(x0 + 18, legend_y, "Text-only", size=12, fill=colors["muted"]))
    parts.append(f'<rect x="{x0+112}" y="{legend_y-10}" width="12" height="12" fill="{colors["red"]}"/>')
    parts.append(svg_text(x0 + 130, legend_y, "Image-linked", size=12, fill=colors["muted"]))

    # Panel C: length distribution.
    lx0, ly0, lw, lh = 950, 120, 200, 190
    parts.append(svg_text(930, 82, "(c) Question length", size=15, weight="700"))
    max_len = max(length_counts)
    length_colors = [colors["green"], colors["yellow"], colors["purple"]]
    for idx, (label, count) in enumerate(zip(length_order, length_counts)):
        bw = 55
        x = lx0 + idx * 65
        h = lh * count / max_len
        y = ly0 + lh - h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{h:.1f}" fill="{length_colors[idx]}"/>')
        parts.append(svg_text(x + bw / 2, y - 8, str(count), size=12, weight="700", anchor="middle"))
        parts.append(svg_text(x + bw / 2, ly0 + lh + 22, label, size=12, anchor="middle", fill=colors["muted"]))
    parts.append(f'<line x1="{lx0-14}" y1="{ly0+lh}" x2="{lx0+lw}" y2="{ly0+lh}" stroke="#999" stroke-width="1"/>')

    parts.append("</svg>")
    path = output_dir / f"{stem}.svg"
    path.write_text("\n".join(parts))
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    records = load_jsonl_records(args.dataset_root)
    summary = summarize_records(records, args.mcq_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / f"{args.stem}_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {stats_path}")

    if "svg" in args.formats:
        make_overview_svg(summary, args.output_dir, args.stem)

    raster_formats = [fmt for fmt in args.formats if fmt != "svg"]
    if raster_formats:
        make_overview_figure(summary, args.output_dir, args.stem, raster_formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
