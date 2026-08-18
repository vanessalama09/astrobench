#!/usr/bin/env python3
"""Plot model performance by benchmark setting and question modality.

This script reads the merged 2017--2026 USAAAO_QA result CSVs and creates a
paper-ready comparison figure:

  1. overall judge score by model, colored by evaluation setting;
  2. text-only vs image-linked question performance for multimodal-capable
     models.

The SVG output uses only the Python standard library. PDF/PNG outputs require
matplotlib and can be generated on the cluster/eval environment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MERGED_DIR = REPO_ROOT / "benchmark_results" / "usaaao_2017_2026_merged"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "figures"

METRICS = ["judge_score", "bleu", "rouge_l", "bertscore_f1", "cosine_similarity"]

MODEL_LABELS = {
    "astrollama-3-8b-aic": "AstroLLaMA 3 8B AIC",
    "astrosage-70b-20251009": "AstroSage 70B",
    "astrosage-8b": "AstroSage 8B",
    "gemma-3-4b": "Gemma 3 4B IT",
    "gemma-4-26b-a4b": "Gemma 4 26B A4B IT",
    "gemma-4-31b": "Gemma 4 31B IT",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.5": "GPT-5.5",
    "gpt-oss-120b": "GPT-OSS 120B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "llama-3.2-1b": "Llama 3.2 1B",
    "meta-llama-3-8b": "Llama 3 8B Inst.",
    "meta-llama-3.1-70b": "Llama 3.1 70B Inst.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=DEFAULT_MERGED_DIR,
        help="Directory containing merged model_summaries.csv and by_model_modality.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures and stats.",
    )
    parser.add_argument(
        "--stem",
        default="usaaao_model_modality_comparison",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="judge_score",
        help="Primary metric to plot.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["svg", "pdf", "png"],
        default=["svg"],
        help="Output formats. SVG is dependency-free; PDF/PNG require matplotlib.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def as_int(row: Dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def label_for(slug: str) -> str:
    return MODEL_LABELS.get(slug, slug)


def compact_multimodal_label(label: str) -> str:
    return (
        label.replace("Gemma 4 26B A4B IT", "G4 26B A4B")
        .replace("Gemma 4 31B IT", "G4 31B")
        .replace("Gemma 3 4B IT", "Gemma 3 4B")
    )


def build_plot_data(summary_rows: List[Dict[str, str]], modality_rows: List[Dict[str, str]], metric: str) -> Dict[str, Any]:
    multimodal_capable_slugs = {
        row["model_slug"]
        for row in modality_rows
        if row.get("group") == "multimodal"
    }

    overall = []
    for row in summary_rows:
        slug = row["model_slug"]
        is_multimodal_capable = slug in multimodal_capable_slugs
        setting = "Full, multimodal-capable" if is_multimodal_capable else "Text-only subset"
        overall.append(
            {
                "model_slug": slug,
                "model": label_for(slug),
                "setting": setting,
                "num_examples": as_int(row, "num_examples"),
                "metric_value": as_float(row, metric),
                "judge_score": as_float(row, "judge_score"),
                "bleu": as_float(row, "bleu"),
                "rouge_l": as_float(row, "rouge_l"),
                "bertscore_f1": as_float(row, "bertscore_f1"),
                "cosine_similarity": as_float(row, "cosine_similarity"),
            }
        )
    overall.sort(key=lambda item: item["metric_value"])

    multimodal_breakdown: Dict[str, Dict[str, Any]] = {}
    for row in modality_rows:
        slug = row["model_slug"]
        if slug not in multimodal_capable_slugs:
            continue
        entry = multimodal_breakdown.setdefault(
            slug,
            {
                "model_slug": slug,
                "model": label_for(slug),
                "text_only": None,
                "image_linked": None,
                "text_only_count": 0,
                "image_linked_count": 0,
            },
        )
        if row.get("group") == "text_only":
            entry["text_only"] = as_float(row, metric)
            entry["text_only_count"] = as_int(row, "count")
        elif row.get("group") == "multimodal":
            entry["image_linked"] = as_float(row, metric)
            entry["image_linked_count"] = as_int(row, "count")

    breakdown = sorted(multimodal_breakdown.values(), key=lambda item: item["model"])
    return {
        "metric": metric,
        "overall": overall,
        "multimodal_breakdown": breakdown,
    }


def metric_label(metric: str) -> str:
    return {
        "judge_score": "Judge score",
        "bleu": "BLEU",
        "rouge_l": "ROUGE-L",
        "bertscore_f1": "BERTScore F1",
        "cosine_similarity": "Cosine similarity",
    }[metric]


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for PDF/PNG output. Generate SVG only, or "
            "run this script in the cluster/eval environment with matplotlib."
        ) from exc
    return plt, Patch


def make_matplotlib_figure(data: Dict[str, Any], output_dir: Path, stem: str, formats: Iterable[str]) -> None:
    plt, Patch = require_matplotlib()
    metric = data["metric"]
    metric_name = metric_label(metric)
    overall = data["overall"]
    breakdown = data["multimodal_breakdown"]

    colors = {
        "text": "#4C78A8",
        "full": "#F58518",
        "text_questions": "#72B7B2",
        "image_questions": "#E45756",
        "grid": "#D8D8D8",
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(12, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax_overall = fig.add_subplot(grid[0, 0])
    ax_breakdown = fig.add_subplot(grid[0, 1])

    y = list(range(len(overall)))
    values = [item["metric_value"] for item in overall]
    bar_colors = [
        colors["full"] if item["setting"].startswith("Full") else colors["text"]
        for item in overall
    ]
    ax_overall.barh(y, values, color=bar_colors, height=0.68)
    ax_overall.set_yticks(y)
    ax_overall.set_yticklabels([item["model"] for item in overall])
    ax_overall.set_xlabel(metric_name)
    ax_overall.set_title("(a) Overall performance by model")
    ax_overall.grid(axis="x", color=colors["grid"], alpha=0.8, linewidth=0.7)
    ax_overall.set_xlim(0, max(values) * 1.18)
    for ypos, item in zip(y, overall):
        ax_overall.text(
            item["metric_value"] + max(values) * 0.015,
            ypos,
            f"{item['metric_value']:.3f}",
            va="center",
            fontsize=8,
        )
    ax_overall.legend(
        handles=[
            Patch(facecolor=colors["full"], label="Multimodal-capable, full set"),
            Patch(facecolor=colors["text"], label="Text-only models, text-only subset"),
        ],
        frameon=False,
        loc="lower right",
    )

    x = list(range(len(breakdown)))
    width = 0.36
    text_values = [item["text_only"] for item in breakdown]
    image_values = [item["image_linked"] for item in breakdown]
    ax_breakdown.bar([i - width / 2 for i in x], text_values, width=width, color=colors["text_questions"], label="Text-only questions")
    ax_breakdown.bar([i + width / 2 for i in x], image_values, width=width, color=colors["image_questions"], label="Image-linked questions")
    ax_breakdown.set_xticks(x)
    ax_breakdown.set_xticklabels([item["model"] for item in breakdown], rotation=25, ha="right")
    ax_breakdown.set_ylabel(metric_name)
    ax_breakdown.set_title("(b) Full-setting models by question type")
    ax_breakdown.grid(axis="y", color=colors["grid"], alpha=0.8, linewidth=0.7)
    ax_breakdown.set_ylim(0, max(text_values + image_values) * 1.22)
    for xpos, value in zip([i - width / 2 for i in x], text_values):
        ax_breakdown.text(xpos, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip([i + width / 2 for i in x], image_values):
        ax_breakdown.text(xpos, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax_breakdown.legend(frameon=False, loc="upper left")

    fig.suptitle("USAAAO_QA model performance by evaluation setting", fontsize=12, weight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"Wrote {path}")
    plt.close(fig)


def svg_text(x: float, y: float, text: str, *, size: int = 14, weight: str = "400", anchor: str = "start", fill: str = "#222222") -> str:
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


def make_svg(data: Dict[str, Any], output_dir: Path, stem: str) -> None:
    metric = data["metric"]
    metric_name = metric_label(metric)
    overall = data["overall"]
    breakdown = data["multimodal_breakdown"]

    width, height = 1200, 670
    colors = {
        "text": "#4C78A8",
        "full": "#F58518",
        "text_questions": "#72B7B2",
        "image_questions": "#E45756",
        "grid": "#D8D8D8",
        "muted": "#666666",
        "axis": "#999999",
    }
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "USAAAO_QA model performance by evaluation setting", size=18, weight="700", anchor="middle"),
    ]

    # Panel A: overall horizontal bars.
    x0, y0, chart_w, row_h = 285, 78, 455, 34
    max_value = max(item["metric_value"] for item in overall)
    parts.append(svg_text(46, 70, "(a) Overall performance by model", size=14, weight="700"))
    for i in range(5):
        value = max_value * i / 4
        x = x0 + chart_w * value / max_value
        parts.append(f'<line x1="{x:.1f}" y1="{y0-16}" x2="{x:.1f}" y2="{y0 + row_h * len(overall) - 7}" stroke="{colors["grid"]}" stroke-width="1"/>')
        parts.append(svg_text(x, y0 + row_h * len(overall) + 14, f"{value:.2f}", size=10, anchor="middle", fill=colors["muted"]))
    for idx, item in enumerate(overall):
        y = y0 + idx * row_h
        bar_w = chart_w * item["metric_value"] / max_value
        fill = colors["full"] if item["setting"].startswith("Full") else colors["text"]
        parts.append(svg_text(x0 - 12, y + 15, item["model"], size=11, anchor="end", fill="#222222"))
        parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="22" fill="{fill}"/>')
        parts.append(svg_text(x0 + bar_w + 7, y + 16, f"{item['metric_value']:.3f}", size=10, fill="#222222"))
    axis_y = y0 + row_h * len(overall) - 7
    parts.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x0+chart_w}" y2="{axis_y}" stroke="{colors["axis"]}" stroke-width="1"/>')
    parts.append(svg_text(x0 + chart_w / 2, axis_y + 37, metric_name, size=11, anchor="middle", fill=colors["muted"]))
    legend_y = 630
    parts.append(f'<rect x="48" y="{legend_y}" width="12" height="12" fill="{colors["full"]}"/>')
    parts.append(svg_text(66, legend_y + 11, "Full-setting models, full set", size=11, fill=colors["muted"]))
    parts.append(f'<rect x="260" y="{legend_y}" width="12" height="12" fill="{colors["text"]}"/>')
    parts.append(svg_text(278, legend_y + 11, "Text-only models, text-only subset", size=11, fill=colors["muted"]))

    # Panel B: multimodal-capable model breakdown.
    bx0, by0, bw, bh = 835, 95, 300, 330
    parts.append(svg_text(805, 70, "(b) Full-setting models by question type", size=14, weight="700"))
    bmax = max(
        max(item["text_only"] or 0, item["image_linked"] or 0)
        for item in breakdown
    )
    for i in range(5):
        value = bmax * i / 4
        y = by0 + bh - bh * value / bmax
        parts.append(f'<line x1="{bx0}" y1="{y:.1f}" x2="{bx0+bw}" y2="{y:.1f}" stroke="{colors["grid"]}" stroke-width="1"/>')
        parts.append(svg_text(bx0 - 10, y + 4, f"{value:.2f}", size=10, anchor="end", fill=colors["muted"]))
    group_w = bw / len(breakdown)
    bar_w = 25
    for idx, item in enumerate(breakdown):
        center = bx0 + group_w * idx + group_w / 2
        text_h = bh * item["text_only"] / bmax
        image_h = bh * item["image_linked"] / bmax
        text_x = center - bar_w - 4
        image_x = center + 4
        parts.append(f'<rect x="{text_x:.1f}" y="{by0+bh-text_h:.1f}" width="{bar_w}" height="{text_h:.1f}" fill="{colors["text_questions"]}"/>')
        parts.append(f'<rect x="{image_x:.1f}" y="{by0+bh-image_h:.1f}" width="{bar_w}" height="{image_h:.1f}" fill="{colors["image_questions"]}"/>')
        parts.append(svg_text(text_x + bar_w / 2, by0 + bh - text_h - 6, f"{item['text_only']:.3f}", size=9, anchor="middle"))
        parts.append(svg_text(image_x + bar_w / 2, by0 + bh - image_h - 6, f"{item['image_linked']:.3f}", size=9, anchor="middle"))
        label = compact_multimodal_label(item["model"])
        parts.append(svg_text(center, by0 + bh + 21, label, size=10, anchor="middle", fill=colors["muted"]))
    parts.append(f'<line x1="{bx0}" y1="{by0+bh}" x2="{bx0+bw}" y2="{by0+bh}" stroke="{colors["axis"]}" stroke-width="1"/>')
    parts.append(f'<rect x="{bx0}" y="462" width="12" height="12" fill="{colors["text_questions"]}"/>')
    parts.append(svg_text(bx0 + 18, 473, "Text-only questions", size=11, fill=colors["muted"]))
    parts.append(f'<rect x="{bx0 + 150}" y="462" width="12" height="12" fill="{colors["image_questions"]}"/>')
    parts.append(svg_text(bx0 + 168, 473, "Image-linked questions", size=11, fill=colors["muted"]))

    parts.append("</svg>")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.svg"
    path.write_text("\n".join(parts))
    print(f"Wrote {path}")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb_fill(hex_color: str, *, stroke: bool = False) -> str:
    hex_color = hex_color.lstrip("#")
    vals = " ".join(f"{int(hex_color[i:i+2], 16) / 255:.4f}" for i in (0, 2, 4))
    return vals + (" RG" if stroke else " rg")


def make_pdf(data: Dict[str, Any], output_dir: Path, stem: str) -> None:
    """Write a vector PDF without requiring matplotlib."""

    metric = data["metric"]
    metric_name = metric_label(metric)
    overall = data["overall"]
    breakdown = data["multimodal_breakdown"]
    width, height = 1200, 670
    colors = {
        "text": "#4C78A8",
        "full": "#F58518",
        "text_questions": "#72B7B2",
        "image_questions": "#E45756",
        "grid": "#D8D8D8",
        "muted": "#666666",
        "axis": "#999999",
        "ink": "#222222",
    }

    objects: List[bytes] = []

    def add(obj: str | bytes) -> int:
        if isinstance(obj, str):
            obj = obj.encode("latin-1")
        objects.append(obj)
        return len(objects)

    def y_coord(svg_y: float) -> float:
        return height - svg_y

    commands: List[str] = []

    def text(
        x: float,
        sy: float,
        value: str,
        *,
        size: int = 10,
        right: bool = False,
        center: bool = False,
        fill: str = "#222222",
    ) -> None:
        text_width = len(value) * size * 0.52
        if right:
            x -= text_width
        if center:
            x -= text_width / 2
        commands.append(
            f"BT {rgb_fill(fill)} /F1 {size} Tf {x:.1f} {y_coord(sy):.1f} Td "
            f"({pdf_escape(value)}) Tj ET"
        )

    def line(x1: float, sy1: float, x2: float, sy2: float, *, stroke: str = "#D8D8D8", lw: float = 1.0) -> None:
        commands.append(
            f"{rgb_fill(stroke, stroke=True)} {lw} w "
            f"{x1:.1f} {y_coord(sy1):.1f} m {x2:.1f} {y_coord(sy2):.1f} l S"
        )

    def rect(x: float, sy: float, w: float, h: float, fill: str) -> None:
        commands.append(f"{rgb_fill(fill)} {x:.1f} {y_coord(sy+h):.1f} {w:.1f} {h:.1f} re f")

    rect(0, 0, width, height, "#FFFFFF")
    text(width / 2, 34, "USAAAO_QA model performance by evaluation setting", size=18, center=True)

    x0, y0, chart_w, row_h = 285, 78, 455, 34
    max_value = max(item["metric_value"] for item in overall)
    text(46, 70, "(a) Overall performance by model", size=14)
    for i in range(5):
        value = max_value * i / 4
        x = x0 + chart_w * value / max_value
        line(x, y0 - 16, x, y0 + row_h * len(overall) - 7, stroke=colors["grid"])
        text(x, y0 + row_h * len(overall) + 14, f"{value:.2f}", size=10, center=True, fill=colors["muted"])
    for idx, item in enumerate(overall):
        sy = y0 + idx * row_h
        bar_w = chart_w * item["metric_value"] / max_value
        fill = colors["full"] if item["setting"].startswith("Full") else colors["text"]
        text(x0 - 12, sy + 15, item["model"], size=11, right=True, fill=colors["ink"])
        rect(x0, sy, bar_w, 22, fill)
        text(x0 + bar_w + 7, sy + 16, f"{item['metric_value']:.3f}", size=10, fill=colors["ink"])
    axis_y = y0 + row_h * len(overall) - 7
    line(x0, axis_y, x0 + chart_w, axis_y, stroke=colors["axis"])
    text(x0 + chart_w / 2, axis_y + 37, metric_name, size=11, center=True, fill=colors["muted"])
    legend_y = 630
    rect(48, legend_y, 12, 12, colors["full"])
    text(66, legend_y + 11, "Full-setting models, full set", size=11, fill=colors["muted"])
    rect(260, legend_y, 12, 12, colors["text"])
    text(278, legend_y + 11, "Text-only models, text-only subset", size=11, fill=colors["muted"])

    bx0, by0, bw, bh = 835, 95, 300, 330
    text(805, 70, "(b) Full-setting models by question type", size=14)
    bmax = max(max(item["text_only"] or 0, item["image_linked"] or 0) for item in breakdown)
    for i in range(5):
        value = bmax * i / 4
        sy = by0 + bh - bh * value / bmax
        line(bx0, sy, bx0 + bw, sy, stroke=colors["grid"])
        text(bx0 - 10, sy + 4, f"{value:.2f}", size=10, right=True, fill=colors["muted"])
    group_w = bw / len(breakdown)
    bar_w = 25
    for idx, item in enumerate(breakdown):
        center = bx0 + group_w * idx + group_w / 2
        text_h = bh * item["text_only"] / bmax
        image_h = bh * item["image_linked"] / bmax
        text_x = center - bar_w - 4
        image_x = center + 4
        rect(text_x, by0 + bh - text_h, bar_w, text_h, colors["text_questions"])
        rect(image_x, by0 + bh - image_h, bar_w, image_h, colors["image_questions"])
        text(text_x + bar_w / 2, by0 + bh - text_h - 6, f"{item['text_only']:.3f}", size=9, center=True)
        text(image_x + bar_w / 2, by0 + bh - image_h - 6, f"{item['image_linked']:.3f}", size=9, center=True)
        label = compact_multimodal_label(item["model"])
        text(center, by0 + bh + 21, label, size=10, center=True, fill=colors["muted"])
    line(bx0, by0 + bh, bx0 + bw, by0 + bh, stroke=colors["axis"])
    rect(bx0, 462, 12, 12, colors["text_questions"])
    text(bx0 + 18, 473, "Text-only questions", size=11, fill=colors["muted"])
    rect(bx0 + 150, 462, 12, 12, colors["image_questions"])
    text(bx0 + 168, 473, "Image-linked questions", size=11, fill=colors["muted"])

    content = "\n".join(commands).encode("latin-1")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    stream = add(f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream")
    page = add(f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>")
    pages = add(f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>")
    objects[page - 1] = f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>".encode("latin-1")
    catalog = add(f"<< /Type /Catalog /Pages {pages} 0 R >>")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("latin-1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(f"trailer << /Size {len(objects)+1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1"))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    path.write_bytes(output)
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    summary_rows = read_csv(args.merged_dir / "model_summaries.csv")
    modality_rows = read_csv(args.merged_dir / "by_model_modality.csv")
    data = build_plot_data(summary_rows, modality_rows, args.metric)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / f"{args.stem}_stats.json"
    stats_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {stats_path}")

    if "svg" in args.formats:
        make_svg(data, args.output_dir, args.stem)

    if "pdf" in args.formats:
        make_pdf(data, args.output_dir, args.stem)

    raster_formats = [fmt for fmt in args.formats if fmt not in {"svg", "pdf"}]
    if raster_formats:
        make_matplotlib_figure(data, args.output_dir, args.stem, raster_formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
