#!/usr/bin/env python3
"""Create a metric-disagreement figure for the USAAAO_QA paper.

The figure compares judge score against reference-based metrics on the fair
204-example text-only subset. It is designed to support the paper's evaluation
claim: lexical and embedding metrics are useful diagnostics, but they are not
reliable stand-ins for judged scientific correctness.

Outputs are dependency-free SVG/PDF plus a JSON stats file. PNG output is
generated from the PDF with macOS `sips` when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MERGED_DIR = REPO_ROOT / "benchmark_results" / "usaaao_2017_2026_merged"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "figures"

METRICS = [
    ("bleu", "BLEU"),
    ("rouge_l", "ROUGE-L"),
    ("bertscore_f1", "BERTScore F1"),
    ("cosine_similarity", "Cosine similarity"),
]

MODEL_LABELS = {
    "astrollama-3-8b-aic": "AstroLLaMA 3 8B",
    "astrosage-70b-20251009": "AstroSage 70B",
    "astrosage-8b": "AstroSage 8B",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemma-3-4b": "Gemma 3 4B",
    "gemma-4-26b-a4b": "Gemma 4 26B",
    "gemma-4-31b": "Gemma 4 31B",
    "gpt-5.5": "GPT-5.5",
    "gpt-oss-120b": "GPT-OSS 120B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "llama-3.2-1b": "Llama 3.2 1B",
    "meta-llama-3-8b": "Llama 3 8B",
    "meta-llama-3.1-70b": "Llama 3.1 70B",
}

MODEL_GROUPS = {
    "astrollama-3-8b-aic": "Open astro",
    "astrosage-70b-20251009": "Open astro",
    "astrosage-8b": "Open astro",
    "claude-sonnet-4-6": "API general",
    "gpt-5.5": "API general",
    "gpt-oss-120b": "API general",
    "gpt-oss-20b": "Open general",
    "llama-3.1-8b": "Open general",
    "llama-3.2-1b": "Open general",
    "meta-llama-3-8b": "Open general",
    "meta-llama-3.1-70b": "Open general",
    "gemma-3-4b": "Open multimodal",
    "gemma-4-26b-a4b": "Open multimodal",
    "gemma-4-31b": "Open multimodal",
}

GROUP_COLORS = {
    "API general": "#B279A2",
    "Open general": "#4C78A8",
    "Open astro": "#54A24B",
    "Open multimodal": "#F58518",
}

LABEL_SLUGS = {
    "gpt-5.5",
    "claude-sonnet-4-6",
    "gpt-oss-120b",
    "astrosage-70b-20251009",
}

LABEL_OFFSETS = {
    ("bleu", "gpt-5.5"): (-7, -8, "end"),
    ("bleu", "claude-sonnet-4-6"): (7, 7, "start"),
    ("bleu", "gpt-oss-120b"): (7, -6, "start"),
    ("bleu", "astrosage-70b-20251009"): (6, -7, "start"),
    ("cosine_similarity", "gpt-5.5"): (-7, -8, "end"),
    ("cosine_similarity", "claude-sonnet-4-6"): (7, 7, "start"),
    ("cosine_similarity", "gpt-oss-120b"): (7, -7, "start"),
    ("cosine_similarity", "astrosage-70b-20251009"): (-7, 8, "end"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="usaaao_metric_disagreement")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["svg", "pdf", "png"],
        default=["svg", "pdf", "png"],
        help="Formats to generate. PNG requires macOS sips.",
    )
    return parser.parse_args()


def read_text_only_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    items: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("group") != "text_only":
            continue
        slug = row["model_slug"]
        item = {
            "model_slug": slug,
            "model": MODEL_LABELS.get(slug, slug),
            "group": MODEL_GROUPS.get(slug, "Other"),
            "judge_score": float(row["judge_score"]),
        }
        for metric, _ in METRICS:
            item[metric] = float(row[metric])
        items.append(item)
    if not items:
        raise ValueError(f"No text_only rows found in {path}")
    return sorted(items, key=lambda item: item["judge_score"])


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_vals = list(xs)
    y_vals = list(ys)
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    numerator = sum((x - mx) * (y - my) for x, y in zip(x_vals, y_vals))
    sx = math.sqrt(sum((x - mx) ** 2 for x in x_vals))
    sy = math.sqrt(sum((y - my) ** 2 for y in y_vals))
    if sx == 0 or sy == 0:
        return float("nan")
    return numerator / (sx * sy)


def linear_fit(xs: Iterable[float], ys: Iterable[float]) -> Tuple[float, float]:
    x_vals = list(xs)
    y_vals = list(ys)
    mx = sum(x_vals) / len(x_vals)
    my = sum(y_vals) / len(y_vals)
    denom = sum((x - mx) ** 2 for x in x_vals)
    if denom == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(x_vals, y_vals)) / denom
    return slope, my - slope * mx


def metric_ranges(items: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    for metric, _ in METRICS:
        values = [float(item[metric]) for item in items]
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.18 if hi > lo else 0.01
        ranges[metric] = (lo - pad, hi + pad)
    return ranges


def escape(text: str) -> str:
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
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(text)}</text>'
    )


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb(hex_color: str, *, stroke: bool = False) -> str:
    hex_color = hex_color.lstrip("#")
    vals = " ".join(f"{int(hex_color[i:i+2], 16) / 255:.4f}" for i in (0, 2, 4))
    return vals + (" RG" if stroke else " rg")


def build_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "subset": "text_only",
        "num_models": len(items),
        "metrics": {},
        "items": items,
    }
    xs = [float(item["judge_score"]) for item in items]
    for metric, label in METRICS:
        ys = [float(item[metric]) for item in items]
        slope, intercept = linear_fit(xs, ys)
        stats["metrics"][metric] = {
            "label": label,
            "pearson_r": pearson(xs, ys),
            "linear_fit_slope": slope,
            "linear_fit_intercept": intercept,
            "min": min(ys),
            "max": max(ys),
        }
    return stats


def panel_layout() -> List[Tuple[str, str, float, float]]:
    return [
        ("bleu", "BLEU", 78, 62),
        ("rouge_l", "ROUGE-L", 350, 62),
        ("bertscore_f1", "BERTScore F1", 78, 274),
        ("cosine_similarity", "Cosine similarity", 350, 274),
    ]


def tick_label(metric: str, value: float) -> str:
    if metric == "bleu":
        return f"{value:.3f}"
    return f"{value:.2f}"


def point_label(item: Dict[str, Any]) -> str:
    return (
        str(item["model"])
        .replace("Claude Sonnet 4.6", "Claude")
        .replace("GPT-OSS 120B", "GPT-OSS")
        .replace("AstroSage 70B", "AstroSage")
    )


def make_svg(items: List[Dict[str, Any]], stats: Dict[str, Any], path: Path) -> None:
    width, height = 620, 500
    panel_w, panel_h = 205, 142
    x_min, x_max = 0.0, 0.74
    ranges = metric_ranges(items)
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 24, "Metric disagreement on fair text-only subset", size=15, weight="700", anchor="middle"),
    ]
    grid = "#D8D8D8"
    muted = "#666666"
    ink = "#222222"

    def sx(x0: float, value: float) -> float:
        return x0 + panel_w * (value - x_min) / (x_max - x_min)

    def sy(y0: float, metric: str, value: float) -> float:
        lo, hi = ranges[metric]
        return y0 + panel_h - panel_h * (value - lo) / (hi - lo)

    for metric, title, x0, y0 in panel_layout():
        lo, hi = ranges[metric]
        parts.append(svg_text(x0, y0 - 18, f"{title} vs judge", size=11, weight="700"))
        r = stats["metrics"][metric]["pearson_r"]
        parts.append(svg_text(x0 + panel_w, y0 - 18, f"r={r:+.2f}", size=9, anchor="end", fill=muted))
        for tick in [0.0, 0.25, 0.50, 0.75]:
            x = sx(x0, min(tick, x_max))
            parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + panel_h}" stroke="{grid}" stroke-width="0.8"/>')
            parts.append(svg_text(x, y0 + panel_h + 14, f"{tick:.2f}", size=8, anchor="middle", fill=muted))
        for idx in range(4):
            val = lo + (hi - lo) * idx / 3
            y = sy(y0, metric, val)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="{grid}" stroke-width="0.8"/>')
            parts.append(svg_text(x0 - 8, y + 3, tick_label(metric, val), size=8, anchor="end", fill=muted))
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#999999" stroke-width="0.8"/>')

        slope = stats["metrics"][metric]["linear_fit_slope"]
        intercept = stats["metrics"][metric]["linear_fit_intercept"]
        y_left = slope * x_min + intercept
        y_right = slope * x_max + intercept
        parts.append(
            f'<line x1="{sx(x0, x_min):.1f}" y1="{sy(y0, metric, y_left):.1f}" '
            f'x2="{sx(x0, x_max):.1f}" y2="{sy(y0, metric, y_right):.1f}" '
            'stroke="#777777" stroke-width="1.2" stroke-dasharray="4 3"/>'
        )

        for item in items:
            x = sx(x0, float(item["judge_score"]))
            y = sy(y0, metric, float(item[metric]))
            color = GROUP_COLORS.get(str(item["group"]), "#999999")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" fill="{color}" stroke="white" stroke-width="0.8"/>')
            if item["model_slug"] in LABEL_SLUGS and metric == "bleu":
                dx, dy, anchor = LABEL_OFFSETS.get((metric, item["model_slug"]), (5, -5, "start"))
                parts.append(svg_text(x + dx, y + dy, point_label(item), size=7, anchor=anchor, fill=ink))
        parts.append(svg_text(x0 + panel_w / 2, y0 + panel_h + 31, "Judge score", size=9, anchor="middle", fill=muted))

    legend_y = 468
    legend_x = 90
    for idx, (group, color) in enumerate(GROUP_COLORS.items()):
        x = legend_x + idx * 128
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="4.5" fill="{color}"/>')
        parts.append(svg_text(x + 9, legend_y + 3, group, size=8, fill=muted))
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def make_pdf(items: List[Dict[str, Any]], stats: Dict[str, Any], path: Path) -> None:
    width, height = 620, 500
    panel_w, panel_h = 205, 142
    x_min, x_max = 0.0, 0.74
    ranges = metric_ranges(items)
    commands: List[str] = []
    grid = "#D8D8D8"
    muted = "#666666"
    ink = "#222222"

    def y_page(sy: float) -> float:
        return height - sy

    def text(x: float, sy_: float, value: str, *, size: int = 9, center: bool = False, right: bool = False, fill: str = "#222222") -> None:
        tw = len(value) * size * 0.52
        if center:
            x -= tw / 2
        if right:
            x -= tw
        commands.append(f"BT {rgb(fill)} /F1 {size} Tf {x:.1f} {y_page(sy_):.1f} Td ({pdf_escape(value)}) Tj ET")

    def line(x1: float, sy1: float, x2: float, sy2: float, *, stroke: str = "#999999", lw: float = 0.8, dash: bool = False) -> None:
        dash_part = "[4 3] 0 d " if dash else "[] 0 d "
        commands.append(
            f"{dash_part}{rgb(stroke, stroke=True)} {lw} w "
            f"{x1:.1f} {y_page(sy1):.1f} m {x2:.1f} {y_page(sy2):.1f} l S"
        )

    def rect(x: float, sy_: float, w: float, h: float, *, fill: str | None = None, stroke: str | None = None) -> None:
        if fill:
            commands.append(f"{rgb(fill)} {x:.1f} {y_page(sy_ + h):.1f} {w:.1f} {h:.1f} re f")
        if stroke:
            commands.append(f"{rgb(stroke, stroke=True)} 0.8 w {x:.1f} {y_page(sy_ + h):.1f} {w:.1f} {h:.1f} re S")

    def circle(cx: float, sy_: float, r: float, fill: str) -> None:
        # Bezier approximation of a filled circle.
        k = 0.5522847498
        cy = y_page(sy_)
        commands.append(
            f"{rgb(fill)} {cx:.1f} {cy + r:.1f} m "
            f"{cx + k*r:.1f} {cy + r:.1f} {cx + r:.1f} {cy + k*r:.1f} {cx + r:.1f} {cy:.1f} c "
            f"{cx + r:.1f} {cy - k*r:.1f} {cx + k*r:.1f} {cy - r:.1f} {cx:.1f} {cy - r:.1f} c "
            f"{cx - k*r:.1f} {cy - r:.1f} {cx - r:.1f} {cy - k*r:.1f} {cx - r:.1f} {cy:.1f} c "
            f"{cx - r:.1f} {cy + k*r:.1f} {cx - k*r:.1f} {cy + r:.1f} {cx:.1f} {cy + r:.1f} c f"
        )

    def sx(x0: float, value: float) -> float:
        return x0 + panel_w * (value - x_min) / (x_max - x_min)

    def sy(y0: float, metric: str, value: float) -> float:
        lo, hi = ranges[metric]
        return y0 + panel_h - panel_h * (value - lo) / (hi - lo)

    rect(0, 0, width, height, fill="#FFFFFF")
    text(width / 2, 24, "Metric disagreement on fair text-only subset", size=15, center=True)

    for metric, title, x0, y0 in panel_layout():
        lo, hi = ranges[metric]
        text(x0, y0 - 18, f"{title} vs judge", size=11)
        r = stats["metrics"][metric]["pearson_r"]
        text(x0 + panel_w, y0 - 18, f"r={r:+.2f}", size=9, right=True, fill=muted)
        for tick in [0.0, 0.25, 0.50, 0.75]:
            x = sx(x0, min(tick, x_max))
            line(x, y0, x, y0 + panel_h, stroke=grid)
            text(x, y0 + panel_h + 14, f"{tick:.2f}", size=8, center=True, fill=muted)
        for idx in range(4):
            val = lo + (hi - lo) * idx / 3
            yy = sy(y0, metric, val)
            line(x0, yy, x0 + panel_w, yy, stroke=grid)
            text(x0 - 8, yy + 3, tick_label(metric, val), size=8, right=True, fill=muted)
        rect(x0, y0, panel_w, panel_h, stroke="#999999")
        slope = stats["metrics"][metric]["linear_fit_slope"]
        intercept = stats["metrics"][metric]["linear_fit_intercept"]
        line(
            sx(x0, x_min),
            sy(y0, metric, slope * x_min + intercept),
            sx(x0, x_max),
            sy(y0, metric, slope * x_max + intercept),
            stroke="#777777",
            lw=1.1,
            dash=True,
        )
        for item in items:
            x = sx(x0, float(item["judge_score"]))
            yy = sy(y0, metric, float(item[metric]))
            circle(x, yy, 4.2, GROUP_COLORS.get(str(item["group"]), "#999999"))
            if item["model_slug"] in LABEL_SLUGS and metric == "bleu":
                dx, dy, anchor = LABEL_OFFSETS.get((metric, item["model_slug"]), (5, -5, "start"))
                text(x + dx, yy + dy, point_label(item), size=7, right=anchor == "end", fill=ink)
        text(x0 + panel_w / 2, y0 + panel_h + 31, "Judge score", size=9, center=True, fill=muted)

    legend_y = 468
    legend_x = 90
    for idx, (group, color) in enumerate(GROUP_COLORS.items()):
        x = legend_x + idx * 128
        circle(x, legend_y, 4.5, color)
        text(x + 9, legend_y + 3, group, size=8, fill=muted)

    content = "\n".join(commands).encode("latin-1")
    objects: List[bytes] = []

    def add(obj: bytes | str) -> int:
        if isinstance(obj, str):
            obj = obj.encode("latin-1")
        objects.append(obj)
        return len(objects)

    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    stream = add(f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream")
    page = add(f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>")
    pages = add(f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>")
    objects[page - 1] = f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {stream} 0 R >>".encode("latin-1")
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


def make_png(pdf_path: Path, png_path: Path) -> None:
    if shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "png", str(pdf_path), "--out", str(png_path)],
            check=True,
        )
    else:
        print("Skipping PNG output: `sips` is not available.")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = read_text_only_rows(args.merged_dir / "by_model_modality.csv")
    stats = build_stats(items)

    stats_path = args.output_dir / f"{args.stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")

    svg_path = args.output_dir / f"{args.stem}.svg"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    png_path = args.output_dir / f"{args.stem}.png"

    if "svg" in args.formats:
        make_svg(items, stats, svg_path)
        print(f"Wrote {svg_path}")
    if "pdf" in args.formats or "png" in args.formats:
        make_pdf(items, stats, pdf_path)
        print(f"Wrote {pdf_path}")
    if "png" in args.formats:
        make_png(pdf_path, png_path)
        if png_path.exists():
            print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
