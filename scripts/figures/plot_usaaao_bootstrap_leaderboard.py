#!/usr/bin/env python3
"""Create a bootstrap-confidence leaderboard for the USAAAO_QA text-only set.

The plot compares every evaluated model on the same 204 text-only examples.
Confidence intervals are computed by bootstrapping per-example judge scores,
not from the already-aggregated summary table.

Default outputs:

  figures/usaaao_bootstrap_leaderboard.pdf
  figures/usaaao_bootstrap_leaderboard.svg
  figures/usaaao_bootstrap_leaderboard.png
  figures/usaaao_bootstrap_leaderboard_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MERGED_DIR = REPO_ROOT / "benchmark_results" / "usaaao_2017_2026_merged"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "figures"

MODEL_LABELS = {
    "astrollama-3-8b-aic": "AstroLLaMA 3 8B AIC",
    "astrosage-70b-20251009": "AstroSage 70B",
    "astrosage-8b": "AstroSage 8B",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemma-3-4b": "Gemma 3 4B IT",
    "gemma-4-26b-a4b": "Gemma 4 26B A4B",
    "gemma-4-31b": "Gemma 4 31B IT",
    "gpt-5.5": "GPT-5.5",
    "gpt-oss-120b": "GPT-OSS 120B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "llama-3.1-8b": "Llama 3.1 8B",
    "llama-3.2-1b": "Llama 3.2 1B",
    "meta-llama-3-8b": "Llama 3 8B Inst.",
    "meta-llama-3.1-70b": "Llama 3.1 70B Inst.",
}

MODEL_GROUPS = {
    "astrollama-3-8b-aic": "Open astro",
    "astrosage-70b-20251009": "Open astro",
    "astrosage-8b": "Open astro",
    "claude-sonnet-4-6": "API general",
    "gemma-3-4b": "Open multimodal",
    "gemma-4-26b-a4b": "Open multimodal",
    "gemma-4-31b": "Open multimodal",
    "gpt-5.5": "API general",
    "gpt-oss-120b": "API general",
    "gpt-oss-20b": "Open general",
    "llama-3.1-8b": "Open general",
    "llama-3.2-1b": "Open general",
    "meta-llama-3-8b": "Open general",
    "meta-llama-3.1-70b": "Open general",
}

GROUP_COLORS = {
    "Open multimodal": "#F58518",
    "Open astro": "#54A24B",
    "API general": "#B279A2",
    "Open general": "#4C78A8",
}

WIDTH = 650
HEIGHT = 420
X_LABEL = 158
X0 = 178
Y0 = 58
CHART_W = 398
ROW_H = 21


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
        default="usaaao_bootstrap_leaderboard",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260806,
        help="Random seed for deterministic confidence intervals.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level. Default: 0.95.",
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


def quantile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolated quantile for sorted finite values."""

    if not sorted_values:
        raise ValueError("Cannot compute quantile of an empty list.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    weight = position - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def load_text_only_scores(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required per-example CSV: {path}")

    scores: Dict[str, List[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model_slug", "has_image", "judge_score"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            if not is_false(row["has_image"]):
                continue
            raw_score = row.get("judge_score", "").strip()
            if raw_score == "":
                continue
            scores[row["model_slug"]].append(float(raw_score))

    if not scores:
        raise ValueError("No text-only judge scores found in all_models_per_example.csv.")
    return dict(scores)


def bootstrap_items(
    scores_by_model: Dict[str, List[float]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples should be at least 100 for stable intervals.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1.")

    rng = random.Random(seed)
    alpha = (1.0 - confidence) / 2.0
    items: List[Dict[str, Any]] = []

    for slug, scores in scores_by_model.items():
        if not scores:
            continue
        n = len(scores)
        observed = sum(scores) / n
        sampled_means = []
        for _ in range(bootstrap_samples):
            total = 0.0
            for _ in range(n):
                total += scores[rng.randrange(n)]
            sampled_means.append(total / n)
        sampled_means.sort()
        items.append(
            {
                "model_slug": slug,
                "model": MODEL_LABELS.get(slug, slug),
                "group": MODEL_GROUPS.get(slug, "Other"),
                "n": n,
                "mean": observed,
                "ci_low": quantile(sampled_means, alpha),
                "ci_high": quantile(sampled_means, 1.0 - alpha),
                "confidence": confidence,
                "bootstrap_samples": bootstrap_samples,
            }
        )

    if not items:
        raise ValueError("No bootstrap items were generated.")
    return sorted(items, key=lambda item: (item["mean"], item["ci_low"]), reverse=True)


def used_groups(items: Iterable[Dict[str, Any]]) -> List[str]:
    groups: List[str] = []
    for item in items:
        group = item["group"]
        if group not in groups:
            groups.append(group)
    return groups


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


def axis_max(items: List[Dict[str, Any]]) -> float:
    high = max(item["ci_high"] for item in items)
    return max(0.10, math.ceil((high + 0.015) / 0.05) * 0.05)


def x_scale(value: float, xmax: float) -> float:
    return X0 + CHART_W * value / xmax


def write_svg(items: List[Dict[str, Any]], path: Path) -> None:
    xmax = axis_max(items)
    axis_y = Y0 + ROW_H * len(items) + 8
    grid_color = "#D8D8D8"
    muted = "#666666"
    ci_color = "#222222"

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(WIDTH / 2, 22, "Bootstrap text-only leaderboard", size=15, weight="700", anchor="middle"),
        svg_text(WIDTH / 2, 39, "Judge score with 95% bootstrap confidence intervals (N=204 examples)", size=10, anchor="middle", fill=muted),
    ]

    tick_count = 4
    for idx in range(tick_count + 1):
        value = xmax * idx / tick_count
        x = x_scale(value, xmax)
        parts.append(f'<line x1="{x:.1f}" y1="{Y0-6}" x2="{x:.1f}" y2="{axis_y:.1f}" stroke="{grid_color}" stroke-width="1"/>')
        parts.append(svg_text(x, axis_y + 17, f"{value:.2f}", size=9, anchor="middle", fill=muted))

    for idx in range(len(items)):
        y = Y0 + idx * ROW_H + 10
        parts.append(f'<line x1="8" y1="{y+7:.1f}" x2="{WIDTH-8}" y2="{y+7:.1f}" stroke="#EEEEEE" stroke-width="0.7"/>')

    for idx, item in enumerate(items):
        y = Y0 + idx * ROW_H + 10
        x_low = x_scale(item["ci_low"], xmax)
        x_high = x_scale(item["ci_high"], xmax)
        x_mean = x_scale(item["mean"], xmax)
        fill = GROUP_COLORS.get(item["group"], "#999999")

        parts.append(svg_text(X_LABEL, y + 4, item["model"], size=9, anchor="end"))
        parts.append(f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="{ci_color}" stroke-opacity="0.52" stroke-width="1.6"/>')
        parts.append(f'<line x1="{x_low:.1f}" y1="{y-4:.1f}" x2="{x_low:.1f}" y2="{y+4:.1f}" stroke="{ci_color}" stroke-opacity="0.52" stroke-width="1.2"/>')
        parts.append(f'<line x1="{x_high:.1f}" y1="{y-4:.1f}" x2="{x_high:.1f}" y2="{y+4:.1f}" stroke="{ci_color}" stroke-opacity="0.52" stroke-width="1.2"/>')
        parts.append(f'<circle cx="{x_mean:.1f}" cy="{y:.1f}" r="4.5" fill="{fill}" stroke="white" stroke-width="0.9"/>')
        parts.append(svg_text(x_high + 7, y + 4, f"{item['mean']:.3f}", size=8, fill="#222222"))

    legend_y = HEIGHT - 12
    legend_x = X0
    cursor = legend_x
    for group in used_groups(items):
        fill = GROUP_COLORS.get(group, "#999999")
        parts.append(f'<circle cx="{cursor+4:.1f}" cy="{legend_y-3:.1f}" r="4" fill="{fill}"/>')
        parts.append(svg_text(cursor + 12, legend_y, group, size=8, fill=muted))
        cursor += 104 if group != "Open multimodal" else 118

    parts.append(f'<line x1="{X0}" y1="{axis_y:.1f}" x2="{X0+CHART_W}" y2="{axis_y:.1f}" stroke="#999999" stroke-width="1"/>')
    parts.append(svg_text(X0 + CHART_W / 2, axis_y + 31, "Judge score", size=10, anchor="middle", fill=muted))
    parts.append("</svg>")
    path.write_text("\n".join(parts))
    print(f"Wrote {path}")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb_fill(hex_color: str, *, stroke: bool = False) -> str:
    hex_color = hex_color.lstrip("#")
    vals = " ".join(f"{int(hex_color[i:i+2], 16) / 255:.4f}" for i in (0, 2, 4))
    return vals + (" RG" if stroke else " rg")


def write_pdf(items: List[Dict[str, Any]], path: Path) -> None:
    """Write a compact vector PDF without requiring matplotlib."""

    objects: List[bytes] = []

    def add(obj: str | bytes) -> int:
        if isinstance(obj, str):
            obj = obj.encode("latin-1")
        objects.append(obj)
        return len(objects)

    def y_coord(svg_y: float) -> float:
        return HEIGHT - svg_y

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
        width = len(value) * size * 0.52
        if right:
            x -= width
        if center:
            x -= width / 2
        commands.append(f"BT {rgb_fill(fill)} /F1 {size} Tf {x:.1f} {y_coord(sy):.1f} Td ({pdf_escape(value)}) Tj ET")

    def line(
        x1: float,
        sy1: float,
        x2: float,
        sy2: float,
        *,
        stroke: str = "#D8D8D8",
        lw: float = 1.0,
    ) -> None:
        commands.append(f"{rgb_fill(stroke, stroke=True)} {lw} w {x1:.1f} {y_coord(sy1):.1f} m {x2:.1f} {y_coord(sy2):.1f} l S")

    def rect(x: float, sy: float, w: float, h: float, fill: str) -> None:
        commands.append(f"{rgb_fill(fill)} {x:.1f} {y_coord(sy+h):.1f} {w:.1f} {h:.1f} re f")

    def circle(cx: float, sy: float, r: float, fill: str) -> None:
        k = 0.5522847498
        c = r * k
        cy = y_coord(sy)
        commands.append(
            f"{rgb_fill(fill)} {cx:.1f} {cy+r:.1f} m "
            f"{cx+c:.1f} {cy+r:.1f} {cx+r:.1f} {cy+c:.1f} {cx+r:.1f} {cy:.1f} c "
            f"{cx+r:.1f} {cy-c:.1f} {cx+c:.1f} {cy-r:.1f} {cx:.1f} {cy-r:.1f} c "
            f"{cx-c:.1f} {cy-r:.1f} {cx-r:.1f} {cy-c:.1f} {cx-r:.1f} {cy:.1f} c "
            f"{cx-r:.1f} {cy+c:.1f} {cx-c:.1f} {cy+r:.1f} {cx:.1f} {cy+r:.1f} c f"
        )

    rect(0, 0, WIDTH, HEIGHT, "#FFFFFF")
    text(WIDTH / 2, 22, "Bootstrap text-only leaderboard", size=15, center=True)
    text(WIDTH / 2, 39, "Judge score with 95% bootstrap confidence intervals (N=204 examples)", size=10, center=True, fill="#666666")

    xmax = axis_max(items)
    axis_y = Y0 + ROW_H * len(items) + 8

    for idx in range(5):
        value = xmax * idx / 4
        x = x_scale(value, xmax)
        line(x, Y0 - 6, x, axis_y)
        text(x, axis_y + 17, f"{value:.2f}", size=9, center=True, fill="#666666")

    for idx in range(len(items)):
        sy = Y0 + idx * ROW_H + 10
        line(8, sy + 7, WIDTH - 8, sy + 7, stroke="#EEEEEE", lw=0.7)

    for idx, item in enumerate(items):
        sy = Y0 + idx * ROW_H + 10
        x_low = x_scale(item["ci_low"], xmax)
        x_high = x_scale(item["ci_high"], xmax)
        x_mean = x_scale(item["mean"], xmax)
        fill = GROUP_COLORS.get(item["group"], "#999999")
        text(X_LABEL, sy + 4, item["model"], size=9, right=True)
        line(x_low, sy, x_high, sy, stroke="#222222", lw=1.4)
        line(x_low, sy - 4, x_low, sy + 4, stroke="#222222", lw=1.0)
        line(x_high, sy - 4, x_high, sy + 4, stroke="#222222", lw=1.0)
        circle(x_mean, sy, 4.5, fill)
        text(x_high + 7, sy + 4, f"{item['mean']:.3f}", size=8)

    legend_y = HEIGHT - 12
    cursor = X0
    for group in used_groups(items):
        fill = GROUP_COLORS.get(group, "#999999")
        circle(cursor + 4, legend_y - 3, 4, fill)
        text(cursor + 12, legend_y, group, size=8, fill="#666666")
        cursor += 104 if group != "Open multimodal" else 118

    line(X0, axis_y, X0 + CHART_W, axis_y, stroke="#999999")
    text(X0 + CHART_W / 2, axis_y + 31, "Judge score", size=10, center=True, fill="#666666")

    content = "\n".join(commands).encode("latin-1")
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
    path.write_bytes(output)
    print(f"Wrote {path}")


def write_png(path: Path, pdf_path: Path) -> None:
    """Write PNG via macOS sips when available."""

    if shutil.which("sips") is None:
        print("Skipping PNG: macOS `sips` is not available and matplotlib is not installed.")
        return
    subprocess.run(["sips", "-s", "format", "png", str(pdf_path), "--out", str(path)], check=True)
    print(f"Wrote {path}")


def main() -> int:
    args = parse_args()
    per_example_path = args.merged_dir / "all_models_per_example.csv"
    scores_by_model = load_text_only_scores(per_example_path)
    items = bootstrap_items(
        scores_by_model,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )

    counts = {item["n"] for item in items}
    if counts != {204}:
        print(f"WARNING: expected every text-only model to have 204 examples; observed counts={sorted(counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "source": str(per_example_path),
        "subset": "text-only",
        "metric": "judge_score",
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
        "items": items,
    }
    stats_path = args.output_dir / f"{args.stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")

    pdf_path = args.output_dir / f"{args.stem}.pdf"
    if "svg" in args.formats:
        write_svg(items, args.output_dir / f"{args.stem}.svg")
    if "pdf" in args.formats or "png" in args.formats:
        write_pdf(items, pdf_path)
    if "png" in args.formats:
        write_png(args.output_dir / f"{args.stem}.png", pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
