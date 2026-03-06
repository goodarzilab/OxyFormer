from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from html import escape
from pathlib import Path


EFFECT_COLORS = {
    "total_effect": "#1f77b4",
    "direct_effect": "#d62728",
}


def load_forest_rows(root_dir: Path, input_path: Path | None) -> list[dict]:
    path = input_path if input_path is not None else root_dir / "outputs" / "phase3" / "site_effect_forest_plot.csv"
    if not path.exists():
        raise FileNotFoundError(f"Site forest-plot table not found at {path}")

    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = row.copy()
            for key in ["support_count", "n_rows", "slope", "slope_se", "ci_lower", "ci_upper", "p_value", "effect_per_sd_treatment", "effect_q90_minus_q10", "effect_q90_minus_q10_pct_of_mean_outcome", "site_rank_within_event"]:
                value = parsed.get(key, "")
                parsed[key] = float(value) if value not in ("", None) else math.nan
            rows.append(parsed)
    return rows


def group_rows_by_event(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["event_type"])].append(row)
    return dict(grouped)


def build_svg(event_type: str, rows: list[dict]) -> str:
    if not rows:
        raise ValueError(f"No rows available for event type {event_type}")

    labels = []
    for row in sorted(rows, key=lambda item: (item["site_rank_within_event"], item["site"])):
        label = f"{row['site']} (n={int(row['n_rows'])})"
        if label not in labels:
            labels.append(label)

    label_to_index = {label: index for index, label in enumerate(labels)}
    effect_offsets = {"total_effect": -8.0, "direct_effect": 8.0}

    x_min = min(row["ci_lower"] for row in rows)
    x_max = max(row["ci_upper"] for row in rows)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    padding = 0.08 * (x_max - x_min)
    x_min -= padding
    x_max += padding

    left_margin = 330
    right_margin = 60
    top_margin = 60
    bottom_margin = 40
    row_height = 34
    plot_width = 540
    plot_height = max(180, row_height * len(labels))
    width = left_margin + plot_width + right_margin
    height = top_margin + plot_height + bottom_margin

    def x_to_px(value: float) -> float:
        return left_margin + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(label: str, effect_type: str) -> float:
        base = top_margin + (label_to_index[label] + 0.5) * row_height
        return base + effect_offsets.get(effect_type, 0.0)

    ticks = []
    tick_count = 6
    for index in range(tick_count + 1):
        value = x_min + (x_max - x_min) * index / tick_count
        ticks.append((value, x_to_px(value)))

    zero_x = x_to_px(0.0)

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="bold">{escape(event_type)} site-specific oxygen effect estimates</text>')

    for label in labels:
        y_center = top_margin + (label_to_index[label] + 0.5) * row_height
        lines.append(f'<line x1="{left_margin}" y1="{y_center:.1f}" x2="{left_margin + plot_width}" y2="{y_center:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        lines.append(f'<text x="{left_margin - 10}" y="{y_center + 5:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="13">{escape(label)}</text>')

    lines.append(f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin + plot_width}" y2="{top_margin}" stroke="#333333" stroke-width="1"/>')
    lines.append(f'<line x1="{left_margin}" y1="{top_margin + plot_height}" x2="{left_margin + plot_width}" y2="{top_margin + plot_height}" stroke="#333333" stroke-width="1"/>')
    lines.append(f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + plot_height}" stroke="#333333" stroke-width="1"/>')
    lines.append(f'<line x1="{left_margin + plot_width}" y1="{top_margin}" x2="{left_margin + plot_width}" y2="{top_margin + plot_height}" stroke="#333333" stroke-width="1"/>')
    lines.append(f'<line x1="{zero_x:.1f}" y1="{top_margin}" x2="{zero_x:.1f}" y2="{top_margin + plot_height}" stroke="#111111" stroke-width="1.2" stroke-dasharray="6,4"/>')

    for tick_value, tick_x in ticks:
        lines.append(f'<line x1="{tick_x:.1f}" y1="{top_margin + plot_height}" x2="{tick_x:.1f}" y2="{top_margin + plot_height + 6}" stroke="#333333" stroke-width="1"/>')
        lines.append(f'<text x="{tick_x:.1f}" y="{top_margin + plot_height + 22}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12">{tick_value:.1f}</text>')

    for row in rows:
        label = f"{row['site']} (n={int(row['n_rows'])})"
        y = y_to_px(label, str(row["effect_type"]))
        color = EFFECT_COLORS.get(str(row["effect_type"]), "#444444")
        x_left = x_to_px(float(row["ci_lower"]))
        x_right = x_to_px(float(row["ci_upper"]))
        x_mid = x_to_px(float(row["slope"]))
        lines.append(f'<line x1="{x_left:.1f}" y1="{y:.1f}" x2="{x_right:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{x_left:.1f}" y1="{y - 4:.1f}" x2="{x_left:.1f}" y2="{y + 4:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{x_right:.1f}" y1="{y - 4:.1f}" x2="{x_right:.1f}" y2="{y + 4:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{x_mid:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')

    legend_x = left_margin + plot_width - 140
    legend_y = 42
    for idx, effect_type in enumerate(["total_effect", "direct_effect"]):
        y = legend_y + idx * 20
        color = EFFECT_COLORS[effect_type]
        label = effect_type.replace("_", " ").title()
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 20}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{legend_x + 10}" cy="{y}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial, Helvetica, sans-serif" font-size="12">{escape(label)}</text>')

    lines.append(f'<text x="{left_margin + plot_width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13">Estimated slope for hypoxia burden</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def run_plot_phase3_site_forest(root_dir: Path, input_path: Path | None = None) -> dict:
    rows = load_forest_rows(root_dir, input_path)
    grouped = group_rows_by_event(rows)
    output_dir = root_dir / "outputs" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}
    for event_type, event_rows in grouped.items():
        output_path = output_dir / f"site_forest_{event_type.strip().lower()}.svg"
        output_path.write_text(build_svg(event_type, event_rows), encoding="utf-8")
        output_paths[event_type] = str(output_path)

    return {
        "input_rows": int(len(rows)),
        "event_types": sorted(grouped.keys()),
        "outputs": output_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render forest plots from Phase 3 site-specific effect outputs.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--input-path", type=Path, default=None, help="Optional path to a site forest-plot CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run_plot_phase3_site_forest(args.root_dir, args.input_path))


if __name__ == "__main__":
    main()
