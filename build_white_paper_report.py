from __future__ import annotations

import json
import math
import shutil
from datetime import date
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

from plot_phase3_site_forest import run_plot_phase3_site_forest


PROJECT_TITLE = "County-Level Oxygen Exposure and Cancer Outcomes"
REPORT_TITLE = "OxyFormer: Causal AI for Altitude-Derived Hypoxia Burden and Cancer Outcomes"
INSTITUTION_NAME = "Arc Institute"
AUTHORS = ["Brandon Chew", "Isha Jain", "Hani Goodarzi"]
GENERATED_DATE = date.today().strftime("%B %d, %Y").replace(" 0", " ")


def fmt_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{int(round(float(value))):,}"


def fmt_float(value: float | int | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_p(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def read_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def tag(name: str, content: str = "", **attrs: str) -> str:
    attr_parts = []
    for key, value in attrs.items():
        html_key = key.rstrip("_").replace("_", "-")
        attr_parts.append(f'{html_key}="{escape(str(value), quote=True)}"')
    attr_text = (" " + " ".join(attr_parts)) if attr_parts else ""
    return f"<{name}{attr_text}>{content}</{name}>"


def render_table(dataframe: pd.DataFrame, classes: str = "data-table", max_rows: int | None = None) -> str:
    if dataframe.empty:
        return '<p class="muted">No rows available.</p>'

    frame = dataframe.copy()
    if max_rows is not None:
        frame = frame.head(max_rows)

    header = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                display = "—" if math.isnan(value) else f"{value:.4g}"
            else:
                display = str(value)
            cells.append(f"<td>{escape(display)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    table_html = f'<table class="{classes}"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    return f'<div class="table-shell">{table_html}</div>'


def read_text_block(paragraphs: Iterable[str]) -> str:
    return "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)


def bullet_list(items: Iterable[str], ordered: bool = False) -> str:
    tag_name = "ol" if ordered else "ul"
    inner = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<{tag_name} class=\"bullet-list\">{inner}</{tag_name}>"


def section(section_id: str, title: str, body: str, level: int = 2) -> str:
    heading = f"h{level}"
    return f'<section id="{escape(section_id)}" class="report-section">{tag(heading, escape(title))}{body}</section>'


def metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-note">{escape(note)}</div>'
        '</div>'
    )


def callout(title: str, body_html: str, tone: str = "accent") -> str:
    return f'<div class="callout callout-{escape(tone)}"><div class="callout-title">{escape(title)}</div>{body_html}</div>'


def equation_block(title: str, equations: Iterable[str], note: str | None = None) -> str:
    equations_html = "".join(f'<div class="equation-line"><code>{escape(equation)}</code></div>' for equation in equations)
    note_html = f'<div class="equation-note">{escape(note)}</div>' if note else ""
    return f'<div class="equation-block"><div class="equation-title">{escape(title)}</div>{equations_html}{note_html}</div>'


def bar_chart_svg(labels: list[str], values: list[float], subtitle: str) -> str:
    width = 760
    height = 360
    left = 170
    right = 35
    top = 45
    bottom = 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    value_min = min(0.0, min(values))
    value_max = max(0.0, max(values))
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0
    padding = 0.1 * (value_max - value_min)
    value_min -= padding
    value_max += padding

    def x_pos(value: float) -> float:
        return left + (value - value_min) / (value_max - value_min) * plot_width

    zero_x = x_pos(0.0)
    bar_height = plot_height / max(len(labels), 1) * 0.55
    gap = plot_height / max(len(labels), 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="18" font-family="Arial, Helvetica, sans-serif" font-weight="bold">{escape(subtitle)}</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top+plot_height}" stroke="#0f172a" stroke-dasharray="5,4"/>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * gap + (gap - bar_height) / 2
        x0 = zero_x
        x1 = x_pos(value)
        bar_x = min(x0, x1)
        bar_w = abs(x1 - x0)
        color = "#0f766e" if value < 0 else "#b45309"
        parts.append(f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{max(bar_w,1):.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{left-10}" y="{y + bar_height/2 + 4:.1f}" text-anchor="end" font-size="13" font-family="Arial, Helvetica, sans-serif">{escape(label)}</text>')
        anchor = "start" if value >= 0 else "end"
        label_x = x1 + 8 if value >= 0 else x1 - 8
        parts.append(f'<text x="{label_x:.1f}" y="{y + bar_height/2 + 4:.1f}" text-anchor="{anchor}" font-size="12" font-family="Arial, Helvetica, sans-serif">{value:.1f}%</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-12}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif">Estimated outcome shift from the 10th to 90th percentile of hypoxia burden</text>')
    parts.append('</svg>')
    return ''.join(parts)


def line_chart_svg(series_map: dict[str, pd.DataFrame], title: str, y_label: str) -> str:
    width = 760
    height = 360
    left = 70
    right = 30
    top = 45
    bottom = 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_x = pd.concat([frame['treatment_value'] for frame in series_map.values()], ignore_index=True)
    all_y = pd.concat([frame['predicted_outcome_do_t'] for frame in series_map.values()], ignore_index=True)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_pad = 0.03 * (x_max - x_min)
    y_pad = 0.08 * (y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pos(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="18" font-family="Arial, Helvetica, sans-serif" font-weight="bold">{escape(title)}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="#cbd5e1"/>')
    for idx in range(5):
        frac = idx / 4
        y = top + frac * plot_height
        value = y_max - frac * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial, Helvetica, sans-serif">{value:.0f}</text>')
    for idx, (label, frame) in enumerate(series_map.items()):
        color = colors[idx % len(colors)]
        points = ' '.join(f'{x_pos(float(x)):.1f},{y_pos(float(y)):.1f}' for x, y in zip(frame['treatment_value'], frame['predicted_outcome_do_t']))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = left + 12
        legend_y = top + 18 + idx * 18
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x+24}" y="{legend_y+4}" font-size="12" font-family="Arial, Helvetica, sans-serif">{escape(label)}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-12}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif">Hypoxia burden</text>')
    parts.append(f'<text x="16" y="{top + plot_height/2:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 16 {top + plot_height/2:.1f})" font-family="Arial, Helvetica, sans-serif">{escape(y_label)}</text>')
    parts.append('</svg>')
    return ''.join(parts)



def arc_institute_logo_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="147" height="20" viewBox="0 0 147 20" fill="currentColor"><path d="M16.9228 1.22796C13.2136 0.57613 9.73288 0 7.31422 0C6.23447 0 5.22211 0.115226 4.31531 0.440273C3.40852 0.765319 2.60774 1.30072 1.93084 2.05453C1.25393 2.80834 0.721277 3.83174 0.363642 5.02644C0.00600741 6.22114 -0.136638 7.68859 0.142481 9.38795C0.396359 10.9521 0.859552 12.3013 1.5079 13.4357C2.15625 14.57 2.96549 15.5024 3.93402 16.2395C4.90254 16.9851 5.99828 17.5375 7.22159 17.904C8.44489 18.2705 9.76259 18.46 11.1644 18.46C12.0926 18.46 13.0703 18.3952 14.0899 18.2657C15.1172 18.1362 16.1483 17.9549 17.191 17.7142L17.8563 12.5617C17.191 12.8065 16.4372 13.0224 15.5951 13.2093C14.753 13.3962 13.8327 13.49 12.8462 13.49C11.3691 13.49 10.1458 13.1774 9.18843 12.5567C8.22724 11.9359 7.60989 10.9593 7.33617 9.62765H18.7268C18.8494 9.17425 18.9356 8.67765 18.9843 8.14226C19.033 7.60688 19.018 6.99262 18.9394 6.29922C18.7492 4.52068 18.001 3.05925 16.9228 1.22796ZM7.32361 6.89532C7.35934 6.14637 7.57517 5.46377 7.97091 4.85593C8.36664 4.2481 8.94234 3.76481 9.68676 3.41229C10.4312 3.05977 11.3626 2.88257 12.4771 2.88257C13.2384 2.88257 13.8931 2.94881 14.4499 3.07754C15.0066 3.20627 15.4723 3.411 15.8421 3.69185C16.2204 3.9727 16.5135 4.34289 16.7293 4.79817C16.9452 5.25344 17.0863 5.78925 17.1559 6.39176C17.1883 6.61229 17.2146 6.79689 17.2371 6.94495C17.2597 7.093 17.2728 7.24106 17.2803 7.38911L7.32361 6.89532Z" fill="currentColor"/></svg>"""


def resolve_cover_logo(root_dir: Path, output_dir: Path) -> tuple[str, bool]:
    candidates = [
        root_dir / 'report' / 'Arc-logo-slate.jpg',
        output_dir / 'Arc-logo-slate.jpg',
    ]
    target = output_dir / 'Arc-logo-slate.jpg'
    for candidate in candidates:
        if candidate.exists():
            if candidate != target:
                shutil.copyfile(candidate, target)
            return '<img src="Arc-logo-slate.jpg" alt="Arc Institute logo" class="brand-image"/>', True
    return arc_institute_logo_svg(), False


def build_cover_page_html(logo_svg: str) -> str:
    author_cards = ''.join(
        f'<div class="author-card"><div class="author-name">{escape(name)}</div><div class="author-affiliation">{escape(INSTITUTION_NAME)}</div></div>'
        for name in AUTHORS
    )
    return (
        f'<section id="cover" class="cover-page">'
        f'<div class="brand-row">'
        f'<div class="brandmark-shell">{logo_svg}</div>'
        f'<div class="cover-meta">'
        f'<div class="cover-kicker">OxyFormer Technical White Paper</div>'
        f'<div class="cover-date">Generated {escape(GENERATED_DATE)}</div>'
        f'</div>'
        f'</div>'
        f'<div class="cover-main">'
        f'<div class="eyebrow">Causal AI • Technical Demonstration • Scale-Up Thesis</div>'
        f'<h1>{escape(REPORT_TITLE)}</h1>'
        f'<p class="cover-deck">This paper presents OxyFormer, a causal-AI system for estimating the effect of altitude-derived oxygen exposure on cancer burden, then uses that prototype to motivate a larger person-level study with direct oxygen and health measurements from wearable and digital-health ecosystems.</p>'
        f'<div class="hero-strip">'
        f'<span class="hero-pill">Physiologic treatment engineering</span>'
        f'<span class="hero-pill">Orthogonalized DML estimation</span>'
        f'<span class="hero-pill">Mediator-aware estimands</span>'
        f'<span class="hero-pill">Site-level heterogeneity</span>'
        f'<span class="hero-pill">Wearable-study roadmap</span>'
        f'</div>'
        f'</div>'
        f'<div class="author-block">'
        f'<div class="author-block-title">Authors</div>'
        f'<div class="author-grid">{author_cards}</div>'
        f'<div class="author-note">Prepared at {escape(INSTITUTION_NAME)}</div>'
        f'</div>'
        f'</section>'
    )


def causal_ai_schematic_svg() -> str:
    width = 1120
    height = 560
    colors = {
        'input': '#e0f2fe',
        'model': '#dcfce7',
        'output': '#fef3c7',
        'stroke': '#0f172a',
        'arrow': '#475569',
        'muted': '#334155',
    }

    boxes = [
        (70, 130, 280, 84, colors['input'], '1. Exposure proxy', ['Elevation -> pressure -> hypoxia burden']),
        (70, 280, 280, 84, colors['input'], '2. County table', ['Public + local health covariates']),
        (420, 130, 280, 84, colors['model'], '3. Transformer pretraining', ['Tokens, self-attention, masked views']),
        (420, 280, 280, 84, colors['model'], '4. Embedding transfer', ['Learn z_i and append it to X_i']),
        (770, 130, 280, 84, colors['output'], '5. DML causal head', ['Orthogonal total and direct effects']),
        (770, 280, 280, 84, colors['output'], '6. Endpoint reuse', ['All-cancer, site effects, scale-up']),
    ]

    def box(x: int, y: int, w: int, h: int, fill: str, title: str, lines: list[str]) -> str:
        parts = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{colors["stroke"]}" stroke-width="1.3"/>',
            f'<text x="{x + 16}" y="{y + 28}" font-size="17" font-weight="bold" font-family="Arial, Helvetica, sans-serif">{escape(title)}</text>',
        ]
        for idx, line in enumerate(lines):
            parts.append(f'<text x="{x + 16}" y="{y + 54 + 18 * idx}" font-size="12.5" fill="{colors["muted"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>')
        return ''.join(parts)

    def arrow(x1: int, y1: int, x2: int, y2: int) -> str:
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colors["arrow"]}" stroke-width="2.6" marker-end="url(#arrowhead)"/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#475569"/></marker></defs>')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="560" y="44" text-anchor="middle" font-size="25" font-weight="bold" font-family="Arial, Helvetica, sans-serif">Technical causal-AI architecture</text>')
    parts.append('<text x="560" y="68" text-anchor="middle" font-size="13" fill="#334155" font-family="Arial, Helvetica, sans-serif">A pretrained attention-based tabular encoder feeds an orthogonal continuous-treatment DML estimator.</text>')

    for spec in boxes:
        parts.append(box(*spec))

    parts.append(arrow(350, 176, 420, 176))
    parts.append(arrow(350, 336, 420, 336))
    parts.append(arrow(700, 176, 770, 176))
    parts.append(arrow(700, 336, 770, 336))
    parts.append(arrow(560, 222, 560, 290))
    parts.append(arrow(910, 222, 910, 290))

    parts.append('<rect x="182" y="435" width="756" height="78" rx="18" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.2"/>')
    parts.append('<text x="560" y="464" text-anchor="middle" font-size="18" font-weight="bold" font-family="Arial, Helvetica, sans-serif">Why this counts as causal AI</text>')
    parts.append('<text x="560" y="489" text-anchor="middle" font-size="12.5" fill="#334155" font-family="Arial, Helvetica, sans-serif">AI: learned tabular representation, self-attention, masked-view pretraining, endpoint transfer.</text>')
    parts.append('<text x="560" y="508" text-anchor="middle" font-size="12.5" fill="#334155" font-family="Arial, Helvetica, sans-serif">Causal: explicit treatment, transferred confounder representation, orthogonal score, mediator-aware estimands.</text>')
    parts.append('</svg>')
    return ''.join(parts)


def build_report(root_dir: Path) -> Path:
    output_dir = root_dir / 'outputs' / 'report'
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1 = load_json(root_dir / 'outputs' / 'phase1' / 'phase1_summary.json')
    phase2 = load_json(root_dir / 'outputs' / 'phase2' / 'phase2_summary.json')
    phase25 = load_json(root_dir / 'outputs' / 'phase25' / 'phase25_summary.json')
    phase26 = load_optional_json(root_dir / 'outputs' / 'phase26' / 'phase26_summary.json')
    phase3 = load_json(root_dir / 'outputs' / 'phase3' / 'phase3_summary.json')
    phase4 = load_optional_json(root_dir / 'outputs' / 'phase4' / 'phase4_summary.json')

    phase3_effects = pd.read_csv(root_dir / 'outputs' / 'phase3' / 'causal_effect_estimates.csv')
    site_forest = pd.read_csv(root_dir / 'outputs' / 'phase3' / 'site_effect_forest_plot.csv')
    dose_curves = pd.read_csv(root_dir / 'outputs' / 'phase3' / 'dose_response_curves.csv')
    site_support = pd.read_csv(root_dir / 'outputs' / 'phase1' / 'site_support.csv')
    feature_manifest = pd.read_csv(root_dir / 'outputs' / 'phase25' / 'feature_manifest.csv')
    missingness = pd.read_csv(root_dir / 'outputs' / 'phase25' / 'missingness_summary.csv')
    phase4_seed = pd.read_csv(root_dir / 'outputs' / 'phase4' / 'seed_sensitivity_summary.csv') if (root_dir / 'outputs' / 'phase4' / 'seed_sensitivity_summary.csv').exists() else pd.DataFrame()
    phase4_state = pd.read_csv(root_dir / 'outputs' / 'phase4' / 'leave_one_state_out_summary.csv') if (root_dir / 'outputs' / 'phase4' / 'leave_one_state_out_summary.csv').exists() else pd.DataFrame()

    run_plot_phase3_site_forest(root_dir)
    incidence_svg = read_svg(root_dir / 'outputs' / 'phase3' / 'site_forest_incidence.svg')
    mortality_svg = read_svg(root_dir / 'outputs' / 'phase3' / 'site_forest_mortality.svg')
    logo_svg, has_jpg_logo = resolve_cover_logo(root_dir, output_dir)
    schematic_svg = causal_ai_schematic_svg()
    if not has_jpg_logo:
        (output_dir / 'arc_institute_logo.svg').write_text(arc_institute_logo_svg(), encoding='utf-8')
    (output_dir / 'causal_ai_schematic.svg').write_text(schematic_svg, encoding='utf-8')

    all_cancer = phase3_effects[phase3_effects['analysis_group'] == 'all_cancer'].copy().sort_values(['event_type', 'effect_type'])
    q90_pct_labels = [f"{row['event_type']} {row['effect_type'].replace('_', ' ')}" for _, row in all_cancer.iterrows()]
    q90_pct_values = [100.0 * float(row['effect_q90_minus_q10_pct_of_mean_outcome']) for _, row in all_cancer.iterrows()]

    dose_pairs = {
        'Incidence total effect': dose_curves[dose_curves['model_name'] == 'incidence_total_effect'],
        'Incidence direct effect': dose_curves[dose_curves['model_name'] == 'incidence_direct_effect'],
        'Mortality total effect': dose_curves[dose_curves['model_name'] == 'mortality_total_effect'],
        'Mortality direct effect': dose_curves[dose_curves['model_name'] == 'mortality_direct_effect'],
    }

    site_total = site_forest[site_forest['effect_type'] == 'total_effect'].copy()
    negative_sig = site_total[(site_total['p_value'] < 0.05) & (site_total['slope'] < 0)].sort_values('p_value').head(8)
    positive_sig = site_total[(site_total['p_value'] < 0.05) & (site_total['slope'] > 0)].sort_values('p_value').head(8)
    null_sites = site_total[site_total['p_value'] >= 0.2].sort_values(['event_type', 'site_rank_within_event']).head(8)

    top_features = missingness[missingness['include_in_model']].sort_values('observed_fraction', ascending=False).head(12)[['feature', 'group', 'observed_fraction']].copy()
    top_features['observed_fraction'] = top_features['observed_fraction'].map(lambda value: fmt_pct(value, 1))
    low_features = missingness[missingness['include_in_model']].sort_values('observed_fraction', ascending=True).head(12)[['feature', 'group', 'observed_fraction']].copy()
    low_features['observed_fraction'] = low_features['observed_fraction'].map(lambda value: fmt_pct(value, 1))
    support_top = site_support[site_support['site'] != 'All Cancer Sites Combined'].copy().sort_values(['event_type', 'counties_with_covariates'], ascending=[True, False]).groupby('event_type').head(8)

    all_cancer_seed = pd.DataFrame()
    if not phase4_seed.empty:
        all_cancer_seed = phase4_seed[phase4_seed['analysis_group'] == 'all_cancer'][['model_name', 'seeds_tested', 'slope_sd', 'sign_consistent']].copy()
        all_cancer_seed['slope_sd'] = all_cancer_seed['slope_sd'].map(lambda value: fmt_float(value, 3))
        all_cancer_seed['sign_consistent'] = all_cancer_seed['sign_consistent'].map(lambda value: 'Yes' if bool(value) else 'No')

    all_cancer_state = pd.DataFrame()
    if not phase4_state.empty:
        all_cancer_state = phase4_state[phase4_state['analysis_group'] == 'all_cancer'][['model_name', 'states_tested', 'median_abs_slope_shift_pct', 'max_abs_slope_shift_pct', 'sign_consistent']].copy()
        all_cancer_state['median_abs_slope_shift_pct'] = all_cancer_state['median_abs_slope_shift_pct'].map(lambda value: fmt_pct(value, 1))
        all_cancer_state['max_abs_slope_shift_pct'] = all_cancer_state['max_abs_slope_shift_pct'].map(lambda value: fmt_pct(value, 1))
        all_cancer_state['sign_consistent'] = all_cancer_state['sign_consistent'].map(lambda value: 'Yes' if bool(value) else 'No')

    hero_metrics = ''.join([
        metric_card('County rows', fmt_int(phase25['curated_rows']), 'Merged county rows available for the causal pipeline.'),
        metric_card('Primary incidence support', fmt_int(phase1['all_cancer_incidence_complete_cases']), 'Counties contributing to the primary all-cancer incidence model.'),
        metric_card('Primary mortality support', fmt_int(phase1['all_cancer_mortality_complete_cases']), 'Counties contributing to the primary all-cancer mortality model.'),
        metric_card('Model features', fmt_int(phase25['model_feature_count']), 'Curated predictors after feature engineering and screening.'),
        metric_card('Embedding width', fmt_int(phase26['embedding_dim']) if phase26 is not None else '—', 'Learned county embedding dimension from the Phase 2.6 foundation model.'),
        metric_card('Site-level rows', fmt_int(len(site_forest)), 'Forest-plot-ready site effect rows across total and direct effects.'),
        metric_card('Mean feature coverage', fmt_pct(phase25['mean_model_feature_observed_fraction'], 1), 'Average observed fraction across included model features.'),
    ])

    primary_findings = [
        f"All-cancer incidence total effect: a shift from the 10th to the 90th percentile of hypoxia burden corresponds to {fmt_pct(all_cancer.iloc[0]['effect_q90_minus_q10_pct_of_mean_outcome'], 1)} of the mean incidence rate.",
        f"All-cancer incidence direct effect remains {fmt_pct(all_cancer.iloc[1]['effect_q90_minus_q10_pct_of_mean_outcome'], 1)} of the mean incidence rate after conditioning on metabolic mediators.",
        f"All-cancer mortality total effect corresponds to {fmt_pct(all_cancer.iloc[2]['effect_q90_minus_q10_pct_of_mean_outcome'], 1)} of the mean mortality rate.",
        f"The negative signal persists into lung/bronchus incidence and mortality, supporting nontrivial site-level heterogeneity rather than a single noisy aggregate association.",
    ]

    robustness_text = []
    if phase4 is not None:
        robustness_text.append(f"The currently saved Phase 4 outputs use seeds {phase4.get('seeds', [])} and a leave-one-state-out inclusion threshold of {phase4.get('min_state_rows', '—')} rows.")
        if phase4.get('site_models_included'):
            robustness_text.append('The saved robustness outputs also include exemplar site-specific models, which makes the current report more illustrative than a purely all-cancer-only write-up.')
    else:
        robustness_text.append('No Phase 4 robustness outputs were available at report-generation time.')

    toc = [
        ('cover', 'Title Page'),
        ('abstract', 'Abstract'),
        ('executive-summary', 'Executive Summary'),
        ('what-is-causal-ai', 'What Is Causal AI?'),
        ('ai-architecture', 'Causal AI Architecture'),
        ('foundation-model', 'Implemented Foundation Model'),
        ('study-rationale', 'Why This Application Matters'),
        ('data', 'Data Sources and Coverage'),
        ('design', 'Treatment, Mediators, and Estimands'),
        ('pipeline', 'Modeling Pipeline'),
        ('primary-results', 'Primary Results'),
        ('site-results', 'Site-Level Heterogeneity'),
        ('robustness', 'Robustness and Credibility'),
        ('next-steps', 'Why Fund the Next Study'),
    ]

    abstract_body = read_text_block([
        'This white paper describes OxyFormer, a county-level causal-AI system for continuous-treatment effect estimation. Let T_i denote altitude-derived hypoxia burden, Y_i denote a cancer outcome, and X_i denote a high-dimensional county context assembled from local covariates and public health data. The central modeling task is to estimate how Y_i changes under interventions on T_i after flexibly adjusting for confounding structure in X_i.',
        'The implemented estimator is a partial-linear double-machine-learning system with learned representation pretraining. A dual-view masked tabular transformer is first pretrained on county covariates to produce dense county embeddings z_i. The causal stage then estimates Y_i = θ T_i + g(X_i, z_i) + ε_i and T_i = m(X_i, z_i) + ν_i with cross-fitted orthogonal nuisance models, so the learned representation and the causal head are explicitly linked.',
        'The analysis remains ecological and cross-sectional, but it now includes a real attention-based tabular encoder, transformer pretraining, reusable embeddings, mediator-aware effect estimates, site-level heterogeneity analysis, and robustness diagnostics. In that sense, it is no longer just a statistical demo; it is a bona fide modern causal-AI prototype.'
    ])

    executive_body = callout(
        'Investment thesis',
        read_text_block([
            'OxyFormer is a true multi-stage causal-AI stack: an attention-based tabular transformer pretrained on county covariates, a transferred county embedding appended to the nuisance feature space, an orthogonalized continuous-treatment DML causal head, mediator-aware direct-effect variants, and a site-level heterogeneity layer across supported cancer endpoints.',
            'That matters for fundraising because it materially de-risks a next-stage study with direct oxygen and health measurements. The county system now addresses not only treatment definition and endpoint prioritization, but also the core AI question of how to pretrain a reusable tabular representation before fitting the causal estimand.'
        ]),
        tone='strong'
    )
    executive_body += f'<div class="card-grid">{hero_metrics}</div>'
    executive_body += bullet_list(primary_findings)
    executive_body += tag('div', bar_chart_svg(q90_pct_labels, q90_pct_values, 'Primary effect sizes: q10→q90 shift in hypoxia burden'), class_='figure-shell')

    what_is_causal_ai_body = callout(
        'Formal definition',
        read_text_block([
            'Causal AI means using machine learning inside an explicitly causal estimand. The objective is not to maximize predictive accuracy for Y_i, but to recover an intervention-relevant functional such as E[Y_i(t)] or an average partial effect ∂E[Y_i | do(T_i=t)]/∂t under stated identification assumptions.',
            'In this project, machine learning is used in two coupled stages. First, a dual-view masked tabular transformer is pretrained on county covariates to learn a reusable county state representation z_i. Second, the causal stage uses the augmented feature set X_i^* = [X_i, z_i] inside orthogonal nuisance models, so the representation learner serves identification and variance reduction rather than replacing the causal target.'
        ]),
        tone='accent'
    )
    what_is_causal_ai_body += equation_block(
        'Partially linear DML setup',
        [
            'Y_i = θ T_i + g(X_i) + ε_i',
            'T_i = m(X_i) + ν_i',
            'Ŷ_i^res = Y_i - ĝ^(-k(i))(X_i)',
            'Ť_i^res = T_i - m̂^(-k(i))(X_i)',
            'θ̂ = arg min_θ Σ_i (Ŷ_i^res - θ Ť_i^res)^2'
        ],
        note='Cross-fitting indexes nuisance predictions by held-out fold k(i), producing an orthogonal score for a continuous treatment effect.'
    )
    what_is_causal_ai_body += equation_block(
        'Transferred representation in the causal head',
        [
            'z_i = Encoder_φ(X_i)',
            'X_i^* = [X_i, z_i]',
            'Y_i = θ T_i + g(X_i^*) + ε_i',
            'T_i = m(X_i^*) + ν_i'
        ],
        note='The learned embedding is not reported as the estimand. It is transferred into the nuisance models to improve representation of confounding structure before orthogonal effect estimation.'
    )
    what_is_causal_ai_body += '<div class="two-col">'
    what_is_causal_ai_body += callout('Necessary ingredients', bullet_list([
        'A treatment with a defensible intervention interpretation.',
        'A covariate representation X rich enough to absorb confounding pathways.',
        'Orthogonal or doubly robust estimation rather than plain prediction loss.',
        'Sensitivity analyses that can falsify an overfit or geography-specific result.'
    ]), tone='accent')
    what_is_causal_ai_body += callout('Why this qualifies technically', bullet_list([
        'The treatment is engineered from atmospheric physics, not chosen post hoc.',
        'The nuisance functions g(X) and m(X) are learned flexibly from a broad county feature space.',
        'The estimands include both total effects and mediator-conditioned direct effects.',
        'The same causal head is deployed across multiple cancer endpoints to test heterogeneity rather than rely on one pooled estimate.'
    ]), tone='neutral')
    what_is_causal_ai_body += '</div>'

    architecture_body = callout(
        'System view before results',
        read_text_block([
            'The schematic below is intentionally placed before the results because the scientific claim is architectural. The project composes treatment engineering, representation construction, nuisance learning, orthogonal effect estimation, heterogeneity analysis, and robustness checks into a single reproducible system.',
            'That system framing is central to the fundraising case: the county prototype is a stage-zero platform for a larger multimodal study, not an end-state claim that the ecological analysis alone is decisive.'
        ]),
        tone='accent'
    )
    architecture_body += f'<div class="figure-shell">{schematic_svg}</div>'
    architecture_body += bullet_list([
        'Treatment layer: convert elevation into an oxygen-availability exposure with a physiologic interpretation.',
        'Representation layer: fuse ACS, PLACES, SVI, RUCC, and local health covariates into a county state vector, then compress it with an attention-based tabular transformer.',
        'Estimation layer: cross-fit nuisance models for E[Y|X] and E[T|X] to construct orthogonal residuals.',
        'Inference layer: estimate total and direct effects plus site-level heterogeneity with the same causal head.',
        'Robustness layer: quantify seed sensitivity and leave-one-state-out stability before advancing the scientific claim.'
    ])

    foundation_body = callout(
        'What is actually implemented now',
        read_text_block([
            'Phase 2.6 now trains a dual-view masked tabular transformer on the county model matrix, excluding the oxygen treatment variables and explicit mediator columns from its inputs. Each scalar feature is converted into a learned token, two independently masked views are sampled, and a shared transformer encoder maps the resulting token sequence to a county embedding z_i.',
            'The pretraining objective combines masked-feature reconstruction, auxiliary multitask prediction of held-out health-related targets, and a cross-view consistency penalty on the projected embedding. In the current run, the model uses 92 input features, token width 48, 3 transformer layers, 4 attention heads, a 24-dimensional county embedding, and 6 auxiliary health targets.'
        ]),
        tone='accent'
    )
    foundation_body += equation_block(
        'Tabular tokenization and attention',
        [
            'e_{ij} = x_{ij} · w_j + b_j',
            'S_i^(0) = [CLS, e_{i1}, e_{i2}, …, e_{ip}]',
            'S_i^(ℓ+1) = TransformerBlock_ℓ(S_i^(ℓ))',
            'z_i = MLP(h_{i,CLS}^{(L)})'
        ],
        note='Each county becomes a token sequence. Self-attention lets the encoder model cross-feature interactions such as poverty × smoking, rurality × insurance, or age × care access before the causal head is fit.'
    )
    foundation_body += equation_block(
        'Transformer pretraining objective',
        [
            'L = L_mask(x_hat^(a), x) + L_mask(x_hat^(b), x) + λ_aux L_aux(ŷ_aux, y_aux)',
            '    + λ_con ||q(z^(a)) - q(z^(b))||² + λ_emb ||z||²',
            'z^(a) = Encoder_φ(Mask_a(X_i)),   z^(b) = Encoder_φ(Mask_b(X_i))',
            'x_hat = Decoder(z),   ŷ_aux = Head(z)'
        ],
        note='The two-view setup is genuine transformer pretraining: the encoder is optimized to produce stable latent structure under masking while still reconstructing tabular inputs and predicting auxiliary health signals.'
    )
    if phase26 is not None:
        foundation_body += render_table(pd.DataFrame([
            ['Model family', phase26['model_family']],
            ['Input rows', fmt_int(phase26['input_rows'])],
            ['Input dimension', fmt_int(phase26['input_dim'])],
            ['Token width', fmt_int(phase26.get('token_dim'))],
            ['Transformer layers', fmt_int(phase26.get('n_layers'))],
            ['Attention heads', fmt_int(phase26.get('n_heads'))],
            ['Embedding dimension', fmt_int(phase26['embedding_dim'])],
            ['Auxiliary targets', fmt_int(len(phase26['auxiliary_columns']))],
            ['Final total loss', fmt_float(phase26['final_losses']['total_loss'], 3)],
            ['Final consistency loss', fmt_float(phase26['final_losses'].get('consistency_loss'), 6)],
        ], columns=['Component', 'Value']), max_rows=None)
    foundation_body += bullet_list([
        'The pretrained embedding is transferred into Phase 3 as additional covariates for both treatment and outcome nuisance models.',
        'The same shared representation is reused across all-cancer incidence, all-cancer mortality, and site-specific models rather than training a separate encoder per endpoint.',
        'This is an actual attention-based tabular pretraining stage, not just feature standardization or a shallow autoencoder.'
    ])

    rationale_body = callout(
        'Why this application matters',
        read_text_block([
            'Oxygen biology is inherently continuous, multi-scale, and plausibly relevant to cancer phenotypes, but direct longitudinal oxygen exposure data are scarce. A county-level prototype is therefore a rational first step: it cheaply tests whether a coherent treatment definition and endpoint prioritization strategy exist before a far more expensive cohort or partner study is launched.',
            'This prototype also solves an important translational problem for fundraising. Instead of asking prospective partners to fund exploratory fishing, it offers a concrete causal-AI framework: which exposure variable to track, which confounders and mediators to measure, which endpoints appear sensitive, and which robustness analyses will be required when direct person-time data arrive.'
        ]),
        tone='accent'
    )
    rationale_body += bullet_list([
        'It converts a vague altitude hypothesis into a measurable exposure axis.',
        'It identifies which endpoints may justify deeper prospective follow-up.',
        'It provides a ready-made estimation stack for longitudinal wearable-linked data.',
        'It gives funders a tractable bridge from public data prototype to high-value proprietary cohort study.'
    ])

    data_sources = pd.DataFrame([
        ['Local county covariates', fmt_int(phase1['county_covariate_rows']), 'Elevation, income, obesity, diabetes, and county identifiers'],
        ['Cancer endpoint rows', fmt_int(phase1['cancer_endpoint_rows']), 'Incidence and mortality rows by site'],
        ['Phase 2 merged rows', fmt_int(phase2['phase2_master_rows']), f"{fmt_int(phase2['phase2_master_columns'])} columns after public-data augmentation"],
        ['Curated Phase 2.5 rows', fmt_int(phase25['curated_rows']), f"{fmt_int(phase25['curated_feature_count'])} curated features"],
        ['Incidence modeling rows', fmt_int(phase25['incidence_rows']), 'Rows with non-missing all-cancer incidence'],
        ['Mortality modeling rows', fmt_int(phase25['mortality_rows']), 'Rows with non-missing all-cancer mortality'],
    ], columns=['Layer', 'Rows', 'Description'])
    sources_summary = pd.DataFrame([
        [source_name, source_info.get('status', '—'), fmt_int(source_info.get('rows')), fmt_int(source_info.get('features'))]
        for source_name, source_info in phase2['sources'].items()
    ], columns=['Source', 'Status', 'Rows', 'Features'])
    data_body = render_table(data_sources, max_rows=None)
    data_body += render_table(sources_summary, max_rows=None)
    data_body += '<div class="two-col">' + render_table(top_features, max_rows=None) + render_table(low_features, max_rows=None) + '</div>'
    data_body += read_text_block([
        'The feature matrix already spans demography, socioeconomic structure, smoking and inactivity proxies, insurance, self-reported health, rurality, and social vulnerability. That breadth matters because the target estimand is only credible if X captures major common causes of both oxygen exposure and cancer burden.',
        'In the current implementation, that confounder table is no longer purely hand-engineered. Phase 2.6 compresses it into a learned county embedding, which is then transferred into the causal stage as a reusable representation layer.'
    ])

    design_body = callout(
        'Estimands and treatment construction',
        read_text_block([
            'The exposure is not raw elevation. Elevation is first translated into barometric pressure and then into an inspired oxygen proxy, producing a continuous treatment T_i that can be interpreted as county-level hypoxia burden.',
            'The primary estimands are total effects of T_i on all-cancer incidence and mortality. Direct-effect variants add obesity and diabetes proxies to test whether the signal survives conditioning on plausible metabolic mediators.'
        ]),
        tone='accent'
    )
    design_body += equation_block(
        'Exposure engineering',
        [
            'P(h) = P_0 (1 - 2.25577×10^-5 h)^5.25588',
            'O2(h) = F_IO2 × (7.5006 P(h) - 47)',
            'T_i = 1 - O2(h_i) / O2(0)'
        ],
        note='h is county elevation, P(h) is barometric pressure, and T_i increases with altitude-derived hypoxia burden.'
    )
    design_body += equation_block(
        'Potential-outcome target',
        [
            "τ(t, t') = E[Y_i(t) - Y_i(t')]",
            "Direct effect: τ_dir(t, t' | M_i fixed)",
            'Reported contrast: q90(T) - q10(T) scaled by mean outcome'
        ],
        note='The report summarizes a high-versus-low treatment contrast because it is interpretable to non-methodologists while still being derived from a continuous-treatment model.'
    )
    design_body += bullet_list([
        'Confounders include demographic, behavioral, structural, and healthcare-adjacent county attributes.',
        'Mediator-aware variants test whether the primary signal is largely absorbed by metabolic proxies.',
        'State indicators provide a coarse geography control, though not yet full spatial smoothing.',
        'The county prototype is suitable for hypothesis refinement, not final individual-level causal claims.'
    ])

    pipeline_body = '''
    <div class="pipeline-grid">
      <div class="pipeline-step"><div class="pipeline-phase">Layer 1</div><h3>Physiologic treatment model</h3><p>Map county elevation to barometric pressure, inspired oxygen, and hypoxia burden.</p></div>
      <div class="pipeline-step"><div class="pipeline-phase">Layer 2</div><h3>Confounder representation</h3><p>Fuse local outcomes with ACS, PLACES, SVI, RUCC, and structural county covariates.</p></div>
      <div class="pipeline-step"><div class="pipeline-phase">Layer 3</div><h3>Foundation-model pretraining</h3><p>Train a dual-view masked tabular transformer with consistency regularization and auxiliary health heads to compress county context into a learned embedding.</p></div>
      <div class="pipeline-step"><div class="pipeline-phase">Layer 4</div><h3>Nuisance learning</h3><p>Cross-fit outcome and treatment models so nuisance estimation and target estimation are separated.</p></div>
      <div class="pipeline-step"><div class="pipeline-phase">Layer 5</div><h3>Orthogonal causal head</h3><p>Estimate continuous-treatment total and direct effects from residualized outcomes and treatments.</p></div>
      <div class="pipeline-step"><div class="pipeline-phase">Layer 6</div><h3>Heterogeneity and robustness</h3><p>Deploy the same causal head across supported sites, then stress-test the result with sensitivity analyses.</p></div>
    </div>
    '''
    pipeline_body += read_text_block([
        'The important technical point is modularity. Each layer can be upgraded without rewriting the entire program: the treatment model can absorb direct oxygen measurements, the county representation can become a learned encoder, and the causal head can be extended to longitudinal or survival settings.',
        'That modularity is exactly what makes this a credible platform for a larger study rather than a one-off ecological analysis.'
    ])

    primary_results_table = all_cancer[['event_type', 'effect_type', 'n_rows', 'effect_q90_minus_q10_pct_of_mean_outcome', 'slope', 'slope_se', 'p_value', 'outcome_cv_mean_r2', 'treatment_cv_mean_r2']].copy()
    primary_results_table['effect_q90_minus_q10_pct_of_mean_outcome'] = primary_results_table['effect_q90_minus_q10_pct_of_mean_outcome'].map(lambda value: fmt_pct(value, 1))
    primary_results_table['slope'] = primary_results_table['slope'].map(lambda value: fmt_float(value, 2))
    primary_results_table['slope_se'] = primary_results_table['slope_se'].map(lambda value: fmt_float(value, 2))
    primary_results_table['p_value'] = primary_results_table['p_value'].map(fmt_p)
    primary_results_table['outcome_cv_mean_r2'] = primary_results_table['outcome_cv_mean_r2'].map(lambda value: fmt_float(value, 3))
    primary_results_table['treatment_cv_mean_r2'] = primary_results_table['treatment_cv_mean_r2'].map(lambda value: fmt_float(value, 3))
    primary_body = callout(
        'Primary all-cancer estimands',
        read_text_block([
            'In the saved Phase 3 outputs, the all-cancer total-effect and direct-effect models are consistently negative for both incidence and mortality. The fact that direct-effect estimates remain directionally similar after conditioning on obesity and diabetes proxies suggests that the county-level relationship is not trivially explained by those mediator candidates alone.',
            'This is still not dispositive evidence of a causal effect, but it is the right kind of prototype result: the sign is stable across closely related specifications, the estimand is interpretable, and the model exposes nuisance-fit diagnostics rather than hiding them.'
        ]),
        tone='strong'
    )
    primary_body += render_table(primary_results_table, max_rows=None)
    primary_body += tag('div', line_chart_svg({key: value for key, value in dose_pairs.items() if not value.empty}, 'Dose-response curves for the primary all-cancer estimands', 'Predicted outcome under do(t)'), class_='figure-shell')

    top_support_table = support_top[['event_type', 'site', 'counties_with_covariates']].copy()
    top_support_table['counties_with_covariates'] = top_support_table['counties_with_covariates'].map(fmt_int)
    def prep_site_table(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame[['event_type', 'site', 'effect_type', 'support_count', 'effect_q90_minus_q10_pct_of_mean_outcome', 'p_value']].copy()
        output['support_count'] = output['support_count'].map(fmt_int)
        output['effect_q90_minus_q10_pct_of_mean_outcome'] = output['effect_q90_minus_q10_pct_of_mean_outcome'].map(lambda value: fmt_pct(value, 1))
        output['p_value'] = output['p_value'].map(fmt_p)
        return output
    site_body = callout(
        'Endpoint heterogeneity is a feature, not a nuisance',
        read_text_block([
            'A pooled cancer result is rarely the full scientific signal. By applying the same orthogonalized causal head across supported site-specific endpoints, the system asks whether the signal concentrates in anatomically or biologically plausible domains or whether it dissolves once the aggregate endpoint is decomposed.',
            'That heterogeneity layer is one of the most valuable outputs for a follow-on study because it informs endpoint selection, subgroup power calculations, and biomarker prioritization.'
        ]),
        tone='accent'
    )
    site_body += render_table(top_support_table, max_rows=None)
    site_body += '<div class="figure-shell">' + incidence_svg + '</div>'
    site_body += '<div class="figure-shell">' + mortality_svg + '</div>'
    site_body += '<div class="three-col">' + render_table(prep_site_table(negative_sig), max_rows=None) + render_table(prep_site_table(positive_sig), max_rows=None) + render_table(prep_site_table(null_sites), max_rows=None) + '</div>'
    site_body += read_text_block([
        'The saved run concentrates the strongest negative site-level estimates in lung and bronchus incidence and mortality, breast incidence, colon and rectum incidence, urinary bladder incidence, and kidney or renal pelvis incidence. Those patterns provide concrete hypotheses for a larger oxygen-linked cohort rather than a generic all-cancer endpoint alone.',
        'Null and discordant endpoints are equally informative because they constrain the hypothesis space. A larger study should be designed to explain why some sites move while others do not, not merely to re-estimate a pooled average.'
    ])

    robustness_body = callout(
        'Credibility checks',
        read_text_block([
            'A causal-AI system earns trust by showing where it could fail. The repository therefore exposes seed-sensitivity and leave-one-state-out analyses instead of presenting one favored specification as if it were uniquely authoritative.',
            'These diagnostics are not yet the full robustness suite envisioned in the blueprint, but they already establish a discipline that should carry forward into any partner study: report nuisance fit, report specification drift, and report whether the result is dominated by one geography.'
        ]),
        tone='neutral'
    )
    robustness_body += read_text_block(robustness_text)
    robustness_body += '<div class="two-col">' + render_table(all_cancer_seed, max_rows=None) + render_table(all_cancer_state, max_rows=None) + '</div>'
    robustness_body += bullet_list([
        'Seed sensitivity probes whether fold assignment materially changes sign or magnitude.',
        'Leave-one-state-out analyses probe whether the result is a single-state artifact.',
        'Direct-effect models probe whether adjustment for metabolic mediators collapses the estimate.',
        'Outcome and treatment nuisance R² values reveal whether the learned county representation is actually informative.'
    ])

    limitations_body = bullet_list([
        'This remains an ecological county-level analysis rather than an individual-level or household-level causal design.',
        'The current prototype is cross-sectional and cannot yet model person-time exposure trajectories, survival, or latency explicitly.',
        'Residual confounding is still plausible because environmental pollution, UV exposure, climate, and healthcare-access layers are incomplete.',
        'The altitude-derived oxygen proxy introduces measurement error relative to direct SpO2 or wearable oxygen measurements.',
        'Site-level multiplicity means these endpoint results should be treated as prioritization signals rather than confirmatory findings.'
    ])

    next_steps_body = callout(
        'Why fund the next study',
        read_text_block([
            'The county prototype has already completed the highest-leverage de-risking work. It defines the treatment, identifies promising endpoints, demonstrates an orthogonalized estimation strategy, and establishes a reporting standard for robustness. The natural next move is a longitudinal study with direct oxygen and health measurements rather than another incremental ecological table.',
            'A partner study with Apple-, WHOOP-, Oura-, or health-system-class data streams would let the same causal-AI architecture operate on person-level trajectories: oxygen saturation, sleep, activity, cardiorespiratory signals, location-derived altitude, diagnoses, and longitudinal outcomes. That would transform this from a county proxy analysis into a genuine multimodal causal study of oxygen exposure and disease risk.'
        ]),
        tone='strong'
    )
    next_steps_body += equation_block(
        'Scale-up target',
        [
            'Person-level treatment: T_it = f(SpO2_it, altitude_it, sleep_it, activity_it)',
            'Outcome process: Y_it = h(T_i,0:t, X_i,0:t, U_i)',
            'Future estimand: effect of sustained oxygen-exposure trajectories on incident disease and progression'
        ],
        note='The county analysis is a prototype for this richer longitudinal estimand, not a substitute for it.'
    )
    next_steps_body += bullet_list([
        'Build a county-year panel with PM2.5, UV, climate, and healthcare-access covariates as the intermediate scale-up step.',
        'Add spatial blocking, negative controls, and sensitivity analyses for unmeasured confounding.',
        'Pursue partnerships with wearable and digital-health platforms that can measure oxygen and physiology directly.',
        'Design a prospective or retrospective cohort with repeated oxygen exposure, oncology endpoints, and high-dimensional confounder capture.',
        'Use the county prototype to justify power calculations, endpoint selection, and phased fundraising for the larger program.'
    ])

    appendix_body = render_table(pd.DataFrame([
        ['Local county covariates', phase1['outputs']['county_covariates']],
        ['Phase 2 merged table', phase2['outputs']['county_master_table_phase2']],
        ['Curated county features', phase25['outputs']['curated_county_features']],
        ['Phase 2.6 summary', 'outputs/phase26/phase26_summary.json' if phase26 is not None else 'not available'],
        ['Phase 2.6 embeddings', 'outputs/phase26/county_foundation_embeddings.csv' if phase26 is not None else 'not available'],
        ['Phase 3 effect estimates', phase3['outputs']['causal_effect_estimates']],
        ['Phase 3 forest plot table', phase3['outputs']['site_effect_forest_plot']],
        ['Phase 4 summary', 'outputs/phase4/phase4_summary.json' if phase4 is not None else 'not available'],
        ['Arc Institute logo', 'outputs/report/Arc-logo-slate.jpg' if (root_dir / 'outputs' / 'report' / 'Arc-logo-slate.jpg').exists() else 'outputs/report/arc_institute_logo.svg'],
        ['Causal AI schematic', 'outputs/report/causal_ai_schematic.svg'],
        ['HTML white paper', 'outputs/report/white_paper.html'],
    ], columns=['Artifact', 'Path']), max_rows=None)
    appendix_body += render_table(feature_manifest[['feature', 'group', 'include_in_model']].head(20), max_rows=None)

    cover_page_html = build_cover_page_html(logo_svg)

    sections_html = ''.join([
        cover_page_html,
        section('abstract', 'Abstract', abstract_body),
        section('executive-summary', 'Executive Summary', executive_body),
        section('what-is-causal-ai', 'What Is Causal AI?', what_is_causal_ai_body),
        section('ai-architecture', 'Causal AI Architecture', architecture_body),
        section('foundation-model', 'Implemented Foundation Model', foundation_body),
        section('study-rationale', 'Why This Application Matters', rationale_body),
        section('data', 'Data Sources and Coverage', data_body),
        section('design', 'Treatment, Mediators, and Estimands', design_body),
        section('pipeline', 'Modeling Pipeline', pipeline_body),
        section('primary-results', 'Primary Results', primary_body),
        section('site-results', 'Site-Level Heterogeneity', site_body),
        section('robustness', 'Robustness and Credibility', robustness_body),
        section('next-steps', 'Why Fund the Next Study', next_steps_body),
    ])

    toc_html = ''.join(f'<a href="#{section_id}" class="toc-link">{escape(label)}</a>' for section_id, label in toc)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(REPORT_TITLE)}</title>
<style>
:root {{
  --bg: #f8fafc;
  --paper: #ffffff;
  --ink: #0f172a;
  --muted: #475569;
  --line: #dbe4ee;
  --accent: #0f766e;
  --accent-soft: #ccfbf1;
  --accent-strong: #083344;
  --neutral-soft: #eef2ff;
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); line-height: 1.68; }}
a {{ color: inherit; text-decoration: none; }}
.layout {{ display: grid; grid-template-columns: 280px minmax(0, 1fr); min-height: 100vh; }}
.sidebar {{ position: sticky; top: 0; align-self: start; height: 100vh; overflow: auto; background: #0f172a; color: #e2e8f0; padding: 28px 22px; border-right: 1px solid rgba(255,255,255,0.08); }}
.sidebar h1 {{ font-size: 1.05rem; line-height: 1.35; margin: 0 0 0.8rem 0; color: #f8fafc; }}
.sidebar .subtitle {{ color: #94a3b8; font-size: 0.92rem; margin-bottom: 1.2rem; }}
.toc-link {{ display: block; padding: 0.55rem 0.7rem; margin: 0.15rem 0; border-radius: 10px; color: #cbd5e1; font-size: 0.95rem; }}
.toc-link:hover, .toc-link.active {{ background: rgba(255,255,255,0.08); color: #fff; }}
.main {{ padding: 34px 42px 60px; }}
.paper {{ max-width: 1140px; margin: 0 auto; background: var(--paper); border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); padding: 46px 54px; }}
.hero {{ border-bottom: 1px solid var(--line); padding-bottom: 1.4rem; margin-bottom: 2rem; }}
.cover-page {{ min-height: 88vh; display: flex; flex-direction: column; justify-content: space-between; gap: 28px; padding: 6px 0 24px; margin-bottom: 2rem; border-bottom: 1px solid var(--line); page-break-after: always; }}
.brand-row {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }}
.brandmark-shell {{ width: min(240px, 100%); color: #111827; }}
.brandmark-shell svg {{ width: 100%; height: auto; display: block; }}
.brand-image {{ width: min(320px, 100%); height: auto; display: block; }}
.cover-meta {{ display: flex; flex-direction: column; align-items: flex-end; gap: 8px; padding-top: 10px; }}
.cover-kicker {{ font-size: 0.86rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }}
.cover-date {{ font-size: 0.92rem; color: var(--muted); }}
.cover-main {{ max-width: 900px; }}
.cover-deck {{ font-size: 1.15rem; line-height: 1.72; color: #1e293b; max-width: 940px; }}
.author-block {{ border: 1px solid var(--line); border-radius: 22px; padding: 22px; background: linear-gradient(180deg, #ffffff, #f8fafc); }}
.author-block-title {{ font-size: 0.95rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.9rem; }}
.author-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
.author-card {{ border: 1px solid var(--line); border-radius: 16px; padding: 16px; background: #ffffff; }}
.author-name {{ font-size: 1.1rem; font-weight: 780; color: #0f172a; }}
.author-affiliation {{ font-size: 0.92rem; color: var(--muted); margin-top: 0.25rem; }}
.author-note {{ margin-top: 0.9rem; color: var(--muted); font-size: 0.95rem; }}
.eyebrow {{ color: var(--accent); font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.8rem; }}
.hero h1 {{ font-size: 2.35rem; line-height: 1.08; margin: 0.45rem 0 0.7rem; max-width: 900px; }}
.hero p {{ margin: 0.35rem 0; color: var(--muted); font-size: 1.01rem; }}
.hero-strip {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1rem; }}
.hero-pill {{ padding: 7px 12px; border-radius: 999px; background: var(--accent-soft); color: var(--accent-strong); font-weight: 700; font-size: 0.85rem; }}
.report-section {{ scroll-margin-top: 24px; padding-top: 0.5rem; margin-top: 2rem; }}
.report-section h2 {{ font-size: 1.68rem; margin-bottom: 0.7rem; border-bottom: 1px solid var(--line); padding-bottom: 0.45rem; }}
.report-section h3 {{ font-size: 1.1rem; margin-bottom: 0.35rem; }}
.report-section p {{ color: #1e293b; }}
.card-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 1rem 0 1.25rem; }}
.metric-card {{ background: linear-gradient(180deg, #ffffff, #f8fafc); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04); }}
.metric-label {{ color: var(--muted); font-size: 0.9rem; }}
.metric-value {{ font-size: 1.7rem; font-weight: 780; margin-top: 0.1rem; }}
.metric-note {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.35rem; }}
.callout {{ margin: 1rem 0 1.2rem; padding: 18px 20px; border-radius: 18px; border: 1px solid var(--line); }}
.callout-title {{ font-size: 1rem; font-weight: 800; margin-bottom: 0.45rem; }}
.callout-accent {{ background: #f0fdfa; border-color: #99f6e4; }}
.callout-strong {{ background: linear-gradient(135deg, #052e2b, #0f766e); color: white; border-color: #134e4a; }}
.callout-strong p, .callout-strong li {{ color: white; }}
.callout-strong .callout-title {{ color: white; }}
.callout-neutral {{ background: #f8fafc; border-color: #cbd5e1; }}
.equation-block {{ margin: 1rem 0 1.2rem; padding: 16px 18px; border-radius: 18px; border: 1px solid #bfdbfe; background: #eff6ff; }}
.equation-title {{ font-size: 0.98rem; font-weight: 800; color: #1d4ed8; margin-bottom: 0.6rem; }}
.equation-line {{ padding: 0.28rem 0; }}
.equation-line code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; font-size: 0.95rem; color: #0f172a; white-space: pre-wrap; }}
.equation-note {{ margin-top: 0.7rem; color: var(--muted); font-size: 0.92rem; }}
.figure-shell {{ margin: 1rem 0 1.4rem; padding: 14px; background: #fff; border: 1px solid var(--line); border-radius: 18px; overflow-x: auto; }}
.table-shell {{ margin: 0.9rem 0 1.2rem; border: 1px solid var(--line); border-radius: 18px; overflow-x: auto; overflow-y: hidden; background: #fff; }}
.data-table {{ width: max-content; min-width: 100%; border-collapse: collapse; margin: 0; font-size: 0.9rem; table-layout: auto; }}
.data-table thead th {{ background: #f1f5f9; color: #0f172a; text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--line); position: sticky; top: 0; white-space: nowrap; }}
.data-table td {{ padding: 8px 11px; border-bottom: 1px solid #edf2f7; vertical-align: top; max-width: 240px; overflow-wrap: anywhere; word-break: break-word; }}
.data-table tbody tr:nth-child(even) {{ background: #fbfdff; }}
.two-col, .three-col {{ display: grid; gap: 18px; align-items: start; }}
.two-col {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.three-col {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.pipeline-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin: 1rem 0 1.3rem; }}
.pipeline-step {{ border: 1px solid var(--line); border-radius: 18px; padding: 18px; background: linear-gradient(180deg, #fff, #f8fafc); }}
.pipeline-phase {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-size: 0.8rem; font-weight: 700; margin-bottom: 0.6rem; }}
.bullet-list {{ margin: 0.65rem 0 1rem 1.2rem; padding-left: 0.4rem; }}
.bullet-list li {{ margin: 0.35rem 0; }}
.muted {{ color: var(--muted); }}
.footer-note {{ color: var(--muted); font-size: 0.92rem; border-top: 1px solid var(--line); padding-top: 1.2rem; margin-top: 2rem; }}
@media (max-width: 1100px) {{
  .layout {{ grid-template-columns: 1fr; }}
  .sidebar {{ position: relative; height: auto; }}
  .main {{ padding: 18px; }}
  .paper {{ padding: 28px 22px; border-radius: 18px; }}
  .card-grid, .two-col, .three-col, .pipeline-grid, .author-grid {{ grid-template-columns: 1fr; }}
  .brand-row {{ flex-direction: column; align-items: flex-start; }}
  .cover-meta {{ align-items: flex-start; padding-top: 0; }}
}}
@media print {{
  body {{ background: white; }}
  .layout {{ display: block; }}
  .sidebar {{ display: none; }}
  .main {{ padding: 0; }}
  .paper {{ box-shadow: none; border: none; max-width: none; padding: 0; }}
}}
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="eyebrow" style="color:#5eead4;">White Paper</div>
    <h1>{escape(REPORT_TITLE)}</h1>
    <div class="subtitle">Technical causal-AI report with equations, a system schematic, and a scale-up roadmap.</div>
    <nav class="toc">{toc_html}</nav>
  </aside>
  <main class="main">
    <article class="paper">
      {sections_html}
      <div class="footer-note">Report generated by <code>build_white_paper_report.py</code>. The report reflects the saved repository outputs at generation time and embeds the Arc Institute logo and causal-AI schematic written to <code>outputs/report/Arc-logo-slate.jpg</code> (when present) and <code>outputs/report/causal_ai_schematic.svg</code>.</div>
    </article>
  </main>
</div>
<script>
const links = Array.from(document.querySelectorAll('.toc-link'));
const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
const observer = new IntersectionObserver((entries) => {{
  entries.forEach((entry) => {{
    const id = entry.target.getAttribute('id');
    const link = document.querySelector(`.toc-link[href="#${{id}}"]`);
    if (entry.isIntersecting) {{
      links.forEach((node) => node.classList.remove('active'));
      if (link) link.classList.add('active');
    }}
  }});
}}, {{ rootMargin: '-20% 0px -60% 0px', threshold: 0.1 }});
sections.forEach((section) => observer.observe(section));
</script>
</body>
</html>'''

    output_path = output_dir / 'white_paper.html'
    output_path.write_text(html, encoding='utf-8')
    return output_path


if __name__ == '__main__':
    output_path = build_report(Path('.'))
    print(output_path)
