from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / 'report'
GENERATED_DIR = REPORT_DIR / 'generated'
OUTPUT_DIR = ROOT / 'outputs' / 'report'
ASSET_DIR = OUTPUT_DIR / 'paper_assets'


def fmt_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    return f"{int(round(float(value))):,}"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    return f"{100.0 * float(value):.{digits}f}\\%"


def fmt_p(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    value = float(value)
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def tex_escape(text: str) -> str:
    escaped = str(text)
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_logo() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [
        REPORT_DIR / 'Arc-logo-slate.jpg',
        OUTPUT_DIR / 'Arc-logo-slate.jpg',
    ]
    for candidate in candidates:
        if candidate.exists():
            target = REPORT_DIR / 'Arc-logo-slate.jpg'
            if candidate != target:
                shutil.copyfile(candidate, target)
            return target
    raise FileNotFoundError('Arc-logo-slate.jpg not found in report/ or outputs/report/.')


def setup_dirs() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def save_primary_effects_figure(primary: pd.DataFrame) -> None:
    labels = [
        'Incidence\nTotal',
        'Incidence\nDirect',
        'Mortality\nTotal',
        'Mortality\nDirect',
    ]
    order = [
        'incidence_total_effect',
        'incidence_direct_effect',
        'mortality_total_effect',
        'mortality_direct_effect',
    ]
    df = primary.set_index('model_name').loc[order].reset_index()
    values = 100.0 * df['effect_q90_minus_q10_pct_of_mean_outcome']
    colors = ['#0f766e', '#14b8a6', '#1d4ed8', '#60a5fa']

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(range(len(values)), values, color=colors, width=0.68)
    ax.axhline(0.0, color='#334155', linewidth=1.0)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel('Effect of q10→q90 hypoxia shift\n(% of mean outcome)')
    ax.set_title('Primary all-cancer effects')
    for idx, (value, pval) in enumerate(zip(values, df['p_value'])):
        va = 'bottom' if value >= 0 else 'top'
        offset = 0.9 if value >= 0 else -0.9
        ax.text(idx, value + offset, f"p={fmt_p(pval)}", ha='center', va=va, fontsize=9)
    ax.set_ylim(-14.0, max(1.6, float(values.max()) + 1.2))
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / 'primary_effects.pdf')
    plt.close(fig)


def save_training_loss_figure(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for column, label, color in [
        ('total_loss', 'Total loss', '#0f766e'),
        ('reconstruction_loss', 'Reconstruction loss', '#1d4ed8'),
        ('auxiliary_loss', 'Auxiliary loss', '#ea580c'),
        ('consistency_loss', 'Consistency loss', '#7c3aed'),
    ]:
        ax.plot(history['epoch'], history[column], label=label, linewidth=2.0, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Transformer pretraining losses')
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / 'foundation_training_losses.pdf')
    plt.close(fig)


def forest_percent_frame(site_df: pd.DataFrame) -> pd.DataFrame:
    df = site_df.copy()
    scale = df['effect_q90_minus_q10_pct_of_mean_outcome'] / df['slope']
    scale = scale.replace([math.inf, -math.inf], float('nan')).fillna(0.0)
    df['effect_pct'] = 100.0 * df['effect_q90_minus_q10_pct_of_mean_outcome']
    df['ci_lower_pct'] = 100.0 * df['ci_lower'] * scale
    df['ci_upper_pct'] = 100.0 * df['ci_upper'] * scale
    return df


def save_site_forest_figure(site_forest: pd.DataFrame, event_type: str, filename: str) -> None:
    df = site_forest[site_forest['event_type'] == event_type].copy()
    df = forest_percent_frame(df)
    sites = list(df.sort_values('site_rank_within_event')['site'].drop_duplicates())
    total = df[df['effect_type'] == 'total_effect'].set_index('site').loc[sites].reset_index()
    direct = df[df['effect_type'] == 'direct_effect'].set_index('site').loc[sites].reset_index()

    y = list(range(len(sites)))
    fig_h = max(4.5, 0.55 * len(sites) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    offset = 0.16

    ax.errorbar(
        total['effect_pct'],
        [value + offset for value in y],
        xerr=[total['effect_pct'] - total['ci_lower_pct'], total['ci_upper_pct'] - total['effect_pct']],
        fmt='o',
        color='#0f766e',
        ecolor='#0f766e',
        elinewidth=1.7,
        capsize=3,
        label='Total effect',
    )
    ax.errorbar(
        direct['effect_pct'],
        [value - offset for value in y],
        xerr=[direct['effect_pct'] - direct['ci_lower_pct'], direct['ci_upper_pct'] - direct['effect_pct']],
        fmt='s',
        color='#ea580c',
        ecolor='#ea580c',
        elinewidth=1.7,
        capsize=3,
        label='Direct effect',
    )

    ax.axvline(0.0, color='#334155', linewidth=1.0)
    ax.set_yticks(y, sites)
    ax.invert_yaxis()
    ax.set_xlabel('Effect of q10→q90 hypoxia shift (% of mean outcome)')
    ax.set_title(f'{event_type} site-level effects')
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(ASSET_DIR / filename)
    plt.close(fig)


def save_dose_response_figure(dose_curves: pd.DataFrame) -> None:
    model_order = [
        ('incidence_total_effect', 'Incidence total', '#0f766e'),
        ('incidence_direct_effect', 'Incidence direct', '#14b8a6'),
        ('mortality_total_effect', 'Mortality total', '#1d4ed8'),
        ('mortality_direct_effect', 'Mortality direct', '#60a5fa'),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for model_name, label, color in model_order:
        df = dose_curves[dose_curves['model_name'] == model_name].copy()
        ax.plot(df['treatment_value'], df['predicted_outcome_do_t'], label=label, linewidth=2.2, color=color)
    ax.set_xlabel('Hypoxia burden')
    ax.set_ylabel('Predicted outcome under do(T=t)')
    ax.set_title('All-cancer dose-response profiles')
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / 'dose_response_all_cancer.pdf')
    plt.close(fig)


def save_schematic_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def box(x: float, y: float, w: float, h: float, title: str, subtitle_lines: list[str], color: str) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                               linewidth=1.2, edgecolor='#0f172a', facecolor=color)
        ax.add_patch(patch)
        ax.text(x + 0.015, y + h - 0.03, title, fontsize=12.2, weight='bold', va='top')
        for idx, line in enumerate(subtitle_lines):
            ax.text(x + 0.015, y + h - 0.075 - 0.036 * idx, line, fontsize=9.4, va='top', color='#334155')

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.7, color='#475569'))

    box(0.05, 0.60, 0.25, 0.14, '1. Exposure proxy', ['Elevation -> pressure -> hypoxia burden'], '#e0f2fe')
    box(0.05, 0.34, 0.25, 0.14, '2. County table', ['Public + local health covariates'], '#e0f2fe')
    box(0.38, 0.60, 0.25, 0.14, '3. Transformer pretraining', ['Tokens, self-attention, masked views'], '#dcfce7')
    box(0.38, 0.34, 0.25, 0.14, '4. Embedding transfer', ['Learn z_i and append it to X_i'], '#dcfce7')
    box(0.71, 0.60, 0.24, 0.14, '5. DML causal head', ['Orthogonal total and direct effects'], '#fef3c7')
    box(0.71, 0.34, 0.24, 0.14, '6. Endpoint reuse', ['All-cancer, site effects, scale-up'], '#fef3c7')

    arrow(0.30, 0.68, 0.38, 0.68)
    arrow(0.30, 0.42, 0.38, 0.42)
    arrow(0.63, 0.68, 0.71, 0.68)
    arrow(0.63, 0.42, 0.71, 0.42)
    arrow(0.505, 0.60, 0.505, 0.50)
    arrow(0.83, 0.60, 0.83, 0.50)

    patch = FancyBboxPatch((0.17, 0.12), 0.66, 0.10, boxstyle='round,pad=0.012,rounding_size=0.02',
                           linewidth=1.1, edgecolor='#94a3b8', facecolor='#f8fafc')
    ax.add_patch(patch)
    ax.text(0.50, 0.185, 'Why this is causal AI', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(0.50, 0.155, 'AI: pretrained representation and endpoint transfer.', ha='center', va='center', fontsize=9.8, color='#334155')
    ax.text(0.50, 0.128, 'Causal: explicit treatment, transferred confounders, orthogonal score.', ha='center', va='center', fontsize=9.8, color='#334155')

    ax.text(0.5, 0.95, 'Technical architecture: transformer pretraining + orthogonal causal estimation', ha='center', fontsize=15, weight='bold')
    ax.text(0.5, 0.915, 'The representation model is transferred into the nuisance stage rather than replacing the causal estimand.', ha='center', fontsize=10, color='#334155')
    fig.tight_layout(pad=0.8)
    fig.savefig(ASSET_DIR / 'technical_architecture.pdf')
    plt.close(fig)


def write_rows_file(path: Path, lines: list[str]) -> None:
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_tables(phase26: dict, primary: pd.DataFrame, site_forest: pd.DataFrame) -> None:
    primary_rows = []
    for _, row in primary.iterrows():
        primary_rows.append(
            ' & '.join([
                tex_escape(str(row['event_type'])),
                tex_escape(str(row['effect_type']).replace('_', ' ')),
                fmt_int(row['n_rows']),
                fmt_int(row['n_covariates']),
                fmt_float(row['slope'], 2),
                f"[{fmt_float(row['ci_lower'], 2)}, {fmt_float(row['ci_upper'], 2)}]",
                fmt_p(row['p_value']),
                fmt_pct(row['effect_q90_minus_q10_pct_of_mean_outcome'], 1),
            ]) + r' \\')
    write_rows_file(GENERATED_DIR / 'primary_results_rows.tex', primary_rows)

    foundation_rows = [
        ' & '.join([
            fmt_int(phase26['input_rows']),
            fmt_int(phase26['input_dim']),
            fmt_int(phase26['embedding_dim']),
            fmt_int(phase26['token_dim']),
            fmt_int(phase26['n_layers']),
            fmt_int(phase26['n_heads']),
            fmt_int(phase26['epochs']),
            fmt_float(phase26['mask_rate'], 2),
            fmt_int(len(phase26['auxiliary_columns'])),
            fmt_float(phase26['final_losses']['total_loss'], 3),
        ]) + r' \\'
    ]
    write_rows_file(GENERATED_DIR / 'foundation_rows.tex', foundation_rows)

    for event_type, filename in [('Incidence', 'incidence_site_rows.tex'), ('Mortality', 'mortality_site_rows.tex')]:
        df = site_forest[(site_forest['event_type'] == event_type) & (site_forest['effect_type'] == 'total_effect')].copy()
        df = forest_percent_frame(df).sort_values('site_rank_within_event')
        lines = []
        for _, row in df.iterrows():
            lines.append(
                ' & '.join([
                    tex_escape(str(row['site'])),
                    fmt_int(row['n_rows']),
                    fmt_float(row['effect_pct'], 1),
                    f"[{fmt_float(row['ci_lower_pct'], 1)}, {fmt_float(row['ci_upper_pct'], 1)}]",
                    fmt_p(row['p_value']),
                ]) + r' \\')
        write_rows_file(GENERATED_DIR / filename, lines)


def build_macros(phase25: dict, phase26: dict, phase3: dict, phase4: dict | None) -> None:
    all_models = pd.DataFrame(phase3['models'])
    all_cancer = all_models[all_models['analysis_group'] == 'all_cancer'].copy().set_index('model_name')
    incidence_total = all_cancer.loc['incidence_total_effect']
    incidence_direct = all_cancer.loc['incidence_direct_effect']
    mortality_total = all_cancer.loc['mortality_total_effect']
    mortality_direct = all_cancer.loc['mortality_direct_effect']

    macros = {
        'CountyRows': fmt_int(phase26['input_rows']),
        'ModelFeatures': fmt_int(phase25['model_feature_count']),
        'ModelMatrixFeatures': fmt_int(phase25['model_matrix_columns']),
        'FoundationInputDim': fmt_int(phase26['input_dim']),
        'FoundationEmbeddingDim': fmt_int(phase26['embedding_dim']),
        'FoundationTokenDim': fmt_int(phase26['token_dim']),
        'FoundationLayers': fmt_int(phase26['n_layers']),
        'FoundationHeads': fmt_int(phase26['n_heads']),
        'FoundationEpochs': fmt_int(phase26['epochs']),
        'FoundationMaskRate': fmt_float(phase26['mask_rate'], 2),
        'FoundationAuxTargets': fmt_int(len(phase26['auxiliary_columns'])),
        'IncidenceN': fmt_int(incidence_total['n_rows']),
        'MortalityN': fmt_int(mortality_total['n_rows']),
        'IncidenceTotalPct': fmt_pct(incidence_total['effect_q90_minus_q10_pct_of_mean_outcome'], 1),
        'IncidenceDirectPct': fmt_pct(incidence_direct['effect_q90_minus_q10_pct_of_mean_outcome'], 1),
        'MortalityTotalPct': fmt_pct(mortality_total['effect_q90_minus_q10_pct_of_mean_outcome'], 1),
        'MortalityDirectPct': fmt_pct(mortality_direct['effect_q90_minus_q10_pct_of_mean_outcome'], 1),
        'IncidencePValue': fmt_p(incidence_total['p_value']),
        'MortalityPValue': fmt_p(mortality_total['p_value']),
        'FoundationLoss': fmt_float(phase26['final_losses']['total_loss'], 3),
        'FoundationReconLoss': fmt_float(phase26['final_losses']['reconstruction_loss'], 3),
        'FoundationAuxLoss': fmt_float(phase26['final_losses']['auxiliary_loss'], 3),
        'FoundationConsistencyLoss': fmt_float(phase26['final_losses']['consistency_loss'], 5),
        'PhaseFourStatesTested': fmt_int(phase4['models'][0]['states_tested']) if phase4 and phase4.get('models') else '--',
        'PhaseFourSeedsTested': fmt_int(len(phase4['seeds'])) if phase4 and phase4.get('seeds') else '--',
    }

    lines = ['% Auto-generated by build_technical_paper_assets.py']
    for name, value in macros.items():
        lines.append(rf'\newcommand{{\{name}}}{{{value}}}')
    (GENERATED_DIR / 'macros.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    setup_dirs()
    resolve_logo()

    phase25 = load_json(ROOT / 'outputs' / 'phase25' / 'phase25_summary.json')
    phase26 = load_json(ROOT / 'outputs' / 'phase26' / 'phase26_summary.json')
    phase3 = load_json(ROOT / 'outputs' / 'phase3' / 'phase3_summary.json')
    phase4_path = ROOT / 'outputs' / 'phase4' / 'phase4_summary.json'
    phase4 = load_json(phase4_path) if phase4_path.exists() else None

    primary = pd.read_csv(ROOT / 'outputs' / 'phase3' / 'causal_effect_estimates.csv')
    primary = primary[primary['analysis_group'] == 'all_cancer'].copy()
    order = ['incidence_total_effect', 'incidence_direct_effect', 'mortality_total_effect', 'mortality_direct_effect']
    primary = primary.set_index('model_name').loc[order].reset_index()

    site_forest = pd.read_csv(ROOT / 'outputs' / 'phase3' / 'site_effect_forest_plot.csv')
    dose_curves = pd.read_csv(ROOT / 'outputs' / 'phase3' / 'dose_response_curves.csv')
    history = pd.read_csv(ROOT / 'outputs' / 'phase26' / 'foundation_training_history.csv')

    build_macros(phase25, phase26, phase3, phase4)
    build_tables(phase26, primary, site_forest)
    save_schematic_figure()
    save_training_loss_figure(history)
    save_primary_effects_figure(primary)
    save_site_forest_figure(site_forest, 'Incidence', 'site_effects_incidence.pdf')
    save_site_forest_figure(site_forest, 'Mortality', 'site_effects_mortality.pdf')
    save_dose_response_figure(dose_curves)


if __name__ == '__main__':
    main()
