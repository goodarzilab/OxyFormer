from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


TREATMENT_COLUMN = "hypoxia_burden"
FOLD_COUNT = 5
RIDGE_ALPHA_OUTCOME = 25.0
RIDGE_ALPHA_TREATMENT = 10.0
RANDOM_SEED = 20260306

BASE_CONFOUNDERS = [
    "median_income_log1p",
    "acs_per_capita_income_log1p",
    "acs_poverty_pct",
    "acs_unemployment_rate",
    "acs_bachelors_plus_pct",
    "acs_less_than_high_school_pct",
    "acs_broadband_pct",
    "acs_owner_occupied_pct",
    "acs_rent_burden_35plus_pct",
    "acs_mean_travel_time_min",
    "acs_median_age_years",
    "acs_under_18_pct",
    "acs_age_65_plus_pct",
    "acs_nonhisp_white_pct",
    "acs_black_pct",
    "acs_hispanic_pct",
    "acs_asian_pct",
    "places_current_smoking_pct",
    "places_lpa_pct",
    "places_no_health_insurance_pct",
    "places_copd_pct",
    "places_poor_physical_health_pct",
    "places_poor_mental_health_pct",
    "svi_rpl_theme1",
    "svi_rpl_theme2",
    "svi_rpl_theme3",
    "svi_rpl_theme4",
    "svi_ep_limeng",
    "svi_ep_noveh",
    "rucc_code_2023",
    "rucc_nonmetro_flag",
]

MEDIATOR_FEATURES = [
    "local_diabetes_pct",
    "local_obesity_pct",
    "places_diabetes_pct",
    "places_obesity_pct",
    "places_high_blood_pressure_pct",
    "places_stroke_pct",
]

BASE_MODEL_SPECS = [
    {
        "model_name": "incidence_total_effect",
        "outcome": "all_cancer_incidence_rate",
        "effect_type": "total_effect",
        "include_mediators": False,
        "analysis_group": "all_cancer",
        "event_type": "Incidence",
        "site": "All Cancer Sites Combined",
        "support_count": np.nan,
        "site_rank_within_event": np.nan,
    },
    {
        "model_name": "incidence_direct_effect",
        "outcome": "all_cancer_incidence_rate",
        "effect_type": "direct_effect",
        "include_mediators": True,
        "analysis_group": "all_cancer",
        "event_type": "Incidence",
        "site": "All Cancer Sites Combined",
        "support_count": np.nan,
        "site_rank_within_event": np.nan,
    },
    {
        "model_name": "mortality_total_effect",
        "outcome": "all_cancer_mortality_rate",
        "effect_type": "total_effect",
        "include_mediators": False,
        "analysis_group": "all_cancer",
        "event_type": "Mortality",
        "site": "All Cancer Sites Combined",
        "support_count": np.nan,
        "site_rank_within_event": np.nan,
    },
    {
        "model_name": "mortality_direct_effect",
        "outcome": "all_cancer_mortality_rate",
        "effect_type": "direct_effect",
        "include_mediators": True,
        "analysis_group": "all_cancer",
        "event_type": "Mortality",
        "site": "All Cancer Sites Combined",
        "support_count": np.nan,
        "site_rank_within_event": np.nan,
    },
]
MODEL_SPECS = BASE_MODEL_SPECS

SITE_SELECTION_RULES = {
    "Incidence": {"min_support": 750, "max_sites": 8},
    "Mortality": {"min_support": 350, "max_sites": 8},
}
SITE_EXCLUSIONS = {
    "All Cancer Sites Combined",
    "Male and Female Breast, <i>in situ</i>",
}


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, vector)]

    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row_index: abs(augmented[row_index][pivot_column]))
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        pivot_value = augmented[pivot_column][pivot_column]
        if abs(pivot_value) < 1e-12:
            raise ValueError("Singular linear system encountered.")

        for column_index in range(pivot_column, size + 1):
            augmented[pivot_column][column_index] /= pivot_value

        for row_index in range(size):
            if row_index == pivot_column:
                continue
            elimination_factor = augmented[row_index][pivot_column]
            if elimination_factor == 0:
                continue
            for column_index in range(pivot_column, size + 1):
                augmented[row_index][column_index] -= elimination_factor * augmented[pivot_column][column_index]

    return [augmented[row_index][size] for row_index in range(size)]


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    inverse_columns = []
    for column_index in range(size):
        basis = [0.0] * size
        basis[column_index] = 1.0
        inverse_columns.append(solve_linear_system([row[:] for row in matrix], basis))
    return [[inverse_columns[column_index][row_index] for column_index in range(size)] for row_index in range(size)]


def normal_p_value_from_z(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def fit_ridge_coefficients(design_matrix: np.ndarray, response: np.ndarray, alpha: float) -> np.ndarray:
    if design_matrix.ndim != 2:
        raise ValueError("design_matrix must be 2D")

    augmented_matrix = np.column_stack([np.ones(design_matrix.shape[0], dtype=float), design_matrix.astype(float)])
    gram = augmented_matrix.T @ augmented_matrix
    rhs = augmented_matrix.T @ response.astype(float)
    penalty = np.eye(gram.shape[0], dtype=float)
    penalty[0, 0] = 0.0
    gram = gram + alpha * penalty
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs
    return coefficients.astype(float)


def predict_ridge(coefficients: np.ndarray, design_matrix: np.ndarray) -> np.ndarray:
    predictions = []
    for row_index in range(design_matrix.shape[0]):
        value = float(coefficients[0])
        for column_index in range(design_matrix.shape[1]):
            value += float(coefficients[column_index + 1]) * float(design_matrix[row_index, column_index])
        predictions.append(value)
    return np.array(predictions, dtype=float)


def prepare_design_frame(dataframe: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    base = dataframe[["fips"] + covariates].copy()
    base["state_fips"] = base["fips"].astype(str).str[:2]
    state_dummies = pd.get_dummies(base["state_fips"], prefix="state", drop_first=True)

    continuous_columns = covariates[:]
    output = pd.concat([base[["fips"] + continuous_columns], state_dummies], axis=1)
    dummy_columns = state_dummies.columns.tolist()
    return output, continuous_columns, dummy_columns


def transform_design(
    design_frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    continuous_columns: list[str],
    dummy_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    train_frame = design_frame.iloc[train_indices]
    test_frame = design_frame.iloc[test_indices]

    train_columns = []
    test_columns = []
    stats = {}
    for column_name in continuous_columns:
        train_values = pd.to_numeric(train_frame[column_name], errors="coerce")
        test_values = pd.to_numeric(test_frame[column_name], errors="coerce")
        median_value = float(train_values.median()) if train_values.notna().any() else 0.0
        train_missing = train_values.isna().astype(float)
        test_missing = test_values.isna().astype(float)
        train_imputed = train_values.fillna(median_value).astype(float)
        test_imputed = test_values.fillna(median_value).astype(float)
        mean_value = float(train_imputed.mean())
        std_value = float(train_imputed.std(ddof=0))
        if std_value == 0.0:
            train_standardized = np.zeros(len(train_imputed), dtype=float)
            test_standardized = np.zeros(len(test_imputed), dtype=float)
        else:
            train_standardized = ((train_imputed - mean_value) / std_value).to_numpy(dtype=float)
            test_standardized = ((test_imputed - mean_value) / std_value).to_numpy(dtype=float)

        train_columns.append(train_standardized)
        train_columns.append(train_missing.to_numpy(dtype=float))
        test_columns.append(test_standardized)
        test_columns.append(test_missing.to_numpy(dtype=float))
        stats[column_name] = {"median": median_value, "mean": mean_value, "std": std_value}

    for column_name in dummy_columns:
        train_columns.append(train_frame[column_name].to_numpy(dtype=float))
        test_columns.append(test_frame[column_name].to_numpy(dtype=float))

    train_matrix = np.column_stack(train_columns) if train_columns else np.empty((len(train_indices), 0), dtype=float)
    test_matrix = np.column_stack(test_columns) if test_columns else np.empty((len(test_indices), 0), dtype=float)
    return train_matrix, test_matrix, stats


def make_folds(row_count: int, fold_count: int, seed: int) -> list[np.ndarray]:
    indices = np.arange(row_count)
    generator = np.random.default_rng(seed)
    generator.shuffle(indices)
    return [np.array(fold, dtype=int) for fold in np.array_split(indices, fold_count)]


def robust_simple_linear(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones(len(x), dtype=float), x.astype(float)])
    xtx = design.T @ design
    xty = design.T @ y.astype(float)
    try:
        coefficients = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(xtx) @ xty

    fitted = design @ coefficients
    residuals = y.astype(float) - fitted
    meat = np.zeros((2, 2), dtype=float)
    for row, residual in zip(design, residuals):
        meat += float(residual ** 2) * np.outer(row, row)

    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    scaling = len(x) / max(len(x) - 2, 1)
    covariance = scaling * (xtx_inv @ meat @ xtx_inv)

    slope = float(coefficients[1])
    slope_se = math.sqrt(max(float(covariance[1, 1]), 0.0))
    z_value = slope / slope_se if slope_se > 0 else float("nan")
    return {
        "intercept": float(coefficients[0]),
        "slope": slope,
        "slope_se": slope_se,
        "z_value": z_value,
        "p_value": normal_p_value_from_z(z_value) if math.isfinite(z_value) else float("nan"),
        "ci_lower": slope - 1.96 * slope_se,
        "ci_upper": slope + 1.96 * slope_se,
        "residual_variance": float(np.mean(residuals ** 2)),
    }


def strip_site_markup(site_name: str) -> str:
    return re.sub(r"<[^>]+>", "", str(site_name))


def sanitize_site_slug(site_name: str) -> str:
    cleaned = strip_site_markup(site_name).lower()
    cleaned = re.sub(r"[^0-9a-z]+", "_", cleaned)
    return cleaned.strip("_")


def outcome_column_for_site(event_type: str, site_name: str) -> str:
    event_slug = event_type.strip().lower()
    return f"{event_slug}_rate__{sanitize_site_slug(site_name)}"


def build_site_feature_table(
    root_dir: Path,
    incidence_support_min: int,
    mortality_support_min: int,
    max_sites_per_event: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_path = root_dir / "outputs" / "phase1" / "site_support.csv"
    cancer_path = root_dir / "outputs" / "phase1" / "cancer_endpoints_long.csv"
    if not support_path.exists():
        raise FileNotFoundError(f"Phase 1 site support table not found at {support_path}")
    if not cancer_path.exists():
        raise FileNotFoundError(f"Phase 1 cancer endpoint table not found at {cancer_path}")

    support = pd.read_csv(support_path)
    cancer = pd.read_csv(cancer_path, dtype={"fips": str})

    support_rules = {
        "Incidence": incidence_support_min,
        "Mortality": mortality_support_min,
    }
    selected_frames = []
    for event_type, min_support in support_rules.items():
        event_support = support[(support["event_type"] == event_type) & (~support["site"].isin(SITE_EXCLUSIONS))].copy()
        event_support = event_support[event_support["counties_with_covariates"] >= min_support]
        event_support = event_support.sort_values(["counties_with_covariates", "site"], ascending=[False, True]).head(max_sites_per_event).reset_index(drop=True)
        if event_support.empty:
            continue
        event_support["site_rank_within_event"] = np.arange(1, len(event_support) + 1)
        event_support["site_clean"] = event_support["site"].map(strip_site_markup)
        event_support["outcome_column"] = event_support.apply(lambda row: outcome_column_for_site(row["event_type"], row["site"]), axis=1)
        selected_frames.append(event_support)

    if not selected_frames:
        empty_table = pd.DataFrame(columns=["fips"])
        empty_manifest = pd.DataFrame(columns=["event_type", "site", "site_clean", "counties_with_covariates", "rows", "site_rank_within_event", "outcome_column"])
        return empty_table, empty_manifest

    selected_support = pd.concat(selected_frames, ignore_index=True)
    selected_pairs = selected_support[["event_type", "site", "outcome_column"]]
    cancer_selected = cancer.merge(selected_pairs, on=["event_type", "site"], how="inner")
    site_wide = cancer_selected.pivot_table(index="fips", columns="outcome_column", values="age_adjusted_rate", aggfunc="first").reset_index()
    site_wide.columns.name = None
    return site_wide, selected_support


def build_site_model_specs(site_manifest: pd.DataFrame) -> list[dict]:
    specs = []
    for _, row in site_manifest.iterrows():
        base_name = f"{row['event_type'].strip().lower()}__{sanitize_site_slug(row['site'])}"
        for effect_type, include_mediators in [("total_effect", False), ("direct_effect", True)]:
            specs.append(
                {
                    "model_name": f"site_{base_name}__{effect_type}",
                    "outcome": row["outcome_column"],
                    "effect_type": effect_type,
                    "include_mediators": include_mediators,
                    "analysis_group": "site",
                    "event_type": row["event_type"],
                    "site": row["site_clean"],
                    "support_count": int(row["counties_with_covariates"]),
                    "site_rank_within_event": int(row["site_rank_within_event"]),
                }
            )
    return specs


def load_foundation_embeddings(root_dir: Path, embedding_path: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    resolved_path = embedding_path if embedding_path is not None else root_dir / "outputs" / "phase26" / "county_foundation_embeddings.csv"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Foundation embedding table not found at {resolved_path}")
    embeddings = pd.read_csv(resolved_path, dtype={"fips": str})
    embedding_columns = [column for column in embeddings.columns if column.startswith("foundation_embedding_")]
    if not embedding_columns:
        raise ValueError(f"No foundation embedding columns were found in {resolved_path}")
    return embeddings[["fips"] + embedding_columns].copy(), embedding_columns


def covariates_for_spec(spec: dict, embedding_columns: list[str] | None = None) -> list[str]:
    covariates = BASE_CONFOUNDERS + (MEDIATOR_FEATURES if spec["include_mediators"] else [])
    if embedding_columns:
        covariates += embedding_columns
    return covariates


def build_site_forest_plot(effect_estimates: pd.DataFrame) -> pd.DataFrame:
    site_rows = effect_estimates[effect_estimates["analysis_group"] == "site"].copy()
    if site_rows.empty:
        return pd.DataFrame(
            columns=[
                "event_type",
                "site",
                "effect_type",
                "model_name",
                "support_count",
                "n_rows",
                "slope",
                "slope_se",
                "ci_lower",
                "ci_upper",
                "p_value",
                "effect_per_sd_treatment",
                "effect_q90_minus_q10",
                "effect_q90_minus_q10_pct_of_mean_outcome",
                "site_rank_within_event",
            ]
        )

    forest = site_rows[
        [
            "event_type",
            "site",
            "effect_type",
            "model_name",
            "support_count",
            "n_rows",
            "slope",
            "slope_se",
            "ci_lower",
            "ci_upper",
            "p_value",
            "effect_per_sd_treatment",
            "effect_q90_minus_q10",
            "effect_q90_minus_q10_pct_of_mean_outcome",
            "site_rank_within_event",
        ]
    ].copy()
    effect_order = {"total_effect": 0, "direct_effect": 1}
    event_order = {"Incidence": 0, "Mortality": 1}
    forest["event_sort"] = forest["event_type"].map(event_order).fillna(99)
    forest["effect_sort"] = forest["effect_type"].map(effect_order).fillna(99)
    forest = forest.sort_values(["event_sort", "site_rank_within_event", "effect_sort", "site"]).drop(columns=["event_sort", "effect_sort"]).reset_index(drop=True)
    return forest


def cross_fitted_partial_linear_dml(
    dataframe: pd.DataFrame,
    outcome_column: str,
    treatment_column: str,
    covariates: list[str],
    fold_count: int,
    alpha_outcome: float,
    alpha_treatment: float,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    analysis = dataframe[["fips", outcome_column, treatment_column] + covariates].copy()
    analysis = analysis.dropna(subset=[outcome_column, treatment_column]).reset_index(drop=True)
    design_frame, continuous_columns, dummy_columns = prepare_design_frame(analysis, covariates)

    y = analysis[outcome_column].astype(float).to_numpy()
    t = analysis[treatment_column].astype(float).to_numpy()
    folds = make_folds(len(analysis), fold_count, seed)
    y_hat = np.zeros(len(analysis), dtype=float)
    t_hat = np.zeros(len(analysis), dtype=float)
    fold_rows = []

    for fold_index, test_indices in enumerate(folds):
        train_mask = np.ones(len(analysis), dtype=bool)
        train_mask[test_indices] = False
        train_indices = np.where(train_mask)[0]
        x_train, x_test, _ = transform_design(design_frame, train_indices, test_indices, continuous_columns, dummy_columns)

        outcome_beta = fit_ridge_coefficients(x_train, y[train_indices], alpha_outcome)
        treatment_beta = fit_ridge_coefficients(x_train, t[train_indices], alpha_treatment)
        y_hat[test_indices] = predict_ridge(outcome_beta, x_test)
        t_hat[test_indices] = predict_ridge(treatment_beta, x_test)

        y_fold = y[test_indices]
        t_fold = t[test_indices]
        y_error = y_fold - y_hat[test_indices]
        t_error = t_fold - t_hat[test_indices]
        outcome_r2 = 1.0 - float(np.sum(y_error ** 2) / np.sum((y_fold - np.mean(y_fold)) ** 2)) if len(y_fold) > 1 and np.sum((y_fold - np.mean(y_fold)) ** 2) > 0 else float("nan")
        treatment_r2 = 1.0 - float(np.sum(t_error ** 2) / np.sum((t_fold - np.mean(t_fold)) ** 2)) if len(t_fold) > 1 and np.sum((t_fold - np.mean(t_fold)) ** 2) > 0 else float("nan")
        fold_rows.append(
            {
                "fold": fold_index,
                "n_test": int(len(test_indices)),
                "outcome_r2": outcome_r2,
                "treatment_r2": treatment_r2,
            }
        )

    y_residual = y - y_hat
    t_residual = t - t_hat
    effect = robust_simple_linear(y_residual, t_residual)
    treatment_sd = float(np.std(t, ddof=0))
    q10 = float(np.quantile(t, 0.10))
    q90 = float(np.quantile(t, 0.90))
    mean_outcome = float(np.mean(y))
    slope = effect["slope"]
    baseline = float(np.mean(y - slope * t))

    effect_summary = {
        "n_rows": int(len(analysis)),
        "n_covariates": int(len(covariates)),
        "n_state_dummies": int(len(dummy_columns)),
        "fold_count": int(fold_count),
        "alpha_outcome": float(alpha_outcome),
        "alpha_treatment": float(alpha_treatment),
        "treatment_mean": float(np.mean(t)),
        "treatment_sd": treatment_sd,
        "treatment_q10": q10,
        "treatment_q90": q90,
        "outcome_mean": mean_outcome,
        "effect_per_unit_treatment": slope,
        "effect_per_sd_treatment": slope * treatment_sd,
        "effect_q90_minus_q10": slope * (q90 - q10),
        "effect_q90_minus_q10_pct_of_mean_outcome": (slope * (q90 - q10) / mean_outcome) if mean_outcome != 0 else float("nan"),
        "outcome_cv_mean_r2": float(np.nanmean([row["outcome_r2"] for row in fold_rows])),
        "treatment_cv_mean_r2": float(np.nanmean([row["treatment_r2"] for row in fold_rows])),
    }
    effect_summary.update(effect)

    grid = np.linspace(float(np.min(t)), float(np.max(t)), 25)
    dose_response = pd.DataFrame(
        {
            "treatment_value": grid,
            "predicted_outcome_do_t": baseline + slope * grid,
            "delta_vs_mean_treatment": slope * (grid - float(np.mean(t))),
        }
    )

    fold_metrics = pd.DataFrame(fold_rows)
    return effect_summary, fold_metrics, dose_response


def run_phase3(
    root_dir: Path,
    include_site_models: bool = False,
    incidence_support_min: int = SITE_SELECTION_RULES["Incidence"]["min_support"],
    mortality_support_min: int = SITE_SELECTION_RULES["Mortality"]["min_support"],
    max_site_models_per_event: int = SITE_SELECTION_RULES["Incidence"]["max_sites"],
    use_foundation_embeddings: bool = False,
    embedding_path: Path | None = None,
) -> dict:
    curated_path = root_dir / "outputs" / "phase25" / "curated_county_features.csv"
    if not curated_path.exists():
        raise FileNotFoundError(f"Curated Phase 2.5 table not found at {curated_path}")

    output_dir = root_dir / "outputs" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    curated = pd.read_csv(curated_path, low_memory=False, dtype={"fips": str})
    embedding_columns: list[str] = []
    resolved_embedding_path: Path | None = None
    if use_foundation_embeddings:
        embedding_frame, embedding_columns = load_foundation_embeddings(root_dir=root_dir, embedding_path=embedding_path)
        resolved_embedding_path = embedding_path if embedding_path is not None else root_dir / "outputs" / "phase26" / "county_foundation_embeddings.csv"
        curated = curated.merge(embedding_frame, on="fips", how="left")
    selected_site_manifest = pd.DataFrame()
    if include_site_models:
        site_table, selected_site_manifest = build_site_feature_table(
            root_dir=root_dir,
            incidence_support_min=incidence_support_min,
            mortality_support_min=mortality_support_min,
            max_sites_per_event=max_site_models_per_event,
        )
        curated = curated.merge(site_table, on="fips", how="left")

    model_specs = BASE_MODEL_SPECS[:]
    if include_site_models and not selected_site_manifest.empty:
        model_specs.extend(build_site_model_specs(selected_site_manifest))

    manifest_rows = []
    effect_rows = []
    fold_frames = []
    dose_frames = []

    for spec in model_specs:
        covariates = covariates_for_spec(spec, embedding_columns=embedding_columns)
        manifest_rows.append(
            {
                "model_name": spec["model_name"],
                "outcome": spec["outcome"],
                "treatment": TREATMENT_COLUMN,
                "effect_type": spec["effect_type"],
                "covariates": json.dumps(covariates),
                "includes_mediators": spec["include_mediators"],
                "uses_foundation_embeddings": bool(embedding_columns),
                "foundation_embedding_count": int(len(embedding_columns)),
                "analysis_group": spec["analysis_group"],
                "event_type": spec["event_type"],
                "site": spec["site"],
                "support_count": spec["support_count"],
                "site_rank_within_event": spec["site_rank_within_event"],
            }
        )
        effect_summary, fold_metrics, dose_response = cross_fitted_partial_linear_dml(
            dataframe=curated,
            outcome_column=spec["outcome"],
            treatment_column=TREATMENT_COLUMN,
            covariates=covariates,
            fold_count=FOLD_COUNT,
            alpha_outcome=RIDGE_ALPHA_OUTCOME,
            alpha_treatment=RIDGE_ALPHA_TREATMENT,
            seed=RANDOM_SEED,
        )
        effect_summary["model_name"] = spec["model_name"]
        effect_summary["outcome"] = spec["outcome"]
        effect_summary["treatment"] = TREATMENT_COLUMN
        effect_summary["effect_type"] = spec["effect_type"]
        effect_summary["analysis_group"] = spec["analysis_group"]
        effect_summary["event_type"] = spec["event_type"]
        effect_summary["site"] = spec["site"]
        effect_summary["support_count"] = spec["support_count"]
        effect_summary["site_rank_within_event"] = spec["site_rank_within_event"]
        effect_rows.append(effect_summary)

        fold_metrics.insert(0, "model_name", spec["model_name"])
        fold_metrics.insert(1, "analysis_group", spec["analysis_group"])
        fold_metrics.insert(2, "event_type", spec["event_type"])
        fold_metrics.insert(3, "site", spec["site"])
        dose_response.insert(0, "model_name", spec["model_name"])
        dose_response.insert(1, "analysis_group", spec["analysis_group"])
        dose_response.insert(2, "event_type", spec["event_type"])
        dose_response.insert(3, "site", spec["site"])
        fold_frames.append(fold_metrics)
        dose_frames.append(dose_response)

    effect_estimates = pd.DataFrame(effect_rows)
    fold_diagnostics = pd.concat(fold_frames, ignore_index=True)
    dose_response = pd.concat(dose_frames, ignore_index=True)
    model_manifest = pd.DataFrame(manifest_rows)
    site_effect_estimates = effect_estimates[effect_estimates["analysis_group"] == "site"].copy()
    site_forest_plot = build_site_forest_plot(effect_estimates)

    effect_estimates.to_csv(output_dir / "causal_effect_estimates.csv", index=False)
    fold_diagnostics.to_csv(output_dir / "fold_diagnostics.csv", index=False)
    dose_response.to_csv(output_dir / "dose_response_curves.csv", index=False)
    model_manifest.to_csv(output_dir / "model_manifest.csv", index=False)
    site_effect_estimates.to_csv(output_dir / "site_causal_effect_estimates.csv", index=False)
    site_forest_plot.to_csv(output_dir / "site_effect_forest_plot.csv", index=False)

    summary = {
        "treatment": TREATMENT_COLUMN,
        "fold_count": FOLD_COUNT,
        "alpha_outcome": RIDGE_ALPHA_OUTCOME,
        "alpha_treatment": RIDGE_ALPHA_TREATMENT,
        "uses_foundation_embeddings": bool(embedding_columns),
        "foundation_embedding_count": int(len(embedding_columns)),
        "foundation_embedding_path": str(resolved_embedding_path) if resolved_embedding_path is not None else None,
        "site_models_included": bool(include_site_models),
        "selected_site_models": selected_site_manifest[
            ["event_type", "site_clean", "counties_with_covariates", "site_rank_within_event", "outcome_column"]
        ].rename(columns={"site_clean": "site"}).to_dict(orient="records") if include_site_models and not selected_site_manifest.empty else [],
        "models": effect_estimates[
            [
                "model_name",
                "analysis_group",
                "event_type",
                "site",
                "effect_type",
                "n_rows",
                "support_count",
                "effect_per_unit_treatment",
                "effect_per_sd_treatment",
                "effect_q90_minus_q10",
                "effect_q90_minus_q10_pct_of_mean_outcome",
                "slope_se",
                "p_value",
                "outcome_cv_mean_r2",
                "treatment_cv_mean_r2",
            ]
        ].to_dict(orient="records"),
        "outputs": {
            "causal_effect_estimates": "outputs/phase3/causal_effect_estimates.csv",
            "fold_diagnostics": "outputs/phase3/fold_diagnostics.csv",
            "dose_response_curves": "outputs/phase3/dose_response_curves.csv",
            "model_manifest": "outputs/phase3/model_manifest.csv",
            "site_causal_effect_estimates": "outputs/phase3/site_causal_effect_estimates.csv",
            "site_effect_forest_plot": "outputs/phase3/site_effect_forest_plot.csv",
            "foundation_embedding_input": str(resolved_embedding_path) if resolved_embedding_path is not None else None,
        },
    }
    (output_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cross-fitted partial-linear DML demo on the curated county feature table.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--include-site-models", action="store_true", help="Also fit supported site-specific cancer outcome models.")
    parser.add_argument("--use-foundation-embeddings", action="store_true", help="Append Phase 2.6 learned county embeddings to the causal covariate set.")
    parser.add_argument("--embedding-path", type=Path, default=None, help="Optional path to a county foundation embedding CSV.")
    parser.add_argument("--site-support-incidence", type=int, default=SITE_SELECTION_RULES["Incidence"]["min_support"], help="Minimum county support required for incidence site models.")
    parser.add_argument("--site-support-mortality", type=int, default=SITE_SELECTION_RULES["Mortality"]["min_support"], help="Minimum county support required for mortality site models.")
    parser.add_argument("--max-site-models-per-event", type=int, default=SITE_SELECTION_RULES["Incidence"]["max_sites"], help="Maximum number of supported sites to model per event type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_phase3(
                root_dir=args.root_dir,
                include_site_models=args.include_site_models,
                incidence_support_min=args.site_support_incidence,
                mortality_support_min=args.site_support_mortality,
                max_site_models_per_event=args.max_site_models_per_event,
                use_foundation_embeddings=args.use_foundation_embeddings,
                embedding_path=args.embedding_path,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
