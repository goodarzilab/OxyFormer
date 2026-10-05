"""B0 partial-linear association benchmark, not the v2 MTP estimator.

Ridge/preprocessing arithmetic retained from initial phase3. Raw approved X is
mandatory; no global embeddings, disease mediators or sufficiency assertion.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from oxyformer.provenance import require


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
    require(covariates and len(set(covariates)) == len(covariates), "explicit unique confounders required")
    require(not set(covariates) & {"fips", outcome_column, treatment_column}, "protected data role in confounders")
    analysis = dataframe[["fips", outcome_column, treatment_column] + covariates].copy()
    analysis = analysis.dropna(subset=[outcome_column, treatment_column]).reset_index(drop=True)
    require(2 <= fold_count <= len(analysis) // 2, "insufficient rows for benchmark folds")
    require(np.isfinite(analysis[[outcome_column, treatment_column]].to_numpy(dtype=float)).all(), "nonfinite outcome or exposure")
    require(np.isfinite(analysis[covariates].to_numpy(dtype=float)[analysis[covariates].notna()]).all(), "nonfinite confounder")
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
            "benchmark_linear_prediction": baseline + slope * grid,
            "delta_vs_mean_treatment": slope * (grid - float(np.mean(t))),
        }
    )

    fold_metrics = pd.DataFrame(fold_rows)
    return effect_summary, fold_metrics, dose_response


def approved_confounders(columns, registry, endpoint, *, use="nuisance"):
    """No guessed default list. The reviewed registry grants each raw-X channel."""
    require(isinstance(columns, (list, tuple)) and columns, "explicit approved confounders required")
    require(len(set(columns)) == len(columns), "duplicate confounders")
    for name in columns:
        registry.require(name, endpoint, use)
        require(not name.startswith("foundation_embedding_"), "global embeddings are not approved raw confounders")
    return list(columns)


def run_leave_one_state_out(dataframe, *, outcome_column, treatment_column,
                            covariates, registry, endpoint, fold_count=5,
                            alpha_outcome=25.0, alpha_treatment=10.0, seed=20260306):
    """Refit preprocessing and both nuisances after EVERY represented deletion.

    Includes small states and states with missing outcomes. Failed refits are
    retained as failures, never silently omitted from the sensitivity report.
    Utah/Colorado are ordinary members of this complete enumeration.
    """
    approved_confounders(covariates, registry, endpoint)
    require(dataframe.fips.map(lambda s: isinstance(s, str) and len(s) == 5 and s.isdigit()).all(),
            "county FIPS must be five-digit strings")
    state = dataframe.fips.str[:2]
    require(len(set(state)) >= 2, "state deletion requires at least two states")
    rows = []
    for deleted_state in sorted(state.unique()):
        subset = dataframe.loc[state != deleted_state].copy().reset_index(drop=True)
        row = {"deleted_state": deleted_state, "deleted_rows": int((state == deleted_state).sum()),
               "retained_ids": subset.fips.tolist()}
        try:
            effect, _, _ = cross_fitted_partial_linear_dml(
                subset, outcome_column, treatment_column, covariates, fold_count,
                alpha_outcome, alpha_treatment, seed)
            require(math.isfinite(effect["slope"]), "nonfinite deletion estimate")
            row.update(status="pass", effect=effect)
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update(status="fail", reason=str(exc))
        rows.append(row)
    return rows
