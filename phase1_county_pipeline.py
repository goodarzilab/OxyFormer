from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEA_LEVEL_PRESSURE_KPA = 101.325
KPA_TO_MMHG = 7.50061683
FIO2 = 0.2095
WATER_VAPOR_PRESSURE_MMHG = 47.0


def extract_fips(area_value: str) -> str | None:
    match = re.search(r"(\d{5})", str(area_value))
    return match.group(1) if match else None


def standard_atmosphere_pressure_kpa(elevation_m: pd.Series) -> pd.Series:
    return SEA_LEVEL_PRESSURE_KPA * np.power(1 - 2.25577e-5 * elevation_m.astype(float), 5.25588)


def inspired_oxygen_proxy_mmhg(elevation_m: pd.Series) -> pd.Series:
    pressure_mmhg = standard_atmosphere_pressure_kpa(elevation_m) * KPA_TO_MMHG
    return FIO2 * (pressure_mmhg - WATER_VAPOR_PRESSURE_MMHG)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, vector)]

    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row_index: abs(augmented[row_index][pivot_column]))
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        pivot_value = augmented[pivot_column][pivot_column]

        if abs(pivot_value) < 1e-12:
            raise ValueError("Singular linear system encountered during Poisson fitting.")

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
    inverse_columns: list[list[float]] = []

    for column_index in range(size):
        basis_vector = [0.0] * size
        basis_vector[column_index] = 1.0
        inverse_columns.append(solve_linear_system([row[:] for row in matrix], basis_vector))

    return [[inverse_columns[column_index][row_index] for column_index in range(size)] for row_index in range(size)]


def normal_p_value_from_z(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def fit_poisson_newton(
    design_matrix: list[list[float]],
    response: list[float],
    max_iter: int = 100,
    tolerance: float = 1e-8,
    ridge: float = 1e-8,
) -> tuple[list[float], list[list[float]], dict[str, float]]:
    row_count = len(response)
    column_count = len(design_matrix[0])
    coefficients = [0.0] * column_count
    coefficients[0] = math.log(max(sum(response) / row_count, 1e-8))

    for _ in range(max_iter):
        linear_predictor = [sum(design_matrix[row_index][column_index] * coefficients[column_index] for column_index in range(column_count)) for row_index in range(row_count)]
        mean_response = [math.exp(min(value, 50.0)) for value in linear_predictor]
        score = [0.0] * column_count
        information = [[0.0] * column_count for _ in range(column_count)]

        for row_index in range(row_count):
            residual = response[row_index] - mean_response[row_index]
            row_values = design_matrix[row_index]

            for column_index in range(column_count):
                score[column_index] += row_values[column_index] * residual

            for left_index in range(column_count):
                left_value = row_values[left_index]
                for right_index in range(column_count):
                    information[left_index][right_index] += left_value * row_values[right_index] * mean_response[row_index]

        for diagonal_index in range(column_count):
            information[diagonal_index][diagonal_index] += ridge

        step = solve_linear_system(information, score)
        updated_coefficients = [coefficients[column_index] + step[column_index] for column_index in range(column_count)]
        max_change = max(abs(updated_coefficients[column_index] - coefficients[column_index]) for column_index in range(column_count))
        coefficients = updated_coefficients

        if max_change < tolerance:
            break

    linear_predictor = [sum(design_matrix[row_index][column_index] * coefficients[column_index] for column_index in range(column_count)) for row_index in range(row_count)]
    mean_response = [math.exp(min(value, 50.0)) for value in linear_predictor]
    information = [[0.0] * column_count for _ in range(column_count)]

    for row_index in range(row_count):
        row_values = design_matrix[row_index]
        for left_index in range(column_count):
            left_value = row_values[left_index]
            for right_index in range(column_count):
                information[left_index][right_index] += left_value * row_values[right_index] * mean_response[row_index]

    for diagonal_index in range(column_count):
        information[diagonal_index][diagonal_index] += ridge

    covariance = invert_matrix(information)
    log_likelihood = 0.0
    for observed_value, expected_value in zip(response, mean_response):
        if observed_value > 0:
            log_likelihood += observed_value * math.log(expected_value) - expected_value - math.lgamma(observed_value + 1.0)
        else:
            log_likelihood += -expected_value

    deviance = 0.0
    for observed_value, expected_value in zip(response, mean_response):
        if observed_value == 0:
            deviance += 2.0 * expected_value
        else:
            deviance += 2.0 * (observed_value * math.log(observed_value / expected_value) - (observed_value - expected_value))

    pearson_chi2 = sum(((observed_value - expected_value) ** 2) / max(expected_value, 1e-12) for observed_value, expected_value in zip(response, mean_response))

    diagnostics = {
        "aic": float(-2.0 * log_likelihood + 2.0 * column_count),
        "deviance": float(deviance),
        "pearson_chi2": float(pearson_chi2),
    }
    return coefficients, covariance, diagnostics


def load_income_table(path: Path) -> pd.DataFrame:
    income = pd.read_csv(path, dtype=str, encoding="utf-8-sig", skiprows=[1])
    income = income[["GEO_ID", "NAME", "S1901_C01_012E"]].copy()
    income["fips"] = income["GEO_ID"].str[-5:]
    income["median_income_2021"] = pd.to_numeric(income["S1901_C01_012E"], errors="coerce")
    income = income.rename(columns={"NAME": "county_label"})
    return income[["fips", "county_label", "median_income_2021"]]


def load_county_covariates(folder: Path) -> pd.DataFrame:
    elevation = pd.read_csv(folder / "counties_x_elevation.csv", dtype={"GEOID": str})
    elevation = elevation[["GEOID", "NAME", "ETOPO_2022_v1_60s_N90W180_bed"]].copy()
    elevation = elevation.rename(
        columns={
            "GEOID": "fips",
            "NAME": "county_name",
            "ETOPO_2022_v1_60s_N90W180_bed": "elevation_m",
        }
    )
    elevation["elevation_m"] = pd.to_numeric(elevation["elevation_m"], errors="coerce")

    income = load_income_table(folder / "ACSST5Y2021.S1901-Data.csv")

    diabetes = pd.read_csv(folder / "DiabetesPercentage.csv", dtype={"County_FIPS": str})
    diabetes["fips"] = diabetes["County_FIPS"].str.zfill(5)
    diabetes["diabetes_pct"] = pd.to_numeric(diabetes["Diagnosed Diabetes Percentage"], errors="coerce")
    diabetes = diabetes[["fips", "diabetes_pct"]]

    obesity = pd.read_csv(folder / "ObesityAll.csv", dtype={"County_FIPS": str})
    obesity["fips"] = obesity["County_FIPS"].str.zfill(5)
    obesity["obesity_pct"] = pd.to_numeric(obesity["Obesity Percentage"], errors="coerce")
    obesity = obesity[["fips", "obesity_pct"]]

    county_covariates = elevation.merge(income, on="fips", how="left")
    county_covariates = county_covariates.merge(diabetes, on="fips", how="left")
    county_covariates = county_covariates.merge(obesity, on="fips", how="left")
    county_covariates["oxygen_proxy_mmhg"] = inspired_oxygen_proxy_mmhg(county_covariates["elevation_m"])
    sea_level_pi02 = inspired_oxygen_proxy_mmhg(pd.Series([0.0])).iloc[0]
    county_covariates["oxygen_fraction_of_sea_level"] = county_covariates["oxygen_proxy_mmhg"] / sea_level_pi02
    county_covariates["hypoxia_burden"] = 1.0 - county_covariates["oxygen_fraction_of_sea_level"]
    return county_covariates.sort_values("fips").reset_index(drop=True)


def load_cancer_table(folder: Path) -> pd.DataFrame:
    usecols = [
        "STATE",
        "AREA",
        "AGE_ADJUSTED_RATE",
        "COUNT",
        "EVENT_TYPE",
        "POPULATION",
        "RACE",
        "SEX",
        "SITE",
        "YEAR",
    ]
    cancer = pd.read_csv(folder / "BYAREA_COUNTY.csv", usecols=usecols, dtype=str)
    cancer["fips"] = cancer["AREA"].map(extract_fips)
    cancer["age_adjusted_rate"] = pd.to_numeric(cancer["AGE_ADJUSTED_RATE"], errors="coerce")
    cancer["count"] = pd.to_numeric(cancer["COUNT"], errors="coerce")
    cancer["population"] = pd.to_numeric(cancer["POPULATION"], errors="coerce")
    cancer = cancer[(cancer["RACE"] == "All Races") & (cancer["SEX"] == "Male and Female")].copy()
    cancer = cancer[cancer["age_adjusted_rate"].notna() & cancer["fips"].notna()].copy()
    cancer = cancer.rename(columns={"STATE": "state_abbr", "EVENT_TYPE": "event_type", "SITE": "site", "YEAR": "year"})
    return cancer[
        [
            "fips",
            "state_abbr",
            "AREA",
            "event_type",
            "site",
            "year",
            "age_adjusted_rate",
            "count",
            "population",
        ]
    ].sort_values(["event_type", "site", "fips"]).reset_index(drop=True)


def build_county_master_table(county_covariates: pd.DataFrame, cancer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cancer = cancer[cancer["site"] == "All Cancer Sites Combined"].copy()
    all_cancer_wide = (
        all_cancer.pivot_table(index="fips", columns="event_type", values="age_adjusted_rate", aggfunc="first")
        .rename(columns={
            "Incidence": "all_cancer_incidence_rate",
            "Mortality": "all_cancer_mortality_rate",
        })
        .reset_index()
    )

    count_wide = (
        all_cancer.pivot_table(index="fips", columns="event_type", values="count", aggfunc="first")
        .rename(columns={
            "Incidence": "all_cancer_incidence_count",
            "Mortality": "all_cancer_mortality_count",
        })
        .reset_index()
    )

    population_wide = (
        all_cancer.pivot_table(index="fips", columns="event_type", values="population", aggfunc="first")
        .rename(columns={
            "Incidence": "all_cancer_incidence_population",
            "Mortality": "all_cancer_mortality_population",
        })
        .reset_index()
    )

    county_master = county_covariates.merge(all_cancer_wide, on="fips", how="left")
    county_master = county_master.merge(count_wide, on="fips", how="left")
    county_master = county_master.merge(population_wide, on="fips", how="left")

    site_support = (
        cancer.merge(county_covariates[["fips"]], on="fips", how="inner")
        .groupby(["event_type", "site"], as_index=False)
        .agg(counties_with_covariates=("fips", "nunique"), rows=("fips", "size"))
        .sort_values(["event_type", "counties_with_covariates", "site"], ascending=[True, False, True])
    )

    return county_master.sort_values("fips").reset_index(drop=True), site_support.reset_index(drop=True)


def fit_poisson_glm(master_table: pd.DataFrame, outcome_column: str, exposure_column: str) -> tuple[dict, pd.DataFrame]:
    model_frame = master_table[
        [outcome_column, exposure_column, "median_income_2021", "diabetes_pct", "obesity_pct"]
    ].dropna()
    term_names = ["const", exposure_column, "median_income_2021", "diabetes_pct", "obesity_pct"]
    design_matrix = []
    for _, row in model_frame.iterrows():
        design_matrix.append(
            [
                1.0,
                float(row[exposure_column]),
                float(row["median_income_2021"]),
                float(row["diabetes_pct"]),
                float(row["obesity_pct"]),
            ]
        )
    response = [float(value) for value in model_frame[outcome_column].tolist()]

    coefficients, covariance, diagnostics = fit_poisson_newton(design_matrix, response)
    standard_errors = [math.sqrt(max(covariance[index][index], 0.0)) for index in range(len(coefficients))]
    z_values = [coefficients[index] / standard_errors[index] if standard_errors[index] > 0 else float("nan") for index in range(len(coefficients))]
    p_values = [normal_p_value_from_z(z_value) if math.isfinite(z_value) else float("nan") for z_value in z_values]
    coefficient_frame = pd.DataFrame(
        {
            "term": term_names,
            "coefficient": coefficients,
            "std_error": standard_errors,
            "z_value": z_values,
            "p_value": p_values,
            "ci_lower": [coefficients[index] - 1.96 * standard_errors[index] for index in range(len(coefficients))],
            "ci_upper": [coefficients[index] + 1.96 * standard_errors[index] for index in range(len(coefficients))],
        }
    )

    metrics = {
        "outcome": outcome_column,
        "exposure": exposure_column,
        "n_rows": int(model_frame.shape[0]),
        "aic": diagnostics["aic"],
        "deviance": diagnostics["deviance"],
        "pearson_chi2": diagnostics["pearson_chi2"],
    }
    return metrics, coefficient_frame


def run_phase1(folder: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    county_covariates = load_county_covariates(folder)
    cancer = load_cancer_table(folder)
    county_master, site_support = build_county_master_table(county_covariates, cancer)

    model_specs = [
        ("all_cancer_mortality_rate", "elevation_m", "mortality_elevation_reproduction"),
        ("all_cancer_incidence_rate", "elevation_m", "incidence_elevation_primary"),
        ("all_cancer_mortality_rate", "oxygen_proxy_mmhg", "mortality_oxygen_proxy"),
        ("all_cancer_incidence_rate", "oxygen_proxy_mmhg", "incidence_oxygen_proxy_primary"),
    ]

    metrics_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []

    for outcome_column, exposure_column, model_name in model_specs:
        metrics, coefficients = fit_poisson_glm(county_master, outcome_column, exposure_column)
        metrics["model_name"] = model_name
        coefficients.insert(0, "model_name", model_name)
        metrics_rows.append(metrics)
        coefficient_frames.append(coefficients)

    metrics_frame = pd.DataFrame(metrics_rows)
    coefficients_frame = pd.concat(coefficient_frames, ignore_index=True)

    county_covariates.to_csv(output_dir / "county_covariates.csv", index=False)
    cancer.to_csv(output_dir / "cancer_endpoints_long.csv", index=False)
    county_master.to_csv(output_dir / "county_master_table.csv", index=False)
    site_support.to_csv(output_dir / "site_support.csv", index=False)
    metrics_frame.to_csv(output_dir / "baseline_model_metrics.csv", index=False)
    coefficients_frame.to_csv(output_dir / "baseline_model_coefficients.csv", index=False)

    summary = {
        "county_covariate_rows": int(county_covariates.shape[0]),
        "cancer_endpoint_rows": int(cancer.shape[0]),
        "county_master_rows": int(county_master.shape[0]),
        "all_cancer_incidence_complete_cases": int(
            county_master[["all_cancer_incidence_rate", "median_income_2021", "diabetes_pct", "obesity_pct", "elevation_m"]]
            .dropna()
            .shape[0]
        ),
        "all_cancer_mortality_complete_cases": int(
            county_master[["all_cancer_mortality_rate", "median_income_2021", "diabetes_pct", "obesity_pct", "elevation_m"]]
            .dropna()
            .shape[0]
        ),
        "oxygen_proxy_mmhg_at_sea_level": float(inspired_oxygen_proxy_mmhg(pd.Series([0.0])).iloc[0]),
        "outputs": {
            "county_covariates": str(output_dir / "county_covariates.csv"),
            "cancer_endpoints_long": str(output_dir / "cancer_endpoints_long.csv"),
            "county_master_table": str(output_dir / "county_master_table.csv"),
            "site_support": str(output_dir / "site_support.csv"),
            "baseline_model_metrics": str(output_dir / "baseline_model_metrics.csv"),
            "baseline_model_coefficients": str(output_dir / "baseline_model_coefficients.csv"),
        },
    }

    with open(output_dir / "phase1_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Brandon county master table and Phase 1 baseline models.")
    parser.add_argument("--folder", type=Path, default=Path("."), help="Folder containing the source CSV files at the project root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1"),
        help="Directory for generated master tables and baseline-model outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_phase1(arguments.folder, arguments.output_dir)
