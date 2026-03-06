from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = [
    "fips",
    "county_name",
    "county_label",
]

OUTCOME_COLUMNS = [
    "all_cancer_incidence_rate",
    "all_cancer_mortality_rate",
    "all_cancer_incidence_count",
    "all_cancer_mortality_count",
    "all_cancer_incidence_population",
    "all_cancer_mortality_population",
]

RAW_FEATURE_SPECS = [
    {"feature": "elevation_m", "raw_column": "elevation_m", "group": "exposure", "description": "County mean elevation in meters."},
    {"feature": "oxygen_proxy_mmhg", "raw_column": "oxygen_proxy_mmhg", "group": "exposure", "description": "Inspired oxygen proxy derived from county elevation."},
    {"feature": "oxygen_fraction_of_sea_level", "raw_column": "oxygen_fraction_of_sea_level", "group": "exposure", "description": "Inspired oxygen proxy as a fraction of sea-level oxygen."},
    {"feature": "hypoxia_burden", "raw_column": "hypoxia_burden", "group": "exposure", "description": "1 - oxygen fraction of sea level."},
    {"feature": "median_income_2021", "raw_column": "median_income_2021", "group": "baseline", "description": "ACS S1901 county median household income used in Phase 1."},
    {"feature": "local_diabetes_pct", "raw_column": "diabetes_pct", "group": "baseline", "description": "Local diabetes prevalence from the original Brandon folder."},
    {"feature": "local_obesity_pct", "raw_column": "obesity_pct", "group": "baseline", "description": "Local obesity prevalence from the original Brandon folder."},
    {"feature": "places_current_smoking_pct", "raw_column": "places_current_smoking_pct", "group": "places", "description": "CDC PLACES crude prevalence of current smoking."},
    {"feature": "places_lpa_pct", "raw_column": "places_lpa_pct", "group": "places", "description": "CDC PLACES crude prevalence of no leisure-time physical activity."},
    {"feature": "places_no_health_insurance_pct", "raw_column": "places_no_health_insurance_pct", "group": "places", "description": "CDC PLACES crude prevalence of no health insurance."},
    {"feature": "places_copd_pct", "raw_column": "places_copd_pct", "group": "places", "description": "CDC PLACES COPD crude prevalence."},
    {"feature": "places_high_blood_pressure_pct", "raw_column": "places_high_blood_pressure_pct", "group": "places", "description": "CDC PLACES high blood pressure crude prevalence."},
    {"feature": "places_diabetes_pct", "raw_column": "places_diabetes_pct", "group": "places", "description": "CDC PLACES diabetes crude prevalence."},
    {"feature": "places_obesity_pct", "raw_column": "places_obesity_pct", "group": "places", "description": "CDC PLACES obesity crude prevalence."},
    {"feature": "places_poor_physical_health_pct", "raw_column": "places_poor_physical_health_pct", "group": "places", "description": "CDC PLACES poor physical health crude prevalence."},
    {"feature": "places_poor_mental_health_pct", "raw_column": "places_poor_mental_health_pct", "group": "places", "description": "CDC PLACES poor mental health crude prevalence."},
    {"feature": "places_stroke_pct", "raw_column": "places_stroke_pct", "group": "places", "description": "CDC PLACES stroke crude prevalence."},
    {"feature": "svi_ep_pov150", "raw_column": "svi_ep_pov150", "group": "svi", "description": "SVI estimated percentage below 150% poverty."},
    {"feature": "svi_ep_unemp", "raw_column": "svi_ep_unemp", "group": "svi", "description": "SVI estimated unemployment percentage."},
    {"feature": "svi_ep_nohsdp", "raw_column": "svi_ep_nohsdp", "group": "svi", "description": "SVI estimated percentage without high school diploma."},
    {"feature": "svi_ep_noinsur", "raw_column": "svi_ep_uninsur", "group": "svi", "description": "SVI estimated percentage without health insurance."},
    {"feature": "svi_ep_age65", "raw_column": "svi_ep_age65", "group": "svi", "description": "SVI estimated percentage age 65 and older."},
    {"feature": "svi_ep_age17", "raw_column": "svi_ep_age17", "group": "svi", "description": "SVI estimated percentage age 17 and younger."},
    {"feature": "svi_ep_disabl", "raw_column": "svi_ep_disabl", "group": "svi", "description": "SVI estimated disability percentage."},
    {"feature": "svi_ep_sngpnt", "raw_column": "svi_ep_sngpnt", "group": "svi", "description": "SVI estimated single-parent household percentage."},
    {"feature": "svi_ep_limeng", "raw_column": "svi_ep_limeng", "group": "svi", "description": "SVI estimated limited-English percentage."},
    {"feature": "svi_ep_minrty", "raw_column": "svi_ep_minrty", "group": "svi", "description": "SVI estimated racial/ethnic minority percentage."},
    {"feature": "svi_ep_noveh", "raw_column": "svi_ep_noveh", "group": "svi", "description": "SVI estimated households without a vehicle."},
    {"feature": "svi_rpl_theme1", "raw_column": "svi_rpl_theme1", "group": "svi", "description": "SVI socioeconomic percentile rank."},
    {"feature": "svi_rpl_theme2", "raw_column": "svi_rpl_theme2", "group": "svi", "description": "SVI household characteristics percentile rank."},
    {"feature": "svi_rpl_theme3", "raw_column": "svi_rpl_theme3", "group": "svi", "description": "SVI minority/language percentile rank."},
    {"feature": "svi_rpl_theme4", "raw_column": "svi_rpl_theme4", "group": "svi", "description": "SVI housing/transport percentile rank."},
    {"feature": "svi_rpl_themes", "raw_column": "svi_rpl_themes", "group": "svi", "description": "SVI overall percentile rank."},
    {"feature": "rucc_code_2023", "raw_column": "rucc_code_2023", "group": "rucc", "description": "USDA 2023 Rural-Urban Continuum Code."},
    {"feature": "acs_per_capita_income_2024", "raw_column": "acs_dp03_dp03_0088e", "group": "acs", "description": "ACS DP03 per-capita income estimate."},
    {"feature": "acs_poverty_pct", "raw_column": "acs_dp03_dp03_0119pe", "group": "acs", "description": "ACS DP03 poverty percentage."},
    {"feature": "acs_no_health_insurance_pct", "raw_column": "acs_dp03_dp03_0099pe", "group": "acs", "description": "ACS DP03 uninsured percentage."},
    {"feature": "acs_unemployment_rate", "raw_column": "acs_dp03_dp03_0009e", "group": "acs", "description": "ACS DP03 unemployment rate estimate."},
    {"feature": "acs_mean_travel_time_min", "raw_column": "acs_dp03_dp03_0025e", "group": "acs", "description": "ACS DP03 mean travel time to work in minutes."},
    {"feature": "acs_bachelors_plus_pct", "raw_column": "acs_dp02_dp02_0068pe", "group": "acs", "description": "ACS DP02 bachelor\'s degree or higher percentage."},
    {"feature": "acs_less_than_9th_pct", "raw_column": "acs_dp02_dp02_0060pe", "group": "acs", "description": "ACS DP02 less than 9th grade percentage."},
    {"feature": "acs_no_diploma_pct", "raw_column": "acs_dp02_dp02_0061pe", "group": "acs", "description": "ACS DP02 9th-12th no diploma percentage."},
    {"feature": "acs_broadband_pct", "raw_column": "acs_dp02_dp02_0154pe", "group": "acs", "description": "ACS DP02 broadband subscription percentage."},
    {"feature": "acs_owner_occupied_pct", "raw_column": "acs_dp04_dp04_0046pe", "group": "acs", "description": "ACS DP04 owner-occupied housing percentage."},
    {"feature": "acs_rent_burden_35plus_pct", "raw_column": "acs_dp04_dp04_0142pe", "group": "acs", "description": "ACS DP04 renter households spending 35%+ income on rent."},
    {"feature": "acs_median_age_years", "raw_column": "acs_dp05_dp05_0018e", "group": "acs", "description": "ACS DP05 median age in years."},
    {"feature": "acs_under_18_pct", "raw_column": "acs_dp05_dp05_0019pe", "group": "acs", "description": "ACS DP05 percentage under age 18."},
    {"feature": "acs_age_65_plus_pct", "raw_column": "acs_dp05_dp05_0024pe", "group": "acs", "description": "ACS DP05 percentage age 65 and older."},
    {"feature": "acs_nonhisp_white_pct", "raw_column": "acs_dp05_dp05_0096pe", "group": "acs", "description": "ACS DP05 non-Hispanic White percentage."},
    {"feature": "acs_black_pct", "raw_column": "acs_dp05_dp05_0045pe", "group": "acs", "description": "ACS DP05 Black population percentage."},
    {"feature": "acs_hispanic_pct", "raw_column": "acs_dp05_dp05_0090pe", "group": "acs", "description": "ACS DP05 Hispanic/Latino percentage."},
    {"feature": "acs_asian_pct", "raw_column": "acs_dp05_dp05_0099pe", "group": "acs", "description": "ACS DP05 non-Hispanic Asian percentage."},
]


DERIVED_SPECS = [
    {"feature": "median_income_log1p", "group": "derived", "description": "log1p of Phase 1 median income."},
    {"feature": "acs_per_capita_income_log1p", "group": "derived", "description": "log1p of ACS per-capita income."},
    {"feature": "acs_less_than_high_school_pct", "group": "derived", "description": "ACS less than high school percentage = less than 9th + no diploma."},
    {"feature": "rucc_nonmetro_flag", "group": "derived", "description": "Indicator for RUCC code >= 4."},
    {"feature": "obesity_places_minus_local", "group": "derived_qc", "description": "PLACES obesity minus local obesity percentage."},
    {"feature": "diabetes_places_minus_local", "group": "derived_qc", "description": "PLACES diabetes minus local diabetes percentage."},
]


MODEL_FEATURE_COLUMNS = [
    "elevation_m",
    "oxygen_proxy_mmhg",
    "oxygen_fraction_of_sea_level",
    "hypoxia_burden",
    "median_income_2021",
    "median_income_log1p",
    "local_diabetes_pct",
    "local_obesity_pct",
    "places_current_smoking_pct",
    "places_lpa_pct",
    "places_no_health_insurance_pct",
    "places_copd_pct",
    "places_high_blood_pressure_pct",
    "places_diabetes_pct",
    "places_obesity_pct",
    "places_poor_physical_health_pct",
    "places_poor_mental_health_pct",
    "places_stroke_pct",
    "svi_ep_pov150",
    "svi_ep_unemp",
    "svi_ep_nohsdp",
    "svi_ep_noinsur",
    "svi_ep_age65",
    "svi_ep_age17",
    "svi_ep_disabl",
    "svi_ep_sngpnt",
    "svi_ep_limeng",
    "svi_ep_minrty",
    "svi_ep_noveh",
    "svi_rpl_theme1",
    "svi_rpl_theme2",
    "svi_rpl_theme3",
    "svi_rpl_theme4",
    "svi_rpl_themes",
    "rucc_code_2023",
    "rucc_nonmetro_flag",
    "acs_per_capita_income_2024",
    "acs_per_capita_income_log1p",
    "acs_poverty_pct",
    "acs_no_health_insurance_pct",
    "acs_unemployment_rate",
    "acs_mean_travel_time_min",
    "acs_bachelors_plus_pct",
    "acs_less_than_9th_pct",
    "acs_no_diploma_pct",
    "acs_less_than_high_school_pct",
    "acs_broadband_pct",
    "acs_owner_occupied_pct",
    "acs_rent_burden_35plus_pct",
    "acs_median_age_years",
    "acs_under_18_pct",
    "acs_age_65_plus_pct",
    "acs_nonhisp_white_pct",
    "acs_black_pct",
    "acs_hispanic_pct",
    "acs_asian_pct",
]


def build_manifest_rows() -> list[dict]:
    rows = []
    for spec in RAW_FEATURE_SPECS:
        rows.append(
            {
                "feature": spec["feature"],
                "group": spec["group"],
                "raw_column": spec["raw_column"],
                "derived_from": "",
                "description": spec["description"],
                "include_in_model": spec["feature"] in MODEL_FEATURE_COLUMNS,
            }
        )
    for spec in DERIVED_SPECS:
        rows.append(
            {
                "feature": spec["feature"],
                "group": spec["group"],
                "raw_column": "",
                "derived_from": "engineered",
                "description": spec["description"],
                "include_in_model": spec["feature"] in MODEL_FEATURE_COLUMNS,
            }
        )
    return rows


def create_curated_features(phase2_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curated = phase2_table[[column for column in IDENTIFIER_COLUMNS + OUTCOME_COLUMNS if column in phase2_table.columns]].copy()

    for spec in RAW_FEATURE_SPECS:
        raw_column = spec["raw_column"]
        curated[spec["feature"]] = pd.to_numeric(phase2_table[raw_column], errors="coerce") if raw_column in phase2_table.columns else np.nan

    curated["rucc_nonmetro_flag"] = np.where(curated["rucc_code_2023"].notna(), (curated["rucc_code_2023"] >= 4).astype(float), np.nan)
    curated["acs_less_than_high_school_pct"] = curated["acs_less_than_9th_pct"] + curated["acs_no_diploma_pct"]
    curated["median_income_log1p"] = np.log1p(curated["median_income_2021"].clip(lower=0))
    curated["acs_per_capita_income_log1p"] = np.log1p(curated["acs_per_capita_income_2024"].clip(lower=0))
    curated["obesity_places_minus_local"] = curated["places_obesity_pct"] - curated["local_obesity_pct"]
    curated["diabetes_places_minus_local"] = curated["places_diabetes_pct"] - curated["local_diabetes_pct"]

    manifest = pd.DataFrame(build_manifest_rows())
    return curated, manifest


def build_missingness_summary(curated: pd.DataFrame, feature_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = len(curated)
    for _, manifest_row in feature_manifest.iterrows():
        feature_name = manifest_row["feature"]
        if feature_name not in curated.columns:
            continue
        observed = int(curated[feature_name].notna().sum())
        rows.append(
            {
                "feature": feature_name,
                "group": manifest_row["group"],
                "observed_rows": observed,
                "missing_rows": int(total_rows - observed),
                "observed_fraction": float(observed / total_rows if total_rows else np.nan),
                "include_in_model": bool(manifest_row["include_in_model"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["include_in_model", "observed_fraction", "feature"], ascending=[False, False, True]).reset_index(drop=True)


def build_model_matrix(curated: pd.DataFrame) -> pd.DataFrame:
    base = curated[[column for column in IDENTIFIER_COLUMNS + OUTCOME_COLUMNS if column in curated.columns]].copy()
    engineered_columns = {}

    for feature_name in MODEL_FEATURE_COLUMNS:
        feature_values = pd.to_numeric(curated[feature_name], errors="coerce")
        engineered_columns[f"{feature_name}__missing"] = feature_values.isna().astype(int)

        median_value = float(feature_values.median()) if feature_values.notna().any() else 0.0
        imputed = feature_values.fillna(median_value)
        mean_value = float(imputed.mean())
        std_value = float(imputed.std(ddof=0))
        if std_value == 0.0:
            standardized = pd.Series(np.zeros(len(imputed)), index=imputed.index)
        else:
            standardized = (imputed - mean_value) / std_value

        engineered_columns[f"z_{feature_name}"] = standardized.astype(float)

    engineered_frame = pd.DataFrame(engineered_columns, index=curated.index)
    return pd.concat([base, engineered_frame], axis=1)


def summarize_phase25(curated: pd.DataFrame, model_matrix: pd.DataFrame, feature_manifest: pd.DataFrame, missingness: pd.DataFrame) -> dict:
    model_missingness = missingness[missingness["include_in_model"]]
    return {
        "curated_rows": int(curated.shape[0]),
        "curated_columns": int(curated.shape[1]),
        "model_matrix_rows": int(model_matrix.shape[0]),
        "model_matrix_columns": int(model_matrix.shape[1]),
        "curated_feature_count": int(feature_manifest.shape[0]),
        "model_feature_count": int(sum(feature_manifest["include_in_model"])),
        "incidence_rows": int(curated["all_cancer_incidence_rate"].notna().sum()),
        "mortality_rows": int(curated["all_cancer_mortality_rate"].notna().sum()),
        "mean_model_feature_observed_fraction": float(model_missingness["observed_fraction"].mean()),
        "min_model_feature_observed_fraction": float(model_missingness["observed_fraction"].min()),
        "outputs": {
            "curated_county_features": "outputs/phase25/curated_county_features.csv",
            "curated_model_matrix": "outputs/phase25/curated_model_matrix.csv",
            "incidence_modeling_table": "outputs/phase25/incidence_modeling_table.csv",
            "mortality_modeling_table": "outputs/phase25/mortality_modeling_table.csv",
            "feature_manifest": "outputs/phase25/feature_manifest.csv",
            "missingness_summary": "outputs/phase25/missingness_summary.csv",
        },
    }


def run_phase25(root_dir: Path) -> dict:
    phase2_path = root_dir / "outputs" / "phase2" / "county_master_table_phase2.csv"
    if not phase2_path.exists():
        raise FileNotFoundError(f"Phase 2 table not found at {phase2_path}")

    output_dir = root_dir / "outputs" / "phase25"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase2_table = pd.read_csv(phase2_path, low_memory=False, dtype={"fips": str})
    curated, feature_manifest = create_curated_features(phase2_table)
    missingness = build_missingness_summary(curated, feature_manifest)
    model_matrix = build_model_matrix(curated)

    incidence_table = model_matrix[model_matrix["all_cancer_incidence_rate"].notna()].reset_index(drop=True)
    mortality_table = model_matrix[model_matrix["all_cancer_mortality_rate"].notna()].reset_index(drop=True)

    curated.to_csv(output_dir / "curated_county_features.csv", index=False)
    model_matrix.to_csv(output_dir / "curated_model_matrix.csv", index=False)
    incidence_table.to_csv(output_dir / "incidence_modeling_table.csv", index=False)
    mortality_table.to_csv(output_dir / "mortality_modeling_table.csv", index=False)
    feature_manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
    missingness.to_csv(output_dir / "missingness_summary.csv", index=False)

    summary = summarize_phase25(curated, model_matrix, feature_manifest, missingness)
    (output_dir / "phase25_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a curated model-ready feature matrix from the Phase 2 county master table.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_phase25(args.root_dir), indent=2))


if __name__ == "__main__":
    main()
