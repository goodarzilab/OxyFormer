from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase3_causal_demo import (
    BASE_MODEL_SPECS,
    RANDOM_SEED,
    RIDGE_ALPHA_OUTCOME,
    RIDGE_ALPHA_TREATMENT,
    SITE_SELECTION_RULES,
    TREATMENT_COLUMN,
    build_site_feature_table,
    build_site_model_specs,
    covariates_for_spec,
    cross_fitted_partial_linear_dml,
    load_foundation_embeddings,
)


DEFAULT_SEEDS = [RANDOM_SEED, RANDOM_SEED + 7, RANDOM_SEED + 14]
DEFAULT_MIN_STATE_ROWS = 20


def parse_seed_values(raw_value: str | None) -> list[int]:
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_SEEDS[:]
    values = []
    for token in raw_value.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("At least one random seed is required.")
    return values


def prepare_curated_and_specs(
    root_dir: Path,
    include_site_models: bool,
    incidence_support_min: int,
    mortality_support_min: int,
    max_site_models_per_event: int,
    use_foundation_embeddings: bool,
    embedding_path: Path | None,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, list[str], Path | None]:
    curated_path = root_dir / "outputs" / "phase25" / "curated_county_features.csv"
    if not curated_path.exists():
        raise FileNotFoundError(f"Curated Phase 2.5 table not found at {curated_path}")

    curated = pd.read_csv(curated_path, low_memory=False, dtype={"fips": str})
    embedding_columns: list[str] = []
    resolved_embedding_path: Path | None = None
    if use_foundation_embeddings:
        embedding_frame, embedding_columns = load_foundation_embeddings(root_dir=root_dir, embedding_path=embedding_path)
        resolved_embedding_path = embedding_path if embedding_path is not None else root_dir / "outputs" / "phase26" / "county_foundation_embeddings.csv"
        curated = curated.merge(embedding_frame, on="fips", how="left")
    model_specs = BASE_MODEL_SPECS[:]
    selected_site_manifest = pd.DataFrame()

    if include_site_models:
        site_table, selected_site_manifest = build_site_feature_table(
            root_dir=root_dir,
            incidence_support_min=incidence_support_min,
            mortality_support_min=mortality_support_min,
            max_sites_per_event=max_site_models_per_event,
        )
        curated = curated.merge(site_table, on="fips", how="left")
        if not selected_site_manifest.empty:
            model_specs.extend(build_site_model_specs(selected_site_manifest))

    return curated, model_specs, selected_site_manifest, embedding_columns, resolved_embedding_path


def run_full_models(curated: pd.DataFrame, model_specs: list[dict], embedding_columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    effect_lookup = {}

    for spec in model_specs:
        effect_summary, _, _ = cross_fitted_partial_linear_dml(
            dataframe=curated,
            outcome_column=spec["outcome"],
            treatment_column=TREATMENT_COLUMN,
            covariates=covariates_for_spec(spec, embedding_columns=embedding_columns),
            fold_count=5,
            alpha_outcome=RIDGE_ALPHA_OUTCOME,
            alpha_treatment=RIDGE_ALPHA_TREATMENT,
            seed=RANDOM_SEED,
        )
        effect_lookup[spec["model_name"]] = effect_summary
        rows.append(
            {
                "model_name": spec["model_name"],
                "outcome": spec["outcome"],
                "analysis_group": spec["analysis_group"],
                "event_type": spec["event_type"],
                "site": spec["site"],
                "effect_type": spec["effect_type"],
                "support_count": spec["support_count"],
                "site_rank_within_event": spec["site_rank_within_event"],
                "n_rows": effect_summary["n_rows"],
                "slope": effect_summary["slope"],
                "slope_se": effect_summary["slope_se"],
                "p_value": effect_summary["p_value"],
                "effect_per_sd_treatment": effect_summary["effect_per_sd_treatment"],
                "effect_q90_minus_q10": effect_summary["effect_q90_minus_q10"],
                "effect_q90_minus_q10_pct_of_mean_outcome": effect_summary["effect_q90_minus_q10_pct_of_mean_outcome"],
                "outcome_cv_mean_r2": effect_summary["outcome_cv_mean_r2"],
                "treatment_cv_mean_r2": effect_summary["treatment_cv_mean_r2"],
            }
        )

    return pd.DataFrame(rows), effect_lookup


def run_seed_sensitivity(curated: pd.DataFrame, model_specs: list[dict], seeds: list[int], embedding_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for spec in model_specs:
        covariates = covariates_for_spec(spec, embedding_columns=embedding_columns)
        for seed in seeds:
            effect_summary, _, _ = cross_fitted_partial_linear_dml(
                dataframe=curated,
                outcome_column=spec["outcome"],
                treatment_column=TREATMENT_COLUMN,
                covariates=covariates,
                fold_count=5,
                alpha_outcome=RIDGE_ALPHA_OUTCOME,
                alpha_treatment=RIDGE_ALPHA_TREATMENT,
                seed=seed,
            )
            rows.append(
                {
                    "model_name": spec["model_name"],
                    "outcome": spec["outcome"],
                    "analysis_group": spec["analysis_group"],
                    "event_type": spec["event_type"],
                    "site": spec["site"],
                    "effect_type": spec["effect_type"],
                    "support_count": spec["support_count"],
                    "seed": seed,
                    "n_rows": effect_summary["n_rows"],
                    "slope": effect_summary["slope"],
                    "slope_se": effect_summary["slope_se"],
                    "p_value": effect_summary["p_value"],
                    "effect_per_sd_treatment": effect_summary["effect_per_sd_treatment"],
                    "effect_q90_minus_q10": effect_summary["effect_q90_minus_q10"],
                    "effect_q90_minus_q10_pct_of_mean_outcome": effect_summary["effect_q90_minus_q10_pct_of_mean_outcome"],
                    "outcome_cv_mean_r2": effect_summary["outcome_cv_mean_r2"],
                    "treatment_cv_mean_r2": effect_summary["treatment_cv_mean_r2"],
                }
            )

    seed_frame = pd.DataFrame(rows)
    summary = (
        seed_frame.groupby(["model_name", "outcome", "analysis_group", "event_type", "site", "effect_type"], as_index=False)
        .agg(
            support_count=("support_count", "first"),
            seeds_tested=("seed", "nunique"),
            slope_mean=("slope", "mean"),
            slope_sd=("slope", "std"),
            slope_min=("slope", "min"),
            slope_max=("slope", "max"),
            min_p_value=("p_value", "min"),
            max_p_value=("p_value", "max"),
            mean_outcome_cv_r2=("outcome_cv_mean_r2", "mean"),
            mean_treatment_cv_r2=("treatment_cv_mean_r2", "mean"),
        )
        .sort_values(["analysis_group", "event_type", "site", "effect_type", "model_name"])
        .reset_index(drop=True)
    )
    summary["slope_sd"] = summary["slope_sd"].fillna(0.0)
    summary["sign_consistent"] = np.where(summary["slope_min"] * summary["slope_max"] > 0, True, False)
    return seed_frame, summary


def run_leave_one_state_out(
    curated: pd.DataFrame,
    model_specs: list[dict],
    full_effect_lookup: dict[str, dict],
    min_state_rows: int,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = curated.copy()
    analysis["state_fips"] = analysis["fips"].astype(str).str[:2]
    state_counts = analysis["state_fips"].value_counts().sort_index()
    eligible_states = state_counts[state_counts >= min_state_rows]

    rows = []
    for spec in model_specs:
        covariates = covariates_for_spec(spec, embedding_columns=embedding_columns)
        full_slope = float(full_effect_lookup[spec["model_name"]]["slope"])

        for state_fips, held_out_rows in eligible_states.items():
            subset = analysis[analysis["state_fips"] != state_fips].reset_index(drop=True)
            effect_summary, _, _ = cross_fitted_partial_linear_dml(
                dataframe=subset,
                outcome_column=spec["outcome"],
                treatment_column=TREATMENT_COLUMN,
                covariates=covariates,
                fold_count=5,
                alpha_outcome=RIDGE_ALPHA_OUTCOME,
                alpha_treatment=RIDGE_ALPHA_TREATMENT,
                seed=RANDOM_SEED,
            )
            slope = float(effect_summary["slope"])
            slope_shift = slope - full_slope
            slope_shift_pct = slope_shift / abs(full_slope) if full_slope != 0 else np.nan
            rows.append(
                {
                    "model_name": spec["model_name"],
                    "outcome": spec["outcome"],
                    "analysis_group": spec["analysis_group"],
                    "event_type": spec["event_type"],
                    "site": spec["site"],
                    "effect_type": spec["effect_type"],
                    "support_count": spec["support_count"],
                    "held_out_state_fips": state_fips,
                    "held_out_rows": int(held_out_rows),
                    "n_rows": effect_summary["n_rows"],
                    "full_slope": full_slope,
                    "slope": slope,
                    "slope_shift": slope_shift,
                    "slope_shift_pct": slope_shift_pct,
                    "slope_se": effect_summary["slope_se"],
                    "p_value": effect_summary["p_value"],
                }
            )

    if not rows:
        empty_details = pd.DataFrame(
            columns=[
                "model_name",
                "outcome",
                "analysis_group",
                "event_type",
                "site",
                "effect_type",
                "support_count",
                "held_out_state_fips",
                "held_out_rows",
                "n_rows",
                "full_slope",
                "slope",
                "slope_shift",
                "slope_shift_pct",
                "slope_se",
                "p_value",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "model_name",
                "outcome",
                "analysis_group",
                "event_type",
                "site",
                "effect_type",
                "support_count",
                "states_tested",
                "median_abs_slope_shift_pct",
                "mean_abs_slope_shift_pct",
                "max_abs_slope_shift_pct",
                "most_influential_state_fips",
                "worst_case_slope",
                "worst_case_p_value",
                "sign_consistent",
            ]
        )
        return empty_details, empty_summary

    state_frame = pd.DataFrame(rows)
    summary_rows = []
    for _, group in state_frame.groupby(["model_name", "outcome", "analysis_group", "event_type", "site", "effect_type"], as_index=False):
        worst_row = group.iloc[group["slope_shift_pct"].abs().fillna(-np.inf).argmax()]
        summary_rows.append(
            {
                "model_name": worst_row["model_name"],
                "outcome": worst_row["outcome"],
                "analysis_group": worst_row["analysis_group"],
                "event_type": worst_row["event_type"],
                "site": worst_row["site"],
                "effect_type": worst_row["effect_type"],
                "support_count": worst_row["support_count"],
                "states_tested": int(group["held_out_state_fips"].nunique()),
                "median_abs_slope_shift_pct": float(group["slope_shift_pct"].abs().median()),
                "mean_abs_slope_shift_pct": float(group["slope_shift_pct"].abs().mean()),
                "max_abs_slope_shift_pct": float(group["slope_shift_pct"].abs().max()),
                "most_influential_state_fips": worst_row["held_out_state_fips"],
                "worst_case_slope": float(worst_row["slope"]),
                "worst_case_p_value": float(worst_row["p_value"]),
                "sign_consistent": bool((group["slope"] > 0).all() or (group["slope"] < 0).all()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["analysis_group", "event_type", "site", "effect_type", "model_name"]).reset_index(drop=True)
    return state_frame, summary


def run_phase4(
    root_dir: Path,
    seeds: list[int],
    min_state_rows: int,
    include_site_models: bool = False,
    incidence_support_min: int = SITE_SELECTION_RULES["Incidence"]["min_support"],
    mortality_support_min: int = SITE_SELECTION_RULES["Mortality"]["min_support"],
    max_site_models_per_event: int = SITE_SELECTION_RULES["Incidence"]["max_sites"],
    use_foundation_embeddings: bool = False,
    embedding_path: Path | None = None,
) -> dict:
    output_dir = root_dir / "outputs" / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)

    curated, model_specs, selected_site_manifest, embedding_columns, resolved_embedding_path = prepare_curated_and_specs(
        root_dir=root_dir,
        include_site_models=include_site_models,
        incidence_support_min=incidence_support_min,
        mortality_support_min=mortality_support_min,
        max_site_models_per_event=max_site_models_per_event,
        use_foundation_embeddings=use_foundation_embeddings,
        embedding_path=embedding_path,
    )
    full_effects, full_effect_lookup = run_full_models(curated, model_specs, embedding_columns)
    seed_details, seed_summary = run_seed_sensitivity(curated, model_specs, seeds, embedding_columns)
    state_details, state_summary = run_leave_one_state_out(curated, model_specs, full_effect_lookup, min_state_rows, embedding_columns)

    full_effects.to_csv(output_dir / "full_model_effects.csv", index=False)
    seed_details.to_csv(output_dir / "seed_sensitivity.csv", index=False)
    seed_summary.to_csv(output_dir / "seed_sensitivity_summary.csv", index=False)
    state_details.to_csv(output_dir / "leave_one_state_out.csv", index=False)
    state_summary.to_csv(output_dir / "leave_one_state_out_summary.csv", index=False)

    summary = {
        "treatment": TREATMENT_COLUMN,
        "seeds": [int(seed) for seed in seeds],
        "min_state_rows": int(min_state_rows),
        "site_models_included": bool(include_site_models),
        "uses_foundation_embeddings": bool(embedding_columns),
        "foundation_embedding_count": int(len(embedding_columns)),
        "foundation_embedding_path": str(resolved_embedding_path) if resolved_embedding_path is not None else None,
        "selected_site_models": selected_site_manifest[
            ["event_type", "site_clean", "counties_with_covariates", "site_rank_within_event", "outcome_column"]
        ].rename(columns={"site_clean": "site"}).to_dict(orient="records") if include_site_models and not selected_site_manifest.empty else [],
        "models": [],
        "outputs": {
            "full_model_effects": "outputs/phase4/full_model_effects.csv",
            "seed_sensitivity": "outputs/phase4/seed_sensitivity.csv",
            "seed_sensitivity_summary": "outputs/phase4/seed_sensitivity_summary.csv",
            "leave_one_state_out": "outputs/phase4/leave_one_state_out.csv",
            "leave_one_state_out_summary": "outputs/phase4/leave_one_state_out_summary.csv",
        },
    }

    for _, full_row in full_effects.iterrows():
        seed_row = seed_summary[seed_summary["model_name"] == full_row["model_name"]].iloc[0]
        matching_state_rows = state_summary[state_summary["model_name"] == full_row["model_name"]]
        if matching_state_rows.empty:
            state_row = {
                "states_tested": 0,
                "sign_consistent": False,
                "max_abs_slope_shift_pct": np.nan,
                "most_influential_state_fips": "",
            }
        else:
            state_row = matching_state_rows.iloc[0]
        summary["models"].append(
            {
                "model_name": full_row["model_name"],
                "analysis_group": full_row["analysis_group"],
                "event_type": full_row["event_type"],
                "site": full_row["site"],
                "effect_type": full_row["effect_type"],
                "support_count": float(full_row["support_count"]) if pd.notna(full_row["support_count"]) else np.nan,
                "full_slope": float(full_row["slope"]),
                "full_p_value": float(full_row["p_value"]),
                "seed_slope_sd": float(seed_row["slope_sd"]),
                "seed_sign_consistent": bool(seed_row["sign_consistent"]),
                "states_tested": int(state_row["states_tested"]),
                "leave_one_state_sign_consistent": bool(state_row["sign_consistent"]),
                "max_abs_state_shift_pct": float(state_row["max_abs_slope_shift_pct"]),
                "most_influential_state_fips": state_row["most_influential_state_fips"],
            }
        )

    (output_dir / "phase4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 robustness checks for the county-level causal demo.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of random seeds for seed-sensitivity checks.")
    parser.add_argument("--min-state-rows", type=int, default=DEFAULT_MIN_STATE_ROWS, help="Minimum rows required to evaluate a leave-one-state-out exclusion.")
    parser.add_argument("--include-site-models", action="store_true", help="Also run robustness checks for supported site-specific cancer outcome models.")
    parser.add_argument("--use-foundation-embeddings", action="store_true", help="Append Phase 2.6 learned county embeddings to the robustness covariate set.")
    parser.add_argument("--embedding-path", type=Path, default=None, help="Optional path to a county foundation embedding CSV.")
    parser.add_argument("--site-support-incidence", type=int, default=SITE_SELECTION_RULES["Incidence"]["min_support"], help="Minimum county support required for incidence site models.")
    parser.add_argument("--site-support-mortality", type=int, default=SITE_SELECTION_RULES["Mortality"]["min_support"], help="Minimum county support required for mortality site models.")
    parser.add_argument("--max-site-models-per-event", type=int, default=SITE_SELECTION_RULES["Incidence"]["max_sites"], help="Maximum number of supported sites to include per event type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_values(args.seeds)
    print(
        json.dumps(
            run_phase4(
                root_dir=args.root_dir,
                seeds=seeds,
                min_state_rows=args.min_state_rows,
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
