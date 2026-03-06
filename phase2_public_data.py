from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests


PLACES_COUNTY_QUERY_URL = (
    "https://services3.arcgis.com/ZvidGQkLaDJxRSJ2/ArcGIS/rest/services/"
    "PLACES__County_Data__GIS_Friendly_Format___2025_release/FeatureServer/1/query"
)
PLACES_COUNTY_CSV_URL = "https://data.cdc.gov/api/views/swc5-untb/rows.csv?accessType=DOWNLOAD"
SVI_COUNTY_QUERY_URL = (
    "https://services3.arcgis.com/ZvidGQkLaDJxRSJ2/ArcGIS/rest/services/"
    "CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/1/query"
)
RUCC_2023_CSV_URL = "https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv?v=49543"
ACS_PROFILE_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5/profile"
ACS_GROUPS = ["DP02", "DP03", "DP04", "DP05"]


def sanitize_column_name(column_name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(column_name).strip().lower())
    return cleaned.strip("_")


def infer_fips_column(dataframe: pd.DataFrame) -> str:
    candidates = [
        "fips",
        "county_fips",
        "countyfips",
        "stcnty",
        "geoid",
        "locationid",
    ]
    available = set(dataframe.columns)
    for candidate in candidates:
        if candidate in available:
            return candidate

    for column_name in dataframe.columns:
        if "fips" in column_name or column_name.endswith("geoid"):
            return column_name

    raise ValueError("Unable to infer a county FIPS column from the downloaded dataset.")


def standardize_fips(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{5})", expand=False).str.zfill(5)


def standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    renamed = dataframe.copy()
    renamed.columns = [sanitize_column_name(column_name) for column_name in renamed.columns]
    return renamed


def read_csv_with_fallback(path: Path, low_memory: bool = False) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, low_memory=low_memory, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=low_memory)


def download_json(url: str, params: dict, timeout: int = 180) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def download_to_path(url: str, destination: Path, timeout: int = 180) -> None:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def fetch_places_county(destination: Path) -> pd.DataFrame:
    errors = []

    try:
        download_to_path(PLACES_COUNTY_CSV_URL, destination)
        dataframe = pd.read_csv(destination, low_memory=False)
        if dataframe.empty:
            raise ValueError("CDC PLACES CSV download returned zero rows.")
        return dataframe
    except Exception as exc:
        errors.append(f"cdc_csv={exc}")

    try:
        return fetch_arcgis_layer(PLACES_COUNTY_QUERY_URL, destination)
    except Exception as exc:
        errors.append(f"arcgis={exc}")

    raise ValueError("Unable to download PLACES county data. " + " | ".join(errors))


def fetch_arcgis_layer(query_url: str, destination: Path, page_size: int = 2000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    last_payload: dict | None = None

    while True:
        payload = download_json(
            query_url,
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": page_size,
            },
        )
        last_payload = payload
        features = payload.get("features", [])
        if not features:
            break

        frames.append(pd.DataFrame([feature["attributes"] for feature in features]))

        if len(features) < page_size:
            break

        offset += len(features)

    if not frames:
        raise ValueError(f"No records returned from ArcGIS layer: {query_url}. Payload preview: {json.dumps(last_payload)[:1000]}")

    dataframe = pd.concat(frames, ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return dataframe


def fetch_acs_group(year: int, group_name: str, destination: Path, api_key: str | None) -> pd.DataFrame:
    params = {
        "get": f"NAME,group({group_name})",
        "for": "county:*",
        "in": "state:*",
    }
    if api_key:
        params["key"] = api_key

    payload = download_json(ACS_PROFILE_BASE_URL.format(year=year), params)
    dataframe = pd.DataFrame(payload[1:], columns=payload[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return dataframe


def fetch_acs_group_metadata(year: int, group_name: str, destination: Path) -> pd.DataFrame:
    payload = download_json(f"{ACS_PROFILE_BASE_URL.format(year=year)}/groups/{group_name}.json", params={})
    variables = payload.get("variables", {})
    records = []
    for variable_name, metadata in variables.items():
        records.append(
            {
                "group": group_name,
                "variable": variable_name,
                "label": metadata.get("label"),
                "concept": metadata.get("concept"),
                "predicate_type": metadata.get("predicateType"),
            }
        )
    dataframe = pd.DataFrame(records).sort_values("variable").reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    return dataframe


def select_existing_columns(dataframe: pd.DataFrame, candidate_columns: list[str]) -> list[str]:
    return [column_name for column_name in candidate_columns if column_name in dataframe.columns]


def normalize_places(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = standardize_columns(read_csv_with_fallback(raw_path, low_memory=False))
    fips_column = infer_fips_column(dataframe)
    dataframe["fips"] = standardize_fips(dataframe[fips_column])

    year_column = "year" if "year" in dataframe.columns else None
    if year_column is not None:
        dataframe[year_column] = pd.to_numeric(dataframe[year_column], errors="coerce")
        latest_year = int(dataframe[year_column].dropna().max())
        dataframe = dataframe[dataframe[year_column] == latest_year].copy()

    if "datavaluetypeid" in dataframe.columns:
        dataframe = dataframe[dataframe["datavaluetypeid"].astype(str).str.lower() == "crdprv"].copy()
    elif "data_value_type" in dataframe.columns:
        dataframe = dataframe[dataframe["data_value_type"].astype(str).str.lower() == "crude prevalence"].copy()

    dataframe["data_value"] = pd.to_numeric(dataframe["data_value"], errors="coerce")
    measure_map = {
        "ACCESS2": "places_no_health_insurance_pct",
        "COPD": "places_copd_pct",
        "CSMOKING": "places_current_smoking_pct",
        "DIABETES": "places_diabetes_pct",
        "BPHIGH": "places_high_blood_pressure_pct",
        "LPA": "places_lpa_pct",
        "OBESITY": "places_obesity_pct",
        "SLEEP": "places_sleep_lt_7h_pct",
        "STROKE": "places_stroke_pct",
        "MHLTH": "places_poor_mental_health_pct",
        "PHLTH": "places_poor_physical_health_pct",
    }
    filtered = dataframe[dataframe["measureid"].isin(measure_map)].copy()
    filtered["feature_name"] = filtered["measureid"].map(measure_map)
    pivoted = filtered.pivot_table(index="fips", columns="feature_name", values="data_value", aggfunc="first").reset_index()

    population = dataframe[["fips", "totalpopulation"]].drop_duplicates(subset="fips").copy()
    population["places_population"] = pd.to_numeric(population["totalpopulation"], errors="coerce")
    output = pivoted.merge(population[["fips", "places_population"]], on="fips", how="left")
    output = output.drop_duplicates(subset="fips").reset_index(drop=True)

    feature_catalog_rows = [{"source": "places", "feature": value, "source_column": key} for key, value in measure_map.items()]
    feature_catalog_rows.append({"source": "places", "feature": "places_population", "source_column": "totalpopulation"})
    feature_catalog = pd.DataFrame(feature_catalog_rows)
    return output, feature_catalog


def normalize_svi(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = standardize_columns(pd.read_csv(raw_path, low_memory=False))
    fips_column = infer_fips_column(dataframe)
    dataframe["fips"] = standardize_fips(dataframe[fips_column])

    selected_columns = [
        "ep_pov150",
        "ep_nohsdp",
        "ep_unemp",
        "ep_uninsur",
        "ep_age65",
        "ep_age17",
        "ep_disabl",
        "ep_sngpnt",
        "ep_limeng",
        "ep_minrty",
        "ep_munit",
        "ep_mobile",
        "ep_crowd",
        "ep_noveh",
        "ep_groupq",
        "rpl_theme1",
        "rpl_theme2",
        "rpl_theme3",
        "rpl_theme4",
        "rpl_themes",
    ]
    existing_columns = select_existing_columns(dataframe, selected_columns)

    output = dataframe[["fips"] + existing_columns].copy()
    renamed_columns = {column_name: f"svi_{column_name}" for column_name in existing_columns}
    output = output.rename(columns=renamed_columns)
    output = output.drop_duplicates(subset="fips").reset_index(drop=True)

    feature_catalog = pd.DataFrame(
        [{"source": "svi", "feature": renamed_columns[column_name], "source_column": column_name} for column_name in existing_columns]
    )
    return output, feature_catalog


def normalize_rucc(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = standardize_columns(read_csv_with_fallback(raw_path, low_memory=False))
    fips_column = infer_fips_column(dataframe)
    dataframe["fips"] = standardize_fips(dataframe[fips_column])

    if {"attribute", "value"}.issubset(dataframe.columns):
        dataframe = dataframe.pivot_table(index="fips", columns="attribute", values="value", aggfunc="first").reset_index()
        dataframe = standardize_columns(dataframe)

    code_column_candidates = ["rucc_2023", "rucc_code_2023", "rucc_code", "rucc"]
    label_column_candidates = ["description", "description_2023", "county_description"]
    population_column_candidates = ["population_2020", "population"]
    code_columns = select_existing_columns(dataframe, code_column_candidates)
    label_columns = select_existing_columns(dataframe, label_column_candidates)
    population_columns = select_existing_columns(dataframe, population_column_candidates)

    selected_columns = ["fips"]
    renamed_columns: dict[str, str] = {}
    if code_columns:
        selected_columns.append(code_columns[0])
        renamed_columns[code_columns[0]] = "rucc_code_2023"
    if label_columns:
        selected_columns.append(label_columns[0])
        renamed_columns[label_columns[0]] = "rucc_description_2023"
    if population_columns:
        selected_columns.append(population_columns[0])
        renamed_columns[population_columns[0]] = "rucc_population_2020"

    output = dataframe[selected_columns].copy().rename(columns=renamed_columns)
    output = output.drop_duplicates(subset="fips").reset_index(drop=True)
    feature_catalog = pd.DataFrame(
        [{"source": "rucc", "feature": value, "source_column": key} for key, value in renamed_columns.items()]
    )
    return output, feature_catalog


def normalize_acs_group(raw_path: Path, group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = standardize_columns(pd.read_csv(raw_path, dtype=str))
    dataframe["fips"] = dataframe["state"].astype(str).str.zfill(2) + dataframe["county"].astype(str).str.zfill(3)

    metadata_rows = []
    keep_columns = ["fips"]
    rename_map = {}
    for column_name in dataframe.columns:
        if column_name in {"name", "state", "county", "fips"}:
            continue
        if not (column_name.endswith("e") or column_name.endswith("m")):
            continue
        keep_columns.append(column_name)
        renamed_name = f"acs_{group_name.lower()}_{column_name}"
        rename_map[column_name] = renamed_name
        metadata_rows.append({"source": f"acs_{group_name.lower()}", "feature": renamed_name, "source_column": column_name})

    output = dataframe[keep_columns].copy().rename(columns=rename_map)
    for column_name in output.columns:
        if column_name == "fips":
            continue
        output[column_name] = pd.to_numeric(output[column_name], errors="coerce")

    output = output.drop_duplicates(subset="fips").reset_index(drop=True)
    return output, pd.DataFrame(metadata_rows)


def run_downloads(root_dir: Path, acs_year: int, api_key: str | None) -> dict:
    raw_dir = root_dir / "data" / "public" / "raw"
    metadata_dir = root_dir / "data" / "public" / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "places": str(raw_dir / "places_county_2025.csv"),
        "svi": str(raw_dir / "svi_2022_county.csv"),
        "rucc": str(raw_dir / "rucc_2023.csv"),
        "acs_groups": {},
        "acs_metadata": {},
    }

    fetch_places_county(raw_dir / "places_county_2025.csv")
    fetch_arcgis_layer(SVI_COUNTY_QUERY_URL, raw_dir / "svi_2022_county.csv")
    download_to_path(RUCC_2023_CSV_URL, raw_dir / "rucc_2023.csv")

    for group_name in ACS_GROUPS:
        group_path = raw_dir / f"acs5_profile_{acs_year}_{group_name.lower()}.csv"
        metadata_path = metadata_dir / f"acs5_profile_{acs_year}_{group_name.lower()}_variables.csv"
        fetch_acs_group(acs_year, group_name, group_path, api_key)
        fetch_acs_group_metadata(acs_year, group_name, metadata_path)
        manifest["acs_groups"][group_name] = str(group_path)
        manifest["acs_metadata"][group_name] = str(metadata_path)

    manifest_path = metadata_dir / "phase2_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_phase2_outputs(root_dir: Path, acs_year: int, allow_missing: bool) -> dict:
    phase1_path = root_dir / "outputs" / "phase1" / "county_master_table.csv"
    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase 1 master table not found at {phase1_path}")

    raw_dir = root_dir / "data" / "public" / "raw"
    output_dir = root_dir / "outputs" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    county_master = pd.read_csv(phase1_path, dtype={"fips": str})
    feature_catalog_frames = []
    source_status = {}

    source_loaders = [
        ("places", raw_dir / "places_county_2025.csv", normalize_places),
        ("svi", raw_dir / "svi_2022_county.csv", normalize_svi),
        ("rucc", raw_dir / "rucc_2023.csv", normalize_rucc),
    ]
    for group_name in ACS_GROUPS:
        source_loaders.append(
            (f"acs_{group_name.lower()}", raw_dir / f"acs5_profile_{acs_year}_{group_name.lower()}.csv", lambda path, g=group_name: normalize_acs_group(path, g))
        )

    for source_name, source_path, loader in source_loaders:
        if not source_path.exists():
            source_status[source_name] = {"status": "missing", "path": str(source_path)}
            if not allow_missing:
                raise FileNotFoundError(f"Required Phase 2 source missing: {source_path}")
            continue

        normalized, feature_catalog = loader(source_path)
        normalized["fips"] = normalized["fips"].astype(str).str.zfill(5)
        county_master = county_master.merge(normalized, on="fips", how="left")
        feature_catalog_frames.append(feature_catalog)
        source_status[source_name] = {
            "status": "loaded",
            "path": str(source_path),
            "rows": int(normalized.shape[0]),
            "features": int(normalized.shape[1] - 1),
        }

    feature_catalog = pd.concat(feature_catalog_frames, ignore_index=True) if feature_catalog_frames else pd.DataFrame(columns=["source", "feature", "source_column"])
    feature_catalog.to_csv(output_dir / "phase2_feature_catalog.csv", index=False)
    county_master.to_csv(output_dir / "county_master_table_phase2.csv", index=False)

    summary = {
        "phase1_master_rows": int(pd.read_csv(phase1_path).shape[0]),
        "phase2_master_rows": int(county_master.shape[0]),
        "phase2_master_columns": int(county_master.shape[1]),
        "sources": source_status,
        "outputs": {
            "county_master_table_phase2": str(output_dir / "county_master_table_phase2.csv"),
            "phase2_feature_catalog": str(output_dir / "phase2_feature_catalog.csv"),
        },
    }
    (output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and merge Phase 2 public county-level data sources.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--acs-year", type=int, default=2024, help="ACS 5-year profile release year to query.")
    parser.add_argument("--acs-api-key", type=str, default=None, help="Optional Census API key.")
    parser.add_argument("--download", action="store_true", help="Download public source files into data/public/raw.")
    parser.add_argument("--build", action="store_true", help="Build the Phase 2 merged county master table.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow build to proceed when some public files are not yet downloaded.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_downloads_flag = args.download or (not args.download and not args.build)
    run_build_flag = args.build or (not args.download and not args.build)

    result = {}
    if run_downloads_flag:
        result["download"] = run_downloads(args.root_dir, args.acs_year, args.acs_api_key)
    if run_build_flag:
        result["build"] = build_phase2_outputs(args.root_dir, args.acs_year, args.allow_missing)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
