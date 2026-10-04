"""Offline synthetic observation-model regressions; no network or study data."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import io
import math

import pandas as pd
import pytest
import yaml

from oxyformer.contracts import SourceManifest
from oxyformer.data.adapters.acs import load_acs, predictor_view
from oxyformer.data.adapters.usaleep import SourceFile, join_outcomes, load_usaleep, mapping_hash
from oxyformer.data.feature_roles import FeatureRule

ROOT = Path(__file__).resolve().parents[1]
IDS = ["01001000100", "01001000200", "01001000300"]
HEADER = ["Tract ID", "STATE2KX", "CNTY2KX", "TRACT2KX", "e(0)", "se(e(0))", "Abridged life table flag"]
# Independent fixture layout from the inspected historical dictionary. These
# constants deliberately do not read sequence positions from the adapter YAML.
TABLES = {"B01001": ("0010", 7, 49), "B03002": ("0013", 38, 21),
          "B15002": ("0040", 90, 35), "C17002": ("0046", 219, 8),
          "B19001": ("0053", 7, 17), "B19013": ("0053", 177, 1),
          "B19301": ("0059", 172, 1)}
WIDTHS = {"0010": 179, "0013": 61, "0040": 223, "0046": 226, "0053": 204, "0059": 191}


@pytest.fixture
def mapping():
    return yaml.safe_load((ROOT / "configs/adapters/us.yaml").read_text())


def csv_text(rows):
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue()


def source(tmp_path, name, text, section, source_id):
    path = tmp_path / name
    payload = text.encode(section["encoding"])
    path.write_bytes(payload)
    manifest = SourceManifest(source_id=source_id, version=section["release"],
                              uri="https://example.invalid/synthetic/" + name,
                              payload_hash=sha256(payload).hexdigest(), license_hash="a" * 64,
                              schema_hash=mapping_hash(section), field_mapping=tuple(section["field_mapping"].items()),
                              mapping_status="reviewed", mapping_review_id="synthetic-test-layout")
    return SourceFile(path, manifest)


def us_bundle(tmp_path, mapping, edit=None):
    rows = [[x, x[:2], x[2:5], x[5:], str(70 + i), "0.5", str(i + 1)] for i, x in enumerate(IDS)]
    if edit:
        edit(rows)
    return [source(tmp_path, "US_A.CSV", csv_text([HEADER] + rows), mapping["usaleep"], "usaleep_2010_2015")]


def geo_line(oid, logrecno, state="al", level="140"):
    line = list(" " * 475)
    for start, width, value in [(1, 6, "ACSSF"), (7, 2, state), (9, 3, level), (12, 2, "00"),
                                (14, 7, logrecno), (26, 2, oid[:2]), (28, 3, oid[2:5]),
                                (41, 6, oid[5:11]), (179, 40, level + "00US" + oid)]:
        line[start - 1:start - 1 + width] = value.ljust(width)
    return "".join(line)


def acs_bundle(tmp_path, mapping, edits=None, ids=IDS, state="al"):
    geos = [geo_line(oid, f"{i+1:07}", state) for i, oid in enumerate(ids)]
    geos.append(geo_line(ids[0] + "1", f"{len(ids)+1:07}", state, "150"))
    raw = {f"g20105{state}.txt": "\n".join(geos) + "\n"}
    for seq, width in WIDTHS.items():
        for kind in ("e", "m"):
            rows = []
            for index in range(len(geos)):
                row = ["ACSSF", f"2010{kind}5", state, "000", seq, f"{index+1:07}"] + ["0"] * (width - 6)
                for table, (number, start, count) in TABLES.items():
                    if number != seq:
                        continue
                    vals = ["1000"] + ["10"] * (count - 1)
                    if table == "B19013":
                        vals = ["50000"]
                    elif table == "B19301":
                        vals = ["25000"]
                    if kind == "m":
                        vals = ["2"] * count
                    row[start - 1:start - 1 + count] = vals
                rows.append(row)
            raw[f"{kind}20105{state}{seq}000.txt"] = csv_text(rows)
    if edits:
        edits(raw)
    return [source(tmp_path, name, text, mapping["acs"], "acs_2006_2010_5yr:" + name)
            for name, text in raw.items()]


def cell_edit(raw, table, line, value, kind="e", row=0):
    seq, start, _ = TABLES[table]
    name = f"{kind}20105al{seq}000.txt"
    rows = list(csv.reader(io.StringIO(raw[name])))
    rows[row][start + line - 2] = value
    raw[name] = csv_text(rows)


def test_only_observed_input_labels_are_primary(tmp_path, mapping):
    primary, audit = load_usaleep(us_bundle(tmp_path, mapping), mapping)
    assert primary.original_id.tolist() == [IDS[0]]
    assert primary.life_expectancy_years.tolist() == [70]
    assert audit["excluded_rows"] == 2
    assert audit["metadata"].mortality_input_kind.tolist() == ["observed", "predicted", "mixed"]
    assert audit["metadata"].standard_error_years.tolist() == [0.5] * 3
    assert audit["metadata"].exclusion_reason.tolist() == ["", "predicted_mortality_inputs", "mixed_mortality_inputs"]
    for field in audit["metadata"].columns:
        for use in ("ssl", "nuisance", "context"):
            with pytest.raises(ValueError):
                audit["registry"].require(field, "usaleep_life_expectancy", use)


def test_exact_ids_and_separate_frames(tmp_path, mapping):
    covariates, audit = load_acs(acs_bundle(tmp_path, mapping), mapping)
    outcomes, _ = load_usaleep(us_bundle(tmp_path, mapping), mapping)
    before = covariates.copy(deep=True)
    joined, join_audit = join_outcomes(covariates, outcomes)
    assert covariates.original_id.tolist() == IDS
    assert joined.original_id.tolist() == [IDS[0]]
    assert join_audit["acs_without_primary_outcome"] == tuple(IDS[1:])
    pd.testing.assert_frame_equal(covariates, before)
    assert len(covariates.columns) == 31
    assert len(audit["excluded_geographies"]) == 1  # no block-group replication
    assert covariates.female_share.tolist() == [0.01] * 3
    assert covariates.age_under_18_share.tolist() == [0.08] * 3
    assert covariates.education_bachelors_plus_share.tolist() == [0.08] * 3
    assert covariates.poverty_ratio_100_199_share.tolist() == [0.04] * 3
    assert covariates.median_household_income.tolist() == [math.asinh(5)] * 3
    assert set(predictor_view(covariates, audit).columns) == set(mapping["acs"]["concepts"])
    for field in ("life_expectancy_years", "mortality_input_flag", "primary_label_available", "moe", "original_id", "county_fips"):
        with pytest.raises(ValueError):
            predictor_view(covariates, audit, [field])


@pytest.mark.parametrize("side", ["covariates", "outcomes"])
def test_duplicate_outcome_join_refused(tmp_path, mapping, side):
    covariates, _ = load_acs(acs_bundle(tmp_path, mapping), mapping)
    outcomes, _ = load_usaleep(us_bundle(tmp_path, mapping), mapping)
    if side == "covariates":
        covariates = pd.concat([covariates, covariates.iloc[:1]])
    else:
        outcomes = pd.concat([outcomes, outcomes])
    with pytest.raises(ValueError, match="duplicate"):
        join_outcomes(covariates, outcomes)


@pytest.mark.parametrize("column,value", [(0, "1001000100"), (1, "1"), (2, "002"),
                                           (4, ""), (4, "."), (4, "-999999999"),
                                           (4, "nan"), (5, "-1"), (6, "4"), (6, "1.0")])
def test_usaleep_unresolved_values_fail_closed(tmp_path, mapping, column, value):
    bundle = us_bundle(tmp_path, mapping, lambda rows: rows[0].__setitem__(column, value))
    with pytest.raises(ValueError):
        load_usaleep(bundle, mapping)


def test_usaleep_duplicate_tract_refused(tmp_path, mapping):
    bundle = us_bundle(tmp_path, mapping, lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(ValueError, match="duplicate"):
        load_usaleep(bundle, mapping)


@pytest.mark.parametrize("mutation", ["flags", "primary_flags", "units", "geography_vintage"])
def test_unresolved_mortality_mapping_refused(tmp_path, mapping, mutation):
    bad = deepcopy(mapping)
    bad["usaleep"][mutation] = {"1": "predicted", "2": "observed", "3": "mixed"} if mutation == "flags" else ["1", "2"] if mutation == "primary_flags" else "days" if mutation == "units" else 2020
    with pytest.raises(ValueError):
        load_usaleep(us_bundle(tmp_path, bad), bad)


@pytest.mark.parametrize("token", ["", "."])
def test_missing_estimates_retained_in_audit(tmp_path, mapping, token):
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B01001", 26, token))
    covariates, audit = load_acs(bundle, mapping)
    assert pd.isna(covariates.female_share.iloc[0])
    row = audit["cells"].query("original_id == @IDS[0] and variable == 'B01001_026'").iloc[0]
    assert row.raw_estimate == token and row.estimate_annotation
    assert len(covariates) == 3


def test_zero_denominator_is_missing(tmp_path, mapping):
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B01001", 1, "0"))
    covariates, audit = load_acs(bundle, mapping)
    assert pd.isna(covariates.female_share.iloc[0])
    assert (IDS[0], "female_share", "unavailable_or_zero_denominator") in audit["missing_concepts"]


@pytest.mark.parametrize("token", ["2499", "250001"])
def test_income_jam_values_are_metadata(tmp_path, mapping, token):
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B19013", 1, token))
    covariates, audit = load_acs(bundle, mapping)
    assert pd.isna(covariates.median_household_income.iloc[0])
    assert token in audit["cells"].raw_estimate.tolist()
    assert audit["cells"].estimate_annotation.str.startswith("income_").any()


@pytest.mark.parametrize("token", ["", ".", "0"])
def test_moe_annotations_do_not_change_predictors(tmp_path, mapping, token):
    reference, _ = load_acs(acs_bundle(tmp_path, mapping), mapping)
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B01001", 26, token, kind="m"))
    covariates, audit = load_acs(bundle, mapping)
    pd.testing.assert_frame_equal(covariates, reference)
    assert audit["cells"].moe_annotation.ne("").any()


@pytest.mark.parametrize("token", ["-666666666", "-1", "(X)", "NaN", "inf"])
def test_foreign_or_unknown_special_values_refused(tmp_path, mapping, token):
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B01001", 26, token))
    with pytest.raises(ValueError):
        load_acs(bundle, mapping)


def test_duplicate_geography_join_refused(tmp_path, mapping):
    def edit(raw):
        raw["g20105al.txt"] += raw["g20105al.txt"].splitlines()[0] + "\n"
    with pytest.raises(ValueError, match="duplicate geography"):
        load_acs(acs_bundle(tmp_path, mapping, edit), mapping)


def test_distinct_geography_keys_cannot_duplicate_tracts(tmp_path, mapping):
    with pytest.raises(ValueError, match="duplicate tract"):
        load_acs(acs_bundle(tmp_path, mapping, ids=[IDS[0], IDS[0]]), mapping)


@pytest.mark.parametrize("kind", ["e", "m"])
def test_duplicate_sequence_join_refused(tmp_path, mapping, kind):
    def edit(raw):
        name = f"{kind}20105al0010000.txt"
        raw[name] += raw[name].splitlines()[0] + "\n"
    with pytest.raises(ValueError, match="duplicate ACS sequence"):
        load_acs(acs_bundle(tmp_path, mapping, edit), mapping)


def test_state_scopes_logrecno(tmp_path, mapping):
    bundle = acs_bundle(tmp_path, mapping)
    bundle += acs_bundle(tmp_path, mapping, ids=["48001000100"], state="tx")
    frame, _ = load_acs(bundle, mapping)
    assert frame.original_id.tolist() == IDS + ["48001000100"]


@pytest.mark.parametrize("missing", ["e20105al0010000.txt", "m20105al0053000.txt", "g20105al.txt"])
def test_missing_source_tables_refused(tmp_path, mapping, missing):
    def edit(raw):
        del raw[missing]
    with pytest.raises(ValueError, match="missing source"):
        load_acs(acs_bundle(tmp_path, mapping, edit), mapping)


def test_missing_table_rows_refused(tmp_path, mapping):
    def edit(raw):
        for kind in ("e", "m"):
            name = f"{kind}20105al0010000.txt"
            raw[name] = "\n".join(raw[name].splitlines()[1:]) + "\n"
    with pytest.raises(ValueError, match="missing source table rows"):
        load_acs(acs_bundle(tmp_path, mapping, edit), mapping)


@pytest.mark.parametrize("mutation", ["role", "unapproved", "definition", "table", "family"])
def test_invalid_feature_registry_refused(tmp_path, mapping, mutation):
    bad = deepcopy(mapping)
    concepts = bad["acs"]["concepts"]
    if mutation == "role":
        concepts["female_share"]["role"] = "outcome_metadata"
    elif mutation == "unapproved":
        concepts["unapproved_health"] = concepts["female_share"].copy()
    elif mutation == "definition":
        concepts["female_share"]["num"] = [2]
    elif mutation == "table":
        del bad["acs"]["tables"]["B01001"]
    else:
        bad["acs"]["masking_families"]["age_structure"] = ["female_share"]
    with pytest.raises(ValueError):
        load_acs(acs_bundle(tmp_path, bad), bad)


def test_merged_roles_deny_metadata_predictors():
    with pytest.raises(ValueError):
        FeatureRule(name="moe", role="outcome_metadata", endpoints=("usaleep_life_expectancy",),
                    uses=("ssl",), approval_id="synthetic")


def test_payload_and_mapping_hashes_bound(tmp_path, mapping):
    bundle = us_bundle(tmp_path, mapping)
    bundle[0].path.write_bytes(bundle[0].path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="payload hash"):
        load_usaleep(bundle, mapping)
    bundle = us_bundle(tmp_path, mapping)
    altered = deepcopy(mapping)
    altered["usaleep"]["inspection"] += " modified"
    with pytest.raises(ValueError, match="mapping hash"):
        load_usaleep(bundle, altered)


def test_unreviewed_source_manifest_refused(tmp_path, mapping):
    from dataclasses import replace
    bundle = us_bundle(tmp_path, mapping)
    bundle[0] = SourceFile(bundle[0].path, replace(bundle[0].manifest, mapping_status="unreviewed", mapping_review_id=None))
    with pytest.raises(ValueError, match="unreviewed"):
        load_usaleep(bundle, mapping)


def test_signed_per_capita_income_retained(tmp_path, mapping):
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B19301", 1, "-500"))
    frame, audit = load_acs(bundle, mapping)
    assert frame.per_capita_income.iloc[0] == math.asinh(-0.05)
    assert -500 in audit["cells"].estimate.tolist()


def test_numeric_join_ids_are_not_coerced():
    with pytest.raises(ValueError, match="strings required"):
        join_outcomes(pd.DataFrame({"original_id": [1001000100]}),
                      pd.DataFrame({"original_id": [IDS[0]], "life_expectancy_years": [70]}))


def test_moe_missing_tract_cannot_silently_drop_covariate_row(tmp_path, mapping):
    def edit(raw):
        raw["m20105al0010000.txt"] = "\n".join(raw["m20105al0010000.txt"].splitlines()[1:]) + "\n"
    with pytest.raises(ValueError, match="cardinality"):
        load_acs(acs_bundle(tmp_path, mapping, edit), mapping)


def test_acs_wrong_release_refused(tmp_path, mapping):
    def edit(raw):
        raw["e20105al0010000.txt"] = raw["e20105al0010000.txt"].replace("2010e5", "2024e5")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_acs(acs_bundle(tmp_path, mapping, edit), mapping)


def test_old_income_boundary_is_not_a_2010_sentinel(tmp_path, mapping):
    bundle = acs_bundle(tmp_path, mapping, lambda raw: cell_edit(raw, "B19013", 1, "200001"))
    frame, audit = load_acs(bundle, mapping)
    assert frame.median_household_income.iloc[0] == math.asinh(200001 / 10000)
    assert not audit["cells"].estimate_annotation.str.startswith("income_").any()


def test_unused_geography_name_bytes_do_not_break_ids(tmp_path, mapping):
    from dataclasses import replace
    bundle = acs_bundle(tmp_path, mapping)
    geo = bundle[0]
    payload = bytearray(geo.path.read_bytes())
    payload[220] = 0xf1  # A source-like single-byte accented name outside ID fields.
    geo.path.write_bytes(payload)
    bundle[0] = SourceFile(geo.path, replace(geo.manifest, payload_hash=sha256(payload).hexdigest()))
    frame, _ = load_acs(bundle, mapping)
    assert frame.original_id.tolist() == IDS
