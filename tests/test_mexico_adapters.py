"""Synthetic only: no network, public payloads, accelerator or campaign."""
import csv
from dataclasses import replace
from hashlib import sha256
from itertools import product
import math
from pathlib import Path
import socket

import pytest
import yaml

from oxyformer.contracts import SourceManifest
from oxyformer.provenance import ContractError, file_hash
from oxyformer.data.adapters.conapo import (
    AGE_COLUMNS, AGE_GROUPS, CONAPO_MAPPING, SEXES, AdapterError, load_denominators,
)
from oxyformer.data.adapters.inegi_edr import (
    EDR_MAPPING, RegistrationRelease, ReviewedCrosswalk, build_cells, load_events,
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network is forbidden in adapter tests")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def write_source(path, rows, columns, mapping, version):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return SourceManifest(source_id=version, version=version,
                          uri="synthetic://fixture", payload_hash=file_hash(path),
                          license_hash=sha256(b"synthetic").hexdigest(),
                          schema_hash=sha256(repr(columns).encode()).hexdigest(),
                          field_mapping=tuple(mapping), mapping_status="reviewed",
                          mapping_review_id="synthetic-dictionary-review")


def event(year=2015, registration=2015, municipality="01001", **changes):
    row = dict(ENT_RESID=municipality[:2], MUN_RESID=municipality[2:],
               ANIO_OCUR=str(year), ANIO_REGIS=str(registration), EDAD="4042",
               SEXO="1", CAUSA_DEF="I219", ENT_OCURR="02", MUN_OCURR="002")
    row.update(changes)
    return row


def releases(tmp_path, rows=None, *, through=2017, coverage=("01001", "01002"),
             vintage="same", missing=()):
    rows = [event()] if rows is None else rows
    result = []
    for year in range(2015, through + 1):
        if year in missing:
            continue
        path = tmp_path / f"edr-{year}.csv"
        manifest = write_source(path, [r for r in rows if int(r["ANIO_REGIS"]) == year],
                                list(event()), EDR_MAPPING, f"edr-{year}-final")
        result.append(RegistrationRelease(str(path), manifest, year, vintage,
                                          tuple(coverage), "synthetic-coverage-review"))
    return result


def population_rows(municipalities=("01001", "01002"), years=(2015,)):
    rows = []
    for municipality, year, sex in product(municipalities, years, ("HOMBRES", "MUJERES")):
        row = dict(CLAVE=str(int(municipality)), CLAVE_ENT=str(int(municipality[:2])),
                   NOM_ENT="Synthetic state", NOM_MUN="Synthetic municipality",
                   SEXO=sex, ANO=str(year), POB_TOTAL="1800", fecha=f"{year}-01-01",
                   etiqueta_estado="Synthetic state")
        row.update({c: "100" for c in AGE_COLUMNS})
        rows.append(row)
    return rows


def denominators(tmp_path, rows=None, *, municipalities=("01001", "01002"),
                 years=(2015,), vintage="same", columns=None):
    rows = population_rows(municipalities, years) if rows is None else rows
    columns = list(population_rows()[0]) if columns is None else columns
    path = tmp_path / "conapo.csv"
    manifest = write_source(path, rows, columns, CONAPO_MAPPING, "conapo-reviewed-revision")
    return load_denominators(path, manifest, municipalities=municipalities,
                             years=years, geography_vintage=vintage)


def loaded(tmp_path, rows=None, **kw):
    return load_events(releases(tmp_path, rows, **kw), occurrence_years=(2015,))


def test_complete_grid_has_genuine_zero_deaths_and_conserves_counts(tmp_path):
    table = build_cells(loaded(tmp_path), denominators(tmp_path))
    assert len(table.cells) == 2 * 2 * 18
    assert sum(c.deaths for c in table.cells) == 1
    assert sum(c.genuine_zero for c in table.cells) == 71
    assert all(c.deaths == 0 and c.genuine_zero for c in table.cells if c.municipality == "01002")
    assert table.audit["population_sum"] == 7200
    assert table.audit["included_events"] == table.audit["allocated_deaths"] == 1


def test_absent_coverage_is_never_zero_filled(tmp_path):
    source = releases(tmp_path, coverage=("01001",))
    events = load_events(source, occurrence_years=(2015,))
    with pytest.raises(AdapterError, match="missing source coverage") as error:
        build_cells(events, denominators(tmp_path))
    assert error.value.audit["missing_coverage"] == [
        ("01002", 2015, 2015, "01002"), ("01002", 2015, 2016, "01002"),
        ("01002", 2015, 2017, "01002")]


def test_one_missing_release_coverage_still_blocks_empty_events(tmp_path):
    source = releases(tmp_path, [])
    source[1] = replace(source[1], covered_municipalities=("01001",))
    with pytest.raises(AdapterError, match="missing source coverage"):
        build_cells(load_events(source, occurrence_years=(2015,)), denominators(tmp_path))


def test_empty_but_covered_release_gives_true_zeros(tmp_path):
    table = build_cells(loaded(tmp_path, []), denominators(tmp_path))
    assert len(table.cells) == 72 and all(c.genuine_zero for c in table.cells)


def test_occurrence_and_registration_stay_distinct_and_filter_reproducibly(tmp_path):
    rows = [event(registration=r) for r in range(2015, 2022)]
    rows += [event(year=2014), event(year=9999)]
    source = releases(tmp_path, rows, through=2021)
    events = load_events(source, occurrence_years=(2015,))
    repeat = load_events(reversed(source), occurrence_years=(2015,))
    assert events.events == repeat.events
    assert events.audit["selection_counts"] == {
        "included": 3, "outside_occurrence_years": 1, "unknown_occurrence_year": 1,
        "late_registration": 4}
    assert sum(events.audit["selection_counts"].values()) == len(rows)
    retained = [e for e in events.events if e.selection == "included"]
    assert [(e.occurrence_year, e.registration_year) for e in retained] == [(2015, 2015), (2015, 2016), (2015, 2017)]
    extended = load_events(source, occurrence_years=(2015,), lag=5)
    assert extended.audit["selection_counts"]["included"] == 6
    assert extended.audit["selection_counts"]["late_registration"] == 1


@pytest.mark.parametrize("lag,through,missing", [(2, 2021, 2021), (5, 2024, 2024), (2, 2021, 2016)])
def test_required_registration_release_absence_blocks(tmp_path, lag, through, missing):
    with pytest.raises(AdapterError, match="missing required registration releases") as error:
        load_events(releases(tmp_path, [], through=through, missing=(missing,)), lag=lag)
    assert error.value.audit["missing_registration_years"] == [missing]


def test_common_lag_for_all_years(tmp_path):
    rows = [event(year=y, registration=y + lag) for y in range(2015, 2020) for lag in (0, 2, 3)]
    source = releases(tmp_path, rows, through=2022)
    table = load_events(source)
    assert table.audit["selection_counts"] == {"included": 10, "late_registration": 5}
    assert table.audit["required_registration_years"] == list(range(2015, 2022))


@pytest.mark.parametrize("lag", [0, 1, 3, -1, True, 2.0])
def test_unapproved_lag_refused(tmp_path, lag):
    with pytest.raises(AdapterError, match="approved lag"):
        load_events(releases(tmp_path), occurrence_years=(2015,), lag=lag)


def test_unknowns_retained_without_redistribution_or_false_zero(tmp_path):
    rows = [event(), event(EDAD="4998"), event(SEXO="9"),
            event(EDAD="4998", SEXO="9", CAUSA_DEF="")]
    events = loaded(tmp_path, rows)
    table = build_cells(events, denominators(tmp_path))
    assert events.audit["unknown_categories"] == {
        "unknown_age": 2, "unknown_sex": 2, "unknown_age_or_sex": 3, "missing_cause": 1}
    assert len(table.unallocated_events) == 3
    assert sum(c.deaths for c in table.cells) == 1
    assert table.audit["allocated_deaths"] + table.audit["unallocated_deaths"] == 4
    assert all(not c.genuine_zero and c.unallocated_deaths == 3
               for c in table.cells if c.municipality == "01001")
    assert all(c.genuine_zero for c in table.cells if c.municipality == "01002")
    assert dict(events.events[0].source_values)["ENT_OCURR"] == "02"
    assert events.events[0].source_municipality == "01001"
    assert events.events[0].cause_code == "I219"


@pytest.mark.parametrize("code,age", [("1001", "00-04"), ("1097", "00-04"), ("1098", "00-04"),
                                     ("2098", "00-04"), ("3098", "00-04"), ("4004", "00-04"),
                                     ("4005", "05-09"), ("4084", "80-84"), ("4085", "85+"),
                                     ("4120", "85+"), ("4998", None)])
def test_actual_age_catalog_boundaries(tmp_path, code, age):
    assert loaded(tmp_path, [event(EDAD=code)]).events[0].age_group == age


def test_identical_public_rows_are_distinct_deaths(tmp_path):
    events = loaded(tmp_path, [event(), event()])
    assert len({e.source_record_id for e in events.events}) == 2
    assert sum(c.deaths for c in build_cells(events, denominators(tmp_path)).cells) == 2


def test_duplicate_release_and_revised_release_cannot_stack(tmp_path):
    source = releases(tmp_path)
    for extra in (source[0], replace(source[0], manifest=replace(source[0].manifest, version="revision2"))):
        with pytest.raises(AdapterError, match="duplicate registration release"):
            load_events([*source, extra], occurrence_years=(2015,))


def test_revision_hash_and_mapping_required(tmp_path):
    source = releases(tmp_path)
    with Path(source[0].path).open("a") as out:
        out.write("\n")
    with pytest.raises(AdapterError, match="release revision changed"):
        load_events(source, occurrence_years=(2015,))
    source = releases(tmp_path)
    source[0] = replace(source[0], manifest=replace(source[0].manifest, mapping_status="unreviewed"))
    with pytest.raises(ContractError, match="unreviewed"):
        load_events(source, occurrence_years=(2015,))


def test_denominator_revision_change_refused(tmp_path):
    table = denominators(tmp_path)
    path = tmp_path / "conapo.csv"
    path.write_text(path.read_text().replace("1800", "1801", 1))
    with pytest.raises(AdapterError, match="release revision changed"):
        load_denominators(path, table.manifest, municipalities=table.municipalities,
                          years=(2015,), geography_vintage="same")


def test_real_archive_lowercase_headers_are_supported_without_losing_source_fields(tmp_path):
    source = releases(tmp_path)
    path = Path(source[0].path)
    row = {k.lower(): v for k, v in event().items()}
    manifest = write_source(path, [row], list(row), EDR_MAPPING, "edr-lowercase")
    source[0] = replace(source[0], manifest=manifest)
    table = load_events(source, occurrence_years=(2015,))
    assert table.events[0].source_municipality == "01001"
    assert dict(table.events[0].source_values)["ent_resid"] == "01"


@pytest.mark.parametrize("changes,match", [({"EDAD": "4000"}, "age code"),
                                          ({"SEXO": "3"}, "sex code"),
                                          ({"ANIO_OCUR": "2016"}, "follows registration")])
def test_bad_source_codes_refused(tmp_path, changes, match):
    with pytest.raises(AdapterError, match=match):
        loaded(tmp_path, [event(**changes)])


def test_registration_year_is_checked_against_release(tmp_path):
    source = releases(tmp_path, [event()])
    source[0] = replace(source[0], registration_year=2018)
    with pytest.raises(AdapterError, match="registration year differs"):
        load_events(source, occurrence_years=(2016,))


@pytest.mark.parametrize("population", ["0", "-1", "nan", "inf", "", "10.5"])
def test_denominator_failures(tmp_path, population):
    rows = population_rows()
    rows[0][AGE_COLUMNS[0]] = population
    if population == "0":
        rows[0]["POB_TOTAL"] = "1700"
    with pytest.raises(AdapterError):
        denominators(tmp_path, rows)


def test_missing_denominator_row_does_not_shrink_target(tmp_path):
    with pytest.raises(AdapterError, match="missing denominator coverage") as error:
        denominators(tmp_path, population_rows()[:-1])
    assert error.value.audit["missing_rows"] == [("01002", 2015, "female")]


def test_denominator_duplicates_and_totals(tmp_path):
    rows = population_rows()
    with pytest.raises(AdapterError, match="duplicate denominator"):
        denominators(tmp_path, [*rows, rows[0]])
    rows[0]["POB_TOTAL"] = "1801"
    with pytest.raises(AdapterError, match="conserve total"):
        denominators(tmp_path, rows)


def test_incompatible_age_bins_and_sex_refused(tmp_path):
    rows = population_rows()
    for row in rows:
        row["POB_00_09"] = row.pop(AGE_COLUMNS[0])
    with pytest.raises(AdapterError, match="incompatible age bins"):
        denominators(tmp_path, rows, columns=list(rows[0]))
    rows = population_rows()
    rows[0]["SEXO"] = "TOTAL"
    with pytest.raises(AdapterError, match="sex category"):
        denominators(tmp_path, rows)


def test_unmatched_geography_reports_every_code_even_unknown_categories(tmp_path):
    rows = [event(municipality="01999"), event(municipality="99000", EDAD="4998"),
            event(municipality="01003"), event(municipality="01003")]
    with pytest.raises(AdapterError, match="unmatched municipality") as error:
        build_cells(loaded(tmp_path, rows), denominators(tmp_path))
    assert error.value.audit["unmatched_events"] == [
        ("same", "01003", 2), ("same", "01999", 1), ("same", "99000", 1)]


def test_boundary_change_blocks_even_when_codes_match(tmp_path):
    with pytest.raises(AdapterError, match="incompatible geographic vintages"):
        build_cells(loaded(tmp_path, vintage="edr-2015"), denominators(tmp_path, vintage="conapo-2022"))


def union_crosswalk():
    return ReviewedCrosswalk((("old", "01001", "stable-union"),
                              ("new", "01001", "stable-union"),
                              ("new", "01002", "stable-union")), "union-v1", "synthetic-review")


def test_reviewed_union_sums_both_denominators_and_counts(tmp_path):
    events = loaded(tmp_path, [event(), event(SEXO="2")], vintage="old", coverage=("01001",))
    table = build_cells(events, denominators(tmp_path, vintage="new"), crosswalk=union_crosswalk())
    assert len(table.cells) == 36
    assert all(c.population == 200 and c.municipality == "stable-union" for c in table.cells)
    assert sum(c.deaths for c in table.cells) == 2
    assert table.audit["population_sum"] == 7200
    assert table.audit["crosswalk_review"] == "synthetic-review"
    assert len(table.audit["crosswalk_hash"]) == 64


def test_partial_denominator_union_refused(tmp_path):
    events = loaded(tmp_path, vintage="old", coverage=("01001",))
    den = denominators(tmp_path, municipalities=("01001",), vintage="new")
    with pytest.raises(AdapterError, match="incomplete stable-union membership"):
        build_cells(events, den, crosswalk=union_crosswalk())


def test_union_requires_all_event_members_even_with_zero_deaths(tmp_path):
    crosswalk = ReviewedCrosswalk((("old", "01001", "union"), ("old", "01002", "union"),
                                  ("new", "01001", "union"), ("new", "01002", "union")),
                                 "union-v1", "synthetic-review")
    with pytest.raises(AdapterError, match="missing source coverage"):
        build_cells(loaded(tmp_path, vintage="old", coverage=("01001",)),
                    denominators(tmp_path, vintage="new"), crosswalk=crosswalk)


@pytest.mark.parametrize("bad", [replace(union_crosswalk(), review_id=""),
                                replace(union_crosswalk(), rows=(*union_crosswalk().rows,
                                                               ("old", "01001", "second-union")))])
def test_unreviewed_or_splitting_crosswalk_refused(tmp_path, bad):
    with pytest.raises(AdapterError, match="crosswalk"):
        build_cells(loaded(tmp_path, vintage="old"), denominators(tmp_path, vintage="new"), crosswalk=bad)


def test_denominator_year_and_state_checks(tmp_path):
    events = loaded(tmp_path)
    with pytest.raises(AdapterError, match="years differ"):
        build_cells(events, denominators(tmp_path, years=(2016,)))
    rows = population_rows()
    rows[0]["CLAVE_ENT"] = "2"
    with pytest.raises(AdapterError, match="municipality/state"):
        denominators(tmp_path, rows)


def test_target_mass_age_weights_and_poisson_exposure_are_separate(tmp_path):
    events, den = loaded(tmp_path), denominators(tmp_path)
    mass = {(d.municipality, d.year, d.age_group, d.sex): 3.0 for d in den.cells}
    standard = {age: 1 / 18 for age in AGE_GROUPS}
    result = build_cells(events, den, target_mass=mass, age_standardization_weights=standard)
    for cell in result.cells:
        assert cell.target_mass == 3.0
        assert cell.age_standardization_weight == 1 / 18
        assert cell.population == 100
        assert cell.poisson_log_offset == pytest.approx(math.log(100))
        assert cell.poisson_loss_multiplier == 0.03
    plain = build_cells(events, den)
    assert all(c.target_mass is None and c.age_standardization_weight is None
               and c.poisson_loss_multiplier is None for c in plain.cells)


def test_invalid_weights_and_death_only_input_cannot_become_population_risk(tmp_path):
    events, den = loaded(tmp_path), denominators(tmp_path)
    with pytest.raises(AdapterError, match="both required"):
        build_cells(events, None)
    with pytest.raises(AdapterError, match="target mass"):
        build_cells(events, den, target_mass={})
    with pytest.raises(AdapterError, match="age-standardization"):
        build_cells(events, den, age_standardization_weights={a: 100 for a in AGE_GROUPS})


def test_manifest_and_config_match_owner_decisions():
    path = Path(__file__).parents[1] / "configs/adapters/mexico.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["registration_windows"]["primary"]["lag"] == 2
    assert config["registration_windows"]["late_registration_check"]["lag"] == 5
    assert tuple(c.upper() for c in config["conapo"]["age_columns"]) == AGE_COLUMNS
