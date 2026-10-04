"""Synthetic ZIP releases only. No downloads, GPU work or real survey records."""
from copy import deepcopy
import csv
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import socket
import zipfile

import pytest

from oxyformer.contracts import EstimandSpec, SourceManifest, source_lineage_hash
from oxyformer.data.adapters import endes
from oxyformer.data.adapters.endes import EndesAudit, EndesBundle, EndesPerson, load_endes
from oxyformer.provenance import ContractError, canonical_json, file_hash


def digest(value):
    return sha256(canonical_json(value).encode()).hexdigest()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ENDES tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def release(tmp_path, monkeypatch):
    config = endes._config()
    catalog = deepcopy(endes._release_catalog())
    monkeypatch.setattr(endes, "_release_catalog", lambda: deepcopy(catalog))
    sequence = 0

    def make(year=2023, mutate=None, mapping=None, nested=None):
        nonlocal sequence
        sequence += 1
        mapping = deepcopy(config["years"][year] if mapping is None else mapping)
        frame_field = "NCONGLOME" if year == 2023 else "NCONGLOME1"
        tables = {m: [{"ID1": str(year)}] for m in config["modules"]}
        households, roster, children = [], [], []
        for i in range(1, 5):
            hh, cluster = f"000{i}0000001", f"00{i}"
            households.append(dict(
                ID1=str(year), HHID=hh, HV001=cluster, HV021=cluster, HV022="01" if i <= 2 else "02",
                HV024="01", UBIGEO=f"01010{i}", HV040=str(100 * i), HV005=str(i * 1000000),
                HV015="1", HV042="1", HV007=str(year), HV033="00012345", **{frame_field: f"000000{i}"},
            ))
            roster.append(dict(ID1=str(year), HHID=hh, HVIDX="01", HV102="1", HV103="1", HV105="2", HV120="1"))
            children.append(dict(ID1=str(year), HHID=hh, HC0="01", HC1="24", HC53="120", HC55="0", HC56="100"))
            if year == 2024:
                children[-1]["HC56A"] = "90"
        # An adult shares the first child's household; never receives child weight.
        roster.append(dict(ID1=str(year), HHID=households[0]["HHID"], HVIDX="02", HV102="1", HV103="1", HV105="35", HV120="0"))
        tables.update(RECH0=households, RECH1=roster, RECH6=children)
        if mutate is not None:
            mutate(tables)
        headers, payloads = {}, {}
        for module, rows in tables.items():
            keys = list(rows[0])
            headers[module] = keys
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            payloads[module] = buffer.getvalue().encode()
        path = tmp_path / f"synthetic-{sequence}-{year}.zip"
        with zipfile.ZipFile(path, "w") as zipped:
            for module, content in payloads.items():
                filename = f"{module}_{year}.csv"
                if nested if nested is not None else year == 2024:
                    inner = io.BytesIO()
                    with zipfile.ZipFile(inner, "w") as inner_zip:
                        inner_zip.writestr(filename, content)
                    zipped.writestr(f"modules/{module}.zip", inner.getvalue())
                else:
                    zipped.writestr(f"release/{module}/{filename}", content)
        acquisition = next(r for r in catalog["resources"] if r["id"] == f"endes_{year}")
        # This is the only test seam: synthetic payload identities replace the
        # trusted acquisition pins. No public API accepts a completeness assertion.
        acquisition["expected_sha256"] = file_hash(path)
        acquisition["expected_bytes"] = path.stat().st_size
        source = SourceManifest(
            source_id=f"endes_{year}", version=str(year), uri=acquisition["url"],
            payload_hash=file_hash(path), license_hash=digest("synthetic-license"), schema_hash=digest(headers),
            field_mapping=tuple((v, k) for k, v in mapping["fields"].items()),
            mapping_status="reviewed", mapping_review_id="synthetic-fixture-only",
        )
        spec = EstimandSpec(
            endpoint=mapping["endpoint"], target_id="synthetic-frozen-child-target", outcome_scale=mapping["outcome_scale"],
            policy_id="synthetic-geography-policy", weight_id=mapping["weight_id"],
            adjustment_schema_hash=digest("synthetic-approved-adjustment"), inference_unit=mapping["inference_unit"],
            source_lineage_hash=source_lineage_hash((source,)),
        )
        bundle = EndesBundle(archive=path, source=source, purpose=mapping["role"])
        return bundle, year, mapping, spec

    return make


def first_child(records):
    return next(r for r in records if r.household_id == "00010000001" and r.person_number == "01")


def test_conversion_weights_lineage_and_roundtrip(release):
    args = release()
    records, audit = load_endes(*args)
    row = first_child(records)
    assert row.hb_g_dl == 12.0  # HC53=120; adjusted HC56=100 must never be used.
    assert row.hc53_raw == "120" and row.hc55_raw == "0"
    assert row.altitude_m == 100.0 and row.altitude_uncertainty_m is None
    assert row.altitude_uncertainty_provenance == "not_quantified_in_year_dictionary"
    assert row.household_id == "00010000001" and row.person_number == "01"
    assert row.psu_id == row.cluster_id == "001" and row.stratum_id == "01"
    assert row.municipality_id == "010101" and row.frame_cluster_id == "0000001"
    assert dict(row.raw_fields)["RECH0.HV033"] == "00012345"
    assert dict(audit.counts) == {"excluded": 1, "measured": 4}
    assert audit.measured_weight_sum == audit.eligible_weight_sum == 10.0
    assert dict(audit.weighted_state_sums) == {"measured": 10.0}
    adult = next(r for r in records if r.person_number == "02")
    assert adult.survey_weight is None and not adult.analysis_eligible
    assert adult.reasons == ("outside_child_biomarker_population",)
    assert len(audit.module_hashes) == 34
    assert set(audit.entity_graph.original_ids) == {r.original_id for r in records}
    assert any(set(component) == {row.original_id, adult.original_id} for component in audit.entity_graph.components())
    assert {link.relation for link in audit.entity_graph.links} == {"household", "psu", "municipality", "repeated_geography"}
    assert len(dict(audit.sampled_psus_by_stratum)["01"]) == 2
    assert audit.spec == args[3]
    assert audit.lineage.source_hashes == (args[0].source.payload_hash,)
    assert audit.source.content_hash == args[0].source.content_hash
    assert "inference from DHS" in audit.source_notes
    audit.assert_inference_ready()
    assert EndesPerson.from_json(row.to_json()) == row
    assert EndesAudit.from_json(audit.to_json()) == audit
    with pytest.raises(FrozenInstanceError):
        row.hb_g_dl = 10
    again = load_endes(*args)
    assert again == (records, audit)


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("raw,status,state,hb,hb_state,status_name,flags", [
    ("999", "9", "biomarker_missing", None, "missing", "missing_unspecified", ()),
    ("999", "3", "eligible_nonmeasurement", None, "missing", "not_present", ()),
    ("999", "4", "eligible_nonmeasurement", None, "missing", "refused", ()),
    ("999", "6", "eligible_nonmeasurement", None, "missing", "other", ()),
    ("120", "9", "discordant", 12., "observed", "missing_unspecified", ("valid_hb_unspecified_status",)),
    ("999", "0", "discordant", None, "missing", "measured", ("measured_status_without_hb",)),
    ("120", "3", "discordant", 12., "observed", "not_present", ("valid_hb_nonmeasurement_status",)),
    ("", "", "not_applicable", None, "not_applicable", "not_applicable", ()),
    ("", "9", "biomarker_missing", None, "not_applicable", "missing_unspecified", ()),
    ("", "0", "discordant", None, "not_applicable", "measured", ("measured_status_without_hb",)),
    ("120", "0", "measured", 12., "observed", "measured", ()),
    ("120", "4", "discordant", 12., "observed", "refused", ("valid_hb_nonmeasurement_status",)),
    ("120", "6", "discordant", 12., "observed", "other", ("valid_hb_nonmeasurement_status",)),
    ("120", "", "discordant", 12., "observed", "not_applicable", ("valid_hb_nonmeasurement_status",)),
    ("999", "", "biomarker_missing", None, "missing", "not_applicable", ()),
    ("", "3", "eligible_nonmeasurement", None, "not_applicable", "not_present", ()),
    ("", "4", "eligible_nonmeasurement", None, "not_applicable", "refused", ()),
    ("", "6", "eligible_nonmeasurement", None, "not_applicable", "other", ()),
])
def test_sentinels_and_discordance(release, year, raw, status, state, hb, hb_state, status_name, flags):
    def mutate(tables):
        tables["RECH6"][0].update(HC53=raw, HC55=status)
    records, audit = load_endes(*release(year, mutate))
    row = first_child(records)
    assert (row.hc53_raw, row.hc55_raw) == (raw, status)
    assert row.state == state and row.hb_g_dl == hb
    assert row.raw_hb_state == hb_state and row.measurement_status == status_name
    assert row.review_flags == flags
    assert row.analysis_eligible == (state == "measured")
    assert row.eligibility == "eligible" and row.survey_weight == 1.0
    assert sum(dict(audit.counts).values()) == 5
    assert audit.eligible_weight_sum == 10.0
    assert audit.measured_weight_sum == (10.0 if state == "measured" else 9.0)
    assert sum(dict(audit.eligible_status_counts).values()) == audit.eligible_count == 4
    assert sum(dict(audit.weighted_eligible_status_sums).values()) == audit.eligible_weight_sum
    if state == "biomarker_missing":
        assert row.reasons == (("unspecified",) if status == "9" else ("missing_hb_blank_status",))
    if flags:
        assert (row.original_id, flags) in audit.review_records
        with pytest.raises(ContractError, match="require review"):
            audit.assert_inference_ready()
    else:
        audit.assert_inference_ready()


@pytest.mark.parametrize("raw", ["0", "-1", "1000", "nan", "Inf", "12.3"])
def test_invalid_raw_encodings_fail(release, raw):
    with pytest.raises(ContractError, match="HC53"):
        load_endes(*release(mutate=lambda t: t["RECH6"][0].update(HC53=raw)))


def test_unknown_measurement_status_not_reclassified(release):
    with pytest.raises(ContractError, match="unknown HC55"):
        load_endes(*release(mutate=lambda t: t["RECH6"][0].update(HC55="5")))


@pytest.mark.parametrize("field", ["HC56", "HC56A"])
def test_adjusted_hb_mapping_rejected_even_if_manifest_calls_it_raw(release, field):
    mapping = deepcopy(endes._config()["years"][2024])
    mapping["fields"]["raw_hb"] = f"RECH6.{field}"
    with pytest.raises(ContractError, match="adjusted Hb cannot substitute"):
        load_endes(*release(2024, mapping=mapping))


@pytest.mark.parametrize("role,value", [
    ("weight", "RECH0.HV028"), ("weight", "REC0111.V005"),
    ("weight", "CSALUD01.PESO15_AMAS"), ("weight", "CSALUD08.Pesomen12"),
    ("frame_cluster_id", "RECH0.NCONGLOME"), ("raw_hb", "RECH5.HA53"),
])
def test_wrong_population_or_cross_year_field_mapping_rejected(release, role, value):
    mapping = deepcopy(endes._config()["years"][2024])
    mapping["fields"][role] = value
    with pytest.raises(ContractError, match="mapping|adjusted Hb"):
        load_endes(*release(2024, mapping=mapping))


@pytest.mark.parametrize("key,value", [("hb_divisor", 1), ("weight_divisor", 1), ("hb_missing", 998),
                                       ("domain", "all_persons"), ("minimum_measurement_age_months", 6)])
def test_unknown_mapping_semantics_fail_closed(release, key, value):
    mapping = deepcopy(endes._config()["years"][2023])
    mapping[key] = value
    with pytest.raises(ContractError, match="unknown or changed year-specific"):
        load_endes(*release(mapping=mapping))


def test_unknown_year_fails_before_archive_access(release):
    bundle, year, mapping, spec = release()
    with pytest.raises(ContractError, match="unknown ENDES year"):
        load_endes(replace(bundle, archive="absent.zip"), 2025, mapping, spec)


@pytest.mark.parametrize("field,value", [("outcome_scale", "g/L"), ("weight_id", "women"),
                                        ("endpoint", "anemia_adjusted"), ("inference_unit", "person"),
                                        ("source_lineage_hash", "0" * 64)])
def test_incompatible_estimand_rejected(release, field, value):
    bundle, year, mapping, spec = release()
    with pytest.raises(ContractError, match="mismatch"):
        load_endes(bundle, year, mapping, replace(spec, **{field: value}))


@pytest.mark.parametrize("module,error", [("RECH0", "duplicate household"), ("RECH1", "duplicate person"),
                                         ("RECH6", "duplicate biomarker")])
def test_duplicate_join_cardinalities(release, module, error):
    with pytest.raises(ContractError, match=error):
        load_endes(*release(mutate=lambda t: t[module].append(deepcopy(t[module][0]))))


@pytest.mark.parametrize("module", ["RECH1", "RECH6"])
def test_orphan_joins_rejected(release, module):
    with pytest.raises(ContractError, match="orphan"):
        load_endes(*release(mutate=lambda t: t[module][0].update(HHID="not-a-household")))


def test_missing_biomarker_record_preserved_for_review(release):
    records, audit = load_endes(*release(mutate=lambda t: t["RECH6"].pop(0)))
    row = first_child(records)
    assert len(records) == 5
    assert row.state == "biomarker_record_absent"
    assert row.eligibility == "eligible"
    assert row.hc53_raw is None and row.hc55_raw is None
    assert row.raw_hb_state == "no_biomarker_record"
    assert row.reasons == ("missing_biomarker_record",)
    audit.assert_inference_ready()
    assert audit.review_records == ((row.original_id, row.review_flags),)
    assert audit.blocking_review_records == ()


@pytest.mark.parametrize("module,key,value,reason", [
    ("RECH6", "HC1", "3", "below_biomarker_measurement_age"),
    ("RECH0", "HV042", "0", "household_not_selected_for_biomarker"),
    ("RECH0", "HV015", "2", "household_interview_incomplete"),
    ("RECH1", "HV103", "0", "not_in_de_facto_population"),
    ("RECH1", "HV120", "0", "outside_child_biomarker_population"),
])
def test_population_exclusions_retain_records(release, module, key, value, reason):
    records, audit = load_endes(*release(mutate=lambda t: t[module][0].update({key: value})))
    row = first_child(records)
    assert row.state == "excluded" and row.eligibility == "excluded"
    assert reason in row.reasons and not row.analysis_eligible
    assert row.survey_weight is None
    assert row.hb_g_dl == 12.0  # Exclusion does not erase the observed raw measurement.
    assert sum(dict(audit.counts).values()) == len(records)
    assert dict(audit.reason_counts)[reason] >= 1


@pytest.mark.parametrize("weight", ["0", "-1", "", "NaN"])
def test_eligible_weight_must_be_positive_finite(release, weight):
    with pytest.raises(ContractError, match="weight|HV005"):
        load_endes(*release(mutate=lambda t: t["RECH0"][0].update(HV005=weight)))


def test_full_survey_psus_survive_domain_attrition(release):
    def mutate(tables):
        tables["RECH6"][1].update(HC53="999", HC55="3")
    records, audit = load_endes(*release(mutate=mutate))
    assert dict(audit.sampled_psus_by_stratum)["01"] == ("001", "002")
    assert dict(audit.measured_psus_by_stratum)["01"] == ("001",)
    assert audit.singleton_strata == ()  # The other sampled PSU contributes zero.
    audit.assert_inference_ready()


def test_singleton_stratum_retained_and_explicitly_blocks_variance(release):
    records, audit = load_endes(*release(mutate=lambda t: t["RECH0"][0].update(HV022="03")))
    assert audit.singleton_strata == ("01", "03")
    assert len(records) == 5
    assert audit.finite_population_status.startswith("not_documented")
    assert audit.replicate_weight_status == "not_documented"
    with pytest.raises(ContractError, match="singleton"):
        audit.assert_inference_ready()


def test_all_missing_still_returns_reproducible_attrition(release):
    def mutate(tables):
        for child in tables["RECH6"]:
            child.update(HC53="999", HC55="9")
    records, audit = load_endes(*release(mutate=mutate))
    assert dict(audit.counts) == {"biomarker_missing": 4, "excluded": 1}
    assert audit.eligible_weight_sum == 10 and audit.measured_weight_sum == 0
    with pytest.raises(ContractError, match="no measured"):
        audit.assert_inference_ready()


def test_replication_and_year_specific_frame_lineage(release):
    records_23, audit_23 = load_endes(*release(2023))
    args_24 = release(2024)
    records_24, audit_24 = load_endes(*args_24)
    assert first_child(records_23).original_id != first_child(records_24).original_id
    assert first_child(records_23).panel_cluster_id == first_child(records_24).panel_cluster_id
    assert first_child(records_23).exposure_unit_id != first_child(records_24).exposure_unit_id
    assert "RECH0.NCONGLOME" in dict(first_child(records_23).raw_fields)
    assert "RECH0.NCONGLOME1" in dict(first_child(records_24).raw_fields)
    assert "RECH6.HC56A" in dict(first_child(records_24).raw_fields)
    assert audit_23.role == "development" and audit_24.role == "locked_replication"
    bundle, year, mapping, spec = args_24
    with pytest.raises(ContractError, match="locked replication"):
        load_endes(replace(bundle, purpose="development"), year, mapping, spec)
    with pytest.raises(ContractError, match="year-specific"):
        load_endes(bundle, year, endes._config()["years"][2023], spec)


@pytest.mark.parametrize("nested", [False, True])
def test_complete_flat_and_nested_inventory(release, nested):
    records, audit = load_endes(*release(nested=nested))
    assert len(audit.module_rows) == 34 and len(records) == 5


def test_incomplete_inventory_not_disguised_by_rehashed_fixture(release):
    with pytest.raises(ContractError, match="incomplete ENDES module inventory"):
        load_endes(*release(mutate=lambda t: t.pop("RECH5")))


def test_no_duplicate_modules_across_nested_archives(release, monkeypatch):
    bundle, year, mapping, spec = release(nested=True)
    with zipfile.ZipFile(bundle.archive, "a") as zipped:
        duplicate = zipped.read("modules/RECH0.zip")
        zipped.writestr("second-copy.zip", duplicate)
    catalog = deepcopy(endes._release_catalog())
    entry = next(r for r in catalog["resources"] if r["id"] == "endes_2023")
    entry.update(expected_sha256=file_hash(bundle.archive), expected_bytes=Path(bundle.archive).stat().st_size)
    monkeypatch.setattr(endes, "_release_catalog", lambda: catalog)
    source = replace(bundle.source, payload_hash=file_hash(bundle.archive))
    spec = replace(spec, source_lineage_hash=source_lineage_hash((source,)))
    with pytest.raises(ContractError, match="duplicate module"):
        load_endes(replace(bundle, source=source), year, mapping, spec)


def test_sample_cannot_self_certify_complete(release, tmp_path):
    bundle, year, mapping, spec = release()
    sample = tmp_path / "portal-sample.zip"
    with zipfile.ZipFile(sample, "w") as zipped:
        zipped.writestr("RECH6_2023.csv", "HC53,HC55\n120,0\n")
    source = replace(bundle.source, payload_hash=file_hash(sample))
    spec = replace(spec, source_lineage_hash=source_lineage_hash((source,)))
    with pytest.raises(ContractError, match="unapproved/sample release"):
        load_endes(replace(bundle, archive=sample, source=source), year, mapping, spec)


def test_payload_tampering_caught(release):
    bundle, year, mapping, spec = release()
    path = Path(bundle.archive)
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(blob)
    with pytest.raises(ContractError, match="payload hash"):
        load_endes(bundle, year, mapping, spec)


def test_unreviewed_source_and_changed_header_identity_refused(release):
    bundle, year, mapping, spec = release()
    for source, error in [(replace(bundle.source, mapping_status="unreviewed"), "unreviewed"),
                          (replace(bundle.source, schema_hash="0" * 64), "schema hash")]:
        new_spec = replace(spec, source_lineage_hash=source_lineage_hash((source,)))
        with pytest.raises(ContractError, match=error):
            load_endes(replace(bundle, source=source), year, mapping, new_spec)


def test_actual_cross_year_column_not_guessed(release):
    def mutate(tables):
        for row in tables["RECH0"]:
            row["NCONGLOME"] = row.pop("NCONGLOME1")
    with pytest.raises(ContractError, match="missing mapped field: RECH0.NCONGLOME1"):
        load_endes(*release(2024, mutate=mutate))


def test_inconsistent_cluster_altitude_rejected(release):
    def mutate(tables):
        tables["RECH0"][1].update(HV001="001", HV021="001", NCONGLOME="0000001")
    with pytest.raises(ContractError, match="inconsistent cluster altitude"):
        load_endes(*release(mutate=mutate))


def test_exact_exposure_fixed_effects_rejected_coarse_variation_allowed(release):
    bundle, year, mapping, spec = release()
    with pytest.raises(ContractError, match="absorb all treatment variation"):
        load_endes(replace(bundle, fixed_effects=("exposure_unit_id",)), year, mapping, spec)
    with pytest.raises(ContractError, match="absorb all treatment variation"):
        load_endes(replace(bundle, fixed_effects=("municipality_id",)), year, mapping, spec)
    records, audit = load_endes(replace(bundle, fixed_effects=("region_id",)), year, mapping, spec)
    assert audit.fixed_effects == ("region_id",)


def test_policy_actions_cannot_depend_on_person_within_same_geography(release):
    bundle, year, mapping, spec = release()
    records, audit = load_endes(bundle, year, mapping, spec)
    actions = tuple((r.original_id, 2.) for r in records)
    _, audit = load_endes(replace(bundle, policy_actions=actions), year, mapping, spec)
    assert dict(audit.policy_actions) == dict(actions)
    inconsistent = tuple((r.original_id, 0. if r.person_number == "02" else 2.) for r in records)
    with pytest.raises(ContractError, match="inconsistent policy actions"):
        load_endes(replace(bundle, policy_actions=inconsistent), year, mapping, spec)
    with pytest.raises(ContractError, match="cover full roster"):
        load_endes(replace(bundle, policy_actions=actions[:-1]), year, mapping, spec)


def test_distinct_cluster_and_psu_identifiers_preserved(release):
    # These are separately mapped identities; their labels need not be equal.
    records, audit = load_endes(*release(mutate=lambda t: t["RECH0"][0].update(HV021="101")))
    row = first_child(records)
    assert row.cluster_id == "001" and row.psu_id == "101"
    assert dict(audit.sampled_psus_by_stratum)["01"] == ("002", "101")
    assert any(link.relation == "psu" and link.entity_id == "101" for link in audit.entity_graph.links)


def test_absent_altitude_is_an_unresolved_exposure_prerequisite(release):
    # Unknown altitude precision is allowed; an absent altitude value cannot be
    # silently assigned an exposure, nor is an alternative assignment approved.
    with pytest.raises(ContractError, match="invalid integer encoding: HV040"):
        load_endes(*release(mutate=lambda t: t["RECH0"][0].update(HV040="")))


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("months", [4, 5])
@pytest.mark.parametrize("raw,status", [("120", "0"), ("999", "9"), ("", "")])
def test_documented_four_month_hb_eligibility(release, year, months, raw, status):
    # Each year's Ficha Tecnica, section 5.2, PDF page 9, explicitly starts
    # hemoglobin measurement at four months. The six-month anemia-indicator
    # population is not the entire raw-Hb measurement population.
    def mutate(tables):
        tables["RECH1"][0]["HV105"] = "0"
        tables["RECH6"][0].update(HC1=str(months), HC53=raw, HC55=status)
    records, audit = load_endes(*release(year, mutate=mutate))
    row = first_child(records)
    assert row.eligibility == "eligible" and row.survey_weight == 1.0
    assert audit.eligible_weight_sum == 10.0
    assert row.analysis_eligible == (status == "0")
    assert row.raw_hb_state == {"120": "observed", "999": "missing", "": "not_applicable"}[raw]
    audit.assert_inference_ready()


def test_psu_identity_cannot_cross_strata(release):
    def mutate(tables):
        tables["RECH0"][0]["HV021"] = "101"
        tables["RECH0"][2]["HV021"] = "101"
    with pytest.raises(ContractError, match="PSU assigned to inconsistent strata"):
        load_endes(*release(mutate=mutate))


@pytest.mark.parametrize("year", [2023, 2024])
def test_missing_hb_blank_status_is_not_not_applicable(release, year):
    records, audit = load_endes(*release(
        year, mutate=lambda t: t["RECH6"][0].update(HC53="999", HC55="")))
    row = first_child(records)
    assert row.state == "biomarker_missing"
    assert row.raw_hb_state == "missing" and row.measurement_status == "not_applicable"
    assert row.reasons == ("missing_hb_blank_status",)
    assert not row.review_flags and not row.analysis_eligible
    assert dict(audit.counts) == {"biomarker_missing": 1, "excluded": 1, "measured": 3}
    assert audit.eligible_weight_sum == 10.0 and audit.measured_weight_sum == 9.0
    audit.assert_inference_ready()


@pytest.mark.parametrize("year", [2023, 2024])
def test_absent_biomarker_keeps_independently_known_eligibility(release, year):
    records, audit = load_endes(*release(year, mutate=lambda t: t["RECH6"].pop(0)))
    row = first_child(records)
    assert row.eligibility == "eligible"
    assert row.state == "biomarker_record_absent"
    assert row.measurement_status == row.raw_hb_state == "no_biomarker_record"
    assert row.hc53_raw is None and row.hc55_raw is None and row.hb_g_dl is None
    assert row.age_months is None and dict(row.raw_fields)["RECH1.HV105"] == "2"
    assert not row.analysis_eligible and row.survey_weight == 1.0
    assert row.reasons == row.review_flags == ("missing_biomarker_record",)
    assert audit.eligible_count == 4 and audit.eligible_weight_sum == 10.0
    assert dict(audit.eligible_status_counts) == {"measured": 3, "no_biomarker_record": 1}
    assert dict(audit.weighted_eligible_status_sums) == {"measured": 9.0, "no_biomarker_record": 1.0}
    audit.assert_inference_ready()
    assert audit.review_records == ((row.original_id, row.review_flags),)
    assert audit.blocking_review_records == ()
    assert EndesAudit.from_json(audit.to_json()) == audit


@pytest.mark.parametrize("year", [2023, 2024])
def test_eligible_status_partition_reconciles_absence_and_all_nonmeasurements(release, year):
    def mutate(tables):
        for index, pair in enumerate([
            ("999", "3"), ("999", "4"), ("999", "6"), ("999", "9"),
            ("999", ""), ("", ""), None,
        ], start=3):
            number = f"{index:02}"
            person = dict(tables["RECH1"][0], HVIDX=number)
            tables["RECH1"].append(person)
            if pair is not None:
                tables["RECH6"].append(dict(tables["RECH6"][0], HC0=number, HC53=pair[0], HC55=pair[1]))
    args = release(year, mutate=mutate)
    records, audit = load_endes(*args)
    expected = {"measured": 4, "not_present": 1, "refused": 1, "other": 1,
                "missing_unspecified": 1, "not_applicable": 2, "no_biomarker_record": 1}
    assert dict(audit.eligible_status_counts) == expected
    assert dict(audit.weighted_eligible_status_sums) == dict(expected, measured=10.0)
    assert sum(expected.values()) == audit.eligible_count == 11
    assert sum(dict(audit.weighted_eligible_status_sums).values()) == audit.eligible_weight_sum == 17.0
    assert dict(audit.counts) == {"measured": 4, "eligible_nonmeasurement": 3,
        "biomarker_missing": 2, "not_applicable": 1, "biomarker_record_absent": 1, "excluded": 1}
    assert sum(dict(audit.counts).values()) == len(records) == 12
    assert audit.measured_weight_sum == 10.0
    absent = next(r for r in records if r.state == "biomarker_record_absent")
    assert absent.eligibility == "eligible" and not absent.analysis_eligible
    assert not any(r.analysis_eligible for r in records if r.state != "measured")
    audit.assert_inference_ready()
    assert audit.review_records == ((absent.original_id, absent.review_flags),)
    assert audit.blocking_review_records == ()
    assert EndesAudit.from_json(audit.to_json()) == audit
    assert load_endes(*args) == (records, audit)


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("age", ["0", "98", ""])
def test_absent_record_does_not_invent_age_eligibility(release, year, age):
    def mutate(tables):
        tables["RECH1"][0]["HV105"] = age
        tables["RECH6"].pop(0)
    records, audit = load_endes(*release(year, mutate=mutate))
    row = first_child(records)
    assert row.eligibility == "unknown" and row.age_months is None
    assert row.state == "biomarker_record_absent" and not row.analysis_eligible
    assert audit.eligible_count == 3 and audit.eligible_weight_sum == 9.0
    assert dict(audit.eligible_status_counts) == {"measured": 3}
    assert dict(audit.counts)["biomarker_record_absent"] == 1
    assert audit.blocking_review_records == audit.review_records == ((row.original_id, row.review_flags),)
    assert EndesAudit.from_json(audit.to_json()) == audit
    with pytest.raises(ContractError, match="require review"):
        audit.assert_inference_ready()


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("age", ["1", "4"])
@pytest.mark.parametrize("absent", [False, True])
def test_roster_age_eligibility_does_not_require_outcome_record(release, year, age, absent):
    def mutate(tables):
        tables["RECH1"][0]["HV105"] = age
        if absent:
            tables["RECH6"].pop(0)
        else:
            tables["RECH6"][0]["HC1"] = ""
    records, audit = load_endes(*release(year, mutate=mutate))
    row = first_child(records)
    assert row.eligibility == "eligible" and row.age_months is None
    assert row.state == ("biomarker_record_absent" if absent else "measured")
    assert row.analysis_eligible is not absent
    assert audit.eligible_weight_sum == 10.0


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("altitude,valid", [(-25, False), (-24, True), (5100, True), (5101, False)])
def test_year_dictionary_altitude_range(release, year, altitude, valid):
    # HV040 row 32 explicitly lists -24:5100: 2023 household PDF p2,
    # 2024 household PDF p3. Bounds are source encoding, not invented support.
    args = release(year, mutate=lambda t: t["RECH0"][0].update(HV040=str(altitude)))
    if valid:
        records, audit = load_endes(*args)
        assert first_child(records).altitude_m == float(altitude)
        audit.assert_inference_ready()
    else:
        with pytest.raises(ContractError, match="altitude outside year-documented range"):
            load_endes(*args)


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("age", ["0", "98", ""])
def test_present_record_unresolved_age_still_blocks(release, year, age):
    def mutate(tables):
        tables["RECH1"][0]["HV105"] = age
        tables["RECH6"][0]["HC1"] = ""
    args = release(year, mutate=mutate)
    records, audit = load_endes(*args)
    row = first_child(records)
    assert row.eligibility == "unknown" and row.state == "eligibility_unknown"
    assert row.review_flags == ("missing_biomarker_age",)
    assert not row.analysis_eligible
    assert audit.blocking_review_records == audit.review_records == ((row.original_id, row.review_flags),)
    assert EndesAudit.from_json(audit.to_json()) == audit
    assert load_endes(*args) == (records, audit)
    with pytest.raises(ContractError, match="require review"):
        audit.assert_inference_ready()


@pytest.mark.parametrize("changes,blocking", [
    ({}, False),
    ({"eligibility": "unknown"}, True),
    ({"state": "discordant"}, True),
    ({"review_flags": ("missing_biomarker_record", "additional_issue")}, True),
    ({"review_flags": ("unrecognized_issue",)}, True),
    ({"review_flags": ()}, False),
])
def test_blocking_review_qualification(release, changes, blocking):
    records, audit = load_endes(*release(mutate=lambda t: t["RECH6"].pop(0)))
    row = replace(first_child(records), **changes)
    assert endes._requires_blocking_review(row) is blocking


@pytest.mark.parametrize("year", [2023, 2024])
@pytest.mark.parametrize("issue,error", [
    ("unknown_eligibility", "require review"),
    ("discordance", "require review"),
    ("singleton", "singleton"),
    ("no_measured", "no measured"),
])
def test_informational_absence_does_not_hide_other_blockers(release, year, issue, error):
    def mutate(tables):
        tables["RECH6"].pop(0)
        if issue == "unknown_eligibility":
            tables["RECH1"][1]["HV105"] = "0"
            tables["RECH6"][0]["HC1"] = ""
        elif issue == "discordance":
            tables["RECH6"][0].update(HC53="999", HC55="0")
        elif issue == "singleton":
            tables["RECH0"][0]["HV022"] = "03"
        else:
            for child in tables["RECH6"]:
                child.update(HC53="999", HC55="9")
    records, audit = load_endes(*release(year, mutate=mutate))
    absent = first_child(records)
    assert absent.eligibility == "eligible" and absent.state == "biomarker_record_absent"
    assert (absent.original_id, absent.review_flags) in audit.review_records
    assert absent.original_id not in dict(audit.blocking_review_records)
    assert bool(audit.blocking_review_records) == (issue in {"unknown_eligibility", "discordance"})
    assert EndesAudit.from_json(audit.to_json()) == audit
    with pytest.raises(ContractError, match=error):
        audit.assert_inference_ready()
