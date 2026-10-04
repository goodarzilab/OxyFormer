"""Offline, year-specific ENDES child-Hb ingestion and auditable attrition.

``load_endes(EndesBundle(...), year, config['years'][year], spec)`` returns
``(tuple[EndesPerson, ...], EndesAudit)``. Read configs/adapters/endes.yaml for
SOURCE evidence, including the explicitly inferred missing-code convention.

The input is a complete local release ZIP, not extracted/sample dataframes.
The reviewed SourceManifest must describe its bytes, all CSV headers (the
SHA-256 of canonical_json({UPPER_MODULE: [headers, ...]})), and the mapping
``tuple((source_field, canonical_role) for role, source_field in fields.items())``.
Acquisition pins and the reviewed mapping are trusted repository configuration.
Tests replace only acquisition identities with synthetic archive identities.

All roster members survive. These are privileged ingestion records, not nuisance
covariates. This adapter grants no predictor permissions, defines no support
policy, and does not fit or compute a variance. Call audit.assert_inference_ready()
before using its measured subset; a pass still requires downstream identification,
frozen target/support, approved adjustment and survey-variance validation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import yaml

from oxyformer.contracts import EstimandSpec, SourceManifest, source_lineage_hash
from oxyformer.data.entity_graph import EntityGraph, EntityLink
from oxyformer.provenance import (
    ArtifactLineage, ContractError, Immutable, canonical_json, file_hash, require,
)

_ROOT = Path(__file__).resolve().parents[4]
_STATUS = {0: "measured", 3: "not_present", 4: "refused", 6: "other", 9: "missing_unspecified"}


@dataclass(frozen=True, kw_only=True)
class EndesBundle:
    archive: Path | str
    source: SourceManifest
    purpose: str
    # Optional downstream requests are checked, never inferred or fitted here.
    fixed_effects: tuple[str, ...] = ()
    policy_actions: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class EndesPerson(Immutable):
    original_id: str
    year: int
    household_id: str
    person_number: str
    cluster_id: str
    psu_id: str
    stratum_id: str
    region_id: str
    municipality_id: str
    frame_cluster_id: str
    exposure_unit_id: str
    panel_cluster_id: str
    altitude_m: float
    altitude_uncertainty_m: float | None
    altitude_provenance: str
    altitude_uncertainty_provenance: str
    hc53_raw: str | None
    hc55_raw: str | None
    raw_hb_state: str
    measurement_status: str
    hb_g_dl: float | None
    age_months: int | None
    eligibility: str
    state: str
    reasons: tuple[str, ...]
    review_flags: tuple[str, ...]
    survey_weight: float | None
    analysis_eligible: bool
    # Qualified original fields retain IDs, sentinels and all design metadata.
    raw_fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EndesAudit(Immutable):
    spec: EstimandSpec
    source: SourceManifest
    year: int
    role: str
    domain: str
    lineage: ArtifactLineage
    entity_graph: EntityGraph
    records_hash: str
    mapping_hash: str
    source_notes: str
    module_hashes: tuple[tuple[str, str], ...]
    module_rows: tuple[tuple[str, int], ...]
    counts: tuple[tuple[str, int], ...]
    reason_counts: tuple[tuple[str, int], ...]
    weighted_state_sums: tuple[tuple[str, float], ...]
    eligible_weight_sum: float
    measured_weight_sum: float
    review_records: tuple[tuple[str, tuple[str, ...]], ...]
    sampled_psus_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    measured_psus_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    singleton_strata: tuple[str, ...]
    finite_population_status: str
    replicate_weight_status: str
    household_without_roster_ids: tuple[str, ...]
    fixed_effects: tuple[str, ...]
    policy_actions: tuple[tuple[str, float], ...]
    inference_blockers: tuple[str, ...]

    def assert_inference_ready(self) -> None:
        """Refuse unresolved ingestion/design issues, without claiming causality."""
        require(not self.inference_blockers, "; ".join(self.inference_blockers))


def _config() -> dict:
    with (_ROOT / "configs/adapters/endes.yaml").open() as stream:
        return yaml.safe_load(stream)


def _release_catalog() -> dict:
    return json.loads((_ROOT / "configs/sources/endes.json").read_text())


def _digest(value) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def _validate_mapping(mapping: dict, expected: dict) -> None:
    require(isinstance(mapping, dict), "ENDES mapping must be a reviewed year mapping")
    require(mapping.get("fields", {}).get("raw_hb") == "RECH6.HC53",
            "adjusted Hb cannot substitute for raw HC53")
    require(mapping == expected, "unknown or changed year-specific ENDES mapping")


def _id(value: str, field: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"missing ID: {field}")
    return value  # Never trim fixed-width IDs or remove leading zeros.


def _integer(value: str, field: str, *, blank=False) -> int | None:
    require(isinstance(value, str), f"invalid numeric field: {field}")
    if blank and not value.strip():
        return None
    require(re.fullmatch(r"[+-]?\d+(?:\.0+)?", value.strip()) is not None,
            f"invalid integer encoding: {field}")
    return int(value.strip().split(".")[0])


def _index(rows, keys, label):
    result = {}
    for row in rows:
        key = tuple(_id(row[k], f"{label}.{k}") for k in keys)
        require(key not in result, f"duplicate {label} join key")
        result[key] = row
    return result


def _read_modules(archive, year, expected_modules):
    """Read flat/nested ZIPs without extraction; caller verifies the outer hash."""
    tables, headers, hashes, counts = {}, {}, {}, {}

    def visit(stream):
        with zipfile.ZipFile(stream) as zipped:
            for member in zipped.infolist():
                if member.is_dir():
                    continue
                if member.filename.lower().endswith(".zip"):
                    visit(io.BytesIO(zipped.read(member)))
                    continue
                if not member.filename.lower().endswith(".csv"):
                    continue
                stem = Path(member.filename).stem.upper()
                require(stem.endswith(f"_{year}"), "CSV release year mismatch")
                module = stem[:-(len(str(year)) + 1)]
                require(module in expected_modules, f"unknown ENDES module: {module}")
                require(module not in headers, f"duplicate module: {module}")
                payload = zipped.read(member)  # CRC checked by zipfile.
                hashes[module] = sha256(payload).hexdigest()
                reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
                names = reader.fieldnames
                require(bool(names) and len(set(names)) == len(names), f"invalid header: {module}")
                headers[module] = names
                rows = []
                count = 0
                for row in reader:
                    require(None not in row and all(v is not None for v in row.values()),
                            f"malformed CSV row: {module}")
                    count += 1
                    if module in {"RECH0", "RECH1", "RECH6"}:
                        require(_integer(row.get("ID1", ""), f"{module}.ID1") == year,
                                f"record release year mismatch: {module}")
                        rows.append(row)
                counts[module] = count
                if module in {"RECH0", "RECH1", "RECH6"}:
                    tables[module] = rows

    try:
        visit(archive)
    except (zipfile.BadZipFile, UnicodeError, csv.Error) as exc:
        raise ContractError("invalid ENDES archive/CSV") from exc
    require(set(headers) == set(expected_modules), "incomplete ENDES module inventory")
    return tables, headers, hashes, counts


def _biomarker(raw_hb, raw_status, mapping):
    if raw_hb is None:
        return None, "no_biomarker_record", "no_biomarker_record", ()
    value = _integer(raw_hb, "HC53", blank=True)
    code = _integer(raw_status, "HC55", blank=True)
    require(code is None or code in _STATUS, "unknown HC55 measurement status")
    status = "not_applicable" if code is None else _STATUS[code]
    flags = []
    if value is None:
        hb, hb_state = None, "not_applicable"
    elif value == mapping["hb_missing"]:
        hb, hb_state = None, "missing"
    else:
        # Only the documented numeric encoding is enforced. Do not invent a
        # physiological trimming threshold or treat other numbers as sentinels.
        require(0 < value < mapping["hb_missing"], "invalid raw HC53 encoding")
        hb, hb_state = value / mapping["hb_divisor"], "observed"
    if code == 0 and hb is None:
        flags.append("measured_status_without_hb")
    elif code != 0 and hb is not None:
        flags.append("valid_hb_unspecified_status" if code == 9 else "valid_hb_nonmeasurement_status")
    return hb, hb_state, status, tuple(flags)


def _check_requests(records, bundle):
    fields = tuple(bundle.fixed_effects)
    require(set(fields) <= {"region_id", "stratum_id", "municipality_id", "exposure_unit_id"},
            "unmapped fixed-effect request")
    if fields:
        groups = defaultdict(set)
        for row in records:
            if row.eligibility == "eligible":
                groups[tuple(getattr(row, f) for f in fields)].add(row.altitude_m)
        require(any(len(values) > 1 for values in groups.values()),
                "fixed effects absorb all treatment variation")
    actions = tuple(bundle.policy_actions)
    if actions:
        action_map = dict(actions)
        require(len(action_map) == len(actions), "duplicate policy action ID")
        require(set(action_map) == {r.original_id for r in records}, "policy actions must cover full roster")
        by_geography = {}
        for row in records:
            action = action_map[row.original_id]
            require(type(action) in (int, float) and math.isfinite(action), "invalid policy action")
            prior = by_geography.setdefault(row.exposure_unit_id, action)
            require(prior == action, "inconsistent policy actions within exposure geography")


def load_endes(bundle: EndesBundle, year: int, mapping: dict,
               estimand_spec: EstimandSpec) -> tuple[tuple[EndesPerson, ...], EndesAudit]:
    """Validate a pinned release; preserve all roster people and attrition reasons.

    Unknown years/mappings and invalid joins raise ContractError. Discordant Hb
    pairs and singleton strata return an inspectable audit that refuses inference.
    2024 requires purpose='locked_replication'; there is no tuning-data option.
    """
    require(type(year) is int, "ENDES year must be an integer")
    config = _config()
    require(year in config["years"], f"unknown ENDES year mapping: {year}")
    _validate_mapping(mapping, config["years"][year])
    require(isinstance(bundle, EndesBundle), "expected EndesBundle")
    require(type(estimand_spec) is EstimandSpec, "expected EstimandSpec")
    require(bundle.purpose == mapping["role"], "release purpose mismatch: 2024 is locked replication")
    source = bundle.source
    source.assert_usable()
    require(source.source_id == f"endes_{year}" and source.version == str(year), "source year mismatch")
    expected_fields = tuple(sorted((v, k) for k, v in mapping["fields"].items()))
    require(source.field_mapping == expected_fields, "SourceManifest mapping mismatch")
    for name in ("endpoint", "outcome_scale", "weight_id", "inference_unit"):
        require(getattr(estimand_spec, name) == mapping[name], f"ENDES {name} mismatch")
    require(estimand_spec.source_lineage_hash == source_lineage_hash((source,)), "source lineage mismatch")
    approvals = yaml.safe_load((_ROOT / "configs/approvals.yaml").read_text())
    require(f"endes_{year}" in approvals["owner_decisions"]["sources_named_in_approved_issues"],
            "ENDES year lacks owner source approval")
    catalog = _release_catalog()
    require(catalog["status"] == "ready", "ENDES acquisition is blocked")
    release = next(r for r in catalog["resources"] if r["id"] == f"endes_{year}")
    require(source.uri == release["url"], "source release URI mismatch")
    require(source.payload_hash == release["expected_sha256"], "unapproved/sample release hash")
    require(Path(bundle.archive).stat().st_size == release["expected_bytes"], "incomplete/sample archive size")
    require(file_hash(bundle.archive) == source.payload_hash, "archive payload hash mismatch")
    tables, headers, hashes, counts = _read_modules(bundle.archive, year, config["modules"])
    require(_digest(headers) == source.schema_hash, "source schema hash mismatch")
    for field in mapping["fields"].values():
        module, name = field.split(".")
        require(name in headers[module], f"missing mapped field: {field}")
    # Metadata fields, when documented in a future reviewed mapping, must be
    # supplied and remain in raw_fields. Current releases document no correction.
    for field in mapping["finite_population_fields"] + mapping["replicate_weight_fields"]:
        module, name = field.split(".")
        require(module == "RECH0" and name in headers[module], f"missing design metadata: {field}")

    def column(role):
        return mapping["fields"][role].split(".")[1]

    households = _index(tables["RECH0"], [column("household_id")], "household")
    roster = _index(tables["RECH1"], [column("roster_household_id"), column("person_number")], "person")
    biomarkers = _index(tables["RECH6"], [column("biomarker_household_id"), column("biomarker_person_number")], "biomarker")
    require(bool(roster), "empty ENDES roster")
    require(all((hh,) in households for hh, person in roster), "orphan roster household join")
    require(set(biomarkers) <= set(roster), "orphan biomarker person join")
    roster_household_ids = {hh for hh, person in roster}
    sampled = defaultdict(set)
    cluster_design = {}
    psu_strata = {}
    for (hh,), row in households.items():
        require(_integer(row[column("household_year")], "HV007") == year, "household interview year mismatch")
        cluster = _id(row[column("cluster_id")], "HV001")
        psu = _id(row[column("psu_id")], "HV021")
        stratum = _id(row[column("stratum_id")], "HV022")
        require(psu_strata.setdefault(psu, stratum) == stratum, "PSU assigned to inconsistent strata")
        altitude = _integer(row[column("altitude_m")], "HV040")
        require(-24 <= altitude <= 5100, "altitude outside year-documented range")
        design = (psu, stratum, altitude, _id(row[column("frame_cluster_id")], column("frame_cluster_id")))
        require(cluster_design.setdefault(cluster, design) == design,
                "inconsistent cluster altitude/design assignment")
        sampled[stratum].add(psu)

    records, links = [], []
    for (hh, person), row in sorted(roster.items()):
        household = households[(hh,)]
        biomarker = biomarkers.get((hh, person))
        oid = canonical_json(["endes", year, hh, person])
        cluster = household[column("cluster_id")]
        raw_hb = None if biomarker is None else biomarker[column("raw_hb")]
        raw_status = None if biomarker is None else biomarker[column("measurement_status")]
        hb, hb_state, status, flags = _biomarker(raw_hb, raw_status, mapping)
        months = None if biomarker is None else _integer(biomarker[column("age_months")], "HC1", blank=True)
        require(months is None or 0 <= months <= mapping["maximum_age_months"], "HC1 outside mapped age domain")
        reasons = []
        interview = _integer(household[column("interview_status")], "HV015")
        selected = _integer(household[column("selected_for_hb")], "HV042")
        slept = _integer(row[column("slept_here")], "HV103")
        child = _integer(row[column("child_eligible")], "HV120")
        require(selected in (0, 1) and slept in (0, 1) and child in (0, 1), "unknown eligibility code")
        if interview != 1:
            reasons.append("household_interview_incomplete")
        if selected != 1:
            reasons.append("household_not_selected_for_biomarker")
        if slept != 1:
            reasons.append("not_in_de_facto_population")
        if child != 1:
            reasons.append("outside_child_biomarker_population")
        if months is not None and months < mapping["minimum_measurement_age_months"]:
            reasons.append("below_biomarker_measurement_age")
        if reasons:
            eligibility, state = "excluded", "excluded"
        elif months is None:
            eligibility, state = "unknown", "eligibility_unknown"
            reasons.append("missing_biomarker_record" if biomarker is None else "missing_biomarker_age")
            flags += (reasons[-1],)
        else:
            eligibility = "eligible"
            if flags:
                state = "discordant"
                reasons.extend(flags)
            elif status == "measured":
                state = "measured"
            elif status in ("not_present", "refused", "other"):
                state = "eligible_nonmeasurement"
                reasons.append(status)
            elif status == "missing_unspecified":
                state = "biomarker_missing"
                reasons.append("unspecified")
            else:
                state = "not_applicable"
                reasons.append("blank_not_applicable")
        weight = None
        if eligibility in ("eligible", "unknown"):
            encoded_weight = _integer(household[column("weight")], "HV005")
            require(encoded_weight > 0, "invalid child biomarker weight")
            weight = encoded_weight / mapping["weight_divisor"]
        raw = {f"RECH0.{k}": v for k, v in household.items()}
        raw.update({f"RECH1.{k}": v for k, v in row.items()})
        if biomarker is not None:
            raw.update({f"RECH6.{k}": v for k, v in biomarker.items()})
        exposure_id = canonical_json(["endes", year, "cluster", cluster])
        panel_id = canonical_json(["endes", "2021-2024", "cluster", cluster])
        municipality = _id(household[column("municipality_id")], "UBIGEO")
        require(re.fullmatch(r"\d{6}", municipality) is not None, "invalid UBIGEO geographic assignment")
        record = EndesPerson(
            original_id=oid, year=year, household_id=hh, person_number=person,
            cluster_id=cluster, psu_id=household[column("psu_id")],
            stratum_id=household[column("stratum_id")], region_id=_id(household[column("region_id")], "HV024"),
            municipality_id=municipality, frame_cluster_id=household[column("frame_cluster_id")],
            exposure_unit_id=exposure_id, panel_cluster_id=panel_id,
            altitude_m=float(_integer(household[column("altitude_m")], "HV040")),
            altitude_uncertainty_m=None, altitude_provenance=f"{source.content_hash}:RECH0.HV040",
            altitude_uncertainty_provenance="not_quantified_in_year_dictionary",
            hc53_raw=raw_hb, hc55_raw=raw_status, raw_hb_state=hb_state,
            measurement_status=status, hb_g_dl=hb, age_months=months,
            eligibility=eligibility, state=state, reasons=tuple(reasons), review_flags=flags,
            survey_weight=weight, analysis_eligible=state == "measured", raw_fields=tuple(sorted(raw.items())),
        )
        records.append(record)
        for relation, namespace, entity in (
            ("household", f"endes:{year}:household", hh),
            ("psu", f"endes:{year}:psu", record.psu_id),
            ("municipality", "peru:ubigeo", municipality),
            ("repeated_geography", "endes:2021-2024:panel", cluster),
        ):
            links.append(EntityLink(observation_id=oid, relation=relation, namespace=namespace, entity_id=entity))
    records = tuple(records)
    _check_requests(records, bundle)
    ids = tuple(r.original_id for r in records)
    graph = EntityGraph(original_ids=ids, links=tuple(links))
    state_counts = Counter(r.state for r in records)
    reason_counts = Counter(reason for r in records for reason in r.reasons)
    weighted = defaultdict(list)
    measured = defaultdict(set)
    for row in records:
        if row.survey_weight is not None:
            weighted[row.state].append(row.survey_weight)
        if row.analysis_eligible:
            measured[row.stratum_id].add(row.psu_id)
    singleton = tuple(sorted(h for h, psus in sampled.items() if len(psus) == 1))
    review = tuple((r.original_id, r.review_flags) for r in records if r.review_flags)
    blockers = []
    if review:
        blockers.append("biomarker/eligibility records require review")
    if singleton:
        blockers.append("singleton strata require a documented variance treatment")
    if not state_counts["measured"]:
        blockers.append("no measured eligible child population")
    mapping_hash = _digest(mapping)
    lineage = ArtifactLineage(
        source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(source.content_hash,),
        split_hash=None, config_hash=_digest(config), model_hash=None,
        environment=(("adapter", "endes-v1"), ("python", sys.version.split()[0])), seed=None, parameter_count=None,
    )
    return records, EndesAudit(
        spec=estimand_spec, source=source, year=year, role=mapping["role"], domain=mapping["domain"],
        lineage=lineage, entity_graph=graph, records_hash=_digest([r.to_dict() for r in records]),
        mapping_hash=mapping_hash, source_notes=canonical_json(config["source_notes"]),
        module_hashes=tuple(sorted(hashes.items())), module_rows=tuple(sorted(counts.items())),
        counts=tuple(sorted(state_counts.items())), reason_counts=tuple(sorted(reason_counts.items())),
        weighted_state_sums=tuple(sorted((state, math.fsum(values)) for state, values in weighted.items())),
        eligible_weight_sum=math.fsum(r.survey_weight for r in records if r.eligibility == "eligible"),
        measured_weight_sum=math.fsum(r.survey_weight for r in records if r.analysis_eligible),
        review_records=review, sampled_psus_by_stratum=tuple(sorted((h, tuple(sorted(p))) for h, p in sampled.items())),
        measured_psus_by_stratum=tuple(sorted((h, tuple(sorted(p))) for h, p in measured.items())),
        singleton_strata=singleton, finite_population_status="not_documented; HV033 preserved, not interpreted as FPC",
        replicate_weight_status="not_documented", household_without_roster_ids=tuple(sorted(
            hh for (hh,) in households if hh not in roster_household_ids)),
        fixed_effects=tuple(bundle.fixed_effects), policy_actions=tuple(sorted(bundle.policy_actions)),
        inference_blockers=tuple(blockers),
    )
