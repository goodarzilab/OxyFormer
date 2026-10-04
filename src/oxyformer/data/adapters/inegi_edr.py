"""EDR observation validation, registration-lag filtering and population cells.

This module produces all-cause counts, never a population cohort or a cause
classifier/risk model. Unknown age/sex/time records remain unallocated. Source files
are already deduplicated by INEGI: identical public rows can be distinct deaths.
We reject repeated registration releases rather than invent a person/event key.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import math

from oxyformer.contracts import SourceManifest
from oxyformer.provenance import canonical_json
from oxyformer.data.adapters.conapo import (
    AGE_GROUPS, SEXES, DenominatorTable, _check, _code, _integer, _read_rows, _years,
)


EDR_MAPPING = (
    ("ENT_RESID", "residence_state"), ("MUN_RESID", "residence_municipality"),
    ("ANIO_OCUR", "occurrence_year"), ("ANIO_REGIS", "registration_year"),
    ("EDAD", "age_code"), ("SEXO", "sex_code"), ("CAUSA_DEF", "cause_code"),
)


@dataclass(frozen=True, slots=True)
class RegistrationRelease:
    """One unfiltered registration-year file plus reviewed coverage evidence.

Coverage declares that the file includes all published records for each listed
source municipality, all occurrence years, ages, sexes and causes. It is not
inferred from municipalities with deaths, and does not certify underregistration
is absent. ``coverage_reference`` identifies the inspection supporting this
assertion. Partial/unknown coverage must not be declared complete.
"""
    path: str
    manifest: SourceManifest
    registration_year: int
    geography_vintage: str
    covered_municipalities: tuple[str, ...]
    coverage_reference: str
    encoding: str = "utf-8-sig"


@dataclass(frozen=True, slots=True)
class DeathEvent:
    source_record_id: str  # file/year/row identity, NOT a public person identifier
    release_year: int
    geography_vintage: str
    source_municipality: str
    occurrence_year: int | None
    registration_year: int
    age_group: str | None
    sex: str | None
    cause_code: str
    selection: str
    source_values: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class EventTable:
    events: tuple[DeathEvent, ...]  # includes late/outside/unknown-time records
    releases: tuple[RegistrationRelease, ...]
    years: tuple[int, ...]
    lag: int
    audit: dict


def _age_group(value: str) -> str | None:
    code = _integer(value, "EDR age code")
    # Actual EDR edad.csv: hours/days/months (including unspecified within
    # that infant unit) all belong to 00-04. 4998 is unspecified years.
    infants = (1001 <= code <= 1023 or 2001 <= code <= 2029
               or 3001 <= code <= 3011 or code in (1097, 1098, 2098, 3098))
    if infants:
        return "00-04"
    if code == 4998:
        return None
    _check(4001 <= code <= 4120, "unsupported EDR age code", age_code=value)
    years = code - 4000
    return AGE_GROUPS[min(years // 5, 17)]


def load_events(releases, *, occurrence_years=range(2015, 2020), lag=2) -> EventTable:
    """Keep t <= registration year <= t+L; require every needed release.

L=2 is primary, L=5 is the owner-approved late-registration check. No completeness
is inferred from event rows. Repeated/revised releases of one year are refused;
all public rows within the selected revision retain their multiplicity.
"""
    years = _years(occurrence_years)
    _check(type(lag) is int and lag in (2, 5), "only approved lag windows 2 and 5 are allowed")
    releases = tuple(releases)
    required = {r for t in years for r in range(t, t + lag + 1)}
    supplied = [r.registration_year for r in releases]
    _check(all(type(r) is int for r in supplied), "invalid release year")
    _check(len(supplied) == len(set(supplied)),
           "duplicate registration release or mixed revisions", release_years=supplied)
    _check(required <= set(supplied), "missing required registration releases",
           missing_registration_years=sorted(required - set(supplied)))
    events = []
    normalized_releases = []
    counts = Counter()
    unknowns = Counter()
    for release in sorted(releases, key=lambda x: x.registration_year):
        _check(bool(release.geography_vintage.strip()) and bool(release.coverage_reference.strip()),
               "reviewed coverage and geographic vintage are required")
        covered = tuple(_code(m, 5) for m in release.covered_municipalities)
        _check(len(set(covered)) == len(covered), "duplicate normalized coverage codes")
        release = replace(release, covered_municipalities=covered)
        normalized_releases.append(release)
        for number, row, raw in _read_rows(release.path, release.manifest, EDR_MAPPING,
                                          release.encoding):
            registration = _integer(row["ANIO_REGIS"], "registration year")
            _check(registration == release.registration_year,
                   "row registration year differs from release", row=number,
                   release_year=release.registration_year, registration_year=registration)
            occurrence = _integer(row["ANIO_OCUR"], "occurrence year")
            if occurrence == 9999:
                occurrence = None
            _check(occurrence is None or 1 <= occurrence <= registration,
                   "occurrence follows registration or is invalid", row=number)
            selection = ("unknown_occurrence_year" if occurrence is None else
                         "outside_occurrence_years" if occurrence not in years else
                         "late_registration" if registration > occurrence + lag else "included")
            age = _age_group(row["EDAD"])
            sex_code = _integer(row["SEXO"], "sex code")
            _check(sex_code in (1, 2, 9), "unsupported EDR sex code", row=number)
            sex = {1: "male", 2: "female", 9: None}[sex_code]
            municipality = _code(row["ENT_RESID"], 2) + _code(row["MUN_RESID"], 3)
            counts[selection] += 1
            if selection == "included":
                unknowns["unknown_age"] += age is None
                unknowns["unknown_sex"] += sex is None
                unknowns["unknown_age_or_sex"] += age is None or sex is None
                unknowns["missing_cause"] += not bool(row["CAUSA_DEF"])
            events.append(DeathEvent(
                f"{registration}:{release.manifest.payload_hash}:{number}", registration,
                release.geography_vintage, municipality, occurrence, registration,
                age, sex, row["CAUSA_DEF"], selection, raw))
    releases = tuple(normalized_releases)
    audit = dict(input_events=len(events), selection_counts=dict(counts),
                 unknown_categories=dict(unknowns), lag=lag, occurrence_years=years,
                 required_registration_years=sorted(required),
                 releases=[dict(year=r.registration_year, version=r.manifest.version,
                                payload_hash=r.manifest.payload_hash,
                                manifest_hash=r.manifest.content_hash,
                                vintage=r.geography_vintage,
                                covered_municipalities=list(r.covered_municipalities),
                                coverage_reference=r.coverage_reference) for r in releases],
                 duplicate_policy="retain published rows; one revision per registration year")
    _check(sum(counts.values()) == len(events), "event count conservation failed")
    return EventTable(tuple(events), releases, years, lag, audit)


@dataclass(frozen=True, slots=True)
class ReviewedCrosswalk:
    """Complete membership of stable unions, reviewed outside this adapter.

Rows are (source_vintage, source_municipality, target_union). Include every
member for every participating vintage, including identity mappings and members
with zero events. Multiple source members may map to one union; splitting a
source municipality is forbidden. review_id attests identical union territory
across vintages. The adapter validates structure, not spatial truth.
"""
    rows: tuple[tuple[str, str, str], ...]
    target_vintage: str
    review_id: str

    def mapping(self):
        _check(bool(self.review_id.strip()) and bool(self.target_vintage.strip()),
               "crosswalk requires an explicit review and target vintage")
        mapping = {}
        for vintage, source, target in self.rows:
            source = _code(source, 5)
            _check(bool(vintage.strip()) and bool(target.strip()), "invalid crosswalk row")
            _check((vintage, source) not in mapping,
                   "duplicate or splitting crosswalk source", source=(vintage, source))
            mapping[vintage, source] = target
        _check(bool(mapping), "empty crosswalk")
        return mapping


@dataclass(frozen=True, slots=True)
class MortalityCell:
    municipality: str
    year: int
    age_group: str
    sex: str
    deaths: int  # fully age/sex classified records only
    population: int
    unallocated_deaths: int  # possible unknown age/sex/time; not additive across cells
    genuine_zero: bool  # no classified deaths AND no possibly relevant unknowns
    target_mass: float | None
    age_standardization_weight: float | None
    poisson_log_offset: float
    poisson_loss_multiplier: float | None  # target_mass / population, plan 3.6


@dataclass(frozen=True, slots=True)
class CellTable:
    cells: tuple[MortalityCell, ...]
    unallocated_events: tuple[DeathEvent, ...]
    audit: dict


def build_cells(events: EventTable, denominators: DenominatorTable, *,
                crosswalk: ReviewedCrosswalk | None = None,
                target_mass=None, age_standardization_weights=None) -> CellTable:
    """Complete the denominator grid only after release and coverage checks.

Key target_mass by (municipality/union, year, age_group, sex). Age-standardization
weights are a separate age-group mapping summing to one; neither is derived
from population. Unspecified weights remain None: this is observation validation,
not a chosen estimand. Poisson exposure is N, log offset is log(N), and the
plan's normalized likelihood multiplier, if mass is supplied, is w/N.

All unmatched residence codes (even on unknown-age/sex records) fail closed.
Unknown age/sex/time records with matched geography remain in unallocated_events;
they are never redistributed. Thus deaths=0 with unallocated records is NOT a
certified zero. This endpoint never exports a decedent classifier as risk.
"""
    _check(isinstance(events, EventTable) and isinstance(denominators, DenominatorTable),
           "event and population denominator tables are both required")
    _check(events.years == denominators.years, "occurrence and denominator years differ")
    required = {r for t in events.years for r in range(t, t + events.lag + 1)}
    vintages = {r.geography_vintage for r in events.releases if r.registration_year in required}
    denominator_vintage = denominators.geography_vintage
    if crosswalk is None:
        _check(vintages == {denominator_vintage}, "incompatible geographic vintages require reviewed crosswalk",
               event_vintages=sorted(vintages), denominator_vintage=denominator_vintage)
        mapping = {(denominator_vintage, m): m for m in denominators.municipalities}
        target_vintage = denominator_vintage
    else:
        mapping = crosswalk.mapping()
        target_vintage = crosswalk.target_vintage
    included = [e for e in events.events if e.selection == "included"]
    # An unknown occurrence year is not assigned to an analysis year. It can
    # still preclude a certified zero in each year compatible with its known
    # registration year and lag. Releases outside all windows are irrelevant.
    unknown_time = [e for e in events.events if e.selection == "unknown_occurrence_year"
                    and e.registration_year in required]
    relevant = included + unknown_time
    unmatched = Counter((e.geography_vintage, e.source_municipality) for e in relevant
                        if (e.geography_vintage, e.source_municipality) not in mapping)
    missing_denominator_mapping = sorted(m for m in denominators.municipalities
                                        if (denominator_vintage, m) not in mapping)
    _check(not unmatched and not missing_denominator_mapping, "unmatched municipality codes",
           unmatched_events=[(v, m, n) for (v, m), n in sorted(unmatched.items())],
           unmatched_denominators=missing_denominator_mapping)
    populations = Counter()
    for d in denominators.cells:
        populations[mapping[denominator_vintage, d.municipality], d.year, d.age_group, d.sex] += d.population
    targets = {key[0] for key in populations}
    # Require every reviewed member on both sides. A union cannot be completed
    # from only the observed subset of its municipalities.
    missing_members = []
    members = defaultdict(set)
    for (vintage, source), target in mapping.items():
        members[vintage, target].add(source)
    for target in sorted(targets):
        for vintage in sorted(vintages | {denominator_vintage}):
            if not members[vintage, target]:
                missing_members.append((vintage, target, "no reviewed members"))
        for source in members[denominator_vintage, target] - set(denominators.municipalities):
            missing_members.append((denominator_vintage, target, source))
    _check(not missing_members, "incomplete stable-union membership", missing_members=missing_members)
    counts = Counter()
    unallocated = []
    unmatched_targets = Counter()
    ambiguous = Counter()
    for event in relevant:
        target = mapping[event.geography_vintage, event.source_municipality]
        if target not in targets:
            unmatched_targets[target] += 1
        elif event.occurrence_year is None:
            unallocated.append(event)
            for year in events.years:
                if year <= event.registration_year <= year + events.lag:
                    ambiguous[target, year] += 1
        elif event.age_group is None or event.sex is None:
            unallocated.append(event)
            ambiguous[target, event.occurrence_year] += 1
        else:
            key = (target, event.occurrence_year, event.age_group, event.sex)
            counts[key] += 1
    _check(not unmatched_targets and set(counts) <= set(populations), "events lack denominator cells",
           unmatched_targets=dict(unmatched_targets), unmatched_cells=sorted(set(counts) - set(populations)))
    release_by_year = {r.registration_year: r for r in events.releases}
    missing_coverage = []
    for target in sorted(targets):
        for year in events.years:
            for registration in range(year, year + events.lag + 1):
                release = release_by_year[registration]
                for source in members[release.geography_vintage, target] - set(release.covered_municipalities):
                    missing_coverage.append((target, year, registration, source))
    _check(not missing_coverage, "missing source coverage cannot become zero deaths",
           missing_coverage=missing_coverage)
    # Also refuse deaths lying outside a release's declared coverage, even when
    # other source municipalities provide enough rows to form the union.
    uncovered_events = [e.source_record_id for e in relevant
                        if e.source_municipality not in release_by_year[e.release_year].covered_municipalities]
    _check(not uncovered_events, "events outside declared coverage", records=uncovered_events)
    _check(all(n > 0 for n in populations.values()), "denominators must be positive")
    _check(sum(counts.values()) + len(unallocated) == len(relevant), "death count conservation failed")
    if target_mass is not None:
        _check(set(target_mass) == set(populations), "target mass must cover exactly the completed cells")
        _check(all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in target_mass.values())
               and sum(target_mass.values()) > 0, "invalid target mass")
    if age_standardization_weights is not None:
        _check(set(age_standardization_weights) == set(AGE_GROUPS)
               and all(type(w) in (int, float) and math.isfinite(w) and w >= 0
                       for w in age_standardization_weights.values())
               and math.isclose(sum(age_standardization_weights.values()), 1.0, rel_tol=1e-9),
               "invalid age-standardization weights")
    cells = []
    for key, n in sorted(populations.items()):
        target, year, age, sex = key
        mass = None if target_mass is None else float(target_mass[key])
        age_weight = None if age_standardization_weights is None else float(age_standardization_weights[age])
        unknown = ambiguous[target, year]
        cells.append(MortalityCell(target, year, age, sex, counts[key], n, unknown,
                                   counts[key] == 0 and unknown == 0, mass, age_weight,
                                   math.log(n), None if mass is None else mass / n))
    audit = dict(input_event_audit=events.audit, denominator_audit=denominators.audit,
                 included_events=len(included), allocated_deaths=sum(counts.values()),
                 unknown_occurrence_year_events=len(unknown_time),
                 unallocated_deaths=len(unallocated), population_sum=sum(populations.values()),
                 genuine_zero_cells=sum(c.genuine_zero for c in cells), missing_coverage=[],
                 unmatched_events=[], target_vintage=target_vintage,
                 crosswalk_review=None if crosswalk is None else crosswalk.review_id,
                 crosswalk_hash=None if crosswalk is None else sha256(canonical_json(
                     [crosswalk.target_vintage, crosswalk.review_id,
                      [(v, s, t) for (v, s), t in sorted(mapping.items())]]).encode()).hexdigest(),
                 observation_model="all-cause age-sex municipal counts with midyear population exposure",
                 death_semantics="classified records; unknown age/sex/time retained separately, never redistributed")
    return CellTable(tuple(cells), tuple(unallocated), audit)
