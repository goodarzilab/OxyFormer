"""DANE banded birth observations; also owns shared birth audit/lineage types.

The input bundle is one local CSV (path or bytes), never an archive containing
both births and deaths. A reviewed SourceManifest must hash those exact bytes
and its schema_hash must describe the ordered string columns (ColumnSpec).
SPSS/Stata extraction belongs upstream and needs its own reviewed provenance.

BirthRelease describes the *particular* vintage, not just its event year.
BirthMapping pins that release, the inspected profile and any expected exposure
artifact. Missing exposure is auditable but cannot pass assert_anchor_ready().
No synthetic permission switch exists in production; tests replace the approvals
reader. Output records are privileged observation data, never nuisance inputs.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
from pathlib import Path
from typing import Literal

import yaml

from oxyformer.contracts import ColumnSpec, SourceManifest, schema_hash
from oxyformer.data.entity_graph import EntityLink
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.provenance import Immutable, canonical_json, check_hash, nonempty, require, unique

_ROOT = Path(__file__).resolve().parents[4]


def _profile(key: str) -> dict:
    profiles = yaml.safe_load((_ROOT / 'configs/adapters/births.yaml').read_text())['profiles']
    require(key in profiles, f'unsupported dictionary profile: {key}; explicit mapping required')
    return profiles[key]


def profile_hash(key: str) -> str:
    """Pin reviewed metadata, including year-specific code/field definitions."""
    return sha256(canonical_json(_profile(key)).encode()).hexdigest()


def _owner_approvals() -> dict:
    return yaml.safe_load((_ROOT / 'configs/approvals.yaml').read_text())


@dataclass(frozen=True, slots=True, kw_only=True)
class BirthRelease(Immutable):
    country: Literal['COL', 'ECU']
    release_id: str
    dictionary_profile: str
    status: Literal['provisional', 'definitive', 'unknown']
    status_reference: str
    occurrence_period: str
    registration_period: str  # explicitly 'not published' when unavailable
    geography_vintage: str
    geography_reference: str
    population: Literal['live_births', 'fetal_deaths', 'mixed']
    selection_reference: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        for name in ('release_id', 'dictionary_profile', 'status_reference', 'occurrence_period',
                     'registration_period', 'geography_vintage', 'geography_reference',
                     'selection_reference'):
            nonempty(getattr(self, name), name)


@dataclass(frozen=True, slots=True, kw_only=True)
class Identifier(Immutable):
    field: str
    kind: Literal['maternal', 'registration', 'household', 'psu', 'outcome_lineage']
    namespace: str  # reviewed scope; never assume a year-local number is global
    documentation: str
    unknown_codes: tuple[str, ...] = ('',)

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.field, self.namespace, self.documentation):
            nonempty(value, 'identifier provenance')


@dataclass(frozen=True, slots=True, kw_only=True)
class PopulationExclusion(Immutable):
    field: str
    values: tuple[str, ...]
    reason: str
    approval_id: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.field, self.reason, self.approval_id):
            nonempty(value, 'population exclusion')
        require(bool(self.values), 'empty exclusion values')


@dataclass(frozen=True, slots=True, kw_only=True)
class BirthMapping(Immutable):
    source: SourceManifest
    release_hash: str
    profile_hash: str
    expected_exposure_hash: str | None = None
    identifiers: tuple[Identifier, ...] = ()
    exclusions: tuple[PopulationExclusion, ...] = ()
    delimiter: str = ','
    encoding: str = 'utf-8-sig'

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.release_hash, self.profile_hash):
            check_hash(value)
        if self.expected_exposure_hash is not None:
            check_hash(self.expected_exposure_hash)
        require(len(self.delimiter) == 1, 'CSV delimiter must be one character')
        unique(tuple(i.field for i in self.identifiers), 'identifier fields')


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidenceAssignment(Immutable):
    residence: tuple[str, ...]
    exposure_mmhg: float
    lineage_namespace: str
    lineage_id: str  # explicit cross-vintage repeated-geography lineage

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.residence), 'empty residence key')
        for value in self.residence + (self.lineage_namespace, self.lineage_id):
            nonempty(value, 'residence assignment')
        require(self.exposure_mmhg >= 0, 'negative exposure')


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidenceExposure(Immutable):
    country: Literal['COL', 'ECU']
    geography_vintage: str
    residence_fields: tuple[str, ...]
    release_hashes: tuple[str, ...]
    source_uri: str
    source_hash: str
    mapping_date: str
    placement: Literal['maternal_residence_point', 'maternal_residence_population_weighted']
    placement_reference: str
    geography_reference: str
    review_id: str
    assignments: tuple[ResidenceAssignment, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.geography_vintage, self.source_uri, self.placement_reference,
                      self.geography_reference, self.review_id):
            nonempty(value, 'exposure provenance')
        date.fromisoformat(self.mapping_date)
        check_hash(self.source_hash)
        require(bool(self.release_hashes) and bool(self.residence_fields), 'exposure scope required')
        for value in self.release_hashes:
            check_hash(value)
        unique(tuple(a.residence for a in self.assignments), 'residence exposure keys')
        require(all(len(a.residence) == len(self.residence_fields) for a in self.assignments),
                'residence key width mismatch')
        forbidden = ('3dep', 'hospital', 'capital_city', 'capital-city')
        require(not any(word in self.source_uri.lower() for word in forbidden),
                'prohibited exposure source')


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightBand(Immutable):
    lower: float | None
    upper: float | None
    upper_inclusive: bool = True

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.lower is None or self.upper is None or self.lower <= self.upper,
                'inverted weight band')
        require(self.lower is None or self.upper != self.lower or self.upper_inclusive,
                'empty weight band')


def classify_lbw(band: WeightBand) -> bool | None:
    """LBW is strictly <2500 g. Bounds are measurements, not midpoints."""
    if band.upper is not None and (band.upper < 2500 or
                                   (band.upper == 2500 and not band.upper_inclusive)):
        return True
    if band.lower is not None and band.lower >= 2500:
        return False
    return None


def _number(raw: str) -> Decimal | None:
    try:
        value = Decimal(raw.strip())
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def _band_weight(raw: str, profile: dict):
    value = _number(raw)
    code = str(int(value)) if value is not None and value == value.to_integral_value() else raw.strip()
    if not raw.strip():
        return None, None, None, 'missing'
    if code in profile['weight_unknown']:
        return None, None, None, 'unknown'
    bounds = profile['bands'].get(code)
    if bounds is None:
        return None, None, None, 'unrecognized_code'
    band = WeightBand(lower=bounds[0], upper=bounds[1])
    lbw = classify_lbw(band)
    return None, band, lbw, 'ambiguous_band' if lbw is None else 'observed_band'


@dataclass(frozen=True, slots=True, kw_only=True)
class BirthRecord(Immutable):
    original_id: str
    release: BirthRelease
    source_hash: str
    mapping_hash: str
    raw: tuple[tuple[str, str], ...]
    outcome_scale: Literal['risk_difference', 'grams']
    weight_raw: str
    birth_weight_g: float | None
    weight_band: WeightBand | None
    lbw: bool | None
    measurement_status: str
    maternal_residence: tuple[str, ...] | None
    delivery_geography: tuple[str, ...] | None
    occurrence: tuple[str, ...]
    registration: tuple[str, ...]
    identifiers: tuple[tuple[str, str, str], ...]
    links: tuple[EntityLink, ...]
    exposure_mmhg: float | None
    exposure_hash: str | None
    exclusion_reasons: tuple[str, ...]

    @property
    def observation_eligible(self) -> bool:
        return not self.exclusion_reasons


@dataclass(frozen=True, slots=True, kw_only=True)
class BirthAudit(Immutable):
    release: BirthRelease
    source: SourceManifest
    mapping_hash: str
    profile_hash: str
    exposure_hash: str | None
    input_count: int
    eligible_count: int
    excluded_count: int
    attrition: tuple[tuple[str, int], ...]  # disjoint, first exclusion reason
    exclusion_counts: tuple[tuple[str, int], ...]  # overlapping reasons
    measurement_counts: tuple[tuple[str, int], ...]
    duplicate_registration_groups: tuple[tuple[str, ...], ...]
    possible_duplicate_rows: tuple[tuple[str, ...], ...]
    blockers: tuple[str, ...]
    feature_registry: FeatureRegistry
    covariate_roles: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]

    def assert_anchor_ready(self) -> None:
        """Observation gate only; fitting also needs approved covariates/splits."""
        require(not self.blockers, 'anchor blocked: ' + '; '.join(self.blockers))
        require(self.eligible_count > 0, 'anchor blocked: no eligible observations')


def _geography(row, fields):
    parts = tuple(row[name].strip() for name in fields)
    return parts if all(parts) else None


def _exposure_lookup(exposure, release, mapping, profile, approvals):
    if exposure is None:
        return {}, ('residence exposure manifest missing',), None
    require(type(exposure) is ResidenceExposure, 'immutable ResidenceExposure required')
    require(mapping.expected_exposure_hash == exposure.content_hash, 'exposure hash mismatch')
    require(exposure.country == release.country, 'exposure country mismatch')
    require(exposure.geography_vintage == release.geography_vintage, 'geography vintage mismatch')
    require(release.content_hash in exposure.release_hashes, 'explicit release exposure mapping required')
    require(exposure.residence_fields == tuple(profile['residence']), 'residence fields mismatch')
    approval = approvals.get('owner_decisions', {}).get('birth_exposure', {}).get(release.country, {})
    if approval.get('status') != 'approved' or approval.get('manifest_hash') != exposure.content_hash:
        return {}, ('residence exposure lacks owner approval',), exposure.content_hash
    return {a.residence: a for a in exposure.assignments}, (), exposure.content_hash


def _load(bundle, release, mapping, exposure_manifest, country, weight_reader, scale):
    require(type(release) is BirthRelease and type(mapping) is BirthMapping, 'birth contracts required')
    require(release.country == country, 'country-specific adapter mismatch')
    require(release.population == 'live_births', 'live-birth-only bundle required; separate fetal deaths')
    require(mapping.release_hash == release.content_hash, 'release mapping mismatch')
    profile = _profile(release.dictionary_profile)
    require(profile['country'] == country, 'dictionary country mismatch')
    require(mapping.profile_hash == profile_hash(release.dictionary_profile), 'dictionary profile hash mismatch')
    mapping.source.assert_usable()
    require(mapping.source.version == release.release_id, 'source release identity mismatch')
    payload = bundle if isinstance(bundle, bytes) else Path(bundle).read_bytes()
    require(sha256(payload).hexdigest() == mapping.source.payload_hash, 'source payload hash mismatch')
    reader = csv.DictReader(io.StringIO(payload.decode(mapping.encoding)), delimiter=mapping.delimiter)
    headers = tuple(reader.fieldnames or ())
    require(bool(headers), 'empty birth schema')
    unique(headers, 'birth columns')
    require(mapping.source.schema_hash == schema_hash(tuple(
        ColumnSpec(name=n, dtype='string', nullable=False) for n in headers)), 'source schema mismatch')
    expected = {profile['weight']: 'weight'}
    for role in ('residence', 'delivery', 'occurrence', 'registration'):
        expected.update({field: f'{role}_{i}' for i, field in enumerate(profile[role])})
    source_fields = dict(mapping.source.field_mapping)
    require(all(source_fields.get(k) == v for k, v in expected.items()),
            'release-specific field mapping mismatch (residence is not delivery)')
    required = set(source_fields) | {i.field for i in mapping.identifiers} | {e.field for e in mapping.exclusions}
    require(required <= set(headers), 'mapped field missing from payload')
    approvals = _owner_approvals()
    for exclusion in mapping.exclusions:
        approved = approvals.get('owner_decisions', {}).get('birth_population_exclusions', [])
        require(exclusion.content_hash in approved, 'population exclusion lacks owner approval')
    lookup, blockers, exposure_hash = _exposure_lookup(
        exposure_manifest, release, mapping, profile, approvals)
    if release.status == 'unknown':
        blockers += ('release status unknown',)
    rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            'ragged CSV records')
    ids = tuple(f'{country}:{release.content_hash}:{mapping.source.payload_hash}:{i + 1}'
                for i in range(len(rows)))
    registration_groups, row_groups = {}, {}
    for oid, row in zip(ids, rows):
        row_groups.setdefault(tuple(row.items()), []).append(oid)
        for ident in mapping.identifiers:
            value = row[ident.field].strip()
            if ident.kind == 'registration' and value not in ident.unknown_codes:
                registration_groups.setdefault((ident.namespace, value), []).append(oid)
    duplicates = tuple(tuple(v) for v in registration_groups.values() if len(set(v)) > 1)
    duplicate_ids = {oid for group in duplicates for oid in group}
    possible = tuple(tuple(v) for v in row_groups.values() if len(v) > 1)
    records = []
    for oid, row in zip(ids, rows):
        grams, band, lbw, status = weight_reader(row[profile['weight']], profile)
        residence = _geography(row, profile['residence'])
        delivery = _geography(row, profile['delivery'])
        assignment = lookup.get(residence)
        reasons = []
        for rule in mapping.exclusions:
            if row[rule.field].strip() in rule.values:
                reasons.append('population:' + rule.reason)
        if oid in duplicate_ids:
            reasons.append('duplicate_registration')
        if (lbw is None if scale == 'risk_difference' else grams is None):
            reasons.append('weight:' + status)
        if residence is None:
            reasons.append('missing_residence')
        elif assignment is None:
            reasons.append('unmapped_residence')
        links, identifiers = [], []
        if residence is not None:
            links.append(EntityLink(observation_id=oid, relation='municipality',
                                    namespace=f'{country}:{release.geography_vintage}',
                                    entity_id=canonical_json(residence)))
        if assignment is not None:
            links.append(EntityLink(observation_id=oid, relation='repeated_geography',
                                    namespace=assignment.lineage_namespace, entity_id=assignment.lineage_id))
        for ident in mapping.identifiers:
            value = row[ident.field].strip()
            identifiers.append((ident.kind, ident.namespace, row[ident.field]))
            if value not in ident.unknown_codes:
                relation = ident.kind if ident.kind in ('household', 'psu') else 'outcome_lineage'
                links.append(EntityLink(observation_id=oid, relation=relation,
                                        namespace=ident.namespace, entity_id=value))
        records.append(BirthRecord(
            original_id=oid, release=release, source_hash=mapping.source.payload_hash,
            mapping_hash=mapping.content_hash, raw=tuple(row.items()), outcome_scale=scale,
            weight_raw=row[profile['weight']], birth_weight_g=grams, weight_band=band, lbw=lbw,
            measurement_status=status, maternal_residence=residence, delivery_geography=delivery,
            occurrence=tuple(row[n] for n in profile['occurrence']),
            registration=tuple(row[n] for n in profile['registration']), identifiers=tuple(identifiers),
            links=tuple(links), exposure_mmhg=assignment.exposure_mmhg if assignment else None,
            exposure_hash=exposure_hash, exclusion_reasons=tuple(dict.fromkeys(reasons))))
    # These semantic classes grant no training permission. Even baseline candidates
    # need owner approval and a reviewed feature registry in the fitting unit.
    roles = {name: role for role, names in profile['covariate_roles'].items() for name in names}
    registry = FeatureRegistry(registry_id=f'{release.dictionary_profile}:unapproved', rules=tuple(
        FeatureRule(name=name, role='predictor' if role == 'baseline_candidate' else 'downstream_health',
                    endpoints=(), uses=(), approval_id=None) for name, role in roles.items()))
    if any(any(x in r.exclusion_reasons for x in ('missing_residence', 'unmapped_residence'))
           and not any(x.startswith('population:') for x in r.exclusion_reasons) for r in records):
        blockers += ('incomplete residence assignment',)
    eligible = sum(r.observation_eligible for r in records)
    audit = BirthAudit(
        release=release, source=mapping.source, mapping_hash=mapping.content_hash,
        profile_hash=mapping.profile_hash, exposure_hash=exposure_hash, input_count=len(rows),
        eligible_count=eligible, excluded_count=len(rows) - eligible,
        attrition=tuple(sorted(Counter(r.exclusion_reasons[0] if r.exclusion_reasons else 'eligible'
                                     for r in records).items())),
        exclusion_counts=tuple(sorted(Counter(x for r in records for x in r.exclusion_reasons).items())),
        measurement_counts=tuple(sorted(Counter(r.measurement_status for r in records).items())),
        duplicate_registration_groups=duplicates, possible_duplicate_rows=possible, blockers=blockers,
        feature_registry=registry, covariate_roles=tuple(sorted(roles.items())), notes=(
            'Live births only: conditioning on live birth does not identify pregnancy survival effects.',
            'Possible identical rows are audited, not automatically deleted: twins may agree on public fields.',
            'No identifiers are inferred from maternal age, birth date or demographic similarity.',
            'No nuisance/SSL/context permissions are granted by this adapter.',
            'Registration periods and missing row dates are not inferred from event/release years.',
            'Observation eligibility does not establish causal, covariate or geographic-split eligibility.'))
    return tuple(records), audit


def load_births(bundle, release, mapping, exposure_manifest):
    """Return (immutable DANE records, audit); grams are always absent."""
    return _load(bundle, release, mapping, exposure_manifest, 'COL', _band_weight, 'risk_difference')
