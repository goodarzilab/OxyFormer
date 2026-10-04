"""Synthetic offline observation-model tests; no source rows or production grants."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import csv
import io
import socket

import pytest

from oxyformer.contracts import ColumnSpec, SourceManifest, schema_hash
from oxyformer.data.adapters import dane_births as dane, inec_births as inec
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.provenance import ContractError


def digest(value):
    return sha256(value.encode()).hexdigest()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('network forbidden in birth adapter tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)


@pytest.fixture
def make(monkeypatch):
    def factory(profile='dane_2023', weights=('4',), overrides=None, identifiers=(), approve=True):
        p = dane._profile(profile)
        country = p['country']
        release = dane.BirthRelease(
            country=country, release_id=f'synthetic:{profile}:v1', dictionary_profile=profile,
            status='provisional', status_reference='synthetic://release-status',
            occurrence_period='synthetic occurrence coverage', registration_period='synthetic registration coverage',
            geography_vintage='synthetic-geography-v1', geography_reference='synthetic://geography',
            population='live_births', selection_reference='synthetic://live-birth-file')
        residence = ('01', '001') if country == 'COL' else ('01', '0101', '010101')
        delivery = ('02', '002') if country == 'COL' else ('02', '0202', '020202')
        fields = {p['weight']: 'weight'}
        for role in ('residence', 'delivery', 'occurrence', 'registration'):
            fields.update({name: f'{role}_{i}' for i, name in enumerate(p[role])})
        rows = []
        for n, weight in enumerate(weights):
            row = {p['weight']: weight}
            row.update(zip(p['residence'], residence))
            row.update(zip(p['delivery'], delivery))
            row.update(zip(p['occurrence'], ('2014', '12', '31')))
            row.update(zip(p['registration'], ('2015', '01', '02')))
            for ident in identifiers:
                row[ident.field] = f'known-{n}'
            if overrides:
                row.update(overrides[n])
            rows.append(row)
        headers = tuple(rows[0])
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        bundle = stream.getvalue().encode()
        source = SourceManifest(
            source_id='synthetic', version=release.release_id, uri='synthetic://csv',
            payload_hash=sha256(bundle).hexdigest(), license_hash=digest('synthetic license'),
            schema_hash=schema_hash(tuple(ColumnSpec(name=n, dtype='string') for n in headers)),
            field_mapping=tuple(fields.items()), mapping_status='reviewed', mapping_review_id='synthetic-only')
        exposure = dane.ResidenceExposure(
            country=country, geography_vintage=release.geography_vintage,
            residence_fields=tuple(p['residence']), release_hashes=(release.content_hash,),
            source_uri='synthetic://residence-survey', source_hash=digest('synthetic exposure source'),
            mapping_date='2026-10-04', placement='maternal_residence_point',
            placement_reference='synthetic://placement', geography_reference='synthetic://boundaries',
            review_id='synthetic-only', assignments=(
                dane.ResidenceAssignment(residence=residence, exposure_mmhg=10,
                                         lineage_namespace='synthetic:geography', lineage_id='original-place'),
                dane.ResidenceAssignment(residence=delivery, exposure_mmhg=40,
                                         lineage_namespace='synthetic:geography', lineage_id='delivery-place')))
        mapping = dane.BirthMapping(source=source, release_hash=release.content_hash,
                                   profile_hash=dane.profile_hash(profile),
                                   expected_exposure_hash=exposure.content_hash, identifiers=identifiers)
        approvals = {'owner_decisions': {'birth_exposure': {
            country: {'status': 'approved', 'manifest_hash': exposure.content_hash}}}} if approve else {}
        monkeypatch.setattr(dane, '_owner_approvals', lambda: approvals)
        return {'bundle': bundle, 'release': release, 'mapping': mapping, 'exposure_manifest': exposure}
    return factory


def load(case, **changes):
    case = {**case, **changes}
    loader = dane.load_births if case['release'].country == 'COL' else inec.load_births
    return loader(**case)


@pytest.mark.parametrize('year', (2023, 2024, 2025))
def test_dane_bands_are_never_fabricated_grams(make, year):
    records, audit = load(make(f'dane_{year}', tuple(map(str, range(1, 10)))))
    assert [r.lbw for r in records] == [True] * 4 + [False] * 4 + [None]
    assert all(r.birth_weight_g is None and r.outcome_scale == 'risk_difference' for r in records)
    assert records[3].weight_band.upper == 2499
    assert records[4].weight_band.lower == 2500
    assert records[-1].weight_raw == '9' and records[-1].measurement_status == 'unknown'
    assert (audit.input_count, audit.eligible_count, audit.excluded_count) == (9, 8, 1)
    assert sum(n for _, n in audit.attrition) == 9
    audit.assert_anchor_ready()


@pytest.mark.parametrize('low,high,inclusive,expected', [
    (None, 2499, True, True), (2000, 2500, False, True),
    (2000, 2500, True, None), (2499, 3000, True, None),
    (2500, None, True, False), (2500, 2500, True, False), (None, None, True, None)])
def test_strict_lbw_band_boundaries(low, high, inclusive, expected):
    assert dane.classify_lbw(dane.WeightBand(lower=low, upper=high, upper_inclusive=inclusive)) is expected


def test_ambiguous_band_survives_with_no_outcome(make, monkeypatch):
    original = dane._profile
    def crossing(key):
        profile = original(key)
        profile['bands']['4'] = [2000, 2600]  # synthetic dictionary variation, not a DANE assertion
        return profile
    monkeypatch.setattr(dane, '_profile', crossing)
    records, audit = load(make())
    assert records[0].weight_band.upper == 2600
    assert records[0].lbw is None and records[0].birth_weight_g is None
    assert records[0].measurement_status == 'ambiguous_band'
    assert audit.excluded_count == 1


@pytest.mark.parametrize('profile,max_grams', [('inec_2015', 5000), ('inec_2024', 5500)])
def test_inec_continuous_measurement_boundaries_and_sentinels(make, profile, max_grams):
    weights = ('499', '500', '2499', '2499.5', '2500', str(max_grams), str(max_grams+1),
               '99', '', 'NaN', 'Infinity', '-10', '30000', 'Sin información')
    records, audit = load(make(profile, weights))
    assert [r.birth_weight_g for r in records[:7]] == [None, 500, 2499, 2499.5, 2500, max_grams, None]
    assert all(r.weight_band is None and r.outcome_scale == 'grams' for r in records)
    assert records[2].lbw is True and records[4].lbw is False
    assert all(r.birth_weight_g is None for r in records[7:])
    assert records[7].measurement_status == ('ambiguous_sentinel' if profile == 'inec_2015' else 'unknown')
    assert [r.weight_raw for r in records] == list(weights)
    assert (audit.input_count, audit.eligible_count, audit.excluded_count) == (14, 5, 9)


def test_invalid_dane_codes_preserve_raw_values(make):
    records, _ = load(make(weights=('4.0', '9', '', '2500', '4.5', 'NaN')))
    assert records[0].lbw is True
    assert [r.measurement_status for r in records[1:]] == ['unknown', 'missing'] + ['unrecognized_code'] * 3
    assert records[3].weight_raw == '2500' and records[3].birth_weight_g is None


@pytest.mark.parametrize('profile', ('dane_2023', 'inec_2024'))
def test_residence_join_uses_residence_not_delivery(make, profile):
    records, _ = load(make(profile, ('2500',) if profile.startswith('inec') else ('4',)))
    assert records[0].exposure_mmhg == 10
    assert records[0].maternal_residence != records[0].delivery_geography
    assert records[0].links[-1].entity_id == 'original-place'


@pytest.mark.parametrize('profile', ('dane_2023', 'inec_2024'))
def test_missing_residence_never_falls_back_to_delivery(make, profile):
    field = dane._profile(profile)['residence'][0]
    weight = '2500' if profile.startswith('inec') else '4'
    records, audit = load(make(profile, (weight, weight), [{field: ''}, {}]))
    assert records[0].delivery_geography is not None
    assert records[0].maternal_residence is None
    assert records[0].exposure_mmhg is None
    assert 'missing_residence' in records[0].exclusion_reasons
    assert records[1].exposure_mmhg == 10
    with pytest.raises(ContractError, match='incomplete residence assignment'):
        audit.assert_anchor_ready()


def test_missing_and_unapproved_exposure_block_anchor(make):
    case = make()
    records, audit = load(case, exposure_manifest=None)
    assert records[0].exposure_mmhg is None
    with pytest.raises(ContractError, match='manifest missing'):
        audit.assert_anchor_ready()
    records, audit = load(make(approve=False))
    assert records[0].exposure_mmhg is None
    with pytest.raises(ContractError, match='owner approval'):
        audit.assert_anchor_ready()


def test_changed_exposure_bytes_and_geography_vintage_refused(make):
    case = make()
    exposure = replace(case['exposure_manifest'], geography_vintage='new-vintage')
    with pytest.raises(ContractError, match='hash mismatch'):
        load(case, exposure_manifest=exposure)
    mapping = replace(case['mapping'], expected_exposure_hash=exposure.content_hash)
    with pytest.raises(ContractError, match='vintage mismatch'):
        load(case, mapping=mapping, exposure_manifest=exposure)


def test_changed_geography_codes_do_not_silently_carry_forward(make):
    records, audit = load(make(overrides=[{'CODMUNRE': '099'}]))
    assert records[0].maternal_residence == ('01', '099')
    assert records[0].exposure_mmhg is None
    with pytest.raises(ContractError, match='incomplete residence'):
        audit.assert_anchor_ready()


def test_delivery_field_substitution_in_reviewed_mapping_is_rejected(make):
    case = make()
    source = case['mapping'].source
    swapped = tuple((name, {'residence_0': 'delivery_0', 'delivery_0': 'residence_0'}.get(dest, dest))
                    for name, dest in source.field_mapping)
    mapping = replace(case['mapping'], source=replace(source, field_mapping=swapped))
    with pytest.raises(ContractError, match='field mapping mismatch'):
        load(case, mapping=mapping)


@pytest.mark.parametrize('source', ('usgs:3dep', 'hospital:altitude', 'capital-city:altitude'))
def test_prohibited_exposure_substitutions(make, source):
    with pytest.raises(ContractError, match='prohibited exposure source'):
        replace(make()['exposure_manifest'], source_uri=source)


def test_prohibited_placement(make):
    with pytest.raises(ContractError):
        replace(make()['exposure_manifest'], placement='hospital')


def test_duplicate_registrations_and_maternal_lineage(make):
    identifiers = tuple(dane.Identifier(field=field, kind=kind, namespace=f'synthetic:{kind}',
                                       documentation='synthetic://identifier-dictionary')
                        for field, kind in [('REG', 'registration'), ('MOM', 'maternal')])
    records, audit = load(make(weights=('4', '5', '6'), identifiers=identifiers,
                              overrides=[{'REG': 'r1', 'MOM': 'm1'}, {'REG': 'r1', 'MOM': 'm1'},
                                         {'REG': 'r2', 'MOM': 'm1'}]))
    assert len(audit.duplicate_registration_groups) == 1
    assert [r.observation_eligible for r in records] == [False, False, True]
    assert records[0].identifiers == (('registration', 'synthetic:registration', 'r1'),
                                      ('maternal', 'synthetic:maternal', 'm1'))
    graph = EntityGraph(original_ids=tuple(r.original_id for r in records),
                        links=tuple(link for r in records for link in r.links))
    assert len(graph.components()) == 1
    with pytest.raises(ContractError, match='lineage crosses'):
        graph.assert_partition({r.original_id: i for i, r in enumerate(records)})
    assert dict(audit.exclusion_counts)['duplicate_registration'] == 2


def test_identical_rows_without_identifier_are_audited_not_invented_duplicates(make):
    records, audit = load(make(weights=('4', '4')))
    assert len({r.original_id for r in records}) == 2
    assert not audit.duplicate_registration_groups
    assert len(audit.possible_duplicate_rows) == 1
    assert all(r.observation_eligible for r in records)


def test_approved_population_exclusion_accounting(make, monkeypatch):
    case = make(weights=('4', '5'), overrides=[{'SCOPE': 'outside'}, {'SCOPE': 'inside'}])
    exclusion = dane.PopulationExclusion(field='SCOPE', values=('outside',),
                                        reason='synthetic-target', approval_id='synthetic-only')
    mapping = replace(case['mapping'], exclusions=(exclusion,))
    with pytest.raises(ContractError, match='population exclusion lacks owner approval'):
        load(case, mapping=mapping)
    approvals = dane._owner_approvals()
    approvals['owner_decisions']['birth_population_exclusions'] = [exclusion.content_hash]
    monkeypatch.setattr(dane, '_owner_approvals', lambda: approvals)
    records, audit = load(case, mapping=mapping)
    assert records[0].exclusion_reasons == ('population:synthetic-target',)
    assert (audit.input_count, audit.eligible_count, audit.excluded_count) == (2, 1, 1)
    assert dict(audit.attrition) == {'eligible': 1, 'population:synthetic-target': 1}


def test_release_identity_status_periods_and_vintage_are_retained(make):
    case = make('inec_2015', ('2500',))
    records, audit = load(case)
    record = records[0]
    assert record.release == case['release'] == audit.release
    assert record.occurrence == ('2014', '12', '31')
    assert record.registration == ('2015', '01', '02')
    definitive = replace(case['release'], status='definitive')
    assert definitive.content_hash != case['release'].content_hash
    with pytest.raises(ContractError, match='release mapping mismatch'):
        load(case, release=definitive)
    other = replace(case['release'], dictionary_profile='inec_2024')
    with pytest.raises(ContractError, match='profile hash mismatch'):
        load(case, release=other, mapping=replace(case['mapping'], release_hash=other.content_hash))


def test_no_silent_new_year_fallback(make):
    case = make()
    release = replace(case['release'], dictionary_profile='dane_2026')
    with pytest.raises(ContractError, match='explicit mapping required'):
        load(case, release=release, mapping=replace(case['mapping'], release_hash=release.content_hash))


def test_no_training_permissions_and_causal_role_distinctions(make):
    _, audit = load(make())
    assert dict(audit.covariate_roles)['EDAD_MADRE'] == 'baseline_candidate'
    assert dict(audit.covariate_roles)['T_GES'] == 'gestational_mediator'
    assert dict(audit.covariate_roles)['APGAR1'] == 'post_outcome'
    for name in ('EDAD_MADRE', 'T_GES', 'APGAR1', 'CODMUNRE'):
        with pytest.raises(ContractError):
            audit.feature_registry.require(name, 'colombia_lbw', 'nuisance')


def test_source_payload_schema_and_review_verification(make):
    case = make()
    with pytest.raises(ContractError, match='payload hash mismatch'):
        load(case, bundle=case['bundle'] + b'\n')
    for change, message in [({'schema_hash': '0'*64}, 'schema mismatch'),
                            ({'mapping_status': 'unreviewed'}, 'unreviewed source mapping')]:
        with pytest.raises(ContractError, match=message):
            load(case, mapping=replace(case['mapping'], source=replace(case['mapping'].source, **change)))


def test_live_birth_selection_is_required(make):
    case = make()
    release = replace(case['release'], population='fetal_deaths')
    with pytest.raises(ContractError, match='live-birth-only'):
        load(case, release=release)


def test_immutable_roundtrip_and_dane_missing_registration_dates(make):
    records, audit = load(make())
    assert records[0].registration == ()
    assert dane.BirthRecord.from_json(records[0].to_json()) == records[0]
    assert dane.BirthAudit.from_json(audit.to_json()) == audit
    with pytest.raises(FrozenInstanceError):
        records[0].exposure_mmhg = 40


def test_empty_file_has_zero_balanced_accounting(make):
    case = make()
    payload = case['bundle'].splitlines(keepends=True)[0]
    mapping = replace(case['mapping'], source=replace(case['mapping'].source,
                                                     payload_hash=sha256(payload).hexdigest()))
    records, audit = load(case, bundle=payload, mapping=mapping)
    assert records == ()
    assert audit.input_count == audit.eligible_count == audit.excluded_count == 0
    with pytest.raises(ContractError, match='no eligible'):
        audit.assert_anchor_ready()


def test_ecuador_parishes_share_canton_link_even_with_missing_parish(make, monkeypatch):
    case = make('inec_2024', ('2500', '2600', '2700'),
                [{}, {'parr_res': '010102'}, {'parr_res': ''}])
    exposure = case['exposure_manifest']
    extra = dane.ResidenceAssignment(residence=('01', '0101', '010102'), exposure_mmhg=12,
                                    lineage_namespace='synthetic:geography', lineage_id='another-parish')
    exposure = replace(exposure, assignments=exposure.assignments + (extra,))
    monkeypatch.setattr(dane, '_owner_approvals', lambda: {'owner_decisions': {'birth_exposure': {
        'ECU': {'status': 'approved', 'manifest_hash': exposure.content_hash}}}})
    records, audit = load(case, exposure_manifest=exposure,
                          mapping=replace(case['mapping'], expected_exposure_hash=exposure.content_hash))
    keys = [[(link.namespace, link.entity_id) for link in r.links if link.relation == 'municipality']
            for r in records]
    assert keys[0] == keys[1] == keys[2]
    assert [r.exposure_mmhg for r in records] == [10, 12, None]
    assert records[2].maternal_residence is None
    assert not records[2].observation_eligible


@pytest.mark.parametrize('contents', (None, '', 'null', '[]', 'owner_decisions: null',
                                      'owner_decisions: {birth_exposure: null}',
                                      'owner_decisions: {birth_exposure: {COL: null}}'))
def test_unavailable_owner_approvals_return_blocked_audit(make, monkeypatch, tmp_path, contents):
    real_reader = dane._owner_approvals
    case = make()
    profile = dane._profile('dane_2023')
    (tmp_path / 'configs').mkdir()
    if contents is not None:
        (tmp_path / 'configs/approvals.yaml').write_text(contents)
    monkeypatch.setattr(dane, '_ROOT', tmp_path)
    monkeypatch.setattr(dane, '_profile', lambda key: profile)
    monkeypatch.setattr(dane, '_owner_approvals', real_reader)
    records, audit = load(case)
    assert records[0].exposure_mmhg is None
    assert audit.input_count == audit.excluded_count == 1
    with pytest.raises(ContractError, match='owner approval'):
        audit.assert_anchor_ready()


@pytest.mark.parametrize('value', (float('inf'), float('-inf'), float('nan')))
def test_shared_immutable_contract_rejects_nonfinite_exposure(value):
    with pytest.raises(ContractError, match='nonfinite number'):
        dane.ResidenceAssignment(residence=('01', '001'), exposure_mmhg=value,
                                 lineage_namespace='synthetic', lineage_id='one')
