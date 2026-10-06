"""Offline tract acquisition and dispatcher admission regressions."""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest
import yaml

from oxyformer.contracts import StageRequest
from oxyformer.data import tract_inputs
from oxyformer.provenance import ContractError, canonical_json, file_hash
from test_us_adapters import acs_bundle, us_bundle, IDS

ROOT = Path(__file__).resolve().parents[1]


def acquisition_fixture(tmp_path, *, omit=None, duplicate=None, outcome_offset=0, extra_acs=False):
    raw = tmp_path / 'raw'
    raw.mkdir(parents=True)
    mapping = yaml.safe_load((ROOT / 'configs/adapters/us.yaml').read_text())
    files = acs_bundle(raw, mapping, ids=IDS + (['01001000400'] if extra_acs else []))
    usa = us_bundle(raw, mapping, edit=lambda rows: [r.__setitem__(4, str(float(r[4]) + outcome_offset)) for r in rows])
    compressed = io.BytesIO()
    with tarfile.open(fileobj=compressed, mode='w:gz') as archive:
        for item in files:
            name = item.path.name
            if name.startswith('g') or name == omit:
                continue
            payload = item.path.read_bytes()
            info = tarfile.TarInfo('nested/' + name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            if name == duplicate:
                archive.addfile(info, io.BytesIO(payload))
    geography = io.BytesIO()
    with zipfile.ZipFile(geography, 'w') as archive:
        geo = next(item.path for item in files if item.path.name.startswith('g'))
        archive.writestr('geog/' + geo.name, geo.read_bytes())
    blobs = {'usaleep_data': usa[0].path.read_bytes(), 'acs_tracts': compressed.getvalue(),
             'acs_geography': geography.getvalue(), 'usaleep_terms': b'USALEEP public terms',
             'acs_terms': b'ACS public terms', 'usaleep_dictionary': b'USALEEP layout',
             'acs_dictionary': b'ACS layout', 'acs_technical': b'ACS technical'}
    for key in ('usaleep', 'acs'):
        mapping[key]['dictionary_sha256'] = sha256(blobs[key + '_dictionary']).hexdigest()
    mapping['acs']['technical_sha256'] = sha256(blobs['acs_technical']).hexdigest()
    source = {'id': 'us', 'resources': []}
    receipt = {'status': 'complete', 'manifest_id': 'us', 'resources': []}
    acquisition = tmp_path / 'acquisition'
    acquisition.mkdir()
    with tarfile.open(acquisition / 'payload.tar', 'w') as outer:
        for rid, data in blobs.items():
            name = 'us/' + rid
            info = tarfile.TarInfo(name)
            info.size = len(data)
            outer.addfile(info, io.BytesIO(data))
            source['resources'].append(dict(id=rid, destination=name, max_bytes=len(data),
                expected_bytes=len(data), expected_sha256=sha256(data).hexdigest(),
                url='https://example.invalid/' + rid))
            receipt['resources'].append(dict(id=rid, destination=name, bytes=len(data),
                sha256=sha256(data).hexdigest()))
    receipt.update(manifest_sha256=sha256(canonical_json(source).encode()).hexdigest(),
                   payload_sha256=file_hash(acquisition / 'payload.tar'),
                   payload_bytes=(acquisition / 'payload.tar').stat().st_size)
    (acquisition / 'receipts.json').write_text(json.dumps(receipt))
    return acquisition, source, mapping


def load_fixture(tmp_path, **kwargs):
    acquisition, source, mapping = acquisition_fixture(tmp_path, **kwargs)
    return tract_inputs.load_us_inputs(acquisition / 'payload.tar', acquisition / 'receipts.json',
                tmp_path / 'scratch', source=source, mapping=mapping, states={'AL': '01'})


def request_fixture(tmp_path):
    config, task = tmp_path / 'config.json', tmp_path / 'task.json'
    config.write_text('{}')
    task.write_text('{}')
    return StageRequest(stage='tract-inputs', config_path=str(config), config_hash=file_hash(config),
        task_path=str(task), task_hash=file_hash(task), dependency_paths=(), dependency_hashes=(),
        output_dir=str(tmp_path / 'out'), code_identity='a' * 40)


def test_real_adapter_fields_stream_into_outcome_free_typed_covariates(tmp_path, monkeypatch):
    def no_extract(*args, **kwargs):
        raise AssertionError('whole archive extraction forbidden')
    monkeypatch.setattr(tarfile.TarFile, 'extractall', no_extract)
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', no_extract)
    frame, metadata, sources, registry, audit = load_fixture(tmp_path)
    assert tuple(frame.original_id) == tuple(IDS)
    assert audit['flag_counts'] == {'1': 1, '2': 1, '3': 1}
    assert 'life_expectancy_years' not in metadata
    manifest, covariates, graph = tract_inputs.typed_covariates(
        frame, metadata, sources, registry, request_fixture(tmp_path))
    assert covariates.original_ids == manifest.original_ids == graph.original_ids == tuple(IDS)
    assert covariates.column('female_share') == (.01, .01, .01)
    assert set(covariates.columns) == set(yaml.safe_load((ROOT / 'configs/adapters/us.yaml').read_text())['acs']['concepts'])
    assert manifest.weight_field is None
    assert all(s.mapping_status == 'reviewed' for s in manifest.sources)
    assert manifest.content_hash in covariates.lineage.parent_hashes
    for name in ['life_expectancy_years', 'mortality_input_flag', 'county_fips', 'oxygen_deficit_mmhg', 'original_id']:
        with pytest.raises(ContractError):
            covariates.column(name)


def test_outcome_perturbation_changes_provenance_but_never_covariate_values(tmp_path):
    results = []
    for label, offset in [('before', 0), ('after', 50)]:
        path = tmp_path / label
        frame, metadata, sources, registry, audit = load_fixture(path, outcome_offset=offset)
        results.append(tract_inputs.typed_covariates(frame, metadata, sources, registry, request_fixture(path)))
    assert results[0][1].values == results[1][1].values
    assert results[0][1].columns == results[1][1].columns
    assert results[0][2] == results[1][2]
    assert results[0][0].content_hash != results[1][0].content_hash


@pytest.mark.parametrize('option', ['omit', 'duplicate'])
def test_missing_or_duplicate_sequence_refused(tmp_path, option):
    with pytest.raises(ContractError, match='missing approved ACS|duplicate or nonregular ACS'):
        load_fixture(tmp_path, **{option: 'e20105al0010000.txt'})


def test_changed_receipt_source_authority_refused(tmp_path):
    acquisition, source, mapping = acquisition_fixture(tmp_path)
    changed = deepcopy(source)
    changed['resources'][0]['url'] = 'https://example.invalid/unreviewed'
    with pytest.raises(ContractError, match='differs from repository'):
        tract_inputs.load_us_inputs(acquisition / 'payload.tar', acquisition / 'receipts.json',
            tmp_path / 'scratch', source=changed, mapping=mapping, states={'AL': '01'})


def test_dispatch_envelope_uses_repository_science_and_owner_authority(tmp_path):
    from dataclasses import replace
    from oxyformer.design.gate import DESIGN_CONFIG, OWNER_APPROVALS, design_configuration
    from oxyformer.execution.runner import read_mapping
    request = request_fixture(tmp_path)
    envelope = {'stage': 'tract-support-gate', 'settings': {'module': 'oxyformer.design.gate',
                'support': {'raw_x_radius': 999}, 'buffers_km': [0]},
                'approvals': read_mapping(OWNER_APPROVALS),
                'input_sources': {str(OWNER_APPROVALS): file_hash(OWNER_APPROVALS)}}
    Path(request.config_path).write_text(json.dumps(envelope))
    request = replace(request, stage='tract-support-gate', config_hash=file_hash(request.config_path))
    assert design_configuration(request) == read_mapping(DESIGN_CONFIG)
    envelope['approvals']['plan_fixed']['support_design_fraction'] = .5
    Path(request.config_path).write_text(json.dumps(envelope))
    with pytest.raises(ContractError, match='approvals differ'):
        design_configuration(request)


def test_direct_api_cannot_override_repository_support_recipe(tmp_path):
    from dataclasses import replace
    from oxyformer.design.gate import DESIGN_CONFIG, design_configuration
    from oxyformer.execution.runner import read_mapping
    request = request_fixture(tmp_path)
    config = read_mapping(DESIGN_CONFIG)
    config['support']['raw_x_radius'] = 999
    Path(request.config_path).write_text(json.dumps(config))
    with pytest.raises(ContractError, match='settings differ'):
        design_configuration(replace(request, stage='tract_design'))


def test_official_usaleep_numeric_identifier_format_is_losslessly_normalized(tmp_path):
    from oxyformer.data.adapters.usaleep import load_usaleep
    mapping = yaml.safe_load((ROOT / 'configs/adapters/us.yaml').read_text())
    def numeric_identifiers(rows):
        for row in rows:
            for column in range(4):
                row[column] = str(int(row[column]))
    frame, audit = load_usaleep(us_bundle(tmp_path, mapping, edit=numeric_identifiers), mapping, numeric_identifiers=True)
    assert frame.original_id.tolist() == [IDS[0]]
    assert audit['metadata'].original_id.tolist() == IDS
    assert audit['metadata'].state_fips.tolist() == ['01'] * 3
    assert audit['metadata'].county_fips.tolist() == ['001'] * 3


@pytest.mark.parametrize('raw', ['1.0', ' 1', '-1', '123456789012'])
def test_usaleep_identifier_normalization_never_coerces_invalid_tokens(tmp_path, raw):
    from oxyformer.data.adapters.usaleep import load_usaleep
    mapping = yaml.safe_load((ROOT / 'configs/adapters/us.yaml').read_text())
    with pytest.raises(ValueError):
        load_usaleep(us_bundle(tmp_path, mapping, edit=lambda rows: rows[0].__setitem__(0, raw)), mapping, numeric_identifiers=True)


def test_unlabeled_acs_pool_is_preserved_before_endpoint_join(tmp_path):
    frame, metadata, sources, registry, audit = load_fixture(tmp_path, extra_acs=True)
    assert len(frame) == 4 and len(metadata) == 3
    assert audit['acs_without_usaleep'] == ['01001000400']
    manifest, covariates, graph = tract_inputs.typed_covariates(
        frame, metadata, sources, registry, request_fixture(tmp_path))
    assert manifest.original_ids == tuple(IDS)
    assert len(frame) == 4  # Typed endpoint construction never mutates the SSL pool.
