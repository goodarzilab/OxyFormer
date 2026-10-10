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


def sf1_fixture(path, *, broken=False, zero=False):
    records = [('040', '01', 0 if zero else 100), ('140', IDS[0], 0 if zero else 100),
               ('101', IDS[0] + '1001', 0 if zero else 25), ('101', IDS[0] + '1002', 0 if zero else 75)]
    geography, segment = [], []
    for n, (level, ident, pop) in enumerate(records, 1):
        line = list(' ' * 500)
        fields = [(0, 'SF1ST '), (6, 'AL'), (8, level), (11, '00'), (13, '000'),
                  (18, f'{n:07}'), (27, '01'), (318, f'{pop:09}'),
                  (336, '+32.0000000' if n == 3 else '+36.0000000'),
                  (347, '-090.0000000' if n == 3 else '-086.0000000')]
        for offset, value in fields:
            line[offset:offset + len(value)] = value
        if level != '040':
            line[29:32], line[54:60] = ident[2:5], ident[5:11]
        if level == '101':
            line[61:65] = ident[11:]
        if broken and n == 3:
            line[336:347] = ' ' * 11
        geography.append(''.join(line))
        segment.append(f'SF1ST,AL,000,01,{n:07},{pop}')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('algeo2010.sf1', '\n'.join(geography) + '\n')
        z.writestr('al000012010.sf1', '\n'.join(segment) + '\n')
    return path


def test_sf1_population_weighted_internal_points_and_no_zero_fallback(tmp_path):
    path = sf1_fixture(tmp_path / 'points.zip')
    assert tract_inputs.sf1_coordinates(path, 'AL', '01') == {IDS[0]: (35., -87.)}
    assert tract_inputs.sf1_coordinates(sf1_fixture(tmp_path / 'zero.zip', zero=True), 'AL', '01') == {}
    with pytest.raises(ContractError, match='internal point'):
        tract_inputs.sf1_coordinates(sf1_fixture(tmp_path / 'bad.zip', broken=True), 'AL', '01')


def test_geography_uses_owner_grid_and_keeps_flags_out_of_predictors(tmp_path):
    from pyproj import Transformer
    from math import floor
    frame, metadata, sources, registry, _ = load_fixture(tmp_path)
    manifest, covariates, _ = tract_inputs.typed_covariates(frame, metadata, sources, registry, request_fixture(tmp_path))
    coords = {oid: (35., -87.) for oid in IDS}
    geography = tract_inputs.typed_geography(manifest, metadata, coords)
    x, y = Transformer.from_crs('EPSG:4269', 'EPSG:5070', always_xy=True).transform(-87., 35.)
    assert {r.subblock for r in geography.rows} == {f'01001:{floor(x / 10000)}:{floor(y / 10000)}'}
    assert [r.assignment_geography for r in geography.rows] == IDS
    assert [r.outcome_flag for r in geography.rows] == [1, 2, 3]
    assert [r.label_available for r in geography.rows] == [True, False, False]
    with pytest.raises(ContractError, match='lacks a population'):
        tract_inputs.typed_geography(manifest, metadata, {})


@pytest.mark.parametrize('change', ['absent', 'placement', 'grid'])
def test_owner_decision_absent_or_changed_refused(tmp_path, monkeypatch, change):
    approvals = yaml.safe_load((ROOT / 'configs/approvals.yaml').read_text())
    if change == 'absent':
        del approvals['owner_decisions']['tract_design']
    elif change == 'placement':
        approvals['owner_decisions']['tract_design']['placement_scenario'] = 'centroid'
    else:
        approvals['owner_decisions']['tract_design']['subblock']['cell_size_m'] = 20000
    (tmp_path / 'configs/execution/tasks').mkdir(parents=True)
    (tmp_path / 'configs/approvals.yaml').write_text(yaml.safe_dump(approvals))
    (tmp_path / tract_inputs.TASK_FILE).write_bytes((ROOT / tract_inputs.TASK_FILE).read_bytes())
    monkeypatch.setattr(tract_inputs, 'ROOT', tmp_path)
    with pytest.raises(ContractError, match='absent or changed'):
        tract_inputs.tract_decisions()


def collected_fixture(tmp_path, *, incomplete=False):
    """Use the merged physical producer, two tracts/scenarios on a tiny raster."""
    from test_exposure import sources, blocks, SPEC, write_raster
    from oxyformer.exposure.build import build_exposure
    import numpy as np
    tile = write_raster(tmp_path / "dem.tif", values=(1500., 1500.))
    frame, quality = build_exposure(sources(tile), blocks(), SPEC)
    # Keep distinct scenario values to detect accidental centroid selection.
    frame.loc[frame.scenario == 'centroid', 'oxygen_deficit_mmhg'] = 99.
    if incomplete:
        from oxyformer.exposure.quality import REASONS
        mask = frame.scenario == 'distributed'
        frame.loc[mask, ['pressure_mmhg','oxygen_deficit_mmhg','elevation_p10_m','elevation_p50_m','elevation_p90_m']] = np.nan
        frame.loc[mask, 'missing_population'] = frame.loc[mask, 'population']
        frame.loc[mask, 'covered_population'] = 0
        frame.loc[mask, 'status'] = 'missing_dem'
        for row in quality['blocks']:
            if row['scenario'] == 'distributed':
                row['covered_population'] = 0
                row['nodata'] = row['population']
    frame.to_parquet(tmp_path / 'atlas.parquet', index=False)
    (tmp_path / 'quality.json').write_text(canonical_json(quality))
    publication = dict(kind='atlas-collect', status='pass', source_identities=quality['source_identities'],
        files={n:file_hash(tmp_path / n) for n in ('atlas.parquet','quality.json')})
    (tmp_path / 'artifact_manifest.json').write_text(canonical_json(publication))
    return {('atlas-collect', n): tmp_path / n for n in ('atlas.parquet','quality.json','artifact_manifest.json')}


def test_collected_atlas_owner_mapping_and_no_imputation(tmp_path):
    from oxyformer.design.gate import collected_atlas
    paths = collected_fixture(tmp_path)
    atlas = collected_atlas(paths)
    assert atlas.coverage_complete
    assert all(r.inhabited_elevation_m == pytest.approx(1500.) and r.exposure_mmhg != 99. for r in atlas.rows)
    assert all(r.allocation_qualified for r in atlas.rows)
    other = tmp_path / 'missing'
    other.mkdir()
    atlas = collected_atlas(collected_fixture(other, incomplete=True))
    assert not atlas.rows and atlas.missing_tract_ids == atlas.expected_tract_ids
    assert not atlas.coverage_complete


def test_registry_and_tasks_declare_census_and_typed_handoff():
    tasks = yaml.safe_load((ROOT / tract_inputs.TASK_FILE).read_text())['tasks']
    stages = yaml.safe_load((ROOT / 'configs/execution/stages.yaml').read_text())['stages']
    for task in tasks:
        assert task['needs'] == stages[task['stage']]['needs']
        assert task['outputs'] == stages[task['stage']]['outputs']
    assert set(tasks[0]['needs']) == {'fetch-us', 'fetch-census'}
    assert tasks[1]['needs']['tract-inputs'] == tasks[0]['outputs']


def envelope_request(tmp_path, stage, roots):
    from dataclasses import replace
    from oxyformer.execution.runner import read_mapping
    task = next(t for t in yaml.safe_load((ROOT / tract_inputs.TASK_FILE).read_text())['tasks'] if t['stage'] == stage)
    settings = read_mapping(ROOT / 'configs/execution/stages.yaml')['stages'][stage]
    config = dict(stage=stage, settings=settings, approvals=read_mapping(ROOT / 'configs/approvals.yaml'),
        input_sources={str(ROOT / 'configs/approvals.yaml'):file_hash(ROOT / 'configs/approvals.yaml')},
        dependencies={k:str(v) for k,v in roots.items()})
    tmp_path.mkdir(parents=True, exist_ok=True)
    request = request_fixture(tmp_path)
    Path(request.config_path).write_text(canonical_json(config))
    Path(request.task_path).write_text(canonical_json(task))
    paths = tuple(str(roots[k] / n) for k,names in task['needs'].items() for n in names)
    Path(request.output_dir).mkdir()
    return replace(request, stage=stage, config_hash=file_hash(request.config_path), task_hash=file_hash(request.task_path),
                   dependency_paths=paths, dependency_hashes=tuple(file_hash(p) for p in paths))


def test_stage_publishes_typed_products_and_gate_accepts_dispatch_handoff(tmp_path, monkeypatch):
    from oxyformer.design import gate
    from oxyformer.provenance import read_artifact
    from oxyformer.contracts import CovariateView
    acquisition, source, mapping = acquisition_fixture(tmp_path / 'raw-acquisition', extra_acs=True)
    census = tmp_path / 'census'
    census.mkdir()
    for n in ('payload.tar','receipts.json'):
        (census / n).write_text('synthetic binding only; SF1 parser tested separately')
    load = tract_inputs.load_us_inputs
    monkeypatch.setattr(tract_inputs, 'load_us_inputs', lambda p,r,s: load(p,r,s,source=source,mapping=mapping,states={'AL':'01'}))
    monkeypatch.setattr(tract_inputs, 'census_coordinates', lambda *args: {oid:(35.,-87.) for oid in IDS})
    request = envelope_request(tmp_path / 'producer', 'tract-inputs', {'fetch-us':acquisition,'fetch-census':census})
    result = tract_inputs.run_stage(request)
    result.verify(request)
    assert result.status == 'pass'
    pool_record = next(a for a in result.artifacts if a.path == 'acs_pool.parquet')
    assert pool_record.lineage.unit_ids == tuple(IDS + ['01001000400'])
    out = Path(request.output_dir)
    view = read_artifact(out / 'covariates.json', CovariateView, file_hash(out / 'covariates.json'))
    assert view.original_ids == tuple(IDS)
    assert view.column('female_share') == (.01, .01, .01)
    atlas_dir = tmp_path / 'atlas'
    atlas_dir.mkdir()
    collected_fixture(atlas_dir)
    gate_request = envelope_request(tmp_path / 'gate', 'tract-support-gate', {'tract-inputs':out,'atlas-collect':atlas_dir})
    assert gate.design_configuration(gate_request)['stage'] == 'tract_design'
    values = gate._inputs(gate_request, json.loads(Path(gate_request.task_path).read_text()))
    assert values['covariates'] == view
    assert values['geography'].data_manifest_hash == values['data_manifest'].content_hash
    assert values['atlas'].rows[0].inhabited_elevation_m == pytest.approx(1500.)
    # Request binding cannot be replaced with an unbound path.
    config = json.loads(Path(gate_request.config_path).read_text())
    config['dependencies']['tract-inputs'] = str(tmp_path / 'unbound')
    Path(gate_request.config_path).write_text(canonical_json(config))
    from dataclasses import replace
    with pytest.raises(ContractError, match='not hash-bound'):
        gate._inputs(replace(gate_request,config_hash=file_hash(gate_request.config_path)), {})


def test_atlas_coverage_is_endpoint_scoped_and_missing_ids_stay_explicit(tmp_path):
    from oxyformer.design.gate import collected_atlas
    paths = collected_fixture(tmp_path)
    endpoint = collected_atlas(paths, (IDS[0],))
    assert endpoint.coverage_complete and endpoint.expected_tract_ids == (IDS[0],)
    missing = collected_atlas(paths, (IDS[0], IDS[1]))
    assert not missing.coverage_complete and missing.missing_tract_ids == (IDS[1],)
    assert tuple(r.tract_id for r in missing.rows) == (IDS[0],)


@pytest.mark.parametrize('pin', ['expected_sha256', 'expected_bytes'])
def test_null_optional_us_acquisition_pin_is_absent_not_mismatched(tmp_path, pin):
    acquisition, source, mapping = acquisition_fixture(tmp_path)
    source['resources'][0][pin] = None
    receipt_path = acquisition / 'receipts.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['manifest_sha256'] = sha256(canonical_json(source).encode()).hexdigest()
    receipt_path.write_text(canonical_json(receipt))
    frame, metadata, *_ = tract_inputs.load_us_inputs(acquisition / 'payload.tar', receipt_path,
        tmp_path / 'scratch', source=source, mapping=mapping, states={'AL': '01'})
    assert tuple(frame.original_id) == tuple(metadata.original_id) == tuple(IDS)


def test_census_acquisition_streams_sf1_and_accepts_null_optional_pins(tmp_path):
    archive = sf1_fixture(tmp_path / 'sf1.zip')
    blob = archive.read_bytes()
    payload = tmp_path / 'payload.tar'
    with tarfile.open(payload, 'w') as tar:
        info = tarfile.TarInfo('census/sf1_al.zip')
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    resource = dict(id='sf1_al', destination='census/sf1_al.zip', bytes=len(blob), sha256=sha256(blob).hexdigest())
    for null in (False, True):
        spec = dict(id=resource['id'], destination=resource['destination'], max_bytes=len(blob),
            expected_bytes=None if null else len(blob), expected_sha256=None if null else resource['sha256'])
        source = dict(id='census', resources=[spec])
        receipt = dict(status='complete', manifest_id='census',
            manifest_sha256=sha256(canonical_json(source).encode()).hexdigest(),
            payload_sha256=file_hash(payload), payload_bytes=payload.stat().st_size, resources=[resource])
        receipt_path = tmp_path / ('receipt-' + str(null) + '.json')
        receipt_path.write_text(canonical_json(receipt))
        scratch = tmp_path / ('scratch-' + str(null))
        scratch.mkdir()
        coordinates = tract_inputs.census_coordinates(payload, receipt_path, scratch,
            source=source, states={'AL': '01'})
        assert coordinates == {IDS[0]: (35., -87.)}
        assert not list(scratch.iterdir())


from test_execution import runtime, acquisition, publication_authority


def test_runner_accepts_identical_approval_copy_for_tract_inputs(runtime, acquisition, tmp_path, monkeypatch):
    import shutil
    from test_execution import commit
    from oxyformer.execution.runner import run
    from oxyformer.contracts import StageResult
    repo, out = runtime
    for name in ('configs/approvals.yaml', 'configs/execution/stages.yaml', tract_inputs.TASK_FILE):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (out / 'code_commit.txt').write_text(commit(repo))
    owner_copy = tmp_path / 'owner-copy.yaml'
    owner_copy.write_bytes((repo / 'configs/approvals.yaml').read_bytes())
    census = tmp_path / 'census-acquisition'
    shutil.copytree(acquisition, census)
    monkeypatch.setenv('SWARM_DEP_FETCH_US', str(acquisition))
    monkeypatch.setenv('SWARM_DEP_FETCH_CENSUS', str(census))
    monkeypatch.setattr(tract_inputs, 'ROOT', repo)
    def admit(request, module, checkout):
        paths = tract_inputs.dispatch_inputs(request)
        assert set(paths) == {(unit, name) for unit in ('fetch-us', 'fetch-census')
                              for name in ('payload.tar', 'receipts.json')}
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                           message='fixture admitted before source parsing')
    result = run('tract-inputs', out, repo, deps_env=True, task_file=repo / tract_inputs.TASK_FILE,
                 task_id='tract-inputs', approvals=owner_copy, execute=admit)
    assert (result.status, result.message) == ('blocked', 'fixture admitted before source parsing')


def test_gate_accepts_identical_approval_copy_but_refuses_changed_bytes(tmp_path):
    from dataclasses import replace
    from oxyformer.design.gate import OWNER_APPROVALS, DESIGN_CONFIG, design_configuration
    from oxyformer.execution.runner import read_mapping
    owner_copy = tmp_path / 'owner-copy.yaml'
    owner_copy.write_bytes(OWNER_APPROVALS.read_bytes())
    request = request_fixture(tmp_path)
    envelope = dict(stage='tract-support-gate', settings={'module':'oxyformer.design.gate'},
        approvals=read_mapping(OWNER_APPROVALS), input_sources={str(owner_copy):file_hash(owner_copy)})
    Path(request.config_path).write_text(canonical_json(envelope))
    request = replace(request, stage='tract-support-gate', config_hash=file_hash(request.config_path))
    assert design_configuration(request) == read_mapping(DESIGN_CONFIG)
    owner_copy.write_text('modified owner document')
    with pytest.raises(ContractError):
        design_configuration(request)


def collected_design_request(tmp_path, values, *, missing, accounted, absent=(), stage='tract-support-gate'):
    """Synthetic collected publication and typed endpoint, through real dispatch."""
    from dataclasses import replace
    import pandas as pd
    from oxyformer.provenance import write_artifact
    inputs, atlas_dir = tmp_path / 'inputs', tmp_path / 'atlas'
    inputs.mkdir(parents=True)
    atlas_dir.mkdir()
    for role in ('data_manifest', 'covariates', 'geography', 'entity_graph'):
        write_artifact(inputs / (role + '.json'), values[role])
    (inputs / 'acs_pool.parquet').write_bytes(b'unused synthetic SSL pool binding')
    (inputs / 'input_audit.json').write_text('{}')
    records = []
    for row in values['atlas'].rows:
        if row.tract_id in absent:
            continue
        omitted = row.tract_id in missing
        records.append(dict(tract_id=row.tract_id, scenario='distributed', population=row.population,
            missing_population=row.population if omitted else 0., status='missing_dem' if omitted else 'complete',
            oxygen_deficit_mmhg=float('nan') if omitted else row.exposure_mmhg,
            elevation_p50_m=float('nan') if omitted else row.inhabited_elevation_m))
    pd.DataFrame(records).to_parquet(atlas_dir / 'atlas.parquet', index=False)
    # The collector writes the same validated omission summary into both files.
    quality = {'missing_dem_tracts': sorted(accounted)}
    (atlas_dir / 'quality.json').write_text(canonical_json(quality))
    publication = dict(kind='atlas-collect', status='pass', **quality,
        source_identities={'synthetic': values['atlas'].source_hashes[0]},
        files={n: file_hash(atlas_dir / n) for n in ('atlas.parquet', 'quality.json')})
    (atlas_dir / 'artifact_manifest.json').write_text(canonical_json(publication))
    req = envelope_request(tmp_path / 'request', 'tract-support-gate',
                           {'tract-inputs': inputs, 'atlas-collect': atlas_dir})
    if stage == 'real-frame-inputs':
        task = yaml.safe_load((ROOT / 'configs/execution/tasks/campaign_profiles.yaml').read_text())['tasks'][0]
        config = json.loads(Path(req.config_path).read_text())
        config['stage'] = stage
        Path(req.config_path).write_text(canonical_json(config))
        Path(req.task_path).write_text(canonical_json(task))
        req = replace(req, stage=stage, config_hash=file_hash(req.config_path), task_hash=file_hash(req.task_path))
    return req


def test_collector_quality_summary_accounts_only_missing_endpoint_tracts(tmp_path):
    from oxyformer.design import gate
    from oxyformer.exposure.build import _coverage_summary
    import pandas as pd
    paths = collected_fixture(tmp_path, incomplete=True)
    quality_path = paths['atlas-collect', 'quality.json']
    quality = json.loads(quality_path.read_text())
    summary = _coverage_summary(pd.read_parquet(paths['atlas-collect', 'atlas.parquet']), quality)
    quality.update(summary)
    quality_path.write_text(canonical_json(quality))
    publication_path = paths['atlas-collect', 'artifact_manifest.json']
    publication = json.loads(publication_path.read_text())
    publication.update(summary)
    publication['files']['quality.json'] = file_hash(quality_path)
    publication_path.write_text(canonical_json(publication))
    expected = (IDS[0], IDS[1])
    values = gate.collected_atlas_inputs(paths, expected)
    values['approvals'] = yaml.safe_load(gate.OWNER_APPROVALS.read_text())
    audit = gate.atlas_coverage(values)
    assert audit['accounted_missing_dem_tract_ids'] == (IDS[0],)
    assert audit['unaccounted_missing_tract_ids'] == (IDS[1],)
    assert not audit['complete']
    assert not values['atlas'].rows  # No missing exposure was filled in.
    quality_path.write_text(quality_path.read_text() + ' ')
    with pytest.raises(ContractError, match='binding mismatch'):
        gate.collected_atlas_inputs(paths, expected)


def test_collected_complete_tract_is_not_excluded_for_other_scenario_omission(tmp_path):
    from oxyformer.design import gate
    paths = collected_fixture(tmp_path)
    publication_path = paths['atlas-collect', 'artifact_manifest.json']
    publication = json.loads(publication_path.read_text())
    publication['missing_dem_tracts'] = [IDS[0]]  # e.g. centroid-only missing coverage.
    publication_path.write_text(canonical_json(publication))
    values = gate.collected_atlas_inputs(paths, (IDS[0],))
    assert values['atlas'].coverage_complete
    assert values['atlas_missing_dem_tract_ids'] == ()


@pytest.mark.parametrize('selected_status', ['absent', 'zero_population'])
@pytest.mark.parametrize('stage', ['tract-support-gate', 'real-frame-inputs'])
def test_other_scenario_dem_does_not_account_for_unexplained_selected_omission(tmp_path, selected_status, stage):
    from dataclasses import replace
    import pandas as pd
    from oxyformer.design import gate
    from oxyformer.design.splits import reserve_design
    from oxyformer.validation import real_frame
    from test_campaign_profiles import real_fixture
    values = real_fixture()
    reserved = set(reserve_design(values['geography'].rows, values['entity_graph']).design_ids)
    missing = next(r.original_id for r in values['geography'].rows if r.original_id not in reserved)
    req = collected_design_request(tmp_path, values, missing=(), accounted=(missing,),
        absent=(missing,) if selected_status == 'absent' else (), stage=stage)
    atlas_dir = tmp_path / 'atlas'
    parquet = atlas_dir / 'atlas.parquet'
    frame = pd.read_parquet(parquet)
    if selected_status == 'zero_population':
        mask = frame.tract_id == missing
        frame.loc[mask, 'population'] = 0.
        frame.loc[mask, 'status'] = 'zero_population'
        frame.loc[mask, ['oxygen_deficit_mmhg', 'elevation_p50_m']] = float('nan')
    # An accurate cross-scenario summary cannot explain why the selected row
    # is absent/zero-population rather than a distributed missing-DEM record.
    centroid = dict(tract_id=missing, scenario='centroid', population=100., missing_population=100.,
                    status='missing_dem', oxygen_deficit_mmhg=float('nan'), elevation_p50_m=float('nan'))
    pd.concat([frame, pd.DataFrame([centroid])], ignore_index=True).to_parquet(parquet, index=False)
    publication_path = atlas_dir / 'artifact_manifest.json'
    publication = json.loads(publication_path.read_text())
    publication['files']['atlas.parquet'] = file_hash(parquet)
    publication_path.write_text(canonical_json(publication))
    req = replace(req, dependency_hashes=tuple(file_hash(path) for path in req.dependency_paths))
    result = (gate if stage == 'tract-support-gate' else real_frame).run_stage(req)
    assert result.status == 'blocked', result.message
    assert 'incomplete atlas coverage' in result.message
