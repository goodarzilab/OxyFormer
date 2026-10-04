"""Offline synthetic rasters and independently assembled scientific checks."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import yaml

from oxyformer.contracts import SourceManifest, StageRequest
from oxyformer.provenance import ContractError, canonical_json, file_hash
from oxyformer.exposure import build
from oxyformer.exposure.build import Geography, RasterSpec, build_exposure, run_stage
from oxyformer.exposure.physics import PHYSICS, deficit_mmhg, inspired_oxygen_mmhg, pressure_mmhg
from oxyformer.exposure.population_allocation import AllocationSpec

CRS = 'EPSG:6933'
CONFIG = Path(__file__).resolve().parents[1] / 'configs/exposure.yaml'
CENTROID = AllocationSpec(scenario='centroid', area_crs=CRS)
DISTRIBUTED = AllocationSpec(scenario='distributed', area_crs=CRS, grid_size_m=1.0)


def source(sid, payload='0' * 64, mapping=None):
    return SourceManifest(source_id=sid, version='synthetic-2010', uri='fixture://' + sid,
                          payload_hash=payload, license_hash='1' * 64, schema_hash='2' * 64,
                          field_mapping=mapping or (('raw_id', 'block_id'), ('raw_tract', 'tract_id'),
                                                     ('raw_population', 'population')),
                          mapping_status='reviewed', mapping_review_id='synthetic-fixture-only')


def raster(path, values=(0, 4000), *, crs=CRS):
    with rasterio.open(path, 'w', driver='GTiff', height=1, width=len(values), count=1,
                       dtype='float32', crs=crs, transform=from_origin(0, 1, 1, 1), nodata=-9999) as ds:
        ds.write(np.asarray([values], dtype=np.float32), 1)
        ds.set_band_unit(1, 'm')
    return RasterSpec(tile_id='tile', source_id='dem', path=str(path), crs=CRS,
                      vertical_unit='m', vertical_datum='NAVD88', nodata=-9999)


def blocks(*, split=False, population=100, state='01'):
    ids = [state + '001000100' + ('0001' if i == 0 else '0002') for i in range(2 if split else 1)]
    return gpd.GeoDataFrame({'block_id': ids, 'tract_id': [i[:11] for i in ids],
                            'population': [population // 2] * 2 if split else [population],
                            'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1)] if split else [box(0, 0, 2, 1)]},
                           crs=CRS)


def core(tmp_path, values=(0, 4000), *, split=False, population=100, spec=DISTRIBUTED):
    rs = raster(tmp_path / 'dem.tif', values)
    geo = Geography(blocks=blocks(split=split, population=population), census_source_ids=('census',), rasters=(rs,))
    return build_exposure((source('census'), source('dem')), geo, spec)


def test_sea_level_reference():
    assert pressure_mmhg(0.0) == 760.0
    assert inspired_oxygen_mmhg(0.0) == pytest.approx(149.2309)
    assert deficit_mmhg(0.0) == 0.0


def test_monotonicity_and_domain():
    z = np.linspace(-500, 11000, 301)
    assert np.all(np.diff(pressure_mmhg(z)) < 0)
    assert np.all(np.diff(deficit_mmhg(z)) > 0)
    assert deficit_mmhg(-100.0) < 0
    for bad in (11001.0, -501.0, np.nan, np.inf):
        with pytest.raises(ContractError):
            pressure_mmhg(bad)


def test_transform_before_average(tmp_path):
    exposure, qc = core(tmp_path)
    # Two different heights: Jensen's inequality separates the two computations.
    p_high = 760 * (1 - 0.0065 * 4000 / 288.15) ** (9.80665 * 0.0289644 / (8.31432 * 0.0065))
    expected = (760 + p_high) / 2
    row = exposure.iloc[0]
    assert row.pressure_mmhg == pytest.approx(expected, abs=1e-10)
    assert row.deficit_mmhg == pytest.approx(0.2093 * (760 - expected), abs=1e-10)
    assert abs(row.pressure_mmhg - pressure_mmhg(2000)) > 5
    assert row.mean_elevation_m == 2000
    assert row.inhabited_p10_m == 0
    assert row.inhabited_p90_m == 4000
    assert 'placement-dependent' in row.quantile_interpretation
    assert qc['population'] == qc['covered_population'] == 100
    assert qc['missing_population'] == 0


def test_sea_level_raster(tmp_path):
    exposure, _ = core(tmp_path, values=(0, 0))
    assert exposure.iloc[0].deficit_mmhg == 0


def test_centroid_is_explicit_scenario(tmp_path):
    distributed, _ = core(tmp_path)
    centroid, qc = core(tmp_path, spec=CENTROID)
    assert centroid.iloc[0].pressure_mmhg == pytest.approx(pressure_mmhg(4000))
    assert centroid.iloc[0].pressure_mmhg != distributed.iloc[0].pressure_mmhg
    assert qc['allocation']['scenario'] == 'centroid'


def test_split_block_invariance(tmp_path):
    unsplit, _ = core(tmp_path)
    split, qc = core(tmp_path, split=True)
    pd.testing.assert_frame_equal(unsplit, split)
    assert len(qc['blocks']) == 2
    assert sum(b['population'] for b in qc['blocks']) == 100


@pytest.mark.parametrize('values,reason', [((0, -9999), 'nodata'), ((0,), 'outside_raster_coverage'),
                                          ((0, 20000), 'outside_physical_domain')])
def test_missing_population_cannot_disappear(tmp_path, values, reason):
    exposure, qc = core(tmp_path, values=values)
    row = exposure.iloc[0]
    assert row.population == 100 and row.covered_population == 50 and row.missing_population == 50
    assert row.status == 'incomplete'
    assert pd.isna(row.pressure_mmhg) and pd.isna(row.deficit_mmhg) and pd.isna(row.inhabited_p90_m)
    assert not qc['coverage_pass']
    assert qc['blocks'][0]['omissions'] == {reason: 50}


def test_all_coverage_missing_is_accounted(tmp_path):
    geo = Geography(blocks=blocks(), census_source_ids=('census',), rasters=())
    exposure, qc = build_exposure((source('census'),), geo, DISTRIBUTED)
    assert exposure.iloc[0].missing_population == qc['population'] == 100
    assert qc['blocks'][0]['omissions'] == {'outside_raster_coverage': 100}


def test_zero_population_kept_without_fabricating_exposure(tmp_path):
    exposure, qc = core(tmp_path, values=(-9999,), population=0)
    assert exposure.iloc[0].status == 'zero_population'
    assert pd.isna(exposure.iloc[0].deficit_mmhg)
    assert qc['coverage_pass'] and len(qc['blocks']) == 1


@pytest.mark.parametrize('change,match', [({'crs': 'EPSG:4326'}, 'CRS mismatch'),
                                         ({'vertical_unit': 'ft'}, 'vertical units'),
                                         ({'vertical_datum': 'ellipsoid'}, 'vertical units'),
                                         ({'nodata': None}, 'nodata mismatch')])
def test_dem_metadata_refused(tmp_path, change, match):
    rs = replace(raster(tmp_path / 'dem.tif'), **change)
    geo = Geography(blocks=blocks(), census_source_ids=('census',), rasters=(rs,))
    with pytest.raises(ContractError, match=match):
        build_exposure((source('census'), source('dem')), geo, DISTRIBUTED)


@pytest.mark.parametrize('kind', ['duplicate', 'wrong_tract', 'fractional_population', 'outcome', 'no_crs'])
def test_invalid_geography_refused(tmp_path, kind):
    table = blocks(split=True)
    if kind == 'duplicate':
        table.loc[1, 'block_id'] = table.loc[0, 'block_id']
    elif kind == 'wrong_tract':
        table.loc[0, 'tract_id'] = '01001000200'
    elif kind == 'fractional_population':
        table['population'] = [1.5, 98.5]
    elif kind == 'outcome':
        table['outcome'] = [0, 1]
    else:
        table = table.set_crs(None, allow_override=True)
    with pytest.raises(ContractError):
        build_exposure((source('census'),), Geography(table, ('census',), ()), DISTRIBUTED)


def test_gradients_and_learned_weights_are_not_inputs():
    import torch
    with pytest.raises(ContractError, match='model tensors'):
        pressure_mmhg(torch.tensor([100.0], requires_grad=True))
    with pytest.raises(TypeError):
        AllocationSpec(scenario='distributed', area_crs=CRS, grid_size_m=1, learned_weights=[1])


def test_physical_configuration_matches_owner_approval():
    approved = yaml.safe_load((CONFIG.parent / 'approvals.yaml').read_text())['owner_decisions']['exposure']
    config = yaml.safe_load(CONFIG.read_text())
    assert config['physical'] == asdict(PHYSICS)
    for key, value in approved.items():
        assert config['physical'][key] == value


def fixture_task(tmp_path, *, archive=False, outcome=0, values=(0, 4000)):
    rs = raster(tmp_path / 'dem.tif', tuple(values) * 2)
    inputs = {'dem': Path(rs.path)}
    for state in ('01', '04'):
        table = blocks(state=state).rename(columns={'block_id': 'raw_id', 'tract_id': 'raw_tract',
                                                    'population': 'raw_population'})
        if state == '04':
            table.geometry = table.geometry.translate(xoff=2)
        table['outcome_that_must_not_enter_service'] = outcome
        path = tmp_path / f'blocks-{state}.parquet'
        table.to_parquet(path, index=False)
        inputs['census-' + state] = path
    bindings = {sid: {'dependency_path': str(path), 'sha256': file_hash(path)} for sid, path in inputs.items()}
    if archive:
        packed = tmp_path / 'payload.tar'
        with tarfile.open(packed, 'w') as tar:
            for sid, path in inputs.items():
                tar.add(path, arcname=path.name)
                bindings[sid]['dependency_path'] = str(packed)
                bindings[sid]['member'] = path.name
            info = tarfile.TarInfo('unused/do-not-extract.bin')
            info.size = 1024
            tar.addfile(info, io.BytesIO(b'x' * 1024))
        dependencies = [packed]
    else:
        dependencies = list(inputs.values())
    manifests = [source(sid, file_hash(bindings[sid]['dependency_path']),
                        mapping=(('band1', 'elevation_m'),) if sid == 'dem' else None).to_dict()
                 for sid in inputs]
    task = {'schema_version': 1, 'review_status': 'reviewed', 'review_id': 'synthetic-only',
            'atlas_id': 'synthetic-two-shards', 'geography_vintage': 2010,
            'allocation': asdict(DISTRIBUTED), 'sources': manifests,
            'shards': [{'id': 'one', 'state_fips': ['01']}, {'id': 'two', 'state_fips': ['04']}],
            'state_controls': {state: {'expected_block_count': 1, 'expected_population': 100}
                               for state in ('01', '04')},
            'blocks': [{'source_id': 'census-' + s, 'state_fips': s, 'resource': bindings['census-' + s],
                        'format': 'geoparquet', 'crs': CRS, 'expected_block_count': 1, 'expected_population': 100}
                       for s in ('01', '04')],
            'rasters': [{'source_id': 'dem', 'tile_id': 'tile', 'shard_ids': ['one', 'two'],
                         'resource': bindings['dem'], 'crs': CRS, 'vertical_unit': 'm',
                         'vertical_datum': 'NAVD88', 'nodata': -9999}]}
    return task, dependencies


def request(tmp_path, task, dependencies, *, stage='exposure-atlas', name='run'):
    task_path = tmp_path / (name + '.json')
    task_path.write_text(canonical_json(task))
    return StageRequest(stage=stage, config_path=str(CONFIG), config_hash=file_hash(CONFIG),
                        task_path=str(task_path), task_hash=file_hash(task_path),
                        dependency_paths=tuple(str(p) for p in dependencies),
                        dependency_hashes=tuple(file_hash(p) for p in dependencies),
                        output_dir=str(tmp_path / name), code_identity='a' * 40)


def make_shards(tmp_path, *, archive=False, prepared=None):
    task, dependencies = fixture_task(tmp_path, archive=archive) if prepared is None else prepared
    refs, outputs = {}, []
    for sid in sorted(s['id'] for s in task['shards']):
        req = request(tmp_path, dict(task, shard_id=sid), dependencies, name=sid)
        result = run_stage(req)
        assert result.status == 'pass', result.message
        result.verify(req)
        root = Path(req.output_dir)
        refs[sid] = {key: str(root / file) for key, file in
                     [('manifest', 'artifact_manifest.json'), ('exposure', 'exposure.parquet'), ('quality', 'quality.json')]}
        outputs.extend(Path(p) for p in refs[sid].values())
    return dict(task, shard_artifacts=refs), outputs


def test_stage_artifacts_and_reproducible_collection(tmp_path):
    task, dependencies = make_shards(tmp_path)
    req = request(tmp_path, task, dependencies, stage='atlas-collect', name='collect')
    result = run_stage(req)
    assert result.status == 'pass', result.message
    result.verify(req)
    assert {a.path for a in result.artifacts} == {'atlas.parquet', 'quality.json', 'artifact_manifest.json'}
    qc = json.loads((Path(req.output_dir) / 'quality.json').read_text())
    assert qc['population'] == qc['covered_population'] == 200
    assert qc['missing_population'] == 0 and len(qc['blocks']) == 2
    reversed_task = dict(task, shard_artifacts=dict(reversed(list(task['shard_artifacts'].items()))))
    rerun = request(tmp_path, reversed_task, dependencies[::-1], stage='atlas-collect', name='repeat')
    assert run_stage(rerun).status == 'pass'
    for filename in ('atlas.parquet', 'quality.json'):
        assert file_hash(Path(req.output_dir) / filename) == file_hash(Path(rerun.output_dir) / filename)


def test_archive_streams_raster_and_extracts_only_selected_block(tmp_path, monkeypatch):
    task, dependencies = fixture_task(tmp_path, archive=True)
    seen = []
    original = build._resource
    def tracked(*args, **kwargs):
        value = original(*args, **kwargs)
        seen.append((value, kwargs.get('raster', False)))
        return value
    monkeypatch.setattr(build, '_resource', tracked)
    req = request(tmp_path, dict(task, shard_id='one'), dependencies)
    result = run_stage(req)
    assert result.status == 'pass', result.message
    assert len(seen) == 2
    assert seen[1][1] and seen[1][0].startswith('/vsitar/')
    assert set(p.name for p in Path(req.output_dir).iterdir()) == {
        'exposure.parquet', 'quality.json', 'artifact_manifest.json'}


def test_outcome_independence_at_adapter_boundary(tmp_path):
    paths = []
    for number in (0, 100000):
        folder = tmp_path / str(number)
        folder.mkdir()
        task, deps = fixture_task(folder, outcome=number)
        req = request(folder, dict(task, shard_id='one'), deps)
        assert run_stage(req).status == 'pass'
        paths.append(Path(req.output_dir) / 'exposure.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(paths[0]), pd.read_parquet(paths[1]))


def test_missing_shard_collection(tmp_path):
    task, dependencies = make_shards(tmp_path)
    del task['shard_artifacts']['two']
    result = run_stage(request(tmp_path, task, dependencies, stage='atlas-collect', name='collect'))
    assert result.status == 'fail' and 'missing or unexpected shard' in result.message


@pytest.mark.parametrize('field,value,match', [('source_identities', {'dem': 'a'*64}, 'source identities'),
                                              ('physical', {}, 'physical specifications'),
                                              ('shard_id', 'two', 'identity/specification')])
def test_collection_rejects_changed_identity(tmp_path, field, value, match):
    task, dependencies = make_shards(tmp_path)
    path = Path(task['shard_artifacts']['one']['manifest'])
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(canonical_json(manifest))
    result = run_stage(request(tmp_path, task, dependencies, stage='atlas-collect', name='collect'))
    assert result.status == 'fail' and match in result.message


def test_collection_rejects_unexpected_tract_omission(tmp_path):
    task, dependencies = make_shards(tmp_path)
    ref = task['shard_artifacts']['one']
    path = Path(ref['exposure'])
    pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    manifest = json.loads(Path(ref['manifest']).read_text())
    manifest['artifacts']['exposure.parquet'] = file_hash(path)
    Path(ref['manifest']).write_text(canonical_json(manifest))
    result = run_stage(request(tmp_path, task, dependencies, stage='atlas-collect', name='collect'))
    assert result.status == 'fail' and 'unexpected omission' in result.message


def test_overlap_in_reviewed_selection_rejected(tmp_path):
    task, dependencies = fixture_task(tmp_path)
    task['shards'][1]['state_fips'] = ['01']
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), dependencies))
    assert result.status == 'fail' and 'overlapping' in result.message


def test_stage_coverage_failure_has_complete_diagnostics(tmp_path):
    task, dependencies = fixture_task(tmp_path, values=(0, -9999))
    req = request(tmp_path, dict(task, shard_id='one'), dependencies)
    result = run_stage(req)
    assert result.status == 'fail' and len(result.artifacts) == 3
    result.verify(req)
    qc = json.loads((Path(req.output_dir) / 'quality.json').read_text())
    assert qc['missing_population'] == 50 and qc['population'] == 100


def test_dependencies_are_checked_before_reads(tmp_path):
    task, dependencies = fixture_task(tmp_path)
    req = request(tmp_path, dict(task, shard_id='one'), dependencies)
    with dependencies[0].open('ab') as out:
        out.write(b'changed')
    result = run_stage(req)
    assert result.status == 'fail' and 'input hash mismatch' in result.message
    assert not Path(req.output_dir).exists()


def test_undeclared_dependency_rejected(tmp_path):
    task, dependencies = fixture_task(tmp_path)
    req = request(tmp_path, dict(task, shard_id='one'), dependencies[1:])
    result = run_stage(req)
    assert result.status == 'fail' and 'explicit request dependency' in result.message


def test_missing_review_is_blocked(tmp_path):
    task, dependencies = fixture_task(tmp_path)
    task['review_status'] = 'unreviewed'
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), dependencies))
    assert result.status == 'blocked'


def test_expected_population_and_count_are_enforced(tmp_path):
    task, dependencies = fixture_task(tmp_path)
    task['blocks'][0]['expected_population'] += 1
    task['state_controls']['01']['expected_population'] += 1
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), dependencies))
    assert result.status == 'fail' and 'changed population total' in result.message


def test_existing_outputs_are_never_overwritten(tmp_path):
    task, dependencies = fixture_task(tmp_path)
    req = request(tmp_path, dict(task, shard_id='one'), dependencies)
    assert run_stage(req).status == 'pass'
    before = file_hash(Path(req.output_dir) / 'exposure.parquet')
    assert run_stage(req).status == 'fail'
    assert file_hash(Path(req.output_dir) / 'exposure.parquet') == before


@pytest.mark.parametrize('scalar', [np.float64(1000), np.float32(1000), np.int64(1000)])
def test_numpy_cpu_scalars_are_valid_physical_inputs(scalar):
    assert pressure_mmhg(scalar) == pytest.approx(pressure_mmhg(1000.0))


def test_positive_area_block_overlap_rejected(tmp_path):
    table = blocks(split=True)
    table.loc[1, 'geometry'] = box(0.5, 0, 1.5, 1)
    with pytest.raises(ContractError, match='overlapping block polygon interiors'):
        build_exposure((source('census'),), Geography(table, ('census',), ()), DISTRIBUTED)


def test_centroid_outside_polygon_uses_explicit_geometry_only_policy(tmp_path):
    table = blocks()
    table.loc[0, 'geometry'] = box(0, 0, 3, 3).difference(box(1, 1, 2, 2))
    path = tmp_path / 'ring.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=3, width=3, count=1, dtype='float32',
                       crs=CRS, transform=from_origin(0, 3, 1, 1), nodata=-9999) as ds:
        values = np.zeros((3, 3), dtype='float32')
        values[1, 1] = -9999
        ds.write(values, 1)
    rs = RasterSpec('ring', 'dem', str(path), CRS, 'm', 'NAVD88', -9999)
    frame, qc = build_exposure((source('census'), source('dem')),
                               Geography(table, ('census',), (rs,)), CENTROID)
    assert qc['coverage_pass'] and qc['covered_population'] == 100
    assert frame.iloc[0].deficit_mmhg == 0
    assert qc['allocation']['centroid_outside_policy'] == 'interior_representative_point'
    # Removing coverage at the fixed interior placement must still fail: placement
    # is determined from geometry, never moved around to search for covered DEM.
    with rasterio.open(path, 'r+') as ds:
        ds.write(np.full((3, 3), -9999, dtype='float32'), 1)
    _, missing = build_exposure((source('census'), source('dem')),
                                Geography(table, ('census',), (rs,)), CENTROID)
    assert missing['missing_population'] == 100


def test_zipped_shapefile_with_reviewed_geoid_prefix(tmp_path):
    import zipfile
    task, dependencies = fixture_task(tmp_path)
    table = blocks().drop(columns='tract_id').rename(columns={'block_id': 'GEOID10', 'population': 'POP10'})
    folder = tmp_path / 'shape'
    folder.mkdir()
    table.to_file(folder / 'blocks.shp')
    archive = tmp_path / 'blocks.zip'
    with zipfile.ZipFile(archive, 'w') as packed:
        for path in folder.iterdir():
            packed.write(path, path.name)
    old_path = task['blocks'][0]['resource']['dependency_path']
    task['blocks'][0].update(format='shapefile_zip', tract_id_from_block_prefix=True,
                             resource={'dependency_path': str(archive), 'sha256': file_hash(archive)})
    for manifest in task['sources']:
        if manifest['payload']['source_id'] == 'census-01':
            manifest['payload'].update(payload_hash=file_hash(archive),
                                       field_mapping=[['GEOID10', 'block_id'], ['POP10', 'population']])
    dependencies = [p for p in dependencies if str(p) != old_path] + [archive]
    req = request(tmp_path, dict(task, shard_id='one'), dependencies)
    result = run_stage(req)
    assert result.status == 'pass', result.message
    frame = pd.read_parquet(Path(req.output_dir) / 'exposure.parquet')
    assert frame.tract_id.tolist() == ['01001000100']


def rebind(task, dependencies, sid, path):
    """Update synthetic reviewed identities after deliberately changing a fixture."""
    digest = file_hash(path)
    for manifest in task['sources']:
        if manifest['payload']['source_id'] == sid:
            manifest['payload']['payload_hash'] = digest
    for binding in task['blocks'] + task['rasters']:
        if binding['source_id'] == sid:
            binding['resource'].update(dependency_path=str(path), sha256=digest)


def edit_shard(task, sid, *, quality=None, frame=None, manifest_change=None):
    """Rebind artifact hashes so negative cases reach semantic collection checks."""
    refs = task['shard_artifacts'][sid]
    manifest = json.loads(Path(refs['manifest']).read_text())
    if quality is not None:
        Path(refs['quality']).write_text(canonical_json(quality))
        manifest['artifacts']['quality.json'] = file_hash(refs['quality'])
    if frame is not None:
        frame.to_parquet(refs['exposure'], index=False)
        manifest['artifacts']['exposure.parquet'] = file_hash(refs['exposure'])
    if manifest_change:
        manifest_change(manifest)
    Path(refs['manifest']).write_text(canonical_json(manifest))


def collect(tmp_path, task, dependencies, *, name='collect', code_identity=None):
    req = request(tmp_path, task, dependencies, stage='atlas-collect', name=name)
    if code_identity is not None:
        req = replace(req, code_identity=code_identity)
    return run_stage(req), Path(req.output_dir)


@pytest.mark.parametrize('left', [0.0, 1.0, 2.0 - 2**-30])
def test_cross_shard_positive_overlap_rejected(tmp_path, left):
    task, deps = fixture_task(tmp_path)
    path = Path(task['blocks'][1]['resource']['dependency_path'])
    table = gpd.read_parquet(path)
    table.geometry = [box(left, 0, left + 2, 1)]
    table.to_parquet(path, index=False)
    rebind(task, deps, 'census-04', path)
    task, deps = make_shards(tmp_path, prepared=(task, deps))
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'fail' and 'overlapping shard polygon interiors' in result.message


def test_zero_population_shard_still_participates_in_overlap_check(tmp_path):
    task, deps = fixture_task(tmp_path)
    path = Path(task['blocks'][1]['resource']['dependency_path'])
    table = gpd.read_parquet(path)
    table.geometry = [box(0, 0, 2, 1)]
    table.raw_population = 0
    table.to_parquet(path, index=False)
    rebind(task, deps, 'census-04', path)
    task['blocks'][1]['expected_population'] = 0
    task['state_controls']['04']['expected_population'] = 0
    task, deps = make_shards(tmp_path, prepared=(task, deps))
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'fail' and 'overlapping shard' in result.message


def test_envelope_overlap_with_disjoint_interiors_passes(tmp_path):
    task, deps = fixture_task(tmp_path)
    geometries = [box(0, 0, 3, 3).difference(box(1, 1, 2, 2)), box(1, 1, 2, 2)]
    for binding, geometry in zip(task['blocks'], geometries):
        path = Path(binding['resource']['dependency_path'])
        table = gpd.read_parquet(path)
        table.geometry = [geometry]
        table.to_parquet(path, index=False)
        rebind(task, deps, binding['source_id'], path)
    path = tmp_path / 'dem.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=3, width=3, count=1, dtype='float32',
                       crs=CRS, transform=from_origin(0, 3, 1, 1), nodata=-9999) as ds:
        ds.write(np.zeros((3, 3), dtype='float32'), 1)
    rebind(task, deps, 'dem', path)
    task, deps = make_shards(tmp_path, prepared=(task, deps))
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'pass', result.message


def test_differently_noded_shared_edges_pass(tmp_path):
    from shapely.geometry import Polygon
    table = blocks(split=True)
    table.loc[1, 'geometry'] = Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0.5), (1, 0)])
    rs = raster(tmp_path / 'dem.tif')
    _, qc = build_exposure((source('census'), source('dem')), Geography(table, ('census',), (rs,)), DISTRIBUTED)
    footprint = build._read_footprint(qc, CRS)
    assert footprint.equals(box(0, 0, 2, 1))


def test_footprint_includes_zero_population_and_uncovered_blocks(tmp_path):
    table = blocks(split=True)
    table.population = [0, 100]
    rs = raster(tmp_path / 'dem.tif', values=(0,))
    _, qc = build_exposure((source('census'), source('dem')), Geography(table, ('census',), (rs,)), DISTRIBUTED)
    assert qc['missing_population'] == 100
    assert build._read_footprint(qc, CRS).equals(box(0, 0, 2, 1))


@pytest.mark.parametrize('problem', ['missing', 'crs', 'digest', 'invalid_wkb'])
def test_bad_footprint_refused(tmp_path, problem):
    task, deps = make_shards(tmp_path)
    qc = json.loads(Path(task['shard_artifacts']['one']['quality']).read_text())
    if problem == 'missing':
        del qc['footprint']
    elif problem == 'crs':
        qc['footprint']['crs'] = 'EPSG:4326'
    elif problem == 'digest':
        qc['footprint']['sha256'] = '0' * 64
    else:
        qc['footprint']['wkb_hex'] = '00'
        qc['footprint']['sha256'] = sha256(b'\x00').hexdigest()
    edit_shard(task, 'one', quality=qc)
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'fail', result.message


def test_mixed_producer_commits_refused(tmp_path):
    task, deps = make_shards(tmp_path)
    edit_shard(task, 'two', manifest_change=lambda m: m.update(code_identity='b' * 40))
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'fail' and 'mixed producer code identities' in result.message


def test_missing_producer_commit_refused(tmp_path):
    task, deps = make_shards(tmp_path)
    edit_shard(task, 'two', manifest_change=lambda m: m.pop('code_identity'))
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'fail' and 'producer code identity' in result.message


def test_distinct_collector_commit_allowed_and_recorded(tmp_path):
    task, deps = make_shards(tmp_path)
    result, root = collect(tmp_path, task, deps, code_identity='b' * 40)
    assert result.status == 'pass', result.message
    qc = json.loads((root / 'quality.json').read_text())
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    assert qc['producer_code_identity'] == manifest['producer_code_identity'] == 'a' * 40
    assert manifest['code_identity'] == 'b' * 40


def multi_binding_task(tmp_path, *, shared_archive=False, overlap=False, wrong_state=False):
    task, deps = fixture_task(tmp_path)
    task['shards'] = [{'id': 'one', 'state_fips': ['01']}]
    task['state_controls'] = {'01': {'expected_block_count': 2, 'expected_population': 100}}
    task['rasters'][0]['shard_ids'] = ['one']
    task['blocks'] = []
    table = blocks(split=True).rename(columns={'block_id': 'raw_id', 'tract_id': 'raw_tract',
                                               'population': 'raw_population'})
    if overlap:
        table.loc[1, 'geometry'] = box(1 - 2**-30, 0, 2 - 2**-30, 1)
    if wrong_state:
        table.loc[1, 'raw_id'] = '040010001000002'
        table.loc[1, 'raw_tract'] = '04001000100'
    county_files = []
    for i in (0, 1):
        path = tmp_path / f'county-{i}.parquet'
        table.iloc[[i]].to_parquet(path, index=False)
        county_files.append(path)
    archive = tmp_path / 'counties.tar'
    if shared_archive:
        with tarfile.open(archive, 'w') as packed:
            for path in county_files:
                packed.add(path, arcname=path.name)
        deps.append(archive)
        task['sources'].append(source('counties', file_hash(archive)).to_dict())
    for i, path in enumerate(county_files):
        if shared_archive:
            sid = 'counties'
            resource = {'dependency_path': str(archive), 'member': path.name, 'sha256': file_hash(path)}
        else:
            sid = f'county-{i}'
            resource = {'dependency_path': str(path), 'sha256': file_hash(path)}
            deps.append(path)
            task['sources'].append(source(sid, file_hash(path)).to_dict())
        task['blocks'].append({'source_id': sid, 'state_fips': '01', 'resource': resource,
                               'format': 'geoparquet', 'crs': CRS,
                               'expected_block_count': 1, 'expected_population': 50})
    return task, deps


@pytest.mark.parametrize('shared_archive', [False, True])
def test_multiple_bindings_same_state_pass_and_repeat(tmp_path, shared_archive):
    task, deps = multi_binding_task(tmp_path, shared_archive=shared_archive)
    req = request(tmp_path, dict(task, shard_id='one'), deps, name='first')
    result = run_stage(req)
    assert result.status == 'pass', result.message
    frame = pd.read_parquet(Path(req.output_dir) / 'exposure.parquet')
    rs = RasterSpec('tile', 'dem', str(tmp_path / 'dem.tif'), CRS, 'm', 'NAVD88', -9999)
    expected, _ = build_exposure((source('census'), source('dem')),
                                 Geography(blocks(), ('census',), (rs,)), DISTRIBUTED)
    pd.testing.assert_frame_equal(frame, expected)
    qc1 = json.loads((Path(req.output_dir) / 'quality.json').read_text())
    reordered = dict(task, blocks=task['blocks'][::-1], shard_id='one')
    req2 = request(tmp_path, reordered, deps[::-1], name='second')
    assert run_stage(req2).status == 'pass'
    qc2 = json.loads((Path(req2.output_dir) / 'quality.json').read_text())
    assert qc1['footprint'] == qc2['footprint']
    collection_task, outputs = make_shards(tmp_path, prepared=(task, deps))
    result, _ = collect(tmp_path, collection_task, outputs)
    assert result.status == 'pass', result.message


def test_dropped_binding_refused_before_adaptation(tmp_path, monkeypatch):
    task, deps = multi_binding_task(tmp_path)
    task['blocks'].pop()
    def should_not_read(*args, **kwargs):
        raise AssertionError('raw adaptation reached despite incomplete declared inventory')
    monkeypatch.setattr(build, '_resource', should_not_read)
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'independent state controls' in result.message


def test_duplicate_resource_refused(tmp_path):
    task, deps = multi_binding_task(tmp_path)
    task['blocks'][1] = deepcopy(task['blocks'][0])
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'duplicate selected block resource' in result.message


def test_offsetting_per_binding_errors_refused(tmp_path):
    task, deps = multi_binding_task(tmp_path)
    task['blocks'][0]['expected_population'] = 40
    task['blocks'][1]['expected_population'] = 60
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'changed population total' in result.message


def test_overlap_across_bindings_refused(tmp_path):
    task, deps = multi_binding_task(tmp_path, overlap=True)
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'overlapping block polygon interiors' in result.message


def test_wrong_state_in_binding_refused(tmp_path):
    task, deps = multi_binding_task(tmp_path, wrong_state=True)
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'state membership mismatch' in result.message


def test_collection_state_controls_not_only_shard_total(tmp_path):
    task, deps = fixture_task(tmp_path)
    task['shards'] = [{'id': 'one', 'state_fips': ['01', '04']}]
    task['rasters'][0]['shard_ids'] = ['one']
    task, deps = make_shards(tmp_path, prepared=(task, deps))
    ref = task['shard_artifacts']['one']
    frame = pd.read_parquet(ref['exposure'])
    qc = json.loads(Path(ref['quality']).read_text())
    for i, population in enumerate((90, 110)):
        qc['blocks'][i]['population'] = population
        qc['blocks'][i]['covered_population'] = float(population)
        frame.loc[i, 'population'] = population
        frame.loc[i, 'covered_population'] = float(population)
    edit_shard(task, 'one', quality=qc, frame=frame)
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'fail' and 'per-state controls mismatch' in result.message


@pytest.mark.parametrize('location', ['allocation', 'blocks', 'rasters'])
def test_malformed_crs_returns_failed_stage(tmp_path, location):
    task, deps = fixture_task(tmp_path)
    if location == 'allocation':
        task['allocation']['area_crs'] = 'EPSG:bogus'
    else:
        task[location][0]['crs'] = 'EPSG:bogus'
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'CRSError' in result.message


def test_unexpected_implementation_error_is_not_swallowed(tmp_path, monkeypatch):
    task, deps = fixture_task(tmp_path)
    def bug(*args, **kwargs):
        raise RuntimeError('unexpected bug')
    monkeypatch.setattr(build, '_task', bug)
    with pytest.raises(RuntimeError, match='unexpected bug'):
        run_stage(request(tmp_path, dict(task, shard_id='one'), deps))


def test_differently_noded_shared_shard_edges_pass(tmp_path):
    from shapely.geometry import Polygon
    task, deps = fixture_task(tmp_path)
    binding = task['blocks'][1]
    path = Path(binding['resource']['dependency_path'])
    table = gpd.read_parquet(path)
    table.geometry = [Polygon([(2, 0), (4, 0), (4, 1), (2, 1), (2, 0.5), (2, 0)])]
    table.to_parquet(path, index=False)
    rebind(task, deps, binding['source_id'], path)
    task, deps = make_shards(tmp_path, prepared=(task, deps))
    result, _ = collect(tmp_path, task, deps)
    assert result.status == 'pass', result.message


def test_duplicate_ids_across_bindings_refused(tmp_path):
    task, deps = multi_binding_task(tmp_path)
    binding = task['blocks'][1]
    path = Path(binding['resource']['dependency_path'])
    table = gpd.read_parquet(path)
    table.raw_id = '010010001000001'
    table.to_parquet(path, index=False)
    rebind(task, deps, binding['source_id'], path)
    result = run_stage(request(tmp_path, dict(task, shard_id='one'), deps))
    assert result.status == 'fail' and 'duplicate 2010 block IDs' in result.message


def test_overlapping_dem_tiles_have_stable_priority_and_nodata_fallback(tmp_path):
    first = replace(raster(tmp_path / 'first.tif', values=(0, -9999)), tile_id='a')
    second = replace(raster(tmp_path / 'second.tif', values=(1000, 2000)), tile_id='b')
    geo = Geography(blocks(), ('census',), (first, second))
    sources = (source('census'), source('dem'))
    frame, qc = build_exposure(sources, geo, DISTRIBUTED)
    expected_pressure = (float(pressure_mmhg(0)) + float(pressure_mmhg(2000))) / 2
    assert frame.iloc[0].pressure_mmhg == pytest.approx(expected_pressure)
    assert qc['missing_population'] == 0
    reversed_frame, reversed_qc = build_exposure(sources, replace(geo, rasters=(second, first)), DISTRIBUTED)
    pd.testing.assert_frame_equal(frame, reversed_frame)
    assert qc == reversed_qc


def test_multipolygon_placement_and_footprint(tmp_path):
    from shapely.geometry import MultiPolygon
    table = blocks()
    table.geometry = [MultiPolygon([box(0, 0, 1, 1), box(3, 0, 4, 1)])]
    rs = raster(tmp_path / 'dem.tif', values=(0, -9999, -9999, 4000))
    frame, qc = build_exposure((source('census'), source('dem')), Geography(table, ('census',), (rs,)), DISTRIBUTED)
    assert qc['coverage_pass'] and qc['population'] == 100
    assert frame.iloc[0].mean_elevation_m == 2000
    assert build._read_footprint(qc, CRS).equals(table.geometry.iloc[0])
