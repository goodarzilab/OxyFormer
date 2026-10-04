"""Offline synthetic tests: no downloaded population, raster, or health data."""
from dataclasses import asdict, replace
from fractions import Fraction
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
import zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Polygon
import yaml
from oxyformer.contracts import StageRequest
from oxyformer.data.source_manifest import load_source
from oxyformer.exposure.build import ExposureSources, build_exposure, run_stage
from oxyformer.exposure.census_blocks import read_census_blocks, read_sf1_population
from oxyformer.exposure.physics import PHYSICS, pressure_mmhg, inspired_oxygen_mmhg, oxygen_deficit_mmhg
from oxyformer.exposure.population_allocation import AllocationSpec, DemTile, RasterSampler, PRIMARY, FALLBACK
from oxyformer.exposure.quality import weighted_quantiles, placement_quantiles
from oxyformer.provenance import ContractError, canonical_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
SPEC = AllocationSpec(grid_size_m=100)
H = 'a' * 64


def write_raster(path, values=(0, 3000), crs='EPSG:5070', nodata=-9999):
    with rasterio.open(path, 'w', driver='GTiff', height=1, width=len(values), count=1,
                       dtype='float64', crs=crs, transform=from_origin(0, 100, 100, 100), nodata=nodata) as ds:
        ds.write(np.array([values], dtype=float), 1)
        ds.set_band_unit(1, 'm')
    return DemTile(resource_id='synthetic', path=str(path), sha256=file_hash(path), crs=crs,
                   nodata=nodata, vertical_unit='m', vertical_datum='NAVD88')


def blocks(pop=(40, 60), state='01'):
    ids = [state + '0010001001001', state + '0010001001002']
    return gpd.GeoDataFrame({'block_id': ids, 'tract_id': [x[:11] for x in ids], 'population': pop},
                           geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs='EPSG:5070')


def sources(tile):
    return ExposureSources(identities=(('synthetic_census', H),), dem_tiles=(tile,))


def test_sea_level_and_owner_constants():
    approved = yaml.safe_load((ROOT / 'configs/approvals.yaml').read_text())['owner_decisions']['exposure']
    for name, value in asdict(PHYSICS).items():
        if name in approved:
            assert value == approved[name]
    assert pressure_mmhg(0) == 760
    assert inspired_oxygen_mmhg(0) == pytest.approx(0.2093 * (760 - 47))
    assert oxygen_deficit_mmhg(0) == 0
    assert pressure_mmhg(0) * PHYSICS.pa_per_mmhg == 101325
    z = np.linspace(-500, 11000, 100)
    assert np.all(np.diff(pressure_mmhg(z)) < 0)
    assert np.all(np.diff(oxygen_deficit_mmhg(z)) > 0)
    assert oxygen_deficit_mmhg(-100) < 0


@pytest.mark.parametrize('height', [np.nan, np.inf, -501, 11001])
def test_invalid_physical_domain(height):
    with pytest.raises(ContractError, match='validity'):
        pressure_mmhg(height)


def test_transform_before_average(tmp_path):
    tile = write_raster(tmp_path / 'dem.tif')
    result, qc = build_exposure(sources(tile), blocks(), SPEC)
    expected = np.dot([0.4, 0.6], pressure_mmhg([0, 3000]))
    wrong = pressure_mmhg(1800)
    assert abs(expected - wrong) > 1
    np.testing.assert_allclose(result.pressure_mmhg, expected, rtol=1e-12)
    np.testing.assert_allclose(result.oxygen_deficit_mmhg, np.dot([0.4, 0.6], oxygen_deficit_mmhg([0, 3000])))
    assert qc['population'] == 100 and qc['block_count'] == 2
    assert set(result.population) == {100}
    assert set(result.covered_population) == {100}
    assert set(result.elevation_p10_m) == {0} and set(result.elevation_p90_m) == {3000}
    assert all('placement-dependent' in x for x in result.quantile_interpretation)


def test_distributed_transform_before_average_and_split_invariance(tmp_path):
    tile = write_raster(tmp_path / 'dem.tif')
    split = blocks(pop=(50, 50))
    whole = split.iloc[:1].copy()
    whole.loc[0, 'population'] = 100
    whole.loc[0, 'geometry'] = box(0, 0, 200, 100)
    spec = replace(SPEC, scenarios=('distributed',))
    a, _ = build_exposure(sources(tile), whole, spec)
    b, _ = build_exposure(sources(tile), split, spec)
    assert a.pressure_mmhg.iloc[0] == pytest.approx(np.mean(pressure_mmhg([0, 3000])))
    for col in ['population', 'pressure_mmhg', 'oxygen_deficit_mmhg', 'elevation_p10_m', 'elevation_p90_m']:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-12)


@pytest.mark.parametrize('bad,reason', [(-9999, 'nodata'), (12000, 'outside_physical_domain'), (np.nan, 'nodata')])
def test_missing_population_cannot_disappear(tmp_path, bad, reason):
    tile = write_raster(tmp_path / 'dem.tif', [0, bad])
    result, qc = build_exposure(sources(tile), blocks(), SPEC)
    assert result.pressure_mmhg.isna().all() and result.oxygen_deficit_mmhg.isna().all()
    assert set(result.population) == {100} and set(result.missing_population) == {60}
    assert sum(r[reason] for r in qc['blocks'] if r['scenario'] == 'distributed') == 60
    assert result.elevation_p10_m.isna().all()


def test_outside_coverage_and_zero_population(tmp_path):
    tile = write_raster(tmp_path / 'dem.tif', [0])
    result, qc = build_exposure(sources(tile), blocks(), SPEC)
    assert set(result.missing_population) == {60}
    assert sum(r['outside_coverage'] for r in qc['blocks']) == 120  # separate scenarios
    result, qc = build_exposure(sources(tile), blocks(pop=(40, 0)), SPEC)
    assert result.pressure_mmhg.eq(760).all() and qc['block_count'] == 2
    result, qc = build_exposure(sources(tile), blocks(pop=(0, 0)), SPEC)
    assert result.status.eq('zero_population').all() and result.pressure_mmhg.isna().all()
    assert len(qc['blocks']) == 4


@pytest.mark.parametrize('change,message', [({'crs': 'EPSG:4326'}, 'CRS mismatch'),
    ({'vertical_unit': 'ft'}, 'metres'), ({'vertical_datum': 'ellipsoid'}, 'datum'),
    ({'nodata': None}, 'nodata mismatch'), ({'sha256': 'b' * 64}, 'hash mismatch'),
    ({'product': FALLBACK}, 'fallback only')])
def test_raster_metadata_validation(tmp_path, change, message):
    tile = write_raster(tmp_path / 'dem.tif')
    with pytest.raises(ContractError, match=message):
        build_exposure(sources(replace(tile, **change)), blocks(), SPEC)


def test_reprojection_and_input_boundary(tmp_path):
    tile = write_raster(tmp_path / 'dem.tif')
    a, _ = build_exposure(sources(tile), blocks(), SPEC)
    b, _ = build_exposure(sources(tile), blocks().to_crs('EPSG:4326'), replace(SPEC, scenarios=('centroid',)))
    assert a.pressure_mmhg.iloc[0] == pytest.approx(b.pressure_mmhg.iloc[0])
    for forbidden in ['outcome', 'learned_weight', 'gradient']:
        contaminated = blocks().assign(**{forbidden: [123, 999]})
        with pytest.raises(ContractError, match='only block'):
            build_exposure(sources(tile), contaminated, SPEC)
    with pytest.raises(ContractError, match='CRS'):
        build_exposure(sources(tile), blocks().set_crs(None, allow_override=True), SPEC)
    with pytest.raises(ContractError, match='tract membership'):
        build_exposure(sources(tile), blocks().assign(tract_id='00000000000'), SPEC)


def write_census_archives(directory, state='AL', fips='01', mismatch=False, missing=False):
    directory.mkdir(exist_ok=True)
    b = blocks(state=fips)
    geo = gpd.GeoDataFrame({'BLOCKID10': b.block_id, 'STATEFP10': fips, 'COUNTYFP10': '001',
        'TRACTCE10': '000100', 'BLOCKCE': ['1001', '1002'], 'POP10': [40, 61 if mismatch else 60]},
        geometry=b.geometry, crs=b.crs)
    stem = f'tabblock2010_{fips}_pophu'
    geo.to_file(directory / (stem + '.shp'))
    block_zip = directory / 'blocks.zip'
    with zipfile.ZipFile(block_zip, 'w') as z:
        for suffix in ('.shp', '.shx', '.dbf', '.prj'):
            z.write(directory / (stem + suffix), stem + suffix)
        z.writestr('unneeded.txt', 'do not extract')
    records = [('040', fips, 100), ('140', fips + '001000100', 100)]
    records += [('101', row.block_id, row.population) for row in b.itertuples()]
    geographic, segment = [], []
    # Independent literal fixture offsets from SF1 geographic header (not reader constants).
    for n, (level, ident, pop) in enumerate(records, 1):
        line = list(' ' * 500)
        for start, value in [(0, 'SF1ST '), (6, state), (8, level), (11, '00'), (13, '000'),
                (16, '00'), (18, f'{n:07}'), (27, fips), (318, f'{pop:09}')]:
            line[start:start + len(value)] = value
        if level != '040':
            line[29:32], line[54:60] = ident[2:5], ident[5:11]
        if level == '101':
            line[61:65] = ident[11:15]
        geographic.append(''.join(line))
        segment.append(f'SF1ST,{state},000,01,{n:07},{pop}')
    sf_zip = directory / 'sf1.zip'
    with zipfile.ZipFile(sf_zip, 'w') as z:
        z.writestr(state.lower() + 'geo2010.sf1', '\n'.join(geographic) + '\n')
        z.writestr(state.lower() + '000012010.sf1', '\n'.join(segment[:-1] if missing else segment) + '\n')
        z.writestr(state.lower() + '000022010.sf1', 'not read')
    return block_zip, sf_zip


def test_census_reader_ids_population_and_inventory(tmp_path):
    bzip, szip = write_census_archives(tmp_path)
    table = read_census_blocks(bzip, szip, state_abbreviation='AL', state_fips='01')
    assert table.block_id.tolist() == blocks().block_id.tolist()
    assert table.population.tolist() == [40, 60]
    assert table.tract_id.tolist() == blocks().tract_id.tolist()
    assert set(p.name for p in tmp_path.iterdir()).isdisjoint({'unneeded.txt', 'al000022010.sf1'})


@pytest.mark.parametrize('mismatch,missing,message', [(True, False, 'TIGER/SF1'), (False, True, 'missing SF1')])
def test_census_join_failures(tmp_path, mismatch, missing, message):
    bzip, szip = write_census_archives(tmp_path, mismatch=mismatch, missing=missing)
    with pytest.raises(ContractError, match=message):
        read_census_blocks(bzip, szip, state_abbreviation='AL', state_fips='01')


def acquisition(directory, name, resource_files):
    tar_path = directory / (name + '.tar')
    resources = []
    with tarfile.open(tar_path, 'w') as tar:
        for rid, path in resource_files:
            destination = name + '/' + rid + path.suffix
            tar.add(path, arcname=destination)
            resources.append(dict(id=rid, destination=destination, sha256=file_hash(path)))
    source = load_source('census') if name == 'census' else {'synthetic': True}
    receipt = dict(status='complete', manifest_id=name, manifest_sha256=sha256(canonical_json(source).encode()).hexdigest(),
                   payload_sha256=file_hash(tar_path), resources=resources)
    receipt_path = directory / (name + '-receipt.json')
    receipt_path.write_text(canonical_json(receipt))
    return receipt_path, tar_path


def request(directory, stage, task, dependencies, config=None):
    directory.mkdir(exist_ok=True)
    task_path = directory / 'task.json'
    task_path.write_text(canonical_json(task))
    config_path = directory / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config or yaml.safe_load((ROOT / 'configs/exposure.yaml').read_text())))
    return StageRequest(stage=stage, config_path=str(config_path), config_hash=file_hash(config_path),
        task_path=str(task_path), task_hash=file_hash(task_path), dependency_paths=tuple(map(str, dependencies)),
        dependency_hashes=tuple(file_hash(p) for p in dependencies), output_dir=str(directory / 'out'), code_identity='b' * 40)


@pytest.fixture
def shard_fixture(tmp_path, monkeypatch):
    # Explicit synthetic DEM inventory; production uses the merged source registry.
    monkeypatch.setattr('oxyformer.exposure.build.load_source',
                        lambda name: {'synthetic': True} if name == 'dem' else load_source(name))
    census_files = []
    for state, fips in [('AL', '01'), ('AZ', '04')]:
        bzip, szip = write_census_archives(tmp_path / state, state, fips)
        census_files.extend([('blocks_' + fips, bzip), ('sf1_' + state.lower(), szip)])
    census = acquisition(tmp_path, 'census', census_files)
    tile = write_raster(tmp_path / 'dem.tif')
    dem = acquisition(tmp_path, 'dem', [('synthetic', Path(tile.path))])
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(canonical_json(dict(schema_version=1, kind='atlas_shards', approval_reference='synthetic-review',
        groups=[dict(id=state, jurisdictions=[state], dem_resources=['synthetic']) for state in ['AL', 'AZ']])))
    deps = [inventory, *census, *dem]
    paths = []
    for state in ['AL', 'AZ']:
        task = dict(schema_version=1, review_status='reviewed', review_id='synthetic', shard_manifest=0, shard_id=state,
                    census=dict(receipt=1, payload=2), dem=dict(receipt=3, payload=4),
                    raster_metadata={'synthetic': dict(crs=tile.crs, nodata=tile.nodata, vertical_unit='m',
                                                       vertical_datum='NAVD88', product=PRIMARY, fallback_reason=None)})
        req = request(tmp_path / ('shard-' + state), 'exposure-atlas', task, deps)
        result = run_stage(req)
        assert result.status == 'pass', result.message
        result.verify(req)
        out = Path(req.output_dir)
        assert sorted(p.name for p in out.iterdir()) == ['artifact_manifest.json', 'exposure.parquet', 'quality.json']
        paths.extend([out / 'artifact_manifest.json', out / 'exposure.parquet', out / 'quality.json'])
    return inventory, paths


def collect_request(directory, inventory, paths, reverse=False):
    binding = [dict(manifest=1, exposure=2, quality=3), dict(manifest=4, exposure=5, quality=6)]
    if reverse:
        binding.reverse()
    task = dict(schema_version=1, review_status='reviewed', review_id='synthetic', shard_manifest=0, shards=binding)
    return request(directory, 'atlas-collect', task, [inventory, *paths])


def test_reproducible_collection(tmp_path, shard_fixture):
    inventory, paths = shard_fixture
    a = collect_request(tmp_path / 'collection-a', inventory, paths)
    b = collect_request(tmp_path / 'collection-b', inventory, paths, reverse=True)
    for req in [a, b]:
        result = run_stage(req)
        assert result.status == 'pass', result.message
        result.verify(req)
    for name in ['atlas.parquet', 'quality.json']:
        assert file_hash(Path(a.output_dir) / name) == file_hash(Path(b.output_dir) / name)
    atlas = pd.read_parquet(Path(a.output_dir) / 'atlas.parquet')
    assert len(atlas) == 4 and atlas.population.sum() == 400


def test_missing_shard_collection(tmp_path, shard_fixture):
    inventory, paths = shard_fixture
    req = collect_request(tmp_path / 'collection', inventory, paths)
    task = json.loads(Path(req.task_path).read_text())
    task['shards'].pop()
    Path(req.task_path).write_text(canonical_json(task))
    req = replace(req, task_hash=file_hash(req.task_path))
    result = run_stage(req)
    assert result.status == 'fail' and 'missing shards' in result.message
    assert not list(Path(req.output_dir).glob('atlas.parquet'))


@pytest.mark.parametrize('kind', ['physical', 'source', 'overlap', 'hash'])
def test_collection_rejects_inconsistency(tmp_path, shard_fixture, kind):
    inventory, paths = shard_fixture
    if kind in ('physical', 'source'):
        quality = json.loads(paths[5].read_text())
        quality['physical_spec']['sea_level_pressure_mmhg'] += 1 if kind == 'physical' else 0
        if kind == 'source':
            quality['source_identities']['dem_receipt'] = 'e' * 64
        paths[5].write_text(canonical_json(quality))
        manifest = json.loads(paths[3].read_text())
        manifest['files']['quality.json'] = file_hash(paths[5])
        manifest['source_identities'] = quality['source_identities']
        paths[3].write_text(canonical_json(manifest))
    if kind == 'overlap':
        for source, destination in zip(paths[:3], paths[3:]):
            destination.write_bytes(source.read_bytes())
    req = collect_request(tmp_path / 'collection', inventory, paths)
    if kind == 'hash':
        paths[2].write_text('{}')
    result = run_stage(req)
    assert result.status == 'fail', result.message
    assert not (Path(req.output_dir) / 'atlas.parquet').exists()


def test_unused_census_name_bytes_do_not_change_numeric_reader(tmp_path):
    _, archive = write_census_archives(tmp_path)
    with zipfile.ZipFile(archive) as z:
        contents = {name: z.read(name) for name in z.namelist()}
    geography = bytearray(contents['algeo2010.sf1'])
    geography[226:230] = b'Pe\xf1a'  # unused NAME, fixed byte positions retained
    contents['algeo2010.sf1'] = bytes(geography)
    with zipfile.ZipFile(archive, 'w') as z:
        for name, data in contents.items():
            z.writestr(name, data)
    table = read_sf1_population(archive, 'AL', '01')
    assert table.block_id.tolist() == blocks().block_id.tolist()
    assert table.population.tolist() == [40, 60]


def test_undeclared_mask_is_ignored_but_internal_mask_is_accounted(tmp_path):
    tile = write_raster(tmp_path / 'dem.tif', [0, 0])
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(tile.path, 'r+') as ds:
            ds.write_mask(np.zeros((1, 2), dtype='uint8'))
    assert Path(tile.path + '.msk').exists()
    assert file_hash(tile.path) == tile.sha256
    result, qc = build_exposure(sources(tile), blocks(), SPEC)
    assert result.pressure_mmhg.eq(760).all() and result.missing_population.eq(0).all()
    # The environment change is scoped, and the ignored sibling still exists.
    with rasterio.open(tile.path) as ds:
        assert ds.read(1, masked=True).mask.all()
    internal = write_raster(tmp_path / 'internal.tif', [0, 0])
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(internal.path, 'r+') as ds:
            ds.write_mask(np.zeros((1, 2), dtype='uint8'))
    internal = replace(internal, sha256=file_hash(internal.path))
    assert not Path(internal.path + '.msk').exists()
    result, qc = build_exposure(sources(internal), blocks(), SPEC)
    assert result.missing_population.eq(100).all() and result.pressure_mmhg.isna().all()


@pytest.mark.parametrize('problem', ['dem_identity', 'crs'])
def test_stage_rejects_wrong_dem_source_and_invalid_crs(tmp_path, shard_fixture, problem):
    inventory, _ = shard_fixture
    task = json.loads((tmp_path / 'shard-AL/task.json').read_text())
    deps = [inventory, tmp_path / 'census-receipt.json', tmp_path / 'census.tar',
            tmp_path / 'dem-receipt.json', tmp_path / 'dem.tar']
    if problem == 'dem_identity':
        receipt = json.loads(deps[3].read_text())
        receipt['manifest_sha256'] = 'c' * 64
        deps[3].write_text(canonical_json(receipt))
    else:
        task['raster_metadata']['synthetic']['crs'] = 'EPSG:invalid'
    req = request(tmp_path / ('reject-' + problem), 'exposure-atlas', task, deps)
    result = run_stage(req)
    assert result.status == 'fail'
    assert ('reviewed source configuration' if problem == 'dem_identity' else 'Invalid projection') in result.message
    assert not (Path(req.output_dir) / 'exposure.parquet').exists()


def test_integral_float32_population_totals_remain_exact(tmp_path):
    count = 16778
    ids = [f'01001000{1 + i // 8389:01d}00{1 + i % 8389:04d}' for i in range(count)]
    pop = np.full(count, 1000, dtype=np.float32)
    pop[-1] = 217
    geography = gpd.GeoDataFrame(dict(block_id=ids, tract_id=[x[:11] for x in ids], population=pop),
        geometry=[box(i * 100, 0, (i + 1) * 100, 100) for i in range(count)], crs='EPSG:5070')
    tile = write_raster(tmp_path / 'large-total.tif', np.zeros(count))
    expected = 16777217
    assert int(geography.population.sum()) != expected  # reproduces float32 accumulation loss
    result, qc = build_exposure(sources(tile), geography, replace(SPEC, scenarios=('centroid',)))
    assert qc['population'] == expected
    assert result.population.sum() == expected
    assert result.pressure_mmhg.eq(760).all()
    # Also exercise loss within a single tract, not just across tract totals.
    compact = blocks(pop=np.array([16777216, 1], dtype=np.float32))
    result, qc = build_exposure(sources(tile), compact, replace(SPEC, scenarios=('centroid',)))
    assert qc['population'] == result.population.iloc[0] == expected


def test_distributed_quantile_exact_boundary(tmp_path):
    tile = write_raster(tmp_path / 'quantile.tif', [0] * 20 + [3000] * 20)
    whole = blocks(pop=(4, 0)).iloc[:1].copy()
    whole.loc[0, 'geometry'] = box(0, 0, 4000, 100)
    result, _ = build_exposure(sources(tile), whole, replace(SPEC, scenarios=('distributed',)))
    assert result.elevation_p50_m.iloc[0] == 0


@pytest.mark.parametrize('weights', [
    [0.1] * 40, [2, 2 + 2**-40], [2 + 2**-40, 2],
    [1, 9], [9, 1], [2**-100, 1, 2**50, 2**50, 1, 2**-100],
])
def test_quantile_matches_exact_left_cdf(weights):
    weights = np.asarray(weights, dtype=float)
    values = np.repeat([0., 3000.], len(weights) // 2)
    exact = [Fraction(float(w)) for w in weights]
    total = sum(exact)
    expected = []
    for q in (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)):
        cumulative = 0
        for z, weight in zip(values, exact):
            cumulative += weight
            if cumulative >= q * total:
                expected.append(z)
                break
    # Exact binary scaling and splitting retain the represented mass. A near
    # tie must not be treated as an exact tie by an arbitrary numeric tolerance.
    for scale in (0.5, 1., 2.):
        assert weighted_quantiles(values, weights * scale) == expected
        assert weighted_quantiles(values[::-1], weights[::-1] * scale) == expected
        assert weighted_quantiles(np.repeat(values, 2), np.repeat(weights * scale / 2, 2)) == expected


@pytest.mark.parametrize('crs_format', ['wkt', 'projjson'])
def test_equivalent_crs_shards_collect_reproducibly(tmp_path, shard_fixture, crs_format):
    inventory, original_paths = shard_fixture
    task = json.loads((tmp_path / 'shard-AZ/task.json').read_text())
    crs = CRS('EPSG:5070')
    declaration = crs.to_wkt() if crs_format == 'wkt' else crs.to_json()
    task['raster_metadata']['synthetic']['crs'] = declaration
    deps = [inventory, tmp_path / 'census-receipt.json', tmp_path / 'census.tar',
            tmp_path / 'dem-receipt.json', tmp_path / 'dem.tar']
    req = request(tmp_path / 'equivalent-AZ', 'exposure-atlas', task, deps)
    result = run_stage(req)
    assert result.status == 'pass', result.message
    out = Path(req.output_dir)
    paths = original_paths[:3] + [out / name for name in ('artifact_manifest.json', 'exposure.parquet', 'quality.json')]
    collections = [collect_request(tmp_path / 'original', inventory, original_paths),
                   collect_request(tmp_path / 'equivalent', inventory, paths),
                   collect_request(tmp_path / 'reversed', inventory, paths, reverse=True)]
    for collection in collections:
        result = run_stage(collection)
        assert result.status == 'pass', result.message
        result.verify(collection)
    for name in ('atlas.parquet', 'quality.json'):
        assert len({file_hash(Path(c.output_dir) / name) for c in collections}) == 1


def test_equivalent_placement_crs_has_same_identity():
    assert AllocationSpec(placement_crs=CRS('EPSG:5070').to_wkt()).content_hash == SPEC.content_hash


@pytest.mark.parametrize('split_at', [1, 33, 49, 99])
def test_nondyadic_block_split_preserves_all_quantiles(tmp_path, split_at):
    tile = write_raster(tmp_path / 'split.tif', [0] * 10 + [1000] * 40 + [2000] * 40 + [3000] * 10)
    split = blocks(pop=(split_at, 100 - split_at))
    split.loc[0, 'geometry'] = box(0, 0, split_at * 100, 100)
    split.loc[1, 'geometry'] = box(split_at * 100, 0, 10000, 100)
    whole = split.iloc[:1].copy()
    whole.loc[0, 'population'] = 100
    whole.loc[0, 'geometry'] = box(0, 0, 10000, 100)
    for geography in (whole, split, split.iloc[::-1]):
        result, _ = build_exposure(sources(tile), geography, replace(SPEC, scenarios=('distributed',)))
        assert result[['elevation_p10_m', 'elevation_p50_m', 'elevation_p90_m']].iloc[0].tolist() == [0, 1000, 2000]
        assert result.pressure_mmhg.iloc[0] == pytest.approx(np.mean(pressure_mmhg([0] * 10 + [1000] * 40 + [2000] * 40 + [3000] * 10)))


@pytest.mark.parametrize('many_cells', [10, 4100])
@pytest.mark.parametrize('many_low', [True, False])
def test_population_quantile_tie_across_different_densities(tmp_path, many_cells, many_low):
    heights = [0] * many_cells + [3000] if many_low else [0] + [3000] * many_cells
    tile = write_raster(tmp_path / 'density.tif', heights)
    geography = blocks(pop=(1, 1))
    boundary = many_cells * 100 if many_low else 100
    geography.loc[0, 'geometry'] = box(0, 0, boundary, 100)
    geography.loc[1, 'geometry'] = box(boundary, 0, (many_cells + 1) * 100, 100)
    result, qc = build_exposure(sources(tile), geography, replace(SPEC, scenarios=('distributed',)))
    # Each elevation has exactly one person, irrespective of cells or batching.
    assert result.elevation_p50_m.iloc[0] == 0
    assert result.pressure_mmhg.iloc[0] == pytest.approx(np.mean(pressure_mmhg([0, 3000])))
    assert qc['population'] == result.covered_population.iloc[0] == 2


@pytest.mark.parametrize('side', [-1, 1])
def test_geometry_derived_near_median_ties_remain_distinct(tmp_path, side):
    tile = write_raster(tmp_path / 'near.tif', [0, 3000])
    epsilon = 2**-30
    geography = blocks(pop=(4, 0)).iloc[:1].copy()
    geography.loc[0, 'geometry'] = box(epsilon if side > 0 else 0, 0,
                                      200 if side > 0 else 200 - epsilon, 100)
    result, _ = build_exposure(sources(tile), geography, replace(SPEC, scenarios=('distributed',)))
    assert result.elevation_p50_m.iloc[0] == (3000 if side > 0 else 0)


def test_exact_area_cdf_uses_its_own_total_and_duplicate_elevations():
    # A permitted tiny geometry-partition discrepancy must not substitute the
    # Census total for the CDF's own exact total; this is not mass redistribution.
    areas = np.array([0.5, 0.5 + 2**-40])
    assert placement_quantiles([(1, 1., np.array([0., 3000.]), areas)])[1] == 3000
    assert placement_quantiles([(1, 1., np.array([0., 0., 3000.]), np.array([0.25, 0.25, 0.5]))])[1] == 0
    assert placement_quantiles([(1, 1., np.array([-0.]), np.array([1.]))]) == [0., 0., 0.]
    assert not np.signbit(placement_quantiles([(1, 1., np.array([-0.]), np.array([1.]))])[0])


def test_exact_area_cdf_many_nondyadic_blocks():
    # Exercises heterogeneous rational denominators, rather than only the cheap
    # single-block/binary-area case. No timing threshold: the suite has timeout.
    distributions = []
    for i in range(1500):
        area = 3. + (i + 1) * 2**-40
        distributions.append((3, area, np.array([0., 1., 2.]), np.array([1., 1., area - 2.])))
    assert placement_quantiles(distributions) == [0., 1., 2.]


@pytest.mark.parametrize('affine', [
    rasterio.Affine(128, 0, 0, 0, -128, 128),
    rasterio.Affine(128, 0, 0, 0, 128, 0),
    rasterio.Affine(-128, 0, 256, 0, -128, 128),
    rasterio.Affine(0, 128, 0, 128, 0, 0),
    rasterio.Affine(128, 64, 0, 0, -128, 128),
])
def test_raster_orientation_uses_actual_pixels(tmp_path, affine):
    tile = write_raster(tmp_path / 'orientation.tif')
    with rasterio.open(tile.path, 'r+') as ds:
        ds.transform = affine
    tile = replace(tile, sha256=file_hash(tile.path))
    # Dyadic transforms make these boundaries exact, including +/- one ulp.
    pixel_xy = [(0, 0), (1, 0.5), (2, 0.5), (0.5, 1), (-0.125, 0.5),
                (np.nextafter(1., 0.), 0.5), (np.nextafter(1., 2.), 0.5)]
    xy = np.array([affine @ p for p in pixel_xy])
    with RasterSampler([tile], 'EPSG:5070') as sampler:
        z, reason = sampler.sample(xy)
    assert z[:2].tolist() == [0, 3000]
    assert reason[2:5].tolist() == ['outside_coverage'] * 3
    # Translation/shear can round the world coordinate itself; compare the
    # represented world points to their exact affine pixel mapping here.
    for i in (5, 6):
        col, _ = (~affine) @ xy[i]
        assert z[i] == (0 if col < 1 else 3000)
    geography = blocks(pop=(40, 60))
    for i in range(2):
        geography.loc[i, 'geometry'] = Polygon([affine @ p for p in [(i, 0), (i+1, 0), (i+1, 1), (i, 1)]])
    result, _ = build_exposure(sources(tile), geography, replace(SPEC, scenarios=('centroid',)))
    assert result.missing_population.eq(0).all()
    assert result.pressure_mmhg.iloc[0] == pytest.approx(np.dot([0.4, 0.6], pressure_mmhg([0, 3000])))


def test_south_up_complete_in_both_scenarios(tmp_path):
    tile = write_raster(tmp_path / 'south.tif', [0])
    with rasterio.open(tile.path, 'r+') as ds:
        ds.transform = rasterio.Affine(100, 0, 0, 0, 100, 0)
    tile = replace(tile, sha256=file_hash(tile.path))
    result, _ = build_exposure(sources(tile), blocks(pop=(100, 0)).iloc[:1], SPEC)
    assert result.missing_population.eq(0).all() and result.pressure_mmhg.eq(760).all()


@pytest.mark.parametrize('affine', [rasterio.Affine(100, 0, 0, 0, 0, 100),
                                   rasterio.Affine(float('nan'), 0, 0, 0, -100, 100)])
def test_invalid_affine_rejected(tmp_path, affine):
    tile = write_raster(tmp_path / 'bad-affine.tif')
    with rasterio.open(tile.path, 'r+') as ds:
        ds.transform = affine
    tile = replace(tile, sha256=file_hash(tile.path))
    with pytest.raises(ContractError, match='affine'):
        build_exposure(sources(tile), blocks(), SPEC)


@pytest.mark.parametrize('embedded,declared,passes', [
    ('EPSG:5070+5703', 'EPSG:5070+5703', True),
    ('EPSG:5070+5703', 'EPSG:5070', True),
    ('EPSG:5070', 'EPSG:5070+5703', True),
    ('EPSG:5070+5773', 'EPSG:5070+5773', False),
    ('EPSG:5070+6360', 'EPSG:5070+6360', False),
    ('EPSG:4979', 'EPSG:4979', False),
    ('EPSG:5070', 'EPSG:5070+5773', False),
])
def test_vertical_crs_corroborates_reviewed_labels(tmp_path, embedded, declared, passes):
    tile = write_raster(tmp_path / 'vertical.tif', [1000, 1000], crs=embedded)
    with rasterio.open(tile.path) as ds:
        assert CRS(ds.crs) == CRS(embedded)  # verify the actual GeoTIFF round-trip
    tile = replace(tile, crs=declared)
    if passes:
        result, _ = build_exposure(sources(tile), blocks(), SPEC)
        np.testing.assert_allclose(result.pressure_mmhg, pressure_mmhg(1000), rtol=1e-12)
    else:
        with pytest.raises(ContractError, match='vertical|ellipsoidal'):
            build_exposure(sources(tile), blocks(), SPEC)


def test_stage_rejects_declared_vertical_contradiction(tmp_path, shard_fixture):
    inventory, _ = shard_fixture
    task = json.loads((tmp_path / 'shard-AL/task.json').read_text())
    task['raster_metadata']['synthetic']['crs'] = 'EPSG:5070+5773'
    deps = [inventory, tmp_path / 'census-receipt.json', tmp_path / 'census.tar',
            tmp_path / 'dem-receipt.json', tmp_path / 'dem.tar']
    req = request(tmp_path / 'bad-vertical-stage', 'exposure-atlas', task, deps)
    result = run_stage(req)
    assert result.status == 'fail' and 'vertical datum' in result.message
    assert not (Path(req.output_dir) / 'exposure.parquet').exists()


@pytest.mark.parametrize('problem', ['depth', 'unknown'])
def test_explicit_depth_and_unknown_vertical_crs_rejected(tmp_path, problem):
    from pyproj.crs import CompoundCRS
    vertical = CRS('EPSG:5703').to_json_dict()
    vertical.pop('id', None)
    if problem == 'depth':
        vertical['coordinate_system']['axis'][0]['direction'] = 'down'
    else:
        vertical['datum'].pop('id', None)
        vertical['datum']['name'] = 'Unidentified vertical datum'
    vertical['name'] = 'Synthetic ' + problem + ' height'
    crs = CompoundCRS('Synthetic vertical test', [CRS('EPSG:5070'), CRS.from_json_dict(vertical)])
    tile = write_raster(tmp_path / 'vertical-custom.tif', crs=crs.to_wkt())
    with rasterio.open(tile.path) as ds:
        embedded = CRS(ds.crs).sub_crs_list[-1]
        assert embedded.is_vertical
        # GeoTIFF's vertical keys normalize a downward axis to upward height in
        # this GDAL version. The reviewed declaration still carries the explicit
        # contradictory direction and must be rejected before sampling.
        if problem == 'depth':
            assert crs.sub_crs_list[-1].axis_info[0].direction == 'down'
        else:
            assert embedded.datum != CRS('EPSG:5703').datum
    with pytest.raises(ContractError, match='vertical'):
        build_exposure(sources(tile), blocks(), SPEC)


def test_gcp_only_raster_rejected(tmp_path):
    from rasterio.control import GroundControlPoint
    tile = write_raster(tmp_path / 'gcp.tif')
    with rasterio.open(tile.path, 'r+') as ds:
        ds.gcps = ([GroundControlPoint(row=0, col=0, x=0, y=100),
                    GroundControlPoint(row=0, col=2, x=200, y=100),
                    GroundControlPoint(row=1, col=0, x=0, y=0)], rasterio.crs.CRS.from_epsg(5070))
    tile = replace(tile, sha256=file_hash(tile.path))
    with pytest.raises(ContractError, match='CRS missing|GCP'):
        build_exposure(sources(tile), blocks(), SPEC)


def test_dispersed_reads_are_bounded_and_nonfinite_points_are_missing(tmp_path, monkeypatch):
    tile = write_raster(tmp_path / 'islands.tif', np.zeros(10001))
    original_open = rasterio.open
    windows = []

    class GuardedDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.dataset.close()
        def __getattr__(self, name):
            return getattr(self.dataset, name)
        def read(self, *args, **kwargs):
            window = kwargs['window']
            assert window.width * window.height <= 4096
            windows.append(window)
            return self.dataset.read(*args, **kwargs)

    monkeypatch.setattr(rasterio, 'open', lambda *a, **k: GuardedDataset(original_open(*a, **k)))
    with RasterSampler([tile], 'EPSG:5070') as sampler:
        z, reason = sampler.sample(np.array([[50, 50], [1000050, 50], [np.inf, 50]]))
    assert z[:2].tolist() == [0, 0] and reason[2] == 'outside_coverage'
    assert len(windows) == 2


@pytest.mark.parametrize('problem', ['missing_decision', 'constant', 'product', 'missing_file'])
def test_stage_requires_current_owner_exposure_approval(tmp_path, shard_fixture, monkeypatch, problem):
    import oxyformer.exposure.physics as physics
    inventory, paths = shard_fixture
    approvals = yaml.safe_load((ROOT / 'configs/approvals.yaml').read_text())
    if problem == 'missing_decision':
        del approvals['owner_decisions']['exposure']
    elif problem == 'constant':
        approvals['owner_decisions']['exposure']['sea_level_pressure_mmhg'] = 750.
    elif problem == 'product':
        approvals['owner_decisions']['exposure']['dem_product'] = 'unapproved-replacement'
    path = tmp_path / 'owner-approvals.yaml'
    if problem != 'missing_file':
        path.write_text(yaml.safe_dump(approvals))
    monkeypatch.setattr(physics, '_APPROVAL_PATH', path, raising=False)
    req = collect_request(tmp_path / 'approval-collection', inventory, paths)
    result = run_stage(req)
    assert result.status == ('blocked' if problem == 'missing_file' else 'fail')
    assert not (Path(req.output_dir) / 'atlas.parquet').exists()
    with pytest.raises((ContractError, FileNotFoundError)):
        build_exposure(sources(write_raster(tmp_path / 'approval-dem.tif')), blocks(), SPEC)


@pytest.mark.parametrize('identity', ['physical_hash', 'allocation_hash'])
def test_collection_rejects_contradictory_manifest_specification(tmp_path, shard_fixture, identity):
    inventory, paths = shard_fixture
    manifest = json.loads(paths[0].read_text())
    manifest[identity] = 'e' * 64
    paths[0].write_text(canonical_json(manifest))
    req = collect_request(tmp_path / 'contradictory-manifest', inventory, paths)
    result = run_stage(req)
    assert result.status == 'fail' and 'inconsistent' in result.message
    assert not (Path(req.output_dir) / 'atlas.parquet').exists()
