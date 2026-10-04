"""Deterministic exposure and the two StageRequest entry points.

Reviewed task JSON schema v1 (this module owns the stage-specific format):
  schema_version=1, review_status='reviewed', review_id, atlas_id,
  geography_vintage=2010, allocation=AllocationSpec fields,
  sources=[SourceManifest.to_dict(), ...],
  shards=[{id, state_fips: [two-digit strings, ...]}],
  state_controls={state: {expected_block_count, expected_population}},
  blocks=[{source_id, state_fips, resource, format, crs,
           expected_block_count, expected_population}],
  rasters=[{source_id, tile_id, shard_ids, resource, crs, vertical_unit,
            vertical_datum, nodata}],
  shard_id (exposure-atlas only),
  shard_artifacts={id: {manifest, exposure, quality}} (atlas-collect only).

Resource: {dependency_path, sha256, member?}. Every path must be an explicit
StageRequest dependency. member selects one regular member of an uncompressed
acquisition payload.tar. SourceManifest.payload_hash binds the entire archive;
sha256 binds the member. No sibling-attempt discovery, downloads or shared writes.

Blocks use geoparquet or shapefile_zip; reviewed SourceManifest field mappings
provide block_id, tract_id and population. A reviewed block binding may instead
explicitly set tract_id_from_block_prefix=true to derive the 2010 tract GEOID
from the first 11 characters of its 15-character block GEOID. No source column
names are guessed. All other raw attributes are excluded before the core service.

All shards share the complete reviewed inventory. Only shard_id/shard_artifacts
are excluded from its digest. Raw resources are read only for the selected shard.
Collection needs all three files of every shard as separately hashed dependencies.
A blocked acquisition inventory is not a reviewed exposure task. State controls
must have an independent reviewed Census basis, not be derived from bindings.

The producer attests that QC footprints equal its verified source geometry.
Collection verifies canonical footprint bytes, CRS, digest and disjoint interiors;
it does not reopen raw sources to reconstruct them. Footprints include all blocks,
including zero-population and uncovered blocks. There is no topology tolerance or
repair and no claim of bitwise reproducibility across GEOS environments. Older
shards without footprint evidence must be rebuilt.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import tarfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
import rasterio
from rasterio.errors import RasterioError
import shapely
from shapely.errors import GEOSException
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import Point, box
from shapely.strtree import STRtree
import yaml

from oxyformer.contracts import SourceManifest, StageRequest, StageResult
from oxyformer.provenance import (
    ArtifactLineage, ArtifactRecord, ContractError, canonical_json, file_hash, require,
)
from oxyformer.exposure.physics import PHYSICS, PhysicalSpec, pressure_mmhg, deficit_mmhg
from oxyformer.exposure.population_allocation import AllocationSpec, placements, validate_blocks, validate_disjoint
from oxyformer.exposure.quality import QUANTILE_LABEL, check_conservation, weighted_quantiles


@dataclass(frozen=True)
class RasterSpec:
    tile_id: str
    source_id: str
    path: str
    crs: str
    vertical_unit: str
    vertical_datum: str
    nodata: float | None


@dataclass(frozen=True)
class Geography:
    blocks: gpd.GeoDataFrame
    census_source_ids: tuple[str, ...]
    rasters: tuple[RasterSpec, ...]
    vintage: int = 2010


class RasterSampler:
    """Bounded handles and single-pixel reads; never loads a national mosaic."""
    def __init__(self, specs, area_crs):
        self.specs = sorted(specs, key=lambda s: s.tile_id)
        require(len({s.tile_id for s in specs}) == len(specs), 'duplicate DEM tile IDs')
        self.handles = OrderedDict()
        self.transforms = []
        footprints = []
        for spec in self.specs:
            require(spec.vertical_unit == 'm' and spec.vertical_datum == 'NAVD88',
                    'DEM vertical units/datum must be explicitly m/NAVD88')
            with rasterio.open(spec.path) as ds:
                require(ds.crs is not None and CRS(ds.crs) == CRS(spec.crs), 'DEM CRS mismatch')
                require(ds.count == 1, 'DEM must have one elevation band')
                require(ds.units[0] in (None, 'm', 'metre', 'meter', 'meters'), 'DEM vertical unit mismatch')
                require(ds.scales == (1.0,) and ds.offsets == (0.0,), 'scaled DEM needs explicit adaptation')
                require(ds.nodata == spec.nodata or (ds.nodata is not None and spec.nodata is not None
                        and math.isnan(ds.nodata) and math.isnan(spec.nodata)), 'DEM nodata mismatch')
                bounds = transform_bounds(ds.crs, area_crs, *ds.bounds, densify_pts=21)
                require(all(math.isfinite(v) for v in bounds), 'invalid DEM extent')
                footprints.append(box(*bounds))
                self.transforms.append(Transformer.from_crs(area_crs, ds.crs, always_xy=True))
        self.index = STRtree(footprints)

    def sample(self, x, y):
        seen_nodata = False
        # Stable tile priority in overedge overlaps, independent of input order.
        for i in sorted(self.index.query(Point(x, y))):
            i = int(i)
            spec = self.specs[i]
            if i not in self.handles:
                if len(self.handles) == 8:
                    self.handles.popitem(last=False)[1].close()
                self.handles[i] = rasterio.open(spec.path)
            self.handles.move_to_end(i)
            ds = self.handles[i]
            xx, yy = self.transforms[i].transform(x, y)
            row, col = ds.index(xx, yy)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                continue
            value = ds.read(1, window=Window(col, row, 1, 1), masked=True)[0, 0]
            if np.ma.is_masked(value) or not np.isfinite(value):
                seen_nodata = True
                continue
            z = float(value)
            if not PHYSICS.minimum_elevation_m <= z <= PHYSICS.maximum_elevation_m:
                return None, 'outside_physical_domain'
            return z, None
        return None, 'nodata' if seen_nodata else 'outside_raster_coverage'

    def close(self):
        for ds in self.handles.values():
            ds.close()
        self.handles.clear()


def _identities(sources):
    require(bool(sources), 'source manifests required')
    require(all(type(s) is SourceManifest for s in sources), 'expected merged SourceManifest contracts')
    require(len({s.source_id for s in sources}) == len(sources), 'duplicate source identities')
    for source in sources:
        source.assert_usable()
    return {s.source_id: s.content_hash for s in sorted(sources, key=lambda s: s.source_id)}


def build_exposure(source_manifests, geography, allocation_spec):
    """Return (tract DataFrame, QC dict) from trusted, already-adapted geography.

    Direct callers own the verified source-to-Geography adaptation; run_stage
    implements it for files/archives. No outcomes, learned weights or tensors
    enter this boundary. Missing population stays in accounting; all full-tract
    statistics are null for incomplete tracts. Zero-population tracts are kept.
    """
    identities = _identities(source_manifests)
    require(type(geography) is Geography and geography.vintage == 2010, '2010 geography required')
    require(type(allocation_spec) is AllocationSpec, 'explicit AllocationSpec required')
    allocation_spec.validate()
    validate_blocks(geography.blocks, check_topology=False)
    require(bool(geography.census_source_ids) and
            set(geography.census_source_ids) <= set(identities), 'Census source identity missing')
    require(all(s.source_id in identities for s in geography.rasters), 'DEM source identity missing')
    blocks = geography.blocks.sort_values('block_id').to_crs(allocation_spec.area_crs)
    validate_blocks(blocks)
    footprint = _make_footprint(blocks.geometry.to_numpy(), allocation_spec.area_crs)
    sampler = RasterSampler(geography.rasters, allocation_spec.area_crs)
    tracts, ledger = {}, []
    try:
        for row in blocks.itertuples():
            require(row.geometry.is_valid and row.geometry.area > 0, 'invalid projected block geometry')
            tract = tracts.setdefault(row.tract_id, {'population': 0, 'covered': [], 'missing': [],
                                                     'z': [], 'pressure': [], 'deficit': []})
            tract['population'] += int(row.population)
            missing, covered, reasons = [], [], {}
            if row.population > 0:
                for x, y, fraction in placements(row.geometry, allocation_spec):
                    mass = row.population * fraction
                    z, reason = sampler.sample(x, y)
                    if reason is not None:
                        missing.append(mass)
                        reasons.setdefault(reason, []).append(mass)
                    else:
                        covered.append(mass)
                        tract['z'].append(z)
                        # Transform EACH location before population averaging.
                        tract['pressure'].append(float(pressure_mmhg(z)))
                        tract['deficit'].append(float(deficit_mmhg(z)))
                        tract['covered'].append(mass)
            missing_mass, covered_mass = math.fsum(missing), math.fsum(covered)
            check_conservation(row.population, covered_mass, missing_mass)
            tract['missing'].append(missing_mass)
            ledger.append({'block_id': row.block_id, 'tract_id': row.tract_id,
                           'population': int(row.population), 'covered_population': covered_mass,
                           'missing_population': missing_mass,
                           'omissions': {key: math.fsum(values) for key, values in sorted(reasons.items())}})
    finally:
        sampler.close()
    rows = []
    for tract_id, values in sorted(tracts.items()):
        population = values['population']
        missing = math.fsum(values['missing'])
        covered = math.fsum(values['covered'])
        check_conservation(population, covered, missing)
        valid = population > 0 and missing == 0
        row = {'tract_id': tract_id, 'scenario': allocation_spec.scenario,
               'population': population, 'covered_population': covered, 'missing_population': missing,
               'status': 'complete' if valid else ('zero_population' if population == 0 else 'incomplete'),
               'pressure_mmhg': None, 'deficit_mmhg': None, 'mean_elevation_m': None,
               'inhabited_p10_m': None, 'inhabited_p50_m': None, 'inhabited_p90_m': None,
               'quantile_interpretation': QUANTILE_LABEL}
        if valid:
            weights = values['covered']
            row['pressure_mmhg'] = math.fsum(p * w for p, w in zip(values['pressure'], weights)) / population
            row['deficit_mmhg'] = math.fsum(d * w for d, w in zip(values['deficit'], weights)) / population
            row['mean_elevation_m'] = math.fsum(z * w for z, w in zip(values['z'], weights)) / population
            q = weighted_quantiles(values['z'], weights)
            row.update(dict(zip(('inhabited_p10_m', 'inhabited_p50_m', 'inhabited_p90_m'), q)))
        rows.append(row)
    qc = {'schema_version': 1, 'source_identities': identities, 'physical': asdict(PHYSICS),
          'allocation': asdict(allocation_spec), 'quantile_interpretation': QUANTILE_LABEL,
          'footprint': footprint,
          'blocks': ledger, 'population': sum(r['population'] for r in ledger),
          'covered_population': math.fsum(r['covered_population'] for r in ledger),
          'missing_population': math.fsum(r['missing_population'] for r in ledger),
          'incomplete_tracts': [r['tract_id'] for r in rows if r['status'] == 'incomplete'],
          'zero_population_tracts': [r['tract_id'] for r in rows if r['status'] == 'zero_population']}
    qc['coverage_pass'] = qc['missing_population'] == 0
    return pd.DataFrame(rows), qc


def _polygon(geometry):
    require(geometry is not None and not geometry.is_empty and geometry.is_valid
            and geometry.geom_type in ('Polygon', 'MultiPolygon')
            and math.isfinite(geometry.area) and geometry.area > 0, 'invalid polygonal footprint')


def _footprint_bytes(geometry):
    return shapely.to_wkb(shapely.normalize(geometry), byte_order=1, output_dimension=2)


def _make_footprint(geometries, crs):
    # General union supports differently noded coincident edges; no coverage-union
    # assumption, precision grid, simplification, tolerance or geometry repair.
    geometry = shapely.union_all(geometries)
    _polygon(geometry)
    raw = _footprint_bytes(geometry)
    return {'crs': crs, 'wkb_hex': raw.hex(), 'sha256': sha256(raw).hexdigest()}


def _read_footprint(qc, crs):
    require('footprint' in qc, 'required footprint missing; rebuild shard')
    spec = qc['footprint']
    require(CRS(spec['crs']) == CRS(crs), 'footprint CRS mismatch')
    raw = bytes.fromhex(spec['wkb_hex'])
    require(sha256(raw).hexdigest() == spec['sha256'], 'footprint digest mismatch')
    geometry = shapely.from_wkb(raw)
    _polygon(geometry)
    require(_footprint_bytes(geometry) == raw, 'noncanonical footprint WKB')
    return geometry


def _state_totals(records, controls, states):
    totals = {state: [0, 0] for state in states}
    for record in records:
        state = record['block_id'][:2]
        require(state in totals, 'unknown state in block ledger')
        totals[state][0] += 1
        totals[state][1] += int(record['population'])
    for state, (count, population) in totals.items():
        control = controls[state]
        require(count == control['expected_block_count'] and population == control['expected_population'],
                f'per-state controls mismatch: {state}')


def _digest(value):
    return sha256(canonical_json(value).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    with path.open('x') as out:
        out.write(canonical_json(value))


def _dependency(request, path):
    require(path in request.dependency_paths, 'path is not an explicit request dependency')
    return Path(path)


def _resource(request, binding, source, scratch, name, *, raster=False):
    path = _dependency(request, binding['dependency_path'])
    index = request.dependency_paths.index(str(path))
    require(source.payload_hash == request.dependency_hashes[index], 'source payload identity mismatch')
    if 'member' not in binding:
        require(binding['sha256'] == request.dependency_hashes[index], 'resource digest mismatch')
        return str(path)
    member = binding['member']
    pure = PurePosixPath(member)
    require(not pure.is_absolute() and '..' not in pure.parts and str(pure) == member,
            'archive member must be a normalized relative path')
    with tarfile.open(path, 'r:') as archive:
        matches = [m for m in archive.getmembers() if m.name == member]
        require(len(matches) == 1 and matches[0].isfile(), 'required regular archive member missing or duplicated')
        digest = sha256()
        target = None if raster else (scratch / name).open('xb')
        try:
            with archive.extractfile(matches[0]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                    if target is not None:
                        target.write(chunk)
        finally:
            if target is not None:
                target.close()
        require(digest.hexdigest() == binding['sha256'], 'archive member hash mismatch')
    return f'/vsitar/{path}/{member}' if raster else str(scratch / name)


class MissingPrerequisite(ValueError):
    pass


def _task(request):
    task = _json(request.task_path)
    require(task.get('schema_version') == 1, 'unsupported exposure task schema')
    if task.get('review_status') != 'reviewed' or not task.get('review_id'):
        raise MissingPrerequisite('reviewed exposure task manifest required')
    require(task.get('geography_vintage') == 2010, 'reviewed geography vintage must be 2010')
    require(isinstance(task.get('atlas_id'), str) and bool(task['atlas_id']), 'atlas ID required')
    shards = task['shards']
    require(bool(shards) and len({s['id'] for s in shards}) == len(shards), 'duplicate/empty shard selection')
    states = [state for shard in shards for state in shard['state_fips']]
    require(len(set(states)) == len(states) and all(isinstance(s, str) and len(s) == 2 and s.isdigit()
                                                   for s in states), 'overlapping/invalid shard jurisdictions')
    require(all(s['state_fips'] for s in shards), 'empty shard jurisdiction selection')
    sources = [SourceManifest.from_json(canonical_json(value)) for value in task['sources']]
    if any(s.mapping_status != 'reviewed' for s in sources):
        raise MissingPrerequisite('reviewed source mappings required')
    _identities(sources)
    require(set(b['state_fips'] for b in task['blocks']) == set(states), 'block inventory omissions or overlaps')
    require(set(task['state_controls']) == set(states), 'independent state controls missing or unexpected')
    for b in task['blocks']:
        require(type(b['expected_block_count']) is int and b['expected_block_count'] > 0
                and type(b['expected_population']) is int and b['expected_population'] >= 0,
                'reviewed block counts and population totals required')
    for state in states:
        control = task['state_controls'][state]
        require(type(control['expected_block_count']) is int and control['expected_block_count'] > 0
                and type(control['expected_population']) is int and control['expected_population'] >= 0,
                'invalid independent state controls')
        selected = [b for b in task['blocks'] if b['state_fips'] == state]
        declared = (sum(b['expected_block_count'] for b in selected),
                    sum(b['expected_population'] for b in selected))
        require(declared == (control['expected_block_count'], control['expected_population']),
                f'declared bindings differ from independent state controls: {state}')
    source_by_id = {source.source_id: source for source in sources}
    resource_keys = [(source_by_id[b['source_id']].payload_hash, b['resource'].get('member'))
                     for b in task['blocks']]
    require(len(resource_keys) == len(set(resource_keys)), 'duplicate selected block resource')
    known_shards = {s['id'] for s in shards}
    require(len({r['tile_id'] for r in task['rasters']}) == len(task['rasters']), 'duplicate raster inventory')
    require(all(r['shard_ids'] and set(r['shard_ids']) <= known_shards for r in task['rasters']),
            'unknown raster shard selection')
    spec_hash = _digest({key: value for key, value in task.items() if key not in ('shard_id', 'shard_artifacts')})
    return task, {s.source_id: s for s in sources}, spec_hash


def _build_shard(request, task, sources, scratch):
    selected = [s for s in task['shards'] if s['id'] == task.get('shard_id')]
    require(len(selected) == 1, 'selected shard is not in reviewed task manifest')
    states = selected[0]['state_fips']
    tables, used_sources = [], []
    for i, binding in enumerate(task['blocks']):
        if binding['state_fips'] not in states:
            continue
        source = sources[binding['source_id']]
        mapping = dict(source.field_mapping)
        inverse = {v: k for k, v in mapping.items()}
        derive_tract = binding.get('tract_id_from_block_prefix') is True
        needed = ('block_id', 'population') if derive_tract else ('block_id', 'tract_id', 'population')
        require(set(needed) <= set(inverse), 'reviewed Census field mapping missing')
        columns = [inverse[k] for k in needed]
        path = _resource(request, binding['resource'], source, scratch, f'blocks-{i}')
        if binding['format'] == 'geoparquet':
            table = gpd.read_parquet(path, columns=columns + ['geometry'])
        elif binding['format'] == 'shapefile_zip':
            table = gpd.read_file('zip://' + path, columns=columns)
        else:
            raise ContractError('unreviewed block file format')
        require(table.crs is not None and CRS(table.crs) == CRS(binding['crs']), 'block CRS mismatch')
        table = table[columns + ['geometry']].rename(columns={c: mapping[c] for c in columns})
        if derive_tract:
            table['tract_id'] = table.block_id.str[:11]
        validate_blocks(table)
        require(table.block_id.str[:2].eq(binding['state_fips']).all(), 'block state membership mismatch')
        require(len(table) == binding['expected_block_count'] and
                sum(int(n) for n in table.population) == binding['expected_population'],
                'unexpected block omission or changed population total')
        tables.append(table.to_crs(task['allocation']['area_crs']))
        used_sources.append(source.source_id)
    require(bool(tables), 'no selected blocks')
    rasters = []
    for i, binding in enumerate(task['rasters']):
        if task['shard_id'] in binding['shard_ids']:
            path = _resource(request, binding['resource'], sources[binding['source_id']], scratch,
                             f'raster-{i}', raster=True)
            rasters.append(RasterSpec(**{k: binding[k] for k in
                                        ('tile_id', 'source_id', 'crs', 'vertical_unit', 'vertical_datum', 'nodata')},
                                      path=path))
    geography = Geography(blocks=gpd.GeoDataFrame(pd.concat(tables, ignore_index=True), geometry='geometry',
                                                 crs=task['allocation']['area_crs']),
                          census_source_ids=tuple(sorted(set(used_sources))), rasters=tuple(rasters))
    _state_totals(geography.blocks.to_dict('records'), task['state_controls'], states)
    return build_exposure(tuple(sources.values()), geography, AllocationSpec(**task['allocation']))


def _validate_ledger(frame, qc, states, expected_count, expected_population):
    ledger = qc['blocks']
    require(len(ledger) == expected_count and len({b['block_id'] for b in ledger}) == len(ledger),
            'unexpected block omission or overlap')
    require(sum(b['population'] for b in ledger) == expected_population == qc['population'],
            'collection population mismatch')
    require({b['block_id'][:2] for b in ledger} == set(states), 'collection jurisdiction mismatch')
    require(not frame.tract_id.duplicated().any() and set(frame.tract_id) == {b['tract_id'] for b in ledger},
            'tract overlap or unexpected omission')
    by_tract = {}
    for b in ledger:
        require(b['tract_id'] == b['block_id'][:11], 'collection tract membership mismatch')
        check_conservation(b['population'], b['covered_population'], b['missing_population'])
        require(math.isclose(math.fsum(b['omissions'].values()), b['missing_population'], abs_tol=1e-8),
                'incomplete omission accounting')
        by_tract.setdefault(b['tract_id'], []).append(b)
    for row in frame.itertuples():
        blocks = by_tract[row.tract_id]
        require(row.population == sum(b['population'] for b in blocks), 'tract population mismatch')
        check_conservation(row.population, row.covered_population, row.missing_population)
        require(math.isclose(row.missing_population, math.fsum(b['missing_population'] for b in blocks),
                             abs_tol=1e-8), 'tract missing population mismatch')
        require(row.status == ('complete' if row.population > 0 else 'zero_population'), 'invalid tract status')
        if row.population > 0:
            require(all(math.isfinite(v) for v in (row.pressure_mmhg, row.deficit_mmhg,
                    row.mean_elevation_m, row.inhabited_p10_m, row.inhabited_p50_m, row.inhabited_p90_m)),
                    'missing tract statistics')
    require(qc['coverage_pass'] and qc['missing_population'] == 0 and
            not qc['incomplete_tracts'] and all(b['missing_population'] == 0 for b in ledger),
            'shard coverage failed; missing coverage cannot be collected as complete')
    check_conservation(qc['population'], qc['covered_population'], qc['missing_population'])


def _collect(request, task, sources, spec_hash, config):
    refs = task.get('shard_artifacts', {})
    require(set(refs) == {s['id'] for s in task['shards']}, 'missing or unexpected shard')
    frames, ledgers, footprints, producer_codes = [], [], [], set()
    for shard in sorted(task['shards'], key=lambda s: s['id']):
        ref = refs[shard['id']]
        manifest_path, exposure_path, quality_path = [_dependency(request, ref[k])
                                                     for k in ('manifest', 'exposure', 'quality')]
        manifest, qc = _json(manifest_path), _json(quality_path)
        producer_code = manifest.get('code_identity')
        require(isinstance(producer_code, str) and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', producer_code)
                is not None, 'invalid or missing producer code identity')
        producer_codes.add(producer_code)
        require(manifest['status'] == 'pass' and manifest['stage'] == 'exposure-atlas', 'nonpassing shard')
        require(manifest['shard_id'] == shard['id'] and manifest['atlas_spec_hash'] == spec_hash,
                'shard identity/specification mismatch')
        require(manifest['config_hash'] == request.config_hash and manifest['physical'] == config['physical']
                and qc['physical'] == config['physical'], 'inconsistent physical specifications')
        require(manifest['source_identities'] == qc['source_identities'] == _identities(tuple(sources.values())),
                'changed source identities')
        require(manifest['allocation'] == qc['allocation'] == asdict(AllocationSpec(**task['allocation'])),
                'inconsistent placement specification')
        require(manifest['artifacts'] == {'exposure.parquet': file_hash(exposure_path),
                                          'quality.json': file_hash(quality_path)}, 'shard artifact hash mismatch')
        footprints.append(_read_footprint(qc, task['allocation']['area_crs']))
        frame = pd.read_parquet(exposure_path)
        require(frame.scenario.eq(task['allocation']['scenario']).all(), 'row scenario mismatch')
        require(frame.quantile_interpretation.eq(QUANTILE_LABEL).all(), 'quantile labeling mismatch')
        selected = [b for b in task['blocks'] if b['state_fips'] in shard['state_fips']]
        _validate_ledger(frame, qc, shard['state_fips'], sum(b['expected_block_count'] for b in selected),
                         sum(b['expected_population'] for b in selected))
        _state_totals(qc['blocks'], task['state_controls'], shard['state_fips'])
        frames.append(frame)
        ledgers.extend(qc['blocks'])
    validate_disjoint(footprints, names=sorted(refs), label='shard')
    require(len(producer_codes) == 1, 'mixed producer code identities')
    require(len({b['block_id'] for b in ledgers}) == len(ledgers), 'overlapping shard blocks')
    frame = pd.concat(frames, ignore_index=True).sort_values('tract_id').reset_index(drop=True)
    require(not frame.tract_id.duplicated().any(), 'overlapping shard tracts')
    qc = {'schema_version': 1, 'source_identities': _identities(tuple(sources.values())),
          'physical': config['physical'], 'allocation': asdict(AllocationSpec(**task['allocation'])),
          'quantile_interpretation': QUANTILE_LABEL, 'blocks': sorted(ledgers, key=lambda b: b['block_id']),
          'population': sum(b['population'] for b in ledgers),
          'covered_population': math.fsum(b['covered_population'] for b in ledgers),
          'missing_population': math.fsum(b['missing_population'] for b in ledgers),
          'coverage_pass': True, 'incomplete_tracts': [],
          'zero_population_tracts': frame.loc[frame.status == 'zero_population', 'tract_id'].tolist(),
          'collected_shards': sorted(refs), 'producer_code_identity': next(iter(producer_codes)),
          'footprint': _make_footprint(footprints, task['allocation']['area_crs'])}
    return frame, qc


def run_stage(request: StageRequest) -> StageResult:
    """Verify dependencies first. A coverage failure writes diagnostics with fail.

    Missing task/source review returns blocked. All other failed validation returns
    fail. Output directory must be fresh; consumers must require status pass.
    """
    scratch = None
    try:
        request.verify_inputs()
        require(request.stage in ('exposure-atlas', 'atlas-collect'), 'unsupported exposure stage')
        config = yaml.safe_load(Path(request.config_path).read_text())
        require(config['schema_version'] == 1 and config['coverage_policy'] ==
                'retain_population_and_null_incomplete_tracts', 'unsupported exposure configuration')
        PhysicalSpec(**config['physical']).validate()
        require(config['physical'] == asdict(PHYSICS), 'complete physical specification required')
        task, sources, spec_hash = _task(request)
        AllocationSpec(**task['allocation']).validate()
        root = Path(request.output_dir)
        require(not root.is_symlink(), 'output directory cannot be a symlink')
        root.mkdir(parents=True, exist_ok=True)
        require(not any(root.iterdir()), 'output directory must be fresh')
        scratch = root / '.inputs'
        scratch.mkdir()
        if request.stage == 'exposure-atlas':
            frame, qc = _build_shard(request, task, sources, scratch)
            filename = 'exposure.parquet'
        else:
            frame, qc = _collect(request, task, sources, spec_hash, config)
            filename = 'atlas.parquet'
        status = 'pass' if qc['coverage_pass'] else 'fail'
        frame.to_parquet(root / filename, index=False)
        _write_json(root / 'quality.json', qc)
        manifest = {'schema_version': 1, 'stage': request.stage, 'status': status,
                    'shard_id': task.get('shard_id') if request.stage == 'exposure-atlas' else None,
                    'atlas_spec_hash': spec_hash, 'source_identities': qc['source_identities'],
                    'physical': config['physical'], 'allocation': qc['allocation'],
                    'request_hash': request.content_hash, 'config_hash': request.config_hash,
                    'code_identity': request.code_identity,
                    'producer_code_identity': qc.get('producer_code_identity', request.code_identity),
                    'artifacts': {filename: file_hash(root / filename), 'quality.json': file_hash(root / 'quality.json')}}
        _write_json(root / 'artifact_manifest.json', manifest)
        lineage = ArtifactLineage(source_hashes=tuple(sorted(s.payload_hash for s in sources.values())),
                                  unit_ids=tuple(frame.tract_id), parent_hashes=request.dependency_hashes,
                                  split_hash=None, config_hash=request.config_hash, model_hash=None,
                                  environment=(('python', platform.python_version()), ('numpy', np.__version__),
                                               ('rasterio', rasterio.__version__), ('geopandas', gpd.__version__),
                                               ('shapely', shapely.__version__), ('geos', shapely.geos_version_string),
                                               ('pyproj', pyproj.__version__)),
                                  seed=None, parameter_count=None)
        artifacts = tuple(ArtifactRecord(path=name, sha256=file_hash(root / name), lineage=lineage,
                                          kind='exposure' if name.endswith('.parquet') else 'exposure_metadata')
                          for name in (filename, 'quality.json', 'artifact_manifest.json'))
        result = StageResult(request_hash=request.content_hash, status=status, artifacts=artifacts,
                             message='Exposure artifacts written' if status == 'pass' else
                                     'DEM coverage failed; missing population retained and affected estimates null')
        result.verify(request)
        return result
    except MissingPrerequisite as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(), message=str(exc))
    except (ContractError, OSError, ValueError, KeyError, TypeError, tarfile.TarError,
            CRSError, RasterioError, GEOSException) as exc:
        return StageResult(request_hash=request.content_hash, status='fail', artifacts=(),
                           message=f'{type(exc).__name__}: {exc}')
    finally:
        if scratch is not None:
            shutil.rmtree(scratch)
