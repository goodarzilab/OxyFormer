"""Deterministic physical exposure and verified shard collection.

Public in-memory API: build_exposure(ExposureSources, block GeoDataFrame,
AllocationSpec) -> (tract/scenario DataFrame, QC dict). Only the four canonical
block columns are accepted. No outcome, learned-weight, or tensor channel exists.

Stage task JSON v1 (all indices address request.dependency_paths):
  review_status: reviewed; review_id: nonempty review reference
  shard_manifest: index of an atlas_shards JSON (groups/id/jurisdictions/dem_resources)
  exposure-atlas additionally: shard_id; census: {receipt: i, payload: j};
    dem: {receipt: i, payload: j}; raster_metadata: {resource_id: {crs, nodata,
    vertical_unit, vertical_datum, product, fallback_reason}}.
  atlas-collect additionally: shards: [{manifest: i, exposure: j, quality: k}].
The exposure config is configs/exposure.yaml. This is the stage's new task
schema, not an alternate StageRequest. Receipts/payloads are those produced by
merged data.source_manifest.fetch_manifest. Every upstream file is an explicit,
hash-verified dependency; filenames adjacent to dependencies are never inferred.
A reviewed task supplies inspected tile metadata, including approved fallback
identities in the eight missing cells. Acquisition readiness is owned upstream.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import tempfile
import numpy as np
import pandas as pd
import rasterio
import yaml
from oxyformer.contracts import StageRequest, StageResult
from oxyformer.data.source_manifest import load_source
from oxyformer.exposure.archives import extract_member
from oxyformer.exposure.census_blocks import read_census_blocks, validate_blocks
from oxyformer.exposure.physics import PHYSICS, pressure_mmhg, oxygen_deficit_mmhg, validate_owner_approval
from oxyformer.exposure.population_allocation import AllocationSpec, DemTile, RasterSampler, placement_batches, FALLBACK
from oxyformer.exposure.quality import QUANTILE_INTERPRETATION, REASONS, validate_accounting, placement_quantiles
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, canonical_json, check_hash, file_hash, require


@dataclass(frozen=True)
class ExposureSources:
    # Stable global acquisition identities, equal across shards of one atlas.
    identities: tuple[tuple[str, str], ...]
    dem_tiles: tuple[DemTile, ...]

    def identity_map(self):
        require(bool(self.identities) and len(dict(self.identities)) == len(self.identities),
                'source identities missing or duplicated')
        for key, digest in self.identities:
            require(isinstance(key, str) and bool(key), 'source identity name missing')
            check_hash(digest)
        return dict(sorted(self.identities))


def build_exposure(source_manifests, geography, allocation_spec):
    """Transform each placement, then weight by fixed Census population.

    Incomplete tracts retain total/covered/missing population and null exposures
    and quantiles. Zero-population blocks remain in QC and never define inhabited
    quantiles. Each scenario has its own mass ledger; scenarios are not pooled.
    """
    require(type(source_manifests) is ExposureSources and type(allocation_spec) is AllocationSpec,
            'expected ExposureSources and AllocationSpec')
    blocks = validate_blocks(geography).to_crs(allocation_spec.placement_crs)
    require(blocks.geometry.is_valid.all(), 'invalid reprojected geometry')
    quality = {'schema_version': 1, 'physical_spec': asdict(PHYSICS),
               'physical_hash': PHYSICS.content_hash, 'allocation': asdict(allocation_spec),
               'allocation_hash': allocation_spec.content_hash,
               'source_identities': source_manifests.identity_map(),
               'quantile_interpretation': QUANTILE_INTERPRETATION,
               'block_count': len(blocks), 'population': sum(int(value) for value in blocks.population), 'blocks': []}
    rows = []
    with RasterSampler(source_manifests.dem_tiles, allocation_spec.placement_crs) as sampler:
        quality['dem_tiles'] = sampler.identities
        for tract_id, tract in blocks.groupby('tract_id', sort=True, observed=True):
            for scenario in allocation_spec.scenarios:
                pressures, deficits, quantile_blocks = [], [], []
                missing_mass = []
                for block in tract.itertuples():
                    pop = int(block.population)
                    ledger = dict(block_id=block.block_id, tract_id=tract_id, scenario=scenario,
                                  population=pop, covered_population=0.0, **{r: 0.0 for r in REASONS})
                    areas_total, weights_total, elevations, placement_areas = [], [], [], []
                    # Uninhabited blocks are retained but need no DEM query.
                    if pop:
                        full_area = block.geometry.area
                        require(math.isfinite(full_area) and full_area > 0, 'block has invalid projected area')
                        # Retain the exact ratio until each finite placement mass
                        # is formed; an intermediate density can overflow.
                        mass_scale = Fraction(pop) / Fraction(full_area)
                        for xy, areas in placement_batches(block.geometry, scenario, allocation_spec):
                            areas_total.append(float(math.fsum(areas)))
                            z, reason = sampler.sample(xy)
                            weights = (np.array([float(pop)]) if scenario == 'centroid' else
                                       np.fromiter((float(mass_scale * Fraction(float(area))) for area in areas),
                                                   dtype=float, count=len(areas)))
                            require(np.isfinite(weights).all() and (weights > 0).all(),
                                    'positive placement mass is zero or nonfinite')
                            weights_total.append(float(math.fsum(weights)))
                            valid = np.isfinite(z)
                            for name in REASONS:
                                ledger[name] += float(math.fsum(weights[reason == name]))
                            if valid.any():
                                # REVIEW/MUTATION TARGET: transform BEFORE population aggregation.
                                pressures.append(float(np.dot(weights[valid], pressure_mmhg(z[valid]))))
                                deficits.append(float(np.dot(weights[valid], oxygen_deficit_mmhg(z[valid]))))
                                elevations.append(z[valid])
                                placement_areas.append(areas[valid])
                        require(math.isclose(math.fsum(areas_total), full_area, rel_tol=1e-10, abs_tol=1e-10),
                                'placement does not conserve block area')
                        require(math.isclose(math.fsum(weights_total), pop, rel_tol=1e-10, abs_tol=1e-8),
                                'placement does not conserve block population')
                        if elevations:
                            quantile_blocks.append((pop, full_area, np.concatenate(elevations),
                                                    np.concatenate(placement_areas)))
                    missing = math.fsum(ledger[r] for r in REASONS)
                    ledger['covered_population'] = max(0.0, pop - missing)
                    missing_mass.append(missing)
                    quality['blocks'].append(ledger)
                # Counts are validated integral; accumulate as Python integers
                # so neither float32 rounding nor fixed-width overflow changes mass.
                total = sum(int(value) for value in tract.population)
                missing = math.fsum(missing_mass)
                complete = total > 0 and missing == 0
                quantiles = placement_quantiles(quantile_blocks) if complete else [None] * 3
                rows.append(dict(tract_id=tract_id, county_id=tract_id[:5], scenario=scenario,
                    block_count=len(tract), population=total, covered_population=max(0.0, total - missing),
                    missing_population=missing, status='complete' if complete else ('zero_population' if total == 0 else 'missing_dem'),
                    pressure_mmhg=math.fsum(pressures) / total if complete else np.nan,
                    oxygen_deficit_mmhg=math.fsum(deficits) / total if complete else np.nan,
                    elevation_p10_m=quantiles[0], elevation_p50_m=quantiles[1], elevation_p90_m=quantiles[2],
                    quantile_interpretation=QUANTILE_INTERPRETATION))
    exposure = pd.DataFrame(rows).sort_values(['tract_id', 'scenario']).reset_index(drop=True)
    for col in ('elevation_p10_m', 'elevation_p50_m', 'elevation_p90_m'):
        exposure[col] = pd.to_numeric(exposure[col], errors='raise').astype(float)
    quality['blocks'].sort(key=lambda r: (r['block_id'], r['scenario']))
    validate_accounting(exposure, quality)
    return exposure, quality


def _json(path):
    return json.loads(Path(path).read_text())


def _dependency(request, index):
    require(type(index) is int and 0 <= index < len(request.dependency_paths), 'invalid dependency index')
    return Path(request.dependency_paths[index])


def _receipt(request, binding):
    receipt_path = _dependency(request, binding['receipt'])
    payload = _dependency(request, binding['payload'])
    receipt = _json(receipt_path)
    require(receipt['status'] == 'complete', 'upstream acquisition incomplete')
    require(receipt['payload_sha256'] == request.dependency_hashes[binding['payload']], 'payload/receipt identity mismatch')
    require(len({r['id'] for r in receipt['resources']}) == len(receipt['resources']), 'duplicate receipt resource')
    return receipt, payload


def _inventory(task, request):
    manifest = _json(_dependency(request, task['shard_manifest']))
    require(manifest['schema_version'] == 1 and manifest['kind'] == 'atlas_shards' and
            bool(manifest.get('approval_reference')), 'reviewed shard manifest required')
    groups = manifest['groups']
    require(groups and len({g['id'] for g in groups}) == len(groups), 'empty/duplicate shard groups')
    jurisdictions = [s for g in groups for s in g['jurisdictions']]
    require(len(set(jurisdictions)) == len(jurisdictions), 'overlapping shard jurisdictions')
    return {g['id']: g for g in groups}


def _merge_quality(parts):
    require(bool(parts), 'no exposure parts')
    result = dict(parts[0])
    for part in parts[1:]:
        for key in ('physical_spec', 'physical_hash', 'allocation', 'allocation_hash',
                    'source_identities', 'quantile_interpretation'):
            require(part[key] == result[key], f'inconsistent {key}')
    result['blocks'] = sorted([row for p in parts for row in p['blocks']], key=lambda r: (r['block_id'], r['scenario']))
    result['population'] = sum(p['population'] for p in parts)
    result['block_count'] = sum(p['block_count'] for p in parts)
    tiles = {}
    for part in parts:
        for tile in part['dem_tiles']:
            old = tiles.setdefault(tile['resource_id'], tile)
            require(old == tile, 'changed DEM source identity')
    result['dem_tiles'] = [tiles[k] for k in sorted(tiles)]
    return result


def _build_shard(request, task, config, groups, output):
    group = groups[task['shard_id']]
    census, census_tar = _receipt(request, task['census'])
    dem, dem_tar = _receipt(request, task['dem'])
    require(census['manifest_id'] == 'census' and dem['manifest_id'] == 'dem', 'acquisition source mismatch')
    dem_source = load_source('dem')
    require(dem['manifest_sha256'] == sha256(canonical_json(dem_source).encode()).hexdigest(),
            'DEM receipt differs from reviewed source configuration')
    source = load_source(config['census_source'])
    require(census['manifest_sha256'] == sha256(canonical_json(source).encode()).hexdigest(),
            'Census receipt differs from reviewed source configuration')
    resources = {r['id']: r for r in dem['resources']}
    tile_ids = sorted(set(group['dem_resources']) | set(group.get('missing_tiles', [])))
    require(set(task['raster_metadata']) == set(tile_ids), 'tile metadata omissions or unexpected tiles')
    tiles = []
    for rid in tile_ids:
        resource = resources[rid]
        tiles.append(DemTile(resource_id=rid, path=str(dem_tar), sha256=resource['sha256'],
                             archive_member=resource['destination'], **task['raster_metadata'][rid]))
    identities = tuple((label + '_' + role, request.dependency_hashes[task[label][role]])
                       for label in ('census', 'dem') for role in ('receipt', 'payload'))
    sources = ExposureSources(identities=identities, dem_tiles=tuple(tiles))
    allocation = AllocationSpec(**config['allocation'])
    frames, qualities = [], []
    census_resources = {r['id']: r for r in census['resources']}
    for state in sorted(group['jurisdictions']):
        fips = config['state_fips'][state]
        # One state's two selected ZIPs, removed before moving to the next state.
        with tempfile.TemporaryDirectory(prefix='.census-', dir=output) as scratch:
            paths = []
            for rid in ('blocks_' + fips, 'sf1_' + state.lower()):
                resource = census_resources[rid]
                path = Path(scratch) / (rid + '.zip')
                extract_member(census_tar, resource['destination'], path, resource['sha256'])
                paths.append(path)
            blocks = read_census_blocks(*paths, state_abbreviation=state, state_fips=fips, source_config=source)
            frame, quality = build_exposure(sources, blocks, allocation)
            frames.append(frame)
            qualities.append(quality)
    return pd.concat(frames, ignore_index=True), _merge_quality(qualities), [task['shard_id']]


def _collect(request, task, config, groups):
    frames, qualities, seen = [], [], set()
    spec = AllocationSpec(**config['allocation'])
    for binding in task['shards']:
        manifest = _json(_dependency(request, binding['manifest']))
        require(manifest['kind'] == 'exposure-atlas' and manifest['status'] == 'pass', 'shard did not pass')
        require(isinstance(manifest['shard_ids'], list) and len(manifest['shard_ids']) == 1,
                'shard manifest requires exactly one shard ID')
        sid = manifest['shard_ids'][0]
        require(manifest['shard_ids'] == [sid] and sid in groups and sid not in seen, 'overlapping or unexpected shards')
        seen.add(sid)
        require(manifest['shard_manifest_hash'] == request.dependency_hashes[task['shard_manifest']], 'shard inventory changed')
        for name, role in (('exposure.parquet', 'exposure'), ('quality.json', 'quality')):
            _dependency(request, binding[role])
            require(manifest['files'][name] == request.dependency_hashes[binding[role]], 'shard artifact hash mismatch')
        quality = _json(_dependency(request, binding['quality']))
        frame = pd.read_parquet(_dependency(request, binding['exposure']))
        require(quality['physical_spec'] == asdict(PHYSICS) and quality['physical_hash'] == PHYSICS.content_hash == manifest['physical_hash'],
                'inconsistent physical specification')
        require(quality['allocation_hash'] == spec.content_hash == manifest['allocation_hash'] and
                canonical_json(quality['allocation']) == canonical_json(asdict(spec)), 'inconsistent allocation specification')
        expected_states = {config['state_fips'][s] for s in groups[sid]['jurisdictions']}
        require({r['block_id'][:2] for r in quality['blocks']} == expected_states, 'unexpected jurisdiction omissions')
        require(manifest['source_identities'] == quality['source_identities'], 'shard source identity mismatch')
        validate_accounting(frame, quality)
        frames.append(frame)
        qualities.append(quality)
    require(seen == set(groups), 'missing shards: ' + ','.join(sorted(set(groups) - seen)))
    exposure = pd.concat(frames, ignore_index=True).sort_values(['tract_id', 'scenario']).reset_index(drop=True)
    quality = _merge_quality(qualities)
    validate_accounting(exposure, quality)
    return exposure, quality, sorted(seen)


def _exception_message(exc):
    """Bound diagnostics without relying on a backend exception's formatter."""
    prefix = f'{type(exc).__module__}.{type(exc).__qualname__}: '
    marker = ' ...[truncated]'
    limit = 4096
    if len(prefix) > limit - len(marker):
        return prefix[:limit - len(marker)] + marker
    try:
        message = str(exc)
    except Exception:
        message = '<exception message unavailable>'
    available = limit - len(prefix)
    if len(message) > available:
        message = message[:available - len(marker)] + marker
    return prefix + message


def run_stage(request: StageRequest) -> StageResult:
    """Create shard/atlas files within a fresh or empty output_dir.

    Ordinary execution exceptions return fail with no declared artifacts;
    FileNotFoundError retains the existing blocked classification. Control-flow
    BaseExceptions propagate. Diagnostics retain the concrete type and original
    text subject to marked truncation; direct reader/build APIs still raise.
    Resource exceptions report execution failure, not scientific invalidity.

    Coverage failure writes and declares its accounting artifacts. An exception
    after publication can leave undeclared files: StageResult is authoritative
    for this invocation, and handlers never delete, overwrite, repair or retry.
    Native crashes or inability to construct a result are outside this boundary.
    """
    try:
        request.verify_inputs()
        require(request.stage in ('exposure-atlas', 'atlas-collect'), 'unsupported exposure stage')
        validate_owner_approval()
        config = yaml.safe_load(Path(request.config_path).read_text())
        require(config['schema_version'] == 1 and config['physical_version'] == PHYSICS.version,
                'inconsistent exposure configuration')
        require(config['approval_reference'] == 'configs/approvals.yaml:owner_decisions.exposure',
                'unexpected owner exposure approval reference')
        task = _json(request.task_path)
        require(task['schema_version'] == 1 and task['review_status'] == 'reviewed' and bool(task['review_id']),
                'reviewed task manifest required')
        groups = _inventory(task, request)
        output = Path(request.output_dir)
        require(not output.is_symlink(), 'output directory cannot be a symlink')
        output.mkdir(parents=True, exist_ok=True)
        require(not any(output.iterdir()), 'exposure requires an empty output directory')
        if request.stage == 'exposure-atlas':
            exposure, quality, shard_ids = _build_shard(request, task, config, groups, output)
            filename = 'exposure.parquet'
        else:
            exposure, quality, shard_ids = _collect(request, task, config, groups)
            filename = 'atlas.parquet'
        exposure = exposure.sort_values(['tract_id', 'scenario']).reset_index(drop=True)
        validate_accounting(exposure, quality)
        validate_owner_approval(use_fallback=any(t['product'] == FALLBACK for t in quality['dem_tiles']))
        status = 'fail' if (exposure.missing_population > 0).any() else 'pass'
        exposure.to_parquet(output / filename, index=False)
        (output / 'quality.json').write_text(canonical_json(quality))
        manifest = dict(schema_version=1, kind=request.stage, status=status, request_hash=request.content_hash,
            code_identity=request.code_identity, shard_ids=shard_ids,
            shard_manifest_hash=request.dependency_hashes[task['shard_manifest']],
            source_identities=quality['source_identities'], physical_hash=PHYSICS.content_hash,
            allocation_hash=quality['allocation_hash'],
            files={name: file_hash(output / name) for name in (filename, 'quality.json')})
        (output / 'artifact_manifest.json').write_text(canonical_json(manifest))
        lineage = ArtifactLineage(source_hashes=tuple(sorted(quality['source_identities'].values())),
            unit_ids=tuple(sorted(exposure.tract_id.unique())), parent_hashes=request.dependency_hashes + (request.task_hash,),
            split_hash=None, config_hash=request.config_hash, model_hash=None,
            environment=(('python', platform.python_version()), ('numpy', np.__version__),
                         ('pandas', pd.__version__), ('rasterio', rasterio.__version__),
                         ('code_identity', request.code_identity)), seed=None, parameter_count=None)
        result = StageResult(request_hash=request.content_hash, status=status,
            artifacts=tuple(ArtifactRecord(path=name, sha256=file_hash(output / name), lineage=lineage, kind=kind)
                            for name, kind in ((filename, 'exposure'), ('quality.json', 'quality'),
                                               ('artifact_manifest.json', 'artifact_manifest'))),
            message='Exposure population and omission accounting written' if status == 'pass' else 'Missing DEM population; exposure withheld')
        result.verify(request)
        return result
    except FileNotFoundError as exc:
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(), message=_exception_message(exc))
    except Exception as exc:
        return StageResult(request_hash=request.content_hash, status='fail', artifacts=(), message=_exception_message(exc))
