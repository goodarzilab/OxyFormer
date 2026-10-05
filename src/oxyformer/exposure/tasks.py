"""Build concrete atlas tasks from committed inventories and inspected DEM headers.

Raster metadata is read from headers by offset and paired XML in an uncompressed
tar. Member bytes are streamed for digest verification without decoding pixels
or extracting files. The stage adapter delegates science to the unchanged API.
"""
from dataclasses import replace
import shutil
import tempfile
from hashlib import sha256
import json
import math
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

from pyproj import CRS
import rasterio
import yaml

from oxyformer.data.source_manifest import load_source, validate_shards
from oxyformer.exposure.population_allocation import PRIMARY, FALLBACK, FALLBACK_CELLS, _validate_vertical_crs
from oxyformer.contracts import StageResult
from oxyformer.execution.paths import atomic_json, atomic_write
from oxyformer.execution.integrity import read_regular
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, canonical_json, file_hash, require

ROOT = Path(__file__).resolve().parents[3]
TASK_FILE = 'configs/execution/tasks/atlas.yaml'
DIVISIONS = {
    'new_england': 'new-england', 'middle_atlantic': 'mid-atlantic',
    'east_north_central': 'east-north', 'west_north_central': 'west-north',
    'south_atlantic': 'south-atlantic', 'east_south_central': 'east-south',
    'west_south_central': 'west-south', 'mountain': 'mountain', 'pacific': 'pacific',
}
SHARD_OUTPUTS = ['exposure.parquet', 'quality.json', 'artifact_manifest.json']


def inspect_dem(acquisition, source=None):
    source = load_source('dem') if source is None else source
    acquisition = Path(acquisition)
    receipt = json.loads(read_regular(acquisition / 'receipts.json'))
    require(receipt.get('status') == 'complete' and receipt.get('manifest_id') == 'dem', 'DEM acquisition incomplete or wrong source')
    require(receipt['manifest_sha256'] == sha256(canonical_json(source).encode()).hexdigest(), 'DEM receipt differs from dem.json')
    resources = {r['id']: r for r in source['resources']}
    acquired = {r['id']: r for r in receipt['resources']}
    require(len(acquired) == len(receipt['resources']) and acquired.keys() == resources.keys(), 'DEM receipt resource inventory mismatch')
    payload = acquisition / 'payload.tar'
    require(payload.stat().st_size == receipt['payload_bytes'], 'DEM payload size mismatch')
    metadata = {}
    with tarfile.open(payload, 'r:') as archive, rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_PAM_ENABLED=False):
        members = {}
        for member in archive:
            require(member.isfile() and member.name not in members, 'duplicate or nonregular DEM member')
            members[member.name] = member
        require(set(members) == {r['destination'] for r in acquired.values()}, 'DEM archive inventory mismatch')
        for rid, declared in resources.items():
            entry = acquired[rid]
            require(entry['destination'] == declared['destination'], f'{rid}: destination differs from dem.json')
            member = members[entry['destination']]
            require(member.size == entry['bytes'] and member.size <= declared['max_bytes'], f'{rid}: size mismatch')
            require(declared.get('expected_bytes', member.size) == member.size, f'{rid}: size differs from dem.json')
            require(declared.get('expected_sha256', entry['sha256']) == entry['sha256'], f'{rid}: digest differs from dem.json')
            # The outer payload hash alone cannot detect a producer that packed
            # changed bytes after recording the individual resource digest.
            digest = sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == entry['sha256'], f'{rid}: resource digest mismatch')
            if declared['format'] != 'tiff':
                continue
            paired = resources[rid + '_metadata']
            paired_entry = acquired[rid + '_metadata']
            xml_bytes = archive.extractfile(members[paired['destination']]).read()
            require(sha256(xml_bytes).hexdigest() == paired_entry['sha256'], f'{rid}: XML digest mismatch')
            xml = ET.fromstring(xml_bytes)
            title = xml.findtext('.//citeinfo/title', '')
            product = declared['product']
            require(product in (PRIMARY, FALLBACK) and paired['product'] == product, f'{rid}: unapproved product')
            resolution = '1/3' if product == PRIMARY else '1'
            require(f'usgs {resolution} arc second {rid} ' in title.lower().replace('-', ' '), f'{rid}: XML product mismatch')
            require(xml.findtext('.//altdatum') == 'North American Vertical Datum of 1988', f'{rid}: XML vertical datum mismatch')
            require(xml.findtext('.//altunits') == source['release_identity']['elevation_units'] == 'meters', f'{rid}: XML vertical unit mismatch')
            require(source['release_identity']['vertical_datum'] == 'NAVD88', 'dem.json vertical datum mismatch')
            require(source['release_identity']['horizontal_datum'] == 'NAD83', 'dem.json horizontal datum mismatch')
            require(xml.findtext('.//horizdn') == 'North American Datum of 1983', f'{rid}: XML horizontal datum mismatch')
            # GDAL sees just this member as a seekable file. It cannot discover
            # undeclared sidecars and never needs to scan/extract the raster.
            uri = f'/vsisubfile/{member.offset_data}_{member.size},{payload.resolve()}'
            with rasterio.open(uri) as ds:
                require(ds.count == 1 and ds.crs is not None, f'{rid}: raster bands/CRS mismatch')
                crs = CRS(ds.crs)
                _validate_vertical_crs(crs)
                require(crs.to_2d() == CRS('EPSG:4269'), f'{rid}: raster CRS differs from dem.json')
                require(ds.units[0] in (None, 'm', 'meter', 'metre') and ds.scales == (1.0,) and ds.offsets == (0.0,), f'{rid}: raster vertical units/scale mismatch')
                expected_resolution = (1 / 3 if product == PRIMARY else 1) / 3600
                # n47w090 stores 0.0000925926 degrees (8e-8 relative rounding).
                require(all(math.isclose(r, expected_resolution, rel_tol=1e-7) for r in ds.res), f'{rid}: raster product resolution mismatch')
                require(ds.nodata is None or math.isfinite(ds.nodata), f'{rid}: nonfinite nodata unsupported by JSON')
                require(product != FALLBACK or rid in FALLBACK_CELLS and rid in source['coverage']['fallback_tiles'], f'{rid}: fallback not approved')
                metadata[rid] = dict(crs=crs.to_string(), nodata=ds.nodata, vertical_unit='m',
                    vertical_datum='NAVD88', product=product,
                    fallback_reason=('Owner-approved one arc-second fallback: no one-third arc-second product for this cell.' if product == FALLBACK else None))
    no_product = source['coverage']['no_product_cells']
    require(not (metadata.keys() & no_product.keys()), 'no-product cell has a raster')
    return {'schema_version': 1, 'dem_manifest_sha256': receipt['manifest_sha256'],
            'dem_payload_sha256': receipt['payload_sha256'], 'raster_metadata': metadata,
            'no_product_cells': no_product}


def build_tasks(inspected, atlas=None):
    source = load_source('dem')
    atlas = json.loads((ROOT / 'configs/sources/atlas_shards.json').read_text()) if atlas is None else atlas
    validate_shards(atlas, source)
    require(inspected['dem_manifest_sha256'] == sha256(canonical_json(source).encode()).hexdigest(), 'inspected metadata differs from dem.json')
    expected = {r['id']: r for r in source['resources'] if r['format'] == 'tiff'}
    require(inspected['raster_metadata'].keys() == expected.keys(), 'inspected raster inventory mismatch')
    require(inspected['no_product_cells'] == source['coverage']['no_product_cells'], 'no-product evidence differs from dem.json')
    for rid, tile in inspected['raster_metadata'].items():
        require(tile['product'] == expected[rid]['product'] and tile['vertical_unit'] == 'm' and
                tile['vertical_datum'] == source['release_identity']['vertical_datum'] and
                CRS(tile['crs']).to_2d() == CRS('EPSG:4269'), f'{rid}: inspected metadata mismatch')
        require(bool(tile['fallback_reason']) == (tile['product'] == FALLBACK), f'{rid}: fallback reason mismatch')
    # Intern equal header records so YAML anchors avoid duplicating 957 identical
    # CRS/unit/nodata declarations in the reviewed task source.
    profiles = {}
    raster_metadata = {rid: profiles.setdefault(canonical_json(tile), tile)
                       for rid, tile in inspected['raster_metadata'].items()}
    reviewed = dict(schema_version=1, review_status='reviewed', review_id='atlas-tasks-implementation-review')
    tasks = [dict(reviewed, id='atlas-inputs', stage='atlas-inputs',
        needs={'fetch-dem': ['payload.tar', 'receipts.json']},
        outputs=['atlas_shards.json', 'raster_metadata.json'])]
    for group in atlas['groups']:
        # Input stage contributes two files + three runner control files.
        needs = {'atlas-inputs': ['atlas_shards.json', 'raster_metadata.json'],
                 'fetch-census': ['payload.tar', 'receipts.json'], 'fetch-dem': ['payload.tar', 'receipts.json']}
        tasks.append(dict(reviewed, id='atlas-' + DIVISIONS[group['id']], stage='exposure-atlas',
            needs=needs, outputs=SHARD_OUTPUTS, shard_manifest=0, inspected_metadata=1,
            shard_id=group['id'], census={'payload': 5, 'receipt': 6}, dem={'payload': 7, 'receipt': 8},
            raster_metadata={rid: raster_metadata[rid] for rid in group['dem_resources']},
            no_product_cells=group.get('no_product_cells', [])))
    # Preserve this explicit order in the checked-in manifest. Each shard
    # contributes three outputs and three runner control files.
    needs = {'atlas-inputs': ['atlas_shards.json', 'raster_metadata.json']}
    bindings = []
    for group in atlas['groups']:
        index = 5 + 6 * len(bindings)
        needs['atlas-' + DIVISIONS[group['id']]] = SHARD_OUTPUTS
        bindings.append(dict(exposure=index, quality=index + 1, manifest=index + 2))
    tasks.append(dict(reviewed, id='atlas-collect', stage='atlas-collect', needs=needs,
        outputs=['atlas.parquet', 'quality.json', 'artifact_manifest.json'], shard_manifest=0, shards=bindings))
    return {'schema_version': 1, 'tasks': tasks}


def write_tasks(path, document):
    # Preserve needs order: it defines StageRequest dependency indices. YAML
    # anchors represent repeated *values*, not missing/uninspected tile records.
    with Path(path).open('w') as stream:
        yaml.safe_dump(document, stream, sort_keys=False)


def run_stage(request):
    request.verify_inputs()
    task = json.loads(Path(request.task_path).read_text())
    committed = yaml.safe_load((ROOT / TASK_FILE).read_text())
    expected = next((t for t in committed['tasks'] if t['id'] == task['id']), None)
    require(task == expected and task['stage'] == request.stage, 'atlas task differs from reviewed configuration')
    out = Path(request.output_dir)
    if request.stage == 'atlas-inputs':
        payload, receipt = map(Path, request.dependency_paths[:2])
        require(payload.name == 'payload.tar' and receipt.name == 'receipts.json' and payload.parent == receipt.parent,
                'atlas input dependency bindings mismatch')
        inspected = inspect_dem(payload.parent)
        require(build_tasks(inspected) == committed, 'actual DEM headers differ from reviewed atlas tasks')
        atomic_write(out, 'atlas_shards.json', (ROOT / 'configs/sources/atlas_shards.json').read_text())
        atomic_json(out, 'raster_metadata.json', inspected)
        lineage = ArtifactLineage(source_hashes=request.dependency_hashes, unit_ids=('atlas-inputs',),
            parent_hashes=(request.task_hash,), split_hash=None, config_hash=request.config_hash,
            model_hash=None, environment=(('code_identity', request.code_identity),), seed=None, parameter_count=None)
        return StageResult(request_hash=request.content_hash, status='pass', message='Inspected atlas inputs published',
            artifacts=tuple(ArtifactRecord(path=name, sha256=file_hash(out / name), lineage=lineage, kind='atlas_input')
                            for name in task['outputs']))
    require(request.stage in ('exposure-atlas', 'atlas-collect'), 'unexpected atlas stage')
    if request.stage == 'exposure-atlas':
        inspected = json.loads(Path(request.dependency_paths[task['inspected_metadata']]).read_text())
        require(task['raster_metadata'] == {rid: inspected['raster_metadata'][rid] for rid in task['raster_metadata']},
                'task metadata differs from inspected upstream headers')
        require(inspected['dem_payload_sha256'] == request.dependency_hashes[task['dem']['payload']],
                'inspected metadata belongs to a different DEM payload')
    # The science API owns an empty product directory and expects exposure.yaml,
    # whereas the runner owns _execution and supplies execution configuration.
    # Persist the derived request and configuration for audit, then rebind only
    # the returned request identity when moving its declared products outward.
    config = atomic_write(out, '_atlas/exposure.yaml', (ROOT / 'configs/exposure.yaml').read_text())
    with tempfile.TemporaryDirectory(prefix='products-', dir=out / '_atlas') as scratch:
        inner = replace(request, config_path=str(config), config_hash=file_hash(config), output_dir=scratch)
        atomic_write(out, '_atlas/request.json', inner.to_json())
        from oxyformer.exposure.build import run_stage as exposure_stage
        result = exposure_stage(inner)
        result.verify(inner)
        for artifact in result.artifacts:
            target = out / artifact.path
            require(not target.exists(), 'atlas output already exists')
            shutil.move(str(Path(scratch) / artifact.path), target)
        return replace(result, request_hash=request.content_hash)
