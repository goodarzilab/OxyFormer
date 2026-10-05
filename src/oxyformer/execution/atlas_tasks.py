"""Build concrete atlas tasks from committed inventories and inspected DEM headers.

Only headers and paired XML metadata are read by offset in an uncompressed tar.
Full payload integrity belongs to runner acquisition admission. No pixel reads,
extraction, downloads, exposure calculations, or approval changes occur here.
"""
from hashlib import sha256
import json
import math
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

from pyproj import CRS
import rasterio

from oxyformer.data.source_manifest import load_source, validate_shards
from oxyformer.exposure.population_allocation import PRIMARY, FALLBACK, FALLBACK_CELLS, _validate_vertical_crs
from oxyformer.provenance import canonical_json, require

ROOT = Path(__file__).resolve().parents[3]
TASK_FILE = 'configs/execution/tasks/atlas.json'
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
    receipt = json.loads((acquisition / 'receipts.json').read_text())
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
            raster_metadata={rid: inspected['raster_metadata'][rid] for rid in group['dem_resources']},
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
    # Compact tile rows keep a complete review practical; never sort needs,
    # whose file order defines StageRequest dependency indices.
    with Path(path).open('w') as stream:
        text = json.dumps(document, separators=(',', ':'), allow_nan=False)
        # One tile per line, plus task boundaries, without reordering mappings.
        import re
        text = re.sub(r',(?="n[0-9]{2}w[0-9]{3}":)', ',\n', text)
        text = text.replace('},{"schema_version":1,"review_status"', '},\n{"schema_version":1,"review_status"')
        stream.write(text + '\n')
