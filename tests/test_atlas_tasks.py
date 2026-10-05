"""Offline metadata, task bindings, admission, and science-adapter regressions."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.data.source_manifest import load_source
from oxyformer.exposure import tasks as atlas_stage
from oxyformer.exposure.tasks import ROOT, TASK_FILE, build_tasks, inspect_dem, PRIMARY, FALLBACK
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, canonical_json, file_hash

# These tests exercise the authorized runner change under the required unit
# command too; the complete execution suite remains independently runnable.
from test_execution import (runtime, acquisition, publication_authority,
    test_complete_acquisition_without_stage_result_is_accepted,
    test_acquisition_rejects_tampering_and_incomplete_receipt,
    test_acquisition_mutation_taints_direct_and_transitive_consumers,
    test_acquisition_cannot_be_rebaselined_after_receipt_and_payload_change,
    test_stage_dependency_without_receipt_is_rejected,
    test_task_cannot_exempt_an_incomplete_stage,
    test_acquisition_exemption_requires_declared_source_receipt)


def tasks():
    return yaml.safe_load((ROOT / TASK_FILE).read_text())


def inspected_fixture():
    source = load_source('dem')
    tiles = {}
    for task in tasks()['tasks']:
        tiles.update(task.get('raster_metadata', {}))
    return dict(schema_version=1, dem_manifest_sha256=sha256(canonical_json(source).encode()).hexdigest(),
        dem_payload_sha256='a' * 64, raster_metadata=tiles, no_product_cells=source['coverage']['no_product_cells'])


def test_committed_tasks_reproduce_and_cover_all_products():
    inspected = inspected_fixture()
    assert build_tasks(inspected) == tasks()
    assert len(inspected['raster_metadata']) == 957
    assert sum(t['product'] == FALLBACK for t in inspected['raster_metadata'].values()) == 3
    assert len(inspected['no_product_cells']) == 5
    assert not inspected['no_product_cells'].keys() & inspected['raster_metadata'].keys()


def test_every_registry_dependency_and_index_is_bound():
    registry = yaml.safe_load((ROOT / 'configs/execution/stages.yaml').read_text())['stages']
    for task in tasks()['tasks']:
        settings = registry[task['stage']]
        for unit, files in settings['needs'].items():
            assert set(files) <= set(task['needs'][unit])
        flattened = []
        for unit, files in task['needs'].items():
            flattened.extend((unit, f) for f in files)
            if unit not in settings.get('acquisition_receipts', {}):
                flattened.extend((unit, '_execution/' + f) for f in ['request.json', 'result.json', 'fingerprint.json'])
        if task['stage'] == 'atlas-inputs':
            continue
        assert flattened[task['shard_manifest']] == ('atlas-inputs', 'atlas_shards.json')
        if task['stage'] == 'exposure-atlas':
            for label in ('census', 'dem'):
                assert flattened[task[label]['receipt']] == ('fetch-' + label, 'receipts.json')
                assert flattened[task[label]['payload']] == ('fetch-' + label, 'payload.tar')
        else:
            assert len(task['shards']) == 9
            for binding in task['shards']:
                units = set()
                for role, filename in [('manifest', 'artifact_manifest.json'), ('exposure', 'exposure.parquet'), ('quality', 'quality.json')]:
                    unit, path = flattened[binding[role]]
                    assert path == filename
                    units.add(unit)
                assert len(units) == 1


@pytest.mark.parametrize('field,value', [('product', FALLBACK), ('crs', 'EPSG:4326'),
    ('vertical_unit', 'ft'), ('vertical_datum', 'EGM96'), ('fallback_reason', 'unapproved')])
def test_task_generation_refuses_metadata_mismatch(field, value):
    inspected = inspected_fixture()
    inspected['raster_metadata']['n25w081'][field] = value
    with pytest.raises(ContractError, match='mismatch'):
        build_tasks(inspected)


def synthetic_dem(tmp_path, *, nodata=-32767.0, crs='EPSG:4269', unit='meters', datum='North American Vertical Datum of 1988', product=PRIMARY):
    rid = 'n46w083' if product == FALLBACK else 'n25w081'
    resolution = 1 / 3600 if product == FALLBACK else 1 / 10800
    raster = tmp_path / 'fixture.tif'
    with rasterio.open(raster, 'w', driver='GTiff', width=2, height=2, count=1,
            dtype='float32', crs=crs, nodata=nodata, transform=from_origin(-83, 46, resolution, resolution)) as ds:
        ds.write(np.array([[1, 2], [3, nodata]], dtype='float32'), 1)
    xml = (f'<metadata><citeinfo><title>USGS {"1" if product == FALLBACK else "1/3"} Arc Second {rid} 20260101</title></citeinfo>'
           f'<altdatum>{datum}</altdatum><altunits>{unit}</altunits><horizdn>North American Datum of 1983</horizdn></metadata>').encode()
    entries = [('dem/' + rid + '.tif', raster.read_bytes(), rid, 'tiff'), ('dem/' + rid + '.xml', xml, rid + '_metadata', 'xml')]
    source = dict(id='dem', resources=[], release_identity={'vertical_datum': 'NAVD88', 'horizontal_datum': 'NAD83', 'elevation_units': 'meters'},
        coverage={'fallback_tiles': [rid] if product == FALLBACK else [], 'no_product_cells': {}})
    resources = []
    with tarfile.open(tmp_path / 'payload.tar', 'w') as tar:
        for name, data, key, fmt in entries:
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            resources.append(dict(id=key, destination=name, bytes=len(data), sha256=sha256(data).hexdigest()))
            source['resources'].append(dict(id=key, destination=name, max_bytes=len(data), expected_bytes=len(data), format=fmt, product=product))
    receipt = dict(status='complete', manifest_id='dem', manifest_sha256=sha256(canonical_json(source).encode()).hexdigest(),
        resources=resources, payload_bytes=(tmp_path / 'payload.tar').stat().st_size, payload_sha256=file_hash(tmp_path / 'payload.tar'))
    (tmp_path / 'receipts.json').write_text(json.dumps(receipt))
    return source, rid


@pytest.mark.parametrize('product', [PRIMARY, FALLBACK])
def test_inspection_reads_actual_headers_and_xml_without_extraction(tmp_path, monkeypatch, product):
    source, rid = synthetic_dem(tmp_path, product=product)
    def forbidden(*args, **kwargs):
        raise AssertionError('whole-archive extraction is forbidden')
    monkeypatch.setattr(tarfile.TarFile, 'extractall', forbidden)
    metadata = inspect_dem(tmp_path, source)['raster_metadata'][rid]
    assert metadata == dict(crs='EPSG:4269', nodata=-32767.0, vertical_unit='m', vertical_datum='NAVD88', product=product,
        fallback_reason='Owner-approved one arc-second fallback: no one-third arc-second product for this cell.' if product == FALLBACK else None)


@pytest.mark.parametrize('change', [{'crs': 'EPSG:4326'}, {'unit': 'feet'}, {'datum': 'EGM96'}])
def test_header_or_xml_mismatch_is_refused(tmp_path, change):
    source, _ = synthetic_dem(tmp_path, **change)
    with pytest.raises(ContractError, match='mismatch|differs'):
        inspect_dem(tmp_path, source)


def test_receipt_must_match_dem_json(tmp_path):
    source, _ = synthetic_dem(tmp_path)
    source['release_identity']['vertical_datum'] = 'EGM96'
    with pytest.raises(ContractError, match='differs from dem.json'):
        inspect_dem(tmp_path, source)


def test_adapter_supplies_science_config_and_empty_output(tmp_path, monkeypatch):
    task = next(t for t in tasks()['tasks'] if t['stage'] == 'atlas-collect')
    task_file = tmp_path / 'task.json'; task_file.write_text(json.dumps(task))
    config = tmp_path / 'config.json'; config.write_text('{}')
    out = tmp_path / 'attempt'; out.mkdir(); (out / 'code_commit.txt').write_text('existing runner controls')
    request = StageRequest(stage='atlas-collect', config_path=str(config), config_hash=file_hash(config),
        task_path=str(task_file), task_hash=file_hash(task_file), dependency_paths=(), dependency_hashes=(), output_dir=str(out), code_identity='a' * 40)
    def science(inner):
        assert not any(Path(inner.output_dir).iterdir())
        assert yaml.safe_load(Path(inner.config_path).read_text())['physical_version']
        assert inner.task_hash == request.task_hash
        lineage = ArtifactLineage(source_hashes=(request.task_hash,), unit_ids=('fixture',), parent_hashes=(),
            split_hash=None, config_hash=inner.config_hash, model_hash=None, environment=(('test', 'fixture'),), seed=None, parameter_count=None)
        artifacts = []
        for name in task['outputs']:
            (Path(inner.output_dir) / name).write_text('synthetic product')
            artifacts.append(ArtifactRecord(path=name, sha256=file_hash(Path(inner.output_dir) / name), lineage=lineage, kind='synthetic'))
        return StageResult(request_hash=inner.content_hash, status='pass', artifacts=tuple(artifacts), message='synthetic')
    monkeypatch.setattr('oxyformer.exposure.build.run_stage', science)
    result = atlas_stage.run_stage(request)
    assert result.status == 'pass'
    result.verify(request)
    assert set(task['outputs']) <= {p.name for p in out.iterdir()}


def test_all_eleven_tasks_admit_through_runner(tmp_path, monkeypatch):
    import runpy
    import shutil
    from test_execution import commit
    repo = tmp_path / 'repo'; repo.mkdir()
    shutil.copytree(ROOT / 'configs', repo / 'configs')
    import subprocess
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    commit(repo)
    monkeypatch.setattr(atlas_stage, 'ROOT', repo)
    roots = []
    for name in ('census', 'dem'):
        root = tmp_path / name; root.mkdir(); roots.append(root)
        (root / 'payload.tar').write_bytes(b'synthetic ' + name.encode())
        (root / 'receipts.json').write_text(json.dumps(dict(status='complete', manifest_id=name,
            payload_sha256=file_hash(root / 'payload.tar'), payload_bytes=(root / 'payload.tar').stat().st_size, resources=[])))
    inspected = inspected_fixture()
    inspected['dem_payload_sha256'] = file_hash(roots[1] / 'payload.tar')
    monkeypatch.setattr(atlas_stage, 'inspect_dem', lambda acquisition: deepcopy(inspected))
    admit = runpy.run_path(str(ROOT / 'scripts/build_atlas_tasks.py'))['admit']
    results = admit(repo, tmp_path / 'admission', *roots)
    assert len(results) == 11
    assert all(r['status'] == 'pass' for r in results)


def test_yaml_owner_dates_remain_json_serializable(tmp_path):
    from oxyformer.execution.runner import read_mapping
    path = tmp_path / 'approvals.yaml'
    path.write_text('schema_version: 1\napproved_on: 2026-10-04\n')
    assert json.loads(canonical_json(read_mapping(path)))['approved_on'] == '2026-10-04'
    # Do not mutate PyYAML globally; other consumers retain their old behavior.
    import datetime
    assert isinstance(yaml.safe_load(path.read_text())['approved_on'], datetime.date)


def test_admission_hash_cache_refuses_changed_payload(tmp_path):
    import runpy
    import os
    from oxyformer.execution import integrity
    frozen = runpy.run_path(str(ROOT / 'scripts/build_atlas_tasks.py'))['frozen_acquisitions']
    root = tmp_path / 'source'; root.mkdir()
    payload = root / 'payload.tar'; payload.write_bytes(b'original')
    before = payload.stat()
    expected = file_hash(payload)
    with pytest.raises(ContractError, match='frozen acquisition changed'):
        with frozen([root]):
            assert integrity.regular_file_hash(payload) == expected
            payload.write_bytes(b'mutation')
            os.utime(payload, ns=(before.st_atime_ns, before.st_mtime_ns))
            integrity.regular_file_hash(payload)


def test_inspection_rejects_stale_member_digest_with_current_payload_hash(tmp_path):
    source, rid = synthetic_dem(tmp_path)
    payload = tmp_path / 'payload.tar'
    with tarfile.open(payload, 'r:') as archive:
        member = archive.getmember('dem/' + rid + '.tif')
    raster = tmp_path / 'fixture.tif'
    with rasterio.open(raster, 'r+') as ds:
        values = ds.read(1)
        values[0, 0] = 1234
        ds.write(values, 1)
    changed = raster.read_bytes()
    assert len(changed) == member.size
    with payload.open('r+b') as stream:
        stream.seek(member.offset_data)
        stream.write(changed)
    receipt = json.loads((tmp_path / 'receipts.json').read_text())
    receipt['payload_sha256'] = file_hash(payload)
    (tmp_path / 'receipts.json').write_text(json.dumps(receipt))
    with pytest.raises(ContractError, match='resource digest mismatch'):
        inspect_dem(tmp_path, source)


@pytest.mark.parametrize('use_helper', [False, True])
def test_final_admission_hash_does_not_depend_on_fingerprint_implementation(tmp_path, monkeypatch, use_helper):
    """Exercise the reviewer-specified helper-backed fingerprint implementation.

    The current implementation hashes directly; both implementations have the
    same public contract and must receive an unpatched final verification.
    """
    import runpy
    from oxyformer.execution import integrity
    original = integrity.fingerprint_tree
    root = tmp_path / 'source'; root.mkdir()
    payload = root / 'payload.tar'; payload.write_bytes(b'original')
    def helper_backed_tree(path, **kwargs):
        tree = original(path, **kwargs)
        tree['payload.tar']['sha256'] = integrity.regular_file_hash(Path(path) / 'payload.tar')
        return tree
    if use_helper:
        monkeypatch.setattr(integrity, 'fingerprint_tree', helper_backed_tree)
    frozen = runpy.run_path(str(ROOT / 'scripts/build_atlas_tasks.py'))['frozen_acquisitions']
    context = frozen([root])
    with pytest.raises(ContractError, match='frozen acquisition changed'):
        with context as snapshots:
            payload.write_bytes(b'mutation')
            # Simulate a stat-invisible storage change, retaining the original
            # content tree but allowing current stat signatures through.
            signature = context.gen.gi_frame.f_locals['signature']
            snapshots[root] = (signature(root), snapshots[root][1])
