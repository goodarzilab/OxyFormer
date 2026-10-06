"""Publish the pinned public elevcan source from the merged US acquisition.

Only the selected small member is read by offset from the uncompressed payload;
no ACS member is extracted. The common runner verifies the outer acquisition.
"""
from hashlib import sha256
import json
from pathlib import Path
import tarfile

import yaml

from oxyformer.contracts import StageResult
from oxyformer.execution.integrity import open_regular, read_regular
from oxyformer.execution.paths import atomic_json, output_path
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, file_hash, require
from .historical import dependency
from .launcher import configuration

ROOT = Path(__file__).resolve().parents[3]
TASK_FILE = 'configs/execution/tasks/legacy.yaml'


def run_stage(request):
    request.verify_inputs()
    task = json.loads(Path(request.task_path).read_text())
    committed = yaml.safe_load((ROOT / TASK_FILE).read_text())
    expected = next(t for t in committed['tasks'] if t['id'] == 'legacy-inputs')
    require(task == expected and request.stage == 'legacy-inputs',
            'legacy input task differs from reviewed configuration')
    payload = dependency(request, {'dependency': 'fetch-us', 'path': 'payload.tar'})
    receipt_path = dependency(request, {'dependency': 'fetch-us', 'path': 'receipts.json'})
    receipt = json.loads(read_regular(receipt_path))
    require(receipt.get('status') == 'complete' and receipt.get('manifest_id') == 'us',
            'complete US acquisition required')
    pin = configuration()['sources']['elevcan-reproduction']
    entries = [r for r in receipt['resources'] if r['id'] == 'elevcan_source']
    require(len(entries) == 1, 'unique elevcan receipt required')
    entry = entries[0]
    name = 'us/elevcan_source.tar.gz'
    require(entry['destination'] == name and entry['sha256'] == pin['archive_sha256']
            and entry['bytes'] == pin['archive_bytes'] and entry['url'] == pin['archive_url'],
            'elevcan receipt differs from pinned source')
    with open_regular(payload) as stream:
        with tarfile.open(fileobj=stream, mode='r:') as archive:
            matches = [m for m in archive if m.name == name]
            require(len(matches) == 1 and matches[0].isfile(), 'unique regular elevcan member required')
            member = matches[0]
            require(member.size == pin['archive_bytes'], 'elevcan archive size mismatch')
            stream.seek(member.offset_data)
            raw = stream.read(member.size)
    require(len(raw) == pin['archive_bytes'] and sha256(raw).hexdigest() == pin['archive_sha256'],
            'elevcan archive SHA-256 mismatch')
    out = Path(request.output_dir)
    with output_path(out, 'elevcan_source.tar.gz').open('xb') as sink:
        sink.write(raw)
    atomic_json(out, 'source_manifest.json', {
        'schema_version': 1, 'source': pin, 'acquisition_receipt_sha256': file_hash(receipt_path),
        'payload_sha256': receipt['payload_sha256'], 'member': name, 'offset': member.offset_data,
        'bytes': len(raw), 'sha256': pin['archive_sha256'],
    })
    lineage = ArtifactLineage(source_hashes=request.dependency_hashes, unit_ids=(task['id'],),
        parent_hashes=(request.content_hash,), split_hash=None, config_hash=request.config_hash,
        model_hash=None, environment=(('code_identity', request.code_identity),), seed=None, parameter_count=None)
    return StageResult(request_hash=request.content_hash, status='pass',
        artifacts=tuple(ArtifactRecord(path=name, sha256=file_hash(out / name), lineage=lineage,
            kind='legacy_input') for name in task['outputs']), message='Pinned elevcan archive published')
