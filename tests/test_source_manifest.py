"""Synthetic, offline tests: never download production data in this suite."""
import copy
import hashlib
import io
import json
from pathlib import Path
import socket
import tarfile
import urllib.error
import urllib.request

import pytest

from oxyformer.data import source_manifest as sm


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('Tests must not contact a network')
    monkeypatch.setattr(socket, 'create_connection', refuse)
    monkeypatch.setattr(socket.socket, 'connect', refuse)


@pytest.fixture
def manifest():
    return {
        'schema_version': 1, 'kind': 'acquisition', 'id': 'synthetic',
        'status': 'ready', 'blockers': [], 'release_identity': {'name': 'synthetic v1'},
        'access_terms': {'rule': 'public_data_only', 'note': 'synthetic fixture'},
        'selection_provenance': {'method': 'synthetic; no real data'},
        'requirements': [{'id': 'all', 'status': 'ready', 'resource_ids': ['data', 'dictionary', 'terms']}],
        'resources': [
            {'id': name, 'url': 'https://example.test/' + name,
             'destination': name + '/content.txt', 'max_bytes': 100,
             'format': 'text', 'role': name, 'release': 'synthetic-v1',
             'availability': 'verified', 'inspection': 'synthetic fixture'}
            for name in ('data', 'dictionary', 'terms')],
    }


class Response(io.BytesIO):
    status = 200

    def __init__(self, data=b'synthetic bytes', headers=None, url='https://example.test/final'):
        super().__init__(data)
        self.headers = headers or {}
        self.url = url
        self.read_sizes = []

    def geturl(self):
        return self.url

    def read(self, size=-1):
        assert 0 < size <= sm.CHUNK
        self.read_sizes.append(size)
        return super().read(size)


def fetch(manifest, tmp_path, **kwargs):
    return sm.fetch_manifest(manifest, tmp_path / 'result', attempt_root=tmp_path, **kwargs)


def test_blocked_manifest_cannot_download(manifest, tmp_path, monkeypatch):
    manifest.update(status='blocked', blockers=['Required release unavailable'])
    calls = []
    def network(*args):
        calls.append(args)
        return Response()
    monkeypatch.setattr(sm, '_open_url', network)
    with pytest.raises(sm.ManifestError, match='Blocked manifest'):
        fetch(manifest, tmp_path)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_streamed_hashes_and_tar_receipts(manifest, tmp_path, monkeypatch):
    payload = b'synthetic\n' * 200000
    digest = hashlib.sha256(payload).hexdigest()
    responses = []
    for resource in manifest['resources']:
        resource.update(max_bytes=len(payload), expected_sha256=digest)
    def network(*args):
        response = Response(payload, {'Content-Length': str(len(payload))})
        responses.append(response)
        return response
    monkeypatch.setattr(sm, '_open_url', network)
    result = fetch(manifest, tmp_path)
    out = tmp_path / 'result'
    assert {p.name for p in out.iterdir()} == {'payload.tar', 'receipts.json', 'download.log'}
    assert out.stat().st_mode & 0o777 == 0o700
    assert result['status'] == 'complete'
    assert result == json.loads((out / 'receipts.json').read_text())
    assert result['payload_sha256'] == hashlib.sha256((out / 'payload.tar').read_bytes()).hexdigest()
    with tarfile.open(out / 'payload.tar') as archive:
        assert archive.getnames() == [r['destination'] for r in manifest['resources']]
        for entry, member in zip(result['resources'], archive.getmembers()):
            assert member.isfile() and member.mode == 0o600
            assert archive.extractfile(member).read() == payload
            assert entry['sha256'] == digest and entry['bytes'] == len(payload)
            assert entry['checksum_status'] == 'matched'
            assert entry['final_url'] == 'https://example.test/final'
    assert all(len(r.read_sizes) > 2 for r in responses)


def test_optional_checksum_is_recorded_not_verified(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response())
    result = fetch(manifest, tmp_path)
    assert all(r['checksum_status'] == 'recorded' and r['expected_sha256'] is None
               for r in result['resources'])


@pytest.mark.parametrize('destination', ['../escape', '/absolute', 'x/../../escape',
                                        'x//y', 'x/./y', 'C:/data', 'x\\y', 'x/'])
def test_traversal_rejected_before_writes(manifest, tmp_path, destination):
    manifest['resources'][0]['destination'] = destination
    with pytest.raises(sm.ManifestError):
        fetch(manifest, tmp_path)
    assert not (tmp_path / 'result').exists()


@pytest.mark.parametrize('url', ['http://example.test/a', 'file:///tmp/a',
                                'https://user:pass@example.test/a', 'https://example.test:80/a'])
def test_non_https_or_credentials_rejected(manifest, url):
    manifest['resources'][0]['url'] = url
    with pytest.raises(sm.ManifestError):
        sm.validate_manifest(manifest)


def test_redirect_to_http_rejected():
    handler = sm.HTTPSRedirectHandler()
    req = urllib.request.Request('https://example.test/file')
    with pytest.raises(sm.ManifestError):
        handler.redirect_request(req, None, 302, 'Moved', {}, 'http://example.test/file')


def test_duplicate_and_prefix_destinations_rejected(manifest):
    for path in ('data/content.txt', 'data', 'data/content.txt/child'):
        changed = copy.deepcopy(manifest)
        changed['resources'][1]['destination'] = path
        with pytest.raises(sm.ManifestError, match='Conflicting'):
            sm.validate_manifest(changed)


@pytest.mark.parametrize('change', [
    {'max_bytes': 0}, {'max_bytes': True}, {'max_bytes': -1},
    {'expected_sha256': 'not-a-hash'}, {'availability': 'unavailable'},
])
def test_resource_contract_invalid(manifest, change):
    manifest['resources'][0].update(change)
    with pytest.raises(sm.ManifestError):
        sm.validate_manifest(manifest)


def test_missing_required_resource_blocks_ready(manifest):
    manifest['requirements'].append({'id': 'missing', 'status': 'blocked',
                                     'resource_ids': [], 'reason': 'Not published'})
    with pytest.raises(sm.ManifestError, match='Unavailable required'):
        sm.validate_manifest(manifest)


@pytest.mark.parametrize('case', ['too_large_header', 'too_large_stream', 'empty', 'hash', 'html', 'zip'])
def test_failed_download_never_publishes_tar(manifest, tmp_path, monkeypatch, case):
    data, headers = b'synthetic bytes', {}
    if case == 'too_large_header': headers = {'Content-Length': '101'}
    if case == 'too_large_stream': data = b'x' * 101
    if case == 'empty': data = b''
    if case == 'hash': manifest['resources'][0]['expected_sha256'] = '0' * 64
    if case == 'html': data = b'<!DOCTYPE html><html>login</html>'
    if case == 'zip': manifest['resources'][0]['format'] = 'zip'
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response(data, headers))
    with pytest.raises(sm.ManifestError):
        fetch(manifest, tmp_path)
    out = tmp_path / 'result'
    assert {p.name for p in out.iterdir()} == {'receipts.json', 'download.log'}
    assert json.loads((out / 'receipts.json').read_text())['status'] == 'failed'


def test_retry_discards_partial_bytes(manifest, tmp_path, monkeypatch):
    calls, sleeps = [], []
    def network(*args):
        calls.append(args)
        if len(calls) == 1:
            return Response(b'partial', {'Content-Length': '20'})
        return Response(b'complete')
    monkeypatch.setattr(sm, '_open_url', network)
    monkeypatch.setattr(sm.time, 'sleep', sleeps.append)
    result = fetch(manifest, tmp_path, attempts=2)
    assert len(calls) == 4 and sleeps == [1]
    assert result['resources'][0]['attempts'] == 2
    assert result['resources'][0]['sha256'] == hashlib.sha256(b'complete').hexdigest()


@pytest.mark.parametrize('code,expected_calls', [(503, 3), (429, 3), (403, 1), (404, 1)])
def test_retries_bounded(manifest, tmp_path, monkeypatch, code, expected_calls):
    calls = []
    def network(*args):
        calls.append(args)
        raise urllib.error.HTTPError(args[0], code, 'synthetic error', {}, None)
    monkeypatch.setattr(sm, '_open_url', network)
    monkeypatch.setattr(sm.time, 'sleep', lambda n: None)
    with pytest.raises(urllib.error.HTTPError):
        fetch(manifest, tmp_path)
    assert len(calls) == expected_calls
    assert not (tmp_path / 'result/payload.tar').exists()


@pytest.mark.parametrize('attempts', [0, 6, True])
def test_retry_limit_validated(manifest, tmp_path, attempts):
    with pytest.raises(sm.ManifestError):
        fetch(manifest, tmp_path, attempts=attempts)


def test_existing_output_is_not_modified(manifest, tmp_path):
    out = tmp_path / 'result'
    out.mkdir()
    (out / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        fetch(manifest, tmp_path)
    assert (out / 'sentinel').read_text() == 'keep'


def test_shared_root_and_symlinks_rejected(manifest, tmp_path):
    root = tmp_path / 'attempt'
    root.mkdir()
    for output in (root, tmp_path / 'outside'):
        with pytest.raises(sm.ManifestError):
            sm.fetch_manifest(manifest, output, attempt_root=root)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(sm.ManifestError):
        sm.fetch_manifest(manifest, root / 'link/result', attempt_root=root)
    assert list(outside.iterdir()) == []


def test_repository_destination_rejected():
    with pytest.raises(sm.ManifestError, match='repository'):
        sm._new_output(sm.REPO_ROOT / 'synthetic-output', sm.REPO_ROOT)


@pytest.mark.parametrize('name', ['us', 'census', 'dem', 'endes', 'births', 'mexico'])
def test_committed_manifest_contract(name):
    manifest = sm.load_source(name)
    assert manifest['id'] == name
    assert manifest['selection_provenance']['inspected_on'] == '2026-10-04'
    assert all(r['inspection'] and r['release'] for r in manifest['resources'])


def test_production_loader_rejects_attempt_manifest(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    with pytest.raises(sm.ManifestError, match='Unknown source'):
        sm.load_source(str(path))


def test_shards_cover_intended_jurisdictions_once():
    atlas = json.loads((sm.REPO_ROOT / 'configs/sources/atlas_shards.json').read_text())
    sm.validate_shards(atlas)
    jurisdictions = [s for g in atlas['groups'] for s in g['jurisdictions']]
    assert len(jurisdictions) == len(set(jurisdictions)) == 49
    assert 'DC' in jurisdictions and 'AK' not in jurisdictions and 'HI' not in jurisdictions
    assert not atlas['national_dem_coverage'] == 'complete'
    for modification in ('duplicate', 'missing', 'wrong_group'):
        broken = copy.deepcopy(atlas)
        if modification == 'duplicate': broken['groups'][0]['jurisdictions'].append('NY')
        elif modification == 'missing': broken['groups'][0]['jurisdictions'].pop()
        else:
            broken['groups'][0]['jurisdictions'][0], broken['groups'][1]['jurisdictions'][0] = (
                broken['groups'][1]['jurisdictions'][0], broken['groups'][0]['jurisdictions'][0])
        with pytest.raises(sm.ManifestError): sm.validate_shards(broken)


def test_cli_validate_is_offline_and_blocked_fetch_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('SWARM_UNIT_DIR', str(tmp_path))
    assert sm.main(['validate', '--source', 'dem']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'
    assert sm.main(['fetch', '--source', 'dem', '--output-dir', str(tmp_path / 'dem')]) == 1
    assert 'Blocked manifest' in capsys.readouterr().err
    assert not (tmp_path / 'dem').exists()
