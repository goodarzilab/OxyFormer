"""Synthetic, offline tests: never download production data in this suite."""
import copy
import hashlib
import http.client
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
        self.headers = {'Content-Length': str(len(data))} if headers is None else headers
        self.framing, self.declared_length, self.content_encoding = sm._framing(self.headers)
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
    if case == 'too_large_stream':
        data = b'x' * 101
        manifest['resources'][0]['expected_bytes'] = 100
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


@pytest.mark.parametrize('name', ['us', 'census', 'dem', 'endes', 'births', 'mexico', 'inec'])
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


def test_close_delimited_without_integrity_is_not_complete(manifest, tmp_path, monkeypatch):
    # EOF alone cannot distinguish a full close-delimited body from a lost suffix.
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response(b'only a prefix', {}))
    with pytest.raises(sm.ManifestError, match='framing|expected'):
        fetch(manifest, tmp_path)
    assert not (tmp_path / 'result/payload.tar').exists()


def test_assigned_symlink_root_can_fetch(manifest, tmp_path, monkeypatch):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'assigned'
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response())
    result = sm.fetch_manifest(manifest, alias / 'result', attempt_root=alias)
    assert result['status'] == 'complete'
    assert (real / 'result/payload.tar').is_file()


@pytest.mark.parametrize('expectation', ['expected_bytes', 'expected_sha256'])
def test_close_delimited_with_expectation_is_accepted(manifest, tmp_path, monkeypatch, expectation):
    data = b'complete synthetic response'
    for resource in manifest['resources']:
        resource[expectation] = len(data) if expectation == 'expected_bytes' else hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response(data, {}))
    result = fetch(manifest, tmp_path)
    assert result['status'] == 'complete'
    assert all(r['transfer_integrity'] == expectation for r in result['resources'])


def chunked_response(body):
    # Exercise the real stdlib chunk parser over a fake socket, without a network.
    class FakeSocket:
        def makefile(self, mode):
            return io.BytesIO(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n' + body)
    response = sm.AcquisitionResponse(FakeSocket())
    response.begin()
    response.url = 'https://example.test/chunked'
    return response


def test_chunked_response_without_length_or_hash_is_accepted(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(sm, '_open_url', lambda *a: chunked_response(b'4\r\ndata\r\n0\r\n\r\n'))
    result = fetch(manifest, tmp_path)
    assert all(r['transfer_integrity'] == 'chunked' and r['bytes'] == 4 for r in result['resources'])


def test_truncated_chunked_response_cannot_complete(manifest, tmp_path, monkeypatch):
    # No terminating zero chunk: the HTTP parser must fail even after a full data chunk.
    monkeypatch.setattr(sm, '_open_url', lambda *a: chunked_response(b'4\r\ndata\r\n'))
    with pytest.raises(http.client.IncompleteRead):
        fetch(manifest, tmp_path, attempts=1)
    assert not (tmp_path / 'result/payload.tar').exists()


def test_symlink_below_assigned_alias_cannot_escape(manifest, tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'assigned'
    alias.symlink_to(real, target_is_directory=True)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    (real / 'escape').symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(sm.ManifestError):
        sm.fetch_manifest(manifest, alias / 'escape/result', attempt_root=alias)
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize('expected_bytes', [0, True, -1, 101])
def test_invalid_expected_bytes_rejected(manifest, expected_bytes):
    manifest['resources'][0]['expected_bytes'] = expected_bytes
    with pytest.raises(sm.ManifestError, match='expected_bytes'):
        sm.validate_manifest(manifest)


def raw_response(headers, body):
    class FakeSocket:
        def makefile(self, mode):
            return io.BytesIO(b'HTTP/1.1 200 OK\r\n' + headers + b'\r\n' + body)
    response = sm.AcquisitionResponse(FakeSocket())
    response.begin()
    response.url = 'https://example.test/parser'
    return response


def framed_response(mode, data=b'data'):
    if mode == 'chunked':
        body = (f'{len(data):x}\r\n'.encode() + data + b'\r\n0\r\n\r\n') if data else b'0\r\n\r\n'
        # Conflicting CL deliberately exercises all prechecks as well as EOF.
        return raw_response(b'Transfer-Encoding: chunked\r\nContent-Length: 100\r\n', body)
    if mode == 'content_length':
        return raw_response(f'Content-Length: {len(data)}\r\n'.encode(), data)
    return raw_response(b'Connection: close\r\n', data)


def assert_failed_artifacts(tmp_path):
    out = tmp_path / 'result'
    assert {p.name for p in out.iterdir()} == {'receipts.json', 'download.log'}
    assert json.loads((out / 'receipts.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('length', [None, '4', '100', '101', '0', 'invalid'])
def test_chunked_ignores_all_raw_content_lengths(manifest, tmp_path, monkeypatch, length):
    headers = b'Transfer-Encoding: chunked\r\n'
    if length is not None:
        headers += f'Content-Length: {length}\r\n'.encode()
    for resource in manifest['resources']:
        resource['expected_bytes'] = 4
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(headers, b'4\r\ndata\r\n0\r\n\r\n'))
    result = fetch(manifest, tmp_path)
    assert all(r['bytes'] == 4 and r['transfer_integrity'] == 'chunked' for r in result['resources'])


@pytest.mark.parametrize('mode', ['chunked', 'content_length', 'close'])
@pytest.mark.parametrize('size,hash_state', [
    (None, None), (4, None), (3, None), (None, 'match'), (None, 'wrong'),
    (4, 'match'), (3, 'match'), (4, 'wrong'),
])
def test_parser_expectation_matrix(manifest, tmp_path, monkeypatch, mode, size, hash_state):
    digest = hashlib.sha256(b'data').hexdigest()
    for resource in manifest['resources']:
        if size is not None: resource['expected_bytes'] = size
        if hash_state: resource['expected_sha256'] = digest if hash_state == 'match' else '0' * 64
    monkeypatch.setattr(sm, '_open_url', lambda *a: framed_response(mode))
    fail = size == 3 or hash_state == 'wrong' or mode == 'close' and size is None and hash_state is None
    if fail:
        with pytest.raises(sm.ManifestError): fetch(manifest, tmp_path, attempts=1)
        assert_failed_artifacts(tmp_path)
    else:
        result = fetch(manifest, tmp_path)
        expected_mode = mode if mode != 'close' else 'expected_bytes' if size else 'expected_sha256'
        assert all(r['bytes'] == 4 and r['sha256'] == digest and r['transfer_integrity'] == expected_mode
                   for r in result['resources'])


@pytest.mark.parametrize('body', [b'4\r\nda', b'4\r\ndata\r\n'])
def test_parser_incomplete_chunk_cannot_be_rescued_by_expectations(manifest, tmp_path, monkeypatch, body):
    for resource in manifest['resources']:
        resource['expected_bytes'] = 4
        resource['expected_sha256'] = hashlib.sha256(b'data').hexdigest()
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(b'Transfer-Encoding: chunked\r\n', body))
    with pytest.raises(http.client.IncompleteRead): fetch(manifest, tmp_path, attempts=1)
    assert_failed_artifacts(tmp_path)


@pytest.mark.parametrize('length', ['invalid', '0', '101'])
def test_nonchunked_length_validation(manifest, tmp_path, monkeypatch, length):
    for resource in manifest['resources']:
        resource['expected_sha256'] = hashlib.sha256(b'data').hexdigest()
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(f'Content-Length: {length}\r\n'.encode(), b'data'))
    with pytest.raises(sm.ManifestError): fetch(manifest, tmp_path, attempts=1)
    assert_failed_artifacts(tmp_path)


@pytest.mark.parametrize('expectation', [None, 'expected_bytes', 'expected_sha256'])
def test_nonchunked_parser_boundary(manifest, tmp_path, monkeypatch, expectation):
    # The parser exposes four bytes; trailing wire bytes are not part of this body.
    data = b'data plus extra wire bytes'
    if expectation:
        for resource in manifest['resources']:
            resource[expectation] = len(data) if expectation == 'expected_bytes' else hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(b'Content-Length: 4\r\n', data))
    if expectation:
        with pytest.raises(sm.ManifestError): fetch(manifest, tmp_path)
        assert_failed_artifacts(tmp_path)
    else:
        result = fetch(manifest, tmp_path)
        assert all(r['bytes'] == 4 and r['sha256'] == hashlib.sha256(b'data').hexdigest()
                   for r in result['resources'])


@pytest.mark.parametrize('mode', ['chunked', 'content_length', 'close'])
@pytest.mark.parametrize('data', [b'', b'data', b'data!'])
def test_parser_byte_boundaries(manifest, tmp_path, monkeypatch, mode, data):
    for resource in manifest['resources']:
        resource['max_bytes'] = 4
        resource['expected_sha256'] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(sm, '_open_url', lambda *a: framed_response(mode, data))
    if len(data) != 4:
        with pytest.raises(sm.ManifestError): fetch(manifest, tmp_path, attempts=1)
        assert_failed_artifacts(tmp_path)
    else:
        result = fetch(manifest, tmp_path)
        assert all(r['bytes'] == 4 for r in result['resources'])


@pytest.mark.parametrize('ows', [' ', '\t', ' \t'])
def test_http_field_optional_whitespace(manifest, tmp_path, monkeypatch, ows):
    data = b'<html>CC-BY 4.0</html>'
    for resource in manifest['resources']:
        resource.update(format='html', expected_bytes=len(data))
    headers = f'Content-Length: {ows}{len(data)}{ows}\r\nContent-Encoding: {ows}identity{ows}\r\n'.encode()
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(headers, data))
    result = fetch(manifest, tmp_path)
    assert all(r['bytes'] == len(data) for r in result['resources'])


def test_short_framed_body_rejected_despite_matching_hash(manifest, tmp_path, monkeypatch):
    data = b'x' * 100
    for resource in manifest['resources']:
        resource.update(max_bytes=300, expected_bytes=100,
                        expected_sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(b'Content-Length: 200\r\n', data))
    with pytest.raises(sm.ManifestError, match='Content-Length disagrees'):
        fetch(manifest, tmp_path, attempts=1)
    assert_failed_artifacts(tmp_path)


def test_parser_truncated_attempt_then_success(manifest, tmp_path, monkeypatch):
    calls = []
    def network(*args):
        calls.append(args)
        if len(calls) == 1:
            return raw_response(b'Content-Length: 8\r\n', b'part')
        return raw_response(b'Content-Length: 4\r\n', b'data')
    monkeypatch.setattr(sm, '_open_url', network)
    monkeypatch.setattr(sm.time, 'sleep', lambda n: None)
    result = fetch(manifest, tmp_path, attempts=2)
    assert len(calls) == 4 and result['resources'][0]['attempts'] == 2
    assert all(r['bytes'] == 4 and r['sha256'] == hashlib.sha256(b'data').hexdigest()
               for r in result['resources'])
    with tarfile.open(tmp_path / 'result/payload.tar') as archive:
        assert all(archive.extractfile(member).read() == b'data' for member in archive.getmembers())
    assert {p.name for p in (tmp_path / 'result').iterdir()} == {'payload.tar', 'receipts.json', 'download.log'}


@pytest.mark.parametrize('name', ['us', 'dem', 'births', 'mexico'])
def test_each_production_manifest_blocks_without_outputs(name, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sm, '_open_url', lambda *a: calls.append(a))
    with pytest.raises(sm.ManifestError, match='Blocked manifest'):
        fetch(sm.load_source(name), tmp_path)
    assert not calls and list(tmp_path.iterdir()) == []


@pytest.fixture
def staged_manifest(manifest, tmp_path, monkeypatch):
    root = tmp_path / 'staging'
    (root / '2023').mkdir(parents=True)
    monkeypatch.setattr(sm, 'DANE_STAGING_DIR', root)
    for r in manifest['resources']:
        data = ('staged ' + r['id']).encode()
        path = root / '2023' / (r['id'] + '.txt')
        path.write_bytes(data)
        path.chmod(0o400)
        r.update(transport='local', local_path='2023/' + path.name,
                 expected_bytes=len(data), expected_sha256=hashlib.sha256(data).hexdigest())
    return manifest


def test_staged_resources_are_read_only_and_hash_audited(staged_manifest, tmp_path):
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in sm.DANE_STAGING_DIR.rglob('*.txt')}
    receipt = fetch(staged_manifest, tmp_path)
    assert receipt['status'] == 'complete'
    with tarfile.open(tmp_path / 'result/payload.tar') as archive:
        for entry in receipt['resources']:
            assert entry['checksum_status'] == 'matched'
            assert entry['transfer_integrity'] == 'staged_size_and_sha256'
            assert archive.extractfile(entry['destination']).read() == before[Path(entry['local_path'])][0]
    assert before == {p: (p.read_bytes(), p.stat().st_mode) for p in before}


@pytest.mark.parametrize('corruption', ['same_size', 'size', 'symlink', 'directory'])
def test_staged_corruption_cannot_publish(staged_manifest, tmp_path, corruption):
    r = staged_manifest['resources'][0]
    path = sm.DANE_STAGING_DIR / r['local_path']
    path.chmod(0o600)
    if corruption == 'same_size': path.write_bytes(b'x' * r['expected_bytes'])
    elif corruption == 'size': path.write_bytes(b'x')
    elif corruption == 'symlink':
        outside = tmp_path / 'outside'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path.unlink()
        path.mkdir()
    with pytest.raises((sm.ManifestError, IsADirectoryError)):
        fetch(staged_manifest, tmp_path)
    assert not (tmp_path / 'result/payload.tar').exists()
    assert json.loads((tmp_path / 'result/receipts.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('change', [{'expected_sha256': None}, {'expected_bytes': None},
                                  {'local_path': '../outside'}, {'local_path': '2022/data.txt'}])
def test_staging_contract_requires_recorded_identity(staged_manifest, change):
    staged_manifest['resources'][0].update(change)
    with pytest.raises(sm.ManifestError): sm.validate_manifest(staged_manifest)


def test_inec_is_usable_independently_of_dane():
    inec = sm.load_source('inec')
    assert inec['status'] == 'ready'
    assert all('dane.gov' not in r['url'] for r in inec['resources'])
    assert {'2015', '2024'} <= {r['release'] for r in inec['resources']}
    for name in ('census', 'endes', 'inec'):
        m = sm.load_source(name)
        assert m['status'] == 'ready' and all(r['availability'] == 'verified' for r in m['resources'])


def test_tile_inventory_expands_exact_paired_urls_and_preserves_year_precision():
    m = sm.load_source('dem')
    resources = {r['id']: r for r in m['resources']}
    assert resources['n33w119']['release'].endswith('_2013')
    assert resources['n33w119']['url'] == sm.DEM_URL + 'n33w119/USGS_13_n33w119.tif'
    assert resources['n33w119_metadata']['url'] == sm.DEM_URL + 'n33w119/USGS_13_n33w119.xml'
    assert all(r['expected_bytes'] == r['max_bytes'] for r in m['resources'] if r['format'] in ('tiff', 'xml'))
    assert m['coverage']['missing_tiles'] and m['status'] == 'blocked'


@pytest.mark.parametrize('change', ['ceiling', 'footprint', 'tile', 'ready'])
def test_shard_contract_rejects_unapproved_or_inconsistent_selection(change):
    a = json.loads((sm.REPO_ROOT / 'configs/sources/atlas_shards.json').read_text())
    g = a['groups'][0]
    if change == 'ceiling': g['max_bytes'] += 1
    elif change == 'footprint': g['dem_footprint'] = None
    elif change == 'tile': g['dem_resources'][0] = 'n00w000'
    else: g.update(status='ready', blockers=[])
    with pytest.raises(sm.ManifestError): sm.validate_shards(a)


@pytest.mark.parametrize('fmt,content', [('tiff', b'II*\x00synthetic'), ('xml', b'<?xml version="1.0"?><metadata/>')])
def test_dem_formats_accept_real_signatures_and_reject_html(fmt, content):
    sm._check_content({'format': fmt}, content, '')
    with pytest.raises(sm.ManifestError): sm._check_content({'format': fmt}, b'<html>challenge</html>', 'text/html')


@pytest.mark.parametrize('coding', ['chunked', ' ChUnKeD \t', '\tchunked '])
def test_production_opener_uses_normalized_decoder(manifest, tmp_path, monkeypatch, coding):
    wire = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: ' + coding.encode() + b'\r\nContent-Length: invalid\r\n\r\n4\r\ndata\r\n0\r\n\r\n'
    class Socket:
        def makefile(self, mode): return io.BytesIO(wire)
        def sendall(self, data): pass
        def close(self): pass
    monkeypatch.setattr(http.client.HTTPSConnection, 'connect', lambda self: setattr(self, 'sock', Socket()))
    result = fetch(manifest, tmp_path)
    assert all(r['bytes'] == 4 and r['sha256'] == hashlib.sha256(b'data').hexdigest()
               and r['transfer_integrity'] == 'chunked' for r in result['resources'])
    with tarfile.open(tmp_path / 'result/payload.tar') as archive:
        assert all(archive.extractfile(m).read() == b'data' for m in archive.getmembers())


@pytest.mark.parametrize('trailers,valid', [(b'\r\n', True), (b'X: y\r\n' * 100 + b'\r\n', True),
    (b'X:' + b'x' * 65530 + b'\r\n\r\n', True), (b'', False),
    (b'X: y\r\n' * 101 + b'\r\n', False), (b'X:' + b'x' * 65531 + b'\r\n\r\n', False)])
def test_chunked_terminal_witness(manifest, tmp_path, monkeypatch, trailers, valid):
    for r in manifest['resources']:
        r.update(expected_bytes=4, expected_sha256=hashlib.sha256(b'data').hexdigest())
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(
        b'Transfer-Encoding: chunked\r\n', b'4\r\ndata\r\n0\r\n' + trailers))
    if valid:
        assert all(r['bytes'] == 4 for r in fetch(manifest, tmp_path)['resources'])
    else:
        with pytest.raises((sm.ManifestError, http.client.IncompleteRead)):
            fetch(manifest, tmp_path, attempts=1)
        assert_failed_artifacts(tmp_path)


@pytest.mark.parametrize('data', [b'\xef\xbb\xbf<?xml version="1.0"?><metadata/>', b'<root/>',
    b'\xff\xfe' + '<root>\u00e9</root>'.encode('utf-16le'),
    b'\xfe\xff' + '<root>\u00e9</root>'.encode('utf-16be'), b'<x:root xmlns:x="urn:test"/>', b'<!DOCTYPE html_data><html_data/>',
    b'<!--' + b'x' * 65537 + b'--><root/>'])
@pytest.mark.parametrize('local', [False, True])
def test_xml_original_bytes_to_eof(manifest, tmp_path, monkeypatch, data, local):
    root = tmp_path / 'staging'
    if local:
        (root / '2023').mkdir(parents=True)
        monkeypatch.setattr(sm, 'DANE_STAGING_DIR', root)
    monkeypatch.setattr(sm, 'CHUNK', 17)  # Splits UTF-16 code units across reads.
    for r in manifest['resources']:
        r.update(format='xml', max_bytes=len(data), expected_bytes=len(data),
                 expected_sha256=hashlib.sha256(data).hexdigest())
        if local:
            r.update(transport='local', local_path='2023/' + r['id'])
            p = root / r['local_path']; p.write_bytes(data); p.chmod(0o400)
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response(data))
    result = fetch(manifest, tmp_path)
    assert all(r['sha256'] == hashlib.sha256(data).hexdigest() for r in result['resources'])
    with tarfile.open(tmp_path / 'result/payload.tar') as archive:
        assert all(archive.extractfile(m).read() == data for m in archive.getmembers())
    if local:
        assert all(p.read_bytes() == data and p.stat().st_mode & 0o777 == 0o400
                   for p in (root / '2023').iterdir())


@pytest.mark.parametrize('data', [b'<root>', b'<root/>junk', b'not xml',
    b'\xef\xbb\xbf<html/>', b'<html xmlns="http://www.w3.org/1999/xhtml"/>',
    b'\xff\xfe' + '<html/>'.encode('utf-16le'),
    b'<!DOCTYPE root [<!ENTITY e SYSTEM "https://example.test/entity">]><root>&e;</root>'])
def test_xml_invalid_or_challenge_never_publishes(manifest, tmp_path, monkeypatch, data):
    for r in manifest['resources']: r.update(format='xml', max_bytes=1000)
    monkeypatch.setattr(sm, '_open_url', lambda *a: Response(data))
    with pytest.raises(sm.ManifestError): fetch(manifest, tmp_path, attempts=1)
    assert_failed_artifacts(tmp_path)


@pytest.mark.parametrize('headers,valid', [(b'Content-Length: 4\r\nContent-Length: 4\r\n', True),
    (b'Content-Length: 4, 4\r\n', True), (b'Content-Length: 4, 5\r\n', False),
    (b'Transfer-Encoding: gzip\r\n', False), (b'Transfer-Encoding: gzip, chunked\r\n', False),
    (b'Content-Encoding: gzip\r\nContent-Length: 4\r\n', False)])
def test_finite_header_profile(manifest, tmp_path, monkeypatch, headers, valid):
    monkeypatch.setattr(sm, '_open_url', lambda *a: raw_response(headers, b'data'))
    if valid: assert all(r['bytes'] == 4 for r in fetch(manifest, tmp_path)['resources'])
    else:
        with pytest.raises(sm.ManifestError): fetch(manifest, tmp_path, attempts=1)
        assert_failed_artifacts(tmp_path)


def test_xml_retry_resets_parser(manifest, tmp_path, monkeypatch):
    calls = []
    for r in manifest['resources']: r.update(format='xml', expected_bytes=7)
    def network(*args):
        calls.append(args)
        return raw_response(b'Content-Length: 7\r\n', b'<root>' if len(calls) == 1 else b'<root/>')
    monkeypatch.setattr(sm, '_open_url', network)
    monkeypatch.setattr(sm.time, 'sleep', lambda n: None)
    result = fetch(manifest, tmp_path, attempts=2)
    assert len(calls) == 4 and result['resources'][0]['attempts'] == 2
    assert all(r['sha256'] == hashlib.sha256(b'<root/>').hexdigest() for r in result['resources'])
