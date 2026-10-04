"""Offline acquisition contracts and a bounded, attempt-local stdlib fetcher.

Production CLI inputs are the committed registry and manifests only. Downloaded
archives are opaque files: this module never extracts third-party archives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.parsers import expat

REPO_ROOT = Path(__file__).resolve().parents[3]
CHUNK = 1024 * 1024
DANE_STAGING_DIR = Path('/mnt/weka/home/hgoodarzi/oxyformer-swarm/staging/dane_births')
DEM_URL = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/'
DIVISIONS = {
    'new_england': 'CT ME MA NH RI VT'.split(),
    'middle_atlantic': 'NJ NY PA'.split(),
    'east_north_central': 'IL IN MI OH WI'.split(),
    'west_north_central': 'IA KS MN MO NE ND SD'.split(),
    'south_atlantic': 'DE DC FL GA MD NC SC VA WV'.split(),
    'east_south_central': 'AL KY MS TN'.split(),
    'west_south_central': 'AR LA OK TX'.split(),
    'mountain': 'AZ CO ID MT NV NM UT WY'.split(),
    'pacific': 'CA OR WA'.split(),
}


class ManifestError(ValueError):
    """Invalid or blocked acquisition; no partial success is implied."""


def _require(condition, message):
    if not condition:
        raise ManifestError(message)


def _positive(value):
    return type(value) is int and value > 0


def _https(url):
    _require(isinstance(url, str) and not any(c.isspace() for c in url),
             'URL must be a whitespace-free HTTPS string')
    parsed = urllib.parse.urlsplit(url)
    _require(parsed.scheme == 'https' and parsed.hostname and not parsed.username
             and not parsed.password and not parsed.fragment and parsed.port in (None, 443),
             'Only credential-free HTTPS URLs on port 443 are supported')
    return parsed


def _destination(value):
    _require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_./-]+', value),
             'Invalid relative destination')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(p not in ('', '.', '..') for p in value.split('/')),
             'Destination traversal is forbidden')
    return path


def validate_manifest(manifest):
    """Validate shape even for blocked contracts; this does not authorize fetch."""
    _require(isinstance(manifest, dict) and manifest.get('schema_version') == 1,
             'Expected manifest schema_version 1')
    _require(manifest.get('kind') == 'acquisition', 'Expected acquisition manifest')
    _require(isinstance(manifest.get('id'), str) and manifest['id'], 'Missing manifest id')
    _require(manifest.get('status') in ('ready', 'blocked'), 'Invalid manifest status')
    blockers = manifest.get('blockers')
    _require(isinstance(blockers, list) and all(isinstance(x, str) and x for x in blockers),
             'blockers must be a list of reasons')
    _require(bool(blockers) == (manifest['status'] == 'blocked'), 'Status/blockers disagree')
    for key in ('release_identity', 'access_terms', 'selection_provenance'):
        _require(isinstance(manifest.get(key), dict) and manifest[key], f'Missing {key}')
    _require(manifest['access_terms'].get('rule') == 'public_data_only', 'Public data only')
    _require('tile_inventory' not in manifest and 'resource_defaults' not in manifest,
             'Load packed production inventories through the committed registry')
    resources = manifest.get('resources')
    _require(isinstance(resources, list), 'resources must be a list')
    seen_ids, seen_paths = set(), set()
    for resource in resources:
        _require(isinstance(resource, dict), 'Invalid resource')
        rid = resource.get('id')
        _require(isinstance(rid, str) and rid and rid not in seen_ids, 'Duplicate/missing resource id')
        seen_ids.add(rid)
        _https(resource.get('url'))
        path = _destination(resource.get('destination'))
        _require(path not in seen_paths and not any(p in seen_paths for p in path.parents)
                 and not any(path in p.parents for p in seen_paths), 'Conflicting destinations')
        seen_paths.add(path)
        _require(_positive(resource.get('max_bytes')), 'Positive integer max_bytes required')
        _require(resource.get('role') in ('data', 'dictionary', 'terms', 'metadata', 'code'),
                 'Invalid resource role')
        _require(resource.get('format') in ('zip', 'pdf', 'text', 'html', 'json', 'csv', 'tar_gz', 'tiff', 'xml'),
                 'Invalid resource format')
        _require(resource.get('availability') in ('verified', 'unverified', 'unavailable'),
                 'Missing resource availability')
        for key in ('release', 'inspection'):
            _require(isinstance(resource.get(key), str) and resource[key], f'Missing resource {key}')
        expected_bytes = resource.get('expected_bytes')
        _require(expected_bytes is None or _positive(expected_bytes)
                 and expected_bytes <= resource['max_bytes'], 'Invalid expected_bytes')
        digest = resource.get('expected_sha256')
        _require(digest is None or isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                 'Invalid expected SHA-256')
        _require(resource.get('transport', 'https') in ('https', 'local'), 'Invalid transport')
        if resource.get('transport') == 'local':
            local = _destination(resource.get('local_path'))
            _require(local.parts[0] in ('2023', '2024', '2025') and len(local.parts) > 1,
                     'Local DANE resource must be under an approved year directory')
            _require(expected_bytes is not None and digest is not None,
                     'Local resources require recorded size and SHA-256')
    requirements = manifest.get('requirements')
    _require(isinstance(requirements, list) and requirements, 'Required resource checklist missing')
    requirement_ids = set()
    for item in requirements:
        _require(isinstance(item, dict) and item.get('id') not in requirement_ids and item.get('id'),
                 'Duplicate/missing requirement id')
        requirement_ids.add(item['id'])
        _require(item.get('status') in ('ready', 'blocked'), 'Invalid requirement status')
        ids = item.get('resource_ids')
        _require(isinstance(ids, list) and all(rid in seen_ids for rid in ids),
                 'Requirement refers to unknown resources')
        if item['status'] == 'ready':
            _require(ids, 'Ready requirement has no resources')
        else:
            _require(item.get('reason'), 'Blocked requirement needs a reason')
    if manifest['status'] == 'ready':
        _require(resources and all(x['status'] == 'ready' for x in requirements),
                 'Unavailable required resource blocks manifest')
        _require(all(r['availability'] == 'verified' for r in resources),
                 'Unverified resource blocks manifest')
        _require({'data', 'dictionary', 'terms'} <= {r['role'] for r in resources},
                 'Ready manifest needs data, dictionary and terms')
    return manifest


def validate_shards(atlas, dem=None):
    _require(atlas.get('schema_version') == 1 and atlas.get('kind') == 'atlas_shards',
             'Invalid atlas schema')
    groups = atlas.get('groups', [])
    _require(len(groups) == 9, 'Exactly nine divisions required')
    _require({g.get('id') for g in groups} == set(DIVISIONS), 'Division ids differ')
    intended = {s for states in DIVISIONS.values() for s in states}
    counts = Counter(s for g in groups for s in g['jurisdictions'])
    _require(set(counts) == intended and set(counts.values()) == {1},
             'Each intended jurisdiction must occur exactly once')
    dem = load_source('dem') if dem is None else validate_manifest(dem)
    resources = {r['id']: r for r in dem['resources']}
    tiles = {r['id'] for r in dem['resources'] if r['role'] == 'data' and r['format'] == 'tiff'}
    assigned = set()
    for group in groups:
        _require(set(group['jurisdictions']) == set(DIVISIONS[group['id']]), 'Wrong division membership')
        _require(group.get('status') in ('ready', 'blocked') and
                 bool(group.get('blockers')) == (group['status'] == 'blocked'),
                 'Invalid shard readiness')
        footprint = group.get('dem_footprint', {})
        _require(isinstance(footprint, dict) and footprint.get('method') == 'state_geometry_intersection' and
                 footprint.get('outcome_blind') is True and
                 footprint.get('boundary_resource') == 'state_boundaries_2010' and
                 footprint['boundary_resource'] in resources and
                 set(footprint.get('jurisdictions', [])) == set(group['jurisdictions']),
                 'Approved outcome-blind footprint required')
        ids = group.get('dem_resources', [])
        _require(ids and len(ids) == len(set(ids)) and all(i in tiles for i in ids),
                 'Unknown or repeated DEM tile')
        assigned.update(ids)
        # Border tiles can be read by both neighboring divisions; jurisdictions remain disjoint.
        budget = sum(resources[i]['max_bytes'] + resources[i + '_metadata']['max_bytes'] for i in ids)
        _require(group.get('max_bytes') == budget and _positive(budget), 'Wrong DEM byte ceiling')
        if group['status'] == 'ready':
            _require(dem['status'] == 'ready', 'Blocked DEM cannot authorize a ready shard')
    _require(assigned == tiles, 'Tile inventory and shards disagree')
    _require(atlas.get('national_dem_coverage') == 'incomplete', 'National DEM coverage is incomplete')
    return atlas


def _unpack_manifest(manifest):
    """Expand compact committed tables; no filenames are discovered at fetch time."""
    manifest = dict(manifest)
    defaults = manifest.pop('resource_defaults', {})
    manifest['resources'] = [dict(defaults, **r) for r in manifest['resources']]
    inventory = manifest.pop('tile_inventory', None)
    if inventory is not None:
        _require(manifest['id'] == 'dem', 'Tile table is only supported for approved DEM')
        _require(inventory['columns'] == ['tile', 'data_bytes', 'metadata_bytes', 'publication_date'],
                 'Unknown tile table layout')
        ids = []
        for tile, size, metadata_size, date in inventory['rows']:
            _require(isinstance(tile, str) and re.fullmatch(r'n[0-9]{2}w[0-9]{3}', tile),
                     'Invalid tile identifier')
            _require(isinstance(date, str) and re.fullmatch(r'(?:[0-9]{4}|[0-9]{8})', date), 'Invalid tile date')
            # Every paired stem below was observed in the USGS current bucket inventory.
            stem = tile + '/USGS_13_' + tile
            for suffix, role, fmt, length in [('', 'data', 'tiff', size),
                                             ('_metadata', 'metadata', 'xml', metadata_size)]:
                rid = tile + suffix
                ids.append(rid)
                manifest['resources'].append({
                    'id': rid, 'release': 'USGS_13_' + tile + '_' + date,
                    'url': DEM_URL + stem + ('.tif' if fmt == 'tiff' else '.xml'),
                    'destination': 'dem/' + tile + ('.tif' if fmt == 'tiff' else '.xml'),
                    'role': role, 'format': fmt, 'max_bytes': length, 'expected_bytes': length,
                    'availability': 'verified',
                    'inspection': inventory['inspection'],
                })
        manifest['requirements'] = [dict(r, resource_ids=ids) if r['id'] == 'approved_tiles'
                                    else r for r in manifest['requirements']]
    return manifest


def load_source(name):
    """Resolve a source id through the committed JSON-compatible YAML registry."""
    registry = json.loads((REPO_ROOT / 'configs/sources.yaml').read_text())
    _require(name in registry['sources'], 'Unknown source id')
    relative = _destination(registry['sources'][name])
    path = (REPO_ROOT / relative).resolve()
    _require(path.is_relative_to(REPO_ROOT / 'configs/sources') and path.suffix == '.json',
             'Production manifests must live under configs/sources')
    return validate_manifest(_unpack_manifest(json.loads(path.read_text())))


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _framing(headers):
    def values(name):
        raw = headers.get_all(name, []) if hasattr(headers, 'get_all') else (
            [headers[name]] if name in headers else [])
        return [v.strip(' \t') for item in raw for v in item.split(',')]
    encoding = ','.join(v.lower() for v in values('Content-Encoding')) or 'identity'
    transfer = values('Transfer-Encoding')
    if transfer:
        _require([v.lower() for v in transfer] == ['chunked'], 'Unsupported Transfer-Encoding')
        return 'chunked', None, encoding
    lengths = values('Content-Length')
    if not lengths:
        return 'close', None, encoding
    _require(all(v.isascii() and v.isdigit() for v in lengths), 'Invalid Content-Length')
    try:
        numbers = [int(v) for v in lengths]
    except ValueError as exc:
        raise ManifestError('Invalid Content-Length') from exc
    _require(len(set(numbers)) == 1, 'Conflicting Content-Length values')
    return 'content_length', numbers[0], encoding


class AcquisitionResponse(http.client.HTTPResponse):
    """One effective framing decision, installed before the first body read."""
    def begin(self):
        if self.headers is not None:
            return
        super().begin()
        self.framing, self.declared_length, self.content_encoding = _framing(self.headers)
        self.chunked = self.framing == 'chunked'
        self.chunk_left = None
        self.length = self.declared_length
        self.will_close = self._check_close() or self.framing == 'close'
        if self.status in (204, 304) or 100 <= self.status < 200 or self._method == 'HEAD':
            self.length, self.chunked = 0, False

    def _read_and_discard_trailer(self):
        total = 0
        for count in range(101):
            line = self.fp.readline(65537 - total)
            total += len(line)
            _require(total <= 65536, 'Trailer byte ceiling exceeded')
            if line == b'\r\n':
                return
            if not line:
                raise http.client.IncompleteRead(b'')
            _require(line.endswith(b'\r\n') and count < 100, 'Invalid or excessive trailers')


class AcquisitionHTTPSConnection(http.client.HTTPSConnection):
    response_class = AcquisitionResponse


class AcquisitionHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(AcquisitionHTTPSConnection, request,
                            context=self._context, check_hostname=self._check_hostname)


def _open_url(url, timeout):
    request = urllib.request.Request(url, headers={
        'User-Agent': 'OxyFormer-source-fetch/1.0', 'Accept-Encoding': 'identity'})
    return urllib.request.build_opener(AcquisitionHTTPSHandler(), HTTPSRedirectHandler()).open(request, timeout=timeout)


def _check_content(resource, prefix, content_type):
    fmt = resource['format']
    lower = prefix.lstrip().lower()
    if fmt != 'html':
        _require('text/html' not in content_type.lower() and (fmt == 'xml' or not lower.startswith(
            (b'<!doctype html', b'<html'))), 'Unexpected HTML/login/challenge response')
    if fmt == 'zip':
        _require(prefix.startswith((b'PK\x03\x04', b'PK\x05\x06')), 'Expected ZIP signature')
    elif fmt == 'pdf':
        _require(prefix.startswith(b'%PDF-'), 'Expected PDF signature')
    elif fmt == 'tiff':
        _require(prefix.startswith((b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+')),
                 'Expected TIFF signature')
    elif fmt == 'tar_gz':
        _require(prefix.startswith(b'\x1f\x8b'), 'Expected gzip signature')


def _xml_parser(resource):
    if resource['format'] != 'xml':
        return None
    parser = expat.ParserCreate(namespace_separator='}')
    def root(name, attributes):
        _require(name.lower() not in ('html', 'http://www.w3.org/1999/xhtml}html'),
                 'Unexpected HTML/XML challenge response')
        parser.StartElementHandler = None
    parser.StartElementHandler = root
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.ExternalEntityRefHandler = lambda *args: 0
    return parser


def _xml_feed(parser, block, final=False):
    if parser is not None:
        try:
            parser.Parse(block, final)
        except expat.ExpatError as exc:
            raise ManifestError('Invalid XML resource') from exc


def _download(resource, path, *, attempts, timeout, log):
    for attempt in range(1, attempts + 1):
        digest, count, prefix = hashlib.sha256(), 0, b''
        xml = _xml_parser(resource)
        log.write(f"resource={resource['id']} attempt={attempt}\n")
        log.flush()
        try:
            with _open_url(resource['url'], timeout) as response, path.open('wb') as output:
                _require(response.status == 200, f'Unexpected HTTP status {response.status}')
                final_url = response.geturl()
                _https(final_url)
                _require(response.content_encoding == 'identity', 'Unexpected transfer content encoding')
                chunked = response.framing == 'chunked'
                length = response.declared_length
                if length is not None:
                    _require(0 < length <= resource['max_bytes'], 'Content-Length exceeds byte ceiling or is empty')
                expected_bytes = resource.get('expected_bytes')
                _require(length is not None or chunked or expected_bytes is not None
                         or resource.get('expected_sha256') is not None,
                         'Missing transfer framing: declare expected_bytes or expected_sha256')
                if expected_bytes is not None and length is not None:
                    _require(length == expected_bytes, 'Content-Length disagrees with expected_bytes')
                while True:
                    block = response.read(min(CHUNK, resource['max_bytes'] - count + 1))
                    if not block:
                        break
                    count += len(block)
                    _require(count <= resource['max_bytes'], 'Stream exceeds byte ceiling')
                    if len(prefix) < 512:
                        prefix += block[:512 - len(prefix)]
                    _xml_feed(xml, block)
                    digest.update(block)
                    output.write(block)
                _require(count > 0, 'Empty resource')
                if length is not None and length != count:
                    raise http.client.IncompleteRead(b'', length - count)
                if expected_bytes is not None:
                    _require(count == expected_bytes, 'Resource length disagrees with expected_bytes')
                _xml_feed(xml, b'', True)
                _check_content(resource, prefix, response.headers.get('Content-Type', ''))
                hexdigest = digest.hexdigest()
                expected = resource.get('expected_sha256')
                _require(expected is None or hexdigest == expected, 'SHA-256 mismatch')
                return {'id': resource['id'], 'url': resource['url'], 'final_url': final_url,
                        'destination': resource['destination'], 'bytes': count, 'sha256': hexdigest,
                        'expected_sha256': expected, 'checksum_status': 'matched' if expected else 'recorded',
                        'expected_bytes': expected_bytes,
                        'transfer_integrity': ('chunked' if chunked else 'content_length'
                            if length is not None else 'expected_bytes' if expected_bytes is not None
                            else 'expected_sha256'),
                        'attempts': attempt, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == attempts:
                raise
            log.write(f'retry HTTP {exc.code}\n')
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            if attempt == attempts:
                raise
            log.write(f'retry {type(exc).__name__}\n')
        time.sleep(min(2 ** (attempt - 1), 4))
    raise AssertionError('unreachable')


def _copy_local(resource, path, *, log):
    """Read an owner-staged DANE resource without changing its bytes or permissions."""
    root = DANE_STAGING_DIR.resolve(strict=True)
    source = root / resource['local_path']
    _require(source.resolve(strict=True) == source and source.is_relative_to(root),
             'Local resource escapes staging or uses a symlink')
    fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    digest, count, prefix = hashlib.sha256(), 0, b''
    xml = _xml_parser(resource)
    with os.fdopen(fd, 'rb') as stream, path.open('wb') as output:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), 'Local resource must be a regular file')
        _require(info.st_size == resource['expected_bytes'], 'Local recorded size mismatch')
        while True:
            block = stream.read(min(CHUNK, resource['max_bytes'] - count + 1))
            if not block:
                break
            count += len(block)
            _require(count <= resource['max_bytes'], 'Local resource exceeds byte ceiling')
            if len(prefix) < 512:
                prefix += block[:512 - len(prefix)]
            _xml_feed(xml, block)
            digest.update(block)
            output.write(block)
    _require(count == resource['expected_bytes'], 'Local recorded size mismatch')
    _require(digest.hexdigest() == resource['expected_sha256'], 'Local SHA-256 mismatch')
    _xml_feed(xml, b'', True)
    _check_content(resource, prefix, '')
    log.write(f"read-only staged resource={resource['id']}\n")
    return {'id': resource['id'], 'url': resource['url'], 'local_path': str(source),
            'destination': resource['destination'], 'bytes': count, 'sha256': digest.hexdigest(),
            'expected_sha256': resource['expected_sha256'], 'expected_bytes': resource['expected_bytes'],
            'checksum_status': 'matched', 'transfer_integrity': 'staged_size_and_sha256',
            'attempts': 1, 'retrieved_at': datetime.now(timezone.utc).isoformat()}


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def _new_output(output_dir, attempt_root):
    assigned_root = Path(attempt_root).absolute()
    root = assigned_root.resolve(strict=True)
    _require(root.is_dir(), 'Attempt root must be a directory')
    output = Path(output_dir).absolute()
    # The coordinator's root assignment is trusted, including its symlink alias.
    # Symlinks inside that root remain forbidden for output destinations.
    if output.is_relative_to(assigned_root):
        output = root / output.relative_to(assigned_root)
    _require(output.resolve() == output and output.is_relative_to(root) and output != root,
             'Output must be a new strict descendant of the attempt root without symlinks')
    _require(not output.is_relative_to(REPO_ROOT), 'Data outputs cannot enter the repository')
    _require(output.parent.is_dir(), 'Output parent must already exist')
    output.mkdir(mode=0o700, exist_ok=False)
    return output


def fetch_manifest(manifest, output_dir, *, attempt_root, attempts=3, timeout=30):
    """Fetch trusted contracts into a fresh private child of an owned attempt root.

    For synthetic callers/tests only, a manifest may be supplied directly. The
    CLI does not accept arbitrary manifest files or shared cache destinations.
    """
    validate_manifest(manifest)
    if manifest['status'] == 'blocked':
        raise ManifestError('Blocked manifest: ' + '; '.join(manifest['blockers']))
    _require(type(attempts) is int and 1 <= attempts <= 5, 'attempts must be 1..5')
    _require(type(timeout) in (int, float) and 0 < timeout <= 120, 'timeout must be in (0, 120]')
    output = _new_output(output_dir, attempt_root)
    stage = output / '.stage'
    stage.mkdir(mode=0o700)
    receipt = {'schema_version': 1, 'manifest_id': manifest['id'],
               'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(),
               'release_identity': manifest['release_identity'], 'status': 'incomplete', 'resources': []}
    archive = output / 'payload.tar'
    try:
        with (output / 'download.log').open('x') as log:
            try:
                for index, resource in enumerate(manifest['resources']):
                    path = stage / str(index)
                    entry = (_copy_local(resource, path, log=log) if resource.get('transport') == 'local'
                             else _download(resource, path, attempts=attempts, timeout=timeout, log=log))
                    receipt['resources'].append(entry)
                    log.write(f"audited {entry['id']} bytes={entry['bytes']} sha256={entry['sha256']}\n")
                with tarfile.open(output / 'payload.tar.part', 'w', format=tarfile.PAX_FORMAT) as tar:
                    for index, resource in enumerate(manifest['resources']):
                        path = stage / str(index)
                        info = tarfile.TarInfo(resource['destination'])
                        info.size = path.stat().st_size
                        info.mode = 0o600
                        with path.open('rb') as stream:
                            tar.addfile(info, stream)
                (output / 'payload.tar.part').rename(archive)
                receipt.update(status='complete', payload_sha256=_hash_file(archive),
                               payload_bytes=archive.stat().st_size)
                log.write('complete\n')
            except Exception as exc:
                receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                log.write(f'failed {type(exc).__name__}: {exc}\n')
                archive.unlink(missing_ok=True)
                (output / 'payload.tar.part').unlink(missing_ok=True)
                raise
    finally:
        shutil.rmtree(stage)
        (output / 'receipts.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'fetch'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    try:
        manifest = load_source(args.source)
        if args.command == 'validate':
            print(json.dumps({'id': manifest['id'], 'status': manifest['status'],
                              'blockers': manifest['blockers']}))
            return 0
        _require(args.output_dir is not None, '--output-dir is required for fetch')
        root = os.environ.get('SWARM_UNIT_DIR')
        _require(root, 'SWARM_UNIT_DIR is required; no shared-root destination is supported')
        receipt = fetch_manifest(manifest, args.output_dir, attempt_root=root,
                                 attempts=args.attempts, timeout=args.timeout)
        print(json.dumps({'status': receipt['status'], 'output_dir': str(args.output_dir)}))
        return 0
    except (ManifestError, OSError, urllib.error.URLError, http.client.HTTPException) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
