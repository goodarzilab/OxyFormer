"""Read-only fingerprints of complete attempt trees, without following links."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile

from oxyformer.contracts import StageResult
from oxyformer.provenance import ArtifactRecord, canonical_json, file_hash, require
from .paths import atomic_json, atomic_write, output_path

# These two control records are written after the snapshot. The passing
# StageResult is the publication authority and binds FINGERPRINT by SHA-256.
FINGERPRINT = "_execution/fingerprint.json"
RESULT = "_execution/result.json"
PUBLICATION_EXCLUSIONS = (FINGERPRINT, RESULT)
# The publisher sets this on its OWN late records before atomic replacement.
# Consumers check the fixed convention, never adopt a current upstream mtime.
CONTROL_MTIME_NS = 0


def _stable(metadata):
    # Identity/times detect changes during the read. Persisted mtime/ctime also
    # expose restored writes; atime is excluded because ordinary reads change it.
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def fingerprint_tree(root, *, exclude=()):
    """Return every entry, including '.', and explicit errors on unreadable paths.

    Regular files bind bytes; symlinks bind the literal target (never its
    referent). Directories bind their names through the complete entry mapping.
    Special files are described but never opened. A caller must reject a
    snapshot containing any error, rather than treating missing data as empty.
    No upstream file is created, chmodded, restored or rewritten.
    """
    root = Path(root)
    entries = {}

    def visit(path, relative):
        if relative in exclude:
            return
        entry = {}
        entries[relative] = entry
        try:
            before = path.lstat()
            kind = stat.S_IFMT(before.st_mode)
            entry.update(type=kind, mode=stat.S_IMODE(before.st_mode),
                         size=before.st_size, sha256=None, target=None,
                         mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)
            if stat.S_ISLNK(kind):
                entry['target'] = os.readlink(path)
            elif stat.S_ISREG(kind):
                digest = sha256()
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, 'rb') as stream:
                    if _stable(os.fstat(stream.fileno())) != _stable(before):
                        raise OSError('entry changed before hashing')
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                    if _stable(os.fstat(stream.fileno())) != _stable(before):
                        raise OSError('entry changed while hashing')
                entry['sha256'] = digest.hexdigest()
            elif stat.S_ISDIR(kind):
                with os.scandir(path) as children:
                    names = sorted(child.name for child in children)
                for name in names:
                    visit(path / name, name if relative == '.' else relative + '/' + name)
            if _stable(path.lstat()) != _stable(before):
                raise OSError('entry changed while fingerprinting')
        except OSError as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'

    visit(root, '.')
    return entries


def changed_paths(before, after):
    """All added, removed, changed or unreadable entries, in stable order."""
    return sorted(name for name in before.keys() | after.keys()
                  if before.get(name) != after.get(name) or 'error' in after.get(name, {}))


def post_execution_check(before):
    """Recompute every dependency, including after an unsuccessful stage.

    Taint is recorded in the caller's receipt, never by writing to the
    dependency. Cross-consumer rejection additionally needs the trusted
    publication fingerprint carried into each consumer request.
    """
    attempts = {}
    for root, expected in before.items():
        actual = fingerprint_tree(root)
        changed = changed_paths(expected, actual)
        attempts[root] = {'status': 'tainted' if changed else 'unchanged',
                          'changed_paths': changed, 'fingerprint': actual}
    return {'status': 'fail' if any(a['changed_paths'] for a in attempts.values()) else 'pass',
            'attempts': attempts}


def publication_view(entries):
    # Publishing the two control records necessarily changes their parent
    # directory's timestamps. Its entries, type, mode and size remain bound.
    return {name: (dict(entry, mtime_ns=None, ctime_ns=None) if name == '_execution' else entry)
            for name, entry in entries.items() if name not in PUBLICATION_EXCLUSIONS}


def publication_tree(root):
    return publication_view(fingerprint_tree(root))


def _replace_control(root, relative, text):
    """Atomically replace only a control file this publisher already reserved."""
    require(relative in PUBLICATION_EXCLUSIONS, 'not a publication control file')
    path = output_path(root, relative)
    require(path.is_file(), 'publication control reservation missing')
    fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(temporary, ns=(CONTROL_MTIME_NS, CONTROL_MTIME_NS))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_result(root, result):
    """Seal a producer's own completed tree; a consumer never calls this.

    Reserve both late control filenames before measuring directory sizes. The
    receipt remains blocked until the fingerprint is complete and stable. Only
    the producer's own two reserved records are replaced, atomically.
    """
    root = Path(root).resolve(strict=True)
    if result.status != 'pass':
        atomic_write(root, RESULT, result.to_json())
        return result
    pending = replace(result, status='blocked', artifacts=(), message='publication incomplete')
    atomic_write(root, RESULT, pending.to_json())
    try:
        atomic_json(root, FINGERPRINT, {})
        for _ in range(3):
            entries = publication_tree(root)
            errors = [str(root / name) for name, entry in entries.items() if 'error' in entry]
            require(not errors, 'publication fingerprint unreadable: ' + ', '.join(errors))
            value = {'schema_version': 1, 'attempt': str(root),
                     'excluded': list(PUBLICATION_EXCLUSIONS), 'entries': entries,
                     'stage_result': result.to_dict(),
                     'control_modes': {name: stat.S_IMODE((root / name).lstat().st_mode)
                                       for name in PUBLICATION_EXCLUSIONS}}
            _replace_control(root, FINGERPRINT, canonical_json(value))
            if publication_tree(root) == entries:
                break
        else:
            raise ValueError('attempt changed during fingerprint publication')
        fingerprint = ArtifactRecord(path=FINGERPRINT, sha256=file_hash(root / FINGERPRINT),
                                     lineage=result.artifacts[0].lineage, kind='attempt_fingerprint')
        published = replace(result, artifacts=(*result.artifacts, fingerprint))
        _replace_control(root, RESULT, published.to_json())
        require(publication_tree(root) == entries, 'attempt changed during result publication')
        return published
    except BaseException as exc:
        failed = replace(result, status='fail', artifacts=(),
                         message='publication failed: ' + (str(exc).strip() or type(exc).__name__))
        _replace_control(root, RESULT, failed.to_json())
        return failed


def verify_published_tree(root, result, expected_hash=None):
    """Read the recorded baseline, hash-check it, then compare current entries.

    The expected digest comes from the producer's passing StageResult, never
    from hashing current data to establish a replacement baseline. The caller
    also carries this record and its expected digest into StageRequest.
    """
    root = Path(root)
    records = [record for record in result.artifacts if record.path == FINGERPRINT]
    require(len(records) == 1 and records[0].kind == 'attempt_fingerprint',
            'dependency publication fingerprint missing')
    require(expected_hash is None or records[0].sha256 == expected_hash,
            'dependency published fingerprint identity changed')
    path = root / FINGERPRINT
    require(not path.is_symlink() and path.resolve().is_relative_to(root),
            'dependency fingerprint escapes attempt')
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == records[0].sha256, 'dependency fingerprint hash mismatch')
    value = json.loads(raw)
    require(isinstance(value, dict) and set(value) == {
                'schema_version', 'attempt', 'excluded', 'entries', 'stage_result', 'control_modes'}
            and value['schema_version'] == 1 and value['attempt'] == str(root)
            and value['excluded'] == list(PUBLICATION_EXCLUSIONS)
            and isinstance(value['entries'], dict), 'invalid dependency fingerprint record')
    original_result = replace(result, artifacts=tuple(a for a in result.artifacts if a.path != FINGERPRINT))
    require(original_result.to_dict() == value['stage_result'], 'dependency result record changed since publication')
    require(bool(original_result.artifacts) and
            records[0].lineage == original_result.artifacts[0].lineage,
            'dependency fingerprint artifact lineage changed since publication')
    actual = fingerprint_tree(root)
    changed = changed_paths(value['entries'], publication_view(actual))
    # The late records cannot hash themselves. Bind the result's canonical
    # contents through the embedded original StageResult, and bind the manifest
    # through its published ArtifactRecord. Check type/mode and the same read.
    for name in PUBLICATION_EXCLUSIONS:
        entry = actual.get(name, {})
        if ('error' in entry or entry.get('type') != stat.S_IFREG or
                entry.get('mode') != value['control_modes'].get(name) or
                entry.get('mtime_ns') != CONTROL_MTIME_NS):
            changed.append(name)
    require(actual.get(FINGERPRINT, {}).get('sha256') == records[0].sha256,
            'dependency fingerprint changed during verification')
    require(actual.get(RESULT, {}).get('sha256') == sha256(result.to_json().encode()).hexdigest(),
            'dependency result record changed during verification')
    require(not changed, 'dependency fingerprint mismatch (tainted): ' +
            ', '.join(str(root / name) for name in sorted(set(changed))))
    # This is the exact tree that passed publication comparison, including
    # the late controls, not a second unverified read that can rebase a race.
    return actual
