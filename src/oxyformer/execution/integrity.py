"""Read-only fingerprints of attempt states, without following links.

Identity binds entries, types, modes, sizes, content hashes and symlink targets.
Timestamps and inode numbers are deliberately excluded: an identical state is
an identical consumer input, even after a rewrite. This is not a write log.
"""
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile

from oxyformer.contracts import StageResult
from oxyformer.provenance import ArtifactRecord, canonical_json, require
from .paths import atomic_json, atomic_write, output_path

# These two control records are written after the snapshot. The passing
# StageResult is the publication authority and binds FINGERPRINT by SHA-256.
FINGERPRINT = "_execution/fingerprint.json"
RESULT = "_execution/result.json"
DEPENDENCY_CHECK = "_execution/dependency_check.json"
PUBLICATION_EXCLUSIONS = (FINGERPRINT, RESULT)


def _stable(metadata):
    # Identity/times guard consistency during one read only. They are never
    # persisted or compared between fingerprint snapshots; atime is ignored.
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def directory_path(path):
    """Check directory components before resolving; never erase a symlink."""
    path = Path(path).absolute()
    for component in (*reversed(path.parents), path):
        require(stat.S_ISDIR(component.lstat().st_mode),
                f'input directory is a symlink or special file: {component}')
    return path.resolve(strict=True)


def regular_file_stat(path):
    path = Path(path)
    directory_path(path.parent)
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f'input is not a regular file: {path}')
    return metadata


@contextmanager
def open_regular(path):
    """Open one stable regular file without following links or blocking on FIFOs."""
    path = Path(path)
    before = regular_file_stat(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode) and _stable(opened) == _stable(before),
                f'input changed before reading: {path}')
        yield stream
        require(_stable(os.fstat(stream.fileno())) == _stable(before)
                and _stable(regular_file_stat(path)) == _stable(before),
                f'input changed while reading: {path}')


def read_regular(path):
    with open_regular(path) as stream:
        return stream.read()


def regular_file_hash(path):
    digest = sha256()
    with open_regular(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(request):
    """StageRequest.verify_inputs checks using nonblocking regular-file reads."""
    for path, digest in zip((request.config_path, request.task_path) + request.dependency_paths,
                            (request.config_hash, request.task_hash) + request.dependency_hashes):
        require(regular_file_hash(path) == digest, f'input hash mismatch: {path}')


def verify_result(result, request):
    """StageResult.verify checks using the same race-safe reader as preflight."""
    require(result.request_hash == request.content_hash, 'stage request mismatch')
    verify_inputs(request)
    root = directory_path(request.output_dir)
    for artifact in result.artifacts:
        path = (root / artifact.path).resolve(strict=True)
        require(path.is_relative_to(root), 'artifact escapes output directory')
        # Keep the declared spelling for the type check; resolution above is
        # only a confinement check and must not hide an in-attempt symlink.
        require(regular_file_hash(root / artifact.path) == artifact.sha256,
                f'artifact hash mismatch: {artifact.path}')


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
                         size=before.st_size, sha256=None, target=None)
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
    # Late control records are bound separately by their producer's published
    # digest, canonical result contents and recorded modes, avoiding self-hashes.
    return {name: entry for name, entry in entries.items()
            if name not in PUBLICATION_EXCLUSIONS}


def publication_tree(root):
    return publication_view(fingerprint_tree(root))


def _restore_control_permissions(path):
    """Restore private control entries, without following links to other trees."""
    metadata = path.lstat()
    directory = stat.S_ISDIR(metadata.st_mode)
    if not directory and not (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1):
        # A hardlink may share an upstream inode; atomic receipt replacement
        # can remove our name without changing that inode's permissions.
        return False
    mode = stat.S_IMODE(metadata.st_mode)
    restored = mode | (0o700 if directory else 0o600)
    changed = restored != mode
    if changed:
        os.chmod(path, restored, follow_symlinks=False)
    if directory:
        for child in path.iterdir():
            changed = _restore_control_permissions(child) or changed
    return changed


def _repair_control_directory(root):
    """Recover only the directory this runner reserved in its own attempt."""
    root = directory_path(root)
    path = root / '_execution'
    try:
        if stat.S_ISDIR(path.lstat().st_mode):
            return _restore_control_permissions(path)
        quarantine = Path(tempfile.mkdtemp(prefix='.control-collision-', dir=root))
        os.rename(path, quarantine / path.name)  # move the entry, never its target
    except FileNotFoundError:
        pass
    path.mkdir()
    return True


def _replace_control(root, relative, text):
    """Publish the current runner's control without opening a colliding entry.

    Only these three final receipt names are replaceable. Directory collisions
    are retained inside this attempt; symlinks/FIFOs are replaced as entries,
    never opened or followed. The caller owns the once-reserved _execution dir.
    """
    require(relative in (*PUBLICATION_EXCLUSIONS, DEPENDENCY_CHECK), 'not a publication control file')
    path = directory_path(root) / relative
    directory_path(path.parent)
    try:
        if stat.S_ISDIR(path.lstat().st_mode):
            quarantine = Path(tempfile.mkdtemp(prefix='.control-collision-', dir=path.parent))
            os.rename(path, quarantine / path.name)
    except FileNotFoundError:
        pass
    fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_result(root, result, *, owned_controls=False):
    """Seal a producer's own completed tree; a consumer never calls this.

    Reserve both late control filenames before measuring directory sizes. The
    receipt remains blocked until the fingerprint is complete and stable. Only
    the producer's own two reserved records are replaced, atomically.
    """
    root = directory_path(root)
    collisions = [str(root / name) for name in PUBLICATION_EXCLUSIONS
                  if os.path.lexists(root / name)]
    # Only the runner that reserved this fresh _execution directory may repair
    # its own receipt slots. Ordinary producer calls retain create-once semantics.
    require(owned_controls or not collisions, 'publication controls already exist')
    if collisions and result.status == 'pass':
        result = replace(result, status='fail', artifacts=(),
                         message='reserved publication control collision: ' + ', '.join(collisions))
    if result.status != 'pass':
        _replace_control(root, RESULT, result.to_json())
        return result
    pending = replace(result, status='blocked', artifacts=(), message='publication incomplete')
    _replace_control(root, RESULT, pending.to_json())
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
        fingerprint = ArtifactRecord(path=FINGERPRINT, sha256=regular_file_hash(root / FINGERPRINT),
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
    root = directory_path(root)
    records = [record for record in result.artifacts if record.path == FINGERPRINT]
    require(len(records) == 1 and records[0].kind == 'attempt_fingerprint',
            'dependency publication fingerprint missing')
    require(expected_hash is None or records[0].sha256 == expected_hash,
            'dependency published fingerprint identity changed')
    path = root / FINGERPRINT
    require(not path.is_symlink() and path.resolve().is_relative_to(root),
            'dependency fingerprint escapes attempt')
    raw = read_regular(path)
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
                entry.get('mode') != value['control_modes'].get(name)):
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
