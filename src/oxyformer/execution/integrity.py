"""Fingerprint entries, types, modes, sizes, bytes and symlink targets.
Content baselines are portable; per-run regular-file identities detect restored writes.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
import errno
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile

from oxyformer.contracts import StageResult
from oxyformer.provenance import ArtifactRecord, ContractError, canonical_json, require
from .paths import atomic_json, atomic_write, output_path, temporary_path

FINGERPRINT = "_execution/fingerprint.json"
RESULT = "_execution/result.json"
DEPENDENCY_CHECK = "_execution/dependency_check.json"
PUBLICATION_EXCLUSIONS = (FINGERPRINT, RESULT)
_ACQUISITION_READS = ContextVar('acquisition_reads', default=())


def publication_receipt(root, *, create=False):
    """Independent runner state; consumers never learn authority from payloads."""
    store = Path(os.environ.get('OXYFORMER_PUBLICATION_STORE',
            Path.home() / 'oxyformer-swarm/state/publications'))
    require(store.is_absolute(), 'publication store must be absolute')
    root = Path(os.path.abspath(root))
    require(not store.resolve().is_relative_to(root) and not root.is_relative_to(store.resolve()),
        'publication store overlaps attempt')
    if create:
        store.mkdir(parents=True, exist_ok=True)
    return directory_path(store) / sha256(str(root).encode()).hexdigest()


def authority_exists(path):
    """Only ENOENT means absent; resource and transport errors must propagate."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def publication_lock(receipt):
    """Serialize durable taints with the publication authority's commit point."""
    fd = os.open(receipt.parent / '.publication.lock',
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'publication lock is not a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def record_publication(root, result, *, dependency_roots=(), dependency_identities=None):
    receipt = publication_receipt(root, create=True)
    identities = check_file_identities(dependency_identities or {})
    refused = [str(Path(parent) / name) for parent, detail in identities.items()
        for name in detail['changed_paths'] + detail['unreadable_paths']]
    require(not refused, 'upstream file identity changed or unreadable: ' + ', '.join(refused))
    with publication_lock(receipt):
        # The authoritative release and permanent taints share one ordering.
        # A recorded upstream taint cannot slip between this check and commit.
        for dependency in dependency_roots:
            marker = Path(str(publication_receipt(dependency)) + '.tainted')
            if authority_exists(marker):
                paths = json.loads(read_regular(marker))
                require(False, 'upstream attempt tainted; changed paths: ' +
                    ', '.join(str(Path(dependency) / name) for name in paths))
        atomic_json(receipt.parent, receipt.name, {'attempt': str(root),
            'result_sha256': sha256(result.to_json().encode()).hexdigest()})


def verify_publication(root, result):
    receipt = publication_receipt(root)
    require(not authority_exists(str(receipt) + '.tainted'), f'tainted upstream fingerprint: {root}')
    require(json.loads(read_regular(receipt)) == {'attempt': str(root),
        'result_sha256': sha256(result.to_json().encode()).hexdigest()},
        f'dependency publication fingerprint authority mismatch: {root}')


def record_taints(check):
    for root, detail in check['attempts'].items():
        if detail['changed_paths']:
            receipt = publication_receipt(root)
            with publication_lock(receipt):
                try:
                    atomic_json(receipt.parent, receipt.name + '.tainted', detail['changed_paths'])
                except FileExistsError:
                    pass  # Taint is permanent; a later observer cannot clear it.


def _acquisition_binding(path):
    """Find a durable acquisition ancestor, without resolving the observed path.

    Both successful and refused reads consult the same immutable authority.
    The authority lives outside acquisitions and cannot recursively bind itself.
    """
    path = Path(os.path.abspath(path))
    store = Path(os.environ.get('OXYFORMER_PUBLICATION_STORE',
        Path.home() / 'oxyformer-swarm/state/publications'))
    if not store.is_absolute() or path.is_relative_to(store):
        return None
    for root, entries in reversed(_ACQUISITION_READS.get()):
        if path.is_relative_to(root):
            return root, str(path.relative_to(root)), entries
    for root in (path, *path.parents):
        if store.is_relative_to(root):
            continue
        baseline = store / (sha256(str(root).encode()).hexdigest() + '.acquisition')
        if authority_exists(baseline):
            value = json.loads(read_regular(baseline))
            require(value['attempt'] == str(root), 'acquisition baseline identity mismatch')
            return root, str(path.relative_to(root)), value['entries']
    return None


def _taint_observation(binding):
    root, relative, _ = binding
    record_taints({'attempts': {str(root): {'changed_paths': [relative]}}})


def observe_acquisition(path, *, metadata=None, digest=None):
    """Compare known fields from a successful read with its durable binding."""
    binding = _acquisition_binding(path)
    if binding is None:
        return
    old = binding[2].get(binding[1])
    known = {} if metadata is None else dict(type=stat.S_IFMT(metadata.st_mode),
        mode=stat.S_IMODE(metadata.st_mode), size=metadata.st_size)
    if digest is not None:
        known['sha256'] = digest
    if old is None or any(old.get(key) != value for key, value in known.items()):
        _taint_observation(binding)
        raise InputChanged(path, 'input hash mismatch' if digest is not None
            else 'input metadata differs from acquisition baseline')


class InputTypeError(ContractError):
    """A path did not have the type required by the read operation."""


@contextmanager
def acquisition_read(path, *, kind=stat.S_IFREG):
    """Preserve namespace/type observations at every reader, including workers.

    A prior durable binding distinguishes changed inputs from an invalid first
    use. Resource/transport errors do not establish mutation. Never rescan an
    observed absence: restoration before that rescan would erase the evidence.
    """
    # Prepare authority before observing input, and share it with nested
    # readers/comparisons. Never discover it after positive evidence exists.
    binding = _acquisition_binding(path)
    token = None
    if binding is not None:
        token = _ACQUISITION_READS.set((*_ACQUISITION_READS.get(), (binding[0], binding[2])))
    try:
        yield
    except (OSError, InputTypeError, InputChanged) as exc:
        if (isinstance(exc, (InputTypeError, InputChanged))
                or exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EISDIR)):
            if binding is not None and binding[2].get(binding[1], {}).get('type') == kind:
                _taint_observation(binding)
        raise
    finally:
        if token is not None:
            _ACQUISITION_READS.reset(token)


def verify_input_hash(path, expected, *, hash_file=None):
    """A wrong request digest is not evidence that the acquisition changed."""
    with acquisition_read(path):
        actual = (regular_file_hash if hash_file is None else hash_file)(path)
        observe_acquisition(path, digest=actual)
        require(actual == expected, f'input hash mismatch: {path}')
    return actual


class InputChanged(ContractError):
    """A reader observed a change, as distinct from invalid input or I/O failure."""

    def __init__(self, path, message):
        self.path = Path(path)
        super().__init__(f'{message}: {path}')


def _stable(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns)


def directory_path(path):
    """Check directory components before resolving; never erase a symlink."""
    path = temporary_path(path)
    with acquisition_read(path, kind=stat.S_IFDIR):
        for component in (*reversed(path.parents), path):
            metadata = component.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise InputTypeError(f'input directory is a symlink or special file: {component}')
            observe_acquisition(component, metadata=metadata)
        return path.resolve(strict=True)


def regular_file_stat(path):
    path = temporary_path(path)
    with acquisition_read(path):
        directory_path(path.parent)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise InputTypeError(f'input is not a regular file: {path}')
        observe_acquisition(path, metadata=metadata)
        return metadata


def _open_observed_regular(path):
    """Open after a successful regular-file stat, retaining namespace evidence."""
    try:
        return os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            raise InputChanged(path, 'input removed or replaced before reading') from exc
        raise  # Transport/resource failures alone do not establish a mutation.


@contextmanager
def open_regular(path):
    """Open one stable regular file without following links or blocking on FIFOs."""
    path = temporary_path(path)
    with acquisition_read(path):
        before = regular_file_stat(path)
        fd = _open_observed_regular(path)
        with os.fdopen(fd, 'rb') as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _stable(opened) != _stable(before):
                raise InputChanged(path, 'input changed before reading')
            yield stream
            try:
                after = regular_file_stat(path)
            except ContractError as exc:
                raise InputChanged(path, 'input type or directory changed while reading') from exc
            except FileNotFoundError as exc:
                raise InputChanged(path, 'input removed while reading') from exc
            if (_stable(os.fstat(stream.fileno())) != _stable(before)
                    or _stable(after) != _stable(before)):
                raise InputChanged(path, 'input changed while reading')


def read_regular(path):
    with open_regular(path) as stream:
        raw = stream.read()
        observe_acquisition(path, digest=sha256(raw).hexdigest())
    return raw


def regular_file_hash(path):
    digest = sha256()
    with open_regular(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        value = digest.hexdigest()
        observe_acquisition(path, digest=value)
    return value


def verify_inputs(request):
    """StageRequest.verify_inputs checks using nonblocking regular-file reads."""
    for path, digest in zip((request.config_path, request.task_path) + request.dependency_paths,
        (request.config_hash, request.task_hash) + request.dependency_hashes):
        verify_input_hash(path, digest)


def verify_result(result, request):
    """StageResult.verify checks using the same race-safe reader as preflight."""
    require(result.request_hash == request.content_hash, 'stage request mismatch')
    verify_inputs(request)
    root = directory_path(request.output_dir)
    for artifact in result.artifacts:
        path = (root / artifact.path).resolve(strict=True)
        require(path.is_relative_to(root), 'artifact escapes output directory')
        require(regular_file_hash(root / artifact.path) == artifact.sha256,
            f'artifact hash mismatch: {artifact.path}')


def fingerprint_tree(root, *, exclude=()):
    """Return every entry, including '.', and explicit errors on unreadable paths.

    Entries omit timestamps and inodes. Those are used only for within-read
    stability checks; the stored view binds types, modes, sizes, bytes and links.
    """
    root = Path(root)
    entries = {}
    pending = [(root, '.', None)]

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
                fd = _open_observed_regular(path)
                with os.fdopen(fd, 'rb') as stream:
                    if _stable(os.fstat(stream.fileno())) != _stable(before):
                        raise InputChanged(path, 'entry changed before hashing')
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                    if _stable(os.fstat(stream.fileno())) != _stable(before):
                        raise InputChanged(path, 'entry changed while hashing')
                entry['sha256'] = digest.hexdigest()
            elif stat.S_ISDIR(kind):
                names = []
                pending.append((path, relative, before))
                try:
                    with os.scandir(path) as children:
                        for child in children:
                            names.append(child.name)
                finally:
                    # A later enumeration error cannot erase names already
                    # observed, including additions restored before this visit.
                    pending.extend((path / name, name if relative == '.' else relative + '/' + name, None)
                        for name in reversed(sorted(names)))
                return
            if _stable(path.lstat()) != _stable(before):
                raise InputChanged(path, 'entry changed while fingerprinting')
        except (OSError, InputChanged) as exc:
            if isinstance(exc, InputChanged):
                entry['changed'] = True
            if isinstance(exc, OSError) and exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                entry['missing'] = True
                if relative != '.' or 'type' in entry:
                    entry['changed'] = True
            entry['error'] = f'{type(exc).__name__}: {exc}'

    while pending:
        path, relative, before = pending.pop()
        if before is None:
            visit(path, relative)
        else:
            try:
                after = path.lstat()
                if _stable(after) != _stable(before):
                    if _stable(after)[:4] == _stable(before)[:4]:
                        entries[relative]['timestamps_only'] = True
                    raise InputChanged(path, 'entry changed while fingerprinting')
            except (OSError, InputChanged) as exc:
                if isinstance(exc, InputChanged):
                    entries[relative]['changed'] = True
                if isinstance(exc, OSError) and exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                    entries[relative]['missing'] = True
                    entries[relative]['changed'] = True
                entries[relative]['error'] = f'{type(exc).__name__}: {exc}'
    return entries


def changed_paths(before, after):
    """All added, removed, changed or unreadable entries, in stable order."""
    return sorted(name for name in before.keys() | after.keys()
        if before.get(name) != after.get(name) or 'error' in after.get(name, {}))


def acquisition_changed_paths(before, after):
    """Return positive differences; an unreadable subtree is not a deletion."""
    changed = []
    for name in before.keys() | after.keys():
        old, new = before.get(name), after.get(name)
        if new is None:
            parents = ('.', *(str(p) for p in Path(name).parents if str(p) != '.'))
            if not any('error' in after.get(parent, {}) for parent in parents):
                changed.append(name)
        elif 'error' in new:
            known = {key: new[key] for key in ('type', 'mode', 'size', 'target', 'sha256')
                if key in new and (key not in ('target', 'sha256') or new[key] is not None)}
            if (old is None or new.get('changed') or new.get('missing')
                    or any(old.get(key) != value for key, value in known.items())):
                changed.append(name)
        elif old != new:
            changed.append(name)
    return sorted(changed)


def _file_identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_ctime_ns)


def snapshot_file_identities(trees):
    """Capture only regular files, once per run, outside portable content baselines."""
    return {root: {name: _file_identity(regular_file_stat(Path(root) / name))
        for name, entry in entries.items() if entry.get('type') == stat.S_IFREG}
        for root, entries in trees.items()}


def check_file_identities(identities, *, observed_changes=None):
    """Retain each kernel-observed change before checking another path.

    atime is changed by reads, and Weka directory timestamps may lag. Neither
    participates. Restoring bytes or mtime cannot restore regular-file ctime.
    Unknown I/O failures refuse without creating a permanent mutation marker.
    """
    attempts = {}
    for root, files in identities.items():
        detail = attempts[root] = {'changed_paths': [], 'unreadable_paths': []}
        for name, expected in files.items():
            path = Path(root) / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                changed = exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
                if not changed:
                    detail['unreadable_paths'].append(name)
            else:
                changed = not stat.S_ISREG(metadata.st_mode) or _file_identity(metadata) != tuple(expected)
            if changed:
                detail['changed_paths'].append(name)
                if observed_changes is not None:
                    observed_changes.append(str(path))
                record_taints({'attempts': {root: {'changed_paths': detail['changed_paths']}}})
    return attempts


def post_execution_check(before, *, observed_changes=None, identities=None):
    """Persist each dependency's evidence before checking the next dependency.

    The optional diagnostic list retains absolute witness paths even if later
    authority reads or marker writes fail before the check can be returned.
    """
    identities = check_file_identities(identities or {}, observed_changes=observed_changes)
    attempts = {}
    for root, expected in before.items():
        # Prepare the comparison authority before observing this tree.
        acquisition = _acquisition_binding(root) is not None
        actual = fingerprint_tree(root)
        changed = (acquisition_changed_paths if acquisition else changed_paths)(expected, actual)
        identity = identities.get(root, {})
        changed = sorted(set(changed) | set(identity.get('changed_paths', [])))
        unreadable = any('error' in entry for entry in actual.values()) or bool(identity.get('unreadable_paths'))
        attempts[root] = {'status': 'tainted' if changed else 'unreadable' if unreadable else 'unchanged',
            'changed_paths': changed, 'fingerprint': actual}
        if changed:
            if observed_changes is not None:
                observed_changes.extend(str(Path(root) / name) for name in changed)
            # A later dependency's I/O error must not erase this observation.
            record_taints({'attempts': {root: attempts[root]}})
    return {'status': 'fail' if any(a['status'] != 'unchanged' for a in attempts.values()) else 'pass',
        'attempts': attempts}


def publication_view(entries):
    return {name: entry for name, entry in entries.items()
        if name not in PUBLICATION_EXCLUSIONS}


def publication_tree(root):
    return publication_view(fingerprint_tree(root))


def _settled_publication_tree(root):
    """Allow our own control-directory timestamps to settle, never rebaseline.

    Weka can expose pre-rename mtime/ctime once after _replace_control, even
    after directory fsync. Only retry a directory-read instability in the
    runner-owned _execution directory. Every entry, byte, mode and size must
    remain identical to the first observation; other errors are not retried.
    Upstream fingerprinting remains strict and never calls this helper.
    """
    first = publication_tree(root)
    control = first.get('_execution', {})
    if not (control.get('type') == stat.S_IFDIR and control.get('timestamps_only')
            and 'error' in control):
        return first
    expected = dict(first, _execution={key: value for key, value in control.items()
        if key not in ('error', 'changed', 'timestamps_only')})
    for _ in range(3):
        current = publication_tree(root)
        if current == expected:
            return current
        detail = current.get('_execution', {})
        if not (detail.get('type') == stat.S_IFDIR and detail.get('timestamps_only') and 'error' in detail):
            break
        normalized = dict(current, _execution={key: value for key, value in detail.items()
            if key not in ('error', 'changed', 'timestamps_only')})
        if normalized != expected:
            break
    return first  # Preserve the refusal and the original evidence.


def _restore_control_permissions(path):
    """Restore private control entries, without following links to other trees."""
    pending = [path]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        directory = stat.S_ISDIR(metadata.st_mode)
        if not directory and not (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1):
            continue
        mode = stat.S_IMODE(metadata.st_mode)
        restored = mode | (0o700 if directory else 0o600)
        if restored != mode:
            os.chmod(path, restored, follow_symlinks=False)
        if directory:
            pending.extend(path.iterdir())


def _repair_control_directory(root):
    """Recover only the directory this runner reserved in its own attempt."""
    root = directory_path(root)
    path = root / '_execution'
    try:
        if stat.S_ISDIR(path.lstat().st_mode):
            _restore_control_permissions(path)
            return False  # Permission restoration is not a directory collision.
        quarantine = Path(tempfile.mkdtemp(prefix='.control-collision-', dir=root))
        os.rename(path, quarantine / path.name)  # move the entry, never its target
    except FileNotFoundError:
        pass
    path.mkdir()
    return True


def _replace_control(root, relative, text):
    """Publish the current runner's control without opening a colliding entry."""
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


def publish_result(root, result, *, owned_controls=False, dependency_roots=(), dependency_identities=None):
    """Seal a producer's own completed tree; a consumer never calls this."""
    root = directory_path(root)
    collisions = [str(root / name) for name in PUBLICATION_EXCLUSIONS
        if os.path.lexists(root / name)]
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
            entries = _settled_publication_tree(root)
            errors = [str(root / name) for name, entry in entries.items() if 'error' in entry]
            require(not errors, 'publication fingerprint unreadable: ' + ', '.join(errors))
            value = {'schema_version': 1, 'attempt': str(root),
                'excluded': list(PUBLICATION_EXCLUSIONS), 'entries': entries,
                'stage_result': result.to_dict(),
                'control_modes': {name: stat.S_IMODE((root / name).lstat().st_mode)
                    for name in PUBLICATION_EXCLUSIONS}}
            _replace_control(root, FINGERPRINT, canonical_json(value))
            if _settled_publication_tree(root) == entries:
                break
        else:
            raise ValueError('attempt changed during fingerprint publication')
        fingerprint = ArtifactRecord(path=FINGERPRINT, sha256=regular_file_hash(root / FINGERPRINT),
            lineage=result.artifacts[0].lineage, kind='attempt_fingerprint')
        published = replace(result, artifacts=(*result.artifacts, fingerprint))
        _replace_control(root, RESULT, published.to_json())
        require(_settled_publication_tree(root) == entries, 'attempt changed during result publication')
        record_publication(root, published, dependency_roots=dependency_roots,
            dependency_identities=dependency_identities)
        return published
    except BaseException as exc:
        failed = replace(result, status='fail', artifacts=(),
            message='publication failed: ' + (str(exc).strip() or type(exc).__name__))
        _replace_control(root, RESULT, failed.to_json())
        return failed


def verify_published_tree(root, result, expected_hash=None):
    """Read the recorded baseline, hash-check it, then compare current entries."""
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
    verify_publication(root, result)
    return actual
