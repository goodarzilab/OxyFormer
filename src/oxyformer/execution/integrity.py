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
import time

from oxyformer.contracts import StageResult
from oxyformer.provenance import ArtifactRecord, ContractError, canonical_json, require, file_hash as provenance_file_hash
from .paths import atomic_json, atomic_write, output_path, temporary_path

FINGERPRINT = "_execution/fingerprint.json"
RESULT = "_execution/result.json"
DEPENDENCY_CHECK = "_execution/dependency_check.json"
PUBLICATION_EXCLUSIONS = (FINGERPRINT, RESULT)
_ACQUISITION_READS = ContextVar('acquisition_reads', default=())
_OBSERVATION_STATES = {}
_RECORDING_REFUSAL = ContextVar('recording_refusal', default=False)
_OBSERVATION_ENV = 'OXYFORMER_OBSERVATION_ATTEMPT'



_HASH_CACHE_ENV = 'OXYFORMER_RUN_HASH_CACHE'
_HASHING_PATHS = ContextVar('hashing_paths', default=())


@contextmanager
def stage_hash_cache():
    """One private cache for admission, workers and post-exit observation.

    Each invocation owns a fresh directory, even when nested in another run.
    Children inherit only its location. The runner reaps them before removing
    it; nothing is reused by a later stage or stored in acquisition attempts.
    """
    previous = os.environ.get(_HASH_CACHE_ENV)
    store = Path(os.environ.get('OXYFORMER_PUBLICATION_STORE',
        Path.home() / 'oxyformer-swarm/state/publications'))
    require(store.is_absolute(), 'publication store must be absolute')
    store.mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix='.run-hashes-', dir=directory_path(store))
    os.environ[_HASH_CACHE_ENV] = directory
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_HASH_CACHE_ENV, None)
        else:
            os.environ[_HASH_CACHE_ENV] = previous
        # Cache records are flat; no recursive traversal or worker cache paths.
        for name in os.listdir(directory):
            os.unlink(Path(directory) / name)
        os.rmdir(directory)


def _invalidate_digest(path):
    directory = os.environ.get(_HASH_CACHE_ENV)
    key = os.fspath(path)
    if directory is None or key in _HASHING_PATHS.get():
        return  # An enclosing cache transaction invalidates its own failure.
    cache = Path(directory) / sha256(key.encode()).hexdigest()
    with cache.open('a+b') as record:
        fcntl.flock(record.fileno(), fcntl.LOCK_EX)
        record.truncate(0)


def _hash_clock_barrier(record, path, before):
    """Close the ctime quantum before reading a version we may memoize.

    Stat timestamps are not change counters: two completed writes in one tick
    can have identical identities. A later timestamp on an independent inode
    of the SAME filesystem establishes an ordering boundary before our read.
    Use filesystem time, not the client's wall clock or an assumed tick size.
    """
    if os.fstat(record.fileno()).st_dev != before.st_dev:
        return False  # No same-filesystem witness: retain full verification.
    deadline = time.monotonic() + 2.0
    while True:
        os.fchmod(record.fileno(), 0o600)
        if os.fstat(record.fileno()).st_ctime_ns > before.st_ctime_ns:
            return True
        require(time.monotonic() < deadline,
            f'filesystem clock cannot order digest observation: {path}')
        time.sleep(0.001)


@contextmanager
def _cached_digest(path):
    """Reuse only a digest bracketed by identical kernel identities.

    The per-path lock covers the complete hash, so simultaneous descendants
    cannot each scan the same file. Exceptions invalidate the entry, including
    failed stats. A different path never shares an entry, even for hard links.
    The ordinary reader still owns namespace, content and refusal observations.
    """
    directory = os.environ.get(_HASH_CACHE_ENV)
    key = os.fspath(path)
    if directory is None or key in _HASHING_PATHS.get():
        yield [None]
        return
    token = _HASHING_PATHS.set((*_HASHING_PATHS.get(), key))
    try:
        cache = Path(directory) / sha256(key.encode()).hexdigest()
        with cache.open('a+b') as record:
            fcntl.flock(record.fileno(), fcntl.LOCK_EX)
            try:
                before = Path(path).lstat()
                if not stat.S_ISREG(before.st_mode):
                    raise InputChanged(path, 'input is not a regular file')
                identity = list(_stable(before))
                record.seek(0)
                raw = record.read()
                saved = json.loads(raw) if raw else None
                value = [saved['digest'] if saved is not None and saved['path'] == key
                    and saved['identity'] == identity else None]
                cacheable = value[0] is not None or _hash_clock_barrier(record, path, before)
                if cacheable and value[0] is None:
                    # Writeback reprotects already-dirty shared mmap pages.
                    # Do this AFTER closing the timestamp quantum: any later
                    # store must fault and acquire a newer ctime before reuse.
                    with open_regular(path) as stream:
                        os.fsync(stream.fileno())
                yield value
                after = Path(path).lstat()
                if _stable(after) != _stable(before):
                    raise _unstable_input(path, 'input changed while hashing', before, after)
                if cacheable and value[0] is not None:
                    record.seek(0)
                    record.truncate()
                    record.write(canonical_json({'path': key, 'identity': identity,
                        'digest': value[0]}).encode())
                    record.flush()
                elif not cacheable:
                    record.seek(0)
                    record.truncate()
                    record.flush()
            except BaseException:
                record.seek(0)
                record.truncate()
                record.flush()
                raise
    finally:
        _HASHING_PATHS.reset(token)


def _observation_state():
    root = os.environ.get(_OBSERVATION_ENV)
    if root is None:
        return None
    if root not in _OBSERVATION_STATES:
        state = _OBSERVATION_STATES[root] = {'root': root, 'trees': {}, 'failures': []}
        try:
            receipt = publication_receipt(root)
            value = json.loads(read_regular(str(receipt) + '.observation-inputs'))
            require(value['attempt'] == root, 'observation attempt mismatch')
            state['trees'] = value['trees']
        except BaseException as exc:
            state['failures'].append(str(exc))
            raise
    return _OBSERVATION_STATES[root]


def _record_refusal(path, error):
    if _RECORDING_REFUSAL.get() or _OBSERVATION_ENV not in os.environ:
        return
    token = _RECORDING_REFUSAL.set(True)
    try:
        state = _observation_state()
        message = f'{path}: {error}'
        state['failures'].append(message)
        receipt = publication_receipt(state['root'])
        try:
            atomic_json(receipt.parent, receipt.name + '.refused', {'message': message})
        except FileExistsError:
            pass  # The first refusal is sufficient and belongs only to this attempt.
    finally:
        _RECORDING_REFUSAL.reset(token)


def require_complete_observations(root):
    state = _observation_state()
    require(state is None or state['root'] == str(root), 'observation attempt mismatch')
    if state is not None:
        require(not state['failures'], 'integrity observation refused: ' + '; '.join(state['failures']))
    receipt = publication_receipt(root)
    marker = Path(str(receipt) + '.refused')
    if authority_exists(marker):
        require(False, 'integrity observation refused: ' + json.loads(read_regular(marker))['message'])


@contextmanager
def observe_dependencies(root, trees):
    """Bind worker reads and retain refusals for this attempt, never its retry."""
    root = str(root)
    receipt = publication_receipt(root, create=True)
    atomic_json(receipt.parent, receipt.name + '.observation-inputs', {'attempt': root, 'trees': trees})
    previous = os.environ.get(_OBSERVATION_ENV)
    _OBSERVATION_STATES[root] = {'root': root, 'trees': trees, 'failures': []}
    os.environ[_OBSERVATION_ENV] = root
    try:
        yield
    finally:
        _OBSERVATION_STATES.pop(root, None)
        if previous is None:
            os.environ.pop(_OBSERVATION_ENV, None)
        else:
            os.environ[_OBSERVATION_ENV] = previous


def run_stage(request):
    """Worker dispatch retains a caught reader failure even if its marker I/O fails."""
    import importlib
    state = _observation_state()
    require(state is not None and state['root'] == request.output_dir, 'worker observation binding missing')
    verify_input_hash(request.config_path, request.config_hash)
    config = json.loads(read_regular(request.config_path))
    module_name = config['settings']['module']
    require(module_name.startswith('oxyformer.') and module_name != __name__, 'invalid observed stage module')
    result = importlib.import_module(module_name).run_stage(request)
    require_complete_observations(request.output_dir)
    return result


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
    identities = check_dependency_identities(dependency_identities or {})
    refused = [str(Path(parent) / name) for parent, detail in identities.items()
        for name in detail['changed_paths'] + detail['unreadable_paths']]
    require(not refused, 'upstream path identity changed or unreadable: ' + ', '.join(refused))
    with publication_lock(receipt):
        require_complete_observations(root)
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
        tainted = detail.get('tainted_paths', detail['changed_paths'])
        if tainted:
            receipt = publication_receipt(root)
            with publication_lock(receipt):
                try:
                    atomic_json(receipt.parent, receipt.name + '.tainted', tainted)
                except FileExistsError:
                    pass  # Taint is permanent; a later observer cannot clear it.


def _acquisition_binding(path, *, use_active=True):
    """Find an active dependency or durable acquisition binding without resolving it.

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
    state = _observation_state() if use_active else None
    if state is not None:
        for root, entries in sorted(state['trees'].items(), key=lambda item: len(item[0]), reverse=True):
            root = Path(root)
            if path.is_relative_to(root):
                name = str(path.relative_to(root))
                if entries.get('.', {}).get('published') is not None:
                    if name not in entries and not name.startswith('_execution/'):
                        continue
                return root, name, entries
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


@contextmanager
def integrity_observation(path):
    """Retain a failed compound observation without attributing an I/O error."""
    try:
        yield
    except (OSError, ContractError) as exc:
        _record_refusal(path, exc)
        raise


def observe_acquisition(path, *, metadata=None, digest=None):
    """Compare known fields from a successful read with its durable binding."""
    binding = _acquisition_binding(path)
    if binding is None:
        return
    old = binding[2].get(binding[1])
    known = {} if metadata is None else dict(type=stat.S_IFMT(metadata.st_mode),
        mode=stat.S_IMODE(metadata.st_mode), size=metadata.st_size)
    if old is not None and old.get('partial') and known.get('type') == stat.S_IFDIR:
        known.pop('size', None)
    if digest is not None:
        known['sha256'] = digest
    if old is None or any(old.get(key) != value for key, value in known.items()):
        content_changed = old is None or any(old.get(key) != value
            for key, value in known.items() if key != 'mode')
        if content_changed:
            _taint_observation(binding)
        raise InputChanged(path, 'input hash mismatch' if digest is not None
            else 'input metadata differs from acquisition baseline', content_changed=content_changed)


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
    binding = None
    token = None
    try:
        binding = _acquisition_binding(path)
        if binding is not None:
            token = _ACQUISITION_READS.set((*_ACQUISITION_READS.get(), (binding[0], binding[2])))
        yield
    except (OSError, ContractError) as exc:
        try:
            _invalidate_digest(path)
            if (isinstance(exc, InputTypeError)
                    or isinstance(exc, InputChanged) and exc.content_changed
                    or isinstance(exc, OSError) and exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EISDIR)):
                if binding is not None and binding[2].get(binding[1], {}).get('type') == kind:
                    _taint_observation(binding)
        finally:
            _record_refusal(path, exc)
        raise
    finally:
        if token is not None:
            _ACQUISITION_READS.reset(token)


def verify_input_hash(path, expected, *, hash_file=None):
    """A wrong request digest is not evidence that the acquisition changed."""
    with acquisition_read(path):
        standard_reader = hash_file in (None, regular_file_hash, provenance_file_hash)
        if not standard_reader:
            # A callback may synchronously delegate to a child using the cache.
            # Retain descriptor/path checks without holding that child's lock.
            with open_regular(path):
                actual = hash_file(path)
        else:
            with _cached_digest(path) as cached:
                if cached[0] is None:
                    cached[0] = (regular_file_hash if hash_file is None else hash_file)(path)
                else:
                    # Reuse bytes, never namespace/type/readability observations.
                    with open_regular(path):
                        pass
                actual = cached[0]
        observe_acquisition(path, digest=actual)
        require(actual == expected, f'input hash mismatch: {path}')
    return actual


class InputChanged(ContractError):
    """A reader observed a change, as distinct from invalid input or I/O failure."""

    def __init__(self, path, message, *, content_changed=True):
        self.path = Path(path)
        self.content_changed = content_changed
        super().__init__(f'{message}: {path}')


def _content_metadata(metadata):
    # Timestamps, inode identity and permissions alone do not prove changed bytes.
    return (stat.S_IFMT(metadata.st_mode),
        metadata.st_size if stat.S_ISREG(metadata.st_mode) else None)



def _unstable_input(path, message, before, *after):
    content_changed = any(_content_metadata(item) != _content_metadata(before) for item in after)
    binding = _acquisition_binding(path)
    if not content_changed and binding is not None:
        # A timestamp alone is not content evidence. If bytes are still changed,
        # retain that positive observation before the caller can restore them.
        # Incomplete reads contribute only the fields actually observed.
        root, name, entries = binding
        current = fingerprint_tree(path)
        content_changed = bool(acquisition_changed_paths(
            {'.': entries[name]} if name in entries else {}, current))
    return InputChanged(path, message, content_changed=content_changed)


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


class _ObservedStream:
    """Retain the digest of bytes actually delivered by a complete stream read.

    Seeking remains supported without retaining file contents in memory. Only
    contiguous reads from offset zero establish a digest of the entire file.
    """

    def __init__(self, stream, path, size):
        self.stream, self.path, self.size = stream, path, size
        self.digest, self.offset = sha256(), 0

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def _read(self, method, *args, into=False):
        with acquisition_read(self.path):
            start = self.stream.tell()
            result = getattr(self.stream, method)(*args)
            raw = memoryview(args[0]).cast('B')[:result] if into else result
            if start == 0:
                self.digest, self.offset = sha256(), 0
            if start != self.offset:
                self.digest = None
            if self.digest is not None:
                self.digest.update(raw)
                self.offset += len(raw)
                if self.offset == self.size:
                    observe_acquisition(self.path, digest=self.digest.hexdigest())
            return result

    def read(self, size=-1):
        return self._read('read', size)

    def read1(self, size=-1):
        return self._read('read1', size)

    def readinto(self, buffer):
        return self._read('readinto', buffer, into=True)

    def readinto1(self, buffer):
        return self._read('readinto1', buffer, into=True)

    def readline(self, size=-1):
        return self._read('readline', size)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def readlines(self, hint=-1):
        lines, size = [], 0
        for line in self:
            lines.append(line)
            size += len(line)
            if hint > 0 and size >= hint:
                break
        return lines


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
                raise _unstable_input(path, 'input changed before reading', before, opened)
            yield _ObservedStream(stream, path, before.st_size)
            try:
                after = regular_file_stat(path)
            except InputTypeError as exc:
                raise InputChanged(path, 'input type or directory changed while reading') from exc
            except FileNotFoundError as exc:
                raise InputChanged(path, 'input removed while reading') from exc
            if (_stable(os.fstat(stream.fileno())) != _stable(before)
                    or _stable(after) != _stable(before)):
                raise _unstable_input(path, 'input changed while reading',
                    before, os.fstat(stream.fileno()), after)


def read_regular(path):
    with open_regular(path) as stream:
        raw = stream.read()
        observe_acquisition(path, digest=sha256(raw).hexdigest())
    return raw


def regular_file_hash(path):
    with acquisition_read(path), _cached_digest(path) as cached:
        with open_regular(path) as stream:
            if cached[0] is None:
                digest = sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                cached[0] = digest.hexdigest()
            value = cached[0]
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


def fingerprint_tree(root, *, exclude=(), include=None):
    """Fingerprint the selected subtrees and their directory ancestors.

    With no selection, include every entry (the acquisition boundary). Partial
    ancestors bind type and mode, but sibling writes cannot affect their size
    or timestamp observations. Entries omit timestamps and inodes; those are
    used only for read stability and separate per-run identity observations.
    """
    root = Path(root)
    entries = {}
    pending = [(root, '.', None)]
    selected = None if include is None else tuple(sorted(set(include)))

    def partial(relative):
        return selected is not None and not any(relative == name or relative.startswith(name + '/')
            for name in selected)

    def signature(metadata, relative):
        if partial(relative) and stat.S_ISDIR(metadata.st_mode):
            return (metadata.st_dev, metadata.st_ino, metadata.st_mode)
        return _stable(metadata)

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
                with _cached_digest(path) as cached:
                    digest = sha256()
                    fd = _open_observed_regular(path)
                    with os.fdopen(fd, 'rb') as stream:
                        opened = os.fstat(stream.fileno())
                        if _stable(opened) != _stable(before):
                            raise InputChanged(path, 'entry changed before hashing',
                                content_changed=_content_metadata(opened) != _content_metadata(before))
                        if cached[0] is None:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                                digest.update(chunk)
                            cached[0] = digest.hexdigest()
                        entry['sha256'] = cached[0]
                        after = os.fstat(stream.fileno())
                        if _stable(after) != _stable(before):
                            raise InputChanged(path, 'entry changed while hashing',
                                content_changed=_content_metadata(after) != _content_metadata(before))
            elif stat.S_ISDIR(kind):
                if partial(relative):
                    # Only traverse selected children; never open or enumerate
                    # coordinator/launcher siblings, even when unreadable.
                    entry.update(size=0, partial=True)
                    if relative == '.':
                        entry['published'] = list(selected)
                    prefix = '' if relative == '.' else relative + '/'
                    names = {name[len(prefix):].split('/')[0] for name in selected
                        if name.startswith(prefix)}
                    pending.append((path, relative, before))
                    pending.extend((path / name, prefix + name, None) for name in reversed(sorted(names)))
                    return
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
            after = path.lstat()
            if _stable(after) != _stable(before):
                raise InputChanged(path, 'entry changed while fingerprinting',
                    content_changed=_content_metadata(after) != _content_metadata(before))
        except (OSError, InputChanged) as exc:
            _invalidate_digest(path)
            if isinstance(exc, InputChanged) and exc.content_changed:
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
                if signature(after, relative) != signature(before, relative):
                    if _stable(after)[:4] == _stable(before)[:4]:
                        entries[relative]['timestamps_only'] = True
                    raise InputChanged(path, 'entry changed while fingerprinting',
                        content_changed=_content_metadata(after) != _content_metadata(before))
            except (OSError, InputChanged) as exc:
                if isinstance(exc, InputChanged) and exc.content_changed:
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
            known = {key: new[key] for key in ('type', 'size', 'target', 'sha256')
                if key in new and (key not in ('target', 'sha256') or new[key] is not None)}
            if (old is None or new.get('changed') or new.get('missing')
                    or any(old.get(key) != value for key, value in known.items())):
                changed.append(name)
        elif old is None or any(old.get(key) != new.get(key)
                for key in ('type', 'size', 'target', 'sha256')):
            changed.append(name)
    return sorted(changed)


def _dependency_identity(metadata):
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode), metadata.st_ctime_ns)


def snapshot_dependency_identities(trees):
    """Capture file identity and namespace witnesses outside content baselines."""
    identities = {}
    for root, entries in trees.items():
        files = identities[root] = {}
        for name, entry in entries.items():
            if entry.get('type') not in (stat.S_IFREG, stat.S_IFDIR):
                continue
            path = Path(root) / name
            record = files[name] = {'identity': _dependency_identity(path.lstat()), 'entry': entry}
            if entry.get('partial'):
                record['parent_identity'] = _dependency_identity(path.parent.lstat())
    return identities


def check_dependency_identities(identities, *, observed_changes=None):
    """Retain each kernel-observed change before checking another path.

    Reads may change atime; it and mtime are excluded. Complete directory
    ctimes retain rename/restore evidence. For shared ancestors, a rename also
    changes the containing directory's ctime; sibling file writes do not.
    Consult both witnesses, including on filesystems without remote inotify.
    Restoring bytes or mtime cannot restore kernel-maintained ctime.
    Identity alone refuses this run. Only observed content/namespace differences
    create permanent taint; failed content observations also remain retryable.
    """
    attempts = {}
    for root, files in identities.items():
        detail = attempts[root] = {'changed_paths': [], 'tainted_paths': [], 'unreadable_paths': []}
        for name, expected in files.items():
            path = Path(root) / name
            tainted = False
            try:
                metadata = path.lstat()
            except OSError as exc:
                _invalidate_digest(path)
                changed = tainted = exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
                if not changed:
                    detail['unreadable_paths'].append(name)
            else:
                identity = _dependency_identity(metadata)
                old = expected['entry']
                changed = identity != tuple(expected['identity'])
                if (changed and old.get('partial') and identity[:3] == tuple(expected['identity'])[:3]
                        and stat.S_IMODE(metadata.st_mode) == old['mode']):
                    try:
                        parent = _dependency_identity(path.parent.lstat())
                    except OSError:
                        detail['unreadable_paths'].append(name)
                        changed = False
                    else:
                        # Both witnesses changing is conservatively refused.
                        # Unrelated edits to both directories can also trigger
                        # this metadata-only refusal, but cannot taint content.
                        changed = parent != tuple(expected['parent_identity'])
                if changed:
                    tainted = (stat.S_IFMT(metadata.st_mode) != old['type']
                        or stat.S_ISREG(metadata.st_mode) and metadata.st_size != old['size'])
                    if not tainted and stat.S_ISREG(metadata.st_mode):
                        # Observe first, then persist once below. A marker-write
                        # failure must propagate, not be mistaken for input I/O.
                        current = fingerprint_tree(path)
                        tainted = bool(acquisition_changed_paths({'.': old}, current))
                        if any('error' in entry for entry in current.values()):
                            detail['unreadable_paths'].append(name)
            if changed:
                detail['changed_paths'].append(name)
                if observed_changes is not None:
                    observed_changes.append(str(path))
                if tainted:
                    detail['tainted_paths'].append(name)
                    record_taints({'attempts': {root: detail}})
    return attempts


def post_execution_check(before, *, observed_changes=None, identities=None):
    """Persist each dependency's evidence before checking the next dependency.

    The optional diagnostic list retains absolute witness paths even if later
    authority reads or marker writes fail before the check can be returned.
    """
    identities = check_dependency_identities(identities or {}, observed_changes=observed_changes)
    attempts = {}
    for root, expected in before.items():
        # Validate any acquisition authority before observing this tree.
        try:
            _acquisition_binding(root, use_active=False)
        except (OSError, ContractError) as exc:
            _record_refusal(root, exc)
            raise
        actual = fingerprint_tree(root, include=expected.get('.', {}).get('published'))
        # Unrelated I/O is not mutation evidence for either dependency kind.
        tainted = acquisition_changed_paths(expected, actual)
        changed = sorted(set(tainted) | {name for name, entry in actual.items()
            if 'error' not in entry and expected.get(name) != entry})
        identity = identities.get(root, {})
        changed = sorted(set(changed) | set(identity.get('changed_paths', [])))
        tainted = sorted(set(tainted) | set(identity.get('tainted_paths', [])))
        unreadable = any('error' in entry for entry in actual.values()) or bool(identity.get('unreadable_paths'))
        attempts[root] = {'status': 'tainted' if tainted else 'changed' if changed else 'unreadable' if unreadable else 'unchanged',
            'changed_paths': changed, 'tainted_paths': tainted, 'fingerprint': actual}
        if changed:
            if observed_changes is not None:
                observed_changes.extend(str(Path(root) / name) for name in changed)
            # A later dependency's I/O error must not erase this observation.
            record_taints({'attempts': {root: attempts[root]}})
        if changed or unreadable:
            witnesses = changed or [name for name, entry in actual.items() if 'error' in entry]
            _record_refusal(root, 'dependency check failed: ' + ', '.join(witnesses))
    return {'status': 'fail' if any(a['status'] != 'unchanged' for a in attempts.values()) else 'pass',
        'attempts': attempts}


def publication_view(entries):
    return {name: entry for name, entry in entries.items()
        if name not in PUBLICATION_EXCLUSIONS}


def _legacy_publication_scope(entries, selected):
    """Project an authenticated v1 baseline without rewriting it or its taint."""
    scoped = {}
    for name, entry in entries.items():
        full = any(name == path or name.startswith(path + '/') for path in selected)
        ancestor = name == '.' or any(path.startswith(name + '/') for path in selected)
        if full or ancestor:
            entry = dict(entry)
            if not full and entry.get('type') == stat.S_IFDIR:
                entry.update(size=0, partial=True)
                if name == '.':
                    entry['published'] = list(selected)
            scoped[name] = entry
    return scoped


def publication_tree(root, artifacts):
    return publication_view(fingerprint_tree(root, include=('_execution', *artifacts)))


def _settled_publication_tree(root, artifacts):
    """Allow our own control-directory timestamps to settle, never rebaseline.

    Weka can expose pre-rename mtime/ctime once after _replace_control, even
    after directory fsync. Only retry a directory-read instability in the
    runner-owned _execution directory. Every entry, byte, mode and size must
    remain identical to the first observation; other errors are not retried.
    Upstream fingerprinting remains strict and never calls this helper.
    """
    first = publication_tree(root, artifacts)
    control = first.get('_execution', {})
    if not (control.get('type') == stat.S_IFDIR and control.get('timestamps_only')
            and 'error' in control):
        return first
    expected = dict(first, _execution={key: value for key, value in control.items()
        if key not in ('error', 'changed', 'timestamps_only')})
    for _ in range(3):
        current = publication_tree(root, artifacts)
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
        artifacts = tuple(a.path for a in result.artifacts)
        for _ in range(3):
            entries = _settled_publication_tree(root, artifacts)
            errors = [str(root / name) for name, entry in entries.items() if 'error' in entry]
            require(not errors, 'publication fingerprint unreadable: ' + ', '.join(errors))
            value = {'schema_version': 2, 'attempt': str(root),
                'excluded': list(PUBLICATION_EXCLUSIONS), 'entries': entries,
                'stage_result': result.to_dict(),
                'control_modes': {name: stat.S_IMODE((root / name).lstat().st_mode)
                    for name in PUBLICATION_EXCLUSIONS}}
            _replace_control(root, FINGERPRINT, canonical_json(value))
            if _settled_publication_tree(root, artifacts) == entries:
                break
        else:
            raise ValueError('attempt changed during fingerprint publication')
        fingerprint = ArtifactRecord(path=FINGERPRINT, sha256=regular_file_hash(root / FINGERPRINT),
            lineage=result.artifacts[0].lineage, kind='attempt_fingerprint')
        published = replace(result, artifacts=(*result.artifacts, fingerprint))
        _replace_control(root, RESULT, published.to_json())
        require(_settled_publication_tree(root, artifacts) == entries, 'attempt changed during result publication')
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
    with integrity_observation(root):
        return _verify_published_tree(root, result, expected_hash)


def _verify_published_tree(root, result, expected_hash):
    root = directory_path(root)
    verify_publication(root, result)
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
        and value['schema_version'] in (1, 2) and value['attempt'] == str(root)
        and value['excluded'] == list(PUBLICATION_EXCLUSIONS)
        and isinstance(value['entries'], dict), 'invalid dependency fingerprint record')
    original_result = replace(result, artifacts=tuple(a for a in result.artifacts if a.path != FINGERPRINT))
    require(original_result.to_dict() == value['stage_result'], 'dependency result record changed since publication')
    require(bool(original_result.artifacts) and
        records[0].lineage == original_result.artifacts[0].lineage,
        'dependency fingerprint artifact lineage changed since publication')
    selected = tuple(sorted({'_execution', *(a.path for a in original_result.artifacts)}))
    expected = value['entries']
    if value['schema_version'] == 1:
        expected = _legacy_publication_scope(expected, selected)
    require(expected.get('.', {}).get('published') == list(selected),
        'dependency publication scope mismatch')
    actual = fingerprint_tree(root, include=selected)
    changed = changed_paths(expected, publication_view(actual))
    tainted = acquisition_changed_paths(expected, publication_view(actual))
    control_hashes = {FINGERPRINT: records[0].sha256, RESULT: sha256(result.to_json().encode()).hexdigest()}
    for name, digest in control_hashes.items():
        entry = actual.get(name, {})
        if (entry.get('missing') or entry.get('changed')
                or entry.get('type', stat.S_IFREG) != stat.S_IFREG
                or entry.get('sha256') is not None and entry['sha256'] != digest):
            tainted.append(name)
    record_taints({'attempts': {str(root): {'changed_paths': sorted(set(tainted))}}})
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
