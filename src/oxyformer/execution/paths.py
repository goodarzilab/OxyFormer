"""Attempt-owned files. These helpers never chmod or write an upstream attempt."""
from contextlib import ExitStack, contextmanager
import json
import os
import re
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from oxyformer.provenance import canonical_json, relative_artifact_path, require


def temporary_path(path):
    """Translate the active TMPDIR FD alias, retaining checks on its suffix."""
    path = Path(path).absolute()
    alias = os.environ.get('TMPDIR', '')
    if re.fullmatch(r'/proc/[0-9]+/fd/[0-9]+', alias) and path.is_relative_to(alias):
        suffix = path.relative_to(alias)
        require('..' not in suffix.parts, 'temporary path traversal')
        path = Path(alias).resolve(strict=True) / suffix
    return path


def output_path(root, relative):
    relative_artifact_path(relative)
    root = Path(root).resolve(strict=True)
    path = root / relative
    require(path.resolve().is_relative_to(root), 'output escapes attempt')
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
        'symlink in output path')
    return path


def atomic_write(root, relative, text):
    """Publish complete bytes once; hard-link publication cannot overwrite a peer."""
    path = output_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path


def atomic_json(root, relative, value):
    return atomic_write(root, relative, canonical_json(value))


def safe_extract(archive, root, relative, *, members=None, max_bytes=10 * 1024**3):
    """Extract selected regular files/directories into a NEW attempt directory."""
    target = output_path(root, relative)
    require(not target.exists(), 'extraction destination already exists')
    selected = None if members is None else set(members)
    if selected is not None:
        for name in selected:
            relative_artifact_path(name)
    entries = []
    from .integrity import open_regular
    with ExitStack() as stack:
        stream = stack.enter_context(open_regular(archive))
        is_zip = zipfile.is_zipfile(stream)
        stream.seek(0)
        handle = stack.enter_context(zipfile.ZipFile(stream) if is_zip else
            tarfile.open(fileobj=stream, mode='r:*'))
        seen = set()
        total = 0
        for item in handle.infolist() if is_zip else handle.getmembers():
            name = (item.filename if is_zip else item.name).rstrip('/')
            while name.startswith('./'):
                name = name[2:]
            directory = item.is_dir() if is_zip else item.isdir()
            if name != '.':
                relative_artifact_path(name)
            require(name not in seen, 'duplicate archive member')
            seen.add(name)
            if is_zip:
                mode = item.external_attr >> 16
                require((mode & 0o170000) in (0, 0o100000, 0o040000), 'archive special file')
                size = item.file_size
            else:
                require(item.isfile() or directory, 'archive special file or link')
                size = item.size
            if name == '.':
                require(directory, 'archive root entry must be a directory')
                continue
            if selected is None or name in selected:
                total += size
                require(total <= max_bytes, 'archive exceeds extraction byte limit')
                entries.append((name, item, directory))
        require(selected is None or selected <= seen, 'requested archive member missing')
        target.mkdir(parents=True)
        try:
            for name, item, directory in entries:
                dest = output_path(target, name)
                if directory:
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    source = handle.open(item) if is_zip else handle.extractfile(item)
                    with source, dest.open('xb') as sink:
                        shutil.copyfileobj(source, sink)
        except BaseException:
            shutil.rmtree(target)
            raise
    return target


@contextmanager
def isolated_caches(root):
    names = ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR',
        'NUMBA_CACHE_DIR', 'TRITON_CACHE_DIR', 'TMPDIR')
    previous = {name: os.environ.get(name) for name in names}
    previous_tempdir = tempfile.tempdir
    directory_fd = None
    try:
        for name in names:
            path = output_path(root, '_execution/cache/' + name.lower())
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        directory_fd = os.open(os.environ['TMPDIR'], os.O_RDONLY | os.O_DIRECTORY)
        # The supervisor retains this alias through all descendant shutdown.
        os.environ['TMPDIR'] = f'/proc/{os.getpid()}/fd/{directory_fd}'
        tempfile.tempdir = os.environ['TMPDIR']
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        if directory_fd is not None:
            os.close(directory_fd)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
