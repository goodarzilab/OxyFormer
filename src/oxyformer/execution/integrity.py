"""Read-only fingerprints of complete attempt trees, without following links."""
from hashlib import sha256
import os
from pathlib import Path
import stat


def _stable(metadata):
    # Times/inodes detect concurrent changes while reading; they are deliberately
    # not persistent fingerprint fields. Reading can update atime harmlessly.
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def fingerprint_tree(root):
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
