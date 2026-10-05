"""Parent-held control expectations and failed-attempt forensic publication."""
import os
from pathlib import Path
import stat
import tempfile

from oxyformer.provenance import require
from .paths import atomic_json, atomic_write


def entry_identity(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)


def file_identity(path):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
            f'execution control is not an unaliased regular file: {path.name}')
    return (*entry_identity(path), value.st_mtime_ns, value.st_ctime_ns)


class ControlRecords:
    """Only the parent holds this snapshot; workers cannot update expectations."""

    def __init__(self, out):
        self.out = Path(out)
        self.root_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.root = entry_identity(self.out)
            self.namespace = entry_identity(self.out / '_execution')
            names = ['code_commit.txt'] + [
                '_execution/' + name + '.json'
                for name in ('config', 'task', 'request', 'environment', 'identity')]
            self.records = {name: (file_identity(self.out / name), (self.out / name).read_bytes())
                            for name in names}
            self.verify()
        except BaseException:
            self.close()
            raise

    def close(self):
        os.close(self.root_fd)

    def verify(self):
        require(entry_identity(self.out) == self.root, 'attempt directory identity changed')
        require(entry_identity(self.out / '_execution') == self.namespace,
                'execution namespace changed')
        for name, (identity, contents) in self.records.items():
            path = self.out / name
            require(file_identity(path) == identity and path.read_bytes() == contents,
                    f'execution control changed: {name}')
        require(not os.path.lexists(self.out / '_execution/result.json'),
                'stage wrote the reserved execution result')

    def reject(self, reason):
        """Called only after proven quiescence. Never follow a substituted entry.

        Preserve the altered namespace, reconstruct parent originals, then let
        the caller publish FAIL. A failed reconstruction leaves no public
        receipt authority; an invalidation error propagates rather than being
        reported as successful rejection.
        """
        require(entry_identity(self.out)[:2] == self.root[:2], 'attempt root was replaced')
        # Restore search/write permission only on this current attempt, using
        # the original descriptor. Never chmod any dependency or link target.
        os.fchmod(self.root_fd, stat.S_IMODE(self.root[2]))
        quarantine = Path(tempfile.mkdtemp(prefix='.execution-rejected-', dir=self.out))
        quarantine_fd = os.open(quarantine, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        missing = []
        try:
            for name, destination in (('_execution', 'entry'), ('code_commit.txt', 'code_commit.txt')):
                try:
                    os.rename(name, destination, src_dir_fd=self.root_fd, dst_dir_fd=quarantine_fd)
                except FileNotFoundError:
                    missing.append(name)
        finally:
            os.close(quarantine_fd)
        atomic_json(quarantine, 'rejection.json', {'reason': reason, 'missing': missing})
        # The public execution entry is now absent, even if reconstruction
        # subsequently fails. The quarantined entry may be a symlink: do not
        # traverse it to recover expected bytes.
        (self.out / '_execution').mkdir()
        for name, (_, contents) in self.records.items():
            atomic_write(self.out, name, contents.decode('utf-8'))
