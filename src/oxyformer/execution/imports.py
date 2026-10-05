"""Load lazy first-party Python sources from the attested Git tree.

The CLI/worker bootstrap is trusted runner code. Before scientific dispatch,
install this finder for oxyformer imports (including lazy sibling imports).
Compile the verified source bytes directly; ignored bytecode is only a cache,
never an independent code authority. Third-party import behavior is unchanged.
"""
import hashlib
import importlib.abc
import importlib.machinery
from pathlib import Path
import sys

from oxyformer.provenance import require
from .identity import git
from .integrity import read_regular


class TrackedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, authority, fullname, path):
        super().__init__(fullname, str(path))
        self.authority = authority

    def get_code(self, fullname):
        # Retain normal source-loader/resource APIs, but never execute a pyc.
        source = self.authority.source(self.path)
        return compile(source, self.path, 'exec', dont_inherit=True)


class TrackedImports(importlib.abc.MetaPathFinder):
    def __init__(self, repo, commit):
        self.repo = Path(repo).resolve(strict=True)
        self.root = self.repo / 'src/oxyformer'
        self.algorithm = git(self.repo, 'rev-parse', '--show-object-format')
        self.blobs = {}
        for entry in git(self.repo, 'ls-tree', '-r', '-z', commit, '--', 'src/oxyformer').split('\0'):
            if not entry:
                continue
            metadata, name = entry.split('\t', 1)
            mode, kind, digest = metadata.split()
            if kind == 'blob' and mode in ('100644', '100755'):
                self.blobs[name] = digest

    def source(self, path):
        path = Path(path).absolute()
        require(path.is_relative_to(self.root) and path.suffix == '.py',
                f'first-party module is not a tracked Python source: {path}')
        expected = self.blobs.get(path.relative_to(self.repo).as_posix())
        require(expected is not None, f'untracked first-party source: {path}')
        source = read_regular(path)
        digest = hashlib.new(self.algorithm, b'blob ' + str(len(source)).encode() + b'\0' + source)
        require(digest.hexdigest() == expected, f'first-party source differs from attested commit: {path}')
        return source

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'oxyformer' and not fullname.startswith('oxyformer.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            raise ModuleNotFoundError(f'no tracked first-party module: {fullname}', name=fullname)
        if spec.origin is None and spec.submodule_search_locations is not None:
            for directory in spec.submodule_search_locations:
                directory = Path(directory).resolve(strict=True)
                require(directory.is_relative_to(self.root), 'namespace outside attested source root')
                prefix = directory.relative_to(self.repo).as_posix() + '/'
                require(any(name.startswith(prefix) for name in self.blobs),
                        f'untracked first-party namespace: {directory}')
            return spec
        self.source(spec.origin)
        spec.loader = TrackedSourceLoader(self, fullname, Path(spec.origin))
        return spec


def install_tracked_imports(repo, commit):
    authority = TrackedImports(repo, commit)
    # Already-loaded bootstrap modules must also be located in tracked source.
    for name, module in list(sys.modules.items()):
        if name == 'oxyformer' or name.startswith('oxyformer.'):
            location = getattr(module, '__file__', None)
            if location is not None:
                authority.source(location)
    sys.meta_path.insert(0, authority)
    return authority
