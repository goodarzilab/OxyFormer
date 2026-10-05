"""Recipe fingerprint v2 covers tracked files except *.md outside src/,
scripts/ and docs/plan/. Tasks cannot override this scope.
"""
import importlib.metadata
from hashlib import new as new_hash, sha256
import os
import stat
from pathlib import Path
import platform
import subprocess
import sys

from oxyformer.provenance import canonical_json, require
from .integrity import open_regular, read_regular

FINGERPRINT_VERSION = 'tracked-science-v2'


def git_bytes(repo, *args):
    # Inspection must never refresh a sealed dependency's index on disk.
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repo), *args])


def git(repo, *args):
    return git_bytes(repo, *args).decode('utf-8').strip()


def scientific_path(name):
    return name.startswith(('src/', 'scripts/', 'docs/plan/')) or not name.endswith('.md')


def verified_checkout(repo):
    """Bind the actual importable checkout to HEAD, including ignored files.

    Git status alone misses ignored additions and assume-unchanged/skip-worktree
    edits. Compare disk blobs directly to HEAD and reject links in code roots:
    a tracked link cannot attest the code or resources at its referent. Stage
    launchers disable bytecode writes; no untracked cache is an identity bypass.
    """
    repo = Path(repo).resolve(strict=True)
    require(Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() == repo,
            'repo must be repository root')
    require(not git(repo, 'status', '--porcelain', '--untracked-files=no'),
            'cloned repository has tracked modifications')
    untracked = git(repo, 'ls-files', '--others', '-z').split('\0')
    extra = sorted(n for n in untracked if n and scientific_path(n))
    require(not extra, 'untracked scientific code/config (including ignored files): ' + ', '.join(extra))
    entries = git(repo, 'ls-tree', '-r', '-z', '--full-tree', 'HEAD').split('\0')
    algorithm = git(repo, 'rev-parse', '--show-object-format')
    for entry in filter(None, entries):
        metadata, name = entry.split('\t', 1)
        mode, kind, expected = metadata.split()
        path = repo / name
        require(kind == 'blob', f'unsupported code identity entry: {name}')
        before = path.lstat()
        if mode == '120000':
            require(not scientific_path(name),
                    f'code identity cannot follow a source symlink: {name}')
            require(stat.S_ISLNK(before.st_mode), f'tracked modifications: {name}')
            data = os.fsencode(os.readlink(path))
            digest = new_hash(algorithm, b'blob ' + str(len(data)).encode() + b'\0' + data)
        else:
            require(mode in ('100644', '100755') and stat.S_ISREG(before.st_mode)
                    and bool(before.st_mode & stat.S_IXUSR) == (mode == '100755'),
                    f'tracked modifications: {name}')
            digest = new_hash(algorithm, b'blob ' + str(before.st_size).encode() + b'\0')
            with open_regular(path) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
        require(digest.hexdigest() == expected, f'tracked modifications: {name}')
    return [e for e in entries if e]


def code_identity(repo, out):
    verified_checkout(repo)
    head = git(repo, 'rev-parse', 'HEAD')
    require(read_regular(Path(out) / 'code_commit.txt').decode('utf-8').strip() == head,
            'code_commit.txt does not match cloned repository HEAD')
    return head


def scientific_fingerprint(repo):
    # Use the same checked tree for recipe locks and execution attestations.
    entries = verified_checkout(repo)
    scientific = sorted(e for e in entries if scientific_path(e.split('\t', 1)[1]))
    return {'algorithm': FINGERPRINT_VERSION,
            'sha256': sha256(canonical_json(scientific).encode()).hexdigest()}


def verify_recipe(repo, lock):
    require(lock.get('scientific_fingerprint') == scientific_fingerprint(repo),
            'recipe scientific code/config drift')


def environment_record():
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(),
            'packages': sorted((d.metadata['Name'], d.version)
                               for d in importlib.metadata.distributions() if d.metadata['Name'])}


def verify_module_origins(repo, modules=None):
    """Attest Python source locations as well as the supplied Git checkout."""
    root = Path(repo).resolve(strict=True) / 'src'
    if modules is None:
        modules = [module for name, module in list(sys.modules.items())
                   if name == 'oxyformer' or name.startswith('oxyformer.')]
    for module in modules:
        location = getattr(module, '__file__', None)
        search_paths = getattr(module, '__path__', None)
        if search_paths is not None:
            for directory in search_paths:
                require(Path(directory).resolve(strict=True).is_relative_to(root),
                        f'loaded oxyformer namespace outside --repo: {directory}')
            if location is None:
                continue
        require(isinstance(location, str) and Path(location).resolve(strict=True).is_relative_to(root),
                f'loaded oxyformer module outside --repo: {location}')
