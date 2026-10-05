"""Recipe fingerprint v1: all tracked files except non-plan documentation.

The scope is fixed here, never selected by a task or recipe. docs/plan/** is
included. Other docs/** and *.md files are excluded. Everything else (including
all src, configs, scripts, build metadata and scientific root scripts) is bound.
"""
import importlib.metadata
from hashlib import sha256
from pathlib import Path
import platform
import subprocess
import sys

from oxyformer.provenance import canonical_json, require

FINGERPRINT_VERSION = 'tracked-science-v1'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def scientific_path(name):
    return name.startswith('docs/plan/') or not (name.startswith('docs/') or name.endswith('.md'))


def code_identity(repo, out):
    repo = Path(repo).resolve(strict=True)
    require(Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() == repo,
            'repo must be repository root')
    head = git(repo, 'rev-parse', 'HEAD')
    require((Path(out) / 'code_commit.txt').read_text().strip() == head,
            'code_commit.txt does not match cloned repository HEAD')
    require(not git(repo, 'status', '--porcelain', '--untracked-files=no'),
            'cloned repository has tracked modifications')
    untracked = git(repo, 'ls-files', '--others', '--exclude-standard', '-z').split('\0')
    require(not any(n and scientific_path(n) for n in untracked), 'untracked scientific code/config')
    return head


def scientific_fingerprint(repo):
    # Git blob IDs bind bytes, path, mode and additions/deletions; include SHA format.
    entries = git(repo, 'ls-tree', '-r', '-z', '--full-tree', 'HEAD').split('\0')
    scientific = sorted(e for e in entries if e and scientific_path(e.split('\t', 1)[1]))
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
        if location is None and getattr(module, '__path__', None) is not None:
            continue  # a namespace package; its loaded children are checked
        require(isinstance(location, str) and Path(location).resolve(strict=True).is_relative_to(root),
                f'loaded oxyformer module outside --repo: {location}')
