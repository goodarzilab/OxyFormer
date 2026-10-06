"""Build inspected atlas tasks, or probe admission with synthetic products.

Admission probes never compute exposure; do not wire their outputs into science.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import subprocess
import yaml
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest.mock import patch

if __name__ == '__main__' and not sys.dont_write_bytecode:
    raise SystemExit('Use python -B')

from oxyformer.contracts import StageResult
from oxyformer.exposure import tasks as atlas_stage
from oxyformer.exposure.tasks import TASK_FILE, build_tasks, inspect_dem, write_tasks
from oxyformer.execution.runner import dependency_variable, read_mapping, run
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, canonical_json, file_hash, require


def admission_execute(request):
    """Reach the science boundary, validate its bindings, emit explicit sentinels."""
    from oxyformer.exposure.build import _inventory, _receipt
    task = json.loads(Path(request.task_path).read_text())
    groups = _inventory(task, request)
    if request.stage == 'exposure-atlas':
        for name in ('census', 'dem'):
            receipt, _ = _receipt(request, task[name])
            require(receipt['manifest_id'] == name, 'acquisition binding points to wrong source')
        group = groups[task['shard_id']]
        require(set(task['raster_metadata']) == set(group['dem_resources']), 'shard raster bindings mismatch')
        shard_ids = [task['shard_id']]
    else:
        shard_ids = []
        for binding in task['shards']:
            parts = [json.loads(Path(request.dependency_paths[binding[role]]).read_text())
                     for role in ('manifest', 'exposure', 'quality')]
            require(parts[0] == parts[1] == parts[2] and parts[0]['admission_only'], 'synthetic collection binding mismatch')
            shard_ids.extend(parts[0]['shard_ids'])
        require(sorted(shard_ids) == sorted(groups), 'collection did not bind every shard exactly once')
    lineage = ArtifactLineage(source_hashes=(request.task_hash,), unit_ids=(task['id'],),
        parent_hashes=request.dependency_hashes, config_hash=request.config_hash, split_hash=None,
        model_hash=None, environment=(('purpose', 'ADMISSION_ONLY'),), seed=None, parameter_count=None)
    artifacts = []
    for name in task['outputs']:
        path = Path(request.output_dir) / name
        path.write_text(canonical_json({'admission_only': True, 'shard_ids': shard_ids}))
        artifacts.append(ArtifactRecord(path=name, sha256=file_hash(path), lineage=lineage, kind='admission_sentinel'))
    return StageResult(request_hash=request.content_hash, status='pass', artifacts=tuple(artifacts),
                       message='ADMISSION_ONLY; synthetic products; no exposure computed')


@contextmanager
def frozen_acquisitions(roots):
    """Hash real read-only acquisitions once for this *synthetic* probe.

    No production verifier is changed. The probe has no raster worker or input
    writer. Every cached read checks the entire acquisition's device/inode,
    mode, size, mtime and ctime. A second full content/tree hash at context exit
    is mandatory: timestamps alone can miss same-size writes. Only after that
    comparison may the probe be called a pass. No scientific products exist.
    """
    import oxyformer.execution.integrity as integrity
    import oxyformer.execution.runner as runner
    import oxyformer.provenance as provenance
    import oxyformer.contracts as contracts

    original_tree, original_hash = integrity.fingerprint_tree, integrity.regular_file_hash
    def signature(root):
        paths = [root, *sorted(root.rglob('*'))]
        return {str(p): tuple(getattr(p.lstat(), field) for field in
                ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')) for p in paths}
    snapshots = {}
    for root in map(lambda p: Path(p).resolve(), roots):
        before = signature(root)
        tree = original_tree(root)
        require(before == signature(root), 'acquisition changed while taking admission snapshot')
        require(not any('error' in entry for entry in tree.values()), 'unreadable admission snapshot')
        snapshots[root] = (before, tree)
        print(str(root) + ': full tree hashed', flush=True)
    def check(root):
        before, tree = snapshots[root]
        require(signature(root) == before, 'frozen acquisition changed during admission')
        return tree
    def tree(root, **kwargs):
        path = Path(root).resolve()
        if path not in snapshots:
            return original_tree(root, **kwargs)
        require(not kwargs.get('exclude'), 'admission acquisition snapshot cannot exclude files')
        return deepcopy(check(path))
    def digest(path):
        path = Path(path).resolve()
        if path.parent in snapshots and path.name == 'payload.tar':
            return check(path.parent)['payload.tar']['sha256']
        return original_hash(path)
    with ExitStack() as stack:
        stack.enter_context(patch.object(integrity, 'fingerprint_tree', tree))
        stack.enter_context(patch.object(runner, 'fingerprint_tree', tree))
        for module, name in [(integrity, 'regular_file_hash'), (runner, 'file_hash'),
                             (provenance, 'file_hash'), (contracts, 'file_hash')]:
            stack.enter_context(patch.object(module, name, digest))
        yield snapshots
    # Restore every helper before the independent final check, including for
    # fingerprint implementations that delegate hashing to module globals.
    for root in snapshots:
        check(root)
        require(original_tree(root) == snapshots[root][1], 'frozen acquisition changed during admission')
        print(str(root) + ': final full tree verified', flush=True)


def admit(repo, out, census, dem, *, finalize=True):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    require(not out.is_relative_to(repo), 'admission output must be outside the clone')
    document = yaml.safe_load((repo / TASK_FILE).read_text())
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    environment = dict(os.environ)
    results = []
    try:
        os.environ['SWARM_DEP_FETCH_CENSUS'] = str(Path(census).resolve())
        os.environ['SWARM_DEP_FETCH_DEM'] = str(Path(dem).resolve())
        os.environ['OXYFORMER_PUBLICATION_STORE'] = str(out / 'publications')
        for task in document['tasks']:
            attempt = out / task['id']; attempt.mkdir()
            (attempt / 'code_commit.txt').write_text(head + '\n')
            os.environ['SWARM_UNIT_DIR'] = str(attempt)
            # Replace science with sentinels. Runner admission, transitive
            # checks and publishing still run; --snapshot-cache memoizes immutable reads.
            with patch('oxyformer.exposure.build.run_stage', admission_execute):
                result = run(task['stage'], attempt, repo, deps_env=True, task_file=repo / TASK_FILE,
                    task_id=task['id'], execute=lambda request, module, clone: atlas_stage.run_stage(request))
            config = json.loads((attempt / '_execution/config.json').read_text())
            approvals = repo / 'configs/approvals.yaml'
            require(config['approvals'] == read_mapping(approvals), 'approval values changed in admission config')
            require(config['input_sources'][str(approvals)] == file_hash(approvals), 'approval hash binding changed')
            results.append(dict(task_id=task['id'], stage=task['stage'], status=result.status,
                message=result.message, attempt=str(attempt),
                approvals={'approved_on': config['approvals']['approved_on'],
                    'sha256': config['input_sources'][str(approvals)],
                    'yaml_timestamp_policy': config['yaml_timestamp_policy']}))
            (out / 'admission.json').write_text(json.dumps({'head': head, 'admission_only': True, 'status': 'incomplete', 'results': results}, indent=2) + '\n')
            require(result.status == 'pass', f'{task["id"]}: {result.message}')
            os.environ[dependency_variable(task['id'])] = str(attempt)
            print(task['id'] + ': ADMISSION_PASS', flush=True)
    finally:
        os.environ.clear(); os.environ.update(environment)
    if finalize:
        record = json.loads((out / 'admission.json').read_text())
        record['status'] = 'pass'
        (out / 'admission.json').write_text(json.dumps(record, indent=2) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem-acquisition', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--admit', action='store_true', help='admission only; write synthetic products')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--census-acquisition', type=Path)
    parser.add_argument('--snapshot-cache', action='store_true', help='memoize between two full tree hashes; admission only')
    args = parser.parse_args()
    if not args.admit:
        write_tasks(args.out, build_tasks(inspect_dem(args.dem_acquisition)))
        return
    require(args.census_acquisition is not None, '--admit requires --census-acquisition')
    from oxyformer.execution.identity import verified_checkout
    verified_checkout(args.repo)  # Refuse dirty code before large snapshot reads.
    if args.snapshot_cache:
        with frozen_acquisitions([args.census_acquisition, args.dem_acquisition]) as snapshots:
            admit(args.repo, args.out, args.census_acquisition, args.dem_acquisition, finalize=False)
        record = json.loads((args.out / 'admission.json').read_text())
        record.update(status='pass', verification='Full acquisition tree hashes before and after; stat checks on every cached reuse')
        (args.out / 'admission.json').write_text(json.dumps(record, indent=2) + '\n')
        (args.out / 'hash-cache.json').write_text(json.dumps({str(root): {'stat_signatures': signatures, 'tree': tree}
            for root, (signatures, tree) in snapshots.items()}, indent=2) + '\n')
    else:
        admit(args.repo, args.out, args.census_acquisition, args.dem_acquisition)


if __name__ == '__main__':
    main()
