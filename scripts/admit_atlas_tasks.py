"""Admission-only atlas probe: real inputs, synthetic products, no raster work.

Never wire these synthetic attempt directories into the scientific plan.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from oxyformer.contracts import StageResult
from oxyformer.execution import atlas_stage
from oxyformer.execution.atlas_tasks import TASK_FILE
from oxyformer.execution.runner import dependency_variable, run
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


def admit(repo, out, census, dem):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    require(not out.is_relative_to(repo), 'admission output must be outside the clone')
    document = json.loads((repo / TASK_FILE).read_text())
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
            # Only the scientific computation is replaced. All runner admission,
            # receipt hashing, fingerprinting, transitive checks and publishing run.
            with patch('oxyformer.exposure.build.run_stage', admission_execute):
                result = run(task['stage'], attempt, repo, deps_env=True, task_file=repo / TASK_FILE,
                    task_id=task['id'], execute=lambda request, module, clone: atlas_stage.run_stage(request))
            results.append(dict(task_id=task['id'], stage=task['stage'], status=result.status,
                                message=result.message, attempt=str(attempt)))
            (out / 'admission.json').write_text(json.dumps({'head': head, 'admission_only': True, 'results': results}, indent=2) + '\n')
            require(result.status == 'pass', f'{task["id"]}: {result.message}')
            os.environ[dependency_variable(task['id'])] = str(attempt)
            print(task['id'] + ': ADMISSION_PASS', flush=True)
    finally:
        os.environ.clear(); os.environ.update(environment)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--dem', type=Path, required=True)
    args = parser.parse_args()
    admit(args.repo, args.out, args.census, args.dem)


if __name__ == '__main__':
    main()
