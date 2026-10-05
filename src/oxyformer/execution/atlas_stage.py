"""Adapt the common runner to the unchanged exposure stage's config/work area."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

from oxyformer.contracts import StageResult
from oxyformer.execution.atlas_tasks import ROOT, TASK_FILE, build_tasks, inspect_dem
from oxyformer.execution.paths import atomic_json, atomic_write
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, file_hash, require


def run_stage(request):
    request.verify_inputs()
    task = json.loads(Path(request.task_path).read_text())
    committed = json.loads((ROOT / TASK_FILE).read_text())
    expected = next((t for t in committed['tasks'] if t['id'] == task['id']), None)
    require(task == expected and task['stage'] == request.stage, 'atlas task differs from reviewed configuration')
    out = Path(request.output_dir)
    if request.stage == 'atlas-inputs':
        payload, receipt = map(Path, request.dependency_paths[:2])
        require(payload.name == 'payload.tar' and receipt.name == 'receipts.json' and payload.parent == receipt.parent,
                'atlas input dependency bindings mismatch')
        inspected = inspect_dem(payload.parent)
        require(build_tasks(inspected) == committed, 'actual DEM headers differ from reviewed atlas tasks')
        atomic_write(out, 'atlas_shards.json', (ROOT / 'configs/sources/atlas_shards.json').read_text())
        atomic_json(out, 'raster_metadata.json', inspected)
        lineage = ArtifactLineage(source_hashes=request.dependency_hashes, unit_ids=('atlas-inputs',),
            parent_hashes=(request.task_hash,), split_hash=None, config_hash=request.config_hash,
            model_hash=None, environment=(('code_identity', request.code_identity),), seed=None, parameter_count=None)
        return StageResult(request_hash=request.content_hash, status='pass', message='Inspected atlas inputs published',
            artifacts=tuple(ArtifactRecord(path=name, sha256=file_hash(out / name), lineage=lineage, kind='atlas_input')
                            for name in task['outputs']))
    require(request.stage in ('exposure-atlas', 'atlas-collect'), 'unexpected atlas stage')
    if request.stage == 'exposure-atlas':
        inspected = json.loads(Path(request.dependency_paths[task['inspected_metadata']]).read_text())
        require(task['raster_metadata'] == {rid: inspected['raster_metadata'][rid] for rid in task['raster_metadata']},
                'task metadata differs from inspected upstream headers')
        require(inspected['dem_payload_sha256'] == request.dependency_hashes[task['dem']['payload']],
                'inspected metadata belongs to a different DEM payload')
    # The science API owns an empty product directory and expects exposure.yaml,
    # whereas the runner owns _execution and supplies execution configuration.
    # Persist the derived request and configuration for audit, then rebind only
    # the returned request identity when moving its declared products outward.
    config = atomic_write(out, '_atlas/exposure.yaml', (ROOT / 'configs/exposure.yaml').read_text())
    with tempfile.TemporaryDirectory(prefix='products-', dir=out / '_atlas') as scratch:
        inner = replace(request, config_path=str(config), config_hash=file_hash(config), output_dir=scratch)
        atomic_write(out, '_atlas/request.json', inner.to_json())
        from oxyformer.exposure.build import run_stage as exposure_stage
        result = exposure_stage(inner)
        result.verify(inner)
        for artifact in result.artifacts:
            target = out / artifact.path
            require(not target.exists(), 'atlas output already exists')
            shutil.move(str(Path(scratch) / artifact.path), target)
        return replace(result, request_hash=request.content_hash)
