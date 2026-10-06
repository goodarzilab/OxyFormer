"""Offline wiring regressions; synthetic archives, never historical computation."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest
import yaml

from oxyformer import legacy
from oxyformer.legacy import historical, tasks as stage
from oxyformer.provenance import ArtifactRecord, ContractError, file_hash
from test_legacy import request_at, synthetic_archive, offline

ROOT = Path(__file__).resolve().parents[1]


def tasks():
    return yaml.safe_load((ROOT / stage.TASK_FILE).read_text())['tasks']


def bind_request(tmp_path, task, roots, files):
    request = request_at(tmp_path, task.get('parameters', {'mode': 'elevcan-reproduction'}),
                         files, stage=task['stage'])
    Path(request.task_path).write_text(json.dumps(task))
    Path(request.config_path).write_text(json.dumps({'dependencies': {k: str(v) for k, v in roots.items()}}))
    return replace(request, task_hash=file_hash(request.task_path), config_hash=file_hash(request.config_path))


@pytest.fixture
def published_inputs(tmp_path, monkeypatch):
    root = tmp_path / 'acquisition'; root.mkdir()
    raw = b'synthetic archive bytes'
    config = deepcopy(stage.configuration())
    pin = config['sources']['elevcan-reproduction']
    pin.update(archive_bytes=len(raw), archive_sha256=sha256(raw).hexdigest())
    monkeypatch.setattr(stage, 'configuration', lambda: config)
    payload = root / 'payload.tar'
    with tarfile.open(payload, 'w') as archive:
        for name, data in [('us/acs_tracts.tar.gz', b'do not read this member'),
                           ('us/elevcan_source.tar.gz', raw)]:
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    receipt = root / 'receipts.json'
    receipt.write_text(json.dumps({'status': 'complete', 'manifest_id': 'us',
        'payload_sha256': file_hash(payload), 'payload_bytes': payload.stat().st_size, 'resources': [{
            'id': 'elevcan_source', 'destination': 'us/elevcan_source.tar.gz',
            'bytes': len(raw), 'sha256': pin['archive_sha256'], 'url': pin['archive_url']}]}))
    req = bind_request(tmp_path / 'request', tasks()[0], {'fetch-us': root}, [payload, receipt])
    return req, raw, config


def test_only_honestly_supplied_tasks_registered():
    registry = yaml.safe_load((ROOT / 'configs/execution/stages.yaml').read_text())['stages']
    assert [t['id'] for t in tasks()] == ['legacy-inputs', 'elevcan-repro']
    for task in tasks():
        assert task['needs'] == registry[task['stage']]['needs']
        assert task['outputs'] == registry[task['stage']]['outputs']
    parameters = tasks()[1]['parameters']
    assert parameters['mode'] == 'elevcan-reproduction'
    assert parameters['allow_download'] is False
    ref = parameters['source_archive']
    assert ref['path'] in tasks()[1]['needs'][ref['dependency']]
    assert Path(parameters['rscript']).is_absolute()
    # Shell logging starts before admission, so it must not be a stage output.
    assert 'run.log' not in tasks()[1]['outputs']
    assert '> "$SWARM_UNIT_DIR/run.log" 2>&1' in (ROOT / 'scripts/slurm/run_stage.sh').read_text()


def test_publish_only_exact_pinned_member(published_inputs, monkeypatch):
    req, raw, _ = published_inputs
    # A generic archive extraction would read ACS: the adapter must seek instead.
    monkeypatch.setattr(tarfile.TarFile, 'extractfile', lambda *a, **k: pytest.fail('must read selected member by offset'))
    result = stage.run_stage(req)
    result.verify(req)
    assert result.status == 'pass'
    out = Path(req.output_dir)
    assert (out / 'elevcan_source.tar.gz').read_bytes() == raw
    assert sorted(p.name for p in out.iterdir()) == ['elevcan_source.tar.gz', 'source_manifest.json']
    provenance = json.loads((out / 'source_manifest.json').read_text())
    assert provenance['sha256'] == sha256(raw).hexdigest()
    assert provenance['acquisition_receipt_sha256'] == file_hash(Path(req.dependency_paths[1]))


def test_member_bytes_must_match_pin_even_if_receipt_claims_it(published_inputs):
    req, raw, _ = published_inputs
    payload = Path(req.dependency_paths[0])
    content = payload.read_bytes().replace(raw, b'X' * len(raw))
    payload.write_bytes(content)
    req = replace(req, dependency_hashes=tuple(file_hash(p) for p in req.dependency_paths))
    with pytest.raises(ContractError, match='SHA-256 mismatch'):
        stage.run_stage(req)
    assert not list(Path(req.output_dir).iterdir())


@pytest.mark.parametrize('field,value', [('status','incomplete'), ('manifest_id','other')])
def test_wrong_acquisition_refused(published_inputs, field, value):
    req, _, _ = published_inputs
    p = Path(req.dependency_paths[1]); receipt=json.loads(p.read_text()); receipt[field]=value
    p.write_text(json.dumps(receipt))
    req=replace(req,dependency_hashes=tuple(file_hash(p) for p in req.dependency_paths))
    with pytest.raises(ContractError, match='complete US acquisition'):
        stage.run_stage(req)


@pytest.mark.parametrize('field,value', [('sha256','0'*64),('bytes',1),('url','https://example.invalid'),('destination','wrong')])
def test_changed_resource_receipt_refused(published_inputs, field, value):
    req, _, _ = published_inputs
    p=Path(req.dependency_paths[1]); receipt=json.loads(p.read_text()); receipt['resources'][0][field]=value
    p.write_text(json.dumps(receipt))
    req=replace(req,dependency_hashes=tuple(file_hash(p) for p in req.dependency_paths))
    with pytest.raises(ContractError, match='differs from pinned'):
        stage.run_stage(req)


def test_duplicate_selected_member_refused(published_inputs):
    req, raw, _ = published_inputs
    p=Path(req.dependency_paths[0])
    with tarfile.open(p,'a') as archive:
        member=tarfile.TarInfo('us/elevcan_source.tar.gz'); member.size=len(raw)
        archive.addfile(member,io.BytesIO(raw))
    req=replace(req,dependency_hashes=tuple(file_hash(p) for p in req.dependency_paths))
    with pytest.raises(ContractError, match='unique regular'):
        stage.run_stage(req)


def test_dependency_reference_resolves_only_declared_file(published_inputs):
    req, _, _ = published_inputs
    assert historical.dependency(req, {'dependency':'fetch-us','path':'payload.tar'}) == Path(req.dependency_paths[0])
    root=Path(req.dependency_paths[0]).parent
    (root/'undeclared').write_text('synthetic')
    with pytest.raises(ContractError, match='hashed StageRequest'):
        historical.dependency(req, {'dependency':'fetch-us','path':'undeclared'})
    with pytest.raises(ContractError, match='not declared'):
        historical.dependency(req, {'dependency':'wrong','path':'payload.tar'})
    with pytest.raises(ContractError):
        historical.dependency(req, {'dependency':'fetch-us','path':'../payload.tar'})


def test_reference_and_plan_outputs_preserve_historical_result(tmp_path, monkeypatch):
    archive, _, _ = synthetic_archive(tmp_path, monkeypatch)
    task={'id':'synthetic','stage':'legacy-reproduction','parameters':{
        'mode':'initial-release','entrypoint':'phase26',
        'source_archive':{'dependency':'source','path':archive.name}}}
    req=bind_request(tmp_path/'request',task,{'source':archive.parent},[archive])
    result=legacy.run_stage(req)
    assert result.status=='pass'
    result.verify(req)
    out=Path(req.output_dir)
    report=json.loads((out/'results.json').read_text())
    assert report==json.loads((out/'legacy/initial-release/report.json').read_text())
    assert report['numerical_agreement'] is None
    manifest=json.loads((out/'artifact_manifest.json').read_text())
    assert manifest['status']=='pass' and manifest['request_hash']==req.content_hash
    records=[ArtifactRecord.from_json(json.dumps(r)) for r in manifest['artifacts']]
    for record in records:
        assert file_hash(out/record.path)==record.sha256
    with tarfile.open(out/'reproduction_bundle.tar','r:') as bundle:
        names=bundle.getnames()
        assert 'results.json' in names
        for record in records:
            if record.path!='reproduction_bundle.tar':
                assert sha256(bundle.extractfile(record.path).read()).hexdigest()==record.sha256
        data=json.load(bundle.extractfile(next(n for n in names if n.endswith('outputs/phase26/result.json'))))
        assert data['historical_auxiliary_weight']==0.35


def test_blocked_stage_exports_diagnostic_not_success(tmp_path):
    req=request_at(tmp_path,{'mode':'initial-release','entrypoint':'phase1'})
    result=legacy.run_stage(req)
    assert result.status=='blocked'
    out=Path(req.output_dir)
    assert json.loads((out/'results.json').read_text())['status']=='blocked'
    assert json.loads((out/'artifact_manifest.json').read_text())['status']=='blocked'
    result.verify(req)

# Use the real admission/publishing lifecycle with a synthetic acquisition.
from test_execution import runtime, publication_authority, commit
from oxyformer.contracts import StageResult
from oxyformer.execution.runner import run


def test_runner_consumes_published_archive_with_portable_reference(runtime, published_inputs, monkeypatch):
    repo, out=runtime
    req, raw, _=published_inputs
    registry=yaml.safe_load((ROOT/'configs/execution/stages.yaml').read_text())
    (repo/'configs/execution/stages.yaml').write_text(yaml.safe_dump(registry))
    taskfile=repo/'configs/execution/legacy.yaml'
    taskfile.write_text((ROOT/stage.TASK_FILE).read_text())
    head=commit(repo); (out/'code_commit.txt').write_text(head+'\n')
    monkeypatch.setenv('SWARM_DEP_FETCH_US', str(Path(req.dependency_paths[0]).parent))
    result=run('legacy-inputs',out,repo,deps_env=True,task_file=taskfile,task_id='legacy-inputs',
               execute=lambda request,*_: stage.run_stage(request))
    assert result.status=='pass',result.message
    monkeypatch.setenv('SWARM_DEP_LEGACY_INPUTS',str(out))
    consumer=out.parent/'consumer'; consumer.mkdir()
    (consumer/'code_commit.txt').write_text(head+'\n')
    reached=[]
    def stop(request,*_):
        task=json.loads(Path(request.task_path).read_text())
        bound=historical.dependency(request,task['parameters']['source_archive'])
        assert bound==out/'elevcan_source.tar.gz'
        assert bound.read_bytes()==raw
        assert request.dependency_hashes[request.dependency_paths.index(str(bound))]==sha256(raw).hexdigest()
        reached.append(request)
        return StageResult(request_hash=request.content_hash,status='blocked',artifacts=(),message='ADMISSION_ONLY')
    result=run('elevcan-reproduction',consumer,repo,deps_env=True,task_file=taskfile,task_id='elevcan-repro',execute=stop)
    assert result.status=='blocked' and reached, result.message
