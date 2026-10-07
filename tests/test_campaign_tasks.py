"""Offline artificial-input and runner-admission checks; no campaign execution."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from oxyformer.contracts import StageResult
from oxyformer.execution.runner import run, verify_dependency_result
from oxyformer.provenance import ContractError
from oxyformer.training import nested_cv
from oxyformer.validation import campaign, coverage, smoke_inputs
from oxyformer.validation.generators import generate_suite_a
from oxyformer.validation.scm import SCMConfig, CovariateFrame
from test_execution import runtime, publication_authority, commit

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'configs/execution/tasks/campaign.yaml'


def tasks():
    return {t['id']: t for t in yaml.safe_load(TASKS.read_text())['tasks']}


def test_registered_inputs_match_pinned_smoke_recipe():
    endpoint, frame = smoke_inputs.build_inputs()
    again = smoke_inputs.build_inputs()
    assert (endpoint.content_hash, frame.content_hash) == tuple(x.content_hash for x in again)
    parameters = tasks()['simulation-smoke']['parameters']
    assert parameters['recipe']['endpoint_hash'] == endpoint.content_hash
    assert parameters['recipe']['frame_hash'] == frame.content_hash
    campaign.validate_recipe(parameters['recipe'], production=False)
    with pytest.raises(ContractError, match='production tuning'):
        campaign.validate_recipe(parameters['recipe'])
    assert parameters['draws'] == coverage.repetition_plan(smoke_inputs.FIXTURE, 'null_effect', 1)
    scenario = SCMConfig(**parameters['scenario'])
    for filename in ('suite_a.yaml', 'final_scenarios.yaml'):
        registered = yaml.safe_load((ROOT / 'configs/validation' / filename).read_text())['scenarios']
        assert scenario in [SCMConfig(**r) for r in registered]
    assert 'campaign-lock' not in tasks()


def test_typed_inputs_rebind_and_preserve_all_nested_partitions(tmp_path):
    endpoint, frame = smoke_inputs.build_inputs()
    endpoint = nested_cv.PreparedEndpoint.from_json(endpoint.to_json())
    frame = CovariateFrame.from_json(frame.to_json())
    parameters = tasks()['simulation-smoke']['parameters']
    sample = generate_suite_a(frame, SCMConfig(**parameters['scenario']), endpoint.policy,
                             seed=parameters['draws'][0]['seed'])
    rebound = coverage.bind_observations(endpoint, sample.observations)
    assert rebound.data.covariates(('x',)).values == endpoint.data.covariates(('x',)).values
    assert rebound.outer.seed_ids == (1103, 2207, 3301)
    for fold in range(5):
        config = rebound.configuration(fold, tmp_path / str(fold),
                                       **nested_cv._settings(parameters['recipe'], rebound))
        assert set(config.inner.split.fold_ids) == {0, 1, 2}
        assert not set(config.inner.split.original_ids) & (set(rebound.outer.original_ids) - set(rebound.outer.training_ids(fold)))
    assert set(frame.region_ids) == {'c', 'd'}
    assert all(w == 1. for w in frame.weights)


def test_real_nested_fold_consumes_published_fixture(tmp_path):
    endpoint, frame = smoke_inputs.build_inputs()
    parameters = tasks()['simulation-smoke']['parameters']
    sample = generate_suite_a(frame, SCMConfig(**parameters['scenario']), endpoint.policy,
                             seed=parameters['draws'][0]['seed'])
    rebound = coverage.bind_observations(endpoint, sample.observations)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = rebound.configuration(0, tmp_path / 'fit',
            **nested_cv._settings(parameters['recipe'], rebound))
        artifact = nested_cv.run_fold(config, rebound.outer, 1103, geography=rebound.geography)
        assert artifact.complete
        view = nested_cv.subset(rebound.data.covariates(('x',)), artifact.prediction_inputs.original_ids)
        prediction = nested_cv.predict(artifact, view, rebound.policy)
        assert set(prediction.original_ids) == (set(rebound.outer.original_ids) - set(rebound.outer.training_ids(0)))
    finally:
        torch.set_num_threads(old)


def runner_setup(runtime):
    repo, out = runtime
    (repo / 'configs/execution/stages.yaml').write_text((ROOT / 'configs/execution/stages.yaml').read_text())
    manifest = repo / 'configs/execution/tasks/campaign.yaml'
    manifest.parent.mkdir()
    manifest.write_text(TASKS.read_text())
    head = commit(repo)
    (out / 'code_commit.txt').write_text(head + '\n')
    return repo, out, manifest


def execute(request, module_name, repo):
    return importlib.import_module(module_name).run_stage(request)


def producer(runtime):
    repo, out, manifest = runner_setup(runtime)
    result = run('simulation-inputs', out, repo, task_file=manifest,
                 task_id='simulation-inputs', execute=execute)
    assert result.status == 'pass', result.message
    verify_dependency_result(out)
    return repo, out, manifest


def consumer_dir(tmp_path):
    out = tmp_path / 'smoke'
    out.mkdir()
    (out / 'code_commit.txt').write_text((tmp_path / 'attempt/code_commit.txt').read_text())
    return out


def test_runner_admits_published_inputs_and_stops_before_compute(runtime, tmp_path, monkeypatch):
    repo, inputs, manifest = producer(runtime)
    monkeypatch.setenv('SWARM_DEP_SIMULATION_INPUTS', str(inputs))
    monkeypatch.setattr(campaign, 'fingerprint', lambda: {'synthetic_test': True})
    reached = []
    def admission(request, module_name, repo):
        assert module_name == 'oxyformer.validation.coverage'
        request.verify_inputs()
        _, recipe, lock, endpoint, frame, scenario, seconds, _ = coverage.prepare_batch(request)
        assert lock is None and seconds == 1800
        assert recipe['nested_cv']['synthetic'] is True
        assert endpoint.content_hash == recipe['endpoint_hash']
        assert frame.content_hash == recipe['frame_hash']
        reached.append(request.content_hash)
        return StageResult(request_hash=request.content_hash, status='blocked', artifacts=(),
                           message='admission-only; stopped before repetitions')
    result = run('simulation-smoke', consumer_dir(tmp_path), repo, deps_env=True,
                 task_file=manifest, task_id='simulation-smoke', execute=admission)
    assert reached and result.message == 'admission-only; stopped before repetitions'
    assert not list((tmp_path / 'smoke').glob('repetitions/*'))
    assert not (tmp_path / 'smoke/result.json').exists()


def test_runner_requires_published_producer(runtime, tmp_path, monkeypatch):
    repo, inputs, manifest = runner_setup(runtime)
    monkeypatch.setenv('SWARM_DEP_SIMULATION_INPUTS', str(inputs))
    endpoint, frame = smoke_inputs.build_inputs()
    (inputs / 'endpoint.json').write_text(endpoint.to_json())
    (inputs / 'frame.json').write_text(frame.to_json())
    (inputs / 'artifact_manifest.json').write_text('{}')
    with pytest.raises(ContractError, match='stage receipt missing'):
        run('simulation-smoke', consumer_dir(tmp_path), repo, deps_env=True,
            task_file=manifest, task_id='simulation-smoke', execute=execute)


def test_input_stage_refuses_unknown_fixture(runtime):
    repo, out, manifest = runner_setup(runtime)
    document = yaml.safe_load(manifest.read_text())
    document['tasks'][0]['parameters']['fixture'] = 'real-tract-substitute'
    # External task selects a different fixture while the repository stays clean.
    external = out / 'external-task.json'
    external.write_text(json.dumps(document))
    result = run('simulation-inputs', out, repo, task_file=external,
                 task_id='simulation-inputs', execute=execute)
    assert result.status == 'blocked' and 'unregistered synthetic fixture' in result.message
    assert not (out / 'endpoint.json').exists()


def test_pinned_recipe_refuses_changed_typed_input(runtime, tmp_path, monkeypatch):
    repo, inputs, manifest = runner_setup(runtime)
    original = smoke_inputs.build_inputs
    def changed():
        endpoint, frame = original()
        return replace(endpoint, treatment_design=replace(endpoint.treatment_design, center=4.)), frame
    monkeypatch.setattr(smoke_inputs, 'build_inputs', changed)
    published = run('simulation-inputs', inputs, repo, task_file=manifest,
                    task_id='simulation-inputs', execute=execute)
    assert published.status == 'pass', published.message
    monkeypatch.setenv('SWARM_DEP_SIMULATION_INPUTS', str(inputs))
    result = run('simulation-smoke', consumer_dir(tmp_path), repo, deps_env=True,
                 task_file=manifest, task_id='simulation-smoke', execute=execute)
    assert result.status == 'blocked' and result.message == 'recipe input drift'
    assert not list((tmp_path / 'smoke').glob('repetitions/*'))
