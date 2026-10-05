"""Offline synthetic acceptance. Never execute historical model training."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from oxyformer import legacy
from oxyformer.contracts import ColumnSpec, DataManifest, EstimandSpec, SourceManifest, SplitManifest, StageRequest, source_lineage_hash
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.legacy import benchmark, historical, launcher
from oxyformer.provenance import ArtifactLineage, ContractError, file_hash

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("test attempted network access")
    monkeypatch.setattr(historical.urllib.request, "urlopen", denied)


@pytest.fixture
def registry():
    return FeatureRegistry(registry_id="synthetic", rules=tuple(
        FeatureRule(name=name, role=role, endpoints=("synthetic",), uses=uses, approval_id="fixture-only")
        for name, role, uses in [("fips", "identifier", ("linkage",)), ("y", "outcome", ("score",)),
                                ("a", "exposure", ("score",)), ("x", "predictor", ("nuisance", "ssl")),
                                ("disease", "downstream_health", ("diagnostic",))]))


@pytest.fixture
def counties():
    # Include UT/CO plus small and large states. A one-row state MUST be deleted.
    states = ["01"] * 12 + ["06"] * 13 + ["08"] * 10 + ["49"] * 11 + ["50"]
    rng = np.random.default_rng(54)
    x = rng.normal(size=len(states))
    a = 0.4 * x + rng.normal(size=len(states))
    y = 2 * a + x + rng.normal(size=len(states))
    return pd.DataFrame({"fips": [s + f"{i:03d}" for i, s in enumerate(states)],
                         "x": x, "a": a, "y": y, "disease": y * 2})


def request_at(tmp_path, parameters, dependencies=(), stage=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    out.mkdir()
    config, task = tmp_path / "config.json", tmp_path / "task.json"
    config.write_text('{}')
    stage = stage or ("elevcan-reproduction" if parameters["mode"] == "elevcan-reproduction" else "legacy-reproduction")
    task.write_text(json.dumps({"id": "synthetic", "stage": stage, "parameters": parameters}))
    return StageRequest(stage=stage, config_path=str(config), config_hash=file_hash(config),
        task_path=str(task), task_hash=file_hash(task), dependency_paths=tuple(map(str, dependencies)),
        dependency_hashes=tuple(file_hash(p) for p in dependencies), output_dir=str(out), code_identity="a" * 40)


def report_for(request):
    return json.loads(next(Path(request.output_dir).glob("legacy/*/report.json")).read_text())


def test_every_state_refits_and_excludes_deleted_state(counties, registry, monkeypatch):
    fits, training_ids = [], []
    original_fit = benchmark.fit_ridge_coefficients
    original_transform = benchmark.transform_design
    def fit(x, y, alpha):
        fits.append((len(x), alpha))
        return original_fit(x, y, alpha)
    def transform(frame, train, test, continuous, dummy):
        training_ids.append(frame.iloc[train].fips.tolist())
        return original_transform(frame, train, test, continuous, dummy)
    monkeypatch.setattr(benchmark, "fit_ridge_coefficients", fit)
    monkeypatch.setattr(benchmark, "transform_design", transform)
    rows = benchmark.run_leave_one_state_out(counties, outcome_column="y", treatment_column="a",
        covariates=["x"], registry=registry, endpoint="synthetic", fold_count=3)
    assert [r["deleted_state"] for r in rows] == ["01", "06", "08", "49", "50"]
    assert len(fits) == 5 * 3 * 2  # both nuisances fitted afresh in each deletion fold
    assert len(training_ids) == 5 * 3
    for i, row in enumerate(rows):
        assert row["status"] == "pass"
        assert all(not oid.startswith(row["deleted_state"]) for ids in training_ids[3*i:3*i+3] for oid in ids)
        retained = counties[~counties.fips.str.startswith(row["deleted_state"])]
        expected, _, _ = benchmark.cross_fitted_partial_linear_dml(retained, "y", "a", ["x"], 3, 25., 10., 20260306)
        assert row["effect"]["slope"] == expected["slope"]
    assert rows[-1]["deleted_rows"] == 1


def test_failed_deletion_is_retained(registry):
    frame = pd.DataFrame({"fips": ["01001", "08001"], "x": [1., 2.], "a": [2., 1.], "y": [3., 2.]})
    rows = benchmark.run_leave_one_state_out(frame, outcome_column="y", treatment_column="a", covariates=["x"],
        registry=registry, endpoint="synthetic", fold_count=2)
    assert len(rows) == 2 and all(r["status"] == "fail" for r in rows)


@pytest.mark.parametrize("columns", [[], ["disease"], ["a"], ["y"], ["foundation_embedding_01"]])
def test_confounds_are_explicit_and_approved(columns, registry):
    with pytest.raises(ContractError):
        benchmark.approved_confounders(columns, registry, "synthetic")
    assert benchmark.approved_confounders(["x"], registry, "synthetic") == ["x"]


def test_foundation_has_no_auxiliary_head():
    import torch
    from oxyformer.legacy.foundation import TabularAttentionFoundationModel
    model = TabularAttentionFoundationModel(2, 8, 4, 2, 1)
    assert model.auxiliary_head is None
    assert not any("auxiliary" in key for key in model.state_dict())
    outputs = model(torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.bool))
    assert "auxiliary" not in outputs
    with pytest.raises(ContractError, match="disabled"):
        TabularAttentionFoundationModel(2, 8, 4, 2, 1, auxiliary_dim=2)


@pytest.mark.parametrize("entry,stage", [("phase1", "exposure-atlas"), ("phase25", "tract-support-gate"),
                                        ("phase26", "primary"), ("phase3", "primary"), ("phase4", "refit-audit")])
def test_cli_delegates_v2_unchanged(entry, stage, tmp_path, monkeypatch):
    import oxyformer.cli
    calls = []
    monkeypatch.setattr(oxyformer.cli, "main", lambda args: calls.append(args) or 2)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"parameters": {"mode": "v2", "entrypoint": entry}}))
    spec = importlib.util.spec_from_file_location("legacy_test_" + entry, REPO / launcher.configuration()["entrypoints"][entry]["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--mode", "v2", "--stage", stage, "--task", str(task), "--repo", "/repo", "--out", "/attempt"]) == 2
    assert calls == [["run-stage", "--stage", stage, "--task", str(task), "--repo", "/repo", "--out", "/attempt"]]


def test_cli_refuses_mode_confusion(tmp_path, monkeypatch):
    import oxyformer.cli
    monkeypatch.setattr(oxyformer.cli, "main", lambda args: pytest.fail("must not delegate mismatched mode"))
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"parameters": {"mode": "initial-release", "entrypoint": "phase3"}}))
    with pytest.raises(SystemExit):
        launcher.main("phase3", ["--mode", "v2", "--stage", "primary", "--task", str(task)])


def synthetic_archive(tmp_path, monkeypatch, *, script=None):
    config = deepcopy(launcher.configuration())
    pin = config["sources"]["initial-release"]
    # A tiny synthetic historical script demonstrates unmodified execution and
    # historical defaults without executing any historical estimator/training.
    script = script or '''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root-dir'); a=p.parse_args()
out=Path(a.root_dir)/'outputs/phase26'; out.mkdir(parents=True)
(out/'result.json').write_text(json.dumps({'historical_auxiliary_weight': 0.35}))
'''
    archive = tmp_path / "synthetic.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for entry in config["entrypoints"].values():
            raw = script.encode()
            entry["sha256_initial"] = sha256(raw).hexdigest()
            info = tarfile.TarInfo(pin["root"] + '/' + entry["path"])
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    pin["archive_sha256"] = file_hash(archive)
    pin["archive_bytes"] = archive.stat().st_size
    config["entrypoints"]["phase26"].update(inputs=[], outputs=["result.json"], python_dependencies=[])
    monkeypatch.setattr(legacy, "configuration", lambda: config)
    return archive, config, script


def test_pins_identify_verified_initial_release_and_elevcan():
    config = launcher.configuration()
    pin = config["sources"]["initial-release"]
    assert pin["commit"] == "2bdf22d568b37e6bb1ff4d2c85bab9f4ff06f277"
    assert pin["archive_sha256"] == "46183d525fc0811b20e75cac954b9383e5e3387b15562eb016062dc3ac102bcf"
    us = json.loads((REPO / 'configs/sources/us.json').read_text())
    upstream = next(r for r in us['resources'] if r['id'] == 'elevcan_source')
    assert config['sources']['elevcan-reproduction']['archive_sha256'] == upstream['expected_sha256']
    assert config['sources']['elevcan-reproduction']['archive_url'] == upstream['url']


def test_historical_source_runs_unmodified_in_isolation(tmp_path, monkeypatch):
    archive, config, script = synthetic_archive(tmp_path, monkeypatch)
    req = request_at(tmp_path / "request", {"mode": "initial-release", "entrypoint": "phase26", "source_archive": str(archive)}, [archive])
    result = legacy.run_stage(req)
    assert result.status == "pass", report_for(req)
    result.verify(req)
    report = report_for(req)
    assert report["numerical_agreement"] is None
    source = Path(req.output_dir) / 'legacy/initial-release/workspace' / config['sources']['initial-release']['root']
    assert (source / 'phase26_foundation_model.py').read_text() == script
    assert json.loads((source / 'outputs/phase26/result.json').read_text())['historical_auxiliary_weight'] == .35
    assert not (tmp_path / 'outputs').exists()
    with pytest.raises(FileExistsError):
        legacy.run_stage(req)


def test_historical_archive_hash_and_script_identity_checked(tmp_path, monkeypatch):
    archive, config, _ = synthetic_archive(tmp_path, monkeypatch)
    config['entrypoints']['phase3']['sha256_initial'] = '0' * 64
    req = request_at(tmp_path / 'request', {"mode": "initial-release", "entrypoint": "phase26", "source_archive": str(archive)}, [archive])
    assert legacy.run_stage(req).status == 'fail'
    assert 'script identity' in report_for(req)['reason']
    archive.write_bytes(b'wrong')
    with pytest.raises(ContractError, match='size mismatch'):
        historical.retrieve_source(config['sources']['initial-release'], tmp_path, archive=archive)


def test_shallow_clone_needs_no_git_and_missing_archive_blocks(tmp_path, monkeypatch):
    def no_process(*args, **kwargs):
        pytest.fail('archive absent: neither git nor training should run')
    monkeypatch.setattr(subprocess, 'run', no_process)
    req = request_at(tmp_path, {"mode": "initial-release", "entrypoint": "phase26"})
    assert legacy.run_stage(req).status == 'blocked'
    report = report_for(req)
    assert report['provisioning_requests'][0]['action'] == 'download_public_archive'
    assert report['numerical_agreement'] is None


def test_explicit_archive_download_is_verified(tmp_path, monkeypatch):
    archive, config, _ = synthetic_archive(tmp_path, monkeypatch)
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(archive.read_bytes())
    monkeypatch.setattr(historical.urllib.request, 'urlopen', fetch)
    root = tmp_path / 'download'; root.mkdir()
    source = historical.retrieve_source(config['sources']['initial-release'], root, allow_download=True)
    assert source.is_dir()
    assert calls == [config['sources']['initial-release']['archive_url']]
    assert file_hash(root / 'source.tar.gz') == file_hash(archive)


def test_python_dependency_block_is_actionable(tmp_path, monkeypatch):
    archive, config, _ = synthetic_archive(tmp_path, monkeypatch)
    config['entrypoints']['phase26']['python_dependencies'] = ['nonexistent_historical_python_module']
    req = request_at(tmp_path / 'request', {"mode": "initial-release", "entrypoint": "phase26", "source_archive": str(archive)}, [archive])
    result = legacy.run_stage(req)
    assert result.status == 'blocked'
    report = report_for(req)
    assert report['provisioning_requests'][0]['action'] == 'install_python_dependencies'
    assert report['numerical_agreement'] is None


def test_missing_r_dependency_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 42, 'glmnet,Hmisc', ''))
    with pytest.raises(historical.PrerequisiteMissing) as exc:
        historical.r_preflight('Rscript', ['glmnet', 'Hmisc'], tmp_path, {})
    assert exc.value.requests[0]['action'] == 'provision_R_dependencies'


def test_root_outputs_refused(tmp_path):
    req = request_at(tmp_path, {"mode": "initial-release", "entrypoint": "phase1"})
    with pytest.raises(ContractError, match='inside the repository'):
        legacy.run_stage(replace(req, output_dir=str(REPO)))


def prepared_data(tmp_path, counties, registry):
    ids = tuple(counties.fips)
    graph = EntityGraph(original_ids=ids, links=())
    source = SourceManifest(source_id='synthetic', version='1', uri='synthetic://fixture', payload_hash='1'*64,
        license_hash='2'*64, schema_hash='3'*64, field_mapping=(('raw_x', 'x'),), mapping_status='reviewed', mapping_review_id='fixture')
    spec = EstimandSpec(endpoint='synthetic', target_id='synthetic', outcome_scale='synthetic', policy_id='synthetic',
        weight_id='unit', adjustment_schema_hash=registry.content_hash, inference_unit='county', source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(), split_hash=None,
        config_hash='4'*64, model_hash=None, environment=(('test', 'synthetic'),), seed=None, parameter_count=None)
    manifest = DataManifest(spec=spec, sources=(source,), schema=tuple(ColumnSpec(name=c, dtype='string' if c=='fips' else 'number') for c in counties.columns),
        registry=registry, original_ids=ids, id_field='fips', outcome_field='y', exposure_field='a', weight_field=None,
        entity_graph_hash=graph.content_hash, lineage=lineage)
    split = SplitManifest(spec=spec, level='outer', original_ids=ids[:-2], fold_ids=tuple(i%3 for i in range(len(ids)-2)),
        design_ids=ids[-2:-1], excluded_ids=ids[-1:], seed_ids=(7,), entity_graph_hash=graph.content_hash,
        lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    paths=[]
    for name, text in [('manifest', manifest.to_json()), ('records', json.dumps(counties.to_dict('records'))),
                       ('entities', graph.to_json()), ('split', split.to_json())]:
        path=tmp_path/(name+'.json'); path.write_text(text); paths.append(path)
    return dict(zip(['manifest','records','entities','split'], map(str, paths))), paths, split


@pytest.mark.parametrize('entry', ['phase3','phase4'])
def test_repaired_stage_benchmark_reports(entry, tmp_path, counties, registry):
    parameters, paths, _ = prepared_data(tmp_path, counties, registry)
    parameters.update(mode='repaired-benchmark', entrypoint=entry, confounders=['x'], fold_count=3)
    req=request_at(tmp_path/'request', parameters, paths)
    result=legacy.run_stage(req)
    assert result.status == 'pass', report_for(req)
    result.verify(req)
    report=report_for(req)
    assert report['mode']=='repaired-benchmark' and report['confounders']==['x']
    assert report['disease_auxiliaries'] is False
    if entry=='phase4':
        assert len(report['deletions'])==5


def test_repaired_foundation_uses_only_approved_training_fold(tmp_path, counties, registry, monkeypatch):
    from oxyformer.legacy import foundation
    parameters, paths, split = prepared_data(tmp_path, counties, registry)
    parameters.update(mode='repaired-benchmark', entrypoint='phase26', confounders=['x'], fold=1)
    seen=[]
    def fit(features, **kwargs):
        seen.append(features.copy())
        return foundation.TabularAttentionFoundationModel(1, 8, 4, 2, 1), []
    monkeypatch.setattr(foundation, 'fit_repaired', fit)
    monkeypatch.setattr(foundation, 'encode_embeddings', lambda model, x: np.zeros((len(x), 4)))
    req=request_at(tmp_path/'request', parameters, paths)
    result=legacy.run_stage(req)
    assert result.status=='pass', report_for(req)
    assert seen[0].shape==(len(split.training_ids(1)),1)
    selected=counties.set_index('fips').loc[list(split.training_ids(1)), 'x'].to_numpy()
    np.testing.assert_allclose(seen[0][:,0],(selected-selected.mean())/selected.std(),rtol=1e-12,atol=1e-12)
    assert report_for(req)['training_ids']==list(split.training_ids(1))


@pytest.mark.parametrize('channel', ['disease','y','a'])
def test_repaired_foundation_rejects_prohibited_predictors(channel, tmp_path, counties, registry):
    parameters, paths, _ = prepared_data(tmp_path, counties, registry)
    parameters.update(mode='repaired-benchmark', entrypoint='phase26', confounders=[channel], fold=1)
    req=request_at(tmp_path/'request',parameters,paths)
    assert legacy.run_stage(req).status=='fail'
    assert 'unapproved ssl' in report_for(req)['reason']


def test_unlabeled_legacy_cli_fails_before_execution(tmp_path, monkeypatch):
    import oxyformer.cli
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / 'synthetic-input'
    marker.write_text('unchanged')
    monkeypatch.setattr(oxyformer.cli, 'main', lambda args: pytest.fail('refused input executed'))
    with pytest.raises(SystemExit) as error:
        launcher.main('phase3', ['--root-dir', '.'])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == 'unchanged'
