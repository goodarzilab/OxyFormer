"""Offline CPU acceptance evidence using synthetic records and real models."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
import yaml

from oxyformer.contracts import ColumnSpec, DataManifest, EstimandSpec, SourceManifest, SplitManifest, source_lineage_hash
from oxyformer.data.entity_graph import EntityGraph, EntityLink
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.data.loaders import load_records
from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy
from oxyformer.design.splits import InnerSplit
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ArtifactLineage, ContractError
from oxyformer.training.calibration import AffineCalibration, CalibrationPartition
from oxyformer.training.checkpoint import CheckpointRequest, load_checkpoint
from oxyformer.training.fit import (
    FitConfig, FoldArtifacts, NuisanceSettings, _Budget, _build, _bundle, _inputs, _partition,
    _predict, _train_one, fit_fold, predict_fold, subset,
)
import oxyformer.training.fit as fitting


def digest(value):
    return sha256(value.encode()).hexdigest()


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous)


def make_case(root, *, identity=False):
    ids = tuple(f"o{i:02d}" for i in range(30)) + ("unlabeled-acs",)
    rules = tuple(FeatureRule(name=n, role=r, endpoints=("synthetic",), uses=u,
        approval_id="synthetic-only") for n, r, u in (
        ("id", "identifier", ("linkage",)), ("y", "outcome", ("score",)),
        ("a", "exposure", ("score",)), ("x", "predictor", ("nuisance", "ssl", "context")),
        ("county", "county", ("county_routing",)), ("w", "outcome_metadata", ("linkage",))))
    registry = FeatureRegistry(registry_id="synthetic", rules=rules)
    links = tuple(EntityLink(observation_id=oid, relation="repeated_geography", namespace="fixture",
                            entity_id=str(i // 2)) for i, oid in enumerate(ids[:-1]))
    graph = EntityGraph(original_ids=ids, links=links)
    source = SourceManifest(source_id="synthetic", version="1", uri="synthetic://no-data",
        payload_hash=digest("synthetic"), license_hash=digest("license"), schema_hash=digest("schema"),
        field_mapping=(("raw_x", "x"),), mapping_status="reviewed", mapping_review_id="synthetic")
    policy = ShiftOrStayPolicy(support_design_hash=digest("design"), components_by_key=(("s", ((0., 10.),)),),
                               delta_mmhg=0. if identity else 2.)
    spec = EstimandSpec(endpoint="synthetic", target_id="synthetic-target", outcome_scale="years",
        policy_id=policy.policy_id, weight_id="synthetic-target-mass", adjustment_schema_hash=registry.content_hash,
        inference_unit="tract", source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(),
        split_hash=None, config_hash=digest("config"), model_hash=None, environment=(("fixture", "cpu"),),
        seed=None, parameter_count=None)
    schema = tuple(ColumnSpec(name=n, dtype=t, nullable=n == "y") for n, t in
                   (("id", "string"), ("y", "number"), ("a", "number"), ("x", "number"),
                    ("county", "string"), ("w", "number")))
    manifest = DataManifest(spec=spec, sources=(source,), schema=schema, registry=registry, original_ids=ids,
        id_field="id", outcome_field="y", exposure_field="a", weight_field="w",
        entity_graph_hash=graph.content_hash, lineage=lineage)
    records = [dict(id=oid, y=2. + (i // 2) % 8 + (i % 3), a=float((i // 2) % 8 + 1),
                    x=float(i % 7), county="c", w=float(i % 4 + 1)) for i, oid in enumerate(ids)]
    records[-1].update(y=None, a=900., x=1e6, w=1e8)
    data = load_records(records, manifest, spec, manifest.schema_hash)
    split = SplitManifest(spec=spec, level="outer", original_ids=ids[:-1], fold_ids=tuple(i // 6 for i in range(30)),
        design_ids=(), excluded_ids=ids[-1:], seed_ids=(1103, 2207, 3301), entity_graph_hash=graph.content_hash,
        lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    train = split.training_ids(0)
    inner_graph = EntityGraph(original_ids=train, links=tuple(link for link in links if link.observation_id in train))
    inner_manifest = replace(manifest, original_ids=train, entity_graph_hash=inner_graph.content_hash,
                             lineage=replace(lineage, unit_ids=train, parent_hashes=(manifest.content_hash,)))
    inner = SplitManifest(spec=spec, level="inner", original_ids=train,
        fold_ids=tuple((i // 2) % 3 for i in range(len(train))), design_ids=(), excluded_ids=(),
        seed_ids=split.seed_ids, entity_graph_hash=inner_graph.content_hash,
        lineage=replace(lineage, unit_ids=train, parent_hashes=(inner_manifest.content_hash,)))
    binding = InnerSplit(outer_fold=0, data_manifest=inner_manifest, entity_graph=inner_graph,
                         split=inner, buffer_excluded_ids=())
    covariates = PolicyCovariates(original_ids=ids, geography_ids=tuple(str(i // 2) for i in range(len(ids))),
                                 support_keys=("s",) * len(ids))
    config = FitConfig(data=data, entity_graph=graph, inner=binding, fold=0, policy=policy,
        policy_covariates=covariates, treatment_design=TreatmentDesign(center=5., scale=5.,
            knots=(0., 2., 4., 6., 8., 10.), design_hash=digest("design")),
        feature_kinds=(("x", "numeric"),), families=(("x",),), county_field="county",
        exposure_assignment_level="tract", output_dir=str(root), ssl_epochs=1,
        settings=NuisanceSettings(batch_size=8, frozen_epochs=1))
    return spec, split, manifest, config


def run(case):
    spec, split, manifest, config = case
    return fit_fold(spec, split, manifest, config, 1103)


def state(artifact):
    return load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)["controller"]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    case = make_case(tmp_path_factory.mktemp("full") / "attempt")
    artifact = run(case)
    torch.set_num_threads(previous)
    return case, artifact


def test_registered_configuration():
    expected = yaml.safe_load(Path("configs/training/nuisance.yaml").read_text())
    actual = NuisanceSettings()
    assert NuisanceSettings(**expected) == actual
    for key, value in expected.items():
        assert getattr(actual, key) == (tuple(value) if isinstance(value, list) else value)
    with pytest.raises(ContractError, match="unregistered"):
        replace(actual, learning_rates=(.1,))


def test_pairs_preserve_grouping_weights_and_endpoint_target(completed):
    case, artifact = completed
    config = case[-1]
    controller = state(artifact)
    assert artifact.complete
    for result in controller["results"]:
        audit = CalibrationPartition.from_json(result["ownership"])
        assert set(audit.evaluation_ids).isdisjoint(audit.fitting_ids + audit.checkpoint_ids)
        for group in config.entity_graph.components():
            assert not (set(group) & set(audit.fitting_ids) and set(group) & set(audit.evaluation_ids))
        assert "unlabeled-acs" not in audit.fitting_ids + audit.evaluation_ids
        assert result["mass"] == sum(_inputs(config, result["ids"]).origin_weights)
    calibration = AffineCalibration.from_json(controller["calibration"])
    assert set(calibration.original_ids) == set(case[1].training_ids(0))
    assert len(calibration.original_ids) == len(set(calibration.original_ids))
    assert calibration.class_prior == .5
    assert Path(artifact.checkpoint.path).with_name("calibration-transfer.json").is_file()
    restored = FoldArtifacts.from_json(artifact.to_json())
    assert restored == artifact


def test_both_treatments_predicted_from_label_free_views(completed):
    case, artifact = completed
    config = case[-1]
    view = subset(config.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    result = predict_fold(artifact, view, config.policy)
    assert result.origin_weights == artifact.prediction_inputs.origin_weights
    assert all(type(value) is float for value in result.mu_a + result.mu_d + result.r_a + result.r_d)
    assert any(a != b for a, b in zip(result.mu_a, result.mu_d))
    controller = state(artifact)
    model = _build(controller["final"]["origin"]).eval()
    calibration = AffineCalibration.from_json(controller["calibration"])
    with torch.no_grad():
        logits = _predict(model, view, artifact.prediction_inputs, config.policy)
    # predict_fold computes calibration and ratios in FP32 before widening for
    # output; use FP32 references and its default assert_close tolerances.
    assert logits.dtype == torch.float32
    expected = (logits * calibration.slope + calibration.intercept).exp()
    torch.testing.assert_close(torch.tensor(result.r_a, dtype=torch.float32), expected[:, 0])
    torch.testing.assert_close(torch.tensor(result.r_d, dtype=torch.float32), expected[:, 1])
    for invalid in (config.data, replace(view, use="ssl")):
        with pytest.raises(ContractError, match="label-free"):
            predict_fold(artifact, invalid, config.policy)
    with pytest.raises(ContractError, match="held-out fold"):
        predict_fold(artifact, config.data.covariates(("x",)), config.policy)


def test_nuisance_parameters_and_offsets_are_independent(completed):
    case, artifact = completed
    controller = state(artifact)
    outcome, origin = (_build(controller["final"][kind]) for kind in ("outcome", "origin"))
    assert {p.data_ptr() for p in outcome.parameters()}.isdisjoint(p.data_ptr() for p in origin.parameters())
    before = next(origin.parameters()).detach().clone()
    with torch.no_grad():
        next(outcome.parameters()).add_(10)
    assert torch.equal(before, next(origin.parameters()))
    with pytest.raises(ContractError, match="permitted training"):
        outcome.group_offsets.update_identity(artifact.prediction_inputs.original_ids,
            torch.zeros(6), torch.zeros(6), torch.ones(6))
    assert set(outcome.group_offsets.training_ids) == set(case[1].training_ids(0))


def test_outer_labels_and_covariates_cannot_change_fitted_models(completed, tmp_path):
    case, artifact = completed
    spec, split, manifest, config = case
    held = set(artifact.prediction_inputs.original_ids)
    rows = tuple((row[0], -9000., row[2], row[3] + 100., row[4], row[5])
                 if row[0] in held else row for row in config.data.rows)
    changed = replace(config, data=replace(config.data, rows=rows), output_dir=str(tmp_path / "changed"))
    other = fit_fold(spec, split, manifest, changed, 1103)
    a, b = state(artifact), state(other)
    assert a["calibration"] == b["calibration"]
    assert a["selection"] == b["selection"]
    for kind in ("outcome", "origin"):
        for name, value in a["final"][kind]["state"].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, b["final"][kind]["state"][name]), name
    original_view = subset(config.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    new_view = subset(changed.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    assert predict_fold(artifact, original_view, config.policy).mu_a != predict_fold(other, new_view, config.policy).mu_a


def test_deterministic_continuation_mid_training_loop(completed, tmp_path):
    case, complete = completed
    spec, split, manifest, config = case
    partial = fit_fold(spec, split, manifest,
        replace(config, output_dir=str(tmp_path / "slice"), max_batches=5), 1103)
    assert not partial.complete
    assert state(partial)["active"] is not None
    assert state(partial)["active"]["progress"]["phase"] in ("train", "evaluate")
    with pytest.raises(ContractError, match="unfinished"):
        predict_fold(partial, config.data.covariates(("x",)), config.policy)
    resumed = fit_fold(spec, split, manifest,
        replace(config, output_dir=str(tmp_path / "resume"), predecessor=partial), 1103)
    a, b = state(complete), state(resumed)
    assert a["results"] == b["results"]
    assert a["calibration"] == b["calibration"]
    for kind in ("outcome", "origin"):
        for name, value in a["final"][kind]["state"].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, b["final"][kind]["state"][name]), name
    assert resumed.checkpoint.predecessor_hash == partial.checkpoint.content_hash


def test_interruption_in_ssl_is_not_complete_and_can_resume(completed, tmp_path):
    case, complete = completed
    spec, split, manifest, config = case
    partial = fit_fold(spec, split, manifest,
        replace(config, output_dir=str(tmp_path / "ssl-slice"), max_batches=1), 1103)
    assert not partial.complete and state(partial)["ssl_pending"] is not None
    resumed = fit_fold(spec, split, manifest,
        replace(config, output_dir=str(tmp_path / "ssl-resume"), predecessor=partial), 1103)
    assert state(resumed)["results"] == state(complete)["results"]


def test_group_splits_and_stopping_leakage_are_rejected(tmp_path):
    spec, split, manifest, config = make_case(tmp_path / "invalid")
    evaluation = tuple(oid for oid, f in zip(config.inner.split.original_ids, config.inner.split.fold_ids) if f == 0)
    bad = replace(config, stopping_ids=((0, evaluation[:2]),))
    with pytest.raises(ContractError, match="inside the fitting partition"):
        fit_fold(spec, split, manifest, bad, 1103)
    fit_ids = config.inner.split.training_ids(0)
    with pytest.raises(ContractError, match="entity lineage crosses"):
        _partition(config, config.inner.split, 0, stopping=fit_ids[:1])


@pytest.mark.parametrize("zero_stop", [False, True])
def test_fitting_only_stopping_and_pair_batch_weights(completed, tmp_path, monkeypatch, zero_stop):
    case, artifact = completed
    config = replace(case[-1], output_dir=str(tmp_path / "stopping"))
    parent = config.inner.split
    stop = parent.training_ids(0)[:2]
    if zero_stop:
        rows = tuple((*row[:-1], 0.) if row[0] in stop else row for row in config.data.rows)
        config = replace(config, data=replace(config.data, rows=rows))
    local, ids, held = _partition(config, parent, 0, stopping=stop)
    # Exercise the real training loop using fresh synthetic initialization.
    view = subset(config.data.covariates(("x",)), ids)
    budget = _Budget(config, CheckpointRequest())
    init = fitting._ssl(config, local, ids, config.data.covariates(("x",)), tmp_path / "init", 1103, budget, None)
    bundle, encoder = _bundle(config, local, ids, init, "origin", config.settings.grid[0])
    calls = []
    original_loss = fitting.paired_origin_loss
    def spy(logits, pairs, *, reduction):
        n = len(pairs.original_ids) // 2
        assert pairs.original_ids[:n] == pairs.original_ids[n:]
        assert pairs.origin_weights[:n] == pairs.origin_weights[n:]
        assert pairs.origin_weights[:n] == _inputs(config, pairs.original_ids[:n]).origin_weights
        calls.append(pairs.original_ids[:n])
        expected = (logits.detach().sigmoid() - logits.new_tensor([0., 1.]))
        expected *= logits.new_tensor(pairs.origin_weights[:n])[:, None]
        expected *= len(ids) / (n * 2 * sum(_inputs(config, ids).origin_weights))
        # This hook observes the derivative of the REAL minibatch objective,
        # before clipping parameter gradients. A batch-mass denominator fails.
        logits.register_hook(lambda gradient: torch.testing.assert_close(gradient, expected))
        return original_loss(logits, pairs, reduction=reduction)
    monkeypatch.setattr(fitting, "paired_origin_loss", spy)
    model, saved, done, _ = _train_one(config, bundle, encoder, view, stop, 2, 1103, budget)
    assert done and saved["progress"]["best_epoch"] in (1, 2)
    if zero_stop:
        assert saved["progress"]["best_epoch"] == 2
        assert saved["progress"]["history"] == [None, None]
    assert set(model.group_offsets.training_ids) == set(ids)
    assert set(x for batch in calls for x in batch) == set(ids)
    assert all(set(batch).isdisjoint(held + stop) for batch in calls)


def test_nonfinite_targets_fail_without_completed_artifact(tmp_path):
    spec, split, manifest, config = make_case(tmp_path / "nonfinite")
    rows = tuple((row[0], 1e40, *row[2:]) if row[0] in split.training_ids(0) else row
                 for row in config.data.rows)
    with pytest.raises(ContractError, match="nonfinite"):
        fit_fold(spec, split, manifest, replace(config, data=replace(config.data, rows=rows)), 1103)
    assert not (tmp_path / "nonfinite/nuisance/fold.json").exists()


def test_requested_checkpoint_does_not_claim_completion(tmp_path):
    case = make_case(tmp_path / "request")
    request = CheckpointRequest()
    request.request()
    artifact = run((*case[:3], replace(case[-1], stop_request=request)))
    assert not artifact.complete and artifact.checkpoint.reason == "requested"
    assert state(artifact)["position"] == 0


def test_identity_policy_does_not_evaluate_exponential_ratio(tmp_path, monkeypatch):
    case = make_case(tmp_path / "identity", identity=True)
    artifact = run(case)
    config = case[-1]
    view = subset(config.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    controller = state(artifact)
    calibration = AffineCalibration.from_json(controller["calibration"])
    # Synthetic boundary reproduction: even these finite origin/calibration
    # outputs cannot change the exactly known ratio under the identity map.
    replacement = replace(calibration, slope=.5, intercept=0.)
    monkeypatch.setattr(AffineCalibration, "from_json", classmethod(lambda cls, text: replacement))
    original_predict = fitting._predict
    def extreme_origin(model, *args, **kwargs):
        if isinstance(model, fitting.OriginTransformer):
            return torch.full((len(view.original_ids), 2), 180.)
        return original_predict(model, *args, **kwargs)
    monkeypatch.setattr(fitting, "_predict", extreme_origin)
    result = predict_fold(artifact, view, config.policy)
    assert result.r_a == result.r_d == (1.,) * len(view.original_ids)


def test_zero_mass_inner_evaluation_fold_is_a_zero_contribution(tmp_path):
    spec, split, manifest, config = make_case(tmp_path / "zero-fold")
    zero_ids = {oid for oid, fold in zip(config.inner.split.original_ids, config.inner.split.fold_ids) if fold == 0}
    rows = tuple((*row[:-1], 0.) if row[0] in zero_ids else row for row in config.data.rows)
    config = replace(config, data=replace(config.data, rows=rows))
    artifact = fit_fold(spec, split, manifest, config, 1103)
    results = state(artifact)["results"]
    assert artifact.complete
    assert all(r["mass"] == 0 and all(m == 0 for m in r["metrics"]) for r in results if r["fold"] == 0)
