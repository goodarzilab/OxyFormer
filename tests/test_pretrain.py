"""Synthetic, offline CPU checks for permissions, masking and fold-local SSL."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import math

import numpy as np
import pytest
import torch
import yaml

from oxyformer.contracts import CovariateView, EstimandSpec, SplitManifest
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.models.tokens import FeatureSpec, FeatureTokenizer
from oxyformer.provenance import ArtifactLineage, ContractError
from oxyformer.training.checkpoint import load_checkpoint
from oxyformer.training.pretrain import (
    PretrainConfig, SSLSettings, balanced_loss, family_mask, fit_preprocessing,
    pretrain, reconstruction_totals,
)

H = "a" * 64


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous)


def make_case(tmp_path):
    columns = ("part", "complement", "category")
    registry = FeatureRegistry(registry_id="synthetic-only", rules=tuple(
        FeatureRule(name=x, role="predictor", endpoints=("synthetic",),
                    uses=("ssl", "nuisance"), approval_id="synthetic-fixture") for x in columns))
    spec = EstimandSpec(endpoint="synthetic", target_id="synthetic", outcome_scale="synthetic",
        policy_id="synthetic", weight_id="unit", adjustment_schema_hash=registry.content_hash,
        inference_unit="synthetic", source_lineage_hash=H)
    ids = tuple(f"t{i}" for i in range(10))
    lineage = ArtifactLineage(source_hashes=(H,), unit_ids=ids, parent_hashes=(), split_hash=None,
                              config_hash=H, model_hash=None, environment=(("fixture", "offline"),),
                              seed=None, parameter_count=None)
    split = SplitManifest(spec=spec, level="inner", original_ids=ids + ("external0", "external1"),
        fold_ids=(1,) * 10 + (0, 0), design_ids=("design",), excluded_ids=("excluded",),
        seed_ids=(1103, 2207), entity_graph_hash=H,
        lineage=replace(lineage, unit_ids=ids + ("external0", "external1", "design", "excluded")))
    view = CovariateView(spec=spec, registry=registry, original_ids=ids, columns=columns,
        values=tuple((float(i), 10.0 - i, "own" if i % 2 else "rent") for i in range(10)),
        use="ssl", lineage=lineage)
    settings = SSLSettings(fold=0, feature_kinds=(("part", "numeric"), ("complement", "numeric"),
        ("category", "categorical")), families=(("part", "complement"), ("category",)),
        batch_size=2, max_epochs=3, dropout=0.2, mask_rate=0.65,
        stopping_ids=("t6", "t7", "t8", "t9"))
    return view, split, PretrainConfig(settings=settings, output_dir=str(tmp_path / "attempt"))


def test_registered_defaults():
    values = yaml.safe_load((Path(__file__).parents[1] / "configs/training/ssl.yaml").read_text())
    settings = SSLSettings(fold=0, feature_kinds=(("x", "numeric"),), families=(("x",),))
    for key, value in values.items():
        assert getattr(settings, key) == (tuple(value) if isinstance(value, list) else value)
    assert settings.mask_rate == .30 and settings.learning_rate == 3e-4
    assert settings.max_epochs == 30 and settings.patience == 5


def test_preprocessing_excludes_stopping_records_and_external_fold(tmp_path):
    view, split, config = make_case(tmp_path)
    values = tuple((10000., -10000., "heldout-only") if oid in config.settings.stopping_ids else row
                   for oid, row in zip(view.original_ids, view.values))
    artifact = pretrain(replace(view, values=values), split, replace(config, max_batches=1), 1103)
    state = load_checkpoint(artifact, artifact.identity)
    features = tuple(FeatureSpec.from_json(x) for x in state["preprocessing"])
    assert features[0].mean == 2.5
    assert features[0].scale == pytest.approx((35 / 12) ** .5)
    assert "heldout-only" not in features[2].categories
    assert set(state["fitting_ids"]).isdisjoint(state["stopping_ids"])
    assert set(state["fitting_ids"] + state["stopping_ids"]) == set(split.training_ids(0))
    assert artifact.identity.preprocessing_hash == FeatureTokenizer(features).preprocessing_hash


@pytest.mark.parametrize("bad_id", ["external0", "design", "excluded"])
def test_external_or_sealed_records_cannot_enter_fit_or_stopping(tmp_path, bad_id):
    view, split, config = make_case(tmp_path)
    ids = (bad_id,) + view.original_ids[1:]
    bad = replace(view, original_ids=ids, lineage=replace(view.lineage, unit_ids=ids))
    with pytest.raises(ContractError, match="permitted training"):
        pretrain(bad, split, config, 1103)
    with pytest.raises(ContractError, match="current fitting partition"):
        pretrain(view, split, replace(config, settings=replace(config.settings, stopping_ids=(bad_id,))), 1103)
    assert not Path(config.output_dir).exists()


@pytest.mark.parametrize("name,role", [
    ("exposure", "exposure"), ("outcome", "outcome"),
    ("disease_aux", "downstream_health"), ("outcome_available", "outcome_metadata"),
    ("mortality_metadata", "outcome_metadata"), ("terrain", "exposure_proxy"),
    ("coordinates", "precise_geography"),
])
def test_prohibited_fields_fail_at_covariate_boundary(tmp_path, name, role):
    view, _, _ = make_case(tmp_path)
    registry = replace(view.registry, rules=view.registry.rules + (
        FeatureRule(name=name, role=role, endpoints=(), uses=(), approval_id=None),))
    with pytest.raises(ContractError, match="unapproved ssl"):
        replace(view, registry=registry, spec=replace(view.spec, adjustment_schema_hash=registry.content_hash),
                columns=view.columns + (name,), values=tuple(row + (1,) for row in view.values))
    with pytest.raises(ContractError, match="forbidden permissions"):
        FeatureRule(name=name, role=role, endpoints=("synthetic",), uses=("ssl",), approval_id="bad")


def test_nuisance_view_and_unknown_config_feature_are_refused(tmp_path):
    view, split, config = make_case(tmp_path)
    with pytest.raises(ContractError, match="SSL view"):
        pretrain(replace(view, use="nuisance"), split, config, 1103)
    with pytest.raises(ContractError, match="feature schema"):
        pretrain(view, split, replace(config, settings=replace(config.settings,
            feature_kinds=(("outcome", "numeric"),), families=(("outcome",),))), 1103)


def test_families_mask_jointly_and_missing_is_not_a_reconstruction_target(tmp_path):
    view, _, config = make_case(tmp_path)
    view = replace(view, values=((None, 10., "rent"),) + view.values[1:])
    features = fit_preprocessing(view, config.settings)
    batch = FeatureTokenizer(features).prepare(view)
    masked = family_mask(batch, ((0, 1), (2,)), .5, generator=torch.Generator().manual_seed(1))
    assert torch.equal(masked.masked[:, 0], masked.masked[:, 1])
    assert not torch.equal(masked.masked[:, 0], masked.masked[:, 2])
    assert torch.equal(masked.missing, batch.missing) and masked.missing[0, 0]
    all_masked = family_mask(batch, ((0, 1), (2,)), 1.0)
    predictions = [torch.zeros(10, 1), torch.zeros(10, 1), torch.zeros(10, 3)]
    sums, counts = reconstruction_totals(predictions, all_masked, features)
    assert counts == [9, 10, 10]
    changed = all_masked.numeric_values.clone()
    changed[0, 0] = float("nan")
    changed_sums, changed_counts = reconstruction_totals(predictions, replace(all_masked, numeric_values=changed), features)
    assert changed_counts == counts
    for x, y in zip(sums, changed_sums):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_huber_cross_entropy_and_equal_family_contributions():
    features = (FeatureSpec(name="x", kind="numeric"), FeatureSpec(name="y", kind="numeric"),
                FeatureSpec(name="z", kind="categorical", categories=("a", "b")))
    from oxyformer.models.tokens import FeatureBatch
    zeros = torch.zeros((1, 3), dtype=torch.bool)
    batch = FeatureBatch(torch.arange(3), torch.tensor([[2., 4., 0.]]),
                         torch.zeros((1, 3), dtype=torch.long), zeros, ~zeros, zeros)
    sums, counts = reconstruction_totals([torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 3)], batch, features)
    assert [float(x) for x in sums] == pytest.approx([1.5, 3.5, math_log_three()])
    assert float(balanced_loss(sums, counts, ((0, 1), (2,)))) == pytest.approx((2.5 + math_log_three()) / 2)


def math_log_three():
    import math
    return math.log(3)


def test_finished_checkpoint_is_frozen_and_has_best_and_latest_state(tmp_path):
    view, split, config = make_case(tmp_path)
    artifact = pretrain(view, split, config, 1103)
    assert artifact.complete and artifact.reason == "max_epochs" and artifact.epoch == 3
    with pytest.raises(FrozenInstanceError):
        artifact.complete = False
    state = load_checkpoint(artifact, artifact.identity)
    assert len(state["progress"]["history"]) == 3
    assert state["best_model"] is not None and state["optimizer"]["state"]
    assert artifact.lineage.parameter_count < 1_000_000
    assert Path(artifact.path).with_suffix(".json").read_text() == artifact.to_json()


def test_patience_is_scientific_stopping_not_execution_budget(tmp_path, monkeypatch):
    import oxyformer.training.pretrain as module
    # Freeze updates so repeated fixed-mask validation scores tie exactly.
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **k: None)
    view, split, config = make_case(tmp_path)
    config = replace(config, settings=replace(config.settings, patience=1, max_epochs=10))
    artifact = module.pretrain(view, split, config, 1103)
    assert artifact.complete and artifact.reason == "patience" and artifact.epoch == 2


@pytest.mark.parametrize("values,expected_mean,expected_scale", [
    ([1e160, -1e160] * 5, 0., 1e160),
    ([1e-200, -1e-200] * 5, 0., 1e-200),
    ([1.6e308] * 10, 1.6e308, 1.),
    ([1e308, -1e308, 1e-308] + [0.] * 7, 1e-309, math.sqrt(.2) * 1e308),
])
def test_population_moments_avoid_intermediate_overflow_and_underflow(
        tmp_path, values, expected_mean, expected_scale):
    view, _, config = make_case(tmp_path)
    view = replace(view, values=tuple((value, -value, row[2])
                                     for value, row in zip(values, view.values)))
    feature = fit_preprocessing(view, config.settings)[0]
    assert math.isclose(feature.mean, expected_mean, rel_tol=1e-14, abs_tol=0.)
    assert math.isclose(feature.scale, expected_scale, rel_tol=1e-14, abs_tol=0.)


@pytest.mark.parametrize("magnitude", [1e160, 1e-200])
def test_extreme_finite_predictors_train_with_unit_standardized_targets(tmp_path, magnitude):
    view, split, config = make_case(tmp_path)
    view = replace(view, values=tuple((magnitude if i % 2 else -magnitude,
                                      -magnitude if i % 2 else magnitude, row[2])
                                     for i, row in enumerate(view.values)))
    artifact = pretrain(view, split, replace(config, settings=replace(config.settings, max_epochs=1)), 1103)
    assert artifact.complete
    state = load_checkpoint(artifact, artifact.identity)
    features = tuple(FeatureSpec.from_json(item) for item in state["preprocessing"])
    batch = FeatureTokenizer(features).prepare(view)
    assert features[0].mean == 0. and features[0].scale == magnitude
    torch.testing.assert_close(batch.numeric_values[:, :2].abs(), torch.ones(10, 2), rtol=0, atol=0)


def test_fitted_moment_arithmetic_restores_numpy_error_policy(tmp_path):
    view, _, config = make_case(tmp_path)
    view = replace(view, values=tuple((1e160 if i % 2 else -1e160, row[1], row[2])
                                     for i, row in enumerate(view.values)))
    with np.errstate(all="raise"):
        feature = fit_preprocessing(view, config.settings)[0]
        assert feature.mean == 0. and feature.scale == 1e160
        assert all(value == "raise" for value in np.geterr().values())


def test_constant_column_uses_exact_observed_value_and_unit_scale(tmp_path):
    view, split, config = make_case(tmp_path)
    view = replace(view, values=tuple((.1, row[1], row[2]) for row in view.values))
    artifact = pretrain(view, split, replace(config, max_batches=1), 1103)
    state = load_checkpoint(artifact, artifact.identity)
    feature = FeatureSpec.from_json(state["preprocessing"][0])
    assert feature.mean == .1 and feature.scale == 1.


def test_tokenizer_cross_unit_overflow_is_refused_before_attempt(tmp_path):
    view, split, config = make_case(tmp_path)
    values = [-1.6e308] * 3 + [1.6e308] + [0.] * 6
    view = replace(view, values=tuple((value, row[1], row[2]) for value, row in zip(values, view.values)))
    config = replace(config, settings=replace(config.settings, stopping_ids=view.original_ids[4:]))
    with pytest.raises(ContractError, match="merged tokenizer.*nonfinite"):
        pretrain(view, split, config, 1103)
    assert not Path(config.output_dir).exists()
