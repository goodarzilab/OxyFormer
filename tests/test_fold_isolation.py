"""Perturb held-out information while holding the fitting problem fixed."""
from dataclasses import replace

import pytest
import torch

from test_nested_cv import full, threads, endpoint, tiny_config, predictions, state
from oxyformer.contracts import ColumnSpec
from oxyformer.data.feature_roles import FeatureRule
from oxyformer.provenance import ContractError
from oxyformer.training import nested_cv as nested
from oxyformer.training.fit import subset
from oxyformer.validation.leakage import assert_fitted_invariant, assert_prediction_invariant


def refit(prepared, root):
    return nested.run_fold(tiny_config(prepared, root), prepared.outer, 1103)


@pytest.mark.parametrize("column", ["y", "x"])
def test_held_out_perturbations(full, tmp_path, column):
    prepared, expected = full
    held = set(expected.prediction_inputs.original_ids)
    names = [c.name for c in prepared.data.manifest.schema]
    changed_id = sorted(held)[0]
    rows = []
    for row in prepared.data.rows:
        row = list(row)
        if row[0] in held and (column == "y" or row[0] == changed_id):
            row[names.index(column)] += 1000.
        rows.append(tuple(row))
    changed = replace(prepared, data=replace(prepared.data, rows=tuple(rows)))
    actual = refit(changed, tmp_path / "changed")
    assert_fitted_invariant(expected, actual)
    before, after = predictions(prepared, expected), predictions(changed, actual)
    assert_prediction_invariant(before, after, except_ids=(changed_id,) if column == "x" else ())
    if column == "x":
        assert before.mu_a != after.mu_a


def rebind(prepared, manifest, rows):
    outer = replace(prepared.outer, lineage=replace(prepared.outer.lineage,
                    parent_hashes=(manifest.content_hash,), unit_ids=manifest.original_ids))
    inners = []
    for inner in prepared.inner:
        im = replace(inner.data_manifest, schema=manifest.schema, registry=manifest.registry,
                     spec=manifest.spec, lineage=replace(inner.data_manifest.lineage,
                     parent_hashes=(manifest.content_hash,)))
        ins = replace(inner.split, spec=manifest.spec,
                      lineage=replace(inner.split.lineage, parent_hashes=(im.content_hash,)))
        inners.append(replace(inner, data_manifest=im, split=ins))
    return replace(prepared, data=replace(prepared.data, manifest=manifest, rows=rows),
        outer=replace(outer, spec=manifest.spec), inner=tuple(inners),
        geography=replace(prepared.geography, data_manifest_hash=manifest.content_hash))


def test_dataset_order_is_not_training_order(full, tmp_path):
    prepared, expected = full
    manifest = prepared.data.manifest
    changed_manifest = replace(manifest, original_ids=manifest.original_ids[::-1],
                               lineage=replace(manifest.lineage, unit_ids=manifest.original_ids[::-1]))
    changed = rebind(prepared, changed_manifest, prepared.data.rows[::-1])
    actual = refit(changed, tmp_path / "reordered")
    assert_fitted_invariant(expected, actual)
    assert_prediction_invariant(predictions(prepared, expected), predictions(changed, actual))


def test_related_outcomes_cannot_enter_predictors(full, tmp_path):
    prepared, _ = full
    manifest = prepared.data.manifest
    registry = replace(manifest.registry, rules=manifest.registry.rules + (FeatureRule(name="related_y", role="outcome",
        endpoints=(manifest.spec.endpoint,), uses=("score",), approval_id="synthetic"),))
    manifest = replace(manifest, schema=manifest.schema + (ColumnSpec(name="related_y", dtype="number", nullable=False),),
        registry=registry, spec=replace(manifest.spec, adjustment_schema_hash=registry.content_hash))
    prepared = rebind(prepared, manifest, tuple(row + (2.,) for row in prepared.data.rows))
    expected = refit(prepared, tmp_path / "before")
    held = set(expected.prediction_inputs.original_ids)
    changed = replace(prepared, data=replace(prepared.data,
        rows=tuple(row[:-1] + ((9000. if row[0] in held else row[-1]),) for row in prepared.data.rows)))
    actual = refit(changed, tmp_path / "after")
    assert_fitted_invariant(expected, actual)
    assert_prediction_invariant(predictions(prepared, expected), predictions(changed, actual))
    with pytest.raises(ContractError):
        changed.data.covariates(("x", "related_y"))


def test_prediction_never_accesses_labels(full, monkeypatch):
    prepared, artifact = full
    view = subset(prepared.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    from oxyformer.data.loaders import LoadedData
    def forbidden(*args, **kwargs):
        raise AssertionError("prediction tried to access privileged labels")
    monkeypatch.setattr(LoadedData, "column", forbidden)
    nested.predict(artifact, view, prepared.policy)
    with pytest.raises(ContractError, match="label-free"):
        nested.predict(artifact, prepared.data, prepared.policy)


def test_stale_context_cache_invalidates(full):
    prepared, artifact = full
    bundle = state(artifact)["final"]["outcome"]
    model = nested._build(bundle).eval()
    view = subset(prepared.data.covariates(("x",)), artifact.prediction_inputs.original_ids)
    with torch.no_grad():
        nested._predict(model, view, artifact.prediction_inputs, prepared.policy)
        assert model.county_context.cache_keys
        model.county_context.seeds.add_(.125)
        actual = nested._predict(model, view, artifact.prediction_inputs, prepared.policy)
        clean = nested._build({**bundle, "state": model.state_dict()}).eval()
        expected = nested._predict(clean, view, artifact.prediction_inputs, prepared.policy)
    assert torch.equal(actual, expected)
    assert clean.county_context is not model.county_context


def test_stopping_and_reference_ownership(full, tmp_path):
    prepared, artifact = full
    outer_held = set(artifact.prediction_inputs.original_ids)
    for result in state(artifact)["results"]:
        audit = nested.CalibrationPartition.from_json(result["ownership"])
        assert outer_held.isdisjoint(audit.fitting_ids + audit.checkpoint_ids + audit.evaluation_ids)
        assert set(audit.fitting_ids).isdisjoint(audit.evaluation_ids)
    for bundle in state(artifact)["final"].values():
        refs = nested.CovariateView.from_json(bundle["references"])
        assert set(refs.original_ids) == set(prepared.outer.training_ids(0))
    cfg = tiny_config(prepared, tmp_path / "bad", stopping_ids=((0, tuple(sorted(outer_held)[:2])),))
    with pytest.raises(ContractError, match="stopping"):
        nested.run_fold(cfg, prepared.outer, 1103)


def test_unseen_county_offsets_are_not_invented(full):
    _, artifact = full
    model = nested._build(state(artifact)["final"]["outcome"])
    with pytest.raises(ContractError):
        model.group_offsets(("unseen",))
