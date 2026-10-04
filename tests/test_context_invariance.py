"""Reference ownership, invariance, gradient and cache tests on synthetic X."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest
import torch

from oxyformer.contracts import CovariateView, EstimandSpec, SplitManifest
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.models.county_context import CountyContext
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.tokens import FeatureSpec
from oxyformer.provenance import ArtifactLineage, ContractError


def digest(text):
    return sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def tiny_cpu_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(2207)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def setup():
    rules = tuple(FeatureRule(name=name, role=role, endpoints=("synthetic",), uses=uses,
                              approval_id="synthetic-only") for name, role, uses in (
        ("x", "predictor", ("context", "nuisance", "ssl")),
        ("cat", "predictor", ("context", "nuisance", "ssl")),
        ("county", "county", ("county_routing",)),
        ("a", "exposure", ("score",)), ("y", "outcome", ("score",)),
        ("residual", "outcome_metadata", ("diagnostic",)),
        ("county_effect", "outcome_metadata", ("diagnostic",)),
        ("latitude", "precise_geography", ("diagnostic",)),
    ))
    registry = FeatureRegistry(registry_id="synthetic-only", rules=rules)
    spec = EstimandSpec(endpoint="synthetic", target_id="synthetic-target", outcome_scale="years",
                        policy_id="synthetic-policy", weight_id="synthetic-unit-weights",
                        adjustment_schema_hash=registry.content_hash, inference_unit="county",
                        source_lineage_hash=digest("source"))
    original_ids = ("t0", "t1", "t2", "t3", "t4", "h0", "h1")
    lineage = ArtifactLineage(source_hashes=(digest("payload"),), unit_ids=original_ids + ("sealed", "excluded"),
                              parent_hashes=(), split_hash=None, config_hash=digest("config"),
                              model_hash=None, environment=(("fixture", "cpu"),), seed=None,
                              parameter_count=None)
    split = SplitManifest(spec=spec, level="outer", original_ids=original_ids,
                          fold_ids=(1, 1, 1, 1, 1, 0, 0), design_ids=("sealed",), excluded_ids=("excluded",),
                          seed_ids=(1103,), entity_graph_hash=digest("graph"), lineage=lineage)
    ids = split.training_ids(0)
    refs = CovariateView(spec=spec, registry=registry, original_ids=ids, columns=("x", "cat"),
                         values=((0., "own"), (1., "rent"), (4., "own"), (-2., "new"), (8., None)),
                         use="context", lineage=replace(lineage, unit_ids=ids, split_hash=split.content_hash))
    features = (FeatureSpec(name="x", kind="numeric", mean=1, scale=2),
                FeatureSpec(name="cat", kind="categorical", categories=("own", "rent")))
    counties = ("c", "c", "c", "d", "singleton")
    return features, refs, split, counties


def make_context(setup, *, refs=None, counties=None, dropout=0.0, split=None):
    features, default_refs, default_split, default_counties = setup
    return CountyContext(FeatureEncoder(features, dropout=dropout), refs or default_refs,
                         split or default_split, 0, counties or default_counties, county_field="county",
                         checkpoint_hash=digest("checkpoint"), dropout=dropout)


def test_prepare_standardizes_and_respects_column_identities(setup):
    features, refs, *_ = setup
    tokenizer = FeatureEncoder(features).tokenizer
    batch = tokenizer.prepare(refs)
    torch.testing.assert_close(batch.numeric_values[:, 0], torch.tensor([-0.5, 0., 1.5, -1.5, 3.5]))
    assert batch.categorical_values[:, 1].tolist() == [0, 1, 0, 2, 0]
    assert batch.missing[:, 1].tolist() == [False, False, False, False, True]
    permuted = replace(refs, columns=refs.columns[::-1], values=tuple(row[::-1] for row in refs.values))
    torch.testing.assert_close(tokenizer(batch), tokenizer(tokenizer.prepare(permuted)))
    assert tokenizer.preprocessing_hash != FeatureEncoder((replace(features[0], mean=2), features[1])).tokenizer.preprocessing_hash


def test_reference_and_query_permutations(setup):
    _, refs, _, counties = setup
    context = make_context(setup).eval()
    order = (4, 2, 0, 3, 1)
    ids = tuple(refs.original_ids[i] for i in order)
    permuted = replace(refs, original_ids=ids, values=tuple(refs.values[i] for i in order),
                       lineage=replace(refs.lineage, unit_ids=ids))
    other = make_context(setup, refs=permuted, counties=tuple(counties[i] for i in order)).eval()
    other.load_state_dict(context.state_dict())
    queries, routes = ("h0", "t1", "h1"), ("c", "c", "d")
    with torch.no_grad():
        expected = context(queries, routes)
        torch.testing.assert_close(other(queries, routes), expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(context(queries[::-1], routes[::-1]), expected.flip(0))
    assert expected.shape == (3, 4, 64)
    # Direct PMA set invariance even without the canonical sorting in forward.
    torch.testing.assert_close(context._pool(("t0", "t1", "t2")),
                               context._pool(("t2", "t0", "t1")), rtol=1e-5, atol=2e-6)


def test_training_query_is_excluded_before_encoding(setup):
    _, refs, _, _ = setup
    context = make_context(setup).eval()
    # Change only t0's own X in the reference store. Its own context cannot change,
    # but another query's context must retain the now-altered t0 reference.
    altered = replace(refs, values=((10000., "rent"),) + refs.values[1:])
    other = make_context(setup, refs=altered).eval()
    other.load_state_dict(context.state_dict())
    torch.testing.assert_close(context(("t0",), ("c",)), other(("t0",), ("c",)), rtol=0, atol=0)
    assert not torch.allclose(context(("t1",), ("c",)), other(("t1",), ("c",)))
    with torch.no_grad():
        context(("t0",), ("c",))
    assert context.cache_keys[0].reference_ids == ("t1", "t2")


def test_empty_unseen_and_singleton_counties_are_constant_zero(setup):
    context = make_context(setup).eval()
    with torch.no_grad():
        value = context(("t4", "new-tract", "h0"), ("singleton", "unseen", "empty"))
    assert value.shape == (3, 4, 64) and value.count_nonzero() == 0
    assert context((), ()).shape == (0, 4, 64)
    assert context(("h1",), ("d",)).abs().sum() > 0
    assert all("county" not in name for name, _ in context.named_parameters())


@pytest.mark.parametrize("bad_id", ["h0", "sealed", "excluded", "unknown"])
def test_reference_ownership_rejects_nontraining_ids(setup, bad_id):
    _, refs, _, _ = setup
    ids = (bad_id,) + refs.original_ids[1:]
    bad = replace(refs, original_ids=ids, lineage=replace(refs.lineage, unit_ids=ids))
    with pytest.raises(ContractError, match="permitted training IDs"):
        make_context(setup, refs=bad)


def test_references_cannot_be_label_selected_subset_or_wrong_split(setup):
    _, refs, split, counties = setup
    ids = refs.original_ids[1:]
    subset = replace(refs, original_ids=ids, values=refs.values[1:], lineage=replace(refs.lineage, unit_ids=ids))
    with pytest.raises(ContractError, match="exactly"):
        make_context(setup, refs=subset, counties=counties[1:])
    bad = replace(refs, lineage=replace(refs.lineage, split_hash=digest("other-split")))
    with pytest.raises(ContractError, match="split mismatch"):
        make_context(setup, refs=bad)
    with pytest.raises(ContractError, match="context CovariateView"):
        make_context(setup, refs=replace(refs, use="nuisance"))
    context = make_context(setup)
    with pytest.raises(ContractError, match="route mismatch"):
        context(("t0",), ("d",))
    for oid in split.design_ids + split.excluded_ids:
        with pytest.raises(ContractError, match="sealed"):
            context((oid,), ("c",))


@pytest.mark.parametrize("field", ["a", "y", "residual", "county_effect", "latitude", "county", "unapproved"])
def test_forbidden_reference_fields_are_rejected(setup, field):
    refs = setup[1]
    with pytest.raises(ContractError, match="unapproved|unknown feature"):
        replace(refs, columns=refs.columns + (field,), values=tuple(row + (99.,) for row in refs.values))
    with pytest.raises(ContractError):
        refs.column(field)


def test_heldout_records_cannot_mutate_references_or_preprocessing(setup):
    features, refs, _, _ = setup
    context = make_context(setup).eval()
    before_state = {k: v.clone() for k, v in context.state_dict().items()}
    original_hash = refs.content_hash
    preprocess_hash = context.encoder.tokenizer.preprocessing_hash
    heldout = replace(refs, original_ids=("h0",), values=((2., "own"),),
                       use="nuisance", lineage=replace(refs.lineage, unit_ids=("h0",)))
    changed = replace(heldout, values=((-999., "rent"),))
    with torch.no_grad():
        first = context(heldout.original_ids, ("c",))
        keys = context.cache_keys
        own_prediction = context.encoder(context.encoder.tokenizer.prepare(heldout))[1]
        altered_prediction = context.encoder(context.encoder.tokenizer.prepare(changed))[1]
        assert not torch.allclose(own_prediction, altered_prediction)
        # A/Y cannot enter either permitted view or the context call signature.
        for forbidden in ("a", "y"):
            with pytest.raises(TypeError):
                context(("h0",), ("c",), **{forbidden: torch.tensor([999.])})
        torch.testing.assert_close(context(changed.original_ids, ("c",)), first, rtol=0, atol=0)
        assert context.cache_keys == keys
        first.zero_()
        assert context(("h0",), ("c",)).abs().sum() > 0
    assert context.references.content_hash == original_hash
    assert context.encoder.tokenizer.preprocessing_hash == preprocess_hash
    for key, value in context.state_dict().items():
        torch.testing.assert_close(value, before_state[key], rtol=0, atol=0)
    with pytest.raises(FrozenInstanceError):
        context.references.values = changed.values


def test_training_recomputes_reference_gradients_after_optimizer_steps(setup):
    context = make_context(setup)
    optimizer = torch.optim.SGD(context.parameters(), lr=0.02)
    weights = torch.linspace(-1, 1, 64)
    versions = []
    for _ in range(2):
        optimizer.zero_grad()
        output = context(("t0", "t1"), ("c", "c"))
        (output * weights).sum().backward()
        gradient = context.encoder.tokenizer.numeric_weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert context.attention.in_proj_weight.grad.abs().sum() > 0
        assert context.seeds.grad.abs().sum() > 0
        assert context.cache_keys == ()
        optimizer.step()
        versions.append(context.encoder.tokenizer.numeric_weight.detach().clone())
    assert not torch.equal(*versions)
    # eval() alone is not permission to detach a graph.
    context.eval()
    optimizer.zero_grad()
    (context(("t0",), ("c",)) * weights).sum().backward()
    assert context.encoder.cls.grad.abs().sum() > 0 and context.cache_keys == ()


def test_cache_binds_model_preprocessing_references_split_and_declared_versions(setup):
    context = make_context(setup).eval()
    with torch.no_grad():
        expected = context(("h0",), ("c",))
        first = context.cache_keys[0]
        torch.testing.assert_close(context(("h0",), ("c",)), expected, rtol=0, atol=0)
        assert context.cache_keys == (first,)
        context.encoder.tokenizer.numeric_weight.add_(0.1)
        updated = context(("h0",), ("c",))
        second = context.cache_keys[0]
        assert second.model_hash != first.model_hash and len(context.cache_keys) == 1
        assert not torch.allclose(updated, expected)
        # PMA changes also invalidate, independently of the reference encoder.
        context.attention.out_proj.bias.add_(torch.linspace(-1, 1, 64))
        context(("h0",), ("c",))
        assert context.cache_keys[0].model_hash != second.model_hash
        state = {k: v.clone() for k, v in context.state_dict().items()}
        state["seeds"] += 0.5 * torch.randn_like(state["seeds"])
        third = context.cache_keys[0]
        context.load_state_dict(state)
        context(("h0",), ("c",))
        assert context.cache_keys[0].model_hash != third.model_hash
        context.set_cache_identity(checkpoint_hash=digest("new-checkpoint"), parameter_version=7)
        assert context.cache_keys == ()
        context(("h0",), ("c",))
        key = context.cache_keys[0]
    assert key.checkpoint_hash == digest("new-checkpoint") and key.parameter_version == 7
    assert key.split_hash == setup[2].content_hash and key.fold == 0
    assert key.reference_ids == ("t0", "t1", "t2")
    assert key.reference_hash == setup[1].content_hash
    assert key.preprocessing_hash == context.encoder.tokenizer.preprocessing_hash
    context.train()
    assert context.cache_keys == ()


def test_context_dropout_and_cache_are_disabled_during_training(setup):
    context = make_context(setup, dropout=0.5)
    assert context.attention.dropout == 0.5
    with torch.no_grad():
        first = context(("t0",), ("c",))
        assert not torch.equal(first, context(("t0",), ("c",)))
        assert context.cache_keys == ()
        context.eval()
        first = context(("h0",), ("c",))
        context.clear_cache()  # Recompute to establish dropout is off, not just cached.
        torch.testing.assert_close(first, context(("h0",), ("c",)), rtol=0, atol=0)
        context.encoder.train()
        context.clear_cache()
        assert not torch.equal(context(("h0",), ("c",)), context(("h0",), ("c",)))
        assert context.cache_keys == ()
