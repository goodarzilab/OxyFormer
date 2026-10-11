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


@pytest.mark.parametrize("cap", [1, 2, 32, 64, 128])
def test_study_cap_is_seedless_permutation_invariant_and_self_out(setup, cap):
    features, refs, split, counties = setup
    context = CountyContext(FeatureEncoder(features), refs, split, 0, counties,
        county_field="county", checkpoint_hash=digest("checkpoint"), reference_cap=cap)
    for query, county in (("t0", "c"), ("t1", "c"), ("t2", "c"), ("h0", "c"),
                          ("t4", "singleton"), ("new", "unseen")):
        expected = tuple(sorted(i for i, c in zip(refs.original_ids, counties)
                                if c == county and i != query))[:cap]
        assert context._reference_ids(query, county) == expected
        torch.manual_seed(3301)
        context._routes = dict(reversed(tuple(context._routes.items())))
        assert context._reference_ids(query, county) == expected
        assert query not in expected
    # Changed approved X cannot affect which references are selected.
    changed = replace(refs, values=tuple((999., row[1]) for row in refs.values))
    other = CountyContext(FeatureEncoder(features), changed, split, 0, counties,
        county_field="county", checkpoint_hash=digest("checkpoint"), reference_cap=cap)
    assert other._reference_ids("t0", "c") == context._reference_ids("t0", "c")


def test_study_cap_none_preserves_tensor_bytes_and_caps_actual_pool(setup, monkeypatch):
    from oxyformer.provenance import canonical_json
    plain = make_context(setup).eval()
    features, refs, split, counties = setup
    explicit = CountyContext(FeatureEncoder(features, dropout=0.), refs, split, 0, counties,
        county_field="county", checkpoint_hash=digest("checkpoint"), dropout=0., reference_cap=None).eval()
    explicit.load_state_dict(plain.state_dict())
    with torch.no_grad():
        a, b = plain(("t0", "h0"), ("c", "c")), explicit(("t0", "h0"), ("c", "c"))
    assert a.numpy().tobytes() == b.numpy().tobytes()
    assert canonical_json([k.reference_ids for k in plain.cache_keys]) == canonical_json([k.reference_ids for k in explicit.cache_keys])
    calls = []
    original = explicit._pool_batch
    def pool(sets):
        calls.extend(sets)
        return original(sets)
    monkeypatch.setattr(explicit, "_pool_batch", pool)
    explicit.reference_cap = 1
    with torch.no_grad():
        explicit(("t0", "h0"), ("c", "c"))
    assert calls == [("t1",), ("t0",)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("training,dropout", [(False, .1), (True, 0.)])
def test_batched_context_matches_scalar_outputs_gradients_and_rng(setup, device, training, dropout):
    from copy import deepcopy
    from oxyformer.training.fit import _numerics
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with _numerics(device):
        torch.manual_seed(1103)
        context = make_context(setup, dropout=dropout).to(device).train(training)
        reference = deepcopy(context)
        queries, routes = ("t0", "h0", "t1", "h1", "t4"), ("c", "c", "c", "d", "singleton")
        torch.manual_seed(2207)
        expected = torch.stack([reference._pool(reference._reference_ids(q, c)) for q, c in zip(queries, routes)])
        expected_rng = torch.get_rng_state().clone()
        weights = torch.linspace(-1, 1, expected.numel(), device=device).reshape_as(expected)
        (expected * weights).sum().backward()
        torch.manual_seed(2207)
        calls = []
        hook = context.encoder.register_forward_hook(lambda module, args, result: calls.append(len(args[0].numeric_values)))
        actual = context(queries, routes)
        hook.remove()
        assert len(calls) == 1
        assert calls[0] == (8 if training and dropout else 4)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        (actual * weights).sum().backward()
        for left, right in zip(context.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)
        torch.manual_seed(2207)
        repeated = context(queries, routes)
        assert actual.detach().cpu().numpy().tobytes() == repeated.detach().cpu().numpy().tobytes()


def test_batched_context_self_exclusion_and_padding(setup, monkeypatch):
    from copy import deepcopy
    context = make_context(setup).eval()
    altered = make_context(setup, refs=replace(setup[1], values=((10000., "rent"),) + setup[1].values[1:])).eval()
    altered.load_state_dict(context.state_dict())
    queries, routes = ("t0", "t1", "h1", "t4"), ("c", "c", "d", "singleton")
    expected = context(queries, routes)
    changed = altered(queries, routes)
    torch.testing.assert_close(expected[0], changed[0], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(expected[1], changed[1])
    assert torch.count_nonzero(expected[3]) == 0
    attention = context.attention.forward
    def poison_padding(query, key, value, **kwargs):
        padding = kwargs["key_padding_mask"]
        assert padding.any(), "fixture must exercise unequal reference lengths"
        key = key.masked_fill(padding[:, :, None], 1000.)
        value = value.masked_fill(padding[:, :, None], -1000.)
        return attention(query, key, value, **kwargs)
    monkeypatch.setattr(context.attention, "forward", poison_padding)
    actual = context(queries, routes)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_occurrence_masks_are_independent_and_have_correct_marginal(setup, device):
    from torch.utils._python_dispatch import TorchDispatchMode
    from oxyformer.training.fit import _numerics
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class RecordMasks(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.masks = []
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            result = operation(*args, **(kwargs or {}))
            if operation == torch.ops.aten.bernoulli_.float:
                self.masks.append(result.detach().cpu().clone())
            elif operation == torch.ops.aten.native_dropout.default:
                self.masks.append(result[1].detach().cpu().clone())
            return result

    with _numerics(device):
        torch.manual_seed(1103)
        context = make_context(setup, dropout=.1).to(device).train()
        # Distinct query IDs have the same three valid training references.
        queries = tuple(f"transfer-{i}" for i in range(256))
        routes = ("c",) * len(queries)
        torch.manual_seed(3301)
        trace = RecordMasks()
        calls = []
        hook = context.encoder.register_forward_hook(lambda module, args, result: calls.append(len(args[0].numeric_values)))
        with trace:
            first = context(queries, routes)
        hook.remove()
        assert calls == [768], "references must remain separate stochastic occurrences"
        mask = trace.masks[0].bool().reshape(256, -1)
        # Every occurrence has its own large mask; no query-wide mask is shared.
        assert len({row.numpy().tobytes() for row in mask}) == 256
        assert abs(float(mask.float().mean()) - .9) < .01
        weights = torch.linspace(-1, 1, first.numel(), device=device).reshape_as(first)
        (first * weights).mean().backward()
        gradients = [p.grad.detach().cpu().numpy().tobytes() for p in context.parameters() if p.grad is not None]
        context.zero_grad(set_to_none=True)
        torch.manual_seed(3301)
        second = context(queries, routes)
        (second * weights).mean().backward()
        assert first.detach().cpu().numpy().tobytes() == second.detach().cpu().numpy().tobytes()
        assert gradients == [p.grad.detach().cpu().numpy().tobytes() for p in context.parameters() if p.grad is not None]
