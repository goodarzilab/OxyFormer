"""Synthetic CPU evidence for complete, independent treatment-query nuisances."""
from dataclasses import replace
from hashlib import sha256

import pytest
import torch
import yaml

from oxyformer.contracts import CovariateView, EstimandSpec, SplitManifest
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.models.county_context import CountyContext
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.likelihoods import CountyOffsets
from oxyformer.models.origin import OriginTransformer, paired_origin_loss
from oxyformer.models.outcome import OutcomeTransformer
from oxyformer.models.riesz import RieszTransformer, riesz_loss
from oxyformer.models.tokens import FeatureSpec
from oxyformer.models.treatment_query import TreatmentBasis, TreatmentDesign
from oxyformer.provenance import ArtifactLineage, ContractError


def digest(text):
    return sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def cpu_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(908)
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def design():
    return TreatmentDesign(center=4., scale=2., knots=(-2., 0., 2., 4., 6., 8.),
                           design_hash=digest("synthetic-design"))


@pytest.fixture
def backbone():
    features = tuple(FeatureSpec(name=f"x{j}", kind="numeric") for j in range(3))
    rules = tuple(FeatureRule(name=f.name, role="predictor", endpoints=("synthetic",),
                              uses=("nuisance", "context"), approval_id="synthetic") for f in features)
    rules += (FeatureRule(name="county", role="county", endpoints=("synthetic",),
                          uses=("county_routing",), approval_id="synthetic"),)
    registry = FeatureRegistry(registry_id="synthetic", rules=rules)
    spec = EstimandSpec(endpoint="synthetic", target_id="synthetic", outcome_scale="years",
                        policy_id="synthetic", weight_id="synthetic", adjustment_schema_hash=registry.content_hash,
                        inference_unit="county", source_lineage_hash=digest("source"))
    ids = ("t0", "t1", "t2", "t3", "h0", "h1")
    lineage = ArtifactLineage(source_hashes=(digest("payload"),), unit_ids=ids + ("sealed",),
                              parent_hashes=(), split_hash=None, config_hash=digest("config"),
                              model_hash=None, environment=(("fixture", "cpu"),), seed=None, parameter_count=None)
    split = SplitManifest(spec=spec, level="outer", original_ids=ids, fold_ids=(1, 1, 1, 1, 0, 0),
                          design_ids=("sealed",), excluded_ids=(), seed_ids=(11,),
                          entity_graph_hash=digest("graph"), lineage=lineage)
    train_ids = split.training_ids(0)
    refs = CovariateView(spec=spec, registry=registry, original_ids=train_ids, columns=("x0", "x1", "x2"),
                         values=((0., 1., 2.), (1., 4., 2.), (-1., 1., 8.), (2., -1., 0.)),
                         use="context", lineage=replace(lineage, unit_ids=train_ids, split_hash=split.content_hash))
    encoder = FeatureEncoder(features, dropout=0.)
    counties = ("c", "c", "d", "d")
    context = CountyContext(encoder, refs, split, 0, counties, county_field="county",
                            checkpoint_hash=digest("checkpoint"), dropout=0.)
    return encoder, context, split, counties


def make_model(kind, design, backbone):
    encoder, context, _, _ = backbone
    return kind(encoder, treatment_design=design, raw_x_dim=3, county_context=context, dropout=0.).eval()


def inputs():
    return (torch.tensor([[[1.], [3.], [5.]], [[2.], [4.], [7.]]]),
            torch.randn(2, 4, 64), torch.randn(2, 3), torch.randn(2, 4, 64))


def predict(model, args):
    return model(*args) if isinstance(model, RieszTransformer) else model(*args, torch.tensor([1., -2.]))


@pytest.mark.parametrize("kind", [OutcomeTransformer, OriginTransformer, RieszTransformer])
def test_query_set_independence_and_zero_cross_dose_gradient(kind, design, backbone):
    model = make_model(kind, design, backbone)
    a, states, raw, context = inputs()
    all_predictions = predict(model, (a, states, raw, context))
    for j in range(3):
        alone = predict(model, (a[:, j:j+1], states, raw, context))
        torch.testing.assert_close(alone[:, 0], all_predictions[:, j], rtol=1e-5, atol=1e-6)
    order = torch.tensor([2, 0, 1, 1])
    torch.testing.assert_close(predict(model, (a[:, order], states, raw, context)), all_predictions[:, order])
    altered = a.clone()
    altered[:, 1:] += 25
    torch.testing.assert_close(predict(model, (altered, states, raw, context))[:, 0], all_predictions[:, 0])
    a.requires_grad_()
    gradient, = torch.autograd.grad(predict(model, (a, states, raw, context))[:, 0].sum(), a)
    assert gradient[:, 0].abs().sum() > 0
    assert gradient[:, 1:].count_nonzero() == 0


@pytest.mark.parametrize("kind", [OutcomeTransformer, OriginTransformer, RieszTransformer])
def test_readout_receives_every_raw_covariate_and_gradients(kind, design, backbone):
    model = make_model(kind, design, backbone)
    a, states, raw, context = inputs()
    raw.requires_grad_()
    seen = []
    hook = model.readout[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    value = predict(model, (a, states, raw, context))
    hook.remove()
    torch.testing.assert_close(seen[0][..., 128:131], raw[:, None].expand(-1, 3, -1))
    gradient, = torch.autograd.grad(value.sum(), raw)
    assert (gradient.abs().sum(0) > 0).all()
    assert model.readout[0].out_features == 64 and model.readout[2].out_features == 32
    with pytest.raises(ContractError, match="complete approved raw-X"):
        predict(model, (a, states, raw[:, :2], context))


def test_independent_full_nuisance_parameters_and_training(design, backbone):
    models = [make_model(kind, design, backbone) for kind in (OutcomeTransformer, OriginTransformer, RieszTransformer)]
    parameter_sets = [{p.data_ptr() for p in model.parameters()} for model in models]
    original = {p.data_ptr() for p in backbone[1].parameters()}
    for i, model in enumerate(models):
        assert model.county_context.encoder is model.encoder
        assert not parameter_sets[i] & original
        assert 100_000 < model.check_parameter_cap() <= 1_000_000
        for j in range(i):
            assert not parameter_sets[i] & parameter_sets[j]
    before = [[p.detach().clone() for p in m.parameters()] for m in models]
    model = models[0]
    refs = model.county_context.references
    batch = model.encoder.tokenizer.prepare(refs)
    a = torch.ones(4, 2, 1)
    context = model.county_context(refs.original_ids, backbone[3])
    value = model(a, batch, torch.tensor(refs.values), context, torch.zeros(4))
    value.square().sum().backward()
    assert model.encoder.cls.grad.abs().sum() > 0
    assert model.county_context.seeds.grad.abs().sum() > 0
    torch.optim.SGD(model.parameters(), lr=.1).step()
    assert any(not torch.equal(old, new) for old, new in zip(before[0], model.parameters()))
    for snapshot, other in zip(before[1:], models[1:]):
        assert all(torch.equal(old, new) for old, new in zip(snapshot, other.parameters()))


def test_parameter_cap_includes_backbone_readout_and_owned_offsets(design, backbone):
    encoder, context, split, counties = backbone
    offsets = CountyOffsets(split, 0, counties, family="identity", exposure_assignment_level="tract")
    model = OutcomeTransformer(encoder, treatment_design=design, raw_x_dim=3,
                               county_context=context, group_offsets=offsets)
    assert model.group_offsets is not offsets
    assert model.group_offsets.values.data_ptr() != offsets.values.data_ptr()
    assert model.check_parameter_cap() == sum(p.numel() for p in model.parameters())
    with pytest.raises(ContractError, match="parameter cap"):
        OutcomeTransformer(encoder, treatment_design=design, raw_x_dim=16000, county_context=context)


def test_spline_is_continuous_frozen_and_checkpointed(design):
    basis = TreatmentBasis(design).double()
    a = torch.tensor([[[4.], [6.]]], dtype=torch.float64, requires_grad=True)
    value = basis(a)
    assert value.shape == (1, 2, 7) and not list(basis.parameters())
    torch.testing.assert_close(value[..., 0], torch.tensor([[0., 1.]], dtype=torch.float64))
    torch.testing.assert_close(value[..., 1:], ((a - a.new_tensor(design.knots)) / 2).clamp_min(0).pow(3))
    knots = a.new_tensor(design.knots).reshape(1, 6, 1)
    torch.testing.assert_close(basis(knots - 1e-8), basis(knots + 1e-8), atol=1e-5, rtol=1e-5)
    with pytest.raises(ContractError, match="override"):
        basis(a, replace(design, center=5.))
    other = TreatmentBasis(replace(design, center=10.))
    other.load_state_dict(basis.state_dict())
    assert other.design == design
    torch.testing.assert_close(other(a), value)


@pytest.mark.parametrize("family,expected", [("identity", -2.), ("bernoulli", torch.sigmoid(torch.tensor(-2.)).item()),
                                              ("poisson", torch.exp(torch.tensor(-2.)).item())])
def test_outcome_endpoint_links(family, expected, design, backbone):
    model = OutcomeTransformer(backbone[0], treatment_design=design, raw_x_dim=3, family=family).eval()
    with torch.no_grad():
        model.readout[-1].weight.zero_()
        model.readout[-1].bias.fill_(-3.)
    result = model.mean(*inputs(), torch.ones(2))
    torch.testing.assert_close(result, torch.full((2, 3), expected))


def test_signed_riesz_and_functional_gradient(design, backbone):
    model = make_model(RieszTransformer, design, backbone)
    for signed in (-2., 3.):
        with torch.no_grad():
            model.readout[-1].weight.zero_()
            model.readout[-1].bias.fill_(signed)
        torch.testing.assert_close(model(*inputs()), torch.full((2, 3), signed))
    a = torch.tensor([-2., .5, 1.], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([3., -.2, 2.], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1., 7., 2.], dtype=torch.float64)
    loss = riesz_loss(a, d, weights)
    torch.testing.assert_close(loss, (weights * (a*a - 2*(d-a))).sum() / weights.sum())
    ga, gd = torch.autograd.grad(loss, (a, d))
    torch.testing.assert_close(ga, weights / weights.sum() * (2*a + 2))
    torch.testing.assert_close(gd, -2*weights/weights.sum())
    identity = riesz_loss(a, a, weights)
    torch.testing.assert_close(identity, (weights*a*a).sum()/weights.sum())


def test_no_monotonic_sign_constraint(design, backbone):
    model = make_model(OutcomeTransformer, design, backbone)
    a, states, raw, context = inputs()
    slopes = []
    for sign in (-1., 1.):
        with torch.no_grad():
            for p in model.readout.parameters():
                p.zero_()
            model.readout[0].weight[0, 128 + 3] = 1
            model.readout[2].weight[0, 0] = 1
            model.readout[4].weight[0, 0] = sign
        predictions = model.mean(a + 8, states, raw, context, torch.zeros(2))
        slopes.append(predictions[:, -1] - predictions[:, 0])
    assert (slopes[0] < 0).all() and (slopes[1] > 0).all()


def test_config_declares_frozen_head_contract():
    from pathlib import Path
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/models/oxyformer_v2.yaml").read_text())
    assert config["nuisance_parameter_cap"] == 1_000_000
    assert config["treatment_query"]["readout_hidden_widths"] == [64, 32]
    assert config["treatment_query"]["spline_values"] == 6
    assert config["treatment_query"]["query_query_attention"] is False


def test_identity_head_offsets_cancel_but_change_residual_predictions(design, backbone):
    model = make_model(OutcomeTransformer, design, backbone)
    args = inputs()
    zero = model.mean(*args, torch.zeros(2))
    gamma = torch.tensor([8., -4.])
    with_offset = model.mean(*args, gamma)
    torch.testing.assert_close(with_offset - zero, gamma[:, None].expand(-1, 3))
    torch.testing.assert_close(with_offset[:, 1] - with_offset[:, 0], zero[:, 1] - zero[:, 0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model.mean(*args, gamma + 3) - with_offset, torch.full((2, 3), 3.))


@pytest.mark.parametrize("kind", [OutcomeTransformer, OriginTransformer, RieszTransformer])
def test_precomputed_states_preserve_padding_and_gradients(kind, design, backbone):
    model = make_model(kind, design, backbone)
    batch = model.encoder.tokenizer.prepare(model.county_context.references)
    padding = batch.padding.clone()
    padding[0, 1:] = True
    padding[1, 2] = True
    batch = replace(batch, padding=padding)
    a, raw, context = torch.ones(4, 2, 1), torch.randn(4, 3), torch.randn(4, 4, 64)
    def run(x):
        if kind is RieszTransformer:
            return model(a, x, raw, context)
        return model(a, x, raw, context, torch.zeros(4))
    direct = run(batch)
    encoded = model.encode(batch)
    cached = run(encoded)
    torch.testing.assert_close(cached, direct, rtol=0, atol=0)
    parameter = model.encoder.tokenizer.numeric_weight
    g_direct, = torch.autograd.grad(direct.sum(), parameter)
    g_cached, = torch.autograd.grad(cached.sum(), parameter)
    torch.testing.assert_close(g_direct, g_cached, rtol=0, atol=0)
    # Changing the caller's mask after encoding cannot change saved states/mask.
    saved = encoded.padding.clone()
    batch.padding.zero_()
    torch.testing.assert_close(encoded.padding, saved)


def test_origin_query_matrix_loss_preserves_policy_pair_order(design, backbone):
    from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy, paired_records
    model = make_model(OriginTransformer, design, backbone)
    policy = ShiftOrStayPolicy(support_design_hash=design.design_hash, components_by_key=(("s", ((0., 10.),)),))
    ids = ("t0", "t1", "t2", "t3")
    cov = PolicyCovariates(original_ids=ids, geography_ids=ids, support_keys=("s",)*4)
    result = policy.apply([1., 2., 8., 9.], cov)
    pairs = paired_records(result, [1., 2., 3., 4.], weight_id="synthetic")
    queries = torch.tensor(list(zip(result.a_mmhg, result.d_mmhg))).unsqueeze(-1)
    batch = model.encoder.tokenizer.prepare(model.county_context.references)
    logits = model.logits(queries, batch, torch.randn(4, 3), torch.randn(4, 4, 64), torch.zeros(4))
    loss = paired_origin_loss(logits, pairs)
    column_order = torch.cat((logits[:, 0], logits[:, 1]))
    torch.testing.assert_close(loss, paired_origin_loss(column_order, pairs))
    torch.testing.assert_close(loss, paired_origin_loss(column_order[:, None], pairs))
    gradient, = torch.autograd.grad(loss, logits)
    weights = logits.new_tensor([1., 2., 3., 4.])[:, None]
    labels = logits.new_tensor([0., 1.])[None, :]
    torch.testing.assert_close(gradient, weights / (2*weights.sum()) * (logits.sigmoid() - labels))
