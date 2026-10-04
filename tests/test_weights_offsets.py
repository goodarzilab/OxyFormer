"""Analytic gradient checks keep target mass distinct from count exposure."""
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from oxyformer.contracts import EstimandSpec, SplitManifest
from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy, paired_records
from oxyformer.models.likelihoods import (CountyOffsets, bernoulli_loss, endpoint_loss,
                                         normalized_poisson_loss, squared_loss)
from oxyformer.models.origin import (calibrated_logit_to_ratio, paired_origin_loss,
                                     probability_to_ratio, weighted_class_prior)
from oxyformer.provenance import ArtifactLineage, ContractError


@pytest.fixture(autouse=True)
def cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def split():
    h = sha256(b"synthetic").hexdigest()
    ids = ("t0", "t1", "t2", "t3", "h0", "h1")
    spec = EstimandSpec(endpoint="synthetic", target_id="synthetic", outcome_scale="years",
                        policy_id="synthetic", weight_id="synthetic", adjustment_schema_hash=h,
                        inference_unit="county", source_lineage_hash=h)
    lineage = ArtifactLineage(source_hashes=(h,), unit_ids=ids + ("sealed",), parent_hashes=(),
                              split_hash=None, config_hash=h, model_hash=None,
                              environment=(("fixture", "cpu"),), seed=None, parameter_count=None)
    return SplitManifest(spec=spec, level="outer", original_ids=ids, fold_ids=(1, 1, 1, 1, 0, 0),
                         design_ids=("sealed",), excluded_ids=(), seed_ids=(11,),
                         entity_graph_hash=h, lineage=lineage)


def vec(values, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def test_normalized_poisson_log_rate_gradient_uses_target_mass():
    log_rate = vec([-.4, -2., .3], grad=True)
    deaths, population, weights = vec([0., 3., 80.]), vec([2., 100., 1000.]), vec([7., 1., 4.])
    loss = normalized_poisson_loss(log_rate, deaths, population, weights)
    expected_nll = -torch.distributions.Poisson(population * log_rate.exp()).log_prob(deaths)
    gradient, = torch.autograd.grad(loss, log_rate)
    torch.testing.assert_close(gradient, weights * (log_rate.exp() - deaths / population), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(loss, (weights / population * expected_nll).sum())


def test_rate_replication_changes_neither_gradient_nor_optimum():
    log_rate = vec([-.4, -2., .3], grad=True)
    deaths, population, weights = vec([0., 3., 80.]), vec([2., 100., 1000.]), vec([7., 1., 4.])
    g1, = torch.autograd.grad(normalized_poisson_loss(log_rate, deaths, population, weights), log_rate)
    g2, = torch.autograd.grad(normalized_poisson_loss(log_rate, 9*deaths, 9*population, weights), log_rate)
    torch.testing.assert_close(g1, g2)
    g3, = torch.autograd.grad(normalized_poisson_loss(log_rate, deaths, population, 3*weights), log_rate)
    torch.testing.assert_close(g3, 3*g1)
    mean_loss = normalized_poisson_loss(log_rate, deaths, population, weights, reduction="mean")
    gm, = torch.autograd.grad(mean_loss, log_rate)
    torch.testing.assert_close(gm, g1 / weights.sum())


@pytest.mark.parametrize("family", ["identity", "bernoulli"])
def test_endpoint_weighted_gradients(family):
    prediction, weights = vec([-.3, .5, 1.2], True), vec([1., 0., 7.])
    target = vec([0., 1., 1. if family == "bernoulli" else .2])
    function = squared_loss if family == "identity" else bernoulli_loss
    loss = function(prediction, target, weights)
    gradient, = torch.autograd.grad(loss, prediction)
    expected = 2*weights*(prediction-target) if family == "identity" else weights*(prediction.sigmoid()-target)
    torch.testing.assert_close(gradient, expected)
    torch.testing.assert_close(function(prediction, target, weights, reduction="none").sum(), loss)


def test_loss_refuses_broadcast_or_missing_population():
    x = vec([1., 2.])
    with pytest.raises(ContractError, match="identical"):
        squared_loss(x[:, None], x, x)
    with pytest.raises(ContractError, match="separate population"):
        endpoint_loss(x, x, x, family="poisson")
    with pytest.raises(ContractError, match="positive"):
        normalized_poisson_loss(x, x, vec([0., 2.]), x)


def test_profiled_identity_offsets_update_and_cancel(split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family="identity",
                            exposure_assignment_level="tract").double()
    ids = split.training_ids(0)
    y, base, weights = vec([4., 10., 2., 8.]), vec([1., 2., 3., 4.], True), vec([1., 3., 2., 2.])
    offsets.update_identity(ids, y, base, weights)
    torch.testing.assert_close(offsets.values, vec([6.75, 1.5]))
    loss = offsets.training_loss(ids, base, y, weights)
    g, = torch.autograd.grad(loss, base)
    torch.testing.assert_close(g, 2*weights*(base + offsets(("c", "c", "d", "d")) - y))
    assert not offsets.values.requires_grad
    assert abs(g[:2].sum()) < 1e-12 and abs(g[2:].sum()) < 1e-12
    observed, shifted = base.detach(), base.detach() + vec([2., -3., 0., 5.])
    gamma = offsets(("c", "c", "d", "d"))
    torch.testing.assert_close((shifted + gamma) - (observed + gamma), shifted-observed)
    offsets.update_identity(ids, y, base.detach() + 2, weights)
    torch.testing.assert_close(offsets.values, vec([4.75, -.5]))
    assert offsets(("unseen",)).item() == 0


@pytest.mark.parametrize("family", ["bernoulli", "poisson"])
def test_joint_link_offsets_solve_target_weighted_score(family, split):
    offsets = CountyOffsets(split, 0, ("c",)*4, family=family,
                            exposure_assignment_level="tract").double()
    ids = split.training_ids(0)
    base, weights = vec([0., 0., 0., 0.]), vec([1., 3., 2., 4.])
    target = vec([0., 1., 0., 1.]) if family == "bernoulli" else vec([0., 4., 20., 60.])
    population = None if family == "bernoulli" else vec([10., 20., 200., 300.])
    optimizer = torch.optim.LBFGS(offsets.parameters(), lr=1., max_iter=60,
                                  tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = offsets.training_loss(ids, base, target, weights, population=population)
        loss.backward()
        return loss
    optimizer.step(closure)
    target_mean = (weights * (target if population is None else target/population)).sum()/weights.sum()
    expected = torch.logit(target_mean) if family == "bernoulli" else target_mean.log()
    torch.testing.assert_close(offsets.values[0], expected, atol=1e-6, rtol=1e-6)
    assert offsets.values.grad.abs().max() < 1e-6


@pytest.mark.parametrize("bad_id", ["h0", "sealed", "unknown"])
def test_offsets_reject_heldout_labels_and_leave_state_unchanged(split, bad_id):
    offsets = CountyOffsets(split, 0, ("c",)*4, family="identity", exposure_assignment_level="tract")
    ids = (bad_id,) + split.training_ids(0)[1:]
    before = offsets.values.clone()
    with pytest.raises(ContractError, match="permitted training IDs"):
        offsets.update_identity(ids, vec([999.]*4), vec([0.]*4), vec([1.]*4))
    with pytest.raises(ContractError, match="permitted training IDs"):
        offsets.training_loss(ids, vec([0.]*4), vec([999.]*4), vec([1.]*4))
    torch.testing.assert_close(offsets.values, before)


def test_offset_ownership_and_exposure_unit_guard(split):
    with pytest.raises(ContractError, match="absorb treatment"):
        CountyOffsets(split, 0, ("c",)*4, family="identity", exposure_assignment_level="county")
    a = CountyOffsets(split, 0, ("c",)*4, family="identity", exposure_assignment_level="tract")
    b = CountyOffsets(replace(split, seed_ids=(12,)), 0, ("c",)*4,
                      family="identity", exposure_assignment_level="tract")
    with pytest.raises(ContractError, match="ownership"):
        b.load_state_dict(a.state_dict())
    with pytest.raises(ContractError, match="all training IDs"):
        a.update_identity(("t0",), vec([1.]), vec([0.]), vec([1.]))


def test_origin_pairs_retain_origin_weights_and_weighted_bce():
    h = sha256(b"synthetic-design").hexdigest()
    policy = ShiftOrStayPolicy(support_design_hash=h, components_by_key=(("s", ((0., 10.),)),))
    covariates = PolicyCovariates(original_ids=("a", "b", "c"), geography_ids=("a", "b", "c"),
                                  support_keys=("s",)*3)
    result = policy.apply([1., 8., 9.], covariates)
    pairs = paired_records(result, [1., 7., 3.], weight_id="synthetic-target")
    assert pairs.origin_weights == (1., 7., 3., 1., 7., 3.)
    assert pairs.a_mmhg == (1., 8., 9., 3., 10., 9.)
    assert weighted_class_prior(pairs) == .5
    logits = vec([-.5, .2, 1., 2., .1, -.3], True)
    loss = paired_origin_loss(logits, pairs)
    gradient, = torch.autograd.grad(loss, logits)
    weights, target = vec(pairs.origin_weights), vec(pairs.transformed)
    torch.testing.assert_close(gradient, weights/weights.sum()*(logits.sigmoid()-target))


@pytest.mark.parametrize("prior", [.2, .5, .8])
def test_explicit_class_prior_correction(prior):
    true_ratio = vec([.1, 1., 2., 10.])
    eta = prior*true_ratio/(1-prior+prior*true_ratio)
    torch.testing.assert_close(probability_to_ratio(eta, class_prior=prior), true_ratio)
    torch.testing.assert_close(calibrated_logit_to_ratio(torch.logit(eta), class_prior=prior), true_ratio)
    assert probability_to_ratio(vec([prior]), class_prior=prior).item() == pytest.approx(1.)
    if prior != .5:
        assert not torch.allclose(probability_to_ratio(eta, class_prior=.5), true_ratio)


def test_origin_offset_training_accepts_both_copies_of_permitted_ids(split):
    offsets = CountyOffsets(split, 0, ("c",)*4, family="bernoulli", exposure_assignment_level="tract").double()
    ids = split.training_ids(0) * 2
    loss = offsets.training_loss(ids, vec([0.]*8), vec([0.]*4 + [1.]*4), vec([1., 2., 3., 4.]*2))
    gradient, = torch.autograd.grad(loss, offsets.values)
    torch.testing.assert_close(gradient, vec([0.]))


def test_identity_profile_repeated_rows_use_supplied_multiplicity_weights(split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family="identity",
                            exposure_assignment_level="tract").double()
    ids = split.training_ids(0)
    y, base, weights = vec([4., 10., 2., 8.]), vec([1., 2., 3., 4.]), vec([1., 3., 2., 2.])
    offsets.update_identity(ids, y, base, weights)
    expected = offsets.values.detach().clone()
    # Splitting each original target mass over repeated rows preserves profiling.
    offsets.update_identity(ids*2, y.repeat(2), base.repeat(2), (weights/2).repeat(2))
    torch.testing.assert_close(offsets.values, expected)
    loss = offsets.training_loss(ids*2, base.repeat(2), y.repeat(2), (weights/2).repeat(2))
    torch.testing.assert_close(loss, offsets.training_loss(ids, base, y, weights))


@pytest.mark.parametrize("death_dtype,population_dtype", [
    (torch.int64, torch.float64), (torch.float64, torch.int64), (torch.int64, torch.int64),
])
def test_natural_integer_count_inputs_preserve_predictor_precision(death_dtype, population_dtype):
    log_rate = vec([-.4, -2., .3], True)
    deaths = torch.tensor([0, 3, 80], dtype=death_dtype)
    population = torch.tensor([2, 100, 1000], dtype=population_dtype)
    weights = vec([7., 1., 4.])
    loss = endpoint_loss(log_rate, deaths, weights, family="poisson", population=population)
    reference = normalized_poisson_loss(log_rate, deaths.double(), population.double(), weights)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
    gradient, = torch.autograd.grad(loss, log_rate)
    torch.testing.assert_close(gradient, weights * (log_rate.exp() - deaths.double()/population.double()),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("entrypoint", ["direct", "endpoint", "offset"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_bernoulli_integer_labels_values_and_gradients(dtype, entrypoint, reduction, split):
    logits = torch.tensor([-.3, .5, 1.2, -1.5], dtype=dtype, requires_grad=True)
    target = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    weights = torch.tensor([1., 3., 0., 7.], dtype=dtype)
    if entrypoint == "direct":
        loss = bernoulli_loss(logits, target, weights, reduction=reduction)
    elif entrypoint == "endpoint":
        loss = endpoint_loss(logits, target, weights, family="bernoulli", reduction=reduction)
    else:
        offsets = CountyOffsets(split, 0, ("c",)*4, family="bernoulli",
                                exposure_assignment_level="tract").to(dtype=dtype)
        loss = offsets.training_loss(split.training_ids(0), logits, target, weights, reduction=reduction)
    floating_target = target.to(dtype)
    rows = weights * (torch.nn.functional.softplus(logits) - floating_target * logits)
    expected = rows if reduction == "none" else rows.sum()
    if reduction == "mean":
        expected = expected / weights.sum()
    torch.testing.assert_close(loss, expected)
    gradient, = torch.autograd.grad(loss.sum(), logits)
    expected_gradient = weights * (logits.sigmoid() - floating_target)
    if reduction == "mean":
        expected_gradient = expected_gradient / weights.sum()
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("target_dtype", [torch.bool, torch.int32, torch.float32, torch.float64])
def test_bernoulli_accepts_binary_label_representations(target_dtype):
    logits = vec([-.3, .5], True)
    target = torch.tensor([0, 1], dtype=target_dtype)
    weights = vec([1., 3.])
    loss = bernoulli_loss(logits, target, weights)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, weights * (logits.sigmoid() - vec([0., 1.])))


@pytest.mark.parametrize("invalid", [.25, -1., 2., 1. + 1e-8, 1e-50, float("nan"), float("inf")])
def test_bernoulli_rejects_nonbinary_labels_before_cast(invalid):
    # These near-endpoint float64 values would round to 0/1 if cast too early.
    target = torch.tensor([0., invalid], dtype=torch.float64)
    with pytest.raises(ContractError, match="Bernoulli|nonfinite"):
        bernoulli_loss(torch.zeros(2, dtype=torch.float32), target, torch.ones(2))
