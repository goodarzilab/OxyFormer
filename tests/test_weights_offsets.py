"""Analytic gradient checks keep target mass distinct from count exposure."""
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from oxyformer.contracts import EstimandSpec, SplitManifest
from oxyformer.design.policies import PolicyCovariates, PolicyPairs, ShiftOrStayPolicy, paired_records
from oxyformer.models.likelihoods import (CountyOffsets, bernoulli_loss, endpoint_loss,
                                         normalized_poisson_loss, squared_loss, weighted_reduce)
from oxyformer.models.origin import (calibrated_logit_to_ratio, paired_origin_loss,
                                     probability_to_ratio, weighted_class_prior)
from oxyformer.models.riesz import riesz_loss
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


@pytest.mark.parametrize("family", ["identity", "bernoulli", "poisson"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_mass_minibatches_have_zero_loss_and_gradient(family, reduction, dtype):
    prediction = torch.tensor([0., 1., 1000., -1000.], dtype=dtype, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1])
    weights = torch.zeros(4, dtype=dtype)
    population = torch.tensor([100, 200, 1000000, 10]) if family == "poisson" else None
    loss = endpoint_loss(prediction, target, weights, family=family,
                         population=population, reduction=reduction)
    expected = torch.zeros_like(prediction) if reduction == "none" else prediction.new_zeros(())
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(loss.sum(), prediction)
    torch.testing.assert_close(gradient, torch.zeros_like(prediction), rtol=0, atol=0)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("excluded_log_rate", [80., 1000.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_poisson_zero_weight_rows_cannot_overflow_loss_or_gradient(reduction, excluded_log_rate, dtype):
    log_rate = torch.tensor([0., excluded_log_rate], dtype=dtype, requires_grad=True)
    loss = normalized_poisson_loss(log_rate, torch.tensor([0, 0]), torch.tensor([1, 1000000]),
                                   torch.tensor([1., 0.], dtype=dtype), reduction=reduction)
    expected = log_rate.new_tensor([1., 0.]) if reduction == "none" else log_rate.new_tensor(1.)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(loss.sum(), log_rate)
    torch.testing.assert_close(gradient, log_rate.new_tensor([1., 0.]), rtol=0, atol=0)


def test_poisson_normalization_avoids_count_mean_overflow_for_active_rows():
    log_rate = torch.tensor([80.], requires_grad=True)
    loss = normalized_poisson_loss(log_rate, torch.tensor([0]), torch.tensor([1000000]), torch.tensor([2.]))
    expected = 2 * log_rate.exp().sum()
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)
    gradient, = torch.autograd.grad(loss, log_rate)
    torch.testing.assert_close(gradient, 2 * log_rate.exp())


@pytest.mark.parametrize("active_weight", [0., 1.])
def test_zero_weight_polynomial_losses_keep_finite_gradients(active_weight):
    from oxyformer.models.riesz import riesz_loss
    weights = torch.tensor([active_weight, 0.])
    a = torch.tensor([1., 1e30], requires_grad=True)
    d = torch.tensor([2., -1e30], requires_grad=True)
    square = squared_loss(a, torch.zeros(2), weights, reduction="mean")
    torch.testing.assert_close(square, torch.tensor(active_weight))
    gradient, = torch.autograd.grad(square, a)
    torch.testing.assert_close(gradient, torch.tensor([2 * active_weight, 0.]))
    riesz = riesz_loss(a, d, weights)
    torch.testing.assert_close(riesz, torch.tensor(-active_weight))
    ga, gd = torch.autograd.grad(riesz, (a, d))
    torch.testing.assert_close(ga, torch.tensor([4 * active_weight, 0.]))
    torch.testing.assert_close(gd, torch.tensor([-2 * active_weight, 0.]))


@pytest.mark.parametrize("family", ["identity", "bernoulli", "poisson"])
def test_positive_fractional_mass_keeps_mean_normalization(family):
    prediction = vec([-.3, .5])
    target = vec([0., 1.])
    weights = vec([.02, .03])
    population = vec([10., 20.]) if family == "poisson" else None
    first = endpoint_loss(prediction, target, weights, family=family, population=population, reduction="mean")
    rescaled = endpoint_loss(prediction, target, weights*100, family=family, population=population, reduction="mean")
    torch.testing.assert_close(first, rescaled)
    with pytest.raises(ContractError, match="invalid target weights"):
        endpoint_loss(prediction, target, vec([-1., 1.]), family=family, population=population)


def origin_pairs_with_weights(weights):
    n = len(weights) // 2
    ids = tuple(f"t{i}" for i in range(n)) * 2
    return PolicyPairs(policy_id="synthetic", weight_id="synthetic", original_ids=ids,
                       geography_ids=ids, support_keys=("s",) * len(weights),
                       a_mmhg=(100.,) * len(weights), transformed=(False,) * n + (True,) * n,
                       origin_weights=tuple(weights.tolist()))


WEIGHTED_ROUTES = [(case, reduction) for case in
                   ("identity", "bernoulli", "poisson", "origin", "reducer")
                   for reduction in ("none", "sum", "mean")] + [("riesz", "mean")]


def weighted_route(case, prediction, target, population, weights, reduction):
    if case == "riesz":
        return riesz_loss(prediction, target, weights)
    if case == "origin":
        return paired_origin_loss(prediction, origin_pairs_with_weights(weights), reduction=reduction)
    if case == "reducer":
        return weighted_reduce(prediction, weights, reduction)
    return endpoint_loss(prediction, target, weights, family=case,
                         population=population if case == "poisson" else None, reduction=reduction)


@pytest.mark.parametrize("case,reduction", WEIGHTED_ROUTES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_weighted_losses_keep_autograd(case, reduction, dtype):
    prediction = torch.empty(0, dtype=dtype, requires_grad=True)
    target = torch.empty(0, dtype=dtype, requires_grad=case == "riesz")
    weights = torch.empty(0, dtype=dtype)
    loss = weighted_route(case, prediction, target, weights, weights, reduction)
    expected = weights if reduction == "none" else weights.new_zeros(())
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    operands = (prediction, target) if case == "riesz" else (prediction,)
    for gradient in torch.autograd.grad(loss.sum(), operands):
        torch.testing.assert_close(gradient, weights, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["identity", "bernoulli", "poisson"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_offset_training_keeps_autograd(family, reduction, dtype, split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family=family,
                            exposure_assignment_level="tract").to(dtype=dtype)
    base = torch.empty(0, dtype=dtype, requires_grad=True)
    empty = torch.empty(0, dtype=dtype)
    loss = offsets.training_loss((), base, empty, empty, reduction=reduction,
                                 population=empty if family == "poisson" else None)
    torch.testing.assert_close(loss, empty if reduction == "none" else empty.new_zeros(()), rtol=0, atol=0)
    operands = (base, offsets.values) if offsets.values.requires_grad else (base,)
    for operand, gradient in zip(operands, torch.autograd.grad(loss.sum(), operands)):
        torch.testing.assert_close(gradient, torch.zeros_like(operand), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_profile_excluded_residual_and_zero_mass_reset(dtype, split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family="identity",
                            exposure_assignment_level="tract").to(dtype=dtype)
    ids = split.training_ids(0)
    extreme = torch.finfo(dtype).max * .75
    target = torch.tensor([extreme, 1., 2., 3.], dtype=dtype)
    base = torch.tensor([-extreme, 0., 0., 0.], dtype=dtype, requires_grad=True)
    weights = torch.tensor([0., 1., 1., 1.], dtype=dtype)
    offsets.update_identity(ids, target, base, weights)
    torch.testing.assert_close(offsets.values, base.new_tensor([1., 2.5]), rtol=0, atol=0)
    loss = offsets.training_loss(ids, base, target, weights)
    torch.testing.assert_close(loss, base.new_tensor(.5), rtol=0, atol=0)
    gradient, = torch.autograd.grad(loss, base)
    torch.testing.assert_close(gradient, base.new_tensor([0., 0., 1., -1.]), rtol=0, atol=0)
    previous = offsets.values.detach().clone()
    offsets.update_identity(ids, target.masked_fill(weights == 0, 0),
                            base.detach().masked_fill(weights == 0, 0), weights)
    torch.testing.assert_close(offsets.values, previous, rtol=0, atol=0)
    weights[:2] = 0
    offsets.update_identity(ids, target, base, weights)
    torch.testing.assert_close(offsets.values, base.new_tensor([0., 2.5]), rtol=0, atol=0)
    offsets.update_identity(ids, target, base, torch.zeros_like(weights))
    torch.testing.assert_close(offsets.values, torch.zeros_like(offsets.values), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["identity", "bernoulli", "poisson"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("zero_mass", [False, True])
def test_offset_composition_excluded_operands(family, reduction, dtype, zero_mass, split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family=family,
                            exposure_assignment_level="tract").to(dtype=dtype)
    extreme = torch.finfo(dtype).max * .75
    with torch.no_grad():
        offsets.values.copy_(torch.tensor([extreme, 0.], dtype=dtype))
    base = torch.tensor([extreme, -extreme, -extreme, 0.], dtype=dtype, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1])
    weights = torch.tensor([0., 1., 0., 1.], dtype=dtype)
    if zero_mass:
        weights.zero_()
    kwargs = dict(population=torch.ones(4, dtype=dtype)) if family == "poisson" else {}
    loss = offsets.training_loss(split.training_ids(0), base, target, weights, reduction=reduction, **kwargs)
    operands = (base, offsets.values) if offsets.values.requires_grad else (base,)
    gradients = torch.autograd.grad(loss.sum(), operands)
    reference = base.new_zeros(4).requires_grad_()
    expected = endpoint_loss(reference, target, weights, family=family, reduction=reduction, **kwargs)
    expected_gradient, = torch.autograd.grad(expected.sum(), reference)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], expected_gradient, rtol=0, atol=0)
    if offsets.values.requires_grad:
        expected_offset = torch.stack((expected_gradient[:2].sum(), expected_gradient[2:].sum()))
        torch.testing.assert_close(gradients[1], expected_offset, rtol=0, atol=0)


@pytest.mark.parametrize("case,reduction", WEIGHTED_ROUTES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("zero_mass", [False, True])
def test_weighted_routes_ignore_valid_excluded_operands(case, reduction, dtype, zero_mass):
    weights = torch.tensor([0., 1., 0., 1.], dtype=dtype)
    if zero_mass:
        weights.zero_()
    excluded = weights == 0
    maximum = torch.finfo(dtype).max * .75
    prediction = torch.tensor([maximum, .5, -maximum, -.3], dtype=dtype)
    target = torch.tensor([0., 1., 1., 0.], dtype=dtype)
    population = torch.ones(4, dtype=dtype)
    if case in ("identity", "riesz"):
        target[0], target[2] = -maximum, maximum
    if case == "poisson":
        prediction[0], prediction[2] = 1000., -1000.
        target[0], target[2] = maximum, maximum
        population[0], population[2] = torch.finfo(dtype).tiny, maximum

    def evaluate(benign):
        p = (prediction.masked_fill(excluded, 0) if benign else prediction).clone().requires_grad_()
        y = (target.masked_fill(excluded, 0) if benign else target).clone().requires_grad_(case == "riesz")
        n = population.masked_fill(excluded, 1) if benign else population
        value = weighted_route(case, p, y, n, weights, reduction)
        return value, torch.autograd.grad(value.sum(), (p, y) if case == "riesz" else (p,))

    actual, gradients = evaluate(False)
    expected, expected_gradients = evaluate(True)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for gradient, reference in zip(gradients, expected_gradients):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        torch.testing.assert_close(gradient[excluded], torch.zeros_like(gradient[excluded]), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["base", "offset", "label", "death", "population"])
def test_offset_exclusion_never_launders_invalid_inputs(invalid, split):
    family = "poisson" if invalid in ("death", "population") else "bernoulli"
    offsets = CountyOffsets(split, 0, ("c", "d", "c", "d"), family=family,
                            exposure_assignment_level="tract")
    base, target, weights = torch.zeros(4), torch.zeros(4), torch.tensor([0., 1., 0., 1.])
    population = torch.ones(4)
    if invalid == "base":
        base[0] = float("nan")
    elif invalid == "offset":
        with torch.no_grad():
            offsets.values[0] = float("nan")
    elif invalid == "label":
        target[0] = .5
    elif invalid == "death":
        target[0] = -1
    else:
        population[0] = 0
    with pytest.raises(ContractError):
        offsets.training_loss(split.training_ids(0), base, target, weights,
                              population=population if family == "poisson" else None)


@pytest.mark.parametrize("method", ["profile", "training"])
@pytest.mark.parametrize("empty_ids", [False, True])
def test_offset_empty_id_tensor_mismatch_rejected(method, empty_ids, split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family="identity",
                            exposure_assignment_level="tract")
    ids = () if empty_ids else split.training_ids(0)
    values = torch.zeros(4 if empty_ids else 0)
    with pytest.raises(ContractError):
        if method == "profile":
            offsets.update_identity(ids, values, values, values)
        else:
            offsets.training_loss(ids, values, values, values)


def test_empty_profile_and_empty_shape_mismatch_remain_invalid(split):
    offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family="identity",
                            exposure_assignment_level="tract")
    empty = torch.empty(0)
    with pytest.raises(ContractError):
        offsets.update_identity((), empty, empty, empty)
    with pytest.raises(ContractError):
        squared_loss(empty, empty.reshape(0, 1), empty)


@pytest.mark.parametrize("size", [0, 4])
def test_class_prior_remains_undefined_without_target_mass(size):
    with pytest.raises(ContractError, match="invalid origin pair weights"):
        weighted_class_prior(origin_pairs_with_weights(torch.zeros(size)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("target_dtype", [torch.bool, torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("entrypoint", ["direct", "endpoint", "profile"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_identity_binary_representations_values_and_gradients(dtype, target_dtype, entrypoint, reduction, split):
    base = torch.tensor([.25, .75, .5, .5], dtype=dtype, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1], dtype=target_dtype)
    weights = torch.tensor([1., 3., 0., 2.], dtype=dtype)
    expected_prediction = base
    if entrypoint == "direct":
        loss = squared_loss(base, target, weights, reduction=reduction)
    elif entrypoint == "endpoint":
        loss = endpoint_loss(base, target, weights, family="identity", reduction=reduction)
    else:
        offsets = CountyOffsets(split, 0, ("c", "c", "d", "d"), family="identity",
                                exposure_assignment_level="tract").to(dtype=dtype)
        offsets.update_identity(split.training_ids(0), target, base, weights)
        torch.testing.assert_close(offsets.values, base.new_tensor([.125, .5]), rtol=0, atol=0)
        loss = offsets.training_loss(split.training_ids(0), base, target, weights, reduction=reduction)
        expected_prediction = base + base.new_tensor([.125, .125, .5, .5])
    residual = expected_prediction - target.to(torch.promote_types(target_dtype, dtype))
    rows = weights * residual.square()
    expected = rows if reduction == "none" else rows.sum()
    expected_gradient = 2 * weights * residual
    if reduction == "mean":
        expected = expected / weights.sum()
        expected_gradient = expected_gradient / weights.sum()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(loss.sum(), base)
    torch.testing.assert_close(gradient, expected_gradient.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, 4])
def test_boolean_identity_zero_mass_keeps_autograd(size):
    base = torch.zeros(size, requires_grad=True)
    target = torch.zeros(size, dtype=torch.bool)
    loss = squared_loss(base, target, torch.zeros(size), reduction="mean")
    torch.testing.assert_close(loss, torch.tensor(0.), rtol=0, atol=0)
    gradient, = torch.autograd.grad(loss, base)
    torch.testing.assert_close(gradient, torch.zeros_like(base), rtol=0, atol=0)


@pytest.mark.parametrize("wide_operand", ["population", "deaths", "weights"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("entrypoint", ["direct", "endpoint"])
@pytest.mark.parametrize("predictor_dtype", [torch.float32, torch.float64])
def test_poisson_mixed_inputs_share_precision(wide_operand, reduction, entrypoint, predictor_dtype):
    f = torch.tensor([0., -.5, 1000.], dtype=predictor_dtype, requires_grad=True)
    deaths = torch.tensor([16777217, 3, 16777217], dtype=torch.int64)
    population = torch.tensor([16777217 if wide_operand == "deaths" else 1, 7, 100], dtype=torch.int64)
    weights = torch.tensor([1., 2., 0.], dtype=torch.float32)
    if wide_operand == "population":
        population = population.double()
    elif wide_operand == "deaths":
        deaths = deaths.double()
    else:
        weights = weights.double()
    if entrypoint == "direct":
        loss = normalized_poisson_loss(f, deaths, population, weights, reduction=reduction)
    else:
        loss = endpoint_loss(f, deaths, weights, family="poisson", population=population, reduction=reduction)

    # Evaluate the count likelihood only on contributing rows. FP64 arithmetic
    # retains the original integer, including the unit above 2**24.
    active = weights > 0
    rate = f.detach()[active].double().exp()
    d, n, w = deaths[active].double(), population[active].double(), weights[active].double()
    rows = torch.zeros(3, dtype=torch.float64)
    rows[active] = -w / n * torch.distributions.Poisson(n * rate).log_prob(d)
    expected = rows if reduction == "none" else rows.sum()
    expected_gradient = torch.zeros(3, dtype=torch.float64)
    expected_gradient[active] = w * (rate - d / n)
    if reduction == "mean":
        expected = expected / w.sum()
        expected_gradient /= w.sum()
    gradient, = torch.autograd.grad(loss.sum(), f)
    # Relative tolerances at magnitude 2**24 would conceal the lost count.
    torch.testing.assert_close(gradient[0], expected_gradient.to(predictor_dtype)[0], rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient.to(predictor_dtype))
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    assert gradient[2] == 0 and bool(torch.isfinite(gradient).all())
