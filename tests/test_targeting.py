"""Independent synthetic references for weighted, frozen-nuisance targeting."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from oxyformer.estimation.targeting import cv_tmle, fit_fluctuation, fluctuation_loss
from oxyformer.provenance import ContractError
from test_scores import make_fixture


def arrays(likelihood):
    y = np.array([0., 1., 0., 1., 1., 0.])
    mu = np.array([.2, .35, .6, .3, .45, .8])
    md = np.array([.3, .4, .55, .6, .7, .5])
    if likelihood == "identity":
        y, mu, md = 3 + 4*y, 2 + 3*mu, 4 + 3*md
    elif likelihood == "poisson":
        y, mu, md = y * .025, mu * .012, md * .014
    r = np.array([0., .4, 1.7, .8, 2.2, 1.1])
    rd = np.array([.7, 1.6, .2, 2.4, 1.3, .6])
    w = np.array([1., 3., 2., 7., 4., 1.])
    population = np.array([100., 800., 40., 1200., 200., 3200.]) if likelihood == "poisson" else None
    return mu, md, r, rd, y, w, population


@pytest.mark.parametrize("likelihood", ["identity", "logistic", "poisson"])
def test_proper_loss_derivative_finite_difference_and_independent_autograd(likelihood):
    mu, md, r, rd, y, w, population = arrays(likelihood)
    epsilon, step = .173, 1e-5
    t = torch.tensor(epsilon, dtype=torch.float64, requires_grad=True)
    tm, tr, ty, tw = [torch.tensor(v, dtype=torch.float64) for v in (mu, r, y, w)]
    if likelihood == "identity":
        prediction = tm + t * tr
        loss = (tw * .5 * (ty - prediction)**2).sum() / tw.sum()
    elif likelihood == "logistic":
        eta = torch.logit(tm) + t * tr
        prediction = torch.sigmoid(eta)
        loss = (tw * torch.nn.functional.binary_cross_entropy_with_logits(eta, ty, reduction="none")).sum()/tw.sum()
    else:
        # Independently differentiate the actual count likelihood with offset,
        # not the reduced rate expression in the implementation.
        tn = torch.tensor(population, dtype=torch.float64)
        counts = tn * ty
        eta = torch.log(tn) + torch.log(tm) + t * tr
        prediction = torch.exp(eta) / tn
        loss = (tw / tn * (torch.exp(eta) - counts*eta + torch.lgamma(counts+1))).sum()/tw.sum()
    loss.backward()
    residual = -float((tw * tr * (ty - prediction)).sum().detach() / tw.sum())
    def objective(e):
        return fluctuation_loss(e, mu, r, y, w, likelihood, population=population)
    numerical = (objective(epsilon + step) - objective(epsilon - step)) / (2*step)
    assert numerical == pytest.approx(residual, rel=2e-7, abs=2e-10)
    assert t.grad.item() == pytest.approx(residual, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("likelihood", ["identity", "logistic", "poisson"])
def test_shifted_ratio_and_residual_corrected_influence(likelihood):
    mu, md, r, rd, y, w, population = arrays(likelihood)
    result = fit_fluctuation(mu, md, r, rd, y, w, likelihood, population=population)
    # Reference optimizer uses explicit loss, not the production loss/moment.
    def means(base, ratio, e):
        if likelihood == "identity":
            return base + e*ratio
        if likelihood == "logistic":
            return expit(logit(base) + e*ratio)
        return base * np.exp(e*ratio)
    def objective(e):
        pred = means(mu, r, e)
        if likelihood == "identity":
            loss = .5 * (y-pred)**2
        elif likelihood == "logistic":
            loss = -y*np.log(pred) - (1-y)*np.log1p(-pred)
        else:
            n, d = population, population*y
            loss = (n*pred - d*np.log(n*pred))/n
        return np.average(loss, weights=w)
    reference = minimize_scalar(objective, bounds=(-2, 3), method="bounded", options={"xatol": 1e-12})
    assert reference.success
    assert result.epsilon == pytest.approx(reference.x, abs=1e-6)
    shifted = means(md, rd, result.epsilon)
    factual = means(mu, r, result.epsilon)
    assert_allclose(result.mu_d, shifted, rtol=1e-13, atol=1e-13)
    assert not np.allclose(shifted, means(md, r, result.epsilon))
    expected_value = np.average(shifted-y, weights=w)
    expected_score = shifted-y+r*(y-factual)
    assert result.value == pytest.approx(expected_value, abs=1e-13)
    assert_allclose(result.scores, expected_score, atol=1e-13)
    assert_allclose(result.influence, w/w.sum()*(expected_score-expected_value), atol=1e-13)
    assert abs(result.moment) < 1e-12
    assert result.score_gap == pytest.approx(result.moment, abs=1e-13)
    assert sum(result.influence) == pytest.approx(result.moment, abs=1e-13)
    assert abs(result.epsilon) > .01


def test_normalized_poisson_exposure_is_not_target_mass():
    mu, md, r, rd, y, w, population = arrays("poisson")
    first = fit_fluctuation(mu, md, r, rd, y, w, "poisson", population=population)
    # Changing N while holding D/N and target mass fixed changes only loss constants.
    second = fit_fluctuation(mu, md, r, rd, y, w, "poisson", population=population[::-1]*10)
    assert first == second
    wrong = fit_fluctuation(mu, md, r, rd, y, w*population, "poisson", population=population)
    assert first.epsilon != pytest.approx(wrong.epsilon)
    assert first.value != pytest.approx(wrong.value)
    with pytest.raises(ContractError, match="population exposure required"):
        fit_fluctuation(mu, md, r, rd, y, w, "poisson")


@pytest.mark.parametrize("likelihood,y,sign", [("logistic", 0., -1), ("logistic", 1., 1), ("poisson", 0., -1)])
def test_boundary_optima_and_zero_ratios(likelihood, y, sign):
    result = fit_fluctuation([.3,.5], [.2,.4], [1,0], [0,2], [y,1-y], [1,2], likelihood,
                             population=[100,10] if likelihood == "poisson" else None)
    assert result.epsilon == sign*np.inf
    assert result.status == "boundary"
    assert result.mu_a == (y, .5)
    assert result.mu_d == (.2, y)
    assert result.moment == 0


def test_flat_moment_is_flagged_without_fabricating_an_update():
    result = fit_fluctuation([1,2], [3,4], [0,0], [1,2], [9,8], [1,1], "identity")
    assert result.status == "flat"
    assert result.epsilon == 0
    assert result.mu_d == (3,4)


@pytest.mark.parametrize("likelihood", ["identity", "logistic", "poisson"])
def test_identity_policy_exact_zero_and_input_arrays_unchanged(likelihood):
    mu, md, r, rd, y, w, population = arrays(likelihood)
    copies = [x.copy() for x in (mu,md,r,rd,y,w)]
    result = fit_fluctuation(mu, md, r, rd, y, w, likelihood, population=population, identity=True)
    assert result.value == result.epsilon == result.moment == result.score_gap == 0
    assert result.scores == result.influence == (0.,)*len(mu)
    assert result.status == "identity"
    for old, new in zip(copies, (mu,md,r,rd,y,w)):
        assert_array_equal(old, new)


def run_fixture(fixture):
    p, data, split, n, weights = fixture
    return cv_tmle(n, data, weights, "identity", n.spec, split=split, policy=p)


def test_pooled_epsilon_seed_averaging_original_id_alignment_and_frozen_inputs():
    fixture = make_fixture()
    p, data, split, n, weights = fixture
    before = n.to_json()
    result = run_fixture(fixture)
    y_by_id = dict(zip(data.manifest.original_ids, data.column("y")))
    qs = np.array([weights[oid] for oid in split.original_ids], dtype=float)
    qs /= qs.sum()
    references, effects, influences = [], [], []
    for seed, diagnostic in result.by_seed:
        idx = [next(i for i, pair in enumerate(zip(n.original_ids,n.seed_ids)) if pair == (oid,seed))
               for oid in split.original_ids]
        mu, md, r, rd = [np.asarray(getattr(n,f))[idx] for f in ("mu_a","mu_d","r_a","r_d")]
        y = np.array([y_by_id[oid] for oid in split.original_ids])
        epsilon = np.sum(qs*r*(y-mu))/np.sum(qs*r*r)
        assert diagnostic.epsilon == pytest.approx(epsilon)
        effect = qs @ (md+epsilon*rd-y)
        h = md+epsilon*rd + r*(y-mu-epsilon*r)-y
        references.append(h)
        effects.append(effect)
        influences.append(qs*(h-effect))
    estimate = result.estimate
    assert len(result.by_seed) == len(split.seed_ids)
    assert estimate.original_ids == split.original_ids
    assert len(estimate.influence) == 4  # not four observations times two seeds
    assert_allclose(estimate.scores, np.mean(references,axis=0))
    assert_allclose(estimate.influence, np.mean(influences,axis=0))
    assert estimate.value == pytest.approx(np.mean(effects))
    assert estimate.standard_error is None
    assert estimate.from_json(estimate.to_json()) == estimate
    assert n.to_json() == before
    # Shuffle/interleave the OOF product without changing any original identity.
    order = [5,0,7,2,4,1,6,3]
    fields = ("original_ids","fold_ids","seed_ids","mu_a","mu_d","r_a","r_d","origin_weights")
    shuffled = replace(n, **{f: tuple(getattr(n,f)[i] for i in order) for f in fields})
    again = run_fixture((p,data,split,shuffled,weights)).estimate
    assert again.value == estimate.value
    assert again.scores == estimate.scores and again.influence == estimate.influence


def test_registered_identical_seed_does_not_multiply_observations_or_change_variance():
    from oxyformer.estimation.covariance import align_estimates, cluster_covariance
    p,data,split,n,w = make_fixture()
    fields = ("original_ids","fold_ids","mu_a","mu_d","r_a","r_d","origin_weights")
    repeated = replace(n, **{f: getattr(n,f)[:4]*2 for f in fields})
    multi = run_fixture((p,data,split,repeated,w)).estimate
    single_split = replace(split, seed_ids=(1103,))
    single = replace(n, **{f: getattr(n,f)[:4] for f in fields}, seed_ids=(1103,)*4,
                     lineage=replace(n.lineage, split_hash=single_split.content_hash))
    one = run_fixture((p,data,single_split,single,w)).estimate
    assert multi.value == one.value and multi.influence == one.influence
    groups = dict(zip(split.original_ids,("c1","c1","c2","c2")))
    def variance(e):
        return cluster_covariance(align_estimates({"y":e}),groups,interpretation="geographic_process").matrix
    assert variance(multi) == variance(one)


def test_artifact_identity_is_exact_and_boundary_checks_still_apply():
    fixture = make_fixture(delta=0)
    result = run_fixture(fixture).estimate
    assert result.value == 0
    assert result.scores == result.influence == (0.,)*4
    p,data,split,n,w = fixture
    with pytest.raises(ContractError,match="policy_id mismatch"):
        run_fixture((replace(p,delta_mmhg=2),data,split,n,w))
    with pytest.raises(ContractError,match="origin weight mismatch"):
        run_fixture((p,data,split,n,dict(w,o1=5)))
    fields = ("original_ids","fold_ids","seed_ids","mu_a","mu_d","r_a","r_d","origin_weights")
    partial = replace(n, **{f:getattr(n,f)[:4] for f in fields})
    with pytest.raises(ContractError,match="incomplete original ID/seed coverage"):
        run_fixture((p,data,split,partial,w))


@pytest.mark.parametrize("likelihood,field,value", [
    ("logistic","mu_a",[0]*6), ("logistic","outcomes",[.5]*6),
    ("poisson","population",[0]*6), ("poisson","mu_d",[-1]*6),
    ("identity","r_d",[-1]*6), ("identity","r_a",[np.nan]*6),
])
def test_invalid_likelihood_inputs_rejected(likelihood,field,value):
    mu,md,r,rd,y,w,pop = arrays(likelihood)
    args = dict(mu_a=mu,mu_d=md,r_a=r,r_d=rd,outcomes=y,weights=w,likelihood=likelihood,population=pop)
    args[field] = value
    with pytest.raises(ContractError):
        fit_fluctuation(**args)


def test_identity_bypasses_link_domain_and_population_requirements():
    binary = fit_fluctuation([0,1],[1,0],[1,2],[2,1],[0,1],[1,2],"logistic",identity=True)
    rate = fit_fluctuation([0,0],[1,2],[1,2],[2,1],[0,0],[1,2],"poisson",identity=True)
    for result in (binary,rate):
        assert result.value == 0 and result.influence == (0.,0.)
        assert result.status == "identity"
