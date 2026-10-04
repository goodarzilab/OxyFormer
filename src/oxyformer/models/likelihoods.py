"""Endpoint losses and training-owned scalar comparison-stratum intercepts.

Weights are target masses. Population is a distinct likelihood exposure. Losses
sum by default; ``mean`` divides by total target mass, never by population.
All-zero-mass and zero-row minibatches contribute zero loss and gradient; ``none``
preserves the input shape. Dataset-level mass requirements belong to the caller.

Zero-weight operands cannot affect objectives, active gradients or profiled
offsets. Validate original inputs before numerical exclusion, then protect
residuals and offset addition before arithmetic. Finite-input, binary-label,
nonnegative-count, positive-population and ownership checks include excluded
rows. This contract covers the weighted endpoint, origin, Riesz and offset
paths in FP32/FP64 when active-only arithmetic is representable. Unweighted
model forwards/inverse links are separate APIs; undefined class priors remain
errors. Profiling still requires nonempty, complete training-ID coverage.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from oxyformer.contracts import SplitManifest
from oxyformer.provenance import require


FAMILIES = ("identity", "bernoulli", "poisson")


def _aligned(prediction: Tensor, target: Tensor, weights: Tensor) -> None:
    require(prediction.shape == target.shape == weights.shape,
            "prediction, target and target weights must have identical shapes")
    require(all(bool(torch.isfinite(x).all()) for x in (prediction, target, weights)),
            "nonfinite likelihood inputs")
    require(bool((weights >= 0).all()), "invalid target weights")


def weighted_reduce(values: Tensor, weights: Tensor, reduction: str = "sum") -> Tensor:
    """Reduce finite per-row values; callers protect upstream arithmetic."""
    require(values.shape == weights.shape, "weight alignment mismatch")
    require(reduction in ("none", "sum", "mean"), "unknown loss reduction")
    weighted = values * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    mass = weights.sum()
    # Preserve every positive mass, including masses below one. Avoid 0/0 for
    # an empty contribution while retaining the predictor autograd connection.
    return weighted.sum() / torch.where(mass > 0, mass, torch.ones_like(mass))


def squared_loss(mean: Tensor, target: Tensor, weights: Tensor, *, reduction="sum") -> Tensor:
    _aligned(mean, target, weights)
    target = target.to(dtype=torch.promote_types(target.dtype, mean.dtype))
    excluded = weights == 0
    residual = mean.masked_fill(excluded, 0) - target.masked_fill(excluded, 0)
    return weighted_reduce(residual.square(), weights, reduction)


def bernoulli_loss(logits: Tensor, target: Tensor, weights: Tensor, *, reduction="sum") -> Tensor:
    _aligned(logits, target, weights)
    require(bool(((target == 0) | (target == 1)).all()), "Bernoulli targets must be 0 or 1")
    target = target.to(dtype=logits.dtype)
    return weighted_reduce(F.binary_cross_entropy_with_logits(logits, target, reduction="none"),
                           weights, reduction)


def normalized_poisson_loss(log_rate: Tensor, deaths: Tensor, population: Tensor,
                            weights: Tensor, *, reduction="sum") -> Tensor:
    """sum_i w_i/N_i * [-log Poisson(D_i; N_i exp(log_rate_i))].

    Includes the exact log-factorial constant. The derivative w.r.t. log rate is
    w_i * (lambda_i - D_i/N_i), including when target mass differs from N_i.
    """
    _aligned(log_rate, deaths, weights)
    require(population.shape == deaths.shape and bool(torch.isfinite(population).all())
            and bool((population > 0).all()), "population must be aligned, finite and positive")
    require(bool((deaths >= 0).all()), "negative death counts")
    # Choose precision jointly before converting integers or branching the
    # predictor graph. A wider population/weight must protect counts and the
    # accumulated log-rate derivative, not only the final multiplication.
    compute_dtype = log_rate.dtype
    for value in (deaths, population, weights):
        compute_dtype = torch.promote_types(compute_dtype, value.dtype)
    log_rate = log_rate.to(dtype=compute_dtype)
    deaths = deaths.to(dtype=compute_dtype)
    population = population.to(dtype=compute_dtype)
    # A zero-mass row must never reach exp/lgamma with extreme values: masking
    # the resulting infinity afterward would still leave NaN backward products.
    excluded = weights == 0
    log_rate = log_rate.masked_fill(excluded, 0)
    deaths = deaths.masked_fill(excluded, 0)
    population = population.masked_fill(excluded, 1)
    log_count_mean = population.log() + log_rate
    # Algebraically NLL(D, N*exp(f))/N, without first forming N*exp(f).
    normalized_nll = (log_rate.exp() - (deaths / population) * log_count_mean
                      + torch.lgamma(deaths + 1) / population)
    return weighted_reduce(normalized_nll, weights, reduction)


def inverse_link(linear_predictor: Tensor, family: str) -> Tensor:
    require(family in FAMILIES, "unknown endpoint family")
    if family == "identity":
        return linear_predictor
    return linear_predictor.sigmoid() if family == "bernoulli" else linear_predictor.exp()


def endpoint_loss(linear_predictor: Tensor, target: Tensor, weights: Tensor, *,
                  family: str, population: Tensor | None = None, reduction="sum") -> Tensor:
    require(family in FAMILIES, "unknown endpoint family")
    if family == "poisson":
        require(population is not None, "count likelihood requires a separate population exposure")
        return normalized_poisson_loss(linear_predictor, target, population, weights, reduction=reduction)
    require(population is None, "population exposure is only used by the count likelihood")
    loss = squared_loss if family == "identity" else bernoulli_loss
    return loss(linear_predictor, target, weights, reduction=reduction)


class CountyOffsets(nn.Module):
    """Scalar county/region/state intercepts, bound to one split's training IDs.

    ``training_counties`` follows split.training_ids(fold). Identity intercepts
    are reprofiled after changes to f_theta using ``update_identity``. Other
    families have trainable link-scale intercepts: optimize ``training_loss``
    jointly with the network. Labels enter only these training-ID-checked APIs.
    Profiling requires nonempty complete training-ID coverage; repeated rows contribute
    their supplied weights, as in training_loss. Callers own multiplicity weights.
    Empty training_loss subsets are allowed with aligned empty vectors.
    Prediction accepts routes alone; unseen strata receive zero. The adapter
    must declare the actual exposure assignment level, so offsets cannot be
    fitted at that exact level. No geographic embedding is constructed.
    """

    def __init__(self, split: SplitManifest, fold: int, training_counties: tuple[str, ...], *,
                 family: str, exposure_assignment_level: str, stratum: str = "county"):
        super().__init__()
        require(family in FAMILIES, "unknown endpoint family")
        require(stratum in ("county", "region", "state"), "unsupported comparison stratum")
        require(bool(exposure_assignment_level) and stratum != exposure_assignment_level,
                "exact exposure-unit offsets would absorb treatment variation")
        self.family = family
        self.split_hash = split.content_hash
        self.training_ids = split.training_ids(fold)
        routes = tuple(training_counties)
        require(len(routes) == len(self.training_ids) and all(isinstance(c, str) and c for c in routes),
                "training county alignment mismatch")
        self._routes = dict(zip(self.training_ids, routes))
        self.counties = tuple(sorted(set(routes)))
        self._indices = {county: i for i, county in enumerate(self.counties)}
        self.values = nn.Parameter(torch.zeros(len(self.counties)), requires_grad=family != "identity")

    def get_extra_state(self):
        return (self.family, self.split_hash, self.training_ids, tuple(self._routes.items()))

    def set_extra_state(self, state):
        require(state == self.get_extra_state(), "offset checkpoint training ownership mismatch")

    def forward(self, counties: tuple[str, ...]) -> Tensor:
        indices = torch.tensor([self._indices.get(c, len(self.counties)) for c in counties],
                               device=self.values.device, dtype=torch.long)
        return torch.cat((self.values, self.values.new_zeros(1)))[indices]

    def _training_routes(self, original_ids: tuple[str, ...], *, complete: bool = False):
        ids = tuple(original_ids)
        require(set(ids) <= set(self.training_ids),
                "offset labels must belong only to permitted training IDs")
        if complete:
            require(bool(ids) and set(ids) == set(self.training_ids), "profiling requires all training IDs")
        return tuple(self._routes[oid] for oid in ids)

    @torch.no_grad()
    def update_identity(self, original_ids: tuple[str, ...], target: Tensor,
                        base_mean: Tensor, weights: Tensor) -> None:
        require(self.family == "identity", "residual profiling requires identity loss")
        routes = self._training_routes(original_ids, complete=True)
        _aligned(base_mean, target, weights)
        require(target.shape == (len(routes),), "training label alignment mismatch")
        target = target.to(dtype=torch.promote_types(target.dtype, base_mean.dtype))
        excluded = weights == 0
        residual = target.masked_fill(excluded, 0) - base_mean.masked_fill(excluded, 0)
        for county, index in self._indices.items():
            mask = torch.tensor([c == county for c in routes], device=target.device)
            mass = weights[mask].sum()
            # A stratum with zero target mass contributes no objective: use zero.
            self.values[index] = (weights[mask] * residual[mask]).sum() / mass if mass > 0 else 0

    def training_loss(self, original_ids: tuple[str, ...], base_predictor: Tensor,
                      target: Tensor, weights: Tensor, *, population=None, reduction="sum") -> Tensor:
        routes = self._training_routes(original_ids)
        require(base_predictor.shape == (len(routes),), "training prediction alignment mismatch")
        _aligned(base_predictor, target, weights)
        offset = self(routes)
        require(bool(torch.isfinite(offset).all()), "nonfinite county offsets")
        excluded = weights == 0
        predictor = base_predictor.masked_fill(excluded, 0) + offset.masked_fill(excluded, 0)
        return endpoint_loss(predictor, target, weights,
                             family=self.family, population=population, reduction=reduction)
