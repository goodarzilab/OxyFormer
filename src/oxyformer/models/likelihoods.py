"""Endpoint losses and training-owned scalar comparison-stratum intercepts.

Weights are target masses. Population is a distinct likelihood exposure. Losses
sum by default; ``mean`` divides by total target mass, never by population.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from oxyformer.contracts import SplitManifest
from oxyformer.provenance import require


FAMILIES = ("identity", "bernoulli", "poisson")


def _aligned(prediction: Tensor, target: Tensor, weights: Tensor) -> None:
    require(prediction.shape == target.shape == weights.shape and prediction.numel() > 0,
            "prediction, target and target weights must have identical nonempty shapes")
    require(all(bool(torch.isfinite(x).all()) for x in (prediction, target, weights)),
            "nonfinite likelihood inputs")
    require(bool((weights >= 0).all()) and bool(weights.sum() > 0), "invalid target weights")


def weighted_reduce(values: Tensor, weights: Tensor, reduction: str = "sum") -> Tensor:
    require(values.shape == weights.shape, "weight alignment mismatch")
    require(reduction in ("none", "sum", "mean"), "unknown loss reduction")
    weighted = values * weights
    if reduction == "none":
        return weighted
    return weighted.sum() if reduction == "sum" else weighted.sum() / weights.sum()


def squared_loss(mean: Tensor, target: Tensor, weights: Tensor, *, reduction="sum") -> Tensor:
    _aligned(mean, target, weights)
    return weighted_reduce((mean - target).square(), weights, reduction)


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
    # Integer unary log/lgamma otherwise default to float32 even for a FP64
    # predictor. Promote before these operations; preserve wider float inputs.
    deaths = deaths.to(dtype=torch.promote_types(deaths.dtype, log_rate.dtype))
    population = population.to(dtype=torch.promote_types(population.dtype, log_rate.dtype))
    log_count_mean = population.log() + log_rate
    nll = log_count_mean.exp() - deaths * log_count_mean + torch.lgamma(deaths + 1)
    return weighted_reduce(nll / population, weights, reduction)


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
    Profiling requires complete training-ID coverage; repeated rows contribute
    their supplied weights, as in training_loss. Callers own multiplicity weights.
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
        require(bool(ids) and set(ids) <= set(self.training_ids),
                "offset labels must belong only to permitted training IDs")
        if complete:
            require(set(ids) == set(self.training_ids), "profiling requires all training IDs")
        return tuple(self._routes[oid] for oid in ids)

    @torch.no_grad()
    def update_identity(self, original_ids: tuple[str, ...], target: Tensor,
                        base_mean: Tensor, weights: Tensor) -> None:
        require(self.family == "identity", "residual profiling requires identity loss")
        routes = self._training_routes(original_ids, complete=True)
        _aligned(base_mean, target, weights)
        require(target.shape == (len(routes),), "training label alignment mismatch")
        residual = target - base_mean
        for county, index in self._indices.items():
            mask = torch.tensor([c == county for c in routes], device=target.device)
            mass = weights[mask].sum()
            # A stratum with zero target mass contributes no objective: use zero.
            self.values[index] = (weights[mask] * residual[mask]).sum() / mass if mass > 0 else 0

    def training_loss(self, original_ids: tuple[str, ...], base_predictor: Tensor,
                      target: Tensor, weights: Tensor, *, population=None, reduction="sum") -> Tensor:
        routes = self._training_routes(original_ids)
        require(base_predictor.shape == (len(routes),), "training prediction alignment mismatch")
        return endpoint_loss(base_predictor + self(routes), target, weights,
                             family=self.family, population=population, reduction=reduction)
