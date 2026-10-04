"""Signed policy-contrast Riesz nuisance; no positivity transform."""
from torch import Tensor

from oxyformer.models.likelihoods import _aligned, weighted_reduce
from oxyformer.models.treatment_query import TreatmentQueryNetwork


def riesz_loss(v_a: Tensor, v_d: Tensor, weights: Tensor) -> Tensor:
    """E_T[v(A,X)^2 - 2 * (v(d(A,X),X) - v(A,X))]."""
    _aligned(v_a, v_d, weights)
    v_a = v_a.masked_fill(weights == 0, 0)
    v_d = v_d.masked_fill(weights == 0, 0)
    return weighted_reduce(v_a.square() - 2 * (v_d - v_a), weights, "mean")


class RieszTransformer(TreatmentQueryNetwork):
    def forward(self, a_query, x_tokens, raw_x, context) -> Tensor:
        return self._predict(a_query, x_tokens, raw_x, context)
