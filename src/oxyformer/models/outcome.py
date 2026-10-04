"""Endpoint-linked treatment-query outcome nuisance."""
from torch import Tensor

from oxyformer.models.likelihoods import FAMILIES, inverse_link
from oxyformer.models.treatment_query import TreatmentQueryNetwork, add_group_offset
from oxyformer.provenance import require


class OutcomeTransformer(TreatmentQueryNetwork):
    def __init__(self, encoder, *, family="identity", **kwargs):
        require(family in FAMILIES, "unknown endpoint family")
        super().__init__(encoder, **kwargs)
        require(self.group_offsets is None or self.group_offsets.family == family,
                "offset likelihood must match outcome likelihood")
        self.family = family

    def linear_predictor(self, a_query, x_tokens, raw_x, context, group_offset, design=None) -> Tensor:
        """Identity mean, Bernoulli logit or log *rate*, before population exposure."""
        return add_group_offset(self._predict(a_query, x_tokens, raw_x, context, design), group_offset)

    def mean(self, a_query, x_tokens, raw_x, context, group_offset, design=None) -> Tensor:
        """Return [B,R] on endpoint scale; counts return rates, not N * rates."""
        return inverse_link(self.linear_predictor(a_query, x_tokens, raw_x, context, group_offset, design),
                            self.family)

    forward = mean
