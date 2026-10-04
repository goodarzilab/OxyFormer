"""Independent origin classifier and explicit calibrated prior correction.

C=1 is the shifted copy. Calibration fitting belongs to inner out-of-fold
outer-training data; these conversion functions only consume calibrated values.
"""
import torch
from torch import Tensor

from oxyformer.design.policies import PolicyPairs
from oxyformer.models.likelihoods import bernoulli_loss
from oxyformer.models.treatment_query import TreatmentQueryNetwork, add_group_offset
from oxyformer.provenance import require


def probability_to_ratio(calibrated_probability: Tensor, *, class_prior: float) -> Tensor:
    require(0 < class_prior < 1, "class prior must lie strictly inside (0,1)")
    require(bool(torch.isfinite(calibrated_probability).all())
            and bool(((calibrated_probability >= 0) & (calibrated_probability < 1)).all()),
            "calibrated probabilities must lie in [0,1); no implicit clipping")
    return ((1 - class_prior) / class_prior
            * calibrated_probability / (1 - calibrated_probability))


def calibrated_logit_to_ratio(calibrated_logits: Tensor, *, class_prior: float) -> Tensor:
    """Equivalent odds conversion without sigmoid rounding at large logits."""
    require(0 < class_prior < 1, "class prior must lie strictly inside (0,1)")
    return calibrated_logits.exp() * ((1 - class_prior) / class_prior)


def paired_origin_loss(logits: Tensor, pairs: PolicyPairs, *, reduction="mean") -> Tensor:
    """Consume authoritative paired_records weights; never reweight destinations.

    Return weighted BCE. Pair construction and split ownership are supplied by
    design.policies and the training unit, respectively.
    """
    require(type(pairs) is PolicyPairs, "expected PolicyPairs")
    labels = logits.new_tensor(pairs.transformed)
    weights = logits.new_tensor(pairs.origin_weights)
    return bernoulli_loss(logits, labels, weights, reduction=reduction)


def weighted_class_prior(pairs: PolicyPairs) -> float:
    weights = torch.tensor(pairs.origin_weights, dtype=torch.float64)
    labels = torch.tensor(pairs.transformed, dtype=torch.float64)
    require(weights.shape == labels.shape and bool((weights >= 0).all()) and weights.sum() > 0,
            "invalid origin pair weights")
    return float((weights * labels).sum() / weights.sum())


class OriginTransformer(TreatmentQueryNetwork):
    def __init__(self, encoder, **kwargs):
        super().__init__(encoder, **kwargs)
        require(self.group_offsets is None or self.group_offsets.family == "bernoulli",
                "origin offsets require the Bernoulli link")

    def logits(self, a_query, x_tokens, raw_x, context, group_offset) -> Tensor:
        return add_group_offset(self._predict(a_query, x_tokens, raw_x, context), group_offset)

    forward = logits
