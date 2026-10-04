"""Fixed, independently named architectural comparisons from plan section 10."""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn

from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.origin import OriginTransformer
from oxyformer.models.outcome import OutcomeTransformer
from oxyformer.models.riesz import RieszTransformer
from oxyformer.models.tokens import FeatureBatch
from oxyformer.models.treatment_query import EncodedFeatures
from oxyformer.provenance import require


@dataclass(frozen=True)
class Variant:
    label: str
    outcome: str
    correction: str
    ssl: bool = True
    pma: bool = True
    status: str = "production_comparison"


VARIANTS = MappingProxyType({
    "A0": Variant("full_oxyformer_v2", "query", "origin"),
    "A1": Variant("no_ssl", "query", "origin", ssl=False),
    "A2": Variant("no_county_pma", "query", "origin", pma=False),
    "A3": Variant("early_treatment_fusion", "early", "early"),
    "A4": Variant("varying_coefficient", "varying", "varying"),
    "A5": Variant("signed_riesz", "query", "signed_riesz"),
    "A6": Variant("tabiclv2_outcome_only", "tabicl", "origin"),
    "A7": Variant("tabiclv2_full_nuisance", "tabicl", "tabicl"),
    "F0": Variant("tabpfn_v2_outcome_only", "tabpfn", "origin"),
    "F1": Variant("tabpfn_v2_full_nuisance", "tabpfn", "tabpfn"),
    "A8": Variant("endpoint_sharing", "deferred", "deferred", status="months_2_3_only"),
    "B0": Variant("conventional_calibration", "ridge_gam_boosting", "plr", status="calibration_only"),
    "D0": Variant("location_red_team", "diagnostic", "diagnostic", status="diagnostic_only"),
})


def _validate(network, a_query, raw_x, context, design):
    basis = network.basis(a_query, design)
    batch = a_query.shape[0]
    require(raw_x.shape == (batch, network.raw_x_dim), "complete approved raw-X bypass required")
    require(context.shape == (batch, 4, 64), "county context must be [B,4,64]")
    if network.county_context is None:
        require(not context.requires_grad and bool((context == 0).all()),
                "nonzero or trainable tokens require an owned county context")
    return basis


class _AlternativeHead:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These alternatives replace treatment-query attention entirely.
        for name in ("query_projection", "query_norm", "memory_norm", "attention",
                     "ff_norm", "ff", "output_norm", "dropout"):
            delattr(self, name)


class _EarlyFusion(_AlternativeHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.treatment_projection = nn.Linear(7, 64)
        self.check_parameter_cap()

    def _predict(self, a_query, x_tokens, raw_x, context, design=None):
        basis = _validate(self, a_query, raw_x, context, design)
        require(type(x_tokens) is FeatureBatch,
                "early fusion requires FeatureBatch; treatment enters before encoder blocks")
        tokens = self.encoder.tokenizer(x_tokens)
        batch, queries, _ = basis.shape
        require(tokens.shape[0] == batch, "feature batch alignment mismatch")
        states = torch.cat((self.encoder.cls.expand(batch, -1, -1), tokens, context), dim=1)
        width = states.shape[1]
        states = (states[:, None] + self.treatment_projection(basis)[:, :, None]).reshape(batch * queries, width, 64)
        padding = torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=tokens.device),
                             x_tokens.padding, torch.zeros((batch, 4), dtype=torch.bool, device=tokens.device)), dim=1)
        padding = padding[:, None].expand(-1, queries, -1).reshape(batch * queries, width)
        for block in self.encoder.blocks:
            states = block(states, src_key_padding_mask=padding)
        states = self.encoder.norm(states).reshape(batch, queries, width, 64)
        inputs = torch.cat((states[:, :, 0], states[:, :, -4:].mean(2),
                            raw_x[:, None].expand(-1, queries, -1), basis), dim=-1)
        return self.readout(inputs).squeeze(-1)


class _VaryingCoefficient(_AlternativeHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readout = nn.Sequential(nn.Linear(128 + self.raw_x_dim, 64), nn.GELU(),
                                     nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8))
        self.check_parameter_cap()

    def _predict(self, a_query, x_tokens, raw_x, context, design=None):
        basis = _validate(self, a_query, raw_x, context, design)
        if isinstance(x_tokens, FeatureBatch):
            x_tokens = self.encode(x_tokens)
        if isinstance(x_tokens, EncodedFeatures):
            x_tokens = x_tokens.states
        require(x_tokens.ndim == 3 and x_tokens.shape[0] == len(raw_x) and
                x_tokens.shape[1] >= 1 and x_tokens.shape[2] == 64, "X states must be CLS-first")
        coefficients = self.readout(torch.cat((x_tokens[:, 0], context.mean(1), raw_x), dim=-1))
        return (coefficients[:, None] * torch.cat((torch.ones_like(basis[..., :1]), basis), dim=-1)).sum(-1)


class EarlyFusionOutcome(_EarlyFusion, OutcomeTransformer):
    pass


class EarlyFusionOrigin(_EarlyFusion, OriginTransformer):
    pass


class VaryingCoefficientOutcome(_VaryingCoefficient, OutcomeTransformer):
    pass


class VaryingCoefficientOrigin(_VaryingCoefficient, OriginTransformer):
    pass


@dataclass(frozen=True)
class NuisanceVariant:
    variant_id: str
    label: str
    outcome: object
    correction: object


def build_variant(variant_id, encoder, *, treatment_design, raw_x_dim, family="identity",
                  county_context=None, outcome_offsets=None, origin_offsets=None,
                  foundation_outcome=None, foundation_origin=None, dropout=.1):
    """Build fresh, independent heads; A1 always discards pretrained weights.

    Foundation arguments are configured adapters (fit by the fold training unit).
    A8/B0/D0 are registered, separately labeled, and cannot be production builds.
    The caller owns optimizer/SSL scheduling and supplies the registered seed.
    """
    require(variant_id in VARIANTS, "unknown registered variant")
    variant = VARIANTS[variant_id]
    require(variant.status == "production_comparison", "variant is not a production nuisance configuration")
    if not variant.ssl:
        encoder = FeatureEncoder(encoder.tokenizer.features, dropout=dropout)
        if county_context is not None:
            county_context = deepcopy(county_context)
            county_context.encoder = encoder
            # PMA initialization must also be fresh, not a learned prior state.
            with torch.no_grad():
                for module in county_context.modules():
                    if isinstance(module, nn.MultiheadAttention):
                        module._reset_parameters()
                    if module is not encoder and hasattr(module, "reset_parameters"):
                        module.reset_parameters()
            nn.init.normal_(county_context.seeds, std=.02)
            county_context.clear_cache()
    if not variant.pma:
        county_context = None
    common = dict(treatment_design=treatment_design, raw_x_dim=raw_x_dim,
                  county_context=county_context, dropout=dropout)
    outcome_classes = {"query": OutcomeTransformer, "early": EarlyFusionOutcome,
                       "varying": VaryingCoefficientOutcome}
    correction_classes = {"origin": OriginTransformer, "early": EarlyFusionOrigin,
                          "varying": VaryingCoefficientOrigin, "signed_riesz": RieszTransformer}
    if variant.outcome in outcome_classes:
        outcome = outcome_classes[variant.outcome](encoder, family=family, group_offsets=outcome_offsets, **common)
    else:
        from oxyformer.models.tabicl_comparator import TabICLComparator
        require(isinstance(foundation_outcome, TabICLComparator) and
                foundation_outcome.package == variant.outcome and foundation_outcome.task == "outcome" and
                foundation_outcome.family == family, "compatible foundation outcome adapter required")
        require(outcome_offsets is None, "foundation outcome does not support county offsets")
        outcome = foundation_outcome
    if variant.correction in correction_classes:
        correction = correction_classes[variant.correction](encoder, group_offsets=origin_offsets, **common)
    else:
        from oxyformer.models.tabicl_comparator import TabICLComparator
        require(isinstance(foundation_origin, TabICLComparator) and
                foundation_origin.package == variant.correction and foundation_origin.task == "origin",
                "compatible foundation origin adapter required")
        require(origin_offsets is None, "foundation origin does not support county offsets")
        correction = foundation_origin
    return NuisanceVariant(variant_id, variant.label, outcome, correction)
