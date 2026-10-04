"""One treatment-to-X attention block; each dose is a separate query row."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from oxyformer.models.county_context import CountyContext
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.likelihoods import CountyOffsets
from oxyformer.models.tokens import FeatureBatch
from oxyformer.provenance import Immutable, check_hash, require


@dataclass(frozen=True, slots=True, kw_only=True)
class TreatmentDesign(Immutable):
    """Already frozen design: mmHg centering, scale and six cubic-spline knots.

    The six spline values are truncated cubic powers ((a-k_j)/scale)_+^3.
    Together with the standardized scalar they provide a continuous treatment
    basis without binning, clipping, refitting knots or enforcing a dose sign.
    ``design_hash`` identifies the frozen design artifact that supplied them.
    """
    center: float
    scale: float
    knots: tuple[float, ...]
    design_hash: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        check_hash(self.design_hash, "treatment design hash")
        require(self.scale > 0, "treatment scale must be positive")
        require(len(self.knots) == 6 and all(a < b for a, b in zip(self.knots, self.knots[1:])),
                "six strictly increasing frozen treatment knots required")


class TreatmentBasis(nn.Module):
    def __init__(self, design: TreatmentDesign):
        super().__init__()
        require(type(design) is TreatmentDesign, "frozen TreatmentDesign required")
        self.design = design

    def get_extra_state(self):
        return self.design.to_json()

    def set_extra_state(self, state):
        self.design = TreatmentDesign.from_json(state)

    def forward(self, a_query: Tensor, design: TreatmentDesign | None = None) -> Tensor:
        require(design is None or design == self.design, "cannot override frozen treatment design")
        require(a_query.ndim == 3 and a_query.shape[-1] == 1 and a_query.is_floating_point(),
                "treatment queries must be [B,R,1] floats")
        require(bool(torch.isfinite(a_query).all()), "nonfinite treatment query")
        standardized = (a_query - self.design.center) / self.design.scale
        spline = ((a_query - a_query.new_tensor(self.design.knots)) / self.design.scale).clamp_min(0).pow(3)
        return torch.cat((standardized, spline), dim=-1)


class TreatmentQueryNetwork(nn.Module):
    """Independent nuisance backbone and mandatory complete raw-X readout.

    Copies supplied initialization; parameters are never shared across nuisance
    instances. With context, pass the same encoder owned by CountyContext;
    the copy preserves that *within-nuisance* alias. Owned offsets, if supplied,
    are copied and included in the full-network parameter cap.

    x_tokens is [B,P+1,64], with CLS first, or a FeatureBatch to encode with this
    network's own encoder. Use ``encode`` and ``county_context`` of this instance
    when precomputing tensors; do not reuse another nuisance's learned states.
    raw_x is [B,raw_x_dim]: all approved covariates in the adapter's frozen
    numerical/missing/category representation, with no learned compression.
    Permission checking and raw-X construction remain the covariate adapter's
    responsibility. context is [B,4,64]. Predictions are deterministic in eval
    mode; training dropout is independent per query, with no dose aggregation.
    """

    def __init__(self, encoder: FeatureEncoder, *, treatment_design: TreatmentDesign,
                 raw_x_dim: int, county_context: CountyContext | None = None,
                 group_offsets: CountyOffsets | None = None, dropout: float = 0.1):
        super().__init__()
        require(type(encoder) is FeatureEncoder, "expected FeatureEncoder")
        require(type(raw_x_dim) is int and raw_x_dim > 0, "complete raw-X width required")
        require(0 <= dropout < 1, "dropout must be in [0,1)")
        require(county_context is None or county_context.encoder is encoder,
                "context must use this nuisance's encoder initialization")
        self.encoder, self.county_context, self.group_offsets = deepcopy((encoder, county_context, group_offsets))
        if self.county_context is not None:
            self.county_context.clear_cache()
        self.raw_x_dim = raw_x_dim
        self.basis = TreatmentBasis(treatment_design)
        self.query_projection = nn.Linear(7, 64)
        self.query_norm = nn.LayerNorm(64)
        self.memory_norm = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(64)
        self.readout = nn.Sequential(nn.Linear(128 + raw_x_dim + 7, 64), nn.GELU(),
                                     nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.check_parameter_cap()

    def check_parameter_cap(self) -> int:
        count = sum(p.numel() for p in self.parameters())
        require(count <= 1_000_000, "complete nuisance network exceeds one-million-parameter cap")
        return count

    def encode(self, batch: FeatureBatch) -> Tensor:
        features, cls = self.encoder(batch)
        return torch.cat((cls.unsqueeze(1), features), dim=1)

    def _predict(self, a_query: Tensor, x_tokens: Tensor | FeatureBatch, raw_x: Tensor,
                 context: Tensor, design: TreatmentDesign | None = None) -> Tensor:
        padding = None
        if isinstance(x_tokens, FeatureBatch):
            padding = x_tokens.padding
            x_tokens = self.encode(x_tokens)
        basis = self.basis(a_query, design)
        batch, queries, _ = basis.shape
        require(x_tokens.ndim == 3 and x_tokens.shape[0] == batch and x_tokens.shape[1] >= 1
                and x_tokens.shape[2] == 64, "X states must be [B,P+1,64] with CLS first")
        require(raw_x.shape == (batch, self.raw_x_dim), "complete approved raw-X bypass required")
        require(context.shape == (batch, 4, 64), "county context must be [B,4,64]")
        memory = self.memory_norm(torch.cat((x_tokens, context), dim=1))
        memory_padding = None
        if padding is not None:
            memory_padding = torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=padding.device),
                                        padding, torch.zeros((batch, 4), dtype=torch.bool,
                                                             device=padding.device)), dim=1)
        query = self.query_projection(basis)
        attended, _ = self.attention(self.query_norm(query), memory, memory,
                                     key_padding_mask=memory_padding, need_weights=False)
        query = query + self.dropout(attended)
        query = query + self.dropout(self.ff(self.ff_norm(query)))
        readout = torch.cat((self.output_norm(query), x_tokens[:, :1].expand(-1, queries, -1),
                             raw_x.unsqueeze(1).expand(-1, queries, -1), basis), dim=-1)
        return self.readout(readout).squeeze(-1)


def add_group_offset(predictor: Tensor, group_offset: Tensor) -> Tensor:
    """One scalar per observation, shared by all queried doses."""
    require(group_offset.shape in ((predictor.shape[0],), (predictor.shape[0], 1)),
            "group offset must be one scalar per observation")
    return predictor + group_offset.reshape(-1, 1)
