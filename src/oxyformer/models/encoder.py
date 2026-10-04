"""Native PyTorch set encoder. No feature-order or geographic encoding."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from oxyformer.models.tokens import FeatureBatch, FeatureSpec, FeatureTokenizer
from oxyformer.provenance import require


class FeatureEncoder(nn.Module):
    """Three pre-LN blocks followed by final LN; return (feature states, CLS).

    Jointly permuting every FeatureBatch column and its feature identifier
    permutes the feature states and preserves CLS in eval mode (up to floating
    point roundoff). Training dropout is stochastic and permutation-invariant
    in distribution. Call eval() for deterministic predictions.
    """

    def __init__(self, features: tuple[FeatureSpec, ...], *, dropout: float = 0.1):
        super().__init__()
        require(0 <= dropout < 1, "dropout must be in [0,1)")
        self.width = 64
        self.tokenizer = FeatureTokenizer(features, width=self.width)
        self.cls = nn.Parameter(torch.empty(1, 1, self.width))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(self.width)
        require(sum(p.numel() for p in self.parameters()) <= 1_000_000,
                "feature encoder exceeds nuisance parameter cap")

    def forward(self, feature_batch: FeatureBatch) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(feature_batch)
        states = torch.cat((self.cls.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        padding = torch.cat((torch.zeros((tokens.shape[0], 1), dtype=torch.bool,
                                         device=tokens.device), feature_batch.padding), dim=1)
        for block in self.blocks:
            states = block(states, src_key_padding_mask=padding)
        states = self.norm(states)
        features = states[:, 1:].masked_fill(feature_batch.padding.unsqueeze(-1), 0)
        return features, states[:, 0]
