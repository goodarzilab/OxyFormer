"""Feature identities and fold-fitted preprocessing, never column positions.

FeatureSpec contains already-fitted training-only preprocessing; fitting it is
owned by the training unit. FeatureBatch carries standardized numerical values
and categorical indices (0..K-1 observed, K unknown). Missingness, artificial
masking and padding are separate boolean tensors. A masked value is never read.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import torch
from torch import Tensor, nn

from oxyformer.contracts import Cell, CovariateView
from oxyformer.provenance import Immutable, canonical_json, nonempty, require, unique


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureSpec(Immutable):
    name: str
    kind: Literal["numeric", "categorical"]
    mean: float = 0.0
    scale: float = 1.0
    categories: tuple[Cell, ...] = ()

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.name, "feature name")
        require(self.scale > 0, "feature scale must be positive")
        require(None not in self.categories, "missing is not an observed category")
        unique(tuple(canonical_json(x) for x in self.categories), "categories")
        if self.kind == "numeric":
            require(not self.categories, "numeric feature cannot have categories")
        else:
            require(self.mean == 0 and self.scale == 1,
                    "categorical feature cannot have numeric preprocessing")


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    feature_ids: Tensor       # [F], stable indices into the tokenizer's schema
    numeric_values: Tensor    # [B,F], standardized; ignored for categorical cells
    categorical_values: Tensor  # [B,F], ignored for numeric cells
    missing: Tensor           # true missingness, independent of artificial masking
    masked: Tensor            # artificial SSL masking; values are suppressed
    padding: Tensor           # excluded from attention; feature ID -1 is permitted

    def to(self, device: torch.device | str) -> FeatureBatch:
        return FeatureBatch(*(x.to(device) for x in (
            self.feature_ids, self.numeric_values, self.categorical_values,
            self.missing, self.masked, self.padding)))


class FeatureTokenizer(nn.Module):
    """Feature-specific numeric slopes, category tables and missing/mask states."""

    def __init__(self, features: tuple[FeatureSpec, ...], width: int = 64):
        super().__init__()
        self.features = tuple(features)
        require(bool(self.features), "feature schema must be nonempty")
        require(all(type(f) is FeatureSpec for f in self.features), "expected FeatureSpec")
        unique(tuple(f.name for f in self.features), "feature names")
        require(width > 0, "width must be positive")
        self.width = width
        count = len(self.features)
        self.identity = nn.Embedding(count, width)
        self.numeric_weight = nn.Parameter(torch.empty(count, width))
        self.missing_state = nn.Embedding(count, width)
        self.masked_state = nn.Embedding(count, width)
        self.categories = nn.ModuleDict({str(i): nn.Embedding(len(f.categories) + 1, width)
                                        for i, f in enumerate(self.features)
                                        if f.kind == "categorical"})
        nn.init.normal_(self.numeric_weight, std=0.02)
        for embedding in (self.identity, self.missing_state, self.masked_state,
                          *self.categories.values()):
            nn.init.normal_(embedding.weight, std=0.02)

    @property
    def preprocessing_hash(self) -> str:
        return sha256(canonical_json([f.to_dict() for f in self.features]).encode()).hexdigest()

    def prepare(self, view: CovariateView) -> FeatureBatch:
        """Convert a permitted immutable view; do not fit or update preprocessing.

        The view must contain exactly the schema's features, in any order.
        Feature permissions are enforced by the merged CovariateView contract.
        """
        require(type(view) is CovariateView, "expected CovariateView")
        require(set(view.columns) == {f.name for f in self.features}, "feature schema mismatch")
        for name in view.columns:
            view.registry.require(name, view.spec.endpoint, view.use)
        shape = (len(view.original_ids), len(self.features))
        numeric = torch.zeros(shape, dtype=self.numeric_weight.dtype,
                              device=self.numeric_weight.device)
        category = torch.zeros(shape, dtype=torch.long, device=numeric.device)
        missing = torch.zeros(shape, dtype=torch.bool, device=numeric.device)
        for j, feature in enumerate(self.features):
            values = view.column(feature.name)
            vocabulary = {canonical_json(value): i for i, value in enumerate(feature.categories)}
            for i, value in enumerate(values):
                if value is None:
                    missing[i, j] = True
                elif feature.kind == "numeric":
                    require(type(value) in (int, float), f"non-numeric value for {feature.name}")
                    numeric[i, j] = (value - feature.mean) / feature.scale
                else:
                    category[i, j] = vocabulary.get(canonical_json(value), len(vocabulary))
        return FeatureBatch(torch.arange(shape[1], device=numeric.device), numeric, category,
                            missing, torch.zeros_like(missing), torch.zeros_like(missing))

    def forward(self, batch: FeatureBatch) -> Tensor:
        require(type(batch) is FeatureBatch, "expected FeatureBatch")
        ids, values = batch.feature_ids, batch.numeric_values
        require(values.ndim == 2 and values.is_floating_point(), "numeric values must be [B,F] floats")
        require(ids.ndim == 1 and ids.dtype == torch.long and ids.numel() == values.shape[1],
                "feature IDs must be [F] integers")
        tensors = (ids, values, batch.categorical_values, batch.missing, batch.masked, batch.padding)
        require(all(x.device == self.numeric_weight.device for x in tensors), "batch device mismatch")
        require(batch.categorical_values.shape == values.shape and batch.categorical_values.dtype == torch.long,
                "categorical values must be [B,F] integers")
        require(all(x.shape == values.shape and x.dtype == torch.bool
                    for x in (batch.missing, batch.masked, batch.padding)), "invalid state masks")
        require(bool(((ids >= -1) & (ids < len(self.features))).all()), "unknown feature ID")
        require(ids[ids >= 0].unique().numel() == (ids >= 0).sum().item(), "duplicate feature IDs")
        require(bool(batch.padding[:, ids == -1].all()), "anonymous feature must be padding")
        safe_ids = ids.clamp_min(0)
        active = ~(batch.missing | batch.masked | batch.padding)
        tokens = self.identity(safe_ids).unsqueeze(0).expand(values.shape[0], -1, -1)
        tokens = (tokens + self.missing_state(safe_ids) * batch.missing.unsqueeze(-1)
                  + self.masked_state(safe_ids) * batch.masked.unsqueeze(-1))
        for j, feature in enumerate(self.features):
            columns = ids == j
            observed = active[:, columns]
            if feature.kind == "numeric":
                x = torch.where(observed, values[:, columns], 0).to(tokens.dtype)
                require(bool(torch.isfinite(x).all()), "nonfinite observed numeric value")
                addition = x.unsqueeze(-1) * self.numeric_weight[j]
            else:
                x = torch.where(observed, batch.categorical_values[:, columns], 0)
                require(bool(((x >= 0) & (x <= len(feature.categories))).all()),
                        "category index outside observed/unknown range")
                addition = self.categories[str(j)](x) * observed.unsqueeze(-1)
            tokens[:, columns] = tokens[:, columns] + addition
        return tokens.masked_fill(batch.padding.unsqueeze(-1), 0)
