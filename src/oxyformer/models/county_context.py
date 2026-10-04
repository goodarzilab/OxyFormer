"""Four PMA seeds over fold-owned X only, with leave-self-out before encoding.

CountyContext owns an immutable context CovariateView and its FeatureEncoder.
Supply the COMPLETE split.training_ids(fold) view: selecting a subset through
labels is not part of this API. County strings are approved routing metadata,
never embeddings. Query input is IDs and county routing ONLY; held-out X, A, Y,
residuals and fitted county effects have no input channel here.

Preprocessing must already have been fitted in this split's training partition
by the training unit. prepare() never fits it. Empty/unseen counties (including
a singleton training county after self-exclusion) yield four constant zeros.
This defines a transfer fallback, not permission to bypass scientific eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

import torch
from torch import Tensor, nn

from oxyformer.contracts import CovariateView, SplitManifest
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.provenance import canonical_json, check_hash, nonempty, require


@dataclass(frozen=True, slots=True)
class ContextCacheKey:
    split_hash: str
    fold: int
    reference_ids: tuple[str, ...]
    preprocessing_hash: str
    checkpoint_hash: str
    parameter_version: int
    model_hash: str
    reference_hash: str


class CountyContext(nn.Module):
    """Reference encoder and four-seed pre-LN pooling-by-attention block.

    forward(query_ids, query_counties) returns [B,4,64]. Known training query
    routes must match their reference route. Design/excluded IDs are refused.
    New IDs are allowed for explicitly separate transfer prediction.

    Cache only under eval() AND no_grad()/inference_mode(). Training and any
    gradient-enabled evaluation recompute the encoder/PMA graph. Cache identity
    includes actual tensor bytes as well as caller checkpoint/parameter version,
    so optimizer steps, load_state_dict and device/dtype changes invalidate it.
    Cached tensors are private; returned outputs cannot mutate the cache.
    Recreate this object with matching references/preprocessing when restoring
    a state_dict; those immutable artifacts are not stored as model tensors.
    """

    def __init__(self, encoder: FeatureEncoder, references: CovariateView,
                 split: SplitManifest, fold: int, reference_counties: tuple[str, ...],
                 *, county_field: str, checkpoint_hash: str, parameter_version: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        require(type(encoder) is FeatureEncoder, "expected FeatureEncoder")
        require(type(references) is CovariateView and references.use == "context",
                "references require a context CovariateView")
        require(type(split) is SplitManifest, "expected SplitManifest")
        split.spec.assert_compatible(references.spec)
        require(set(references.original_ids) == set(split.training_ids(fold)),
                "references must contain exactly the permitted training IDs")
        require(references.lineage.split_hash == split.content_hash, "reference split mismatch")
        require(set(references.columns) == {f.name for f in encoder.tokenizer.features},
                "reference feature schema mismatch")
        for name in references.columns:
            references.registry.require(name, references.spec.endpoint, "context")
        rule = references.registry.require(county_field, references.spec.endpoint, "county_routing")
        require(rule.role == "county", "county routing requires county role")
        counties = tuple(reference_counties)
        require(len(counties) == len(references.original_ids), "reference county alignment mismatch")
        for county in counties:
            nonempty(county, "county route")
        require(0 <= dropout < 1, "dropout must be in [0,1)")
        self.encoder = encoder
        self._references = references
        self._split = split
        self._fold = fold
        self._routes = dict(zip(references.original_ids, counties))
        self._rows = {oid: i for i, oid in enumerate(references.original_ids)}
        self.seeds = nn.Parameter(torch.empty(1, 4, 64))
        nn.init.normal_(self.seeds, std=0.02)
        self.reference_norm = nn.LayerNorm(64)
        self.seed_norm = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(64)
        self._cache: dict[ContextCacheKey, Tensor] = {}
        self._cache_generation = None
        self.set_cache_identity(checkpoint_hash=checkpoint_hash, parameter_version=parameter_version)
        require(sum(p.numel() for p in self.parameters()) <= 1_000_000,
                "backbone exceeds nuisance parameter cap; heads must check the complete network")

    @property
    def references(self) -> CovariateView:
        return self._references

    @property
    def cache_keys(self) -> tuple[ContextCacheKey, ...]:
        return tuple(self._cache)

    def set_cache_identity(self, *, checkpoint_hash: str, parameter_version: int) -> None:
        check_hash(checkpoint_hash, "checkpoint hash")
        require(type(parameter_version) is int and parameter_version >= 0, "invalid parameter version")
        self.checkpoint_hash = checkpoint_hash
        self.parameter_version = parameter_version
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_generation = None

    def train(self, mode: bool = True):
        self.clear_cache()
        return super().train(mode)

    def _model_hash(self) -> str:
        digest = sha256()
        for name, value in self.state_dict().items():
            digest.update(canonical_json([name, str(value.dtype), str(value.device), list(value.shape)]).encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _reference_ids(self, query_id: str, county: str) -> tuple[str, ...]:
        return tuple(sorted(oid for oid, route in self._routes.items()
                            if route == county and oid != query_id))

    def _pool(self, ids: tuple[str, ...]) -> Tensor:
        if not ids:
            return self.seeds.new_zeros((4, 64))
        rows = tuple(self._references.values[self._rows[oid]] for oid in ids)
        view = replace(self._references, original_ids=ids, values=rows,
                       lineage=replace(self._references.lineage, unit_ids=ids))
        _, z = self.encoder(self.encoder.tokenizer.prepare(view))
        memory = self.reference_norm(z.unsqueeze(0))
        seeds = self.seeds
        attended, _ = self.attention(self.seed_norm(seeds), memory, memory, need_weights=False)
        states = seeds + self.dropout(attended)
        states = states + self.dropout(self.ff(self.ff_norm(states)))
        return self.output_norm(states).squeeze(0)

    def forward(self, query_ids: tuple[str, ...], query_counties: tuple[str, ...]) -> Tensor:
        ids, counties = tuple(query_ids), tuple(query_counties)
        require(len(ids) == len(counties), "query county alignment mismatch")
        for oid, county in zip(ids, counties):
            nonempty(oid, "query ID")
            nonempty(county, "county route")
            require(oid not in self._split.design_ids + self._split.excluded_ids, "sealed query ID")
            require(oid not in self._routes or self._routes[oid] == county, "training query route mismatch")
        # Disabling only the parent flag must never cache a stochastic child.
        cacheable = not torch.is_grad_enabled() and all(not m.training for m in self.modules())
        if cacheable:
            generation = (self._split.content_hash, self._fold, self.encoder.tokenizer.preprocessing_hash,
                          self.checkpoint_hash, self.parameter_version, self._model_hash(),
                          self._references.content_hash, tuple(sorted(self._routes.items())))
            if generation != self._cache_generation:
                self.clear_cache()
                self._cache_generation = generation
        outputs = []
        for oid, county in zip(ids, counties):
            reference_ids = self._reference_ids(oid, county)
            if cacheable:
                key = ContextCacheKey(generation[0], self._fold, reference_ids, generation[2],
                                      self.checkpoint_hash, self.parameter_version, generation[5], generation[6])
                if key not in self._cache:
                    self._cache[key] = self._pool(reference_ids).detach().clone()
                outputs.append(self._cache[key])
            else:
                outputs.append(self._pool(reference_ids))
        # stack copies: caller mutation never changes a cached tensor.
        return torch.stack(outputs) if outputs else self.seeds.new_empty((0, 4, 64))
