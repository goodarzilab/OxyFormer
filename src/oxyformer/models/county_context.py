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
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from oxyformer.contracts import CovariateView, SplitManifest
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.tokens import FeatureBatch
from oxyformer.provenance import canonical_json, check_hash, nonempty, require


class _DropoutReplay(TorchDispatchMode):
    """Apply already drawn masks through the unchanged dropout/autograd kernels.

    The masks were drawn query-first, with the original tensor strides. Their
    batched application is layer-first. This mode does not consume any RNG.
    """

    def __init__(self, masks):
        super().__init__()
        self.masks = iter(masks)
        self.remaining = len(masks)

    def _take(self, values, probability):
        require(self.remaining > 0, "unexpected context dropout operation")
        expected, mask = next(self.masks)
        self.remaining -= 1
        require(abs(probability - expected) < 1e-12 and values.shape == mask.shape,
                "context dropout replay shape/probability mismatch")
        return mask

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if operation == torch.ops.aten.bernoulli_.float:
            values = args[0]
            keep = args[1] if len(args) > 1 else kwargs.get("p", .5)
            # CPU dropout asks for a Bernoulli tensor and scales it afterward.
            mask = self._take(values, 1 - keep)
            return values.copy_(mask.ne(0).to(values.dtype))
        if operation == torch.ops.aten.native_dropout.default:
            values, probability, training = args
            if training and 0 < probability < 1:
                mask = self._take(values, probability)
                return values * mask, mask.ne(0)
        return operation(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        result = super().__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            require(self.remaining == 0, "unused context dropout masks")
        return result


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
                 dropout: float = 0.1, reference_cap: int | None = None):
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
        require(reference_cap is None or (type(reference_cap) is int and reference_cap > 0),
                "reference cap must be a positive integer or None")
        self.reference_cap = reference_cap
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
        # Raw, parameter-independent inputs are reusable in training too. Prepare
        # only requested rows so self-excluded/unused values are never converted.
        self._prepared: FeatureBatch | None = None
        self._prepared_ids: set[str] = set()
        self._prepared_identity = None
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
        # Study only: seedless/outcome-blind canonical IDs, self removed first.
        ids = tuple(sorted(oid for oid, route in self._routes.items()
                           if route == county and oid != query_id))
        return ids if self.reference_cap is None else ids[:self.reference_cap]

    def _reference_batch(self, ids: tuple[str, ...]) -> FeatureBatch:
        tokenizer = self.encoder.tokenizer
        weight = tokenizer.numeric_weight
        identity = (self._references, tokenizer.features, weight.device, weight.dtype)
        previous = self._prepared_identity
        # Immutable specifications are identified by object, not Python value
        # equality: categorical 1 and 1.0 (and signed-zero means) can compare
        # equal while preparing different tensor bytes.
        if (previous is None or previous[0] is not identity[0]
                or previous[1] is not identity[1] or previous[2:] != identity[2:]):
            self._prepared = None
            self._prepared_ids.clear()
            self._prepared_identity = identity
        pending = tuple(oid for oid in ids if oid not in self._prepared_ids)
        if pending:
            rows = tuple(self._references.values[self._rows[oid]] for oid in pending)
            view = replace(self._references, original_ids=pending, values=rows,
                           lineage=replace(self._references.lineage, unit_ids=pending))
            # An inference call may populate inputs subsequently used by autograd.
            # Keep the retained raw tensors ordinary tensors, even in that case.
            with torch.inference_mode(False):
                prepared = tokenizer.prepare(view)
                if self._prepared is None:
                    shape = (len(self._references.original_ids), len(tokenizer.features))
                    self._prepared = FeatureBatch(prepared.feature_ids, *(
                        value.new_empty(shape) for value in (
                            prepared.numeric_values, prepared.categorical_values,
                            prepared.missing, prepared.masked, prepared.padding)))
                positions = torch.tensor([self._rows[oid] for oid in pending],
                                         device=weight.device, dtype=torch.long)
                for field in ("numeric_values", "categorical_values", "missing", "masked", "padding"):
                    getattr(self._prepared, field).index_copy_(0, positions, getattr(prepared, field))
            self._prepared_ids.update(pending)
        positions = torch.tensor([self._rows[oid] for oid in ids],
                                 device=weight.device, dtype=torch.long)
        return FeatureBatch(self._prepared.feature_ids, *(
            getattr(self._prepared, field).index_select(0, positions)
            for field in ("numeric_values", "categorical_values", "missing", "masked", "padding")))

    def _pool(self, ids: tuple[str, ...]) -> Tensor:
        if not ids:
            return self.seeds.new_zeros((4, 64))
        _, z = self.encoder(self._reference_batch(ids))
        memory = self.reference_norm(z.unsqueeze(0))
        seeds = self.seeds
        attended, _ = self.attention(self.seed_norm(seeds), memory, memory, need_weights=False)
        states = seeds + self.dropout(attended)
        states = states + self.dropout(self.ff(self.ff_norm(states)))
        return self.output_norm(states).squeeze(0)

    def _dropout_masks(self, lengths: tuple[int, ...], width: int):
        """Draw masks in scalar-path order, including MHA's transposed strides."""
        tokens = len(self.encoder.tokenizer.features) + 1
        encoder_masks, pool_masks = [], []

        def draw(module, shape, *, transpose=False):
            probability = module.dropout if isinstance(module, nn.MultiheadAttention) else module.p
            if not module.training or probability == 0:
                return None
            values = self.seeds.new_ones(shape)
            if transpose:
                values = values.transpose(0, 1)
            # Calling ordinary dropout retains the caller's RNG policy, including
            # fit._ReferenceDropout's CPU stream when the model lives on CUDA.
            return probability, F.dropout(values, probability, training=True)

        for count in lengths:
            enc, pool = [], []
            for block in self.encoder.blocks:
                enc.extend((
                    draw(block.self_attn, (count, block.self_attn.num_heads, tokens, tokens)),
                    draw(block.dropout1, (tokens, count, 64), transpose=True),
                    draw(block.dropout, (count, tokens, block.linear1.out_features)),
                    draw(block.dropout2, (count, tokens, 64))))
            pool.extend((
                draw(self.attention, (1, self.attention.num_heads, 4, count)),
                draw(self.dropout, (4, 1, 64), transpose=True),
                draw(self.ff[2], (1, 4, self.ff[0].out_features)),
                draw(self.dropout, (1, 4, 64))))
            if pool[0] is not None:
                probability, mask = pool[0]
                pool[0] = probability, F.pad(mask, (0, width - count))
            encoder_masks.append([entry for entry in enc if entry is not None])
            pool_masks.append([entry for entry in pool if entry is not None])

        def combine(rows):
            return [(column[0][0], torch.cat([entry[1] for entry in column], dim=0))
                    for column in zip(*rows)]
        return combine(encoder_masks), combine(pool_masks)

    def _pool_occurrences(self, reference_sets: tuple[tuple[str, ...], ...]) -> Tensor:
        """One encoder call over stochastic occurrences, with original RNG draws."""
        active = [i for i, ids in enumerate(reference_sets) if ids]
        if not active:
            return self.seeds.new_zeros((len(reference_sets), 4, 64))
        lengths = tuple(len(reference_sets[i]) for i in active)
        width = max(lengths)
        union = tuple(sorted({oid for ids in reference_sets for oid in ids}))
        raw = self._reference_batch(union)
        lookup = {oid: i for i, oid in enumerate(union)}
        occurrences = torch.tensor([lookup[oid] for i in active for oid in reference_sets[i]],
                                   device=self.seeds.device, dtype=torch.long)
        batch = FeatureBatch(raw.feature_ids, *(getattr(raw, field).index_select(0, occurrences)
            for field in ("numeric_values", "categorical_values", "missing", "masked", "padding")))
        encoder_masks, pool_masks = self._dropout_masks(lengths, width)
        with _DropoutReplay(encoder_masks):
            _, encoded = self.encoder(batch)
        indices, offset = [], 0
        for count in lengths:
            indices.append(list(range(offset, offset + count)) + [0] * (width - count))
            offset += count
        index = torch.tensor(indices, device=encoded.device, dtype=torch.long)
        padding = torch.arange(width, device=encoded.device)[None, :] >= torch.tensor(
            lengths, device=encoded.device)[:, None]
        memory = self.reference_norm(encoded[index]).masked_fill(padding[:, :, None], 0)
        seeds = self.seeds.expand(len(active), -1, -1)
        with _DropoutReplay(pool_masks):
            attended, _ = self.attention(self.seed_norm(seeds), memory, memory,
                                         key_padding_mask=padding, need_weights=False)
            states = seeds + self.dropout(attended)
            states = states + self.dropout(self.ff(self.ff_norm(states)))
        pooled = self.output_norm(states)
        outputs = [self.seeds.new_zeros((4, 64)) for _ in reference_sets]
        for position, value in zip(active, pooled.unbind()):
            outputs[position] = value
        return torch.stack(outputs)

    def _pool_batch(self, reference_sets: tuple[tuple[str, ...], ...]) -> Tensor:
        """Encode the needed union once, then pool masked sets in tensor batches.

        Bound the padded PMA workspace independently of prediction-pass size;
        this does not change training minibatches or parameter updates.
        """
        if not reference_sets:
            return self.seeds.new_empty((0, 4, 64))
        # Stochastic reference occurrences retain independent encoder draws.
        stochastic = any((isinstance(m, nn.Dropout) and m.training and m.p > 0)
                         or (isinstance(m, nn.MultiheadAttention) and m.training and m.dropout > 0)
                         for m in self.modules())
        if stochastic:
            return self._pool_occurrences(reference_sets)
        union = tuple(sorted({oid for ids in reference_sets for oid in ids}))
        if not union:
            return self.seeds.new_zeros((len(reference_sets), 4, 64))
        _, encoded = self.encoder(self._reference_batch(union))
        lookup = {oid: i for i, oid in enumerate(union)}
        outputs = [self.seeds.new_zeros((4, 64)) for _ in reference_sets]
        active = [i for i, ids in enumerate(reference_sets) if ids]
        # Similar set lengths limit padding, while output order stays unchanged.
        active.sort(key=lambda i: len(reference_sets[i]))
        start = 0
        while start < len(active):
            end = start + 1
            while end < len(active) and (end - start + 1) * len(reference_sets[active[end]]) <= 262144:
                end += 1
            selected = active[start:end]
            width = max(len(reference_sets[i]) for i in selected)
            indices = [[lookup[oid] for oid in reference_sets[i]]
                       + [0] * (width - len(reference_sets[i])) for i in selected]
            index = torch.tensor(indices, dtype=torch.long, device=encoded.device)
            lengths = torch.tensor([len(reference_sets[i]) for i in selected], device=encoded.device)
            padding = torch.arange(width, device=encoded.device)[None, :] >= lengths[:, None]
            memory = self.reference_norm(encoded[index])
            # Padding has no value or gradient channel, even before attention.
            memory = memory.masked_fill(padding[:, :, None], 0)
            seeds = self.seeds.expand(len(selected), -1, -1)
            attended, _ = self.attention(self.seed_norm(seeds), memory, memory,
                                         key_padding_mask=padding, need_weights=False)
            states = seeds + self.dropout(attended)
            states = states + self.dropout(self.ff(self.ff_norm(states)))
            pooled = self.output_norm(states)
            for position, value in zip(selected, pooled.unbind()):
                outputs[position] = value
            start = end
        return torch.stack(outputs)

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
        county_ids = {}
        for oid, county in sorted(self._routes.items()):
            county_ids.setdefault(county, []).append(oid)
        reference_sets = tuple(tuple(ref for ref in county_ids.get(county, ()) if ref != oid)
                               [:self.reference_cap] for oid, county in zip(ids, counties))
        if not cacheable:
            return self._pool_batch(reference_sets)
        keys = tuple(ContextCacheKey(generation[0], self._fold, refs, generation[2],
                     self.checkpoint_hash, self.parameter_version, generation[5], generation[6])
                     for refs in reference_sets)
        missing = tuple(dict.fromkeys(key for key in keys if key not in self._cache))
        if missing:
            pooled = self._pool_batch(tuple(key.reference_ids for key in missing))
            for key, value in zip(missing, pooled.unbind()):
                self._cache[key] = value.detach().clone()
        # stack copies: caller mutation never changes a cached tensor.
        return torch.stack([self._cache[key] for key in keys]) if keys else self.seeds.new_empty((0, 4, 64))
