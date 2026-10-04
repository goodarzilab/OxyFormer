"""Fold-local covariate-only masked pretraining with exact slice continuation.

Pass a reviewed SSL CovariateView containing exactly split.training_ids(fold).
The split producer must already have validated geographic/entity separation (see
INTERFACES.md). Feature kinds and semantic families are explicit reviewed config,
never inferred from column names. Runtime limits are separate from SSLSettings.

Each invocation claims output_dir/ssl exclusively. Resume into a NEW attempt;
there is no global cache. CheckpointArtifact.complete distinguishes scientific
completion from a resumable execution slice. Its archive contains both the latest
model (paired with optimizer state) and best_model (for downstream initialization).
The four-hour bound is cooperative at minibatch boundaries, including validation;
reserve enough checkpoint_margin_seconds for the longest batch and serialization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from oxyformer.contracts import CovariateView, SplitManifest
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.tokens import FeatureBatch, FeatureSpec
from oxyformer.provenance import ArtifactLineage, Immutable, canonical_json, require, unique
from oxyformer.training.checkpoint import (
    CheckpointArtifact, CheckpointIdentity, CheckpointRequest,
    capture_rng, load_checkpoint, restore_rng, save_checkpoint,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SSLSettings(Immutable):
    fold: int
    feature_kinds: tuple[tuple[str, Literal["numeric", "categorical"]], ...]
    families: tuple[tuple[str, ...], ...]
    mask_rate: float = 0.30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.10
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    validation_fraction: float = 0.10
    stopping_ids: tuple[str, ...] = ()

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.fold >= 0, "negative fitting fold")
        require(0 < self.mask_rate <= 1, "mask rate must be in (0,1]")
        require(self.learning_rate > 0 and self.weight_decay >= 0, "invalid optimizer settings")
        require(0 <= self.dropout < 1, "invalid dropout")
        require(self.batch_size > 0 and 0 < self.max_epochs <= 30 and self.patience > 0,
                "invalid training bounds")
        require(0 < self.validation_fraction < 1, "invalid stopping fraction")
        unique(tuple(k for k, _ in self.feature_kinds), "feature kinds")
        require(bool(self.families) and all(self.families), "semantic families must be explicit")
        flattened = tuple(name for family in self.families for name in family)
        unique(flattened, "family membership")
        require(set(flattened) == {k for k, _ in self.feature_kinds}, "family/schema mismatch")
        unique(self.stopping_ids, "stopping IDs")


@dataclass(frozen=True, kw_only=True)
class PretrainConfig:
    settings: SSLSettings
    output_dir: str
    predecessor: CheckpointArtifact | None = None
    slice_seconds: float = 4 * 60 * 60
    checkpoint_margin_seconds: float = 120
    max_batches: int | None = None  # Execution budget, including validation batches.
    stop_request: CheckpointRequest | None = None
    controller_state: dict | None = None  # JSON-only nested fold/tuning controller state.
    device: str = "cpu"

    def __post_init__(self):
        require(type(self.settings) is SSLSettings, "expected SSLSettings")
        require(Path(self.output_dir).is_absolute(), "attempt output_dir must be absolute")
        require(math.isfinite(self.slice_seconds) and 0 < self.slice_seconds <= 14400,
                "slice must be at most four hours")
        require(0 <= self.checkpoint_margin_seconds < self.slice_seconds, "invalid checkpoint margin")
        require(self.max_batches is None or (type(self.max_batches) is int and self.max_batches > 0),
                "invalid slice batch budget")


class StatefulSampler:
    """A single-process sampler: explicit permutation, cursor and generator state."""

    def __init__(self, size: int, seed: int):
        require(size > 0, "empty fitting sampler")
        self.size = size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.order: list[int] = []
        self.cursor = 0

    def indices(self, batch_size: int) -> list[int]:
        if not self.order:
            self.order = torch.randperm(self.size, generator=self.generator).tolist()
        # The caller advances only after a complete optimizer transaction.
        return self.order[self.cursor:self.cursor + batch_size]

    def finish_epoch(self):
        require(self.cursor == self.size, "incomplete sampler epoch")
        self.order = []
        self.cursor = 0

    def state_dict(self) -> dict:
        return {"size": self.size, "order": list(self.order), "cursor": self.cursor,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state: dict):
        require(state["size"] == self.size, "sampler size mismatch")
        order, cursor = state["order"], state["cursor"]
        require((not order and cursor == 0) or
                (sorted(order) == list(range(self.size)) and 0 <= cursor <= self.size),
                "invalid sampler permutation/cursor")
        self.order, self.cursor = list(order), cursor
        self.generator.set_state(state["generator"])


def _subset(view: CovariateView, ids: tuple[str, ...]) -> CovariateView:
    by_id = dict(zip(view.original_ids, view.values))
    return replace(view, original_ids=ids, values=tuple(by_id[x] for x in ids),
                   lineage=replace(view.lineage, unit_ids=ids,
                                   parent_hashes=(view.content_hash,)))


def fit_preprocessing(view: CovariateView, settings: SSLSettings) -> tuple[FeatureSpec, ...]:
    """Fit on the optimization subset only, excluding the stopping subset."""
    kinds = dict(settings.feature_kinds)
    result = []
    for name in view.columns:
        values = [x for x in view.column(name) if x is not None]
        if kinds[name] == "numeric":
            require(all(type(x) in (int, float) for x in values), f"non-numeric feature: {name}")
            mean = float(np.mean(values)) if values else 0.0
            scale = float(np.std(values)) if values else 1.0
            result.append(FeatureSpec(name=name, kind="numeric", mean=mean, scale=scale or 1.0))
        else:
            categories = {canonical_json(x): x for x in values}
            result.append(FeatureSpec(name=name, kind="categorical",
                                      categories=tuple(categories[k] for k in sorted(categories))))
    return tuple(result)


def family_mask(batch: FeatureBatch, families: tuple[tuple[int, ...], ...], rate: float,
                *, generator: torch.Generator | None = None) -> FeatureBatch:
    """One Bernoulli draw per record/family, shared by totals and complements.

    True missingness remains separate, including when a missing family is masked.
    Missing targets and padding never contribute a reconstruction loss.
    """
    draws = torch.rand((batch.numeric_values.shape[0], len(families)), generator=generator) < rate
    masked = torch.zeros_like(batch.missing)
    for j, members in enumerate(families):
        masked[:, list(members)] = draws[:, j:j + 1].to(masked.device)
    return replace(batch, masked=masked & ~batch.padding)


class MaskedReconstructor(nn.Module):
    def __init__(self, features: tuple[FeatureSpec, ...], dropout: float):
        super().__init__()
        self.encoder = FeatureEncoder(features, dropout=dropout)
        self.heads = nn.ModuleList(nn.Linear(self.encoder.width,
                                  1 if f.kind == "numeric" else len(f.categories) + 1)
                                  for f in features)
        require(sum(p.numel() for p in self.parameters()) <= 1_000_000,
                "SSL network exceeds parameter cap")

    def forward(self, batch: FeatureBatch) -> list[torch.Tensor]:
        hidden, _ = self.encoder(batch)
        return [head(hidden[:, j]) for j, head in enumerate(self.heads)]


def reconstruction_totals(predictions, batch: FeatureBatch, features):
    sums, counts = [], []
    for j, feature in enumerate(features):
        valid = batch.masked[:, j] & ~batch.missing[:, j] & ~batch.padding[:, j]
        count = int(valid.sum())
        counts.append(count)
        if not count:
            sums.append(predictions[j].sum() * 0)
        elif feature.kind == "numeric":
            sums.append(F.huber_loss(predictions[j][valid, 0], batch.numeric_values[valid, j],
                                     reduction="sum", delta=1.0))
        else:
            sums.append(F.cross_entropy(predictions[j][valid], batch.categorical_values[valid, j],
                                        reduction="sum"))
    return sums, counts


def balanced_loss(sums, counts, families):
    contributions = []
    for family in families:
        observed = [sums[j] / counts[j] for j in family if counts[j]]
        if observed:
            contributions.append(sum(observed) / len(observed))
    return sum(contributions) / len(contributions) if contributions else None


def _batch(batch: FeatureBatch, indices) -> FeatureBatch:
    return FeatureBatch(batch.feature_ids, *(x[indices] for x in (
        batch.numeric_values, batch.categorical_values, batch.missing, batch.masked, batch.padding)))


def scientific_code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    entries = [(str(p.relative_to(root)), sha256(p.read_bytes()).hexdigest())
               for p in sorted(root.rglob("*.py"))]
    return sha256(canonical_json(entries).encode()).hexdigest()


def environment_identity(device: torch.device) -> tuple[tuple[str, str], ...]:
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    values = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(), "torch": str(torch.__version__),
              "numpy": np.__version__, "packages": sha256(canonical_json(packages).encode()).hexdigest(),
              "torch_build": sha256(torch.__config__.show().encode()).hexdigest(),
              "threads": str(torch.get_num_threads()), "interop_threads": str(torch.get_num_interop_threads()),
              "device": str(device), "deterministic": str(torch.are_deterministic_algorithms_enabled()),
              "matmul_precision": torch.get_float32_matmul_precision(),
              "cuda": str(torch.version.cuda), "cudnn": str(torch.backends.cudnn.version()),
              "cudnn_deterministic": str(torch.backends.cudnn.deterministic),
              "cudnn_benchmark": str(torch.backends.cudnn.benchmark),
              "matmul_tf32": str(torch.backends.cuda.matmul.allow_tf32),
              "cudnn_tf32": str(torch.backends.cudnn.allow_tf32)}
    if device.type == "cuda":
        values["accelerator"] = str(torch.cuda.get_device_properties(device))
    return tuple(sorted(values.items()))


def _model_hash(state: dict) -> str:
    digest = sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(canonical_json([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _controller_state(value: dict | None) -> dict:
    # JSON roundtrip rejects custom objects/tensors and copies caller-owned state.
    require(value is None or type(value) is dict, "controller state must be a JSON object")
    return json.loads(canonical_json(value or {}))


def pretrain(training_covariate_view: CovariateView, split_manifest: SplitManifest,
             config: PretrainConfig, seed: int) -> CheckpointArtifact:
    """Fit/continue approved X only; return an immutable, hash-validated artifact."""
    started = time.monotonic()
    require(type(training_covariate_view) is CovariateView, "expected CovariateView")
    require(type(split_manifest) is SplitManifest, "expected SplitManifest")
    require(type(config) is PretrainConfig, "expected PretrainConfig")
    view, split, settings = training_covariate_view, split_manifest, config.settings
    view.spec.assert_compatible(split.spec)
    require(view.use == "ssl", "SSL view required")
    require(type(seed) is int and seed in split.seed_ids, "unregistered SSL seed")
    allowed = split.training_ids(settings.fold)
    require(set(view.original_ids) == set(allowed), "view must contain exactly the permitted training records")
    require(set(view.columns) == {k for k, _ in settings.feature_kinds}, "SSL feature schema mismatch")
    for name in view.columns:
        rule = view.registry.require(name, view.spec.endpoint, "ssl")
        require(rule.role == "predictor", "only approved predictor fields may enter SSL")
    require(len(allowed) >= 2, "SSL stopping requires at least two training records")
    if settings.stopping_ids:
        stopping_ids = settings.stopping_ids
        require(set(stopping_ids) < set(allowed), "stopping IDs must be within the current fitting partition")
    else:
        count = min(len(allowed) - 1, max(1, int(len(allowed) * settings.validation_fraction)))
        stopping_ids = tuple(sorted(allowed, key=lambda oid: sha256(
            canonical_json([seed, oid, "ssl-stopping"]).encode()).digest())[:count])
    stopping_set = set(stopping_ids)
    fitting_ids = tuple(oid for oid in view.original_ids if oid not in stopping_set)
    fitting, stopping = _subset(view, fitting_ids), _subset(view, stopping_ids)
    features = fit_preprocessing(fitting, settings)
    preprocessing = tuple(f.to_json() for f in features)
    preprocessing_hash = sha256(canonical_json([f.to_dict() for f in features]).encode()).hexdigest()
    device = torch.device(config.device)
    require(device.type in ("cpu", "cuda"), "unsupported training device")
    identity = CheckpointIdentity(training_ids=view.original_ids, data_hash=view.content_hash,
        split_hash=split.content_hash, config_hash=settings.content_hash,
        preprocessing_hash=preprocessing_hash, seed=seed,
        scientific_code_hash=scientific_code_fingerprint(), environment=environment_identity(device))
    loaded = load_checkpoint(config.predecessor, identity) if config.predecessor else None
    if loaded is not None:
        require(loaded["preprocessing"] == preprocessing, "stale preprocessing state")
        require(loaded["fitting_ids"] == fitting_ids and loaded["stopping_ids"] == stopping_ids,
                "checkpoint fitting/stopping partition mismatch")
        if config.controller_state is not None:
            require(loaded["controller"] == _controller_state(config.controller_state),
                    "cannot replace resumed controller state")
    # Claim an exclusive directory even when the enclosing StageRequest root exists.
    root = Path(config.output_dir).resolve() / "ssl"
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700, exist_ok=False)
    request = config.stop_request or CheckpointRequest()
    with request.signals():
        return _run(view, settings, config, seed, features, preprocessing, identity,
                    fitting, stopping, loaded, root, request, started, device)


def _run(view, settings, config, seed, features, preprocessing, identity,
         fitting, stopping, loaded, root, request, started, device):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    model = MaskedReconstructor(features, settings.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                                 weight_decay=settings.weight_decay)
    # Fixed registered LR; explicit scheduler state supports generic continuation.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    sampler = StatefulSampler(len(fitting.original_ids), seed)
    train_batch = model.encoder.tokenizer.prepare(fitting)
    validation_batch = model.encoder.tokenizer.prepare(stopping)
    family_indices = tuple(tuple(view.columns.index(name) for name in family) for family in settings.families)
    validation_batch = family_mask(validation_batch, family_indices, settings.mask_rate,
                                   generator=torch.Generator().manual_seed(seed + 1))
    # Fixed stopping masks must expose at least one observed target per available
    # family, even in tiny folds. This does not alter training mask probabilities.
    for members in family_indices:
        available = (~validation_batch.missing[:, list(members)]).any(dim=1)
        selected = (validation_batch.masked[:, list(members)] &
                    ~validation_batch.missing[:, list(members)]).any(dim=1)
        if bool(available.any()) and not bool(selected.any()):
            first = int(torch.where(available)[0][0])
            validation_batch.masked[first, list(members)] = True
    require(bool((~train_batch.missing).any()), "no observed fitting targets")
    require(bool((~validation_batch.missing).any()), "no observed stopping targets")
    progress = {"epoch": 0, "step": 0, "phase": "train", "validation_cursor": 0,
                "validation_sums": [0.0] * len(features), "validation_counts": [0] * len(features),
                "best_loss": None, "bad_epochs": 0, "history": []}
    best_model = None
    controller = _controller_state(config.controller_state)
    if loaded is not None:
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        sampler.load_state_dict(loaded["sampler"])
        progress, best_model, controller = loaded["progress"], loaded["best_model"], loaded["controller"]
        # Restore last, after construction and preprocessing. No stochastic work
        # may occur between this restoration and the next logical training batch.
        restore_rng(loaded["rng"])
    completed = bool(config.predecessor and config.predecessor.complete)
    reason = config.predecessor.reason if completed else None
    slice_batches = 0
    while not completed:
        if request.requested:
            reason = "requested"
            break
        if time.monotonic() - started >= config.slice_seconds - config.checkpoint_margin_seconds:
            reason = "slice_limit"
            break
        if config.max_batches is not None and slice_batches >= config.max_batches:
            reason = "batch_limit"
            break
        if progress["phase"] == "train":
            indices = sampler.indices(settings.batch_size)
            masked = family_mask(_batch(train_batch, indices), family_indices, settings.mask_rate)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            sums, counts = reconstruction_totals(model(masked), masked, features)
            loss = balanced_loss(sums, counts, family_indices)
            if loss is not None:
                require(bool(torch.isfinite(loss)), "nonfinite SSL loss")
                loss.backward()
                optimizer.step()
            sampler.cursor += len(indices)
            progress["step"] += 1
            if sampler.cursor == sampler.size:
                progress["phase"] = "validation"
        else:
            start = progress["validation_cursor"]
            end = min(start + settings.batch_size, len(stopping.original_ids))
            batch = _batch(validation_batch, slice(start, end))
            model.eval()
            with torch.no_grad():
                sums, counts = reconstruction_totals(model(batch), batch, features)
            for j in range(len(features)):
                progress["validation_sums"][j] += float(sums[j])
                progress["validation_counts"][j] += counts[j]
            progress["validation_cursor"] = end
            if end == len(stopping.original_ids):
                score = balanced_loss(progress["validation_sums"], progress["validation_counts"], family_indices)
                require(score is not None and math.isfinite(score), "nonfinite/empty stopping loss")
                progress["history"].append(score)
                if progress["best_loss"] is None or score < progress["best_loss"]:
                    progress["best_loss"], progress["bad_epochs"] = score, 0
                    best_model = deepcopy(model.state_dict())
                else:
                    progress["bad_epochs"] += 1
                progress["epoch"] += 1
                scheduler.step()
                completed = progress["epoch"] >= settings.max_epochs or progress["bad_epochs"] >= settings.patience
                if completed:
                    reason = "max_epochs" if progress["epoch"] >= settings.max_epochs else "patience"
                else:
                    sampler.finish_epoch()
                    progress.update(phase="train", validation_cursor=0,
                                    validation_sums=[0.0] * len(features), validation_counts=[0] * len(features))
        slice_batches += 1
    state = {"model": model.state_dict(), "best_model": best_model,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "rng": capture_rng(), "sampler": sampler.state_dict(), "progress": progress,
             "preprocessing": preprocessing, "fitting_ids": fitting.original_ids,
             "stopping_ids": stopping.original_ids, "controller": controller}
    parents = (view.content_hash, identity.split_hash)
    if config.predecessor:
        parents += (config.predecessor.content_hash,)
    lineage = ArtifactLineage(source_hashes=view.lineage.source_hashes, unit_ids=view.original_ids,
        parent_hashes=parents, split_hash=identity.split_hash, config_hash=identity.config_hash,
        model_hash=_model_hash(model.state_dict()), environment=identity.environment, seed=seed,
        parameter_count=sum(p.numel() for p in model.parameters()))
    return save_checkpoint(root, identity=identity, lineage=lineage, state=state,
                           complete=completed, reason=reason, predecessor=config.predecessor)
