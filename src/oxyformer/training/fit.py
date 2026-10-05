"""Fold-local weighted nuisance fitting and label-free prediction.

FitConfig is the runtime binding: DataManifest remains metadata, LoadedData is
privileged and used only by fitting, and PredictionInputs contains only the
query's treatment, policy routing, county routing and original target weights.
FoldArtifacts is an immutable descriptor, never a live trainable model.

Inner calibration evaluations do not select their own epochs. Use either the
prespecified frozen_epochs (default 150), or explicit stopping IDs contained in
that inner fitting partition. Outer refits use the median selected inner epoch
count, frozen before refitting, and every outer-training original. The grid is
ranked by factual loss (origin log loss then Brier), never an estimated effect.
Each calibration fold chooses its own origin grid using only its stopping IDs,
or its fitting loss when epochs were independently frozen. Its evaluation IDs
cannot choose the grid supplying their logits. The final refit grid uses the
pooled inner evaluation scores; transfer diagnostics audit that refitted model.

A new output_dir is required on each invocation/continuation. Interruption saves
nested controller, optimizer, scheduler, sampler, RNG, current and best model
states together. Only complete fold artifacts can predict. Model forwards,
parameters, calibration and ratios are FP32; weighted loss/offset reductions
use FP64 to preserve finite objectives across target-weight scales. Python
float outputs preserve prediction precision for downstream FP64 arithmetic.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from fractions import Fraction
import math
from pathlib import Path
import statistics
import time
from typing import Literal

import torch
from torch.nn import functional as F

from oxyformer.contracts import CovariateView, DataManifest, EstimandSpec, OOFNuisances, SplitManifest
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import LoadedData, validate_split
from oxyformer.design.policies import PolicyCovariates, ShiftOrStayPolicy, paired_records
from oxyformer.design.splits import InnerSplit
from oxyformer.models.county_context import CountyContext
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.likelihoods import CountyOffsets, endpoint_loss
from oxyformer.models.origin import OriginTransformer, paired_origin_loss
from oxyformer.models.outcome import OutcomeTransformer
from oxyformer.models.tokens import FeatureSpec
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ArtifactLineage, Immutable, canonical_json, file_hash, require, write_artifact
from oxyformer.training.calibration import (
    AffineCalibration, CalibrationPartition, fit_affine, paired_tensors, pair_metrics, transfer_diagnostics,
    CALIBRATION_LOGIT_ABS_MAX, CALIBRATION_LOSS_TOLERANCE,
)
from oxyformer.training.checkpoint import (
    CheckpointArtifact, CheckpointIdentity, CheckpointRequest, capture_rng, load_checkpoint,
    model_state_hash, restore_rng, save_checkpoint,
)
from oxyformer.training.pretrain import (
    PretrainConfig, SSLSettings, StatefulSampler, environment_identity, pretrain,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NuisanceSettings(Immutable):
    learning_rates: tuple[float, ...] = (3e-4, 1e-3)
    dropouts: tuple[float, ...] = (0., .10)
    weight_decay: float = 1e-4
    max_epochs: int = 150
    patience: int = 15
    gradient_norm: float = 1.
    batch_size: int = 256
    frozen_epochs: int = 150
    calibration_logit_abs_max: float = CALIBRATION_LOGIT_ABS_MAX
    calibration_loss_tolerance: float = CALIBRATION_LOSS_TOLERANCE

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.learning_rates == (3e-4, 1e-3) and self.dropouts == (0., .10),
                "unregistered nuisance grid")
        require(self.weight_decay == 1e-4 and self.max_epochs == 150 and
                self.patience == 15 and self.gradient_norm == 1., "unregistered optimization bounds")
        require(self.calibration_logit_abs_max == CALIBRATION_LOGIT_ABS_MAX and
                self.calibration_loss_tolerance == CALIBRATION_LOSS_TOLERANCE,
                "unregistered calibration numerical bounds")
        require(self.batch_size > 0 and 0 < self.frozen_epochs <= self.max_epochs,
                "invalid batch size or frozen epoch schedule")

    @property
    def grid(self):
        return tuple((lr, dropout) for lr in self.learning_rates for dropout in self.dropouts)


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionInputs(Immutable):
    """Treatment and routing are separate from the approved-X predictor view."""
    original_ids: tuple[str, ...]
    a_mmhg: tuple[float, ...]
    counties: tuple[str, ...]
    origin_weights: tuple[float, ...]
    policy_covariates: PolicyCovariates

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.original_ids == self.policy_covariates.original_ids, "query policy IDs mismatch")
        require(len(self.original_ids) == len(self.a_mmhg) == len(self.counties) == len(self.origin_weights),
                "query metadata alignment mismatch")
        require(all(self.counties), "empty county route")
        require(all(w >= 0 for w in self.origin_weights),
                "invalid query target weights")


@dataclass(frozen=True, slots=True, kw_only=True)
class FoldArtifacts(Immutable):
    spec: EstimandSpec
    split: SplitManifest
    data_manifest: DataManifest
    fold: int
    seed: int
    prediction_inputs: PredictionInputs
    checkpoint: CheckpointArtifact

    @property
    def complete(self):
        return self.checkpoint.complete


@dataclass(frozen=True, kw_only=True)
class FitConfig:
    """Explicit runtime resources; no data is inferred from source URI strings.

    inner is the authoritative InnerSplit for this outer fold. stopping_ids maps
    inner fold numbers to predeclared, grouped stopping originals. Missing entries
    use frozen_epochs. The same stopping IDs exclude preprocessing, SSL, context,
    gradient updates and offset fitting. SSL has its own grouped, training-local
    stopping split. Runtime budgets and predecessor do not change science identity.
    """
    data: LoadedData
    entity_graph: EntityGraph
    inner: InnerSplit
    fold: int
    policy: ShiftOrStayPolicy
    policy_covariates: PolicyCovariates
    treatment_design: TreatmentDesign
    feature_kinds: tuple[tuple[str, Literal["numeric", "categorical"]], ...]
    families: tuple[tuple[str, ...], ...]
    county_field: str
    exposure_assignment_level: str
    output_dir: str
    family: Literal["identity", "bernoulli", "poisson"] = "identity"
    population_field: str | None = None
    settings: NuisanceSettings = NuisanceSettings()
    stopping_ids: tuple[tuple[int, tuple[str, ...]], ...] = ()
    ssl_epochs: int = 30
    predecessor: FoldArtifacts | None = None
    stop_request: CheckpointRequest | None = None
    max_batches: int | None = None
    slice_seconds: float = 14400.
    checkpoint_margin_seconds: float = 120.

    def __post_init__(self):
        require(type(self.data) is LoadedData and type(self.inner) is InnerSplit,
                "explicit LoadedData and InnerSplit required")
        require(self.family in ("identity", "bernoulli", "poisson"), "unknown endpoint likelihood")
        require((self.family == "poisson") == (self.population_field is not None),
                "Poisson needs a separately declared population exposure")
        require(0 < self.ssl_epochs <= 30, "invalid SSL epoch bound")
        require(Path(self.output_dir).is_absolute(), "absolute attempt output_dir required")
        require(self.max_batches is None or (type(self.max_batches) is int and self.max_batches > 0),
                "invalid batch budget")
        require(math.isfinite(self.slice_seconds) and 0 < self.slice_seconds <= 14400 and
                0 <= self.checkpoint_margin_seconds < self.slice_seconds, "invalid slice bound")
        require(len(dict(self.stopping_ids)) == len(self.stopping_ids), "duplicate stopping fold")


def _hash(value):
    return sha256(canonical_json(value).encode()).hexdigest()


def subset(view, ids, *, use=None, split_hash=None):
    """Copy only permitted scalar covariates; no label-bearing backing object."""
    rows = dict(zip(view.original_ids, view.values))
    return replace(view, original_ids=tuple(ids), values=tuple(rows[x] for x in ids),
        use=use or view.use, lineage=replace(view.lineage, unit_ids=tuple(ids),
            parent_hashes=(view.content_hash,), split_hash=split_hash or view.lineage.split_hash))


def _values(config, name, ids):
    values = dict(zip(config.data.manifest.original_ids, config.data.column(name)))
    return tuple(values[oid] for oid in ids)


def _inputs(config, ids):
    cov = config.policy_covariates
    by_id = {oid: (g, k) for oid, g, k in zip(cov.original_ids, cov.geography_ids, cov.support_keys)}
    routes = dict(zip(config.data.manifest.original_ids, config.data.county_routing(config.county_field)))
    manifest = config.data.manifest
    return PredictionInputs(original_ids=tuple(ids), a_mmhg=_values(config, manifest.exposure_field, ids),
        counties=tuple(routes[oid] for oid in ids),
        origin_weights=_values(config, manifest.weight_field, ids) if manifest.weight_field else (1.,) * len(ids),
        policy_covariates=PolicyCovariates(original_ids=tuple(ids),
            geography_ids=tuple(by_id[x][0] for x in ids), support_keys=tuple(by_id[x][1] for x in ids)))


def _take_inputs(inputs, ids):
    positions = {oid: i for i, oid in enumerate(inputs.original_ids)}
    ix = [positions[oid] for oid in ids]
    return PredictionInputs(original_ids=tuple(ids), a_mmhg=tuple(inputs.a_mmhg[i] for i in ix),
        counties=tuple(inputs.counties[i] for i in ix),
        origin_weights=tuple(inputs.origin_weights[i] for i in ix),
        policy_covariates=PolicyCovariates(original_ids=tuple(ids),
            geography_ids=tuple(inputs.policy_covariates.geography_ids[i] for i in ix),
            support_keys=tuple(inputs.policy_covariates.support_keys[i] for i in ix)))


def _partition(config, parent, fold, *, stopping=()):
    """Derive a fitting-only manifest; sealed/excluded IDs retain their status."""
    stopping = tuple(stopping)
    allowed = parent.training_ids(fold)
    require(len(set(stopping)) == len(stopping) and set(stopping) < set(allowed),
            "stopping records must be inside the fitting partition")
    fitting = tuple(oid for oid in allowed if oid not in set(stopping))
    held = tuple(oid for oid, f in zip(parent.original_ids, parent.fold_ids) if f == fold)
    require(set(held).isdisjoint(stopping), "calibration records entered checkpoint selection")
    # The original geographic folds and whole entity links remain intact. Only
    # fitting originals can be assigned to the extra stopping partition.
    graph = config.entity_graph if parent.level == "outer" else config.inner.entity_graph
    assignments = {oid: ("fit" if oid in fitting else "stop" if oid in stopping else "eval")
                   for oid in parent.original_ids}
    assignments.update((oid, "sealed") for oid in parent.design_ids)
    assignments.update((oid, "excluded") for oid in parent.excluded_ids)
    graph.assert_partition(assignments)
    derived = replace(parent, fold_ids=tuple(1 if oid in fitting else 0 for oid in parent.original_ids),
        lineage=replace(parent.lineage, parent_hashes=(parent.content_hash,)))
    return derived, fitting, held


def _raw(batch, features):
    pieces = []
    for j, feature in enumerate(features):
        if feature.kind == "numeric":
            pieces.append(batch.numeric_values[:, j:j + 1])
        else:
            category = F.one_hot(batch.categorical_values[:, j], len(feature.categories) + 1).float()
            pieces.append(category.masked_fill(batch.missing[:, j:j + 1], 0))
        pieces.append(batch.missing[:, j:j + 1].float())
    result = torch.cat(pieces, dim=1)
    require(bool(torch.isfinite(result).all()), "nonfinite raw covariate bypass")
    return result


def _build(bundle, *, encoder_state=None):
    features = tuple(FeatureSpec.from_json(value) for value in bundle["preprocessing"])
    encoder = FeatureEncoder(features, dropout=bundle["dropout"]).float()
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    split = SplitManifest.from_json(bundle["split"])
    references = CovariateView.from_json(bundle["references"])
    context = CountyContext(encoder, references, split, 0, tuple(bundle["counties"]),
        county_field=bundle["county_field"], checkpoint_hash=bundle["ssl_hash"], dropout=bundle["dropout"])
    family = bundle["family"] if bundle["kind"] == "outcome" else "bernoulli"
    offsets = CountyOffsets(split, 0, tuple(bundle["counties"]), family=family,
                            exposure_assignment_level=bundle["exposure_assignment_level"])
    width = sum((1 if f.kind == "numeric" else len(f.categories) + 1) + 1 for f in features)
    model_type = OutcomeTransformer if bundle["kind"] == "outcome" else OriginTransformer
    kwargs = {"family": family} if bundle["kind"] == "outcome" else {}
    model = model_type(encoder, treatment_design=TreatmentDesign.from_json(bundle["treatment_design"]),
        raw_x_dim=width, county_context=context, group_offsets=offsets,
        dropout=bundle["dropout"], **kwargs).float()
    if "state" in bundle:
        model.load_state_dict(bundle["state"], strict=True)
    return model


def _predict(model, view, inputs, policy, *, outcome_mean=False, base_only=False):
    require(type(view) is CovariateView and view.use == "nuisance", "label-free nuisance CovariateView required")
    require(view.original_ids == inputs.original_ids, "prediction alignment mismatch")
    result = policy.apply(inputs.a_mmhg, inputs.policy_covariates)
    query = torch.tensor(tuple(zip(result.a_mmhg, result.d_mmhg)), dtype=torch.float32).unsqueeze(-1)
    batch = model.encoder.tokenizer.prepare(view)
    raw = _raw(batch, model.encoder.tokenizer.features)
    context = model.county_context(inputs.original_ids, inputs.counties)
    offset = torch.zeros(len(view.original_ids)) if base_only else model.group_offsets(inputs.counties)
    if isinstance(model, OutcomeTransformer):
        method = model.mean if outcome_mean else model.linear_predictor
    else:
        method = model.logits
    prediction = method(query, batch, raw, context, offset)
    require(bool(torch.isfinite(prediction).all()), "nonfinite nuisance predictions")
    return prediction


def _weight_unit(weights):
    """Choose arithmetic units with room for FP64 products and reciprocals.

    Original weights remain in inputs/pair records. A common positive factor
    cancels from a weighted objective; it must cancel before autograd would
    have to represent an overflowing reciprocal of the original total mass.
    """
    largest = max(weights, default=0.)
    limits = torch.finfo(torch.float64)
    return largest if (0 < largest < math.sqrt(limits.tiny) or
                       largest > math.sqrt(limits.max)) else 1.


def _clip_gradient_norm(parameters, maximum, *, scale=1.):
    """Apply the registered cap without overflowing an FP32 norm reduction."""
    with torch.no_grad():
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            return
        norm = torch.stack([g.double().norm() for g in gradients]).norm()
        require(bool(torch.isfinite(norm)), "nonfinite nuisance gradients")
        if norm == 0:
            return
        # For g_scaled = scale * g, this is exactly the registered rule
        # g * min(1, maximum / (norm(g) + 1e-6)). Do not materialize an
        # overflowing unscaled FP32 gradient, and scale the stabilizer too.
        factor = torch.minimum(maximum / (norm + 1e-6 * scale),
                               norm.new_tensor(scale).reciprocal())
        for gradient in gradients:
            gradient.copy_(gradient.double() * factor)


def _backward_and_clip(loss, parameters, maximum):
    """Retry backward on the same graph before committing a minibatch.

    A finite FP64 loss can have derivatives exceeding FP32 storage before
    clipping. Power-of-two scaling moves these derivatives into range. Keep
    the forward graph (including its dropout masks); retries neither consume
    RNG nor advance optimizer, scheduler, sampler, or checkpoint progress.
    """
    parameters = tuple(parameters)
    # Span the FP64 exponent range in bounded steps. Ordinary batches use
    # scale=1 and one backward pass; parameters/gradient storage stay FP32.
    for exponent in range(0, 1025, 32):
        scale = math.ldexp(1., -exponent)
        for parameter in parameters:
            parameter.grad = None
        loss.backward(gradient=loss.new_tensor(scale), retain_graph=True)
        if all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters):
            _clip_gradient_norm(parameters, maximum, scale=scale)
            return scale
    require(False, "nonfinite nuisance gradients after scaled backward")


def _pooled_metrics(results):
    """Order represented fold metrics without rounding away positive mass.

    Reporting floats cannot distinguish every weighted score. Integer ratios
    retain exact raw mass through checkpoints; Fraction compares the existing
    finite metric values under the same weighted objective, without a cutoff
    or tie tolerance. The caller keeps its loss/Brier/grid-index ordering.
    """
    masses = [Fraction(*result["mass_ratio"]) for result in results]
    total = sum(masses)
    require(total > 0, "inner evaluation pool has no target mass")
    return tuple(sum(Fraction(r["metrics"][j]) * mass for r, mass in zip(results, masses)) / total
                 for j in range(len(results[0]["metrics"])))


def _unit_products(values, weights, unit, multiplier=None):
    """Form signed products before a positive weight can underflow alone."""
    exponent = values.abs().log() + weights.log() - math.log(unit)
    sign = values.sign()
    if multiplier is not None:
        exponent = exponent + multiplier.abs().log()
        sign = sign * multiplier.sign()
    return exponent.exp() * sign


class _LossInUnits(torch.autograd.Function):
    """Carry a local likelihood derivative through extreme weight units.

    The authoritative likelihood supplies both per-row losses and derivatives.
    Combining its derivative with raw mass before division avoids an underflowed
    weight becoming zero before a large residual can contribute to backward.
    """
    @staticmethod
    def forward(ctx, prediction, losses, derivatives, weights, unit):
        ctx.save_for_backward(derivatives, weights)
        ctx.unit = unit
        return _unit_products(losses, weights, unit)

    @staticmethod
    def backward(ctx, gradient):
        derivatives, weights = ctx.saved_tensors
        return _unit_products(derivatives, weights, ctx.unit, gradient), None, None, None, None


def _loss_in_units(prediction, weights, unit, evaluate, reduction):
    scaled = weights / unit
    # Keep ordinary arithmetic and the merged likelihood's normal autograd
    # path. Use the wider product representation when normalized mass becomes
    # subnormal or zero, before this rounding can alter a row's contribution.
    if not bool(((weights > 0) & (scaled < torch.finfo(weights.dtype).tiny)).any()):
        return evaluate(prediction, scaled, reduction)
    with torch.enable_grad():
        local = prediction.detach().requires_grad_(True)
        # Original weights were validated first. Indicator weights expose the
        # merged per-row likelihood while retaining its zero-row protection.
        losses = evaluate(local, (weights > 0).to(weights.dtype), "none")
        derivatives, = torch.autograd.grad(losses.sum(), local)
    weighted = _LossInUnits.apply(prediction, losses.detach(), derivatives, weights, unit)
    if reduction == "none":
        return weighted
    total = weighted.sum()
    mass = scaled.sum()
    return total / torch.where(mass > 0, mass, torch.ones_like(mass)) if reduction == "mean" else total


def _outcome_loss(config, prediction, ids, *, reduction="sum", weight_unit=1.):
    # Accumulate target-weighted losses before normalization in FP64. Model
    # predictions and target validity remain governed by their FP32 contract.
    prediction = prediction.double()
    weights = torch.tensor(_inputs(config, ids).origin_weights, dtype=torch.float64)
    if reduction == "mean":
        weight_unit = _weight_unit(weights.tolist())
    target = torch.tensor(_values(config, config.data.manifest.outcome_field, ids), dtype=torch.float32)
    population = (torch.tensor(_values(config, config.population_field, ids), dtype=torch.float32)
                  if config.population_field else None)
    return _loss_in_units(prediction, weights, weight_unit,
        lambda p, w, r: endpoint_loss(p, target, w, family=config.family,
                                      population=population, reduction=r), reduction)


def _origin_loss(prediction, pairs, weight_unit):
    weights = torch.tensor(pairs.origin_weights, dtype=torch.float64)
    scaled = weights / weight_unit
    if bool(((weights > 0) & (scaled < torch.finfo(weights.dtype).tiny)).any()):
        # PolicyPairs order is all observed, then all shifted. Preserve raw
        # pairs and weights; only the local likelihood's arithmetic is scaled.
        prediction = prediction.transpose(0, 1).reshape(-1)
        return _loss_in_units(prediction, weights, weight_unit,
            lambda p, w, r: paired_origin_loss(p, replace(pairs, origin_weights=tuple(w.tolist())),
                                               reduction=r), "sum")
    numerical_pairs = (pairs if weight_unit == 1. else replace(pairs,
        origin_weights=tuple(w / weight_unit for w in pairs.origin_weights)))
    return paired_origin_loss(prediction, numerical_pairs, reduction="sum")


def _profile(model, config, view, inputs):
    if isinstance(model, OutcomeTransformer) and config.family == "identity":
        model.eval()
        with torch.no_grad():
            base = _predict(model, view, inputs, config.policy, base_only=True)[:, 0]
            target = torch.tensor(_values(config, config.data.manifest.outcome_field, view.original_ids),
                                  dtype=torch.float32)
            # Each offset profiles its own weighted mean. Use units local to
            # that county so a different county's mass cannot erase its support.
            by_county = {}
            for county, weight in zip(inputs.counties, inputs.origin_weights):
                by_county.setdefault(county, []).append(weight)
            units = {county: _weight_unit(weights) for county, weights in by_county.items()}
            weights = torch.tensor([w / units[c] for c, w in
                                    zip(inputs.counties, inputs.origin_weights)], dtype=torch.float64)
            model.group_offsets.update_identity(view.original_ids, target.double(), base.double(), weights)


class _Budget:
    def __init__(self, config, request):
        self.config, self.request, self.started, self.batches = config, request, time.monotonic(), 0

    def reason(self):
        if self.request.requested:
            return "requested"
        if time.monotonic() - self.started >= self.config.slice_seconds - self.config.checkpoint_margin_seconds:
            return "slice_limit"
        if self.config.max_batches is not None and self.batches >= self.config.max_batches:
            return "batch_limit"
        return None


def _train_one(config, bundle, encoder_state, view, stopping, epochs, seed, budget, saved=None):
    """One transaction per original-record minibatch, including both copies."""
    torch.manual_seed(seed)
    model = _build(bundle, encoder_state=encoder_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=bundle["learning_rate"],
                                 weight_decay=config.settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)
    inputs = _inputs(config, view.original_ids)
    require(sum(inputs.origin_weights) > 0, "fitting partition has no target mass")
    weight_unit = _weight_unit(inputs.origin_weights)
    original_mass = sum(w / weight_unit for w in inputs.origin_weights)
    sampler = StatefulSampler(len(view.original_ids), seed)
    progress = dict(epoch=0, step=0, phase="start", best_loss=None, best_epoch=0, bad_epochs=0, history=[])
    best = None
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampler.load_state_dict(saved["sampler"])
        progress, best = saved["progress"], saved["best"]
        restore_rng(saved["rng"])
    complete, reason = False, None
    while not complete:
        reason = budget.reason()
        if reason:
            break
        if progress["phase"] == "start":
            _profile(model, config, view, inputs)
            progress["phase"] = "train"
        if progress["phase"] == "train":
            indices = sampler.indices(config.settings.batch_size)
            ids = tuple(view.original_ids[i] for i in indices)
            part = subset(view, ids)
            metadata = _inputs(config, ids)
            if sum(metadata.origin_weights) == 0:
                # A sampled zero-mass group has no objective and no decay update.
                sampler.cursor += len(indices)
                progress["step"] += 1
                budget.batches += 1
                if sampler.cursor == sampler.size:
                    progress["phase"] = "evaluate"
                continue
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted = _predict(model, part, metadata, config.policy)
            if bundle["kind"] == "outcome":
                loss = _outcome_loss(config, predicted[:, 0], ids, weight_unit=weight_unit)
                mass = original_mass
            else:
                pairs = paired_records(config.policy.apply(metadata.a_mmhg, metadata.policy_covariates),
                                       metadata.origin_weights, weight_id=config.data.manifest.spec.weight_id)
                # The authoritative records retain raw origin weights. Only
                # the numerical loss receives weights in common arithmetic
                # units, using the SAME unit as the full fitting denominator.
                loss = _origin_loss(predicted.double(), pairs, weight_unit)
                mass = 2 * original_mass
            # Uniform sampling of ORIGINALS estimates the global weighted loss.
            # Dividing each batch by its own mass would optimize a different law.
            loss = loss * (len(view.original_ids) / (len(ids) * mass))
            require(bool(torch.isfinite(loss)), "nonfinite nuisance loss")
            _backward_and_clip(loss, model.parameters(), config.settings.gradient_norm)
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                    "nonfinite nuisance parameters")
            sampler.cursor += len(indices)
            progress["step"] += 1
            budget.batches += 1
            if sampler.cursor == sampler.size:
                progress["phase"] = "evaluate"
            continue
        _profile(model, config, view, inputs)
        model.eval()
        score = None
        # A zero-mass stopping set has no checkpoint-selection information;
        # retain the final epoch of the already-declared training bound.
        if stopping and sum(_inputs(config, stopping).origin_weights) > 0:
            stop_view = subset(config.data.covariates(view.columns), stopping)
            stop_inputs = _inputs(config, stopping)
            with torch.no_grad():
                predicted = _predict(model, stop_view, stop_inputs, config.policy)
                score = (float(_outcome_loss(config, predicted[:, 0], stopping, reduction="mean"))
                         if bundle["kind"] == "outcome" else pair_metrics(predicted, stop_inputs.origin_weights)[0])
            require(math.isfinite(score), "nonfinite stopping loss")
        progress["epoch"] += 1
        progress["history"].append(score)
        if score is None or progress["best_loss"] is None or score < progress["best_loss"]:
            progress.update(best_loss=score, best_epoch=progress["epoch"], bad_epochs=0)
            best = deepcopy(model.state_dict())
        else:
            progress["bad_epochs"] += 1
        scheduler.step()
        complete = progress["epoch"] >= epochs or progress["bad_epochs"] >= config.settings.patience
        if complete:
            reason = "max_epochs" if progress["epoch"] >= epochs else "patience"
        else:
            sampler.finish_epoch()
            progress["phase"] = "start"
    state = dict(model=deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), rng=capture_rng(),
        progress=progress, best=best)
    if complete:
        model.load_state_dict(best, strict=True)
        model.eval()
    return model, state, complete, reason


def _validate(spec, split, manifest, config, seed):
    require(type(config) is FitConfig and type(manifest) is DataManifest, "FitConfig and DataManifest required")
    spec.assert_compatible(manifest.spec)
    require(config.data.manifest == manifest, "training data manifest mismatch")
    validate_split(split, manifest, config.entity_graph)
    require(split.level == "outer" and len(set(split.fold_ids)) == 5, "five outer folds required")
    require(seed in (1103, 2207, 3301) and seed in split.seed_ids, "unregistered nuisance seed")
    require(config.inner.outer_fold == config.fold, "wrong inner parent fold")
    validate_split(config.inner.split, config.inner.data_manifest, config.inner.entity_graph)
    spec.assert_compatible(config.inner.split.spec)
    require(config.inner.split.level == "inner" and set(config.inner.split.fold_ids) == {0, 1, 2},
            "three inner geographic folds required")
    allowed = split.training_ids(config.fold)
    require(set(config.inner.data_manifest.original_ids) == set(allowed), "inner split escapes outer training")
    for field in ("schema", "registry", "sources", "id_field", "outcome_field", "exposure_field", "weight_field"):
        require(getattr(config.inner.data_manifest, field) == getattr(manifest, field), "inner data identity mismatch")
    require(config.policy.policy_id == spec.policy_id, "policy mismatch")
    require(set(config.policy_covariates.original_ids) == set(manifest.original_ids), "policy metadata coverage mismatch")
    require(set(dict(config.stopping_ids)) <= {0, 1, 2}, "unknown stopping fold")
    require(bool(config.feature_kinds) and len(dict(config.feature_kinds)) == len(config.feature_kinds),
            "invalid feature configuration")
    for name, _ in config.feature_kinds:
        for use in ("nuisance", "ssl", "context"):
            manifest.registry.require(name, spec.endpoint, use)
    # Construction checks every prediction-metadata value without accessing Y.
    _inputs(config, split.original_ids)
    for name in (manifest.outcome_field,) + ((config.population_field,) if config.population_field else ()):
        require(all(type(x) in (int, float) and math.isfinite(x) for x in _values(config, name, allowed)),
                "missing/nonfinite target or likelihood exposure in outer training")
    return allowed


def _science_identity(config, split, manifest, seed, allowed):
    data_rows = dict(zip(manifest.original_ids, config.data.rows))
    data_hash = _hash([manifest.content_hash, [(oid, data_rows[oid]) for oid in allowed]])
    science = dict(settings=config.settings.to_dict(), family=config.family,
        population_field=config.population_field, fold=config.fold, inner=config.inner.to_dict(),
        features=config.feature_kinds, families=config.families, county_field=config.county_field,
        exposure_assignment_level=config.exposure_assignment_level,
        policy=config.policy.to_dict(), treatment_design=config.treatment_design.to_dict(),
        stopping_ids=config.stopping_ids, ssl_epochs=config.ssl_epochs,
        policy_training_inputs=_inputs(config, allowed).to_dict())
    # Hash the executed package source, never generated outputs or path names.
    package = Path(__file__).resolve().parents[1]
    code = _hash([(str(path.relative_to(package)), file_hash(path))
                  for path in sorted(package.rglob("*.py"))])
    return CheckpointIdentity(training_ids=allowed, data_hash=data_hash, split_hash=split.content_hash,
        config_hash=_hash(science), preprocessing_hash=_hash([config.feature_kinds, config.families]),
        seed=seed, scientific_code_hash=code, environment=environment_identity(torch.device("cpu")))


@contextmanager
def _numerics():
    caller = capture_rng()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"), torch.inference_mode(False), torch.enable_grad(), torch.autocast("cpu", enabled=False):
            yield
    finally:
        torch.set_default_dtype(dtype)
        restore_rng(caller)


def _ssl(config, split, fitting, view, root, seed, budget, predecessor):
    groups = [g for g in config.entity_graph.components() if set(g) <= set(fitting)]
    require(len(groups) >= 2, "SSL needs two independent fitting components")
    # A deterministic component draw, shared by all later transformed copies.
    stop = min(groups, key=lambda group: _hash([seed, group, "ssl-stopping"]))
    settings = SSLSettings(fold=0, feature_kinds=config.feature_kinds, families=config.families,
                          max_epochs=config.ssl_epochs, stopping_ids=stop)
    remaining = (None if config.max_batches is None else config.max_batches - budget.batches)
    seconds = max(.01, config.slice_seconds - (time.monotonic() - budget.started))
    ssl_config = PretrainConfig(settings=settings, output_dir=str(root),
        predecessor=predecessor, stop_request=budget.request, max_batches=remaining,
        slice_seconds=seconds, checkpoint_margin_seconds=min(config.checkpoint_margin_seconds, seconds / 2))
    artifact = pretrain(subset(view, fitting, use="ssl"), split, ssl_config, seed)
    # SSL validation transactions are also charged to the enclosing slice.
    state = load_checkpoint(artifact, artifact.identity)
    before = load_checkpoint(predecessor, predecessor.identity) if predecessor else None
    def transactions(saved):
        if saved is None:
            return 0
        progress = saved["progress"]
        size = len(saved["stopping_ids"])
        count = progress["step"] + progress["epoch"] * math.ceil(size / settings.batch_size)
        if progress["phase"] == "validation" and progress["validation_cursor"] < size:
            count += math.ceil(progress["validation_cursor"] / settings.batch_size)
        return count
    budget.batches += transactions(state) - transactions(before)
    return artifact


def _bundle(config, split, fitting, initialization, kind, hyperparameters):
    require(initialization.complete and set(initialization.identity.training_ids) == set(fitting),
            "complete partition-owned SSL initialization required")
    require(initialization.identity.split_hash == split.content_hash, "SSL split identity mismatch")
    state = load_checkpoint(initialization, initialization.identity)
    encoder_state = {k.removeprefix("encoder."): v for k, v in state["best_model"].items()
                     if k.startswith("encoder.")}
    columns = tuple(name for name, _ in config.feature_kinds)
    refs = subset(config.data.covariates(columns, use="context"), fitting, split_hash=split.content_hash)
    lr, dropout = hyperparameters
    bundle = dict(preprocessing=state["preprocessing"], split=split.to_json(), references=refs.to_json(),
        counties=_inputs(config, fitting).counties, county_field=config.county_field,
        exposure_assignment_level=config.exposure_assignment_level, family=config.family,
        kind=kind, treatment_design=config.treatment_design.to_json(),
        ssl_hash=initialization.content_hash, learning_rate=lr, dropout=dropout)
    return bundle, encoder_state


def _tensor_state(controller):
    """The generic checkpoint model hash binds tensors; archive hash binds extras.

    Native model state_dicts also contain immutable treatment/offset metadata.
    Keep these exact extra states in controller; never drop or pickle them.
    """
    tensors = {}
    active = controller.get("active")
    if active is not None:
        tensors.update(("active." + name, value) for name, value in active["model"].items()
                       if isinstance(value, torch.Tensor))
    for kind, bundle in controller["final"].items():
        tensors.update((kind + "." + name, value) for name, value in bundle["state"].items()
                       if isinstance(value, torch.Tensor))
    return tensors or {"controller_position": torch.tensor(controller["position"], dtype=torch.int64)}


def _lineage(manifest, identity, ids, parents, *, model_hash=None, parameter_count=None):
    return ArtifactLineage(source_hashes=manifest.lineage.source_hashes, unit_ids=tuple(ids),
        parent_hashes=tuple(parents), split_hash=identity.split_hash, config_hash=identity.config_hash,
        model_hash=model_hash, environment=identity.environment, seed=identity.seed,
        parameter_count=parameter_count)


def fit_fold(spec, split, data_manifest, model_config, seed) -> FoldArtifacts:
    """Fit or continue one fold, using only outer-training labels and covariates."""
    config = model_config
    allowed = _validate(spec, split, data_manifest, config, seed)
    with _numerics():
        identity = _science_identity(config, split, data_manifest, seed, allowed)
        controller = dict(position=0, initializations={}, ssl_pending=None, active=None,
                          results=[], final={}, selection=None, calibration=None)
        if config.predecessor is not None:
            require(config.predecessor.spec == spec and config.predecessor.fold == config.fold and
                    config.predecessor.seed == seed, "continuation fold mismatch")
            require(not config.predecessor.complete, "fold is already complete")
            controller = load_checkpoint(config.predecessor.checkpoint, identity)["controller"]
        root = Path(config.output_dir).resolve() / "nuisance"
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=False)
        request = config.stop_request or CheckpointRequest()
        budget = _Budget(config, request)
        with request.signals():
            controller, complete, reason = _fit_controller(config, split, data_manifest, identity,
                                                          controller, root, budget)
        tensors = _tensor_state(controller)
        progress = dict(epoch=controller["position"], step=controller["position"])
        lineage = _lineage(data_manifest, identity, allowed,
            (data_manifest.content_hash, split.content_hash) +
            ((config.predecessor.checkpoint.content_hash,) if config.predecessor else ()),
            model_hash=model_state_hash(tensors), parameter_count=controller.get("parameter_count", 0))
        checkpoint = save_checkpoint(root, identity=identity, lineage=lineage,
            state=dict(model=tensors, controller=controller, progress=progress), complete=complete,
            reason=reason, predecessor=config.predecessor.checkpoint if config.predecessor else None)
        held = tuple(oid for oid, fold in zip(split.original_ids, split.fold_ids) if fold == config.fold)
        artifact = FoldArtifacts(spec=spec, split=split, data_manifest=data_manifest, fold=config.fold,
            seed=seed, prediction_inputs=_inputs(config, held), checkpoint=checkpoint)
        write_artifact(root / "fold.json", artifact)
        return artifact


def _fit_controller(config, outer, manifest, identity, controller, root, budget):
    columns = tuple(name for name, _ in config.feature_kinds)
    all_view = config.data.covariates(columns)
    outer_ids = outer.training_ids(config.fold)
    tasks = []
    partitions = {}
    audits = {}
    for fold in range(3):
        stopping = tuple(dict(config.stopping_ids).get(fold, ()))
        local, fitting, held = _partition(config, config.inner.split, fold, stopping=stopping)
        audit = CalibrationPartition(fold=fold, evaluation_ids=held, fitting_ids=fitting,
            checkpoint_ids=stopping, frozen_epochs=None if stopping else config.settings.frozen_epochs)
        audit.validate(outer_ids)
        partitions[fold] = (local, fitting, held, stopping)
        audits[fold] = audit
        tasks.append(("ssl", fold, None))
        tasks.extend((kind, fold, grid) for kind in ("outcome", "origin") for grid in range(4))
    outer_local, fitting, held = _partition(config, outer, config.fold)
    partitions[3] = (outer_local, fitting, held, ())
    tasks.extend((("select", 3, None), ("ssl", 3, None), ("outcome", 3, None), ("origin", 3, None)))
    while controller["position"] < len(tasks):
        reason = budget.reason()
        if reason:
            return controller, False, reason
        kind, fold, grid = tasks[controller["position"]]
        local, fitting, held, stopping = partitions[fold]
        if kind == "select":
            selected = {}
            for model_kind in ("outcome", "origin"):
                candidates = []
                for g in range(4):
                    values = [r for r in controller["results"] if r["kind"] == model_kind and r["grid"] == g]
                    scores = _pooled_metrics(values)
                    candidates.append((scores, g))
                chosen = min(candidates)[1]
                values = [r for r in controller["results"] if r["kind"] == model_kind and r["grid"] == chosen]
                selected[model_kind] = dict(grid=chosen, epochs=int(statistics.median(r["epochs"] for r in values)))
            # Each OOF classifier is chosen without consulting its own
            # evaluation scores, including indirectly through other folds.
            origin = [min((r for r in controller["results"]
                           if r["kind"] == "origin" and r["fold"] == fold),
                          key=lambda r: (r["selection_metrics"], r["grid"]))
                      for fold in range(3)]
            selected["origin"]["calibration_grids"] = tuple(r["grid"] for r in origin)
            ids = tuple(oid for r in origin for oid in r["ids"])
            logits = [z for r in origin for z in r["predictions"]]
            weights = _inputs(config, ids).origin_weights
            calibration = fit_affine(logits, weights, original_ids=ids,
                fold_ids=tuple(r["fold"] for r in origin for _ in r["ids"]),
                partitions=tuple(audits.values()), outer_training_ids=outer_ids,
                lineage=_lineage(manifest, identity, ids, (manifest.content_hash, config.inner.content_hash)))
            controller.update(selection=selected, calibration=calibration.to_json(), oof_logits=logits)
        elif kind == "ssl":
            pending = (CheckpointArtifact.from_json(controller["ssl_pending"])
                       if controller["ssl_pending"] else None)
            artifact = _ssl(config, local, fitting, all_view, root / f"ssl-{fold}",
                            identity.seed, budget, pending)
            if not artifact.complete:
                controller["ssl_pending"] = artifact.to_json()
                return controller, False, artifact.reason
            controller["initializations"][fold] = artifact.to_json()
            controller["ssl_pending"] = None
        else:
            if fold == 3:
                grid = controller["selection"][kind]["grid"]
                epochs = controller["selection"][kind]["epochs"]
            else:
                epochs = config.settings.max_epochs if stopping else config.settings.frozen_epochs
            initialization = CheckpointArtifact.from_json(controller["initializations"][fold])
            bundle, encoder_state = _bundle(config, local, fitting, initialization, kind, config.settings.grid[grid])
            view = subset(all_view, fitting)
            model, state, complete, reason = _train_one(config, bundle, encoder_state, view, stopping,
                epochs, identity.seed, budget, controller["active"])
            controller["parameter_count"] = sum(p.numel() for p in model.parameters())
            if not complete:
                controller["active"] = state
                return controller, False, reason
            controller["active"] = None
            if fold == 3:
                with torch.no_grad():
                    _predict(model, view, _inputs(config, fitting), config.policy,
                             outcome_mean=kind == "outcome")
                bundle["state"] = deepcopy(model.state_dict())
                controller["final"][kind] = bundle
            else:
                metadata = _inputs(config, held)
                with torch.no_grad():
                    predicted = _predict(model, subset(all_view, held), metadata, config.policy)
                    metrics = ((float(_outcome_loss(config, predicted[:, 0], held, reduction="mean")),)
                               if kind == "outcome" else pair_metrics(predicted, metadata.origin_weights))
                require(all(math.isfinite(m) for m in metrics), "nonfinite held-out factual loss")
                mass_unit = _weight_unit(metadata.origin_weights)
                exact_mass = sum(map(Fraction, metadata.origin_weights), Fraction())
                result = dict(kind=kind, fold=fold, grid=grid, ids=held,
                    predictions=predicted.tolist(), metrics=metrics,
                    mass=sum(w / mass_unit for w in metadata.origin_weights), mass_unit=mass_unit,
                    mass_ratio=(exact_mass.numerator, exact_mass.denominator),
                    epochs=state["progress"]["best_epoch"], ownership=audits[fold].to_json())
                if kind == "origin":
                    # With frozen epochs, an in-sample fitting score may choose
                    # a grid, but never a calibration evaluation score. Explicit
                    # stopping partitions provide an independent tuning score.
                    selection_ids = stopping or fitting
                    require(set(selection_ids).isdisjoint(held),
                            "calibration records entered grid selection")
                    selection_inputs = _inputs(config, selection_ids)
                    with torch.no_grad():
                        selection_logits = _predict(model, subset(all_view, selection_ids),
                                                    selection_inputs, config.policy)
                    result.update(selection_ids=selection_ids,
                        selection_metrics=pair_metrics(selection_logits, selection_inputs.origin_weights))
                controller["results"].append(result)
        controller["position"] += 1
    calibration = AffineCalibration.from_json(controller["calibration"])
    model = _build(controller["final"]["origin"]).eval()
    metadata = _inputs(config, calibration.original_ids)
    with torch.no_grad():
        refit = _predict(model, subset(all_view, calibration.original_ids), metadata, config.policy)
    positive_refit, _, _ = paired_tensors(refit, metadata.origin_weights)
    calibration.ratios(positive_refit)
    diagnostics = transfer_diagnostics(calibration, controller["oof_logits"], refit,
        metadata.origin_weights, lineage=_lineage(manifest, identity, calibration.original_ids,
            (calibration.content_hash, model_state_hash(_tensor_state(controller)))))
    controller["transfer_diagnostics"] = diagnostics.to_json()
    controller["parameter_count"] = sum(sum(p.numel() for p in _build(bundle).parameters())
                                          for bundle in controller["final"].values())
    write_artifact(root / "calibration-transfer.json", diagnostics)
    write_artifact(root / "calibration.json", calibration)
    return controller, True, "max_epochs"


def predict_fold(artifacts, covariate_view, policy) -> OOFNuisances:
    """Only immutable, label-free X reaches a reconstructed frozen model."""
    require(type(artifacts) is FoldArtifacts and artifacts.complete, "unfinished fold cannot predict")
    require(type(covariate_view) is CovariateView and covariate_view.use == "nuisance",
            "label-free nuisance CovariateView required")
    artifacts.spec.assert_compatible(covariate_view.spec)
    require(type(policy) is ShiftOrStayPolicy and policy.policy_id == artifacts.spec.policy_id, "policy mismatch")
    require(set(covariate_view.original_ids) == set(artifacts.prediction_inputs.original_ids),
            "prediction view must contain exactly this outer held-out fold")
    inputs = _take_inputs(artifacts.prediction_inputs, covariate_view.original_ids)
    with _numerics(), torch.no_grad():
        controller = load_checkpoint(artifacts.checkpoint, artifacts.checkpoint.identity)["controller"]
        outcome = _build(controller["final"]["outcome"]).eval()
        mu = _predict(outcome, covariate_view, inputs, policy, outcome_mean=True).double()
        calibration = AffineCalibration.from_json(controller["calibration"])
        # Apply the known identity before evaluating an unnecessary classifier
        # or exponential; neither can improve an exactly known unit ratio.
        if policy.is_identity:
            ratio = torch.ones_like(mu)
        else:
            origin = _build(controller["final"]["origin"]).eval()
            logits = _predict(origin, covariate_view, inputs, policy)
            ratio = calibration.ratios(logits).double()
        ids = covariate_view.original_ids
        lineage = replace(artifacts.checkpoint.lineage, unit_ids=ids,
            parent_hashes=(artifacts.data_manifest.content_hash, artifacts.checkpoint.content_hash,
                           covariate_view.content_hash, calibration.content_hash))
        return OOFNuisances(spec=artifacts.spec, original_ids=ids, fold_ids=(artifacts.fold,) * len(ids),
            seed_ids=(artifacts.seed,) * len(ids), mu_a=tuple(mu[:, 0].tolist()), mu_d=tuple(mu[:, 1].tolist()),
            r_a=tuple(ratio[:, 0].tolist()), r_d=tuple(ratio[:, 1].tolist()),
            origin_weights=inputs.origin_weights, lineage=lineage)
