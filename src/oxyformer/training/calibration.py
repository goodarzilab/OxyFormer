"""Weighted affine-logit calibration from independently fitted inner OOF pairs.

Every row represents one original observation and both origin classes. Ownership
is checked on original IDs, before any numerical optimization. Final-refit
transfer statistics are diagnostics only and never update the calibration.
Alignment, ownership, finiteness and nonnegative weights cover every original
row. Weighted arithmetic uses only raw positive-weight rows, without replacing
excluded observations by placeholders. Public logits/ratios remain unweighted.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from oxyformer.provenance import ArtifactLineage, Immutable, require, unique


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationPartition(Immutable):
    fold: int
    evaluation_ids: tuple[str, ...]
    fitting_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    frozen_epochs: int | None

    def validate(self, outer_training_ids):
        evaluation, fitting, stopping = map(set, (
            self.evaluation_ids, self.fitting_ids, self.checkpoint_ids))
        for ids in (self.evaluation_ids, self.fitting_ids, self.checkpoint_ids):
            unique(ids, "calibration partition IDs")
        require(bool(evaluation) and bool(fitting), "empty calibration partition")
        require(evaluation.isdisjoint(fitting), "calibration records entered fitting")
        require(evaluation.isdisjoint(stopping), "calibration records entered checkpoint selection")
        require(fitting.isdisjoint(stopping), "fitting and stopping must be separate")
        require(evaluation | fitting | stopping <= set(outer_training_ids),
                "calibration partition outside outer training")
        require((bool(stopping) and self.frozen_epochs is None) or
                (not stopping and self.frozen_epochs is not None and self.frozen_epochs > 0),
                "declare fitting-only stopping records or independently frozen epochs")


@dataclass(frozen=True, slots=True, kw_only=True)
class AffineCalibration(Immutable):
    slope: float
    intercept: float
    class_prior: float
    original_ids: tuple[str, ...]
    partitions: tuple[CalibrationPartition, ...]
    lineage: ArtifactLineage
    # Store the optimized affine map in normalized coordinates. Expanding it
    # into raw-logit coefficients can overflow or erase small contrasts.
    input_offset: float = 0.
    input_scale: float = 1.

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.input_scale > 0, "invalid calibration input scale")

    def logits(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        require(bool(torch.isfinite(values).all()), "nonfinite calibration logits")
        with torch.autocast(values.device.type, enabled=False):
            if self.slope == 0:
                result = torch.full_like(values, self.intercept)
            else:
                centered = values - self.input_offset
                normalized = centered / self.input_scale
                overflow = ~torch.isfinite(centered)
                if bool(overflow.any()):
                    # Opposite extreme signs can overflow subtraction even
                    # when the difference in normalized units is finite.
                    normalized = torch.where(overflow,
                        values / self.input_scale - self.input_offset / self.input_scale,
                        normalized)
                result = normalized * self.slope + self.intercept
        require(bool(torch.isfinite(result).all()), "nonfinite calibrated logits")
        return result

    def ratios(self, values):
        result = self.logits(values).exp() * ((1 - self.class_prior) / self.class_prior)
        require(bool(torch.isfinite(result).all()), "nonfinite calibrated ratio")
        return result


def paired_tensors(logits, weights, *, allow_zero_mass=False):
    """Validate all originals, then return only their positive-weight pairs.

    Select support before normalizing weights; rounded normalized mass must
    never decide which original logits undergo calibration and ratio audits.
    Callers retain the complete original IDs and provenance separately.
    """
    z = torch.as_tensor(logits, dtype=torch.float32).detach()
    w = torch.as_tensor(weights, dtype=torch.float64, device=z.device).detach()
    require(z.ndim == 2 and z.shape[1] == 2 and w.shape == (len(z),),
            "calibration requires [original, observed/shifted] logits and origin weights")
    require(bool(torch.isfinite(z).all()) and bool(torch.isfinite(w).all()),
            "nonfinite calibration inputs")
    require(bool((w >= 0).all()), "invalid calibration target weights")
    positive = w > 0
    require(allow_zero_mass or bool(positive.any()), "invalid calibration target weights")
    z, w = z[positive], w[positive]
    labels = torch.tensor([0., 1.], device=z.device).expand_as(z)
    # Retain raw mass until reduction. Even FP64 normalization can erase a
    # positive row whose weighted loss or root-mean-square is representable.
    return z, labels, w[:, None].expand_as(z)


def _log_weights(weights):
    return weights.log() - weights.max().log()


def _weighted_mean(values, weights, *, root=False):
    """Reduce nonnegative observations without prematurely rounding mass."""
    if not values.numel():
        return values.new_zeros((), dtype=torch.float64)
    log_weights = _log_weights(weights).flatten()
    log_mean = (torch.logsumexp(values.double().flatten().log() + log_weights, 0)
                - torch.logsumexp(log_weights, 0))
    # A representable RMSE may have an unrepresentable mean square.
    return (log_mean / 2 if root else log_mean).exp()


def _weighted_score(values, weights):
    """Accumulate signed score terms before conversion to FP32 parameters."""
    log_weights = _log_weights(weights).flatten()
    values = values.double().flatten()
    terms = (values.abs().log() + log_weights - torch.logsumexp(log_weights, 0)).exp()
    # fsum preserves cancellation between paired/original signed contributions.
    return math.fsum((terms * values.sign()).tolist())


def pair_metrics(logits, weights):
    # A zero-mass evaluation partition contributes exactly zero to ranking.
    return _pair_metrics(*paired_tensors(logits, weights, allow_zero_mass=True))


def _pair_metrics(z, y, w):
    """Score already validated positive-weight pairs, including empty support."""
    loss = float(_weighted_mean(F.binary_cross_entropy_with_logits(z, y, reduction="none"), w))
    brier = float(_weighted_mean((z.sigmoid() - y).square(), w))
    require(math.isfinite(loss) and math.isfinite(brier), "nonfinite calibration score")
    return loss, brier


def _weighted_median(values, mass):
    """Choose an observed coordinate without averaging away small contrasts."""
    order = values.argsort()
    cumulative = torch.logcumsumexp(_log_weights(mass)[order], 0)
    index = torch.searchsorted(cumulative, cumulative[-1] - math.log(2))
    return values[order[index]]


def fit_affine(logits, weights, *, original_ids, fold_ids, partitions,
               outer_training_ids, lineage):
    """Fit two FP32 parameters; no clipping, sign constraint, or effect input.

    The affine map is standardized internally for conditioning. An exactly
    constant input has the canonical constant map at the balanced class prior.
    Separated inputs may lack a finite MLE; finite optimization is required and
    subsequent ratio overflow is an explicit failure, never implicit clipping.
    """
    ids = tuple(original_ids)
    unique(ids, "calibration original IDs")
    require(len(ids) == len(fold_ids) == len(logits), "calibration ID alignment mismatch")
    require(set(ids) <= set(outer_training_ids), "calibration outside outer training")
    require(set(lineage.unit_ids) == set(ids), "calibration lineage mismatch")
    by_fold = {}
    for partition in partitions:
        partition.validate(outer_training_ids)
        require(partition.fold not in by_fold, "duplicate calibration fold")
        by_fold[partition.fold] = partition
    require(set(fold_ids) == set(by_fold), "calibration fold coverage mismatch")
    for fold, partition in by_fold.items():
        require({oid for oid, f in zip(ids, fold_ids) if f == fold} == set(partition.evaluation_ids),
                "calibration predictions do not match held-out partition")
    z, labels, mass = paired_tensors(logits, weights)
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(z.device.type, enabled=False):
        # Robust coordinates preserve ordinary contrasts when a tiny-weight
        # original has an extreme logit. These statistics only condition the
        # same affine objective; every positive-weight pair remains in it.
        offset = _weighted_median(z.flatten(), mass.flatten())
        # Coordinate statistics may use FP64; the realized map and both
        # optimized parameters remain FP32. Keep the observed offset even if
        # an opposite extreme would overflow FP32 subtraction.
        deviations = (z.double() - offset.double()).abs()
        varying = deviations > 0
        magnitude = (_weighted_median(deviations[varying], mass[varying])
                     if bool(varying.any()) else deviations.new_zeros(()))
        maximum = torch.finfo(z.dtype).max
        magnitude = torch.maximum(magnitude, deviations.max() / (maximum / 2))
        magnitude = magnitude.clamp(max=maximum).float()
        if float(magnitude) == 0:
            slope, intercept, magnitude = 0., 0., torch.ones_like(magnitude)
        else:
            coordinates = AffineCalibration(slope=1., intercept=0., class_prior=.5,
                original_ids=ids, partitions=tuple(partitions), lineage=lineage,
                input_offset=float(offset), input_scale=float(magnitude))
            # Fit exactly the normalization used by the public FP32 map,
            # including its overflow-safe subtraction on opposite extremes.
            x = coordinates.logits(z)
            parameter = torch.zeros(2, dtype=torch.float32, device=z.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([parameter], lr=1., max_iter=100,
                                         tolerance_grad=1e-7, tolerance_change=1e-9,
                                         line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                with torch.no_grad():
                    calibrated = parameter[0] * x + parameter[1]
                    loss = _weighted_mean(F.binary_cross_entropy_with_logits(
                        calibrated, labels, reduction="none"), mass)
                    require(bool(torch.isfinite(loss)), "nonfinite affine objective")
                    residual = (calibrated.sigmoid() - labels).double()
                    # Form scores before reducing raw weights: casting tiny
                    # normalized mass to FP32 first loses large x * mass terms.
                    scores = ((residual * x.double()).mean(1), residual.mean(1))
                    parameter.grad = parameter.new_tensor([
                        _weighted_score(score, mass[:, 0]) for score in scores])
                require(bool(torch.isfinite(parameter.grad).all()), "nonfinite affine gradient")
                return loss

            optimizer.step(closure)
            slope = float(parameter[0].detach())
            intercept = float(parameter[1].detach())
    require(math.isfinite(slope) and math.isfinite(intercept), "nonfinite affine coefficients")
    # A slope below one can bring an overflowing normalized input back into
    # range, but the public FP32 map validates after that intermediate. Move
    # powers of two into BOTH stored coefficients until the slope no longer
    # shrinks the normalized value, or every finite FP32 difference fits in
    # the chosen scale. Binary rescaling preserves the fitted affine map and
    # avoids rounding a subnormal input scale through division by the slope.
    factor = 1.
    while slope != 0 and abs(slope) * factor < 1. and float(magnitude) * factor < 2.:
        factor *= 2.
    slope = float(torch.tensor(slope * factor, dtype=torch.float32))
    magnitude = magnitude.new_tensor(float(magnitude) * factor)
    result = AffineCalibration(slope=slope, intercept=intercept, class_prior=.5,
        original_ids=ids, partitions=tuple(partitions), lineage=lineage,
        input_offset=float(offset), input_scale=float(magnitude))
    _pair_metrics(result.logits(z), labels, mass)
    result.ratios(z)  # Numerical validity includes the actual ratio, not just BCE.
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class TransferDiagnostics(Immutable):
    original_ids: tuple[str, ...]
    metrics: tuple[tuple[str, float], ...]
    evaluation_kind: str
    lineage: ArtifactLineage


def transfer_diagnostics(calibration, oof_logits, refit_logits, weights, *, lineage):
    """Describe transfer on training originals; these are NOT held-out refit scores."""
    z, labels, mass = paired_tensors(oof_logits, weights)
    final, _, _ = paired_tensors(refit_logits, weights)
    require(final.shape == z.shape, "transfer alignment mismatch")
    before = _pair_metrics(calibration.logits(z), labels, mass)
    after = _pair_metrics(calibration.logits(final), labels, mass)
    # Diagnostics can accumulate squared FP32 differences in FP64 without
    # changing the FP32 calibration or ratio computation.
    delta = float(_weighted_mean((final.double() - z.double()).square(), mass, root=True))
    require(math.isfinite(delta), "nonfinite calibration transfer")
    return TransferDiagnostics(original_ids=calibration.original_ids,
        metrics=(("inner_oof_log_loss", before[0]), ("inner_oof_brier", before[1]),
                 ("refit_training_log_loss", after[0]), ("refit_training_brier", after[1]),
                 ("weighted_logit_rmse", delta)),
        evaluation_kind="outer_training_refit_in_sample; diagnostic_only", lineage=lineage)
