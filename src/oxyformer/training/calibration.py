"""Weighted affine-logit calibration from independently fitted inner OOF pairs.

Every row represents one original observation and both origin classes. Ownership
is checked on original IDs, before any numerical optimization. Final-refit
transfer statistics are diagnostics only and never update the calibration.
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

    def logits(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        require(bool(torch.isfinite(values).all()), "nonfinite calibration logits")
        with torch.autocast(values.device.type, enabled=False):
            result = values * self.slope + self.intercept
        require(bool(torch.isfinite(result).all()), "nonfinite calibrated logits")
        return result

    def ratios(self, values):
        result = self.logits(values).exp() * ((1 - self.class_prior) / self.class_prior)
        require(bool(torch.isfinite(result).all()), "nonfinite calibrated ratio")
        return result


def paired_tensors(logits, weights, *, allow_zero_mass=False):
    z = torch.as_tensor(logits, dtype=torch.float32).detach()
    w = torch.as_tensor(weights, dtype=torch.float32, device=z.device).detach()
    require(z.ndim == 2 and z.shape[1] == 2 and w.shape == (len(z),),
            "calibration requires [original, observed/shifted] logits and origin weights")
    require(bool(torch.isfinite(z).all()) and bool(torch.isfinite(w).all()),
            "nonfinite calibration inputs")
    total = w.sum()
    require(bool((w >= 0).all()) and bool(torch.isfinite(total)) and
            (allow_zero_mass or bool(total > 0)), "invalid calibration target weights")
    z = z.masked_fill(w[:, None] == 0, 0)
    labels = torch.tensor([0., 1.], device=z.device).expand_as(z)
    # Normalize before duplication; both copies always retain the same mass.
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    return z, labels, (w / denominator)[:, None].expand_as(z) / 2


def pair_metrics(logits, weights):
    # A zero-mass evaluation partition contributes exactly zero to ranking.
    z, y, w = paired_tensors(logits, weights, allow_zero_mass=True)
    loss = float((F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum())
    brier = float(((z.sigmoid() - y).square() * w).sum())
    require(math.isfinite(loss) and math.isfinite(brier), "nonfinite calibration score")
    return loss, brier


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
        # Normalize magnitude before squaring: valid FP32 logits can have a
        # representable standard deviation but unrepresentable squared values.
        magnitude = z.abs().max()
        magnitude = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
        normalized = z / magnitude
        center = (mass * normalized).sum()
        scale = ((normalized - center).square() * mass).sum().sqrt()
        require(bool(torch.isfinite(scale)), "nonfinite calibration scale")
        if float(scale) == 0:
            slope, intercept = 0., 0.
        else:
            x = (normalized - center) / scale
            parameter = torch.zeros(2, dtype=torch.float32, device=z.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([parameter], lr=1., max_iter=100,
                                         tolerance_grad=1e-7, tolerance_change=1e-9,
                                         line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                loss = (F.binary_cross_entropy_with_logits(
                    parameter[0] * x + parameter[1], labels, reduction="none") * mass).sum()
                require(bool(torch.isfinite(loss)), "nonfinite affine objective")
                loss.backward()
                require(bool(torch.isfinite(parameter.grad).all()), "nonfinite affine gradient")
                return loss

            optimizer.step(closure)
            slope = float(((parameter[0] / scale) / magnitude).detach())
            intercept = float((parameter[1] - parameter[0] * center / scale).detach())
    require(math.isfinite(slope) and math.isfinite(intercept), "nonfinite affine coefficients")
    result = AffineCalibration(slope=slope, intercept=intercept, class_prior=.5,
        original_ids=ids, partitions=tuple(partitions), lineage=lineage)
    pair_metrics(result.logits(z), weights)
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
    before = pair_metrics(calibration.logits(oof_logits), weights)
    after = pair_metrics(calibration.logits(refit_logits), weights)
    z, _, mass = paired_tensors(oof_logits, weights)
    final, _, _ = paired_tensors(refit_logits, weights)
    require(final.shape == z.shape, "transfer alignment mismatch")
    delta = float(((final - z).square() * mass).sum().sqrt())
    require(math.isfinite(delta), "nonfinite calibration transfer")
    return TransferDiagnostics(original_ids=calibration.original_ids,
        metrics=(("inner_oof_log_loss", before[0]), ("inner_oof_brier", before[1]),
                 ("refit_training_log_loss", after[0]), ("refit_training_brier", after[1]),
                 ("weighted_logit_rmse", delta)),
        evaluation_kind="outer_training_refit_in_sample; diagnostic_only", lineage=lineage)
