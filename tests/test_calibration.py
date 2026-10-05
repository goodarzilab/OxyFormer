"""Synthetic CPU checks of weighting and calibration ownership."""
from dataclasses import replace

import pytest
import torch

from oxyformer.provenance import ArtifactLineage, ContractError
from oxyformer.training.calibration import (
    CalibrationPartition, fit_affine, pair_metrics, transfer_diagnostics,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def lineage(ids):
    return ArtifactLineage(source_hashes=("a" * 64,), unit_ids=tuple(ids), parent_hashes=(),
        split_hash="b" * 64, config_hash="c" * 64, model_hash=None,
        environment=(("fixture", "cpu"),), seed=1103, parameter_count=None)


def inputs():
    ids = ("a", "b", "c", "d")
    partitions = (
        CalibrationPartition(fold=0, evaluation_ids=ids[:2], fitting_ids=("c",),
                             checkpoint_ids=("d",), frozen_epochs=None),
        CalibrationPartition(fold=1, evaluation_ids=ids[2:], fitting_ids=("a",),
                             checkpoint_ids=("b",), frozen_epochs=None),
    )
    return dict(original_ids=ids, fold_ids=(0, 0, 1, 1), partitions=partitions,
                outer_training_ids=ids, lineage=lineage(ids))


def test_weighted_affine_reduces_loss_and_solves_weighted_score_equations():
    z = torch.tensor([[-2., 1.], [2., -.5], [-1., 2.], [.7, -.8]])
    weights = [8., 1., 5., 2.]
    fitted = fit_affine(z, weights, **inputs())
    assert pair_metrics(fitted.logits(z), weights)[0] < pair_metrics(z, weights)[0]
    assert fitted.logits(z).dtype == torch.float32
    # Independent score equations; FP32 line-search losses quantize near the optimum.
    p = fitted.logits(z).double().sigmoid()
    residual = p - torch.tensor([0., 1.], dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)[:, None]
    assert abs(float((residual * w).sum() / (2 * w.sum()))) < 5e-5
    assert abs(float((residual * w * z).sum() / (2 * w.sum()))) < 5e-5
    scaled = fit_affine(z, [v * 10 for v in weights], **inputs())
    assert scaled.slope == pytest.approx(fitted.slope, abs=1e-5)
    assert scaled.intercept == pytest.approx(fitted.intercept, abs=1e-5)
    unweighted = fit_affine(z, [1.] * 4, **inputs())
    assert abs(fitted.slope - unweighted.slope) > .1


def test_calibration_records_cannot_select_their_checkpoint():
    values = inputs()
    first = replace(values["partitions"][0], checkpoint_ids=("a", "d"))
    values["partitions"] = (first, values["partitions"][1])
    with pytest.raises(ContractError, match="calibration records entered checkpoint selection"):
        fit_affine([[-1., 1.]] * 4, [1.] * 4, **values)


def test_calibration_records_cannot_enter_fitting_or_outer_holdout():
    values = inputs()
    bad = replace(values["partitions"][0], fitting_ids=("a", "c"))
    values["partitions"] = (bad, values["partitions"][1])
    with pytest.raises(ContractError, match="entered fitting"):
        fit_affine([[-1., 1.]] * 4, [1.] * 4, **values)
    values = inputs()
    values["outer_training_ids"] = ("a", "b", "c")
    with pytest.raises(ContractError, match="outside outer training"):
        fit_affine([[-1., 1.]] * 4, [1.] * 4, **values)


def test_frozen_epochs_and_complete_fold_ownership():
    values = inputs()
    values["partitions"] = tuple(replace(p, fitting_ids=p.fitting_ids + p.checkpoint_ids,
        checkpoint_ids=(), frozen_epochs=2) for p in values["partitions"])
    fit_affine([[1., 1.]] * 4, [1.] * 4, **values)
    values["fold_ids"] = (1, 0, 1, 1)
    with pytest.raises(ContractError, match="held-out partition"):
        fit_affine([[1., 1.]] * 4, [1.] * 4, **values)


@pytest.mark.parametrize("z,w", [
    ([[float("nan"), 0.]] * 4, [1.] * 4),
    ([[0., float("inf")]] * 4, [1.] * 4),
    ([[0., 0.]] * 4, [0.] * 4),
    ([[0., 0.]] * 4, [1., -1., 1., 1.]),
    ([[0., 0.]] * 4, [1., float("inf"), 1., 1.]),
])
def test_nonfinite_and_invalid_mass_fail(z, w):
    with pytest.raises(ContractError):
        fit_affine(z, w, **inputs())


def test_constant_logits_and_transfer_do_not_refit_calibration():
    calibration = fit_affine([[9., 9.]] * 4, [1., 2., 3., 4.], **inputs())
    assert calibration.slope == calibration.intercept == 0
    assert torch.equal(calibration.ratios([[9., 9.]]), torch.ones(1, 2))
    before = calibration.to_json()
    result = transfer_diagnostics(calibration, [[9., 9.]] * 4,
        [[-4., 20.]] * 4, [1., 2., 3., 4.], lineage=lineage(calibration.original_ids))
    assert calibration.to_json() == before
    assert "in_sample" in result.evaluation_kind
    assert dict(result.metrics)["weighted_logit_rmse"] > 1
    explosive = replace(calibration, slope=1.)
    with pytest.raises(ContractError, match="nonfinite calibrated ratio"):
        explosive.ratios([[1000., 1000.]])


def test_zero_mass_extreme_records_cannot_change_calibration():
    weights = [1., 1., 1., 0.]
    logits = [[-2., 1.], [2., -.5], [-1., 2.], [0., 0.]]
    baseline = fit_affine(logits, weights, **inputs())
    logits[-1] = [1e38, -1e38]
    other = fit_affine(logits, weights, **inputs())
    assert baseline.slope == other.slope and baseline.intercept == other.intercept
