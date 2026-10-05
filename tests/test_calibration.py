"""Synthetic CPU checks of weighting and calibration ownership."""
from dataclasses import replace

import pytest
import torch

from oxyformer.provenance import ArtifactLineage, ContractError
from oxyformer.training.calibration import (
    CalibrationPartition, fit_affine, pair_metrics, transfer_diagnostics,
)


def legacy_fit_affine(logits, weights, **kwargs):
    """Retain historical numerical regressions beyond the new public domain.

    Each such input must first fail at the public entry point. The private
    numerical kernel still runs the old assertions without changing tolerances.
    All supported inputs continue through the public fitting API.
    """
    from oxyformer.training.calibration import _fit_affine, CALIBRATION_LOGIT_ABS_MAX
    raw = torch.as_tensor(logits, dtype=torch.float64)
    if bool(torch.isfinite(raw).all()) and bool((raw.abs() > CALIBRATION_LOGIT_ABS_MAX).any()):
        with pytest.raises(ContractError, match='calibration fitting.*1000000'):
            fit_affine(logits, weights, **kwargs)
        return _fit_affine(logits, weights, **kwargs)
    return fit_affine(logits, weights, **kwargs)


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
    fitted = legacy_fit_affine(z, weights, **inputs())
    assert pair_metrics(fitted.logits(z), weights)[0] < pair_metrics(z, weights)[0]
    assert fitted.logits(z).dtype == torch.float32
    # Independent score equations; FP32 line-search losses quantize near the optimum.
    p = fitted.logits(z).double().sigmoid()
    residual = p - torch.tensor([0., 1.], dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)[:, None]
    assert abs(float((residual * w).sum() / (2 * w.sum()))) < 5e-5
    assert abs(float((residual * w * z).sum() / (2 * w.sum()))) < 5e-5
    scaled = legacy_fit_affine(z, [v * 10 for v in weights], **inputs())
    assert scaled.slope == pytest.approx(fitted.slope, abs=1e-5)
    assert scaled.intercept == pytest.approx(fitted.intercept, abs=1e-5)
    unweighted = legacy_fit_affine(z, [1.] * 4, **inputs())
    assert abs(fitted.slope - unweighted.slope) > .1


def test_calibration_records_cannot_select_their_checkpoint():
    values = inputs()
    first = replace(values["partitions"][0], checkpoint_ids=("a", "d"))
    values["partitions"] = (first, values["partitions"][1])
    with pytest.raises(ContractError, match="calibration records entered checkpoint selection"):
        legacy_fit_affine([[-1., 1.]] * 4, [1.] * 4, **values)


def test_calibration_records_cannot_enter_fitting_or_outer_holdout():
    values = inputs()
    bad = replace(values["partitions"][0], fitting_ids=("a", "c"))
    values["partitions"] = (bad, values["partitions"][1])
    with pytest.raises(ContractError, match="entered fitting"):
        legacy_fit_affine([[-1., 1.]] * 4, [1.] * 4, **values)
    values = inputs()
    values["outer_training_ids"] = ("a", "b", "c")
    with pytest.raises(ContractError, match="outside outer training"):
        legacy_fit_affine([[-1., 1.]] * 4, [1.] * 4, **values)


def test_frozen_epochs_and_complete_fold_ownership():
    values = inputs()
    values["partitions"] = tuple(replace(p, fitting_ids=p.fitting_ids + p.checkpoint_ids,
        checkpoint_ids=(), frozen_epochs=2) for p in values["partitions"])
    legacy_fit_affine([[1., 1.]] * 4, [1.] * 4, **values)
    values["fold_ids"] = (1, 0, 1, 1)
    with pytest.raises(ContractError, match="held-out partition"):
        legacy_fit_affine([[1., 1.]] * 4, [1.] * 4, **values)


@pytest.mark.parametrize("z,w", [
    ([[float("nan"), 0.]] * 4, [1.] * 4),
    ([[0., float("inf")]] * 4, [1.] * 4),
    ([[0., 0.]] * 4, [0.] * 4),
    ([[0., 0.]] * 4, [1., -1., 1., 1.]),
    ([[0., 0.]] * 4, [1., float("inf"), 1., 1.]),
])
def test_nonfinite_and_invalid_mass_fail(z, w):
    with pytest.raises(ContractError):
        legacy_fit_affine(z, w, **inputs())


def test_constant_logits_and_transfer_do_not_refit_calibration():
    calibration = legacy_fit_affine([[9., 9.]] * 4, [1., 2., 3., 4.], **inputs())
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
    baseline = legacy_fit_affine(logits, weights, **inputs())
    logits[-1] = [1e38, -1e38]
    other = legacy_fit_affine(logits, weights, **inputs())
    assert baseline.slope == other.slope and baseline.intercept == other.intercept


@pytest.mark.parametrize("scale", [1e20, 1e-25, 1e-39])
def test_nonconstant_extreme_logits_preserve_finite_calibration(scale):
    logits, values = nonseparable_case()
    baseline = legacy_fit_affine(logits, [1.] * 6, **values)
    scaled = legacy_fit_affine(logits * scale, [1.] * 6, **values)
    assert scaled.slope != 0
    torch.testing.assert_close(scaled.ratios(logits * scale), baseline.ratios(logits))


def nonseparable_case():
    ids = tuple("abcdef")
    values = dict(original_ids=ids, fold_ids=(0, 0, 1, 1, 2, 2),
        partitions=tuple(CalibrationPartition(fold=fold, evaluation_ids=ids[2*fold:2*fold+2],
            fitting_ids=tuple(oid for oid in ids if oid not in ids[2*fold:2*fold+2]),
            checkpoint_ids=(), frozen_epochs=1) for fold in range(3)),
        outer_training_ids=ids, lineage=lineage(ids))
    logits = torch.tensor([[-1., 1.], [-1., 1.], [-1., 1.], [1., -1.], [0., 0.], [0., 0.]])
    return logits, values


def test_large_common_offset_preserves_fitted_calibrated_contrasts():
    logits, values = nonseparable_case()
    baseline = legacy_fit_affine(logits, [1.] * 6, **values)
    shifted = 1000. + 2. ** -13 * logits
    fitted = legacy_fit_affine(shifted, [1.] * 6, **values)
    torch.testing.assert_close(fitted.ratios(shifted), baseline.ratios(logits))
    restored = type(fitted).from_json(fitted.to_json())
    torch.testing.assert_close(restored.ratios(shifted), fitted.ratios(shifted))


def test_transfer_rmse_preserves_representable_large_differences():
    logits, values = nonseparable_case()
    logits *= 1e20
    fitted = legacy_fit_affine(logits, [1.] * 6, **values)
    result = transfer_diagnostics(fitted, logits, torch.zeros_like(logits), [1.] * 6,
                                  lineage=lineage(values["original_ids"]))
    expected = float(logits.double().square().mean().sqrt())
    assert dict(result.metrics)["weighted_logit_rmse"] == pytest.approx(expected, rel=1e-6)


def with_zero_weight_original():
    logits, values = nonseparable_case()
    ids = values["original_ids"] + ("g",)
    values.update(original_ids=ids, fold_ids=values["fold_ids"] + (0,),
        partitions=(replace(values["partitions"][0],
            evaluation_ids=values["partitions"][0].evaluation_ids + ("g",)),
            *values["partitions"][1:]), outer_training_ids=ids, lineage=lineage(ids))
    return torch.cat((logits, torch.zeros(1, 2))), values


def test_valid_zero_weight_original_never_evaluates_a_placeholder():
    base_logits, base_values = nonseparable_case()
    logits, values = with_zero_weight_original()
    base_logits, logits = -12. + .125 * base_logits, -12. + .125 * logits
    baseline = legacy_fit_affine(base_logits, [1.] * 6, **base_values)
    # All actual ratios are finite; only the old fabricated raw zero overflows.
    assert torch.isfinite(baseline.ratios(logits)).all()
    fitted = legacy_fit_affine(logits, [1.] * 6 + [0.], **values)
    for field in ("slope", "intercept", "input_offset", "input_scale"):
        torch.testing.assert_close(torch.tensor(getattr(fitted, field)),
                                   torch.tensor(getattr(baseline, field)))
    torch.testing.assert_close(fitted.ratios(logits[:6]), baseline.ratios(base_logits))
    assert fitted.original_ids == values["original_ids"]
    assert fitted.partitions == values["partitions"]
    assert fitted.lineage == values["lineage"]


@pytest.mark.parametrize("zero_logits", [[0., 0.], [1e38, -1e38]])
def test_weighted_paths_use_only_original_positive_weight_rows(zero_logits):
    base, base_values = nonseparable_case()
    base = -12. + .125 * base
    logits, values = with_zero_weight_original()
    logits[:6], logits[6] = base, torch.tensor(zero_logits)
    weights = [1.] * 6 + [0.]
    baseline = legacy_fit_affine(base, [1.] * 6, **base_values)
    fitted = legacy_fit_affine(logits, weights, **values)
    torch.testing.assert_close(fitted.ratios(base), baseline.ratios(base))
    assert pair_metrics(logits, weights) == pair_metrics(base, [1.] * 6)
    assert pair_metrics(logits, [0.] * 7) == (0., 0.)
    refit = logits.clone()
    refit[:6] += .125
    expected = transfer_diagnostics(baseline, base, refit[:6], [1.] * 6,
                                    lineage=base_values["lineage"])
    actual = transfer_diagnostics(fitted, logits, refit, weights, lineage=values["lineage"])
    assert actual.metrics == expected.metrics
    assert actual.original_ids == values["original_ids"]
    assert actual.lineage == values["lineage"]
    # Public prediction still evaluates every requested row, independent of mass.
    with pytest.raises(ContractError, match="nonfinite calibrated"):
        fitted.ratios(logits[6:])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_zero_weight_raw_nonfinite_rows_remain_invalid(bad):
    logits, values = with_zero_weight_original()
    weights = [1.] * 6 + [0.]
    calibration = legacy_fit_affine(logits, weights, **values)
    invalid = logits.clone()
    invalid[-1, 0] = bad
    with pytest.raises(ContractError, match="nonfinite"):
        legacy_fit_affine(invalid, weights, **values)
    with pytest.raises(ContractError, match="nonfinite"):
        pair_metrics(invalid, weights)
    for oof, refit in ((invalid, logits), (logits, invalid)):
        with pytest.raises(ContractError, match="nonfinite"):
            transfer_diagnostics(calibration, oof, refit, weights, lineage=values["lineage"])


def test_zero_weight_rows_retain_weight_and_ownership_checks():
    logits, values = with_zero_weight_original()
    with pytest.raises(ContractError, match="target weights"):
        legacy_fit_affine(logits, [1.] * 6 + [-1.], **values)
    with pytest.raises(ContractError, match="held-out partition"):
        legacy_fit_affine(logits, [1.] * 6 + [0.], **(values | {"fold_ids": (0, 0, 1, 1, 2, 2, 1)}))
    with pytest.raises(ContractError, match="lineage mismatch"):
        legacy_fit_affine(logits, [1.] * 6 + [0.],
                   **(values | {"lineage": lineage(values["original_ids"][:-1])}))


def test_mixed_scale_positive_weight_logits_preserve_ordinary_contrasts():
    base, base_values = nonseparable_case()
    logits, values = with_zero_weight_original()
    logits[-1] = -1e20
    weights = [1.] * 6 + [1e-30]
    baseline = legacy_fit_affine(base, [1.] * 6, **base_values)
    fitted = legacy_fit_affine(logits, weights, **values)
    # The identity map is already strictly better than the false constant fit.
    assert pair_metrics(fitted.logits(logits), weights)[0] <= pair_metrics(logits, weights)[0]
    torch.testing.assert_close(fitted.ratios(logits[:6]), baseline.ratios(base))
    assert torch.isfinite(fitted.ratios(logits)).all()


@pytest.mark.parametrize("weights", [[1e38] * 6, [2e38, 2e38, 0., 0.]])
def test_finite_weights_with_overflowing_sum_preserve_weighted_work(weights):
    values = nonseparable_case()[1] if len(weights) == 6 else inputs()
    logits = torch.zeros(len(weights), 2)
    reference_weights = [w / max(weights) for w in weights]
    baseline = legacy_fit_affine(logits, reference_weights, **values)
    fitted = legacy_fit_affine(logits, weights, **values)
    assert fitted.slope == fitted.intercept == 0.
    torch.testing.assert_close(fitted.ratios(logits), baseline.ratios(logits))
    assert pair_metrics(logits, weights) == pair_metrics(logits, reference_weights)
    actual = transfer_diagnostics(fitted, logits, logits + 2., weights, lineage=values["lineage"])
    reference = transfer_diagnostics(baseline, logits, logits + 2., reference_weights,
                                     lineage=values["lineage"])
    assert actual.metrics == reference.metrics


def test_mixed_scale_conditioning_keeps_normalized_inputs_finite():
    logits, values = with_zero_weight_original()
    logits[:6] *= 1e-20
    logits[-1] = -1e20
    weights = [1.] * 6 + [1e-38]
    base, base_values = nonseparable_case()
    feasible = replace(legacy_fit_affine(base, [1.] * 6, **base_values),
                       slope=4e-22, intercept=0., input_offset=0., input_scale=1.)
    assert torch.isfinite(feasible.ratios(logits)).all()
    fitted = legacy_fit_affine(logits, weights, **values)
    assert torch.isfinite(fitted.ratios(logits)).all()
    assert pair_metrics(fitted.logits(logits), weights)[0] <= pair_metrics(
        torch.zeros_like(logits), weights)[0]


def test_opposite_extreme_logits_do_not_erase_weighted_contrasts():
    logits, values = with_zero_weight_original()
    c, e = float(2 ** 126), float(2 ** 103)
    logits[:] = torch.tensor([[c, c+e], [c, c+e], [c, c+e], [c+e, c],
                              [c, c], [c, c], [-3*c, -3*c]])
    weights = [1.] * 6 + [1e-38]
    fitted = legacy_fit_affine(logits, weights, **values)
    feasible = replace(fitted, slope=float(torch.log(torch.tensor(5.))),
                       intercept=float(torch.log(torch.tensor(.6))),
                       input_offset=c, input_scale=e)
    assert torch.isfinite(feasible.ratios(logits)).all()
    # A strictly better finite FP32 map rules out the false constant fit.
    assert pair_metrics(fitted.logits(logits), weights)[0] <= (
        pair_metrics(feasible.logits(logits), weights)[0] + 1e-6)


@pytest.mark.parametrize("large,small", [(1e38, 1e-20), (1e300, 1e-300)])
def test_transfer_preserves_positive_mass_before_normalization(large, small):
    logits, values = with_zero_weight_original()
    logits.zero_()
    weights = [large] * 6 + [small]
    fitted = legacy_fit_affine(logits, weights, **values)
    refit = logits.clone()
    refit[-1] = -1e38
    result = transfer_diagnostics(fitted, logits, refit, weights, lineage=values['lineage'])
    # Take square roots before the division in this independent reference:
    # the normalized mass itself need not be representable even in FP64.
    expected = abs(float(refit[-1, 0])) * (small ** .5 / large ** .5) / 6 ** .5
    assert expected > 0
    assert dict(result.metrics)['weighted_logit_rmse'] == pytest.approx(expected, rel=1e-6, abs=0.)


@pytest.mark.parametrize("weight", [1e39, 1e-50])
def test_raw_finite_weight_scale_preserves_the_fit(weight):
    logits, values = nonseparable_case()
    baseline = legacy_fit_affine(logits, [1.] * 6, **values)
    fitted = legacy_fit_affine(logits, [weight] * 6, **values)
    torch.testing.assert_close(fitted.ratios(logits), baseline.ratios(logits))
    assert pair_metrics(logits, [weight] * 6) == pair_metrics(logits, [1.] * 6)


def test_raw_negative_weight_cannot_round_to_zero_before_validation():
    logits, values = with_zero_weight_original()
    weights = [1.] * 6 + [-1e-50]
    calibration = legacy_fit_affine(logits, [1.] * 7, **values)
    with pytest.raises(ContractError, match="target weights"):
        legacy_fit_affine(logits, weights, **values)
    with pytest.raises(ContractError, match="target weights"):
        pair_metrics(logits, weights)
    with pytest.raises(ContractError, match="target weights"):
        transfer_diagnostics(calibration, logits, logits, weights, lineage=values['lineage'])


def test_raw_positive_weight_retains_its_ratio_audit():
    logits, values = with_zero_weight_original()
    logits[-1] = 1e20
    # This raw positive row has negligible loss but an overflowing ratio.
    with pytest.raises(ContractError, match="nonfinite calibrated ratio"):
        legacy_fit_affine(logits, [1.] * 6 + [1e-50], **values)


def test_subnormal_calibration_transfers_without_intermediate_normalization_overflow():
    _, values = nonseparable_case()
    core = torch.tensor([[-1., 1.]] * 3 + [[1., -1.]] * 2 + [[0., 0.]])
    z = core * 1e-39
    calibration = legacy_fit_affine(z, [1.] * 6, **values)
    # Independent FP32 order multiplies before dividing, so the result fits
    # even though the old intermediate normalized value overflowed.
    baseline = legacy_fit_affine(core, [1.] * 6, **values)
    refit = torch.full_like(z, -.5)
    scale = float(z.abs().max())
    expected = (refit * (baseline.slope / baseline.input_scale)) / scale + baseline.intercept
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(calibration.logits(refit), expected)
    assert torch.equal(calibration.ratios(refit), torch.zeros_like(refit))
    diagnostics = transfer_diagnostics(calibration, z, refit, [1.] * 6, lineage=values['lineage'])
    assert dict(diagnostics.metrics)['weighted_logit_rmse'] == pytest.approx(.5)


@pytest.mark.parametrize('scale', [1e-44, 1e-20])
def test_power_of_two_calibration_coordinates_preserve_small_slope_transfer(scale):
    from oxyformer.training.calibration import AffineCalibration
    _, values = nonseparable_case()
    core = torch.tensor([[-1., 1.]] * 3 + [[1., -1.]] * 2 + [[0., 0.]])
    z = core * scale
    actual_scale = float(z.abs().max())
    fitted = legacy_fit_affine(z, [1.] * 6, **values)
    baseline = legacy_fit_affine(core, [1.] * 6, **values)
    refit = torch.full_like(z, -actual_scale * 4e38)
    gain = baseline.slope / baseline.input_scale
    expected = (refit * gain) / actual_scale + baseline.intercept
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(fitted.logits(refit), expected)
    torch.testing.assert_close(fitted.ratios(z), baseline.ratios(core))
    restored = AffineCalibration.from_json(fitted.to_json())
    assert torch.equal(restored.logits(refit), fitted.logits(refit))


@pytest.mark.parametrize('magnitude', [1e12, 1e22, 1e32])
def test_mixed_scale_nonseparable_calibration_keeps_line_search_finite(magnitude):
    from oxyformer.training.calibration import AffineCalibration
    ids = tuple('abcdefgh')
    folds = (0, 0, 1, 1, 2, 2, 2, 2)
    partitions = tuple(CalibrationPartition(fold=fold,
        evaluation_ids=tuple(oid for oid, f in zip(ids, folds) if f == fold),
        fitting_ids=tuple(oid for oid, f in zip(ids, folds) if f != fold),
        checkpoint_ids=(), frozen_epochs=1) for fold in range(3))
    large = float(torch.tensor(magnitude, dtype=torch.float32))
    z = torch.tensor([[-1., 1.], [1., -1.], [0., 0.], [0., 0.],
                      [-large, large], [-large, large], [-large, large], [large, -large]])
    weights = [1.] * 4 + [1e-4] * 4
    provenance = lineage(ids)
    feasible = AffineCalibration(slope=float(torch.log(torch.tensor(3.))) / large,
        intercept=0., class_prior=.5, original_ids=ids, partitions=partitions, lineage=provenance)
    expected_loss = pair_metrics(feasible.logits(z), weights)[0]
    constant_loss = pair_metrics(torch.zeros_like(z), weights)[0]
    assert expected_loss < constant_loss - 1e-5
    fitted = legacy_fit_affine(z, weights, original_ids=ids, fold_ids=folds, partitions=partitions,
                        outer_training_ids=ids, lineage=provenance)
    assert fitted.logits(z).dtype == fitted.ratios(z).dtype == torch.float32
    actual_loss = pair_metrics(fitted.logits(z), weights)[0]
    # Loss observations are computed in FP32; retain its default tolerances.
    torch.testing.assert_close(torch.tensor(actual_loss, dtype=torch.float32),
                               torch.tensor(expected_loss, dtype=torch.float32))
    assert actual_loss < constant_loss - 1e-5


@pytest.mark.parametrize('bad', [1_000_001., -1_000_001., -1e32])
@pytest.mark.parametrize('weight', [0., 1.])
def test_fitting_rejects_logits_outside_supported_domain_before_optimizer(monkeypatch, bad, weight):
    z, values = with_zero_weight_original()
    z[-1, 0] = bad
    calls = []
    monkeypatch.setattr(torch.optim.LBFGS, 'step', lambda *args: calls.append(True))
    with pytest.raises(ContractError, match='calibration fitting.*1000000'):
        fit_affine(z, [1.] * 6 + [weight], **values)
    assert calls == []


@pytest.mark.parametrize('sign', [1., -1.])
def test_negligible_weight_cannot_freeze_ordinary_calibration(sign):
    import math
    from oxyformer.training.calibration import AffineCalibration
    z, values = with_zero_weight_original()
    z[:6] *= 1e-26 * sign
    z[-1] = torch.tensor([-1e6 * sign, 0.])
    weights = [1.] * 6 + [1e-12]
    feasible = AffineCalibration(slope=sign * math.log(3), intercept=0., class_prior=.5,
        original_ids=values['original_ids'], partitions=values['partitions'],
        lineage=values['lineage'], input_scale=float(z[0].abs().max()))
    assert torch.isfinite(feasible.ratios(z)).all()
    fitted = fit_affine(z, weights, **values)
    actual, comparison = pair_metrics(fitted.logits(z), weights)[0], pair_metrics(feasible.logits(z), weights)[0]
    assert actual <= comparison + 1e-6
    torch.testing.assert_close(torch.tensor(actual, dtype=torch.float32),
                               torch.tensor(comparison, dtype=torch.float32))


def test_stalled_calibration_is_refused_when_a_feasible_map_is_better(monkeypatch):
    # Emulate successful optimizer return without an update, not an exception.
    # The old fitter silently publishes the strictly suboptimal constant map.
    monkeypatch.setattr(torch.optim.LBFGS, 'step', lambda self, closure: closure())
    z, values = nonseparable_case()
    with pytest.raises(ContractError, match='affine calibration did not converge'):
        fit_affine(z, [1.] * 6, **values)


@pytest.mark.parametrize('value', [-1_000_000., 1_000_000.])
def test_fitting_domain_includes_its_boundary_and_checks_before_fp32_rounding(value):
    z = torch.full((4, 2), value, dtype=torch.float64)
    fitted = fit_affine(z, [1.] * 4, **inputs())
    assert torch.equal(fitted.ratios(z), torch.ones(4, 2))
    z[0, 0] += .001 if value > 0 else -.001
    # This violation rounds back onto the boundary in FP32.
    assert float(z.float()[0, 0]) == value
    with pytest.raises(ContractError, match='calibration fitting.*1000000'):
        fit_affine(z, [0., 1., 1., 1.], **inputs())
