"""Adversarial comparisons of frozen nuisance states and aligned predictions."""
from dataclasses import replace

import torch

from oxyformer.provenance import require
from oxyformer.training.checkpoint import load_checkpoint, model_state_hash


def model_hashes(artifact):
    require(artifact.complete, "only completed nuisance states can be compared")
    state = load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)["controller"]
    return {kind: model_state_hash({name: value for name, value in bundle["state"].items()
                                   if isinstance(value, torch.Tensor)})
            for kind, bundle in state["final"].items()}


def assert_fitted_invariant(before, after):
    """Compare numerical models, tuning and calibration; opaque lineage may differ."""
    assert model_hashes(before) == model_hashes(after), "held-out perturbation changed fitted models"
    a, b = [load_checkpoint(x.checkpoint, x.checkpoint.identity)["controller"] for x in (before, after)]
    assert a["selection"] == b["selection"], "held-out perturbation changed tuning"
    for kind in a["final"]:
        assert a["final"][kind]["preprocessing"] == b["final"][kind]["preprocessing"], "preprocessing changed"
        from oxyformer.contracts import CovariateView
        ra, rb = [CovariateView.from_json(s["final"][kind]["references"]) for s in (a, b)]
        assert (ra.original_ids, ra.columns, ra.values) == (rb.original_ids, rb.columns, rb.values), "references changed"
    from oxyformer.training.calibration import AffineCalibration
    ca, cb = [AffineCalibration.from_json(x["calibration"]) for x in (a, b)]
    assert replace(ca, lineage=cb.lineage) == cb, "held-out perturbation changed calibration"


def assert_prediction_invariant(before, after, *, except_ids=()):
    columns = ("mu_a", "mu_d", "r_a", "r_d", "origin_weights")
    def rows(value):
        return {key: row for key, *row in zip(zip(value.original_ids, value.seed_ids),
                    *(getattr(value, name) for name in columns)) if key[0] not in except_ids}
    assert rows(before) == rows(after), "unrelated predictions changed"


def audit_precision(reference, candidate, *, absolute_tolerance, relative_tolerance):
    """Report measured FP32-relative errors; never enable BF16 as a side effect."""
    require(reference.dtype == torch.float32, "FP32 reference required")
    require(reference.shape == candidate.shape and reference.numel() > 0, "precision shape mismatch")
    require(absolute_tolerance > 0 and relative_tolerance > 0, "positive audit tolerances required")
    a, b = reference.detach().double(), candidate.detach().double()
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "nonfinite precision audit")
    error = (a - b).abs()
    allowed = absolute_tolerance + relative_tolerance * a.abs()
    return {"passed": bool((error <= allowed).all()), "max_absolute_error": float(error.max()),
            "max_scaled_error": float((error / allowed).max()),
            "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
            "bf16_enabled": False}
