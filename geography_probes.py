"""Separate, held-out diagnostic exposure probes. Never a nuisance adapter.

Views copy diagnostic values and permissions. They cannot expose labels or
be converted into primary CovariateViews. A probe is a linear reconstruction
baseline, not a conditional-density or causal-identification certificate.
"""
from dataclasses import dataclass

import numpy as np

from oxyformer.data.feature_roles import FeatureRegistry
from oxyformer.provenance import ContractError, Immutable, require, unique


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticView(Immutable):
    endpoint: str
    registry: FeatureRegistry
    original_ids: tuple[str, ...]
    columns: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.original_ids) and bool(self.columns), "empty diagnostic view")
        unique(self.original_ids, "diagnostic original IDs")
        unique(self.columns, "diagnostic columns")
        require(len(self.values) == len(self.original_ids) and all(len(v) == len(self.columns) for v in self.values),
                "diagnostic alignment mismatch")
        for name in self.columns:
            self.registry.require(name, self.endpoint, "diagnostic")

    def as_covariates(self, *args, **kwargs):
        raise ContractError("Diagnostic geography and probe outputs cannot become nuisance inputs")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeResult(Immutable):
    view_hash: str
    train_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    heldout_predictions: tuple[float, ...]
    mse: float
    interpretation: str = "diagnostic-only held-out exposure reconstruction; no nuisance permission"

    def as_covariates(self, *args, **kwargs):
        raise ContractError("Diagnostic geography and probe outputs cannot become nuisance inputs")


def exposure_probe(view, exposure, train_ids, heldout_ids):
    require(type(view) is DiagnosticView, "separate DiagnosticView required")
    train_ids, heldout_ids = tuple(train_ids), tuple(heldout_ids)
    require(bool(train_ids) and bool(heldout_ids), "probe partitions required")
    unique(train_ids + heldout_ids, "probe partition IDs")
    require(set(train_ids + heldout_ids) == set(view.original_ids), "probe partition coverage mismatch")
    require(set(exposure) == set(view.original_ids), "exposure ID mismatch")
    rows = {oid: values for oid, values in zip(view.original_ids, view.values)}
    train = np.asarray([rows[i] for i in train_ids], dtype=np.float64)
    test = np.asarray([rows[i] for i in heldout_ids], dtype=np.float64)
    y = np.asarray([exposure[i] for i in train_ids], dtype=np.float64)
    truth = np.asarray([exposure[i] for i in heldout_ids], dtype=np.float64)
    require(bool(np.isfinite(y).all() and np.isfinite(truth).all()), "nonfinite probe exposure")
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(train)), (train - mean) / scale))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = np.column_stack((np.ones(len(test)), (test - mean) / scale)) @ coefficients
    return ProbeResult(view_hash=view.content_hash, train_ids=train_ids, heldout_ids=heldout_ids,
                       heldout_predictions=tuple(prediction.tolist()), mse=float(np.mean((truth - prediction) ** 2)))
