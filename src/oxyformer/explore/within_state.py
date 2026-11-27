"""Exploratory within-state USALEEP readout: ``usaleep_within_state_d2_c2``.

Exploratory; certifies no release criterion.

Owner decision ``a0d288a4-fedc-5d64-ab91-817f99584f5e`` (2026-11-27, recorded as
``owner_decisions.within_state_primary_amendment`` in ``configs/approvals.yaml``)
redefined the exploratory primary comparison stratum from county to state. This
module packages the cross-fitted one-step runner for that estimand:

* support: outcome-blind interior-bin ``2.0 IQR`` support within each state
  (:func:`compute_within_state_support`), using the ``design/support.py`` bin
  rules without the within-county and 25 km restrictions;
* folds: whole-county atomic outer folds balanced along each state's exposure
  gradient (:func:`make_balanced_county_folds`);
* features: exact one-hot state fixed effects plus within-state centred exposure
  (:func:`encode_features_onehot`), so no learner sees cross-state exposure
  contrasts;
* nuisances: pluggable ``Q`` regressors and ``g`` origin classifiers
  (:func:`fit_cross_fitted_within_state`). ``A6-Q`` is the frozen TabICLv2
  regressor; ``A7-g`` is the frozen TabICLv2 origin classifier with
  county-grouped inner out-of-fold Platt calibration. The amendment's
  ``density_ratio_benchmark_gate`` (G-SIM) completed on 2026-11-27 (Slurm
  175340 G1_within_county, 175341 G2_within_state; W&B fbod6ooi, c8dkdzvr):
  ``A7_tabicl_origin`` passed the prespecified G2_within_state gate (9/10 wins
  vs A0, RMSE 0.8455 vs 0.9992, calibration slope 0.847, null RMSE 0.0242);
  ``A5_riesz_nn`` failed (0/10) and is dropped. ``B0_gbm_origin`` also passed
  G2 and stays the ``g`` sensitivity comparator, per the gate's rule that a
  ratio estimator is never chosen on max ``r_a``, ESS or sign alone;
* inference and reporting: one-step and Hajek estimates, iid, county-clustered
  and spatial Conley standard errors, influence concentration, and the
  permanent reporting invariants (per-fold plug-in versus EIF correction,
  landing ESS, maximum ratio, delivered rather than nominal shift)
  (:func:`compute_conley_inference`, :func:`score_within_state_estimate`).

``tabicl``, ``torch`` and ``wandb`` are imported only when a TabICL learner is
built or a real run starts, so the module imports and its offline tests run
without them.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module, metadata
import json
from math import floor
import os
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oxyformer.provenance import file_hash, require

ESTIMAND_ID = 'usaleep_within_state_d2_c2'
OWNER_DECISION_ID = 'a0d288a4-fedc-5d64-ab91-817f99584f5e'
LABELS = dict(exploratory=True, estimand_id=ESTIMAND_ID, owner_decision=OWNER_DECISION_ID,
              effect_release_authorized=False, certifies_release_criteria=False)

# Context caps for the frozen TabICLv2 learners.
MAX_Q_CONTEXT_ROWS = 30_000
MAX_G_ORIGINAL_ROWS = 15_000          # 15,000 originals -> at most 30,000 paired rows.
MIN_STATE_SD = 0.25                   # mmHg floor for the within-state exposure scale.
PLATT_C = 10.0
PLATT_LOGIT_CLIP = 10.0
ODDS_LOG_CLIP = (-8.0, 6.0)           # numerical guard only; no trimming of the ratio.
PROBA_CLIP = 1e-4
Z95 = 1.959964
CONLEY_BANDWIDTHS_KM = (50.0, 100.0, 250.0)
EARTH_RADIUS_KM = 6371.0088
NULL_SEED = 20261127
MIN_STRATUM_ROWS = 20

TABICL_REPOSITORY = 'jingang/TabICL'
TABICL_REVISION = '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
TABICL_REGRESSOR_FILE = 'tabicl-regressor-v2-20260212.ckpt'
TABICL_CLASSIFIER_FILE = 'tabicl-classifier-v2-20260212.ckpt'

PRIMARY_LABEL = 'A6-Q + A7-g (primary TabICLv2)'
NULL_LABEL = 'NULL permuted-A within state (A6-Q + A7-g)'


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


RegressorFactory = Callable[[int], Regressor]
ClassifierFactory = Callable[[int], Classifier]


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WithinStateSupport:
    """Outcome-blind within-state support; every array is aligned with the input rows."""
    supported: np.ndarray       # query rows whose own exposure bin is an occupied interior bin
    moved: np.ndarray           # supported rows whose whole shift path stays on occupied interior bins
    scales: np.ndarray          # per-covariate IQR used for the caliper (1.0 where the IQR is zero)
    zero_iqr_columns: np.ndarray  # covariates whose pool IQR is zero
    delta: float
    caliper: float
    zero_iqr_mode: str = 'unit_scale'

    @property
    def n_targets(self) -> int:
        return int(self.supported.sum())

    @property
    def n_movers(self) -> int:
        return int(self.moved.sum())

    @property
    def delivered_shift(self) -> float:
        """Mean shift over targets actually delivered by shift-or-stay (never the nominal delta)."""
        require(self.n_targets > 0, 'no supported targets')
        return float(self.delta * self.n_movers / self.n_targets)


ZERO_IQR_MODES = ('unit_scale', 'exact')


def covariate_iqr(X: np.ndarray) -> np.ndarray:
    """Per-column observed interquartile range."""
    return np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)


def covariate_scales(X: np.ndarray) -> np.ndarray:
    """Per-column observed IQR; a column whose IQR is exactly 0 gets scale 1.0 (see ``zero_iqr_mode``)."""
    iqr = covariate_iqr(X)
    return np.where(iqr > 0.0, iqr, 1.0)


def compute_within_state_support(
    a: np.ndarray, X: np.ndarray, states: np.ndarray, query: np.ndarray | None = None, *,
    delta: float = 2.0, caliper: float = 2.0, bin_width: float = 1.0, min_bin_tracts: int = 2,
    zero_iqr_mode: Literal['unit_scale', 'exact'] = 'unit_scale',
) -> WithinStateSupport:
    """Interior-bin support inside each state; reads exposure and covariates only.

    For a query row, its neighbours are the other rows of the same state with the
    identical missingness pattern whose observed covariates all lie within
    ``caliper * IQR`` (L-infinity). A neighbour bin ``b = floor(a / bin_width)`` is
    occupied when ``first < b < last`` over the neighbour bins and it holds at least
    ``min_bin_tracts`` neighbours (``design/support.py`` rules; tails are never
    support). The query is supported when its own bin is occupied, and moves when
    every bin from its own to ``floor((a + delta) / bin_width)`` is occupied
    (``support.recheck_support``'s path rule); otherwise it stays.

    ``query`` marks the rows to classify (default: all). Only rows with a finite
    exposure form the pool: they alone are neighbours and set the IQR scales. A row
    without exposure is outside the atlas; it can never stand in for a tail bin.

    ``zero_iqr_mode`` sets the rule for a covariate whose pool IQR is zero:
    ``'unit_scale'`` (default, the verified exploratory census code) uses scale 1.0,
    so the caliper is ``caliper`` raw units; ``'exact'`` requires exact equality,
    as ``design/support.py`` ``neighbors`` does for ``scale == 0``.
    """
    a = np.asarray(a, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    states = np.asarray(states)
    n = len(a)
    query = np.ones(n, dtype=bool) if query is None else np.asarray(query, dtype=bool)
    require(X.ndim == 2 and X.shape[0] == n and states.shape == (n,) and query.shape == (n,),
            'support inputs are misaligned')
    pool = np.isfinite(a)
    require(pool[query].all(), 'support queries require finite exposures')
    require(not np.isinf(X).any(), 'support covariates must be finite or missing')
    require(np.isfinite(delta) and delta >= 0, 'delta must be finite and nonnegative')
    require(np.isfinite(caliper) and caliper > 0 and bin_width > 0 and min_bin_tracts >= 2,
            'invalid support recipe')
    require(zero_iqr_mode in ZERO_IQR_MODES, f'zero_iqr_mode must be one of {ZERO_IQR_MODES}')

    iqr = covariate_iqr(X[pool])
    zero_iqr = ~(iqr > 0.0)       # exactly zero (or no observed values), as support.py's scale == 0
    scales = np.where(zero_iqr, 1.0, iqr)
    # Per-column L-infinity tolerance in raw units (never divided by a tiny IQR); 0 means exact equality.
    tolerance = caliper * np.where(zero_iqr, 0.0 if zero_iqr_mode == 'exact' else 1.0, iqr)
    missing = np.isnan(X)
    _, pattern = np.unique(missing, axis=0, return_inverse=True)
    pattern = pattern.reshape(-1)
    bins = np.zeros(n, dtype=np.int64)
    bins[pool] = np.floor(a[pool] / bin_width).astype(np.int64)
    supported = np.zeros(n, dtype=bool)
    moved = np.zeros(n, dtype=bool)

    for state in np.unique(states):
        rows = np.flatnonzero(states == state)
        for q in rows[query[rows]]:
            candidates = rows[pool[rows] & (pattern[rows] == pattern[q]) & (rows != q)]
            if len(candidates) == 0:
                continue
            observed = ~missing[q]
            diff = np.abs(X[np.ix_(candidates, observed)] - X[q, observed])
            near = (diff <= tolerance[observed]).all(axis=1)
            neighbour_bins = bins[candidates[near]]
            if len(neighbour_bins) == 0:
                continue
            first, last = int(neighbour_bins.min()), int(neighbour_bins.max())
            values, counts = np.unique(neighbour_bins, return_counts=True)
            occupied = {int(b) for b, c in zip(values, counts) if first < b < last and c >= min_bin_tracts}
            origin = int(bins[q])
            if origin not in occupied:
                continue
            supported[q] = True
            destination = int(floor((a[q] + delta) / bin_width))
            moved[q] = all(b in occupied for b in range(origin, destination + 1))

    return WithinStateSupport(supported=supported, moved=moved, scales=scales, zero_iqr_columns=zero_iqr,
                              delta=float(delta), caliper=float(caliper), zero_iqr_mode=zero_iqr_mode)


# ---------------------------------------------------------------------------
# Folds, features, subsampling
# ---------------------------------------------------------------------------

def balanced_county_fold_ids(counties: np.ndarray, states: np.ndarray, a: np.ndarray,
                             n_splits: int = 5) -> np.ndarray:
    """Fold per row: counties sorted by mean exposure inside each state, dealt round-robin.

    The deal starts at ``state_index % n_splits`` so small states do not all fill fold 0.
    """
    counties, states = np.asarray(counties), np.asarray(states)
    a = np.asarray(a, dtype=np.float64)
    require(n_splits >= 2 and counties.shape == states.shape == a.shape, 'invalid fold inputs')
    state_of: dict[Any, Any] = {}
    for c, s in zip(counties, states):
        require(state_of.setdefault(c, s) == s, f'county {c} spans more than one state')
    labels, inverse = np.unique(counties, return_inverse=True)
    means = np.bincount(inverse, weights=a) / np.bincount(inverse)
    county_mean = dict(zip(labels.tolist(), means.tolist()))
    fold_of: dict[Any, int] = {}
    for si, state in enumerate(sorted(set(state_of.values()))):
        members = sorted((c for c, s in state_of.items() if s == state), key=lambda c: (county_mean[c], c))
        for rank, c in enumerate(members):
            fold_of[c] = (rank + si) % n_splits
    return np.asarray([fold_of[c] for c in counties.tolist()], dtype=int)


def make_balanced_county_folds(counties: np.ndarray, states: np.ndarray, a: np.ndarray,
                               n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Whole-county atomic outer folds as ``(train_rows, eval_rows)`` pairs.

    Each county falls in exactly one evaluation fold, so train and eval never share
    a county, and each state's exposure gradient is spread across all folds.
    """
    fold_ids = balanced_county_fold_ids(counties, states, a, n_splits)
    counties = np.asarray(counties)
    splits = []
    for fold in range(n_splits):
        ev, tr = np.flatnonzero(fold_ids == fold), np.flatnonzero(fold_ids != fold)
        require(set(counties[tr].tolist()).isdisjoint(counties[ev].tolist()), f'fold {fold} shares a county')
        splits.append((tr, ev))
    return splits


@dataclass(frozen=True)
class StateStats:
    """Training-partition exposure location and scale per state."""
    means: Mapping[Any, float]
    sds: Mapping[Any, float]
    global_mean: float
    global_sd: float

    def lookup(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray([self.means.get(s, self.global_mean) for s in states.tolist()], dtype=np.float64)
        sd = np.asarray([self.sds.get(s, self.global_sd) for s in states.tolist()], dtype=np.float64)
        return m, sd


def fit_state_stats(a_train: np.ndarray, states_train: np.ndarray) -> StateStats:
    """Mean and sample SD (floored at 0.25 mmHg; 1.0 for a single row) of exposure per state."""
    a_train = np.asarray(a_train, dtype=np.float64)
    states_train = np.asarray(states_train)
    require(len(a_train) > 0 and a_train.shape == states_train.shape, 'invalid state-stat inputs')
    means, sds = {}, {}
    for s in np.unique(states_train).tolist():
        values = a_train[states_train == s]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
        means[s], sds[s] = float(np.mean(values)), max(sd, MIN_STATE_SD)
    return StateStats(means=means, sds=sds, global_mean=float(np.mean(a_train)),
                      global_sd=float(np.std(a_train)) or 1.0)


def state_onehot(states: np.ndarray, all_states: Sequence[Any]) -> np.ndarray:
    """Exact indicator per state in ``all_states``; an unlisted state gets an all-zero row."""
    index = {s: i for i, s in enumerate(all_states)}
    out = np.zeros((len(states), len(all_states)), dtype=np.float64)
    for r, s in enumerate(np.asarray(states).tolist()):
        if s in index:
            out[r, index[s]] = 1.0
    return out


def encode_features_onehot(a: np.ndarray, X_imputed: np.ndarray, states: np.ndarray, stats: StateStats,
                           all_states: Sequence[Any], *, include_raw_a: bool = False) -> np.ndarray:
    """Columns: [raw A (opt-in)] + centred A + standardised A + imputed X + state one-hot.

    Raw A carries between-state exposure differences, so it is excluded by default.
    """
    a = np.asarray(a, dtype=np.float64)
    states = np.asarray(states)
    require(X_imputed.shape[0] == len(a) == len(states), 'feature inputs are misaligned')
    mean, sd = stats.lookup(states)
    centred = a - mean
    columns = [centred, centred / np.maximum(sd, MIN_STATE_SD)]
    if include_raw_a:
        columns.insert(0, a)
    return np.column_stack(columns + [X_imputed, state_onehot(states, all_states)])


def deterministic_subsample_idx(ids: np.ndarray, cap: int, salt: str) -> np.ndarray:
    """Sorted row positions of the ``cap`` ids with the smallest ``sha256(salt:id)``.

    Outcome-blind: reads ids only; all rows are kept when ``len(ids) <= cap``.
    """
    require(type(cap) is int and cap >= 1, 'cap must be a positive integer')
    ids = np.asarray(ids)
    if len(ids) <= cap:
        return np.arange(len(ids))
    hashes = np.asarray([sha256(f'{salt}:{oid}'.encode()).hexdigest() for oid in ids.tolist()])
    return np.sort(np.argsort(hashes, kind='stable')[:cap])


def permute_within_state(a: np.ndarray, moved: np.ndarray, states: np.ndarray,
                         seed: int = NULL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Falsification null: permute A across all rows of each state; Y, X, target and moved stay fixed.

    ``a_null`` is then independent of (Y, X, target, moved) given state, so the true effect
    is zero; stratifying by target would let ``a_null`` carry the target/non-target
    exposure contrast into a pooled Q. ``moved`` is returned unchanged, so every state
    keeps its target mover count and the null's delivered shift is the original's.
    """
    a, moved, states = np.asarray(a, dtype=np.float64), np.asarray(moved, dtype=bool), np.asarray(states)
    require(moved.shape == states.shape == a.shape, 'permutation inputs are misaligned')
    rng = np.random.default_rng(seed)
    a_null = a.copy()
    for s in sorted(np.unique(states).tolist()):
        rows = np.flatnonzero(states == s)
        a_null[rows] = a[rng.permutation(rows)]
    return a_null, moved.copy()


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QLearner:
    """Outcome regressor; fold seed = ``seed + fold``; ``context_cap`` bounds training rows."""
    name: str
    factory: RegressorFactory
    seed: int
    context_cap: int | None = None
    salt: str = 'ws-v2-q'


@dataclass(frozen=True)
class GLearner:
    """Origin classifier for the density ratio; fold seed = ``seed + 10 * fold``."""
    name: str
    factory: ClassifierFactory
    seed: int
    original_cap: int | None = None
    score: Literal['proba', 'decision'] = 'proba'
    inner_splits: int = 3
    salt: str = 'a7-v2'


def _version_pin() -> None:
    from oxyformer.models.tabicl_comparator import PACKAGE_PINS
    try:
        version = metadata.version('tabicl')
    except metadata.PackageNotFoundError as exc:
        raise ImportError('tabicl is required for A6-Q / A7-g learners') from exc
    require(version == PACKAGE_PINS['tabicl'], f'tabicl {version} differs from the approved pin')


def tabicl_regressor_factory(checkpoint: str | Path, device: str, n_estimators: int = 8) -> RegressorFactory:
    """A6-Q: frozen TabICLv2 regressor; ``tabicl`` is imported on first construction."""
    def make(seed: int) -> Regressor:
        _version_pin()
        module = import_module('tabicl')
        return module.TabICLRegressor(model_path=str(checkpoint), checkpoint_version=TABICL_REGRESSOR_FILE,
                                      allow_auto_download=False, device=str(device), random_state=seed,
                                      n_estimators=n_estimators)
    return make


def tabicl_classifier_factory(checkpoint: str | Path, device: str, n_estimators: int = 4) -> ClassifierFactory:
    """A7-g: frozen TabICLv2 classifier; ``tabicl`` is imported on first construction."""
    def make(seed: int) -> Classifier:
        _version_pin()
        module = import_module('tabicl')
        return module.TabICLClassifier(model_path=str(checkpoint), checkpoint_version=TABICL_CLASSIFIER_FILE,
                                       allow_auto_download=False, device=str(device), random_state=seed,
                                       n_estimators=n_estimators)
    return make


def gbm_regressor_factory(seed: int) -> Regressor:
    """B0_hist_gbm Q comparator."""
    return HistGradientBoostingRegressor(max_iter=350, max_depth=6, learning_rate=0.05,
                                         min_samples_leaf=20, random_state=seed)


def gbm_classifier_factory(seed: int) -> Classifier:
    """B0_hist_gbm origin classifier."""
    return HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.05, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=seed)


def logit_classifier_factory(seed: int) -> Classifier:
    """Smooth regularised logistic origin classifier (deterministic; seed unused)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=300))


def default_q_learners(regressor_checkpoint: str | Path, device: str) -> list[QLearner]:
    return [QLearner('A6-Q', tabicl_regressor_factory(regressor_checkpoint, device, 8), seed=1103,
                     context_cap=MAX_Q_CONTEXT_ROWS),
            QLearner('B0_hist_gbm', gbm_regressor_factory, seed=1103)]


def default_g_learners(classifier_checkpoint: str | Path, device: str) -> list[GLearner]:
    return [GLearner('A7-g', tabicl_classifier_factory(classifier_checkpoint, device, 4), seed=2207,
                     original_cap=MAX_G_ORIGINAL_ROWS),
            GLearner('B0_gbm_origin', gbm_classifier_factory, seed=3301),
            GLearner('B0_logit_origin', logit_classifier_factory, seed=4401, score='decision')]


def default_null_learners(regressor_checkpoint: str | Path, classifier_checkpoint: str | Path,
                          device: str) -> tuple[list[QLearner], list[GLearner]]:
    return ([QLearner('A6-Q', tabicl_regressor_factory(regressor_checkpoint, device, 4), seed=5501,
                      context_cap=MAX_Q_CONTEXT_ROWS)],
            [GLearner('A7-g', tabicl_classifier_factory(classifier_checkpoint, device, 4), seed=6601,
                      original_cap=MAX_G_ORIGINAL_ROWS)])


# ---------------------------------------------------------------------------
# Density ratio by origin classification
# ---------------------------------------------------------------------------

def _raw_logits(model: Any, X: np.ndarray, score: str) -> np.ndarray:
    if score == 'decision':
        return np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1], PROBA_CLIP, 1.0 - PROBA_CLIP)
    return logit(p)


def platt_calibrate_odds(raw_logits_oof: np.ndarray, labels_oof: np.ndarray,
                         raw_logits_eval: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit Platt scaling on out-of-fold logits; return calibrated odds ``r = p / (1 - p)`` at eval rows."""
    clip = PLATT_LOGIT_CLIP
    model = LogisticRegression(C=PLATT_C, solver='lbfgs')
    model.fit(np.clip(raw_logits_oof, -clip, clip).reshape(-1, 1), np.asarray(labels_oof, dtype=int))
    calibrated = model.decision_function(np.clip(raw_logits_eval, -clip, clip).reshape(-1, 1))
    return (np.exp(np.clip(calibrated, *ODDS_LOG_CLIP)),
            dict(slope=float(model.coef_[0, 0]), intercept=float(model.intercept_[0])))


def fit_platt_calibrated_origin(
    F0: np.ndarray, F1: np.ndarray, F_eval: np.ndarray, counties: np.ndarray,
    make_classifier: ClassifierFactory, *, seed: int, ids: np.ndarray | None = None,
    cap: int | None = None, salt: str = 'a7-v2', inner_splits: int = 3,
    score: Literal['proba', 'decision'] = 'proba',
) -> tuple[np.ndarray, dict]:
    """Density ratio ``dP(A_d, X) / dP(A, X)`` at eval rows by classifying origin (0) vs shifted (1).

    Each original row contributes its factual pair (``F0``, label 0) and shifted pair
    (``F1``, label 1); both carry the row's county. Inner folds group by county, so a
    county's factual and shifted pairs are never split across inner train and
    validation, matching the outer folds' county atomicity. Platt scaling is fitted on
    the inner out-of-fold logits; the final classifier sees every (capped) pair.
    """
    counties = np.asarray(counties)
    require(len(F0) == len(F1) == len(counties) and F0.shape[1] == F1.shape[1] == F_eval.shape[1],
            'origin-classifier inputs are misaligned')
    keep = np.arange(len(F0))
    if cap is not None:
        require(ids is not None and len(ids) == len(F0), 'capped origin fits require row ids')
        keep = deterministic_subsample_idx(np.asarray(ids), cap=cap, salt=f'{salt}-{seed}')
    X_pair = np.vstack([F0[keep], F1[keep]])
    y_pair = np.concatenate([np.zeros(len(keep), dtype=int), np.ones(len(keep), dtype=int)])
    c_pair = np.concatenate([counties[keep], counties[keep]])
    require(len(np.unique(c_pair)) >= inner_splits, 'fewer counties than inner folds')

    oof = np.zeros(len(y_pair), dtype=np.float64)
    inner = []
    for i, (tr, va) in enumerate(GroupKFold(n_splits=inner_splits).split(X_pair, y_pair, groups=c_pair)):
        shared = set(c_pair[tr].tolist()) & set(c_pair[va].tolist())
        require(not shared, f'inner fold {i} shares counties')
        model = make_classifier(seed + i)
        model.fit(X_pair[tr], y_pair[tr])
        oof[va] = _raw_logits(model, X_pair[va], score)
        inner.append(dict(n_train=int(len(tr)), n_val=int(len(va)), shared_counties=len(shared)))
    final = make_classifier(seed + 99)
    final.fit(X_pair, y_pair)
    ratio, platt = platt_calibrate_odds(oof, y_pair, _raw_logits(final, F_eval, score))
    return ratio, dict(platt, n_originals=int(len(keep)), n_pairs=int(len(y_pair)), inner_folds=inner)


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WithinStateInputs:
    """Eligible (target or training) rows; ``target`` rows are scored, every row may train Q."""
    ids: np.ndarray
    a: np.ndarray
    y: np.ndarray
    X: np.ndarray
    counties: np.ndarray
    states: np.ndarray
    target: np.ndarray
    moved: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ids)
        require(all(len(v) == n for v in (self.a, self.y, self.X, self.counties, self.states, self.target,
                                          self.moved)), 'within-state inputs are misaligned')
        require(len(set(np.asarray(self.ids).tolist())) == n, 'duplicate row ids')
        require(np.isfinite(self.a).all() and np.isfinite(self.y).all(), 'nonfinite exposure or outcome')
        require(bool(np.asarray(self.target).any()), 'no targets')

    def shifted(self, delta: float) -> np.ndarray:
        return self.a + np.where(self.moved, delta, 0.0)


@dataclass
class CrossFitResult:
    target_ids: np.ndarray
    fold_ids: np.ndarray
    mu_a: dict[str, np.ndarray] = field(default_factory=dict)
    mu_d: dict[str, np.ndarray] = field(default_factory=dict)
    r_a: dict[str, np.ndarray] = field(default_factory=dict)
    platt: dict[str, list[dict]] = field(default_factory=dict)
    folds: list[dict] = field(default_factory=list)


def _r2(y: np.ndarray, mu: np.ndarray) -> float:
    total = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0.0
    return float(1.0 - np.sum((y - mu) ** 2) / total) if total > 0 else float('nan')


def fit_cross_fitted_within_state(
    data: WithinStateInputs, splits: Sequence[tuple[np.ndarray, np.ndarray]], *, delta: float,
    q_learners: Sequence[QLearner], g_learners: Sequence[GLearner],
    all_states: Sequence[Any] | None = None, log: Callable[[str], None] | None = None,
) -> CrossFitResult:
    """Out-of-fold ``mu_a``, ``mu_d`` and ``r_a`` at every target for each learner.

    Per outer fold: median imputation and state statistics come from the training
    partition only; Q trains on every training row; g trains on training targets
    (the target law), pairing each with its shifted copy.
    """
    require(np.isfinite(delta) and delta > 0, 'delta must be positive')
    names = [q.name for q in q_learners] + [g.name for g in g_learners]
    require(len(set(q.name for q in q_learners)) == len(q_learners)
            and len(set(g.name for g in g_learners)) == len(g_learners), f'duplicate learner names {names}')
    n = len(data.ids)
    evaluated = np.concatenate([ev for _, ev in splits]) if splits else np.array([], dtype=int)
    require(len(evaluated) == n and set(evaluated.tolist()) == set(range(n)),
            'outer folds must evaluate every row exactly once')
    all_states = sorted(np.unique(data.states).tolist()) if all_states is None else list(all_states)
    target = np.asarray(data.target, dtype=bool)
    t_pos = np.full(n, -1, dtype=int)
    t_pos[target] = np.arange(int(target.sum()))
    n_t = int(target.sum())
    a_d = data.shifted(delta)

    result = CrossFitResult(target_ids=np.asarray(data.ids)[target], fold_ids=np.full(n_t, -1, dtype=int))
    for q in q_learners:
        result.mu_a[q.name], result.mu_d[q.name] = np.zeros(n_t), np.zeros(n_t)
    for g in g_learners:
        result.r_a[g.name], result.platt[g.name] = np.zeros(n_t), []

    for fold, (tr, ev) in enumerate(splits):
        started = time.monotonic()
        require(set(data.counties[tr].tolist()).isdisjoint(data.counties[ev].tolist()),
                f'fold {fold} county leakage')
        ev_t, tr_t = ev[target[ev]], tr[target[tr]]
        slots = t_pos[ev_t]
        result.fold_ids[slots] = fold
        moved_ev = data.moved[ev_t]
        record: dict[str, Any] = dict(fold=fold, n_train=int(len(tr)), n_train_targets=int(len(tr_t)),
                                      n_eval_targets=int(len(ev_t)), shared_counties=0, q_context_rows={},
                                      skipped=len(ev_t) == 0)
        if record['skipped']:
            # Nothing to predict: no nuisance is fitted, so the fold costs nothing and changes no target.
            record['seconds'] = time.monotonic() - started
            result.folds.append(record)
            if log is not None:
                log(f'[fold {fold}] skipped: no evaluation targets')
            continue
        imputer = SimpleImputer(strategy='median').fit(data.X[tr])
        X_tr, X_tr_t, X_ev_t = (imputer.transform(data.X[rows]) for rows in (tr, tr_t, ev_t))

        stats = fit_state_stats(data.a[tr], data.states[tr])
        Fq_tr = encode_features_onehot(data.a[tr], X_tr, data.states[tr], stats, all_states)
        Fq_a = encode_features_onehot(data.a[ev_t], X_ev_t, data.states[ev_t], stats, all_states)
        Fq_d = encode_features_onehot(a_d[ev_t], X_ev_t, data.states[ev_t], stats, all_states)
        for q in q_learners:
            keep = (np.arange(len(tr)) if q.context_cap is None else
                    deterministic_subsample_idx(data.ids[tr], cap=q.context_cap, salt=f'{q.salt}-f{fold}'))
            model = q.factory(q.seed + fold)
            model.fit(Fq_tr[keep], data.y[tr][keep])
            mu_a = np.asarray(model.predict(Fq_a), dtype=np.float64).reshape(-1)
            mu_d = np.where(moved_ev, np.asarray(model.predict(Fq_d), dtype=np.float64).reshape(-1), mu_a)
            require(np.isfinite(mu_a).all() and np.isfinite(mu_d).all(), f'{q.name}: nonfinite prediction')
            result.mu_a[q.name][slots], result.mu_d[q.name][slots] = mu_a, mu_d
            record['q_context_rows'][q.name] = int(len(keep))

        if g_learners:
            stats_t = fit_state_stats(data.a[tr_t], data.states[tr_t])
            Fg_0 = encode_features_onehot(data.a[tr_t], X_tr_t, data.states[tr_t], stats_t, all_states)
            Fg_1 = encode_features_onehot(a_d[tr_t], X_tr_t, data.states[tr_t], stats_t, all_states)
            Fg_ev = encode_features_onehot(data.a[ev_t], X_ev_t, data.states[ev_t], stats_t, all_states)
        for g in g_learners:
            ratio, diag = fit_platt_calibrated_origin(
                Fg_0, Fg_1, Fg_ev, data.counties[tr_t], g.factory, seed=g.seed + 10 * fold,
                ids=data.ids[tr_t], cap=g.original_cap, salt=g.salt, inner_splits=g.inner_splits, score=g.score)
            require(np.isfinite(ratio).all() and (ratio > 0).all(), f'{g.name}: invalid density ratio')
            result.r_a[g.name][slots] = ratio
            result.platt[g.name].append(dict(diag, fold=fold))

        record['seconds'] = time.monotonic() - started
        result.folds.append(record)
        if log is not None:
            y_ev = data.y[ev_t]
            q_text = ' '.join(f'{q.name}_R2={_r2(y_ev, result.mu_a[q.name][slots]):.4f}' for q in q_learners)
            g_text = ' '.join(f'{g.name}(mean/max/slope)={result.r_a[g.name][slots].mean():.3f}/'
                              f'{result.r_a[g.name][slots].max():.1f}/{result.platt[g.name][-1]["slope"]:.2f}'
                              for g in g_learners if len(slots))
            log(f'[fold {fold}] ({record["seconds"]:.1f}s) n={len(slots)} {q_text} {g_text}')
    require((result.fold_ids >= 0).all(), 'a target was never evaluated')
    return result


# ---------------------------------------------------------------------------
# Inference and reporting
# ---------------------------------------------------------------------------

def haversine_km(coords_rad: np.ndarray) -> np.ndarray:
    lat, lon = coords_rad[:, 0:1], coords_rad[:, 1:2]
    h = np.sin((lat - lat.T) / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def compute_conley_inference(eif_centered: np.ndarray, counties: np.ndarray,
                             county_locs: Mapping[str, tuple[float, float]],
                             bandwidths_km: Sequence[float] = CONLEY_BANDWIDTHS_KM) -> dict:
    """SEs of the mean of ``eif_centered``: iid, county-clustered, and spatial Conley.

    County scores ``U_g = sum_i u_i / n`` enter a Bartlett kernel ``max(0, 1 - d / bw)``
    on county-centroid distance. Counties without a centroid only correlate with
    themselves. Each spatial variance is floored at the clustered variance, so a
    spatial SE is never below the clustered SE. ``s_max`` is the largest county share
    ``U_g^2 / sum U_g^2`` and ``G_eff = 1 / sum s_g^2``.
    """
    eif_centered = np.asarray(eif_centered, dtype=np.float64)
    counties = np.asarray(counties).astype(str)
    n = len(eif_centered)
    require(n > 0 and counties.shape == (n,), 'inference inputs are misaligned')
    u = eif_centered / n
    labels, inverse = np.unique(counties, return_inverse=True)
    U = np.bincount(inverse, weights=u, minlength=len(labels))
    D = float(np.sum(U ** 2))
    shares = U ** 2 / D if D > 0 else np.zeros_like(U)
    concentration = float(np.sum(shares ** 2))
    order = np.argsort(shares, kind='stable')[::-1][:5]
    located = np.asarray([c in county_locs for c in labels.tolist()], dtype=bool)
    coords = np.radians(np.asarray([county_locs[c] if ok else (0.0, 0.0)
                                    for c, ok in zip(labels.tolist(), located)], dtype=np.float64).reshape(-1, 2))
    distance = haversine_km(coords)
    out = dict(se_iid=float(np.sqrt(np.sum(u ** 2))), se_cluster=float(np.sqrt(D)), n_counties=int(len(labels)),
               loc_coverage=float(located.mean()) if len(labels) else 0.0)
    for bw in bandwidths_km:
        K = np.maximum(0.0, 1.0 - distance / bw)
        K[~located, :] = 0.0
        K[:, ~located] = 0.0
        np.fill_diagonal(K, 1.0)
        out[f'se_spatial_{int(bw)}km'] = float(np.sqrt(max(float(U @ (K @ U)), D)))
    out.update(s_max=float(shares.max()) if len(shares) else 0.0,
               g_eff=float(1.0 / concentration) if concentration > 0 else 0.0,
               top5_counties=[dict(county=str(labels[i]), U_g=float(U[i]), s_g=float(shares[i])) for i in order])
    return out


def _landing_ess(r_a: np.ndarray) -> tuple[float, float]:
    """Kish ESS of the positive excess weights ``max(r_a - 1, 0)``, and its percentage of their count."""
    excess = np.maximum(r_a - 1.0, 0.0)
    positive = excess[excess > 0]
    if not len(positive):
        return 0.0, 0.0
    ess = float(positive.sum() ** 2 / np.sum(positive ** 2))
    return ess, float(100.0 * ess / len(positive))


def _wmse(r_a: np.ndarray, residual_sq: np.ndarray) -> float:
    w = (r_a - 1.0) ** 2
    return float(np.sum(w * residual_sq) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(residual_sq))


def score_within_state_estimate(
    label: str, y: np.ndarray, mu_a: np.ndarray, mu_d: np.ndarray, r_a: np.ndarray, fold_ids: np.ndarray,
    counties: np.ndarray, county_locs: Mapping[str, tuple[float, float]], delivered_shift: float,
    flags: np.ndarray | None = None,
) -> dict:
    """One-step and Hajek estimates with every permanent reporting invariant.

    ``psi = mean(mu_d - mu_a + (r_a - 1)(y - mu_a))``; per-mmHg values divide by the
    delivered shift. Flag strata (USALEEP abridged life-table flag 1 vs 3) are post-hoc
    descriptive only; the flag is never a model feature.
    """
    y, mu_a, mu_d, r_a = (np.asarray(v, dtype=np.float64) for v in (y, mu_a, mu_d, r_a))
    fold_ids, counties = np.asarray(fold_ids), np.asarray(counties)
    n = len(y)
    require(all(len(v) == n for v in (mu_a, mu_d, r_a, fold_ids, counties)), 'scoring inputs are misaligned')
    require(np.isfinite(delivered_shift) and delivered_shift > 0, 'delivered shift must be positive')
    require(np.isfinite(r_a).all() and (r_a > 0).all(), 'density ratio must be positive and finite')
    plugin = mu_d - mu_a
    correction = (r_a - 1.0) * (y - mu_a)
    eif = plugin + correction
    psi = float(np.mean(eif))
    inf = compute_conley_inference(eif - psi, counties, county_locs)
    # Hajek: psi = mean(plugin) + mean(r e) / mean(r) - mean(e), e = y - mu_a. Its influence
    # function includes the delta-method term for estimating mean(r): (r / r_bar)(e - c).
    mean_r = float(np.mean(r_a))
    residual = y - mu_a
    c_hajek = float(np.mean(r_a * residual)) / mean_r
    psi_hajek = float(np.mean(plugin)) + c_hajek - float(np.mean(residual))
    eif_hajek = plugin + (r_a / mean_r) * (residual - c_hajek) - residual
    inf_hajek = compute_conley_inference(eif_hajek - np.mean(eif_hajek), counties, county_locs)
    residual_sq = (y - mu_a) ** 2
    ess, ess_pct = _landing_ess(r_a)
    per_mmhg = lambda v: v / delivered_shift  # noqa: E731

    per_fold = {}
    for f in np.unique(fold_ids).tolist():
        m = fold_ids == f
        ess_f, ess_pct_f = _landing_ess(r_a[m])
        per_fold[str(f)] = dict(n=int(m.sum()), plugin=float(np.mean(plugin[m])),
            correction=float(np.mean(correction[m])), one_step=float(np.mean(eif[m])),
            r2=_r2(y[m], mu_a[m]), wmse=_wmse(r_a[m], residual_sq[m]), mean_r_a=float(np.mean(r_a[m])),
            max_r_a=float(np.max(r_a[m])), landing_ess=ess_f, landing_ess_pct=ess_pct_f)

    strata = {}
    if flags is not None:
        flags = np.asarray(flags)
        require(flags.shape == (n,), 'flags are misaligned')
        for flag in (1, 3):
            m = flags == flag
            if m.sum() >= MIN_STRATUM_ROWS:
                est = float(np.mean(eif[m]))
                inf_f = compute_conley_inference(eif[m] - est, counties[m], county_locs)
                strata[f'flag_{flag}'] = dict(n=int(m.sum()), plugin=float(np.mean(plugin[m])),
                    correction=float(np.mean(correction[m])), one_step=est, per_mmhg=per_mmhg(est),
                    se_cluster_per_mmhg=per_mmhg(inf_f['se_cluster']),
                    se_sp100_per_mmhg=per_mmhg(inf_f['se_spatial_100km']))

    sp100 = inf['se_spatial_100km']
    return dict(
        label=label, n=n, delivered_shift_mmhg=float(delivered_shift), r2=_r2(y, mu_a),
        wmse=_wmse(r_a, residual_sq), mean_r_a=mean_r, p99_r_a=float(np.percentile(r_a, 99)),
        max_r_a=float(np.max(r_a)), landing_ess=ess, landing_ess_pct=ess_pct,
        plugin=float(np.mean(plugin)), correction=float(np.mean(correction)),
        one_step=dict(psi=psi, per_mmhg=per_mmhg(psi), se_iid_per_mmhg=per_mmhg(inf['se_iid']),
                      se_cluster_per_mmhg=per_mmhg(inf['se_cluster']),
                      se_sp50_per_mmhg=per_mmhg(inf['se_spatial_50km']), se_sp100_per_mmhg=per_mmhg(sp100),
                      se_sp250_per_mmhg=per_mmhg(inf['se_spatial_250km']),
                      ci95_sp100_per_mmhg=[per_mmhg(psi - Z95 * sp100), per_mmhg(psi + Z95 * sp100)],
                      inference=inf),
        hajek_one_step=dict(psi=psi_hajek, per_mmhg=per_mmhg(psi_hajek),
                            se_cluster_per_mmhg=per_mmhg(inf_hajek['se_cluster']),
                            se_sp100_per_mmhg=per_mmhg(inf_hajek['se_spatial_100km']), inference=inf_hajek),
        per_fold=per_fold, flag_stratified_descriptive=strata)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DEFAULT_COMBOS = (
    (PRIMARY_LABEL, 'A6-Q', 'A7-g'),
    ('A6-Q + B0_gbm_origin (g sensitivity)', 'A6-Q', 'B0_gbm_origin'),
    ('A6-Q + B0_logit_origin (smooth g sensitivity)', 'A6-Q', 'B0_logit_origin'),
    ('B0_hist_gbm + A7-g (Q sensitivity)', 'B0_hist_gbm', 'A7-g'),
    ('B0_hist_gbm + B0_gbm_origin (classical)', 'B0_hist_gbm', 'B0_gbm_origin'),
)


def run_within_state(
    data: WithinStateInputs, *, delta: float, county_locs: Mapping[str, tuple[float, float]],
    q_learners: Sequence[QLearner], g_learners: Sequence[GLearner],
    null_learners: tuple[Sequence[QLearner], Sequence[GLearner]] | None = None,
    combos: Sequence[tuple[str, str, str]] = DEFAULT_COMBOS, flags: np.ndarray | None = None,
    n_splits: int = 5, null_seed: int = NULL_SEED, log: Callable[[str], None] | None = None,
) -> dict:
    """Cross-fit every learner on shared folds, then score each (Q, g) combination and the null.

    ``flags`` aligns with the target rows. The null permutes A within state (moved fixed),
    reuses the outer folds and scores its first Q with its first g.
    """
    require(n_splits >= 2, 'at least two outer folds required')
    splits = make_balanced_county_folds(data.counties, data.states, data.a, n_splits)
    all_states = sorted(np.unique(data.states).tolist())
    n_t = int(np.asarray(data.target).sum())
    n_m = int(np.asarray(data.moved)[np.asarray(data.target)].sum())
    require(n_m > 0, 'no movers among targets')
    delivered = float(delta * n_m / n_t)
    fits = fit_cross_fitted_within_state(data, splits, delta=delta, q_learners=q_learners,
                                         g_learners=g_learners, all_states=all_states, log=log)
    target = np.asarray(data.target, dtype=bool)
    y_t, c_t = data.y[target], np.asarray(data.counties)[target]
    results = {}
    for label, q, g in combos:
        require(q in fits.mu_a and g in fits.r_a, f'{label}: unknown learner {q} or {g}')
        results[label] = score_within_state_estimate(label, y_t, fits.mu_a[q], fits.mu_d[q], fits.r_a[g],
                                                     fits.fold_ids, c_t, county_locs, delivered, flags)
    null = None
    if null_learners is not None:
        nq, ng = null_learners
        require(bool(nq) and bool(ng), 'null requires a Q and a g learner')
        a_null, moved_null = permute_within_state(data.a, data.moved, data.states, null_seed)
        null_delivered = float(delta * moved_null[target].mean())
        null_data = WithinStateInputs(ids=data.ids, a=a_null, y=data.y, X=data.X, counties=data.counties,
                                      states=data.states, target=data.target, moved=moved_null)
        null = fit_cross_fitted_within_state(null_data, splits, delta=delta, q_learners=nq, g_learners=ng,
                                             all_states=all_states, log=log)
        results[NULL_LABEL] = score_within_state_estimate(
            NULL_LABEL, y_t, null.mu_a[nq[0].name], null.mu_d[nq[0].name], null.r_a[ng[0].name],
            null.fold_ids, c_t, county_locs, null_delivered, flags)
    return dict(results=results, fits=fits, null=null, n_targets=n_t, n_movers=n_m, delivered_shift=delivered,
                null_delivered_shift=null_delivered if null is not None else None,
                folds=fits.folds, platt=fits.platt)


# ---------------------------------------------------------------------------
# Real-data loading and CLI
# ---------------------------------------------------------------------------

def load_geography_metadata(path: str | Path) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """County centroids (mean tract latitude/longitude) and tract outcome flags from geography.json."""
    rows = json.loads(Path(path).read_text())['payload']['rows']
    lats: dict[str, list[float]] = {}
    lons: dict[str, list[float]] = {}
    flags: dict[str, int] = {}
    for row in rows:
        oid = str(row['original_id']).strip().zfill(11)
        county = str(row['county']).strip().zfill(5)
        flags[oid] = int(row['outcome_flag'])
        lat, lon = row.get('latitude'), row.get('longitude')
        if lat is not None and lon is not None:
            lats.setdefault(county, []).append(float(lat))
            lons.setdefault(county, []).append(float(lon))
    return {c: (float(np.mean(lats[c])), float(np.mean(lons[c]))) for c in lats}, flags


@dataclass(frozen=True)
class EndpointFrame:
    ids: np.ndarray
    a: np.ndarray
    y: np.ndarray
    X: np.ndarray
    counties: np.ndarray
    states: np.ndarray
    eligible: np.ndarray
    feature_names: tuple[str, ...]


def endpoint_frame(endpoint: Any) -> EndpointFrame:
    """Frame arrays from a prepared endpoint; eligible = outer role ``target`` or ``training``."""
    manifest = endpoint.data.manifest
    names = tuple(name for name, _ in endpoint.feature_kinds)
    ids = np.asarray(manifest.original_ids, dtype=object)
    a = np.asarray(endpoint.data.column(manifest.exposure_field), dtype=np.float64)
    y = np.asarray([np.nan if v is None else float(v) for v in endpoint.data.column(manifest.outcome_field)])
    counties = np.asarray([str(c).zfill(5) for c in endpoint.data.county_routing(endpoint.county_field)])
    X = np.asarray([[np.nan if v is None else float(v) for v in row]
                    for row in endpoint.data.covariates(names).values], dtype=np.float64)
    roles = dict(zip(endpoint.outer.frame_ids, endpoint.outer.membership_roles))
    eligible = np.asarray([roles.get(oid) in ('target', 'training') for oid in ids], dtype=bool)
    return EndpointFrame(ids=ids, a=a, y=y, X=X, counties=counties, states=np.asarray([c[:2] for c in counties]),
                         eligible=eligible, feature_names=names)


def default_checkpoint(filename: str) -> Path:
    hub = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache' / 'huggingface' / 'hub'))
    return hub / f'models--{TABICL_REPOSITORY.replace("/", "--")}' / 'snapshots' / TABICL_REVISION / filename


def verify_checkpoint(path: str | Path, filename: str) -> Path:
    """Require the registered TabICL identity: filename, revision pin and SHA-256."""
    from oxyformer.models.tabicl_comparator import REGISTERED_CHECKPOINTS
    path = Path(path)
    registered = {(r[4], r[5]) for r in REGISTERED_CHECKPOINTS
                  if r[0] == 'tabicl' and r[2] == TABICL_REPOSITORY and r[3] == TABICL_REVISION}
    digests = [d for name, d in registered if name == filename]
    require(len(digests) == 1, f'{filename} is not a registered TabICL checkpoint')
    require(path.name == filename and path.is_file(), f'{path}: local {filename} required; downloads are forbidden')
    require(file_hash(path) == digests[0], f'{path}: checkpoint SHA-256 mismatch')
    return path


def check_support_settings(amendment: Mapping[str, Any], approvals: Mapping[str, Any], delta: float,
                           caliper: float) -> None:
    """The amendment's recipe runs as-is; other delta/caliper need the sensitivity approval bounds."""
    from oxyformer.explore import real_fit
    recipe = amendment['support_recipe']
    design = dict(support=dict(bin_width_mmhg=float(recipe['bin_width_mmhg']),
                               min_bin_tracts=int(recipe['min_bin_tracts']),
                               raw_x_radius=float(recipe['raw_x_radius_iqr'])))
    if (delta, caliper) != (float(recipe['delta_mmhg']), float(recipe['raw_x_radius_iqr'])):
        real_fit.validate_support_settings(design, approvals, delta, caliper)


def _readout(payload: dict, wall_seconds: float) -> str:
    lines = [f'# Exploratory readout: `{ESTIMAND_ID}`', '',
             f'**Exploratory; certifies no release criterion.** Owner decision `{OWNER_DECISION_ID}`.',
             'G-SIM passed A7_tabicl_origin on G2_within_state (Slurm 175341); B0_gbm_origin, which also '
             'passed, is the g sensitivity comparator.',
             f'Support zero-IQR mode: `{payload["config"]["zero_iqr_mode"]}`.', '',
             f'Identification: {payload["identification_disclosure"]}', '',
             f'Support: `{payload["n_targets"]:,}` targets, `{payload["n_movers"]:,}` movers; nominal shift '
             f'`{payload["delta"]}` mmHg, **delivered shift `{payload["delivered_shift_mmhg"]:.4f}` mmHg** '
             f'(estimates per delivered mmHg). Wall time `{wall_seconds:.1f}` s.',
             f'Census vs approved outcome-blind census: `{payload["config"]["census"]}`.', '',
             '## One-step estimates', '',
             '| Model | OOF R^2 | wMSE | Plug-in (y) | Correction (y) | One-step (y/mmHg) | Cluster SE | '
             'Sp-50 SE | Sp-100 SE | Sp-250 SE | 95% CI (Sp-100) | Mean / p99 / max r_a | Landing ESS (%) | '
             's_max / G_eff |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for label, res in payload['results'].items():
        o, inf = res['one_step'], res['one_step']['inference']
        lo, hi = o['ci95_sp100_per_mmhg']
        lines.append(
            f'| `{label}` | {res["r2"]:.4f} | {res["wmse"]:.3f} | {res["plugin"]:+.4f} | {res["correction"]:+.4f} | '
            f'**{o["per_mmhg"]:+.4f}** | {o["se_cluster_per_mmhg"]:.4f} | {o["se_sp50_per_mmhg"]:.4f} | '
            f'{o["se_sp100_per_mmhg"]:.4f} | {o["se_sp250_per_mmhg"]:.4f} | [{lo:+.4f}, {hi:+.4f}] | '
            f'{res["mean_r_a"]:.3f} / {res["p99_r_a"]:.2f} / {res["max_r_a"]:.1f} | '
            f'{res["landing_ess"]:.0f} ({res["landing_ess_pct"]:.1f}%) | {inf["s_max"]:.3f} / {inf["g_eff"]:.1f} |')
    for label, res in payload['results'].items():
        lines += ['', f'## Per-fold decomposition: `{label}`', '',
                  '| Fold | n | OOF R^2 | wMSE | Plug-in (y) | Correction (y) | One-step (y) | Mean r_a | Max r_a | '
                  'Landing ESS (%) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for f, v in res['per_fold'].items():
            lines.append(f'| {f} | {v["n"]:,} | {v["r2"]:.4f} | {v["wmse"]:.3f} | {v["plugin"]:+.4f} | '
                         f'{v["correction"]:+.4f} | {v["one_step"]:+.4f} | {v["mean_r_a"]:.3f} | '
                         f'{v["max_r_a"]:.1f} | {v["landing_ess"]:.0f} ({v["landing_ess_pct"]:.1f}%) |')
        if res['flag_stratified_descriptive']:
            lines += ['', 'Post-hoc flag strata (descriptive; flag never in X):', '']
            for k, v in res['flag_stratified_descriptive'].items():
                lines.append(f'- `{k}`: n={v["n"]:,}, plug-in {v["plugin"]:+.4f}, correction {v["correction"]:+.4f}, '
                             f'one-step {v["per_mmhg"]:+.4f} y/mmHg (cluster SE {v["se_cluster_per_mmhg"]:.4f}, '
                             f'Sp-100 SE {v["se_sp100_per_mmhg"]:.4f})')
    lines += ['', '## Hajek-normalised ratio sensitivity (`r_a / mean(r_a)`)', '']
    for label, res in payload['results'].items():
        h = res['hajek_one_step']
        lines.append(f'- `{label}`: {h["per_mmhg"]:+.4f} ± {h["se_sp100_per_mmhg"]:.4f} y/mmHg (Sp-100; cluster '
                     f'{h["se_cluster_per_mmhg"]:.4f}; raw mean r_a {res["mean_r_a"]:.4f})')
    return '\n'.join(lines) + '\n'


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m oxyformer.explore.within_state', description=__doc__.split('\n')[0])
    parser.add_argument('--prepared-dir', type=Path, required=True, help='prepared real endpoint directory')
    parser.add_argument('--geography-json', type=Path, required=True, help='tract-inputs geography.json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--delta', type=float, default=2.0, help='nominal shift in mmHg')
    parser.add_argument('--caliper', type=float, default=2.0, help='raw-X caliper in IQR units')
    parser.add_argument('--zero-iqr-mode', choices=ZERO_IQR_MODES, default='unit_scale',
                        help="zero-IQR covariates: 'unit_scale' (scale 1.0) or 'exact' (design/support.py equality)")
    parser.add_argument('--wandb', action='store_true', help='log the run to W&B (goodarzilab/oxyformer-v2)')
    parser.add_argument('--device', default=None, help='torch device (default: cuda:0 when available)')
    parser.add_argument('--regressor-checkpoint', type=Path, default=default_checkpoint(TABICL_REGRESSOR_FILE))
    parser.add_argument('--classifier-checkpoint', type=Path, default=default_checkpoint(TABICL_CLASSIFIER_FILE))
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--skip-null', action='store_true', help='skip the permuted-A falsification null')
    parser.add_argument('--allow-census-drift', action='store_true',
                        help='run although the support census differs from the approved outcome-blind census')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from oxyformer.design import gate
    from oxyformer.execution.runner import read_mapping
    from oxyformer.explore import real_fit

    started = time.monotonic()
    out = real_fit.write_directory(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    approvals = read_mapping(gate.OWNER_APPROVALS)
    amendment = approvals['owner_decisions']['within_state_primary_amendment']
    require(amendment['new_estimand_id'] == ESTIMAND_ID, 'within-state amendment missing')
    check_support_settings(amendment, approvals, args.delta, args.caliper)
    recipe = amendment['support_recipe']

    county_locs, flag_map = load_geography_metadata(args.geography_json)
    endpoint, _, audit = real_fit.load_prepared(args.prepared_dir)
    frame = endpoint_frame(endpoint)
    support = compute_within_state_support(frame.a, frame.X, frame.states, frame.eligible, delta=args.delta,
                                           caliper=args.caliper, bin_width=float(recipe['bin_width_mmhg']),
                                           min_bin_tracts=int(recipe['min_bin_tracts']),
                                           zero_iqr_mode=args.zero_iqr_mode)
    target = support.supported & frame.eligible
    approved = amendment['outcome_blind_census']
    census = dict(targets=support.n_targets, movers=support.n_movers, delivered_shift_mmhg=support.delivered_shift,
                  approved_targets=approved['target_tracts'], approved_movers=approved['mover_tracts'],
                  zero_iqr_mode=args.zero_iqr_mode,
                  zero_iqr_columns=[n for n, z in zip(frame.feature_names, support.zero_iqr_columns) if z],
                  approved_recipe=(args.delta, args.caliper) == (float(recipe['delta_mmhg']),
                                                                 float(recipe['raw_x_radius_iqr'])))
    census['matches_approval'] = census['approved_recipe'] and (support.n_targets, support.n_movers) == (
        approved['target_tracts'], approved['mover_tracts'])
    require(census['matches_approval'] or not census['approved_recipe'] or args.allow_census_drift,
            f'support census {support.n_targets}/{support.n_movers} differs from the approved '
            f'{approved["target_tracts"]}/{approved["mover_tracts"]}; pass --allow-census-drift to run and record it')
    eligible = frame.eligible
    require(np.isfinite(frame.y[eligible]).all(), 'nonfinite outcomes among eligible rows')
    data = WithinStateInputs(ids=frame.ids[eligible], a=frame.a[eligible], y=frame.y[eligible], X=frame.X[eligible],
                             counties=frame.counties[eligible], states=frame.states[eligible],
                             target=target[eligible], moved=support.moved[eligible])
    t_ids = data.ids[data.target]
    flags = np.asarray([flag_map.get(str(oid).strip().zfill(11), -1) for oid in t_ids.tolist()], dtype=int)

    import torch
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    reg = verify_checkpoint(args.regressor_checkpoint, TABICL_REGRESSOR_FILE)
    cls = verify_checkpoint(args.classifier_checkpoint, TABICL_CLASSIFIER_FILE)
    config = dict(LABELS, delta=args.delta, caliper=args.caliper, zero_iqr_mode=args.zero_iqr_mode,
                  n_splits=args.n_splits, census=census,
                  n_targets=support.n_targets, n_movers=support.n_movers, delivered_shift_mmhg=support.delivered_shift,
                  prepared_hash=audit['prepared_hash'], checkout=real_fit.checkout_identity(), device=device,
                  regressor=TABICL_REGRESSOR_FILE, classifier=TABICL_CLASSIFIER_FILE, revision=TABICL_REVISION)
    run = None
    if args.wandb:
        wandb = import_module('wandb')
        run = wandb.init(entity='goodarzilab', project='oxyformer-v2', name=f'explore-{ESTIMAND_ID}',
                         config=config, dir=str(out), tags=['exploratory', 'within-state-d2-c2', 'A6-Q', 'A7-g'])
    print(f'{support.n_targets} targets, {support.n_movers} movers, delivered {support.delivered_shift:.4f} mmHg; '
          f'{len(county_locs)} county centroids', flush=True)
    outcome = run_within_state(
        data, delta=args.delta, county_locs=county_locs, q_learners=default_q_learners(reg, device),
        g_learners=default_g_learners(cls, device),
        null_learners=None if args.skip_null else default_null_learners(reg, cls, device),
        flags=flags, n_splits=args.n_splits, log=lambda line: print(line, flush=True))

    fits, null = outcome['fits'], outcome['null']
    arrays = dict(target_ids=t_ids.astype(str), counties=data.counties[data.target], flags=flags,
                  fold_ids=fits.fold_ids, y=data.y[data.target], moved=data.moved[data.target])
    for kind in ('mu_a', 'mu_d', 'r_a'):
        for name, values in getattr(fits, kind).items():
            arrays[f'{kind}__{name}'] = values
        for name, values in (getattr(null, kind).items() if null is not None else ()):
            arrays[f'null_{kind}__{name}'] = values
    np.savez_compressed(out / 'oof_nuisances.npz', **arrays)
    payload = dict(LABELS, config=config, delta=args.delta, n_targets=outcome['n_targets'],
                   n_movers=outcome['n_movers'], delivered_shift_mmhg=outcome['delivered_shift'],
                   identification_disclosure=amendment['identification_disclosure'], results=outcome['results'],
                   folds=outcome['folds'], platt=outcome['platt'],
                   null_platt=null.platt if null is not None else None)
    (out / f'{ESTIMAND_ID}.json').write_text(json.dumps(payload, indent=2) + '\n')
    readout = _readout(payload, time.monotonic() - started)
    (out / f'{ESTIMAND_ID}.md').write_text(readout)
    print(readout, flush=True)
    if run is not None:
        for label, res in outcome['results'].items():
            run.log({f'{label}/{k}': v for k, v in dict(
                per_mmhg=res['one_step']['per_mmhg'], se_cluster_per_mmhg=res['one_step']['se_cluster_per_mmhg'],
                se_sp100_per_mmhg=res['one_step']['se_sp100_per_mmhg'], r2=res['r2'], wmse=res['wmse'],
                mean_r_a=res['mean_r_a'], max_r_a=res['max_r_a'], landing_ess_pct=res['landing_ess_pct']).items()})
        run.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
