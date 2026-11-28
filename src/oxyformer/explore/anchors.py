"""Exploratory external altitude-response anchors: Peru ENDES, Ecuador INEC, Colombia DANE.

Exploratory; certifies no release criterion.

This module packages three standalone anchor runs as reusable, tested code:

* Peru ENDES 2023-2024 child haemoglobin (owner decision
  ``50386c29-4341-5098-80bc-3f13588f9956``): raw unadjusted ``HC53 / 10`` g/dL with
  ``HC55 == 0`` and child age 4-59 months. WHO altitude-adjusted ``HC56`` / ``HC57``
  and anthropometry are rejected.
* Ecuador INEC 2024 live births (same owner decision): birthweight and low
  birthweight, placed at maternal residence canton ``cant_res``.
* Colombia DANE 2023-2025 live births (owner, "Go per recommendation",
  2026-11-28T00:08Z): low birthweight, placed at maternal residence municipality
  ``CODPTORE + CODMUNRE``.

``configs/approvals.yaml`` ``owner_decisions.birth_anchor_recipe`` fixes the
residence fields, covariates and exclusions for the two birth anchors. The
interpretation is "consistency of a residence-pressure association with birth
outcomes, not oxygen-specific identification".

Exposure. :func:`elevation_m_to_pio2_deficit_mmhg` applies the approved US Standard
Atmosphere 1976 constants (``owner_decisions.exposure``). Fine elevations are
transformed *before* any population averaging
(:func:`population_weighted_pio2_deficit`). The deficit is concave in elevation, so
averaging elevations first would overstate the exposure of a heterogeneous area
(:func:`jensen_gap_mmhg`).

Estimator. :func:`run_anchor` is a within-stratum modified-treatment-policy one-step
estimator. It reuses the reviewed primitives of
:mod:`oxyformer.explore.within_state` and keeps four invariants:

(a) exact one-hot stratum fixed effects with within-stratum centred exposure and no
    raw ``A`` (``encode_features_onehot(include_raw_a=False)``), so no learner sees
    cross-stratum exposure contrasts;
(b) whole-cluster atomic 5-fold outer CV balanced along each stratum's exposure
    gradient (``make_balanced_county_folds`` with clusters as the atoms);
(c) cluster-grouped inner 3-fold out-of-fold Platt calibration of the origin
    classifiers ``A7-g`` (TabICLv2) and ``B0_gbm_origin``, fitted per weighting scheme
    under that scheme's target law (:func:`fit_weighted_platt_calibrated_origin`);
(d) a cluster-level within-stratum permuted-``A`` falsification null
    (:func:`permute_cluster_within_stratum`). Exposure is assigned per cluster, so it
    is permuted across clusters, never across rows; ``Y``, ``X``, target and moved
    stay fixed.

Continuous outcomes use ``A6-Q`` (frozen TabICLv2 regressor) with a
``HistGradientBoostingRegressor`` comparator. Binary 0/1 outcomes use the frozen
TabICLv2 *classifier* through ``predict_proba[:, 1]`` with a
``HistGradientBoostingClassifier`` comparator (:class:`ProbabilityQ`); a regressor
is never fitted to a 0/1 outcome. Estimates are reported per delivered (not nominal)
mmHg. They are given under equal-cluster weights and equal-unit weights
(Horvitz-Thompson expansion when clusters were subsampled), plus survey weights
where a design supplies them. The per-mmHg estimand is the ratio
``beta = psi / s_bar`` of the effect to the delivered shift
``s_bar = sum w delta moved``, and ``s_bar`` varies with the sample, so standard
errors aggregate the ratio's delta-method influence function by cluster
(:func:`per_mmhg_influence`); the Hajek numerator also carries the delta-method
term for the estimated density-ratio mean. Each estimate carries the permanent reporting invariants: per-fold plug-in
and EIF correction, landing ESS, maximum ``r_a``, ``s_max`` and ``G_eff``, and the
delivered shift.

Support. :func:`compute_cluster_support` reuses ``compute_within_state_support`` on
one row per cluster, so a bin counts as replicated only when at least two *other*
clusters of the stratum occupy it. This is the ``design/support.py`` neighbour
convention, and it is stricter than the standalone scripts, which counted the query
cluster itself (birth anchors) or applied a child-level covariate caliper (Peru).
Censuses can therefore differ slightly from those exploratory runs.

``tabicl``, ``torch`` and ``wandb`` are imported only when a TabICL learner is
built, a device is resolved or a W&B run starts. The module and its offline tests
run without them.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from importlib import import_module
import inspect
import json
from pathlib import Path
import re
from typing import Any, Literal
import unicodedata

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from oxyformer.explore.within_state import (
    MAX_G_ORIGINAL_ROWS,
    MAX_Q_CONTEXT_ROWS,
    ODDS_LOG_CLIP,
    PLATT_C,
    PLATT_LOGIT_CLIP,
    TABICL_CLASSIFIER_FILE,
    TABICL_REGRESSOR_FILE,
    Z95,
    ClassifierFactory,
    CrossFitResult,
    GLearner,
    QLearner,
    RegressorFactory,
    WithinStateInputs,
    WithinStateSupport,
    _raw_logits,
    compute_within_state_support,
    default_checkpoint,
    deterministic_subsample_idx,
    encode_features_onehot,
    fit_cross_fitted_within_state,
    fit_platt_calibrated_origin,
    fit_state_stats,
    gbm_classifier_factory,
    gbm_regressor_factory,
    make_balanced_county_folds,
    tabicl_classifier_factory,
    tabicl_regressor_factory,
    verify_checkpoint,
)

EXPLORATORY_NOTICE = 'Exploratory; certifies no release criterion.'
LABELS = dict(exploratory=True, notice=EXPLORATORY_NOTICE, effect_release_authorized=False,
              certifies_release_criteria=False)
OWNER_DECISION_PERU_ECUADOR = '50386c29-4341-5098-80bc-3f13588f9956'
OWNER_APPROVAL_COLOMBIA = 'owner "Go per recommendation", 2026-11-28T00:08Z'
INTERPRETATION = ('consistency of a residence-pressure association with birth outcomes, '
                  'not oxygen-specific identification')
NULL_SEED = 20261127
EXPOSURE_TOLERANCE_MMHG = 1e-9

OutcomeKind = Literal['continuous', 'binary']
OUTCOME_KINDS = ('continuous', 'binary')


def _check(condition: bool, message: str) -> None:
    """Guardrail check: a violated anchor recipe or input contract raises ``ValueError``."""
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Exposure physics (owner_decisions.exposure; US Standard Atmosphere 1976)
# ---------------------------------------------------------------------------

P0 = 760.0            # sea-level pressure, mmHg
T0 = 288.15           # sea-level temperature, K
L = 0.0065            # tropospheric lapse rate, K/m
g0 = 9.80665          # gravity, m/s^2
M = 0.0289644         # molar mass of air, kg/mol
R = 8.31432           # US Standard Atmosphere 1976 gas constant, J/(mol K)
fO2 = 0.2093          # inspired oxygen fraction
pH2O = 47.0           # water vapour pressure at body temperature, mmHg
PRESSURE_EXPONENT = g0 * M / (R * L)
ELEVATION_DOMAIN_M = (-500.0, 11000.0)   # exposure/physics.py validity domain


def barometric_pressure_mmhg(elevation_m: Any) -> np.ndarray:
    """``P_B(z) = P0 * (1 - L z / T0) ** (g0 M / (R L))`` on the approved validity domain."""
    z = np.asarray(elevation_m, dtype=np.float64)
    lo, hi = ELEVATION_DOMAIN_M
    _check(bool(np.isfinite(z).all()), 'elevation must be finite')
    _check(bool(((z >= lo) & (z <= hi)).all()), f'elevation outside the approved domain [{lo}, {hi}] m')
    return P0 * (1.0 - L * z / T0) ** PRESSURE_EXPONENT


def elevation_m_to_pio2_deficit_mmhg(elevation_m: Any) -> np.ndarray:
    """Inspired-oxygen deficit ``P_IO2(0) - P_IO2(z)`` in mmHg, ``P_IO2 = fO2 (P_B - pH2O)``."""
    return fO2 * (P0 - pH2O) - fO2 * (barometric_pressure_mmhg(elevation_m) - pH2O)


def _positive_weights(weights: Any, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    _check(w.shape == (n,), 'weights are misaligned with elevations')
    _check(bool(np.isfinite(w).all() and (w >= 0).all()), 'weights must be finite and nonnegative')
    return w


def population_weighted_pio2_deficit(elevation_m: Any, weights: Any,
                                     groups: Any = None) -> float | dict[Any, float]:
    """Transform-before-aggregate: each fine cell becomes a deficit, then cells are weight-averaged.

    With ``groups`` (area codes per cell) the result maps each area to its average.
    """
    z = np.asarray(elevation_m, dtype=np.float64).reshape(-1)
    deficit = elevation_m_to_pio2_deficit_mmhg(z)       # transform first, per fine cell
    w = _positive_weights(np.asarray(weights).reshape(-1), len(z))
    if groups is None:
        _check(w.sum() > 0, 'weights sum to zero')
        return float(np.sum(w * deficit) / np.sum(w))
    groups = np.asarray(groups).reshape(-1)
    _check(groups.shape == z.shape, 'groups are misaligned with elevations')
    labels, inverse = np.unique(groups, return_inverse=True)
    total = np.bincount(inverse, weights=w, minlength=len(labels))
    _check(bool((total > 0).all()), 'an area has zero total weight')
    mean = np.bincount(inverse, weights=w * deficit, minlength=len(labels)) / total
    return {label: float(v) for label, v in zip(labels.tolist(), mean.tolist())}


def aggregate_then_transform_pio2_deficit(elevation_m: Any, weights: Any) -> float:
    """The forbidden ordering (average elevation, then transform), kept only as a diagnostic."""
    z = np.asarray(elevation_m, dtype=np.float64).reshape(-1)
    w = _positive_weights(np.asarray(weights).reshape(-1), len(z))
    _check(w.sum() > 0, 'weights sum to zero')
    return float(elevation_m_to_pio2_deficit_mmhg(np.sum(w * z) / np.sum(w)))


def jensen_gap_mmhg(elevation_m: Any, weights: Any) -> float:
    """Transform-first minus aggregate-first; never positive because the deficit is concave in z."""
    return population_weighted_pio2_deficit(elevation_m, weights) - aggregate_then_transform_pio2_deficit(
        elevation_m, weights)


# ---------------------------------------------------------------------------
# Approved anchor recipes and guardrails
# ---------------------------------------------------------------------------

PERU_CHILD_COVARIATES = ('HC1', 'HC27', 'HC61', 'HC62', 'HC64')
PERU_HOUSEHOLD_COVARIATES = ('HV009', 'HV014', 'HV025')
PERU_ASSET_COVARIATES = ('HV201', 'HV205', 'HV206', 'HV207', 'HV208', 'HV209', 'HV213', 'HV214', 'HV215',
                         'HV216', 'HV219', 'HV220', 'HV270', 'HV271')
PERU_BASELINE_COVARIATES = PERU_CHILD_COVARIATES + PERU_HOUSEHOLD_COVARIATES + PERU_ASSET_COVARIATES + ('year',)
# configs/adapters/endes.yaml adjusted_hb_fields (2023 and 2024) plus anthropometry.
PERU_FORBIDDEN_FIELDS = frozenset({
    'HC56', 'HC56A', 'HC57', 'HA56', 'HA56A', 'V456', 'V456A', 'HW56', 'HW56A',
    'HC2', 'HC3', 'HC70', 'HC71', 'HC72', 'HC73'})

ECUADOR_PRIMARY_COVARIATES = ('edad_mad', 'niv_inst', 'etnia')
ECUADOR_SENSITIVITY_COVARIATES = ('est_civil',)
ECUADOR_FORBIDDEN_FIELDS = frozenset({
    'sem_gest', 'con_pren', 'tipo_part', 'apgar1', 'apgar5', 'talla', 'hijos_viv', 'hijos_mvt', 'hijos_mva',
    'num_emb', 'num_par', 'hij_viv', 'hij_vivm', 'hij_nacm',
    'prov_nac', 'cant_nac', 'parr_nac', 'prov_insc', 'cant_insc', 'parr_insc'})
ECUADOR_OUTCOMES = {'peso_g': 'continuous', 'lbw': 'binary'}
ECUADOR_BIRTHWEIGHT_RANGE_G = (500.0, 5500.0)
# Exposure-table columns that place a canton in its province; canton names alone are never matched
# (BOLIVAR and OLMEDO each name cantons in two provinces).
ECUADOR_EXPOSURE_PROVINCE_COLUMNS = ('province', 'prov_res', 'prov_key', 'provincia', 'ADM1')
ECUADOR_EXPOSURE_DPA_COLUMNS = ('cant_code', 'dpa_canton')     # 4-digit DPA canton code (province + canton)
# INEC DPA province codes -> province names as normalise_area_name writes them (24 provinces + zone 90).
ECUADOR_DPA_PROVINCES = {
    '01': 'AZUAY', '02': 'BOLIVAR', '03': 'CANAR', '04': 'CARCHI', '05': 'COTOPAXI', '06': 'CHIMBORAZO',
    '07': 'EL ORO', '08': 'ESMERALDAS', '09': 'GUAYAS', '10': 'IMBABURA', '11': 'LOJA', '12': 'LOS RIOS',
    '13': 'MANABI', '14': 'MORONA SANTIAGO', '15': 'NAPO', '16': 'PASTAZA', '17': 'PICHINCHA',
    '18': 'TUNGURAHUA', '19': 'ZAMORA CHINCHIPE', '20': 'GALAPAGOS', '21': 'SUCUMBIOS', '22': 'ORELLANA',
    '23': 'SANTO DOMINGO DE LOS TSACHILAS', '24': 'SANTA ELENA', '90': 'ZONAS NO DELIMITADAS',
}
ECUADOR_MATERNAL_AGE_RANGE = (10.0, 60.0)
# Reviewed INEC -> geoBoundaries ADM2 canton name concordance (normalised names).
CANTON_ALIASES = {
    'EL EMPALME': 'EMPALME',
    'SALITRE (URBINA JADO)': 'SALITRE',
    'YANTZAZA (YANZATZA)': 'YANTZAZA',
    'ALFREDO BAQUERIZO MORENO (JUJAN)': 'ALFREDO BAQUERIZO MORENO',
    'GENERAL ANTONIO ELIZALDE (BUCAY)': 'GNRAL ANTONIO ELIZALDE',
    'CORONEL MARCELINO MARIDUENA': 'CRNEL MARCELINO MARIDUENA',
}

COLOMBIA_PRIMARY_COVARIATES = ('EDAD_MADRE', 'NIV_EDUM')
# Verbatim birth_anchor_recipe.covariates.sensitivity.COL; nothing else (e.g. SEXO) is an approved covariate.
COLOMBIA_SENSITIVITY_COVARIATES = ('EST_CIVM',)
# Per-column DANE "sin informacion" codes after integer normalisation; blank and NaN are always missing.
# NIV_EDUM 9 is a valid level (profesional), so only 99 is missing there.
COLOMBIA_MISSING_CODES = {
    'EDAD_MADRE': frozenset({'9', '99', ''}),
    'NIV_EDUM': frozenset({'99', ''}),
    'EST_CIVM': frozenset({'9', '99', ''}),
}
COLOMBIA_FORBIDDEN_FIELDS = frozenset({
    'T_GES', 'T_GES_AGRU_CIE', 'NUMCONSUL', 'TIPO_PARTO', 'MUL_PARTO', 'APGAR1', 'APGAR2', 'TALLA_NAC',
    'IDPERTET', 'COD_DPTO', 'COD_MUNIC', 'SIT_PARTO', 'ATEN_PAR', 'N_HIJOSV', 'N_EMB'})
COLOMBIA_OUTCOMES = {'lbw': 'binary', 'vlbw': 'binary', 'bw_mid_g': 'continuous'}
COLOMBIA_BAND_MIDPOINTS_G = {'1': 750.0, '2': 1250.0, '3': 1750.0, '4': 2250.0, '5': 2750.0, '6': 3250.0,
                             '7': 3750.0, '8': 4250.0}
COLOMBIA_LBW_BANDS = ('1', '2', '3', '4')     # < 2,500 g
COLOMBIA_VLBW_BANDS = ('1', '2')              # < 1,500 g
YEAR_TERM = 'year'


def _forbidden_hits(names: Sequence[str], forbidden: frozenset[str]) -> list[str]:
    folded = {f.casefold() for f in forbidden}
    return sorted({str(n) for n in names if str(n).casefold() in folded})


def _coerce_spec(spec: Any, cls: type, overrides: Mapping[str, Any]) -> Any:
    if spec is None:
        spec = cls()
    elif isinstance(spec, Mapping):
        unknown = set(spec) - {f.name for f in fields(cls)}
        _check(not unknown, f'unknown {cls.__name__} fields: {sorted(unknown)}')
        spec = cls(**spec)
    _check(isinstance(spec, cls), f'expected a {cls.__name__}')
    if overrides:
        spec = replace(spec, **overrides)
    return replace(spec, covariates=tuple(spec.covariates))


def _check_covariates(anchor: str, covariates: Sequence[str], forbidden: frozenset[str],
                      primary: Sequence[str], allowed: Sequence[str]) -> None:
    covariates = list(covariates)
    hits = _forbidden_hits(covariates, forbidden)
    _check(not hits, f'{anchor}: excluded covariates {hits} are forbidden by the approved recipe')
    _check(len(covariates) == len(set(covariates)), f'{anchor}: duplicate covariates')
    unapproved = sorted(set(covariates) - set(allowed))
    _check(not unapproved, f'{anchor}: covariates {unapproved} are not approved baseline covariates')
    missing = [c for c in primary if c not in covariates]
    _check(not missing, f'{anchor}: approved primary covariates {missing} are missing')


@dataclass(frozen=True)
class PeruEndesSpec:
    """Peru ENDES child haemoglobin recipe (configs/adapters/endes.yaml); the defaults are the only approved values."""
    outcome_field: str = 'HC53'
    hb_divisor: float = 10.0
    status_field: str = 'HC55'
    status_value: int = 0
    age_field: str = 'HC1'
    age_months: tuple[int, int] = (4, 59)
    exposure_field: str = 'HV040'
    cluster_field: str = 'HV001'
    stratum_field: str = 'HV024'
    weight_field: str = 'HV005'
    weight_divisor: float = 1_000_000.0
    covariates: tuple[str, ...] = PERU_BASELINE_COVARIATES


def validate_peru_endes_spec(spec: PeruEndesSpec | Mapping[str, Any] | None = None, **overrides: Any) -> PeruEndesSpec:
    """Raw unadjusted ``HC53 / 10`` with ``HC55 == 0`` at 4-59 months only; no adjusted Hb or anthropometry."""
    spec = _coerce_spec(spec, PeruEndesSpec, overrides)
    named = [spec.outcome_field, spec.status_field, spec.age_field, spec.exposure_field, spec.cluster_field,
             spec.stratum_field, spec.weight_field]
    hits = _forbidden_hits(named + list(spec.covariates), PERU_FORBIDDEN_FIELDS)
    _check(not hits, f'peru_endes: {hits} are forbidden (WHO altitude-adjusted Hb or anthropometry); '
                     'only raw unadjusted HC53 / 10 is approved')
    expected = dict(outcome_field='HC53', status_field='HC55', age_field='HC1', exposure_field='HV040',
                    cluster_field='HV001', stratum_field='HV024', weight_field='HV005')
    for name, value in expected.items():
        _check(getattr(spec, name) == value, f'peru_endes: {name} must be {value!r}, got {getattr(spec, name)!r}')
    _check(spec.hb_divisor == 10.0, 'peru_endes: haemoglobin is HC53 / 10 g/dL')
    _check(spec.status_value == 0, 'peru_endes: only measured haemoglobin (HC55 == 0) is approved')
    _check(tuple(spec.age_months) == (4, 59), 'peru_endes: child age domain is 4-59 months')
    _check(spec.weight_divisor == 1_000_000.0, 'peru_endes: survey weight is HV005 / 1,000,000')
    _check(not {spec.outcome_field, spec.status_field} & set(spec.covariates),
           'peru_endes: outcome fields cannot be covariates')
    _check_covariates('peru_endes', spec.covariates, PERU_FORBIDDEN_FIELDS, (), PERU_BASELINE_COVARIATES)
    return spec


@dataclass(frozen=True)
class EcuadorInecSpec:
    """Ecuador INEC 2024 births recipe (birth_anchor_recipe); maternal residence canton only."""
    outcome: str = 'peso_g'
    birthweight_field: str = 'peso'
    residence_province_field: str = 'prov_res'
    residence_canton_field: str = 'cant_res'
    covariates: tuple[str, ...] = ECUADOR_PRIMARY_COVARIATES


def validate_ecuador_inec_spec(spec: EcuadorInecSpec | Mapping[str, Any] | None = None,
                               **overrides: Any) -> EcuadorInecSpec:
    """Residence ``prov_res`` / ``cant_res`` only; ``edad_mad``, ``niv_inst``, ``etnia`` (+ ``est_civil``)."""
    spec = _coerce_spec(spec, EcuadorInecSpec, overrides)
    named = [spec.birthweight_field, spec.residence_province_field, spec.residence_canton_field]
    hits = _forbidden_hits(named, ECUADOR_FORBIDDEN_FIELDS)
    _check(not hits, f'ecuador_inec: {hits} are forbidden (delivery/registration geography or post-outcome '
                     'fields); exposure is placed at maternal residence cant_res only')
    _check(spec.residence_canton_field == 'cant_res', 'ecuador_inec: residence canton field must be cant_res')
    _check(spec.residence_province_field == 'prov_res', 'ecuador_inec: residence province field must be prov_res')
    _check(spec.birthweight_field == 'peso', 'ecuador_inec: birthweight field must be peso')
    _check(spec.outcome in ECUADOR_OUTCOMES, f'ecuador_inec: outcome must be one of {sorted(ECUADOR_OUTCOMES)}')
    _check_covariates('ecuador_inec', spec.covariates, ECUADOR_FORBIDDEN_FIELDS, ECUADOR_PRIMARY_COVARIATES,
                      ECUADOR_PRIMARY_COVARIATES + ECUADOR_SENSITIVITY_COVARIATES)
    return spec


@dataclass(frozen=True)
class ColombiaDaneSpec:
    """Colombia DANE 2023-2025 births recipe (birth_anchor_recipe); maternal residence municipality only."""
    outcome: str = 'lbw'
    birthweight_field: str = 'PESO_NAC'
    residence_department_field: str = 'CODPTORE'
    residence_municipality_field: str = 'CODMUNRE'
    covariates: tuple[str, ...] = (YEAR_TERM,) + COLOMBIA_PRIMARY_COVARIATES


def validate_colombia_dane_spec(spec: ColombiaDaneSpec | Mapping[str, Any] | None = None,
                                **overrides: Any) -> ColombiaDaneSpec:
    """Residence ``CODPTORE + CODMUNRE`` only; ``EDAD_MADRE``, ``NIV_EDUM`` (+ sensitivity ``EST_CIVM``)."""
    spec = _coerce_spec(spec, ColombiaDaneSpec, overrides)
    named = [spec.birthweight_field, spec.residence_department_field, spec.residence_municipality_field]
    hits = _forbidden_hits(named, COLOMBIA_FORBIDDEN_FIELDS)
    _check(not hits, f'colombia_dane: {hits} are forbidden (occurrence geography or post-outcome fields); '
                     'exposure is placed at maternal residence CODPTORE + CODMUNRE only')
    _check(spec.residence_department_field == 'CODPTORE', 'colombia_dane: residence department must be CODPTORE')
    _check(spec.residence_municipality_field == 'CODMUNRE', 'colombia_dane: residence municipality must be CODMUNRE')
    _check(spec.birthweight_field == 'PESO_NAC', 'colombia_dane: birthweight field must be PESO_NAC')
    _check(spec.outcome in COLOMBIA_OUTCOMES, f'colombia_dane: outcome must be one of {sorted(COLOMBIA_OUTCOMES)}')
    _check_covariates('colombia_dane', spec.covariates, COLOMBIA_FORBIDDEN_FIELDS, COLOMBIA_PRIMARY_COVARIATES,
                      (YEAR_TERM,) + COLOMBIA_PRIMARY_COVARIATES + COLOMBIA_SENSITIVITY_COVARIATES)
    return spec


# ---------------------------------------------------------------------------
# Loaders (raw microdata already read into DataFrames; nothing is written or shared)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorFrame:
    """Analysis rows for one anchor; ``a`` is constant within each cluster and clusters nest in strata."""
    anchor: str
    ids: np.ndarray
    a: np.ndarray
    y: np.ndarray
    X: np.ndarray
    clusters: np.ndarray
    strata: np.ndarray
    feature_names: tuple[str, ...]
    outcome: str
    outcome_kind: OutcomeKind
    survey_weights: np.ndarray | None = None
    audit: Mapping[str, Any] | None = None


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    _check(not missing, f'{what} is missing columns {missing}')


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).lstrip('﻿').strip() for c in frame.columns]
    return frame


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def integer_codes(series: pd.Series, field: str) -> pd.Series:
    """Integer survey codes as canonical decimal strings; missing values stay ``NaN``.

    ``3``, ``'3'``, ``' 03 '`` and ``3.0`` (a float column produced by a missing cell)
    all become ``'3'``; ``None``, ``NaN`` and blank strings are missing. A present value
    that is not an integer (``3.5``, ``'abc'``, ``inf``) raises ``ValueError`` rather
    than silently becoming a new code. Callers zero-pad where a code has a fixed width.
    """
    text = series.astype('string').str.strip()
    missing = (text.isna() | (text == '')).to_numpy(dtype=bool)
    numbers = pd.to_numeric(text.fillna('nan').astype(object).where(~missing, 'nan'),
                            errors='coerce').to_numpy(dtype=np.float64)
    with np.errstate(invalid='ignore'):
        invalid = ~missing & ~(np.isfinite(numbers) & (numbers == np.round(numbers)))
    _check(not invalid.any(), f'{field}: non-integer codes {sorted(set(text[invalid].tolist()))[:5]}')
    out = np.full(len(text), np.nan, dtype=object)
    out[~missing] = [str(v) for v in numbers[~missing].astype(np.int64).tolist()]
    return pd.Series(out, index=series.index, dtype=object)


_INTEGRAL_TEXT = re.compile(r'^[+-]?\d+(?:\.0*)?$')


def _normalize_hhid(series: pd.Series, what: str) -> pd.Series:
    """ENDES household ids comparable across modules read as text or as numbers.

    Values are stripped; an integral number (``1``, ``1.0``, ``'1.0'``, ``'0001'``) becomes
    its integer string (``'1'``) so a float-typed module still joins a text-typed one.
    Genuine DHS text ids such as ``'000101 02'`` are kept exactly as stripped. Missing or
    non-integral numeric ids raise ``ValueError``. Spellings of one household (``'1'`` and
    ``'0001'``) may coexist; :func:`_require_unique_rows` checks the household-level and
    child-level tables for genuine duplicates after normalisation.
    """
    def one(value: Any) -> Any:
        if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{what}: boolean HHID')
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            _check(float(value).is_integer(), f'{what}: non-integral numeric HHID {value!r}')
            return str(int(value))
        text = str(value).strip()
        if not text:
            return None
        return str(int(text.split('.')[0])) if _INTEGRAL_TEXT.match(text) else text
    out = series.astype(object).map(one).astype(object)
    _check(bool(out.notna().all()), f'{what}: missing HHID')
    return out


def _require_unique_rows(frame: pd.DataFrame, key: Sequence[str], what: str) -> pd.DataFrame:
    """Drop exact duplicate rows, then require one row per ``key`` (rows with a missing key part are exempt)."""
    frame = frame.drop_duplicates()
    complete = frame[list(key)].notna().all(axis=1)
    clash = frame.loc[complete].duplicated(list(key), keep=False)
    example = frame.loc[complete].loc[clash, list(key)].head(2).values.tolist()
    _check(not bool(clash.any()), f'{what}: {int(clash.sum())} rows share {"/".join(key)} after normalisation '
                                  f'with conflicting values, e.g. {example}')
    return frame


def _zfill(codes: pd.Series, width: int) -> pd.Series:
    """Zero-pad normalised codes; missing codes stay missing (safe on all-missing columns)."""
    return codes.map(lambda v: v.zfill(width) if isinstance(v, str) else np.nan).astype(object)


def load_peru_endes_frame(modules_by_year: Mapping[int, Mapping[str, pd.DataFrame]],
                          spec: PeruEndesSpec | Mapping[str, Any] | None = None) -> AnchorFrame:
    """Pooled ENDES child biomarker rows from ``{year: {'RECH0', 'RECH23', 'RECH6'}}`` DataFrames.

    Only the approved columns are read: the outcome is ``HC53 / 10`` g/dL for measured
    children (``HC55 == 0``, valid ``HC53`` in 30-250) aged 4-59 months (``HC1``), and
    the exposure is the deficit at cluster altitude ``HV040``. Clusters are
    year-prefixed ``HV001`` (panels share numbers across years); strata are the
    departments ``HV024``; survey weights are ``HV005 / 1e6``. Adjusted Hb and
    anthropometry columns present in RECH6 are never touched.
    """
    spec = validate_peru_endes_spec(spec)
    _check(bool(modules_by_year), 'peru_endes: no survey years supplied')
    # Requested covariates per module: immutable and never rebound inside the year loop.
    child_covariates = tuple(c for c in spec.covariates if c in PERU_CHILD_COVARIATES)
    household_covariates = tuple(c for c in spec.covariates if c in PERU_HOUSEHOLD_COVARIATES)
    asset_covariates = tuple(c for c in spec.covariates if c in PERU_ASSET_COVARIATES)
    frames, audit = [], dict(years={}, absent_covariates={})
    for year in sorted(modules_by_year):
        modules = modules_by_year[year]
        _check({'RECH0', 'RECH23', 'RECH6'} <= set(modules), f'peru_endes {year}: RECH0, RECH23 and RECH6 required')
        r0, r23, r6 = (_clean_columns(modules[k]) for k in ('RECH0', 'RECH23', 'RECH6'))
        base6 = ['HHID', 'HC0', spec.outcome_field, spec.status_field, spec.age_field]
        base0 = ['HHID', spec.cluster_field, spec.weight_field, spec.stratum_field, spec.exposure_field]
        _require_columns(r6, base6, f'peru_endes {year} RECH6')
        _require_columns(r0, base0, f'peru_endes {year} RECH0')
        _require_columns(r23, ['HHID'], f'peru_endes {year} RECH23')
        keep6 = list(dict.fromkeys(base6 + [c for c in child_covariates if c in r6.columns]))
        keep0 = base0 + [c for c in household_covariates if c in r0.columns]
        keep23 = ['HHID'] + [c for c in asset_covariates if c in r23.columns]
        for name, frame in (('RECH0', r0), ('RECH23', r23), ('RECH6', r6)):
            frame['HHID'] = _normalize_hhid(frame['HHID'], f'peru_endes {year} {name}')
        r6['_hc0'] = integer_codes(r6['HC0'], f'peru_endes {year} HC0')
        house0 = _require_unique_rows(r0[keep0], ['HHID'], f'peru_endes {year} RECH0 households')
        house23 = _require_unique_rows(r23[keep23], ['HHID'], f'peru_endes {year} RECH23 households')
        # Rows are compared on normalised keys only (HHID is already normalised; raw HC0 '1' vs '01' is dropped),
        # so an exact repeat spelled differently is deduplicated and only conflicting values raise.
        dedup_cols6 = [c for c in keep6 if c != 'HC0'] + ['_hc0']
        children = _require_unique_rows(r6[dedup_cols6], ['HHID', '_hc0'], f'peru_endes {year} RECH6 children')
        merged = children.merge(house0, on='HHID', how='inner').merge(house23, on='HHID', how='left')
        status, raw_hb = _numeric(merged[spec.status_field]), _numeric(merged[spec.outcome_field])
        age, altitude = _numeric(merged[spec.age_field]), _numeric(merged[spec.exposure_field])
        weight = _numeric(merged[spec.weight_field]) / spec.weight_divisor
        lo, hi = spec.age_months
        valid = ((status == spec.status_value) & raw_hb.between(30, 250) & age.between(lo, hi)
                 & altitude.between(-50, 5500) & (weight > 0))
        merged = merged.loc[valid].reset_index(drop=True)
        merged[YEAR_TERM] = float(year)
        merged['_y'] = _numeric(merged[spec.outcome_field]) / spec.hb_divisor
        merged['_a'] = elevation_m_to_pio2_deficit_mmhg(_numeric(merged[spec.exposure_field]).to_numpy(np.float64))
        merged['_w'] = _numeric(merged[spec.weight_field]) / spec.weight_divisor
        stratum = integer_codes(merged[spec.stratum_field], f'peru_endes {year} {spec.stratum_field}')
        cluster = integer_codes(merged[spec.cluster_field], f'peru_endes {year} {spec.cluster_field}')
        line = merged['_hc0']
        _check(not (stratum.isna() | cluster.isna() | line.isna()).any(),
               f'peru_endes {year}: missing {spec.stratum_field}, {spec.cluster_field} or HC0 codes')
        merged['_stratum'] = _zfill(stratum, 2)
        merged['_cluster'] = f'{year}:' + cluster
        merged['_id'] = f'{year}:' + merged['HHID'] + ':' + line
        audit['years'][int(year)] = dict(n_rech6=int(len(r6)), n_eligible=int(len(merged)))
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)
    feature_names = tuple(c for c in spec.covariates if c in df.columns)
    audit['absent_covariates'] = [c for c in spec.covariates if c not in df.columns]
    _check(not _forbidden_hits(feature_names, PERU_FORBIDDEN_FIELDS), 'peru_endes: forbidden column leaked')
    _check(len(df) > 0, 'peru_endes: no eligible children')
    X = df[list(feature_names)].apply(_numeric).to_numpy(dtype=np.float64)
    a, clusters = df['_a'].to_numpy(np.float64), df['_cluster'].to_numpy()
    cluster_table(a, clusters, df['_stratum'].to_numpy())   # HV040 is documented cluster altitude
    audit.update(n_rows=int(len(df)), n_clusters=int(df['_cluster'].nunique()),
                 n_strata=int(df['_stratum'].nunique()))
    return AnchorFrame(anchor='peru_endes', ids=df['_id'].to_numpy(dtype=object), a=a,
                       y=df['_y'].to_numpy(np.float64), X=X, clusters=clusters, strata=df['_stratum'].to_numpy(),
                       feature_names=feature_names, outcome='hb_g_dl', outcome_kind='continuous',
                       survey_weights=df['_w'].to_numpy(np.float64), audit=audit)


_MISSING_AREA_TEXT = frozenset({'', 'NAN', 'NONE', '<NA>'})


_DECIMAL_TEXT = re.compile(r'^[+-]?\d*\.\d+$')


def _is_missing_value(value: Any) -> bool:
    return (value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value))
            or (isinstance(value, str) and normalise_area_name(value) in _MISSING_AREA_TEXT))


def _try_int_code(value: Any) -> str | None:
    """Canonical integer string of one integral value (``13``, ``13.0``, ``'13'``, ``'013'`` -> ``'13'``).

    Returns ``None`` for missing values and text (a name); a non-integral number
    (``13.5``, ``'13.5'``, ``inf``) raises ``ValueError``. Judged per value, never per column.
    """
    if _is_missing_value(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        _check(float(value).is_integer(), f'non-integral numeric code {value!r}')
        return str(int(value))
    text = str(value).strip()
    _check(not _DECIMAL_TEXT.match(text) or _INTEGRAL_TEXT.match(text), f'non-integral numeric code {value!r}')
    return str(int(text.split('.')[0])) if _INTEGRAL_TEXT.match(text) else None


def _try_dpa_code(value: Any) -> str | None:
    """4-digit zero-padded DPA canton code (``1701``, ``1701.0``, ``'1701'``, ``'0101'``) or ``None``."""
    code = _try_int_code(value)
    return f'{int(code):04d}' if code is not None and 1 <= int(code) <= 9999 else None


def _area_key(value: Any) -> Any:
    """One normalised area name; integral codes become integer strings; missing stays ``NaN``.

    Integral numeric codes in any spelling (``17``, ``17.0``, ``'17.0'``, ``'017'``) become the
    integer string (``'17'``) before name normalisation, which would otherwise turn ``17.0``
    into ``'17 0'``. A non-integral number raises ``ValueError``.
    """
    if _is_missing_value(value):
        return np.nan
    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, (bool, np.bool_)):
        _check(float(value).is_integer(), f'non-integral numeric area code {value!r}')
        return str(int(value))
    stripped = str(value).strip()
    if _INTEGRAL_TEXT.match(stripped):
        return str(int(stripped.split('.')[0]))
    return normalise_area_name(stripped)


def _canonical_ecuador_province(value: Any) -> str | None:
    """One province label for codes and names: ``17``, ``17.0``, ``'017'`` and ``'Pichincha'`` -> ``'PICHINCHA'``.

    Missing -> ``None``. An integral code must be an INEC DPA province code, else
    ``ValueError``. Text is normalised; an official name is returned as such and any other
    name is kept as normalised text (it can never agree with a DPA code).
    """
    if _is_missing_value(value):
        return None
    code = _try_int_code(value)
    if code is not None:
        padded = f'{int(code):02d}'
        _check(padded in ECUADOR_DPA_PROVINCES, f'ecuador_inec: unrecognized DPA province code {value!r}')
        return ECUADOR_DPA_PROVINCES[padded]
    name = _area_key(value)
    return name if isinstance(name, str) else None


def _canton_key(value: Any) -> Any:
    """Normalised canton name with the reviewed INEC -> geoBoundaries aliases, for births and exposure alike."""
    name = _area_key(value)
    return CANTON_ALIASES.get(name, name) if isinstance(name, str) else name


def _area_keys(series: pd.Series) -> pd.Series:
    """:func:`_area_key` per value; missing, blank or stringified-missing values stay ``NaN`` (never ``'NAN'``)."""
    return series.astype(object).map(_area_key).astype(object)


def normalise_area_name(value: Any) -> str:
    """ASCII upper-case, hyphens and dots to spaces, single-spaced (Ecuador canton matching)."""
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii').upper().strip()
    return ' '.join(text.replace('-', ' ').replace('.', ' ').split())


def _clean_category(series: pd.Series, missing: Mapping[str, str], field: str) -> pd.Series:
    """Per-element category labels: integral codes in any spelling become one label, text is kept.

    ``'1'``, ``1``, ``1.0``, ``'1.0'`` and ``' 01 '`` all become ``'1'`` whatever the column
    dtype (object columns can mix them); text such as ``'Primaria'`` is kept as stripped;
    missing values become ``'MISSING'``. A non-integral number raises ``ValueError``.
    """
    def one(value: Any) -> str:
        if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return 'MISSING'
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            _check(float(value).is_integer(), f'{field}: non-integral numeric category {value!r}')
            return str(int(value))
        text = str(value).strip()
        return str(int(text.split('.')[0])) if _INTEGRAL_TEXT.match(text) else text
    return series.astype(object).map(one).astype(object).replace(dict(missing))


def _onehot(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[np.ndarray], list[str]]:
    parts, names = [], []
    for col in columns:
        dummies = pd.get_dummies(frame[col], prefix=col, dtype=np.float64)
        parts.append(dummies.to_numpy(dtype=np.float64))
        names.extend(dummies.columns.tolist())
    return parts, names


def _resolve_unmatched(bad: Mapping[str, int], on_unmatched: str, what: str) -> None:
    _check(on_unmatched in ('raise', 'drop'), "on_unmatched must be 'raise' or 'drop'")
    _check(not bad or on_unmatched == 'drop',
           f'{what}: unmatched or ambiguous area codes stop the run (birth_anchor_recipe.unmatched_codes): '
           f'{dict(sorted(bad.items())[:10])}; pass on_unmatched="drop" to exclude and audit them')


def load_ecuador_inec_frame(births: pd.DataFrame, canton_exposure: pd.DataFrame,
                            spec: EcuadorInecSpec | Mapping[str, Any] | None = None, *,
                            stratum: Literal['province', 'national'] = 'province',
                            on_unmatched: Literal['raise', 'drop'] = 'raise') -> AnchorFrame:
    """INEC births placed at the maternal residence canton; only approved columns are read.

    ``canton_exposure`` carries the transform-before-aggregate ``oxygen_deficit_mmhg``
    and must place every canton in its province, because canton names repeat across
    provinces. Two lookups are built from it up front:

    * a DPA lookup from a ``cant_code`` / ``dpa_canton`` column (4-digit codes; a numeric
      exposure province must equal the code's first two digits);
    * a name lookup from ``shapeName`` plus a province column (``province``, ``prov_res``,
      ``prov_key``, ``provincia`` or ``ADM1``), keyed on (province, canton name).

    An exposure table supporting neither raises ``ValueError``: canton-only matching is
    never used. Births are matched one row at a time, never by guessing a mode for a
    whole column: a ``cant_res`` that is a DPA code uses the DPA lookup, and that row's
    ``prov_res`` must agree with the code (numeric) or with the exposure row's province
    name (text), else ``ValueError``; any other ``cant_res`` uses the name lookup. A
    birth's cluster and stratum come from the exposure row it matches, so births reaching
    one canton by code and by name share a cluster. Repeated keys and unmatched cantons
    stop the run unless ``on_unmatched='drop'``, which excludes and audits them.
    """
    spec = validate_ecuador_inec_spec(spec)
    _check(stratum in ('province', 'national'), "stratum must be 'province' or 'national'")
    read = [spec.residence_province_field, spec.residence_canton_field, spec.birthweight_field] + list(spec.covariates)
    births = _clean_columns(births)
    _require_columns(births, read, 'ecuador_inec births')
    df = births[read].copy()                       # never touches delivery geography or post-outcome fields
    n_raw = len(df)
    df['_peso'] = _numeric(df[spec.birthweight_field])
    df = df.loc[df['_peso'].between(*ECUADOR_BIRTHWEIGHT_RANGE_G)].reset_index(drop=True)
    n_valid = len(df)

    exposure = _clean_columns(canton_exposure)
    _require_columns(exposure, ['oxygen_deficit_mmhg'], 'ecuador_inec canton exposure')
    exposure = exposure.loc[exposure['oxygen_deficit_mmhg'].notna()].reset_index(drop=True)
    deficits = exposure['oxygen_deficit_mmhg'].astype(float).to_numpy()
    _check(bool(np.isfinite(deficits).all()), 'ecuador_inec: exposure deficits must be finite')
    province_col = next((c for c in ECUADOR_EXPOSURE_PROVINCE_COLUMNS if c in exposure.columns), None)
    dpa_col = next((c for c in ECUADOR_EXPOSURE_DPA_COLUMNS if c in exposure.columns), None)
    name_col = 'shapeName' if 'shapeName' in exposure.columns else None
    has_names = province_col is not None and name_col is not None
    _check(dpa_col is not None or has_names,
           'ecuador_inec: the canton exposure table needs a province column '
           f'{ECUADOR_EXPOSURE_PROVINCE_COLUMNS} with shapeName, or a 4-digit DPA canton code '
           f'{ECUADOR_EXPOSURE_DPA_COLUMNS}; canton names repeat across provinces, '
           'so canton-only matching is never used')

    # Exposure lookups, parsed row by row up front. Every province (exposure and births, code or name) goes
    # through _canonical_ecuador_province; a DPA row's stratum is always the official province of its code.
    # Cluster labels are stratum::DPA code, else stratum::canton name.
    rows: list[tuple[str | None, str | None, dict]] = []      # (DPA code, name key, record) per usable exposure row
    exposure_unusable = 0
    for i in range(len(exposure)):
        raw_province = exposure.at[i, province_col] if province_col else None
        exp_province = _canonical_ecuador_province(raw_province)
        dpa = None
        if dpa_col is not None:
            raw_code = exposure.at[i, dpa_col]
            dpa = _try_dpa_code(raw_code)
            _check(dpa is not None or _is_missing_value(raw_code),
                   f'ecuador_inec: exposure {dpa_col} {raw_code!r} is not a 4-digit DPA canton code')
        if dpa is not None:
            _check(dpa[:2] in ECUADOR_DPA_PROVINCES,
                   f'ecuador_inec: unrecognized DPA province code {dpa[:2]!r} in exposure {dpa_col} {dpa}')
            official = ECUADOR_DPA_PROVINCES[dpa[:2]]
            _check(exp_province is None or exp_province == official,
                   f'ecuador_inec: exposure DPA code {dpa} lies outside its own province {raw_province!r} '
                   f'(DPA province {official!r})')
            exp_province = official
        cant_key = _canton_key(exposure.at[i, name_col]) if has_names else np.nan
        named = exp_province is not None and isinstance(cant_key, str) and has_names
        if dpa is None and not named:
            exposure_unusable += 1                 # neither a code nor a (province, canton) key: places nothing
            continue
        record = dict(deficit=float(deficits[i]), stratum=exp_province,
                      cluster=f'{exp_province}::{dpa if dpa is not None else cant_key}')
        rows.append((dpa, f'{exp_province}::{cant_key}' if named else None, record))
    # A repeated DPA code or (province, canton) key is ambiguous, and so is every key of any row carrying an
    # ambiguous key (to a fixed point), so no lookup can pick one side of a conflicting exposure.
    dpa_counts = pd.Series([d for d, _, _ in rows if d], dtype=object).value_counts()
    name_counts = pd.Series([n for _, n, _ in rows if n], dtype=object).value_counts()
    bad_dpa, bad_name = set(dpa_counts.index[dpa_counts > 1]), set(name_counts.index[name_counts > 1])
    while tainted := [(d, n) for d, n, _ in rows if (d in bad_dpa or n in bad_name)
                      and ((d and d not in bad_dpa) or (n and n not in bad_name))]:
        bad_dpa |= {d for d, _ in tainted if d}
        bad_name |= {n for _, n in tainted if n}
    ambiguous = sorted([f'DPA:{d}' for d in bad_dpa] + list(bad_name))
    dpa_lookup = {d: r for d, _, r in rows if d and d not in bad_dpa} if dpa_col is not None else None
    name_lookup = {n: r for _, n, r in rows if n and n not in bad_name} if has_names else None

    # Births, one row at a time.
    records: list[dict | None] = []
    modes: list[str | None] = []
    conflicts: list[str] = []
    n_missing_area = 0
    for prov_value, cant_value in zip(df[spec.residence_province_field].tolist(),
                                      df[spec.residence_canton_field].tolist()):
        dpa = _try_dpa_code(cant_value) if dpa_lookup is not None else None
        birth_province = _canonical_ecuador_province(prov_value)
        record, mode = None, None
        if dpa is not None:
            _check(dpa[:2] in ECUADOR_DPA_PROVINCES,
                   f'ecuador_inec: unrecognized DPA province code {dpa[:2]!r} in cant_res {cant_value!r}')
            official = ECUADOR_DPA_PROVINCES[dpa[:2]]
            if birth_province is not None and birth_province != official:   # one check for codes and names
                conflicts.append(f'prov_res {prov_value!r} ({birth_province!r}) vs DPA province {official!r} '
                                 f'of cant_res {cant_value!r}')
            record, mode = dpa_lookup.get(dpa), 'dpa_code'
        elif name_lookup is not None:
            cant_name = _canton_key(cant_value)
            if birth_province is not None and isinstance(cant_name, str):
                record, mode = name_lookup.get(f'{birth_province}::{cant_name}'), f'province_name:{province_col}'
            else:
                n_missing_area += 1                # missing province or canton: unmatched
        elif _is_missing_value(cant_value):
            n_missing_area += 1
        records.append(record)
        modes.append(mode if record is not None else None)
    _check(not conflicts, f'ecuador_inec: {len(conflicts)} births have a cant_res DPA code outside their prov_res '
                          f'province, e.g. {conflicts[:3]}')

    matched = np.asarray([r is not None for r in records], dtype=bool)
    raw_canton = df[spec.residence_canton_field]
    bad = (raw_canton.astype(str).where(_area_keys(raw_canton).notna(), '<missing>')
           .loc[~matched].value_counts().to_dict())
    _resolve_unmatched(bad, on_unmatched, 'ecuador_inec')
    kept = [r for r in records if r is not None]
    used_modes = {m for m in modes if m is not None}
    matching = '+'.join(m for m in ('dpa_code', f'province_name:{province_col}') if m in used_modes)
    df = df.loc[matched].reset_index(drop=True)
    df['_a'] = [r['deficit'] for r in kept]
    df['_cluster'] = [r['cluster'] for r in kept]
    province = pd.Series([r['stratum'] for r in kept], dtype=object)
    df['_stratum'] = province if stratum == 'province' else 'ECU_ALL'
    df['edad_mad'] = _numeric(df['edad_mad'])
    df.loc[~df['edad_mad'].between(*ECUADOR_MATERNAL_AGE_RANGE), 'edad_mad'] = np.nan
    missing = {'Sin información': 'MISSING', 'Ignorado': 'MISSING', '': 'MISSING'}
    categorical = [c for c in spec.covariates if c != 'edad_mad']
    for col in categorical:
        df[col] = _clean_category(df[col], missing, f'ecuador_inec {col}')
    parts, names = _onehot(df, categorical)
    X = np.hstack([df[['edad_mad']].to_numpy(dtype=np.float64)] + parts)
    names = ['edad_mad'] + names
    y = df['_peso'].to_numpy(np.float64) if spec.outcome == 'peso_g' else (df['_peso'] < 2500.0).to_numpy(np.float64)
    a, clusters = df['_a'].to_numpy(np.float64), df['_cluster'].to_numpy()
    _check(len(df) > 0, 'ecuador_inec: no matched births')
    cluster_table(a, clusters, df['_stratum'].to_numpy())
    audit = dict(n_raw=n_raw, n_valid_birthweight=n_valid, n_matched_residence=int(len(df)),
                 match_rate=float(len(df) / n_valid) if n_valid else 0.0, n_cantons=int(df['_cluster'].nunique()),
                 n_provinces=int(province.nunique()), ambiguous_exposure_names=sorted(ambiguous),
                 matching=matching, n_matched_by_dpa=sum(m == 'dpa_code' for m in modes),
                 n_matched_by_name=sum(m is not None and m != 'dpa_code' for m in modes),
                 n_births_missing_residence=n_missing_area, n_exposure_rows_missing_area=exposure_unusable,
                 excluded_unmatched_cantons={str(k): int(v) for k, v in bad.items()}, stratum=stratum,
                 interpretation=INTERPRETATION)
    return AnchorFrame(anchor='ecuador_inec', ids=np.asarray([f'ecu2024:{i}' for i in range(len(df))], dtype=object),
                       a=a, y=y, X=X, clusters=clusters, strata=df['_stratum'].to_numpy(), feature_names=tuple(names),
                       outcome=spec.outcome, outcome_kind=ECUADOR_OUTCOMES[spec.outcome], audit=audit)


def load_colombia_dane_frame(births: pd.DataFrame, muni_exposure: pd.DataFrame,
                             spec: ColombiaDaneSpec | Mapping[str, Any] | None = None, *,
                             on_unmatched: Literal['raise', 'drop'] = 'raise') -> AnchorFrame:
    """DANE births placed at the maternal residence municipality; only approved columns are read.

    ``muni_exposure`` carries the 5-digit DIVIPOLA ``cod_mpio`` and the
    transform-before-aggregate ``oxygen_deficit_mmhg``. Strata are residence
    departments. ``year`` comes from a ``year`` column, else ``ANO``.
    """
    spec = validate_colombia_dane_spec(spec)
    births = _clean_columns(births)
    year_col = 'year' if 'year' in births.columns else 'ANO'
    maternal = [c for c in spec.covariates if c != YEAR_TERM]
    read = [spec.residence_department_field, spec.residence_municipality_field, spec.birthweight_field] + maternal
    if YEAR_TERM in spec.covariates:
        read.append(year_col)
    _require_columns(births, read, 'colombia_dane births')
    df = births[read].copy()                       # never touches occurrence geography or post-outcome fields
    n_raw = len(df)
    df['_band'] = integer_codes(df[spec.birthweight_field], f'colombia_dane {spec.birthweight_field}')
    df = df.loc[df['_band'].isin(set(COLOMBIA_BAND_MIDPOINTS_G))].reset_index(drop=True)
    n_valid = len(df)

    exposure = _clean_columns(muni_exposure)
    _require_columns(exposure, ['cod_mpio', 'oxygen_deficit_mmhg'], 'colombia_dane municipality exposure')
    exposure = exposure.loc[exposure['oxygen_deficit_mmhg'].notna()]
    exposure_codes = integer_codes(exposure['cod_mpio'], 'colombia_dane exposure cod_mpio')
    _check(not exposure_codes.isna().any(), 'colombia_dane: missing municipality codes in the exposure table')
    exposure_codes = _zfill(exposure_codes, 5)
    _check(exposure_codes.is_unique, 'colombia_dane: repeated municipality codes in the exposure table')
    deficit = dict(zip(exposure_codes, exposure['oxygen_deficit_mmhg'].astype(float)))

    dept = _zfill(integer_codes(df[spec.residence_department_field], 'colombia_dane CODPTORE'), 2)
    muni = dept + _zfill(integer_codes(df[spec.residence_municipality_field], 'colombia_dane CODMUNRE'), 3)
    matched = muni.isin(set(deficit)).to_numpy(dtype=bool)
    bad = muni.loc[~matched].fillna('<missing>').value_counts().to_dict()
    _resolve_unmatched(bad, on_unmatched, 'colombia_dane')
    df, dept, muni = (v.loc[matched].reset_index(drop=True) for v in (df, dept, muni))
    _check(len(df) > 0, 'colombia_dane: no matched births')

    outcome = {'lbw': df['_band'].isin(COLOMBIA_LBW_BANDS).astype(np.float64),
               'vlbw': df['_band'].isin(COLOMBIA_VLBW_BANDS).astype(np.float64),
               'bw_mid_g': df['_band'].map(COLOMBIA_BAND_MIDPOINTS_G).astype(np.float64)}[spec.outcome]
    for col in maternal:
        level_codes = integer_codes(df[col], f'colombia_dane {col}')
        df[col] = level_codes.where(level_codes.notna() & ~level_codes.isin(COLOMBIA_MISSING_CODES[col]), 'MISSING')
    parts, names = _onehot(df, maternal)
    if YEAR_TERM in spec.covariates:
        year = _numeric(integer_codes(df[year_col], f'colombia_dane {year_col}')).to_numpy(np.float64)
        parts.insert(0, (year - 2024.0).reshape(-1, 1))
        names.insert(0, 'year_c')
    X = np.hstack(parts)
    a = muni.map(deficit).to_numpy(np.float64)
    clusters = muni.to_numpy()
    cluster_table(a, clusters, dept.to_numpy())
    audit = dict(n_raw=n_raw, n_valid_birthweight=n_valid, n_matched_residence=int(len(df)),
                 match_rate=float(len(df) / n_valid) if n_valid else 0.0, n_departments=int(dept.nunique()),
                 n_municipalities=int(muni.nunique()), outcome_rate=float(outcome.mean()),
                 excluded_unmatched_municipalities={str(k): int(v) for k, v in bad.items()},
                 interpretation=INTERPRETATION)
    return AnchorFrame(anchor='colombia_dane', ids=np.asarray([f'col:{i}' for i in range(len(df))], dtype=object),
                       a=a, y=outcome.to_numpy(np.float64), X=X, clusters=clusters, strata=dept.to_numpy(),
                       feature_names=tuple(names), outcome=spec.outcome,
                       outcome_kind=COLOMBIA_OUTCOMES[spec.outcome], audit=audit)


# ---------------------------------------------------------------------------
# Cluster-level design: support, subsampling, permutation null
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterTable:
    labels: np.ndarray       # sorted unique cluster labels
    inverse: np.ndarray      # row -> cluster position
    a: np.ndarray            # exposure per cluster
    strata: np.ndarray       # stratum per cluster


def cluster_table(a: Any, clusters: Any, strata: Any) -> ClusterTable:
    """One row per cluster; requires constant exposure within a cluster and clusters nested in strata."""
    a, clusters, strata = np.asarray(a, dtype=np.float64), np.asarray(clusters), np.asarray(strata)
    _check(a.shape == clusters.shape == strata.shape and a.ndim == 1, 'cluster inputs are misaligned')
    _check(bool(np.isfinite(a).all()), 'exposure must be finite')
    labels, first, inverse = np.unique(clusters, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    lo = np.full(len(labels), np.inf)
    hi = np.full(len(labels), -np.inf)
    np.minimum.at(lo, inverse, a)
    np.maximum.at(hi, inverse, a)
    _check(bool((hi - lo <= EXPOSURE_TOLERANCE_MMHG).all()),
           'exposure varies within a cluster; cluster-level designs need cluster-assigned exposure')
    _check(bool((strata == strata[first][inverse]).all()), 'a cluster spans more than one stratum')
    return ClusterTable(labels=labels, inverse=inverse, a=a[first], strata=strata[first])


@dataclass(frozen=True)
class ClusterSupport:
    supported: np.ndarray            # per row
    moved: np.ndarray                # per row
    cluster_support: WithinStateSupport
    clusters: ClusterTable

    @property
    def n_targets(self) -> int:
        return int(self.supported.sum())

    @property
    def n_movers(self) -> int:
        return int(self.moved.sum())

    @property
    def delivered_shift(self) -> float:
        _check(self.n_targets > 0, 'no supported targets')
        return float(self.cluster_support.delta * self.n_movers / self.n_targets)


def compute_cluster_support(a: Any, strata: Any, clusters: Any, *, delta: float, bin_width: float = 2.0,
                            min_bin_clusters: int = 2, caliper: float = 2.0,
                            cluster_covariates: np.ndarray | None = None) -> ClusterSupport:
    """Outcome-blind interior-bin support over clusters, broadcast to their rows.

    Delegates to ``compute_within_state_support`` on one row per cluster: a bin is
    occupied when at least ``min_bin_clusters`` *other* clusters of the stratum lie in
    it, strictly inside the stratum's first and last bins; a cluster moves when every
    bin on its shift path is occupied. ``cluster_covariates`` (one row per sorted
    cluster label) optionally adds the ``caliper``-IQR neighbourhood.
    """
    table = cluster_table(a, clusters, strata)
    X_c = np.zeros((len(table.labels), 1)) if cluster_covariates is None else np.asarray(cluster_covariates, float)
    _check(X_c.ndim == 2 and X_c.shape[0] == len(table.labels), 'cluster covariates need one row per cluster')
    support = compute_within_state_support(table.a, X_c, table.strata, delta=delta, caliper=caliper,
                                           bin_width=bin_width, min_bin_tracts=min_bin_clusters)
    return ClusterSupport(supported=support.supported[table.inverse], moved=support.moved[table.inverse],
                          cluster_support=support, clusters=table)


def cluster_capped_subsample(ids: Any, clusters: Any, per_cluster_cap: int,
                             salt: str = 'anchor-eval') -> tuple[np.ndarray, dict[Any, int]]:
    """Outcome-blind hash-ranked subsample of at most ``per_cluster_cap`` rows per cluster.

    Every cluster is kept. Returns the sorted kept row positions and each cluster's full
    row count, for Horvitz-Thompson equal-unit weights ``N_g / n_g``.
    """
    ids, clusters = np.asarray(ids), np.asarray(clusters)
    _check(ids.shape == clusters.shape, 'ids and clusters are misaligned')
    labels, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse.reshape(-1), kind='stable')
    groups = np.split(order, np.cumsum(np.bincount(inverse.reshape(-1), minlength=len(labels)))[:-1])
    keep, counts = [], {}
    for label, rows in zip(labels.tolist(), groups):
        counts[label] = int(len(rows))
        keep.append(rows[deterministic_subsample_idx(ids[rows], cap=per_cluster_cap, salt=f'{salt}:{label}')])
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int), counts


def permute_cluster_within_stratum(a: Any, moved: Any, clusters: Any, strata: Any,
                                   seed: int = NULL_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Falsification null: permute cluster exposures across the clusters of each stratum.

    Exposure is assigned per cluster (PSU altitude, canton or municipality deficit),
    so every row of a cluster receives that cluster's single permuted value; a row-level
    permutation would give rows of one cluster different exposures and break the
    design. Each stratum keeps its multiset of cluster exposures. ``Y``, ``X``, target
    and ``moved`` stay fixed (``moved`` is returned as a copy), so the null delivers
    the original shift and its true effect is zero given the stratum.
    """
    moved = np.asarray(moved, dtype=bool)
    table = cluster_table(a, clusters, strata)
    _check(moved.shape == table.inverse.shape, 'moved is misaligned')
    rng = np.random.default_rng(seed)
    a_cluster = table.a.copy()
    for stratum in sorted(np.unique(table.strata).tolist()):
        members = np.flatnonzero(table.strata == stratum)
        a_cluster[members] = table.a[rng.permutation(members)]
    return a_cluster[table.inverse], moved.copy()


# ---------------------------------------------------------------------------
# Outcome routing: classifiers for binary outcomes, regressors for continuous ones
# ---------------------------------------------------------------------------

class ProbabilityQ:
    """Binary-outcome ``Q``: wraps a classifier and predicts ``predict_proba[:, class 1]``."""

    def __init__(self, classifier: Any) -> None:
        _check(callable(getattr(classifier, 'predict_proba', None)),
               'binary outcomes need a classifier with predict_proba; a regressor is never fitted to 0/1 outcomes')
        _check(not type(classifier).__name__.endswith('Regressor'),
               f'{type(classifier).__name__} is a regressor; binary outcomes need a classifier')
        self.classifier = classifier
        self._constant_prob: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ProbabilityQ':
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        _check(len(y) > 0 and len(X) == len(y), 'binary Q training rows are empty or misaligned')
        _check(bool(np.isin(y, (0.0, 1.0)).all()), 'binary Q requires 0/1 outcomes')
        unique = np.unique(y)
        if len(unique) == 1:
            self._constant_prob = float(unique[0])
            return self
        self._constant_prob = None
        self.classifier.fit(X, y.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self._constant_prob is not None:
            return np.full(len(X), self._constant_prob, dtype=np.float64)
        proba = np.asarray(self.classifier.predict_proba(X), dtype=np.float64)
        classes = [int(c) for c in getattr(self.classifier, 'classes_', (0, 1))]
        _check(1 in classes and proba.ndim == 2 and proba.shape[1] == len(classes), 'classifier lacks class 1')
        return proba[:, classes.index(1)]


@dataclass(frozen=True)
class BinaryQFactory:
    """Seed -> :class:`ProbabilityQ` around ``make_classifier(seed)``; the only Q accepted for 0/1 outcomes."""
    make_classifier: ClassifierFactory
    outcome_kind: str = 'binary'

    def __call__(self, seed: int) -> ProbabilityQ:
        return ProbabilityQ(self.make_classifier(seed))


def binary_q_factory(make_classifier: ClassifierFactory) -> BinaryQFactory:
    return BinaryQFactory(make_classifier)


def _q_kind(q: QLearner) -> str:
    return getattr(q.factory, 'outcome_kind', 'continuous')


def _guard_binary(factory: RegressorFactory) -> RegressorFactory:
    def make(seed: int) -> Any:
        model = factory(seed)
        _check(isinstance(model, ProbabilityQ), 'binary outcomes are modelled only through predict_proba')
        return model
    return make


def route_q_learners(outcome_kind: str, q_learners: Sequence[QLearner]) -> list[QLearner]:
    """Check every Q matches the outcome kind; binary Q factories are re-checked at construction."""
    _check(outcome_kind in OUTCOME_KINDS, f'outcome_kind must be one of {OUTCOME_KINDS}')
    _check(bool(q_learners), 'at least one Q learner is required')
    for q in q_learners:
        _check(_q_kind(q) == outcome_kind,
               f'{q.name}: a {_q_kind(q)} Q learner cannot model a {outcome_kind} outcome'
               + ('; wrap a classifier with binary_q_factory (predict_proba)' if outcome_kind == 'binary' else ''))
    if outcome_kind == 'continuous':
        return list(q_learners)
    return [replace(q, factory=_guard_binary(q.factory)) for q in q_learners]


def default_anchor_q_learners(outcome_kind: str, regressor_checkpoint: str | Path,
                              classifier_checkpoint: str | Path, device: str,
                              n_estimators: int = 4) -> list[QLearner]:
    """``A6-Q`` and ``B0_hist_gbm``: TabICLv2 regressor / GBM regressor, or their classifier forms for 0/1 outcomes."""
    _check(outcome_kind in OUTCOME_KINDS, f'outcome_kind must be one of {OUTCOME_KINDS}')
    if outcome_kind == 'binary':
        return [QLearner('A6-Q', binary_q_factory(tabicl_classifier_factory(classifier_checkpoint, device,
                                                                            n_estimators)),
                         seed=1103, context_cap=MAX_Q_CONTEXT_ROWS),
                QLearner('B0_hist_gbm', binary_q_factory(gbm_classifier_factory), seed=1103)]
    return [QLearner('A6-Q', tabicl_regressor_factory(regressor_checkpoint, device, n_estimators), seed=1103,
                     context_cap=MAX_Q_CONTEXT_ROWS),
            QLearner('B0_hist_gbm', gbm_regressor_factory, seed=1103)]


def default_anchor_g_learners(classifier_checkpoint: str | Path, device: str,
                              n_estimators: int = 4) -> list[GLearner]:
    """``A7-g`` (TabICLv2 origin classifier) and the ``B0_gbm_origin`` comparator."""
    return [GLearner('A7-g', tabicl_classifier_factory(classifier_checkpoint, device, n_estimators), seed=2207,
                     original_cap=MAX_G_ORIGINAL_ROWS),
            GLearner('B0_gbm_origin', gbm_classifier_factory, seed=3301)]


def default_anchor_null_learners(outcome_kind: str, regressor_checkpoint: str | Path,
                                 classifier_checkpoint: str | Path, device: str,
                                 n_estimators: int = 4) -> tuple[list[QLearner], list[GLearner]]:
    """The primary ``A6-Q + A7-g`` pair with fresh seeds, for the falsification null."""
    q = default_anchor_q_learners(outcome_kind, regressor_checkpoint, classifier_checkpoint, device, n_estimators)[0]
    g = default_anchor_g_learners(classifier_checkpoint, device, n_estimators)[0]
    return [replace(q, seed=4409)], [replace(g, seed=5501)]


def resolve_tabicl_checkpoints(regressor: str | Path | None = None,
                               classifier: str | Path | None = None) -> tuple[Path, Path]:
    """Registered local TabICLv2 checkpoints (filename, revision and SHA-256 verified; no downloads)."""
    return (verify_checkpoint(regressor or default_checkpoint(TABICL_REGRESSOR_FILE), TABICL_REGRESSOR_FILE),
            verify_checkpoint(classifier or default_checkpoint(TABICL_CLASSIFIER_FILE), TABICL_CLASSIFIER_FILE))


def resolve_device(device: str | None = None) -> str:
    """``device`` or ``cuda:0`` when available; ``torch`` is imported only here."""
    if device:
        return device
    torch = import_module('torch')
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Inputs, weights, inference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorInputs:
    """Eligible rows: ``target`` rows are scored; every row may train Q. Clusters nest in strata."""
    ids: np.ndarray
    a: np.ndarray
    y: np.ndarray
    X: np.ndarray
    clusters: np.ndarray
    strata: np.ndarray
    target: np.ndarray
    moved: np.ndarray
    outcome_kind: str = 'continuous'

    def __post_init__(self) -> None:
        _check(self.outcome_kind in OUTCOME_KINDS, f'outcome_kind must be one of {OUTCOME_KINDS}')
        y = np.asarray(self.y, dtype=np.float64)
        if self.outcome_kind == 'binary':           # binary is strict; an explicit continuous kind may take any values
            _check(bool(np.isfinite(y).all() and np.isin(y, (0.0, 1.0)).all()),
                   'a binary anchor outcome must be coded 0/1')
        cluster_table(self.a, self.clusters, self.strata)
        _check(not (np.asarray(self.moved, bool) & ~np.asarray(self.target, bool)).any(),
               'moved rows must be targets')

    def to_within_state(self) -> WithinStateInputs:
        return WithinStateInputs(ids=np.asarray(self.ids), a=np.asarray(self.a, dtype=np.float64),
                                 y=np.asarray(self.y, dtype=np.float64), X=np.asarray(self.X, dtype=np.float64),
                                 counties=np.asarray(self.clusters), states=np.asarray(self.strata),
                                 target=np.asarray(self.target, dtype=bool), moved=np.asarray(self.moved, dtype=bool))


def anchor_inputs_from_frame(frame: AnchorFrame, support: ClusterSupport,
                             rows: np.ndarray | None = None) -> AnchorInputs:
    """Inputs over ``rows`` (default all) of ``frame``; support targets and movers come from ``support``."""
    rows = np.arange(len(frame.ids)) if rows is None else np.asarray(rows)
    return AnchorInputs(ids=frame.ids[rows], a=frame.a[rows], y=frame.y[rows], X=frame.X[rows],
                        clusters=frame.clusters[rows], strata=frame.strata[rows],
                        target=support.supported[rows], moved=support.moved[rows], outcome_kind=frame.outcome_kind)


WEIGHT_SCHEMES = ('equal_cluster', 'equal_unit', 'survey')


def anchor_weights(scheme: str, clusters: Any, *, survey_weights: Any = None,
                   population_counts: Mapping[Any, int] | None = None) -> np.ndarray:
    """Row weights over target rows.

    ``equal_cluster``: ``1 / n_g`` (every PSU, canton or municipality counts once);
    ``equal_unit``: 1 per row, or the Horvitz-Thompson expansion ``N_g / n_g`` when
    ``population_counts`` gives each subsampled cluster's full count;
    ``survey``: the supplied design weights (e.g. ENDES ``HV005 / 1e6``), multiplied by
    the same expansion ``N_g / n_g`` when ``population_counts`` is given.
    """
    clusters = np.asarray(clusters)
    _check(scheme in WEIGHT_SCHEMES, f'weight scheme must be one of {WEIGHT_SCHEMES}')
    _check(len(clusters) > 0, 'no target rows to weight')
    labels, inverse, n_g = np.unique(clusters, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if scheme == 'equal_cluster':
        return 1.0 / n_g[inverse]
    expansion = np.ones(len(clusters))
    if population_counts is not None:
        missing = [c for c in labels.tolist() if c not in population_counts]
        _check(not missing, f'population counts are missing clusters {missing[:5]}')
        full = np.asarray([population_counts[c] for c in labels.tolist()], dtype=np.float64)
        _check(bool((full >= n_g).all()), 'a population count is below its sampled count')
        expansion = (full / n_g)[inverse]
    if scheme == 'equal_unit':
        return expansion
    w = np.asarray(survey_weights, dtype=np.float64) if survey_weights is not None else None
    _check(w is not None and w.shape == clusters.shape, 'survey weights are required and must align with targets')
    _check(bool(np.isfinite(w).all() and (w > 0).all()), 'survey weights must be finite and positive')
    return w * expansion


def _normalised(weights: Any, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    _check(w.shape == (n,), 'weights are misaligned')
    _check(bool(np.isfinite(w).all() and (w >= 0).all()), 'weights must be finite and nonnegative')
    _check(float(w.sum()) > 0, 'weights sum to zero')
    return w / w.sum()


def one_step_influence(y: np.ndarray, mu_a: np.ndarray, mu_d: np.ndarray, r_a: np.ndarray,
                       weights: Any) -> tuple[float, np.ndarray]:
    """Weighted one-step ``psi = sum w (mu_d - mu_a + (r_a - 1)(y - mu_a))`` and row influence ``w (phi - psi)``."""
    w = _normalised(weights, len(y))
    phi = mu_d - mu_a + (r_a - 1.0) * (y - mu_a)
    psi = float(np.sum(w * phi))
    return psi, w * (phi - psi)


def hajek_influence(y: np.ndarray, mu_a: np.ndarray, mu_d: np.ndarray, r_a: np.ndarray,
                    weights: Any) -> tuple[float, np.ndarray]:
    """Hajek one-step ``sum w p + sum w r e / sum w r - sum w e`` and its delta-method row influence.

    With ``e = y - mu_a`` and ``c = sum w r e / sum w r`` the linearisation is
    ``w [(p - p_bar) + (r / r_bar)(e - c) - (e - e_bar)]``; the second term carries
    the estimation of the ratio mean.
    """
    w = _normalised(weights, len(y))
    plugin, e = mu_d - mu_a, y - mu_a
    r_bar = float(np.sum(w * r_a))
    _check(r_bar > 0, 'the weighted ratio mean must be positive')
    c = float(np.sum(w * r_a * e)) / r_bar
    psi = float(np.sum(w * plugin)) + c - float(np.sum(w * e))
    phi = plugin + (r_a / r_bar) * (e - c) - e
    return psi, w * (phi - float(np.sum(w * phi)))


def cluster_influence(u: Any, clusters: Any, top: int = 5, *, allow_zero: bool = False) -> dict:
    """Cluster aggregation ``U_g = sum_{i in g} u_i``: SEs, ``s_max = max s_g`` and ``G_eff = 1 / sum s_g^2``.

    ``s_g = U_g^2 / D`` with ``D = sum U_g^2`` (the influence-concentration gate's
    definitions). ``D`` must be positive unless ``allow_zero``: an estimate with no
    sampling variability then reports zero SEs and undefined (``None``) concentration.
    """
    u, clusters = np.asarray(u, dtype=np.float64), np.asarray(clusters)
    _check(u.shape == clusters.shape and len(u) > 0, 'influence inputs are misaligned')
    _check(bool(np.isfinite(u).all()), 'nonfinite influence contribution')
    labels, inverse = np.unique(clusters, return_inverse=True)
    U = np.bincount(inverse.reshape(-1), weights=u, minlength=len(labels))
    D = float(np.sum(U ** 2))
    _check(D > 0 or allow_zero, 'cluster influence sums to zero (D = 0)')
    if D == 0:
        return dict(se_iid=float(np.sqrt(np.sum(u ** 2))), se_cluster=0.0, n_clusters=int(len(labels)),
                    s_max=None, g_eff=None, top_clusters=[])
    shares = U ** 2 / D
    order = np.argsort(shares, kind='stable')[::-1][:top]
    return dict(se_iid=float(np.sqrt(np.sum(u ** 2))), se_cluster=float(np.sqrt(D)), n_clusters=int(len(labels)),
                s_max=float(shares.max()), g_eff=float(1.0 / np.sum(shares ** 2)),
                top_clusters=[dict(cluster=str(labels[i]), U_g=float(U[i]), s_g=float(shares[i])) for i in order])


def per_mmhg_influence(y: Any, mu_a: Any, mu_d: Any, r_a: Any, moved: Any, delta: float, weights: Any, *,
                       hajek: bool = False) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Per-delivered-mmHg ratio ``beta = psi / s_bar`` and its delta-method row influence.

    ``s_i = delta * moved_i`` and ``s_bar = sum w s`` (normalised weights). With ``u``
    the centred numerator influence (one-step, or Hajek with ``hajek=True``) the ratio
    influence is ``[u_i - beta * w_i * (s_i - s_bar)] / s_bar``; for the one-step this is
    ``w_i (phi_i - beta s_i) / s_bar``. Both influences sum to zero.

    Returns ``(beta, psi, s_bar, u_ratio, u_numerator)``.
    """
    y, mu_a, mu_d, r_a = (np.asarray(v, dtype=np.float64) for v in (y, mu_a, mu_d, r_a))
    moved = np.asarray(moved, dtype=bool)
    _check(moved.shape == y.shape, 'moved is misaligned')
    _check(np.isfinite(delta) and delta > 0, 'delta must be positive')
    w = _normalised(weights, len(y))
    s = float(delta) * moved.astype(np.float64)
    s_bar = float(np.sum(w * s))
    _check(s_bar > 0, 'no weighted movers: the delivered shift is zero')
    psi, u_num = (hajek_influence if hajek else one_step_influence)(y, mu_a, mu_d, r_a, w)
    beta = psi / s_bar
    return beta, psi, s_bar, (u_num - beta * w * (s - s_bar)) / s_bar, u_num


def _weighted_landing_ess(r_a: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Kish ESS of the target-weighted landing mass ``w * max(r_a - 1, 0)`` over its positive entries, and its
    percentage of their count; uniform weights give ``within_state._landing_ess(r_a)`` exactly."""
    land = np.asarray(weights, dtype=np.float64) * np.maximum(np.asarray(r_a, dtype=np.float64) - 1.0, 0.0)
    positive = land[land > 0]
    if not len(positive):
        return 0.0, 0.0
    ess = float(positive.sum() ** 2 / np.sum(positive ** 2))
    return ess, float(100.0 * ess / len(positive))


def _weighted_ratio_ess(r_a: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Kish ESS of the full weighted ratio ``w * r_a`` over all rows, and its percentage of the row count."""
    v = np.asarray(weights, dtype=np.float64) * np.asarray(r_a, dtype=np.float64)
    ess = float(v.sum() ** 2 / np.sum(v ** 2)) if np.sum(v ** 2) > 0 else 0.0
    return ess, float(100.0 * ess / len(v)) if len(v) else 0.0


def _weighted_r2(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    total = float(np.sum(w * (y - np.sum(w * y) / np.sum(w)) ** 2))
    return float(1.0 - np.sum(w * (y - mu) ** 2) / total) if total > 0 else float('nan')


def score_anchor_estimate(label: str, y: Any, mu_a: Any, mu_d: Any, r_a: Any, fold_ids: Any, clusters: Any,
                          moved: Any, delta: float, weights: Any) -> dict:
    """Weighted one-step and Hajek estimates per delivered mmHg with every permanent reporting invariant."""
    y, mu_a, mu_d, r_a = (np.asarray(v, dtype=np.float64) for v in (y, mu_a, mu_d, r_a))
    fold_ids, clusters, moved = np.asarray(fold_ids), np.asarray(clusters), np.asarray(moved, dtype=bool)
    n = len(y)
    _check(n > 0 and all(len(v) == n for v in (mu_a, mu_d, r_a, fold_ids, clusters, moved)),
           'scoring inputs are misaligned or empty')
    _check(bool((fold_ids >= 0).all()), 'a target was never evaluated by an outer fold')
    _check(bool(np.isfinite(mu_a).all() and np.isfinite(mu_d).all()), 'nonfinite outcome predictions')
    _check(bool(np.isfinite(r_a).all() and (r_a > 0).all()), 'density ratio must be positive and finite')
    _check(np.isfinite(delta) and delta > 0, 'delta must be positive')
    w = _normalised(weights, n)
    plugin, correction = mu_d - mu_a, (r_a - 1.0) * (y - mu_a)
    # Per-mmHg inference uses the ratio influence (delivered shift estimated); the fixed-shift
    # numerator SE divided by the delivered shift is kept as a diagnostic only.
    beta, psi, delivered, u_ratio, u = per_mmhg_influence(y, mu_a, mu_d, r_a, moved, delta, w)
    inf, inf_fixed = cluster_influence(u_ratio, clusters, allow_zero=True), cluster_influence(u, clusters,
                                                                                              allow_zero=True)
    beta_h, psi_h, _, u_ratio_h, u_h = per_mmhg_influence(y, mu_a, mu_d, r_a, moved, delta, w, hajek=True)
    inf_h, inf_h_fixed = (cluster_influence(u_ratio_h, clusters, allow_zero=True),
                          cluster_influence(u_h, clusters, allow_zero=True))
    r2 = _weighted_r2(y, mu_a, w)
    ess, ess_pct = _weighted_landing_ess(r_a, w)
    ratio_ess, ratio_ess_pct = _weighted_ratio_ess(r_a, w)

    per_fold = {}
    for f in sorted(np.unique(fold_ids).tolist()):
        m = fold_ids == f
        share = float(w[m].sum())
        ess_f, ess_pct_f = _weighted_landing_ess(r_a[m], w[m])
        ratio_ess_f, ratio_ess_pct_f = _weighted_ratio_ess(r_a[m], w[m])
        if share <= 0.0:
            per_fold[str(f)] = dict(n=int(m.sum()), weight_share=0.0, plugin=None, correction=None,
                                    one_step=None, delivered_shift_mmhg=None, r2=None,
                                    mean_r_a=float(np.mean(r_a[m])), max_r_a=float(np.max(r_a[m])),
                                    landing_ess=0.0, landing_ess_pct=0.0,
                                    weighted_ratio_ess=0.0, weighted_ratio_ess_pct=0.0)
            continue
        wf = w[m] / share
        per_fold[str(f)] = dict(n=int(m.sum()), weight_share=share, plugin=float(np.sum(wf * plugin[m])),
                                correction=float(np.sum(wf * correction[m])),
                                one_step=float(np.sum(wf * (plugin[m] + correction[m]))),
                                delivered_shift_mmhg=float(delta * np.sum(wf * moved[m])),
                                r2=_weighted_r2(y[m], mu_a[m], wf),
                                mean_r_a=float(np.mean(r_a[m])), max_r_a=float(np.max(r_a[m])),
                                landing_ess=ess_f, landing_ess_pct=ess_pct_f,
                                weighted_ratio_ess=ratio_ess_f, weighted_ratio_ess_pct=ratio_ess_pct_f)
    se, se_h = inf['se_cluster'], inf_h['se_cluster']
    return dict(
        label=label, n_targets=n, n_movers=int(moved.sum()), n_clusters=inf['n_clusters'],
        delivered_shift_mmhg=delivered, nominal_shift_mmhg=float(delta), r2=r2,
        mean_r_a=float(np.mean(r_a)), weighted_mean_r_a=float(np.sum(w * r_a)),
        p99_r_a=float(np.percentile(r_a, 99)), max_r_a=float(np.max(r_a)), landing_ess=ess, landing_ess_pct=ess_pct,
        weighted_ratio_ess=ratio_ess, weighted_ratio_ess_pct=ratio_ess_pct,
        plugin=float(np.sum(w * plugin)), correction=float(np.sum(w * correction)),
        s_max=inf['s_max'], g_eff=inf['g_eff'],
        one_step=dict(psi=psi, per_mmhg=beta, se_iid_per_mmhg=inf['se_iid'], se_cluster_per_mmhg=se,
                      ci95_cluster_per_mmhg=[beta - Z95 * se, beta + Z95 * se], inference=inf,
                      se_fixed_shift_per_mmhg=inf_fixed['se_cluster'] / delivered,
                      inference_fixed_shift=inf_fixed),
        hajek_one_step=dict(psi=psi_h, per_mmhg=beta_h, se_cluster_per_mmhg=se_h,
                            ci95_cluster_per_mmhg=[beta_h - Z95 * se_h, beta_h + Z95 * se_h],
                            s_max=inf_h['s_max'], g_eff=inf_h['g_eff'], inference=inf_h,
                            se_fixed_shift_per_mmhg=inf_h_fixed['se_cluster'] / delivered,
                            inference_fixed_shift=inf_h_fixed),
        per_fold=per_fold)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _accepts_sample_weight(model: Any) -> bool:
    fit = getattr(model, 'fit', None)
    return callable(fit) and 'sample_weight' in inspect.signature(fit).parameters


def _fit_classifier(model: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray, weighted: bool) -> None:
    if weighted:
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)


def fit_weighted_platt_calibrated_origin(
    F0: np.ndarray, F1: np.ndarray, F_eval: np.ndarray, clusters: np.ndarray, weights: np.ndarray,
    make_classifier: ClassifierFactory, *, seed: int, ids: np.ndarray | None = None, cap: int | None = None,
    salt: str = 'a7-v2', inner_splits: int = 3, score: Literal['proba', 'decision'] = 'proba',
) -> tuple[np.ndarray, dict]:
    """Target-weighted density ratio ``r_w = dP_w(A_d, X) / dP_w(A, X)`` at eval rows by origin classification.

    ``weights`` are the training targets' target-law weights (any positive scale). Uniform
    weights reuse ``within_state.fit_platt_calibrated_origin`` unchanged. Otherwise each
    original row's factual (label 0) and shifted (label 1) pair carries its weight:

    Rows are first capped by the outcome-blind hash rank when ``cap`` is set. The inner
    folds then split the original rows by cluster, before any resampling, and out-of-fold
    logits are taken on the original validation pairs; Platt scaling is fitted on them with
    the pair weights. Training (inner and final) depends on the classifier:

    * a classifier whose ``fit`` takes ``sample_weight`` is trained on weighted pairs;
    * any other classifier (e.g. TabICL) is trained on a deterministic
      probability-proportional-to-weight resample drawn only from the training rows, which
      represents ``P_w`` restricted to them.
    """
    clusters, w = np.asarray(clusters), np.asarray(weights, dtype=np.float64)
    _check(len(F0) == len(F1) == len(clusters) == len(w) and F0.shape[1] == F1.shape[1] == F_eval.shape[1],
           'origin-classifier inputs are misaligned')
    _check(bool(np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0), 'g weights must be finite, >= 0, not all 0')
    w = w / w.mean()
    if np.allclose(w, 1.0, rtol=0.0, atol=1e-12):
        ratio, diag = fit_platt_calibrated_origin(F0, F1, F_eval, clusters, make_classifier, seed=seed, ids=ids,
                                                  cap=cap, salt=salt, inner_splits=inner_splits, score=score)
        return ratio, dict(diag, weighting='uniform')
    keep = np.arange(len(F0))
    if cap is not None:                             # outcome-blind hash cap on originals, as within_state
        _check(ids is not None and len(ids) == len(F0), 'capped origin fits require row ids')
        keep = deterministic_subsample_idx(np.asarray(ids), cap=cap, salt=f'{salt}-{seed}')
    F0, F1, clusters, w = F0[keep], F1[keep], clusters[keep], w[keep] / w[keep].mean()
    _check(len(np.unique(clusters)) >= inner_splits, 'fewer clusters than inner folds')
    first = make_classifier(seed)                  # also the first inner model (seed + 0)
    weighted = _accepts_sample_weight(first)

    def pairs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.vstack([F0[rows], F1[rows]]),
                np.concatenate([np.zeros(len(rows), dtype=int), np.ones(len(rows), dtype=int)]))

    def fit_on(model: Any, rows: np.ndarray, draw_seed: list[int]) -> None:
        if weighted:                                # weighted pairs
            X, y = pairs(rows)
            model.fit(X, y, sample_weight=np.concatenate([w[rows], w[rows]]))
        else:                                       # P_w by a deterministic weight-proportional resample of rows
            p = w[rows] / w[rows].sum()
            drawn = np.sort(np.random.default_rng(draw_seed).choice(rows, size=len(rows), replace=True, p=p))
            model.fit(*pairs(drawn))

    # Inner folds split the ORIGINAL rows by cluster before any resampling, so extreme weights can never
    # collapse the cluster groups; out-of-fold logits are taken on the original validation pairs.
    oof = np.zeros(2 * len(F0), dtype=np.float64)
    inner = []
    for i, (tr, va) in enumerate(GroupKFold(n_splits=inner_splits).split(F0, groups=clusters)):
        _check(not set(clusters[tr].tolist()) & set(clusters[va].tolist()), f'inner fold {i} shares clusters')
        model = first if i == 0 else make_classifier(seed + i)
        fit_on(model, tr, [seed, i])
        X_va, _ = pairs(va)
        logits = _raw_logits(model, X_va, score)
        oof[va], oof[len(F0) + va] = logits[:len(va)], logits[len(va):]
        inner.append(dict(n_train=int(len(tr)), n_val=int(len(va))))
    final = make_classifier(seed + 99)
    fit_on(final, np.arange(len(F0)), [seed, 99])
    y_pair = np.concatenate([np.zeros(len(F0), dtype=int), np.ones(len(F0), dtype=int)])
    clip = PLATT_LOGIT_CLIP
    platt = LogisticRegression(C=PLATT_C, solver='lbfgs')
    platt.fit(np.clip(oof, -clip, clip).reshape(-1, 1), y_pair, sample_weight=np.concatenate([w, w]))
    calibrated = platt.decision_function(np.clip(_raw_logits(final, F_eval, score), -clip, clip).reshape(-1, 1))
    ratio = np.exp(np.clip(calibrated, *ODDS_LOG_CLIP))
    return ratio, dict(slope=float(platt.coef_[0, 0]), intercept=float(platt.intercept_[0]),
                       n_originals=int(len(F0)), n_pairs=int(2 * len(F0)), inner_folds=inner,
                       weighting='sample_weight' if weighted else 'weighted_resample')


def fit_weighted_g_cross_fit(
    data: WithinStateInputs, splits: Sequence[tuple[np.ndarray, np.ndarray]], *, delta: float,
    g_learners: Sequence[GLearner], target_weights: np.ndarray, all_states: Sequence[Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    """Out-of-fold ``r_w`` at every target for each g, under one scheme's target weights.

    Same folds and features as ``fit_cross_fitted_within_state`` (training-partition
    imputation, training-target state statistics, one-hot strata, no raw A). In fold ``k``
    the training targets' weights are normalised to mean 1.
    """
    target = np.asarray(data.target, dtype=bool)
    n_t = int(target.sum())
    w_t = np.asarray(target_weights, dtype=np.float64)
    _check(w_t.shape == (n_t,), 'target weights must give one weight per target')
    t_pos = np.full(len(data.ids), -1, dtype=int)
    t_pos[target] = np.arange(n_t)
    a_d = data.shifted(delta)
    ratios = {g.name: np.zeros(n_t) for g in g_learners}
    platt: dict[str, list[dict]] = {g.name: [] for g in g_learners}
    covered = np.zeros(n_t, dtype=bool)
    for fold, (tr, ev) in enumerate(splits):
        ev_t, tr_t = ev[target[ev]], tr[target[tr]]
        if not len(ev_t):
            continue
        slots = t_pos[ev_t]
        imputer = SimpleImputer(strategy='median').fit(data.X[tr])
        X_tr_t, X_ev_t = imputer.transform(data.X[tr_t]), imputer.transform(data.X[ev_t])
        stats = fit_state_stats(data.a[tr_t], data.states[tr_t])
        F0 = encode_features_onehot(data.a[tr_t], X_tr_t, data.states[tr_t], stats, all_states)
        F1 = encode_features_onehot(a_d[tr_t], X_tr_t, data.states[tr_t], stats, all_states)
        F_ev = encode_features_onehot(data.a[ev_t], X_ev_t, data.states[ev_t], stats, all_states)
        w_fold = w_t[t_pos[tr_t]]
        _check(w_fold.sum() > 0, f'fold {fold}: training targets carry no weight')
        for g in g_learners:
            ratio, diag = fit_weighted_platt_calibrated_origin(
                F0, F1, F_ev, data.counties[tr_t], w_fold / w_fold.mean(), g.factory, seed=g.seed + 10 * fold,
                ids=data.ids[tr_t], cap=g.original_cap, salt=g.salt, inner_splits=g.inner_splits, score=g.score)
            _check(bool(np.isfinite(ratio).all() and (ratio > 0).all()), f'{g.name}: invalid density ratio')
            ratios[g.name][slots] = ratio
            platt[g.name].append(dict(diag, fold=fold))
        covered[slots] = True
    _check(bool(covered.all()), 'a target was never evaluated')
    return ratios, platt


def _scheme_g_fits(data: WithinStateInputs, splits: Sequence[tuple[np.ndarray, np.ndarray]], delta: float,
                   g_learners: Sequence[GLearner], schemes: Mapping[str, Any],
                   all_states: Sequence[Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    """One weighted g cross-fit per distinct normalised target-weight vector, keyed back to every scheme."""
    cache: dict[bytes, tuple[dict, dict]] = {}
    ratios, platt = {}, {}
    for name, w in schemes.items():
        w = np.asarray(w, dtype=np.float64)
        key = np.round(w / w.mean(), 12).tobytes()
        if key not in cache:
            cache[key] = fit_weighted_g_cross_fit(data, splits, delta=delta, g_learners=g_learners,
                                                  target_weights=w, all_states=all_states)
        ratios[name], platt[name] = cache[key]
    return ratios, platt


ANCHOR_PRIMARY_LABEL = 'A6-Q + A7-g (primary TabICLv2)'
ANCHOR_NULL_LABEL = 'NULL cluster-permuted A within stratum'
ANCHOR_COMBOS = (
    (ANCHOR_PRIMARY_LABEL, 'A6-Q', 'A7-g'),
    ('A6-Q + B0_gbm_origin (g sensitivity)', 'A6-Q', 'B0_gbm_origin'),
    ('B0_hist_gbm + A7-g (Q sensitivity)', 'B0_hist_gbm', 'A7-g'),
    ('B0_hist_gbm + B0_gbm_origin (classical)', 'B0_hist_gbm', 'B0_gbm_origin'),
)


def run_anchor(
    data: AnchorInputs, *, delta: float, q_learners: Sequence[QLearner], g_learners: Sequence[GLearner],
    null_learners: tuple[Sequence[QLearner], Sequence[GLearner]] | None = None,
    combos: Sequence[tuple[str, str, str]] = ANCHOR_COMBOS, weight_schemes: Mapping[str, Any] | None = None,
    n_splits: int = 5, null_seed: int = NULL_SEED, log: Callable[[str], None] | None = None,
) -> dict:
    """Cross-fit every learner on shared whole-cluster folds; score each (Q, g) under each weight scheme.

    ``weight_schemes`` maps a name to target-row weights (default: ``equal_cluster``
    and unweighted ``equal_unit``). Q is fitted once (``E[Y | A, X]`` does not depend on
    the target weights). Each scheme defines its own target law ``P_w(A, X)``, so g is
    cross-fitted per scheme under that scheme's weights (:func:`fit_weighted_g_cross_fit`;
    schemes with identical normalised weights share one fit) and every estimate uses its
    own scheme's ``r_w``. The null permutes cluster exposures within strata (``moved``
    fixed), reuses the outer folds and scores its first Q with its first g.
    """
    _check(n_splits >= 2, 'at least two outer folds are required')
    q_routed = route_q_learners(data.outcome_kind, q_learners)
    ws_data = data.to_within_state()
    splits = make_balanced_county_folds(ws_data.counties, ws_data.states, ws_data.a, n_splits)
    all_strata = sorted(np.unique(ws_data.states).tolist())
    target = ws_data.target
    n_t, n_m = int(target.sum()), int(ws_data.moved[target].sum())
    _check(n_m > 0, 'no movers among targets')
    c_t, y_t, moved_t = ws_data.counties[target], ws_data.y[target], ws_data.moved[target]
    schemes = dict(weight_schemes) if weight_schemes is not None else dict(
        equal_cluster=anchor_weights('equal_cluster', c_t), equal_unit=anchor_weights('equal_unit', c_t))
    _check(bool(schemes), 'at least one weight scheme is required')
    for name, w in schemes.items():
        _check(np.asarray(w).shape == (n_t,), f'weight scheme {name} must give one weight per target')

    _check(len({g.name for g in g_learners}) == len(g_learners), 'duplicate g learner names')
    fits = fit_cross_fitted_within_state(ws_data, splits, delta=delta, q_learners=q_routed, g_learners=[],
                                         all_states=all_strata, log=log)
    r_w, platt = _scheme_g_fits(ws_data, splits, delta, g_learners, schemes, all_strata)
    results: dict[str, dict[str, dict]] = {name: {} for name in schemes}
    for label, q, g in combos:
        _check(q in fits.mu_a and g in {x.name for x in g_learners}, f'{label}: unknown learner {q} or {g}')
        for name, w in schemes.items():
            results[name][label] = score_anchor_estimate(label, y_t, fits.mu_a[q], fits.mu_d[q], r_w[name][g],
                                                         fits.fold_ids, c_t, moved_t, delta, w)
    null: CrossFitResult | None = None
    null_r_w = null_platt = None
    if null_learners is not None:
        nq, ng = null_learners
        _check(bool(nq) and bool(ng), 'the null requires a Q and a g learner')
        nq = route_q_learners(data.outcome_kind, nq)
        a_null, moved_null = permute_cluster_within_stratum(ws_data.a, ws_data.moved, ws_data.counties,
                                                            ws_data.states, null_seed)
        null_data = replace(ws_data, a=a_null, moved=moved_null)
        null = fit_cross_fitted_within_state(null_data, splits, delta=delta, q_learners=nq, g_learners=[],
                                             all_states=all_strata, log=log)
        null_r_w, null_platt = _scheme_g_fits(null_data, splits, delta, ng[:1], schemes, all_strata)
        for name, w in schemes.items():
            results[name][ANCHOR_NULL_LABEL] = score_anchor_estimate(
                ANCHOR_NULL_LABEL, y_t, null.mu_a[nq[0].name], null.mu_d[nq[0].name], null_r_w[name][ng[0].name],
                null.fold_ids, c_t, moved_null[target], delta, w)
    return dict(LABELS, outcome_kind=data.outcome_kind, delta=float(delta), n_targets=n_t, n_movers=n_m,
                n_clusters=int(len(np.unique(c_t))), n_strata=len(all_strata),
                delivered_shift_mmhg=float(delta * n_m / n_t), results=results, folds=fits.folds,
                platt=platt, null_platt=null_platt, r_a=r_w, null_r_a=null_r_w, fits=fits, null=null,
                null_seed=int(null_seed) if null is not None else None)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def anchor_payload(anchor: str, outcome: str, outcome_units: str, run: Mapping[str, Any], *,
                   config: Mapping[str, Any] | None = None, audit: Mapping[str, Any] | None = None) -> dict:
    """JSON-safe payload: exploratory labels, census, every estimate and fold record (no row-level arrays)."""
    keep = ('outcome_kind', 'delta', 'n_targets', 'n_movers', 'n_clusters', 'n_strata', 'delivered_shift_mmhg',
            'results', 'folds', 'platt', 'null_platt', 'null_seed')
    payload = dict(LABELS, anchor=anchor, outcome=outcome, outcome_units=outcome_units,
                   primary_label=ANCHOR_PRIMARY_LABEL, null_label=ANCHOR_NULL_LABEL,
                   **{k: run[k] for k in keep if k in run}, config=dict(config or {}), audit=dict(audit or {}))
    payload = _jsonable(payload)
    json.dumps(payload, allow_nan=True)
    return payload


def _fmt(value: Any, spec: str = '+.4f') -> str:
    return 'n/a' if value is None or (isinstance(value, float) and not np.isfinite(value)) else format(value, spec)


def render_anchor_markdown(payload: Mapping[str, Any]) -> str:
    """Markdown readout carrying the exploratory notice and every permanent reporting invariant."""
    with_null = [scheme for scheme, models in payload['results'].items() if ANCHOR_NULL_LABEL in models]
    without_null = [scheme for scheme in payload['results'] if scheme not in with_null]
    if with_null:
        seed = payload.get('null_seed')
        null_text = ('Falsification: cluster-level within-stratum permuted-A null'
                     + (f' (seed `{seed}`)' if seed is not None else '')
                     + f', reported as `{ANCHOR_NULL_LABEL}` under weights {", ".join(f"`{s}`" for s in with_null)}.')
        if without_null:
            null_text += (' No permutation null estimate is present under weights '
                          + ', '.join(f'`{s}`' for s in without_null) + '.')
    else:
        null_text = 'Falsification: no permutation null was run for this readout (no null estimate in the results).'
    lines = [f'# Exploratory external anchor: `{payload["anchor"]}` ({payload["outcome"]})', '',
             f'**{EXPLORATORY_NOTICE}**', '',
             f'Outcome `{payload["outcome"]}` ({payload["outcome_kind"]}, {payload["outcome_units"]}); estimates are '
             f'per **delivered** mmHg (nominal shift `{payload["delta"]}` mmHg, unweighted delivered shift '
             f'`{payload["delivered_shift_mmhg"]:.4f}` mmHg). Targets `{payload["n_targets"]:,}`, movers '
             f'`{payload["n_movers"]:,}`, clusters `{payload["n_clusters"]:,}`, strata `{payload["n_strata"]}`.',
             'Design: exact one-hot stratum FE (no raw A), whole-cluster balanced 5-fold CV, cluster-grouped inner '
             '3-fold Platt calibration. ' + null_text, '',
             '## One-step estimates', '',
             '| Weights | Model | Delivered shift | OOF R^2 | Plug-in | Correction | One-step (per mmHg) | '
             'Cluster SE | 95% CI | Hajek (per mmHg) | Mean / max r_a | Landing ESS (%) | s_max / G_eff |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|']
    for scheme, models in payload['results'].items():
        for label, res in models.items():
            o, h = res['one_step'], res['hajek_one_step']
            lo, hi = o['ci95_cluster_per_mmhg']
            lines.append(
                f'| `{scheme}` | `{label}` | {res["delivered_shift_mmhg"]:.4f} | {_fmt(res["r2"], ".4f")} | '
                f'{res["plugin"]:+.5f} | {res["correction"]:+.5f} | **{o["per_mmhg"]:+.5f}** | '
                f'{o["se_cluster_per_mmhg"]:.5f} | [{lo:+.5f}, {hi:+.5f}] | {h["per_mmhg"]:+.5f} | '
                f'{res["mean_r_a"]:.3f} / {res["max_r_a"]:.1f} | {res["landing_ess"]:.0f} '
                f'({res["landing_ess_pct"]:.1f}%) | {_fmt(res["s_max"], ".3f")} / {_fmt(res["g_eff"], ".1f")} |')
    for scheme, models in payload['results'].items():
        for label, res in models.items():
            lines += ['', f'## Per-fold decomposition: `{scheme}` / `{label}`', '',
                      '| Fold | n | Weight share | Delivered shift | Plug-in | Correction | One-step | OOF R^2 | '
                      'Mean r_a | Max r_a | Landing ESS (%) |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
            for f, v in res['per_fold'].items():
                lines.append(f'| {f} | {v["n"]:,} | {v["weight_share"]:.3f} | '
                             f'{_fmt(v["delivered_shift_mmhg"], ".4f")} | '
                             f'{_fmt(v["plugin"], "+.5f")} | {_fmt(v["correction"], "+.5f")} | '
                             f'{_fmt(v["one_step"], "+.5f")} | '
                             f'{_fmt(v["r2"], ".4f")} | {v["mean_r_a"]:.3f} | {v["max_r_a"]:.1f} | '
                             f'{v["landing_ess"]:.0f} ({v["landing_ess_pct"]:.1f}%) |')
    skipped = [str(f['fold']) for f in payload.get('folds', []) if f.get('skipped')]
    if skipped:
        lines += ['', f'Outer folds without evaluation targets (skipped, no nuisance fitted): {", ".join(skipped)}.']
    return '\n'.join(lines) + '\n'


def write_anchor_outputs(out_dir: str | Path, stem: str, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` (aggregate estimates only; never microdata)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / f'{stem}.json', out / f'{stem}.md'
    json_path.write_text(json.dumps(payload, indent=2) + '\n')
    md_path.write_text(render_anchor_markdown(payload))
    return json_path, md_path


def log_anchor_to_wandb(payload: Mapping[str, Any], *, name: str, tags: Sequence[str] = (),
                        entity: str = 'goodarzilab', project: str = 'oxyformer-v2') -> None:
    """Log each estimate's headline numbers; ``wandb`` is imported only here."""
    wandb = import_module('wandb')
    config = dict(LABELS, anchor=payload['anchor'], outcome=payload['outcome'], delta=payload['delta'],
                  **payload.get('config', {}))
    run = wandb.init(entity=entity, project=project, name=name, config=config,
                     tags=['exploratory', 'anchor', payload['anchor'], *tags])
    for scheme, models in payload['results'].items():
        for label, res in models.items():
            run.log({f'{scheme}/{label}/{k}': v for k, v in dict(
                per_mmhg=res['one_step']['per_mmhg'], se_cluster_per_mmhg=res['one_step']['se_cluster_per_mmhg'],
                delivered_shift_mmhg=res['delivered_shift_mmhg'], r2=res['r2'], max_r_a=res['max_r_a'],
                landing_ess_pct=res['landing_ess_pct'], s_max=res['s_max'], g_eff=res['g_eff']).items()})
    run.finish()

