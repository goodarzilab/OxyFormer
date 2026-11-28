"""Offline tests of the exploratory external anchors: CPU only, no tabicl, torch, wandb, network or microdata."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
import yaml

from oxyformer.explore import anchors as an
from oxyformer.explore import within_state as ws
from oxyformer.exposure import physics

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def linear_q(name, seed=0):
    return ws.QLearner(name, lambda s: LinearRegression(), seed=seed)


def logistic_q(name, seed=0):
    return ws.QLearner(name, an.binary_q_factory(lambda s: LogisticRegression(C=1.0, max_iter=1000)), seed=seed)


def logistic_g(name, seed=0):
    return ws.GLearner(name, lambda s: LogisticRegression(C=1.0, max_iter=1000), seed=seed)


def cluster_design(seed=11, n_strata=4, clusters_per_stratum=30, rows=6, stratum_gap=10.0, spread=8.0,
                   vary_rows=True):
    """Cluster-assigned exposure; strata sit ``stratum_gap`` mmHg apart (strong between-stratum confounding).

    Cluster sizes vary in ``rows +- 3`` (when ``vary_rows``), so equal-cluster and equal-unit weights differ.
    """
    rng = np.random.default_rng(seed)
    ids, clusters, strata, a, a_s, re = [], [], [], [], [], []
    for s in range(n_strata):
        for c in range(clusters_per_stratum):
            a_c = stratum_gap * s + rng.uniform(0.0, spread)
            effect_c = rng.normal(0.0, 0.3)
            size = int(rng.integers(max(rows - 3, 2), rows + 4)) if vary_rows else rows
            for t in range(size):
                ids.append(f'S{s}C{c:03d}R{t}')
                clusters.append(f'S{s}C{c:03d}')
                strata.append(f'S{s}')
                a.append(a_c)
                a_s.append(stratum_gap * s)
                re.append(effect_c)
    X = rng.normal(size=(len(ids), 2))
    X[rng.random(len(ids)) < 0.05, 1] = np.nan
    return dict(ids=np.asarray(ids, dtype=object), clusters=np.asarray(clusters), strata=np.asarray(strata),
                a=np.asarray(a), a_s=np.asarray(a_s), re=np.asarray(re), X=X, rng=rng)


def continuous_outcome(d, effect=0.5, confound=3.0):
    noise = d['rng'].normal(0.0, 1.0, len(d['a']))
    return (confound * d['a_s'] + effect * (d['a'] - d['a_s']) + d['X'][:, 0] + 0.5 * np.nan_to_num(d['X'][:, 1])
            + d['re'] + noise)


def binary_outcome(d, slope=0.5):
    logit = -1.0 + 0.05 * d['a_s'] + slope * (d['a'] - d['a_s'] - 4.0) + 0.5 * d['X'][:, 0] + d['re']
    return (d['rng'].random(len(d['a'])) < expit(logit)).astype(float)


def anchor_inputs(d, y, kind, delta=2.0):
    support = an.compute_cluster_support(d['a'], d['strata'], d['clusters'], delta=delta, bin_width=2.0)
    return an.AnchorInputs(ids=d['ids'], a=d['a'], y=y, X=d['X'], clusters=d['clusters'], strata=d['strata'],
                           target=support.supported, moved=support.moved, outcome_kind=kind), support


# ---------------------------------------------------------------------------
# exploratory labelling and lazy imports
# ---------------------------------------------------------------------------

def test_module_docstring_and_labels_carry_exploratory_notice():
    assert an.EXPLORATORY_NOTICE == 'Exploratory; certifies no release criterion.'
    assert an.EXPLORATORY_NOTICE in an.__doc__
    assert an.LABELS['exploratory'] is True and an.LABELS['certifies_release_criteria'] is False
    assert an.LABELS['effect_release_authorized'] is False and an.LABELS['notice'] == an.EXPLORATORY_NOTICE


def test_module_imports_without_tabicl_torch_or_wandb():
    code = ('import sys, oxyformer.explore.anchors; '
            'bad = [m for m in ("tabicl", "torch", "wandb") if m in sys.modules]; assert not bad, bad')
    env = dict(os.environ, PYTHONPATH=str(Path(an.__file__).resolve().parents[2]))
    subprocess.run([sys.executable, '-c', code], check=True, env=env)


def test_default_learners_build_without_importing_tabicl():
    q = an.default_anchor_q_learners('binary', 'r.ckpt', 'c.ckpt', 'cpu')
    assert [x.name for x in q] == ['A6-Q', 'B0_hist_gbm'] and all(x.factory.outcome_kind == 'binary' for x in q)
    q = an.default_anchor_q_learners('continuous', 'r.ckpt', 'c.ckpt', 'cpu')
    assert all(getattr(x.factory, 'outcome_kind', 'continuous') == 'continuous' for x in q)
    assert [g.name for g in an.default_anchor_g_learners('c.ckpt', 'cpu')] == ['A7-g', 'B0_gbm_origin']
    nq, ng = an.default_anchor_null_learners('binary', 'r.ckpt', 'c.ckpt', 'cpu')
    assert nq[0].name == 'A6-Q' and nq[0].factory.outcome_kind == 'binary' and ng[0].name == 'A7-g'
    assert an.resolve_device('cpu') == 'cpu'


# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976 physics
# ---------------------------------------------------------------------------

def test_physics_constants_are_the_approved_owner_exposure_constants():
    approved = yaml.safe_load((REPO / 'configs' / 'approvals.yaml').read_text())['owner_decisions']['exposure']
    assert (an.P0, an.T0, an.L, an.g0, an.M, an.R, an.fO2, an.pH2O) == (
        760.0, 288.15, 0.0065, 9.80665, 0.0289644, 8.31432, 0.2093, 47.0)
    assert (an.P0, an.T0, an.L) == (approved['sea_level_pressure_mmhg'], approved['sea_level_temperature_k'],
                                     approved['lapse_rate_k_per_m'])
    assert (an.g0, an.M, an.R) == (approved['gravity_m_per_s2'], approved['molar_mass_air_kg_per_mol'],
                                   approved['gas_constant_j_per_mol_k'])
    assert (an.fO2, an.pH2O) == (approved['inspired_o2_fraction'], approved['water_vapour_pressure_mmhg'])


def test_pio2_deficit_exact_values():
    assert an.elevation_m_to_pio2_deficit_mmhg(0.0) == 0.0
    exponent = 9.80665 * 0.0289644 / (8.31432 * 0.0065)
    expected = 0.2093 * (760.0 - 760.0 * (1.0 - 0.0065 * 2500.0 / 288.15) ** exponent)
    assert float(an.elevation_m_to_pio2_deficit_mmhg(2500.0)) == pytest.approx(expected, rel=1e-12)
    assert float(an.elevation_m_to_pio2_deficit_mmhg(2500.0)) == pytest.approx(41.825, abs=1e-3)
    assert float(an.elevation_m_to_pio2_deficit_mmhg(2309.0)) == pytest.approx(39.0, abs=0.02)
    grid = np.linspace(-400.0, 5100.0, 56)
    np.testing.assert_allclose(an.elevation_m_to_pio2_deficit_mmhg(grid), physics.oxygen_deficit_mmhg(grid),
                               rtol=0, atol=1e-10)
    assert (np.diff(an.elevation_m_to_pio2_deficit_mmhg(grid)) > 0).all()
    for bad in (np.nan, -600.0, 12000.0):
        with pytest.raises(ValueError):
            an.elevation_m_to_pio2_deficit_mmhg([0.0, bad])


def test_transform_before_aggregate_and_jensen_gap():
    z = np.array([200.0, 1500.0, 3800.0, 4300.0])
    w = np.array([5.0, 1.0, 2.0, 0.5])
    transform_first = an.population_weighted_pio2_deficit(z, w)
    assert transform_first == pytest.approx(float(np.sum(w * an.elevation_m_to_pio2_deficit_mmhg(z)) / w.sum()))
    aggregate_first = an.aggregate_then_transform_pio2_deficit(z, w)
    assert aggregate_first == pytest.approx(float(an.elevation_m_to_pio2_deficit_mmhg(np.sum(w * z) / w.sum())))
    # The deficit is concave in elevation: averaging elevations first overstates a heterogeneous area.
    assert transform_first < aggregate_first
    gap = an.jensen_gap_mmhg(z, w)
    assert gap == pytest.approx(transform_first - aggregate_first) and gap < -0.1
    flat = np.full(4, 2500.0)
    assert an.jensen_gap_mmhg(flat, w) == pytest.approx(0.0, abs=1e-12)
    per_area = an.population_weighted_pio2_deficit(z, w, groups=['A', 'A', 'B', 'B'])
    assert per_area['A'] == pytest.approx(an.population_weighted_pio2_deficit(z[:2], w[:2]))
    assert per_area['B'] == pytest.approx(an.population_weighted_pio2_deficit(z[2:], w[2:]))
    with pytest.raises(ValueError, match='zero total weight'):
        an.population_weighted_pio2_deficit(z, [1.0, 1.0, 0.0, 0.0], groups=['A', 'A', 'B', 'B'])
    with pytest.raises(ValueError):
        an.population_weighted_pio2_deficit(z, [1.0, -1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# guardrails
# ---------------------------------------------------------------------------

def test_peru_spec_defaults_are_the_approved_recipe():
    spec = an.validate_peru_endes_spec()
    assert (spec.outcome_field, spec.hb_divisor, spec.status_field, spec.status_value) == ('HC53', 10.0, 'HC55', 0)
    assert tuple(spec.age_months) == (4, 59) and spec.exposure_field == 'HV040'
    assert not set(spec.covariates) & an.PERU_FORBIDDEN_FIELDS


@pytest.mark.parametrize('override', [
    dict(outcome_field='HC56'), dict(outcome_field='HC57'), dict(outcome_field='hc57'),
    dict(covariates=an.PERU_BASELINE_COVARIATES + ('HC70',)), dict(covariates=an.PERU_BASELINE_COVARIATES + ('HC2',)),
    dict(covariates=an.PERU_BASELINE_COVARIATES + ('HC3',)), dict(covariates=('HC1', 'HC56')),
    dict(covariates=an.PERU_BASELINE_COVARIATES + ('HV040',)), dict(status_value=1), dict(hb_divisor=1.0),
    dict(age_months=(6, 59)), dict(age_months=(4, 71)), dict(exposure_field='HV041'),
    dict(covariates=('HC1', 'HC1')),
])
def test_peru_spec_rejects_adjusted_hb_anthropometry_and_recipe_changes(override):
    with pytest.raises(ValueError):
        an.validate_peru_endes_spec(**override)


def test_peru_spec_rejects_unknown_mapping_keys():
    with pytest.raises(ValueError, match='unknown'):
        an.validate_peru_endes_spec({'adjusted_field': 'HC56'})


@pytest.mark.parametrize('override', [
    dict(residence_canton_field='cant_nac'), dict(residence_province_field='prov_nac'),
    dict(residence_canton_field='cant_insc'),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('sem_gest',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('con_pren',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('talla',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('tipo_part',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('apgar5',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('hijos_viv',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('SEM_GEST',)),
    dict(covariates=an.ECUADOR_PRIMARY_COVARIATES + ('sexo',)),
    dict(covariates=('edad_mad', 'niv_inst')), dict(outcome='apgar1'), dict(birthweight_field='talla'),
])
def test_ecuador_spec_rejects_delivery_geography_post_outcome_and_unapproved(override):
    with pytest.raises(ValueError):
        an.validate_ecuador_inec_spec(**override)


def test_ecuador_spec_accepts_primary_and_est_civil_sensitivity():
    assert an.validate_ecuador_inec_spec().covariates == ('edad_mad', 'niv_inst', 'etnia')
    spec = an.validate_ecuador_inec_spec(covariates=['edad_mad', 'niv_inst', 'etnia', 'est_civil'], outcome='lbw')
    assert spec.covariates[-1] == 'est_civil' and spec.outcome == 'lbw'


@pytest.mark.parametrize('override', [
    dict(residence_municipality_field='COD_MUNIC'), dict(residence_department_field='COD_DPTO'),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'T_GES')),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'IDPERTET')),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'APGAR1')),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'NUMCONSUL')),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'TALLA_NAC')),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'N_HIJOSV')),
    dict(covariates=('EDAD_MADRE', 'NIV_EDUM', 'idpertet')),
    dict(covariates=('year', 'EDAD_MADRE')), dict(covariates=('EDAD_MADRE', 'NIV_EDUM', 'FECHA_NAC')),
    dict(outcome='APGAR2'), dict(birthweight_field='TALLA_NAC'),
])
def test_colombia_spec_rejects_occurrence_geography_post_outcome_and_unapproved(override):
    with pytest.raises(ValueError):
        an.validate_colombia_dane_spec(**override)


def test_colombia_spec_accepts_primary_and_sensitivity():
    spec = an.validate_colombia_dane_spec(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'EST_CIVM', 'SEXO'))
    assert spec.residence_department_field == 'CODPTORE' and spec.residence_municipality_field == 'CODMUNRE'
    assert an.validate_colombia_dane_spec(covariates=('EDAD_MADRE', 'NIV_EDUM')).outcome == 'lbw'


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def peru_modules(year=2023, hc56='999', hv040_second=None):
    r0 = pd.DataFrame({'﻿HHID': ['1', '2', '3', '4', '5'], 'HV001': ['10', '10', '20', '30', '30'],
                       'HV005': ['1000000', '2000000', '1500000', '1000000', '500000'],
                       'HV024': ['1', '1', '2', '2', '2'],
                       'HV040': ['100', '100' if hv040_second is None else hv040_second, '2500', '3800', '3800'],
                       'HV009': ['4', '5', '6', '3', '2'], 'HV014': ['1', '2', '1', '0', '1'],
                       'HV025': ['1', '1', '2', '2', '2']})
    r23 = pd.DataFrame({'HHID': ['1', '2', '3', '4', '5'], 'HV270': ['1', '2', '3', '4', '5'],
                        'HV271': ['-100', '0', '100', '200', '300']})
    r6 = pd.DataFrame({'HHID': [' 1', '1', '2', '3', '4', '5', '5'], 'HC0': ['1', '2', '1', '1', '1', '1', '2'],
                       'HC1': ['12', '3', '30', '40', '58', '20', '24'],
                       'HC53': ['110', '120', '125', '140', '999', '131', '118'],
                       'HC55': ['0', '0', '0', '0', '0', '1', '0'],
                       'HC56': [hc56] * 7, 'HC57': ['1'] * 7, 'HC70': ['-9999'] * 7, 'HC2': ['80'] * 7,
                       'HC27': ['1', '2', '1', '2', '1', '2', '1'], 'HC61': ['3'] * 7, 'HC62': [''] * 7,
                       'HC64': ['1'] * 7})
    return {'RECH0': r0, 'RECH23': r23, 'RECH6': r6}


def test_peru_loader_reads_raw_hc53_only_and_ignores_adjusted_columns():
    frame = an.load_peru_endes_frame({2023: peru_modules()})
    # kept: (1,1) age 12; (2,1); (3,1); (5,2). Dropped: age 3 m, HC53=999, HC55=1.
    assert list(frame.ids) == ['2023:1:1', '2023:2:1', '2023:3:1', '2023:5:2']
    np.testing.assert_allclose(frame.y, [11.0, 12.5, 14.0, 11.8])
    np.testing.assert_allclose(frame.a, an.elevation_m_to_pio2_deficit_mmhg([100, 100, 2500, 3800]))
    np.testing.assert_allclose(frame.survey_weights, [1.0, 2.0, 1.5, 0.5])
    assert list(frame.clusters) == ['2023:10', '2023:10', '2023:20', '2023:30']
    assert list(frame.strata) == ['01', '01', '02', '02']
    assert frame.outcome == 'hb_g_dl' and frame.outcome_kind == 'continuous'
    assert not set(frame.feature_names) & an.PERU_FORBIDDEN_FIELDS
    assert set(frame.feature_names) <= set(an.PERU_BASELINE_COVARIATES) and 'year' in frame.feature_names
    assert 'HV201' in frame.audit['absent_covariates']
    changed = an.load_peru_endes_frame({2023: peru_modules(hc56='50')})
    np.testing.assert_array_equal(changed.y, frame.y)
    np.testing.assert_array_equal(np.nan_to_num(changed.X), np.nan_to_num(frame.X))


def test_peru_loader_pools_years_with_distinct_panel_clusters_and_rejects_bad_recipes():
    frame = an.load_peru_endes_frame({2023: peru_modules(2023), 2024: peru_modules(2024)})
    assert len(frame.ids) == 8 and len(set(frame.ids)) == 8
    assert {'2023:10', '2024:10'} <= set(frame.clusters.tolist())
    with pytest.raises(ValueError, match='forbidden'):
        an.load_peru_endes_frame({2023: peru_modules()}, spec={'outcome_field': 'HC57'})
    with pytest.raises(ValueError, match='exposure varies within a cluster'):
        an.load_peru_endes_frame({2023: peru_modules(hv040_second='900')})
    modules = peru_modules()
    modules['RECH6'] = modules['RECH6'].drop(columns=['HC53'])
    with pytest.raises(ValueError, match='missing columns'):
        an.load_peru_endes_frame({2023: modules})


def ecuador_tables(with_province=False):
    births = pd.DataFrame({
        'prov_res': ['Pichincha', 'Pichincha', 'Guayas', 'Guayas', 'Manabí', 'Pichincha', 'Guayas'],
        'cant_res': ['Quito', 'Quito', 'El Empalme', 'Guayaquil', 'Bolívar', 'Cayambe', 'Guayaquil'],
        'cant_nac': ['Guayaquil', 'Guayaquil', 'Quito', 'Quito', 'Quito', 'Quito', 'Quito'],
        'peso': ['3100', '2400', '3300', '2900', '3000', '450', '2450'],
        'edad_mad': ['25', '70', '31', '19', '28', '30', '22'],
        'niv_inst': ['Superior', 'Ignorado', 'Primaria', 'Secundaria', 'Primaria', 'Superior', ''],
        'etnia': ['Mestiza', 'Indígena', 'Mestiza', 'Afro', 'Mestiza', 'Mestiza', 'Sin información'],
        'est_civil': ['Casada', 'Soltera', 'Unida', 'Soltera', 'Casada', 'Casada', 'Unida'],
        'sem_gest': ['39', '30', '40', '38', '39', '25', '37'], 'talla': ['50'] * 7})
    exposure = pd.DataFrame({'shapeName': ['Quito', 'Empalme', 'Guayaquil', 'Bolivar', 'Bolivar', 'Cayambe'],
                             'oxygen_deficit_mmhg': [42.0, 1.0, 0.5, 30.0, 5.0, 44.0]})
    if with_province:
        exposure['province'] = ['Pichincha', 'Guayas', 'Guayas', 'Carchi', 'Manabi', 'Pichincha']
    return births, exposure


def test_ecuador_loader_places_exposure_at_maternal_residence_and_stops_on_ambiguity():
    births, exposure = ecuador_tables()
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(births, exposure)
    frame = an.load_ecuador_inec_frame(births, exposure, on_unmatched='drop')
    assert frame.audit['ambiguous_exposure_names'] == ['BOLIVAR']
    assert frame.audit['excluded_unmatched_cantons'] == {'Bolívar': 1}
    # residence cantons Quito, Quito, El Empalme (alias), Guayaquil, Guayaquil; cant_nac never used
    np.testing.assert_allclose(frame.a, [42.0, 42.0, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(frame.y, [3100.0, 2400.0, 3300.0, 2900.0, 2450.0])
    assert list(frame.clusters) == ['PICHINCHA::QUITO', 'PICHINCHA::QUITO', 'GUAYAS::EMPALME',
                                    'GUAYAS::GUAYAQUIL', 'GUAYAS::GUAYAQUIL']
    assert list(frame.strata) == ['PICHINCHA', 'PICHINCHA', 'GUAYAS', 'GUAYAS', 'GUAYAS']
    assert np.isnan(frame.X[1, 0]) and frame.feature_names[0] == 'edad_mad'
    assert 'niv_inst_MISSING' in frame.feature_names and 'etnia_MISSING' in frame.feature_names
    assert not an._forbidden_hits(frame.feature_names, an.ECUADOR_FORBIDDEN_FIELDS)
    assert not any(n.startswith('est_civil') for n in frame.feature_names)
    lbw = an.load_ecuador_inec_frame(births, exposure, {'outcome': 'lbw'}, on_unmatched='drop', stratum='national')
    np.testing.assert_array_equal(lbw.y, [0.0, 1.0, 0.0, 0.0, 1.0])
    assert lbw.outcome_kind == 'binary' and set(lbw.strata) == {'ECU_ALL'}
    with pytest.raises(ValueError, match='cant_nac'):
        an.load_ecuador_inec_frame(births, exposure, {'residence_canton_field': 'cant_nac'})


def test_ecuador_loader_disambiguates_repeated_canton_names_by_province():
    births, exposure = ecuador_tables(with_province=True)
    frame = an.load_ecuador_inec_frame(births, exposure)
    bolivar = frame.clusters == 'MANABI::BOLIVAR'
    assert bolivar.sum() == 1 and frame.a[bolivar][0] == 5.0
    assert frame.audit['ambiguous_exposure_names'] == []


def colombia_tables():
    births = pd.DataFrame({
        'CODPTORE': ['5', '05', '11', '11', '25', '05'], 'CODMUNRE': ['1', '001', '1', '001', '999', '2'],
        'COD_DPTO': ['11'] * 6, 'COD_MUNIC': ['001'] * 6,
        'PESO_NAC': ['3', '6', '01', '7', '5', '9'], 'ANO': ['2023', '2024', '2025', '2025', '2023', '2024'],
        'EDAD_MADRE': ['3', '4', '9', '2', '3', '3'], 'NIV_EDUM': ['04', '06', '99', '13', '4', '4'],
        'SEXO': ['1', '2', '1', '2', '1', '1'], 'T_GES': ['4'] * 6, 'IDPERTET': ['6'] * 6})
    exposure = pd.DataFrame({'cod_mpio': ['05001', '11001', '5002'], 'oxygen_deficit_mmhg': [28.18, 41.0, 36.1]})
    return births, exposure


def test_colombia_loader_places_exposure_at_maternal_residence_divipola():
    births, exposure = colombia_tables()
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_colombia_dane_frame(births, exposure)
    frame = an.load_colombia_dane_frame(births, exposure, on_unmatched='drop')
    # PESO_NAC 9 is not a band; 25999 is unmatched. COD_DPTO/COD_MUNIC (11001) never place a birth.
    assert frame.audit['excluded_unmatched_municipalities'] == {'25999': 1}
    assert list(frame.clusters) == ['05001', '05001', '11001', '11001']
    np.testing.assert_allclose(frame.a, [28.18, 28.18, 41.0, 41.0])
    np.testing.assert_array_equal(frame.y, [1.0, 0.0, 1.0, 0.0])
    assert list(frame.strata) == ['05', '05', '11', '11'] and frame.outcome_kind == 'binary'
    assert frame.feature_names[0] == 'year_c'
    np.testing.assert_allclose(frame.X[:, 0], [-1.0, 0.0, 1.0, 1.0])
    assert 'EDAD_MADRE_MISSING' in frame.feature_names and 'NIV_EDUM_MISSING' in frame.feature_names
    assert not an._forbidden_hits(frame.feature_names, an.COLOMBIA_FORBIDDEN_FIELDS)
    assert not any(n.startswith('SEXO') for n in frame.feature_names)
    mid = an.load_colombia_dane_frame(births, exposure, {'outcome': 'bw_mid_g'}, on_unmatched='drop')
    np.testing.assert_allclose(mid.y, [1750.0, 3250.0, 750.0, 3750.0])
    assert mid.outcome_kind == 'continuous'
    vlbw = an.load_colombia_dane_frame(births, exposure, {'outcome': 'vlbw'}, on_unmatched='drop')
    np.testing.assert_array_equal(vlbw.y, [0.0, 0.0, 1.0, 0.0])
    sens = an.load_colombia_dane_frame(births, exposure, {'covariates': ('year', 'EDAD_MADRE', 'NIV_EDUM', 'SEXO')},
                                       on_unmatched='drop')
    assert any(n.startswith('SEXO_') for n in sens.feature_names)
    for bad in ({'residence_municipality_field': 'COD_MUNIC'}, {'covariates': ('EDAD_MADRE', 'NIV_EDUM', 'T_GES')}):
        with pytest.raises(ValueError):
            an.load_colombia_dane_frame(births, exposure, bad, on_unmatched='drop')


# ---------------------------------------------------------------------------
# cluster support, subsampling
# ---------------------------------------------------------------------------

def support_frame():
    """Stratum A: bins 0 | 1 1 1 | 2 2 2 | 3 (width 2); stratum B: one lone cluster."""
    exposures = {'a0': 0.5, 'a1': 2.1, 'a2': 2.5, 'a3': 3.0, 'a4': 4.2, 'a5': 4.9, 'a6': 5.5, 'a7': 6.3, 'b0': 2.2}
    clusters = np.repeat(list(exposures), 3)
    a = np.asarray([exposures[c] for c in clusters])
    strata = np.asarray(['A' if c.startswith('a') else 'B' for c in clusters])
    return a, strata, clusters


def test_cluster_support_counts_other_clusters_and_broadcasts_to_rows():
    a, strata, clusters = support_frame()
    s = an.compute_cluster_support(a, strata, clusters, delta=2.0, bin_width=2.0)
    sup = {c: bool(v) for c, v in zip(clusters, s.supported)}
    mov = {c: bool(v) for c, v in zip(clusters, s.moved)}
    assert not sup['a0'] and not sup['a7'] and not sup['b0']       # tails and a lone stratum never support
    assert all(sup[c] for c in ('a1', 'a2', 'a3', 'a4', 'a5', 'a6'))
    assert mov['a1'] and mov['a2'] and mov['a3']                    # +2 lands in occupied bin 2
    assert not mov['a4'] and not mov['a5'] and not mov['a6']        # +2 lands in tail bin 3: stay
    assert s.n_targets == 18 and s.n_movers == 9 and s.delivered_shift == pytest.approx(1.0)
    for c in np.unique(clusters):                                   # whole clusters share support
        assert len(set(s.supported[clusters == c])) == 1 and len(set(s.moved[clusters == c])) == 1


def test_cluster_support_needs_two_other_clusters_and_cluster_assigned_exposure():
    a, strata, clusters = support_frame()
    keep = clusters != 'a3'                                         # bin 1 now holds a1, a2 only
    s = an.compute_cluster_support(a[keep], strata[keep], clusters[keep], delta=2.0, bin_width=2.0)
    assert not s.supported[clusters[keep] == 'a1'].any()            # only one *other* cluster in its bin
    bad = a.copy()
    bad[0] += 0.3
    with pytest.raises(ValueError, match='exposure varies within a cluster'):
        an.compute_cluster_support(bad, strata, clusters, delta=2.0)
    spans = strata.copy()
    spans[0] = 'B'
    with pytest.raises(ValueError, match='spans more than one stratum'):
        an.compute_cluster_support(a, spans, clusters, delta=2.0)


def test_cluster_capped_subsample_is_outcome_blind_and_keeps_every_cluster():
    clusters = np.asarray(['m1'] * 10 + ['m2'] * 2 + ['m3'] * 5)
    ids = np.asarray([f'col:{i}' for i in range(len(clusters))], dtype=object)
    idx, counts = an.cluster_capped_subsample(ids, clusters, per_cluster_cap=3, salt='t')
    assert counts == {'m1': 10, 'm2': 2, 'm3': 5}
    kept = pd.Series(clusters[idx]).value_counts().to_dict()
    assert kept == {'m1': 3, 'm2': 2, 'm3': 3} and (np.diff(idx) > 0).all()
    again, _ = an.cluster_capped_subsample(ids, clusters, per_cluster_cap=3, salt='t')
    np.testing.assert_array_equal(idx, again)
    perm = np.random.default_rng(0).permutation(len(ids))
    shuffled, _ = an.cluster_capped_subsample(ids[perm], clusters[perm], per_cluster_cap=3, salt='t')
    assert set(ids[perm][shuffled].tolist()) == set(ids[idx].tolist())   # depends on ids only, never row order


# ---------------------------------------------------------------------------
# cluster-level within-stratum permutation null
# ---------------------------------------------------------------------------

def test_permute_cluster_within_stratum_preserves_design():
    d = cluster_design(seed=3, n_strata=3, clusters_per_stratum=8, rows=4)
    moved = d['rng'].random(len(d['a'])) < 0.5
    a_null, moved_null = an.permute_cluster_within_stratum(d['a'], moved, d['clusters'], d['strata'], seed=5)
    np.testing.assert_array_equal(moved_null, moved)
    assert moved_null is not moved
    for c in np.unique(d['clusters']):                              # every row of a cluster shares one value
        assert np.ptp(a_null[d['clusters'] == c]) == 0.0
    for s in np.unique(d['strata']):                                # each stratum keeps its cluster exposures
        m = d['strata'] == s
        before = np.sort(pd.Series(d['a'][m]).groupby(d['clusters'][m]).first().to_numpy())
        after = np.sort(pd.Series(a_null[m]).groupby(d['clusters'][m]).first().to_numpy())
        np.testing.assert_array_equal(before, after)
    assert not np.array_equal(a_null, d['a'])
    again, _ = an.permute_cluster_within_stratum(d['a'], moved, d['clusters'], d['strata'], seed=5)
    np.testing.assert_array_equal(again, a_null)
    other, _ = an.permute_cluster_within_stratum(d['a'], moved, d['clusters'], d['strata'], seed=6)
    assert not np.array_equal(other, a_null)
    # A row-level permutation (within_state's tract null) would split clusters: the reason for the cluster null.
    row_null, _ = ws.permute_within_state(d['a'], moved, d['strata'], seed=5)
    assert any(np.ptp(row_null[d['clusters'] == c]) > 0 for c in np.unique(d['clusters']))


def test_permute_cluster_within_stratum_rejects_row_assigned_exposure():
    d = cluster_design(seed=3, n_strata=2, clusters_per_stratum=4, rows=3)
    a = d['a'].copy()
    a[1] += 1.0
    with pytest.raises(ValueError, match='exposure varies within a cluster'):
        an.permute_cluster_within_stratum(a, np.zeros(len(a), bool), d['clusters'], d['strata'])
    with pytest.raises(ValueError, match='misaligned'):
        an.permute_cluster_within_stratum(d['a'], np.zeros(3, bool), d['clusters'], d['strata'])


def test_cluster_null_recovers_zero_under_strong_stratum_confounding():
    d = cluster_design(seed=11)
    y = continuous_outcome(d, effect=0.5, confound=3.0)
    data, _ = anchor_inputs(d, y, 'continuous')
    combos = [('lin + logit', 'lin', 'logit')]
    run = an.run_anchor(data, delta=2.0, q_learners=[linear_q('lin')], g_learners=[logistic_g('logit')],
                        null_learners=([linear_q('lin', 7)], [logistic_g('logit', 9)]), combos=combos)
    for scheme in ('equal_cluster', 'equal_unit'):
        res = run['results'][scheme]['lin + logit']
        real, null = res['one_step'], run['results'][scheme][an.ANCHOR_NULL_LABEL]['one_step']
        assert res['plugin'] / res['delivered_shift_mmhg'] == pytest.approx(0.5, abs=0.08)
        lo, hi = real['ci95_cluster_per_mmhg']
        assert lo < 0.5 < hi                                      # the true effect is covered
        assert abs(null['per_mmhg']) < 4.0 * null['se_cluster_per_mmhg'] and abs(null['per_mmhg']) < 0.1
    assert run['null_seed'] == an.NULL_SEED
    null_res = run['results']['equal_cluster'][an.ANCHOR_NULL_LABEL]
    assert null_res['delivered_shift_mmhg'] == pytest.approx(run['results']['equal_cluster']['lin + logit'][
        'delivered_shift_mmhg'])


# ---------------------------------------------------------------------------
# binary outcome routing
# ---------------------------------------------------------------------------

class SpyClassifier(LogisticRegression):
    proba_calls: list = []

    def predict(self, X):
        raise AssertionError('Q must use predict_proba on binary outcomes')

    def predict_proba(self, X):
        SpyClassifier.proba_calls.append(len(X))
        return super().predict_proba(X)


def test_binary_outcomes_route_through_classifier_predict_proba():
    d = cluster_design(seed=21, rows=8)
    y = binary_outcome(d)
    data, _ = anchor_inputs(d, y, 'binary')
    SpyClassifier.proba_calls.clear()
    q = ws.QLearner('spy', an.binary_q_factory(lambda s: SpyClassifier(max_iter=1000)), seed=0)
    run = an.run_anchor(data, delta=2.0, q_learners=[q], g_learners=[logistic_g('logit')],
                        combos=[('spy + logit', 'spy', 'logit')])
    assert len(SpyClassifier.proba_calls) >= 10                     # mu_a and mu_d in each of 5 folds
    fits = run['fits']
    for mu in (fits.mu_a['spy'], fits.mu_d['spy']):
        assert ((mu > 0) & (mu < 1)).all()


def test_binary_outcomes_never_accept_a_regressor():
    d = cluster_design(seed=21, n_strata=2, clusters_per_stratum=20, rows=4)
    y = binary_outcome(d)
    data, _ = anchor_inputs(d, y, 'binary')
    with pytest.raises(ValueError, match='continuous Q learner cannot model a binary outcome'):
        an.run_anchor(data, delta=2.0, q_learners=[linear_q('lin')], g_learners=[logistic_g('g')],
                      combos=[('x', 'lin', 'g')])
    with pytest.raises(ValueError, match='predict_proba'):
        an.ProbabilityQ(LinearRegression())
    with pytest.raises(ValueError, match='predict_proba'):
        an.binary_q_factory(lambda s: LinearRegression())(0)

    class Impostor:                                                 # claims binary, builds a regressor
        outcome_kind = 'binary'

        def __call__(self, seed):
            return LinearRegression()
    with pytest.raises(ValueError, match='only through predict_proba'):
        an.run_anchor(data, delta=2.0, q_learners=[ws.QLearner('imp', Impostor(), 0)], g_learners=[logistic_g('g')],
                      combos=[('x', 'imp', 'g')])
    cont = an.AnchorInputs(ids=d['ids'], a=d['a'], y=continuous_outcome(d), X=d['X'], clusters=d['clusters'],
                           strata=d['strata'], target=data.target, moved=data.moved, outcome_kind='continuous')
    with pytest.raises(ValueError, match='binary Q learner cannot model a continuous outcome'):
        an.run_anchor(cont, delta=2.0, q_learners=[logistic_q('lq')], g_learners=[logistic_g('g')],
                      combos=[('x', 'lq', 'g')])


def test_outcome_kind_must_match_outcome_coding():
    d = cluster_design(seed=2, n_strata=2, clusters_per_stratum=6, rows=3)
    common = dict(ids=d['ids'], a=d['a'], X=d['X'], clusters=d['clusters'], strata=d['strata'],
                  target=np.ones(len(d['a']), bool), moved=np.zeros(len(d['a']), bool))
    with pytest.raises(ValueError, match='0/1 outcome declared continuous'):
        an.AnchorInputs(y=binary_outcome(d), outcome_kind='continuous', **common)
    with pytest.raises(ValueError, match='coded 0/1'):
        an.AnchorInputs(y=continuous_outcome(d), outcome_kind='binary', **common)
    with pytest.raises(ValueError, match='outcome_kind'):
        an.AnchorInputs(y=continuous_outcome(d), outcome_kind='count', **common)
    q = an.ProbabilityQ(LogisticRegression())
    with pytest.raises(ValueError, match='single class'):
        q.fit(np.zeros((4, 1)), np.ones(4))
    with pytest.raises(ValueError, match='0/1'):
        q.fit(np.zeros((4, 1)), np.array([0.0, 1.0, 2.0, 1.0]))


# ---------------------------------------------------------------------------
# weights, influence, SEs
# ---------------------------------------------------------------------------

def test_cluster_influence_exact_se_s_max_g_eff():
    u = np.array([1.0, -1.0, 2.0, 0.5])
    inf = an.cluster_influence(u, np.array(['a', 'a', 'b', 'c']))
    D = 0.0 ** 2 + 2.0 ** 2 + 0.5 ** 2
    shares = np.array([0.0, 4.0, 0.25]) / D
    assert inf['se_cluster'] == pytest.approx(np.sqrt(D)) and inf['se_iid'] == pytest.approx(np.sqrt(6.25))
    assert inf['s_max'] == pytest.approx(4.0 / D) and inf['g_eff'] == pytest.approx(1.0 / np.sum(shares ** 2))
    assert inf['n_clusters'] == 3 and inf['top_clusters'][0]['cluster'] == 'b'
    with pytest.raises(ValueError, match='D = 0'):
        an.cluster_influence(np.array([1.0, -1.0]), np.array(['a', 'a']))
    with pytest.raises(ValueError, match='nonfinite'):
        an.cluster_influence(np.array([np.nan, 1.0]), np.array(['a', 'b']))


def nuisance_draw(seed=4, n=60):
    rng = np.random.default_rng(seed)
    mu_a = rng.normal(size=n)
    return dict(y=mu_a + rng.normal(size=n), mu_a=mu_a, mu_d=mu_a + rng.normal(0.5, 0.2, n),
                r_a=np.exp(rng.normal(0.0, 0.5, n)), W=rng.uniform(0.2, 3.0, n))


@pytest.mark.parametrize('influence', [an.one_step_influence, an.hajek_influence])
def test_influence_is_the_exact_derivative_of_the_weighted_estimator(influence):
    """``u_j = W_j * d psi / d W_j``: the delta-method linearisation, including the Hajek ratio-mean term."""
    v = nuisance_draw()
    psi, u = influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], v['W'])
    assert u.sum() == pytest.approx(0.0, abs=1e-12)
    eps = 1e-6
    for j in range(0, 60, 7):
        up, down = v['W'].copy(), v['W'].copy()
        up[j] += eps
        down[j] -= eps
        slope = (influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], up)[0]
                 - influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], down)[0]) / (2 * eps)
        assert u[j] == pytest.approx(v['W'][j] * slope, rel=1e-5, abs=1e-10)


def test_hajek_matches_closed_form_and_differs_from_one_step_when_ratio_mean_drifts():
    v = nuisance_draw(seed=8)
    w = v['W'] / v['W'].sum()
    e = v['y'] - v['mu_a']
    expected = (np.sum(w * (v['mu_d'] - v['mu_a'])) + np.sum(w * v['r_a'] * e) / np.sum(w * v['r_a'])
                - np.sum(w * e))
    psi_h, _ = an.hajek_influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], v['W'])
    psi, _ = an.one_step_influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], v['W'])
    assert psi_h == pytest.approx(expected) and psi_h != pytest.approx(psi)
    ones = np.ones_like(v['r_a'])
    assert an.hajek_influence(v['y'], v['mu_a'], v['mu_d'], ones, v['W'])[0] == pytest.approx(
        an.one_step_influence(v['y'], v['mu_a'], v['mu_d'], ones, v['W'])[0])


def test_anchor_weights_equal_cluster_equal_unit_ht_and_survey():
    clusters = np.array(['m1'] * 4 + ['m2'] * 1 + ['m3'] * 5)
    eq_c = an.anchor_weights('equal_cluster', clusters)
    assert {c: eq_c[clusters == c].sum() for c in ('m1', 'm2', 'm3')} == pytest.approx(
        {'m1': 1.0, 'm2': 1.0, 'm3': 1.0})
    np.testing.assert_array_equal(an.anchor_weights('equal_unit', clusters), np.ones(10))
    ht = an.anchor_weights('equal_unit', clusters, population_counts={'m1': 40, 'm2': 1, 'm3': 500})
    assert {c: ht[clusters == c].sum() for c in ('m1', 'm2', 'm3')} == pytest.approx(
        {'m1': 40.0, 'm2': 1.0, 'm3': 500.0})
    sw = np.linspace(0.5, 2.0, 10)
    np.testing.assert_array_equal(an.anchor_weights('survey', clusters, survey_weights=sw), sw)
    with pytest.raises(ValueError):
        an.anchor_weights('survey', clusters)
    with pytest.raises(ValueError):
        an.anchor_weights('survey', clusters, survey_weights=-sw)
    with pytest.raises(ValueError, match='below its sampled count'):
        an.anchor_weights('equal_unit', clusters, population_counts={'m1': 3, 'm2': 1, 'm3': 5})
    with pytest.raises(ValueError, match='weight scheme'):
        an.anchor_weights('equal_birthweight', clusters)


def test_equal_cluster_versus_equal_unit_influence_concentration():
    """One dominant city: equal-unit weighting concentrates influence; equal-cluster spreads it."""
    rng = np.random.default_rng(1)
    clusters = np.array(['city'] * 200 + [f'm{i:02d}' for i in range(40) for _ in range(5)])
    phi = np.where(clusters == 'city', 1.0, 0.0) + rng.normal(0.0, 1.0, len(clusters))
    zeros, ones = np.zeros(len(phi)), np.ones(len(phi))
    moved = ones.astype(bool)
    fold_ids = np.arange(len(phi)) % 5
    res = {s: an.score_anchor_estimate('x', zeros, zeros, phi, ones, fold_ids, clusters, moved, 2.0,
                                       an.anchor_weights(s, clusters))
           for s in ('equal_cluster', 'equal_unit')}
    assert res['equal_unit']['s_max'] > 0.5 > res['equal_cluster']['s_max']
    assert res['equal_cluster']['g_eff'] > 3 * res['equal_unit']['g_eff']
    cluster_means = pd.Series(phi).groupby(clusters).mean()
    assert res['equal_cluster']['one_step']['psi'] == pytest.approx(cluster_means.mean())
    assert res['equal_unit']['one_step']['psi'] == pytest.approx(phi.mean())
    # Horvitz-Thompson expansion of a capped subsample recovers the full-population mean.
    ht = an.anchor_weights('equal_unit', clusters, population_counts={c: (2000 if c == 'city' else 5)
                                                                      for c in np.unique(clusters)})
    psi_ht = an.score_anchor_estimate('x', zeros, zeros, phi, ones, fold_ids, clusters, moved, 2.0, ht)
    expected = (2000 * cluster_means['city'] + 5 * cluster_means.drop('city').sum()) / (2000 + 5 * 40)
    assert psi_ht['one_step']['psi'] == pytest.approx(expected)


def test_score_guards_empty_folds_zero_weight_and_no_movers():
    v = nuisance_draw(n=20)
    clusters = np.repeat([f'c{i}' for i in range(10)], 2)
    moved = np.arange(20) % 2 == 0
    folds = np.arange(20) % 4
    args = (v['y'], v['mu_a'], v['mu_d'], v['r_a'])
    an.score_anchor_estimate('ok', *args, folds, clusters, moved, 2.0, np.ones(20))
    with pytest.raises(ValueError, match='never evaluated'):
        an.score_anchor_estimate('x', *args, np.where(folds == 3, -1, folds), clusters, moved, 2.0, np.ones(20))
    with pytest.raises(ValueError, match='sum to zero'):
        an.score_anchor_estimate('x', *args, folds, clusters, moved, 2.0, np.zeros(20))
    with pytest.raises(ValueError, match='zero target weight'):
        an.score_anchor_estimate('x', *args, folds, clusters, moved, 2.0, np.where(folds == 2, 0.0, 1.0))
    with pytest.raises(ValueError, match='delivered shift is zero'):
        an.score_anchor_estimate('x', *args, folds, clusters, np.zeros(20, bool), 2.0, np.ones(20))
    with pytest.raises(ValueError, match='density ratio'):
        an.score_anchor_estimate('x', *args[:3], -v['r_a'], folds, clusters, moved, 2.0, np.ones(20))


def test_outer_fold_without_targets_is_skipped_and_reported():
    d = cluster_design(seed=11)
    y = continuous_outcome(d)
    fold = ws.balanced_county_fold_ids(d['clusters'], d['strata'], d['a'], 5)
    support = an.compute_cluster_support(d['a'], d['strata'], d['clusters'], delta=2.0)
    target = support.supported & (fold != 4)
    data = an.AnchorInputs(ids=d['ids'], a=d['a'], y=y, X=d['X'], clusters=d['clusters'], strata=d['strata'],
                           target=target, moved=support.moved & target, outcome_kind='continuous')
    run = an.run_anchor(data, delta=2.0, q_learners=[linear_q('lin')], g_learners=[logistic_g('logit')],
                        combos=[('lin + logit', 'lin', 'logit')])
    assert run['folds'][4]['skipped'] and not any(f['skipped'] for f in run['folds'][:4])
    res = run['results']['equal_cluster']['lin + logit']
    assert set(res['per_fold']) == {'0', '1', '2', '3'}
    payload = an.anchor_payload('synthetic', 'y', 'units', run)
    assert 'skipped, no nuisance fitted' in an.render_anchor_markdown(payload)


# ---------------------------------------------------------------------------
# end-to-end synthetic runs and reporting invariants
# ---------------------------------------------------------------------------

INVARIANT_KEYS = ('plugin', 'correction', 'delivered_shift_mmhg', 'landing_ess', 'landing_ess_pct', 'max_r_a',
                  'mean_r_a', 's_max', 'g_eff', 'per_fold', 'one_step', 'hajek_one_step', 'r2', 'n_clusters')
FOLD_KEYS = ('n', 'weight_share', 'plugin', 'correction', 'one_step', 'delivered_shift_mmhg', 'max_r_a',
             'mean_r_a', 'landing_ess', 'landing_ess_pct', 'r2')


def check_reporting_invariants(run, data, delta, schemes, labels):
    target = np.asarray(data.target)
    moved_t = np.asarray(data.moved)[target]
    for scheme in schemes:
        w = an.anchor_weights(scheme, np.asarray(data.clusters)[target])
        w = w / w.sum()
        for label in labels:
            res = run['results'][scheme][label]
            for key in INVARIANT_KEYS:
                assert key in res, (scheme, label, key)
            assert res['delivered_shift_mmhg'] == pytest.approx(delta * float(np.sum(w * moved_t)))
            assert 0 < res['delivered_shift_mmhg'] < delta == res['nominal_shift_mmhg']
            psi = res['one_step']['psi']
            assert res['plugin'] + res['correction'] == pytest.approx(psi)
            assert res['one_step']['per_mmhg'] == pytest.approx(psi / res['delivered_shift_mmhg'])
            assert len(res['per_fold']) == 5
            assert sum(f['weight_share'] for f in res['per_fold'].values()) == pytest.approx(1.0)
            assert sum(f['weight_share'] * f['one_step'] for f in res['per_fold'].values()) == pytest.approx(psi)
            assert sum(f['weight_share'] * f['plugin'] for f in res['per_fold'].values()) == pytest.approx(
                res['plugin'])
            for f in res['per_fold'].values():
                for key in FOLD_KEYS:
                    assert key in f
                assert f['plugin'] + f['correction'] == pytest.approx(f['one_step'])
            assert 0 < res['s_max'] <= 1 and 1 <= res['g_eff'] <= res['n_clusters']
            assert res['max_r_a'] >= res['mean_r_a'] > 0 and 0 <= res['landing_ess_pct'] <= 100
            lo, hi = res['one_step']['ci95_cluster_per_mmhg']
            assert lo < res['one_step']['per_mmhg'] < hi


def test_end_to_end_continuous_anchor_with_payload_and_markdown(tmp_path):
    d = cluster_design(seed=11)
    y = continuous_outcome(d, effect=0.5)
    data, support = anchor_inputs(d, y, 'continuous')
    assert 0 < support.n_movers < support.n_targets < len(y)
    q = [linear_q('A6-Q'), ws.QLearner('B0_hist_gbm', ws.gbm_regressor_factory, seed=1103)]
    g = [logistic_g('A7-g', 2207), logistic_g('B0_gbm_origin', 3301)]
    run = an.run_anchor(data, delta=2.0, q_learners=q, g_learners=g,
                        null_learners=([linear_q('A6-Q', 4409)], [logistic_g('A7-g', 5501)]))
    labels = [c[0] for c in an.ANCHOR_COMBOS] + [an.ANCHOR_NULL_LABEL]
    check_reporting_invariants(run, data, 2.0, ('equal_cluster', 'equal_unit'), labels)
    primary = run['results']['equal_cluster'][an.ANCHOR_PRIMARY_LABEL]
    lo, hi = primary['one_step']['ci95_cluster_per_mmhg']
    assert lo < 0.5 < hi and primary['plugin'] / primary['delivered_shift_mmhg'] == pytest.approx(0.5, abs=0.08)
    assert primary['one_step']['psi'] != pytest.approx(
        run['results']['equal_unit'][an.ANCHOR_PRIMARY_LABEL]['one_step']['psi'])
    assert abs(run['results']['equal_cluster'][an.ANCHOR_NULL_LABEL]['one_step']['per_mmhg']) < 0.1
    assert run['n_targets'] == support.n_targets and run['n_movers'] == support.n_movers
    assert run['delivered_shift_mmhg'] == pytest.approx(support.delivered_shift)
    assert all(p['slope'] > 0 for p in run['platt']['A7-g'])

    payload = an.anchor_payload('synthetic_continuous', 'y', 'units', run, config=dict(delta=2.0),
                                audit=dict(n=len(y)))
    assert payload['notice'] == an.EXPLORATORY_NOTICE and payload['certifies_release_criteria'] is False
    assert 'fits' not in payload and 'null' not in payload
    assert json.loads(json.dumps(payload))['results']['equal_unit'][an.ANCHOR_PRIMARY_LABEL]['per_fold']
    json_path, md_path = an.write_anchor_outputs(tmp_path, 'synthetic', payload)
    assert json.loads(json_path.read_text())['notice'] == an.EXPLORATORY_NOTICE
    md = md_path.read_text()
    assert an.EXPLORATORY_NOTICE in md and 'delivered' in md and 'Per-fold decomposition' in md
    assert 's_max / G_eff' in md and 'Landing ESS' in md and an.ANCHOR_NULL_LABEL in md


def test_end_to_end_binary_anchor_uses_classifiers_and_survey_weights():
    d = cluster_design(seed=31, rows=10)
    y = binary_outcome(d, slope=0.5)
    data, support = anchor_inputs(d, y, 'binary')
    q = [logistic_q('A6-Q'), ws.QLearner('B0_hist_gbm', an.binary_q_factory(ws.gbm_classifier_factory), seed=1103)]
    g = [logistic_g('A7-g', 2207), logistic_g('B0_gbm_origin', 3301)]
    c_t = d['clusters'][support.supported]
    survey = np.random.default_rng(0).uniform(0.5, 2.0, int(support.supported.sum()))
    schemes = dict(equal_cluster=an.anchor_weights('equal_cluster', c_t),
                   equal_unit=an.anchor_weights('equal_unit', c_t),
                   survey=an.anchor_weights('survey', c_t, survey_weights=survey))
    run = an.run_anchor(data, delta=2.0, q_learners=q, g_learners=g, weight_schemes=schemes,
                        null_learners=([logistic_q('A6-Q', 4409)], [logistic_g('A7-g', 5501)]))
    labels = [c[0] for c in an.ANCHOR_COMBOS] + [an.ANCHOR_NULL_LABEL]
    check_reporting_invariants(run, data, 2.0, ('equal_cluster', 'equal_unit'), labels)
    for name in ('A6-Q', 'B0_hist_gbm'):
        mu = np.concatenate([run['fits'].mu_a[name], run['fits'].mu_d[name]])
        assert ((mu >= 0) & (mu <= 1)).all()
    for scheme in schemes:
        real = run['results'][scheme][an.ANCHOR_PRIMARY_LABEL]['one_step']
        null = run['results'][scheme][an.ANCHOR_NULL_LABEL]['one_step']
        assert real['per_mmhg'] > 2.0 * real['se_cluster_per_mmhg']    # higher exposure raises the risk
        assert abs(null['per_mmhg']) < 4.0 * null['se_cluster_per_mmhg']
    payload = an.anchor_payload('synthetic_binary', 'lbw', 'probability', run)
    assert payload['outcome_kind'] == 'binary'
    assert set(payload['results']) == {'equal_cluster', 'equal_unit', 'survey'}
    assert an.EXPLORATORY_NOTICE in an.render_anchor_markdown(payload)


def test_frame_to_inputs_round_trip():
    births, exposure = colombia_tables()
    frame = an.load_colombia_dane_frame(births, exposure, on_unmatched='drop')
    support = an.compute_cluster_support(frame.a, frame.strata, frame.clusters, delta=2.0)
    inputs = an.anchor_inputs_from_frame(frame, support)
    assert inputs.outcome_kind == 'binary' and len(inputs.ids) == len(frame.ids)
    np.testing.assert_array_equal(inputs.target, support.supported)
