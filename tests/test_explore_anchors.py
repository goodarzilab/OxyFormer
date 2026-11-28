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
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'SEXO')),
    dict(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'EST_CIVM', 'SEXO')),
])
def test_colombia_spec_rejects_occurrence_geography_post_outcome_and_unapproved(override):
    with pytest.raises(ValueError):
        an.validate_colombia_dane_spec(**override)


def test_colombia_spec_accepts_primary_and_sensitivity():
    spec = an.validate_colombia_dane_spec(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'EST_CIVM'))
    assert spec.residence_department_field == 'CODPTORE' and spec.residence_municipality_field == 'CODMUNRE'
    assert an.validate_colombia_dane_spec(covariates=('EDAD_MADRE', 'NIV_EDUM')).outcome == 'lbw'


def test_birth_covariate_lists_match_approvals_verbatim():
    """Review round 1: SEXO was admitted beyond birth_anchor_recipe; the approved lists are copied verbatim."""
    recipe = yaml.safe_load((REPO / 'configs' / 'approvals.yaml').read_text())['owner_decisions'][
        'birth_anchor_recipe']['covariates']
    assert list(an.COLOMBIA_PRIMARY_COVARIATES) == recipe['primary']['COL']
    assert list(an.COLOMBIA_SENSITIVITY_COVARIATES) == recipe['sensitivity']['COL'] == ['EST_CIVM']
    assert list(an.ECUADOR_PRIMARY_COVARIATES) == recipe['primary']['ECU']
    assert list(an.ECUADOR_SENSITIVITY_COVARIATES) == recipe['sensitivity']['ECU']
    with pytest.raises(ValueError, match="'SEXO'"):
        an.validate_colombia_dane_spec(covariates=('year', 'EDAD_MADRE', 'NIV_EDUM', 'SEXO'))


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


def ecuador_tables(province_column='province'):
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
    if province_column is not None:
        exposure[province_column] = ['Pichincha', 'Guayas', 'Guayas', 'Carchi', 'Manabi', 'Pichincha']
    return births, exposure


def test_ecuador_loader_places_exposure_at_maternal_residence_and_stops_on_ambiguity():
    births, exposure = ecuador_tables()
    repeated = pd.DataFrame({'shapeName': ['Bolivar'], 'oxygen_deficit_mmhg': [6.0], 'province': ['Manabi']})
    exposure = pd.concat([exposure, repeated], ignore_index=True)     # Manabi BOLIVAR listed twice
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(births, exposure)
    frame = an.load_ecuador_inec_frame(births, exposure, on_unmatched='drop')
    assert frame.audit['ambiguous_exposure_names'] == ['MANABI::BOLIVAR']
    assert frame.audit['matching'] == 'province_name:province'
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
    births, exposure = ecuador_tables()
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
        'EST_CIVM': ['1', '9', '2', '99', '1', '1'], 'SEXO': ['1', '2', '1', '2', '1', '1'],
        'T_GES': ['4'] * 6, 'IDPERTET': ['6'] * 6})
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
    sens = an.load_colombia_dane_frame(births, exposure,
                                       {'covariates': ('year', 'EDAD_MADRE', 'NIV_EDUM', 'EST_CIVM')},
                                       on_unmatched='drop')
    assert {'EST_CIVM_1', 'EST_CIVM_2', 'EST_CIVM_MISSING'} <= set(sens.feature_names)
    assert not any(n.startswith('SEXO') for n in sens.feature_names)
    for bad in ({'residence_municipality_field': 'COD_MUNIC'}, {'covariates': ('EDAD_MADRE', 'NIV_EDUM', 'T_GES')},
                {'covariates': ('year', 'EDAD_MADRE', 'NIV_EDUM', 'SEXO')}):
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
    explicit = an.AnchorInputs(y=binary_outcome(d), outcome_kind='continuous', **common)   # caller's choice
    assert explicit.outcome_kind == 'continuous'
    with pytest.raises(ValueError, match='coded 0/1'):
        an.AnchorInputs(y=continuous_outcome(d), outcome_kind='binary', **common)
    with pytest.raises(ValueError, match='outcome_kind'):
        an.AnchorInputs(y=continuous_outcome(d), outcome_kind='count', **common)
    q_zero = an.ProbabilityQ(ws.gbm_classifier_factory(0))
    q_zero.fit(np.zeros((4, 1)), np.zeros(4))
    np.testing.assert_array_equal(q_zero.predict(np.zeros((3, 1))), np.zeros(3))
    q_one = an.ProbabilityQ(LogisticRegression())
    q_one.fit(np.zeros((4, 1)), np.ones(4))
    np.testing.assert_array_equal(q_one.predict(np.zeros((3, 1))), np.ones(3))
    with pytest.raises(ValueError, match='0/1'):
        q_one.fit(np.zeros((4, 1)), np.array([0.0, 1.0, 2.0, 1.0]))


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
    zero_fold = an.score_anchor_estimate('ok', *args, folds, clusters, moved, 2.0, np.where(folds == 2, 0.0, 1.0))
    assert zero_fold['per_fold']['2']['weight_share'] == 0.0
    assert zero_fold['per_fold']['2']['plugin'] is None
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
    assert all(p['slope'] > 0 for scheme in run['platt'].values() for p in scheme['A7-g'])

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


# ---------------------------------------------------------------------------
# review round 1 regressions: integer codes, DANE missing codes
# ---------------------------------------------------------------------------

def test_integer_codes_normalise_integral_floats_and_reject_non_integers():
    out = an.integer_codes(pd.Series([3, ' 03 ', 3.0, '3.0', None, np.nan, '', 99, '0']), 'f')
    assert out.tolist()[:4] == ['3', '3', '3', '3'] and out.tolist()[-2:] == ['99', '0']
    assert out.iloc[4:7].isna().all()
    np.testing.assert_array_equal(an.integer_codes(pd.Series([5.0, np.nan]), 'f').isna(), [False, True])
    for bad in ([3.5], ['abc'], [np.inf], ['3.5']):
        with pytest.raises(ValueError, match='non-integer codes'):
            an.integer_codes(pd.Series(bad, dtype=object), 'f')


def test_colombia_float_peso_nac_with_missing_keeps_valid_band_three_birth():
    """Review counterexample: PESO_NAC [3, None] is read as float [3.0, NaN]; '3.0' must not drop the birth."""
    births = pd.DataFrame({'CODPTORE': ['05', '05'], 'CODMUNRE': ['001', '001'], 'PESO_NAC': [3, None],
                           'ANO': ['2023', '2023'], 'EDAD_MADRE': ['3', '3'], 'NIV_EDUM': ['4', '4']})
    assert births['PESO_NAC'].dtype.kind == 'f'
    _, exposure = colombia_tables()
    frame = an.load_colombia_dane_frame(births, exposure)
    assert list(frame.clusters) == ['05001'] and list(frame.y) == [1.0]
    assert frame.audit['n_valid_birthweight'] == 1 and frame.audit['excluded_unmatched_municipalities'] == {}


def test_colombia_float_codes_everywhere_match_string_codes():
    births, exposure = colombia_tables()
    keep = ['CODPTORE', 'CODMUNRE', 'PESO_NAC', 'ANO', 'EDAD_MADRE', 'NIV_EDUM', 'EST_CIVM']
    floats = births[keep].apply(lambda c: pd.to_numeric(c).astype(float))
    floats.loc[len(floats)] = [5.0, 1.0, 4.0, np.nan, np.nan, np.nan, np.nan]   # a sparse row forces float dtype
    spec = {'covariates': ('year', 'EDAD_MADRE', 'NIV_EDUM', 'EST_CIVM')}
    as_text = an.load_colombia_dane_frame(births, exposure, spec, on_unmatched='drop')
    as_float = an.load_colombia_dane_frame(floats, exposure.assign(cod_mpio=[5001.0, 11001.0, 5002.0]), spec,
                                           on_unmatched='drop')
    assert list(as_float.clusters) == list(as_text.clusters) + ['05001']
    np.testing.assert_array_equal(as_float.y, list(as_text.y) + [1.0])
    assert set(as_float.feature_names) == set(as_text.feature_names)
    last = dict(zip(as_float.feature_names, as_float.X[-1]))
    assert np.isnan(last['year_c']) and last['EDAD_MADRE_MISSING'] == last['NIV_EDUM_MISSING'] == 1.0
    assert last['EST_CIVM_MISSING'] == 1.0
    bad = births.copy()
    bad.loc[0, 'PESO_NAC'] = '3.5'
    with pytest.raises(ValueError, match='non-integer codes'):
        an.load_colombia_dane_frame(bad, exposure, on_unmatched='drop')
    blank = births.copy()
    blank.loc[1, 'CODMUNRE'] = None
    with pytest.raises(ValueError, match='<missing>'):
        an.load_colombia_dane_frame(blank, exposure)


def test_colombia_niv_edum_nine_is_profesional_not_missing():
    """Review counterexample: NIV_EDUM '09' is a valid level (profesional); only '99'/blank/NaN are missing."""
    births, exposure = colombia_tables()
    births['NIV_EDUM'] = ['09', '09', '04', '13', '4', '4']
    frame = an.load_colombia_dane_frame(births, exposure, on_unmatched='drop')
    names = list(frame.feature_names)
    assert 'NIV_EDUM_9' in names and 'NIV_EDUM_MISSING' not in names
    np.testing.assert_array_equal(frame.X[:, names.index('NIV_EDUM_9')], [1.0, 1.0, 0.0, 0.0])
    births = births.iloc[:4].copy()
    births['NIV_EDUM'] = ['99', None, '9', '']
    births['EDAD_MADRE'] = ['9', '99', '3', None]
    births['EST_CIVM'] = ['9', '99', '2', '1']
    frame = an.load_colombia_dane_frame(births, exposure, {'covariates': ('year', 'EDAD_MADRE', 'NIV_EDUM',
                                                                          'EST_CIVM')})
    col = dict(zip(frame.feature_names, frame.X.T))
    np.testing.assert_array_equal(col['NIV_EDUM_MISSING'], [1.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(col['NIV_EDUM_9'], [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(col['EDAD_MADRE_MISSING'], [1.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(col['EST_CIVM_MISSING'], [1.0, 1.0, 0.0, 0.0])
    assert an.COLOMBIA_MISSING_CODES == {'EDAD_MADRE': {'9', '99', ''}, 'NIV_EDUM': {'99', ''},
                                         'EST_CIVM': {'9', '99', ''}}


def test_peru_and_ecuador_float_codes_match_string_codes():
    text = peru_modules()
    floats = {k: v.copy() for k, v in text.items()}
    for col in ('HV001', 'HV024'):
        floats['RECH0'][col] = pd.to_numeric(floats['RECH0'][col]).astype(float)
    floats['RECH6']['HC0'] = pd.to_numeric(floats['RECH6']['HC0']).astype(float)
    a, b = an.load_peru_endes_frame({2023: text}), an.load_peru_endes_frame({2023: floats})
    assert list(b.ids) == list(a.ids) == ['2023:1:1', '2023:2:1', '2023:3:1', '2023:5:2']
    assert list(b.clusters) == list(a.clusters) and list(b.strata) == list(a.strata) == ['01', '01', '02', '02']
    births, exposure = ecuador_tables()
    births['niv_inst'] = [1.0, np.nan, 2.0, 3.0, 1.0, 2.0, 3.0]
    frame = an.load_ecuador_inec_frame(births, exposure, on_unmatched='drop')
    assert {'niv_inst_1', 'niv_inst_2', 'niv_inst_3', 'niv_inst_MISSING'} <= set(frame.feature_names)
    assert not any(n.endswith('.0') for n in frame.feature_names)


# ---------------------------------------------------------------------------
# review round 1 regression: per-delivered-mmHg ratio delta method
# ---------------------------------------------------------------------------

def test_per_mmhg_se_uses_ratio_delta_method():
    """Review counterexample: phi_i == s_i exactly, so beta = 1 in every resample and its SE is 0."""
    zeros, ones = np.zeros(4), np.ones(4)
    mu_d = np.array([2.0, 2.0, 0.0, 0.0])
    moved = np.array([True, True, False, False])
    clusters = np.array(['g1', 'g2', 'g3', 'g4'])
    res = an.score_anchor_estimate('ratio', zeros, zeros, mu_d, ones, np.array([0, 1, 0, 1]), clusters, moved,
                                   2.0, ones)
    assert res['one_step']['per_mmhg'] == pytest.approx(1.0, abs=1e-12)
    assert res['one_step']['se_cluster_per_mmhg'] == pytest.approx(0.0, abs=1e-12)
    assert res['hajek_one_step']['per_mmhg'] == pytest.approx(1.0, abs=1e-12)
    assert res['hajek_one_step']['se_cluster_per_mmhg'] == pytest.approx(0.0, abs=1e-12)
    assert res['one_step']['ci95_cluster_per_mmhg'] == pytest.approx([1.0, 1.0], abs=1e-12)
    assert res['s_max'] is None and res['g_eff'] is None              # concentration undefined when D = 0
    assert res['one_step']['se_fixed_shift_per_mmhg'] == pytest.approx(0.5)   # the old, wrong fixed-shift SE
    run = dict(outcome_kind='continuous', delta=2.0, n_targets=4, n_movers=2, n_clusters=4, n_strata=1,
               delivered_shift_mmhg=1.0, results={'equal_unit': {'ratio': res}}, folds=[])
    md = an.render_anchor_markdown(an.anchor_payload('degenerate', 'y', 'units', run))
    assert 'n/a / n/a' in md


@pytest.mark.parametrize('hajek', [False, True])
def test_per_mmhg_influence_is_the_exact_derivative_of_the_ratio(hajek):
    """``u_j = W_j * d beta / d W_j`` with ``beta = psi(W) / s_bar(W)`` and a varying delivered shift."""
    v = nuisance_draw(seed=12)
    moved = np.random.default_rng(3).random(60) < 0.6

    def beta(W):
        return an.per_mmhg_influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], moved, 2.0, W, hajek=hajek)[0]
    b, psi, s_bar, u, u_num = an.per_mmhg_influence(v['y'], v['mu_a'], v['mu_d'], v['r_a'], moved, 2.0, v['W'],
                                                    hajek=hajek)
    assert b == pytest.approx(psi / s_bar) and u.sum() == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(u, u_num / s_bar)                          # the shift's own variability matters
    eps = 1e-6
    for j in range(0, 60, 7):
        up, down = v['W'].copy(), v['W'].copy()
        up[j] += eps
        down[j] -= eps
        assert u[j] == pytest.approx(v['W'][j] * (beta(up) - beta(down)) / (2 * eps), rel=1e-5, abs=1e-10)
    w = v['W'] / v['W'].sum()
    res = an.score_anchor_estimate('x', v['y'], v['mu_a'], v['mu_d'], v['r_a'], np.arange(60) % 5,
                                   np.repeat(np.arange(20), 3), moved, 2.0, v['W'])
    key = 'hajek_one_step' if hajek else 'one_step'
    U = np.bincount(np.repeat(np.arange(20), 3), weights=u)
    assert res[key]['se_cluster_per_mmhg'] == pytest.approx(np.sqrt(np.sum(U ** 2)))
    assert res['delivered_shift_mmhg'] == pytest.approx(2.0 * np.sum(w * moved))


# ---------------------------------------------------------------------------
# review round 2 regressions: ENDES HHID joins, Ecuador province-qualified matching
# ---------------------------------------------------------------------------

def test_normalize_hhid_canonicalises_integral_numbers_and_keeps_dhs_text_ids():
    for value, expected in ((1, '1'), (1.0, '1'), ('1.0', '1'), (' 0001 ', '1'), ('000101 02', '000101 02'),
                            (' 12AB ', '12AB'), (np.int64(7), '7')):
        assert an._normalize_hhid(pd.Series([value], dtype=object), 'x').tolist() == [expected]
    same = an._normalize_hhid(pd.Series([1, 1.0, '1.0', '2', ' 000101 02 '], dtype=object), 'x')
    assert same.tolist() == ['1', '1', '1', '2', '000101 02']        # one household spelled three ways
    for bad, match in (([None], 'missing HHID'), ([''], 'missing HHID'), ([1.5], 'non-integral')):
        with pytest.raises(ValueError, match=match):
            an._normalize_hhid(pd.Series(bad, dtype=object), 'x')
    # Spellings are only canonicalised here; duplicate households are judged structurally by the loader.
    assert an._normalize_hhid(pd.Series(['0101', '101'], dtype=object), 'x').tolist() == ['101', '101']


def test_peru_float_hhid_in_one_module_still_joins_text_hhid_in_another():
    """Review counterexample: RECH0 HHID as float (1.0) and RECH6 HHID as text ('1') must still join."""
    text = an.load_peru_endes_frame({2023: peru_modules()})
    modules = peru_modules()
    r0_hhid = '\ufeffHHID'                                                   # the fixture's BOM header
    modules['RECH0'][r0_hhid] = pd.to_numeric(modules['RECH0'][r0_hhid]).astype(float)
    assert modules['RECH0'][r0_hhid].dtype.kind == 'f' and pd.api.types.is_string_dtype(modules['RECH6']['HHID'])
    mixed = an.load_peru_endes_frame({2023: modules})
    assert len(mixed.ids) == 4 and list(mixed.ids) == list(text.ids)
    np.testing.assert_allclose(mixed.a, text.a)
    np.testing.assert_allclose(mixed.survey_weights, text.survey_weights)
    modules['RECH23']['HHID'] = ['1.0', '2.0', '3.0', '4.0', '5.0']           # assets still merge
    assets = an.load_peru_endes_frame({2023: modules})
    np.testing.assert_array_equal(np.nan_to_num(assets.X), np.nan_to_num(text.X))
    dhs = peru_modules()
    for name, col in (('RECH0', r0_hhid), ('RECH23', 'HHID'), ('RECH6', 'HHID')):
        dhs[name][col] = dhs[name][col].str.strip().map(lambda h: f'  0001{h} 02 ')
    spaced = an.load_peru_endes_frame({2023: dhs})
    assert list(spaced.ids) == ['2023:00011 02:1', '2023:00012 02:1', '2023:00013 02:1', '2023:00015 02:2']


def test_ecuador_manabi_bolivar_never_matches_carchi_bolivar():
    """Review counterexample: a province-less exposure table must not place Manabi BOLIVAR at Carchi's deficit."""
    births, _ = ecuador_tables()
    manabi = births.iloc[[4]].reset_index(drop=True)
    assert (manabi.loc[0, 'prov_res'], manabi.loc[0, 'cant_res']) == ('Manabí', 'Bolívar')
    carchi_only = pd.DataFrame({'shapeName': ['Bolivar'], 'oxygen_deficit_mmhg': [30.0]})
    for on_unmatched in ('raise', 'drop'):
        with pytest.raises(ValueError, match='needs a province column'):
            an.load_ecuador_inec_frame(manabi, carchi_only, on_unmatched=on_unmatched)
    labelled = carchi_only.assign(ADM1=['Carchi'])
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(manabi, labelled)
    dropped = an.load_ecuador_inec_frame(pd.concat([manabi, births.iloc[[0]]], ignore_index=True),
                                         pd.concat([labelled, pd.DataFrame({'shapeName': ['Quito'], 'ADM1': [
                                             'Pichincha'], 'oxygen_deficit_mmhg': [42.0]})], ignore_index=True),
                                         on_unmatched='drop')
    assert list(dropped.a) == [42.0] and dropped.audit['excluded_unmatched_cantons'] == {'Bolívar': 1}
    births_all, exposure = ecuador_tables(province_column=None)
    with pytest.raises(ValueError, match='needs a province column'):
        an.load_ecuador_inec_frame(births_all, exposure, on_unmatched='drop')


@pytest.mark.parametrize('column', ['province', 'prov_res', 'prov_key', 'provincia', 'ADM1'])
def test_ecuador_accepts_each_province_column_name(column):
    assert column in an.ECUADOR_EXPOSURE_PROVINCE_COLUMNS
    births, exposure = ecuador_tables(province_column=column)
    frame = an.load_ecuador_inec_frame(births, exposure)
    assert frame.a[frame.clusters == 'MANABI::BOLIVAR'].tolist() == [5.0]
    assert frame.audit['matching'] == f'province_name:{column}'


def test_ecuador_dpa_canton_codes_place_births_without_names():
    births, _ = ecuador_tables()
    births = births.iloc[[0, 1, 4]].reset_index(drop=True)       # Quito, Quito, Manabi Bolivar
    births['prov_res'] = ['17', '17', '13']
    births['cant_res'] = ['1701', '1701', 1302.0]
    exposure = pd.DataFrame({'dpa_canton': ['1701', '0402', '1302'], 'oxygen_deficit_mmhg': [42.0, 30.0, 5.0]})
    frame = an.load_ecuador_inec_frame(births, exposure)
    assert frame.audit['matching'] == 'dpa_code'
    np.testing.assert_allclose(frame.a, [42.0, 42.0, 5.0])
    assert list(frame.clusters) == ['PICHINCHA::1701', 'PICHINCHA::1701', 'MANABI::1302']
    assert list(frame.strata) == ['PICHINCHA', 'PICHINCHA', 'MANABI']          # official DPA province names
    clash = births.assign(prov_res=['17', '17', '04'])           # code 1302 is Manabi, not Carchi
    with pytest.raises(ValueError, match='outside their prov_res province'):
        an.load_ecuador_inec_frame(clash, exposure)
    names, _ = ecuador_tables()                                  # names cannot use a DPA-only table: unmatched
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(names, exposure)


# ---------------------------------------------------------------------------
# review round 3 regressions
# ---------------------------------------------------------------------------

def test_ecuador_missing_province_or_canton_never_matches_as_nan():
    """Review counterexample: prov_res NaN + Bolivar must not match an exposure row province NaN + Bolivar."""
    births, _ = ecuador_tables()
    births = births.iloc[[0, 4]].reset_index(drop=True)          # Quito; Bolivar
    births.loc[1, 'prov_res'] = np.nan
    exposure = pd.DataFrame({'shapeName': ['Quito', 'Bolivar'], 'province': ['Pichincha', np.nan],
                             'oxygen_deficit_mmhg': [42.0, 30.0]})
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(births, exposure)
    frame = an.load_ecuador_inec_frame(births, exposure, on_unmatched='drop')
    assert list(frame.a) == [42.0] and 'NAN::BOLIVAR' not in set(frame.clusters.tolist())
    assert frame.audit['excluded_unmatched_cantons'] == {'Bolívar': 1}
    assert frame.audit['n_births_missing_residence'] == 1 and frame.audit['n_exposure_rows_missing_area'] == 1
    for blank_birth, blank_exposure in (('', ''), ('  ', None), ('nan', 'NaN'), (None, np.nan)):
        b, e = births.copy(), exposure.copy()
        b.loc[1, 'prov_res'], e.loc[1, 'province'] = blank_birth, blank_exposure
        with pytest.raises(ValueError, match='unmatched or ambiguous'):
            an.load_ecuador_inec_frame(b, e)
    no_canton = births.copy()
    no_canton.loc[1, ['prov_res', 'cant_res']] = ['Pichincha', np.nan]
    with pytest.raises(ValueError, match="'<missing>': 1"):
        an.load_ecuador_inec_frame(no_canton, exposure)


def test_peru_two_spellings_of_one_household_keep_both_children():
    """Review counterexample: RECH6 HHIDs '1' and '0001' (distinct HC0) are one household with two children."""
    modules = peru_modules()
    r6 = modules['RECH6']
    r6.loc[1, ['HHID', 'HC1']] = ['0001', '20']                  # second child of household 1, now eligible
    frame = an.load_peru_endes_frame({2023: modules})
    assert list(frame.ids) == ['2023:1:1', '2023:1:2', '2023:2:1', '2023:3:1', '2023:5:2']
    assert list(frame.clusters[:2]) == ['2023:10', '2023:10']
    np.testing.assert_allclose(frame.y[:2], [11.0, 12.0])
    np.testing.assert_allclose(frame.survey_weights[:2], [1.0, 1.0])


def test_peru_structural_duplicates_are_rejected_after_normalisation():
    dup_child = peru_modules()
    dup_child['RECH6'].loc[1, ['HHID', 'HC0', 'HC1']] = ['0001', '01', '20']   # same household, same child line
    with pytest.raises(ValueError, match='RECH6 children'):
        an.load_peru_endes_frame({2023: dup_child})
    dup_house = peru_modules()
    r0 = dup_house['RECH0']
    extra = r0.iloc[[0]].copy()
    extra.iloc[0, 0] = '0001'                                     # household 1 again, different altitude
    extra['HV040'] = '900'
    dup_house['RECH0'] = pd.concat([r0, extra], ignore_index=True)
    with pytest.raises(ValueError, match='RECH0 households'):
        an.load_peru_endes_frame({2023: dup_house})
    dup_assets = peru_modules()
    dup_assets['RECH23'] = pd.concat([dup_assets['RECH23'], pd.DataFrame(
        {'HHID': ['0002'], 'HV270': ['5'], 'HV271': ['999']})], ignore_index=True)
    with pytest.raises(ValueError, match='RECH23 households'):
        an.load_peru_endes_frame({2023: dup_assets})
    exact = peru_modules()
    exact['RECH0'] = pd.concat([exact['RECH0'], exact['RECH0'].iloc[[0]]], ignore_index=True)
    assert len(an.load_peru_endes_frame({2023: exact}).ids) == 4      # an exact repeat is harmless


def test_survey_weights_take_the_horvitz_thompson_expansion():
    """Review counterexample: A (N=100, 1 sampled, w=1, y=1), B (N=10, 1 sampled, w=1, y=0)."""
    clusters = np.array(['A', 'B'])
    w = an.anchor_weights('survey', clusters, survey_weights=np.ones(2), population_counts={'A': 100, 'B': 10})
    np.testing.assert_array_equal(w, [100.0, 10.0])
    y = np.array([1.0, 0.0])
    assert float(np.sum(w * y) / np.sum(w)) == pytest.approx(100 / 110)
    np.testing.assert_array_equal(an.anchor_weights('survey', clusters, survey_weights=[2.0, 3.0]), [2.0, 3.0])
    expanded = an.anchor_weights('survey', np.array(['A', 'A', 'B']), survey_weights=[2.0, 4.0, 3.0],
                                 population_counts={'A': 10, 'B': 3})
    np.testing.assert_allclose(expanded, [10.0, 20.0, 9.0])
    with pytest.raises(ValueError, match='below its sampled count'):
        an.anchor_weights('survey', np.array(['A', 'A']), survey_weights=[1.0, 1.0], population_counts={'A': 1})
    with pytest.raises(ValueError, match='missing clusters'):
        an.anchor_weights('survey', clusters, survey_weights=np.ones(2), population_counts={'A': 100})


def test_markdown_states_the_permutation_null_only_when_it_ran():
    d = cluster_design(seed=11, n_strata=3, clusters_per_stratum=24, rows=5)
    data, _ = anchor_inputs(d, continuous_outcome(d), 'continuous')
    kwargs = dict(delta=2.0, q_learners=[linear_q('lin')], g_learners=[logistic_g('logit')],
                  combos=[('lin + logit', 'lin', 'logit')])
    without = an.render_anchor_markdown(an.anchor_payload('s', 'y', 'u', an.run_anchor(data, **kwargs)))
    assert 'no permutation null was run' in without
    assert 'permuted-A null' not in without and an.ANCHOR_NULL_LABEL not in without
    with_null = an.render_anchor_markdown(an.anchor_payload('s', 'y', 'u', an.run_anchor(
        data, null_learners=([linear_q('lin', 7)], [logistic_g('logit', 9)]), **kwargs)))
    assert 'cluster-level within-stratum permuted-A null' in with_null and an.ANCHOR_NULL_LABEL in with_null
    assert f'seed `{an.NULL_SEED}`' in with_null and 'no permutation null was run' not in with_null


# ---------------------------------------------------------------------------
# fresh-cycle round 1 regressions
# ---------------------------------------------------------------------------

def test_peru_exact_child_repeat_with_respelled_hc0_is_deduplicated():
    """Review counterexample: the first RECH6 row repeated with HC0 '01' is the same child, not a conflict."""
    base = an.load_peru_endes_frame({2023: peru_modules()})
    modules = peru_modules()
    repeat = modules['RECH6'].iloc[[0]].copy()
    repeat['HC0'] = '01'
    modules['RECH6'] = pd.concat([modules['RECH6'], repeat], ignore_index=True)
    frame = an.load_peru_endes_frame({2023: modules})
    assert list(frame.ids) == list(base.ids) == ['2023:1:1', '2023:2:1', '2023:3:1', '2023:5:2']
    np.testing.assert_allclose(frame.y, base.y)
    respelled = peru_modules()
    repeat = respelled['RECH6'].iloc[[0]].copy()
    repeat['HHID'], repeat['HC0'] = '0001', '01'
    respelled['RECH6'] = pd.concat([respelled['RECH6'], repeat], ignore_index=True)
    assert list(an.load_peru_endes_frame({2023: respelled}).ids) == list(base.ids)
    conflict = peru_modules()
    repeat = conflict['RECH6'].iloc[[0]].copy()
    repeat['HC0'], repeat['HC53'] = '01', '115'                  # same child, different haemoglobin
    conflict['RECH6'] = pd.concat([conflict['RECH6'], repeat], ignore_index=True)
    with pytest.raises(ValueError, match='RECH6 children'):
        an.load_peru_endes_frame({2023: conflict})


def test_clean_category_normalises_mixed_object_codes_per_element():
    """Review counterexample: an object column mixing '1' and 1.0 is one category, not two."""
    out = an._clean_category(pd.Series(['1', 1.0], dtype=object), {}, 'niv_inst')
    assert out.tolist() == ['1', '1']
    parts, names = an._onehot(pd.DataFrame({'niv_inst': out}), ['niv_inst'])
    assert names == ['niv_inst_1'] and parts[0].shape == (2, 1)
    mixed = pd.Series([1, 1.0, '1.0', ' 01 ', 'Primaria', ' Superior ', None, np.nan, '', 'Ignorado', 2],
                      dtype=object)
    cleaned = an._clean_category(mixed, {'Ignorado': 'MISSING', '': 'MISSING'}, 'niv_inst')
    assert cleaned.tolist() == ['1', '1', '1', '1', 'Primaria', 'Superior', 'MISSING', 'MISSING', 'MISSING',
                                'MISSING', '2']
    with pytest.raises(ValueError, match='non-integral numeric category'):
        an._clean_category(pd.Series([1.5], dtype=object), {}, 'niv_inst')
    births, exposure = ecuador_tables()
    births['niv_inst'] = pd.Series(['1', 1.0, '2', 2.0, 3.0, '9', '01'], dtype=object)   # row 5 (450 g) drops
    frame = an.load_ecuador_inec_frame(births, exposure)
    assert sorted(n for n in frame.feature_names if n.startswith('niv_inst')) == ['niv_inst_1', 'niv_inst_2',
                                                                                 'niv_inst_3']


def test_markdown_null_claim_follows_null_estimates_present_in_results():
    d = cluster_design(seed=11, n_strata=3, clusters_per_stratum=24, rows=5)
    data, _ = anchor_inputs(d, continuous_outcome(d), 'continuous')
    run = an.run_anchor(data, delta=2.0, q_learners=[linear_q('lin')], g_learners=[logistic_g('logit')],
                        combos=[('lin + logit', 'lin', 'logit')],
                        null_learners=([linear_q('lin', 7)], [logistic_g('logit', 9)]))
    payload = an.anchor_payload('s', 'y', 'u', run)
    full = an.render_anchor_markdown(payload)
    assert 'permuted-A null' in full and 'under weights `equal_cluster`, `equal_unit`.' in full
    # A stored seed alone never implies a null: only the null estimate in results does.
    stripped = json.loads(json.dumps(payload))
    for models in stripped['results'].values():
        models.pop(an.ANCHOR_NULL_LABEL)
    assert stripped['null_seed'] == an.NULL_SEED
    none = an.render_anchor_markdown(stripped)
    assert 'no permutation null was run' in none and 'permuted-A null' not in none
    partial = json.loads(json.dumps(payload))
    partial['results']['equal_unit'].pop(an.ANCHOR_NULL_LABEL)
    some = an.render_anchor_markdown(partial)
    assert 'under weights `equal_cluster`.' in some
    assert 'No permutation null estimate is present under weights `equal_unit`.' in some
    unseeded = json.loads(json.dumps(payload))
    unseeded['null_seed'] = None
    assert 'seed `' not in an.render_anchor_markdown(unseeded)


# ---------------------------------------------------------------------------
# cycle-2 round 2 regressions: year-loop shadowing, numeric area codes
# ---------------------------------------------------------------------------

def test_peru_pooled_years_keep_child_covariates_in_every_year():
    """Review counterexample (a): both years keep HC27, HC61, HC62, HC64 and match their single-year loads."""
    frame = an.load_peru_endes_frame({2023: peru_modules(2023), 2024: peru_modules(2024)})
    names = list(frame.feature_names)
    assert {'HC27', 'HC61', 'HC62', 'HC64'} <= set(names) and frame.audit['absent_covariates'] == ['HV201'] + [
        c for c in an.PERU_ASSET_COVARIATES if c not in ('HV201', 'HV270', 'HV271')]
    for year in (2023, 2024):
        rows = np.char.startswith(frame.ids.astype(str), f'{year}:')
        assert rows.sum() == 4
        for col in ('HC27', 'HC61', 'HC64'):
            assert np.isfinite(frame.X[rows, names.index(col)]).all(), (year, col)
        single = an.load_peru_endes_frame({year: peru_modules(year)})
        cols = [names.index(c) for c in single.feature_names]
        np.testing.assert_array_equal(np.nan_to_num(frame.X[rows][:, cols]), np.nan_to_num(single.X))


def test_peru_child_covariate_absent_in_first_year_still_loads_for_later_years():
    """Review counterexample (b): HC27 missing from 2023 RECH6 stays a feature, NaN for 2023, values for 2024."""
    early = peru_modules(2023)
    early['RECH6'] = early['RECH6'].drop(columns=['HC27'])
    frame = an.load_peru_endes_frame({2023: early, 2024: peru_modules(2024)})
    names = list(frame.feature_names)
    assert 'HC27' in names and 'HC27' not in frame.audit['absent_covariates']
    hc27 = frame.X[:, names.index('HC27')]
    in_2023 = np.char.startswith(frame.ids.astype(str), '2023:')
    assert np.isnan(hc27[in_2023]).all()
    np.testing.assert_array_equal(hc27[~in_2023], [1.0, 1.0, 2.0, 1.0])     # children (1,1), (2,1), (3,1), (5,2)
    for col in ('HC61', 'HC64'):                                          # other child covariates in both years
        assert np.isfinite(frame.X[:, names.index(col)]).all()


def test_area_keys_canonicalise_integral_numeric_codes():
    keys = an._area_keys(pd.Series([17, 17.0, '17.0', ' 017 ', 'Pichincha', np.nan, 'nan', ''], dtype=object))
    assert keys.tolist()[:5] == ['17', '17', '17', '17', 'PICHINCHA'] and keys.iloc[5:].isna().all()
    with pytest.raises(ValueError, match='non-integral numeric area code'):
        an._area_keys(pd.Series([17.5], dtype=object))


def test_ecuador_numeric_province_codes_match_across_int_and_float():
    """Review counterexample: births prov_res 17 against exposure province 17.0 is one key, 'PICHINCHA::QUITO'."""
    births, _ = ecuador_tables()
    births = births.iloc[[0]].reset_index(drop=True)
    births['prov_res'] = [17]
    exposure = pd.DataFrame({'shapeName': ['Quito'], 'province': [17.0], 'oxygen_deficit_mmhg': [42.0]})
    frame = an.load_ecuador_inec_frame(births, exposure)
    assert len(frame.ids) == 1 and list(frame.clusters) == ['PICHINCHA::QUITO'] and list(frame.a) == [42.0]
    assert list(frame.strata) == ['PICHINCHA']
    for spelling in ('017', '17.0', ' 17 '):
        b = births.copy()
        b['prov_res'] = [spelling]
        assert list(an.load_ecuador_inec_frame(b, exposure.assign(province=[17])).clusters) == [
            'PICHINCHA::QUITO']


# ---------------------------------------------------------------------------
# cycle-2 round 3 regressions: per-row DPA / name matching, no column-mode guessing
# ---------------------------------------------------------------------------

def quito_births(prov_res, cant_res):
    births, _ = ecuador_tables()
    births = births.iloc[[0, 1]].reset_index(drop=True)          # two valid Quito births
    births['prov_res'] = pd.Series(prov_res, dtype=object)
    births['cant_res'] = pd.Series(cant_res, dtype=object)
    return births


def test_try_code_helpers_judge_each_value():
    for value in (1701, 1701.0, '1701', ' 1701 ', '1701.0', np.int64(1701)):
        assert an._try_dpa_code(value) == '1701'
    assert an._try_dpa_code('0101') == '0101' and an._try_dpa_code(101) == '0101'
    for value in ('Quito', None, np.nan, '', 10000, 0, -5):
        assert an._try_dpa_code(value) is None
    for value in (13, 13.0, '13', '013', ' 13 '):
        assert an._try_int_code(value) == '13'
    assert an._try_int_code('Pichincha') is None and an._try_int_code(np.nan) is None
    for bad in (1701.5, '1701.5', 13.5, np.inf):
        with pytest.raises(ValueError, match='non-integral'):
            an._try_int_code(bad)
        with pytest.raises(ValueError, match='non-integral'):
            an._try_dpa_code(bad)


def test_ecuador_numeric_province_conflict_raises_even_when_another_row_is_text():
    """Review counterexample 1: prov_res [13, 'Pichincha'] with cant_res [1701, 1701] is a province conflict."""
    births = quito_births([13, 'Pichincha'], [1701, 1701])
    exposure = pd.DataFrame({'dpa_canton': [1701], 'oxygen_deficit_mmhg': [42.0]})
    for on_unmatched in ('raise', 'drop'):
        with pytest.raises(ValueError, match='outside their prov_res province'):
            an.load_ecuador_inec_frame(births, exposure, on_unmatched=on_unmatched)
    agreeing = quito_births([17, 'Pichincha'], [1701, 1701])
    frame = an.load_ecuador_inec_frame(agreeing, exposure)
    assert list(frame.a) == [42.0, 42.0] and list(frame.clusters) == ['PICHINCHA::1701', 'PICHINCHA::1701']


def test_ecuador_mixed_code_and_name_cantons_match_the_same_canton():
    """Review counterexample 2: cant_res [1701, 'Quito'] both reach Quito (one by DPA code, one by name)."""
    births = quito_births(['Pichincha', 'Pichincha'], [1701, 'Quito'])
    exposure = pd.DataFrame({'dpa_canton': [1701], 'shapeName': ['Quito'], 'province': ['Pichincha'],
                             'oxygen_deficit_mmhg': [42.0]})
    frame = an.load_ecuador_inec_frame(births, exposure)
    assert len(frame.ids) == 2 and list(frame.a) == [42.0, 42.0]
    assert list(frame.clusters) == ['PICHINCHA::1701', 'PICHINCHA::1701'] and list(frame.strata) == ['PICHINCHA'] * 2
    assert frame.audit['matching'] == 'dpa_code+province_name:province'
    assert (frame.audit['n_matched_by_dpa'], frame.audit['n_matched_by_name']) == (1, 1)
    mixed_province = quito_births([17, 'Pichincha'], [1701, 'Quito'])         # numeric and text provinces too
    assert list(an.load_ecuador_inec_frame(mixed_province, exposure).a) == [42.0, 42.0]


def test_ecuador_textual_province_conflict_with_dpa_row_raises():
    """Review bonus: prov_res 'Manabi' with cant_res 1701 contradicts the exposure row's province Pichincha."""
    births = quito_births(['Manabi', 'Pichincha'], [1701, 1701])
    exposure = pd.DataFrame({'dpa_canton': [1701], 'province': ['Pichincha'], 'oxygen_deficit_mmhg': [42.0]})
    with pytest.raises(ValueError, match="DPA province 'PICHINCHA'"):
        an.load_ecuador_inec_frame(births, exposure)
    ok = an.load_ecuador_inec_frame(quito_births(['Pichincha', None], [1701, 1701]), exposure)
    assert list(ok.a) == [42.0, 42.0]                                        # missing prov_res: the code places it


def test_ecuador_exposure_lookups_are_validated_up_front():
    births = quito_births(['Pichincha', 'Pichincha'], [1701, 1701])
    with pytest.raises(ValueError, match='not a 4-digit DPA canton code'):
        an.load_ecuador_inec_frame(births, pd.DataFrame({'dpa_canton': ['Quito'], 'oxygen_deficit_mmhg': [42.0]}))
    with pytest.raises(ValueError, match='outside its own province'):
        an.load_ecuador_inec_frame(births, pd.DataFrame({'dpa_canton': [1701], 'province': [13],
                                                         'oxygen_deficit_mmhg': [42.0]}))
    repeated = pd.DataFrame({'dpa_canton': [1701, 1701.0], 'oxygen_deficit_mmhg': [42.0, 40.0]})
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(births, repeated)
    # The ambiguous DPA code also taints its row's name key: neither lookup may use either row.
    named_repeat = pd.concat([repeated.assign(shapeName=[np.nan, 'Quito'], province=[np.nan, 'Pichincha']),
                              pd.DataFrame({'dpa_canton': [1702], 'oxygen_deficit_mmhg': [41.0]})], ignore_index=True)
    frame = an.load_ecuador_inec_frame(pd.concat([births, quito_births(['Pichincha'] * 2, ['Quito'] * 2),
                                                  quito_births(['Pichincha'] * 2, [1702, 1702])], ignore_index=True),
                                       named_repeat, on_unmatched='drop')
    assert frame.audit['ambiguous_exposure_names'] == ['DPA:1701', 'PICHINCHA::QUITO'] and list(frame.a) == [41.0] * 2
    with pytest.raises(ValueError, match='non-integral'):
        an.load_ecuador_inec_frame(quito_births(['Pichincha'] * 2, [1701.5, 1701]),
                                   pd.DataFrame({'dpa_canton': [1701], 'oxygen_deficit_mmhg': [42.0]}))


def test_anchors_module_has_no_value_error_fallbacks():
    """Audit: no loader guesses a mode by catching ValueError over a whole column."""
    import inspect
    source = inspect.getsource(an)
    assert 'except ValueError' not in source and 'try:' not in source


# ---------------------------------------------------------------------------
# cycle-3 round 1 regression: textual prov_res verified against the official DPA province
# ---------------------------------------------------------------------------

def test_ecuador_textual_province_checked_against_dpa_province_without_exposure_names():
    """Review counterexample: prov_res 'Manabi' with cant_res 1701 (Pichincha) against a DPA-only table raises."""
    dpa_only = pd.DataFrame({'dpa_canton': [1701], 'oxygen_deficit_mmhg': [42.0]})
    with pytest.raises(ValueError, match="DPA province 'PICHINCHA'"):
        an.load_ecuador_inec_frame(quito_births(['Manabi', 'Manabi'], [1701, 1701]), dpa_only)
    frame = an.load_ecuador_inec_frame(quito_births(['Pichincha', 'Pichincha'], [1701, 1701]), dpa_only)
    assert list(frame.a) == [42.0, 42.0] and list(frame.clusters) == ['PICHINCHA::1701', 'PICHINCHA::1701']
    accented = an.load_ecuador_inec_frame(quito_births(['Manabí', 'manabi'], [1302, '1302']),
                                          pd.DataFrame({'dpa_canton': ['1302'], 'oxygen_deficit_mmhg': [5.0]}))
    assert list(accented.a) == [5.0, 5.0]
    # Unmatched rows are checked too, and a name that nothing can verify is an error, not a pass.
    with pytest.raises(ValueError, match="DPA province 'PICHINCHA'"):
        an.load_ecuador_inec_frame(quito_births(['Manabi', 'Pichincha'], [1799, 1701]), dpa_only,
                                   on_unmatched='drop')
    with pytest.raises(ValueError, match='unrecognized DPA province code'):
        an.load_ecuador_inec_frame(quito_births(['Pichincha', 'Pichincha'], [9901, 1701]),
                                   pd.DataFrame({'dpa_canton': [9901, 1701], 'oxygen_deficit_mmhg': [3.0, 42.0]}))
    assert an.ECUADOR_DPA_PROVINCES['13'] == an.normalise_area_name('Manabí')
    assert an.ECUADOR_DPA_PROVINCES['23'] == an.normalise_area_name('Santo Domingo de los Tsáchilas')
    assert len(an.ECUADOR_DPA_PROVINCES) == 25


def test_ecuador_exposure_province_name_must_agree_with_its_dpa_code():
    births = quito_births(['Pichincha', 'Pichincha'], [1701, 1701])
    with pytest.raises(ValueError, match="DPA province 'PICHINCHA'"):
        an.load_ecuador_inec_frame(births, pd.DataFrame({'dpa_canton': [1701], 'province': ['Manabi'],
                                                         'oxygen_deficit_mmhg': [42.0]}))
    frame = an.load_ecuador_inec_frame(births, pd.DataFrame({'dpa_canton': [1701], 'province': ['Pichincha'],
                                                             'oxygen_deficit_mmhg': [42.0]}))
    assert list(frame.clusters) == ['PICHINCHA::1701'] * 2


# ---------------------------------------------------------------------------
# cycle-3 round 2 regressions: one canonical Ecuador province for codes and names
# ---------------------------------------------------------------------------

def test_canonical_ecuador_province_unifies_codes_and_names():
    for value in (17, 17.0, '17', '017', ' 17 ', 'Pichincha', 'PICHINCHA', ' pichincha '):
        assert an._canonical_ecuador_province(value) == 'PICHINCHA'
    assert an._canonical_ecuador_province(1) == 'AZUAY' and an._canonical_ecuador_province('Cañar') == 'CANAR'
    assert an._canonical_ecuador_province(90) == 'ZONAS NO DELIMITADAS'
    for missing in (None, np.nan, '', '  ', 'nan'):
        assert an._canonical_ecuador_province(missing) is None
    assert an._canonical_ecuador_province('Provincia Sintetica') == 'PROVINCIA SINTETICA'   # kept, never official
    for bad in (99, '99', 25, 0):
        with pytest.raises(ValueError, match='unrecognized DPA province code'):
            an._canonical_ecuador_province(bad)


def test_ecuador_numeric_prov_res_matches_a_textual_exposure_province_by_name():
    """Review counterexample 1: prov_res 17 with cant_res 'Quito' matches province 'Pichincha', shapeName 'Quito'."""
    exposure = pd.DataFrame({'dpa_canton': [1701], 'province': ['Pichincha'], 'shapeName': ['Quito'],
                             'oxygen_deficit_mmhg': [42.0]})
    frame = an.load_ecuador_inec_frame(quito_births([17, 17], ['Quito', 'Quito']), exposure)
    assert len(frame.ids) == 2 and list(frame.a) == [42.0, 42.0]
    assert list(frame.clusters) == ['PICHINCHA::1701'] * 2 and list(frame.strata) == ['PICHINCHA'] * 2
    assert frame.audit['matching'] == 'province_name:province'
    names_only = exposure.drop(columns=['dpa_canton'])
    named = an.load_ecuador_inec_frame(quito_births([17, 'Pichincha'], ['Quito', 'Quito']), names_only)
    assert list(named.clusters) == ['PICHINCHA::QUITO'] * 2


def test_ecuador_dpa_rows_share_the_official_stratum_with_or_without_exposure_province():
    """Review counterexample 2: 1701 (province 'Pichincha') and 1702 (no province) are both stratum PICHINCHA."""
    exposure = pd.DataFrame({'dpa_canton': [1701, 1702], 'province': ['Pichincha', None],
                             'oxygen_deficit_mmhg': [42.0, 40.0]})
    frame = an.load_ecuador_inec_frame(quito_births([17, 17], [1701, 1702]), exposure)
    assert list(frame.strata) == ['PICHINCHA', 'PICHINCHA']
    assert list(frame.clusters) == ['PICHINCHA::1701', 'PICHINCHA::1702'] and list(frame.a) == [42.0, 40.0]
    numeric = an.load_ecuador_inec_frame(quito_births([17, 'Pichincha'], [1701, 1702]),
                                         exposure.assign(province=[17.0, np.nan]))
    assert list(numeric.strata) == ['PICHINCHA', 'PICHINCHA']


def test_ecuador_unrecognized_dpa_province_code_raises():
    """Review counterexample 3: prov_res 99 with cant_res 9901 against dpa_canton 9901 is not a DPA province."""
    with pytest.raises(ValueError, match="unrecognized DPA province code '99'"):
        an.load_ecuador_inec_frame(quito_births([99, 99], [9901, 9901]),
                                   pd.DataFrame({'dpa_canton': [9901], 'oxygen_deficit_mmhg': [3.0]}))
    with pytest.raises(ValueError, match="unrecognized DPA province code '99'"):     # birth-side prefix too
        an.load_ecuador_inec_frame(quito_births([None, None], [9901, 1701]),
                                   pd.DataFrame({'dpa_canton': [1701], 'oxygen_deficit_mmhg': [42.0]}),
                                   on_unmatched='drop')
    with pytest.raises(ValueError, match='unrecognized DPA province code 99'):        # numeric prov_res by name
        an.load_ecuador_inec_frame(quito_births([99, 99], ['Quito', 'Quito']),
                                   pd.DataFrame({'province': ['Pichincha'], 'shapeName': ['Quito'],
                                                 'oxygen_deficit_mmhg': [42.0]}))


# ---------------------------------------------------------------------------
# cycle-3 round 3 regression: an ambiguous DPA code taints its row's name key
# ---------------------------------------------------------------------------

def test_ecuador_conflicting_dpa_rows_cannot_leak_through_the_name_lookup():
    """Review counterexample: DPA 1701 at 42.0 (named Quito) and 40.0 (unnamed) must not match Quito by name."""
    exposure = pd.DataFrame([
        {'dpa_canton': 1701, 'province': 'Pichincha', 'shapeName': 'Quito', 'oxygen_deficit_mmhg': 42.0},
        {'dpa_canton': 1701, 'province': 'Pichincha', 'shapeName': None, 'oxygen_deficit_mmhg': 40.0}])
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(quito_births(['Pichincha', 'Pichincha'], ['Quito', 'Quito']), exposure)
    # Symmetric: a repeated name key taints the DPA codes of its rows, and the taint propagates transitively.
    chained = pd.DataFrame([
        {'dpa_canton': 1701, 'province': 'Pichincha', 'shapeName': 'Quito', 'oxygen_deficit_mmhg': 42.0},
        {'dpa_canton': 1702, 'province': 'Pichincha', 'shapeName': 'Quito', 'oxygen_deficit_mmhg': 40.0},
        {'dpa_canton': 1702, 'province': 'Pichincha', 'shapeName': 'Cayambe', 'oxygen_deficit_mmhg': 44.0},
        {'dpa_canton': 1703, 'province': 'Pichincha', 'shapeName': 'Cayambe', 'oxygen_deficit_mmhg': 45.0},
        {'dpa_canton': 1704, 'province': 'Pichincha', 'shapeName': 'Mejia', 'oxygen_deficit_mmhg': 39.0}])
    births = quito_births(['Pichincha'] * 2, [1701, 'Cayambe'])
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(births, chained)
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(quito_births(['Pichincha'] * 2, [1703, 1703]), chained)
    clean = an.load_ecuador_inec_frame(quito_births(['Pichincha'] * 2, [1704, 'Mejia']), chained)
    assert list(clean.a) == [39.0, 39.0]
    assert clean.audit['ambiguous_exposure_names'] == ['DPA:1701', 'DPA:1702', 'DPA:1703', 'PICHINCHA::CAYAMBE',
                                                       'PICHINCHA::QUITO']


# ---------------------------------------------------------------------------
# cycle-4 round 1 regression: canton aliases apply to exposure names too
# ---------------------------------------------------------------------------

def test_ecuador_canton_aliases_apply_to_births_and_exposure_alike():
    """Review counterexample: cant_res 'El Empalme' matches exposure shapeName 'El Empalme' (and 'Empalme')."""
    births, _ = ecuador_tables()
    births = births.iloc[[2]].reset_index(drop=True)
    assert (births.loc[0, 'prov_res'], births.loc[0, 'cant_res']) == ('Guayas', 'El Empalme')
    for shape in ('El Empalme', 'Empalme'):
        exposure = pd.DataFrame({'province': ['Guayas'], 'shapeName': [shape], 'oxygen_deficit_mmhg': [1.0]})
        frame = an.load_ecuador_inec_frame(births, exposure)
        assert list(frame.a) == [1.0] and list(frame.clusters) == ['GUAYAS::EMPALME']
    both = pd.DataFrame({'province': ['Guayas', 'Guayas'], 'shapeName': ['El Empalme', 'Empalme'],
                         'oxygen_deficit_mmhg': [1.0, 1.2]})                 # one canton listed twice
    with pytest.raises(ValueError, match='unmatched or ambiguous'):
        an.load_ecuador_inec_frame(births, both)


# ---------------------------------------------------------------------------
# cycle-4 round 2 regressions: scheme-specific target-weighted g; explicit continuous 0/1
# ---------------------------------------------------------------------------

class ZeroRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class NoWeightLogit:
    """A classifier whose fit takes no sample_weight (as TabICL's): forces the weighted resample path."""

    def __init__(self):
        self.model = LogisticRegression(C=100.0, max_iter=2000)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def two_level_design():
    """20 low clusters (a = 0) x 50 rows, half moved; 10 high clusters (a = 1) x 1 row, unmoved; y = a."""
    ids, clusters, a, moved = [], [], [], []
    for c in range(20):
        for t in range(50):
            ids.append(f'L{c:02d}R{t:02d}')
            clusters.append(f'L{c:02d}')
            a.append(0.0)
            moved.append(c % 2 == 0)
    for c in range(10):
        ids.append(f'H{c}')
        clusters.append(f'H{c}')
        a.append(1.0)
        moved.append(False)
    n = len(ids)
    a = np.asarray(a)
    return an.AnchorInputs(ids=np.asarray(ids, dtype=object), a=a, y=a.copy(), X=np.zeros((n, 1)),
                           clusters=np.asarray(clusters), strata=np.asarray(['S'] * n),
                           target=np.ones(n, dtype=bool), moved=np.asarray(moved), outcome_kind='continuous')


def test_density_ratio_is_fitted_under_each_schemes_target_law():
    """Review counterexample: equal-cluster r at high exposure is ~2 (not 51) and its effect ~1/mmHg (not 50).

    The weighted classifier recovers odds 2.0 / 51 exactly here; the cluster-grouped inner Platt step shrinks
    both towards 1 on this tiny design (one-row clusters pack into one GroupKFold fold), so the bounds test the
    order of magnitude and the exact values are pinned by the equal-size survey-weighted design below.
    """
    data = two_level_design()
    run = an.run_anchor(data, delta=1.0, q_learners=[ws.QLearner('zero', lambda s: ZeroRegressor(), 0)],
                        g_learners=[ws.GLearner('logit', lambda s: LogisticRegression(C=100.0, max_iter=2000), 0)],
                        combos=[('zero + logit', 'zero', 'logit')])
    high = data.a == 1.0
    r_cluster = run['r_a']['equal_cluster']['logit']
    r_unit = run['r_a']['equal_unit']['logit']
    assert 1.3 < np.mean(r_cluster[high]) < 3.0 and np.mean(r_unit[high]) > 15.0
    assert np.mean(r_unit[high]) / np.mean(r_cluster[high]) > 10.0
    for scheme in ('equal_cluster', 'equal_unit'):
        assert 0.4 < run['results'][scheme]['zero + logit']['one_step']['per_mmhg'] < 1.6
    assert {p['weighting'] for p in run['platt']['equal_cluster']['logit']} == {'sample_weight'}
    assert {p['weighting'] for p in run['platt']['equal_unit']['logit']} == {'uniform'}
    # The defect: reusing the equal-unit ratio under equal-cluster weights inflates the effect ~50-fold.
    w_cluster = an.anchor_weights('equal_cluster', data.clusters)
    zeros = np.zeros(len(data.a))
    shared = an.score_anchor_estimate('shared', data.y, zeros, zeros, r_unit, run['fits'].fold_ids, data.clusters,
                                      data.moved, 1.0, w_cluster)
    assert shared['one_step']['per_mmhg'] > 15.0


def test_density_ratio_recovers_each_target_law_exactly_with_equal_size_clusters():
    """The same law with one-row clusters and survey weights 50 / 1: r = 2 (uniform) and 51 (survey), 1/mmHg each."""
    n_low, n_high = 200, 100
    ids = np.asarray([f'L{c:03d}' for c in range(n_low)] + [f'H{c:03d}' for c in range(n_high)], dtype=object)
    a = np.r_[np.zeros(n_low), np.ones(n_high)]
    data = an.AnchorInputs(ids=ids, a=a, y=a.copy(), X=np.zeros((len(a), 1)), clusters=ids.astype(str),
                           strata=np.asarray(['S'] * len(a)), target=np.ones(len(a), bool),
                           moved=np.r_[np.arange(n_low) % 2 == 0, np.zeros(n_high, bool)], outcome_kind='continuous')
    survey = np.r_[np.full(n_low, 50.0), np.ones(n_high)]
    schemes = dict(unit=an.anchor_weights('equal_unit', ids.astype(str)),
                   survey=an.anchor_weights('survey', ids.astype(str), survey_weights=survey))
    run = an.run_anchor(data, delta=1.0, q_learners=[ws.QLearner('zero', lambda s: ZeroRegressor(), 0)],
                        g_learners=[ws.GLearner('logit', lambda s: LogisticRegression(C=100.0, max_iter=2000), 0)],
                        combos=[('z', 'zero', 'logit')], weight_schemes=schemes)
    assert np.mean(run['r_a']['unit']['logit'][a == 1]) == pytest.approx(2.0, rel=0.03)
    assert np.mean(run['r_a']['survey']['logit'][a == 1]) == pytest.approx(51.0, rel=0.03)
    assert np.mean(run['r_a']['survey']['logit'][a == 0]) == pytest.approx(0.5, rel=0.03)
    for scheme in schemes:
        assert run['results'][scheme]['z']['one_step']['per_mmhg'] == pytest.approx(1.0, abs=0.03)
    zeros = np.zeros(len(a))
    shared = an.score_anchor_estimate('shared', a, zeros, zeros, run['r_a']['unit']['logit'], run['fits'].fold_ids,
                                      ids.astype(str), data.moved, 1.0, schemes['survey'])
    assert shared['one_step']['per_mmhg'] < 0.1                          # the uniform ratio under survey weights


def test_weighted_resample_path_for_classifiers_without_sample_weight():
    data = two_level_design()
    kwargs = dict(delta=1.0, q_learners=[ws.QLearner('zero', lambda s: ZeroRegressor(), 0)],
                  g_learners=[ws.GLearner('nw', lambda s: NoWeightLogit(), 0)], combos=[('zero + nw', 'zero', 'nw')])
    run = an.run_anchor(data, **kwargs)
    assert {p['weighting'] for p in run['platt']['equal_cluster']['nw']} == {'weighted_resample'}
    assert 1.3 < np.mean(run['r_a']['equal_cluster']['nw'][data.a == 1.0]) < 3.0
    assert 0.4 < run['results']['equal_cluster']['zero + nw']['one_step']['per_mmhg'] < 1.6
    again = an.run_anchor(data, **kwargs)
    np.testing.assert_array_equal(again['r_a']['equal_cluster']['nw'], run['r_a']['equal_cluster']['nw'])


def test_schemes_with_identical_normalised_weights_share_one_g_fit():
    d = cluster_design(seed=11, n_strata=3, clusters_per_stratum=24, rows=5, vary_rows=False)
    data, _ = anchor_inputs(d, continuous_outcome(d), 'continuous')
    run = an.run_anchor(data, delta=2.0, q_learners=[linear_q('lin')], g_learners=[logistic_g('logit')],
                        combos=[('lin + logit', 'lin', 'logit')])
    assert run['r_a']['equal_cluster'] is run['r_a']['equal_unit']         # equal cluster sizes: one fit
    res = run['results']
    assert res['equal_cluster']['lin + logit']['one_step']['psi'] == pytest.approx(
        res['equal_unit']['lin + logit']['one_step']['psi'])


def test_explicit_continuous_outcome_may_take_zero_one_values():
    d = cluster_design(seed=2, n_strata=2, clusters_per_stratum=6, rows=3)
    common = dict(ids=d['ids'], a=d['a'], X=d['X'], clusters=d['clusters'], strata=d['strata'],
                  target=np.ones(len(d['a']), bool), moved=np.zeros(len(d['a']), bool))
    y01 = binary_outcome(d)
    assert an.AnchorInputs(y=y01, outcome_kind='continuous', **common).outcome_kind == 'continuous'
    with pytest.raises(ValueError, match='coded 0/1'):
        an.AnchorInputs(y=np.where(y01 == 1, 2.0, 0.0), outcome_kind='binary', **common)
    binary = an.AnchorInputs(y=y01, outcome_kind='binary', **common)
    with pytest.raises(ValueError, match='continuous Q learner cannot model a binary outcome'):
        an.route_q_learners(binary.outcome_kind, [linear_q('lin')])


# ---------------------------------------------------------------------------
# cycle-4 round 3 regressions: inner folds before resampling; weighted landing ESS
# ---------------------------------------------------------------------------

def test_weighted_origin_fit_splits_original_clusters_before_resampling():
    """Review counterexample: weights [1e20, 1, 1, 1] over 4 clusters must not collapse the inner cluster groups."""
    args = (np.zeros((4, 1)), np.ones((4, 1)), np.zeros((1, 1)), np.asarray(['a', 'b', 'c', 'd']),
            np.asarray([1e20, 1.0, 1.0, 1.0]))
    for factory, mode in ((lambda s: NoWeightLogit(), 'weighted_resample'),
                          (lambda s: LogisticRegression(max_iter=1000), 'sample_weight')):
        ratio, diag = an.fit_weighted_platt_calibrated_origin(*args, factory, seed=0, inner_splits=3)
        assert diag['weighting'] == mode and len(diag['inner_folds']) == 3
        assert diag['n_originals'] == 4 and np.isfinite(ratio).all() and (ratio > 0).all()
    again, _ = an.fit_weighted_platt_calibrated_origin(*args, lambda s: NoWeightLogit(), seed=0, inner_splits=3)
    np.testing.assert_array_equal(again, an.fit_weighted_platt_calibrated_origin(
        *args, lambda s: NoWeightLogit(), seed=0, inner_splits=3)[0])
    with pytest.raises(ValueError, match='fewer clusters than inner folds'):
        an.fit_weighted_platt_calibrated_origin(np.zeros((4, 1)), np.ones((4, 1)), np.zeros((1, 1)),
                                                np.asarray(['a', 'a', 'b', 'b']), np.asarray([5.0, 1, 1, 1]),
                                                lambda s: NoWeightLogit(), seed=0, inner_splits=3)


def test_landing_ess_uses_target_weights():
    """Review counterexample (w = [1000, 1, 1]): the weighted landing mass gives ESS 1002^2 / (1000^2 + 2)."""
    expected = 1002 ** 2 / (1000 ** 2 + 2)
    zeros, clusters = np.zeros(3), np.asarray(['c1', 'c2', 'c3'])
    moved, folds, w = np.asarray([True, False, False]), np.zeros(3, dtype=int), np.asarray([1000.0, 1.0, 1.0])

    def score(r_a, weights):
        return an.score_anchor_estimate('ess', zeros, zeros, np.asarray([1.0, 0.0, 0.0]), np.asarray(r_a), folds,
                                        clusters, moved, 1.0, weights)
    landed = score([3.0, 3.0, 3.0], w)                     # unweighted landing ESS here was 3
    assert landed['landing_ess'] == pytest.approx(expected)
    assert landed['per_fold']['0']['landing_ess'] == pytest.approx(expected)
    assert landed['landing_ess_pct'] == pytest.approx(100.0 * expected / 3)
    flat = score([1.0, 1.0, 1.0], w)                       # r = 1: nothing lands; the full-ratio ESS still sees w
    assert flat['landing_ess'] == 0.0
    assert flat['weighted_ratio_ess'] == pytest.approx(expected)
    assert flat['per_fold']['0']['weighted_ratio_ess'] == pytest.approx(expected)
    r = np.random.default_rng(0).lognormal(0.0, 0.7, 3)
    assert score(r, np.ones(3))['landing_ess'] == pytest.approx(ws._landing_ess(r)[0])   # uniform: unchanged


def test_score_anchor_estimate_allows_zero_weight_outer_fold():
    """Review counterexample: weights [1, 1, 0, 0] across 2 folds keep overall beta=0.5 and mark fold 1 n/a."""
    res = an.score_anchor_estimate(
        'x', np.array([0.0, 1.0, 0.0, 1.0]), np.zeros(4), np.ones(4), np.ones(4),
        np.array([0, 0, 1, 1]), np.array(['a', 'b', 'c', 'd']),
        np.ones(4, dtype=bool), 2.0, np.array([1.0, 1.0, 0.0, 0.0]),
    )
    assert res['one_step']['per_mmhg'] == pytest.approx(0.5)
    assert res['per_fold']['0']['weight_share'] == pytest.approx(1.0)
    assert res['per_fold']['1']['weight_share'] == 0.0
    assert res['per_fold']['1']['plugin'] is None
    assert res['per_fold']['1']['one_step'] is None
    assert res['per_fold']['1']['delivered_shift_mmhg'] is None
    payload = an.anchor_payload(
        anchor='test', outcome='y', outcome_units='u',
        run=dict(outcome_kind='continuous', delta=2.0, n_targets=4, n_movers=4, n_clusters=4, n_strata=1,
                 delivered_shift_mmhg=2.0, results={'survey': {'x': res}}, folds=[]),
    )
    md = an.render_anchor_markdown(payload)
    assert '| 1 | 2 | 0.000 | n/a | n/a | n/a | n/a | n/a |' in md


def test_capped_weighted_origin_excludes_zero_weight_rows_before_capping():
    """Review counterexample: zeroing the cap-selected rows must not leave zero total capped weight."""
    F0, F1 = np.zeros((36, 1)), np.ones((36, 1))
    clusters = np.repeat(np.arange(6), 6)
    ids = np.array([str(i) for i in range(36)])
    seed, cap, salt = 7, 6, 'a7-v2'
    keep = ws.deterministic_subsample_idx(ids, cap=cap, salt=f'{salt}-{seed}')
    w = np.ones(36, dtype=np.float64)
    w[keep] = 0.0
    ratio, diag = an.fit_weighted_platt_calibrated_origin(
        F0, F1, F0[:1], clusters, w, lambda s: LogisticRegression(), seed=seed, ids=ids, cap=cap, salt=salt
    )
    assert np.isfinite(ratio).all() and (ratio > 0).all()
    assert diag['n_originals'] == cap
    # Also verify non-uniform positive weights after excluding zero-weight rows
    w_nonunif = np.where(w > 0, np.linspace(1.0, 3.0, 36), 0.0)
    ratio2, diag2 = an.fit_weighted_platt_calibrated_origin(
        F0, F1, F0[:1], clusters, w_nonunif, lambda s: LogisticRegression(), seed=seed, ids=ids, cap=cap, salt=salt
    )
    assert np.isfinite(ratio2).all() and (ratio2 > 0).all()
    assert diag2['n_originals'] == cap
