"""Offline tests of the exploratory within-state runner: CPU only, no tabicl, no network, no checkpoint."""
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from oxyformer.design.support import SupportRecipe, connected_components
from oxyformer.explore import within_state as ws
from oxyformer.provenance import ContractError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def oracle_support(a, X, states, *, delta, caliper, bin_width=1.0, min_bin_tracts=2):
    """Independent reference built on design.support.connected_components."""
    scales = ws.covariate_scales(X)
    recipe = SupportRecipe(bin_width_mmhg=bin_width, min_bin_tracts=min_bin_tracts, raw_x_radius=caliper)
    missing = np.isnan(X)
    supported = np.zeros(len(a), dtype=bool)
    moved = np.zeros(len(a), dtype=bool)
    for q in range(len(a)):
        nbrs = [r for r in range(len(a)) if r != q and states[r] == states[q]
                and (missing[r] == missing[q]).all()
                and all(abs(X[r, j] - X[q, j]) <= caliper * scales[j] for j in np.flatnonzero(~missing[q]))]
        components = connected_components([a[r] for r in nbrs], nbrs, recipe)
        for lo, hi in components:
            if lo <= a[q] < hi:
                supported[q] = True
                moved[q] = a[q] + delta < hi
    return supported, moved


def line_frame(extra_a=(), extra_x=(), extra_states=()):
    """One state; base bins 0,1,1,2,2,3,(3),4 near X; queries at a = 1.0, 2.0, 1.999."""
    base_a = [0.5, 1.2, 1.5, 2.2, 2.5, 3.2, 3.5, 4.5]
    query_a = [1.0, 2.0, 1.999]
    a = np.asarray(base_a + query_a + list(extra_a), dtype=float)
    near = np.linspace(0.0, 1.0, len(base_a) + len(query_a))
    X = np.column_stack([near, near[::-1]])
    if len(extra_x):
        X = np.vstack([X, np.asarray(extra_x, dtype=float)])
    states = np.asarray(['01'] * (len(base_a) + len(query_a)) + list(extra_states))
    return a, X, states


def synthetic_inputs(seed=7, n_states=6, counties_per_state=12, tracts=15, effect=0.5, cross_state=3.0):
    """Strong cross-state confounding: state mean A drives state mean Y with slope ``cross_state``."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_states):
        a_state = 2.0 * s
        for c in range(counties_per_state):
            county = f'{s + 1:02d}{c:03d}'
            a_county = a_state + rng.normal(0.0, 0.5)
            for t in range(tracts):
                rows.append((f'{county}{t:06d}', county, f'{s + 1:02d}', a_state, a_county + rng.normal(0.0, 1.0)))
    ids = np.asarray([r[0] for r in rows], dtype=object)
    counties = np.asarray([r[1] for r in rows])
    states = np.asarray([r[2] for r in rows])
    a_state = np.asarray([r[3] for r in rows])
    a = np.asarray([r[4] for r in rows])
    X = rng.normal(size=(len(rows), 2))
    X[rng.random(len(rows)) < 0.05, 1] = np.nan
    y = cross_state * a_state + effect * (a - a_state) + X[:, 0] + np.nan_to_num(X[:, 1]) * 0.5 \
        + rng.normal(0.0, 1.0, len(rows))
    moved = rng.random(len(rows)) < 0.7
    return ws.WithinStateInputs(ids=ids, a=a, y=y, X=X, counties=counties, states=states,
                                target=np.ones(len(rows), dtype=bool), moved=moved)


def linear_q(name='lin', seed=0, cap=None):
    return ws.QLearner(name, lambda s: LinearRegression(), seed=seed, context_cap=cap)


def logistic_g(name='logit', seed=0, cap=None):
    return ws.GLearner(name, lambda s: LogisticRegression(C=1.0, max_iter=500), seed=seed, original_cap=cap)


# ---------------------------------------------------------------------------
# (a) support
# ---------------------------------------------------------------------------

def test_support_shift_or_stay_boundary_and_tails():
    a, X, states = line_frame()
    s = ws.compute_within_state_support(a, X, states, delta=2.0, caliper=1000.0)
    by_a = {round(v, 3): (bool(sup), bool(mv)) for v, sup, mv in zip(a, s.supported, s.moved)}
    assert by_a[0.5] == (False, False) and by_a[4.5] == (False, False)  # first and last bins are tails
    assert by_a[1.0] == (True, True)          # a + delta = 3.0 lands in occupied bin 3
    assert by_a[1.999] == (True, True)        # 3.999 still inside the component [1, 4)
    assert by_a[2.0] == (True, False)         # 4.0 == component end: whole shift must stay inside, so stay
    assert by_a[2.5] == (True, False)         # 4.5 beyond the last interior bin
    assert by_a[3.2] == (False, False)        # its own bin holds only one other tract
    assert s.n_movers == int(s.moved.sum()) and (s.moved <= s.supported).all()
    assert s.delivered_shift == pytest.approx(2.0 * s.n_movers / s.n_targets)
    assert s.delivered_shift < 2.0


def test_support_caliper_excludes_far_covariates():
    # Drop the near 3.5 replicate; bin 3 is then replicated only by a far-X row.
    a, X, states = line_frame(extra_a=[3.6], extra_x=[[100.0, 100.0]], extra_states=['01'])
    keep = a != 3.5
    a, X, states = a[keep], X[keep], states[keep]
    q = int(np.flatnonzero(a == 1.0)[0])
    tight = ws.compute_within_state_support(a, X, states, delta=2.0, caliper=3.0)
    loose = ws.compute_within_state_support(a, X, states, delta=2.0, caliper=1e6)
    assert tight.supported[q] and not tight.moved[q]   # far row is outside 2 IQR: bin 3 unreplicated
    assert loose.supported[q] and loose.moved[q]


def test_support_requires_exact_missingness_and_same_state():
    a, X, states = line_frame(extra_a=[3.6], extra_x=[[0.5, np.nan]], extra_states=['01'])
    keep = a != 3.5
    a, X, states = a[keep], X[keep], states[keep]
    q = int(np.flatnonzero(a == 1.0)[0])
    s = ws.compute_within_state_support(a, X, states, delta=2.0, caliper=1e6)
    assert s.supported[q] and not s.moved[q]           # different missingness pattern never a neighbour
    X_obs = X.copy()
    X_obs[-1, 1] = 0.5
    assert ws.compute_within_state_support(a, X_obs, states, delta=2.0, caliper=1e6).moved[q]
    other = states.copy()
    other[-1] = '02'
    assert not ws.compute_within_state_support(a, X_obs, other, delta=2.0, caliper=1e6).moved[q]


def test_rows_without_exposure_never_stand_in_for_a_tail():
    # Bin 0 replicated (0.5, 0.6, 0.7): a phantom bin below it would make bin 0 interior.
    a, X, states = line_frame(extra_a=[0.6, 0.7, np.nan], extra_x=[[0.4, 0.6]] * 3, extra_states=['01'] * 3)
    finite = np.isfinite(a)
    s = ws.compute_within_state_support(a, X, states, finite, delta=2.0, caliper=1e6)
    kept = ws.compute_within_state_support(a[finite], X[finite], states[finite], delta=2.0, caliper=1e6)
    np.testing.assert_array_equal(s.supported[finite], kept.supported)
    np.testing.assert_array_equal(s.moved[finite], kept.moved)
    assert not s.supported[a == 0.5].any() and not s.supported[~finite].any()
    a_phantom = np.where(finite, a, -1e9)                  # what a NaN -> INT_MIN bin cast would do
    assert ws.compute_within_state_support(a_phantom, X, states, finite, delta=2.0, caliper=1e6).supported[a == 0.5].all()
    with pytest.raises(ContractError, match='finite exposures'):
        ws.compute_within_state_support(a, X, states, delta=2.0, caliper=1e6)


def test_support_matches_design_support_oracle():
    rng = np.random.default_rng(3)
    n = 240
    states = np.asarray(['01', '02', '03'])[rng.integers(0, 3, n)]
    a = rng.normal(5.0, 1.6, n) + (states == '02') * 2.0
    X = rng.normal(size=(n, 3))
    X[rng.random(n) < 0.15, 2] = np.nan
    query = rng.random(n) < 0.8
    for delta, caliper in ((2.0, 2.0), (1.0, 1.0), (3.0, 3.0)):
        s = ws.compute_within_state_support(a, X, states, query, delta=delta, caliper=caliper)
        sup, mv = oracle_support(a, X, states, delta=delta, caliper=caliper)
        np.testing.assert_array_equal(s.supported, sup & query)
        np.testing.assert_array_equal(s.moved, mv & query)
        assert s.n_targets > 0 and 0 < s.n_movers < s.n_targets


# ---------------------------------------------------------------------------
# (b) balanced county folds
# ---------------------------------------------------------------------------

def test_balanced_county_folds_are_disjoint_and_elevation_balanced():
    rng = np.random.default_rng(11)
    counties, states, a = [], [], []
    for s, n_counties in enumerate((10, 15, 23, 7)):
        for c in range(n_counties):
            mean = 3.0 * s + rng.normal(0.0, 2.0)
            for _ in range(int(rng.integers(1, 6))):
                counties.append(f'{s + 1:02d}{c:03d}')
                states.append(f'{s + 1:02d}')
                a.append(mean + rng.normal(0.0, 0.3))
    counties, states, a = np.asarray(counties), np.asarray(states), np.asarray(a)
    splits = ws.make_balanced_county_folds(counties, states, a, n_splits=5)
    assert len(splits) == 5
    evaluated = np.concatenate([ev for _, ev in splits])
    assert sorted(evaluated.tolist()) == list(range(len(a)))
    for tr, ev in splits:
        assert set(counties[tr]).isdisjoint(counties[ev])
        assert len(tr) + len(ev) == len(a)
    fold_ids = ws.balanced_county_fold_ids(counties, states, a, 5)
    for s in np.unique(states):
        members = sorted(set(counties[states == s]), key=lambda c: (a[counties == c].mean(), c))
        folds = [int(fold_ids[counties == c][0]) for c in members]
        for start in range(0, len(members) - len(members) % 5, 5):
            assert sorted(folds[start:start + 5]) == [0, 1, 2, 3, 4]   # one county per exposure block
        for f in range(5):     # each fold spans the state's exposure gradient
            assert abs(sum(x == f for x in folds) - len(members) / 5) < 1


def test_county_spanning_states_is_refused():
    with pytest.raises(ContractError, match='spans more than one state'):
        ws.make_balanced_county_folds(np.asarray(['01001', '01001']), np.asarray(['01', '02']),
                                      np.asarray([1.0, 2.0]))


# ---------------------------------------------------------------------------
# (c) deterministic outcome-blind subsampling
# ---------------------------------------------------------------------------

def test_deterministic_subsample_is_stable_and_capped():
    ids = np.asarray([f'{i:011d}' for i in range(500)], dtype=object)
    first = ws.deterministic_subsample_idx(ids, 120, 'salt')
    np.testing.assert_array_equal(first, ws.deterministic_subsample_idx(ids, 120, 'salt'))
    assert len(first) == 120 and (np.diff(first) > 0).all()
    perm = np.random.default_rng(0).permutation(len(ids))
    assert set(ids[perm][ws.deterministic_subsample_idx(ids[perm], 120, 'salt')]) == set(ids[first])
    assert set(ws.deterministic_subsample_idx(ids, 120, 'other')) != set(first)
    np.testing.assert_array_equal(ws.deterministic_subsample_idx(ids[:50], 120, 'salt'), np.arange(50))
    for bad in (0, -1, 1.5):
        with pytest.raises(ContractError):
            ws.deterministic_subsample_idx(ids, bad, 'salt')


def test_learner_context_caps_bind_and_tabicl_caps_are_registered(tmp_path):
    assert ws.MAX_Q_CONTEXT_ROWS == 30_000 and 2 * ws.MAX_G_ORIGINAL_ROWS == 30_000
    q = {x.name: x for x in ws.default_q_learners(tmp_path / 'r.ckpt', 'cpu')}
    g = {x.name: x for x in ws.default_g_learners(tmp_path / 'c.ckpt', 'cpu')}
    assert q['A6-Q'].context_cap == ws.MAX_Q_CONTEXT_ROWS and g['A7-g'].original_cap == ws.MAX_G_ORIGINAL_ROWS

    data = synthetic_inputs(n_states=3, counties_per_state=8, tracts=10)
    seen = {'q': [], 'g': []}

    class RecQ(LinearRegression):
        def fit(self, X, y):
            seen['q'].append(len(X))
            return super().fit(X, y)

    class RecG(LogisticRegression):
        def fit(self, X, y):
            seen['g'].append(len(X))
            return super().fit(X, y)

    splits = ws.make_balanced_county_folds(data.counties, data.states, data.a, 5)
    ws.fit_cross_fitted_within_state(
        data, splits, delta=2.0, q_learners=[ws.QLearner('q', lambda s: RecQ(), 0, context_cap=100)],
        g_learners=[ws.GLearner('g', lambda s: RecG(max_iter=500), 0, original_cap=60)])
    assert len(seen['q']) == 5 and max(seen['q']) <= 100
    assert len(seen['g']) == 5 * 4 and max(seen['g']) <= 2 * 60


def test_module_imports_without_tabicl_or_wandb(monkeypatch, tmp_path):
    code = ('import sys, oxyformer.explore.within_state; '
            'assert "tabicl" not in sys.modules and "wandb" not in sys.modules, sorted(sys.modules)')
    env = dict(os.environ, PYTHONPATH=str(Path(ws.__file__).resolve().parents[2]))
    subprocess.run([sys.executable, '-c', code], check=True, env=env)
    make = ws.tabicl_regressor_factory(tmp_path / ws.TABICL_REGRESSOR_FILE, 'cpu')   # no import yet

    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(ws.metadata, 'version', missing)
    with pytest.raises(ImportError, match='tabicl is required'):
        make(0)


# ---------------------------------------------------------------------------
# (d) one-hot state FE and the within-state permuted-A null
# ---------------------------------------------------------------------------

def test_state_stats_and_onehot_encoding():
    a = np.asarray([1.0, 3.0, 10.0])
    states = np.asarray(['01', '01', '02'])
    stats = ws.fit_state_stats(a, states)
    assert stats.means == {'01': 2.0, '02': 10.0}
    assert stats.sds['01'] == pytest.approx(np.sqrt(2.0)) and stats.sds['02'] == 1.0
    F = ws.encode_features_onehot(np.asarray([4.0, 11.0, 5.0]), np.zeros((3, 2)), np.asarray(['01', '02', '09']),
                                  stats, ['01', '02'])
    assert F.shape == (3, 2 + 2 + 2)
    np.testing.assert_allclose(F[:, 0], [2.0, 1.0, 5.0 - stats.global_mean])   # centred, never raw A
    np.testing.assert_array_equal(F[:, -2:], [[1, 0], [0, 1], [0, 0]])
    raw = ws.encode_features_onehot(np.asarray([4.0]), np.zeros((1, 2)), np.asarray(['01']), stats, ['01', '02'],
                                    include_raw_a=True)
    assert raw.shape == (1, 7) and raw[0, 0] == 4.0


def test_permuted_null_is_near_zero_despite_cross_state_confounding():
    data = synthetic_inputs()
    a_null, moved_null = ws.permute_within_state(data.a, data.moved, data.states, seed=5)
    a_again, _ = ws.permute_within_state(data.a, data.moved, data.states, seed=5)
    np.testing.assert_array_equal(a_null, a_again)
    for s in np.unique(data.states):
        m = data.states == s
        assert sorted(a_null[m]) == sorted(data.a[m]) and moved_null[m].sum() == data.moved[m].sum()
    null = ws.WithinStateInputs(ids=data.ids, a=a_null, y=data.y, X=data.X, counties=data.counties,
                                states=data.states, target=data.target, moved=moved_null)
    delivered = 2.0 * moved_null.mean()

    def estimate(inputs):
        splits = ws.make_balanced_county_folds(inputs.counties, inputs.states, inputs.a, 5)
        fit = ws.fit_cross_fitted_within_state(inputs, splits, delta=2.0, q_learners=[linear_q()],
                                               g_learners=[logistic_g()])
        return ws.score_within_state_estimate('x', inputs.y, fit.mu_a['lin'], fit.mu_d['lin'], fit.r_a['logit'],
                                              fit.fold_ids, inputs.counties, {}, delivered)

    res = estimate(null)
    assert abs(res['plugin']) < 0.1 and abs(res['one_step']['psi']) < 0.1
    assert abs(res['one_step']['per_mmhg']) < 3 * res['one_step']['se_cluster_per_mmhg'] + 0.02
    # Power check: the same data without state FE shows the 3 y/mmHg cross-state slope.
    Z = np.column_stack([np.ones(len(a_null)), a_null, np.nan_to_num(data.X)])
    slope = np.linalg.lstsq(Z, data.y, rcond=None)[0][1]
    assert slope > 2.0
    # And the unpermuted data recover the 0.5 y/mmHg within-state effect.
    real = estimate(data)
    assert real['one_step']['per_mmhg'] == pytest.approx(0.5, abs=0.15)


# ---------------------------------------------------------------------------
# (e) county-grouped inner Platt calibration
# ---------------------------------------------------------------------------

def test_platt_origin_inner_folds_are_county_atomic_and_null_odds_near_one():
    rng = np.random.default_rng(21)
    n = 900
    county_code = rng.integers(0, 30, n).astype(float)
    counties = np.asarray([f'01{int(c):03d}' for c in county_code])
    F0 = np.column_stack([county_code, rng.normal(size=(n, 3))])
    F1 = np.column_stack([county_code, rng.normal(size=(n, 3))])      # same law: balanced null
    F_ev = np.column_stack([np.full(200, -1.0), rng.normal(size=(200, 3))])
    calls = []

    class Recorder:
        def __init__(self):
            self.model, self.fit_counties, self.pred_counties = LogisticRegression(max_iter=500), set(), set()
            calls.append(self)

        def fit(self, X, y):
            self.fit_counties = set(X[:, 0].tolist())
            self.model.fit(X[:, 1:], y)
            return self

        def predict_proba(self, X):
            self.pred_counties |= set(X[:, 0].tolist())
            return self.model.predict_proba(X[:, 1:])

    ratio, diag = ws.fit_platt_calibrated_origin(F0, F1, F_ev, counties, lambda s: Recorder(), seed=0)
    assert len(calls) == 4
    for inner in calls[:3]:
        assert inner.fit_counties and inner.pred_counties
        assert inner.fit_counties.isdisjoint(inner.pred_counties)
    assert set().union(*(c.pred_counties for c in calls[:3])) == set(county_code.tolist())
    assert calls[3].fit_counties == set(county_code.tolist())
    assert [f['shared_counties'] for f in diag['inner_folds']] == [0, 0, 0]
    assert np.isfinite(ratio).all() and (ratio > 0).all()
    assert ratio.mean() == pytest.approx(1.0, abs=0.1)
    assert diag['n_pairs'] == 2 * n and np.isfinite([diag['slope'], diag['intercept']]).all()

    ids = np.asarray([f'id{i}' for i in range(n)], dtype=object)
    _, capped = ws.fit_platt_calibrated_origin(F0, F1, F_ev, counties, lambda s: Recorder(), seed=0, ids=ids, cap=150)
    assert capped['n_originals'] == 150 and capped['n_pairs'] == 300


# ---------------------------------------------------------------------------
# (f) Conley inference
# ---------------------------------------------------------------------------

def test_conley_exact_concentration():
    inf = ws.compute_conley_inference(np.asarray([1.0, 1.0, 2.0, -1.0]), np.asarray(['A', 'A', 'B', 'C']), {})
    assert inf['se_iid'] == pytest.approx(np.sqrt(7.0) / 4.0)
    assert inf['se_cluster'] == pytest.approx(0.75)
    assert inf['s_max'] == pytest.approx(4.0 / 9.0)
    assert inf['g_eff'] == pytest.approx(81.0 / 33.0)
    assert inf['loc_coverage'] == 0.0 and inf['n_counties'] == 3
    for bw in (50, 100, 250):
        assert inf[f'se_spatial_{bw}km'] == pytest.approx(inf['se_cluster'])   # no centroid: cluster only
    assert inf['top5_counties'][0]['s_g'] == pytest.approx(4.0 / 9.0)


def test_conley_monotone_under_positive_spatial_correlation_and_missing_centroids():
    n_counties = 40
    counties = np.repeat([f'01{i:03d}' for i in range(n_counties)], 5)
    eif = np.repeat(1.0 + np.sin(np.arange(n_counties) / 6.0), 5)        # smooth, positive across space
    eif = eif - eif.mean() + 0.8 * np.repeat(np.cos(np.arange(n_counties) / 9.0), 5)
    locs = {f'01{i:03d}': (35.0, -100.0 + 0.3 * i) for i in range(n_counties)}   # ~27 km spacing
    inf = ws.compute_conley_inference(eif, counties, locs)
    assert inf['loc_coverage'] == 1.0
    assert inf['se_spatial_250km'] >= inf['se_spatial_100km'] >= inf['se_spatial_50km'] >= inf['se_cluster']
    assert inf['se_spatial_250km'] > inf['se_cluster']
    partial = {k: v for i, (k, v) in enumerate(locs.items()) if i % 4}
    inf_p = ws.compute_conley_inference(eif, counties, partial)
    assert inf_p['loc_coverage'] == pytest.approx(0.75)
    assert all(np.isfinite(inf_p[f'se_spatial_{bw}km']) and inf_p[f'se_spatial_{bw}km'] >= inf_p['se_cluster']
               for bw in (50, 100, 250))
    assert inf_p['se_cluster'] == pytest.approx(inf['se_cluster'])


# ---------------------------------------------------------------------------
# (g) reporting invariants and orchestration
# ---------------------------------------------------------------------------

def test_score_reports_every_permanent_invariant():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.normal(size=n)
    mu_a = y + rng.normal(0.0, 0.5, n)
    mu_d = mu_a + 0.1
    r_a = np.exp(rng.normal(0.0, 0.3, n))
    folds = np.arange(n) % 5
    counties = np.asarray([f'01{i % 30:03d}' for i in range(n)])
    flags = np.where(np.arange(n) % 2, 1, 3)
    res = ws.score_within_state_estimate('lbl', y, mu_a, mu_d, r_a, folds, counties, {}, 1.42, flags)
    for key in ('delivered_shift_mmhg', 'r2', 'wmse', 'mean_r_a', 'p99_r_a', 'max_r_a', 'landing_ess',
                'landing_ess_pct', 'plugin', 'correction', 'one_step', 'hajek_one_step', 'per_fold',
                'flag_stratified_descriptive'):
        assert key in res
    assert res['delivered_shift_mmhg'] == 1.42
    assert res['one_step']['per_mmhg'] == pytest.approx(res['one_step']['psi'] / 1.42)
    assert res['plugin'] == pytest.approx(0.1)
    assert res['one_step']['psi'] == pytest.approx(res['plugin'] + res['correction'])
    assert res['max_r_a'] == pytest.approx(r_a.max())
    for key in ('se_iid_per_mmhg', 'se_cluster_per_mmhg', 'se_sp50_per_mmhg', 'se_sp100_per_mmhg',
                'se_sp250_per_mmhg', 'ci95_sp100_per_mmhg', 'inference'):
        assert key in res['one_step']
    assert {'s_max', 'g_eff'} <= set(res['one_step']['inference'])
    assert set(res['per_fold']) == {'0', '1', '2', '3', '4'}
    for fold in res['per_fold'].values():
        assert {'n', 'plugin', 'correction', 'one_step', 'r2', 'wmse', 'mean_r_a', 'max_r_a', 'landing_ess',
                'landing_ess_pct'} <= set(fold)
        assert fold['one_step'] == pytest.approx(fold['plugin'] + fold['correction'])
    assert sum(f['n'] for f in res['per_fold'].values()) == n
    assert set(res['flag_stratified_descriptive']) == {'flag_1', 'flag_3'}
    assert res['flag_stratified_descriptive']['flag_1']['n'] == 150
    assert ws.score_within_state_estimate('x', y, mu_a, mu_d, r_a, folds, counties, {}, 1.42)[
        'flag_stratified_descriptive'] == {}
    with pytest.raises(ContractError, match='delivered shift'):
        ws.score_within_state_estimate('x', y, mu_a, mu_d, r_a, folds, counties, {}, 0.0)


def test_run_within_state_end_to_end_with_injected_learners():
    data = synthetic_inputs(n_states=4, counties_per_state=10, tracts=8)
    flags = np.where(np.arange(len(data.ids)) % 3, 1, 3)
    out = ws.run_within_state(
        data, delta=2.0, county_locs={}, q_learners=[linear_q('A6-Q'), linear_q('B0_hist_gbm')],
        g_learners=[logistic_g('A7-g'), logistic_g('B0_gbm_origin'), logistic_g('B0_logit_origin')],
        null_learners=([linear_q('A6-Q')], [logistic_g('A7-g')]), flags=flags)
    assert set(out['results']) == {label for label, _, _ in ws.DEFAULT_COMBOS} | {ws.NULL_LABEL}
    assert out['delivered_shift'] == pytest.approx(2.0 * data.moved.mean())
    assert [f['shared_counties'] for f in out['folds']] == [0] * 5
    primary = out['results'][ws.PRIMARY_LABEL]
    assert primary['delivered_shift_mmhg'] == out['delivered_shift']
    assert set(primary['flag_stratified_descriptive']) == {'flag_1', 'flag_3'}
    assert abs(out['results'][ws.NULL_LABEL]['one_step']['per_mmhg']) < 0.3
    payload = dict(ws.LABELS, config=dict(census=dict(targets=out['n_targets'])), delta=2.0,
                   n_targets=out['n_targets'], n_movers=out['n_movers'], delivered_shift_mmhg=out['delivered_shift'],
                   identification_disclosure='state FE only', results=out['results'], folds=out['folds'],
                   platt=out['platt'], null_platt=out['null'].platt)
    assert json.loads(json.dumps(payload))['certifies_release_criteria'] is False
    readout = ws._readout(payload, 1.0)
    assert 'certifies no release criterion' in readout and 'delivered shift' in readout
    assert all(f'`{label}`' in readout for label in out['results'])


def test_cli_defaults():
    args = ws.parse_args(['--prepared-dir', 'p', '--geography-json', 'g.json', '--out-dir', 'o'])
    assert (args.delta, args.caliper, args.wandb, args.n_splits) == (2.0, 2.0, False, 5)
    assert not args.allow_census_drift and not args.skip_null
    assert args.regressor_checkpoint.name == ws.TABICL_REGRESSOR_FILE
    assert ws.TABICL_REVISION in str(args.classifier_checkpoint)
