"""Offline synthetic tests for the exploratory A6-Q TabICLv2 Q adapter.

The TabICL backend is injected, so these tests need neither tabicl nor the
checkpoint. A real-checkpoint smoke runs separately on a CPU allocation.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_nested_cv import endpoint
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.contracts import SplitManifest
from oxyformer.design.splits import InnerSplit
from oxyformer.explore import q_refit as driver
from oxyformer.explore import tabicl_q as adapter
from oxyformer.models import tabicl_comparator
from oxyformer.provenance import ContractError, file_hash
from oxyformer.training.checkpoint import CheckpointRequest

RECIPE = dict(nested_cv=dict(ssl_epochs=1, frozen_epochs=1, synthetic=True))
WEIGHTS = b'synthetic regressor weights, not a model'


def five_fold(root, *, outcome=None):
    """The nested_cv fixture with five inner splits, unit weights and non-movers.

    In every outer fold one assignment-geography pair has a = 9.5, above the
    support's shift limit (10 - delta), so the policy leaves it unchanged.
    """
    p = endpoint(root)
    manifest = p.data.manifest
    names = [c.name for c in manifest.schema]
    rows = []
    for i, row in enumerate(p.data.rows):
        row = list(row)
        row[names.index('w')] = 1.
        if i < 30 and i % 6 in (4, 5):
            row[names.index('a')] = 9.5
        if outcome is not None and row[names.index('y')] is not None:
            row[names.index('y')] = outcome(row[names.index('id')], row[names.index('y')])
        rows.append(tuple(row))
    inners = []
    for fold in range(5):
        train = p.outer.training_ids(fold)
        graph = EntityGraph(original_ids=train,
                            links=tuple(x for x in p.entity_graph.links if x.observation_id in train))
        data = replace(manifest, original_ids=train, entity_graph_hash=graph.content_hash,
                       lineage=replace(manifest.lineage, unit_ids=train, parent_hashes=(manifest.content_hash,)))
        split = SplitManifest(spec=manifest.spec, level='inner', original_ids=train,
            fold_ids=tuple((i // 2) % 3 for i in range(len(train))), design_ids=(), excluded_ids=(),
            seed_ids=p.outer.seed_ids, entity_graph_hash=graph.content_hash,
            lineage=replace(manifest.lineage, unit_ids=train, parent_hashes=(data.content_hash,)))
        inners.append(InnerSplit(outer_fold=fold, data_manifest=data, entity_graph=graph, split=split,
                                 buffer_excluded_ids=()))
    return replace(p, data=replace(p.data, rows=tuple(rows)), inner=tuple(inners))


def bound(root, p):
    root.mkdir(parents=True)
    (root / 'endpoint.json').write_text(p.to_json())
    driver.atomic_record(root / 'recipe.json', RECIPE)
    driver.atomic_record(root / 'binding.json', dict(chain_dir=str(root / 'synthetic-chain'),
        frame_hash='2' * 64, archive_hash='3' * 64, endpoint_hash=p.content_hash,
        recipe_hash=driver.coverage.digest(RECIPE)))
    return root


class Estimator:
    def __init__(self, backend, kwargs):
        self.backend, self.kwargs = backend, kwargs

    def fit(self, X, y):
        log = self.backend.log
        log['fits'] += 1
        log['grad'].append(torch.is_grad_enabled())
        self.X, self.y = np.array(X), np.array(y)
        return self

    def predict(self, X, output_type='mean'):
        assert output_type == 'mean'
        self.backend.log['rows'] += len(X)
        self.backend.time += len(X)
        if self.backend.kind == 'dose':
            return X[:, 0].copy()
        weights = np.exp(-((X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1) / 10.)
        out = weights @ self.y / weights.sum(1) + 0.01 * X[:, 0]
        if self.backend.kind == 'transductive':
            out = out + X[:, 0].mean()
        return out


class Backend:
    """Records every call; a version or estimator call counts as a backend call."""

    def __init__(self, kind='kernel', version='2.2.0'):
        self.kind, self._version, self.time = kind, version, 0.
        self.log = dict(calls=0, fits=0, rows=0, grad=[], kwargs=[], copies=[])

    def version(self):
        self.log['calls'] += 1
        return self._version

    def estimator(self, **kwargs):
        self.log['calls'] += 1
        path = Path(kwargs['model_path'])
        self.log['kwargs'].append(kwargs)
        self.log['copies'].append(dict(path=str(path), sha256=file_hash(path), mode=path.stat().st_mode & 0o777))
        return Estimator(self, kwargs)


@pytest.fixture
def weights(tmp_path, monkeypatch):
    path = tmp_path / 'provisioned' / adapter.REGRESSOR['filename']
    path.parent.mkdir()
    path.write_bytes(WEIGHTS)
    identity = dict(adapter.REGRESSOR, sha256=file_hash(path))
    monkeypatch.setattr(adapter, 'REGRESSOR', identity)
    monkeypatch.setattr(adapter, 'REGISTERED_CHECKPOINTS', adapter.REGISTERED_CHECKPOINTS | {
        tuple(identity[f] for f in adapter.CHECKPOINT_IDENTITY_FIELDS)})
    monkeypatch.setattr(driver, 'checkout_identity', lambda: '4' * 40)
    monkeypatch.setattr(driver, 'resolve_device', lambda device: torch.device('cpu'))
    return path


def use(monkeypatch, backend):
    monkeypatch.setattr(adapter, 'BACKEND', backend)
    return backend


def fit(out, fold, weights, rows=None, slice_seconds=600.):
    return driver.fit(out, fold, 'A6-Q', 'cell-d', slice_seconds, 'none', device='cpu',
                      checkpoint=None if weights is None else str(weights), context_rows=rows)


def run_all(out, weights, rows=None):
    return [fit(out, fold, weights, rows)['parts'][0] for fold in range(5)]


def context(p, root, fold, rows, salt=adapter.SALT):
    return adapter.build_context(p, RECIPE, fold, rows, root / f'context-{fold}', salt=salt)


# T1: explicit, deterministic, outcome-blind sampling with refusal bounds.
def test_registered_regressor_identity():
    assert adapter.registered_identity() == adapter.REGRESSOR
    registered = [x for x in tabicl_comparator.REGISTERED_CHECKPOINTS if x[4] == adapter.REGRESSOR['filename']]
    assert registered == [tuple(adapter.REGRESSOR[f] for f in tabicl_comparator.CHECKPOINT_IDENTITY_FIELDS)]
    assert adapter.MAX_CONTEXT_ROWS == tabicl_comparator.TabICLComparator.max_context_rows
    assert adapter.MAX_COLUMNS == tabicl_comparator.TabICLComparator.max_features


def test_sampling_rule():
    ids = tuple(f'r{i}' for i in range(50))
    assert adapter.select_context(ids, 0, 50) == ids == adapter.select_context(ids, 0, 30000)
    picked = adapter.select_context(ids, 0, 7)
    assert len(picked) == 7 and set(picked) < set(ids)
    assert list(picked) == [x for x in ids if x in picked]
    expected = sorted(ids, key=lambda x: adapter.rank(adapter.SALT, 0, x))[:7]
    assert set(picked) == set(expected)
    assert adapter.select_context(ids, 0, 7) == picked
    assert adapter.select_context(ids, 1, 7) != picked
    assert adapter.select_context(ids, 0, 7, salt='other') != picked
    for rows in (1, 0, -3, 30001, True, 2.0, None):
        with pytest.raises(ContractError, match='context rows'):
            adapter.select_context(ids, 0, rows)
    with pytest.raises(ContractError, match='duplicate'):
        adapter.select_context(ids + ids[:1], 0, 7)


@pytest.mark.parametrize('rows', [2, 10, 24, 30000])
def test_context_rows_are_outer_training_rows(tmp_path, rows):
    p = five_fold(tmp_path / 'e')
    for fold in range(5):
        c = context(p, tmp_path, fold, rows)
        training = p.outer.training_ids(fold)
        assert c.ids == (training if rows >= len(training) else adapter.select_context(training, fold, rows))
        assert len(c.ids) == min(rows, len(training))
        assert set(c.ids).isdisjoint(p.outer.evaluation_ids(fold))
        assert c.targets == p.outer.evaluation_ids(fold)
        assert c.record['candidates'] == len(training) and c.record['selected'] == len(c.ids)
        assert c.record['columns'] == ['a', 'x'] and c.record['salt'] == adapter.SALT
        assert c.record['selected_ids_sha256'] == adapter.digest(list(c.ids))
        assert c.matrix.shape == (len(c.ids), 2)
        data = dict(zip(p.data.manifest.original_ids, p.data.rows))
        assert c.matrix.tolist() == [[data[i][2], data[i][3]] for i in c.ids]
        assert c.labels.tolist() == [data[i][1] for i in c.ids]


def test_selection_ignores_outcomes_treatments_and_covariates(tmp_path):
    p = five_fold(tmp_path / 'e')
    names = [c.name for c in p.data.manifest.schema]

    def perturbed(column, change):
        j = names.index(column)
        rows = tuple(tuple(change(v) if k == j and v is not None else v for k, v in enumerate(row))
                     for row in p.data.rows)
        return replace(p, data=replace(p.data, rows=rows))

    for fold in range(5):
        base = context(p, tmp_path / 'base', fold, 10)
        for column, change in (('y', lambda v: -7. * v + 3.), ('a', lambda v: v + 0.25), ('x', lambda v: v * 3.)):
            other = context(perturbed(column, change), tmp_path / column, fold, 10)
            assert other.ids == base.ids
            assert other.record['selected_ids_sha256'] == base.record['selected_ids_sha256']
            assert other.record['context_sha256'] != base.record['context_sha256']
        assert context(p, tmp_path / 'salt', fold, 10, salt='other').ids != base.ids
    assert context(p, tmp_path / 'f0', 0, 10).ids != context(p, tmp_path / 'f1', 1, 10).ids


def test_column_limit(tmp_path, monkeypatch):
    p = five_fold(tmp_path / 'e')
    wide = replace(p, feature_kinds=tuple((f'x{i}', 'numeric') for i in range(500)))
    with pytest.raises(ContractError, match='500 columns'):
        context(wide, tmp_path, 0, 10)
    monkeypatch.setattr(adapter, 'MAX_COLUMNS', 2)
    assert context(p, tmp_path, 0, 10).record['columns'] == ['a', 'x']
    monkeypatch.setattr(adapter, 'MAX_COLUMNS', 1)
    with pytest.raises(ContractError, match='columns'):
        context(p, tmp_path, 0, 10)
    with pytest.raises(ContractError, match='numeric'):
        context(replace(p, feature_kinds=(('x', 'categorical'),)), tmp_path, 0, 10)


def test_cli_context_rows_bounds(tmp_path, weights, monkeypatch):
    use(monkeypatch, Backend())
    out = bound(tmp_path / 'q', five_fold(tmp_path / 'e'))
    for rows in (1, 30001):
        with pytest.raises(ContractError, match='context rows'):
            fit(out, 0, weights, rows)
    assert not (out / 'candidate.json').exists()


# T2: no held-out outcome reaches its own fold's predictions.
def test_held_out_outcomes_do_not_leak(tmp_path, weights, monkeypatch):
    use(monkeypatch, Backend())
    p = five_fold(tmp_path / 'e')
    expected = run_all(bound(tmp_path / 'base', p), weights, rows=10)
    for fold in range(5):
        held = set(p.outer.evaluation_ids(fold))
        changed = five_fold(tmp_path / f'e{fold}', outcome=lambda oid, y: y + 1234. if oid in held else y)
        assert fit(bound(tmp_path / f'held-{fold}', changed), fold, weights, 10)['parts'][0] == expected[fold]
    # Positive control: a context row's outcome does change that fold's predictions.
    row = context(p, tmp_path, 0, 10).ids[0]
    changed = five_fold(tmp_path / 'ctx', outcome=lambda oid, y: y + 1234. if oid == row else y)
    assert fit(bound(tmp_path / 'ctx-q', changed), 0, weights, 10)['parts'][0]['mu_a'] != expected[0]['mu_a']


# T3: factual and policy queries share one context; composition is irrelevant.
def test_queries_follow_policy_apply(tmp_path, weights, monkeypatch):
    use(monkeypatch, Backend('dose'))
    p = five_fold(tmp_path / 'e')
    parts = run_all(bound(tmp_path / 'q', p), weights)
    a = dict(zip(p.data.manifest.original_ids, p.data.column('a')))
    cov = p.policy_covariates
    keys = dict(zip(cov.original_ids, zip(cov.geography_ids, cov.support_keys)))
    non_movers = 0
    for fold, part in enumerate(parts):
        ids = p.outer.evaluation_ids(fold)
        assert part['original_ids'] == list(ids)
        covariates = replace(cov, original_ids=ids, geography_ids=tuple(keys[i][0] for i in ids),
                             support_keys=tuple(keys[i][1] for i in ids))
        action = p.policy.apply(tuple(a[i] for i in ids), covariates)
        assert part['mu_a'] == list(action.a_mmhg)
        assert part['mu_d'] == list(action.d_mmhg)
        for x, y, moved in zip(part['mu_a'], part['mu_d'], action.moved):
            if not moved:
                non_movers += 1
                assert x == y
        assert any(action.moved)
    assert non_movers == 10


def test_one_context_and_query_composition(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend('transductive'))
    p = five_fold(tmp_path / 'e')
    c = context(p, tmp_path, 2, 10)
    estimator = adapter.fit_estimator(backend, str(weights), c, 'cpu')
    together = adapter.predict_targets(estimator, c, range(len(c.targets)), 'cpu')
    for i in range(len(c.targets)):
        alone = adapter.predict_targets(estimator, c, [i], 'cpu')
        assert alone == ([together[0][i]], [together[1][i]])
    assert adapter.predict_targets(estimator, c, [4, 0, 1], 'cpu') == (
        [together[0][i] for i in (4, 0, 1)], [together[1][i] for i in (4, 0, 1)])
    # A fold's slice fits one context, used by both query sets.
    fresh = use(monkeypatch, Backend())
    fit(bound(tmp_path / 'q', p), 2, weights, 10)
    assert fresh.log['fits'] == 1 and len(fresh.log['kwargs']) == 1


# T4: pinned, verified weights; no download; no training.
def test_weights_refuse_before_any_backend_call(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend())
    p = five_fold(tmp_path / 'e')
    out = bound(tmp_path / 'q', p)
    missing = tmp_path / 'absent' / adapter.REGRESSOR['filename']
    with pytest.raises(ContractError, match='missing'):
        fit(out, 0, missing)
    with pytest.raises(ContractError, match='missing'):
        adapter.verified_checkpoint(missing)
    renamed = tmp_path / 'other.ckpt'
    renamed.write_bytes(WEIGHTS)
    with pytest.raises(ContractError, match='filename'):
        fit(out, 0, renamed)
    with pytest.raises(ContractError, match='A6-Q requires'):
        fit(out, 0, None)
    wrong = tmp_path / 'wrong' / adapter.REGRESSOR['filename']
    wrong.parent.mkdir()
    wrong.write_bytes(WEIGHTS + b'!')
    with pytest.raises(ContractError, match='hash mismatch'):
        fit(out, 0, wrong)
    assert backend.log['calls'] == 0
    # A file replaced after admission still refuses before the backend.
    weights_copy = weights.read_bytes()
    weights.write_bytes(weights_copy[::-1])
    with pytest.raises(ContractError, match='hash mismatch'):
        fit(bound(tmp_path / 'q2', p), 0, weights)
    weights.write_bytes(weights_copy)
    assert backend.log['calls'] == 0
    for field, value in (('revision', '0' * 40), ('sha256', '1' * 64), ('filename', 'tabicl-classifier-v2-20260212.ckpt')):
        monkeypatch.setattr(adapter, 'REGRESSOR', dict(adapter.REGRESSOR, **{field: value}))
        with pytest.raises(ContractError, match='unregistered'):
            fit(bound(tmp_path / f'q-{field}', p), 0, weights)
        with pytest.raises(ContractError, match='unregistered'):
            adapter.verified_checkpoint(weights)
    assert backend.log['calls'] == 0 and backend.log['fits'] == 0 and backend.log['rows'] == 0


def test_backend_gets_private_verified_copy_and_never_trains(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend())

    def refuse(*args, **kwargs):
        pytest.fail('A6-Q constructed an optimizer')
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', refuse)
    out = bound(tmp_path / 'q', five_fold(tmp_path / 'e'))
    run_all(out, weights)
    assert backend.log['fits'] == 5 and backend.log['grad'] == [False] * 5
    for kwargs, copy in zip(backend.log['kwargs'], backend.log['copies']):
        assert kwargs['allow_auto_download'] is False and kwargs['device'] == 'cpu'
        assert kwargs['checkpoint_version'] == adapter.REGRESSOR['filename']
        assert kwargs['random_state'] == 1103
        assert {k: v for k, v in kwargs.items() if k in adapter.ESTIMATOR} == adapter.ESTIMATOR
        assert copy['path'] != str(weights) and Path(copy['path']).name == weights.name
        assert copy['sha256'] == adapter.REGRESSOR['sha256'] and copy['mode'] == 0o400
        assert not Path(copy['path']).exists()
    identity = json.loads((out / 'candidate.json').read_text())
    assert identity['tabicl']['checkpoint'] == adapter.REGRESSOR
    assert str(weights) not in (out / 'candidate.json').read_text()
    assert str(weights.parent) not in (out / 'fold-0' / 'progress.json').read_text()


# T5: batch-boundary slices resume to the uninterrupted part.
def test_resume_at_batch_boundary(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend())
    monkeypatch.setattr(adapter, 'clock', lambda: backend.time)
    monkeypatch.setattr(adapter, 'QUERY_BATCH', 2)
    p = five_fold(tmp_path / 'e')
    whole = fit(bound(tmp_path / 'whole', p), 0, weights, 10)
    assert whole['status'] == 'complete' and len(whole['batches']) == 3
    rows_whole = backend.log['rows']
    out = bound(tmp_path / 'q', p)
    first = fit(out, 0, weights, 10, slice_seconds=6.)
    assert first['status'] == 'incomplete' and len(first['batches']) == 1
    assert first['slices'][0]['stop_reason'] == 'deadline'
    assert first['parts'] == []
    journal = json.loads((out / 'fold-0' / 'progress.json').read_text())
    assert journal['batches'] == first['batches']
    rows_first = backend.log['rows'] - rows_whole
    second = fit(out, 0, weights, 10)
    assert second['status'] == 'complete' and second['parts'] == whole['parts']
    assert [b['slice'] for b in second['batches']] == [0, 1, 1]
    assert second['batches'][0] == first['batches'][0]
    assert backend.log['rows'] - rows_whole - rows_first == sum(
        len(b['mu_a']) + sum(1 for x, y in zip(b['mu_a'], b['mu_d']) if x != y) for b in second['batches'][1:])
    assert backend.log['rows'] - rows_whole == rows_whole
    assert set(second['phase_seconds']) == {'checkpoint', 'context', 'fit', 'prediction'}
    assert all({'context_seconds', 'fit_seconds', 'wall_seconds'} <= set(s) for s in second['slices'])
    assert all(b['wall_seconds'] > 0 for b in second['batches'])
    calls = backend.log['fits']
    assert fit(out, 0, weights, 10) == second and backend.log['fits'] == calls


def test_resume_refuses_changed_runtime_or_context(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend())
    monkeypatch.setattr(adapter, 'clock', lambda: backend.time)
    monkeypatch.setattr(adapter, 'QUERY_BATCH', 2)
    p = five_fold(tmp_path / 'e')
    out = bound(tmp_path / 'q', p)
    assert fit(out, 0, weights, 10, slice_seconds=6.)['status'] == 'incomplete'
    path = out / 'fold-0' / 'progress.json'
    saved = path.read_text()
    real = adapter.runtime_fingerprint
    monkeypatch.setattr(adapter, 'runtime_fingerprint', lambda b, d: dict(real(b, d), torch='0.0'))
    with pytest.raises(ContractError, match='runtime fingerprint'):
        fit(out, 0, weights, 10)
    monkeypatch.setattr(adapter, 'runtime_fingerprint', real)
    selection = adapter.select_context
    monkeypatch.setattr(adapter, 'select_context', lambda ids, fold, rows, salt: selection(ids, fold, rows, salt + '!'))
    with pytest.raises(ContractError, match='context digest'):
        fit(out, 0, weights, 10)
    monkeypatch.setattr(adapter, 'select_context', selection)
    journal = json.loads(saved)
    journal['batches'][0]['stop'] = 1
    path.write_text(json.dumps(journal))
    with pytest.raises(ContractError, match='batch plan'):
        fit(out, 0, weights, 10)
    path.write_text(saved)
    assert path.read_text() == saved
    assert fit(out, 0, weights, 10)['status'] == 'complete'


def test_main_returns_75_for_an_incomplete_slice(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend())
    monkeypatch.setattr(adapter, 'clock', lambda: backend.time)
    monkeypatch.setattr(adapter, 'QUERY_BATCH', 2)
    out = bound(tmp_path / 'q', five_fold(tmp_path / 'e'))
    args = ['fit', str(out), '--fold', '0', '--variant', 'A6-Q', '--study', 'cell-d', '--run-logging', 'none',
            '--checkpoint', str(weights), '--context-rows', '10', '--slice-seconds']
    assert driver.main(args + ['6']) == 75
    assert driver.main(args + ['600']) == 0


def test_signal_stops_at_batch_boundary(tmp_path, weights, monkeypatch):
    use(monkeypatch, Backend())
    monkeypatch.setattr(adapter, 'QUERY_BATCH', 2)
    out = bound(tmp_path / 'q', five_fold(tmp_path / 'e'))
    request = CheckpointRequest()
    request.request()
    monkeypatch.setattr(driver, 'CheckpointRequest', lambda: request)
    record = fit(out, 0, weights, 10)
    assert record['status'] == 'incomplete' and record['batches'] == []
    assert record['slices'][-1]['stop_reason'] == 'signal'


def test_deadline_rule():
    assert not adapter.out_of_time(0., 100., 0., 10.)
    assert not adapter.out_of_time(80., 100., 10., 10.)
    assert adapter.out_of_time(80.5, 100., 10., 10.)
    assert adapter.out_of_time(95., 100., 0., 10.)


# T6: complete five-fold run, rerun identity and exact q_parts.v1 output.
def test_five_folds_collect_and_rerun(tmp_path, weights, monkeypatch):
    backend = use(monkeypatch, Backend())
    configs, events = [], []

    class Logger:
        def log(self, event):
            events.append(event)

        def finish(self):
            pass

    def logger_for(mode, config, directory=None):
        configs.append(config)
        return Logger() if mode == 'wandb' else None
    monkeypatch.setattr(driver, 'logger_for', logger_for)
    p = five_fold(tmp_path / 'e')
    out = bound(tmp_path / 'q', p)
    args = ['--variant', 'A6-Q', '--study', 'cell-e', '--slice-seconds', '600', '--run-logging', 'wandb',
            '--checkpoint', str(weights)]
    assert all(driver.main(['fit', str(out), '--fold', str(k)] + args) == 0 for k in range(5))
    assert driver.main(['collect', str(out)]) == 0
    result = json.loads((out / 'q_parts.json').read_text())
    assert set(result) == {'schema', 'chain_dir', 'frame_hash', 'variant', 'study', 'code_commit', 'parts'}
    assert result['schema'] == 'oxyformer.q_parts.v1' and result['variant'] == 'A6-Q'
    assert result['study'] == driver.STUDIES['cell-e']
    assert all(set(part) == driver.PART_KEYS for part in result['parts'])
    assert [part['original_ids'] for part in result['parts']] == [list(p.outer.evaluation_ids(k)) for k in range(5)]
    assert backend.log['fits'] == 5
    for fold, config in enumerate(configs):
        assert config['fold'] == fold and config['variant'] == 'A6-Q' and config['study'] == driver.STUDIES['cell-e']
        assert config['chain_dir'] == str(out / 'synthetic-chain')
        assert config['tabicl']['context_rows'] == 30000 and config['tabicl']['salt'] == adapter.SALT
        assert config['tabicl']['checkpoint']['sha256'] == adapter.REGRESSOR['sha256']
    assert len(events) == 5 and all(e['complete'] for e in events)
    rerun = bound(tmp_path / 'rerun', p)
    assert run_all(rerun, weights) == result['parts']


def test_existing_variants_refuse_adapter_arguments(tmp_path, weights):
    out = bound(tmp_path / 'q', five_fold(tmp_path / 'e'))
    with pytest.raises(ContractError, match='apply only to A6-Q'):
        driver.fit(out, 0, 'A0', 'cell-d', 600., 'none', device='cpu', checkpoint=str(weights))
    with pytest.raises(ContractError, match='apply only to A6-Q'):
        driver.fit(out, 0, 'A4-Q', 'cell-d', 600., 'none', device='cpu', context_rows=10)
    assert not (out / 'candidate.json').exists()
    assert driver.VARIANTS == ('A0', 'A0-SW', 'A3-Q', 'A4-Q', 'A6-Q')
    assert driver.nested_cv.Q_VARIANTS == ('A0', 'A0-SW', 'A3-Q', 'A4-Q')


def test_real_backend_refuses_without_tabicl(monkeypatch):
    with pytest.raises(ContractError, match='pin'):
        adapter.runtime_fingerprint(Backend(version='2.2.1'), 'cpu')

    def absent(name):
        raise adapter.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(adapter.metadata, 'version', absent)
    with pytest.raises(ContractError, match='tabicl'):
        adapter.TabICLBackend().version()
    with pytest.raises(ContractError, match='tabicl'):
        adapter.TabICLBackend().estimator()
