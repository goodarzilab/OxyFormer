"""Synthetic CPU-only comparator evidence; no network or model downloads."""
from dataclasses import replace
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from oxyformer.contracts import CovariateView, EstimandSpec, SplitManifest
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.design.policies import PolicyPairs
from oxyformer.models import tabicl_comparator as icl
from oxyformer.models import tabpfn_comparator as pfn
from oxyformer.models.ablations import VARIANTS, build_variant
from oxyformer.models.county_context import CountyContext
from oxyformer.models.encoder import FeatureEncoder
from oxyformer.models.likelihoods import CountyOffsets
from oxyformer.models.tokens import FeatureSpec
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ArtifactLineage, ContractError


def digest(text):
    return sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def isolated_cpu(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("network is forbidden in comparator tests")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(47)
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def fold():
    rules = tuple(FeatureRule(name=x, role="predictor", endpoints=("synthetic",),
                              uses=("nuisance", "context"), approval_id="fixture") for x in ("x0", "x1"))
    rules += (FeatureRule(name="county", role="county", endpoints=("synthetic",),
                          uses=("county_routing",), approval_id="fixture"),)
    registry = FeatureRegistry(registry_id="fixture", rules=rules)
    spec = EstimandSpec(endpoint="synthetic", target_id="synthetic", outcome_scale="years",
                        policy_id="fixture", weight_id="unit", adjustment_schema_hash=registry.content_hash,
                        inference_unit="county", source_lineage_hash=digest("source"))
    ids = ("t0", "t1", "t2", "t3", "h0", "h1")
    lineage = ArtifactLineage(source_hashes=(digest("payload"),), unit_ids=ids + ("sealed",),
                              parent_hashes=(), split_hash=None, config_hash=digest("config"),
                              model_hash=None, environment=(("fixture", "cpu"),), seed=None, parameter_count=None)
    split = SplitManifest(spec=spec, level="outer", original_ids=ids, fold_ids=(1, 1, 1, 1, 0, 0),
                          design_ids=("sealed",), excluded_ids=(), seed_ids=(11,),
                          entity_graph_hash=digest("graph"), lineage=lineage)
    def view(ids, values):
        return CovariateView(spec=spec, registry=registry, original_ids=ids, columns=("x0", "x1"),
                             values=values, use="nuisance",
                             lineage=replace(lineage, unit_ids=ids, split_hash=split.content_hash))
    train = view(ids[:4], ((0., 1.), (1., 4.), (-1., 8.), (2., -1.)))
    held = view(ids[4:], ((10., 11.), (20., -2.)))
    return train, held, split


class MockEstimator:
    """Deliberately query-batch-sensitive to test adapter isolation."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)
        self.classes_ = np.array([1, 0])  # class order must be inspected

    def fit(self, x, y):
        self.x, self.y = x.copy(), y.copy()
        self.preprocess_mean = np.nanmean(x, axis=0)
        return self

    def predict(self, x, *, output_type="mean"):
        assert output_type == "mean"
        return np.nan_to_num(x).sum(1) + self.y.mean() + x[:, 0].mean()

    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-self.predict(x) / 30))
        return np.column_stack((p, 1 - p))


@pytest.fixture(params=[icl.TabICLComparator, pfn.TabPFNComparator], ids=["tabicl", "tabpfn"])
def backend(request, monkeypatch, tmp_path):
    cls = request.param
    env = tuple(sorted((name, icl.PACKAGE_PINS[cls.package] if name == cls.package else "fixture")
                       for name in ("python", *icl.RUNTIME_PACKAGES, cls.package)))
    monkeypatch.setattr(icl, "runtime_environment", lambda package: env)
    # Replace the production registry with exact synthetic identities, never a
    # public bypass of checkpoint validation or a real model download.
    identities = frozenset((cls.package, icl.PACKAGE_PINS[cls.package], "fixture/models", "a"*40,
                           f"{cls.package}-{role}-fixed.ckpt",
                           sha256(b"synthetic checkpoint bytes only").hexdigest())
                          for role in ("regressor", "classifier"))
    monkeypatch.setattr(icl, "REGISTERED_CHECKPOINTS", identities)
    model = SimpleNamespace(weights=[1])
    loader_calls = []
    def load(path, **kwargs):
        assert kwargs['download_if_not_exists'] is False
        assert kwargs['version'] == 'v2'
        loader_calls.append(kwargs)
        return [model], SimpleNamespace(borders=[0]), [SimpleNamespace()], SimpleNamespace()
    module = SimpleNamespace(TabICLRegressor=MockEstimator, TabICLClassifier=MockEstimator,
                             TabPFNRegressor=MockEstimator, TabPFNClassifier=MockEstimator,
                             ModelSpecs=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(icl, "import_module", lambda name: module)
    monkeypatch.setattr(pfn, "import_module", lambda name: SimpleNamespace(load_model_criterion_config=load)
                        if name == "tabpfn.model_loading" else module)
    def make(task="outcome", family="identity"):
        filename = f"{cls.package}-{'regressor' if family == 'identity' else 'classifier'}-fixed.ckpt"
        path = tmp_path / filename
        path.write_bytes(b"synthetic checkpoint bytes only")
        checkpoint = icl.Checkpoint(package=cls.package, package_version=icl.PACKAGE_PINS[cls.package],
            repository="fixture/models", revision="a" * 40, filename=filename,
            sha256=sha256(path.read_bytes()).hexdigest(), path=str(path), environment=env)
        return cls(checkpoint, task=task, family=family, seed=11)
    return make, loader_calls


def fitted(backend, fold, **kwargs):
    train, held, split = fold
    model = backend[0](**kwargs)
    y = [0., 1., 0., 1.] if model.family == "bernoulli" else [1., 3., 7., 2.]
    return model.fit_outcome(train, split, 0, [0., 1., 2., 3.], y,
                             sample_weight=np.ones(4), weight_semantics="unit")


def query(model, view, doses=None):
    a = torch.tensor([[[1.], [3.]], [[2.], [4.]]]) if doses is None else doses
    args = (a, view, torch.tensor(view.values), None, None)
    return model.mean(*args) if model.task == "outcome" else model.probability(*args)


def test_local_only_pinned_constructors(backend, fold):
    model = fitted(backend, fold)
    kwargs = model._estimator.kwargs
    assert kwargs['device'] == 'cpu' and kwargs['random_state'] == 11
    assert kwargs['n_estimators'] == 8
    if model.package == 'tabicl':
        assert kwargs['allow_auto_download'] is False
        assert kwargs['checkpoint_version'] == model.checkpoint.filename
        assert kwargs['model_path'] == model.checkpoint.path
    else:
        assert len(backend[1]) == 1
        other = fitted(backend, fold)
        assert kwargs['model_path'].model is not other._estimator.kwargs['model_path'].model


@pytest.mark.parametrize('revision', ['main', 'latest', 'v2', '', 'a' * 39])
def test_floating_checkpoints_rejected(backend, revision):
    with pytest.raises(ContractError, match='floating'):
        replace(backend[0]().checkpoint, revision=revision)


def test_missing_or_corrupt_checkpoint_blocks_before_import(backend, fold, monkeypatch):
    model = backend[0]()
    monkeypatch.setattr(model, '_make_estimator', lambda: pytest.fail('must fail before import'))
    path = Path(model.checkpoint.path)
    path.write_bytes(b'wrong bytes')
    with pytest.raises(ContractError, match='checkpoint hash mismatch'):
        model.fit_outcome(fold[0], fold[2], 0, np.arange(4), np.arange(4), sample_weight=np.ones(4), weight_semantics='unit')
    path.unlink()
    with pytest.raises(ContractError, match='local checkpoint missing'):
        model.checkpoint.verify()


def test_missing_optional_dependency_is_actionable(backend, fold, monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(icl, 'import_module', missing)
    monkeypatch.setattr(pfn, 'import_module', missing)
    with pytest.raises(ContractError, match='missing optional dependency'):
        fitted(backend, fold)
    monkeypatch.undo()
    monkeypatch.setattr(icl.metadata, 'version', lambda name: (_ for _ in ()).throw(metadata.PackageNotFoundError(name)))
    with pytest.raises(ContractError, match='missing optional dependency'):
        icl.runtime_environment('tabicl')


def test_primary_imports_do_not_require_comparators():
    code = '''
import sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('tabicl', 'tabpfn'):
            raise AssertionError('optional dependency imported: ' + fullname)
sys.meta_path.insert(0, Deny())
import oxyformer.contracts
import oxyformer.models.outcome
import oxyformer.models.origin
import oxyformer.models.ablations
import oxyformer.models.tabicl_comparator
import oxyformer.models.tabpfn_comparator
assert 'tabicl' not in sys.modules and 'tabpfn' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=45)


@pytest.mark.parametrize('weights,semantics', [([1, 2, 1, 1], 'unit'), ([1]*4, 'survey'),
    ([1]*4, 'target'), ([0]*4, 'unit'), ([2]*4, 'unit'), ([1, 1, float('nan'), 1], 'unit')])
def test_unsupported_sample_weights_are_not_discarded(backend, fold, weights, semantics):
    with pytest.raises(ContractError, match='unsupported.*weight'):
        backend[0]().fit_outcome(fold[0], fold[2], 0, np.arange(4), np.arange(4),
                                sample_weight=weights, weight_semantics=semantics)


def test_fold_context_is_exact_and_copied(backend, fold):
    train, held, split = fold
    a, y = np.arange(4.), np.arange(4.)
    model = backend[0]().fit_outcome(train, split, 0, a, y, sample_weight=np.ones(4), weight_semantics='unit')
    before = query(model, held)
    a[:] = 1000; y[:] = 1000
    torch.testing.assert_close(query(model, held), before)
    assert model.training_ids == train.original_ids
    assert np.array_equal(model._estimator.preprocess_mean, np.column_stack((np.arange(4.), train.values)).mean(0))
    with pytest.raises(ContractError, match='already fitted'):
        model.fit_outcome(train, split, 0, a, y, sample_weight=np.ones(4), weight_semantics='unit')
    with pytest.raises(ContractError, match='fold training IDs'):
        backend[0]().fit_outcome(held, split, 0, [1, 2], [1, 2], sample_weight=[1, 1], weight_semantics='unit')
    other = backend[0]().fit_outcome(held, split, 1, [1, 2], [20, 40], sample_weight=[1, 1], weight_semantics='unit')
    assert other.context_hash != model.context_hash
    assert other._estimator is not model._estimator
    torch.testing.assert_close(query(model, held), before)


@pytest.mark.parametrize('family', ['identity', 'bernoulli'])
def test_query_consistency(backend, fold, family):
    if family == 'bernoulli':
        fold = outcome_scale_fold(fold, 'risk_difference')
    model = fitted(backend, fold, family=family)
    view = fold[1]
    a = torch.tensor([[[1.], [3.]], [[2.], [4.]]])
    all_values = query(model, view, a)
    for j in range(2):
        torch.testing.assert_close(query(model, view, a[:, j:j+1])[:, 0], all_values[:, j])
    torch.testing.assert_close(query(model, view, a[:, [1, 0, 1]]), all_values[:, [1, 0, 1]])
    for i in range(2):
        single = replace(view, original_ids=(view.original_ids[i],), values=(view.values[i],),
                         lineage=replace(view.lineage, unit_ids=(view.original_ids[i],)))
        torch.testing.assert_close(query(model, single, a[i:i+1]), all_values[i:i+1])


def make_pairs(train):
    return PolicyPairs(policy_id=train.spec.policy_id, weight_id=train.spec.weight_id,
        original_ids=train.original_ids*2, geography_ids=train.original_ids*2,
        support_keys=('s',)*8, a_mmhg=(0., 1., 2., 3., 2., 3., 4., 5.),
        transformed=(False,)*4+(True,)*4, origin_weights=(1.,)*8)


def test_origin_pair_semantics_and_weights(backend, fold):
    train, held, split = fold
    pairs = make_pairs(train)
    model = backend[0](task='origin', family='bernoulli').fit_origin(train, split, 0, pairs, weight_semantics='unit')
    assert np.array_equal(model._estimator.x[:, 1:], np.tile(train.values, (2, 1)))
    assert np.array_equal(model._estimator.y, pairs.transformed)
    prob = query(model, held)
    a = torch.tensor([[[1.], [3.]], [[2.], [4.]]])
    torch.testing.assert_close(model.logits(a, held, torch.tensor(held.values), None, None).sigmoid(), prob)
    for weights in ((2.,)*8, (1.,)*7+(2.,)):
        with pytest.raises(ContractError, match='unsupported sample weights'):
            backend[0](task='origin', family='bernoulli').fit_origin(train, split, 0,
                replace(pairs, origin_weights=weights), weight_semantics='unit')


def test_runtime_mismatch_on_fit_and_predict(backend, fold, monkeypatch):
    model = fitted(backend, fold)
    monkeypatch.setattr(icl, 'runtime_environment', lambda package: (('python', 'wrong'),))
    with pytest.raises(ContractError, match='runtime fingerprint mismatch'):
        query(model, fold[1])
    with pytest.raises(ContractError, match='runtime fingerprint mismatch'):
        fitted(backend, fold)


@pytest.mark.parametrize('family', ['poisson', 'binomial', 'negative_binomial'])
def test_unsupported_outcome_families(backend, family):
    with pytest.raises(ContractError, match='unsupported outcome family'):
        backend[0](family=family)


def test_context_limits_and_query_compatibility(backend, fold):
    model = backend[0]()
    model.max_context_rows = 3
    with pytest.raises(ContractError, match='context limit'):
        model.fit_outcome(fold[0], fold[2], 0, np.arange(4), np.arange(4), sample_weight=[1]*4, weight_semantics='unit')
    model = fitted(backend, fold)
    held = fold[1]
    wrong = replace(held, spec=replace(held.spec, weight_id='survey'))
    with pytest.raises(ContractError):
        query(model, wrong)
    a = torch.zeros(2, 1, 1)
    with pytest.raises(ContractError, match='raw-X'):
        model.mean(a, held, torch.zeros(2, 1), None, None)
    with pytest.raises(ContractError, match='offsets'):
        model.mean(a, held, torch.tensor(held.values), None, torch.ones(2))


@pytest.fixture
def architecture(fold):
    train, _, split = fold
    encoder = FeatureEncoder(tuple(FeatureSpec(name=n, kind='numeric') for n in train.columns), dropout=0.)
    context = CountyContext(encoder, replace(train, use='context'), split, 0, ('c',)*4,
                            county_field='county', checkpoint_hash=digest('weights'), dropout=0.)
    design = TreatmentDesign(center=0., scale=2., knots=(-2., 0., 2., 4., 6., 8.), design_hash=digest('design'))
    return encoder, context, design


@pytest.mark.parametrize('variant_id', ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
def test_architectural_variants_raw_x_independence_and_query_consistency(architecture, fold, variant_id):
    encoder, context, design = architecture
    pair = build_variant(variant_id, encoder, treatment_design=design, raw_x_dim=2,
                         county_context=context, dropout=0.)
    assert pair.label == VARIANTS[variant_id].label
    first = {p.data_ptr() for p in pair.outcome.parameters()}
    assert not first & {p.data_ptr() for p in pair.correction.parameters()}
    for model in (pair.outcome, pair.correction):
        model.eval()
        assert model.check_parameter_cap() <= 1_000_000
        view = fold[0]
        tokens = model.encoder.tokenizer.prepare(view)
        raw = torch.tensor(view.values, requires_grad=True)
        ctx = torch.zeros(4, 4, 64) if variant_id == 'A2' else model.county_context(view.original_ids, ('c',)*4)
        a = torch.arange(8.).reshape(4, 2, 1).requires_grad_()
        args = (tokens, raw, ctx)
        def run(doses):
            return model(doses, *args) if variant_id == 'A5' and model is pair.correction else model(doses, *args, torch.zeros(4))
        values = run(a)
        torch.testing.assert_close(run(a[:, :1])[:, 0], values[:, 0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(run(a[:, [1, 0, 1]]), values[:, [1, 0, 1]], atol=1e-5, rtol=1e-5)
        grad_raw, grad_a = torch.autograd.grad(values[:, 0].sum(), (raw, a))
        assert (grad_raw.abs().sum(0) > 0).all()
        assert grad_a[:, 1].count_nonzero() == 0
        assert grad_a[:, 0].abs().sum() > 0
        with pytest.raises(ContractError, match='raw-X'):
            model._predict(a, tokens, raw[:, :1], ctx)


def test_no_ssl_reinitializes_and_no_pma_retains_offsets(architecture, fold):
    encoder, context, design = architecture
    with torch.no_grad():
        for p in context.parameters(): p.fill_(123.)
    pair = build_variant('A1', encoder, treatment_design=design, raw_x_dim=2, county_context=context, dropout=0.)
    assert all(not torch.all(p == 123.) for p in pair.outcome.parameters())
    assert pair.outcome.encoder is pair.outcome.county_context.encoder
    offsets = CountyOffsets(fold[2], 0, ('c',)*4, family='identity', exposure_assignment_level='tract')
    pair = build_variant('A2', encoder, treatment_design=design, raw_x_dim=2, county_context=context,
                         outcome_offsets=offsets, dropout=0.)
    assert pair.outcome.county_context is None and pair.outcome.group_offsets is not None


def test_signed_riesz_and_varying_coefficient_can_have_either_sign(architecture, fold):
    encoder, context, design = architecture
    for variant in ('A4', 'A5'):
        model = build_variant(variant, encoder, treatment_design=design, raw_x_dim=2,
                              county_context=context, dropout=0.).correction.eval()
        batch = model.encoder.tokenizer.prepare(fold[0])
        args = (torch.ones(4, 2, 1), batch, torch.tensor(fold[0].values), torch.zeros(4, 4, 64))
        with torch.no_grad():
            for parameter in model.readout.parameters(): parameter.zero_()
            for sign in (-2., 3.):
                model.readout[-1].bias.zero_()
                model.readout[-1].bias[0] = sign
                result = model(*args) if variant == 'A5' else model.logits(*args, torch.zeros(4))
                torch.testing.assert_close(result, torch.full((4, 2), sign))


def test_registered_foundation_variants(backend, architecture):
    encoder, context, design = architecture
    outcome = backend[0]()
    origin = backend[0](task='origin', family='bernoulli')
    names = ('A6', 'A7') if outcome.package == 'tabicl' else ('F0', 'F1')
    pairs = [build_variant(name, encoder, treatment_design=design, raw_x_dim=2,
                          county_context=context, foundation_outcome=outcome, foundation_origin=origin)
             for name in names]
    assert pairs[0].outcome is outcome and pairs[1].correction is origin
    assert pairs[0].label != pairs[1].label


def test_configuration_labels_pins_and_nonproduction_registry(architecture):
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root/'configs/models/ablations.yaml').read_text())
    assert set(config['variants']) == set(VARIANTS)
    for key, value in config['variants'].items():
        assert value['label'] == VARIANTS[key].label
    config = yaml.safe_load((root/'configs/models/foundations.yaml').read_text())
    assert config['offline_only'] and config['weight_semantics'] == 'unit_only'
    assert len(config['checkpoints']) == 4
    assert icl.REGISTERED_CHECKPOINTS == frozenset(
        tuple(item[field] for field in icl.CHECKPOINT_IDENTITY_FIELDS)
        for item in config['checkpoints'].values())
    for item in config['checkpoints'].values():
        assert item['package_version'] == icl.PACKAGE_PINS[item['package']]
        assert len(item['sha256']) == 64 and len(item['revision']) == 40
    for name in ('A8', 'B0', 'D0'):
        with pytest.raises(ContractError, match='not a production'):
            build_variant(name, architecture[0], treatment_design=architecture[2], raw_x_dim=2)


def test_numeric_raw_x_accepts_float32_rounding_and_missingness(backend, fold):
    model = fitted(backend, fold)
    held = replace(fold[1], values=((0.1, None), (0.2, 0.3)))
    raw = torch.tensor([[0.1, float("nan")], [0.2, 0.3]], dtype=torch.float32)
    result = model.mean(torch.ones(2, 1, 1), held, raw, None, None)
    assert torch.isfinite(result).all()


def test_checkpoint_task_and_package_pins(backend):
    model = backend[0]()
    with pytest.raises(ContractError, match="package version"):
        replace(model.checkpoint, package_version="latest")
    with pytest.raises(ContractError, match="task/family"):
        type(model)(model.checkpoint, task="origin", family="bernoulli", seed=11)


def test_training_queries_cannot_be_emitted_as_oof(backend, fold):
    model = fitted(backend, fold)
    train = fold[0]
    with pytest.raises(ContractError, match='held-out'):
        query(model, train, torch.ones(4, 1, 1))


@pytest.mark.parametrize('name', ['A0', 'A1', 'A3', 'A4', 'A5'])
def test_pma_variant_cannot_silently_be_no_pma(architecture, name):
    with pytest.raises(ContractError, match='requires county context'):
        build_variant(name, architecture[0], treatment_design=architecture[2], raw_x_dim=2)


@pytest.mark.parametrize('cell', [np.float64(1.), np.int64(1), np.bool_(True), float('nan')])
def test_merged_view_requires_python_scalar_cells(fold, cell):
    # The merged contract fails before _matrix sees a NumPy scalar or NaN.
    # Convert scalars to builtins and missing values to None in the data adapter.
    with pytest.raises(ContractError, match='value does not match'):
        replace(fold[0], values=((cell, 1.), (2., 3.), (4., 5.), (6., 7.)))


def test_integer_raw_x_is_valid_numeric_input(backend, fold):
    model = fitted(backend, fold)
    held = replace(fold[1], values=((1, 2), (3, 4)))
    raw = torch.tensor(held.values)
    assert raw.dtype == torch.int64
    integer = model.mean(torch.ones(2, 1, 1), held, raw, None, None)
    floating = model.mean(torch.ones(2, 1, 1), held, raw.float(), None, None)
    torch.testing.assert_close(integer, floating)


def test_numpy_integer_seed_is_canonicalized(backend, fold):
    prototype = backend[0]()
    model = type(prototype)(prototype.checkpoint, task='outcome', family='identity', seed=np.int64(11))
    model.fit_outcome(fold[0], fold[2], 0, np.arange(4), np.arange(4),
                      sample_weight=np.ones(4), weight_semantics='unit')
    assert type(model.seed) is int and model.seed == 11


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool])
def test_equivalent_numeric_representations_preserve_predictions(backend, fold, dtype):
    model = fitted(backend, fold)
    held = replace(fold[1], values=((0, 1), (1, 0)))
    result = model.mean(torch.ones(2, 1, 1), held, torch.tensor(held.values, dtype=dtype), None, None)
    torch.testing.assert_close(result, query(model, held, torch.ones(2, 1, 1)))


def test_integer_raw_x_does_not_truncate_fractional_view_values(backend, fold):
    model = fitted(backend, fold)
    held = replace(fold[1], values=((0.1, 0.2), (0.3, 0.4)))
    with pytest.raises(ContractError, match='raw-X'):
        model.mean(torch.ones(2, 1, 1), held, torch.zeros(2, 2, dtype=torch.int64), None, None)


@pytest.mark.parametrize('seed', [-1, 1.5, True, np.bool_(True)])
def test_invalid_seed_values_remain_blocked(backend, seed):
    prototype = backend[0]()
    with pytest.raises(ContractError, match='integer seed'):
        type(prototype)(prototype.checkpoint, task='outcome', family='identity', seed=seed)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_real_floating_query_representations(backend, fold, dtype):
    model = fitted(backend, fold)
    result = query(model, fold[1], torch.ones(2, 1, 1, dtype=dtype))
    assert result.dtype == dtype and torch.isfinite(result).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_origin_near_boundary_logits_before_query_cast(backend, fold, monkeypatch, dtype):
    train, held, split = fold
    model = backend[0](task='origin', family='bernoulli').fit_origin(
        train, split, 0, make_pairs(train), weight_semantics='unit')
    held = replace(held, original_ids=('h0',), values=((10., 11.),),
                   lineage=replace(held.lineage, unit_ids=('h0',)))
    predict_proba = model._estimator.predict_proba
    monkeypatch.setattr(model._estimator, 'predict_proba',
                        lambda row: predict_proba(row).astype(np.float32))
    p = model._estimator.predict_proba(np.array([[100., 10., 11.]]))[0, 0]
    assert float(p) == 0.9993788599967957
    args = (torch.full((1, 1, 1), 100., dtype=dtype), held,
            torch.tensor(held.values, dtype=dtype), None, None)
    probability = model.probability(*args)
    assert probability.dtype == dtype
    if dtype == torch.bfloat16:
        assert probability.item() == 1.  # Reporting may round; logit must not.
    actual = model.logits(*args)
    expected = torch.tensor([[np.log(np.float64(p)) - np.log1p(-np.float64(p))]], dtype=dtype)
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('backend_dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('near', [0., 1.])
def test_origin_preserves_backend_precision_before_logit(backend, fold, monkeypatch, backend_dtype, near):
    train, held, split = fold
    model = backend[0](task='origin', family='bernoulli').fit_origin(
        train, split, 0, make_pairs(train), weight_semantics='unit')
    p = np.nextafter(backend_dtype(near), backend_dtype(1. - near))
    probs = np.array([[p, 1 - p]], dtype=backend_dtype)  # Mock classes are [1, 0].
    monkeypatch.setattr(model._estimator, 'predict_proba', lambda row: probs.copy())
    a = torch.ones(2, 1, 1, dtype=torch.float64)
    actual = model.logits(a, held, torch.tensor(held.values), None, None)
    wide = np.asarray(p, dtype=np.result_type(backend_dtype, np.float64))
    expected = float(np.log(wide) - np.log1p(-wide))
    assert actual.dtype == a.dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.full((2, 1), expected, dtype=a.dtype), rtol=1e-14, atol=0)


@pytest.mark.parametrize('boundary', [0., 1.])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_origin_genuine_boundary_probability_still_blocks(backend, fold, monkeypatch, boundary, dtype):
    train, held, split = fold
    model = backend[0](task='origin', family='bernoulli').fit_origin(
        train, split, 0, make_pairs(train), weight_semantics='unit')
    monkeypatch.setattr(model._estimator, 'predict_proba',
                        lambda row: np.array([[boundary, 1 - boundary]]))
    args = (torch.ones(2, 1, 1, dtype=dtype), held, torch.tensor(held.values), None, None)
    assert (model.probability(*args) == boundary).all()
    with pytest.raises(ContractError, match='boundary probability; no implicit clipping'):
        model.logits(*args)


def outcome_scale_fold(fold, scale):
    train, held, split = fold
    spec = replace(train.spec, outcome_scale=scale)
    split = replace(split, spec=spec)
    views = [replace(view, spec=spec, lineage=replace(view.lineage, split_hash=split.content_hash))
             for view in (train, held)]
    return *views, split


@pytest.mark.parametrize('family,scale', [
    ('identity', 'count'), ('identity', 'rate'), ('identity', 'risk_difference'),
    ('identity', 'unknown'), ('bernoulli', 'count'), ('bernoulli', 'years'),
    ('bernoulli', 'binomial'), ('bernoulli', 'unknown'),
])
def test_unsupported_target_scales_block_before_backend(backend, fold, monkeypatch, family, scale):
    train, _, split = outcome_scale_fold(fold, scale)
    model = backend[0](family=family)
    calls = []
    make = model._make_estimator
    def record_backend():
        calls.append(True)
        return make()
    monkeypatch.setattr(model, '_make_estimator', record_backend)
    labels = [0, 1, 2, 3] if family == 'identity' else [0, 1, 0, 1]
    with pytest.raises(ContractError, match='unsupported outcome scale'):
        model.fit_outcome(train, split, 0, np.arange(4.), labels,
                          sample_weight=np.ones(4), weight_semantics='unit')
    assert not calls and model._estimator is None


@pytest.mark.parametrize('family,scale', [
    ('identity', 'years'), ('identity', 'grams'), ('identity', 'g/dL'),
    ('bernoulli', 'risk_difference'),
])
def test_supported_target_scales_keep_endpoint_units(backend, fold, family, scale):
    fold = outcome_scale_fold(fold, scale)
    model = fitted(backend, fold, family=family)
    assert model.spec.outcome_scale == scale
    assert torch.isfinite(query(model, fold[1])).all()


@pytest.mark.parametrize('direction', [0, 2])
@pytest.mark.parametrize('container', ['array', 'list'])
def test_extended_precision_nonunit_weights_are_not_discarded(backend, fold, direction, container):
    weights = np.ones(4, dtype=np.longdouble)
    weights[1] = np.nextafter(np.longdouble(1), np.longdouble(direction))
    assert weights[1] != 1
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        assert np.float64(weights[1]) == 1
    if container == 'list':
        weights = list(weights)
    with pytest.raises(ContractError, match='unsupported.*weight'):
        backend[0]().fit_outcome(fold[0], fold[2], 0, np.arange(4.), np.arange(4.),
                                sample_weight=weights, weight_semantics='unit')


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble, np.int64])
def test_exact_unit_weights_are_accepted(backend, fold, dtype):
    model = backend[0]().fit_outcome(fold[0], fold[2], 0, np.arange(4.), np.arange(4.),
                                    sample_weight=np.ones(4, dtype=dtype), weight_semantics='unit')
    assert model._estimator is not None


@pytest.mark.parametrize('variant,expected', [('A3', 966273), ('A4', 965544)])
def test_alternative_cap_uses_final_architecture(architecture, variant, expected):
    encoder, context, design = architecture
    pair = build_variant(variant, encoder, treatment_design=design, raw_x_dim=12810,
                         county_context=context, dropout=0.)
    for model in (pair.outcome, pair.correction):
        assert sum(p.numel() for p in model.parameters()) == expected
        assert model.check_parameter_cap() == expected
    # The next width above the final architecture's cap must still be rejected.
    over_width = 12810 + (1_000_000 - expected) // 64 + 1
    for model in (pair.outcome, pair.correction):
        with pytest.raises(ContractError, match='one-million-parameter cap'):
            type(model)(encoder, treatment_design=design, raw_x_dim=over_width,
                        county_context=context, dropout=0.)


def test_unregistered_checkpoint_identity_blocks_before_fit(backend, fold, tmp_path):
    model = backend[0]()
    alternate = tmp_path / 'alternate-regressor.ckpt'
    alternate.write_bytes(b'a different synthetic checkpoint')
    fitted_before = len(MockEstimator.instances)
    with pytest.raises(ContractError, match='unregistered checkpoint identity'):
        checkpoint = replace(model.checkpoint, repository='fixture/alternate', revision='b'*40,
                             filename=alternate.name, path=str(alternate),
                             sha256=sha256(alternate.read_bytes()).hexdigest())
        type(model)(checkpoint, task='outcome', family='identity', seed=11).fit_outcome(
            fold[0], fold[2], 0, np.arange(4.), np.arange(4.),
            sample_weight=np.ones(4), weight_semantics='unit')
    assert len(MockEstimator.instances) == fitted_before


@pytest.mark.parametrize('field,value', [
    ('repository', 'fixture/alternate'), ('revision', 'b'*40), ('sha256', 'b'*64),
    ('filename', 'alternate-regressor.ckpt'),
])
def test_checkpoint_registration_binds_each_identity_field(backend, field, value):
    checkpoint = backend[0]().checkpoint
    update = {field: value}
    if field == 'filename':
        update['path'] = str(Path(checkpoint.path).with_name(value))
    with pytest.raises(ContractError, match='unregistered checkpoint identity'):
        replace(checkpoint, **update)


def test_signed_riesz_rejects_unused_origin_offsets(architecture, fold):
    encoder, context, design = architecture
    offsets = CountyOffsets(fold[2], 0, ('c',)*4, family='bernoulli',
                            exposure_assignment_level='tract')
    with pytest.raises(ContractError, match='signed Riesz.*offset'):
        build_variant('A5', encoder, treatment_design=design, raw_x_dim=2,
                      county_context=context, origin_offsets=offsets, dropout=0.)


def test_selected_checkpoint_identities_accept_local_provisioning_paths(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1]/'configs/models/foundations.yaml').read_text())
    for item in config['checkpoints'].values():
        environment = tuple((name, item['package_version'] if name == item['package'] else 'fixture')
                            for name in ('python', *icl.RUNTIME_PACKAGES, item['package']))
        identity = {field: item[field] for field in icl.CHECKPOINT_IDENTITY_FIELDS}
        checkpoint = icl.Checkpoint(**identity, path=str(tmp_path/item['filename']), environment=environment)
        assert checkpoint.path == str(tmp_path/item['filename'])
        assert checkpoint.package_version == item['package_version']
