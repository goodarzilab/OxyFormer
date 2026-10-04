"""Synthetic score algebra, nuisance-error and artifact-boundary acceptance."""
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from oxyformer.contracts import (
    ColumnSpec, DataManifest, EstimandSpec, OOFNuisances, SourceManifest, SplitManifest,
    source_lineage_hash,
)
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule
from oxyformer.data.loaders import load_records
from oxyformer.design.policies import ShiftOrStayPolicy
from oxyformer.estimation.influence import influence_contributions
from oxyformer.estimation.mtp import one_step, one_step_scores, pushforward_ratio
from oxyformer.provenance import ArtifactLineage, ContractError
from oxyformer.validation.analytic_truth import UniformShiftTruth


def make_fixture(delta=2, components=((0, 10),)):
    policy = ShiftOrStayPolicy(support_design_hash="a" * 64, delta_mmhg=delta,
                               components_by_key=(("s", components),))
    ids = ("o1", "o2", "o3", "o4")
    source = SourceManifest(source_id="synthetic", version="1", uri="synthetic://fixture",
                            payload_hash="1" * 64, license_hash="2" * 64, schema_hash="3" * 64,
                            field_mapping=(("raw_id", "id"),), mapping_status="reviewed",
                            mapping_review_id="synthetic-only")
    registry = FeatureRegistry(registry_id="synthetic", rules=tuple(
        FeatureRule(name=name, role=role, endpoints=("synthetic",), uses=(use,), approval_id="fixture")
        for name, role, use in (("id", "identifier", "linkage"), ("y", "outcome", "score"),
                                ("a", "exposure", "score"), ("w", "outcome_metadata", "linkage"))))
    spec = EstimandSpec(endpoint="synthetic", target_id="fixture", outcome_scale="years",
                        policy_id=policy.policy_id, weight_id="fixture-origin-weights",
                        adjustment_schema_hash=registry.content_hash, inference_unit="geography",
                        source_lineage_hash=source_lineage_hash((source,)))
    lineage = ArtifactLineage(source_hashes=(source.payload_hash,), unit_ids=ids, parent_hashes=(),
                              split_hash=None, config_hash="4" * 64, model_hash=None,
                              environment=(("python", "synthetic"),), seed=None, parameter_count=None)
    manifest = DataManifest(spec=spec, sources=(source,), schema=tuple(
        ColumnSpec(name=name, dtype=dtype) for name, dtype in
        (("id", "string"), ("y", "number"), ("a", "number"), ("w", "number"))),
        registry=registry, original_ids=ids, id_field="id", outcome_field="y", exposure_field="a",
        weight_field="w", entity_graph_hash="5" * 64, lineage=lineage)
    data = load_records([dict(id=oid, y=float(y), a=float(a), w=float(w))
                         for oid, y, a, w in zip(ids, [4, 7, 12, 14], [1, 5, 9, 9], [1, 2, 3, 4])],
                        manifest, spec, manifest.schema_hash)
    split = SplitManifest(spec=spec, level="outer", original_ids=ids, fold_ids=(0, 1, 0, 1),
                          design_ids=(), excluded_ids=(), seed_ids=(1103, 2207),
                          entity_graph_hash=manifest.entity_graph_hash,
                          lineage=replace(lineage, parent_hashes=(manifest.content_hash,)))
    n = OOFNuisances(spec=spec, original_ids=ids[::-1] * 2, fold_ids=(1, 0, 1, 0) * 2,
                     seed_ids=(1103,) * 4 + (2207,) * 4,
                     mu_a=(9, 9, 5, 1, 10, 8, 6, 2), mu_d=(9, 9, 7, 3, 10, 8, 8, 4),
                     r_a=(2, 2, 1, 0, 1.5, 2.5, 1.2, 0.2), r_d=(2,) * 8,
                     origin_weights=(4, 3, 2, 1) * 2,
                     lineage=replace(lineage, parent_hashes=(manifest.content_hash,),
                                     split_hash=split.content_hash))
    weights = dict(zip(ids, [1, 2, 3, 4]))
    return policy, data, split, n, weights


def estimate(fixture):
    policy, data, split, n, weights = fixture
    return one_step(n, data, weights, n.spec, split=split, policy=policy)


def test_score_algebra_and_unchanged_incoming_correction():
    mu = np.array([1, 5, 9], dtype=float)
    md, r, y = np.array([3, 7, 9]), np.array([0, 1, 2]), np.array([4, 7, 12])
    h = one_step_scores(mu, md, r, y)
    assert_array_equal(h, [-1, 2, 3])
    assert_allclose(h, md + r * (y - mu) - y)
    # The last observation stays, but incoming shifted mass gives correction 3.
    assert h[-1] == 3
    value, influence = influence_contributions(h, [1, 2, 3])
    assert value == pytest.approx(2)
    assert_allclose(influence, [-0.5, 0, 0.5])


def test_estimator_averages_scores_and_influences_by_original_id():
    fixture = make_fixture()
    policy, data, split, n, weights = fixture
    e = estimate(fixture)
    expected_seed_scores = np.array([[-1, 2, 3, 5], [0.4, 2.2, 6, 2]])
    expected = expected_seed_scores.mean(axis=0)
    assert e.original_ids == split.original_ids
    assert_allclose(e.scores, expected)
    assert e.value == pytest.approx(np.average(expected, weights=[1, 2, 3, 4]))
    expected_u = [influence_contributions(row, [1, 2, 3, 4])[1] for row in expected_seed_scores]
    assert_allclose(e.influence, np.mean(expected_u, axis=0))
    assert sum(e.influence) == pytest.approx(0, abs=1e-15)
    assert e.standard_error is None
    assert e.seed_ids == split.seed_ids
    assert {policy.content_hash, n.content_hash, data.content_hash, split.content_hash} == set(e.lineage.parent_hashes)
    assert e.lineage.seed is e.lineage.model_hash is e.lineage.parameter_count is None
    assert e.from_json(e.to_json()) == e
    # Show averaging nuisances before multiplying would give a different answer.
    wrong = one_step_scores(np.mean(np.array(n.mu_a).reshape(2, 4), axis=0),
                            np.mean(np.array(n.mu_d).reshape(2, 4), axis=0),
                            np.mean(np.array(n.r_a).reshape(2, 4), axis=0), [14, 12, 7, 4])
    assert not np.allclose(wrong[::-1], expected)


@pytest.mark.parametrize("delta,components", [(0, ((0, 10),)), (2, ((0, 1), (3, 4)))])
def test_declared_identity_is_exactly_zero_despite_nuisance_disagreement(delta, components):
    fixture = make_fixture(delta, components)
    assert fixture[0].is_identity
    e = estimate(fixture)
    assert e.value == 0.0
    assert e.scores == e.influence == (0.0,) * 4
    assert_array_equal(one_step_scores([1e308], [-1e308], [1e308], [-1e308], identity=True), [0])


def test_sample_with_nobody_moved_is_not_declared_identity():
    fixture = make_fixture()
    p, data, split, n, w = fixture
    n = replace(n, mu_d=n.mu_a)
    e = estimate((p, data, split, n, w))
    assert not p.is_identity
    assert e.value != 0


def quadrature():
    # Integrate polynomial scores exactly across every ratio discontinuity.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    a, q = [], []
    for lo, hi in [(0, 2), (2, 8), (8, 10)]:
        a.extend((lo + (nodes + 1) * (hi - lo) / 2).tolist())
        q.extend((weights * (hi - lo) / 20).tolist())
    return np.array(a), np.array(q)


@pytest.mark.parametrize("mu_error,ratio_error", [(0, 0), (0, 0.4), (0.7, 0), (0.7, 0.4), (-0.3, 0.8)])
def test_either_oracle_and_mixed_error_remainder(mu_error, ratio_error):
    a, q = quadrature()
    d = np.where(a <= 8, a + 2, a)
    true_mu = lambda a: 5 + 1.5 * a + 0.2 * a**2
    error = lambda a: mu_error * (1 + a + 0.1 * a**2)
    p = make_fixture()[0]
    r = pushforward_ratio(p, a, ("s",) * len(a), UniformShiftTruth().density, exposure_law="continuous")
    rhat = r + ratio_error * (1 + 0.02 * a)
    muhat, mdhat = true_mu(a) + error(a), true_mu(d) + error(d)
    # Paired +/- noise has conditional mean zero at every quadrature node.
    h = (one_step_scores(muhat, mdhat, rhat, true_mu(a) + 0.5)
         + one_step_scores(muhat, mdhat, rhat, true_mu(a) - 0.5)) / 2
    truth = q @ (true_mu(d) - true_mu(a))
    remainder = -q @ ((rhat - r) * error(a))
    assert q @ h - truth == pytest.approx(remainder, abs=2e-13)
    if mu_error == 0 or ratio_error == 0:
        assert q @ h == pytest.approx(truth, abs=2e-13)


def test_linear_truth_is_1_point_6_beta():
    a, q = quadrature()
    d = np.where(a <= 8, a + 2, a)
    beta = 3.0
    h = one_step_scores(beta * a, beta * d, UniformShiftTruth().ratio(a), beta * a)
    assert q @ h == pytest.approx(1.6 * beta)
    assert q @ h != pytest.approx(2 * beta)


@pytest.mark.parametrize("field,value", [("outcome_scale", "grams"), ("policy_id", "other"),
                                        ("weight_id", "population"), ("target_id", "other"),
                                        ("adjustment_schema_hash", "f" * 64)])
def test_mismatched_estimand_is_rejected(field, value):
    p, data, split, n, w = make_fixture()
    with pytest.raises(ContractError, match=f"{field} mismatch"):
        one_step(n, data, w, replace(n.spec, **{field: value}), split=split, policy=p)
    with pytest.raises(ContractError, match=f"{field} mismatch"):
        estimate((p, data, split, replace(n, spec=replace(n.spec, **{field: value})), w))


def test_wrong_policy_and_weights_rejected_even_for_identity():
    p, data, split, n, w = make_fixture(0)
    with pytest.raises(ContractError, match="policy_id mismatch"):
        estimate((replace(p, delta_mmhg=2), data, split, n, w))
    with pytest.raises(ContractError, match="origin weight mismatch"):
        estimate((p, data, split, n, dict(w, o1=5)))
    with pytest.raises(ContractError, match="observation IDs"):
        estimate((p, data, split, n, {"wrong": 1}))
    with pytest.raises(ContractError, match="invalid origin weights"):
        estimate((p, data, split, n, dict(w, o1=-1)))


def test_incomplete_seeds_and_wrong_fold_are_rejected():
    p, data, split, n, w = make_fixture()
    vector_fields = ("original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")
    partial = replace(n, **{field: getattr(n, field)[:4] for field in vector_fields})
    with pytest.raises(ContractError, match="incomplete original ID/seed coverage"):
        estimate((p, data, split, partial, w))
    wrong = replace(n, fold_ids=tuple(1 - fold for fold in n.fold_ids))
    with pytest.raises(ContractError, match="held-out fold mismatch"):
        estimate((p, data, split, wrong, w))
    with pytest.raises(ContractError, match="duplicate original ID/seed"):
        replace(n, seed_ids=(1103,) * 8)


def test_fp64_arithmetic_and_weight_scale_invariance():
    # FP32 multiplication loses the +1 before the subtracting plugin contrast.
    h = one_step_scores(np.array([0], dtype=np.float32), np.array([-1e8], dtype=np.float32),
                        np.array([1e8], dtype=np.float32), np.array([-1], dtype=np.float32))
    assert h.dtype == np.float64
    assert h[0] == -199999999
    delta, u = influence_contributions([1, 3], [1, 2])
    large_delta, large_u = influence_contributions([1, 3], [1e308, 1.5e308])
    assert np.isfinite(large_delta) and np.isfinite(large_u).all()
    assert_allclose(influence_contributions([1, 3], [100, 200])[1], u)
    assert influence_contributions([1, 3], [100, 200])[0] == delta


@pytest.mark.parametrize("argument,value", [("ratios", [-1]), ("mu_a", [np.nan]),
                                           ("outcomes", [np.inf]), ("mu_d", [1, 2])])
def test_invalid_score_inputs_are_rejected(argument, value):
    args = dict(mu_a=[1], mu_d=[2], ratios=[1], outcomes=[3])
    args[argument] = value
    with pytest.raises(ContractError):
        one_step_scores(**args)
