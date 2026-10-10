"""Synthetic CPU chain through the merged scientific APIs, never a release.

One-epoch fits exercise all five outer folds, three seeds and four tuning
choices. They are an explicitly shortened test fixture, not coverage evidence.
The common CLI has separately documented prerequisite defects; no substitute
dispatcher, owner approval or production adapter is fabricated here.
"""
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from oxyformer.contracts import OOFNuisances, StageRequest
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import load_records, validate_oof
from oxyformer.design.eligibility import AtlasRow, GeographyRow
from oxyformer.design.gate import FrozenDesign, run_stage as design_stage
from oxyformer.design.policies import PolicyCovariates
from oxyformer.design.splits import SplitScenario
from oxyformer.estimation.covariance import align_estimates, cluster_covariance, spatial_sensitivities
from oxyformer.estimation.mtp import one_step
from oxyformer.estimation.targeting import cv_tmle
from oxyformer.exposure.physics import PHYSICS, oxygen_deficit_mmhg, validate_owner_approval
from oxyformer.models.treatment_query import TreatmentDesign
from oxyformer.provenance import ContractError, canonical_json, file_hash, write_artifact
from oxyformer.reporting import run_stage as report_stage
from oxyformer.reporting.records import ExpectedTasks, ReportBundle, TaskReceipts
from oxyformer.training import nested_cv as nested
from oxyformer.training.checkpoint import load_checkpoint
from oxyformer.training.fit import NuisanceSettings, subset
from oxyformer.validation.leakage import assert_fitted_invariant, assert_prediction_invariant
from test_design import make_inputs, make_request

ROOT = Path(__file__).resolve().parents[1]


def exposure_design_inputs():
    """Two synthetic counties; sealed blocks alone carry the support tails."""
    validate_owner_approval()
    template = make_inputs()
    geography, atlas = [], []
    elevations = {}
    for county in ("c1", "c2"):
        names = [f"b{i:02}" for i in range(7)]
        sealed = set(sorted(names, key=lambda b: sha256(canonical_json([1103, county, b]).encode()).hexdigest())[:2])
        for block, name in enumerate(names):
            # High-altitude synthetic bins keep the approved 300 m relief
            # screen while needing only four interior doses per subblock.
            for dose in (range(100, 106) if name in sealed else range(101, 105)):
                oid = f"{county}-{name}-{dose:02}"
                # Invert the fixed formula only to place a synthetic exposure
                # gradient inside each bin; the forward service computes A.
                p = PHYSICS
                exponent = p.gravity_m_per_s2 * p.molar_mass_air_kg_per_mol / (p.gas_constant_j_per_mol_k * p.lapse_rate_k_per_m)
                z = p.sea_level_temperature_k / p.lapse_rate_k_per_m * (1 - (1 - (dose + .25) / (p.inspired_o2_fraction * p.sea_level_pressure_mmhg)) ** (1 / exponent))
                # Two equal-population locations: transform before averaging.
                locations = np.array([z - 2., z + 2.])
                a = float(np.average(oxygen_deficit_mmhg(locations), weights=[1., 1.]))
                elevations[oid] = locations
                geography.append(GeographyRow(original_id=oid, tract_id=oid, county=county,
                    state="s1" if county == "c1" else "s2", subblock=name, assignment_geography=oid,
                    latitude=40. if county == "c1" else 42., longitude=-100 + block * .003,
                    outcome_flag=1 if dose % 2 == 0 else 3, label_available=True))
                atlas.append(AtlasRow(tract_id=oid, exposure_mmhg=a, inhabited_elevation_m=float(z),
                                      population=2, allocation_qualified=True))
    ids = tuple(r.original_id for r in geography)
    graph = EntityGraph(original_ids=ids, links=())
    manifest = replace(template["data_manifest"], original_ids=ids, entity_graph_hash=graph.content_hash,
                       lineage=replace(template["data_manifest"].lineage, unit_ids=ids))
    result = dict(data_manifest=manifest, entity_graph=graph,
        geography=replace(template["geography"], rows=tuple(geography), data_manifest_hash=manifest.content_hash),
        atlas=replace(template["atlas"], rows=tuple(atlas), expected_tract_ids=ids),
        covariates=replace(template["covariates"], original_ids=ids, values=((.5,),) * len(ids),
            lineage=replace(template["covariates"].lineage, unit_ids=ids, parent_hashes=(manifest.content_hash,))))
    return result, elevations


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root = tmp_path_factory.mktemp("end-to-end")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            yield build_chain(root)
    finally:
        torch.set_num_threads(old_threads)


def fit_partition(arguments):
    """Independent CPU processes keep RNG and torch state isolated per fit."""
    prepared, root, fold, seed = arguments
    torch.set_num_threads(1)

    def config(path, **kwargs):
        return prepared.configuration(fold, path, ssl_epochs=1,
            settings=NuisanceSettings(batch_size=256, frozen_epochs=1), **kwargs)

    location = root / f"fold-{fold}-seed-{seed}"
    artifact = nested.run_fold(config(location), prepared.outer, seed, geography=prepared.geography)
    assert artifact.complete
    controller = load_checkpoint(artifact.checkpoint, artifact.checkpoint.identity)["controller"]
    assert controller["counts"]["nuisance_fits"] == 26
    assert controller["counts"]["ssl_fits"] == 4
    ids = artifact.prediction_inputs.original_ids
    view = subset(prepared.data.covariates(("female_share",)), ids)
    prediction = nested.predict(artifact, view, prepared.policy)
    if fold == 0 and seed == nested.SEEDS[0]:
        partial_root = root / "interrupted"
        partial = nested.run_fold(config(partial_root / "work", max_batches=1), prepared.outer,
                                  seed, geography=prepared.geography)
        assert not partial.complete
        archive = root / "continuation.tar"
        binding = {"endpoint": prepared.content_hash, "design": prepared.treatment_design.design_hash}
        nested.export_continuation(partial, archive, binding=binding, task_id="first",
            chain=dict(owner="synthetic-chain", step=0, predecessor=None), allowed_root=partial_root)
        shutil.rmtree(partial_root)  # Resume cannot rely on the old attempt.
        destination = root / "continued"
        destination.mkdir()
        restored = nested.import_continuation(archive, destination, binding=binding,
            chain=dict(owner="synthetic-chain", step=1, predecessor="first"))
        resumed = nested.run_fold(config(destination / "work", predecessor=restored), prepared.outer,
                                  seed, geography=prepared.geography)
        assert_fitted_invariant(artifact, resumed)
        assert_prediction_invariant(prediction, nested.predict(resumed, view, prepared.policy))
    return prediction


def build_chain(root):
    inputs, elevations = exposure_design_inputs()
    request = make_request(root / "design", inputs)
    result = design_stage(request)
    assert result.status == "pass", result.message
    result.verify(request)
    output = Path(request.output_dir)
    design = FrozenDesign.from_json((output / "design.json").read_text())
    scenario = SplitScenario.from_json(canonical_json(json.loads((output / "splits.json").read_text())["scenarios"][0]))
    manifest = scenario.data_manifest
    by_id = {r.tract_id: r for r in inputs["atlas"].rows}
    geo = inputs["geography"]
    records = [dict(id=r.original_id, y=70. + .2 * by_id[r.tract_id].exposure_mmhg + (j % 3) * .1,
                    a=by_id[r.tract_id].exposure_mmhg, female_share=.5, county=r.county)
               for j, r in enumerate(geo.rows)]
    data = load_records(records, manifest, manifest.spec, manifest.schema_hash)
    knots = design.support.spline_knots
    prepared = nested.PreparedEndpoint(data=data, entity_graph=inputs["entity_graph"],
        geography=replace(geo, data_manifest_hash=manifest.content_hash), outer=scenario.outer, inner=scenario.inner,
        policy=design.support.policy,
        policy_covariates=PolicyCovariates(original_ids=manifest.original_ids,
            geography_ids=tuple(r.assignment_geography for r in geo.rows),
            support_keys=tuple(r.assignment_geography for r in geo.rows)),
        treatment_design=TreatmentDesign(center=(knots[0] + knots[-1]) / 2,
            scale=knots[-1] - knots[0], knots=knots, design_hash=design.content_hash),
        feature_kinds=(("female_share", "numeric"),), families=(("female_share",),),
        county_field="county", exposure_assignment_level="tract")
    arguments = [(prepared, root, fold, seed) for fold in range(5) for seed in nested.SEEDS]
    with ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(fit_partition, arguments))
    fields = ("original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")
    lineage = replace(parts[0].lineage, unit_ids=prepared.outer.original_ids,
        parent_hashes=(manifest.content_hash, *(p.content_hash for p in parts)),
        split_hash=prepared.outer.content_hash, model_hash=None, seed=None, parameter_count=None)
    oof = OOFNuisances(spec=manifest.spec, lineage=lineage,
                      **{f: tuple(x for p in parts for x in getattr(p, f)) for f in fields})
    validate_oof(oof, data, prepared.outer)
    weights = dict.fromkeys(prepared.outer.original_ids, 1.)
    one = one_step(oof, data, weights, manifest.spec, split=prepared.outer, policy=prepared.policy)
    targeted = cv_tmle(oof, data, weights, "identity", manifest.spec,
                       split=prepared.outer, policy=prepared.policy).estimate
    return root, inputs, elevations, prepared, oof, one, targeted


def test_exposure_design_fit_continuation_score_targeting_covariance_reporting(chain):
    root, inputs, elevations, prepared, oof, one, targeted = chain
    ids = prepared.outer.original_ids
    assert len(oof.original_ids) == len(ids) * 3
    assert one.original_ids == targeted.original_ids == ids
    assert one.spec == targeted.spec == prepared.data.manifest.spec == oof.spec
    assert one.spec.policy_id == prepared.policy.policy_id
    assert set(ids).isdisjoint(prepared.outer.design_ids)
    assert len(one.influence) == len(ids)  # Seeds are not independent people.
    for row in inputs["atlas"].rows:
        assert row.exposure_mmhg == float(np.average(oxygen_deficit_mmhg(elevations[row.tract_id])))
    geo = {r.original_id: r for r in prepared.geography.rows}
    groups = {i: geo[i].county for i in ids}
    locations = {"c1": (40., -100.), "c2": (42., -100.)}
    aligned = align_estimates({"one_step": one, "cv_tmle": targeted})
    covariance = cluster_covariance(aligned, groups, interpretation="geographic_process")
    assert covariance.dependence_units == 2 and np.asarray(covariance.matrix).shape == (2, 2)
    assert set(spatial_sensitivities(aligned, groups, locations, interpretation="geographic_process")) == {50, 100, 200}
    row_by_id = dict(zip(prepared.data.manifest.original_ids, prepared.data.column("a")))
    a = tuple(row_by_id[i] for i in ids)
    pc = PolicyCovariates(original_ids=ids, geography_ids=ids, support_keys=ids)
    shifted = prepared.policy.apply(a, pc).d_mmhg
    ratios = {(i, s): r for i, s, r in zip(oof.original_ids, oof.seed_ids, oof.r_a)}
    bundle = ReportBundle(spec=one.spec, sources=prepared.data.manifest.sources, original_ids=ids,
        attrition=(("source", len(prepared.data.rows)), ("target", len(ids))), weights=(1.,) * len(ids),
        observed_exposure=a, shifted_exposure=shifted, seed_ids=nested.SEEDS,
        ratios=tuple(tuple(ratios[i, s] for i in ids) for s in nested.SEEDS),
        balance_basis_id="synthetic-frozen-constant-dose", balance_names=("constant", "A"),
        balance_observed=tuple((1., x) for x in a), balance_shifted=tuple((1., x) for x in shifted),
        estimates=(one, targeted), counties=tuple(groups[i] for i in ids),
        states=tuple(geo[i].state for i in ids), county_locations=tuple(locations.items()), coverage=())
    expected = ExpectedTasks(stage="tract_release", spec=one.spec, seed_ids=nested.SEEDS,
                             tasks=(), coverage_scenarios=("unrun-final-scenario",))
    paths = {}
    for name, value in (("bundle", bundle), ("manifest", expected), ("receipts", TaskReceipts(items=()))):
        path = root / f"{name}.json"
        write_artifact(path, value)
        paths[name] = str(path)
    paths["approvals"] = str(ROOT / "configs/approvals.yaml")
    task = root / "report-task.json"
    task.write_text(canonical_json(paths))
    config = ROOT / "configs/reporting.yaml"
    request = StageRequest(stage="tract_release", config_path=str(config), config_hash=file_hash(config),
        task_path=str(task), task_hash=file_hash(task), dependency_paths=tuple(paths.values()),
        dependency_hashes=tuple(file_hash(p) for p in paths.values()), output_dir=str(root / "report"), code_identity="a" * 40)
    result = report_stage(request)
    result.verify(request)
    report = json.loads((Path(request.output_dir) / "report.json").read_text())
    assert result.status in {"blocked", "fail"}
    assert not report["releasable"] and report["evidence_label"] == "diagnostic-only"
    assert {e["method"] for e in report["estimators"]} == {one.method, targeted.method}
    assert any(g["status"] == "missing" for g in report["gates"])
    assert {a.path for a in result.artifacts} == {"report.json", "report.html", "estimators.svg"}


def test_incomplete_predictions_and_changed_identity_block_scoring(chain):
    _, _, _, prepared, oof, _, _ = chain
    fields = ("original_ids", "fold_ids", "seed_ids", "mu_a", "mu_d", "r_a", "r_d", "origin_weights")
    partial = replace(oof, **{f: getattr(oof, f)[:-1] for f in fields})
    weights = dict.fromkeys(prepared.outer.original_ids, 1.)
    with pytest.raises(ContractError, match="incomplete original ID/seed coverage"):
        one_step(partial, prepared.data, weights, oof.spec, split=prepared.outer, policy=prepared.policy)
    with pytest.raises(ContractError, match="target_id"):
        one_step(oof, prepared.data, weights, replace(oof.spec, target_id="different-target"),
                 split=prepared.outer, policy=prepared.policy)


def test_incomplete_exposure_blocks_design(tmp_path):
    inputs, _ = exposure_design_inputs()
    atlas = inputs["atlas"]
    inputs["atlas"] = replace(atlas, coverage_complete=False,
        expected_tract_ids=atlas.expected_tract_ids + ("missing-tract",), missing_tract_ids=("missing-tract",))
    request = make_request(tmp_path, inputs)
    result = design_stage(request)
    result.verify(request)
    assert result.status == "blocked"
    assert not json.loads((Path(request.output_dir) / "gate.json").read_text())["effect_release_authorized"]
