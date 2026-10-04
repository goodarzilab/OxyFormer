"""Geographic split minima, grouping and buffer tests independent of outcomes."""
from dataclasses import replace

import pytest

from oxyformer.data.entity_graph import EntityGraph, EntityLink
from oxyformer.data.loaders import validate_split
from oxyformer.design.policies import paired_records
from oxyformer.design.splits import (
    Reservation, buffer_exclusions, build_scenario, deletion_refit_ids,
    dependence_groups, geographic_folds, reserve_design,
)
from oxyformer.design.support import freeze_support, policy_for
from test_design import digest, make_inputs
from oxyformer.provenance import ContractError


def split_fixture(n=30, counties=1):
    v = make_inputs()
    template = v["geography"].rows[0]
    rows = tuple(replace(template, original_id=f"o{i:03}", tract_id=f"t{i:03}", subblock=f"b{i:03}",
                         assignment_geography=f"g{i:03}", county=f"c{i // n}",
                         latitude=30 + (i // n) * 10, longitude=-100 + i * .1)
                 for i in range(n * counties))
    ids = tuple(r.original_id for r in rows)
    graph = EntityGraph(original_ids=ids, links=())
    manifest = replace(v["data_manifest"], original_ids=ids, entity_graph_hash=graph.content_hash,
                       lineage=replace(v["data_manifest"].lineage, unit_ids=ids))
    reservation = Reservation(design_ids=(), selected_subblocks=(), sealed_subblocks=(),
                               total_subblocks=len(rows), groups=dependence_groups(rows, graph))
    return rows, graph, manifest, reservation


def scenario_fixture(n=30):
    rows, graph, manifest, reservation = split_fixture(n)
    return build_scenario(rows, graph, manifest, reservation, manifest.original_ids, {},
                          radius_km=0, policy_id=digest("policy"))


def test_eight_training_tract_screen():
    # Nine tracts can have 4/5 per inner fitting partition yet only seven in
    # some outer-training partitions. This isolates the eight-tract screen.
    result = scenario_fixture(9)
    assert result.status == "fail"
    assert result.outer is None
    assert {reason for _, reason in result.exclusions} == {"fewer_than_eight_outer_training_tracts"}
    assert any(a.outer_training_tracts == 7 for a in result.count_audit)
    assert all(min(a.inner_fitting_tracts) >= 4 for a in result.count_audit)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_small_counties_never_pass_preliminary_count_as_operational_gate(n):
    result = scenario_fixture(n)
    assert result.status == "fail"
    assert len(result.exclusions) == n


def test_exact_minima_and_all_folds():
    result = scenario_fixture(10)
    assert result.status == "pass"
    assert set(result.outer.fold_ids) == set(range(5))
    assert all(a.outer_training_tracts == 8 for a in result.count_audit)
    assert all(min(a.inner_fitting_tracts) >= 4 for a in result.count_audit)
    for inner in result.inner:
        assert set(inner.split.fold_ids) == {0, 1, 2}
        outer_held = {oid for oid, fold in zip(result.outer.original_ids, result.outer.fold_ids) if fold == inner.outer_fold}
        assert not outer_held.intersection(inner.split.lineage.unit_ids)
        validate_split(inner.split, inner.data_manifest, inner.entity_graph)


def test_inner_minimum_is_checked_separately():
    rows, graph, manifest, _ = split_fixture(14)
    # One large component leaves highly imbalanced inner training partitions.
    links = tuple(EntityLink(observation_id=r.original_id, relation="household", namespace="synthetic",
                             entity_id="big") for r in rows[:8])
    graph = replace(graph, links=links)
    manifest = replace(manifest, entity_graph_hash=graph.content_hash)
    reservation = Reservation(design_ids=(), selected_subblocks=(), sealed_subblocks=(), total_subblocks=14,
                               groups=dependence_groups(rows, graph))
    result = build_scenario(rows, graph, manifest, reservation, manifest.original_ids, {},
                            radius_km=0, policy_id=digest("policy"))
    assert result.status == "fail"
    assert any(min(a.inner_fitting_tracts) < 4 for a in result.count_audit)


@pytest.mark.parametrize("relation", ["household", "psu", "municipality", "repeated_geography", "outcome_lineage"])
def test_lineage_never_crosses_design_or_folds(relation):
    rows, graph, manifest, _ = split_fixture(50)
    graph = replace(graph, links=tuple(EntityLink(observation_id=rows[i].original_id, relation=relation,
                                                  namespace="common-outcome-system", entity_id="linked")
                                       for i in (0, 49)))
    manifest = replace(manifest, entity_graph_hash=graph.content_hash)
    reservation = reserve_design(rows, graph)
    candidates = set(manifest.original_ids) - set(reservation.design_ids)
    result = build_scenario(rows, graph, manifest, reservation, candidates, {}, radius_km=0,
                            policy_id=digest("policy"))
    assert result.status == "pass"
    validate_split(result.outer, result.data_manifest, graph)
    assignments = dict(zip(result.outer.original_ids, result.outer.fold_ids))
    assignments.update(dict.fromkeys(result.outer.design_ids, "design"))
    assignments.update(dict.fromkeys(result.outer.excluded_ids, "excluded"))
    assert assignments[rows[0].original_id] == assignments[rows[49].original_id]
    for inner in result.inner:
        validate_split(inner.split, inner.data_manifest, inner.entity_graph)


def test_design_seals_transitive_links_and_reports_rounding():
    rows, graph, _, _ = split_fixture(11)
    original = reserve_design(rows, graph)
    assert len(original.selected_subblocks) == 3
    first = original.design_ids[0]
    unsealed = [r.original_id for r in rows if r.original_id not in original.design_ids]
    graph = replace(graph, links=(
        EntityLink(observation_id=first, relation="psu", namespace="p", entity_id="a"),
        EntityLink(observation_id=unsealed[0], relation="psu", namespace="p", entity_id="a"),
        EntityLink(observation_id=unsealed[0], relation="outcome_lineage", namespace="o", entity_id="b"),
        EntityLink(observation_id=unsealed[1], relation="outcome_lineage", namespace="o", entity_id="b"),
    ))
    reserved = reserve_design(rows, graph)
    assert set(unsealed[:2]) <= set(reserved.design_ids)
    assert len(reserved.sealed_subblocks) > len(reserved.selected_subblocks)


def test_buffer_induced_exclusions_are_frozen_and_disclosed():
    rows, graph, manifest, reservation = split_fixture(30)
    # All cross-fold boundaries are within 10 km in this county.
    rows = tuple(replace(r, longitude=-100 + i * .001) for i, r in enumerate(rows))
    base = build_scenario(rows, graph, manifest, reservation, manifest.original_ids, {},
                          radius_km=0, policy_id=digest("policy"))
    buffered = build_scenario(rows, graph, manifest, reservation, manifest.original_ids, {},
                              radius_km=10, policy_id=digest("policy"))
    assert base.status == "pass" and buffered.status == "fail"
    assert buffered.target_id != base.target_id
    assert len(buffered.exclusions) == 30
    assert {reason for _, reason in buffered.exclusions} == {"outer_buffer"}


def test_buffer_removes_entire_component():
    rows, graph, _, _ = split_fixture(6)
    rows = tuple(replace(r, latitude=30, longitude=-100 + i * .01) for i, r in enumerate(rows))
    groups = ((rows[0].original_id, rows[-1].original_id),) + tuple((r.original_id,) for r in rows[1:-1])
    folds = {r.original_id: 0 if i in (0, 5) else 1 for i, r in enumerate(rows)}
    excluded = buffer_exclusions(rows, folds, groups, 10)
    assert {rows[0].original_id, rows[-1].original_id} <= excluded


def test_whole_geography_deletion_returns_refit_frame():
    rows, *_ = split_fixture(10, counties=2)
    retained = deletion_refit_ids(rows, county="c0")
    assert retained == tuple(r.original_id for r in rows if r.county == "c1")
    with pytest.raises(ContractError):
        deletion_refit_ids(rows)


def test_original_and_shifted_pairs_share_split_and_origin_identity():
    v = make_inputs()
    rows = v["geography"].rows
    reservation = reserve_design(rows, v["entity_graph"])
    atlas = {r.tract_id: r for r in v["atlas"].rows}
    support = freeze_support(rows, reservation.design_ids, atlas, v["covariates"])
    chosen = [r for r in rows if r.original_id not in reservation.design_ids]
    folds = geographic_folds(chosen, reservation.groups, 5)
    pairs = paired_records(policy_for(chosen, atlas, support), [1] * len(chosen), weight_id="equal-tract")
    n = len(chosen)
    assert pairs.original_ids[:n] == pairs.original_ids[n:]
    assert tuple(folds[oid] for oid in pairs.original_ids[:n]) == tuple(folds[oid] for oid in pairs.original_ids[n:])



def test_twenty_five_km_buffer_can_preserve_a_supported_geographic_split():
    rows, graph, manifest, reservation = split_fixture(60)
    # Separated subblocks avoid all 25-km cross-fold boundaries. Every outer
    # partition still has ample tracts for all three buffered inner fits.
    rows = tuple(replace(r, longitude=-120 + i * .4) for i, r in enumerate(rows))
    result = build_scenario(rows, graph, manifest, reservation, manifest.original_ids, {},
                            radius_km=25, policy_id=digest("policy"))
    assert result.status == "pass"
    assert not result.exclusions
    validate_split(result.outer, result.data_manifest, graph)
    assert all(a.outer_training_tracts >= 8 and min(a.inner_fitting_tracts) >= 4
               for a in result.count_audit)
    for inner in result.inner:
        assert not inner.buffer_excluded_ids


def test_small_county_exclusion_does_not_remove_adequate_county():
    rows, graph, manifest, reservation = split_fixture(30, counties=2)
    # Exclude all but nine of the second county before split assignment.
    candidates = tuple(r.original_id for r in rows if r.county == "c0" or r.original_id < "o039")
    result = build_scenario(rows, graph, manifest, reservation, candidates, {},
                            radius_km=0, policy_id=digest("policy"))
    assert result.status == "pass"
    by_id = {r.original_id: r for r in rows}
    assert {by_id[oid].county for oid in result.outer.original_ids} == {"c0"}
    assert len(result.exclusions) == 30
