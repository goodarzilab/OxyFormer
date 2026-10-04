"""Immutable within-county geographic splits and explicit buffer targets.

For each radius, both sides of every cross-fold boundary within that radius
leave the frozen target (whole dependence components). This conservative,
common-target convention lets the authoritative SplitManifest.training_ids()
remain correct; no caller needs a hidden, fold-dependent exclusion overlay.
Inner splits use the same convention inside their permitted outer partition.
Whole-geography deletion is an analysis refit, never unseen-county prediction.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from math import ceil
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from oxyformer.contracts import DataManifest, SplitManifest
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import validate_split
from oxyformer.design.eligibility import county_screen
from oxyformer.design.support import recheck_support
from oxyformer.provenance import Immutable, canonical_json, require

SEEDS = (1103, 2207, 3301)


@dataclass(frozen=True, slots=True, kw_only=True)
class Reservation(Immutable):
    design_ids: tuple[str, ...]
    selected_subblocks: tuple[tuple[str, str], ...]
    sealed_subblocks: tuple[tuple[str, str], ...]
    total_subblocks: int
    groups: tuple[tuple[str, ...], ...]


def dependence_groups(rows, graph):
    ids = {r.original_id for r in rows}
    require(ids == set(graph.original_ids), "geography/entity graph mismatch")
    parent = {oid: oid for oid in ids}

    def root(oid):
        while parent[oid] != oid:
            parent[oid] = parent[parent[oid]]
            oid = parent[oid]
        return oid

    def join(group):
        for oid in group[1:]:
            parent[root(oid)] = root(group[0])

    for group in graph.components():
        join(group)
    for key in (lambda r: (r.county, r.subblock), lambda r: r.assignment_geography,
                lambda r: r.tract_id):
        groups = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row.original_id)
        for group in groups.values():
            join(group)
    groups = {}
    for oid in sorted(ids):
        groups.setdefault(root(oid), []).append(oid)
    return tuple(sorted(tuple(group) for group in groups.values()))


def close_groups(ids, groups):
    ids = set(ids)
    return {oid for group in groups if ids.intersection(group) for oid in group}


def reserve_design(rows, graph):
    """Select 20% per county (round up), then seal the complete link closure.

    Rounding and additional subblocks sealed by dependence links are disclosed.
    Selection hashes only county/subblock IDs and the prespecified seed.
    """
    groups = dependence_groups(rows, graph)
    blocks = {}
    for row in rows:
        blocks.setdefault(row.county, set()).add(row.subblock)
    selected = set()
    for county, names in sorted(blocks.items()):
        ordered = sorted(names, key=lambda name: sha256(
            canonical_json([SEEDS[0], county, name]).encode()).hexdigest())
        selected.update((county, name) for name in ordered[:ceil(.2 * len(names))])
    ids = close_groups({r.original_id for r in rows if (r.county, r.subblock) in selected}, groups)
    sealed = {(r.county, r.subblock) for r in rows if r.original_id in ids}
    return Reservation(design_ids=tuple(sorted(ids)), selected_subblocks=tuple(sorted(selected)),
                       sealed_subblocks=tuple(sorted(sealed)),
                       total_subblocks=sum(map(len, blocks.values())), groups=groups)


def geographic_folds(rows, groups, count):
    """Contiguous latitude/longitude ordered components, grouped within counties.

    Cross-county components have one anchor county and one fold. The subsequent
    per-county checks reject any resulting unsupported county; links never split.
    """
    by_id = {r.original_id: r for r in rows}
    counties = {}
    for group in groups:
        present = tuple(oid for oid in group if oid in by_id)
        if not present:
            continue
        require(len(present) == len(group), "partial dependence component")
        county = min(by_id[oid].county for oid in present)
        counties.setdefault(county, []).append(present)
    folds = {}
    for county, members in sorted(counties.items()):
        members.sort(key=lambda g: (sum(by_id[oid].latitude for oid in g) / len(g),
                                   sum(by_id[oid].longitude for oid in g) / len(g), g))
        for index, group in enumerate(members):
            fold = min(count - 1, index * count // len(members))
            folds.update((oid, fold) for oid in group)
    return folds


def buffer_exclusions(rows, folds, groups, radius_km):
    require(radius_km in (0, 10, 25), "unapproved buffer radius")
    if radius_km == 0 or len(rows) < 2:
        return set()
    radians = np.radians([(r.latitude, r.longitude) for r in rows])
    lat, lon = radians[:, 0], radians[:, 1]
    xyz = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(radius_km / (2 * 6371.0088)))
    remove = set()
    for i, j in pairs:
        if folds[rows[i].original_id] != folds[rows[j].original_id]:
            remove.update((rows[i].original_id, rows[j].original_id))
    return close_groups(remove, groups)


def tract_count(rows, county):
    return len({r.tract_id for r in rows if r.county == county
                and r.outcome_flag == 1 and r.label_available})


@dataclass(frozen=True, slots=True, kw_only=True)
class CountAudit(Immutable):
    county: str
    outer_fold: int
    outer_training_tracts: int
    inner_fitting_tracts: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class InnerSplit(Immutable):
    outer_fold: int
    data_manifest: DataManifest
    entity_graph: EntityGraph
    split: SplitManifest
    buffer_excluded_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitScenario(Immutable):
    buffer_km: float
    status: Literal["pass", "fail"]
    target_id: str
    data_manifest: DataManifest
    outer: SplitManifest | None
    inner: tuple[InnerSplit, ...]
    exclusions: tuple[tuple[str, str], ...]
    count_audit: tuple[CountAudit, ...]


def _manifest(manifest, ids, graph):
    return replace(manifest, original_ids=tuple(ids), entity_graph_hash=graph.content_hash,
                   lineage=replace(manifest.lineage, unit_ids=tuple(ids),
                                   parent_hashes=(manifest.content_hash,)))


def _split(manifest, ids, folds, design_ids, excluded_ids, *, level):
    split = SplitManifest(spec=manifest.spec, level=level, original_ids=tuple(ids),
                          fold_ids=tuple(folds[oid] for oid in ids), design_ids=tuple(design_ids),
                          excluded_ids=tuple(excluded_ids), seed_ids=SEEDS,
                          entity_graph_hash=manifest.entity_graph_hash,
                          lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))
    return split


def build_scenario(rows, graph, manifest, reservation, candidate_ids, initial_reasons,
                   *, radius_km, policy_id, atlas=None, covariates=None, support=None):
    """Freeze eligibility to a fixed point before any fitting is possible.

    Passing a frozen support record enables the required conditional rechecks.
    The low-level split-only API can be exercised without an exposure model;
    run_stage always supplies support, atlas and covariates.
    """
    require(radius_km in (0, 10, 25), "unapproved buffer radius")
    by_id = {r.original_id: r for r in rows}
    all_ids, design_ids = set(by_id), set(reservation.design_ids)
    require(set(manifest.original_ids) == all_ids, "manifest/geography mismatch")
    require(not design_ids.intersection(candidate_ids), "design labels in target")
    require(set(candidate_ids) <= all_ids, "unknown target ID")
    groups = reservation.groups
    reasons = dict(initial_reasons)
    excluded = close_groups(all_ids - set(candidate_ids) - design_ids, groups) - design_ids
    for oid in excluded:
        reasons.setdefault(oid, "ineligible_or_linked_to_ineligible")
    active = all_ids - design_ids - excluded
    folds = geographic_folds([by_id[oid] for oid in sorted(active)], groups, 5)
    buffered = buffer_exclusions([by_id[oid] for oid in sorted(active)], folds, groups, radius_km)
    for oid in buffered:
        reasons[oid] = "outer_buffer"
    active -= buffered
    audits, inner_plans = [], []
    while active:
        current = [by_id[oid] for oid in sorted(active)]
        counties = sorted({r.county for r in current})
        failures = {}
        unsupported = {}
        audits, inner_plans = [], []
        if atlas is not None:
            for county, screen in county_screen(current, atlas).items():
                if screen["reasons"]:
                    failures[county] = "post_exclusion_geographic_screen"
        for county in counties:
            if {folds[r.original_id] for r in current if r.county == county} != set(range(5)):
                failures[county] = "missing_within_county_outer_fold"
        for outer in range(5):
            training = [r for r in current if folds[r.original_id] != outer]
            evaluation = [r for r in current if folds[r.original_id] == outer]
            train_ids = {r.original_id for r in training}
            inner_folds = geographic_folds(training, groups, 3)
            inner_buffer = buffer_exclusions(training, inner_folds, groups, radius_km)
            inner_rows = [r for r in training if r.original_id not in inner_buffer]
            for county in counties:
                n_outer = tract_count(training, county)
                n_inner = tuple(tract_count([r for r in inner_rows
                                            if inner_folds[r.original_id] != fold], county)
                                for fold in range(3))
                audits.append(CountAudit(county=county, outer_fold=outer,
                                         outer_training_tracts=n_outer, inner_fitting_tracts=n_inner))
                if n_outer < 8:  # Mutation acceptance test must detect bypass of this gate.
                    failures[county] = "fewer_than_eight_outer_training_tracts"
                elif min(n_inner) < 4:
                    failures[county] = "fewer_than_four_inner_fitting_tracts"
                elif {inner_folds[r.original_id] for r in inner_rows if r.county == county} != {0, 1, 2}:
                    failures[county] = "missing_within_county_inner_fold"
            if support is not None:
                failed = recheck_support(evaluation, training, atlas, covariates, support)
                for oid in failed:
                    unsupported.setdefault(oid, "outer_conditional_support")
                for fold in range(3):
                    held = [r for r in inner_rows if inner_folds[r.original_id] == fold]
                    fit = [r for r in inner_rows if inner_folds[r.original_id] != fold]
                    for oid in recheck_support(held, fit, atlas, covariates, support):
                        unsupported.setdefault(oid, "inner_conditional_support")
            inner_plans.append((outer, train_ids, inner_folds, inner_buffer))
        if not failures and not unsupported:
            break
        # A row-level support failure is not a county-level failure. Exclude
        # its complete geographic/dependence component, then recompute all
        # county minima and conditional support on the remaining fixed target.
        removed = close_groups({r.original_id for r in current if r.county in failures}
                               | set(unsupported), groups)
        for oid in removed:
            reason = failures.get(by_id[oid].county, unsupported.get(oid, "linked_component_exclusion"))
            reasons.setdefault(oid, reason)
        active -= removed
    excluded = all_ids - active - design_ids
    for oid in excluded:
        reasons.setdefault(oid, "ineligible_or_linked_to_ineligible")
    target_id = sha256(canonical_json({"ids": sorted(active), "design_ids": sorted(design_ids),
                                      "buffer_km": float(radius_km), "policy": policy_id,
                                      "folds": sorted((oid, folds[oid]) for oid in active)}).encode()).hexdigest()
    final_manifest = replace(manifest, spec=replace(manifest.spec, target_id=target_id, policy_id=policy_id),
                             lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))
    outer_split, inners = None, []
    if active:
        outer_split = _split(final_manifest, sorted(active), folds, sorted(design_ids), sorted(excluded), level="outer")
        validate_split(outer_split, final_manifest, graph)
        for outer, train_ids, inner_folds, inner_buffer in inner_plans:
            inner_graph = EntityGraph(original_ids=tuple(sorted(train_ids)),
                                      links=tuple(link for link in graph.links if link.observation_id in train_ids))
            inner_manifest = _manifest(final_manifest, sorted(train_ids), inner_graph)
            split = _split(inner_manifest, sorted(train_ids - inner_buffer), inner_folds, (),
                           sorted(inner_buffer), level="inner")
            validate_split(split, inner_manifest, inner_graph)
            inners.append(InnerSplit(outer_fold=outer, data_manifest=inner_manifest,
                                     entity_graph=inner_graph, split=split,
                                     buffer_excluded_ids=tuple(sorted(inner_buffer))))
    return SplitScenario(buffer_km=float(radius_km), status="pass" if active else "fail", target_id=target_id,
                         data_manifest=final_manifest, outer=outer_split, inner=tuple(inners),
                         exclusions=tuple(sorted((oid, reasons[oid]) for oid in excluded)),
                         count_audit=tuple(audits))


def deletion_refit_ids(rows, *, county=None, state=None):
    """Return the frame for a new full fit; no prediction or county offsets."""
    require((county is None) != (state is None), "select exactly one deletion geography")
    return tuple(r.original_id for r in rows if (r.county != county if county is not None else r.state != state))
