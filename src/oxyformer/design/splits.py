"""Immutable within-county geographic splits and explicit buffer targets.

For each radius, both sides of every cross-fold boundary within that radius
leave the frame (whole dependence components). Membership roles keep eligible
non-target rows available for fitting and leave other frame rows inert. Explicit
training_exclusions record the inner buffers for each outer refit.
Inner splits use the same convention inside their permitted outer partition.
Whole-geography deletion is an analysis refit, never unseen-county prediction.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from math import ceil
from typing import Literal

import json

import numpy as np
from scipy.spatial import cKDTree

from oxyformer.contracts import DataManifest, SplitManifest
from oxyformer.data.adapters.usaleep import primary_outcome_flags
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.data.loaders import validate_split
from oxyformer.design.eligibility import county_screen, usable
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
    flags = primary_outcome_flags()
    return len({r.tract_id for r in rows if r.county == county
                and r.outcome_flag in flags and r.label_available})


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
    target_hash: str = ""
    layer_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    estimand_id: str = "usaleep-a2-2026-10-10"
    outer_buffer_excluded_ids: tuple[str, ...] = ()



@dataclass(frozen=True, slots=True, kw_only=True)
class SplitMembership(Immutable):
    """Outcome-blind fold decisions; source provenance is attached at binding."""
    level: Literal["outer", "inner"]
    original_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    design_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    seed_ids: tuple[int, ...]
    entity_graph_hash: str
    frame_ids: tuple[str, ...]
    frame_fold_ids: tuple[int, ...]
    membership_roles: tuple[Literal["target", "training", "inert"], ...]
    training_exclusions: tuple[tuple[int, tuple[str, ...]], ...]

    @classmethod
    def from_split(cls, split):
        payload = split.to_dict()["payload"]
        return cls(**{k: v for k, v in payload.items() if k not in ("spec", "lineage")})

    def bind(self, manifest, graph):
        split = SplitManifest(**self.to_dict()["payload"], spec=manifest.spec,
                              lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))
        validate_split(split, manifest, graph)
        return split


@dataclass(frozen=True, slots=True, kw_only=True)
class InnerMembership(Immutable):
    outer_fold: int
    frame_ids: tuple[str, ...]
    split: SplitMembership
    buffer_excluded_ids: tuple[str, ...]


def _binding_hash(manifest):
    # Scientific decisions cannot hash an outcome-bearing source payload. The
    # stage receipt binds those exact sources; binding below restores them to
    # every fitting manifest, without discarding or inventing source lineage.
    return sha256(canonical_json({
        "schema": manifest.schema_hash, "registry": manifest.registry.content_hash,
        "ids": manifest.original_ids, "graph": manifest.entity_graph_hash,
        "endpoint": manifest.spec.endpoint, "scale": manifest.spec.outcome_scale,
        "weight": manifest.spec.weight_id, "inference_unit": manifest.spec.inference_unit,
        "fields": [manifest.id_field, manifest.outcome_field, manifest.exposure_field, manifest.weight_field],
    }).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitScenarioDesign(Immutable):
    """Published design decisions, independent of Y and its payload hashes.

    Verify the stage receipt against its request before reading this record,
    then bind to that request's DataManifest and EntityGraph before fitting.
    The resulting SplitScenario retains the full authoritative provenance.
    """
    binding_hash: str
    buffer_km: float
    status: Literal["pass", "fail"]
    target_id: str
    policy_id: str
    outer: SplitMembership | None
    inner: tuple[InnerMembership, ...]
    exclusions: tuple[tuple[str, str], ...]
    count_audit: tuple[CountAudit, ...]
    target_hash: str
    layer_ids: tuple[tuple[str, tuple[str, ...]], ...]
    estimand_id: str
    outer_buffer_excluded_ids: tuple[str, ...] = ()

    @classmethod
    def from_scenario(cls, scenario):
        return cls(binding_hash=_binding_hash(scenario.data_manifest), buffer_km=scenario.buffer_km,
                   status=scenario.status, target_id=scenario.target_id,
                   policy_id=scenario.data_manifest.spec.policy_id,
                   outer=SplitMembership.from_split(scenario.outer) if scenario.outer else None,
                   inner=tuple(InnerMembership(outer_fold=i.outer_fold, frame_ids=i.data_manifest.original_ids,
                               split=SplitMembership.from_split(i.split), buffer_excluded_ids=i.buffer_excluded_ids)
                               for i in scenario.inner), exclusions=scenario.exclusions, count_audit=scenario.count_audit,
                   target_hash=scenario.target_hash, layer_ids=scenario.layer_ids, estimand_id=scenario.estimand_id,
                   outer_buffer_excluded_ids=scenario.outer_buffer_excluded_ids)

    def bind(self, manifest, graph):
        require(_binding_hash(manifest) == self.binding_hash, "split design/data binding mismatch")
        require(graph.content_hash == manifest.entity_graph_hash, "split design/entity graph mismatch")
        final = replace(manifest, spec=replace(manifest.spec, target_id=self.target_id, policy_id=self.policy_id),
                        lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))
        outer = self.outer.bind(final, graph) if self.outer else None
        inners = []
        for i in self.inner:
            require(outer is not None and set(i.frame_ids) == set(outer.frame_training_ids(i.outer_fold)),
                    "inner design escapes outer training frame")
            ids = set(i.frame_ids)
            inner_graph = EntityGraph(original_ids=i.frame_ids,
                                      links=tuple(link for link in graph.links if link.observation_id in ids))
            inner_manifest = _manifest(final, i.frame_ids, inner_graph)
            inners.append(InnerSplit(outer_fold=i.outer_fold, data_manifest=inner_manifest, entity_graph=inner_graph,
                                    split=i.split.bind(inner_manifest, inner_graph),
                                    buffer_excluded_ids=i.buffer_excluded_ids))
        return SplitScenario(buffer_km=self.buffer_km, status=self.status, target_id=self.target_id,
                             data_manifest=final, outer=outer, inner=tuple(inners), exclusions=self.exclusions,
                             count_audit=self.count_audit, target_hash=self.target_hash,
                             layer_ids=self.layer_ids, estimand_id=self.estimand_id,
                             outer_buffer_excluded_ids=self.outer_buffer_excluded_ids)

def _manifest(manifest, ids, graph):
    return replace(manifest, original_ids=tuple(ids), entity_graph_hash=graph.content_hash,
                   lineage=replace(manifest.lineage, unit_ids=tuple(ids),
                                   parent_hashes=(manifest.content_hash,)))


def _split(manifest, ids, folds, design_ids, excluded_ids, *, level, targets, labelled,
           training_exclusions=()):
    target_ids = tuple(i for i in ids if i in targets)
    return SplitManifest(spec=manifest.spec, level=level, original_ids=target_ids,
                         fold_ids=tuple(folds[oid] for oid in target_ids),
                         frame_ids=tuple(ids), frame_fold_ids=tuple(folds[oid] for oid in ids),
                         design_ids=tuple(design_ids),
                         excluded_ids=tuple(excluded_ids), seed_ids=SEEDS,
                         membership_roles=tuple("target" if i in targets else
                                                "training" if i in labelled else "inert" for i in ids),
                         training_exclusions=tuple(training_exclusions),
                         entity_graph_hash=manifest.entity_graph_hash,
                         lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))


def build_scenario(rows, graph, manifest, reservation, candidate_ids, initial_reasons,
                   *, radius_km, policy_id, atlas=None, covariates=None, support=None):
    """A2: frame-atomic folds and buffers, tract-level target exclusions.

    Inner geography is fixed on the whole outer-training frame before label
    selection. The same labelled rows, minus inner buffers, supply each outer
    nuisance fit, its eight-tract audit and its conditional support recheck.
    """
    require(radius_km in (0, 10, 25), "unapproved buffer radius")
    by_id = {r.original_id: r for r in rows}
    all_ids, design_ids = set(by_id), set(reservation.design_ids)
    require(set(manifest.original_ids) == all_ids, "manifest/geography mismatch")
    require(not design_ids.intersection(candidate_ids), "design labels in target")
    require(set(candidate_ids) <= all_ids, "unknown target ID")
    groups = reservation.groups
    reasons = dict(initial_reasons)
    frame = [r for r in rows if r.original_id not in design_ids]
    folds = geographic_folds(frame, groups, 5)
    buffered = buffer_exclusions(frame, folds, groups, radius_km)
    frame = [r for r in frame if r.original_id not in buffered]
    frame_ids = {r.original_id for r in frame}
    flags = primary_outcome_flags()
    labelled = {r.original_id for r in frame if (usable(r, atlas) if atlas is not None else
                r.outcome_flag in flags and r.label_available)}
    active = set(candidate_ids) & labelled - buffered
    for oid in buffered:
        reasons.setdefault(oid, "outer_buffer")
    for oid in all_ids - design_ids - set(candidate_ids):
        reasons.setdefault(oid, "initial_target_exclusion")
    for oid in set(candidate_ids) - labelled - buffered:
        reasons.setdefault(oid, "flag_label_or_allocation_unusable")
    layers = [("buffers", tuple(sorted(active)))]
    inner_plans = []
    for outer in range(5):
        training_frame = [r for r in frame if folds[r.original_id] != outer]
        inner_folds = geographic_folds(training_frame, groups, 3)
        inner_buffer = buffer_exclusions(training_frame, inner_folds, groups, radius_km)
        inner_plans.append((outer, {r.original_id for r in training_frame}, inner_folds, inner_buffer))
    audits = []
    minima_removed, recheck_removed = set(), set()
    while active:
        counties = sorted({by_id[i].county for i in active})
        fitting = [r for r in frame if r.original_id in labelled and r.county in counties]
        failures, unsupported, audits = {}, {}, []
        if atlas is not None:
            screens = county_screen(fitting, atlas)
            for county in counties:
                if county not in screens or screens[county]["reasons"]:
                    failures[county] = "post_exclusion_geographic_screen"
        for county in counties:
            if {folds[r.original_id] for r in fitting if r.county == county} != set(range(5)):
                failures[county] = "missing_within_county_outer_fold"
        for outer, _, inner_folds, inner_buffer in inner_plans:
            training = [r for r in fitting if folds[r.original_id] != outer
                        and r.original_id not in inner_buffer]
            evaluation = [by_id[i] for i in sorted(active) if folds[i] == outer]
            for county in counties:
                n_outer = tract_count(training, county)
                n_inner = tuple(tract_count([r for r in training
                                            if inner_folds[r.original_id] != fold], county)
                                for fold in range(3))
                audits.append(CountAudit(county=county, outer_fold=outer,
                                         outer_training_tracts=n_outer, inner_fitting_tracts=n_inner))
                if n_outer < 8:  # Mutation acceptance test must detect bypass of this gate.
                    failures[county] = "fewer_than_eight_outer_training_tracts"
                elif min(n_inner) < 4:
                    failures[county] = "fewer_than_four_inner_fitting_tracts"
                elif {inner_folds[r.original_id] for r in training if r.county == county} != {0, 1, 2}:
                    failures[county] = "missing_within_county_inner_fold"
            if support is not None:
                for oid in recheck_support(evaluation, training, atlas, covariates, support):
                    unsupported.setdefault(oid, "outer_conditional_support")
                for fold in range(3):
                    held = [r for r in training if r.original_id in active and inner_folds[r.original_id] == fold]
                    fit = [r for r in training if inner_folds[r.original_id] != fold]
                    for oid in recheck_support(held, fit, atlas, covariates, support):
                        unsupported.setdefault(oid, "inner_conditional_support")
        removed_minima = {i for i in active if by_id[i].county in failures}
        removed_support = active.intersection(unsupported)
        if not removed_minima and not removed_support:
            break
        minima_removed |= removed_minima
        recheck_removed |= removed_support
        for oid in removed_minima | removed_support:
            reasons.setdefault(oid, failures.get(by_id[oid].county, unsupported.get(oid)))
        active -= removed_minima | removed_support
    layers.extend((("fold_minima", tuple(sorted(set(layers[0][1]) - minima_removed))),
                   ("recheck", tuple(sorted(active)))))
    excluded = all_ids - active - design_ids
    for oid in excluded:
        reasons.setdefault(oid, "initial_target_exclusion")
    estimand_id = "usaleep-a2-2026-10-10"
    target_id = sha256(canonical_json({"estimand": estimand_id, "ids": sorted(active),
                                      "design_ids": sorted(design_ids), "buffer_km": float(radius_km),
                                      "policy": policy_id, "folds": sorted(folds.items())}).encode()).hexdigest()
    final_manifest = replace(manifest, spec=replace(manifest.spec, target_id=target_id, policy_id=policy_id),
                             lineage=replace(manifest.lineage, parent_hashes=(manifest.content_hash,)))
    outer_split, inners = None, []
    if active:
        outer_split = _split(final_manifest, sorted(frame_ids), folds, sorted(design_ids), sorted(buffered),
                             level="outer", targets=active, labelled=labelled,
                             training_exclusions=tuple((o, tuple(sorted(b))) for o, _, _, b in inner_plans))
        validate_split(outer_split, final_manifest, graph)
        for outer, train_ids, inner_folds, inner_buffer in inner_plans:
            inner_graph = EntityGraph(original_ids=tuple(sorted(train_ids)),
                                      links=tuple(link for link in graph.links if link.observation_id in train_ids))
            inner_manifest = _manifest(final_manifest, sorted(train_ids), inner_graph)
            split = _split(inner_manifest, sorted(train_ids - inner_buffer), inner_folds, (),
                           sorted(inner_buffer), level="inner", targets=active, labelled=labelled)
            validate_split(split, inner_manifest, inner_graph)
            inners.append(InnerSplit(outer_fold=outer, data_manifest=inner_manifest,
                                     entity_graph=inner_graph, split=split,
                                     buffer_excluded_ids=tuple(sorted(inner_buffer))))
    return SplitScenario(buffer_km=float(radius_km), status="pass" if active else "fail", target_id=target_id,
                         data_manifest=final_manifest, outer=outer_split, inner=tuple(inners),
                         exclusions=tuple(sorted((oid, reasons[oid]) for oid in excluded)),
                         count_audit=tuple(audits), layer_ids=tuple(layers), estimand_id=estimand_id,
                         outer_buffer_excluded_ids=tuple(sorted(buffered)),
                         target_hash=sha256(json.dumps(sorted(active)).encode()).hexdigest()[:16])


def deletion_refit_ids(rows, *, county=None, state=None):
    """Return the frame for a new full fit; no prediction or county offsets."""
    require((county is None) != (state is None), "select exactly one deletion geography")
    return tuple(r.original_id for r in rows if (r.county != county if county is not None else r.state != state))
