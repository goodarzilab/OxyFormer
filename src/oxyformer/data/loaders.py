"""Local adapter boundaries. No source fields, downloads or inference are guessed."""
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from oxyformer.contracts import (
    Cell, CovariateView, DataManifest, EstimandSpec, OOFNuisances, SplitManifest,
)
from oxyformer.data.entity_graph import EntityGraph
from oxyformer.provenance import Immutable, require


def validate_manifest(manifest: DataManifest, expected: EstimandSpec, expected_schema_hash: str) -> None:
    expected.assert_compatible(manifest.spec)
    require(manifest.schema_hash == expected_schema_hash, "data schema mismatch")
    for source in manifest.sources:
        source.assert_usable()


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedData(Immutable):
    """Privileged score/training-side data; never pass this object to prediction."""
    manifest: DataManifest
    rows: tuple[tuple[Cell, ...], ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        validate_manifest(self.manifest, self.manifest.spec, self.manifest.schema_hash)
        schema = self.manifest.schema
        require(len(self.rows) == len(self.manifest.original_ids), "record count mismatch")
        require(all(len(row) == len(schema) for row in self.rows), "record schema mismatch")
        for row in self.rows:
            for column, value in zip(schema, row):
                if value is None:
                    require(column.nullable, f"null {column.name}")
                    continue
                types = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}
                require(type(value) in types[column.dtype], f"dtype mismatch: {column.name}")
        id_index = tuple(c.name for c in schema).index(self.manifest.id_field)
        require(tuple(row[id_index] for row in self.rows) == self.manifest.original_ids,
                "original-observation order mismatch")

    def column(self, name: str) -> tuple[Cell, ...]:
        names = tuple(c.name for c in self.manifest.schema)
        require(name in names, f"unknown column: {name}")
        return tuple(row[names.index(name)] for row in self.rows)

    def covariates(self, columns: Sequence[str], use="nuisance") -> CovariateView:
        names = tuple(c.name for c in self.manifest.schema)
        for name in columns:
            self.manifest.registry.require(name, self.manifest.spec.endpoint, use)
            require(name in names, f"unknown column: {name}")
        indices = tuple(names.index(name) for name in columns)
        lineage = replace(self.manifest.lineage,
                          parent_hashes=self.manifest.lineage.parent_hashes + (self.manifest.content_hash,))
        return CovariateView(spec=self.manifest.spec, registry=self.manifest.registry,
                             original_ids=self.manifest.original_ids, columns=tuple(columns),
                             values=tuple(tuple(row[i] for i in indices) for row in self.rows),
                             use=use, lineage=lineage)

    def county_routing(self, name: str) -> tuple[Cell, ...]:
        self.manifest.registry.require(name, self.manifest.spec.endpoint, "county_routing")
        return self.column(name)


def load_records(records: Sequence[Mapping[str, Cell]], manifest: DataManifest,
                 expected_spec: EstimandSpec, expected_schema_hash: str) -> LoadedData:
    validate_manifest(manifest, expected_spec, expected_schema_hash)
    names = tuple(column.name for column in manifest.schema)
    rows = []
    for record in records:
        require(set(record) == set(names), "record schema mismatch")
        rows.append(tuple(record[name] for name in names))
    return LoadedData(manifest=manifest, rows=tuple(rows))


def _validate_split_identity(split: SplitManifest, manifest: DataManifest) -> None:
    manifest.spec.assert_compatible(split.spec)
    require(manifest.entity_graph_hash == split.entity_graph_hash, "entity graph hash mismatch")
    require(set(manifest.original_ids) == set(split.lineage.unit_ids),
            "split does not cover original observations")
    require(manifest.content_hash in split.lineage.parent_hashes, "missing data parent")
    require(set(split.lineage.source_hashes) == set(manifest.lineage.source_hashes), "split source mismatch")


def validate_split(split: SplitManifest, manifest: DataManifest, graph: EntityGraph) -> None:
    _validate_split_identity(split, manifest)
    require(manifest.entity_graph_hash == graph.content_hash, "entity graph hash mismatch")
    require(set(manifest.original_ids) == set(graph.original_ids), "entity graph ID mismatch")
    assignments = dict(zip(split.original_ids, split.fold_ids))
    assignments.update({x: "design" for x in split.design_ids})
    assignments.update({x: "excluded" for x in split.excluded_ids})
    graph.assert_partition(assignments)


def validate_oof(nuisances: OOFNuisances, data: LoadedData, split: SplitManifest) -> None:
    manifest = data.manifest
    manifest.spec.assert_compatible(nuisances.spec)
    _validate_split_identity(split, manifest)
    require(nuisances.lineage.split_hash == split.content_hash, "OOF split mismatch")
    require(manifest.content_hash in nuisances.lineage.parent_hashes, "missing data parent")
    require(set(nuisances.lineage.source_hashes) == set(manifest.lineage.source_hashes), "OOF source mismatch")
    expected = {(x, seed) for x in split.original_ids for seed in split.seed_ids}
    require(set(zip(nuisances.original_ids, nuisances.seed_ids)) == expected, "incomplete original ID/seed coverage")
    folds = dict(zip(split.original_ids, split.fold_ids))
    for oid, fold in zip(nuisances.original_ids, nuisances.fold_ids):
        require(folds[oid] == fold, "held-out fold mismatch")
    if manifest.weight_field is None:
        weights = dict.fromkeys(manifest.original_ids, 1.0)
    else:
        weights = dict(zip(manifest.original_ids, data.column(manifest.weight_field)))
    for oid, weight in zip(nuisances.original_ids, nuisances.origin_weights):
        require(weights[oid] == weight, "origin weight mismatch")


def join_outcomes(nuisances: OOFNuisances, data: LoadedData, split: SplitManifest,
                  expected_spec: EstimandSpec) -> tuple[Cell, ...]:
    """Score/targeting-only operation, aligned by original ID, never row position."""
    expected_spec.assert_compatible(data.manifest.spec)
    validate_oof(nuisances, data, split)
    outcomes = dict(zip(data.manifest.original_ids, data.column(data.manifest.outcome_field)))
    require(all(outcomes[oid] is not None for oid in nuisances.original_ids), "missing held-out outcome")
    return tuple(outcomes[oid] for oid in nuisances.original_ids)
