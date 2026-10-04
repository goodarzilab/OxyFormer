"""Proposed internal APIs. These contracts do not assert scientific eligibility."""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
import re
from typing import Literal, Protocol

from oxyformer.data.feature_roles import FeatureRegistry
from oxyformer.provenance import (
    ArtifactLineage, ArtifactRecord, Immutable, canonical_json, check_hash,
    file_hash, nonempty, require, unique,
)

Cell = str | int | float | bool | None


def _ids(ids: tuple[str, ...], name: str = "original IDs") -> None:
    require(bool(ids), f"empty {name}")
    unique(ids, name)
    for value in ids:
        nonempty(value, name)


def _aligned(lineage: ArtifactLineage, ids: tuple[str, ...]) -> None:
    require(set(lineage.unit_ids) == set(ids), "lineage unit IDs do not align")


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimandSpec(Immutable):
    endpoint: str
    target_id: str
    outcome_scale: str
    policy_id: str
    weight_id: str
    adjustment_schema_hash: str
    inference_unit: str
    source_lineage_hash: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        for field in fields(self):
            nonempty(getattr(self, field.name), field.name)
        check_hash(self.adjustment_schema_hash, "adjustment schema hash")
        check_hash(self.source_lineage_hash, "source lineage hash")

    def assert_compatible(self, other: EstimandSpec) -> None:
        require(type(other) is EstimandSpec, "expected EstimandSpec")
        require(self.policy_id == other.policy_id, "policy_id mismatch")
        for field in fields(self):
            if field.name != "policy_id":
                require(getattr(self, field.name) == getattr(other, field.name),
                        f"{field.name} mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceManifest(Immutable):
    source_id: str
    version: str
    uri: str
    payload_hash: str
    license_hash: str
    schema_hash: str
    field_mapping: tuple[tuple[str, str], ...]
    mapping_status: Literal["unreviewed", "reviewed"]
    mapping_review_id: str | None

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.source_id, self.version, self.uri):
            nonempty(value, "source identity")
        for value in (self.payload_hash, self.license_hash, self.schema_hash):
            check_hash(value)
        unique(tuple(key for key, _ in self.field_mapping), "source mapping keys")
        unique(tuple(value for _, value in self.field_mapping), "source mapping destinations")
        for pair in self.field_mapping:
            for value in pair:
                nonempty(value, "mapping field")
        object.__setattr__(self, "field_mapping", tuple(sorted(self.field_mapping)))
        if self.mapping_status == "reviewed":
            require(bool(self.field_mapping) and bool(self.mapping_review_id and self.mapping_review_id.strip()),
                    "reviewed mapping requires fields and review identity")

    def assert_usable(self) -> None:
        require(self.mapping_status == "reviewed", f"unreviewed source mapping: {self.source_id}")


def source_lineage_hash(sources: tuple[SourceManifest, ...]) -> str:
    """Bind source bytes, mapping, licence, schema and review provenance."""
    values = sorted(source.content_hash for source in sources)
    return sha256(canonical_json(values).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnSpec(Immutable):
    name: str
    dtype: Literal["string", "integer", "number", "boolean"]
    nullable: bool = False

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.name, "column name")


def schema_hash(schema: tuple[ColumnSpec, ...]) -> str:
    return sha256(canonical_json([c.to_dict() for c in schema]).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class DataManifest(Immutable):
    spec: EstimandSpec
    sources: tuple[SourceManifest, ...]
    schema: tuple[ColumnSpec, ...]
    registry: FeatureRegistry
    original_ids: tuple[str, ...]
    id_field: str
    outcome_field: str
    exposure_field: str
    weight_field: str | None
    entity_graph_hash: str
    lineage: ArtifactLineage

    def __post_init__(self):
        Immutable.__post_init__(self)
        _ids(self.original_ids)
        _aligned(self.lineage, self.original_ids)
        require(bool(self.sources), "sources required")
        unique(tuple(s.source_id for s in self.sources), "source IDs")
        check_hash(self.entity_graph_hash)
        require(source_lineage_hash(self.sources) == self.spec.source_lineage_hash, "source lineage mismatch")
        require(set(self.lineage.source_hashes) == {s.payload_hash for s in self.sources}, "source hashes mismatch")
        require(self.registry.content_hash == self.spec.adjustment_schema_hash, "adjustment schema mismatch")
        names = tuple(c.name for c in self.schema)
        unique(names, "schema columns")
        special = (self.id_field, self.outcome_field, self.exposure_field)
        if self.weight_field is not None:
            special += (self.weight_field,)
        unique(special, "data roles")
        require(set(special) <= set(names), "missing data role columns")
        roles = {rule.name: rule.role for rule in self.registry.rules}
        require(roles.get(self.id_field) == "identifier", "ID field role mismatch")
        require(roles.get(self.outcome_field) == "outcome", "outcome field role mismatch")
        require(roles.get(self.exposure_field) == "exposure", "exposure field role mismatch")
        # No ordinary predictor may alias a protected data role, including weights.
        require(self.weight_field is None or roles.get(self.weight_field) != "predictor",
                "weights cannot be ordinary predictors")
        require(next(c for c in self.schema if c.name == self.id_field).dtype == "string",
                "original ID column must be string")

    @property
    def schema_hash(self) -> str:
        return schema_hash(self.schema)


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitManifest(Immutable):
    spec: EstimandSpec
    level: Literal["outer", "inner"]
    original_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    design_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    seed_ids: tuple[int, ...]
    entity_graph_hash: str
    lineage: ArtifactLineage

    def __post_init__(self):
        Immutable.__post_init__(self)
        _ids(self.original_ids)
        require(len(self.fold_ids) == len(self.original_ids), "fold alignment mismatch")
        require(all(x >= 0 for x in self.fold_ids), "negative fold")
        require(len(set(self.fold_ids)) >= 2, "cross-fitting requires multiple folds")
        require(bool(self.seed_ids), "seeds required")
        unique(self.seed_ids, "seeds")
        require(all(x >= 0 for x in self.seed_ids), "negative seed")
        all_ids = self.original_ids + self.design_ids + self.excluded_ids
        _ids(all_ids, "split IDs")
        _aligned(self.lineage, all_ids)
        check_hash(self.entity_graph_hash)

    def training_ids(self, fold: int) -> tuple[str, ...]:
        require(fold in self.fold_ids, "unknown fold")
        return tuple(x for x, k in zip(self.original_ids, self.fold_ids) if k != fold)


@dataclass(frozen=True, slots=True, kw_only=True)
class CovariateView(Immutable):
    spec: EstimandSpec
    registry: FeatureRegistry
    original_ids: tuple[str, ...]
    columns: tuple[str, ...]
    values: tuple[tuple[Cell, ...], ...]
    use: Literal["nuisance", "ssl", "context"]
    lineage: ArtifactLineage

    def __post_init__(self):
        Immutable.__post_init__(self)
        _ids(self.original_ids)
        _aligned(self.lineage, self.original_ids)
        unique(self.columns, "covariate columns")
        require(self.registry.content_hash == self.spec.adjustment_schema_hash, "adjustment schema mismatch")
        for name in self.columns:
            self.registry.require(name, self.spec.endpoint, self.use)
        require(len(self.values) == len(self.original_ids), "covariate row alignment mismatch")
        require(all(len(row) == len(self.columns) for row in self.values), "covariate width mismatch")

    def column(self, name: str) -> tuple[Cell, ...]:
        # Permission check precedes lookup, including access to absent labels.
        self.registry.require(name, self.spec.endpoint, self.use)
        require(name in self.columns, f"column absent: {name}")
        index = self.columns.index(name)
        return tuple(row[index] for row in self.values)

    def __getitem__(self, name: str) -> tuple[Cell, ...]:
        return self.column(name)


@dataclass(frozen=True, slots=True, kw_only=True)
class OOFNuisances(Immutable):
    spec: EstimandSpec
    original_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    seed_ids: tuple[int, ...]
    mu_a: tuple[float, ...]
    mu_d: tuple[float, ...]
    r_a: tuple[float, ...]
    r_d: tuple[float, ...]
    origin_weights: tuple[float, ...]
    lineage: ArtifactLineage

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.original_ids), "empty OOF predictions")
        _aligned(self.lineage, self.original_ids)
        vectors = (self.fold_ids, self.seed_ids, self.mu_a, self.mu_d,
                   self.r_a, self.r_d, self.origin_weights)
        require(all(len(x) == len(self.original_ids) for x in vectors), "OOF alignment mismatch")
        unique(tuple(zip(self.original_ids, self.seed_ids)), "original ID/seed pairs")
        require(all(x >= 0 for x in self.fold_ids + self.seed_ids), "negative fold or seed")
        require(all(x >= 0 for x in self.r_a + self.r_d), "negative density ratio")
        require(all(x >= 0 for x in self.origin_weights) and sum(self.origin_weights) > 0,
                "invalid origin weights")
        require(self.lineage.split_hash is not None, "OOF split hash required")
        by_id = {}
        for oid, fold, weight in zip(self.original_ids, self.fold_ids, self.origin_weights):
            previous = by_id.setdefault(oid, (fold, weight))
            require(previous == (fold, weight), "fold or origin weight changes across seeds")


@dataclass(frozen=True, slots=True, kw_only=True)
class Estimate(Immutable):
    spec: EstimandSpec
    method: str
    value: float
    standard_error: float | None
    original_ids: tuple[str, ...]
    scores: tuple[float, ...]
    influence: tuple[float, ...]
    seed_ids: tuple[int, ...]
    lineage: ArtifactLineage

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.method, "estimation method")
        _ids(self.original_ids)
        _aligned(self.lineage, self.original_ids)
        require(len(self.scores) == len(self.influence) == len(self.original_ids), "estimate alignment mismatch")
        require(bool(self.seed_ids), "seed provenance required")
        unique(self.seed_ids, "seeds")
        if self.standard_error is not None:
            require(self.standard_error >= 0, "negative standard error")


@dataclass(frozen=True, slots=True, kw_only=True)
class StageRequest(Immutable):
    stage: str
    config_path: str
    config_hash: str
    task_path: str
    task_hash: str
    dependency_paths: tuple[str, ...]
    dependency_hashes: tuple[str, ...]
    output_dir: str
    code_identity: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.stage, "stage")
        for value in (self.config_path, self.task_path, self.output_dir) + self.dependency_paths:
            require(Path(value).is_absolute(), "request paths must be absolute")
        unique(self.dependency_paths, "dependency paths")
        require(len(self.dependency_paths) == len(self.dependency_hashes), "dependency alignment mismatch")
        for value in (self.config_hash, self.task_hash) + self.dependency_hashes:
            check_hash(value)
        require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.code_identity) is not None,
                "code identity must be a full commit hash")

    def verify_inputs(self) -> None:
        for path, digest in zip((self.config_path, self.task_path) + self.dependency_paths,
                                (self.config_hash, self.task_hash) + self.dependency_hashes):
            require(file_hash(path) == digest, f"input hash mismatch: {path}")


@dataclass(frozen=True, slots=True, kw_only=True)
class StageResult(Immutable):
    request_hash: str
    status: Literal["pass", "fail", "blocked"]
    artifacts: tuple[ArtifactRecord, ...]
    message: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        check_hash(self.request_hash)
        nonempty(self.message, "stage message")
        unique(tuple(a.path for a in self.artifacts), "artifact paths")
        if self.status == "pass":
            require(bool(self.artifacts), "passing stage must declare artifacts")

    def verify(self, request: StageRequest) -> None:
        require(self.request_hash == request.content_hash, "stage request mismatch")
        request.verify_inputs()
        root = Path(request.output_dir).resolve(strict=True)
        for artifact in self.artifacts:
            path = (root / artifact.path).resolve(strict=True)
            require(path.is_relative_to(root), "artifact escapes output directory")
            require(file_hash(path) == artifact.sha256, f"artifact hash mismatch: {artifact.path}")


class StageRunner(Protocol):
    """Each stage module exports this method as a module-level function."""

    def run_stage(self, request: StageRequest) -> StageResult: ...
