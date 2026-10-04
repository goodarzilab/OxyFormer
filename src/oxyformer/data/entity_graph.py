"""Dependence links are namespaced, transitive and independent of predictors."""
from dataclasses import dataclass
from typing import Literal

from oxyformer.provenance import Immutable, nonempty, require, unique

Relation = Literal["household", "psu", "municipality", "repeated_geography", "outcome_lineage"]


@dataclass(frozen=True, slots=True, kw_only=True)
class EntityLink(Immutable):
    observation_id: str
    relation: Relation
    namespace: str
    entity_id: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.observation_id, self.namespace, self.entity_id):
            nonempty(value, "entity link")


@dataclass(frozen=True, slots=True, kw_only=True)
class EntityGraph(Immutable):
    original_ids: tuple[str, ...]
    links: tuple[EntityLink, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.original_ids), "empty entity graph")
        unique(self.original_ids, "original IDs")
        unique(self.links, "entity links")
        for value in self.original_ids:
            nonempty(value, "original ID")
        require(all(link.observation_id in self.original_ids for link in self.links), "unknown linked ID")
        object.__setattr__(self, "links", tuple(sorted(
            self.links, key=lambda x: (x.observation_id, x.relation, x.namespace, x.entity_id))))

    def components(self) -> tuple[tuple[str, ...], ...]:
        parents = {x: x for x in self.original_ids}

        def root(x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        entities = {}
        for link in self.links:
            key = (link.relation, link.namespace, link.entity_id)
            other = entities.setdefault(key, link.observation_id)
            parents[root(link.observation_id)] = root(other)
        groups = {}
        for x in self.original_ids:
            groups.setdefault(root(x), []).append(x)
        return tuple(sorted(tuple(sorted(group)) for group in groups.values()))

    def assert_partition(self, assignments: dict[str, object]) -> None:
        require(set(assignments) == set(self.original_ids), "partition must cover whole entity graph")
        for group in self.components():
            require(len({assignments[x] for x in group}) == 1, "entity lineage crosses partitions")
