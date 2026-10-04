"""Endpoint permissions are explicit approvals, never inferred from names."""
from dataclasses import dataclass
from typing import Literal

from oxyformer.provenance import Immutable, nonempty, require, unique

Role = Literal["predictor", "exposure", "exposure_proxy", "precise_geography",
               "downstream_health", "outcome", "outcome_metadata", "county", "identifier"]
Use = Literal["nuisance", "ssl", "context", "county_routing", "diagnostic", "score", "linkage"]
_ALLOWED = {
    "predictor": {"nuisance", "ssl", "context", "diagnostic"},
    "exposure": {"diagnostic", "score"},
    "exposure_proxy": {"diagnostic"},
    "precise_geography": {"diagnostic", "linkage"},
    "downstream_health": {"diagnostic"},
    "outcome": {"score"},
    "outcome_metadata": {"diagnostic", "linkage"},
    "county": {"county_routing", "linkage", "diagnostic"},
    "identifier": {"linkage"},
}


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureRule(Immutable):
    name: str
    role: Role
    endpoints: tuple[str, ...]
    uses: tuple[Use, ...]
    approval_id: str | None

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.name, "feature name")
        unique(self.endpoints, "endpoints")
        unique(self.uses, "uses")
        require(set(self.uses) <= _ALLOWED[self.role], f"forbidden permissions for {self.role}")
        if self.uses:
            require(bool(self.endpoints) and bool(self.approval_id and self.approval_id.strip()),
                    "permissions require endpoint approval")
        for endpoint in self.endpoints:
            nonempty(endpoint, "endpoint")


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureRegistry(Immutable):
    registry_id: str
    rules: tuple[FeatureRule, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.registry_id, "registry ID")
        unique(tuple(rule.name for rule in self.rules), "feature names")
        object.__setattr__(self, "rules", tuple(sorted(self.rules, key=lambda r: r.name)))

    def require(self, name: str, endpoint: str, use: Use) -> FeatureRule:
        rules = {rule.name: rule for rule in self.rules}
        require(name in rules, f"unknown feature: {name}")
        rule = rules[name]
        require(endpoint in rule.endpoints and use in rule.uses and bool(rule.approval_id),
                f"unapproved {use} access to {name} for {endpoint}")
        return rule
