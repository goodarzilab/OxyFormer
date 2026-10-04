"""Reporting handoffs; these objects grant no nuisance-model permissions."""
from dataclasses import dataclass
from typing import Literal

from oxyformer.contracts import Estimate, EstimandSpec, SourceManifest, StageRequest, StageResult
from oxyformer.provenance import Immutable, check_hash, nonempty, require, unique

STAGE_GATES = {
    "anchor_review": ("legacy_reproduction", "analytic", "coverage", "physiology_anchor", "birth_anchor"),
    "audit_collection": ("support", "coverage", "refit_audit", "spatial_sensitivity", "estimator_agreement", "geography_isolation"),
}
STAGE_GATES["tract_release"] = tuple(dict.fromkeys(
    STAGE_GATES["anchor_review"] + STAGE_GATES["audit_collection"] + ("overlap",)))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedTask(Immutable):
    task_id: str
    gate: str
    request_hash: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.task_id, "task ID")
        nonempty(self.gate, "gate")
        check_hash(self.request_hash)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedTasks(Immutable):
    stage: Literal["anchor_review", "audit_collection", "tract_release"]
    spec: EstimandSpec
    seed_ids: tuple[int, ...]
    tasks: tuple[ExpectedTask, ...]
    coverage_scenarios: tuple[str, ...]
    # Frozen complete endpoint-country family; empty means not yet registered.
    mortality_family: tuple[str, ...] = ()

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.seed_ids), "registered seeds required")
        unique(self.seed_ids, "registered seeds")
        unique(tuple(t.task_id for t in self.tasks), "expected tasks")
        unique(self.coverage_scenarios, "coverage scenarios")
        unique(self.mortality_family, "mortality family")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskReceipt(Immutable):
    task_id: str
    request: StageRequest
    result: StageResult


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskReceipts(Immutable):
    items: tuple[TaskReceipt, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        unique(tuple(t.task_id for t in self.items), "received tasks")


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageScenario(Immutable):
    scenario_id: str
    repetitions: int
    independent_repetitions: bool
    production_tuning_and_stopping: bool
    coverage_one_sided_95_lower_bound: float
    null_rejection_upper_bound: float
    abs_bias_over_empirical_sd: float
    mean_se_over_empirical_sd: float
    numerical_failure_upper_bound: float
    all_failures_published: bool
    registered_retry_rules: tuple[str, ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.scenario_id, "scenario")
        require(self.repetitions > 0, "positive repetition count required")
        for x in (self.coverage_one_sided_95_lower_bound, self.null_rejection_upper_bound,
                  self.numerical_failure_upper_bound):
            require(0 <= x <= 1, "probability bound outside [0,1]")
        require(self.abs_bias_over_empirical_sd >= 0 and self.mean_se_over_empirical_sd >= 0,
                "negative coverage diagnostic")


@dataclass(frozen=True, slots=True, kw_only=True)
class Sensitivity(Immutable):
    name: str
    estimate: Estimate
    target_change: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.name, "sensitivity")
        nonempty(self.target_change, "target change disclosure (use unchanged when applicable)")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportBundle(Immutable):
    spec: EstimandSpec
    sources: tuple[SourceManifest, ...]
    original_ids: tuple[str, ...]
    # Sequential remaining counts: first source frame, last frozen target.
    attrition: tuple[tuple[str, int], ...]
    weights: tuple[float, ...]
    observed_exposure: tuple[float, ...]
    shifted_exposure: tuple[float, ...]
    seed_ids: tuple[int, ...]
    ratios: tuple[tuple[float, ...], ...]  # seed x original ID
    balance_basis_id: str
    balance_names: tuple[str, ...]
    balance_observed: tuple[tuple[float, ...], ...]  # original ID x frozen function
    balance_shifted: tuple[tuple[float, ...], ...]
    estimates: tuple[Estimate, ...]
    counties: tuple[str, ...]
    states: tuple[str, ...]
    county_locations: tuple[tuple[str, tuple[float, float]], ...]
    coverage: tuple[CoverageScenario, ...]
    sensitivities: tuple[Sensitivity, ...] = ()
    p_values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.original_ids), "empty reporting target")
        unique(self.original_ids, "reporting original IDs")
        unique(self.seed_ids, "reporting seeds")
        unique(tuple(e.method for e in self.estimates), "reported estimators")
        unique(tuple(s.name for s in self.sensitivities), "sensitivity names")
        unique(tuple(x[0] for x in self.p_values), "endpoint p-values")
        unique(tuple(x.scenario_id for x in self.coverage), "coverage scenarios")
