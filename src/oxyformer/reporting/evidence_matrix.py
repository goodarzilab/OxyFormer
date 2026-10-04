"""Release decisions require scientific gates, complete fan-in and external approval.

Trust boundary: the coordinator freezes the expected-task manifest and owner
approval registry. Upstream producers truthfully describe science metadata;
we verify identities, files, numerical criteria and full task coverage. Neither
hashes nor a StageResult pass authenticate a human or prove identification.
"""
from dataclasses import asdict
import math

from oxyformer.provenance import ContractError, require
from oxyformer.reporting.diagnostics import summarize
from oxyformer.reporting.records import CV_TMLE_METHODS, STAGE_GATES

LIMITATIONS = (
    "Altitude bundles environmental and social exposures; predictive fit does not identify an oxygen-specific causal effect.",
    "Residence at observation is not lifetime or pregnancy-long exposure; migration and survivor/live-birth selection remain unresolved.",
    "Flag-1 selection changes the target. Historical smoking is imperfectly measured. No geographic treatment recommendation follows.",
    "Uncertainty concerns a declared geographic stochastic process, not literal census sampling error; upstream measurement uncertainty is separate.",
    "Expected anchor signs are never acceptance criteria. Every estimator, seed and spatial sensitivity remains visible.",
    "Coordinates, terrain, location embeddings and probe outputs are diagnostic-only and cannot be nuisance predictors.",
)
HOLM_FAMILY = ("usaleep_life_expectancy", "us_lung_incidence")
MORTALITY_ENDPOINTS = ("all_cause", "lung_cancer", "ihd", "stroke", "copd", "diabetes", "kidney_disease", "alzheimer_disease")


def multiplicity(p_values, mortality_family):
    """Never adjust only the available subset and call the family complete."""
    values = dict(p_values)
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in values.values()), "invalid p-value")
    if mortality_family:
        pairs = [entry.split(":", 1) for entry in mortality_family]
        require(all(len(pair) == 2 and pair[0] and pair[1] in MORTALITY_ENDPOINTS for pair in pairs),
                "mortality family requires country:endpoint registry entries")
        countries = {pair[0] for pair in pairs}
        require(set(mortality_family) == {f"{country}:{endpoint}" for country in countries for endpoint in MORTALITY_ENDPOINTS},
                "incomplete registered mortality endpoint-country family")
    reports = {}
    for name, family, method in (("tract_lung", HOLM_FAMILY, "Holm"),
                                  ("adult_mortality", mortality_family, "BY")):
        available = {e: values[e] for e in family if e in values}
        complete = bool(family) and len(available) == len(family)
        adjusted = {}
        if complete:
            ordered = sorted(available, key=available.get)
            n = len(ordered)
            if method == "Holm":
                previous = 0.0
                for i, endpoint in enumerate(ordered):
                    previous = max(previous, min(1.0, (n - i) * available[endpoint]))
                    adjusted[endpoint] = previous
            else:
                harmonic = sum(1 / i for i in range(1, n + 1))
                previous = 1.0
                for i in reversed(range(n)):
                    endpoint = ordered[i]
                    previous = min(previous, n * harmonic * available[endpoint] / (i + 1))
                    adjusted[endpoint] = previous
        reports[name] = {"method": method, "alpha": .05, "family": list(family), "complete": complete,
                         "available_p_values": available, "adjusted_p_values": adjusted,
                         "missing": [e for e in family if e not in available],
                         "status": "complete" if complete else "unfinished"}
    reports["adult_mortality"]["prespecified_endpoints"] = list(MORTALITY_ENDPOINTS)
    reports["adult_mortality"]["registration"] = "Endpoint-country family must be frozen externally; no country list inferred."
    reports["separate_families"] = ["physiological anchors", "birth anchors", "genetic evidence", "randomized trials"]
    return reports


def external_approval(approvals, gate, scope):
    """Approvals live in the owner registry, never in producer result payloads.

    reporting_approvals entries bind bundle/manifest/receipts/config hashes, a
    gate and a stage. A stale, rejected or unexplained approval cannot release.
    """
    records = approvals.get("owner_decisions", {}).get("reporting_approvals", [])
    matches = [record for record in records if record.get("gate") == gate and
               all(record.get(k) == v for k, v in scope.items())]
    require(len(matches) <= 1, "contradictory or duplicate external approval")
    return bool(matches and matches[0].get("status") == "approved" and
                matches[0].get("reviewer") and matches[0].get("reference"))


def coverage_decisions(bundle, manifest, approved):
    gates, warnings = [], []
    required = ("min_repetitions_per_scenario", "coverage_one_sided_95_lower_bound_min",
                "null_rejection_upper_bound_max", "abs_bias_over_empirical_sd_target",
                "abs_bias_over_empirical_sd_investigate_above", "mean_se_over_empirical_sd_range",
                "numerical_failure_upper_bound_max", "publish_all_failures_and_registered_retry_rules")
    if not isinstance(approved, dict) or not all(k in approved for k in required):
        return [{"gate": "coverage_parameters", "status": "blocked", "reason": "owner release_gates approval missing"}], [], False
    require(approved["publish_all_failures_and_registered_retry_rules"] is True, "contradictory publication approval")
    lo, hi = approved["mean_se_over_empirical_sd_range"]
    require(0 <= lo <= hi and approved["min_repetitions_per_scenario"] > 0, "invalid coverage approval")
    scenarios = {s.scenario_id: s for s in bundle.coverage}
    require(set(scenarios) <= set(manifest.coverage_scenarios), "unregistered coverage scenario")
    if not manifest.coverage_scenarios:
        gates.append({"gate": "coverage_manifest", "status": "missing", "reason": "no selected final scenarios registered"})
    investigate = False
    for name in manifest.coverage_scenarios:
        s = scenarios.get(name)
        if s is None:
            gates.append({"gate": f"coverage:{name}", "status": "missing", "reason": "scenario absent"})
            continue
        failures = []
        checks = {
            "independent production repetitions": s.independent_repetitions and s.production_tuning_and_stopping,
            "repetition count": s.repetitions >= approved["min_repetitions_per_scenario"],
            "coverage lower bound": s.coverage_one_sided_95_lower_bound >= approved["coverage_one_sided_95_lower_bound_min"],
            "null rejection upper bound": s.null_rejection_upper_bound <= approved["null_rejection_upper_bound_max"],
            "SE calibration": lo <= s.mean_se_over_empirical_sd <= hi,
            "numerical failure bound": s.numerical_failure_upper_bound <= approved["numerical_failure_upper_bound_max"],
            "failures and retry rules published": s.all_failures_published and bool(s.registered_retry_rules),
        }
        failures = [label for label, passed in checks.items() if not passed]
        gates.append({"gate": f"coverage:{name}", "status": "failed" if failures else "pass",
                      "reason": ", ".join(failures) or "all approved numerical criteria met", "metrics": asdict(s)})
        if s.abs_bias_over_empirical_sd > approved["abs_bias_over_empirical_sd_target"]:
            warnings.append(f"{name}: bias/SD exceeds target; target is not a hard maximum")
        if s.abs_bias_over_empirical_sd > approved["abs_bias_over_empirical_sd_investigate_above"]:
            investigate = True
    return gates, warnings, investigate


def evaluate(bundle, manifest, receipts, approvals, config_hash):
    """Pure report assembly apart from verifying upstream immutable artifacts."""
    report = {"schema_version": 1, "stage": manifest.stage, "state": "blocked", "releasable": False,
              "evidence_label": "diagnostic-only", "limitations": list(LIMITATIONS), "gates": [],
              "estimators": [asdict(e) for e in bundle.estimates], "warnings": [],
              "bundle_hash": bundle.content_hash, "manifest_hash": manifest.content_hash,
              "receipts_hash": receipts.content_hash, "config_hash": config_hash}
    gates = report["gates"]
    scope = {k: report[k] for k in ("stage", "bundle_hash", "manifest_hash", "receipts_hash", "config_hash")}
    try:
        report["diagnostics"] = summarize(bundle, manifest)
        report["multiplicity"] = multiplicity(bundle.p_values, manifest.mortality_family)
        methods = [e.method for e in bundle.estimates]
        paired = "mtp_one_step" in methods and any(m in CV_TMLE_METHODS for m in methods)
        gates.append({"gate": "paired_estimators", "status": "pass" if paired else "missing", "reason": "one-step and CV-TMLE both required"})
        received = {t.task_id: t for t in receipts.items}
        expected = {t.task_id: t for t in manifest.tasks}
        require(set(received) <= set(expected), "unexpected task receipt")
        required_gates = STAGE_GATES[manifest.stage]
        for gate in required_gates:
            if not any(t.gate == gate for t in manifest.tasks):
                gates.append({"gate": gate, "status": "missing", "reason": "expected-task manifest lacks required gate"})
        for task in manifest.tasks:
            receipt = received.get(task.task_id)
            if receipt is None:
                gates.append({"gate": task.gate, "task_id": task.task_id, "status": "missing", "reason": "expected task absent"})
                continue
            require(receipt.request.content_hash == task.request_hash, "task request identity mismatch")
            try:
                receipt.result.verify(receipt.request)
                status = {"pass": "pass", "fail": "failed", "blocked": "blocked"}[receipt.result.status]
                reason = receipt.result.message
            except FileNotFoundError:
                status, reason = "missing", "upstream input or artifact file absent"
            gates.append({"gate": task.gate, "task_id": task.task_id, "status": status, "reason": reason})
        owner = approvals.get("owner_decisions", {})
        coverage, warnings, investigate = coverage_decisions(bundle, manifest, owner.get("release_gates"))
        gates.extend(coverage)
        report["warnings"].extend(warnings)
        concentration = owner.get("influence_concentration_gate")
        if not concentration or not all(k in concentration for k in ("applies_to", "definitions", "s_max_max", "g_eff_min", "require", "on_failure")):
            gates.append({"gate": "influence_concentration", "status": "blocked", "reason": "owner influence_concentration_gate approval missing"})
        else:
            require(set(concentration["applies_to"]) == {"one_step", "cv_tmle"}, "contradictory estimator concentration approval")
            require(0 < concentration["s_max_max"] <= 1 and concentration["g_eff_min"] >= 1, "invalid concentration approval")
            for method, metric in report["diagnostics"]["information"].items():
                if method != "mtp_one_step" and method not in CV_TMLE_METHODS:
                    continue
                passed = metric["positive_D"] and metric["s_max"] <= concentration["s_max_max"] and metric["G_eff"] >= concentration["g_eff_min"]
                gates.append({"gate": f"influence_concentration:{method}", "status": "pass" if passed else "failed",
                              "reason": "approved county s_max/G_eff gate; never trim, cap or reweight post hoc"})
            report["approved_concentration_gate"] = concentration
        approval_gates = list(required_gates) + ["expected_manifest"]
        if investigate:
            approval_gates.append("bias_investigation")
        if manifest.stage == "tract_release":
            approval_gates.append("tract_release")
        for gate in approval_gates:
            ok = external_approval(approvals, gate, scope)
            gates.append({"gate": f"approval:{gate}", "status": "pass" if ok else "blocked",
                          "reason": "external scoped approval verified" if ok else "external scoped approval missing"})
    except (ContractError, ValueError, TypeError, KeyError, OverflowError) as exc:
        gates.append({"gate": "input_consistency", "status": "failed", "reason": str(exc)})
    statuses = {g["status"] for g in gates}
    if "failed" in statuses:
        report["state"] = "failed"
    elif "missing" in statuses:
        report["state"] = "missing"
    elif "blocked" in statuses:
        report["state"] = "blocked"
    else:
        report["state"] = "released" if manifest.stage == "tract_release" else "exploratory"
    report["releasable"] = report["state"] == "released"
    report["evidence_label"] = "confirmatory release" if report["releasable"] else "diagnostic-only"
    return report
