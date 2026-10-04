# Reporting handoff

`oxyformer.reporting.run_stage(StageRequest)` accepts `anchor_review`,
`audit_collection`, or `tract_release`. The task JSON maps `bundle`, `manifest`,
`receipts`, and `approvals` to absolute paths. These must be exactly the
request's `dependency_paths`, with their corresponding `dependency_hashes`.
Config is `configs/reporting.yaml`. Approvals must resolve to the repository's
read-only `configs/approvals.yaml`. Results are created once in `output_dir`:
`report.json`, self-contained `report.html`, and `estimators.svg`. Existing
outputs are never overwritten. No reporting path loads outcomes into a model.

Use canonical `ReportBundle`, `ExpectedTasks`, and `TaskReceipts` artifacts from
`records.py`. The merged `Estimate` contract contains final scores and normalized
influence contributions averaged across seeds by original ID. Reporting aligns
IDs, checks all estimand dimensions and seed/split identities, and uses the
merged covariance functions. It does not average nuisance predictions or treat
seeds as independent observations. A bundle's ratios are seed by original ID;
functional balance arrays are original ID by frozen function. Attrition contains
sequential remaining counts ending at the frozen target count. Changed-target
sensitivities retain their own entire Estimate and require an explicit disclosure.

The expected-task manifest is frozen externally. Every task binds an upstream
StageRequest hash and a gate role. Collection verifies the actual StageResult,
its inputs, declared artifacts, hashes and status. All expected tasks must be
present even when another task with the same role has already passed. Required
roles are in `STAGE_GATES`; production task builders may add repetitions, folds,
scenarios and other tasks but cannot remove the required roles. Coverage must
contain every scenario in `coverage_scenarios`. Bounds and production-procedure
metadata come from the validated simulation producer, not reporting itself.

Scientific parameters are read at runtime from
`owner_decisions.release_gates` and `owner_decisions.influence_concentration_gate`.
Fixtures mirror PR #18; the launch base intentionally predates that approval.
The approved concentration definition is county-based: sum final contributions
within county first, then square. Report D, s_max, G_eff and ranked counties for
one-step and CV-TMLE separately. State shares sum these county information
shares and are descriptive. This uses the orchestrator's clarified PR #18 gate
instead of inventing a state/block metric or a different information definition.
Zero D fails this confirmatory gate, including a scientifically exact identity
contrast. Failure retains the point estimates as diagnostic-only; no trimming,
ratio capping, reweighting or favorable-estimator selection is performed.

ESS is `(sum v)^2 / sum(v^2)` for nonnegative weights only. Target `v=w` and
ratio `v=w*r` ESS are distinct. Signed correction weights have absolute-mass
concentration and no sampling-weight ESS. Full, moved, and affected subsets
are shown per seed. An unchanged record may be affected by incoming mass.
Ratio p99 > 10 and ESS fraction < .25 are warnings; a scoped overlap review
must establish whether the observed regime is validated. Functional balance
compares `E_T[r*f(A,X)]` with `E_T[f(d(A,X),X)]` over the full target. It is not
reinterpreted as a conditional identity within the moved subset.

External decisions are separate from computation. The owner registry may contain
`owner_decisions.reporting_approvals`, a list of records with these fields:

```text
gate, status: approved, reviewer, reference,
stage, bundle_hash, manifest_hash, receipts_hash, config_hash
```

All hashes bind the exact report inputs; stale approvals cannot transfer. Every
required gate, plus `expected_manifest`, needs an external decision. Tract
release additionally requires `tract_release`. Bias/SD > the approved
investigation threshold additionally requires `bias_investigation`; exceeding
the lower bias target alone is a warning. Estimator disagreement always remains
visible and requires `estimator_agreement` review. Reporting invents no material-
disagreement threshold. Expected physiological or birth-anchor signs are never
acceptance criteria. A failed upstream gate cannot be waived by an approval.
No run approvals are fabricated by this code or added to the owner file.

Report states distinguish `missing`, `blocked`, `failed`, `exploratory`, and
`released`. Only the last is releasable. Missing inputs produce a blocked
StageResult; executed failures produce fail. Successful anchor/audit collection
returns pass with exploratory evidence, not release. Consumers must check both
StageResult status and the report's explicit release state. Limits remain visible
in every rendering. A pass verifies a finite declared contract, not identification.

The fixed Holm family is tract life expectancy and US lung incidence at .05.
The later BY .05 registry uses `country:endpoint` entries for all eight registered
mortality endpoints in each registered country. Missing p-values leave a family
unfinished and produce no subset-adjusted claims. Anchors and genetic/trial
families remain separate. Supplied p-values must belong to the prespecified
analysis; reporting does not choose among estimators or bandwidths.

The three existing report/asset scripts accept `--v2-request REQUEST.json`,
using this same isolated stage and outputs. Their legacy mode remains available
and is visibly marked exploratory. The root `geography_probes.py` uses copied
DiagnosticViews and a held-out linear reconstruction baseline. It grants no
training permissions: the shared FeatureRegistry/CovariateView contract refuses
precise geography and exposure proxies as nuisances. Renaming forbidden values
and lying about their semantic role is outside that contract; upstream adapters
remain responsible for truthful roles. Probe fit cannot see held-out exposure.
