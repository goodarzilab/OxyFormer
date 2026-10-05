# Reporting handoff

`oxyformer.reporting.run_stage(StageRequest)` accepts `anchor_review`,
`audit_collection`, or `tract_release`. The task JSON maps `bundle`, `manifest`,
`receipts`, and `approvals` to absolute paths. These must be exactly the
request's `dependency_paths`, with their corresponding `dependency_hashes`.
Config is `configs/reporting.yaml`. Approvals must resolve to the repository's
read-only `configs/approvals.yaml`. Source checkouts bind this registry to the
code's repository. A non-editable installation uses the repository
`configs/reporting.yaml` at the hash-bound `StageRequest.config_path`; its
adjacent `approvals.yaml` is the only accepted registry. Installed code never
looks for owner approvals beside `site-packages`. Output protection uses that
same repository. Results are created once in `output_dir`:
`report.json`, self-contained `report.html`, and `estimators.svg`. Existing
outputs are never overwritten. Identical reruns reverify inputs and existing
bytes and return the same result; matching partial publications are completed.
Conflicting files or an output directory that contains inputs, or overlaps
repository src/, configs/, outputs/ or report/, return a failed StageResult with
no stale artifact references. No reporting path loads outcomes into a model.

Use canonical `ReportBundle`, `ExpectedTasks`, and `TaskReceipts` artifacts from
`records.py`. The merged `Estimate` contract contains final scores and normalized
influence contributions averaged across seeds by original ID. Reporting aligns
IDs, checks all estimand dimensions and seed/split identities, and uses the
merged covariance functions. The recorded `primary_input_consistency` gate
enforces plan section 4.4 for
`mtp_one_step` and every primary `CV_TMLE_METHODS` member. Under the merged
producer convention, at least four parents are required and their ordered
prefix must match: initial OOFNuisances, LoadedData, SplitManifest, policy.
Parent index 2 must also equal each estimate's `split_hash`. Additional parent
hashes remain visible and may differ. A mismatch fails release while retaining
all estimates, influence/covariance, overlap, coverage and sensitivity diagnostics.
Nonprimary comparators and disclosed sensitivities retain their existing handling.
A different registered split belongs in `sensitivities`, where it
remains visible with its own provenance. It does not average nuisance predictions or treat
seeds as independent observations. A bundle's ratios are seed by original ID;
functional balance arrays are original ID by frozen function. Attrition contains
sequential remaining counts ending at the frozen target count. Changed-target
sensitivities retain their own entire Estimate and require an explicit disclosure.
Incorrect disclosures record a failed gate while retaining every sensitivity
estimate and the primary diagnostics. Raw sensitivity records also remain
visible if another consistency check interrupts diagnostic assembly.

`counties` contains globally unique dependence-unit identifiers, such as full
county FIPS or state-qualified names (`AR:Benton`, `MO:Benton`), with the same
keys in `county_locations`. These are group identities for the merged covariance
API, not display names. A county ID cannot identify two states; silently merging
different counties would corrupt both covariance and concentration diagnostics.

The expected-task manifest is frozen externally. Every task binds an upstream
StageRequest hash and a gate role. Collection verifies the actual StageResult,
its inputs, declared artifacts, hashes and status. All expected tasks must be
present even when another task with the same role has already passed. Required
roles are in `STAGE_GATES`; production task builders may add repetitions, folds,
scenarios and other tasks but cannot remove the required roles. Coverage must
contain every scenario in `coverage_scenarios`. Bounds and production-procedure
metadata come from the validated simulation producer, not reporting itself.
Each expected task with role `coverage` must publish canonical `CoverageScenario`
artifacts with kind `coverage_scenario`, serialized through `write_artifact` and
listed in its verified `StageResult`. Scenario-level summaries belong to this
role; raw repetition tasks can use a separate role in the complete manifest.
Every reported coverage record must exactly match its verified artifact.
Missing summaries block release, and contradictory summaries fail it even when
the task status and external approvals say pass. Reports disclose the task,
request and artifact identities used for coverage decisions. This uses the
existing reporting record and merged artifact API; no numerical threshold changes.

Scientific parameters are read at runtime from
`owner_decisions.release_gates` and `owner_decisions.influence_concentration_gate`.
Fixtures mirror the approved PR #18 parameters in the owner registry.
The approved concentration definition is county-based: sum final contributions
within county first, then square. County sums and concentration metrics use exact
rational arithmetic on the supplied binary64 contributions. Exact `s_max_exact`
and `G_eff_exact` ratios are compared with the owner's decimal thresholds, so
rounding cannot change an inclusive boundary decision; no tolerance is added.
Floating diagnostics and `D_scientific` are presentation values. County totals
are converted once to floats for the unchanged merged cluster/spatial formulas. Report D, s_max, G_eff and ranked counties for
one-step and CV-TMLE separately. State shares sum these county information
shares and are descriptive. This uses the orchestrator's clarified PR #18 gate
instead of inventing a state/block metric or a different information definition.
Zero D fails this confirmatory gate, including a scientifically exact identity
contrast. Positive D outside float64 range is preserved as `D_scientific`
with a null numeric D; positivity is determined from the exact county sums. The family alias
`cv_tmle` and the merged likelihood-specific names use identical gates.
Failure retains the point estimates as diagnostic-only; no trimming,
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
and is visibly marked exploratory. The module `oxyformer.validation.geography_probes` uses copied
DiagnosticViews and a held-out linear reconstruction baseline. It grants no
training permissions: the shared FeatureRegistry/CovariateView contract refuses
precise geography and exposure proxies as nuisances. Renaming forbidden values
and lying about their semantic role is outside that contract; upstream adapters
remain responsible for truthful roles. Probe fit cannot see held-out exposure.

Mandatory publication fields are checked for content. Retry rules must contain
at least one rule, and every entry must be non-whitespace text; an explicit
`no retries` entry is valid. Blank rules remain representable in an incomplete
coverage artifact but fail release. Attrition steps, geography identifiers and
balance-function names must also contain text, in addition to their existing
alignment and numerical checks. External reviewer/reference fields and the
owner concentration definition, requirement and failure-action text must be
nonempty strings. This validates publication completeness, not the scientific
truth of a producer's prose or the authenticity of a human approval.

The stage authenticates the task and bundle against the frozen request before
checking aggregate prerequisites. Thus an absent or changed receipt, manifest,
config or approval file cannot hide an authenticated estimate. The report keeps
all estimates and sensitivities as diagnostic-only; an independently verified,
compatible manifest also permits the existing derived diagnostics. Missing or
changed task/bundle bytes provide no authenticated estimates to display. Complete
request verification, the fixed owner-registry path, every scientific gate and
a second input verification still precede release. Output isolation and immutable
publication remain required even for refusal reports.

Finite inputs can still overflow during derived diagnostic arithmetic. The
complete derived record is checked for finite JSON values before attachment to
an evaluation or refusal report. A nonfinite derived value fails the scientific
report instead of reaching publication as released evidence; authenticated
estimates and sensitivities remain publishable with the failure reason.
