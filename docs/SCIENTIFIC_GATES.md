# Scientific admission and release

The fixed numerical authority is [configs/approvals.yaml](../configs/approvals.yaml).
Read it from the merged clone; never edit it or infer a missing approval.
The [frozen plan](plan/OXYFORMER_V2_PLAN.md) wins over task shorthand.

| Decision | Required evidence | What it permits |
| --- | --- | --- |
| Source permission | Public access, applicable terms, immutable payload/licence/schema hashes, reviewed mapping and joins | Acquisition/use within those terms; no predictor permission |
| Feature/policy approval | Endpoint-specific approved concepts and transformations; routing, support, shift, weights, target and outcome scale | The specified design and nuisance inputs; no locked fit or release |
| Recipe lock | Passed prerequisites, complete production timing, frozen code/environment/input identities, full work manifest and budget | Bounded execution of that exact recipe after all required allocations |
| Final release authorization | Complete scoped expected-task receipts, all scientific gates and owner reporting approvals bound to bundle/manifest/receipts/config | The authorized evidence report; not shared-root promotion |

Owner approval of TabPFN versions does not allow floating checkpoints, unreviewed
payloads or an unsupported observation model. The current nested runner blocks
A5–A7 and F0/F1 until resumable signed-correction/foundation interfaces exist. Their
required scientific comparisons cannot be counted as completed by omission.

## Gates in order

1. Verify source permission, raw fields, code/weight licences, reviewed joins
   and provenance. Missing fields, ambiguous geography or a missing approval
   block use. A named source is not a reviewed mapping.
2. Freeze the outcome-sealed 20% design reservation, two-mmHg shift-or-stay
   policy and supported target. Use pressure before population aggregation;
   no learned exposure weights, clipping, gap bridging or post hoc trimming.
   Enforce owner-approved primary outcome flags (1 and 3), inhabited relief/local support, five
   geographic outer folds within counties, three inner folds, at least eight
   outer-training and four inner-fitting tracts per county after exclusions.
3. Verify fold-nested preprocessing/SSL/tuning/calibration, independent
   nuisance weights and mandatory approved raw-X inputs. Held-out labels join
   only after nuisance freezing. County is a coarse routing/offset stratum;
   terrain, coordinates, satellite features and outcome metadata never become
   nuisance predictors.
4. Pass analytic identities, leakage tests, known-truth calibration and legacy
   reproduction. Predictive agreement cannot choose between observationally
   equivalent causal worlds. Smoke tests are implementation evidence only.
5. Lock the production recipe before screening. Screening evaluates that
   recipe without modifying it. Changes require a new prospective lock,
   independent evaluation draws and revalidated timing/approvals.
6. Complete every selected final scenario with at least **1,000 independent
   repetitions** using production tuning/stopping, not three neural seeds
   counted as three repetitions. Publish every failure and registered retry
   rule; missing repetitions or incomplete chains cannot disappear from the
   denominator.
7. Complete the physiological and birth anchors, full deletion/refit audits,
   overlap and spatial sensitivity checks, estimator agreement and geographic
   isolation. Expected anchor signs are not acceptance criteria. A failed
   support, coverage or anchor gate stops progression and goes to the owner.
8. Bind all expected task IDs/request hashes and receive the scoped final
   authorization. Hash verification alone does not establish scientific
   validity, completeness or release permission.

## Numerical release criteria

These are approved acceptance criteria, not inferential guarantees:

| Metric | Requirement |
| --- | --- |
| Coverage | One-sided 95% Monte Carlo lower bound ≥0.925 |
| Null rejection | Upper confidence bound ≤0.075 |
| Absolute bias / empirical SD | Target ≤0.10; investigate above 0.20 |
| Mean estimated SE / empirical SD | 0.90–1.10 |
| Numerical failure probability | Upper confidence bound ≤0.01 |
| Influence concentration, each estimator | Finite contributions, positive sum of squared county totals, maximum county share ≤0.10, effective county count ≥30 |

Compute county influence totals after seed aggregation by original observation.
Retain both one-step and CV-TMLE and the full cross-endpoint covariance;
report the registered 50/100/200-km spatial sensitivities. Ratio p99 >10 or
ESS <25% are warning thresholds, not universal release cutoffs. Examine the
moved portion separately. Do not choose a favorable seed, architecture,
bandwidth or support threshold to pass. A revised policy needs a new estimand
ID and locked analysis. Whole-geography deletion means refitting, not predicting
an unseen county with an invented intercept.

Multiplicity remains Holm 0.05 for tract life expectancy/US lung incidence and
BY FDR 0.05 for the registered adult mortality endpoint-country family. Anchors
and genetic/trial conclusions have separate families. Residential, genetic and
randomized-drug evidence remain separate; do not pool participant rows or
translate a geographic association into an oxygen-treatment recommendation.

The [execution blockers](EXECUTION.md#current-admission-status) currently prevent
production admission. Missing evidence must produce a diagnostic-only report
or a blocked/failed result. The synthetic integration report intentionally
lacks coverage, sufficient counties and release approvals: its finite
estimates must never authorize release. Shared-root promotion is absent from
this plan and requires the PI if requested later.

## USALEEP outcome-flag amendment (2026-10-10)

The owner decision `owner_decisions.outcome_flags` in
[approvals.yaml](../configs/approvals.yaml) amends frozen plan sections 4.1 and
5.3. `plan_fixed.primary_outcome_flags: [1]` remains as the historical value
with its amendment comment; primary labels, eligibility, training and evaluation
now use `primary: [1, 3]`. Flag 1 means observed age-specific death rates at all
ages, flag 2 predicted rates at all ages, and flag 3 a combination of observed
and predicted rates. Flag 2 has no primary label. Missing or malformed owner
decisions refuse execution, with no fallback to the historical restriction.

The unchanged `plan_fixed.min_flag1_tracts_per_county: 4` now counts distinct
primary-eligible tracts, as required by `county_minimum_counts: primary`.
Relief, distance, reservation, support, buffer and split rules are unchanged.
The motivating gate report found 4,421 flag-1 tracts among 66,718 nationally,
290 counties with at least four, and zero evaluable tracts after the screens.
Flags 1 and 3 provide 58,242 tracts and 2,078 counties with at least four;
these input counts do not establish support or authorize an effect release.

The registered `sensitivity: flag1_only` is a subset restricted to mortality
input flag 1. The tract metadata retains `mortality_input_flag` (called
`outcome_flag` in `GeographyRow`), and the gate support report retains that
per-ID mapping plus counts for the input frame, primary-eligible frame, sealed
design set and each buffer's evaluation target. Later sensitivity work can
select the flag-1 subset without acquiring new inputs. This change registers
and preserves the subset; it does not run the sensitivity analysis. Flags remain
diagnostic/selection metadata and never enter nuisance predictors.

## Tract design amendment A2 (2026-10-10)

The owner decision `owner_decisions.tract_design_amendment` in
[approvals.yaml](../configs/approvals.yaml) amends sections 4.1 and 5.1 for the
first US tract application. The frozen design was infeasible as specified:
tract-gate-r2 retained no evaluable tract. Its primary target ID
`e2999854bbeb2d232ca59b46e058b55e69b9ca6c2ddb0896d805f018350181d5`
remains in the gate's estimand registry with that status. The new estimand ID
is `usaleep-a2-2026-10-10`, with an analysis locked to A2. Each buffer scenario
also has a content-derived target ID. Missing or malformed A2 approvals refuse
execution. The original `tract_design` decision and execution task pin remain
unchanged.

Eligibility closes at the tract: an unusable or unsupported tract leaves the
target without removing its neighbours. Dependence links, assignment geography
and county by 10 km subblocks remain atomic for folds and buffers across the
whole non-reserved frame. Buffers exclude both sides of cross-fold boundaries,
including non-target frame rows. Primary-eligible labelled non-target rows
remain available for outcome regression and SSL. Flag-2 and other ineligible
rows are inert. Classifier fitting and calibration use target-distribution
rows; held-out target rows and their linked rows cannot enter their training
partition. The eight outer-training and four inner-fitting tract minima, fold
presence and post-exclusion county screens count labelled training rows. Outer
training also excludes its inner buffers. Conditional support is rechecked
against the same permitted labelled rows until target exclusions stabilise.

Support freezes once, before folds, using exposure, county and every approved
covariate from all allocation-qualified frame tracts. The caliper is 2 IQR on
every covariate, with 1 mmHg bins and two distinct tracts per bin; tail bins are
discarded and support gaps remain gaps. Its hash records the pool definition,
pool IDs, sealed reservation IDs and radius. The 20% geographic reservation
remains outcome-sealed and never enters estimation. Historical support settings
and `plan_fixed.min_inhabited_relief_m: 300` remain recorded in approvals.
County P90–P10 inhabited relief is now a diagnostic. The four-tract minimum and
local comparison within 25 km still screen counties.

The Y-perturbation acceptance criterion concerns decisions, not serialized
provenance (orchestrator clarification, 2026-10-10). Under a Y-only change,
`design.json`, `support_report.json`, target hashes, estimand IDs, fold
assignments, exclusions, buffers, policy IDs and support hashes are identical.
`splits.json` retains the existing source-bound `SplitScenario` interface and
may differ only in honest provenance. The canonical regression comparison
strips exactly `spec.source_lineage_hash`, `lineage.source_hashes`, and
`lineage.parent_hashes` from each outer/inner split and data manifest, plus
`DataManifest.sources[*].payload_hash`. It retains every other field, including
source schema/review metadata, graph/config hashes and all scientific decisions.
No production provenance is removed, rewritten or weakened by this comparison.

A joint outcome-blind feasibility census is required before approval of any new
gate design or wider threshold. Reports retain counts per layer and county,
flags, target hashes, maximum county share, effective counties
`(sum n)^2 / sum n^2`, county relief quantiles, and target tracts in counties
below 50, 100 and 300 m relief. A2 is count-informed: the owner and orchestrator
saw outcome-blind feasibility counts before approval. The approved replay
retained 1,752 tracts in 158 counties and 33 states, target hash
`c8ecb139b57de78d`; 1,724 tracts were in counties below 300 m relief and 79 were
flag 1. This is a different, predominantly low-relief population. Inference is
conditional on the realised frame because support is no longer estimated on
an independent reserved sample.

The 10 and 25 km scenarios are reported even when they contain no evaluable
tract, as in the approved replay. The declared 3-IQR fallback is recorded but
is not run by this gate. The 2 mmHg shift-or-stay policy, equal-tract weights,
release criteria and flag-1-only sensitivity remain fixed. Structural county
counts are proxies, not the fitted influence-concentration release checks;
ratio p99 and ESS on the moved target portion remain required downstream.

### A2 validation-harness ruling (orchestrator, 2026-10-11)

The semi-synthetic frame includes the target and its labelled training rows.
Rows with nonempty frozen support sample exposure by the registered SCM, including
supported training-only rows. A training-only row with empty support instead keeps
its observed atlas exposure and receives Y through the same SCM outcome and
latent-state machinery. It has zero estimand weight and never enters the
origin-classifier fit. Inert rows receive no Y. No new assignment law or support
interval is introduced, and no outcome enters the design.

The existing endpoint exposure column supplies fixed doses; artifact schemas and
frozen support/policy identities do not change. Nonempty-support validation and
the support-gaps/heterogeneous-eligibility switches concern sampled rows only.
Fixed rows bypass `AssignmentLaw`; existing measurement-error semantics recover
true dose from the fixed recorded dose for outcome generation. The real-frame
report counts sampled and fixed-exposure rows. On the approved historical frame
these are 7,061 sampled (1,752 targets plus 5,309 training-only) and 36,927 fixed.
Truth, coverage and release statistics remain target-only. This ruling addresses
a validation-harness consequence of A2 and will be disclosed to the owner in the
orchestrator's hourly report.

The orchestrator's second ruling of 2026-10-11 makes `support_gaps` and
`heterogeneous_eligibility` declarations of frame facts, checked by
`validate_policy`. For real-frame scenarios the profile registry declares them
`from_frame`; profile construction, campaign locking and budget estimation use
those same derived values. Fully synthetic scenarios retain their explicit
switches and mismatch checks. On the approved historical frame both declarations
are true: 282 sampled rows have gaps (34 target, 248 training-only), and 4,226
have no positive-width shift interval (1,035 target, 3,191 training-only).
The real-frame report records both values and these membership counts.
`bridge_support_gaps` remains false; the support, target and estimand are unchanged.
