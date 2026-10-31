# Execution and verification

The [frozen science plan](plan/OXYFORMER_V2_PLAN.md), [owner mandate](orchestrator-mandate.md)
and read-only [approvals](../configs/approvals.yaml) govern this run. The merged
[interfaces](INTERFACES.md) define artifact names. A passing unit test is not a
clinical release, a compute allocation, or permission to access a source.

## Current admission status

Production campaign admission is **blocked** at the audited merge base
`00c340adcf961068257d3f17749f2623683d533f`. The runner's owner-approval YAML date
serialization and reporting task/config handoff have been corrected upstream.
They must not be carried forward as current defects. Reporting accepts the
registered hyphenated stage names and binds the clone's reporting config and
owner registry; missing scientific evidence or scoped approvals still blocks it.

The remaining boundaries below must be resolved before admitting an executable
campaign. The integration fixture's direct scientific APIs do not certify the
registered runner route. Reproductions use synthetic inputs only; their
receipts, script and exact results belong in the attempt's `SUMMARY.md` and
`pytest.log`, outside Git.

| Boundary | Current contract and reproduction |
| --- | --- |
| Runner → design | The unchanged `tract-support-gate` registration reaches `design.gate.run_stage` with an execution envelope. It returns `invalid design stage configuration`: design requires `config.schema_version == 1` and `config.stage == request.stage == "tract_design"`. The direct design control passes; translating only the name still fails. |
| Design input roles and config | The registered needs supply raw acquisition and atlas files. Design requires canonical `DataManifest`, `CovariateView`, `GeographyTable`, `CollectedAtlas`, `EntityGraph` and a hash-bound owner file in `task.dependencies`. Isolated probes return `task must bind approvals...`, `task dependency is not hash-bound by StageRequest`, and, after adding just the direct schema/name, `missing or contradictory fixed approval: support_design_fraction`. No approval is actually missing from the owner registry; the adapter does not read the runner envelope. |
| Generic expansion → nested slices | `expand_campaign` emits scheduler resources but no `task.slice` execution limit. A 120-second leaf therefore reaches a stage whose default is 14,400 seconds with a 120-second checkpoint margin. Extra per-slice limits/outputs in the spec are ignored; all slices share `work.outputs`. Declaring terminal nuisances on a partial slice produces `stage omitted declared outputs`. Common checkpoint-only outputs require a separate terminal-completeness check. |
| Registered comparisons → nested stage | A5, A6, A7, F0 and F1 each return `blocked`: the density-ratio checkpoint interface cannot represent their signed corrections or foundation state. Their direct model interfaces do not provide an exact campaign continuation path. |
| Generic expansion → report collector | The generated `audit-collect` task has `expected_leaves`, but lacks `bundle`, `manifest`, `receipts` and `approvals`. The registered reporting worker returns `invalid reporting task fields`. A separately constructed complete reporting task is supported; the generic expander does not construct it. |
| Coverage collector → reporting evidence | `validation.campaign` publishes `result.json`, `gate.json` and an artifact manifest. Reporting requires canonical artifacts of kind `coverage_scenario`. A passing synthetic final collection still yields `coverage task has no CoverageScenario artifact` at that boundary. Copying metric values into a report cannot replace verified producer evidence. |

Source inspection also finds no registered producer of the typed design inputs
or `PreparedEndpoint`, and no registered score/targeting/covariance stage that
assembles a `ReportBundle`. These APIs remain callable directly; the synthetic
fixture explicitly assembles their records. That fixture is not a reviewed
production adapter. The nested and coverage runners are CPU-only; coverage
batches complete repetitions and has no cross-leaf interrupted-repetition
continuation. Neither limitation authorizes shortened tuning or assumed GPU
performance.

Production admission requires the design implementation/registration to accept
the merged runner contract and bind the real typed inputs. Direct API tests do
not establish an executable coordinator path. Continue offline verification of
the scientific APIs and plan-integrity checks while retaining these production
blocks. Reassess admission after merged corrections; the early atlas/support
path must remain independent of this documentation/integration unit.

## Coordinator contract

A code unit is **DONE only with its merged receipt** for the approved repository
and target branch `dev`. A pushed branch, open PR, passing review or local
`completion.json` is insufficient. Code dependencies therefore provide merge
barriers. Code-attempt directories are never acquisition inputs.

Every repository-running Slurm unit, including collectors and continuations,
starts with the fixed prefix emitted by `execution.campaign.stage_command`:

```sh
set -euo pipefail
: "${SWARM_UNIT_DIR:?}"
export GIT_NO_REPLACE_OBJECTS=1
git clone --depth 1 --branch dev https://github.com/goodarzilab/OxyFormer.git "$SWARM_UNIT_DIR/src"
git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SWARM_UNIT_DIR/src/src"
cd "$SWARM_UNIT_DIR/src"
```

The registered command then uses that clone's CLI with `--repo`, `--out`,
`--deps-env`, explicit task selection and the clone's owner registry. The
`--prepared` shell path is only for a worker that already performed the same
clone/identity preparation; it is not permission to reuse an unreviewed tree.

Cloning `dev` selects the launch revision; it does not override the locked
scientific identity. Before invoking a locked stage, `execution.runner` calls
`verify_recipe` and requires the clone's `tracked-science-v2` fingerprint to
match the recipe. It also checks the scientific fingerprints of locked upstream
attempts, including transitive dependencies. The fingerprint covers tracked
files except `*.md` outside `src/`, `scripts/` and `docs/plan/`. Each clone's
full commit is still recorded and bound to its stage request.

If an ordinary merge advances `dev` with changed source, config or frozen-plan
content, a later leaf refuses with `recipe scientific code/config drift` before
producing scientific output. Do not combine results from different scientific
fingerprints or rewrite the old lock. A changed scientific recipe requires a
new prospective lock, timing and authorizations. A documentation-only merge
can preserve the scientific fingerprint and pass the generic lock check;
portable nested continuation additionally binds its full request code identity
and may refuse that change. A clone prefix alone is not evidence that either
identity check passed.

Only normalized dependency variables resolve upstream work: for example,
`atlas-mid-atlantic` becomes `SWARM_DEP_ATLAS_MID_ATLANTIC`. IDs must not collide
after normalization. Never search run directories for the newest output or
guess an upstream path. Verify producer IDs, passing receipts, declared files,
hashes and immutable attempt fingerprints. A stage receipt's `pass` status is
required in addition to `StageResult.verify(request)`.

Acquisition executes merged source manifests and reviewed adapters. It retains
source receipts and seals its own attempt; consumers never seal an upstream
attempt. The early fetch → nine-division atlas → atlas collection → tract
support path must not depend on `release-docs`, `campaign-lock` or final release.
Check the external coordinator DAG as well as the repository registry; local
registry tests cannot certify coordinator state.

## Scientific handoffs

`StageRequest` uses `dependency_paths`, `dependency_hashes`, `output_dir` and
`code_identity`. These merged names supersede older prompt shorthand. Each
request binds absolute input paths and hashes; artifacts use confined relative
paths. `pass`, `blocked` and `fail` are distinct. Never consume a failed or
blocked stage as a satisfied prerequisite.

The test chain computes pressure before population aggregation, runs the sealed
design/support gate, loads its frozen target and splits, fits the complete
five-fold × three-seed schedule, resumes one interrupted fit from a portable
checkpoint, assembles OOF predictions, joins outcomes for one-step/CV-TMLE,
aligns covariance by original ID, and emits a diagnostic report. It also tests
missing atlas coverage, a missing OOF row and a changed target. All data are
synthetic. One-epoch test fits cannot establish production calibration.

`PreparedEndpoint` is the fitting boundary supplied by reviewed adapters and
design. It binds data, sources, entity graph, geography, policy, treatment
knots, outer/inner splits and permitted feature families. Coordinates remain
in split/inference metadata, never predictor views. The same original ID,
fold, seed, origin weight and estimand must survive prediction assembly.
Seed averaging does not create independent observations.

A successful nested slice may contain only `continuation.tar`, `progress.json`
and `artifact_manifest.json`. Require `progress.complete == true` before
consuming terminal `nuisances.parquet`, `model_bundle.tar` and `metrics.json`.
Resume validates model/optimizer/scheduler/RNG/sampler, preprocessing,
calibration, fold/reference identities, code/environment/source/recipe hashes
and progress. New slices get new attempt roots and consecutive predecessor
links; they may not shorten tuning or restart from an approximate state.

## Offline verification

The commands below use this lab's installed interpreters and Slurm settings.
On another host, provision and verify the equivalent environments and use its
site-approved scheduler settings; these absolute interpreter paths are local
to this run. Run from this checkout on a compute node. The login node is for
editing, Git, GitHub and review-gate calls only. No tests, mutations, probes or acquisition
reads belong there. This command preserves the required inner test command:

```sh
srun --partition=standard --account=root --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=32G --time=01:30:00 \
  timeout 3600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -B -m pytest -q tests
```

Before every review round repeat the complete suite with
`/mnt/weka/home/hgoodarzi/oxyformer-swarm/envs/ci-cpu/bin/python`. The owner
replaced the obsolete ten-minute complete-suite limit with a 3,600-second
timeout; the measured eight-CPU full runs in this attempt took approximately
32–33 minutes. Record failures and build-specific skips; do not relax tolerances to hide CPU float32 differences.
Also run `tests/test_end_to_end.py tests/test_plan_integrity.py` **without `-B`**
in both environments before review, matching CI's ordinary interpreter mode.
Do not make tests depend on bytecode flags. The integration additions must add
well under a minute; shortened synthetic timing cannot admit production work.

For the required mutation, use an isolated copy of
`test_final_coverage_collector_dependencies` with its repository root preserved.
Remove a leaf from the expanded final collector's `needs`, run that test, and require the
`collector omitted required leaf dependency` failure. Revert the injection,
rerun the test and complete suites, and record both results outside Git.

Review the complete delta from the coordinator-recorded base with the author
excluded. Commit/push the anchored attempt branch, open a PR into `dev` only
after REVIEW_PASS, and stop; the worker never merges or enables auto-merge.
Write the declared evidence into `$SWARM_UNIT_DIR`, with `completion.json`
last. That code receipt does not admit a scientific campaign.

## Production nuisance cost diagnosis

A quiet tuning directory does not establish a stalled fit. The nested controller
retains minibatch state in memory and publishes an enclosing checkpoint at a
slice boundary. Inspect its active epoch, step, sampler cursor and counts before
attributing elapsed time to checkpoint I/O.

The real-frame fold-0/inner-0 outcome grid-0 fit uses 19,434 labelled fitting
rows, including 837 target rows, and 76 original-record minibatches per epoch.
The A2 tract-design amendment uses the A0 architecture; it is not the A2
no-context ablation. In bounded pre-repair profiles, repeated construction and
validation of leave-self-out views dominated CPU time, while repeated scalar
reference preparation dominated H100 host time. A 75-reference, 30-feature
context call made 2,250 host-to-device scalar transfers. No per-minibatch durable
checkpoint writes appeared in those supervised slices.

`CountyContext` now retains only parameter-independent prepared reference
inputs. Rows are prepared lazily with the existing scalar tokenizer, preserving
its conversion and overflow behavior and leaving excluded values untouched.
Each query gathers its own ordered rows before running the unchanged encoder
and attention graph. Optimizer steps still recompute all learned activations
and gradients. Preprocessing, device or dtype changes invalidate the prepared
inputs. Immutable preprocessing objects are compared by identity: categorical
`1` and `1.0`, or signed-zero means, must not collide through Python value
equality. Inference-mode preparation retains ordinary tensors usable by later
autograd calls. These inputs are private runtime caches, absent from checkpoint
tensor state. Neither offset-profile pass is removed.

The owner-authorized equivalence criterion compares scientific values and
decisions exactly while retaining truthful production provenance. A source edit
must change `scientific_code_hash`; enclosing checkpoint parent links also
change. The attempt-local canonical comparison checks full SSL training state,
both nuisance tensor states, tuning/epoch decisions, counts, preprocessing,
calibration, audits and predictions at A and d(A,X). Its only removed provenance
paths are the nuisance and four SSL identity source hashes, the two final
bundles' validated `ssl_hash` parent-descriptor links, the prediction's
validated enclosing-checkpoint parent link, and explicit checkpoint predecessor
links (including a matching final SSL lineage parent when present). Artifact paths, ZIP timestamps and elapsed compute are retained as separate transport/measurement metadata, not
scientific values. Checkpoint format remains version 1 with uncompressed tensor
members; archive byte counts can change with embedded paths and elapsed-time
text. Production fingerprints and resume identity checks are unchanged.

The registered recipe remains FP32 with BF16 disabled, the same deterministic
backend settings, original batch size 256, four grid points, 150 frozen epochs,
three inner folds, five outer folds and three seeds. Full-fit and campaign
projections from bounded profiles are extrapolations. They cannot establish
campaign feasibility or authorize different epochs, stopping rules, precision,
batch semantics, profile budgets or the four-GPU-hour leaf cap.

### Opt-in recipe cost study

`oxyformer.training.fit.RecipeStudy` is an explicit study parameter object for
`FitConfig(study=...)` and `PreparedEndpoint.configuration(..., study=...)`.
These are candidate amendments for an owner decision, not registered values.
Campaign configuration, approvals, profiles, scenarios and allocation caps do
not consume these overrides. `coverage.estimate_repetition(..., study=...)` is
available to a separately invoked study; registered callers omit it.

| Parameter | Default (existing behavior) | Study choices |
| --- | --- | --- |
| `reference_cap` | `None`: all permitted labelled county references | 32, 64, 128 (positive integers supported for synthetic diagnostics) |
| `frozen_epochs` | `None`: inherit `NuisanceSettings.frozen_epochs`, registered 150 | 150, 60, 30 |
| `grid_points` | 4 | 2: retain both registered learning rates, each at dropout 0 |
| `seed_count` | 3: 1103, 2207, 3301 | 1: retain 1103 |

The reference cap takes the first K sorted `original_id` values in the permitted
county after removing the query itself. It uses no outcomes, exposure, weights
or random state. This preserves the existing ordering and keeps precise geography
outside the nuisance interface (plan 3.1); it is not nearest-distance selection.
It caps each context call only, retaining all fitting rows, preprocessing/SSL
ownership, both heads and county offset passes. The selected cap travels in model
bundles so checkpoint reload and prediction use the same rule. In the two-point
study, grid indices are 0/1 for learning rates 0.0003/0.001; registered indices
remain 0/1/2/3. All three inner and five outer folds remain in repetition studies.

Frozen fitting has no stopping score, resets its bad-epoch counter, and reaches
its frozen epoch count. This study does not add patience-based stopping. Explicit
training-local stopping sets retain the registered maximum 150 and patience 15;
outer refits use the median selected inner epoch. SSL retains maximum 30 and
patience five. A frozen-epoch override does not shorten explicitly stopped fits
or SSL. One-seed studies average only the selected neural seed and bind the
aggregate scoring split to that seed; original fold manifests remain unchanged.
Neural seeds are not simulation repetitions. No final replication count changes.

Default study fields are omitted from scientific identity and bundle payloads,
preserving existing scientific values and decisions; changed package source has
its truthful new fingerprint. Nondefault settings enter scientific identity, so
a different study cannot resume a registered nuisance checkpoint. Default
base/current comparisons allow only source fingerprints and validated enclosing
checkpoint parent links, using the diagnosis's canonical scientific comparator.
Study artifacts, timing slices, projections and candidate prediction differences
belong in the attempt directory, never in this repository. A cost projection is
not campaign admission or evidence of statistical adequacy. Candidate C (target
plus supported training rows) is not implemented: the prior attempt's registered
`PreparedEndpoint.validate` rejected all five folds under plan 5.1 minima.

### Batched county context and numerical comparison

County context constructs sorted, fold-owned reference IDs per query, removes the
query before applying an optional RecipeStudy cap, and retains the raw-input
cache. Empty reference sets return four zero tokens. In evaluation and when all
context dropout is disabled, the encoder processes the deduplicated union once.
PMA gathers each query's reference states and masks padding. PMA workspaces are
bounded at 262,144 reference slots per tensor batch (a single larger reference
set is retained whole); this is an execution workspace limit, not a reference
cap or a change to the original-record minibatch size.

Training with nonzero dropout instead encodes separate `(query, reference)`
occurrences in one call. Masks are drawn independently in the original query
order, with the original tensor strides, then applied in batched layer order.
This retains the original RNG stream and Bernoulli keep probabilities. The
replay comparison on CUDA exceeded the forward tolerance at one near-zero
context element (absolute difference 1.27e-6); the tolerance was not loosened.
The owner subsequently accepted PR #63's three documented exception categories:
that context element, intermediate base-only prediction drift after optimizer
updates, and the two-grid/dropout-zero calibration differences. This acceptance
applies to PR #63 only; subsequent changes must report every new strict exception
against the current path for a new owner decision. Seed-paired short fits retain
the actual numerical and decision differences. Native layer-first mask generation was also evaluated, but failed
the existing extreme-weight invariance regression, so original-order mask
replay is retained.

The registered CPU random stream on CUDA, FP32 arithmetic, deterministic kernels,
seeds, grid, epochs, stopping, loss normalization and optimizer remain in force.
Padding is masked and empty contexts are zero. The private scalar `_pool`
implementation remains a comparison oracle; ordinary prediction and training
use the batched path.

Both identity-outcome offset passes still predict the whole fitting partition
in eval mode, once before and once after each epoch. They compute both doses,
then profile county offsets from factual-dose residuals with FP64 reductions.
Model-output caching remains restricted to eval with gradients disabled and
binds actual model bytes, reference IDs/content, preprocessing, split, checkpoint
and parameter version. Source changes produce a new scientific code hash; old
production continuations must still pass the existing identity checks.

The owner-approved 2026-10-11 comparison criterion for this change is elementwise
forward relative tolerance `1e-5` with absolute floor `1e-6`. The orchestrator's
clarification permits gradient relative tolerance `1e-4` with absolute floor
`1e-5` on matched minibatches. Test assertions use the conventional
`abs(actual-reference) <= atol + rtol*abs(reference)` bound and report maximum
absolute and raw relative differences separately. The orchestrator clarified that continuous fitted quantities, including affine
calibration coefficients, county offsets, logits, means and losses, use the
forward tolerance. Discrete tuning and epoch decisions, counts, preprocessing
categories, stopping IDs and audit outcomes must remain identical. Repeatability
requires matching tensor bytes on repeated runs under the same deterministic
settings. A comparison may normalize explicit enclosing-checkpoint provenance
links only after validating them; it must retain the differing source hashes in
its report and must never bypass a production resume check.

Run the fit/nested-CV/context/token tests with and without `-B` in both the shared
CUDA-build environment (with CUDA hidden) and the CI CPU-only environment. Run
CUDA comparisons on an H100 and the full suite as CI does, without `-B`. All tests,
synthetic reproductions, real-frame reads and timed measurements run on Slurm
`standard`/`root`. Bounded cost slices report cold setup separately, match ordinary
minibatch step IDs, distinguish measured epochs from projections, retain the two
offset-pass costs and include the registered and requested RecipeStudy settings.
An incomplete equivalence check or a different stochastic realization must be
reported explicitly; deterministic repetition alone is not equivalence.

### Fit preparation cost profile

The next profile isolates immutable preparation from the already batched model.
Use the same saved fold-0/inner-0 outcome grid-0 SSL initialization, observations,
seed 1103, 19,434 fitting originals, 30 features and 76 minibatches per epoch.
Keep the 1,200-second supervised slice and record cold setup separately. Run
cProfile over that slice and a separate torch.profiler capture of an ordinary
256-original minibatch. Report host wall ranges, device kernel duration, kernel
launch counts and synchronisation separately: asynchronous GPU durations must
not be added to the host ranges that contain them.

Record the parameter count of the instantiated nuisance model separately from
the saved SSL descriptor. In these slices the SSL checkpoint reports 110,238
parameters, while the complete outcome nuisance has 193,143 in both the baseline
and candidate; adding context, treatment readout and county offsets changes the
object being counted. A shared initialization does not make these counts equal.

The implementation now creates one fit-local covariate index and parent content
hash, then uses ordinary validated CovariateView and lineage constructors for
every sampled subset. A second fit-local object applies the authoritative frozen
policy once to fitting inputs and selects its original results by ID. Reuse
requires the identical policy object and matching exposure bytes, geography and
support key. Changed inputs, including signed zero, or records outside the fit
use the policy API again. These objects contain no learned states, never cross
fits and do not enter checkpoints. Each resume reconstructs them from validated
inputs; production source/resume identity checks remain in force.

Token preparation keeps the Python scalar standardisation and canonical category
lookup, including distinct numeric/string/boolean categories, missingness and
signed zero. Numeric values use the original scalar PyTorch assignment into a
CPU buffer in the destination dtype, then transfer the complete buffer once.
This retains the native scalar rounding, subnormal and overflow behavior for
every value instead of duplicating selected parts of that conversion with a
Python-list tensor constructor. Category and missingness arrays still transfer
in bulk. Raw target weights retain their separate existing FP64 arithmetic;
they never enter the numeric token buffer. Encoder, PMA, dropout draws,
loss reductions, clipping, optimizer updates and both full offset passes retain
their existing mathematics.

Before review, run the required matrix on Slurm `standard`/`root`, eight CPUs,
with each of these interpreters and with both ordinary mode and `-B`:

```sh
timeout 1200s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -B -m pytest -q \
  tests/test_fit.py tests/test_nested_cv.py tests/test_context_invariance.py tests/test_tokens.py
# Repeat without -B, and repeat both commands with:
# /mnt/weka/home/hgoodarzi/oxyformer-swarm/envs/ci-cpu/bin/python
```

Run the complete suite once with the CI CPU wheel and CI's ordinary interpreter
mode, inside an eight-CPU Slurm allocation:

```sh
timeout 3600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/weka/home/hgoodarzi/oxyformer-swarm/envs/ci-cpu/bin/python -m pytest -q
```

Use separate H100 allocations with `--gres=gpu:1`, each limited to at most one
hour and all GPU allocations totaling less than four hours, for real saved-state
replays, synthetic nested-fit comparisons and repeated GPU tests. Compare every
captured intermediate output, loss and gradient as well as full-partition means,
offsets, calibration, discrete decisions and repeated scientific bytes. Preserve
every exceeding element with its reference value, candidate value and bound;
historical PR #63 exceptions confer no waiver. Verify source fingerprints and
enclosing provenance links before normalizing only transport/identity metadata.

The conversion regression grid compares storage bytes against scalar assignment
and Python binary64 subtraction/division followed by an explicit dtype cast.
It includes the smallest subnormal, `1e-311`, FP32 subnormals, signed zero,
`1e307`, negative counterparts and both sides of the FP32 finite boundary.
Run the unchanged extreme-weight complete-fold regression as well. Diagnostic
CPU-dispatch and flush-to-zero probes belong in isolated Slurm processes; they
must not change the production deterministic or precision settings. A cluster
pass does not replace checking the actual GitHub PR run on its own CPU.

Keep `fit_cost_profile_2.json`, `equivalence_report.json`, `pytest.log` and the
job receipts in the attempt directory. Cost projections retain the previous
reference-row workload scaling and SSL proxy, with explicit limitations, for the
registered recipe and the K=32/E=30/G=2/S=1 and K=None/E=60/G=2/S=1 study cells.
Report the 5,400/1,700 H100-hour comparisons and
`floor(5400 / (scenario_count * repetition_hours))` for 4, 2 and 1 scenarios.
This arithmetic does not amend the frozen minimum of 1,000 repetitions per final
scenario or authorize a campaign or a larger leaf.

### Accelerator identity and existing artifacts

CUDA environment identity records the device model and architecture directly
from `name`, `major`, `minor`, `total_memory`, `multi_processor_count` and
`L2_cache_size`. Physical-card UUID and PCI identifiers are excluded, so Slurm
may resume a continuation on another card with exactly the same recorded model
properties. All other environment fields and exact checkpoint, fingerprint and
continuation comparisons are unchanged; different models remain incompatible.

There is no compatibility shim or migration for artifacts written before this
fix. Checkpoints also bind `scientific_code_hash`; coverage and portable nested
continuations bind the request's `code_identity` (the Git commit). Those existing
code bindings already reject pre-fix artifacts. The orchestrator must restart
the screening profile chains from slot 0 after merge; this change does not
consume their old checkpoints or continuations.

### Coverage-owned continuation chains

Coverage work may use the generic campaign `slices` chain. A work item that
uses chained profiling declares `continuation.tar` on **every** slice, including
the terminal slice. Its immutable parameters retain the draws, recipe, scenario,
wall budget and optional `RecipeStudy` cell. The owner/step/predecessor fields
come from the generic expansion; no new instance type or scheduler is involved.
Archive-free single-leaf execution remains available. Budget evidence from both
legacy and chained producers must now pass the runner-finalized loader.

The coverage archive binds draw IDs and simulation seeds, scenario payload,
endpoint/frame/observation hashes, recipe, study cell, lock, code identity, wall
budget and scientific/environment fingerprints. Import resolves only the named predecessor
from `StageRequest.dependency_paths`; the executor authenticates its declared
outputs. Digest-named archive members carry the original observations/truth
receipts, completed `(fold, seed)` prediction parts and the unfinished fold's
nested-CV continuation. Embedded old checkpoint paths are never opened. Each
predecessor must have the same owner and exactly the preceding step. Binding
drift, gaps, reordered parts and corrupt blobs are refusals, never new draws.

A bounded coverage leaf uses the shared compute budget described below,
reserving measured export/publication costs, the safety factor, and
a finalization reserve of at least 300 seconds. The in-fold checkpoint margin
is measured separately, inside the compute deadline. The 14,400-second bound
applies to the complete allocation, including setup, not a fold's cumulative
training. Epochs, tuning, seeds and estimators are unchanged. Only after all
prediction parts exist does the usual terminal aggregation run. Incomplete
repetitions have no `repetitions/<id>/result.json`; a valid checkpointed slice
passes execution while its batch summary remains incomplete. If a leaf crashes
before export, retry in a fresh attempt root with the same task and exported
predecessor. Never consume partial files in the failed root.

The runner finalizes exactly one `budget_record.json` after the final wall
check. Its terminal `StageResult` lists that record and its SHA256. The record
binds the request, declared wall, task and continuation owner/step, recipe, lock,
scenario and draws. It contains reconciled per-leaf compute, setup, import,
export and publication measurements, outcome counts, and chain totals.
`coverage_measurements.json` is worker input to finalization, with no downstream
authority. `result.json`, `timing.json` and `runtime.json` remain diagnostics and
scientific payloads; their budget flags cannot authorize work.
Execution and finalization classify a chain from the producer task's archive
output or predecessor, preserving the archive-free generic step-zero path.
The same task supplies the effective compute contract, including a bootstrap
ceiling; worker scope tags and copied contracts cannot enlarge that allowance.

`coverage.load_budget` is the only budget-evidence loader. It requires a
passing terminal result and the existing independent publication seal, verifies
the declared producer request/task and record hash, checks the wall/bindings,
and recomputes eligibility from measurements and counts. Within this same loader,
the recorded chain mode and per-leaf contracts must match the producer task,
and the hashed scientific recipe and outcomes must agree with the binding and counts;
diagnostic summary flags remain irrelevant. Missing, truncated,
pre-seal, over-budget and reused evidence is refused. Stored eligibility,
complete and success flags are not trusted. Scientific payload reads follow
this gate. Authoritative profiles, leaf deadlines, predecessor import, collection and
ordinary dependency reuse/retry use this loader. Measurement-only calibration
is the explicit sizing exception described below. A lock
carries the profile record identity and declares its original sealed inputs so
a leaf can revalidate the measurements before spending its compute allowance.

A partial or unsuccessful profile chain cannot justify a lock. Chained sizing
uses one capacity calculation. The execution window is
`C = wall_seconds - 300 - sum(overheads)`; the safety factor is recorded for
provenance but never shortens execution. Sizing uses
`usable = C - checkpoint_margin_seconds() - max(PHASE_UNIT_SECONDS.values())`
and `net = usable - resume_overhead_seconds`. With today's declared bounds the
margin is 690 s, the partial-unit allowance is 330 s and the checkpoint write
bound is 30 s. Net capacity must fit a 330 s unit plus its checkpoint or sizing
refuses. For `r` repetitions, `DEMAND = r * seconds_per_repetition +
publication_verification_seconds` (the last term is paid once per chain), and
`slices = ceil(safety_factor * DEMAND / net) + 1`. The spare counts against the
same allocation and forty-leaf cap. The resume constant is shared across
scenarios: the maximum completed-first-repetition excess per leaf over the
profile rate, with a minimum of 810 s. This floor raises the required 800 s
minimum above the final null-effect snapshot's 809.556315 s excess per leaf.
Collection still counts the last leaf's accumulated terminal records once,
retains all draws in denominators, and lists incomplete chains.

Screening estimate and lock accept `setup_calibration`, a list of
`{dependency, path: budget_record.json}` references, one per sealed campaign
leaf. Each record contributes its **last** leaf, avoiding repeated chain
history. Setup is `max(1.25 * P90, max)`; the other phase allowances are the
maximum of campaign calibration and profile measurements. The dedicated
`read_leaf_measurements` verifies the budget hash and publication receipt and
reports its generation. It grants no execution-budget authority. Only estimate
and lock may declare these measurement dependencies; the runner retains all
publication/hash/fingerprint checks, skips only their budget-authority load,
and publishes `measurement_only_dependencies.json` in the consumer result.
Profile chains and ordinary budget consumers retain their existing checks.

Without calibration records, estimate and lock require one or two receipts from
`campaign-setup-probe` in `calibration` mode. Its inputs are the four terminal
profile receipts and real-frame inputs, never an estimate or lock. Receipt
input digests must match the inputs being sized. The smaller completed draw is
used: setup is `max(probe + 60 s, profile allowance) * 1.5`. All draws, the
chosen draw and its ratio to the profile allowance are recorded. The estimate
reports requested counts beyond forty and recommends a second draw when only
one exists. The fallback deliberately refuses a batch that fits only without
headroom; the refusal names the unpadded count and the remedy of declaring
sealed campaign-shaped calibration records. Further draws after two need an
owner decision.

Screening locks no longer perform in-process admission. They publish immutable
outputs with `gate.json.admitted: false`. The separate CPU
`campaign-setup-probe` in `admission` mode measures clone, environment startup
and the runner's full verification of the first expanded leaf's input set,
including lock outputs and excluding its not-yet-created admission receipt.
The launcher sets `OXYFORMER_SETUP_STARTED` before cloning. The new
`admission-resumed` mode measures that base plus a synthetic predecessor at
`depth: 1` or `depth: 2`. Each logical probe runs in two separate jobs, with
separate clones, requests and publications. Its final receipt carries both
`setup_draws` and their minimum, `setup_seconds`. One slow repeat cannot raise
the minimum. Every receipt binds the lock, estimate, scenario and base input
digests. The second draw additionally authenticates the first draw's receipt.

The lock's `expanded_units.json.admission` declares eight setup jobs (six draws
and two synthetic predecessor builders) followed by `campaign-admit`.
`task_manifest.json.admission_tasks` declares their inputs and outputs. The
builders run on CPU in the probe stage's internal `admission-predecessor` mode.
They require sealed profile leaves 1 and 2 from the same scenario, scientific
tree and environment as the final lock. Depth 1 uses the fresh probe's sealed
dependency check, bound through its request snapshot and current publication;
depth 2 uses the depth-1 probe's check, including the synthetic grand-predecessor.
The profile's old dependency receipt never substitutes for the final lineage.
A resumed probe refuses `reuse-miss` when the re-walk exceeds the one added
predecessor publication, and requires positive ancestor reuse.

`campaign-admit` requires three final two-draw receipts: `fresh`, `resumed_1`,
and `resumed_2`. It checks equality of batch-level spec `inputs` (per-leaf
`needs` differ), all receipt bindings, shape evidence and reuse results.
Allowance is `max(fresh minimum, resumed-1 minimum, resumed-2 minimum) + 60 s`.
It applies the fit predicate below to every locked batch. Estimate and lock
outputs plus the serialized admission candidate must also fit within 50 MB.
Refusal never resizes or mutates the lock. Ordering remains estimate, lock,
probes, admit, leaves; admission measurements never feed back into sizing.

`admission.json` is the only authority for screening admission. Every expanded
screening leaf and collector declares it as an input and verifies `admitted`,
`lock_hash` and `estimate_hash`. This is a deliberate leaf dispatch contract
change. The lock gate remains false permanently. Specs without an
`admission_receipt` (including bootstrap profiles and final `build_lock`)
retain their expansion contract. Final `build_lock` shares the new window and
slice arithmetic but retains its existing in-process admission.

The three probes sample the first expanded scenario; shared spec inputs cover
the common base lineage. Each scenario's first resumed leaf independently
checks its own locked publication shape. No throughput guarantee is claimed.
The scientific fingerprint freeze is unchanged: old profiles require
an owner-approved re-profile on the final tree before a new estimate/lock run.
The sealed fixtures prove sizing; CPU probe exercises can verify the old
campaign's dependency shape without authorizing its science.

`budget.expected_predecessor_shapes` records archive bytes and regular-file
count from each scenario's first sealed profile publication and expanded leaf
outputs. Each early profile must match the terminal profile's chain owner and
recipe, and its current publication hash must appear in the terminal profile's
declared inputs or request-bound dependency snapshot. Equal task IDs or measured
times from another attempt do not establish this relationship.
For a fast scenario, archive bytes include one copy per potentially
started repetition: `min(draw count, max(1, ceil(sized usable compute / measured
seconds per repetition)))`. Record count replaces the profile's individually
published repetition records with one slot per potentially completed draw.
Synthetic archives cycle the appropriate authenticated profile archive's bytes
to the locked byte count (a final prefix is allowed); additional record slots
copy authenticated profile result metadata. These are hashing workloads, never
scientific continuation inputs. Their stage, ownership and budget bindings
cannot authorize a scientific successor. A missing same-tree profile refuses
the probe. The sealed synthetic publication must be within 10% of the locked
shape in both dimensions. The first real resumed leaf compares its predecessor
against that **same** locked reference: above 10% it records
`admission-shape-exceeded` and runs; above 50% it writes an unsealed diagnostic
budget with `admission-shape-mismatch` and refuses before continuation import
or compute. Exactly 10% and 50% are inclusive. Smaller real publications are
allowed. These are conservative size estimates, not forecasts of payload
content or throughput; scenarios whose first-leaf publication exceeds this
envelope require remeasurement.

### Setup projection and runtime stop policy

The shared `projected_setup_allowance` implementation serves estimate, lock and
admission. Estimate groups calibration records by chain origin, scenario,
recipe, lock, chain owner, scientific fingerprint, environment and budget
generation. With at least three resumed observations in one group, growth is
the nonnegative median of **all** pairwise slopes using actual leaf indices.
For longest expanded chain length `L` and last measured index `n`, projected
setup is `max(observed resumed setups) + growth * max(0, L - n)`. Allowance is
`max(fresh setup, projected setup) + 60 s`. Fewer than three resumed observations
are explicitly `unmeasured`. Lock construction recomputes the same estimate.
The unchanged fit predicate is `window - allowance - resume reserve >= sized
usable compute`. A refusal names fresh, points, growth, projected setup, `L`
and the allowance ceiling. Admission uses the same function without extrapolation;
its two-depth difference is recorded only as informative growth.

At the reviewed month-1 ceiling of 4,629 seconds, fresh 2,811 and resumed
5,599 / 5,792 / 6,320 project to 16,774.5 at `L=33` (growth 360.5), hence refuse
at allowance 16,834.5. A concave 3,000 / 3,600 / 4,000 chain also refuses,
even if it might later plateau. An end-point 2,800 / 2,800 / 3,100 transient
refuses at `L=33` but fits at `L=10`. This is an explicit conservative policy;
three observations cannot distinguish a future plateau from continuing growth.
Late growth may still pass projection and is bounded by the runtime stop rule.

New screening locks declare `budget.setup_safety_net`. Each resumed leaf's
measured setup is an overrun only above the admitted allowance plus 10% of the
chain's **actual fresh setup**. Two consecutive overruns or three in total
cause the next leaf to refuse before continuation import or compute, writing
`allowance-exceeded` in its unsealed diagnostic budget. The collector reports
the chain as stopped and refuses to seal. Thus at most three overrunning
leaves (12 GPU-hours) can be spent, and sustained growth stops after two.
Budgets retain their formula, generation, owner factor and compute accounting;
these diagnostics never grant execution or reuse authority. The science,
continuation semantics, fingerprint freeze, 40-leaf and four-GPU limits remain
unchanged.

The bounds remain 4 GPU-hours per leaf and 40 leaves **per campaign spec**.
Screening and final instances are still formed as before, and all generated
resources remain `standard`/`root`. Per-scenario instances and month-1 sizing
belong to the subsequent campaign-binding unit.

The synthetic equivalence regression runs one reference repetition and a chain
of three leaves, cutting inside a fold and between folds. It compares floating
values by their binary64 bytes and records discrete fields separately. Only the
explicitly enumerated timing/provenance fields in its report are excluded.
`OXYFORMER_EQUIVALENCE_REPORT` selects the external report path;
`OXYFORMER_CHAIN_DEVICE=cuda` selects the single-device H100 check. CPU is the
default. Run the required coverage/campaign/profile/nested-CV files with and
without `-B` in both the shared and CI CPU environments, plus the full CI-style
suite, on Slurm. Tests use synthetic inputs and sealed budget measurements, with no network access.

### Shared coverage leaf budget

The month-1 screening safety factor is read from the owner decision
`owner_decisions.screening_month1_safety_factor` in `configs/approvals.yaml`.
Both profile task emitters use that approval. `campaign-estimate` and
`campaign-lock` reject a task factor that differs from their dispatched approval;
screening validation also requires that decision to match the repository's.
Their `budget_estimate.json` and `budget.json` record the normalized decision as
`owner_safety_factor_decision`. Changing the factor requires an owner decision
recorded by the orchestrator; workers never edit the approvals file. Bootstrap
profile chain tasks and their sealed receipts are unchanged by this sizing input.

For coverage-owned chains, the lock stores `leaf_budget` in each batch and its
expanded tasks. Sizing, locked preflight admission, and execution all use
`C = wall_seconds - LEAF_FINALIZATION_SECONDS - sum(overheads)`.
Overheads name setup, import, export, and publication separately. The immutable
lock and continuation binding authenticate this compute ceiling. Sizing subtracts
the declared in-fold checkpoint margin, partial-unit allowance and resume cost
from C on every leaf, since fitting stops before that ceiling. The worker still meters all actual compute against C;
unused headroom cannot enlarge a successor's sealed allowance.

The whole leaf shares one compute deadline. Setup and import are each charged
once; each positive overrun reduces C and underruns never increase it. Export
and publication remain reserved, along with the checkpoint margin. The safety
factor multiplies sizing demand only. A compute overrun refuses usable publication. Observed finalization or
wall overruns fail the leaf and cannot justify a profile lock. The collector
refuses over-budget leaves. Checkpoint boundaries are cooperative;
uninterruptible operations that overrun are detected, not certified as within
budget. The scheduler remains the hard wall limit.

Bootstrap chain profiles need no prior measurement. They may declare an
immutable `compute_ceiling_seconds` below the wall minus finalization reserve;
otherwise that bound applies. Their separate phase measurements supply the
next lock. Partial, unsuccessful, or known-overrun measurements never admit a
lock. Existing archive-free single-leaf producers retain their legacy behavior:
generic step-zero metadata alone does not promise an archive. New locks built
from chained profiles explicitly declare `continuation.tar`. Malformed archive
declaration hardening is deferred to a follow-up.

The final wall measurement includes worker return and runner verification.
Post-check record writing and terminal publication use `min(30 seconds,
wall_seconds / 20)` of the existing checkpoint/finalization reserve; the final
check requires elapsed plus that reserved interval to fit the wall. This
allocation does not enlarge compute or change the shared C formula. Actual
post-check duration is not claimed as measured in a record written before it.
Slurm enforces the allocation's hard limit. Publication first writes provisional
bytes and then the independent seal; no terminal result is changed after that
seal. A provisional passing result without the seal admits nothing.

The worker's bounded deadline and final wall check, the runner's finalization,
and the sealed-budget loader share the pre-publication limit
`wall_seconds - min(30, wall_seconds / 20)`. At a 600-second wall, elapsed 570
passes the wall check and 571 fails; the reserved publication time remains inside
the existing finalization reserve. This check does not enlarge the separate
compute ceiling or the usable capacity used by sizing. Legacy
CPU profiles retain their planning-only training deadline and must pass the same
final budget check to supply usable measurements.

An omitted or explicitly null `leaf_budget` selects bootstrap accounting in
execution, finalization and loading. Only a non-null measured contract charges
export/publication phase overruns against compute. The sealed scientific payload's
scenario (with registered defaults expanded), mode and recipe hash must match the
producer task's identities; profile admission then requires those same identities
to match the scenario and recipe being locked.

Month-1 screening binding (2026-10-13 owner decision): the real-frame stage now
publishes `screening_profile_tasks`, using the recorded 32/30/2/1 study cell.
The existing `profile_tasks` helper remains the production profiling interface
for separately authorized later work. Missing or malformed repository or
dispatched screening decisions, including `month1_sizing`, are refused; this
code never writes `configs/approvals.yaml`.

Bootstrap profiling declares one draw per scenario across 40 sequential slots
of at most four GPU-hours. Once complete, the same terminal record is carried
through remaining slots without fitting or drawing again. If still incomplete
at the terminal slot, estimation refuses the profile. Estimation also refuses a
reference to an earlier completed slot: its receipt must account for all 40
declared bootstrap slots. Every slot declares the
coverage-owned continuation archive and finalized budget receipt. Each scenario
has one sequential chain; at most four single-GPU profiles can run together.
The estimate task depends on all four terminal receipts and admits nothing.

After reporting the estimate's per-scenario counts to the owner, the
orchestrator constructs a separate `campaign-lock` task from the estimate task's
parameters, adding `campaign_id`, `evaluation_namespace`, `retry_rules`,
`prior_locks`, `cpu_wall_hours_budget` and `estimate_input` (a declared dependency
reference to `budget_estimate.json`). Do not supply manual repetition counts or
batch sizes. Counts are maximized against the recorded four-hour leaf cap. A
shorter requested wall is accepted only when it can schedule those same counts
within forty leaves per scenario; it cannot silently reduce them.
The lock rechecks the report, records its SHA-256, and measures
setup on the expanded leaf shapes before admission. Setup beyond the measured
profile allowance is refused and requires new measurements.

The resulting four campaign specs each contain one sequential continuation
chain and its collector. Their combined dependency graph holds at most four
GPUs. Collectors use CPU and verify the locked scientific code and sealed
profile receipts; they do not fit or require the accelerator environment. Leaf
reservations sum to at most 640 GPU-hours, checked against the 3,600-hour ceiling. Profiles and retries
remain separately accounted work. All task resources and expanded Slurm units
use `standard` / `root`; screening refuses other partitions. Diagnostic campaigns
use the separately approved partition described below.
Screening publications carry the recorded final-cell deferral and certify no
release criterion. Production/final code paths retain their existing gates.

Chained coverage allocations reserve 300 seconds for clone, environment and
runner preparation (the 2026-10-14 slot-0 jobs measured 94 seconds). A full leaf
has a 14,100-second stage wall inside a 14,400-second Slurm allocation. Its
sealed budget reserves at least 300 seconds within that stage for export and
publication. `coverage.leaf_allocation` is shared by screening profiles and
campaign expansion; GPU-hour accounting includes the complete allocation.
Sizing also pays the in-fold stopping headroom on every leaf; that time remains
inside the compute ceiling authenticated by the sealed budget.

The nested chained coverage fit checks its absolute compute deadline with a margin of
at least 120 seconds, twice the largest declared or observed phase transaction,
and the checkpoint-write bound. SSL uses its existing cooperative stop-request
check; nuisance fitting and evaluation record their existing transaction
boundaries. Completed nuisance training can checkpoint before evaluation and
resume evaluation without another optimizer update. Timing is execution metadata
and does not enter the scientific recipe.

The bounds use the real-frame D3 measurements from 2026-10-14, Slurm jobs
89923, 89980 and 90034, on one H100 with eight CPUs. The largest measured
intervals used to set the bounds were:

| Phase | Observed seconds | Declared bound, seconds |
| --- | ---: | ---: |
| SSL, including its preparation | 50.643 | 120 |
| Nuisance, including fold configuration and identity preparation | 150.883 | 330 |
| Evaluation, including calibration and held-out prediction | 44.217 | 120 |
| In-fold checkpoint write | 1.529 | 30 |
| Checkpoint return through sealed publication | 4.099 | 300 |

Phase bounds round twice the observations upward to a 30-second boundary,
retaining conservative phase floors. The shared stopping margin is therefore
690 seconds initially: twice the largest phase bound plus 30 seconds for the
checkpoint. Each leaf raises that margin when its own largest observed unit or
checkpoint write requires more headroom. Sizing subtracts the same declared
margin from every leaf's compute capacity. Archive-free repetitions keep their existing
short admission margin because they have no successor to resume an early
checkpoint.

The measurement chains used revision
`bb4aaee493119eef19546aeffd985147c14d7aa4`, keeping the owner study cell;
SSL and nuisance deadline checkpoints both published and their successors
advanced the saved batch count. The independent evaluation probe used
`6a9f760af08f3ee8e1331cc7d38f6d60062d9d47`, including the initial fold-preparation
interval. Only that probe used one SSL and nuisance epoch through the existing
fit controls. It completed all twelve inner evaluations, final refits,
calibration, and outer prediction. Production profiles and campaigns retain
the frozen study settings.

The attempt's external `deadline_report.json` records these measurements,
subsequent checks on the finalized source, source revisions, per-leaf Slurm
allocations and publication times, and cumulative GPU accounting including
failed and canceled probe jobs. It lives under
`runs/leaf-deadline-margins-r2/1c4dee1ab8d11248` in the coordinator project,
alongside `SUMMARY.md` and the final validation logs. Generated reports,
checkpoints and study inputs are not repository files.

New leaf budgets carry `formula: "arc-2612-window"` and must equal the current
formula. Unmarked budgets are accepted only at exactly factor 1.0 with the
matching compute value. All others are refused as "leaf budget sealed under
the pre-ARC-2612 window formula"; old factor-2 campaign records are usable only
as non-authority calibration measurements. No receipt is rebound or resealed.

Dependencies declared under an estimate or lock task's `setup_calibration` are
measurement inputs, so they and ancestors reachable only through those edges do
not enter that consumer's recipe drift comparison. A full dependency walk computes
this exclusion per consumer: any ordinary path to a record keeps
that record scientific. Publication receipts, artifact hashes, request identity,
pass status and attempt ownership remain verified throughout the walk; the existing
budget-authority skips remain recorded in `measurement_only_dependencies.json`.
The estimate's `budget_estimate.json` and the lock's `budget.json` retain each
calibration record's hash, task ID, budget generation, scientific fingerprint and
environment hash as sealed provenance. Scientific profiles, estimates and locks
retain their existing fingerprint equality requirements.

### Dependency verification receipts and diagnostics

Fresh leaves, collectors, estimates, locks, admission and fresh setup probes keep
the full verification walk on current bytes. Resumed `coverage` and
`simulation-smoke` leaves, and `admission-resumed` probes, can attest transitive
ancestors using these sealed receipts:

* The estimate's version-2 `measurement_only_dependencies.json` contains the
  measurement-only publication snapshots and the same `calibration.records`
  hashes, fingerprints and environment hashes as `budget_estimate.json`. Its
  publication is verified now. Calibration record hashes must also agree with
  the estimate's declared request inputs. Ordinary edges do not acquire the
  measurement-only exception. Older estimates without this attestation require
  another valid receipt or the full walk.
* The predecessor's publication is verified now, including its
  `_execution/dependency_check.json`. Its passing check must match the startup
  `_execution/dependencies.json`, which must itself be hash-bound by the
  predecessor's request. Each shared ancestor's publication hash must agree with
  the current producer manifest. Missing or inconsistent entries receive a full
  verification; change and taint fields must be empty lists, and an unbound
  receipt cannot authorize reuse. True post-seal changes
  to the predecessor's own receipt fail its current publication check.
  Before pruning, the runner authenticates each attested publication's small
  manifest, request and config records and checks its declared
  parent hashes against the receipt. A parent absent from both receipt maps
  still receives a full walk. Snapshot entries must match the publication's
  manifest; malformed entries fall back to current verification. This metadata
  must remain readable. Reuse avoids rehashing transitive payloads, not the
  metadata needed to establish which publications the receipt covers.
  The terminal result is reconstructed from the manifest's original result and
  fingerprint artifact, then checked against both the sealed snapshot and the
  independent publication authority. Unused stored result and task payloads are
  attested rather than reread; the sealed consumer already checked the producer
  identity, and the authenticated parent edge binds its publication hash.
  Their post-seal drift has the same full-walk collector backstop as other
  unused transitive payloads. Direct task/result inputs remain checked now.
  Direct-parent identities are read once from their currently verified
  publications and cached within this walk, including when older leaves share
  those parents; the cache never carries across attempts.
* Continuation validation retains its owner, consecutive step, stage, campaign,
  parameters and recipe checks. For resumed leaves it uses the current walk's
  verified publications and budget-authority cache. This removes a second full
  lineage traversal; its evidence is the current verified predecessor publication
  and the same bound receipt, not an unchecked continuation flag.

These exceptions never substitute evidence for a leaf's direct inputs. The
continuation archive, predecessor records, profiles, real-frame inputs, estimate
outputs and lock retain current hash, publication and recipe checks. The current
leaf's science fingerprint freeze and sealed budget formula/generation are
unchanged. Shared transitive recipe comparisons are attested by the predecessor's
same-recipe check; direct recipe comparisons remain current. Publication
post-checks and authority checks cover the currently verified roots. The next
receipt also retains the attested snapshots, labelled `attested_by`, without
claiming that their bytes were re-read. Read observations still cover these
snapshots if a worker actually accesses them.

The trust boundary is deliberate: a transitive-only ancestor whose stored bytes
no attempt in the leaf's lineage reads any more is attested by its sealed
consumer's receipt. A post-seal modification changes no consumed result and is
**not detected by a resumed leaf**. Every chain's fresh leaf and the repetition's
collector perform the full walk on current bytes; the collector detects such
drift and refuses to seal the repetition. The pre-existing verify-at-setup /
read-at-compute window is unchanged and outside this change's scope. This is not
a defense against forging the independent publication authority or changing the
trusted runner itself.

`_execution/dependency_check.json` adds `timings`: exclusive per-root wall seconds
and calls for `hashing`, `publication_receipt`, `recipe_drift` and `lineage_walk`.
Within the ancestry walk, hashing includes safe-read observations and repeated
hashes. `total_seconds` covers that interval; normal request construction and the
final `verify_inputs` pass occur outside it and are not included. These timings
do not claim to measure all setup I/O or total production setup. The
diagnostic continuation subtotal overlaps these components. Direct-input
bookkeeping without an active publication is named `direct_inputs`. Resumed
timings also report `ancestors_reused`, `ancestors_rewalked`, current
`verified_roots`, source publication hashes and `reuse_miss`. Counts refer to
attempts: a predecessor adds one publication. More than one transitive ancestor
re-walked is recorded as `reuse-miss` in the leaf's budget record. Required direct
input checks are not counted as transitive re-walks.

Run `python -m oxyformer.execution.runner --verify-only ATTEMPT --scratch NEW_DIR`
on a CPU compute node to replay a sealed attempt's declared inputs. `--successor`
uses the unique next leaf already declared by its sealed lock, with that sealed
attempt as predecessor. `--full-walk` disables receipt reuse for a diagnostic
comparison. The resolved scratch path must be new and outside the input attempts.
The diagnostic authenticates its source before timing, uses the source's sealed
recipe for ancestry comparisons, never calls `load_lock`, never computes and
never publishes or seals. It writes only the scratch dependency-check report;
refused diagnostic reads do not write taints into the original campaign.
The leaf diagnostic supports stage-publication inputs, including their existing
sealed acquisition ancestry. Attempts with direct acquisition inputs are outside
this diagnostic's envelope and are refused before any scratch output is written.

### Month-2 diagnostic campaigns

`campaign_mode: diagnostic` is bound to the read-only
`owner_decisions.month2_followon` block. It certifies no release criterion.
`real_frame.diagnostic_profile_tasks` emits profiles for the approved scenarios,
then estimate and lock tasks. The caller supplies the campaign ID, fresh evaluation
namespace, prior-lock references and CPU budget. The prior locks must include
exactly one campaign matching `draws_continue_from`.

Each approved repetition runs in its own sequential continuation chain. Sizing
uses the same measured leaf costs and publication reserve as screening, with a
default safety factor of 2.0, at most 40 leaves per chain and four GPU-hours per
leaf. Counts are never reduced to fit a budget; infeasible chains are refused.
The month-1 total-hour ceilings do not apply. GPU concurrency is campaign-wide,
read from the decision (currently 16 on `preemptible`). Expanded leaves use
`--no-requeue`, three attempts and restart in a fresh attempt root after
preemption. First leaves restart from initial inputs; later leaves import only
the predecessor's sealed continuation, never the interrupted attempt's files.

Admission still requires the three probe minima, using the first approved
scenario. Diagnostic collection reports bias, empirical SD, SE/SD and exact
binomial bounds, with `production_equivalent: false` and
`certifies_release_criteria: false`. Final release gates and their repetition
minimum remain unchanged.
