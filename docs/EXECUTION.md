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

### Coverage-owned continuation chains

Coverage work may use the generic campaign `slices` chain. A work item that
uses chained profiling declares `continuation.tar` on **every** slice, including
the terminal slice. Its immutable parameters retain the draws, recipe, scenario,
wall budget and optional `RecipeStudy` cell. The owner/step/predecessor fields
come from the generic expansion; no new instance type or scheduler is involved.
Legacy single-leaf profiles and locks retain their existing interface.

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
`min(120 seconds, wall_seconds / 10)`. Nested-CV retains its checkpoint margin.
The 14,400-second bound applies to a slice, not to a fold's cumulative
training. Epochs, tuning, seeds and estimators are unchanged. Only after all
prediction parts exist does the usual terminal aggregation run. Incomplete
repetitions have no `repetitions/<id>/result.json`; a valid checkpointed slice
passes execution while its batch summary remains incomplete. If a leaf crashes
before export, retry in a fresh attempt root with the same task and exported
predecessor. Never consume partial files in the failed root.

Chained `timing.json` uses `measurement_scope: complete_chain_return`. It lists
per-leaf compute, setup, import, export, publication and total seconds, as well
as the chain total and accumulated compute per completed repetition. Timing and
runtime receipt writes and the outer executor/scheduler are explicitly excluded.
A partial or unsuccessful chain cannot justify a lock. For a chained profile,
lock sizing pays the maximum measured cost of each overhead phase on every
slice and applies the declared safety factor. Usable compute per slice is
`(wall_seconds - checkpoint_reserve) / safety_factor - sum(max_phase_overheads)`;
the slice count is the ceiling of measured repetition compute divided by this
capacity. Admission also measures actual locked setup and refuses if it exceeds
the allocation. Collection counts the last leaf's accumulated terminal records
once, retains all declared draws in denominators, and lists incomplete chains.

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
suite, on Slurm. Tests use only synthetic fixtures and no network access.

### Shared coverage leaf budget

For coverage-owned chains, the lock stores `leaf_budget` in each batch and its
expanded tasks. Sizing, locked preflight admission, and execution all use
`C = (wall_seconds - min(120, wall_seconds / 10)) / safety_factor - sum(overheads)`.
Overheads name setup, import, export, and publication separately. The immutable
lock and continuation binding authenticate the budget, including the compute
ceiling. A 600-second wall, factor 1.2, and 100 seconds of overhead admit 350
seconds of compute: 800 seconds of work requires three leaves.

The whole leaf shares one compute deadline. Setup and import are each charged
once; each positive overrun reduces C and underruns never increase it. Export
and publication remain reserved, along with the checkpoint margin and safety
factor. A compute overrun refuses usable publication. Observed finalization or
wall overruns fail the leaf and cannot justify a profile lock. The collector
refuses over-budget chained leaves. Checkpoint boundaries are cooperative;
uninterruptible operations that overrun are detected, not certified as within
budget. The scheduler remains the hard wall limit.

Bootstrap chain profiles need no prior measurement. They may declare an
immutable `compute_ceiling_seconds` below the wall minus checkpoint reserve;
otherwise that bound applies. Their separate phase measurements supply the
next lock. Partial, unsuccessful, or known-overrun measurements never admit a
lock. Existing archive-free single-leaf producers retain their legacy behavior:
generic step-zero metadata alone does not promise an archive. New locks built
from chained profiles explicitly declare `continuation.tar`. Malformed archive
declaration hardening is deferred to a follow-up.
