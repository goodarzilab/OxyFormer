# Month-2 recipe cost measurements

This module supplies measurement and arithmetic tools for the five cells in the
reviewed recipe-cost criteria. It changes no registered recipe, training module,
approval, or campaign. It does not choose final cells or repetition counts.

Run preparation and tests on Slurm CPU nodes. Run each scoring or fitting
sub-measurement on one H100 in a separate held job on standard/root, with an
eight-hour allocation. Only the orchestrator's node guard releases jobs.
The driver permits two GPU measurement jobs concurrently. GPU jobs disable
automatic requeue; any retry is a separate driver-recorded attempt, bounded by
the two-attempt limit.

The checkout must be clean, and preparation and submission require the recorded
month-two owner decision. Measurements record their commit and the fixed
measurement tree. The count derivation reads the task-list expressions in that tree's
nested controller and the fold/seed iteration and scoring calls in coverage.
It checks that the runtime nested controller, coverage, fit and pretraining
sources match the measurement tree's bytes. Calibration also compares these four
sources at the sealed receipt's commit; differing protected sources make the
result uncalibrated even when their operation counts match.

Preparation authenticates the sealed null-effect profile and its real-frame
dependencies, generates the profile's null-effect draw, and binds the observations
through coverage's existing adapter. The scientific values from the scoring
measurement are discarded. Scoring inputs are synthetic predictions on the full
target-by-seed product; only timing is retained. The merged split contract defines
original_ids as target_ids; extra training and inert rows live in frame_original_ids
and do not contribute scoring predictions. Sealed StageRequest dependency paths
must be absolute. Preparation resolves its own profile and output roots, and the
driver resolves its measurement root, so metadata and child commands remain valid
when fitting jobs run in a separate runtime checkout.

Fitting uses the existing nested controller and its compute counters. Each cell
measures inner SSL, first-grid inner outcome and origin, outer SSL, and first-grid
outer outcome and origin in separate jobs. Outer refits use the first grid point
at the cell's epoch count, as clarified by the orchestrator. The cost of inner
selection enters the projection through counted inner fits; running the entire
tuning grid is not a prerequisite for timing a representative outer refit.

For an independent task, the harness creates an explicitly timing-only controller
checkpoint at that task position, with fixed first-grid settings and genuine SSL
initialization for the corresponding training partition. It contains no invented
tuning scores or calibration results. These incomplete controller artifacts never
claim a finished scientific procedure. The final origin refit stops before the
calibration-transfer step, which requires actual tuning OOF predictions.
Epoch slices use existing budget phase checks. Nuisance timings are differences
of the controller's cumulative compute.wall_seconds. SSL uses the existing
whole-call phase bracket. Slice checkpoints and their durable descriptors permit
a second attempt after interruption. A resumed SSL slice runs only the remaining
transactions of its partial epoch before saving a timed boundary. Actual epoch
progress is retained separately from the last timed boundary; a legacy record
without a boundary cannot supply an extrapolation rate. Full training epochs
without the final fit evaluation are incomplete measurements. The frozen plan specifies a maximum of
thirty SSL epochs with patience five. Native patience completion is measured
at its actual epoch count; this plan rule takes precedence over wording that
would require every SSL fit to execute thirty epochs. Construction work inside the controller
is included; repeated reconstruction can therefore affect calibration.

For each cell the projection is the sum of each counted operation times its
measured duration, divided by seconds per hour. Inner outcome and origin fits
have separate terms, as do their outer refits. Scoring contributes once per
repetition. Inner and outer SSL use separate measured terms and counts. The first
inner partition represents other inner partitions. These are projection
assumptions, not measurements of a complete repetition. Each timed fit uses
registered seed 1103; the projection multiplies by the cell's seed count. The
first hyperparameter grid point represents the other grid points in each
inner-fit timing term. Each refit record also gives its fixed grid index, learning
rate, dropout and epoch count; no grid selection is performed for that timing.

Peak GPU memory is the largest recorded PyTorch allocator peak. CUDA context,
non-PyTorch allocations and unreturned interrupted calls are excluded. Missing
observations are null. Fit summaries sum returned durable slices; whole-job wall
time is retained separately in slurm_jobs.json.

Calibration compares the screening anchor with the single complete-repetition
compute duration in the sealed profile. Frame, cell and operation identity must
match. The inclusive ratio window is 0.80 to 1.25. A failed calibration propagates
the uncalibrated basis to projections and feasibility numbers.

Campaign overhead uses each of the 140 campaign-D leaves once. Its numerator
is the reserved GPU allocation derived from the sealed stage budget through
coverage.leaf_allocation: stage time plus setup, rounded to whole minutes. Its
denominator sums each leaf's own compute entry, without double-counting cumulative
predecessor histories. Measured runner wall time is retained separately; it
excludes reserved final publication and is not the allocation numerator. Failed
scheduler attempts are outside this ratio over the sealed campaign leaves.

The feasibility arithmetic spans 39, 100, 300 and 1,000 repetitions, with one and
four scenarios. Total allocated GPU-hours divided by sixteen is an idealised
wall-clock lower bound. It assumes perfect utilisation and does not predict
queue delays. The study itself uses no more than two GPUs.

CPU fixtures validate arithmetic, source-walk responsiveness, basis propagation
and actual controller epoch continuation. They do not validate GPU timings,
Slurm placement or successful completion of the real-frame measurements.
The unit's external output directory, not this documentation, contains the
measurement evidence and any completed report.

The driver records a unique submission intent before invoking sbatch. After a
lost reply it reconciles that job name against accounting and the live queue.
An unresolved submission stops further submissions until it can be reconciled;
it is never silently retried. The ledger and driver lock enforce the concurrency
limit across driver restarts. Requeued and other nonterminal scheduler states
retain their concurrency slots. Publication writes slurm_jobs.json from the
measurement ledger while retaining separately recorded CPU checks.

When continuing existing fits, run the driver with --fit-checkout pointing to the
original clean runtime checkout and --harness-checkout pointing to a clean frozen
checkout containing the fixed-grid timing harness. All fitting jobs execute
that harness as a script while importing the original runtime package; their
records bind both commits. Each new attempt records its execution harness commit.
This preserves the package identity of real SSL
checkpoints already measured before the scheduling clarification. Scoring jobs also
execute the harness script from the fitting checkout. Child commands explicitly
use PYTHONPATH=src. Never edit
either frozen checkout while its checkpoint chain is live.

The lifecycle identity guard reads `inputs.json` before preparation, scheduling,
and reporting. It loads the endpoint from its absolute path, checks its stored
file digest, and recomputes the endpoint content hash for comparison to journal
bindings. It checks the counted work, recipe, study dimensions and fit position.
Scoring now records an endpoint binding too. Legacy scores without that evidence
are preserved but excluded with an error; matching measurements are required.
Preparation with identical inputs is a no-op even after a harness commit changes;
changed inputs are refused before any overwrite. Failed terms retain their raw
counters but their numeric summaries carry `failed_measurement` labels.
Clean alternate Git histories are accepted; the protected-source byte checks
remain required. CPU lifecycle tests exercise the actual fitting and scoring
calls with synthetic inputs; they do not establish real GPU timings or placement.

The measurement tree remains the original study base
`f25e8853ba691883b447e0f89e9ce334f84f56ac`, as required for the orchestrator's
primary attempt-4 measurements. The successor launch base
`57d0a04e419b0bc2c8bffb3adb46b586297746f4` defines the complete PR review range,
not a relabelling of earlier timings. Fitting and scoring run the harness script from the frozen
measurement runtime with `PYTHONPATH=src`; its protected files must still match
the measurement tree. Preparation and read-only consumers derive counts directly
from that Git tree, so a newer reporting checkout does not invalidate earlier
measurements. Pass the frozen runtime as `--fit-checkout` when driving from the
new harness checkout. Verify prior artifacts against their stored bindings,
preserve and exclude unverifiable records, and measure only missing terms after
the primary driver's ledger has no pending work. Across both roots the bound is
two GPU jobs, with separate held eight-hour jobs and the existing node guard.

Scoring receipts record the scheduler job ID and both runtime and harness commits.
The scoring producer checks both checkouts before timing; every CLI entry also
checks the checkout containing the harness script, including when invoked by
absolute path from the frozen runtime.

Artifact reuse also resolves each recorded runtime commit and compares its four
protected source blobs with the counted tree. Missing commits or mismatched
sources are refused before scheduling or projection, including for legacy fits.
Runtime and harness commits are execution provenance rather than scientific
continuation keys. A second attempt records its actual commits while preserving
the original journal binding and attempt history. Documentation-only checkout
changes therefore permit continuation. The existing checkpoint validator still
checks the entire executed package source and scientific configuration; this
does not make checkpoints from changed fitting code interchangeable.
