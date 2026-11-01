# Month-2 recipe cost measurements

This module supplies measurement and arithmetic tools for the five cells in the
reviewed recipe-cost criteria. It changes no registered recipe, training module,
approval, or campaign. It does not choose final cells or repetition counts.

Run preparation and tests on Slurm CPU nodes. Run each scoring or fitting
sub-measurement on one H100 in a separate held job on standard/root, with an
eight-hour allocation. Only the orchestrator's node guard releases jobs.
The driver permits two GPU measurement jobs concurrently.

The checkout must be clean. Measurements record their commit and the fixed
launch tree. The count derivation reads the task-list expressions in that tree's
nested controller and the fold/seed iteration and scoring calls in coverage.
It checks that the runtime protected sources match those bytes.

Preparation authenticates the sealed null-effect profile and its real-frame
dependencies, generates the profile's null-effect draw, and binds the observations
through coverage's existing adapter. The scientific values from the scoring
measurement are discarded. Scoring inputs are synthetic predictions on the full
target-by-seed product; only timing is retained.

Fitting proceeds through the genuine nested controller. Each task has its own
job, including the prerequisite tuning tasks needed to reach an outer refit.
The driver never fabricates selection results or checkpoints to skip tuning.
Epoch slices use existing budget phase checks. Nuisance timings are differences
of the controller's cumulative compute.wall_seconds. SSL uses the existing
whole-call phase bracket. Slice checkpoints and their durable descriptors permit
a second attempt after interruption. Construction work inside the controller
is included; repeated reconstruction can therefore affect calibration.

For each cell the projection is the sum of each counted operation times its
measured duration, divided by seconds per hour. Inner outcome and origin fits
have separate terms, as do their outer refits. Scoring contributes once per
repetition. The representative SSL term is the outer SSL fit, and the first
inner partition represents other inner partitions. These are projection
assumptions, not measurements of a complete repetition.

Calibration compares the screening anchor with the single complete-repetition
compute duration in the sealed profile. Frame, cell and operation identity must
match. The inclusive ratio window is 0.80 to 1.25. A failed calibration propagates
the uncalibrated basis to projections and feasibility numbers.

Campaign overhead uses each of the 140 campaign-D leaves once, taking its own
last entry rather than summing cumulative predecessor histories. The budget
record's measured runner wall interval excludes reserved final publication;
it is not a replacement for scheduler accounting of failed attempts.

The feasibility arithmetic spans 39, 100, 300 and 1,000 repetitions, with one and
four scenarios. Total allocated GPU-hours divided by sixteen is an idealised
wall-clock lower bound. It assumes perfect utilisation and does not predict
queue delays. The study itself uses no more than two GPUs.

CPU fixtures validate arithmetic, source-walk responsiveness, basis propagation
and actual controller epoch continuation. They do not validate GPU timings,
Slurm placement or successful completion of the real-frame measurements.
The unit's external output directory, not this documentation, contains the
measurement evidence and any completed report.
