# Exploratory real USALEEP fits

The owner decision of 2026-11-19 permits reading outcomes on the full approved
tract frame. No outcome-sealed reserve is withheld. Support design and inference
therefore use the same tracts. Results are exploratory, certify no release
criteria, and do not authorize effect release or establish a causal clinical
recommendation. The sealed campaign paths retain their existing behavior.

Use the pinned Python environment and checkout. Run every command in a scheduler
allocation; this driver never submits a job. Preparation and scoring need CPU;
a real fit invocation needs one GPU. The orchestrator chooses allocations.

```sh
python -m oxyformer.explore.real_fit prepare \
  --tract-inputs /path/to/tract-inputs --atlas /path/to/atlas-collect \
  --usaleep-tar /path/to/fetch-us/payload.tar \
  --usaleep-receipts /path/to/fetch-us/receipts.json --out /path/to/prepared

python -m oxyformer.explore.real_fit fit --prepared /path/to/prepared \
  --fold 0 --seed 1103 --state /path/to/fold-0 \
  --slice-seconds 10800 --run-logging wandb

python -m oxyformer.explore.real_fit score --prepared /path/to/prepared \
  --state /path/to/fold-0 --state /path/to/fold-1 --state /path/to/fold-2 \
  --state /path/to/fold-3 --state /path/to/fold-4 --out /path/to/result \
  --run-logging wandb
```

Repeat fit for folds 0 through 4, each with its own state directory. Repeat each
invocation until its progress.json says complete. Each call creates a fresh slice
and atomically replaces the journal. SIGTERM and SIGUSR1 request a cooperative
checkpoint; allow time after signaling for checkpoint serialization. A completed
fold is a no-op. An interrupted process can leave an orphan slice; keep it for
diagnosis and resume from the last journal. Never share one state between folds
or concurrent invocations. Resume requires the identical clean checkout,
environment, device and thread count. Keep prepared inputs and slice files in
place. Identity mismatches refuse; recover by restoring the original environment
or using a new state directory for a new run.
Fit state and score output paths that resolve inside the checkout must already
be ignored by Git; otherwise they refuse before writing, so use a state directory
outside the checkout.

Cell d uses reference cap 32, 60 nuisance epochs, two grid points and seed 1103.
Extending the cell-d approval block beyond its four registered fields needs an owner decision.
SSL remains 30 epochs. The cell-d study overrides the registered 150-epoch base
settings to 60 effective nuisance epochs. The five outer and three inner folds, approved primary
flags 1 and 3, and approved covariates are unchanged. Treatment centering/scaling
uses the allocation-qualified support pool. County centroids use target tracts;
the primary spatial bandwidth is 100 km.

W&B is loaded only when requested, with entity goodarzilab and project oxyformer-v2.
The orchestrator installs it before running; a missing package refuses before
training. W&B reads its credential from the environment and keeps its local run
files under the validated state/output directory. Driver logging records
configuration, slice fold/checkpoint position/stop reason/wall time and final
estimates; it has no per-epoch curves.
`checkpoint_position` is the nested-CV controller position; `train_epoch` is
included only when active model progress is present in the existing checkpoint
metadata. `--run-logging none` is an explicit recorded choice for synthetic fits
and scoring; real fits refuse it before creating state or starting training,
as required by the owner rule.

Read result.json for both one_step and cv_tmle, all pilot SE definitions, counts,
recipe and identities. result.md gives the primary readout. Different SEs target
different sources of variation: a small frame-conditional residual SE is not a
certificate of geographic coverage. prepare_audit.json records source hashes,
flag counts and the exclusions at each design layer.

Preparation accepts `--delta-mmhg` (default 2.0) and
`--support-caliper-iqr` (default the approved `raw_x_radius`, 2.0 IQR).
The owner's 2026-11-22 `policy_support_sensitivity` decision permits delta 1.0
or 2.0 and a caliper from the approved value through 3.0 IQR. Non-default
settings require that decision; bin width and per-bin tract minimum stay fixed.
The original design is still validated against approvals before applying these
exploratory overrides. Fit and score use the prepared policy without new flags.

Both settings are recorded in `prepare_audit.json`; the endpoint carries delta
and the frozen support components. No endpoint schema change or sidecar is used
(orchestrator narrowing of S3, 2026-11-22). Older default preparations still load.

Every preparation and score reports `n_targets`, `n_moved_targets`,
`shifted_fraction` (moved / targets) and `achieved_mean_shift_mmhg` (fraction times
delta). The denominator is the final target set for that preparation, which can
change with either override. A small shifted fraction means the policy leaves
most targets unchanged. Delta 1 answers a different question from delta 2;
it is never rescaled into a 2-mmHg claim. A wider caliper loosens neighbour
comparability. Results remain exploratory and certify nothing.

The score's `policy_diagnostics` contains per-estimator value and primary SE
rescaled per mover and per achieved mmHg. These are descriptive rescalings, not
new estimands. Among movers it also reports Kish ESS of the scored `r_d`, the
largest tract's share of its sum and its 99th percentile. With no movers these
diagnostics are null; ESS and share are also null if all mover ratios are zero.
