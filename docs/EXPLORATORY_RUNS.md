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

## Outcome-only simulation refits

`oxyformer.explore.q_refit` binds the simulated observations from one saved
terminal diagnostic or pilot chain. It refits Q with the frozen policy and
folds, and trains no density-ratio model or calibration. The chain's saved g is
used later by the separate screening harness. These results are exploratory
and certify nothing. This path does not fit real-data endpoints.

Candidates are A0 (matched baseline), A0-SW (A0 with support-weighted loss),
A3-Q (registered early-fusion outcome head), and A4-Q (registered
varying-coefficient outcome head). Each network is capped at 1,000,000
parameters. A0-SW multiplies factual loss by
`1 + shift_mask(a) + shift_mask(a - delta)`, using the endpoint's active delta
and frozen support. Empty support contributes one. The same weights govern
held-out grid selection; no density ratio enters them. SSL and inner tuning
remain fold-local. Cell d uses 60 nuisance epochs; cell e uses 30; both use
reference cap 32, two grid points, one seed (1103), and 30 SSL epochs.

Run binding and collection on CPU scheduler allocations. Each production fit
invocation needs one GPU allocation; the driver submits no jobs.

```sh
python -m oxyformer.explore.q_refit bind /path/to/terminal-chain /path/to/q-A4
python -m oxyformer.explore.q_refit fit /path/to/q-A4 --fold 0 \
  --variant A4-Q --study cell-d --slice-seconds 10800 --run-logging wandb
# Repeat fit for folds 0..4, resuming each fold until complete.
python -m oxyformer.explore.q_refit collect /path/to/q-A4
```

Use a separate OUT for each chain, variant and study. Binding authenticates the
archive blobs and endpoint/recipe dependencies and writes the bound endpoint,
recipe and provenance without changing the chain. Fit preserves its predecessor
and completed tasks between slices of at most three hours. Exit 75 means the
slice checkpointed before completion; repeat the same command. Exit zero means
the fold is complete. Each `fold-K/progress.json` records task counts, slices,
checkpoint position, and cumulative wall seconds for each SSL fold, inner
outcome grid fit, outer outcome fit and prediction. Resume requires the same
clean commit, environment, device, inputs and logging mode. Keep slice files
and bound inputs in place. Restore the original environment or use a new OUT
if an identity check refuses. State locking prevents concurrent writers.
W&B uses entity `goodarzilab`, project `oxyformer-v2`; config records chain,
variant, study and fold. Logging `none` is accepted only for synthetic tests.

Collection requires five complete ordered folds and writes `q_parts.json`.
The `oxyformer.q_parts.v1` object has exactly `schema`, `chain_dir` (resolved
terminal chain path), `frame_hash` (saved state binding), `variant`, `study`
(the four RecipeStudy values), `code_commit`, and `parts`. Each part has exactly
`fold`, `seed`, `original_ids`, `mu_a`, and `mu_d`; IDs and predictions retain the
saved target order. The file contains no g predictions, truth or estimates.

### A6-Q: frozen TabICLv2 outcome-only candidate

`--variant A6-Q` routes a fold to `oxyformer.explore.tabicl_q` instead of the
nested fitting controller. It fits no network and updates no weight: the pinned
TabICLv2 regressor is conditioned in context on fold K's training rows and
predicts `mu_a` and `mu_d` for that fold's held-out targets. The owner allowed
it as a Q candidate on 2026-11-24. The registered science-plan comparator
(`models/tabicl_comparator.py`) is separate and unchanged; its CPU and
no-subsampling rules govern the plan comparison, not this candidate. Results
are exploratory and certify nothing.

Context rule. The candidate rows are exactly the rows the Q-only outer outcome
fit of fold K trains on (`outer.training_ids(K)`, derived through the same
partition). With at most C rows (`--context-rows`, 2 to 30,000, default
30,000), all are used in their order. Otherwise exactly C are kept: those with
the smallest SHA-256 of `[salt, K, original_id]`, in their original order. The
salt `oxyformer.explore.tabicl_q.context.v1` is fixed and recorded. Selection
reads IDs only, never an outcome, treatment, covariate or prediction, so it
drops 9% to 25% of each real fold's rows without outcome-selected bias. No
held-out target enters its own fold's context. Columns are the treatment in
mmHg followed by the endpoint's numeric nuisance covariates in
`feature_kinds` order; more than 500 columns, categorical covariates or
non-unit weights refuse. One fitted context serves both query sets. Each
target is queried alone at `action.a_mmhg` and `action.d_mmhg` from
`policy.apply`; a non-mover's `mu_d` is its `mu_a`.

Checkpoint. `--checkpoint PATH` names a local copy of the registered regressor
`tabicl-regressor-v2-20260212.ckpt` (repository `jingang/TabICL`, revision
`4dcd344e…`, SHA-256 `0db9cb53…`). Each slice copies it to a private
directory, verifies the copy's SHA-256 and passes only the copy to TabICL with
automatic download disabled. A missing file, another filename, a wrong hash
or an unregistered identity refuses before any prediction. The candidate
identity records the checkpoint's registered fields and SHA-256, C, the salt,
the estimator settings (8 estimators, `random_state` 1103, KV cache, no AMP or
FlashAttention 3) and the numerics (deterministic kernels, IEEE FP32), never
the path. tabicl 2.2.0 must be installed; CI does not install it.

```sh
python -m oxyformer.explore.q_refit bind /path/to/terminal-chain /path/to/q-A6
python -m oxyformer.explore.q_refit fit /path/to/q-A6 --fold 0 \
  --variant A6-Q --study cell-d --slice-seconds 10800 --run-logging wandb \
  --checkpoint /path/to/tabicl-regressor-v2-20260212.ckpt
# Repeat fit for folds 0..4 until each exits 0.
python -m oxyformer.explore.q_refit collect /path/to/q-A6
```

`--study` is required and recorded as for the other variants; A6-Q uses none
of its values. Each fit needs one GPU allocation for a saved endpoint. Exit 75
means the slice stopped at a query-batch boundary (deadline or SIGTERM/SIGUSR1)
before the fold completed; repeat the same command. Exit 0 means the fold is
complete. A slice starts another batch only while the longest batch of this
slice still fits before the deadline less min(60 s, slice/10). Conditioning is
admitted only if the time left before that margin covers the last recorded
context-fit seconds (0 before any slice has conditioned) plus the batch
estimate, which is zero before this slice's first batch. The stop request and
this admission rule are checked before the estimator is constructed and again
after construction, immediately before `estimator.fit`; if either trips, the
slice ends with `stop_reason` `signal` or `deadline` without conditioning.
Each slice journals its checkpoint, context, estimator-construction and fit
seconds, whether it conditioned, and the admission seconds it required, before
its first batch (`stop_reason` `running` until the slice ends). If a resumed
slice can never cover the recorded fit, it keeps ending with `deadline`; rerun
with a longer `--slice-seconds` (up to 10,800).

Accepted residual. `estimator.fit` is one synchronous TabICL call and cannot be
interrupted mid-call by the cooperative stop flag. A signal that arrives during
the fit takes effect at the next batch boundary; if Slurm kills the slice
during the fit, no partial batch or part has been written, so the journal
stays resumable and the next slice redoes the fit. The time lost is that
slice's conditioning, bounded by the mandate's at most 4 GPU-hours lost per
Slurm attempt.

A complete fold returns its journaled record before `--checkpoint` is checked,
so rerunning it without the checkpoint, or after the file is removed, loads no
weights and predicts nothing; `--context-rows` may then be omitted, and an
explicit value must match. A fold with work left always validates the
checkpoint (explicit path, registered identity, SHA-256 of a private copy)
before any backend call. Every candidate training row, sampled or not, must
carry a unit weight, and a journal with saved batches must carry its runtime
and context provenance, or the fit refuses. The private checkpoint copy
(about 114 MB) is written under `TMPDIR`; point `TMPDIR` elsewhere if the
system temporary directory is small. Any refusal exits non-zero with the
reason. Predictions run in
batches of 32 targets, each saved atomically in `fold-K/progress.json` with its
wall seconds. The journal also records the context (candidate count, C,
selected count, salt, columns, selected-ID and context digests), the runtime
fingerprint from the first slice (Python, torch, numpy, scipy, scikit-learn,
tabicl, device), and per-slice checkpoint, context and fit seconds. A resumed
slice rebuilds and refits the context, refuses a changed digest or
fingerprint, and skips completed batches, so the completed part equals an
uninterrupted run's on the same device and environment. W&B config records
chain, variant, study, fold, C, the salt and the checkpoint SHA-256. Collect
needs no A6-Q case: `q_parts.json` has the same exact keys.
