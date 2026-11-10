# Run logging

`FitConfig.run_logger` / `PretrainConfig.run_logger` default to `None`.
Use a fresh `run_logging.WandbLogger(base_directory)` per attempt; nested SSL
shares the fold logger. Off and `run_logging.noop` skip telemetry. W&B is lazy;
provision it only in the separate `oxyformer-p3` clone from conda-forge.

During fitting the hook buffers redacted JSON events in memory. Filesystem,
Git provenance lookup and SDK operations happen after checkpoint creation,
so a slow logging filesystem or SDK outage cannot consume the fitting slice.
W&B start records receive Git provenance before publication. Both source checkouts and recipes stay
pinned. SDK initialization always uses offline mode, including when
`WANDB_MODE=online`, and is attempted once.

Each logger allocates a unique `attempt-*` subdirectory under its base. Its
`status_path` identifies its own `logging-status.json`; sharing a base never
overwrites another attempt. Null `logging_ok` means SDK processing is pending;
true/false records final logging completeness, separately from `sync_ok`.
Sync follows the saved checkpoint and offline SDK finish, with a 120-second
timeout. Credentials come only from runtime `WANDB_API_KEY`; never dump the
environment. Redaction covers secret fields and numeric representations of the
runtime key. Online uploads remain unverified.

The retained `spool` has a sealed event count, digest and capture-completeness
flag. Failed capture stays incomplete after replay. Partial or modified spools
are refused before SDK publication. Recover a complete captured spool after an
SDK failure with `WandbLogger.replay("/path/to/events-complete.jsonl")` from the
clone, without retraining. Replay has its own status and preserves the original.
For an existing W&B offline run, use `python -m wandb sync /path/to/offline-run`.

`CONFIG_SCHEMA` classifies every runtime dataclass field and rejects unknown
fields or instance keys. Structured settings use their contract serializers;
data uses pins and lineage; the logger is excluded. Initial stop state is
`none` or `requested`; checkpoint events record `honoured` for a requested stop.
Snapshots read stored request state without invoking deadline polling. Evaluation
events cover stopping, held-out and selection partitions, retaining numeric
values and the selection partition's original IDs.

Each GPU window synchronizes and resets the Torch allocator peak immediately
before a training step, then samples after the optimizer update. It includes
allocations live at window start; CUDA context and allocations outside Torch's
allocator are excluded. CPU peaks are zero. The logger owns the device's peak
counters during these windows; allocator doubles verify successive windows.

Both loops test off/no-op/on and SDK failure with deterministic CPU kernels,
bitwise backward losses and complete checkpoint SHA-256, fixed paths/ZIP dates,
and unchanged Python/NumPy/Torch RNG. Clock doubles also test slow filesystem,
provenance and SDK failures. GPU nondeterminism remains; hardware measurements
apply only to the hardware used.

The narrowed contract accepts exactly `WandbLogger`, `noop`, or `None`.
Generic callables and logger subclasses are rejected with `TypeError` before
training starts. Construct one fresh logger per attempt; event callbacks are
no longer an extension interface.

The five settled contract questions are:

- Non-interference covers checkpoint bytes, per-step numerics and the returned
  training artifact. Finish runs after checkpoint persistence, with a 180-second
  delivery deadline. On Linux this uses a temporary SIGALRM handler; a threaded
  caller or an existing real-time timer refuses delivery rather than replacing
  another component's timer. A delivery timeout marks logging incomplete;
  a pending/null status also means incomplete. Scientific results stay usable.
- Credential redaction includes exact runtime key text, its decimal integer,
  and digit strings differing only by leading zeros, in values and mapping keys.
- SSL `train_step` means an applied optimizer update. Skipped batches emit
  `batch_skipped`; both events carry cumulative `processed_batches`.
- Config inspection uses the caller's original object before copying. Extra
  instance keys refuse telemetry while training proceeds: `logging_ok=False`,
  a `refused: undeclared config key <name>` status event and a flagged start
  snapshot are retained at finish. Undeclared values are never exported.
- Git uses one subprocess in its own process group with a ten-second timeout.
  The group is killed and its leader reaped after the lookup. Timeout/error
  records `git_revision = 'unavailable: <reason>'` (also in `git_sha`), marks
  logging incomplete and continues finish.
