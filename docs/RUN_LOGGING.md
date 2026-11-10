# Run logging

`FitConfig.run_logger` / `PretrainConfig.run_logger` default to `None`.
Use a fresh `run_logging.WandbLogger(base_directory)` per attempt; nested SSL
shares the fold logger. Off and `run_logging.noop` skip telemetry. W&B is lazy;
provision it only in the separate `oxyformer-p3` clone from conda-forge.

During fitting the hook buffers redacted JSON events in memory. Filesystem,
Git provenance lookup and SDK operations happen after checkpoint creation,
so a slow logging filesystem or SDK outage cannot consume the fitting slice.
Custom callbacks receive deferred Git provenance in `run_finish`; W&B start
records receive it before publication. Both source checkouts and recipes stay
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
data uses pins and lineage; the callback is excluded. Initial stop state is
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
