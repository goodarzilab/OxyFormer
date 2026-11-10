# Run logging

`FitConfig.run_logger` / `PretrainConfig.run_logger` default to `None`.
Use a fresh `run_logging.WandbLogger(directory)` per attempt; nested SSL shares
it. W&B is imported lazily. Provision W&B in the separate `oxyformer-p3`
clone from conda-forge; never modify the pinned training environment.

The hook spools redacted JSONL events locally during training. SDK initialization
and replay into an offline W&B run happen only after checkpoint creation,
including when `WANDB_MODE=online`. Failed initialization is attempted once;
it cannot consume the fitting time slice. Finish/sync follow checkpoints
(120-second sync timeout). `logging-status.json` records `logging_ok` / `sync_ok` as separate
completeness facts. Retain offline runs and retry with
`python -m wandb sync /path/to/offline-run` from the clone. If SDK initialization
failed, recover the retained JSONL file with
`WandbLogger.replay("/path/to/events-....jsonl")` from the clone, without retraining.
A null `logging_ok` means SDK processing is pending; true/false records its final
completeness independently of the scientific checkpoint.
Credentials come only from runtime `WANDB_API_KEY`; never dump the environment.
Events redact secrets. Service uploads remain unverified.

`CONFIG_SCHEMA` classifies every runtime dataclass field; unknown fields fail
projection and mark W&B logging incomplete. Declared structured settings are
included through their contract serializers, data through pins and lineage,
and the callback is excluded. Initial stop state is `none` or `requested`;
checkpoint events record `honoured` when the stop reason is `requested`.
Snapshots read stored request state without invoking deadline polling.

Each GPU logging window starts immediately before a training step, synchronizes
and resets the Torch allocator peak, and ends at the step-event sample after
the optimizer update. The peak includes allocations live at window start.
CUDA context and allocations outside Torch's allocator are excluded. CPU peaks
are zero; allocator-double tests verify independent successive windows.
This logger owns the device's allocator peak counters during these windows.

Both loops test off/no-op/on with deterministic CPU kernels, bitwise losses and
checkpoint SHA-256, fixed paths/ZIP dates, and unchanged Python/NumPy/Torch RNG.
GPU nondeterminism remains; measurements apply only to that hardware.
