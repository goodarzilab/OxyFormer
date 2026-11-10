# Run logging

`FitConfig.run_logger`/`PretrainConfig.run_logger` default to `None` (no-op).
Use fresh `run_logging.WandbLogger(directory)` per attempt; nested SSL shares it.
`WANDB_MODE` defaults to `offline`. Finish/sync follow checkpoints (120-second
sync timeout). `logging-status.json` records `logging_ok`/`sync_ok` facts;
failures preserve results. Retry with `python -m wandb sync /path/to/offline-run`.

Install from **conda-forge only**. Credentials come only from runtime
`WANDB_API_KEY`: never persist it or dump the environment. Events redact secrets;
errors are suppressed and tests grep repo/run files. Upload is unverified.

Both loops test off/no-op/on with deterministic CPU kernels, bitwise losses and
checkpoint SHA-256, fixed paths/ZIP dates, and unchanged Python/NumPy/Torch RNG.
GPU nondeterminism remains; measurements apply only to that hardware.
