# Run logging

`FitConfig.run_logger` and `PretrainConfig.run_logger` default to `None`.
Pass a fresh `run_logging.WandbLogger(directory)` per attempt; `noop` disables
logging. Nested SSL shares the fold logger.

W&B targets `goodarzilab/oxyformer-v2`; `WANDB_MODE` defaults to `offline`.
After checkpoint publication, finish precedes sync (120-second timeout).
`logging-status.json` and `logging_ok`/`sync_ok` report separate completeness
facts; null sync means no attempt. Failures preserve training results. Retry:

```sh
python -m wandb sync --entity goodarzilab --project oxyformer-v2 /path/to/offline-run
```

Install optional W&B from **conda-forge only**. Only runtime `WANDB_API_KEY`
supplies credentials for online/sync; never persist it or dump the environment.
Events redact credentials; error output is suppressed. Tests grep repo/run files.
Uploads are unverified: tests use offline SDK files and simulated sync.

Both loops test off/no-op/on with deterministic CPU kernels, bitwise losses and
checkpoint SHA-256. Tests fix paths/ZIP dates affecting archive bytes. Callbacks
preserve Python/NumPy/Torch RNG state. GPU nondeterminism remains a limit;
throughput/memory measurements apply only to that hardware.
