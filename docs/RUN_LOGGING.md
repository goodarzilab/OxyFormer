# Run logging

`FitConfig.run_logger` and `PretrainConfig.run_logger` default to `None`.
Pass a fresh `run_logging.WandbLogger(directory)` per attempt; `noop` explicitly
disables logging. Nested SSL shares the fold logger.

The target is `goodarzilab/oxyformer-v2`; `WANDB_MODE` defaults to `offline`.
After checkpoint publication, finish precedes sync (120-second timeout).
`logging-status.json` and `logging_ok`/`sync_ok` are separate completeness facts:
null means no sync attempted. Failures preserve results and offline files. Retry:

```sh
python -m wandb sync --entity goodarzilab --project oxyformer-v2 /path/to/offline-run
```

Steps record loss, LR, applied gradient norm (after existing clipping), throughput
excluding logging, and cumulative allocator peak bytes (CPU: zero). SSL batches
without targets log null loss/zero norm. Evaluations, selections and checkpoint
paths/hashes are recorded. Start events include config/hashes, Git/package/model/data pins, seed/fold,
hardware/node/job ID and resume lineage, without raw observations.

Install optional `wandb==0.30.0` from **conda-forge only**, as recorded in the
manifests. Credentials come only from runtime `WANDB_API_KEY`; online mode
requires it. Never persist credentials or dump the environment.
Events redact credential fields/current-key occurrences; errors and sync output
are suppressed. Tests grep repository/run files, including real SDK output.
Upload is unverified; tests use offline files and simulated sync.

Both loops test off/no-op/on with deterministic CPU kernels, bitwise step losses
and complete checkpoint SHA-256. Tests fix paths/ZIP dates, which affect archive bytes. Callbacks return no training decisions and preserve
Python/NumPy/Torch RNG state even on failure. GPU nondeterminism remains a limit; measurements apply to that hardware.
