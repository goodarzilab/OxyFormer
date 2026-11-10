# Run logging

`FitConfig.run_logger` and `PretrainConfig.run_logger` default to `None`.
Pass a fresh `run_logging.WandbLogger(directory)` per attempt; `noop` explicitly
disables logging. Nested SSL shares the fold logger. Estimators, collectors and
campaigns are unchanged.

The target is `goodarzilab/oxyformer-v2`; `WANDB_MODE` defaults to `offline`.
After checkpoint publication, finish precedes sync (120-second timeout).
`logging-status.json` and `logging_ok`/`sync_ok` are separate completeness facts:
null means no sync attempted. Failures preserve training results and offline
files. Retry with:

```sh
python -m wandb sync --entity goodarzilab --project oxyformer-v2 /path/to/offline-run
```

Steps record loss, LR, applied gradient norm (after existing clipping), throughput
excluding logging, and cumulative allocator peak bytes (CPU: zero). SSL batches
without targets log null loss/zero norm. Evaluations, selections and checkpoint
paths/hashes are recorded. Start events include non-secret recipe/runtime config,
its hash, scientific/package/Git hashes, model/data/source/split pins, seed/fold,
hardware/node/job ID and resume lineage. No raw observations are uploaded.

Install optional `wandb==0.30.0` from **conda-forge only**, as recorded in the
manifests. Credentials come only from runtime `WANDB_API_KEY`; online mode
requires it. Never persist it, dump the environment or embed a login literal.
Events redact credential fields/current-key occurrences; errors and sync output
are suppressed. Tests grep repository/run files, including real SDK output.
Online upload remains unverified: tests use offline files and simulated sync.

Both loops test off/no-op/on with deterministic CPU kernels, bitwise step losses
and complete checkpoint SHA-256. Paths and ZIP dates are fixed in tests because
they affect archive bytes. Callbacks return no training decisions and preserve
Python/NumPy/Torch RNG state even on failure. GPU nondeterminism outside test
mode remains a limit; throughput/memory measurements apply only to that hardware.
