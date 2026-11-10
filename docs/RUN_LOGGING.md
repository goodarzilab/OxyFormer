# Run logging

`FitConfig.run_logger` and `PretrainConfig.run_logger` default to `None`.
`run_logging.noop` is an explicit no-op. For a new run, construct
`WandbLogger(attempt_logging_directory)` and pass it as `run_logger`.
Standalone SSL and fold fitting finish and sync after writing their checkpoint;
nested SSL shares the fold logger and leaves finishing to the fold.
Use a fresh logger and output directory for every invocation or continuation.
No campaign, estimator or collector wiring is changed here.

The adapter targets entity `goodarzilab`, project `oxyformer-v2`. It defaults to
`WANDB_MODE=offline`, including on compute nodes. An explicit environment setting
can override it. After SDK finish, the adapter invokes `python -m wandb sync`
with a 120-second timeout and suppressed subprocess output. Offline files remain
available for a later login-node sync if egress or authentication is unavailable:

```sh
python -m wandb sync --entity goodarzilab --project oxyformer-v2 /path/to/offline-run
```

`logging-status.json` and the logger's `logging_ok` / `sync_ok` properties are
separate completeness facts. `sync_ok: null` means no upload was attempted.
Logging or sync errors never change checkpoint completeness or training return
values. Retry sync from the preserved offline directory; inspect the scientific
checkpoint independently. Online service access has not been tested with owner
credentials; tests use real offline SDK files and a simulated sync failure.

Each actual update logs detached train loss, learning rate, gradient norm of the
gradients applied (after the existing nuisance clipping), samples per second,
and the selected device's torch allocator peak bytes (zero on CPU). An SSL batch
with no masked reconstruction targets records null loss and zero gradient norm.
Stopping evaluations, held-out nuisance evaluations, selected epochs, and final
checkpoint paths/hashes are recorded. Peak memory is cumulative allocator peak,
not an estimate of total device memory. Throughput covers the minibatch update
and device synchronization, excluding logging time. These measurements make no
claims about other hardware.

Run-start events contain declared recipe/runtime config and its snapshot hash,
scientific identity/config hash, executed package hash, Git SHA, model settings,
source/data/split pins and hashes, seed, fold, hardware/node, Slurm job ID,
output path and predecessor lineage. Raw training observations are not logged.
Live budget/signal resources are excluded; their declared settings are retained.

Provision the optional `wandb==0.30.0` dependency from **conda-forge only**, as in
`requirements/run-logging.in`; `pyproject.toml` records the optional dependency.
Credentials come only from `WANDB_API_KEY` at runtime. Never call `wandb.login`
with a literal, persist credentials, or dump environment variables. Credential
fields and occurrences of the current environment key are redacted from events.
Error text is suppressed because it may contain credentials. The hygiene test
scans the repository and synthetic run tree for the injected key and credential
assignment patterns; bare Git and checkpoint hashes are provenance, not keys.

Both training suites compare disabled, no-op and enabled logging under deterministic
CPU kernels: all per-step losses and checkpoint SHA-256 values must match bitwise.
The same attempt path is reused sequentially because paths are part of checkpoint
lineage. The tests also freeze ZIP member timestamps: production archives include
wall-clock dates, which otherwise change archive hashes independently of numerics. The hook gets JSON values only; callback return values cannot affect
training. Global Python, NumPy and Torch RNG states are restored around SDK calls,
including failures. A separate test deliberately draws randomness and raises.
GPU nondeterminism outside deterministic test mode remains an accepted limit.
