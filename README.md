# OxyFormer

OxyFormer v2 is a research implementation of support-qualified continuous
exposure-shift estimation. Fold-nested transformers estimate outcome responses
and density ratios; cross-fitted one-step and CV-TMLE estimates retain aligned
geographic uncertainty. Geographic prediction does not establish an
oxygen-specific intervention effect, and this software is not for patient care.

The [frozen science plan](docs/plan/OXYFORMER_V2_PLAN.md),
[owner mandate](docs/orchestrator-mandate.md) and read-only
[approvals registry](configs/approvals.yaml) govern this run.

| Guide | Purpose |
| --- | --- |
| [Execution](docs/EXECUTION.md) | Merged receipts, attempt clones, scientific interfaces, CPU verification and current integration blockers |
| [Scientific gates](docs/SCIENTIFIC_GATES.md) | Source/feature permissions, recipe lock, complete evidence and release authorization |
| [Phase B](docs/PHASE_B.md) | Measured campaign admission, bounded leaves, exact continuation and complete collectors |
| [Artifact interfaces](docs/INTERFACES.md) | Immutable scientific identities and merged request/result field names |

**Production campaign admission remains blocked** by the documented merged
dispatcher/stage interface gaps and missing campaign-specific allocations.
The synthetic end-to-end test exercises the merged direct scientific APIs;
its shortened fits and deliberately diagnostic report are not production
coverage or clinical release evidence. See the execution guide for exact
reproductions and scope limits.

The current package is under `src/oxyformer/`. Tests use synthetic fixtures,
CPU execution and no network. Run from the checkout on a Slurm compute node:

```sh
srun --partition=standard --account=root --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --time=00:15:00 \
  timeout 600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -m pytest -q tests
```

Repeat with `/mnt/weka/home/hgoodarzi/oxyformer-swarm/envs/ci-cpu/bin/python`
before every review. Never run suites, probes, mutations or acquisition reads
on the login node. A passing suite verifies implementation behavior; scientific
release still requires every gate and scoped owner approval.

Historical phase scripts and assets remain for explicit legacy reproduction.
Their county embeddings, ridge/PLR estimator and older adjustment set describe
that historical pipeline, not the v2 primary estimator. Consult
`configs/legacy_entrypoints.yaml` and the merged legacy launcher before use.
Root data CSVs, `outputs/` and `report/` are read-only for this run. Keep
downloads, caches, checkpoints and generated reports in attempt-owned storage;
never commit them. Shared-root promotion is absent from this plan and requires
the PI if requested later.

See [LICENSE](LICENSE) for the repository's restrictive proprietary terms.
Source and model licences apply independently.
