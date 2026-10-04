# Registered foundation and architecture comparisons

Built with PriorLabs-TabPFN.

The frozen plan sections 2, 8.4 and 10, its adoption note, and
`configs/approvals.yaml` authorize TabICLv2 and all TabPFN versions for this
nonprofit research project. This implementation selects TabPFN **v2 weights**
with package **9.1.0**, and TabICLv2 weights with package **2.2.0**. This is a
reproducibility choice, not a legal exclusion of later approved versions.
Official sources and installed source were inspected on 2026-10-04. No study
data, checkpoints or generated results belong in Git.

## Exact identities and provisioning

`configs/models/foundations.yaml` records four concrete checkpoint filenames,
immutable Hugging Face revisions, byte lengths and published LFS SHA-256 values.
These are published expected identities; this unit has **not downloaded or
executed the real checkpoints**. A provisioner must verify the actual local
bytes before use. Package pins live in `requirements/comparators.in`; installing
these optional packages is not part of primary package import or this unit.

| Package | Selected checkpoint | SHA-256 |
| --- | --- | --- |
| tabicl 2.2.0 | tabicl-classifier-v2-20260212.ckpt | bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0 |
| tabicl 2.2.0 | tabicl-regressor-v2-20260212.ckpt | 0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a |
| tabpfn 9.1.0 | tabpfn-v2-classifier-finetuned-zk73skhh.ckpt | cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49 |
| tabpfn 9.1.0 | tabpfn-v2-regressor.ckpt | 2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736 |

The coordinator receives `comparator_runtime_request.json` in `SWARM_UNIT_DIR`.
It archives exact inspected licence/model-card/third-party-notice text and hashes,
checkpoint metadata, and the request for an isolated provisioned runtime. The
shared environment is unchanged. Provisioning must retain wheels, transitive
versions and a lock; after a synthetic checkpoint smoke test it must record
`runtime_environment(package)` in the runtime manifest, along with CPU/platform
and threading information. A consumer constructs `Checkpoint` using that saved
fingerprint, not a new observation substituted for the expected environment.
`Checkpoint.verify()` compares Python, torch, numpy, scipy, scikit-learn and the
comparator package exactly and hashes local checkpoint bytes. Missing packages,
files, floating revisions or mismatched fingerprints block that comparator.
The expected runtime manifest and model registry are trusted configuration.

Official references: [TabICL 2.2.0 release](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0),
[TabICL model card](https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/README.md),
[TabPFN 9.1.0 source](https://github.com/PriorLabs/TabPFN/tree/v9.1.0),
[TabPFN local/offline documentation](https://github.com/PriorLabs/TabPFN/blob/v9.1.0/README.md).

## Selected terms and attribution archive

TabICL's selected model card declares BSD-3-Clause and requests citations to
TabICL (Qu et al., 2025) and TabICLv2 (Qu et al., 2026). The exact code licence,
linked by that card, is archived with SHA-256
`c959a1db8bc4b2f41c315a9b077c822372a4a84e1a83636bdb962715aaae6874`.
Preserve its copyright, conditions and disclaimer in source redistribution,
reproduce them in binary distribution materials, and do not imply endorsement.
[Selected TabICL licence](https://github.com/soda-inria/tabicl/blob/v2.2.0/LICENSE).

TabPFN 9.1.0 code is Apache-2.0, exact licence SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
Preserve the licence and relevant notices, identify changed files, and carry
applicable attribution when distributing. The third-party notices are separately
archived with SHA-256
`445dd62f4403e21d439674a892feddd2cf8c81061eddc5e94844b12cc98b30d3`.
[Code terms](https://github.com/PriorLabs/TabPFN/blob/v9.1.0/LICENSE),
[third-party notices](https://github.com/PriorLabs/TabPFN/blob/v9.1.0/THIRD-PARTY-NOTICES.md).

Both selected TabPFN v2 repositories carry identical Prior Labs License 1.1
(May 2025) bytes with SHA-256
`08b1851b4b6136b5fb0cb124d3258e0d91e3e1623f1be5904b77e4d270d112d8`.
These terms include Apache-style preservation requirements and section 10's
additional attribution. Distribution or making available covered material,
products or services requires a licence copy and prominent
“Built with PriorLabs-TabPFN” attribution on related documentation/interfaces.
Distributed AI models improved using the source, weights or outputs must also
begin their names with “TabPFN”. Internal benchmarking/testing without external
communication is explicitly excepted from section 10 attribution. The exact
archived text controls; this paragraph is an operational digest, not a replacement.
[Classifier weight terms](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/f851f2a3c941544733b712d8c0f96dfae9b28862/LICENSE.txt),
[regressor weight terms](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/4972a65a1b30806315c6f92499959ffbfc69a673/LICENSE.txt).

## Adapter contract and supported targets

`TabICLComparator` and `TabPFNComparator` expose the merged argument order:
`mean(a_query, x_tokens, raw_x, context, group_offset, design=None)` and
`logits(a_query, x_tokens, raw_x, context, group_offset)`, with an additional
`probability(...)` method for the origin classifier. As in the merged heads,
outputs have shape `[B,R]` for queries `[B,R,1]`. In these adapters `x_tokens`
is a nuisance `CovariateView`, and `raw_x` is its complete numeric matrix in
column order. No feature is replaced with a learned embedding. `None` cells
are NaN; numeric and binary covariates are supported. Categorical strings must
not be silently encoded by this adapter. County/geography is not a predictor.

- Identity outcomes return a predictive **mean**, never a median or effect.
  Binary Bernoulli outcomes return class-1 probability and require both classes
  in context. Poisson, negative-binomial and aggregated binomial targets are
  unsupported: counts cannot be silently reinterpreted as continuous rates.
- Both package `fit(X,y)` signatures lack sample weights. Callers must explicitly
  supply `weight_semantics="unit"` and all-one weights; target/survey semantics,
  unequal weights, zero weights and non-unit constants all block the comparator.
  Neither replication nor unweighted fitting substitutes for weighted training.
- `fit_outcome` consumes exactly `split.training_ids(fold)` in order. `fit_origin`
  consumes authoritative `PolicyPairs`: original rows then shifted rows, the same
  origin weights and X on both copies, and matching policy/weight identities.
  No labels from query views are accepted. Each fold needs a new adapter instance;
  refits are rejected. The context hash binds view, split, fold, training arrays,
  labels, model runtime, task, family and seed. Seeds must be registered.
- TabICL uses eight ensemble members, training-fitted none/power preprocessing,
  Latin feature permutations, package outlier handling (threshold 4), CPU FP32
  and no automatic checkpoint downloads. TabPFN uses the checkpoint's pinned
  inference/preprocessing configuration, eight fixed ensemble members, CPU FP32,
  and `fit_preprocessors`; its 9.1.0 loader receives
  `download_if_not_exists=False`, then isolated `ModelSpecs` objects bypass the
  estimator's download-capable path. No remote client is imported.
- Conservative operational caps are 30,000 training-context rows for TabICL,
  1,000 for the registered TabPFN CPU configuration, and 500 input columns
  including treatment. Origin pairs count twice. These are project caps, not
  claims of architectural maxima. Exceeding them blocks without subsampling or
  changing the target. A faster/larger runtime requires separately registered
  settings and validation, not an automatic override.
- A single query row is sent to the fitted estimator at a time, preventing query
  batching, duplication, ordering or other rows' missingness from changing a
  prediction. This is intentionally conservative and may be slow. Package-native
  missingness handling is retained; TabICL can mask all-NaN query columns.
- Nonzero/trainable PMA context and external group offsets are unsupported and
  rejected. These comparators cannot be substituted into a configuration that
  requires those inputs. They use raw mmHg and reject a treatment-design override.
  Probabilities are uncalibrated; fit calibration on inner training OOF records
  before using the existing prior-corrected ratio conversion. No implicit clipping,
  targeting or ratio fitting happens here.

Feature approvals, target-weight semantics, label alignment and authenticity of
runtime manifests remain responsibilities of the trusted data/training producer.
The adapters verify the merged view/split/spec boundaries; they cannot infer
whether a caller falsely labeled population weights as unit weights.

## Fixed variants

`build_variant` constructs independent nuisance heads and checks their existing
one-million-parameter cap. `VARIANTS` and `configs/models/ablations.yaml` use
independent names. A0 is primary. A1 resets encoder/PMA initialization and skips
SSL in the training schedule. A2 removes PMA while retaining county offsets and
raw X. A3 injects the frozen treatment basis before the feature encoder's blocks,
independently per query. A4 predicts unconstrained coefficients of an intercept
and the frozen seven-value treatment basis from encoded X, context and raw X.
A5 uses the merged signed Riesz head/loss without a positivity transform.

A6/F0 use TabICLv2/TabPFN outcomes with the primary transformer origin nuisance.
A7/F1 use compatible foundation outcomes and origin probabilities. Both roles
must independently pass target, fold, context, weight and calibration checks.
A8 remains the plan's months 2–3 task; B0 conventional learners remain calibration
benchmarks and D0 geographic inputs remain diagnostic-only. The factory refuses
these three as production configurations. The full approved raw-X bypass remains
in every production transformer variant. Architecture, seed, effect direction or
statistical significance must not be used to select a favorable effect estimate.

## Verification boundary

`tests/test_comparators.py` uses synthetic immutable contracts, mocked checkpoint
bytes and mocked optional packages with networking denied. It exercises local
loader flags, missing dependencies, primary import isolation, checkpoint/runtime
mismatches, unsupported targets/weights, fold isolation, query consistency and
architecture distinctions. This establishes adapter behavior, not real checkpoint
accuracy, performance, numerical reproducibility across platforms, or causal
validity. Real checkpoint smoke tests belong to the provisioning request.
