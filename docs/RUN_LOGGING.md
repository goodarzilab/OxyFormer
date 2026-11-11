# Run logging

Use a fresh exact WandbLogger(base_directory) per attempt; nested SSL shares it.
None/noop disable telemetry. Other callables/subclasses fail before training.
Install the optional SDK only in the separate oxyformer-p3 environment.

Capture is redacted JSON in RAM: loss, LR, applied gradient norm, throughput,
Torch peak, evaluations, selection, declared config/provenance, seed/fold,
hardware/job, checkpoint pins/paths and resume lineage. SSL updates/skips carry
processed_batches; only updates emit train_step. CONFIG_SCHEMA classifies every
field. Inspect the original config before copying: extras refuse telemetry with
logging_ok=False and a flagged snapshot. Sealed refusal survives dropped events
and replay. Stop snapshots read stored none/requested/honoured state, not polling.

All I/O follows scientific checkpoint persistence. Git owns a process group and
ten-second timeout; kill/reap it before starting SDK delivery. Errors record
unavailable: reason. The disposable SDK/spool worker has a 180-second deadline
and is killed/reaped on completion/timeout, isolating native stalls and RNG.

Lazy SDK init is attempted once, always offline, targeting goodarzilab/oxyformer-v2.
Each attempt owns its directory/status. Pending/null means incomplete; logging_ok
and sync_ok are separate. Post-fit sync has a 120-second timeout and needs runtime
WANDB_API_KEY. Never dump credentials/environment. Matching requires >=32-character
keys; secret-labelled fields remain redacted and credential-bearing paths refuse.
Spools encode protocol counts/digests as boolean bits, avoiding numeric-key
collisions, and seal capture/refusal; invalid spools cannot claim completion.
WandbLogger.replay("/path/to/events-complete.jsonl") preserves failures and original
status. Sync existing runs with python -m wandb sync /path/to/offline-run.

GPU windows synchronize/reset Torch peak before the step and sample after update,
including live allocations. Non-Torch allocations/context are excluded; CPU is zero.
The logger owns peak counters. Hardware measurements and GPU nondeterminism are
not universal guarantees; online upload is unverified. The following rulings apply,
with A2/B2 replacing affected sentences. No deadline refund, checkpoint format
change, predecessor deletion or archive-hash normalization is used.

Requirement provenance: orchestrator ruling 2026-11-11

RULING A (N2, capture time versus checkpoint invariance). Per-step capture is in-memory only and bounded: each capture call must cost well under one millisecond on the CPU test double (add a test that times 1,000 capture calls and asserts a mean below 1 ms, with no I/O, Git, subprocess, network or SDK call). Bitwise invariance (C4: logging off / no-op / on identical per-step losses and checkpoint SHA-256) is REQUIRED for fits bounded by step, batch or epoch counts (batch_limit, max_epochs, patience, requested). For wall-clock slice_limit fits, capture time counts against the slice like any other in-process work: the slice boundary may move by the bounded capture cost, but per-step numerics are unchanged and the resumed trajectory reaches the same result (test: logging-on versus logging-off under slice_limit produce identical per-step losses for their common steps, and after resume to the same batch count identical final checkpoint SHA-256). Document this as an accepted limit of wall-clock slices. The reviewer's 35-second capture double is OUTSIDE the envelope (capture cost is bounded by the new test): record it in evidence.md as out-of-envelope with the measured bound; do not refund or exclude capture time from the scientific deadline (that would change deadline accounting, which N2 forbids).

RULING B (N3, numeric retention versus exact-key secrecy). The credential envelope is a Weights & Biases API key of at least 32 characters (real keys are 40). Redaction covers the exact key, its decimal-integer form and its leading-zero variants, in values and mapping keys, only for keys that meet that length; metric values are never altered unless they contain such a match. A 5-digit synthetic key is outside the envelope: tests use 40-character keys (the existing leading-zero regression keeps WANDB_API_KEY constructed at runtime as `"0" + "1234567890" * 3 + "123456789"`). Record the reviewer's 12345/123456 case in evidence.md as out-of-envelope. C3 (no key, token or key-shaped string in any file, log, config dump or the repository) stands unchanged.

Requirement provenance: orchestrator clarification 2026-11-11

CLARIFICATION A2 (to RULING A, slice_limit resume test). Checkpoint archives embed resume ancestry (metadata.predecessor_hash and ArtifactLineage.parent_hashes, src/oxyformer/training/checkpoint.py, outside this unit's scope), so two runs whose wall-clock slices cut at different steps legitimately have different archive SHA-256 values even when training is bitwise identical. Therefore: for slice_limit fits the invariance test compares (i) the full per-step loss trajectory, (ii) the complete encoded training state and (iii) every tensor payload byte of the final checkpoint, plus all archive metadata EXCEPT the lineage and predecessor_hash keys; those three must be identical between logging off and logging on, and the lineage/predecessor difference is recorded as expected. Full archive SHA-256 equality stays REQUIRED for count-bounded fits (batch_limit, max_epochs, patience, requested), whose ancestry is identical by construction. Your check-ruling-a-lineage.py measurement (0.255 ms mean capture; identical losses, state and tensors; only lineage and predecessor_hash differ) already satisfies A2; turn it into the regression. Do not change checkpoint.py, drop predecessor binding or normalize archive hashes.

CLARIFICATION B2 (to RULING B, documentation). docs/RUN_LOGGING.md and SUMMARY.md must never contain a literal key-shaped string. Express the leading-zero example compositionally, for example: "a synthetic 40-digit key consisting of the digit 0 followed by the digits 1-9 and 0 repeated to length 40", or in code form '"0" + "1234567890" * 3 + "123456789"', and keep building the test fixture at runtime as it does today. The retained test_secret_hygiene_in_run_tree_and_repository stays unchanged and must pass on the docs. Where my ruling text is quoted in your records, replace the literal assignment with that compositional description.
