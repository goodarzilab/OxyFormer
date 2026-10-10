# Stage admission and upstream observations

A stage's admission verifies the published results in its dependency lineage,
including acquisition receipts and the acquisition trees bound into producer
requests. Acquisition roots include all entries, including unlisted files.
Published stage roots use the existing publication scope: artifacts and execution
controls, with directory ancestors; coordinator-owned siblings are excluded.

A consumer re-verifies acquisitions that its producers read even when those
acquisitions are not among its own `StageRequest.dependency_paths`. The stage
runner threat model requires that no consumer accept a changed upstream attempt.
A producer's immutable receipt proves which acquisition bytes were admitted; it
does not prove that those bytes are still present. Dropping the recursive check
would let changed acquisitions pass through a previously published producer.
The same recursively collected roots remain in the consumer's post-exit check.

## Cost and lifetime

The runner creates one private temporary digest cache in the independent
publication store before admission. Its
worker process tree inherits the cache location. Each path has a lock spanning
lookup, a possible full read, and the final stat. Concurrent workers therefore
share the first completed digest. The runner removes the cache after its worker
and descendants exit and final observations finish. Every new `run` invocation,
including another invocation in the same interpreter, starts an empty cache.
Standalone verification outside a run has no cache.

Only a stable regular-file digest is reusable: its path, device, inode, size,
nanosecond mtime and nanosecond ctime must still match. Mode is checked as well.
The identity is checked before and after use; full reads also retain their
existing descriptor and pathname comparisons. A different path, even a hard
link, gets a separate digest. Changed identities require a full read; an
incomplete observation refuses the run and invalidates the affected digest.
The cache also covers repeated integrity hashes of small request/control files;
it never replaces actual reads of content needed by scientific computations.
No digests are reused between runs or written into upstream acquisitions.
Custom hash callbacks still execute and are checked against their requested
digests, but their return values never enter the shared SHA-256 cache. They run
outside the cache lock so callbacks may delegate to child readers without
deadlocking; descriptor and pathname comparisons still bracket the callback.

Matching stat tuples alone do not close a filesystem timestamp quantum: two
same-size writes can retain identical ctime and restored mtime. Before hashing
a version for reuse, the runner changes metadata on its private cache record
until that independent inode has a strictly later ctime than the input. This
uses the filesystem's own clock and ordering, without assuming its tick size or
comparing a remote filesystem clock with the client clock. It never writes to
the input. The original pre-read identity remains the comparison baseline
through this barrier and the subsequent hash. See the
[Linux timestamp ordering documentation](https://cdn.kernel.org/doc/html/latest/filesystems/multigrain-ts.html).

After establishing that clock boundary and before hashing, the reader calls
`fsync` on the input's read descriptor. Linux writeback cleans and write-protects
previously dirty shared mappings; a later mapped store must fault and update the
file's timestamps. Without this step, an already-writable mmap page could change
bytes without another ctime update. The ordering matters: reprotection follows
the timestamp barrier, so a subsequent write cannot share the admitted ctime
quantum. The reader does not write input bytes. A writeback error refuses the
observation and invalidates the digest. See the
[Linux writeback implementation](https://kernel.googlesource.com/pub/scm/linux/kernel/git/stable/linux-stable.git/+/refs/tags/v5.4.51/mm/page-writeback.c).

The witness must be on the same device as the input. Registered acquisitions and
their independent publication store share the run's state filesystem. Files on
another device retain full verification without digest reuse. An unreadable
witness, or a filesystem clock that cannot establish ordering within two seconds,
refuses the observation without claiming that input content changed.

Byte reuse does not skip tree enumeration, namespace checks, acquisition
baseline comparisons, publication authority/taint checks, or the independent
before/after-run identity snapshots. Restoring bytes or mtime cannot restore
kernel ctime. Metadata-only changes refuse the observing run without permanently
tainting the dependency. An I/O failure refuses that run; only positive evidence
of changed content or namespace establishes permanent taint. Later consumers can
retry with byte-identical restored inputs, subject to existing recorded taints.
These boundaries implement the owner's “Detection boundary” and “Restored
inputs” decisions in the stage runner threat model; ancestor-directory evasion
by a hostile local actor remains outside that model.

## Synthetic call-count regression

`tests/test_execution.py::test_acquisition_hash_once_per_stage_chain` creates an
11.5 MiB synthetic payload, one or three producer stages, and a consumer of every
producer. It counts full digest streams across `fingerprint_tree`,
`regular_file_hash`, and the provenance helper used by scientific stage
preflight. The payload is synthetic; no production acquisition is read.

At launch base `7a691dec30b90531ab937bb6fc9a98a7a4366236`, each inline producer
reads the payload six times: acquisition fingerprint, request digest,
runner preflight, stage preflight, post-exit fingerprint, and result verification.
The consumer reads it `2 * producers + 1` times: each producer's acquisition
fingerprint and request verification, then one post-exit fingerprint. Three
producers therefore cause seven reads in their consumer. The real CLI adds a
separate worker preflight for a directly consumed acquisition.

The regression expects exactly one full digest stream per unchanged payload in
each run, including post-exit observation. A second CLI regression counts the
runner, supervisor and worker together. Identity changes, failed stats,
in-flight writes and separate-run lifetime have dedicated regressions;
existing tests continue to exercise changed/restored content, namespace changes,
metadata-only refusals and retry behavior.

The timestamp-collision regression uses real `copy2` overwrites with restored
mtime and equal-length contents; it does not mock ctime. It demonstrated changed
bytes accepted with an identical stat tuple before the ordering barrier. The
custom-digest regression demonstrated a non-SHA callback value returned by a
subsequent built-in SHA-256 read before callback values were separated.

Shared-mapping and child-callback regressions reproduce unchanged-stat mmap
corruption and a parent/child lock deadlock before their respective repairs.
