# Execution attempts and verification depth

A running stage fully verifies each acquisition it declares, at admission and
post-exit. Its request binds the acquisition file hashes and the full admitted
snapshot in `_execution/dependencies.json`. The runner records its post-exit
observation in `_execution/dependency_check.json`. Existing content-change,
restored-write, metadata-only and observation-error detection and taint rules
continue to apply to these directly declared inputs. No digest cache is used.

Under the owner's **Verification depth** decision of 2026-10-10, a consumer
verifies every stage publication in its complete lineage, including results,
published fingerprints, independent publication receipts, taint markers,
artifacts and execution controls. For a producer's acquisition inputs it checks
the recorded evidence: the request binds the admitted snapshot and its file
hashes, and the successful dependency check binds its final fingerprint to that
same snapshot. Both control files are authenticated by the producer's published
fingerprint. All other request inputs retain their live hash checks.

The consumer does not resolve, open or hash a transitive acquisition, and does
not add it to its admission snapshots or post-exit observations. Producers'
publications remain usable when those acquisition bytes change or their payload
or entire directory is archived or deleted. A stage that declares that same
acquisition still refuses a missing or changed input on its next run. Likewise,
an acquisition taint does not retroactively invalidate a previously verified
producer publication; a tainted or changed **stage publication** still refuses
its consumers. Direct declarations take precedence when a stage depends on both
a publication and its acquisition.

This boundary assumes consumers compute from their declared inputs and verified
publications. Detecting later changes to acquisitions they do not declare belongs
to the declaring stages. It is not protection against a hostile local actor.
