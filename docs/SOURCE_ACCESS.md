# Source acquisition

Manifests record inspected releases, terms, dictionaries, selection and ceilings.
Ready covers acquisition, not ingestion: Census, ENDES and INEC are ready. US flag
dictionary/ACS index and Mexico lag L/CONAPO inputs remain blocked.

Outcome-blind DEM selection covers 954 tiles; eight missing state-intersecting
cells block DEM and every shard. Nine groups cover 48 contiguous states plus DC
once; border tiles may overlap. Current USGS URLs are mutable. No national
coverage claim, AK/HI/territories, or scientific approval changes.

DANE staging is read-only and empty. Local resources require a relative year
path, size and SHA-256. Colombia awaits the approvals.yaml fallback determination;
INEC works independently. No CAPTCHA bypass or application-based access.

```sh
PYTHONPATH=src python -m oxyformer.data.source_manifest fetch --source inec --output-dir "$SWARM_UNIT_DIR/inec-acquisition"
```

Blocked sources refuse before writes/network. Output must be a fresh private
attempt descendant. Success writes payload.tar, receipts.json and download.log;
failures remove partial payloads. Hashes, ceilings and 1–5 retries bound transfer.
No extraction/shared writes. Tests are offline; evidence stays in SWARM_UNIT_DIR.

The adapter normalizes framing before decoding. Identity content only; chunked,
consistent length, or close-with-expectation. Trailers require terminal CRLF and
≤64KiB/100 lines. Tree-free XML parsing reaches EOF without external retrieval;
HTML roots fail. Expectations never excuse framing failure.
