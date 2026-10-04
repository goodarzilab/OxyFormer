# Source acquisition contracts

Inspected 2026-10-04 against plan sections 3.1 and 7–9, the mandate and read-only
approvals. Ready means every acquisition resource was inspected. Future transfer
success and scientific validity remain separate.

| Source | Status and remaining limits |
| --- | --- |
| `us` | Blocked: accessible USALEEP File A flag dictionary not located; ACS tract ZIP suffix inspection did not yield a central directory. FTP File A, ACS geography/documentation and pinned elevcan source/license inspected. |
| `census` | Ready: 49 block and 49 matching SF1 archive indexes inspected, with bulk SF1 layout, TIGER dictionary and terms. Full transfers, population reconciliation and joins remain downstream. |
| `dem` | Blocked: full-resolution 2010 state polygons intersect 962 cells; 954 approved 3DEP 1/3-arc-second tiles have inspected TIFF headers and XML. Eight cells lack current tiles. National raster coverage is incomplete. |
| `endes` | Ready: complete 2023/2024 archives, nested ZIP CRCs, 34 module headers per year, three dictionaries per year and ODbL terms inspected. 2024 remains locked replication. |
| `births` | Blocked: no DANE files staged; Colombia drop awaits the orchestrator determination required by approvals.yaml. No CAPTCHA bypass. |
| `inec` | Ready independently of DANE: Ecuador 2024/2015 archives, dictionaries and terms inspected. ENV and fetal deaths remain separate. |
| `mexico` | Blocked: EDR registration 2015–2019 archives and dictionaries inspected, but common lag L/later-period years are unset. CONAPO returns an HTTP-200 challenge; denominators, terms and geographic compatibility remain unverified. |

Committed source IDs, shared defaults and compact DEM rows expand before
validation/hashing. Tile rows record TIFF/XML byte ceilings and XML dates at
observed precision. Fixed HTTPS templates reproduce inspected bucket keys.

## Source constraints

Terms and dictionary observations are recorded in each manifest. Ready status
covers acquisition; units, joins, weights and population reconciliation remain
ingestion checks. No scientific approvals were changed.

DEM selection uses full-resolution TIGER2010 state geometry without outcomes,
flags or eligibility filters. The 48 contiguous states and DC each occur once;
border tiles may serve multiple groups. TIFF/XML sums bound group transfers.
Eight missing cells need inhabited-block reconciliation, so all shards remain
blocked. AK/HI/territories are excluded. Selected XMLs report NAD83, NAVD88 and
meters; preserve their exact footprints and use constraints. Current USGS URLs
are mutable despite recorded dates and sizes.

DANE local resources use `transport: "local"`, a `local_path` under the approved
2023–2025 staging folders, and mandatory `expected_bytes`/`expected_sha256`.
The HTTPS URL is catalog provenance. The reader rejects symlinks, traversal and
nonregular files, then copies and hashes without changing staged bytes or modes.
Staging alone does not establish terms/dictionary completeness. Colombia awaits
the orchestrator's fallback determination; INEC works independently.

Preserve ENDES ODbL conditions and locked 2024 replication. INEC permits scientific
and statistical use with attribution and aggregate reporting; retain stricter
2015 no-redistribution/no-reidentification conditions despite the CC-BY footer.
Mexico requires registration years through 2019+L with equal occurrence cutoffs;
no lag or CONAPO compatibility is assumed. Preserve INEGI attribution.

## CLI and integrity

```sh
PYTHONPATH=src /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python \
  -m oxyformer.data.source_manifest fetch --source inec \
  --output-dir "$SWARM_UNIT_DIR/inec-acquisition" --attempts 3 --timeout 30
```

`validate` checks structure, even if blocked. `fetch` rejects blocked contracts
before network/writes. Output must be a fresh strict descendant of the assigned
exclusive attempt root, outside the repo. Trusted root aliases resolve once;
symlinks below it and existing destinations are refused. No shared cache writes
or promotion. Contracts/root assignment are trusted; hostile concurrent filesystem
replacement is outside scope.

Success writes `payload.tar`, `receipts.json`, `download.log`. Archives stay
opaque. Every resource and the tar receive streamed SHA-256 hashes; receipts
record manifest identity, byte counts and provenance. An observed hash audits
bytes; a declared hash checks an expectation. HTTPS redirects, signature checks,
byte ceilings and framing checks prevent known failed transfers from publishing
a complete receipt. Parser-selected chunking takes precedence over Content-Length;
unframed bodies need an independent length/hash expectation. Retries are bounded
to 1–5; 403/404 and content/hash failures are not retried. Timeout is per socket
operation. Failure retains the log and failed receipt and removes the tar.

```sh
timeout 600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -m pytest -q tests/test_source_manifest.py
```

Tests use synthetic fixtures and forbid networking. Mutation verification disables
blocked-manifest rejection, requires its regression to fail, then restores bytes
and reruns tests. All test/review/completion evidence belongs in `$SWARM_UNIT_DIR`.
