# Source acquisition contracts

Inspected 2026-10-04 against frozen plan sections 7–9, section 3.1, the mandate
and read-only approvals. Ready means every acquisition resource was inspected;
future transfer success and scientific ingestion validity remain separate.
No real data or generated reports enter Git.

| Source | Status and remaining limits |
| --- | --- |
| `us` | Blocked: accessible USALEEP File A flag dictionary not located; ACS tract ZIP suffix inspection did not yield a central directory. FTP File A, ACS geography/documentation and pinned elevcan source/license inspected. |
| `census` | Ready: 49 block and 49 matching SF1 archive indexes inspected, with bulk SF1 layout, TIGER dictionary and terms. Full transfers, population reconciliation and joins remain downstream. |
| `dem` | Blocked: full-resolution 2010 state polygons intersect 962 cells; 954 approved 3DEP 1/3-arc-second tiles have inspected TIFF headers and XML. Eight cells lack current tiles. National raster coverage is incomplete. |
| `endes` | Ready: complete 2023/2024 archives, nested ZIP CRCs, 34 module headers per year, three dictionaries per year and ODbL terms inspected. 2024 remains locked replication. |
| `births` | Blocked: no DANE files staged; Colombia drop awaits the orchestrator determination required by approvals.yaml. No CAPTCHA bypass. |
| `inec` | Ready independently of DANE: Ecuador 2024/2015 archives, dictionaries and terms inspected. ENV and fetal deaths remain separate. |
| `mexico` | Blocked: EDR registration 2015–2019 archives and dictionaries inspected, but common lag L/later-period years are unset. CONAPO returns an HTTP-200 challenge; denominators, terms and geographic compatibility remain unverified. |

Only committed source IDs are accepted. Shared defaults and compact DEM rows
expand before validation and hashing. Each observed tile records TIFF/XML byte
ceilings and XML publication date (including year-only precision); the fixed
HTTPS template reproduces inspected bucket keys. No runtime source discovery.

## Source constraints

Source terms and dictionary observations are in each manifest; science approvals stay fixed.

DEM selection intersects full-resolution TIGER2010 state geometry without
outcomes, flags or eligibility filters. Nine disjoint jurisdiction groups cover
the 48 contiguous states plus DC once. Border tiles can serve multiple groups;
each group caps its TIFF plus XML bytes. Eight missing cells cannot be dismissed
as water without inhabited-block reconciliation; all shards remain blocked.
AK/HI/territories are excluded. Every selected XML reports NAD83, NAVD88 and
meters. Preserve its exact overlapped footprint and use constraints. Current
USGS URLs can change; dates and observed sizes do not make them immutable.

ENDES raw/adjusted Hb and altitude definitions are recorded in endes.json.
Survey scaling, missing values and joins remain ingestion checks. Preserve ODbL
conditions; no application-based DHS source is substituted.

DANE catalogs 876/878/915 require human CAPTCHA acquisition. Local resources use
`transport: "local"`, `local_path` under an approved staging year (2023–2025),
and mandatory `expected_bytes` and `expected_sha256`. Their HTTPS URL is catalog
provenance only. The reader rejects symlinks/traversal and nonregular files and
copies verified bytes into its private attempt; staged permissions/bytes remain
unchanged. Staging alone does not establish terms or dictionary completeness.
No files are staged and Colombia has not been declared dropped.

INEC permits scientific/statistical use with attribution and aggregate reporting.
Retain the stricter 2015 no-redistribution/no-reidentification conditions despite
the general CC-BY footer. Units and residence joins remain ingestion gates.
Mexico occurrence years need registration releases through 2019+L with equal
cutoffs; no lag or CONAPO compatibility is assumed. Preserve INEGI attribution.

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
