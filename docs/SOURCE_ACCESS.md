# Source acquisition

Manifests distinguish resource inspection from downstream ingestion and scientific
coverage. Census, ENDES, INEC and DANE births are acquisition-ready. Every ready
requirement names verified resources; any unresolved requirement blocks its source
before network access or output creation. INEC remains independently usable.

The 2026-10-04 inspection implements the approvals in `configs/approvals.yaml`:

- **DANE births:** all nine owner-staged ZIPs for 2023–2025 have exact sizes,
  SHA-256 hashes, member inventories and verified ZIP CRCs. Catalogs 876, 878 and
  915 returned their dictionary and terms pages to a non-browser HTTPS client.
  Each year includes births, nonfetal deaths and fetal deaths. Attribution and
  restrictions on redistribution remain recorded; Colombia is not dropped.
- **US:** the staged USALEEP record layout is 70,785 bytes, SHA-256
  `7612e16609d73f958ff6618ae70aab51d6502745c093c307a7f217c967e24431`.
  File A flag definitions were inspected. The approved endpoint covariate registry
  is referenced; raw acquisition does not authorize additional predictors or
  flag-2/3 training. Only the ACS giant-archive structure requirement remains
  blocked: HTTP 206 ranges report 3,369,803,296 bytes, but the object starts with
  gzip magic despite its `.zip` name. Matching suffix and absolute tail ranges
  (65,557 bytes and 1 MiB) contain no ZIP/ZIP64 end record or central directory.
  This is an inspection of byte ranges, not a complete download or verified archive.
- **DEM:** retain all 954 inspected 1/3 arc-second tiles. The approved 1 arc-second
  fallback is restricted to the original eight missing cells. Only `n46w083`,
  `n48w086` and `n49w088` have inspectable objects: record each product, TIFF size
  and ceiling, paired XML size/hash, and year-only 2013 publication date. Their
  combined 24,549,593 bytes are added to the east-north-central shard budget.
  `n27w080`, `n29w091`, `n40w074`, `n41w072` and `n43w070` have no objects in either
  current or historical USGS 1 arc-second TIFF listings (HTTP 200, untruncated,
  empty listings). These five requirements keep DEM and the atlas blocked;
  no fallback size is invented. Nine divisions cover 48 contiguous states plus
  DC once; border tiles can overlap. No inhabited-block raster coverage is
  certified. AK, HI and territories are excluded; current USGS URLs are mutable.
- **Mexico:** full ZIP integrity, sizes, hashes and embedded EDR dictionaries were
  inspected for registration years 2015–2024. The approved primary lag is two
  years (2015–2021 registrations); the five-year check uses 2015–2024 registrations
  for occurrence years 2015–2019. The CONAPO midyear municipality/sex/five-year-age
  CSV independently matches 36,658,654 bytes and SHA-256
  `1a8f07be08de082a0c33404f0fbce9d9292845a8c8290153a6ad2e1889bab31a`.
  Its official datos.gob.mx catalog/API declares CC BY 4.0. Mexico stays blocked
  only for the unverified CONAPO field dictionary (official datastore request
  returned HTTP 403 Access Denied) and an uninspected geographic crosswalk to EDR.
  Shared key names alone do not establish denominator compatibility. Old fetal
  dictionary links for EDR 2015/2016 are replaced by the inspected dictionaries
  embedded in the corresponding general-death archives.

Local resources use `transport=local`, a `staging_key` from
`owner_decisions.manual_acquisition`, and a relative `local_path`. Only the
approval supplies the root; the USALEEP approval also fixes the filename and
DANE retains its approved year directories. Size and SHA-256 are mandatory and
rechecked during copying. Staged files are opened read-only; bytes and permissions
are never changed. Traversal, symlinks and nonregular files fail. No CAPTCHA
bypass or application-based access is used. Live HTML/catalog snapshot hashes
are observational inspection evidence, not immutable transfer expectations.

```sh
PYTHONPATH=src python -m oxyformer.data.source_manifest fetch --source births --output-dir "$SWARM_UNIT_DIR/births-acquisition"
```

Output must be a fresh private attempt descendant. Success writes `payload.tar`,
`receipts.json` and `download.log`; failure removes partial payloads. Hashes,
ceilings and 1–5 retries bound transfer. No extraction or shared writes. Tests
use synthetic fixtures without network calls; inspection evidence lives outside
Git in `SWARM_UNIT_DIR/access_report.json`.

Redirects fail before a target request; redirect/error bodies stay unread.
HTTP HTML needs an HTML media type. The adapter normalizes payload framing:
chunked, consistent length, or close-with-expectation. Trailers require terminal
CRLF and ≤64KiB/100 lines. Tree-free XML parsing reaches EOF without external
retrieval; HTML roots fail; media-type parameters do not change the type.
Expectations never excuse framing failure.
