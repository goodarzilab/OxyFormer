"""Local, reviewed USALEEP File A ingestion; no acquisition or eligibility fitting.

``load_usaleep(bundle, mapping)`` returns (primary_outcomes, audit). ``bundle``
is a sequence of SourceFile records (one File A); ``mapping`` is the entire
configs/adapters/us.yaml document. Each manifest binds its local bytes and the
source section's canonical JSON hash. Audits are privileged diagnostic material,
never covariate views. County/support/split eligibility is applied downstream.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from oxyformer.contracts import SourceManifest
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule


OWNER_APPROVALS = Path(__file__).resolve().parents[4] / "configs" / "approvals.yaml"


def primary_outcome_flags(approvals=None):
    """Read the owner amendment; a legacy plan/mapping value is never a fallback."""
    if approvals is None:
        # Cache parsing, not file contents: changed or missing approvals are
        # observed on every call, including repeated per-tract eligibility checks.
        return _primary_flags_from_text(OWNER_APPROVALS.read_text())
    owner = approvals.get("owner_decisions") if isinstance(approvals, dict) else None
    decision = owner.get("outcome_flags") if isinstance(owner, dict) else None
    valid = isinstance(decision, dict)
    if valid:
        for key, expected in (("primary", [1, 3]), ("excluded", [2])):
            values = decision.get(key)
            valid = valid and isinstance(values, list) and values == expected and all(
                type(flag) is int for flag in values)
        valid = (valid and decision.get("county_minimum_counts") == "primary"
                 and decision.get("sensitivity") == "flag1_only")
    _require(valid, "missing or malformed owner_decisions.outcome_flags")
    return tuple(decision["primary"])


def _reject_duplicate_approval_keys(text):
    """Inspect explicit keys before construction discards duplicates.

    Composition preserves keys and aliases without applying YAML merges. Valid
    merge overrides stay valid; visiting an alias twice is not a duplicate key.
    """
    pending = [yaml.compose(text, Loader=yaml.SafeLoader)]
    visited = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, yaml.MappingNode):
            keys = [(key.tag, key.value) for key, _ in node.value
                    if isinstance(key, yaml.ScalarNode)]
            _require(len(keys) == len(set(keys)),
                     "duplicate YAML key in owner_decisions.outcome_flags approval document")
            pending.extend(child for pair in node.value for child in pair)
        elif isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)


@lru_cache(maxsize=1)
def _primary_flags_from_text(text):
    try:
        _reject_duplicate_approval_keys(text)
        approvals = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError("malformed owner_decisions.outcome_flags document") from exc
    _require(isinstance(approvals, dict), "missing or malformed owner_decisions.outcome_flags")
    return primary_outcome_flags(approvals)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    manifest: SourceManifest


def mapping_hash(section):
    return sha256(json.dumps(section, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mapping(mapping, source):
    _require(mapping.get("schema_version") == 1, "unresolved mapping version")
    section = mapping[source]
    _require(section["geography_vintage"] == 2010, "2010 tract geography required")
    _require(section["mapping_status"] == "reviewed", "unreviewed mapping")
    _require(section.get("dictionary_sha256") and section.get("dictionary_url"),
             "missing source dictionary evidence")
    return section


def _read(source_file, section, source_id):
    manifest = source_file.manifest
    manifest.assert_usable()
    _require(manifest.source_id == source_id, "wrong source release")
    _require(manifest.version == section["release"], "wrong source vintage")
    _require(manifest.schema_hash == mapping_hash(section), "mapping hash mismatch")
    _require(dict(manifest.field_mapping) == section["field_mapping"],
             "source field mapping mismatch")
    payload = Path(source_file.path).read_bytes()
    _require(sha256(payload).hexdigest() == manifest.payload_hash, "source payload hash mismatch")
    return payload.decode(section["encoding"])


def _ids(frame, name="frame"):
    _require("original_id" in frame, f"missing original_id in {name}")
    _require(all(isinstance(x, str) and re.fullmatch(r"[0-9]{11}", x)
                 for x in frame.original_id), f"invalid tract ID in {name}; strings required")
    _require(not frame.original_id.duplicated().any(), f"duplicate tract IDs in {name}")


def join_outcomes(covariates, outcomes):
    """Explicit score/training-side inner join; never mutate the unlabeled frame.

    Return joined rows and IDs excluded on each side. Support and split rules are
    still required. IDs are never coerced and duplicate keys fail before joining.
    """
    _ids(covariates, "covariates")
    _ids(outcomes, "outcomes")
    _require(tuple(outcomes.columns) == ("original_id", "life_expectancy_years"),
             "expected primary outcome frame")
    _require(not (set(covariates) & (set(outcomes) - {"original_id"})),
             "outcome already present in covariates")
    joined = covariates.merge(outcomes, on="original_id", how="inner", validate="one_to_one", sort=False)
    covariate_ids, outcome_ids = set(covariates.original_id), set(outcomes.original_id)
    return joined, {
        "acs_without_primary_outcome": tuple(x for x in covariates.original_id
                                              if x not in outcome_ids),
        "primary_outcome_without_acs": tuple(x for x in outcomes.original_id
                                             if x not in covariate_ids),
        "join_cardinality": "one_to_one",
    }


def load_usaleep(bundle, mapping, *, numeric_identifiers=False):
    """Validate raw File A and retain owner-approved flags in the primary frame.

    numeric_identifiers enables the pinned public CSV's unpadded numeric FIPS
    representation. Existing direct callers keep the strict fixed-width API.

    A missing/unknown outcome, SE or mortality flag fails closed: the inspected
    release contains only finite estimates/SEs and flags 1/2/3, with no documented
    missing token. Absence of a tract is audited by the later explicit join.
    """
    primary_flags = primary_outcome_flags()
    section = _mapping(mapping, "usaleep")
    expected_fields = {
        "Tract ID": "original_id", "STATE2KX": "state_fips", "CNTY2KX": "county_fips",
        "TRACT2KX": "tract_code", "e(0)": "life_expectancy_years",
        "se(e(0))": "standard_error_years", "Abridged life table flag": "mortality_input_flag",
    }
    _require(section["field_mapping"] == expected_fields, "unresolved USALEEP fields")
    _require(section["flags"] == {"1": "observed", "2": "predicted", "3": "mixed"},
             "mortality flag interpretation differs from inspected CDC layout")
    # Retain the reviewed mapping/hash, whose original primary declaration is
    # amended by owner_decisions.outcome_flags, not by rewriting source metadata.
    _require(section["primary_flags"] == ["1"], "reviewed legacy primary mapping differs")
    _require(section["units"] == "years", "USALEEP units must be years")
    _require(len(bundle) == 1, "exactly one USALEEP File A required")
    reader = csv.DictReader(io.StringIO(_read(bundle[0], section, "usaleep_2010_2015")))
    _require(reader.fieldnames == list(expected_fields), "USALEEP header differs from inspected release")
    records = list(reader)
    _require(bool(records), "empty USALEEP File A")
    _require(all(set(r) == set(expected_fields) and None not in r.values() for r in records),
             "malformed USALEEP record")
    frame = pd.DataFrame(records).rename(columns=expected_fields)
    # The pinned CDC CSV writes these numeric identifiers without leading
    # zeroes. Restore only their documented fixed widths, preserving strings;
    # decimal, signed, whitespace and over-width tokens are never coerced.
    for column, width in (("original_id", 11), ("state_fips", 2),
                          ("county_fips", 3), ("tract_code", 6)):
        pattern = rf"[0-9]{{1,{width}}}" if numeric_identifiers else rf"[0-9]{{{width}}}"
        _require(frame[column].str.fullmatch(pattern).all(), f"invalid {column}")
        if numeric_identifiers:
            frame[column] = frame[column].str.zfill(width)
    _ids(frame, "USALEEP")
    _require((frame.original_id == frame.state_fips + frame.county_fips + frame.tract_code).all(),
             "USALEEP 2010 tract components disagree")
    _require(frame.mortality_input_flag.isin(section["flags"]).all(), "unknown mortality flag")
    for column in ("life_expectancy_years", "standard_error_years"):
        _require(frame[column].str.fullmatch(r"[0-9]+(?:\.[0-9]+)?").all(),
                 f"unresolved special value in {column}")
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        _require(frame[column].map(lambda x: float('-inf') < x < float('inf')).all(),
                 f"nonfinite {column}")
    _require((frame.life_expectancy_years > 0).all(), "nonpositive life expectancy")
    eligible = frame.mortality_input_flag.isin([str(flag) for flag in primary_flags])
    primary = frame.loc[eligible, ["original_id", "life_expectancy_years"]].reset_index(drop=True)
    frame["mortality_input_kind"] = frame.mortality_input_flag.map(section["flags"])
    frame["primary_label_available"] = eligible
    frame["exclusion_reason"] = frame.mortality_input_kind.map(
        {"observed": "", "predicted": "predicted_mortality_inputs", "mixed": "mixed_mortality_inputs"})
    frame.loc[eligible, "exclusion_reason"] = ""
    registry = FeatureRegistry(registry_id="usaleep-observation-metadata", rules=tuple(
        FeatureRule(name=name, role="outcome" if name == "life_expectancy_years" else
                    "identifier" if name == "original_id" else "outcome_metadata",
                    endpoints=("usaleep_life_expectancy",),
                    uses=("score",) if name == "life_expectancy_years" else
                    ("linkage",) if name == "original_id" else ("diagnostic",),
                    approval_id=section["approval_id"])
        for name in frame.columns))
    return primary, {"metadata": frame, "registry": registry,
                     "mapping_hash": mapping_hash(section),
                     "source_manifests": tuple(f.manifest for f in bundle),
                     "primary_rows": len(primary), "excluded_rows": int((~eligible).sum())}
