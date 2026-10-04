"""Historical ACS Summary File adapter, independent of outcome availability.

Input is a sequence of SourceFile objects for extracted g20105ss.txt geography
files and the six selected e/m20105ssNNNN000.txt sequence pairs. These are the
original fixed-width/CSV payloads, not API exports or prejoined tract tables.
The caller supplies reviewed manifests for each payload; this adapter never
fetches or extracts an archive. Acquisition of the complete US archive remains
blocked in configs/sources/us.json until its separate verification is resolved.

Return (covariates, audit): covariates has original_id plus approved concepts.
Raw estimates, MOEs, special-value annotations and geography remain in audit.
Use predictor_view for permission-checked access. No fitting, imputation or
standardization is performed here; each belongs inside training folds.
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import re

import pandas as pd
import yaml

from oxyformer.data.adapters.usaleep import _ids, _mapping, _read, _require, mapping_hash
from oxyformer.data.feature_roles import FeatureRegistry, FeatureRule


ENDPOINT = "usaleep_life_expectancy"
APPROVALS = Path(__file__).resolve().parents[4] / "configs/approvals.yaml"


def _registry(section):
    approvals = yaml.safe_load(APPROVALS.read_text())
    approved = approvals["owner_decisions"]["endpoint_covariates"][ENDPOINT]
    _require(approved["source"] == "acs_2006_2010_5yr", "historical ACS approval missing")
    concepts = section["concepts"]
    _require(set(concepts) == set(approved["concepts"]), "unapproved or missing ACS concepts")
    _require(section["masking_families"] == approved["masking_families"],
             "semantic masking families differ from approval")
    rules = []
    for name, concept in concepts.items():
        expected = approved["concepts"][name]
        _require({k: concept[k] for k in expected} == expected, f"unapproved definition: {name}")
        _require(concept["role"] == "predictor", f"invalid feature role: {name}")
        _require(concept["approval_id"] == f"configs/approvals.yaml#owner_decisions.endpoint_covariates.{ENDPOINT}",
                 f"missing concept approval: {name}")
        _require(concept["family"] in section["masking_families"] and
                 name in section["masking_families"][concept["family"]], "missing masking family")
        table = section["tables"].get(concept["table"])
        _require(table is not None, f"missing source table: {concept['table']}")
        for line in concept["num"] + ([concept["den"]] if "den" in concept else []):
            _require(str(line) in table["lines"], f"unresolved source line: {name}/{line}")
        rules.append(FeatureRule(name=name, role=concept["role"], endpoints=(ENDPOINT,),
                                 uses=("ssl", "nuisance", "context", "diagnostic"),
                                 approval_id=concept["approval_id"]))
    return FeatureRegistry(registry_id="acs-2006-2010-usaleep-approved", rules=tuple(rules))


def predictor_view(covariates, audit, columns=None, use="ssl"):
    """Copy only explicitly permitted concept values, indexed by tract string ID."""
    _require(use in ("ssl", "nuisance", "context"), "invalid predictor use")
    _ids(covariates, "ACS covariates")
    names = list(columns) if columns is not None else [r.name for r in audit["registry"].rules]
    _require(len(names) == len(set(names)), "duplicate predictor columns")
    for name in names:
        audit["registry"].require(name, ENDPOINT, use)
    return covariates.set_index("original_id").loc[:, names].copy(deep=True)


def _geography(text, state, layout):
    records = {}
    for line in text.splitlines():
        _require(len(line) >= 218, "short ACS geography record")
        row = {name: line[start - 1:start - 1 + width].strip()
               for name, (start, width) in layout.items()}
        _require(row["FILEID"] == "ACSSF" and row["STUSAB"].lower() == state,
                 "ACS geography file identity mismatch")
        _require(re.fullmatch(r"[0-9]{7}", row["LOGRECNO"]), "invalid geographic LOGRECNO")
        key = (state, row["LOGRECNO"])
        _require(key not in records, "duplicate geography join key")
        records[key] = row
    _require(bool(records), "empty ACS geography file")
    return records


def _sequence(text, state, number, kind, width):
    records = {}
    for row in csv.reader(io.StringIO(text)):
        _require(len(row) == width, "missing source table or wrong ACS sequence width")
        _require(row[:2] == ["ACSSF", f"2010{kind}5"] and row[2].lower() == state
                 and row[3:5] == ["000", number], "ACS sequence identity mismatch")
        _require(re.fullmatch(r"[0-9]{7}", row[5]), "invalid sequence LOGRECNO")
        key = (state, row[5])
        _require(key not in records, "duplicate ACS sequence join key")
        records[key] = row
    _require(bool(records), "empty required ACS sequence")
    return records


def _value(raw, table, kind):
    # These are Summary File tokens, not contemporary Census API sentinels.
    if raw == "":
        return None, "unavailable"
    if raw == ".":
        return None, "insufficient_sample_or_unavailable_median_moe"
    _require(re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", raw), f"unresolved ACS special value: {raw!r}")
    value = float(raw)
    _require(math.isfinite(value), "nonfinite ACS value")
    if kind == "M":
        # -1 is documented only for B00001/B00002/B98/B99, none approved here.
        _require(value >= 0, "invalid MOE for approved ACS table")
        return value, "controlled_estimate" if value == 0 else ""
    if table["unit"] == "persons" or table["unit"] == "households":
        _require(value >= 0, "unresolved negative estimate in approved ACS table")
        _require(value.is_integer(), "noninteger ACS count")
    if table.get("median_jam_values"):
        _require(value >= 0, "invalid negative median-income estimate")
    if table.get("median_jam_values") and raw in table["median_jam_values"]:
        return None, table["median_jam_values"][raw]
    return value, ""


def load_acs(bundle, mapping):
    """Build the entire supplied tract covariate frame before any outcome join."""
    section = _mapping(mapping, "acs")
    registry = _registry(section)
    _require(section["release"] == "2006-2010 ACS 5-year Summary File", "wrong ACS release")
    _require(section["geography_layout"] == {
        "FILEID": [1, 6], "STUSAB": [7, 2], "SUMLEVEL": [9, 3], "COMPONENT": [12, 2],
        "LOGRECNO": [14, 7], "STATE": [26, 2], "COUNTY": [28, 3],
        "TRACT": [41, 6], "GEOID": [179, 40]}, "unresolved geography layout")
    geography, sequences, source_ids = {}, {}, set()
    required_sequences = {table["sequence"] for table in section["tables"].values()}
    for source_file in bundle:
        name = Path(source_file.path).name
        _require(name not in source_ids, "duplicate ACS source file")
        source_ids.add(name)
        text = _read(source_file, section, "acs_2006_2010_5yr:" + name)
        geo_match = re.fullmatch(r"g20105([a-z]{2})\.txt", name)
        data_match = re.fullmatch(r"([em])20105([a-z]{2})([0-9]{4})000\.txt", name)
        if geo_match:
            state = geo_match[1]
            _require(state not in geography, "duplicate geography file")
            geography[state] = _geography(text, state, section["geography_layout"])
        elif data_match:
            kind, state, number = data_match.groups()
            _require(number in required_sequences, "unmapped source sequence")
            sequences[state, number, kind] = _sequence(
                text, state, number, kind, section["sequence_widths"][number])
        else:
            raise ValueError(f"unresolved historical ACS filename: {name}")
    _require(bool(geography), "missing source geography")
    expected = {(state, seq, kind) for state in geography for seq in required_sequences for kind in ("e", "m")}
    _require(set(sequences) == expected, "missing source tables or unmatched ACS states")
    tract_rows, excluded = {}, []
    for state, rows in geography.items():
        for key, row in rows.items():
            if row["SUMLEVEL"] != "140" or row["COMPONENT"] != "00":
                excluded.append((state, row["LOGRECNO"], row["SUMLEVEL"], row["COMPONENT"]))
                continue
            _require(row["STATE"] == section["state_fips"].get(state),
                     "tract state FIPS disagrees with source-file postal state")
            geoid = row["GEOID"]
            _require(re.fullmatch(r"14000US[0-9]{11}", geoid), "invalid 2010 tract GEOID")
            original_id = geoid[7:]
            _require(original_id == row["STATE"] + row["COUNTY"] + row["TRACT"],
                     "ACS tract geographic components disagree")
            tract_rows[key] = original_id
        tract_keys = {key for key in tract_rows if key[0] == state}
        for seq in required_sequences:
            estimates, moes = sequences[state, seq, "e"], sequences[state, seq, "m"]
            _require(set(estimates) == set(moes), "estimate/MOE join cardinality mismatch")
            _require(set(estimates) <= set(rows), "sequence record missing geography")
            _require(tract_keys <= set(estimates), "missing source table rows for tract frame")
    frame = pd.DataFrame({"original_id": list(tract_rows.values())})
    _ids(frame, "ACS geography")
    _require(not frame.empty, "no 2010 census tracts in ACS bundle")
    estimates_by_tract, cells = {}, []
    for key, original_id in tract_rows.items():
        state = key[0]
        estimates_by_tract[original_id] = {}
        for table_id, table in section["tables"].items():
            seq = table["sequence"]
            for line in table["lines"]:
                offset = table["start_position"] + int(line) - 2
                variable = f"{table_id}_{int(line):03}"
                raw_e, raw_m = (sequences[state, seq, kind][key][offset] for kind in ("e", "m"))
                estimate, ea = _value(raw_e, table, "E")
                moe, ma = _value(raw_m, table, "M")
                estimates_by_tract[original_id][variable] = estimate
                cells.append((original_id, variable, raw_e, raw_m, estimate, moe, ea, ma))
    missing = []
    for name, concept in section["concepts"].items():
        values = []
        for original_id in frame.original_id:
            source = estimates_by_tract[original_id]
            numerator = [source[f"{concept['table']}_{line:03}"] for line in concept["num"]]
            denominator = source[f"{concept['table']}_{concept['den']:03}"] if "den" in concept else None
            reason = ""
            if any(x is None for x in numerator):
                value, reason = None, "unavailable_numerator"
            elif concept["transform"] == "share":
                if denominator is None or denominator == 0:
                    value, reason = None, "unavailable_or_zero_denominator"
                else:
                    value = sum(numerator) / denominator
                    _require(0 <= value <= 1, f"invalid denominator/share for {name}")
            else:
                _require(concept["transform"] == "income" and len(numerator) == 1,
                         f"unresolved transform: {name}")
                value = math.asinh(numerator[0] / 10000)
            values.append(value)
            if reason:
                missing.append((original_id, name, reason))
        frame[name] = pd.Series(values, dtype="float64")
    return frame, {
        "registry": registry, "mapping_hash": mapping_hash(section),
        "source_manifests": tuple(f.manifest for f in bundle),
        "masking_families": section["masking_families"],
        "cells": pd.DataFrame(cells, columns=["original_id", "variable", "raw_estimate", "raw_moe",
                                              "estimate", "moe", "estimate_annotation", "moe_annotation"]),
        "missing_concepts": tuple(missing), "excluded_geographies": tuple(excluded),
        "geography": pd.DataFrame({"original_id": frame.original_id,
                                    "county_fips": frame.original_id.str[:5]}),
        "join_cardinality": "one_to_one", "geography_vintage": 2010,
    }
