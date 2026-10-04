"""Local CONAPO municipal denominators; no acquisition or mortality estimation.

The inspected datos.gob.mx resource is a wide, midyear population table, with
18 five-year age groups (85+ is open). ``fecha`` is a display date, NOT a
release date. A reviewed SourceManifest pins the actual file revision by hash.
See configs/adapters/mexico.yaml for dictionary URLs and inspection evidence.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

from oxyformer.contracts import SourceManifest
from oxyformer.provenance import ContractError, file_hash


AGE_GROUPS = tuple(f"{n:02d}-{n + 4:02d}" for n in range(0, 85, 5)) + ("85+",)
AGE_COLUMNS = (
    "POB_00_04", "POB_05_09", "POB_010_014", "POB_015_019",
    *(f"POB_{n}_{n + 4}" for n in range(20, 85, 5)), "POB_85_MM",
)
SEXES = ("male", "female")
CONAPO_MAPPING = (
    ("CLAVE", "municipality"), ("CLAVE_ENT", "state"),
    ("SEXO", "sex"), ("ANO", "year"), ("POB_TOTAL", "population_total"),
    *zip(AGE_COLUMNS, AGE_GROUPS),
)


class AdapterError(ContractError):
    """A refused join/load, with machine-readable evidence in ``audit``."""

    def __init__(self, message: str, **audit):
        super().__init__(message)
        self.audit = audit


def _check(condition, message, **audit):
    if not condition:
        raise AdapterError(message, **audit)


def _code(value: str, width: int) -> str:
    text = str(value).strip()
    _check(bool(re.fullmatch(r"[0-9]{1," + str(width) + r"}", text)),
           "invalid geography code", value=value)
    return text.zfill(width)


def _integer(value: str, name: str) -> int:
    _check(bool(re.fullmatch(r"[0-9]+", str(value).strip())),
           f"invalid {name}", value=value)
    return int(value)


def _years(values) -> tuple[int, ...]:
    result = tuple(values)
    _check(bool(result) and len(set(result)) == len(result)
           and all(type(y) is int and 2015 <= y <= 2019 for y in result),
           "occurrence years must be a nonempty subset of approved 2015-2019")
    return tuple(sorted(result))


def _read_rows(path, manifest, mapping, encoding):
    """Read only local CSV; verify bytes and reviewed required field meanings."""
    manifest.assert_usable()
    _check(file_hash(Path(path)) == manifest.payload_hash,
           "source payload hash mismatch: release revision changed",
           version=manifest.version)
    approved = {k.upper(): v for k, v in manifest.field_mapping}
    _check(len(approved) == len(manifest.field_mapping)
           and all(approved.get(k) == v for k, v in mapping),
           "reviewed source mapping does not match dictionary")
    with Path(path).open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        upper = [name.upper() for name in names]
        _check(len(set(upper)) == len(upper), "duplicate CSV columns")
        _check(set(k for k, _ in mapping) <= set(upper),
               "missing source columns or incompatible age bins",
               missing_columns=sorted(set(k for k, _ in mapping) - set(upper)))
        for number, row in enumerate(reader, 1):
            _check(None not in row and all(v is not None for v in row.values()),
                   "malformed CSV row", row=number)
            yield number, {k.upper(): v.strip() for k, v in row.items()}, tuple(row.items())


@dataclass(frozen=True, slots=True)
class Denominator:
    municipality: str
    year: int
    age_group: str
    sex: str
    population: int
    source_state: str
    source_municipality: str


@dataclass(frozen=True, slots=True)
class DenominatorTable:
    cells: tuple[Denominator, ...]
    years: tuple[int, ...]
    municipalities: tuple[str, ...]
    geography_vintage: str
    manifest: SourceManifest
    audit: dict


def load_denominators(path, manifest: SourceManifest, *, municipalities,
                      geography_vintage: str, years=range(2015, 2020),
                      encoding="utf-8-sig") -> DenominatorTable:
    """Load a frozen target's complete municipality × year × sex × age grid.

``municipalities`` is an explicit source-code universe; never inferred from
available rows. Missing rows, repeated rows/revisions, totals that disagree
with age cells, and nonpositive selected denominators refuse with audits.
Unselected rows are counted, not treated as part of the target. No network.
"""
    years = _years(years)
    municipalities = tuple(_code(m, 5) for m in municipalities)
    _check(bool(municipalities) and len(set(municipalities)) == len(municipalities),
           "municipalities must be a nonempty unique source universe")
    _check(bool(geography_vintage.strip()), "geographic vintage is required")
    expected = set(product(municipalities, years, SEXES))
    selected = set()
    cells = []
    invalid_population = []
    input_rows = outside = 0
    for number, row, raw in _read_rows(path, manifest, CONAPO_MAPPING, encoding):
        input_rows += 1
        municipality = _code(row["CLAVE"], 5)
        year = _integer(row["ANO"], "population year")
        if municipality not in municipalities or year not in years:
            outside += 1
            continue
        state = _code(row["CLAVE_ENT"], 2)
        _check(1 <= int(state) <= 32 and municipality[:2] == state
               and 1 <= int(municipality[2:]) <= 998,
               "inconsistent CONAPO municipality/state", row=number)
        _check(row["SEXO"] in ("HOMBRES", "MUJERES"),
               "incompatible denominator sex category", row=number, sex=row["SEXO"])
        sex = {"HOMBRES": "male", "MUJERES": "female"}[row["SEXO"]]
        key = (municipality, year, sex)
        _check(key not in selected, "duplicate denominator row or mixed revisions", key=key)
        selected.add(key)
        population = [_integer(row[col], col) for col in AGE_COLUMNS]
        total = _integer(row["POB_TOTAL"], "population total")
        _check(sum(population) == total, "population age counts do not conserve total", key=key)
        for age, n in zip(AGE_GROUPS, population):
            if n <= 0:
                invalid_population.append((municipality, year, age, sex, n))
            cells.append(Denominator(municipality, year, age, sex, n,
                                     row["CLAVE_ENT"], row["CLAVE"]))
    audit = dict(input_rows=input_rows, outside_target_rows=outside,
                 selected_rows=len(selected), missing_rows=sorted(expected - selected),
                 nonpositive_denominators=invalid_population,
                 release_version=manifest.version, source_hash=manifest.payload_hash,
                 manifest_hash=manifest.content_hash, geography_vintage=geography_vintage)
    _check(not audit["missing_rows"], "missing denominator coverage", **audit)
    _check(not invalid_population, "denominators must be positive", **audit)
    audit["population_sum"] = sum(c.population for c in cells)
    return DenominatorTable(tuple(cells), years, tuple(sorted(municipalities)),
                            geography_vintage, manifest, audit)
