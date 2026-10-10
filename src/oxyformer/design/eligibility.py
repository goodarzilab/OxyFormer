"""Outcome-free, reviewed inputs and preliminary geographic eligibility.

A validated-data producer publishes a DataManifest, approved CovariateView,
EntityGraph and GeographyTable. An atlas producer publishes CollectedAtlas.
These explicit stage inputs contain no Y values. Flag and label-availability
metadata are supplied by the validator; the gate never derives them from Y.
"""
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

import numpy as np

from oxyformer.data.adapters.usaleep import primary_outcome_flags
from oxyformer.provenance import Immutable, check_hash, nonempty, require, unique


@dataclass(frozen=True, slots=True, kw_only=True)
class GeographyRow(Immutable):
    original_id: str
    tract_id: str
    county: str
    state: str
    subblock: str
    assignment_geography: str
    latitude: float
    longitude: float
    outcome_flag: int
    label_available: bool

    def __post_init__(self):
        Immutable.__post_init__(self)
        for value in (self.original_id, self.tract_id, self.county, self.state,
                      self.subblock, self.assignment_geography):
            nonempty(value, "geography identity")
        require(-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180,
                "invalid geographic coordinates")
        require(self.outcome_flag in (1, 2, 3), "unknown outcome flag")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeographyTable(Immutable):
    rows: tuple[GeographyRow, ...]
    data_manifest_hash: str
    county_field: str
    approval_reference: str
    mapping_review_id: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.rows), "empty geography")
        unique(tuple(r.original_id for r in self.rows), "geography observations")
        check_hash(self.data_manifest_hash)
        for text in (self.county_field, self.approval_reference, self.mapping_review_id):
            nonempty(text, "reviewed geography binding")
        for field in ("tract_id", "assignment_geography"):
            seen = {}
            for row in self.rows:
                # An assignment area can contain several tract centroids and
                # subblocks. Only repetitions of the SAME tract must retain
                # identical geometry; both group types share county/state.
                value = (row.county, row.state)
                if field == "tract_id":
                    value += (row.subblock, row.latitude, row.longitude)
                require(seen.setdefault(getattr(row, field), value) == value,
                        f"inconsistent repeated {field}")


@dataclass(frozen=True, slots=True, kw_only=True)
class AtlasRow(Immutable):
    tract_id: str
    exposure_mmhg: float
    inhabited_elevation_m: float
    population: float
    allocation_qualified: bool

    def __post_init__(self):
        Immutable.__post_init__(self)
        nonempty(self.tract_id, "atlas tract")
        require(self.population >= 0, "negative population")


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectedAtlas(Immutable):
    rows: tuple[AtlasRow, ...]
    source_hashes: tuple[str, ...]
    footprint: str
    expected_tract_ids: tuple[str, ...]
    missing_tract_ids: tuple[str, ...]
    coverage_complete: bool
    mapping_review_id: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        ids = tuple(r.tract_id for r in self.rows)
        for values in (ids, self.expected_tract_ids, self.missing_tract_ids):
            unique(values, "atlas tracts")
        require(bool(self.expected_tract_ids), "atlas coverage frame required")
        require(set(ids).isdisjoint(self.missing_tract_ids), "contradictory atlas coverage")
        require(set(ids) | set(self.missing_tract_ids) == set(self.expected_tract_ids),
                "atlas coverage accounting mismatch")
        require(self.coverage_complete == (not self.missing_tract_ids), "inconsistent atlas completeness")
        require(bool(self.source_hashes), "atlas sources required")
        for digest in self.source_hashes:
            check_hash(digest)
        nonempty(self.footprint, "approved atlas footprint")
        nonempty(self.mapping_review_id, "atlas review")


def distance_km(a, b):
    lat1, lat2 = radians(a.latitude), radians(b.latitude)
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin(
        radians(b.longitude - a.longitude) / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(min(1.0, max(0.0, h))))


def usable(row, atlas):
    return _usable(row, atlas, primary_outcome_flags())


def _usable(row, atlas, flags):
    item = atlas.get(row.tract_id)
    return (row.outcome_flag in flags and row.label_available and item is not None
            and item.allocation_qualified and item.population > 0)


def county_screen(rows, atlas):
    """Fixed four-tract / 300 m / 25 km screens on allocation-qualified inputs."""
    flags = primary_outcome_flags()
    counties = {}
    for row in rows:
        counties.setdefault(row.county, {})
        if _usable(row, atlas, flags):
            counties[row.county][row.tract_id] = row
    result = {}
    for county, tracts in sorted(counties.items()):
        values = tuple(tracts.values())
        heights = [atlas[r.tract_id].inhabited_elevation_m for r in values]
        spread = float(np.quantile(heights, .9) - np.quantile(heights, .1)) if heights else 0.0
        local = any(atlas[a.tract_id].exposure_mmhg != atlas[b.tract_id].exposure_mmhg
                    and distance_km(a, b) <= 25.0
                    for i, a in enumerate(values) for b in values[i + 1:])
        reasons = []
        if len(values) < 4:
            reasons.append("fewer_than_four_primary_eligible_tracts")
        if spread < 300:
            reasons.append("inhabited_relief_below_300m")
        if not local:
            reasons.append("no_local_comparison_within_25km")
        result[county] = {"tract_count": len(values), "p90_p10_m": spread,
                          "local_comparison": local, "reasons": reasons}
    return result
