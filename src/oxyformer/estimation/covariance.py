"""Aligned multivariate cluster, geographic-process and survey covariance.

Inputs are normalized contributions u_i, already averaged across seeds by
original observation, as stored in Estimate. Never divide them by n again.
A literal fixed frame of places has no sampling variance by itself: these
geographic-process estimates require a stochastic-process interpretation.
Fixed-frame descriptive uncertainty needs a separate contract. No upstream
measurement SE, denominator error or exposure error is automatically added.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from oxyformer.contracts import Estimate
from oxyformer.provenance import require

EARTH_RADIUS_KM = 6371.0088
SPATIAL_BANDWIDTHS_KM = (50.0, 100.0, 200.0)


@dataclass(frozen=True)
class AlignedInfluence:
    original_ids: tuple[str, ...]
    endpoints: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        ids, endpoints = tuple(self.original_ids), tuple(self.endpoints)
        require(bool(ids) and len(set(ids)) == len(ids), "unique original observation IDs required")
        require(bool(endpoints) and len(set(endpoints)) == len(endpoints), "unique endpoint labels required")
        matrix = np.asarray(self.values, dtype=np.float64)
        require(matrix.shape == (len(ids), len(endpoints)), "influence alignment mismatch")
        require(bool(np.isfinite(matrix).all()), "nonfinite influence")
        object.__setattr__(self, "original_ids", ids)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "values", tuple(tuple(row) for row in matrix.tolist()))


def align_estimates(estimates: Mapping[str, Estimate]) -> AlignedInfluence:
    """Align full endpoint vectors by ID; never silently intersect populations.

    Each Estimate is already averaged over seeds. Different endpoints may have
    different units/targets, but a joint covariance requires an explicitly common
    set of original units. Adapters for different populations must declare a
    union/zero-contribution contract separately rather than inventing one here.
    """
    require(bool(estimates), "estimates required")
    ids = next(iter(estimates.values())).original_ids
    columns = []
    for estimate in estimates.values():
        require(set(estimate.original_ids) == set(ids), "endpoint original IDs mismatch")
        lookup = dict(zip(estimate.original_ids, estimate.influence))
        columns.append([lookup[oid] for oid in ids])
    return AlignedInfluence(ids, tuple(estimates), tuple(map(tuple, np.asarray(columns).T)))


@dataclass(frozen=True)
class Covariance:
    endpoints: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    method: str
    interpretation: str
    dependence_units: int
    details: tuple[tuple[str, str], ...]

    @property
    def standard_errors(self):
        # Only roundoff at zero may be negative for a PSD matrix product.
        return tuple(np.sqrt(np.maximum(np.diag(self.matrix), 0.0)).tolist())


def _result(influence, matrix, method, interpretation, units, **details):
    matrix = (matrix + matrix.T) / 2
    require(bool(np.isfinite(matrix).all()), "nonfinite covariance")
    return Covariance(influence.endpoints, tuple(map(tuple, matrix.tolist())), method,
                      interpretation, units, tuple((key, str(value)) for key, value in details.items()))


def _keys(mapping, ids, name):
    require(set(mapping) == set(ids), f"{name} observation IDs mismatch")
    require(all(isinstance(mapping[oid], str) and mapping[oid] for oid in ids), f"invalid {name} labels")


def _groups(influence, groups):
    _keys(groups, influence.original_ids, "cluster")
    labels = tuple(dict.fromkeys(groups[oid] for oid in influence.original_ids))
    require(len(labels) >= 2, "at least two independent clusters required")
    positions = {label: i for i, label in enumerate(labels)}
    totals = np.zeros((len(labels), len(influence.endpoints)), dtype=np.float64)
    for oid, row in zip(influence.original_ids, influence.values):
        totals[positions[groups[oid]]] += row
    return labels, totals


def cluster_covariance(influence: AlignedInfluence, groups: Mapping[str, str], *, interpretation: str) -> Covariance:
    """G/(G-1) sum_g U_g U_g^T; groups are explicit dependence units.

    The small-sample factor uses clusters, never observation or seed count.
    This formula assumes independent clusters under the stated interpretation.
    It is not a universal spatial-field theorem or a fixed-census sampling SE.
    """
    require(interpretation in ("geographic_process", "independent_cluster_sampling"),
            "fixed-frame or unspecified uncertainty requires a separate contract")
    labels, totals = _groups(influence, groups)
    factor = len(labels) / (len(labels) - 1)
    return _result(influence, factor * (totals.T @ totals), "cluster", interpretation,
                   len(labels), finite_cluster_factor=factor)


def earth_centered_coordinates(locations) -> np.ndarray:
    """(latitude, longitude) degrees -> spherical Earth-centered xyz in km.

    Radius is 6371.0088 km. Distances are Euclidean chords of this sphere, not
    ellipsoidal ECEF or geodesic arc lengths. Antimeridian/pole pairs therefore
    have no longitude discontinuity. No claim of identical geodesic bandwidth.
    """
    points = np.asarray(locations, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) > 0,
            "locations must be latitude/longitude pairs")
    require(bool(np.isfinite(points).all()), "nonfinite locations")
    require(bool((np.abs(points[:, 0]) <= 90).all() & (np.abs(points[:, 1]) <= 180).all()),
            "latitude/longitude outside degree bounds")
    lat, lon = np.deg2rad(points).T
    return EARTH_RADIUS_KM * np.column_stack((np.cos(lat) * np.cos(lon),
                                             np.cos(lat) * np.sin(lon), np.sin(lat)))


def spatial_kernel(locations, bandwidth: float) -> np.ndarray:
    """PSD Gaussian K=exp(-||xyz_g-xyz_h||^2/(2*b^2)), b in chord km.

    The Gaussian on Euclidean R^3 has a nonnegative Fourier transform, hence
    its restriction to any set of sphere coordinates is positive semidefinite.
    b is a scale (K=exp(-1/2) at distance b), not a hard cutoff. No eigenvalue
    clipping or distance-kernel substitution is used to obtain PSD.
    """
    require(np.isfinite(bandwidth) and bandwidth > 0, "positive finite bandwidth required")
    xyz = earth_centered_coordinates(locations)
    difference = (xyz[:, None, :] - xyz[None, :, :]) / bandwidth
    with np.errstate(over="ignore"):
        distance_squared = np.sum(difference**2, axis=2)
    return np.exp(-0.5 * distance_squared)


def spatial_covariance(influence: AlignedInfluence, groups: Mapping[str, str],
                       locations: Mapping[str, tuple[float, float]], bandwidth: float, *,
                       interpretation: str, finite_cluster_correction=False) -> Covariance:
    """Sum K_b(g,h) U_g U_h^T on declared cluster locations (lat/lon).

    Aggregate within counties first; the kernel adds cross-county dependence.
    The frozen spatial formula has no G/(G-1). The optional correction multiplies
    the whole PSD matrix, is explicit, and is recorded for sensitivity analyses.
    Locations must be predeclared group representatives, not outcome-selected.
    """
    require(interpretation == "geographic_process", "spatial covariance requires geographic_process interpretation")
    labels, totals = _groups(influence, groups)
    require(set(locations) == set(labels), "cluster location IDs mismatch")
    kernel = spatial_kernel([locations[label] for label in labels], bandwidth)
    factor = len(labels) / (len(labels) - 1) if finite_cluster_correction else 1.0
    return _result(influence, factor * (totals.T @ kernel @ totals), "spatial_gaussian_chord",
                   interpretation, len(labels), bandwidth_km=bandwidth,
                   distance="spherical Earth-centered Euclidean chord km", radius_km=EARTH_RADIUS_KM,
                   finite_cluster_factor=factor)


def spatial_sensitivities(influence, groups, locations, *, interpretation):
    """Prespecified 50/100/200-km Gaussian chord scales, without selecting a winner."""
    return {b: spatial_covariance(influence, groups, locations, b, interpretation=interpretation)
            for b in SPATIAL_BANDWIDTHS_KM}


def survey_covariance(influence: AlignedInfluence, strata: Mapping[str, str], psu: Mapping[str, str], *,
                      singleton="raise", finite_population_fraction: Mapping[str, float] | None = None) -> Covariance:
    """Stratum-centered PSU Taylor covariance, with optional stratum FPC (1-f_h).

    PSU IDs are scoped within strata. Weights are already in u_i; never reweight
    here. The default with-replacement approximation has f_h=0. Supplied f_h
    must be known PSU sampling fractions for every stratum under a design that
    supports this correction; they are not inferred from observation counts.
    This is not a replicate-weight variance estimator.

    A singleton raises by default. ``singleton='certainty'`` explicitly asserts
    every singleton is a certainty PSU with zero first-stage variance; it cannot
    estimate unprovided lower-stage uncertainty. If FPCs are supplied its f must
    be 1. Noncertainty singletons require an upstream, documented design choice
    such as collapsing strata, never automatic zeroing or guessed replication.
    """
    require(singleton in ("raise", "certainty"), "unknown singleton-stratum policy")
    ids = influence.original_ids
    _keys(strata, ids, "stratum")
    _keys(psu, ids, "PSU")
    totals = {}
    for oid, row in zip(ids, influence.values):
        key = (strata[oid], psu[oid])
        totals.setdefault(key, np.zeros(len(influence.endpoints), dtype=np.float64))[:] += row
    require(len(totals) >= 2, "at least two PSUs required")
    labels = tuple(dict.fromkeys(strata[oid] for oid in ids))
    if finite_population_fraction is None:
        fractions = dict.fromkeys(labels, 0.0)
    else:
        require(set(finite_population_fraction) == set(labels), "FPC stratum IDs mismatch")
        fractions = dict(finite_population_fraction)
        require(all(np.isfinite(f) and 0 <= f <= 1 for f in fractions.values()), "invalid PSU sampling fraction")
    covariance = np.zeros((len(influence.endpoints), len(influence.endpoints)), dtype=np.float64)
    singleton_labels = []
    for label in labels:
        u = np.asarray([value for (h, _), value in totals.items() if h == label])
        m = len(u)
        if m == 1:
            require(singleton == "certainty", f"singleton stratum: {label}")
            require(finite_population_fraction is None or fractions[label] == 1,
                    "certainty singleton requires FPC sampling fraction 1")
            singleton_labels.append(label)
            continue
        centered = u - u.mean(axis=0)
        covariance += (1 - fractions[label]) * m / (m - 1) * (centered.T @ centered)
    return _result(influence, covariance, "stratified_psu", "survey_design", len(totals),
                   singleton=singleton, certainty_strata=tuple(singleton_labels),
                   finite_population_fraction=tuple(fractions.items()))
