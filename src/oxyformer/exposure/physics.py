"""Owner-approved standard-atmosphere exposure, never a weather prediction.

Reference inspected: US Standard Atmosphere 1976, NASA-TM-X-74335, tables 1/2
and the lower-atmosphere hydrostatic equation:
https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf
The standard uses geopotential height. The owner-approved formula explicitly
uses DEM metres above mean sea level directly; we retain that approximation,
not an unapproved geopotential conversion. NAVD88 is the approved CONUS DEM
vertical reference, not ellipsoidal GPS height. Domain: -500 to 11000 m, a
conservative subset of the standard's lower-atmosphere table on printed p. 52 (-1000 m upward).

Inspired (humidified at 37 C) oxygen is FiO2*(PB-47 mmHg), not alveolar oxygen:
https://pubmed.ncbi.nlm.nih.gov/26735235/ . No Pa/mmHg conversions are implicit;
the hydrostatic exponent uses SI constants and its pressure ratio is unitless.
"""
from dataclasses import asdict, dataclass

import numpy as np

from oxyformer.provenance import require


@dataclass(frozen=True)
class PhysicalSpec:
    version: str = 'owner-2026-10-04-isa-v1'
    dem_product: str = 'usgs_3dep_one_third_arc_second_seamless'
    pressure_model: str = 'international_standard_atmosphere_troposphere'
    sea_level_pressure_mmhg: float = 760.0
    sea_level_temperature_k: float = 288.15
    lapse_rate_k_per_m: float = 0.0065
    gravity_m_per_s2: float = 9.80665
    molar_mass_air_kg_per_mol: float = 0.0289644
    gas_constant_j_per_mol_k: float = 8.31432
    inspired_o2_fraction: float = 0.2093
    water_vapour_pressure_mmhg: float = 47.0
    minimum_elevation_m: float = -500.0
    maximum_elevation_m: float = 11000.0
    reference_elevation_m: float = 0.0
    elevation_convention: str = 'NAVD88_metres_used_directly_as_height'

    def validate(self):
        require(asdict(self) == asdict(PHYSICS), 'physical specification differs from approved v1')


PHYSICS = PhysicalSpec()


def _elevations(elevation_m):
    # Do not detach tensors or accept objects whose conversion could hide gradients.
    require(type(elevation_m) in (int, float, list, tuple, np.ndarray) or
            isinstance(elevation_m, (np.integer, np.floating)),
            'elevation must be plain CPU numbers, not model tensors')
    values = np.asarray(elevation_m)
    require(values.dtype.kind in 'iuf', 'elevation must be numeric')
    values = values.astype(np.float64)
    require(np.isfinite(values).all(), 'nonfinite elevation')
    require(((values >= PHYSICS.minimum_elevation_m) &
             (values <= PHYSICS.maximum_elevation_m)).all(), 'elevation outside physical validity domain')
    return values


def pressure_mmhg(elevation_m):
    """Barometric pressure in mmHg for ground elevation in metres."""
    z = _elevations(elevation_m)
    p = PHYSICS
    exponent = p.gravity_m_per_s2 * p.molar_mass_air_kg_per_mol / (
        p.gas_constant_j_per_mol_k * p.lapse_rate_k_per_m)
    return p.sea_level_pressure_mmhg * (1 - p.lapse_rate_k_per_m * z /
                                       p.sea_level_temperature_k) ** exponent


def inspired_oxygen_mmhg(elevation_m):
    return PHYSICS.inspired_o2_fraction * (pressure_mmhg(elevation_m) -
                                          PHYSICS.water_vapour_pressure_mmhg)


def deficit_mmhg(elevation_m):
    """Sea-level inspired oxygen minus local inspired oxygen, in mmHg.

    Negative deficits below sea level are preserved rather than clipped.
    """
    reference = PHYSICS.inspired_o2_fraction * (PHYSICS.sea_level_pressure_mmhg -
                                               PHYSICS.water_vapour_pressure_mmhg)
    return reference - inspired_oxygen_mmhg(elevation_m)
