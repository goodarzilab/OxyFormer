"""Owner-approved, outcome-free elevation proxy; pressures are in mmHg.

US Standard Atmosphere 1976, part 1, tables 2/4 and hydrostatic equation:
https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf
The standard uses geopotential height. approvals.yaml explicitly prescribes DEM
metres above MSL directly as z, so this is the approved near-surface approximation,
not a full geometric-to-geopotential atmosphere or measured weather. NAVD88 DEM
heights approximate MSL. The negative-height extension is limited to -500 m;
above 11 km the tropospheric lapse law is not used. Inspired oxygen is humidified
at the approved fixed 47 mmHg water pressure, not alveolar oxygen.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import numpy as np
from oxyformer.provenance import canonical_json, require


@dataclass(frozen=True)
class PhysicalSpec:
    version: str = 'owner-exposure-2026-10-04-v1'
    sea_level_pressure_mmhg: float = 760.0
    sea_level_temperature_k: float = 288.15
    lapse_rate_k_per_m: float = 0.0065
    gravity_m_per_s2: float = 9.80665
    molar_mass_air_kg_per_mol: float = 0.0289644
    gas_constant_j_per_mol_k: float = 8.31432
    inspired_o2_fraction: float = 0.2093
    water_vapour_pressure_mmhg: float = 47.0
    reference_elevation_m: float = 0.0
    minimum_elevation_m: float = -500.0
    maximum_elevation_m: float = 11000.0
    elevation_convention: str = 'DEM orthometric metres used directly as z'
    # Standard atmosphere defines 101325 Pa = 760 mmHg for this proxy.
    pa_per_mmhg: float = 101325.0 / 760.0

    @property
    def content_hash(self):
        return sha256(canonical_json(asdict(self)).encode()).hexdigest()


PHYSICS = PhysicalSpec()


def pressure_mmhg(elevation_m):
    """P0*(1-L*z/T0)**(g*M/(R*L)); no tensors or gradients accepted."""
    require(not hasattr(elevation_m, 'detach'), 'model tensors cannot enter exposure')
    z = np.asarray(elevation_m, dtype=np.float64)
    p = PHYSICS
    require(np.all(np.isfinite(z)) and np.all((z >= p.minimum_elevation_m) &
            (z <= p.maximum_elevation_m)), 'elevation outside physical validity domain')
    return p.sea_level_pressure_mmhg * (1 - p.lapse_rate_k_per_m * z /
        p.sea_level_temperature_k) ** (p.gravity_m_per_s2 * p.molar_mass_air_kg_per_mol /
                                      (p.gas_constant_j_per_mol_k * p.lapse_rate_k_per_m))


def inspired_oxygen_mmhg(elevation_m):
    return PHYSICS.inspired_o2_fraction * (pressure_mmhg(elevation_m) -
                                          PHYSICS.water_vapour_pressure_mmhg)


def oxygen_deficit_mmhg(elevation_m):
    return inspired_oxygen_mmhg(PHYSICS.reference_elevation_m) - inspired_oxygen_mmhg(elevation_m)
