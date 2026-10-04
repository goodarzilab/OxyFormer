"""Fixed, outcome-free population placement scenarios on 2010 Census blocks.

A distributed scenario is uniform area placement on a globally anchored grid
in an explicitly selected equal-area CRS. Each clipped cell is represented by
an interior point. This is an assumption, not observed habitation. Splitting a
block along grid boundaries preserves its placements and exposure when the
split populations retain the same area density.

The centroid scenario uses the geometric centroid when covered by the polygon,
otherwise a geometry-only interior representative point. This policy is explicit
in AllocationSpec and in every output manifest; it never consults DEM coverage.
"""
from dataclasses import dataclass
import math
import re

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import box
from shapely.strtree import STRtree

from oxyformer.provenance import require


@dataclass(frozen=True)
class AllocationSpec:
    scenario: str
    area_crs: str
    grid_size_m: float | None = None
    version: str = 'fixed-area-placement-v1'
    centroid_outside_policy: str = 'interior_representative_point'

    def validate(self):
        require(self.version == 'fixed-area-placement-v1', 'unknown allocation version')
        require(self.centroid_outside_policy == 'interior_representative_point',
                'unknown centroid outside-polygon policy')
        require(self.scenario in ('centroid', 'distributed'), 'unknown placement scenario')
        crs = CRS.from_user_input(self.area_crs)
        require(crs.is_projected and all(a.unit_name == 'metre' for a in crs.axis_info),
                'placement CRS must be projected in metres')
        operation = crs.coordinate_operation
        require(operation is not None and 'equal area' in operation.method_name.lower().replace('-', ' '),
                'placement CRS must use an equal-area projection')
        if self.scenario == 'distributed':
            require(type(self.grid_size_m) in (int, float) and math.isfinite(self.grid_size_m)
                    and self.grid_size_m > 0, 'distributed scenario needs a positive fixed grid size')
        else:
            require(self.grid_size_m is None, 'centroid scenario does not use a grid')


def validate_blocks(blocks, *, check_topology=True):
    require(isinstance(blocks, gpd.GeoDataFrame), 'geography must contain a GeoDataFrame')
    require(set(blocks.columns) == {'block_id', 'tract_id', 'population', 'geometry'},
            'exposure boundary permits only block_id, tract_id, population and geometry')
    require(blocks.crs is not None, 'block CRS missing')
    require(len(blocks) > 0, 'empty block geography')
    require(not blocks.block_id.duplicated().any(), 'duplicate 2010 block IDs')
    for row in blocks.itertuples():
        require(isinstance(row.block_id, str) and re.fullmatch(r'[0-9]{15}', row.block_id) is not None,
                'block ID must be a 15-digit 2010 string')
        require(isinstance(row.tract_id, str) and row.tract_id == row.block_id[:11],
                '2010 tract membership does not match block ID')
        require(isinstance(row.population, (int, np.integer)) and not isinstance(row.population, bool)
                and row.population >= 0, 'population must be a nonnegative Census integer')
        geom = row.geometry
        require(geom is not None and not geom.is_empty and geom.is_valid
                and geom.geom_type in ('Polygon', 'MultiPolygon'), 'invalid block geometry')

    if check_topology:
        validate_disjoint(blocks.geometry.to_numpy(), names=blocks.block_id.tolist())


def validate_disjoint(geometries, *, names=None, label='block'):
    """Reject positive-area interiors; shared boundaries are permitted."""
    names = list(range(len(geometries))) if names is None else names
    tree = STRtree(geometries)
    for i, geometry in enumerate(geometries):
        for j in tree.query(geometry):
            if j > i:
                overlap_area = geometry.intersection(geometries[j]).area
                require(overlap_area == 0,
                        f'overlapping {label} polygon interiors: {names[i]} / {names[j]}, area={overlap_area}')


def placements(geometry, spec):
    """Yield (x, y, population fraction) in spec.area_crs; no DEM-dependent weights."""
    if spec.scenario == 'centroid':
        point = geometry.centroid
        if not geometry.covers(point):
            point = geometry.representative_point()
        yield point.x, point.y, 1.0
        return
    size = spec.grid_size_m
    left, bottom, right, top = geometry.bounds
    pieces = []
    for ix in range(math.floor(left / size), math.ceil(right / size)):
        for iy in range(math.floor(bottom / size), math.ceil(top / size)):
            cut = geometry.intersection(box(ix * size, iy * size, (ix + 1) * size, (iy + 1) * size))
            if not cut.is_empty and cut.area > 0:
                point = cut.representative_point()
                pieces.append((point.x, point.y, cut.area))
    total = math.fsum(area for _, _, area in pieces)
    require(total > 0 and math.isclose(total, geometry.area, rel_tol=1e-9, abs_tol=1e-6),
            'allocation geometry area not conserved')
    for x, y, area in pieces:
        yield x, y, area / total
