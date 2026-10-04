"""Fixed area-based placement and bounded raster sampling, without outcome inputs."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from shapely.geometry import box
from oxyformer.exposure.physics import PHYSICS
from oxyformer.provenance import canonical_json, check_hash, file_hash, require

PRIMARY = 'usgs_3dep_one_third_arc_second_seamless'
FALLBACK = 'usgs_3dep_one_arc_second_seamless'
FALLBACK_CELLS = frozenset(('n43w070', 'n40w074', 'n41w072', 'n46w083',
                          'n48w086', 'n49w088', 'n27w080', 'n29w091'))


@dataclass(frozen=True)
class AllocationSpec:
    scenarios: tuple[str, ...] = ('centroid', 'distributed')
    placement_crs: str = 'EPSG:5070'
    grid_size_m: float = 100.0
    grid_origin_m: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, 'scenarios', tuple(self.scenarios))
        object.__setattr__(self, 'grid_origin_m', tuple(self.grid_origin_m))
        require(bool(self.scenarios) and len(set(self.scenarios)) == len(self.scenarios) and
                set(self.scenarios) <= {'centroid', 'distributed'}, 'invalid placement scenarios')
        require(CRS(self.placement_crs) == CRS('EPSG:5070'), 'placement CRS must be CONUS equal-area EPSG:5070')
        require(math.isfinite(self.grid_size_m) and self.grid_size_m > 0 and
                len(self.grid_origin_m) == 2 and all(math.isfinite(x) for x in self.grid_origin_m),
                'invalid placement grid')

    @property
    def content_hash(self):
        return sha256(canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class DemTile:
    resource_id: str
    path: str
    sha256: str
    crs: str
    nodata: float | None
    vertical_unit: str
    vertical_datum: str
    product: str = PRIMARY
    fallback_reason: str | None = None
    archive_member: str | None = None

    def identity(self):
        return {k: v for k, v in asdict(self).items() if k != 'path'}


class RasterSampler:
    """Open verified local rasters once; read only sampled windows through GDAL.

    Reviewed tile metadata must declare CRS, nodata and vertical units/datum even
    when GeoTIFF carries no vertical-unit tag. Embedded tags must agree if present.
    Overlapping border pixels use first valid tile in sorted resource-ID order.
    """
    def __init__(self, tiles, placement_crs):
        self.tiles = sorted(tiles, key=lambda tile: tile.resource_id)
        self.placement_crs = placement_crs
        self.stack = ExitStack()
        self.datasets = []

    def __enter__(self):
        try:
            # Bind reads to the verified TIFF itself. GDAL otherwise discovers
            # undeclared .msk/.aux.xml siblings, including inside a tar archive.
            # Internal TIFF masks remain part of the verified bytes and are used.
            self.stack.enter_context(rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                                                 GDAL_PAM_ENABLED=False))
            require(self.tiles and len({t.resource_id for t in self.tiles}) == len(self.tiles),
                    'empty or duplicate DEM resources')
            for tile in self.tiles:
                check_hash(tile.sha256)
                require(Path(tile.path).is_file(), 'DEM payload missing')
                if tile.archive_member is None:
                    digest, raster_path = file_hash(tile.path), tile.path
                else:
                    from oxyformer.exposure.archives import member_hash
                    digest = member_hash(tile.path, tile.archive_member)
                    raster_path = f'/vsitar/{Path(tile.path).resolve()}/{tile.archive_member}'
                require(digest == tile.sha256, 'DEM hash mismatch')
                require(tile.vertical_unit == 'm' and tile.vertical_datum == 'NAVD88',
                        'DEM requires explicit metres and NAVD88 vertical datum')
                require(tile.product in (PRIMARY, FALLBACK), 'unapproved DEM product')
                if tile.product == FALLBACK:
                    require(tile.resource_id in FALLBACK_CELLS and bool(tile.fallback_reason),
                            'fallback only approved for eight missing cells, with reason')
                else:
                    require(tile.fallback_reason is None, 'primary tile has fallback reason')
                ds = self.stack.enter_context(rasterio.open(raster_path))
                require(ds.count == 1 and ds.crs is not None and CRS(ds.crs) == CRS(tile.crs),
                        'DEM CRS mismatch or wrong band count')
                require(ds.nodata == tile.nodata or (ds.nodata is not None and tile.nodata is not None
                        and math.isnan(ds.nodata) and math.isnan(tile.nodata)), 'DEM nodata mismatch')
                require(ds.units[0] in (None, 'm', 'metre', 'meter') and ds.scales == (1.0,) and
                        ds.offsets == (0.0,), 'DEM vertical units/scale mismatch')
                transform = Transformer.from_crs(self.placement_crs, ds.crs, always_xy=True)
                self.datasets.append((ds, transform))
            return self
        except Exception:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def sample(self, xy):
        z = np.full(len(xy), np.nan)
        reason = np.full(len(xy), 'outside_coverage', dtype=object)
        for ds, transform in self.datasets:
            xx, yy = transform.transform(xy[:, 0], xy[:, 1])
            indices = np.flatnonzero(np.isnan(z) & (xx >= ds.bounds.left) &
                (xx < ds.bounds.right) & (yy > ds.bounds.bottom) & (yy <= ds.bounds.top))
            if not len(indices):
                continue
            values = np.ma.concatenate(list(ds.sample(zip(xx[indices], yy[indices]), indexes=1, masked=True)))
            data = values.astype(float).filled(np.nan)
            valid = ~np.ma.getmaskarray(values) & np.isfinite(data)
            domain = (data >= PHYSICS.minimum_elevation_m) & (data <= PHYSICS.maximum_elevation_m)
            reason[indices] = np.where(valid & ~domain, 'outside_physical_domain', 'nodata')
            selected = indices[valid & domain]
            z[selected] = data[valid & domain]
            reason[selected] = 'covered'
        return z, reason


def placement_batches(polygon, scenario, spec, batch_size=4096):
    """Yield (xy, area fractions), including locations outside raster coverage.

    A global fixed grid makes area allocation invariant to block subdivision when
    pieces follow grid boundaries and population follows area. Arbitrary new
    polygon splits can move centroids: this is a placement sensitivity, not data.
    """
    require(polygon.area > 0, 'block has zero projected area')
    if scenario == 'centroid':
        point = polygon.centroid
        yield np.array([[point.x, point.y]]), np.ones(1)
        return
    size = spec.grid_size_m
    ox, oy = spec.grid_origin_m
    left, bottom, right, top = polygon.bounds
    points, fractions = [], []
    for iy in range(math.floor((bottom - oy) / size), math.ceil((top - oy) / size)):
        for ix in range(math.floor((left - ox) / size), math.ceil((right - ox) / size)):
            cell = box(ox + ix * size, oy + iy * size, ox + (ix + 1) * size, oy + (iy + 1) * size)
            piece = polygon.intersection(cell)
            if piece.area <= 0:
                continue
            point = piece.centroid
            points.append((point.x, point.y))
            fractions.append(piece.area / polygon.area)
            if len(points) == batch_size:
                yield np.asarray(points), np.asarray(fractions)
                points, fractions = [], []
    if points:
        yield np.asarray(points), np.asarray(fractions)
