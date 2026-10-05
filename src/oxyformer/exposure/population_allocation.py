"""Fixed area-based placement and bounded raster sampling, without outcome inputs."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from fractions import Fraction
import math
from pathlib import Path
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from oxyformer.exposure.physics import PHYSICS, APPROVED_DEM_PRODUCT, validate_owner_approval
from oxyformer.provenance import canonical_json, check_hash, file_hash, require

PRIMARY = APPROVED_DEM_PRODUCT
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
        # Canonical identity for accepted equivalent CRS and numeric spellings.
        object.__setattr__(self, 'placement_crs', 'EPSG:5070')
        object.__setattr__(self, 'grid_size_m', float(self.grid_size_m))
        object.__setattr__(self, 'grid_origin_m', tuple(float(x) for x in self.grid_origin_m))

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

    def identity(self, verified_crs):
        identity = {k: v for k, v in asdict(self).items() if k != 'path'}
        # The raster bytes anchor identity, independently of the task's accepted
        # equivalent CRS spelling. The original declaration stays in task_hash.
        identity['crs'] = CRS(verified_crs).to_wkt()
        return identity


def _validate_vertical_crs(crs):
    """Corroborate explicit height semantics; a 2D CRS relies on reviewed labels.

    pyproj CRS sub_crs_list/axis_info/Datum equality expose the PROJ semantics:
    https://pyproj4.github.io/pyproj/stable/api/crs/crs.html
    """
    if crs.is_bound:
        _validate_vertical_crs(crs.source_crs)
    elif crs.is_compound:
        for component in crs.sub_crs_list:
            _validate_vertical_crs(component)
    elif crs.is_vertical:
        require(crs.datum == CRS('EPSG:5703').datum,
                'DEM explicit vertical datum must be NAVD88')
        require(len(crs.axis_info) == 1 and crs.axis_info[0].direction == 'up' and
                crs.axis_info[0].unit_conversion_factor == 1.0,
                'DEM vertical axis must be upward metres')
    else:
        require(len(crs.axis_info) == 2, 'DEM explicit ellipsoidal/3D height is not NAVD88')


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
        self.identities = []

    def __enter__(self):
        try:
            validate_owner_approval(use_fallback=any(t.product == FALLBACK for t in self.tiles))
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
                require(ds.count == 1 and ds.crs is not None, 'DEM CRS missing or wrong band count')
                embedded, declared = CRS(ds.crs), CRS(tile.crs)
                _validate_vertical_crs(embedded)
                _validate_vertical_crs(declared)
                require(embedded.to_2d() == declared.to_2d(), 'DEM CRS mismatch')
                require(all(math.isfinite(v) for v in ds.transform),
                        'DEM requires a finite invertible affine transform')
                require(not (ds.transform.is_identity and (ds.gcps[0] or ds.rpcs)),
                        'DEM requires affine rather than GCP/RPC-only georeferencing')
                a, b, c, d, e, f = (Fraction(v) for v in tuple(ds.transform)[:6])
                determinant = a * e - b * d
                require(determinant != 0, 'DEM requires a finite invertible affine transform')
                inverse = (e / determinant, -b / determinant, (b*f - c*e) / determinant,
                           -d / determinant, a / determinant, (c*d - a*f) / determinant)
                require(ds.nodata == tile.nodata or (ds.nodata is not None and tile.nodata is not None
                        and math.isnan(ds.nodata) and math.isnan(tile.nodata)), 'DEM nodata mismatch')
                require(ds.units[0] in (None, 'm', 'metre', 'meter') and ds.scales == (1.0,) and
                        ds.offsets == (0.0,), 'DEM vertical units/scale mismatch')
                transform = Transformer.from_crs(self.placement_crs, embedded.to_2d(), always_xy=True)
                self.datasets.append((ds, transform, inverse))
                self.identities.append(tile.identity(ds.crs))
            return self
        except Exception:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def sample(self, xy):
        z = np.full(len(xy), np.nan)
        reason = np.full(len(xy), 'outside_coverage', dtype=object)
        for ds, transform, inverse in self.datasets:
            xx, yy = transform.transform(xy[:, 0], xy[:, 1])
            # Pixel membership is discontinuous. Preserve the stored affine and
            # represented transformed coordinates exactly until integer flooring:
            # even 0.1*x can otherwise round an interior point into adjacent nodata.
            # https://gdal.org/en/stable/tutorials/geotransforms_tut.html
            finite = np.isfinite(xx) & np.isfinite(yy)
            indices, pixels = [], []
            a, b, c, d, e, f = inverse
            for index in np.flatnonzero(np.isnan(z) & finite):
                x, y = Fraction(float(xx[index])), Fraction(float(yy[index]))
                col, row = math.floor(a*x + b*y + c), math.floor(d*x + e*y + f)
                if 0 <= col < ds.width and 0 <= row < ds.height:
                    indices.append(index)
                    pixels.append((row, col))
            if not indices:
                continue
            indices = np.asarray(indices)
            # Dispersed locations must not create an unbounded bounding window.
            values = np.ma.concatenate([ds.read(1, window=Window(col, row, 1, 1), masked=True).reshape(-1)
                                        for row, col in pixels])
            data = values.astype(float).filled(np.nan)
            valid = ~np.ma.getmaskarray(values) & np.isfinite(data)
            domain = (data >= PHYSICS.minimum_elevation_m) & (data <= PHYSICS.maximum_elevation_m)
            reason[indices] = np.where(valid & ~domain, 'outside_physical_domain', 'nodata')
            selected = indices[valid & domain]
            z[selected] = data[valid & domain]
            reason[selected] = 'covered'
        return z, reason


def placement_batches(polygon, scenario, spec, batch_size=4096):
    """Yield (xy, raw square-metre areas), including locations outside raster coverage.

    A global fixed grid makes area allocation invariant to block subdivision when
    pieces follow grid boundaries, population follows area, and represented
    intersection areas stay identical. Arbitrary polygon splits can move centroids: this is a placement sensitivity, not data.
    """
    require(math.isfinite(polygon.area) and polygon.area > 0, 'block has invalid projected area')
    if scenario == 'centroid':
        point = polygon.centroid
        yield np.array([[point.x, point.y]]), np.array([polygon.area])
        return
    # Preserve represented bounds through subtraction/division: floating
    # cancellation at a translated grid origin can otherwise omit a real strip.
    size = Fraction(spec.grid_size_m)
    ox, oy = (Fraction(value) for value in spec.grid_origin_m)
    left, bottom, right, top = (Fraction(value) for value in polygon.bounds)
    points, areas = [], []
    for iy in range(math.floor((bottom - oy) / size), math.ceil((top - oy) / size)):
        for ix in range(math.floor((left - ox) / size), math.ceil((right - ox) / size)):
            # Round each canonical grid line only once when entering GEOS.
            cell = box(float(ox + ix * size), float(oy + iy * size),
                       float(ox + (ix + 1) * size), float(oy + (iy + 1) * size))
            piece = polygon.intersection(cell)
            if piece.area <= 0:
                continue
            point = piece.centroid
            points.append((point.x, point.y))
            areas.append(piece.area)
            if len(points) == batch_size:
                yield np.asarray(points), np.asarray(areas)
                points, areas = [], []
    if points:
        yield np.asarray(points), np.asarray(areas)
