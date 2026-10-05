"""Numerical domain of placement, separate from physical constants and weights.

GEOS defines represented projected geometry, intersections, inclusion and areas.
Its binary64 vertices define exact planar centroids. A certified identical CRS
uses those rationals directly; otherwise the explicit boundary is float ->
PROJ -> represented float output. Affine pixel classification is exact for that
coordinate representation. Neither clipping nor geodetic projection is claimed
exact. Moment areas below are used only for location, never population weights.
"""
from fractions import Fraction
from pyproj import CRS
from oxyformer.provenance import require

NUMERICAL_POLICY = 'represented_geometry_centroid_proj64_v1'
NUMERICAL_DOMAIN = (
    'GEOS represented projected geometry, intersections, inclusion and population areas; '
    'exact filled-area centroid of represented vertices; certified identical CRS uses '
    'exact centroid and affine; otherwise nearest binary64 centroid -> PROJ binary64 '
    'output -> exact affine. No ideal-clipping, exact-projection or habitation claim.'
)
SAMPLING_MODES = ('exact_identity', 'proj_binary64')
MODE_ACCOUNTING = (
    'Per-scenario location participation by sampling mode, once per location per mode. '
    'A location assessed in both modes contributes to both; these masses are not additive '
    'across modes and do not replace the population/omission ledger.'
)


def _rings(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == 'Polygon':
        yield geometry.exterior.coords, 1
        for ring in geometry.interiors:
            yield ring.coords, -1
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _rings(part)


def exact_centroid(geometry):
    """Filled-area centroid of represented vertices, with integer moments.

    All binary64 coordinates share a power-of-two scale. Integer shoelace
    moments avoid per-vertex rational reductions and preserve both coordinates.
    Ring role, not stored winding, determines shell/hole sign. Polygonal members
    of collections contribute; lines and points have no area contribution.
    """
    rings = list(_rings(geometry))
    scale = max((float(v).as_integer_ratio()[1] for coords, _ in rings
                 for point in coords for v in point[:2]), default=1)
    area2 = mx = my = 0
    for coords, role in rings:
        def scaled(point):
            ratios = [float(v).as_integer_ratio() for v in point[:2]]
            return tuple(n * (scale // d) for n, d in ratios)
        iterator = iter(coords)
        x0, y0 = scaled(next(iterator))
        ring_area = ring_mx = ring_my = 0
        for point in iterator:
            x1, y1 = scaled(point)
            cross = x0*y1 - x1*y0
            ring_area += cross
            ring_mx += (x0+x1)*cross
            ring_my += (y0+y1)*cross
            x0, y0 = x1, y1
        sign = role if ring_area >= 0 else -role
        area2 += sign * ring_area
        mx += sign * ring_mx
        my += sign * ring_my
    require(area2 > 0, 'represented polygon has nonpositive exact moment area')
    return Fraction(mx, 3*area2*scale), Fraction(my, 3*area2*scale)


def _dynamic(value):
    if isinstance(value, dict):
        return (str(value.get('type', '')).startswith('Dynamic') or
                bool({'frame_reference_epoch', 'coordinate_epoch'} & value.keys()) or
                any(_dynamic(v) for v in value.values()))
    return isinstance(value, list) and any(_dynamic(v) for v in value)


def _crs_definition(crs):
    """Conservative comparison: remove only known descriptive object paths."""
    doc = crs.to_json_dict()
    if crs.is_bound or doc['type'] != 'ProjectedCRS' or _dynamic(doc):
        return None

    def descriptive_root(node):
        for key in ('$schema', 'name', 'id', 'scope', 'area', 'bbox', 'usages', 'remarks'):
            node.pop(key, None)
        for axis in node.get('coordinate_system', {}).get('axis', []):
            axis.pop('name', None)
            axis.pop('abbreviation', None)
    descriptive_root(doc)
    descriptive_root(doc['base_crs'])
    for key in ('name', 'id'):
        doc['conversion'].pop(key, None)
    return doc


def _numeric_parameters(crs):
    # Also compare exposed binary64 values before PROJJSON's decimal formatting.
    # This avoids granting identity solely because serialization rounded a value.
    geo = crs.geodetic_crs
    ellipsoid, meridian = crs.ellipsoid, crs.prime_meridian
    return (tuple((p.value, p.unit_conversion_factor) for p in crs.coordinate_operation.params),
            tuple(a.unit_conversion_factor for a in crs.axis_info),
            tuple(a.unit_conversion_factor for a in geo.axis_info),
            ellipsoid.semi_major_metre, ellipsoid.semi_minor_metre, ellipsoid.inverse_flattening,
            meridian.longitude, meridian.unit_conversion_factor)


def same_horizontal_crs(left, right):
    """Certify the supported static projected identity case, without tolerance.

    Semantic equality alone tolerates changed parameters. Strict PROJ object
    identity rejects harmless TIFF labels. Preserve every computational JSON
    field and exposed numeric parameter; unknown differences use PROJ instead.
    Bound/dynamic CRS never receive the shortcut.
    """
    left, right = CRS(left).to_2d(), CRS(right).to_2d()
    if left != right:
        return False
    a, b = _crs_definition(left), _crs_definition(right)
    return (a is not None and b is not None and a == b and
            _numeric_parameters(left) == _numeric_parameters(right))
