"""INEC continuous-grams observation model, distinct from DANE LBW bands.

Uses the shared immutable birth/audit contracts defined in dane_births. The
observation scale and measurement parser are Ecuador-specific. A release-specific
profile binds fields, sentinels and documented range; no cross-year carry-forward.
"""
from oxyformer.data.adapters.dane_births import _load, _number


def _gram_weight(raw, profile):
    if not raw.strip():
        return None, None, None, 'missing'
    value = _number(raw)
    code = str(int(value)) if value is not None and value == value.to_integral_value() else raw.strip()
    if code in profile['weight_unknown']:
        return None, None, None, 'unknown'
    if code in profile.get('weight_ambiguous', []):
        return None, None, None, 'ambiguous_sentinel'
    if value is None:
        return None, None, None, 'invalid_grams'
    lower, upper = profile['gram_range']
    if not lower <= value <= upper:
        return None, None, None, 'outside_documented_range'
    return float(value), None, bool(value < 2500), 'observed_grams'


def load_births(bundle, release, mapping, exposure_manifest):
    """Return (immutable INEC records, audit), retaining raw invalid weights."""
    return _load(bundle, release, mapping, exposure_manifest, 'ECU', _gram_weight, 'grams')
