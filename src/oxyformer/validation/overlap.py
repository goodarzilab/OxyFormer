"""Target, ratio and signed-correction diagnostics with distinct meanings.

ESS(v) = (sum v)^2 / sum(v^2) is used only for nonnegative target weights w
or ratio weights w*r. Signed w*(r-1) is not a sampling weight: report signed
mass, absolute mass and absolute concentration, never an ordinary weight ESS.
"""
from fractions import Fraction

import numpy as np

from oxyformer.design.policies import fp64_vector
from oxyformer.estimation.influence import normalized_weights
from oxyformer.provenance import ContractError, nonempty, require


def weight_diagnostics(values):
    v = fp64_vector(values, "nonnegative weights")
    require(bool((v >= 0).all()), "negative sampling weight")
    if not np.any(v):
        return {"ess": None, "ess_fraction": None, "max_share": None}
    z = v / v.max()
    ess = float(z.sum() ** 2 / (z @ z))
    return {"ess": ess, "ess_fraction": ess / len(v), "max_share": float(z.max() / z.sum())}


def signed_diagnostics(values):
    v = fp64_vector(values, "signed correction weights")
    mass = float(np.abs(v).sum())
    require(np.isfinite(mass), "nonfinite absolute correction mass")
    return {"signed_sum": float(v.sum()), "absolute_mass": mass,
            "max_absolute_share": float(np.abs(v).max() / mass) if mass else None,
            "ess": None, "ess_reason": "Signed correction is not a sampling weight."}



def _functional_expectations(weights, ratios, observed, shifted):
    """Round only final expectations, after exact binary64 products and sums.

    A finite basis can have very large, canceling weighted contributions. Raw
    target masses also avoid overflow or rounding in a floating normalization.
    This evaluates the same weighted expectations without changing a tolerance.
    """
    masses = [Fraction.from_float(float(w)) for w in weights]
    total = sum(masses, Fraction(0))
    ratio_masses = [w * Fraction.from_float(float(r)) for w, r in zip(masses, ratios)]

    def expectation(mass, column):
        value = sum((w * Fraction.from_float(float(f)) for w, f in zip(mass, column)), Fraction(0)) / total
        try:
            return float(value)
        except OverflowError as exc:
            raise ContractError("nonfinite functional balance expectation") from exc

    return (np.array([expectation(ratio_masses, column) for column in observed.T]),
            np.array([expectation(masses, column) for column in shifted.T]))


def overlap_report(weights, ratios, observed, shifted, names, f_a, f_d):
    a = fp64_vector(observed, "observed exposure")
    d = fp64_vector(shifted, "shifted exposure", len(a))
    r = fp64_vector(ratios, "density ratios", len(a))
    require(bool((r >= 0).all()), "negative density ratios")
    q = normalized_weights(weights, len(a))
    fa, fd = np.asarray(f_a, dtype=np.float64), np.asarray(f_d, dtype=np.float64)
    require(bool(names) and len(set(names)) == len(names), "frozen balance functions required")
    for name in names:
        require(isinstance(name, str), "balance function name must be text")
        nonempty(name, "balance function name")
    require(fa.shape == fd.shape == (len(a), len(names)), "balance function alignment mismatch")
    require(bool(np.isfinite(fa).all() and np.isfinite(fd).all()), "nonfinite balance functions")
    moved = d != a
    affected = moved | (r != 1)  # unchanged records can receive incoming mass
    subsets = {}
    for name, mask in (("all", np.ones(len(a), dtype=bool)), ("moved", moved), ("affected", affected)):
        if not mask.any():
            subsets[name] = {"count": 0, "target_mass": 0.0, "diagnostics": None}
            continue
        target = weight_diagnostics(q[mask])
        ratio = weight_diagnostics((q * r)[mask])
        p99 = float(np.quantile(r[mask], .99))
        warnings = ["ratio p99 > 10"] if p99 > 10 else []
        for label, diagnostic in (("target", target), ("ratio", ratio)):
            if diagnostic["ess_fraction"] is not None and diagnostic["ess_fraction"] < .25:
                warnings.append(f"{label} ESS < 25% of subset records")
        subsets[name] = {"count": int(mask.sum()), "target_mass": float(q[mask].sum()),
                         "target_weights": target, "ratio_weights": ratio,
                         "signed_correction": signed_diagnostics((q * (r - 1))[mask]),
                         "ratio_p99": p99, "warnings": warnings}
    left, right = _functional_expectations(weights, r, fa, fd)
    require(bool(np.isfinite(left).all() and np.isfinite(right).all()), "nonfinite functional balance")
    shift = float(q @ (d - a))
    require(np.isfinite(shift), "nonfinite achieved shift")
    return {"moved_fraction": float(q @ moved), "affected_fraction": float(q @ affected),
            "achieved_shift": shift, "subsets": subsets,
            "functional_balance": [dict(function=n, ratio_expectation=float(l), shifted_expectation=float(v),
                                         difference=float(l - v)) for n, l, v in zip(names, left, right)],
            "formulas": {"ess": "(sum v)^2 / sum(v^2), v>=0; target v=w, ratio v=w*r",
                         "signed": "c_i=(w_i/W)*(r_i-1); max(abs(c))/sum(abs(c)); no ESS",
                         "balance": "E_T[r*f(A,X)] versus E_T[f(d(A,X),X)]"},
            "limitation": "p99 is an empirical record quantile. Thresholds are warnings, not universal cutoffs. "
                          "Balance is a full-target identity, not a conditional moved-subset identity."}
