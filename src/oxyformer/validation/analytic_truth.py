"""Independent closed-form truths for synthetic continuous uniform fixtures."""
from dataclasses import dataclass

import numpy as np

from oxyformer.design.policies import fp64_vector
from oxyformer.provenance import Immutable, require


@dataclass(frozen=True, slots=True, kw_only=True)
class UniformShiftTruth(Immutable):
    lower: float = 0.0
    upper: float = 10.0
    delta: float = 2.0

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(self.lower < self.upper and self.delta >= 0, "invalid uniform truth parameters")

    def density(self, a, support_keys=None) -> np.ndarray:
        a = fp64_vector(a, "exposure")
        return ((a >= self.lower) & (a <= self.upper)).astype(np.float64) / (self.upper - self.lower)

    def ratio(self, a) -> np.ndarray:
        """Independent preimage counts, including overlap for a long shift."""
        a = fp64_vector(a, "exposure")
        require(bool(((a >= self.lower) & (a <= self.upper)).all()), "outside uniform law")
        if self.delta == 0 or self.delta > self.upper - self.lower:
            return np.ones_like(a)
        return ((a >= self.lower + self.delta).astype(np.float64)
                + (a > self.upper - self.delta).astype(np.float64))

    @property
    def moved_fraction(self) -> float:
        if self.delta == 0:
            return 0.0
        return max(1.0 - self.delta / (self.upper - self.lower), 0.0)

    def linear_contrast(self, beta: float) -> float:
        require(np.isfinite(beta), "nonfinite slope")
        return float(beta * self.delta * self.moved_fraction)
