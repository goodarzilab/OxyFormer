"""FP64 normalized contributions; clustering/covariance belongs to its adapter."""
import numpy as np

from oxyformer.design.policies import fp64_vector, origin_weights
from oxyformer.provenance import require


def normalized_weights(weights, size: int) -> np.ndarray:
    w = origin_weights(weights, size)
    # Scaling avoids overflow without clipping or changing relative target mass.
    w = w / w.max()
    return w / w.sum(dtype=np.float64)


def influence_contributions(scores, weights) -> tuple[float, np.ndarray]:
    """Return Delta and u_i = (w_i / W) (H_i - Delta), in input ID order."""
    h = fp64_vector(scores, "scores")
    w = normalized_weights(weights, len(h))
    with np.errstate(over="ignore", invalid="ignore"):
        delta = float(w @ h)
        influence = w * (h - delta)
    require(np.isfinite(delta) and bool(np.isfinite(influence).all()), "nonfinite influence")
    return delta, influence
