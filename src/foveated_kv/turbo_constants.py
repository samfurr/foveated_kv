"""TurboQuant constants: rotation matrix Pi, QJL matrix S, Lloyd-Max centroids.

All matrices are deterministic from seed. Shared across layers — generated
once per (dim, seed) pair and cached.

The Lloyd-Max centroids are for the unit-sphere marginal distribution:
  f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
which converges to N(0, 1/d) for large d. For 2-bit quantization (4 levels),
centroids are computed via iterative Lloyd-Max on this PDF.
"""

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class TurboConstants:
    """Precomputed TurboQuant constants for a given head dimension."""

    Pi: mx.array  # (D, D) float32 — orthogonal rotation
    S: mx.array  # (D, D) float32 — QJL random projection
    centroids: mx.array  # (4,) float32 — Lloyd-Max centroids
    boundaries: mx.array  # (3,) float32 — decision boundaries (midpoints)


@lru_cache(maxsize=4)
def get_turbo_constants(d: int, pi_seed: int = 42, s_seed: int = 137) -> TurboConstants:
    """Get or compute TurboQuant constants for head dimension d."""
    Pi = _rotation_matrix(d, pi_seed)
    S = _qjl_matrix(d, s_seed)
    centroids, boundaries = _lloyd_max_centroids(d, bits=2)
    return TurboConstants(
        Pi=mx.array(Pi, dtype=mx.float32),
        S=mx.array(S, dtype=mx.float32),
        centroids=mx.array(centroids, dtype=mx.float32),
        boundaries=mx.array(boundaries, dtype=mx.float32),
    )


def _rotation_matrix(d: int, seed: int) -> np.ndarray:
    """Deterministic d x d orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(H)
    # Fix sign ambiguity: ensure diagonal of R is positive
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[np.newaxis, :]
    return Q


def _qjl_matrix(d: int, seed: int) -> np.ndarray:
    """Deterministic d x d random Gaussian matrix for QJL projection."""
    rng = np.random.RandomState(seed)
    return rng.randn(d, d).astype(np.float32) / np.sqrt(d)


def _lloyd_max_centroids(
    d: int, bits: int = 2, n_iters: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max centroids for unit-sphere marginal distribution.

    For d-dimensional unit sphere, each coordinate after random rotation
    follows Beta((d-1)/2, (d-1)/2) scaled to [-1, 1]. For large d, this
    is approximately N(0, 1/d).

    Returns (centroids, boundaries) as numpy arrays.
    """
    n_levels = 2**bits
    sigma = 1.0 / np.sqrt(d)

    # Initialize centroids at quantiles of N(0, sigma^2)
    from scipy.stats import norm

    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    init_boundaries = norm.ppf(quantiles, scale=sigma)

    # Lloyd-Max iteration on Gaussian PDF (good approx for large d)
    boundaries = init_boundaries.copy()
    for _ in range(n_iters):
        # Extended boundaries with ±inf
        ext = np.concatenate([[-np.inf], boundaries, [np.inf]])
        centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = ext[i], ext[i + 1]
            # Conditional expectation E[X | lo < X < hi] under N(0, sigma^2)
            p_lo = norm.pdf(lo, scale=sigma)
            p_hi = norm.pdf(hi, scale=sigma)
            prob = norm.cdf(hi, scale=sigma) - norm.cdf(lo, scale=sigma)
            if prob > 1e-15:
                centroids[i] = sigma**2 * (p_lo - p_hi) / prob
            else:
                centroids[i] = (lo + hi) / 2 if np.isfinite(lo + hi) else 0
        # Update boundaries as midpoints
        boundaries = (centroids[:-1] + centroids[1:]) / 2

    return centroids.astype(np.float32), boundaries.astype(np.float32)
