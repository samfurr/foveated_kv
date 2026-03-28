"""TurboQuant compression and dequantization for KV cache.

Two-stage key compression:
  Stage 1 (Lloyd-Max): random rotation + 2-bit scalar quantization per dim
  Stage 2 (QJL): 1-bit sign quantization of residual random projections

Value compression: 2-bit symmetric group quantization (group_size=32).
"""

import math

import mlx.core as mx

from .turbo_constants import TurboConstants

# 2-bit symmetric value dequant levels: {0,1,2,3} -> {-1, -1/3, 1/3, 1}
_VAL_LEVELS = mx.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=mx.float32)


# ---------------------------------------------------------------------------
# Key compression (3.25 bits/dim)
# ---------------------------------------------------------------------------


def turbo_compress_keys(
    keys: mx.array,
    tc: TurboConstants,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compress keys via TurboQuant (Lloyd-Max + QJL).

    Args:
        keys: (B, H, N, D) float16/bf16/float32
        tc: precomputed TurboConstants

    Returns:
        indices_packed: (B, H, N, D//4) uint8 — 2-bit Lloyd-Max indices
        signs_packed:   (B, H, N, D//8) uint8 — 1-bit QJL signs
        norms:          (B, H, N) float16 — ||k||
        gamma:          (B, H, N) float16 — ||residual||
    """
    B, H, N, D = keys.shape
    k = keys.astype(mx.float32)
    Pi = tc.Pi  # (D, D)
    centroids = tc.centroids  # (4,)
    boundaries = tc.boundaries  # (3,)
    S = tc.S  # (D, D)

    # Key norms
    norms = mx.sqrt(mx.sum(k * k, axis=-1))  # (B, H, N)
    k_normed = k / mx.maximum(mx.expand_dims(norms, -1), 1e-8)

    # Stage 1: rotate and quantize
    y = k_normed @ Pi.T  # (B, H, N, D)

    # Quantize each coordinate to nearest of 4 centroids via boundaries
    # boundaries[0] < boundaries[1] < boundaries[2]
    # idx: 0 if y < b0, 1 if b0 <= y < b1, 2 if b1 <= y < b2, 3 if y >= b2
    b0, b1, b2 = boundaries[0], boundaries[1], boundaries[2]
    idx = (y >= b0).astype(mx.uint8) + (y >= b1).astype(mx.uint8) + (y >= b2).astype(mx.uint8)
    # idx is (B, H, N, D) uint8 with values 0-3

    # Pack 4 indices per byte (2 bits each)
    idx_r = idx.reshape(B, H, N, D // 4, 4)
    indices_packed = (
        idx_r[..., 0]
        | (idx_r[..., 1] << 2)
        | (idx_r[..., 2] << 4)
        | (idx_r[..., 3] << 6)
    ).astype(mx.uint8)  # (B, H, N, D//4)

    # Stage 2: QJL residual correction
    # Dequant stage 1 to get k_hat
    centroid_vals = centroids[idx.astype(mx.uint32)]  # (B, H, N, D) — lookup
    y_dequant = centroid_vals  # rotated-space reconstruction
    k_hat = mx.expand_dims(norms, -1) * (y_dequant @ Pi)  # rotate back: (B, H, N, D)

    residual = k - k_hat  # (B, H, N, D)
    gamma = mx.sqrt(mx.sum(residual * residual, axis=-1))  # (B, H, N)

    # Project residual and take signs
    projected = residual @ S.T  # (B, H, N, D)
    sign_bits = (projected >= 0).astype(mx.uint8)  # 1 if positive, 0 if negative

    # Pack 8 sign bits per byte
    sign_r = sign_bits.reshape(B, H, N, D // 8, 8)
    signs_packed = sign_r[..., 0]
    for i in range(1, 8):
        signs_packed = signs_packed | (sign_r[..., i] << i)
    signs_packed = signs_packed.astype(mx.uint8)  # (B, H, N, D//8)

    return indices_packed, signs_packed, norms.astype(mx.float16), gamma.astype(mx.float16)


# ---------------------------------------------------------------------------
# Key dequantization (full reconstruction for fallback path)
# ---------------------------------------------------------------------------


def turbo_dequant_keys(
    indices_packed: mx.array,
    signs_packed: mx.array,
    norms: mx.array,
    gamma: mx.array,
    tc: TurboConstants,
) -> mx.array:
    """Full key reconstruction from TurboQuant compressed data.

    Returns: (B, H, N, D) float16
    """
    B, H, N, D4 = indices_packed.shape
    D = D4 * 4
    Pi = tc.Pi
    centroids = tc.centroids
    S = tc.S

    # Unpack 2-bit indices
    idx = _unpack_2bit(indices_packed, D)  # (B, H, N, D) uint8

    # Stage 1 reconstruction
    centroid_vals = centroids[idx.astype(mx.uint32)]  # (B, H, N, D)
    norms_f = norms.astype(mx.float32)
    k_mse = mx.expand_dims(norms_f, -1) * (centroid_vals @ Pi)  # (B, H, N, D)

    # Stage 2 QJL reconstruction (approximate)
    signs = _unpack_1bit(signs_packed, D)  # (B, H, N, D) float32 {-1, +1}
    gamma_f = gamma.astype(mx.float32)
    # Reconstruct residual direction from sign sketch: r_hat = gamma * S^T @ signs / D
    # This is an approximate reconstruction; the exact residual is lost.
    r_hat = mx.expand_dims(gamma_f, -1) * (signs @ S) / D  # (B, H, N, D)

    return (k_mse + r_hat).astype(mx.float16)


# ---------------------------------------------------------------------------
# Value compression (2-bit symmetric group quantization)
# ---------------------------------------------------------------------------


def turbo_compress_values(
    values: mx.array,
    group_size: int = 32,
) -> tuple[mx.array, mx.array]:
    """Compress values via 2-bit symmetric group quantization.

    Args:
        values: (B, H, N, D) float16/bf16/float32
        group_size: quantization group size (default 32)

    Returns:
        packed: (B, H, N, D//4) uint8 — 2-bit packed
        scales: (B, H, N, n_groups) float16 — per-group scale
    """
    B, H, N, D = values.shape
    n_groups = D // group_size
    v = values.astype(mx.float32).reshape(B, H, N, n_groups, group_size)

    # Per-group scale = max(abs(group))
    scales = mx.max(mx.abs(v), axis=-1)  # (B, H, N, n_groups)
    scales = mx.maximum(scales, 1e-8)

    # Quantize: val/scale maps to [-1, 1], then to levels {-1, -1/3, 1/3, 1}
    # Nearest level: round((val/scale + 1) * 1.5) clipped to [0, 3]
    normalized = v / mx.expand_dims(scales, -1)  # [-1, 1]
    idx = mx.clip(mx.round((normalized + 1.0) * 1.5), 0, 3).astype(mx.uint8)
    # idx: 0=-1, 1=-1/3, 2=1/3, 3=1

    idx_flat = idx.reshape(B, H, N, D)

    # Pack 4 values per byte (2 bits each)
    idx_r = idx_flat.reshape(B, H, N, D // 4, 4)
    packed = (
        idx_r[..., 0]
        | (idx_r[..., 1] << 2)
        | (idx_r[..., 2] << 4)
        | (idx_r[..., 3] << 6)
    ).astype(mx.uint8)

    return packed, scales.astype(mx.float16)


def turbo_dequant_values(
    packed: mx.array,
    scales: mx.array,
    group_size: int = 32,
) -> mx.array:
    """Dequantize 2-bit symmetric group quantized values.

    Returns: (B, H, N, D) float16
    """
    B, H, N, D4 = packed.shape
    D = D4 * 4
    n_groups = D // group_size

    # Unpack 2-bit indices
    idx = _unpack_2bit(packed, D)  # (B, H, N, D) uint8

    # Map indices to levels
    levels = _VAL_LEVELS[idx.astype(mx.uint32)]  # (B, H, N, D) float32

    # Apply per-group scale
    levels_grouped = levels.reshape(B, H, N, n_groups, group_size)
    scales_f = mx.expand_dims(scales.astype(mx.float32), -1)
    result = (levels_grouped * scales_f).reshape(B, H, N, D)

    return result.astype(mx.float16)


# ---------------------------------------------------------------------------
# Optimized attention score (no full dequant)
# ---------------------------------------------------------------------------


def turbo_score_keys(
    query: mx.array,
    indices_packed: mx.array,
    signs_packed: mx.array,
    norms: mx.array,
    gamma: mx.array,
    tc: TurboConstants,
) -> mx.array:
    """Compute attention logits directly on TurboQuant compressed keys.

    No full key dequantization — computes dot products via codebook lookup
    and QJL sign inner product.

    Args:
        query: (B, H_q, 1, D) float32/16
        indices_packed: (B, H_kv, N, D//4) uint8
        signs_packed: (B, H_kv, N, D//8) uint8
        norms: (B, H_kv, N) float16
        gamma: (B, H_kv, N) float16
        tc: TurboConstants

    Returns:
        logits: (B, H_kv, N) float32 — raw attention scores (pre-softmax)
    """
    B, H_kv, N, D4 = indices_packed.shape
    D = D4 * 4
    q = query.astype(mx.float32)
    if q.ndim == 4:
        q = q[:, :, 0, :]  # (B, H_q, D)

    Pi = tc.Pi  # (D, D)
    S = tc.S  # (D, D)
    centroids = tc.centroids  # (4,)

    # Pre-rotate query (once, amortized over all far tokens)
    q_rot = q @ Pi.T  # (B, H_q, D)

    # Pre-sketch query for QJL
    q_sketch = q @ S.T  # (B, H_q, D)

    # Unpack indices and signs
    idx = _unpack_2bit(indices_packed, D)  # (B, H_kv, N, D)
    signs = _unpack_1bit(signs_packed, D)  # (B, H_kv, N, D) {-1, +1}

    # MSE score: norm_k * sum_j(q_rot[j] * centroids[idx[j]])
    centroid_vals = centroids[idx.astype(mx.uint32)]  # (B, H_kv, N, D)
    # For GQA: q_rot is (B, H_q, D), centroid_vals is (B, H_kv, N, D)
    # Average query heads per KV head group for scoring
    H_q = q_rot.shape[1]
    if H_q != H_kv:
        group_size = H_q // H_kv
        q_rot_kv = mx.mean(q_rot.reshape(B, H_kv, group_size, D), axis=2)
        q_sketch_kv = mx.mean(q_sketch.reshape(B, H_kv, group_size, D), axis=2)
    else:
        q_rot_kv = q_rot
        q_sketch_kv = q_sketch

    # (B, H_kv, D) x (B, H_kv, N, D) -> (B, H_kv, N)
    score_mse = norms.astype(mx.float32) * mx.sum(
        mx.expand_dims(q_rot_kv, 2) * centroid_vals, axis=-1
    )

    # QJL score: sqrt(pi/2) / D * gamma * sum_j(sign[j] * q_sketch[j])
    sign_dot = mx.sum(mx.expand_dims(q_sketch_kv, 2) * signs, axis=-1)
    score_qjl = (math.sqrt(math.pi / 2) / D) * gamma.astype(mx.float32) * sign_dot

    return score_mse + score_qjl


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------


def _unpack_2bit(packed: mx.array, D: int) -> mx.array:
    """Unpack 2-bit values from byte array.

    packed: (..., D//4) uint8 → (..., D) uint8 with values 0-3
    """
    shape = packed.shape[:-1]
    p = packed.astype(mx.uint32)
    v0 = p & 0x3
    v1 = (p >> 2) & 0x3
    v2 = (p >> 4) & 0x3
    v3 = (p >> 6) & 0x3
    return mx.stack([v0, v1, v2, v3], axis=-1).reshape(*shape, D).astype(mx.uint8)


def _unpack_1bit(packed: mx.array, D: int) -> mx.array:
    """Unpack 1-bit signs from byte array.

    packed: (..., D//8) uint8 → (..., D) float32 with values {-1, +1}
    """
    shape = packed.shape[:-1]
    p = packed.astype(mx.uint32)
    bits = []
    for i in range(8):
        bits.append((p >> i) & 1)
    # Stack and reshape
    raw = mx.stack(bits, axis=-1).reshape(*shape, D)  # 0 or 1
    return (2.0 * raw.astype(mx.float32) - 1.0)  # {-1, +1}
