"""
MLX-native quantization utilities for foveated KV cache.

Mirrors the PyTorch quantize.py API exactly:
  - INT8 per-channel (keys): scale/zero along last dim, computed across tokens
  - INT8 per-token (peripheral values): scale/zero per token row
  - INT4 per-token packed (far values): 2 values per byte, nibble packing

All functions return (quantized, scale, zero_point) tuples.
"""

import mlx.core as mx


def quantize_int8_per_channel(
    tensor: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize fp16/fp32 tensor to INT8 with per-channel (last dim) scaling.

    Args:
        tensor: (..., N, D) float tensor. N can be 0.

    Returns:
        quantized: (..., N, D) uint8
        scale: (..., D) float — per-channel scale
        zero_point: (..., D) float — per-channel zero
    """
    if tensor.shape[-2] == 0:
        D = tensor.shape[-1]
        batch_shape = tensor.shape[:-2]
        return (
            tensor.astype(mx.uint8),
            mx.ones(batch_shape + (D,), dtype=tensor.dtype),
            mx.zeros(batch_shape + (D,), dtype=tensor.dtype),
        )
    vmin = mx.min(tensor, axis=-2, keepdims=True)
    vmax = mx.max(tensor, axis=-2, keepdims=True)
    scale = (vmax - vmin) / 255.0
    scale = mx.maximum(scale, 1e-8)
    zero_point = vmin
    quantized = mx.clip(mx.round((tensor - zero_point) / scale), 0, 255).astype(
        mx.uint8
    )
    return quantized, mx.squeeze(scale, axis=-2), mx.squeeze(zero_point, axis=-2)


def dequantize_int8_per_channel(
    quantized: mx.array, scale: mx.array, zero_point: mx.array
) -> mx.array:
    """Dequantize INT8 back to float16.

    Args:
        quantized: (..., N, D) uint8
        scale: (..., D) float
        zero_point: (..., D) float

    Returns:
        (..., N, D) float16
    """
    return (
        quantized.astype(mx.float32)
        * mx.expand_dims(scale, axis=-2)
        + mx.expand_dims(zero_point, axis=-2)
    ).astype(mx.float16)


def quantize_int8_per_token(
    tensor: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize with per-token (per-row) scaling. Better for values.

    Args:
        tensor: (..., N, D) float. N can be 0.

    Returns:
        quantized: (..., N, D) uint8
        scale: (..., N, 1) float
        zero_point: (..., N, 1) float
    """
    if tensor.shape[-2] == 0:
        batch_shape = tensor.shape[:-1]
        return (
            tensor.astype(mx.uint8),
            mx.zeros(batch_shape + (1,), dtype=tensor.dtype),
            mx.zeros(batch_shape + (1,), dtype=tensor.dtype),
        )
    vmin = mx.min(tensor, axis=-1, keepdims=True)
    vmax = mx.max(tensor, axis=-1, keepdims=True)
    scale = (vmax - vmin) / 255.0
    scale = mx.maximum(scale, 1e-8)
    zero_point = vmin
    quantized = mx.clip(mx.round((tensor - zero_point) / scale), 0, 255).astype(
        mx.uint8
    )
    return quantized, scale, zero_point


def dequantize_int8_per_token(
    quantized: mx.array, scale: mx.array, zero_point: mx.array
) -> mx.array:
    """Dequantize per-token INT8."""
    return (quantized.astype(mx.float32) * scale + zero_point).astype(mx.float16)


def quantize_int4_per_token(
    tensor: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize to INT4, packed 2 values per byte, with per-token scaling.

    Args:
        tensor: (..., N, D) float — D must be even, N can be 0

    Returns:
        packed: (..., N, D//2) uint8 — two INT4 values per byte
        scale: (..., N, 1) float
        zero_point: (..., N, 1) float
    """
    assert (
        tensor.shape[-1] % 2 == 0
    ), f"D must be even for INT4 packing, got {tensor.shape[-1]}"
    if tensor.shape[-2] == 0:
        D_half = tensor.shape[-1] // 2
        batch_shape = tensor.shape[:-1]
        packed = mx.zeros(batch_shape + (D_half,), dtype=mx.uint8)
        return (
            packed,
            mx.zeros(batch_shape + (1,), dtype=tensor.dtype),
            mx.zeros(batch_shape + (1,), dtype=tensor.dtype),
        )
    vmin = mx.min(tensor, axis=-1, keepdims=True)
    vmax = mx.max(tensor, axis=-1, keepdims=True)
    scale = (vmax - vmin) / 15.0
    scale = mx.maximum(scale, 1e-8)
    zero_point = vmin
    normalized = mx.clip(mx.round((tensor - zero_point) / scale), 0, 15).astype(
        mx.uint8
    )
    # Pack: even indices in low nibble, odd indices in high nibble
    low = normalized[..., 0::2]  # even elements
    high = normalized[..., 1::2]  # odd elements
    packed = mx.bitwise_or(low, mx.left_shift(high, mx.array(4, dtype=mx.uint8)))
    return packed, scale, zero_point


def dequantize_int4_per_token(
    packed: mx.array, scale: mx.array, zero_point: mx.array
) -> mx.array:
    """Dequantize packed INT4 back to float16.

    Args:
        packed: (..., N, D//2) uint8
        scale: (..., N, 1) float
        zero_point: (..., N, 1) float

    Returns:
        (..., N, D) float16
    """
    low = mx.bitwise_and(packed, mx.array(0x0F, dtype=mx.uint8))
    high = mx.bitwise_and(
        mx.right_shift(packed, mx.array(4, dtype=mx.uint8)),
        mx.array(0x0F, dtype=mx.uint8),
    )
    # Interleave back to original order: [low0, high0, low1, high1, ...]
    D_half = packed.shape[-1]
    batch_shape = packed.shape[:-1]
    # Stack and reshape to interleave
    # low: (..., D_half), high: (..., D_half)
    # Want: (..., D) where result[..., 0::2] = low, result[..., 1::2] = high
    unpacked = mx.zeros(batch_shape + (D_half * 2,), dtype=mx.uint8)
    # MLX doesn't support slice assignment, so use concatenation + reshape
    # Stack along last dim: (... D_half, 2) then reshape to (... D)
    interleaved = mx.stack([low, high], axis=-1)  # (..., D_half, 2)
    unpacked = interleaved.reshape(batch_shape + (D_half * 2,))
    return (unpacked.astype(mx.float32) * scale + zero_point).astype(mx.float16)
