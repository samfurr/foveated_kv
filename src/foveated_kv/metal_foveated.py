"""
Metal kernel constants for the 2-tier foveated attention system.

The actual kernels are now precompiled into a .metallib and dispatched via
the C++ FoveatedPrimitive extension. This module provides constants used
by the Python fallback path.
"""

# Max overrides per KV head for spike promotion
MAX_OV = 32

# Split-K parameters
_BASE_SPLIT_SIZE = 256
_MAX_SPLITS = 16
DEFAULT_SPLIT_SIZE = _BASE_SPLIT_SIZE


def optimal_split_size(s_total: int) -> int:
    """Compute split size that keeps num_splits capped at _MAX_SPLITS."""
    if s_total <= _BASE_SPLIT_SIZE * _MAX_SPLITS:
        return _BASE_SPLIT_SIZE
    return ((s_total + _MAX_SPLITS - 1) // _MAX_SPLITS + 255) // 256 * 256


def is_available() -> bool:
    """Check if the Metal foveated kernel can be used.

    With the 2-tier system, the C++ extension with precompiled metallib
    is required for the fused kernel path. The Python fallback uses
    simple reconstruct + SDPA instead.
    """
    try:
        from .mlx_foveated import _cpp_available
        return _cpp_available
    except Exception:
        return False


def foveated_attention_metal(*args, **kwargs):
    """Legacy API — no longer used. The C++ FoveatedHandle is the fast path."""
    raise NotImplementedError(
        "The 3-tier JIT Metal kernel has been replaced by the 2-tier "
        "precompiled metallib. Use the C++ FoveatedHandle via mlx_foveated."
    )
