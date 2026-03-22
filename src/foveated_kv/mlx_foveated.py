"""
MLX-native FoveatedKV: 2-tier importance-adaptive KV cache.

Native Apple Silicon implementation using MLX arrays and Metal-accelerated
attention.

Two tiers:
  Near: fp16 K + fp16 V  (high-importance: sinks, window, recent middle)
  Far K: fp8 E4M3 (register-level encode/decode, ~12.5% relative error)
  Far V: int4 per-token (nibble-packed, per-token scale+zero)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

# Capture the ORIGINAL sdpa before any monkey-patching happens.
_original_sdpa = mx.fast.scaled_dot_product_attention

# Optional C++ extension (precompiled metallib + FoveatedPrimitive + CompressHandle)
_cpp_available = False
_FoveatedHandle = None
_CompressHandle = None
_CppTierConfig = None
_PromotionPipeline = None
_metallib_path = ""
try:
    import os as _os
    import sys as _sys

    if "src" not in _sys.path:
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
    from foveated_ext import FoveatedHandle as _FoveatedHandle
    from foveated_ext import CompressHandle as _CompressHandle
    from foveated_ext import TierConfig as _CppTierConfig
    from foveated_ext import PromotionPipeline as _PromotionPipeline

    import importlib.util as _ilu

    _ext_spec = _ilu.find_spec("foveated_ext")
    if _ext_spec and _ext_spec.origin:
        _candidate = _os.path.join(
            _os.path.dirname(_ext_spec.origin), "foveated_attn.metallib"
        )
        if not _os.path.exists(_candidate):
            _build_candidate = _os.path.join(
                _os.path.dirname(__file__),
                "..",
                "..",
                "build",
                "foveated_attn.metallib",
            )
            if _os.path.exists(_build_candidate):
                _candidate = _os.path.abspath(_build_candidate)
            else:
                _build_candidate2 = _os.path.join(
                    _os.path.dirname(__file__),
                    "..",
                    "..",
                    "build_ext",
                    "foveated_attn.metallib",
                )
                if _os.path.exists(_build_candidate2):
                    _candidate = _os.path.abspath(_build_candidate2)
        if _os.path.exists(_candidate):
            _metallib_path = _candidate
            _cpp_available = True
except ImportError:
    pass


def _e4m3_to_fp16(fp8_data: mx.array) -> mx.array:
    """Reconstruct fp16 from fp8 E4M3 data.

    E4M3: sign(1) + exp(4, bias=7) + mantissa(3)
    FP16: sign(1) + exp(5, bias=15) + mantissa(10)
    Rebias exponent (+8), zero-pad mantissa (3→10 bits).
    """
    v = fp8_data.astype(mx.uint32)
    sign = v >> 7
    exp8 = (v >> 3) & 0xF
    mant = v & 0x7
    # Rebias exponent: E4M3 bias=7 → fp16 bias=15 (add 8)
    # Zero exponent stays zero (preserve zeros)
    exp16 = mx.where(exp8 > 0, exp8 + 8, mx.zeros_like(exp8))
    fp16_bits = (sign << 15) | (exp16 << 10) | (mant << 7)
    return fp16_bits.astype(mx.uint16).view(mx.float16)


def _dequant_int4_per_token(packed: mx.array, scale: mx.array, zero: mx.array) -> mx.array:
    """Dequantize int4 per-token packed data to fp16.

    packed: (B, H, N, D//2) uint8 — nibble-packed
    scale: (B, H, N) float16 — per-token
    zero: (B, H, N) float16 — per-token
    """
    lo = (packed & 0x0F).astype(mx.float16)
    hi = ((packed >> 4) & 0x0F).astype(mx.float16)
    # Interleave: [lo0, hi0, lo1, hi1, ...]
    B, H, N, D_half = packed.shape
    interleaved = mx.stack([lo, hi], axis=-1).reshape(B, H, N, D_half * 2)
    s = mx.expand_dims(scale, axis=-1)
    z = mx.expand_dims(zero, axis=-1)
    return interleaved * s + z


@dataclass
class MLXTierConfig:
    """Configuration for 2-tier foveated cache."""

    near_pct: float = 0.10  # 10% near (fp16), 90% far (fp8 K + int4 V)
    n_sinks: int = 4
    window_size: int = 32
    promo_headroom_pct: float = 0.5
    promo_headroom_min: int = 8

    def tier_boundaries(self, S: int) -> dict:
        """Compute tier boundaries for sequence length S.

        Returns dict with: n_sinks, window, near_from_mid, far_count,
        R_actual, N_near_padded, mid_start, mid_end, near_mid_start,
        far_src_offset.
        """
        n_sinks = min(self.n_sinks, S)
        window = min(self.window_size, max(S - n_sinks, 0))
        near_reserved = n_sinks + window
        R_total = max(int(S * self.near_pct), near_reserved)
        far_count = S - R_total
        if far_count < 0:
            far_count = 0
            R_total = S

        mid_start = n_sinks
        mid_end = S - window if window > 0 else S
        mid_len = max(mid_end - mid_start, 0)
        near_from_mid = min(max(R_total - near_reserved, 0), mid_len)

        R_actual = n_sinks + near_from_mid + window
        headroom = max(int(R_actual * self.promo_headroom_pct), self.promo_headroom_min)

        # Near mid start: pick the most recent middle tokens for near tier
        near_mid_start = mid_end - near_from_mid

        # Far range: [mid_start, near_mid_start) — oldest middle tokens
        actual_far_count = near_mid_start - mid_start

        return {
            "n_sinks": n_sinks,
            "window": window,
            "near_from_mid": near_from_mid,
            "far_count": actual_far_count,
            "R_actual": R_actual,
            "N_near_padded": R_actual + headroom,
            "mid_start": mid_start,
            "mid_end": mid_end,
            "near_mid_start": near_mid_start,
            "far_src_offset": mid_start,
        }


@dataclass
class MLXFoveatedLayer:
    """Single-layer 2-tier foveated KV store.

    Near: fp16 K + V (sinks, window, high-importance middle)
    Far K: fp8 E4M3 (register-level dequant in Metal kernel)
    Far V: int4 per-token (nibble-packed, per-token scale+zero)
    """

    # Near: fp16
    near_k: mx.array   # (B, H, N_near, D) float16
    near_v: mx.array   # (B, H, N_near, D) float16
    near_idx: mx.array  # (B, H, N_near) int32

    # Far K: fp8 E4M3
    far_k: mx.array    # (B, H, F, D) uint8
    # Far V: int4 per-token packed
    far_v: mx.array    # (B, H, F, D//2) uint8
    far_v_scale: mx.array  # (B, H, F) float16
    far_v_zero: mx.array   # (B, H, F) float16
    far_idx: mx.array  # (B, H, F) int32

    # Archive: exact fp16 for lossless promotion (unified memory)
    archive_k: mx.array = field(repr=False)  # (B, H, F, D) float16
    archive_v: mx.array = field(repr=False)
    archive_idx: mx.array = field(repr=False)

    # Per-head valid count for near tier (includes padding headroom)
    near_valid: Optional[mx.array] = None  # (H_kv,) int32

    def __post_init__(self):
        max_pos = -1
        for idx_tensor in [self.near_idx, self.far_idx]:
            if idx_tensor.size > 0:
                max_pos = max(max_pos, int(mx.max(idx_tensor).item()))
        self._next_pos = max_pos + 1

        self._decode_k_buf: list[mx.array] = []
        self._decode_v_buf: list[mx.array] = []
        self._decode_k_cached: Optional[mx.array] = None
        self._decode_v_cached: Optional[mx.array] = None

        if self.near_valid is None:
            N = self.near_k.shape[2]
            H = self.near_k.shape[1]
            self.near_valid = mx.full((H,), N, dtype=mx.int32)

        self._kcache = None

    @property
    def total_tokens(self) -> int:
        n_near = (
            int(mx.max(self.near_valid).item())
            if self.near_valid is not None
            else self.near_k.shape[2]
        )
        return n_near + self.far_k.shape[2]

    def attend(self, query: mx.array) -> mx.array:
        """Mixed-precision attention: near (fp16) + far (reconstructed fp16).

        Args:
            query: (B, H_q, 1, D) float16

        Returns:
            (B, H_q, 1, D) float16
        """
        # Reconstruct far K: fp8 E4M3 → fp16
        far_k_fp = _e4m3_to_fp16(self.far_k)
        # Reconstruct far V: int4 per-token → fp16
        far_v_fp = _dequant_int4_per_token(self.far_v, self.far_v_scale, self.far_v_zero)

        # Concatenate near (valid only, no padding) + far
        eff_k = self.effective_near_k
        eff_v = self.effective_near_v
        all_k = mx.concatenate([eff_k, far_k_fp], axis=2)
        all_v = mx.concatenate([eff_v, far_v_fp], axis=2)

        # GQA: expand K,V to match query head count
        n_q_heads = query.shape[1]
        n_kv_heads = all_k.shape[1]
        if n_q_heads != n_kv_heads:
            assert n_q_heads % n_kv_heads == 0
            group_size = n_q_heads // n_kv_heads
            all_k = mx.repeat(all_k, group_size, axis=1)
            all_v = mx.repeat(all_v, group_size, axis=1)

        scale = 1.0 / math.sqrt(query.shape[-1])
        return _original_sdpa(query, all_k, all_v, scale=scale)

    def _ensure_kcache(self):
        """Build kernel cache on first call."""
        if self._kcache is not None:
            return

        B = self.near_k.shape[0]
        H_kv = self.near_k.shape[1]
        D = self.near_k.shape[-1]

        empty_dk = mx.zeros((B, H_kv, 0, D), dtype=mx.float16)
        empty_dv = mx.zeros((B, H_kv, 0, D), dtype=mx.float16)
        mx.eval(empty_dk, empty_dv)

        if _cpp_available:
            self._kcache = {
                "cpp_handle": _FoveatedHandle(
                    self.near_k,
                    self.near_v,
                    self.far_k,
                    self.far_v,
                    self.far_v_scale,
                    self.far_v_zero,
                    self.near_valid,
                    spike_margin=10.0,
                    metallib_path=_metallib_path,
                ),
                "B": B,
                "H_kv": H_kv,
                "D": D,
                "empty_decode": (empty_dk, empty_dv),
            }
        else:
            # Python fallback — no custom kernel, use reconstruct + SDPA
            self._kcache = {
                "B": B,
                "H_kv": H_kv,
                "D": D,
                "empty_decode": (empty_dk, empty_dv),
            }

    def _dispatch_kernel(self, query: mx.array):
        """Dispatch fused kernel. Returns (output, spike_flags, spike_tokens)."""
        self._ensure_kcache()
        c = self._kcache

        # --- C++ fast path (no override buffers — promotions in near headroom) ---
        if "cpp_handle" in c:
            dk = self.decode_k
            if dk is None:
                dk, dv = c["empty_decode"]
            else:
                dv = self.decode_v

            return c["cpp_handle"](query, dk, dv)

        # --- Python fallback: reconstruct + SDPA ---
        out = self.attend(query)
        B = query.shape[0]
        H_q = query.shape[1]
        return out, mx.zeros((B, H_q), dtype=mx.int32), mx.full((B, H_q), -1, dtype=mx.int32)

    def attend_fused(self, query: mx.array) -> mx.array:
        """Fused Metal kernel — cached fast path."""
        out, _, _ = self._dispatch_kernel(query)
        return out

    def attend_fused_with_spikes(
        self, query: mx.array, spike_margin: float = 0.5
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Fused attention + spike detection — cached fast path."""
        return self._dispatch_kernel(query)

    def add_token(self, new_k: mx.array, new_v: mx.array) -> None:
        """Add a newly generated token to the decode buffer."""
        k16 = new_k.astype(mx.float16)
        v16 = new_v.astype(mx.float16)
        self._decode_k_buf.append(k16)
        self._decode_v_buf.append(v16)
        # Incrementally build cached decode tensors (avoid re-concatenating
        # the full list on every dispatch — O(1) instead of O(n_tokens))
        if self._decode_k_cached is None:
            self._decode_k_cached = k16
            self._decode_v_cached = v16
        else:
            self._decode_k_cached = mx.concatenate(
                [self._decode_k_cached, k16], axis=2
            )
            self._decode_v_cached = mx.concatenate(
                [self._decode_v_cached, v16], axis=2
            )
        self._next_pos += 1

    @property
    def decode_k(self) -> Optional[mx.array]:
        return self._decode_k_cached

    @property
    def decode_v(self) -> Optional[mx.array]:
        return self._decode_v_cached

    @property
    def effective_near_k(self) -> mx.array:
        """Valid near + decode buffer, padding stripped."""
        if self.near_valid is not None:
            max_valid = int(mx.max(self.near_valid).item())
            valid_k = self.near_k[:, :, :max_valid, :]
        else:
            valid_k = self.near_k
        dk = self.decode_k
        return valid_k if dk is None else mx.concatenate([valid_k, dk], axis=2)

    @property
    def effective_near_v(self) -> mx.array:
        if self.near_valid is not None:
            max_valid = int(mx.max(self.near_valid).item())
            valid_v = self.near_v[:, :, :max_valid, :]
        else:
            valid_v = self.near_v
        dv = self.decode_v
        return valid_v if dv is None else mx.concatenate([valid_v, dv], axis=2)

    def detect_spikes(self, query: mx.array, margin: float = 0.5) -> Optional[mx.array]:
        """Check if any far token scores above weakest near token."""
        D = query.shape[-1]
        q = self._query_to_kv_heads(query).astype(mx.float32)

        eff_k = self.effective_near_k
        near_scores = mx.sum(
            mx.expand_dims(q, axis=2) * eff_k.astype(mx.float32), axis=-1
        ) / math.sqrt(D)
        min_near = mx.min(near_scores, axis=-1, keepdims=True)
        threshold = min_near + margin

        if self.far_k.shape[2] == 0:
            return None

        far_k_fp = _e4m3_to_fp16(self.far_k)
        far_scores = mx.sum(
            mx.expand_dims(q, axis=2) * far_k_fp.astype(mx.float32), axis=-1
        ) / math.sqrt(D)

        spike_mask = far_scores > threshold
        mx.eval(spike_mask)

        if not mx.any(spike_mask).item():
            return None

        B_s, H_s, F_s = spike_mask.shape
        results = []
        for b in range(B_s):
            for h in range(H_s):
                head_mask = spike_mask[b, h]
                mx.eval(head_mask)
                if mx.any(head_mask).item():
                    head_scores = far_scores[b, h]
                    mx.eval(head_scores)
                    best = int(mx.argmax(head_scores).item())
                    results.append([b, h, best])
        if not results:
            return None
        return mx.array(results, dtype=mx.int32)

    def _query_to_kv_heads(self, query: mx.array) -> mx.array:
        q = mx.squeeze(query, axis=2) if query.ndim == 4 else query
        n_q = q.shape[1]
        n_kv = self.near_k.shape[1]
        if n_q == n_kv:
            return q
        group_size = n_q // n_kv
        return mx.mean(q.reshape(q.shape[0], n_kv, group_size, q.shape[-1]), axis=2)

    def memory_bytes(self) -> dict:
        def _bytes(arr: mx.array) -> int:
            return arr.size * arr.dtype.size

        near = _bytes(self.near_k) + _bytes(self.near_v)
        far = (_bytes(self.far_k) + _bytes(self.far_v)
               + _bytes(self.far_v_scale) + _bytes(self.far_v_zero))
        archive = _bytes(self.archive_k) + _bytes(self.archive_v)
        return {
            "near": near,
            "far": far,
            "archive": archive,
            "total_quantized": near + far,
            "total_with_archive": near + far + archive,
        }


def _fp16_to_e4m3(data: mx.array) -> mx.array:
    """Convert fp16 to fp8 E4M3 with rounding. Pure MLX ops.

    E4M3: sign(1) + exp(4, bias=7) + mantissa(3)
    From fp16: rebias exponent (15→7), round mantissa (10→3 bits).
    Matches the Metal kernel's fp16_to_e4m3 exactly.
    """
    bits = data.astype(mx.float16).view(mx.uint16).astype(mx.int32)
    sign = (bits >> 15) & 1
    exp16 = (bits >> 10) & 0x1F
    mant16 = bits & 0x3FF

    # Rebias exponent: fp16 bias=15, E4M3 bias=7, delta=-8
    # Use signed int32 so exp16 < 8 gives negative (handled below)
    exp8_raw = exp16 - 8

    # Round mantissa: 10 → 3 bits
    trunc = mant16 >> 7
    round_bit = (mant16 >> 6) & 1
    sticky = (mant16 & 0x3F) != 0
    # Round-to-nearest-even: round up if (round_bit && (sticky || odd))
    do_round = (round_bit & (sticky | (trunc & 1))).astype(mx.int32)
    mant3 = trunc + do_round
    # Mantissa overflow (8 → 0, carry into exponent)
    carry = (mant3 >= 8).astype(mx.int32)
    mant3 = mx.where(carry > 0, mx.zeros_like(mant3), mant3)
    exp8 = exp8_raw + carry

    # Clamp: underflow → zero, overflow/inf/nan → E4M3 max (0x7E = exp14,mant6)
    underflow = (exp8 <= 0) | (exp16 == 0)
    overflow = (exp8 >= 15) | (exp16 == 31)

    exp8 = mx.where(underflow, mx.zeros_like(exp8), exp8)
    mant3 = mx.where(underflow, mx.zeros_like(mant3), mant3)
    exp8 = mx.where(overflow, mx.full(exp8.shape, 14, dtype=mx.int32), exp8)
    mant3 = mx.where(overflow, mx.full(mant3.shape, 6, dtype=mx.int32), mant3)

    result = (sign << 7) | (exp8 << 3) | mant3
    return result.astype(mx.uint8)


def _quantize_int4_per_token(data: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize fp16 to int4 per-token with nibble packing.

    Args:
        data: (B, H, N, D) float16

    Returns:
        packed: (B, H, N, D//2) uint8 — nibble-packed
        scale: (B, H, N) float16 — per-token
        zero: (B, H, N) float16 — per-token
    """
    B, H, N, D = data.shape
    fp = data.astype(mx.float32)

    # Per-token min/max
    tok_min = mx.min(fp, axis=-1, keepdims=True)
    tok_max = mx.max(fp, axis=-1, keepdims=True)

    # Scale and zero: map [min, max] → [0, 15]
    scale = (tok_max - tok_min) / 15.0
    scale = mx.where(scale == 0, mx.ones_like(scale), scale)
    zero = tok_min

    # Quantize to [0, 15]
    quantized = mx.clip(mx.round((fp - zero) / scale), 0, 15).astype(mx.uint8)

    # Nibble pack: even indices → low nibble, odd indices → high nibble
    even = quantized[:, :, :, 0::2]
    odd = quantized[:, :, :, 1::2]
    packed = even | (odd << 4)

    return packed, mx.squeeze(scale, axis=-1).astype(mx.float16), mx.squeeze(zero, axis=-1).astype(mx.float16)


class MLXFoveatedKVCache:
    """Multi-layer 2-tier foveated KV cache.

    Usage:
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, layer_idx)
        stats = cache.compress()
        output = cache.attend(layer_idx, query)
    """

    def __init__(self, cfg: Optional[MLXTierConfig] = None):
        self.cfg = cfg or MLXTierConfig()
        self.layers: list[Optional[MLXFoveatedLayer]] = []
        self._prefill_keys: list[Optional[mx.array]] = []
        self._prefill_values: list[Optional[mx.array]] = []
        self._compressed = False
        self.seq_length: int = 0

    def update(self, key_states: mx.array, value_states: mx.array, layer_idx: int):
        while len(self._prefill_keys) <= layer_idx:
            self._prefill_keys.append(None)
            self._prefill_values.append(None)
            self.layers.append(None)

        if self._prefill_keys[layer_idx] is None:
            self._prefill_keys[layer_idx] = key_states
            self._prefill_values[layer_idx] = value_states
        else:
            self._prefill_keys[layer_idx] = mx.concatenate(
                [self._prefill_keys[layer_idx], key_states], axis=2
            )
            self._prefill_values[layer_idx] = mx.concatenate(
                [self._prefill_values[layer_idx], value_states], axis=2
            )

    def compress(self, query: Optional[mx.array] = None) -> dict:
        """Compress all layers into 2-tier foveated cache."""
        if not self._prefill_keys:
            return {"compressed": False}

        if _cpp_available and _CompressHandle is not None:
            return self._compress_cpp()

        total_before = 0
        total_after = 0
        seq_length = 0

        for layer_idx in range(len(self._prefill_keys)):
            keys = self._prefill_keys[layer_idx]
            values = self._prefill_values[layer_idx]
            if keys is None:
                continue

            B, H, S, D = keys.shape
            seq_length = S
            total_before += keys.size * keys.dtype.size * 2

            layer = self._assign_and_build_layer(keys, values, B, H, S, D)
            self.layers[layer_idx] = layer

            mem = layer.memory_bytes()
            total_after += mem["total_quantized"]

        self.seq_length = seq_length
        self._prefill_keys.clear()
        self._prefill_values.clear()
        self._compressed = True

        return {
            "compressed": True,
            "n_layers": len(self.layers),
            "before_mb": total_before / (1024 * 1024),
            "after_mb": total_after / (1024 * 1024),
            "compression": total_before / max(total_after, 1),
        }

    def _compress_cpp(self) -> dict:
        """Fast compression via precompiled Metal kernels."""
        cfg = self.cfg

        cpp_cfg = _CppTierConfig()
        cpp_cfg.near_pct = cfg.near_pct
        cpp_cfg.n_sinks = cfg.n_sinks
        cpp_cfg.window_size = cfg.window_size
        cpp_cfg.promo_headroom_pct = cfg.promo_headroom_pct
        cpp_cfg.promo_headroom_min = cfg.promo_headroom_min

        handle = _CompressHandle(cpp_cfg, _metallib_path)

        valid = [(i, self._prefill_keys[i], self._prefill_values[i])
                 for i in range(len(self._prefill_keys))
                 if self._prefill_keys[i] is not None]

        all_keys = [k for _, k, _ in valid]
        all_values = [v for _, _, v in valid]

        compressed = handle.compress_all(all_keys, all_values)

        total_before = 0
        total_after = 0
        seq_length = 0

        for (layer_idx, keys, values), cl in zip(valid, compressed):
            B, H, S, D = keys.shape
            seq_length = S
            total_before += keys.size * keys.dtype.size * 2

            tb = cfg.tier_boundaries(S)
            n_sinks = tb["n_sinks"]
            window = tb["window"]
            near_from_mid = tb["near_from_mid"]
            far_count = tb["far_count"]
            mid_start = tb["mid_start"]
            mid_end = tb["mid_end"]
            near_mid_start = tb["near_mid_start"]

            def bcast(idx_1d):
                n = idx_1d.size
                if n == 0:
                    return mx.zeros((B, H, 0), dtype=mx.int32)
                return mx.broadcast_to(
                    idx_1d.reshape(1, 1, -1), (B, H, n)
                ).astype(mx.int32)

            # Near indices
            parts_near = []
            if n_sinks > 0:
                parts_near.append(mx.arange(n_sinks))
            if near_from_mid > 0:
                parts_near.append(mx.arange(near_mid_start, mid_end))
            if window > 0:
                parts_near.append(mx.arange(S - window, S))
            near_idx = bcast(
                mx.concatenate(parts_near) if parts_near
                else mx.zeros((0,), dtype=mx.int32)
            )

            # Far indices
            far_idx = (
                bcast(mx.arange(mid_start, mid_start + far_count))
                if far_count > 0
                else mx.zeros((B, H, 0), dtype=mx.int32)
            )

            # Pad near indices for headroom
            N_near_padded = cl.near_k.shape[2]
            R_actual = cl.n_near_actual
            headroom = N_near_padded - R_actual
            pad_idx = mx.full((B, H, headroom), -1, dtype=mx.int32)
            near_idx = mx.concatenate([near_idx, pad_idx], axis=-1)

            # Archive: fp16 originals of far tokens for lossless promotion
            n_arc = far_idx.shape[-1]
            if n_arc > 0:
                idx_exp = mx.broadcast_to(
                    mx.expand_dims(far_idx, axis=-1), (B, H, n_arc, D)
                )
                arc_k = mx.take_along_axis(keys, idx_exp, axis=2)
                arc_v = mx.take_along_axis(values, idx_exp, axis=2)
            else:
                arc_k = mx.zeros((B, H, 0, D), dtype=keys.dtype)
                arc_v = mx.zeros((B, H, 0, D), dtype=values.dtype)

            mx.eval(near_idx, far_idx, arc_k, arc_v)

            layer = MLXFoveatedLayer(
                near_k=cl.near_k,
                near_v=cl.near_v,
                near_idx=near_idx,
                far_k=cl.far_k,
                far_v=cl.far_v,
                far_v_scale=cl.far_v_scale,
                far_v_zero=cl.far_v_zero,
                far_idx=far_idx,
                archive_k=arc_k,
                archive_v=arc_v,
                archive_idx=far_idx,
                near_valid=cl.near_valid,
            )
            self.layers[layer_idx] = layer
            mem = layer.memory_bytes()
            total_after += mem["total_quantized"]

        self.seq_length = seq_length
        self._prefill_keys.clear()
        self._prefill_values.clear()
        self._compressed = True

        return {
            "compressed": True,
            "n_layers": len(self.layers),
            "before_mb": total_before / (1024 * 1024),
            "after_mb": total_after / (1024 * 1024),
            "compression": total_before / max(total_after, 1),
        }

    def _assign_and_build_layer(
        self,
        keys: mx.array,
        values: mx.array,
        B: int,
        H: int,
        S: int,
        D: int,
    ) -> MLXFoveatedLayer:
        """Assign tiers by recency and build an MLXFoveatedLayer.

        Near: sinks + recent middle + window (fp16)
        Far: oldest middle (fp8 E4M3 K + int4 per-token V)
        """
        cfg = self.cfg
        tb = cfg.tier_boundaries(S)
        n_sinks = tb["n_sinks"]
        window = tb["window"]
        near_from_mid = tb["near_from_mid"]
        far_count = tb["far_count"]
        mid_start = tb["mid_start"]
        mid_end = tb["mid_end"]
        near_mid_start = tb["near_mid_start"]

        def bcast(idx_1d):
            n = idx_1d.size
            if n == 0:
                return mx.zeros((B, H, 0), dtype=mx.int32)
            return mx.broadcast_to(idx_1d.reshape(1, 1, -1), (B, H, n)).astype(mx.int32)

        # Near indices
        parts_near = []
        if n_sinks > 0:
            parts_near.append(mx.arange(n_sinks))
        if near_from_mid > 0:
            parts_near.append(mx.arange(near_mid_start, mid_end))
        if window > 0:
            parts_near.append(mx.arange(S - window, S))
        near_idx = bcast(
            mx.concatenate(parts_near) if parts_near
            else mx.zeros((0,), dtype=mx.int32)
        )

        # Far indices
        far_idx = (
            bcast(mx.arange(mid_start, mid_start + far_count))
            if far_count > 0
            else mx.zeros((B, H, 0), dtype=mx.int32)
        )

        # Gather K, V
        def gather_kv(idx, n):
            if n == 0:
                return (
                    mx.zeros((B, H, 0, D), dtype=keys.dtype),
                    mx.zeros((B, H, 0, D), dtype=values.dtype),
                )
            idx_exp = mx.broadcast_to(mx.expand_dims(idx, axis=-1), (B, H, n, D))
            k = mx.take_along_axis(keys, idx_exp, axis=2)
            v = mx.take_along_axis(values, idx_exp, axis=2)
            return k, v

        near_k, near_v = gather_kv(near_idx, near_idx.shape[-1])
        far_k_fp, far_v_fp = gather_kv(far_idx, far_idx.shape[-1])

        # Archive: fp16 originals of far tokens for promotion
        arc_k, arc_v = far_k_fp, far_v_fp

        # Far K: fp16 → fp8 E4M3
        # Far V: fp16 → int4 per-token (nibble-packed + scale/zero)
        if far_k_fp.shape[2] > 0:
            far_k = _fp16_to_e4m3(far_k_fp.astype(mx.float16))
            far_v, far_v_scale, far_v_zero = _quantize_int4_per_token(
                far_v_fp.astype(mx.float16)
            )
        else:
            far_k = mx.zeros((B, H, 0, D), dtype=mx.uint8)
            far_v = mx.zeros((B, H, 0, D // 2), dtype=mx.uint8)
            far_v_scale = mx.zeros((B, H, 0), dtype=mx.float16)
            far_v_zero = mx.zeros((B, H, 0), dtype=mx.float16)

        # Pad near with zero slots for promotion headroom
        R_actual = near_k.shape[2]
        headroom = max(int(R_actual * cfg.promo_headroom_pct), cfg.promo_headroom_min)

        pad_k = mx.zeros((B, H, headroom, D), dtype=near_k.dtype)
        pad_v = mx.zeros((B, H, headroom, D), dtype=near_v.dtype)
        pad_idx = mx.full((B, H, headroom), -1, dtype=mx.int32)
        near_k = mx.concatenate([near_k, pad_k], axis=2)
        near_v = mx.concatenate([near_v, pad_v], axis=2)
        near_idx = mx.concatenate([near_idx, pad_idx], axis=-1)
        near_valid = mx.full((H,), R_actual, dtype=mx.int32)

        mx.eval(
            near_k, near_v, near_idx, near_valid,
            far_k, far_v, far_v_scale, far_v_zero, far_idx,
            arc_k, arc_v,
        )

        return MLXFoveatedLayer(
            near_k=near_k,
            near_v=near_v,
            near_idx=near_idx,
            far_k=far_k,
            far_v=far_v,
            far_v_scale=far_v_scale,
            far_v_zero=far_v_zero,
            far_idx=far_idx,
            archive_k=arc_k,
            archive_v=arc_v,
            archive_idx=far_idx,
            near_valid=near_valid,
        )

    def attend(self, layer_idx: int, query: mx.array) -> mx.array:
        layer = self.layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} not initialized")
        return layer.attend(query)

    def attend_fused(self, layer_idx: int, query: mx.array) -> mx.array:
        layer = self.layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} not initialized")
        return layer.attend_fused(query)

    def memory_bytes(self) -> dict:
        totals = {"near": 0, "far": 0, "archive": 0}
        for layer in self.layers:
            if layer is None:
                continue
            mem = layer.memory_bytes()
            for k in totals:
                totals[k] += mem[k]
        totals["total_quantized"] = totals["near"] + totals["far"]
        totals["total_with_archive"] = totals["total_quantized"] + totals["archive"]
        return totals


def standard_attention_mlx(
    query: mx.array, keys: mx.array, values: mx.array
) -> mx.array:
    """Standard fp16 attention using MLX SDPA (baseline)."""
    n_q = query.shape[1]
    n_kv = keys.shape[1]
    if n_q != n_kv:
        group_size = n_q // n_kv
        keys = mx.repeat(keys, group_size, axis=1)
        values = mx.repeat(values, group_size, axis=1)

    scale = 1.0 / math.sqrt(query.shape[-1])
    return mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale)
