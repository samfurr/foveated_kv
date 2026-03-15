"""
MLX-native FoveatedKV: Importance-adaptive mixed-precision KV cache.

Native Apple Silicon implementation using MLX arrays and Metal-accelerated
attention. Mirrors the PyTorch FoveatedLayer/FoveatedKVCache API for
benchmarking comparisons.

Key MLX advantages:
  - Unified memory: no CPU↔GPU transfers, archive is just another array
  - Lazy evaluation: dequant + attention can be fused by the compiler
  - mx.fast.scaled_dot_product_attention: optimized Metal FlashAttention
  - Custom Metal kernels: fused in-register dequant for zero intermediate materialization

Tiers (same as PyTorch):
  Foveal:         fp16 K + fp16 V  (high-attention tokens)
  Peripheral:     INT8 K + INT8 V  (medium-attention, 2x savings)
  Far peripheral: INT8 K + INT4 V  (low-attention, asymmetric)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

# Capture the ORIGINAL sdpa before any monkey-patching happens.
# Used by attend() to avoid recursion when the SDPA interceptor is installed.
_original_sdpa = mx.fast.scaled_dot_product_attention

from .mlx_quantize import (
    dequantize_int4_per_token,
    dequantize_int8_per_channel,
    dequantize_int8_per_token,
    quantize_int4_per_token,
    quantize_int8_per_channel,
    quantize_int8_per_token,
)

# Optional fused Metal kernel (custom Metal shader with in-register dequant)
try:
    from .metal_foveated import foveated_attention_metal, is_available as _metal_is_available
    _metal_available = _metal_is_available()
except Exception:
    _metal_available = False


@dataclass
class MLXTierConfig:
    """Configuration for foveated tier sizes."""

    foveal_pct: float = 0.02
    periph_pct: float = 0.18
    n_sinks: int = 4
    window_size: int = 32
    promo_headroom_pct: float = 0.5  # extra foveal slots per head for promotion
    promo_headroom_min: int = 8      # minimum padding slots


@dataclass
class MLXFoveatedLayer:
    """Single-layer foveated KV store with three precision tiers.

    On Apple Silicon with unified memory, all tensors live in the same
    address space — no CPU/GPU distinction. The 'archive' is simply
    another MLX array that can be accessed without transfer overhead.
    """

    # Foveal: fp16
    foveal_k: mx.array  # (B, H, R, D) float16
    foveal_v: mx.array  # (B, H, R, D) float16
    foveal_idx: mx.array  # (B, H, R) int32

    # Peripheral: INT8 K + INT8 V
    periph_k: mx.array  # (B, H, M, D) uint8
    periph_v: mx.array  # (B, H, M, D) uint8
    periph_k_scale: mx.array  # (B, H, D)
    periph_k_zero: mx.array
    periph_v_scale: mx.array  # (B, H, M, 1)
    periph_v_zero: mx.array
    periph_idx: mx.array  # (B, H, M) int32

    # Far peripheral: INT8 K + INT4 V
    far_k: mx.array  # (B, H, F, D) uint8
    far_v: mx.array  # (B, H, F, D//2) uint8 packed
    far_k_scale: mx.array
    far_k_zero: mx.array
    far_v_scale: mx.array  # (B, H, F, 1)
    far_v_zero: mx.array
    far_idx: mx.array  # (B, H, F) int32

    # Archive: exact fp16 for lossless promotion (unified memory — no transfer cost)
    archive_k: mx.array = field(repr=False)  # (B, H, S_arc, D) float16
    archive_v: mx.array = field(repr=False)
    archive_idx: mx.array = field(repr=False)

    # Per-head valid count: tracks real (non-padding) foveal tokens per head.
    # foveal_k has shape (B, H, N_fov_max, D) where N_fov_max includes padding.
    # foveal_valid[h] = number of real tokens in head h (rest are zero padding).
    foveal_valid: Optional[mx.array] = None  # (H_kv,) int32

    def __post_init__(self):
        max_pos = -1
        for idx_tensor in [self.foveal_idx, self.periph_idx, self.far_idx]:
            if idx_tensor.size > 0:
                max_pos = max(max_pos, int(mx.max(idx_tensor).item()))
        self._next_pos = max_pos + 1
        self._decode_k_buf: list[mx.array] = []
        self._decode_v_buf: list[mx.array] = []
        # Default foveal_valid: all slots valid (no padding)
        if self.foveal_valid is None:
            H = self.foveal_k.shape[1]
            N = self.foveal_k.shape[2]
            self.foveal_valid = mx.full((H,), N, dtype=mx.int32)
        # Override buffer for promoted far-tier tokens (set by async promoter)
        self.overrides = None
        # Kernel cache (built lazily on first attend_fused call)
        self._kcache = None

    @property
    def total_tokens(self) -> int:
        # Use max valid count (not padded shape) for accurate token counting
        n_fov = int(mx.max(self.foveal_valid).item()) if self.foveal_valid is not None else self.foveal_k.shape[2]
        return n_fov + self.periph_k.shape[2] + self.far_k.shape[2]

    def attend(self, query: mx.array) -> mx.array:
        """Mixed-precision attention over all tiers.

        Dequantizes quantized tiers to fp16, concatenates, and runs
        optimized Metal SDPA. MLX's lazy evaluation fuses the dequant
        elementwise ops, avoiding full materialization.

        Args:
            query: (B, H_q, 1, D) float16

        Returns:
            (B, H_q, 1, D) float16
        """
        # Dequant peripheral
        periph_k_fp = dequantize_int8_per_channel(
            self.periph_k, self.periph_k_scale, self.periph_k_zero
        )
        periph_v_fp = dequantize_int8_per_token(
            self.periph_v, self.periph_v_scale, self.periph_v_zero
        )

        # Dequant far
        far_k_fp = dequantize_int8_per_channel(
            self.far_k, self.far_k_scale, self.far_k_zero
        )
        far_v_fp = dequantize_int4_per_token(
            self.far_v, self.far_v_scale, self.far_v_zero
        )

        # Concatenate all tiers.
        # effective_foveal strips padding (returns [valid + decode] only),
        # so no attention mask needed — all positions are real tokens.
        eff_k = self.effective_foveal_k
        eff_v = self.effective_foveal_v
        all_k = mx.concatenate([eff_k, periph_k_fp, far_k_fp], axis=2)
        all_v = mx.concatenate([eff_v, periph_v_fp, far_v_fp], axis=2)

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
        """Build kernel cache on first call. Caches static inputs + kernels."""
        if self._kcache is not None:
            return
        from .metal_foveated import _get_splitk_kernels, DEFAULT_SPLIT_SIZE, MAX_OV

        B = self.foveal_k.shape[0]
        H_kv = self.foveal_k.shape[1]
        H_q = H_kv  # updated on first real call
        D = self.foveal_k.shape[-1]
        N_fov = self.foveal_k.shape[2]
        N_per = self.periph_k.shape[2]
        N_far = self.far_k.shape[2]

        # Pre-reshape the static arrays that _prepare_inputs would reshape each call
        pv_s = self.periph_v_scale.reshape(B, H_kv, max(N_per, 0))
        pv_z = self.periph_v_zero.reshape(B, H_kv, max(N_per, 0))
        fv_s = self.far_v_scale.reshape(B, H_kv, max(N_far, 0))
        fv_z = self.far_v_zero.reshape(B, H_kv, max(N_far, 0))
        fov_valid_u32 = self.foveal_valid.astype(mx.uint32)

        # Static input list (positions 1-15 in the kernel's input order)
        # Position 0 = rt_params (dynamic), these start at "query" slot
        static = [
            self.foveal_k, self.foveal_v,
            self.periph_k, self.periph_v,
            self.periph_k_scale, self.periph_k_zero, pv_s, pv_z,
            self.far_k, self.far_v,
            self.far_k_scale, self.far_k_zero, fv_s, fv_z,
            fov_valid_u32,
        ]

        # Pre-build zero override + empty decode arrays
        zero_ov_k = mx.zeros((H_kv, MAX_OV, D), dtype=mx.float16)
        zero_ov_v = mx.zeros((H_kv, MAX_OV, D), dtype=mx.float16)
        zero_ov_idx = mx.zeros((H_kv, MAX_OV), dtype=mx.int32)
        zero_ov_cnt = mx.zeros((H_kv,), dtype=mx.int32)
        empty_dk = mx.zeros((B, H_kv, 0, D), dtype=mx.float16)
        empty_dv = mx.zeros((B, H_kv, 0, D), dtype=mx.float16)

        self._kcache = {
            'B': B, 'H_kv': H_kv, 'D': D,
            'N_fov': N_fov, 'N_per': N_per, 'N_far': N_far,
            'split_size': DEFAULT_SPLIT_SIZE,
            'static': static,
            'zero_ov': (zero_ov_k, zero_ov_v, zero_ov_idx, zero_ov_cnt),
            'empty_decode': (empty_dk, empty_dv),
            'kernels': {},  # keyed by H_q (GQA ratio might vary)
        }

    def _dispatch_kernel(self, query: mx.array):
        """Fast kernel dispatch using cached static inputs.

        Skips foveated_attention_metal → _prepare_inputs → _run_splitk.
        Returns (output, spike_flags, spike_tokens).
        """
        from .metal_foveated import _get_splitk_kernels

        self._ensure_kcache()
        c = self._kcache
        B, H_kv, D = c['B'], c['H_kv'], c['D']
        N_fov, N_per, N_far = c['N_fov'], c['N_per'], c['N_far']
        split_size = c['split_size']
        H_q = query.shape[1]
        total_bh_q = B * H_q

        # Get or cache kernels for this H_q
        if H_q not in c['kernels']:
            c['kernels'][H_q] = _get_splitk_kernels(
                N_fov, N_per, N_far, D, H_q, H_kv, 0.5, split_size,
            )
        sk_kernel, red_kernel = c['kernels'][H_q]

        # Dynamic: query
        q_flat = query.reshape(total_bh_q, D)

        # Dynamic: decode buffer
        dk = self.decode_k
        n_decode = dk.shape[2] if dk is not None else 0
        if dk is None:
            dk, dv = c['empty_decode']
        else:
            dv = self.decode_v

        # Dynamic: overrides (only convert when active)
        if self.overrides is not None:
            ov = self.overrides
            live = ov._live
            if ov._count[live].any():
                ov_k = mx.array(ov._k[live])
                ov_v = mx.array(ov._v[live])
                ov_idx = mx.array(ov._far_idx[live])
                ov_cnt = mx.array(ov._count[live])
            else:
                ov_k, ov_v, ov_idx, ov_cnt = c['zero_ov']
        else:
            ov_k, ov_v, ov_idx, ov_cnt = c['zero_ov']

        # Build input list: [rt_params, query, ...static..., decode, overrides]
        S_total = N_fov + N_per + N_far + n_decode
        num_splits = (S_total + split_size - 1) // split_size
        partial_size = num_splits * total_bh_q

        sk_rt = mx.array([total_bh_q, n_decode], dtype=mx.uint32)

        partials = sk_kernel(
            inputs=[sk_rt, q_flat] + c['static'] + [dk, dv, ov_k, ov_v, ov_idx, ov_cnt],
            output_shapes=[
                (partial_size, D), (partial_size,), (partial_size,),
                (partial_size,), (partial_size,), (partial_size,),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32,
                           mx.float32, mx.float32, mx.int32],
            grid=(num_splits * total_bh_q * 32, 1, 1),
            threadgroup=(32, 1, 1),
            init_value=0.0,
        )

        red_rt = mx.array([num_splits, total_bh_q], dtype=mx.uint32)
        outputs = red_kernel(
            inputs=[red_rt] + list(partials),
            output_shapes=[(total_bh_q, D), (total_bh_q,), (total_bh_q,)],
            output_dtypes=[mx.float16, mx.int32, mx.int32],
            grid=(total_bh_q * 32, 1, 1),
            threadgroup=(32, 1, 1),
        )

        return (
            outputs[0].reshape(B, H_q, 1, D),
            outputs[1].reshape(B, H_q),
            outputs[2].reshape(B, H_q),
        )

    def attend_fused(self, query: mx.array) -> mx.array:
        """Fused Metal kernel — cached fast path."""
        if not _metal_available:
            return self.attend(query)
        out, _, _ = self._dispatch_kernel(query)
        return out

    def attend_fused_with_spikes(
        self, query: mx.array, spike_margin: float = 0.5
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Fused attention + spike detection — cached fast path."""
        if not _metal_available:
            return self.attend(query), None, None
        return self._dispatch_kernel(query)

    def add_token(self, new_k: mx.array, new_v: mx.array) -> None:
        """Add a newly generated token to the decode buffer.

        O(1) — just appends to a Python list. The actual concatenation
        happens lazily in _effective_foveal_k/v when the kernel needs it.

        Args:
            new_k: (B, H, 1, D) float16
            new_v: (B, H, 1, D) float16
        """
        # O(1) append to decode buffer. Foveal stays fixed (compile-time N_FOV).
        self._decode_k_buf.append(new_k)
        self._decode_v_buf.append(new_v)
        self._next_pos += 1

    @property
    def decode_k(self) -> Optional[mx.array]:
        """Decode buffer K — new tokens since compression. None if empty."""
        if not self._decode_k_buf:
            return None
        return mx.concatenate(self._decode_k_buf, axis=2)

    @property
    def decode_v(self) -> Optional[mx.array]:
        if not self._decode_v_buf:
            return None
        return mx.concatenate(self._decode_v_buf, axis=2)

    @property
    def effective_foveal_k(self) -> mx.array:
        """Valid foveal + decode buffer, padding stripped.

        Layout: [valid_foveal(R) | decode_tokens(N)] — no padding.
        Used by the unfused SDPA path (model's own attention, no mask).
        The fused kernel uses self.foveal_k directly (with padding).
        """
        # Strip padding: only valid foveal tokens
        if self.foveal_valid is not None:
            max_valid = int(mx.max(self.foveal_valid).item())
            valid_k = self.foveal_k[:, :, :max_valid, :]
        else:
            valid_k = self.foveal_k
        dk = self.decode_k
        return valid_k if dk is None else mx.concatenate([valid_k, dk], axis=2)

    @property
    def effective_foveal_v(self) -> mx.array:
        if self.foveal_valid is not None:
            max_valid = int(mx.max(self.foveal_valid).item())
            valid_v = self.foveal_v[:, :, :max_valid, :]
        else:
            valid_v = self.foveal_v
        dv = self.decode_v
        return valid_v if dv is None else mx.concatenate([valid_v, dv], axis=2)

    def detect_spikes(
        self, query: mx.array, margin: float = 0.5
    ) -> Optional[mx.array]:
        """Check if any far-peripheral token scores above weakest foveal.

        Args:
            query: (B, H_q, 1, D) float16
            margin: score margin above min foveal

        Returns:
            (N, 3) array of (batch, head, far_local_idx) or None.
        """
        D = query.shape[-1]
        q = self._query_to_kv_heads(query).astype(mx.float32)

        # Score foveal (exact fp16, includes decode buffer)
        eff_k = self.effective_foveal_k
        fov_scores = (
            mx.sum(
                mx.expand_dims(q, axis=2) * eff_k.astype(mx.float32), axis=-1
            )
            / math.sqrt(D)
        )
        min_fov = mx.min(fov_scores, axis=-1, keepdims=True)
        threshold = min_fov + margin

        if self.far_k.shape[2] == 0:
            return None

        # Dequant far keys and score
        far_k_fp = dequantize_int8_per_channel(
            self.far_k, self.far_k_scale, self.far_k_zero
        )
        far_scores = (
            mx.sum(
                mx.expand_dims(q, axis=2) * far_k_fp.astype(mx.float32), axis=-1
            )
            / math.sqrt(D)
        )

        spike_mask = far_scores > threshold
        mx.eval(spike_mask)

        if not mx.any(spike_mask).item():
            return None

        # Extract (batch, head, far_local_idx) for spiking tokens
        # MLX has no argwhere, so iterate over the small B,H dims
        B_s, H_s, F_s = spike_mask.shape
        results = []
        for b in range(B_s):
            for h in range(H_s):
                head_mask = spike_mask[b, h]
                mx.eval(head_mask)
                if mx.any(head_mask).item():
                    # Find the max-scoring far token for this head
                    head_scores = far_scores[b, h]
                    mx.eval(head_scores)
                    best = int(mx.argmax(head_scores).item())
                    results.append([b, h, best])
        if not results:
            return None
        return mx.array(results, dtype=mx.int32)

    def _query_to_kv_heads(self, query: mx.array) -> mx.array:
        """Reduce query heads to KV heads."""
        q = mx.squeeze(query, axis=2) if query.ndim == 4 else query
        n_q = q.shape[1]
        n_kv = self.foveal_k.shape[1]
        if n_q == n_kv:
            return q
        group_size = n_q // n_kv
        return mx.mean(q.reshape(q.shape[0], n_kv, group_size, q.shape[-1]), axis=2)

    def memory_bytes(self) -> dict:
        """Compute memory usage by tier."""

        def _bytes(arr: mx.array) -> int:
            return arr.size * arr.dtype.size

        foveal = _bytes(self.foveal_k) + _bytes(self.foveal_v)
        periph = (
            _bytes(self.periph_k)
            + _bytes(self.periph_v)
            + _bytes(self.periph_k_scale)
            + _bytes(self.periph_k_zero)
            + _bytes(self.periph_v_scale)
            + _bytes(self.periph_v_zero)
        )
        far = (
            _bytes(self.far_k)
            + _bytes(self.far_v)
            + _bytes(self.far_k_scale)
            + _bytes(self.far_k_zero)
            + _bytes(self.far_v_scale)
            + _bytes(self.far_v_zero)
        )
        archive = _bytes(self.archive_k) + _bytes(self.archive_v)
        return {
            "foveal": foveal,
            "peripheral": periph,
            "far": far,
            "archive": archive,
            "total_quantized": foveal + periph + far,
            "total_with_archive": foveal + periph + far + archive,
        }


class MLXFoveatedKVCache:
    """Multi-layer foveated KV cache for MLX.

    Usage:
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, layer_idx)  # store prefill K,V
        stats = cache.compress()               # assign tiers + quantize
        output = cache.attend(layer_idx, query) # mixed-precision attention
    """

    def __init__(self, cfg: Optional[MLXTierConfig] = None):
        self.cfg = cfg or MLXTierConfig()
        self.layers: list[Optional[MLXFoveatedLayer]] = []
        self._prefill_keys: list[Optional[mx.array]] = []
        self._prefill_values: list[Optional[mx.array]] = []
        self._compressed = False
        self.seq_length: int = 0

    def update(self, key_states: mx.array, value_states: mx.array, layer_idx: int):
        """Store prefill K,V for a layer."""
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
        """Compress all layers: assign tiers by recency, quantize, archive.

        Pure positional assignment — sinks and recent window are foveal,
        remaining tokens assigned by recency (most recent → peripheral,
        oldest → far). No scoring needed. Promotion handles any important
        tokens that end up in the far tier.

        Returns:
            dict with compression stats.
        """
        if not self._prefill_keys:
            return {"compressed": False}

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

    def _assign_and_build_layer(
        self,
        keys: mx.array,
        values: mx.array,
        B: int,
        H: int,
        S: int,
        D: int,
    ) -> MLXFoveatedLayer:
        """Assign tiers by pure recency and build an MLXFoveatedLayer.

        Layout: [sinks | far (oldest) | peripheral | foveal_mid | window]
        Sinks and window are always foveal. Middle tokens are assigned by
        recency: most recent → foveal overflow → peripheral → far.
        Deterministic — no scoring, no argpartition non-determinism.
        """
        cfg = self.cfg
        n_sinks = min(cfg.n_sinks, S)
        window = min(cfg.window_size, max(S - n_sinks, 0))

        foveal_reserved = n_sinks + window
        R_total = max(int(S * cfg.foveal_pct), foveal_reserved)
        M_total = int(S * cfg.periph_pct)
        F_total = S - R_total - M_total
        if F_total < 0:
            M_total = S - R_total
            F_total = 0

        # Middle region: between sinks and window
        mid_start = n_sinks
        mid_end = S - window if window > 0 else S
        mid_len = max(mid_end - mid_start, 0)

        # Assign middle by recency (newest = closest to window)
        fov_from_mid = min(max(R_total - foveal_reserved, 0), mid_len)
        per_count = min(M_total, mid_len - fov_from_mid)
        far_count = mid_len - fov_from_mid - per_count

        # Build position indices — simple arange slices
        def bcast(idx_1d):
            n = idx_1d.size
            if n == 0:
                return mx.zeros((B, H, 0), dtype=mx.int32)
            return mx.broadcast_to(idx_1d.reshape(1, 1, -1), (B, H, n)).astype(mx.int32)

        # Foveal: [sinks] + [most recent middle] + [window]
        parts_fov = []
        if n_sinks > 0:
            parts_fov.append(mx.arange(n_sinks))
        if fov_from_mid > 0:
            parts_fov.append(mx.arange(mid_end - fov_from_mid, mid_end))
        if window > 0:
            parts_fov.append(mx.arange(S - window, S))
        foveal_idx = bcast(mx.concatenate(parts_fov) if parts_fov else mx.zeros((0,), dtype=mx.int32))

        # Peripheral: next most recent middle
        per_boundary = mid_end - fov_from_mid
        periph_idx = bcast(mx.arange(per_boundary - per_count, per_boundary)) if per_count > 0 else mx.zeros((B, H, 0), dtype=mx.int32)

        # Far: oldest middle tokens
        far_idx = bcast(mx.arange(mid_start, mid_start + far_count)) if far_count > 0 else mx.zeros((B, H, 0), dtype=mx.int32)

        # Gather K, V
        def gather_kv(idx, n):
            if n == 0:
                return (
                    mx.zeros((B, H, 0, D), dtype=keys.dtype),
                    mx.zeros((B, H, 0, D), dtype=values.dtype),
                )
            # idx: (B, H, n) int32
            idx_exp = mx.broadcast_to(
                mx.expand_dims(idx, axis=-1), (B, H, n, D)
            )
            k = mx.take_along_axis(keys, idx_exp, axis=2)
            v = mx.take_along_axis(values, idx_exp, axis=2)
            return k, v

        fov_k, fov_v = gather_kv(foveal_idx, foveal_idx.shape[-1])
        per_k, per_v = gather_kv(periph_idx, periph_idx.shape[-1])
        far_k, far_v = gather_kv(far_idx, far_idx.shape[-1])

        # Archive non-foveal (unified memory — just keep a copy)
        non_foveal_idx = mx.concatenate([periph_idx, far_idx], axis=-1)
        arc_k, arc_v = gather_kv(non_foveal_idx, non_foveal_idx.shape[-1])

        # Quantize
        if per_k.shape[2] > 0:
            per_k_q, per_k_s, per_k_z = quantize_int8_per_channel(per_k)
            per_v_q, per_v_s, per_v_z = quantize_int8_per_token(per_v)
        else:
            per_k_q = mx.zeros((B, H, 0, D), dtype=mx.uint8)
            per_v_q = mx.zeros((B, H, 0, D), dtype=mx.uint8)
            per_k_s = per_k_z = mx.zeros((B, H, D), dtype=mx.float16)
            per_v_s = per_v_z = mx.zeros((B, H, 0, 1), dtype=mx.float16)

        if far_k.shape[2] > 0:
            far_k_q, far_k_s, far_k_z = quantize_int8_per_channel(far_k)
            far_v_q, far_v_s, far_v_z = quantize_int4_per_token(far_v)
        else:
            far_k_q = mx.zeros((B, H, 0, D), dtype=mx.uint8)
            far_v_q = mx.zeros((B, H, 0, D // 2), dtype=mx.uint8)
            far_k_s = far_k_z = mx.zeros((B, H, D), dtype=mx.float16)
            far_v_s = far_v_z = mx.zeros((B, H, 0, 1), dtype=mx.float16)

        # Pad foveal with zero slots for per-head promotion headroom
        R_actual = fov_k.shape[2]
        headroom = max(int(R_actual * cfg.promo_headroom_pct), cfg.promo_headroom_min)
        N_fov_max = R_actual + headroom

        # Zero-padded slots. For the Metal kernel: exp(0) = 1 per slot in softmax,
        # but with only ~headroom padding vs ~S total tokens, the bias is <1%.
        # For the Python SDPA path: an attention mask blocks padding positions.
        pad_k = mx.zeros((B, H, headroom, D), dtype=fov_k.dtype)
        pad_v = mx.zeros((B, H, headroom, D), dtype=fov_v.dtype)
        pad_idx = mx.full((B, H, headroom), -1, dtype=mx.int32)
        fov_k = mx.concatenate([fov_k, pad_k], axis=2)
        fov_v = mx.concatenate([fov_v, pad_v], axis=2)
        foveal_idx = mx.concatenate([foveal_idx, pad_idx], axis=-1)
        foveal_valid = mx.full((H,), R_actual, dtype=mx.int32)

        # Force evaluation before building layer
        mx.eval(
            fov_k, fov_v, foveal_idx, foveal_valid,
            per_k_q, per_v_q, per_k_s, per_k_z, per_v_s, per_v_z, periph_idx,
            far_k_q, far_v_q, far_k_s, far_k_z, far_v_s, far_v_z, far_idx,
            arc_k, arc_v, non_foveal_idx,
        )

        layer = MLXFoveatedLayer(
            foveal_k=fov_k,
            foveal_v=fov_v,
            foveal_idx=foveal_idx,
            periph_k=per_k_q,
            periph_v=per_v_q,
            periph_k_scale=per_k_s,
            periph_k_zero=per_k_z,
            periph_v_scale=per_v_s,
            periph_v_zero=per_v_z,
            periph_idx=periph_idx,
            far_k=far_k_q,
            far_v=far_v_q,
            far_k_scale=far_k_s,
            far_k_zero=far_k_z,
            far_v_scale=far_v_s,
            far_v_zero=far_v_z,
            far_idx=far_idx,
            archive_k=arc_k,
            archive_v=arc_v,
            archive_idx=non_foveal_idx,
            foveal_valid=foveal_valid,
        )
        return layer

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
        totals = {"foveal": 0, "peripheral": 0, "far": 0, "archive": 0}
        for layer in self.layers:
            if layer is None:
                continue
            mem = layer.memory_bytes()
            for k in totals:
                totals[k] += mem[k]
        totals["total_quantized"] = (
            totals["foveal"] + totals["peripheral"] + totals["far"]
        )
        totals["total_with_archive"] = totals["total_quantized"] + totals["archive"]
        return totals


def _gather_indices_from_mask(
    all_indices: mx.array, mask: mx.array, B: int, H: int, n: int
) -> mx.array:
    """Extract indices where mask is True, returning (B, H, n) tensor."""
    if n == 0:
        return mx.zeros((B, H, 0), dtype=mx.int32)

    # For each (b, h), gather the indices where mask is True
    # Use argsort on the mask (descending) to push True values first
    # Then take the first n
    sort_key = (-mask.astype(mx.int32)).astype(mx.float32)
    sorted_order = mx.argsort(sort_key, axis=-1)
    result = mx.take_along_axis(all_indices, sorted_order[:, :, :n], axis=-1)
    return result.astype(mx.int32)


def standard_attention_mlx(
    query: mx.array, keys: mx.array, values: mx.array
) -> mx.array:
    """Standard fp16 attention using MLX SDPA (baseline for benchmarking).

    Args:
        query: (B, H_q, 1, D) float16
        keys: (B, H_kv, S, D) float16
        values: (B, H_kv, S, D) float16

    Returns:
        (B, H_q, 1, D) float16
    """
    n_q = query.shape[1]
    n_kv = keys.shape[1]
    if n_q != n_kv:
        group_size = n_q // n_kv
        keys = mx.repeat(keys, group_size, axis=1)
        values = mx.repeat(values, group_size, axis=1)

    scale = 1.0 / math.sqrt(query.shape[-1])
    return mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale)
