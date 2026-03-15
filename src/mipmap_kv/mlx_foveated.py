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

    def __post_init__(self):
        max_pos = -1
        for idx_tensor in [self.foveal_idx, self.periph_idx, self.far_idx]:
            if idx_tensor.size > 0:
                max_pos = max(max_pos, int(mx.max(idx_tensor).item()))
        self._next_pos = max_pos + 1
        # Decode token buffer: O(1) append, single concat when kernel needs it.
        # Avoids O(n²) chained concatenations from repeated add_token calls.
        self._decode_k_buf: list[mx.array] = []
        self._decode_v_buf: list[mx.array] = []

    @property
    def total_tokens(self) -> int:
        return self.foveal_k.shape[2] + self.periph_k.shape[2] + self.far_k.shape[2]

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

        # Concatenate all tiers (effective foveal includes decode buffer)
        all_k = mx.concatenate([self.effective_foveal_k, periph_k_fp, far_k_fp], axis=2)
        all_v = mx.concatenate([self.effective_foveal_v, periph_v_fp, far_v_fp], axis=2)

        # GQA: expand K,V to match query head count
        n_q_heads = query.shape[1]
        n_kv_heads = all_k.shape[1]
        if n_q_heads != n_kv_heads:
            assert n_q_heads % n_kv_heads == 0
            group_size = n_q_heads // n_kv_heads
            # Repeat along head dimension
            all_k = mx.repeat(all_k, group_size, axis=1)
            all_v = mx.repeat(all_v, group_size, axis=1)

        scale = 1.0 / math.sqrt(query.shape[-1])
        return mx.fast.scaled_dot_product_attention(
            query, all_k, all_v, scale=scale
        )

    def attend_fused(self, query: mx.array) -> mx.array:
        """Fused Metal kernel: loads quantized data, dequants in registers.

        Zero intermediate fp16 materialization — the key bandwidth optimization.
        Falls back to eager dequant+SDPA if Metal kernel unavailable.

        Args:
            query: (B, H_q, 1, D) float16

        Returns:
            (B, H_q, 1, D) float16
        """
        if not _metal_available:
            return self.attend(query)

        out, _, _ = foveated_attention_metal(
            query,
            self.foveal_k, self.foveal_v,
            self.periph_k, self.periph_v,
            self.periph_k_scale, self.periph_k_zero,
            self.periph_v_scale, self.periph_v_zero,
            self.far_k, self.far_v,
            self.far_k_scale, self.far_k_zero,
            self.far_v_scale, self.far_v_zero,
            decode_k=self.decode_k, decode_v=self.decode_v,
        )
        return out

    def attend_fused_with_spikes(
        self, query: mx.array, spike_margin: float = 0.5
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Fused attention + spike detection in one kernel pass."""
        if not _metal_available:
            return self.attend(query), None, None

        return foveated_attention_metal(
            query,
            self.foveal_k, self.foveal_v,
            self.periph_k, self.periph_v,
            self.periph_k_scale, self.periph_k_zero,
            self.periph_v_scale, self.periph_v_zero,
            self.far_k, self.far_v,
            self.far_k_scale, self.far_k_zero,
            self.far_v_scale, self.far_v_zero,
            spike_margin=spike_margin,
            decode_k=self.decode_k, decode_v=self.decode_v,
        )

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
        """Foveal + decode buffer (for unfused/reference attention path)."""
        dk = self.decode_k
        return self.foveal_k if dk is None else mx.concatenate([self.foveal_k, dk], axis=2)

    @property
    def effective_foveal_v(self) -> mx.array:
        dv = self.decode_v
        return self.foveal_v if dv is None else mx.concatenate([self.foveal_v, dv], axis=2)

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
        """Compress all layers: assign tiers, quantize, archive.

        Args:
            query: (B, H_kv, D) scoring query. If None, uses mean of keys.

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

            if query is not None:
                q = mx.squeeze(query, axis=2) if query.ndim == 4 else query
            else:
                q = mx.mean(keys, axis=2)

            scores = (
                mx.sum(
                    mx.expand_dims(q.astype(mx.float32), axis=2)
                    * keys.astype(mx.float32),
                    axis=-1,
                )
                / math.sqrt(D)
            )

            layer = self._assign_and_build_layer(keys, values, scores, B, H, S, D)
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
        scores: mx.array,
        B: int,
        H: int,
        S: int,
        D: int,
    ) -> MLXFoveatedLayer:
        """Assign tiers and build an MLXFoveatedLayer."""
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

        middle_start = n_sinks
        middle_end = S - window if window > 0 else S
        middle_len = max(middle_end - middle_start, 0)

        foveal_from_middle = max(R_total - foveal_reserved, 0)

        if middle_len > 0:
            middle_scores = scores[:, :, middle_start:middle_end]

            total_from_middle = min(foveal_from_middle + M_total, middle_len)
            if total_from_middle > 0:
                # topk: MLX doesn't have topk directly, use argpartition + sort
                top_mid_idx = mx.argpartition(
                    -middle_scores, kth=total_from_middle - 1, axis=-1
                )[:, :, :total_from_middle]
                # Sort the selected indices by score (descending)
                selected_scores = mx.take_along_axis(
                    middle_scores, top_mid_idx, axis=-1
                )
                sort_order = mx.argsort(-selected_scores, axis=-1)
                top_mid_idx = mx.take_along_axis(top_mid_idx, sort_order, axis=-1)
                top_mid_idx = top_mid_idx + middle_start

                k_fov = min(foveal_from_middle, total_from_middle)
                fov_mid_idx = top_mid_idx[:, :, :k_fov]
                per_mid_idx = top_mid_idx[:, :, k_fov:]
            else:
                fov_mid_idx = mx.zeros((B, H, 0), dtype=mx.int32)
                per_mid_idx = mx.zeros((B, H, 0), dtype=mx.int32)
        else:
            fov_mid_idx = mx.zeros((B, H, 0), dtype=mx.int32)
            per_mid_idx = mx.zeros((B, H, 0), dtype=mx.int32)

        # Assemble foveal indices
        parts_fov = []
        if n_sinks > 0:
            sink_idx = mx.broadcast_to(
                mx.arange(n_sinks).reshape(1, 1, n_sinks), (B, H, n_sinks)
            )
            parts_fov.append(sink_idx)
        if fov_mid_idx.shape[-1] > 0:
            parts_fov.append(fov_mid_idx)
        if window > 0:
            win_idx = mx.broadcast_to(
                mx.arange(S - window, S).reshape(1, 1, window), (B, H, window)
            )
            parts_fov.append(win_idx)

        if parts_fov:
            foveal_idx = mx.concatenate(parts_fov, axis=-1).astype(mx.int32)
        else:
            foveal_idx = mx.zeros((B, H, 0), dtype=mx.int32)

        periph_idx = per_mid_idx.astype(mx.int32)

        # Far indices: everything not assigned
        all_assigned = mx.concatenate([foveal_idx, periph_idx], axis=-1)
        all_indices = mx.broadcast_to(mx.arange(S).reshape(1, 1, S), (B, H, S))

        # Build assigned mask
        assigned_mask = mx.zeros((B, H, S), dtype=mx.bool_)
        # Scatter True at assigned positions
        mx.eval(all_assigned)
        assigned_flat = all_assigned.reshape(B * H, -1)
        mask_flat = mx.zeros((B * H, S), dtype=mx.bool_)

        # Use a loop-free approach: create index tensors
        for i in range(all_assigned.shape[-1]):
            col_idx = all_assigned[:, :, i : i + 1]  # (B, H, 1)
            one_hot = mx.zeros((B, H, S), dtype=mx.bool_)
            # Scatter via equality
            assigned_mask = assigned_mask | (all_indices == col_idx)

        far_mask = ~assigned_mask
        mx.eval(far_mask)

        # Gather indices for far tier
        # Count far tokens per (B, H)
        n_far = F_total
        # We know exact count: S - R_total - M_total
        # Extract far indices using the mask
        far_idx = _gather_indices_from_mask(all_indices, far_mask, B, H, n_far)

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

        # Force evaluation before building layer
        mx.eval(
            fov_k, fov_v, foveal_idx,
            per_k_q, per_v_q, per_k_s, per_k_z, per_v_s, per_v_z, periph_idx,
            far_k_q, far_v_q, far_k_s, far_k_z, far_v_s, far_v_z, far_idx,
            arc_k, arc_v, non_foveal_idx,
        )

        return MLXFoveatedLayer(
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
