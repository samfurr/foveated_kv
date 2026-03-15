"""Tests for MLX native foveated KV cache implementation."""

import math

import pytest

mx = pytest.importorskip("mlx.core")

from mipmap_kv.mlx_quantize import (
    dequantize_int4_per_token,
    dequantize_int8_per_channel,
    dequantize_int8_per_token,
    quantize_int4_per_token,
    quantize_int8_per_channel,
    quantize_int8_per_token,
)
from mipmap_kv.mlx_foveated import (
    MLXFoveatedKVCache,
    MLXFoveatedLayer,
    MLXTierConfig,
    standard_attention_mlx,
)


# --- Quantization roundtrip tests ---


class TestMLXQuantization:
    def test_int8_per_channel_roundtrip(self):
        x = mx.random.normal((1, 2, 32, 64)).astype(mx.float16)
        mx.eval(x)
        q, s, z = quantize_int8_per_channel(x)
        mx.eval(q, s, z)
        assert q.dtype == mx.uint8
        assert q.shape == x.shape
        assert s.shape == (1, 2, 64)  # per-channel
        recon = dequantize_int8_per_channel(q, s, z)
        mx.eval(recon)
        assert recon.dtype == mx.float16
        cos = _cosine(recon, x)
        assert cos > 0.99, f"INT8 per-channel cosine {cos:.4f} too low"

    def test_int8_per_token_roundtrip(self):
        x = mx.random.normal((1, 2, 32, 64)).astype(mx.float16)
        mx.eval(x)
        q, s, z = quantize_int8_per_token(x)
        mx.eval(q, s, z)
        assert q.dtype == mx.uint8
        assert s.shape == (1, 2, 32, 1)  # per-token
        recon = dequantize_int8_per_token(q, s, z)
        mx.eval(recon)
        cos = _cosine(recon, x)
        assert cos > 0.99, f"INT8 per-token cosine {cos:.4f} too low"

    def test_int4_per_token_roundtrip(self):
        x = mx.random.normal((1, 2, 32, 64)).astype(mx.float16)
        mx.eval(x)
        packed, s, z = quantize_int4_per_token(x)
        mx.eval(packed, s, z)
        assert packed.dtype == mx.uint8
        assert packed.shape == (1, 2, 32, 32)  # D//2
        recon = dequantize_int4_per_token(packed, s, z)
        mx.eval(recon)
        assert recon.shape == x.shape
        cos = _cosine(recon, x)
        assert cos > 0.90, f"INT4 per-token cosine {cos:.4f} too low"

    def test_int4_requires_even_dim(self):
        x = mx.random.normal((1, 2, 32, 63)).astype(mx.float16)
        mx.eval(x)
        with pytest.raises(AssertionError, match="D must be even"):
            quantize_int4_per_token(x)

    def test_empty_tensor_per_channel(self):
        x = mx.zeros((1, 2, 0, 64), dtype=mx.float16)
        q, s, z = quantize_int8_per_channel(x)
        mx.eval(q, s, z)
        assert q.shape == (1, 2, 0, 64)
        assert s.shape == (1, 2, 64)

    def test_empty_tensor_per_token(self):
        x = mx.zeros((1, 2, 0, 64), dtype=mx.float16)
        q, s, z = quantize_int8_per_token(x)
        mx.eval(q, s, z)
        assert q.shape == (1, 2, 0, 64)

    def test_int8_preserves_score_ordering(self):
        """INT8 quantization should preserve relative attention score ordering."""
        keys = mx.random.normal((1, 1, 64, 32)).astype(mx.float16)
        query = mx.random.normal((1, 1, 32)).astype(mx.float16)
        mx.eval(keys, query)

        # Exact scores
        scores_exact = mx.sum(
            mx.expand_dims(query, axis=2).astype(mx.float32)
            * keys.astype(mx.float32),
            axis=-1,
        )

        # Quantized scores
        q, s, z = quantize_int8_per_channel(keys)
        mx.eval(q, s, z)
        recon = dequantize_int8_per_channel(q, s, z)
        scores_quant = mx.sum(
            mx.expand_dims(query, axis=2).astype(mx.float32)
            * recon.astype(mx.float32),
            axis=-1,
        )

        mx.eval(scores_exact, scores_quant)

        # Top-k ordering should be preserved
        k = 8
        top_exact = mx.argsort(-scores_exact, axis=-1)[:, :, :k]
        top_quant = mx.argsort(-scores_quant, axis=-1)[:, :, :k]
        mx.eval(top_exact, top_quant)

        # At least 6/8 of top-8 should overlap
        exact_set = set(top_exact[0, 0].tolist())
        quant_set = set(top_quant[0, 0].tolist())
        overlap = len(exact_set & quant_set)
        assert overlap >= 6, f"Only {overlap}/8 top-k overlap"


# --- Cache tests ---


class TestMLXFoveatedCache:
    def _make_cache(
        self, S=256, H_kv=2, D=64, cfg=None
    ) -> tuple[MLXFoveatedKVCache, mx.array, mx.array]:
        cfg = cfg or MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
        B = 1
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        mx.eval(keys, values)
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        return cache, keys, values

    def test_compress_creates_tiers(self):
        cache, _, _ = self._make_cache()
        stats = cache.compress()
        assert stats["compressed"]
        layer = cache.layers[0]
        assert layer is not None
        assert layer.foveal_k.shape[2] > 0
        assert layer.periph_k.shape[2] > 0
        assert layer.far_k.shape[2] > 0
        assert layer.total_tokens == 256

    def test_tier_sizes_match_config(self):
        S = 1000
        cfg = MLXTierConfig(foveal_pct=0.05, periph_pct=0.25, n_sinks=4, window_size=32)
        cache, _, _ = self._make_cache(S=S, cfg=cfg)
        cache.compress()
        layer = cache.layers[0]
        # Foveal: 50 valid + headroom padding
        assert int(mx.max(layer.foveal_valid).item()) == 50
        assert layer.foveal_k.shape[2] > 50  # padded
        # Peripheral: 250
        assert layer.periph_k.shape[2] == 250
        # Far: 700
        assert layer.far_k.shape[2] == 700
        # Padding slots are zeros
        pad_start = int(layer.foveal_valid[0].item())
        assert mx.all(layer.foveal_k[0, 0, pad_start:] == 0).item()
        assert mx.all(layer.foveal_v[0, 0, pad_start:] == 0).item()

    def test_attend_output_shape(self):
        cache, _, _ = self._make_cache(S=256, H_kv=2, D=64)
        cache.compress()
        query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)  # GQA 4:1
        mx.eval(query)
        out = cache.attend(0, query)
        mx.eval(out)
        assert out.shape == (1, 8, 1, 64)

    def test_attend_quality_vs_exact(self):
        """Foveated attention should closely match exact fp16."""
        S, H_kv, D = 256, 2, 64
        cache, keys, values = self._make_cache(S=S, H_kv=H_kv, D=D)
        cache.compress()

        query = mx.random.normal((1, 2, 1, D)).astype(mx.float16)
        mx.eval(query)

        fov_out = cache.attend(0, query)
        ref_out = standard_attention_mlx(query, keys, values)
        mx.eval(fov_out, ref_out)

        cos = _cosine(fov_out, ref_out)
        assert cos > 0.99, f"Attention cosine {cos:.4f} too low"

    def test_archive_exists(self):
        cache, _, _ = self._make_cache()
        cache.compress()
        layer = cache.layers[0]
        assert layer.archive_k.shape[2] > 0
        assert layer.archive_v.shape[2] > 0

    def test_archive_is_fp16(self):
        cache, _, _ = self._make_cache()
        cache.compress()
        layer = cache.layers[0]
        assert layer.archive_k.dtype == mx.float16
        assert layer.archive_v.dtype == mx.float16

    def test_memory_tracking(self):
        cache, _, _ = self._make_cache(S=512)
        cache.compress()
        mem = cache.memory_bytes()
        assert mem["total_quantized"] > 0
        assert mem["foveal"] > 0
        assert mem["peripheral"] > 0
        assert mem["far"] > 0
        # Quantized should be less than full fp16
        fp16_bytes = 1 * 2 * 512 * 64 * 2 * 2  # B * H * S * D * 2bytes * (K+V)
        assert mem["total_quantized"] < fp16_bytes

    def test_add_token(self):
        cache, _, _ = self._make_cache(S=256, H_kv=2, D=64)
        cache.compress()
        layer = cache.layers[0]
        n_before = layer.total_tokens
        arc_before = layer.archive_k.shape[2]

        new_k = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
        new_v = mx.random.normal((1, 2, 1, 64)).astype(mx.float16)
        mx.eval(new_k, new_v)
        layer.add_token(new_k, new_v)

        # Decode buffer holds new token; effective foveal = valid + 1 decode
        assert len(layer._decode_k_buf) == 1
        valid = int(mx.max(layer.foveal_valid).item())
        assert layer.effective_foveal_k.shape[2] == valid + 1

    def test_multi_layer(self):
        cfg = MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
        cache = MLXFoveatedKVCache(cfg)
        for i in range(4):
            keys = mx.random.normal((1, 2, 128, 64)).astype(mx.float16)
            values = mx.random.normal((1, 2, 128, 64)).astype(mx.float16)
            mx.eval(keys, values)
            cache.update(keys, values, i)
        cache.compress()
        assert len(cache.layers) == 4
        for layer in cache.layers:
            assert layer is not None
            assert layer.total_tokens == 128

    def test_gqa_attention(self):
        """GQA: 8 query heads, 2 KV heads."""
        cache, _, _ = self._make_cache(S=256, H_kv=2, D=64)
        cache.compress()
        query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
        mx.eval(query)
        out = cache.attend(0, query)
        mx.eval(out)
        assert out.shape == (1, 8, 1, 64)

    def test_spike_detection_with_extreme_token(self):
        """A far token with extreme key should trigger spike."""
        B, H, S, D = 1, 2, 256, 64
        cfg = MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
        keys = mx.random.normal((B, H, S, D)).astype(mx.float16) * 0.1
        values = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(keys, values)

        # Make one token in the likely-far region extremely aligned with query
        query_dir = mx.random.normal((D,)).astype(mx.float16)
        query_dir = query_dir / (mx.sqrt(mx.sum(query_dir * query_dir)) + 1e-8)
        mx.eval(query_dir)

        # Place extreme key at position 100 (likely far tier)
        keys_list = keys.tolist()
        for h in range(H):
            for d in range(D):
                keys_list[0][h][100][d] = float(query_dir[d].item()) * 10.0
        keys = mx.array(keys_list, dtype=mx.float16)
        mx.eval(keys)

        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()

        query = mx.expand_dims(mx.expand_dims(query_dir, axis=0), axis=0)
        query = mx.expand_dims(query, axis=2).astype(mx.float16)  # (1, 1, 1, D)
        query = mx.broadcast_to(query, (B, H, 1, D))
        mx.eval(query)

        spikes = cache.layers[0].detect_spikes(query, margin=0.5)
        # If the extreme token ended up in far tier, we should detect a spike
        # (if it ended up in foveal/periph, no spike expected — test is probabilistic)
        far_idx = cache.layers[0].far_idx
        mx.eval(far_idx)
        is_in_far = mx.any(far_idx == 100).item()
        if is_in_far:
            assert spikes is not None, "Expected spike for extreme far token"


# --- Fused Metal kernel tests ---


class TestFusedMetalKernel:
    def _make_cache(self, S=256, H_kv=2, D=64, H_q=8, cfg=None):
        cfg = cfg or MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
        B = 1
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        mx.eval(keys, values, query)
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()
        return cache, keys, values, query

    def test_fused_available(self):
        from mipmap_kv.metal_foveated import is_available
        assert is_available(), "Metal fused kernel should be available on Apple Silicon"

    def test_fused_matches_reference(self):
        """Fused Metal kernel should match dequant+SDPA reference closely."""
        cache, _, _, query = self._make_cache(S=256, H_kv=2, D=64, H_q=8)
        ref_out = cache.attend(0, query)
        fused_out = cache.attend_fused(0, query)
        mx.eval(ref_out, fused_out)
        cos = _cosine(fused_out, ref_out)
        assert cos > 0.99999, f"Fused vs reference cosine {cos:.6f}"

    def test_fused_d128(self):
        """Test with D=128 (Qwen2.5-7B head dim)."""
        cache, _, _, query = self._make_cache(S=512, H_kv=2, D=128, H_q=8)
        ref_out = cache.attend(0, query)
        fused_out = cache.attend_fused(0, query)
        mx.eval(ref_out, fused_out)
        cos = _cosine(fused_out, ref_out)
        assert cos > 0.99999, f"Fused D=128 cosine {cos:.6f}"

    def test_fused_no_gqa(self):
        """H_q == H_kv (no GQA)."""
        cache, _, _, query = self._make_cache(S=256, H_kv=4, D=64, H_q=4)
        ref_out = cache.attend(0, query)
        fused_out = cache.attend_fused(0, query)
        mx.eval(ref_out, fused_out)
        cos = _cosine(fused_out, ref_out)
        assert cos > 0.99999, f"No-GQA fused cosine {cos:.6f}"

    def test_fused_quality_vs_exact(self):
        """Fused should match exact fp16 as closely as reference does."""
        cache, keys, values, query = self._make_cache(S=512, H_kv=2, D=128, H_q=8)
        fused_out = cache.attend_fused(0, query)
        exact_out = standard_attention_mlx(query, keys, values)
        mx.eval(fused_out, exact_out)
        cos = _cosine(fused_out, exact_out)
        assert cos > 0.99, f"Fused vs exact cosine {cos:.6f}"

    def test_spike_detection(self):
        """Fused kernel spike output shape is correct."""
        cache, _, _, query = self._make_cache(S=256, H_kv=2, D=64, H_q=8)
        out, flags, tokens = cache.layers[0].attend_fused_with_spikes(query)
        mx.eval(out, flags, tokens)
        assert out.shape == (1, 8, 1, 64)
        assert flags.shape == (1, 8)
        assert tokens.shape == (1, 8)


# --- Standard attention baseline test ---


class TestStandardAttention:
    def test_output_shape(self):
        B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
        query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        mx.eval(query, keys, values)

        out = standard_attention_mlx(query, keys, values)
        mx.eval(out)
        assert out.shape == (B, H_q, 1, D)

    def test_self_attention(self):
        """Non-GQA: same heads for Q, K, V."""
        B, H, S, D = 1, 4, 128, 64
        query = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        keys = mx.random.normal((B, H, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(query, keys, values)

        out = standard_attention_mlx(query, keys, values)
        mx.eval(out)
        assert out.shape == (B, H, 1, D)


# --- Helpers ---


def _cosine(a: mx.array, b: mx.array) -> float:
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    result = dot / (norm_a * norm_b + 1e-8)
    mx.eval(result)
    return result.item()
