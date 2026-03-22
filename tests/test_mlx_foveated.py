"""Tests for MLX native foveated KV cache implementation."""

import math

import pytest

mx = pytest.importorskip("mlx.core")

from foveated_kv.mlx_quantize import (
    dequantize_int4_per_token,
    dequantize_int8_per_channel,
    dequantize_int8_per_token,
    quantize_int4_per_token,
    quantize_int8_per_channel,
    quantize_int8_per_token,
)
from foveated_kv.mlx_foveated import (
    MLXFoveatedKVCache,
    MLXFoveatedLayer,
    MLXTierConfig,
    standard_attention_mlx,
    _fp16_to_e4m3,
    _e4m3_to_fp16,
    _quantize_int4_per_token,
    _dequant_int4_per_token,
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


# --- fp8 E4M3 + int4 roundtrip tests ---


class TestFP8E4M3:
    def test_roundtrip_cosine(self):
        x = mx.random.normal((1, 2, 32, 64)).astype(mx.float16)
        mx.eval(x)
        encoded = _fp16_to_e4m3(x)
        mx.eval(encoded)
        assert encoded.dtype == mx.uint8
        assert encoded.shape == x.shape
        decoded = _e4m3_to_fp16(encoded)
        mx.eval(decoded)
        assert decoded.dtype == mx.float16
        cos = _cosine(decoded, x)
        assert cos > 0.95, f"fp8 E4M3 roundtrip cosine {cos:.4f} too low"

    def test_zeros_preserved(self):
        x = mx.zeros((1, 1, 4, 8), dtype=mx.float16)
        encoded = _fp16_to_e4m3(x)
        decoded = _e4m3_to_fp16(encoded)
        mx.eval(decoded)
        assert mx.all(decoded == 0).item()

    def test_score_ordering_preserved(self):
        """fp8 E4M3 K should preserve top-k attention ordering."""
        keys = mx.random.normal((1, 1, 64, 32)).astype(mx.float16)
        query = mx.random.normal((1, 1, 32)).astype(mx.float16)
        mx.eval(keys, query)

        scores_exact = mx.sum(
            mx.expand_dims(query, axis=2).astype(mx.float32)
            * keys.astype(mx.float32),
            axis=-1,
        )

        encoded = _fp16_to_e4m3(keys)
        decoded = _e4m3_to_fp16(encoded)
        scores_quant = mx.sum(
            mx.expand_dims(query, axis=2).astype(mx.float32)
            * decoded.astype(mx.float32),
            axis=-1,
        )
        mx.eval(scores_exact, scores_quant)

        k = 8
        top_exact = mx.argsort(-scores_exact, axis=-1)[:, :, :k]
        top_quant = mx.argsort(-scores_quant, axis=-1)[:, :, :k]
        mx.eval(top_exact, top_quant)

        exact_set = set(top_exact[0, 0].tolist())
        quant_set = set(top_quant[0, 0].tolist())
        overlap = len(exact_set & quant_set)
        assert overlap >= 5, f"Only {overlap}/8 top-k overlap for fp8 E4M3"


class TestInt4PerToken:
    def test_roundtrip_cosine(self):
        x = mx.random.normal((1, 2, 32, 64)).astype(mx.float16)
        mx.eval(x)
        packed, scale, zero = _quantize_int4_per_token(x)
        mx.eval(packed, scale, zero)
        assert packed.dtype == mx.uint8
        assert packed.shape == (1, 2, 32, 32)  # D//2
        assert scale.shape == (1, 2, 32)
        assert zero.shape == (1, 2, 32)
        recon = _dequant_int4_per_token(packed, scale, zero)
        mx.eval(recon)
        assert recon.shape == x.shape
        cos = _cosine(recon, x)
        assert cos > 0.90, f"int4 per-token roundtrip cosine {cos:.4f} too low"

    def test_zeros(self):
        x = mx.zeros((1, 1, 4, 8), dtype=mx.float16)
        packed, scale, zero = _quantize_int4_per_token(x)
        recon = _dequant_int4_per_token(packed, scale, zero)
        mx.eval(recon)
        assert mx.allclose(recon, x, atol=1e-4).item()


# --- Cache tests ---


class TestMLXFoveatedCache:
    def _make_cache(
        self, S=256, H_kv=2, D=64, cfg=None
    ) -> tuple[MLXFoveatedKVCache, mx.array, mx.array]:
        cfg = cfg or MLXTierConfig()
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
        assert layer.near_k.shape[2] > 0
        assert layer.far_k.shape[2] > 0
        assert layer.total_tokens == 256

    def test_tier_sizes_match_config(self):
        S = 1000
        cfg = MLXTierConfig(near_pct=0.10, n_sinks=4, window_size=32)
        cache, _, _ = self._make_cache(S=S, cfg=cfg)
        cache.compress()
        layer = cache.layers[0]
        # Near: 100 valid + headroom padding
        assert int(mx.max(layer.near_valid).item()) == 100
        assert layer.near_k.shape[2] > 100  # padded
        # Far: 900
        assert layer.far_k.shape[2] == 900
        # Padding slots are zeros
        pad_start = int(layer.near_valid[0].item())
        assert mx.all(layer.near_k[0, 0, pad_start:] == 0).item()
        assert mx.all(layer.near_v[0, 0, pad_start:] == 0).item()

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
        assert mem["near"] > 0
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

        # Decode buffer holds new token; effective near = valid + 1 decode
        assert len(layer._decode_k_buf) == 1
        valid = int(mx.max(layer.near_valid).item())
        assert layer.effective_near_k.shape[2] == valid + 1

    def test_multi_layer(self):
        cfg = MLXTierConfig()
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
        cfg = MLXTierConfig()
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
        cfg = cfg or MLXTierConfig()
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
        from foveated_kv.metal_foveated import is_available
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

    def test_fused_bfloat16_inputs(self):
        """Regression: bfloat16 model inputs must produce correct results.

        Scale/zero arrays from quantization inherit the input dtype. The Metal
        kernel reads them as float16. If build_blob doesn't normalize to fp16,
        bfloat16 bits get misinterpreted (different exponent width) and
        dequantization produces garbage.
        """
        B, H_kv, S, D, H_q = 1, 2, 256, 64, 8
        cfg = MLXTierConfig()
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.bfloat16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.bfloat16)
        query = mx.random.normal((B, H_q, 1, D)).astype(mx.bfloat16)
        mx.eval(keys, values, query)

        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()

        # Reference: dequant + SDPA (uses astype internally, always correct)
        ref_out = cache.attend(0, query)
        # Fused kernel through C++ blob path
        fused_out = cache.attend_fused(0, query)
        mx.eval(ref_out, fused_out)

        cos = _cosine(fused_out, ref_out)
        assert cos > 0.9999, f"bf16 fused vs reference cosine {cos:.6f} — scale/zero dtype bug?"


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


# --- SDPA fallback tests ---


class TestSDPAFallback:
    """Tests for graceful fallback when the fused Metal kernel fails."""

    def _make_fused_setup(self, S=256, H_kv=2, D=64, H_q=8):
        from foveated_kv.mlx_generate import (
            _fused_state,
            _fused_sdpa_interceptor,
            install_fused_sdpa,
            uninstall_fused_sdpa,
            FusedCacheWrapper,
        )
        cfg = MLXTierConfig()
        keys = mx.random.normal((1, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((1, H_kv, S, D)).astype(mx.float16)
        mx.eval(keys, values)
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()
        return cache, _fused_state, install_fused_sdpa, uninstall_fused_sdpa, FusedCacheWrapper

    def test_fallback_on_kernel_failure(self):
        """Metal kernel exception → falls back to standard SDPA."""
        cache, state, install, uninstall, Wrapper = self._make_fused_setup()
        install(cache)
        try:
            # Force attend_fused_with_spikes to throw
            original_method = cache.layers[0].attend_fused_with_spikes
            cache.layers[0].attend_fused_with_spikes = lambda q: (_ for _ in ()).throw(
                RuntimeError("Metal kernel failed")
            )

            query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
            mx.eval(query)

            from foveated_kv.mlx_generate import _fused_sdpa_interceptor
            out = _fused_sdpa_interceptor(
                query, query, query, scale=1.0 / (64 ** 0.5), mask=None
            )
            mx.eval(out)
            # Should produce valid output via fallback
            assert out.shape == (1, 8, 1, 64)
            cache.layers[0].attend_fused_with_spikes = original_method
        finally:
            uninstall()

    def test_permanent_disable_after_failure(self):
        """After first failure, fused is permanently disabled."""
        cache, state, install, uninstall, Wrapper = self._make_fused_setup()
        install(cache)
        try:
            original_method = cache.layers[0].attend_fused_with_spikes
            cache.layers[0].attend_fused_with_spikes = lambda q: (_ for _ in ()).throw(
                RuntimeError("Metal kernel failed")
            )

            query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
            mx.eval(query)

            from foveated_kv.mlx_generate import _fused_sdpa_interceptor
            _fused_sdpa_interceptor(query, query, query, scale=0.125, mask=None)
            assert state._fused_disabled is True

            # Restore method — should still be disabled
            cache.layers[0].attend_fused_with_spikes = original_method
            state._layer_counter = 0
            out = _fused_sdpa_interceptor(query, query, query, scale=0.125, mask=None)
            mx.eval(out)
            assert state._fused_disabled is True
            assert out.shape == (1, 8, 1, 64)
        finally:
            uninstall()

    def test_warning_logged_once(self, caplog):
        """Warning is logged exactly once on first failure."""
        import logging
        cache, state, install, uninstall, Wrapper = self._make_fused_setup()
        install(cache)
        try:
            cache.layers[0].attend_fused_with_spikes = lambda q: (_ for _ in ()).throw(
                RuntimeError("Metal kernel failed")
            )

            query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
            mx.eval(query)

            from foveated_kv.mlx_generate import _fused_sdpa_interceptor
            with caplog.at_level(logging.WARNING, logger="foveated_kv"):
                # First call — triggers warning
                _fused_sdpa_interceptor(query, query, query, scale=0.125, mask=None)
                # Second call — fused already disabled, no new warning
                state._layer_counter = 0
                _fused_sdpa_interceptor(query, query, query, scale=0.125, mask=None)

            warning_count = sum(
                1 for r in caplog.records
                if "falling back to standard SDPA" in r.message
            )
            assert warning_count == 1, f"Expected 1 warning, got {warning_count}"
        finally:
            uninstall()

    def test_fallback_produces_correct_output(self):
        """Fallback output matches standard SDPA exactly."""
        S, H_kv, D, H_q = 256, 2, 64, 8
        cache, state, install, uninstall, Wrapper = self._make_fused_setup(
            S=S, H_kv=H_kv, D=D, H_q=H_q
        )

        keys = mx.random.normal((1, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((1, H_kv, S, D)).astype(mx.float16)
        query = mx.random.normal((1, H_q, 1, D)).astype(mx.float16)
        mx.eval(keys, values, query)

        # Standard SDPA reference
        ref = state.original_sdpa or mx.fast.scaled_dot_product_attention
        ref_out = ref(query, keys, values, scale=1.0 / (D ** 0.5))
        mx.eval(ref_out)

        # Force fallback
        install(cache)
        try:
            state._fused_disabled = True
            from foveated_kv.mlx_generate import _fused_sdpa_interceptor
            fb_out = _fused_sdpa_interceptor(
                query, keys, values, scale=1.0 / (D ** 0.5), mask=None
            )
            mx.eval(fb_out)
        finally:
            uninstall()

        cos = _cosine(fb_out, ref_out)
        assert cos > 0.99999, f"Fallback vs standard cosine {cos:.6f}"


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
