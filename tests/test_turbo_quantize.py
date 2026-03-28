"""Tests for TurboQuant compression and dequantization."""

import math
import sys
import os

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveated_kv.turbo_constants import (
    TurboConstants,
    get_turbo_constants,
    _rotation_matrix,
    _lloyd_max_centroids,
)
from foveated_kv.turbo_quantize import (
    turbo_compress_keys,
    turbo_dequant_keys,
    turbo_compress_values,
    turbo_dequant_values,
    turbo_score_keys,
    _unpack_2bit,
    _unpack_1bit,
)
from foveated_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig


def _cosine(a: mx.array, b: mx.array) -> float:
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    na = mx.sqrt(mx.sum(a_f * a_f))
    nb = mx.sqrt(mx.sum(b_f * b_f))
    result = dot / (na * nb + 1e-8)
    mx.eval(result)
    return result.item()


class TestTurboConstants:
    def test_rotation_matrix_orthogonal(self):
        Pi = _rotation_matrix(128, seed=42)
        eye = Pi @ Pi.T
        assert np.allclose(eye, np.eye(128), atol=1e-5)

    def test_rotation_matrix_deterministic(self):
        Pi1 = _rotation_matrix(128, seed=42)
        Pi2 = _rotation_matrix(128, seed=42)
        assert np.array_equal(Pi1, Pi2)

    def test_different_seeds_different_matrices(self):
        Pi1 = _rotation_matrix(128, seed=42)
        Pi2 = _rotation_matrix(128, seed=99)
        assert not np.allclose(Pi1, Pi2)

    def test_centroids_symmetric(self):
        centroids, _ = _lloyd_max_centroids(128, bits=2)
        assert np.allclose(centroids, -centroids[::-1], atol=1e-6)

    def test_centroids_count(self):
        centroids, boundaries = _lloyd_max_centroids(128, bits=2)
        assert len(centroids) == 4
        assert len(boundaries) == 3

    def test_centroids_sorted(self):
        centroids, boundaries = _lloyd_max_centroids(128, bits=2)
        assert all(centroids[i] < centroids[i + 1] for i in range(3))
        assert all(boundaries[i] < boundaries[i + 1] for i in range(2))

    def test_get_turbo_constants_cached(self):
        tc1 = get_turbo_constants(128)
        tc2 = get_turbo_constants(128)
        assert tc1 is tc2

    def test_get_turbo_constants_shapes(self):
        tc = get_turbo_constants(64)
        assert tc.Pi.shape == (64, 64)
        assert tc.S.shape == (64, 64)
        assert tc.centroids.shape == (4,)
        assert tc.boundaries.shape == (3,)


class TestBitPacking:
    def test_2bit_roundtrip(self):
        original = mx.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=mx.uint8).reshape(1, 1, 1, 8)
        packed = (
            original[..., 0::4]
            | (original[..., 1::4] << 2)
            | (original[..., 2::4] << 4)
            | (original[..., 3::4] << 6)
        ).astype(mx.uint8)
        unpacked = _unpack_2bit(packed, 8)
        mx.eval(unpacked)
        assert mx.array_equal(unpacked, original)

    def test_1bit_roundtrip(self):
        bits = mx.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=mx.uint8)
        # Pack 8 bits into 1 byte
        byte_val = bits[0]
        for i in range(1, 8):
            byte_val = byte_val | (bits[i] << i)
        packed = byte_val.reshape(1, 1, 1, 1).astype(mx.uint8)  # (1,1,1,D//8) where D=8
        unpacked = _unpack_1bit(packed, 8)
        mx.eval(unpacked)
        expected = (2.0 * bits.astype(mx.float32) - 1.0).reshape(1, 1, 1, 8)
        mx.eval(expected)
        assert mx.array_equal(unpacked, expected)


class TestTurboKeyCompression:
    def test_roundtrip_shapes(self):
        B, H, N, D = 1, 2, 100, 128
        keys = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(keys)
        tc = get_turbo_constants(D)
        idx, signs, norms, gamma = turbo_compress_keys(keys, tc)
        mx.eval(idx, signs, norms, gamma)
        assert idx.shape == (B, H, N, D // 4)
        assert signs.shape == (B, H, N, D // 8)
        assert norms.shape == (B, H, N)
        assert gamma.shape == (B, H, N)

    def test_roundtrip_cosine(self):
        B, H, N, D = 1, 2, 100, 128
        keys = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(keys)
        tc = get_turbo_constants(D)
        idx, signs, norms, gamma = turbo_compress_keys(keys, tc)
        recon = turbo_dequant_keys(idx, signs, norms, gamma, tc)
        mx.eval(recon)
        cos = _cosine(keys, recon)
        assert cos > 0.80, f"Key roundtrip cosine {cos:.4f} too low"

    def test_norms_preserved(self):
        B, H, N, D = 1, 2, 50, 64
        keys = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(keys)
        tc = get_turbo_constants(D)
        _, _, norms, _ = turbo_compress_keys(keys, tc)
        mx.eval(norms)
        expected_norms = mx.sqrt(mx.sum(keys.astype(mx.float32) ** 2, axis=-1))
        mx.eval(expected_norms)
        cos = _cosine(norms, expected_norms.astype(mx.float16))
        assert cos > 0.999, f"Norm preservation cosine {cos:.6f}"

    def test_empty_input(self):
        tc = get_turbo_constants(64)
        keys = mx.zeros((1, 2, 0, 64), dtype=mx.float16)
        idx, signs, norms, gamma = turbo_compress_keys(keys, tc)
        mx.eval(idx, signs, norms, gamma)
        assert idx.shape == (1, 2, 0, 16)
        assert signs.shape == (1, 2, 0, 8)

    def test_d64_and_d128(self):
        for D in [64, 128]:
            tc = get_turbo_constants(D)
            keys = mx.random.normal((1, 1, 10, D)).astype(mx.float16)
            mx.eval(keys)
            idx, signs, norms, gamma = turbo_compress_keys(keys, tc)
            recon = turbo_dequant_keys(idx, signs, norms, gamma, tc)
            mx.eval(recon)
            cos = _cosine(keys, recon)
            assert cos > 0.75, f"D={D}: cosine {cos:.4f}"


class TestTurboValueCompression:
    def test_roundtrip_shapes(self):
        B, H, N, D = 1, 2, 100, 128
        values = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(values)
        packed, scales = turbo_compress_values(values)
        mx.eval(packed, scales)
        assert packed.shape == (B, H, N, D // 4)
        assert scales.shape == (B, H, N, D // 32)

    def test_roundtrip_cosine(self):
        B, H, N, D = 1, 2, 100, 128
        values = mx.random.normal((B, H, N, D)).astype(mx.float16)
        mx.eval(values)
        packed, scales = turbo_compress_values(values)
        recon = turbo_dequant_values(packed, scales)
        mx.eval(recon)
        cos = _cosine(values, recon)
        assert cos > 0.70, f"Value roundtrip cosine {cos:.4f} too low"

    def test_zeros(self):
        values = mx.zeros((1, 1, 10, 64), dtype=mx.float16)
        packed, scales = turbo_compress_values(values)
        recon = turbo_dequant_values(packed, scales)
        mx.eval(recon)
        assert mx.max(mx.abs(recon)).item() < 1e-6


class TestTurboScoreKeys:
    def test_score_matches_dequant(self):
        B, H_kv, N, D = 1, 2, 50, 128
        H_q = H_kv
        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        mx.eval(keys, query)

        tc = get_turbo_constants(D)
        idx, signs, norms, gamma = turbo_compress_keys(keys, tc)

        # Score via optimized path
        scores_turbo = turbo_score_keys(query, idx, signs, norms, gamma, tc)
        mx.eval(scores_turbo)

        # Score via full dequant + matmul
        recon = turbo_dequant_keys(idx, signs, norms, gamma, tc)
        q_f = query[:, :, 0, :].astype(mx.float32)
        scores_dequant = mx.sum(
            mx.expand_dims(q_f, 2) * recon.astype(mx.float32), axis=-1
        )
        mx.eval(scores_dequant)

        cos = _cosine(scores_turbo, scores_dequant)
        assert cos > 0.95, f"Score vs dequant cosine {cos:.4f}"


class TestTurboFoveatedCache:
    def test_compress_creates_turbo_tiers(self):
        B, H, S, D = 1, 2, 256, 64
        cfg = MLXTierConfig(compress_method="turbo")
        keys = mx.random.normal((B, H, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(keys, values)

        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()

        layer = cache.layers[0]
        assert layer.compress_method == "turbo"
        assert layer.turbo_far_k_indices is not None
        assert layer.turbo_far_k_signs is not None
        assert layer.turbo_far_k_norm is not None
        assert layer.turbo_far_k_gamma is not None
        assert layer.turbo_far_v_packed is not None
        assert layer.turbo_far_v_scale is not None
        # fp8/int4 fields should be empty
        assert layer.far_k.shape[2] == 0

    def test_attend_quality_vs_exact(self):
        B, H_kv, S, D = 1, 2, 256, 64
        H_q = 8
        cfg = MLXTierConfig(compress_method="turbo")
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        mx.eval(keys, values, query)

        # Exact reference
        from foveated_kv.mlx_foveated import standard_attention_mlx
        ref = standard_attention_mlx(query, keys, values)
        mx.eval(ref)

        # Turbo foveated
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()
        fov_out = cache.attend(0, query)
        mx.eval(fov_out)

        cos = _cosine(fov_out, ref)
        assert cos > 0.85, f"Turbo attend vs exact cosine {cos:.4f}"

    def test_memory_less_than_fp8(self):
        B, H, S, D = 1, 2, 1024, 128
        keys = mx.random.normal((B, H, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.eval(keys, values)

        cfg_fp8 = MLXTierConfig(compress_method="fp8")
        cache_fp8 = MLXFoveatedKVCache(cfg_fp8)
        cache_fp8.update(keys, values, 0)
        cache_fp8.compress()
        mem_fp8 = cache_fp8.memory_bytes()

        cfg_turbo = MLXTierConfig(compress_method="turbo")
        cache_turbo = MLXFoveatedKVCache(cfg_turbo)
        cache_turbo.update(keys, values, 0)
        cache_turbo.compress()
        mem_turbo = cache_turbo.memory_bytes()

        assert mem_turbo["total_quantized"] < mem_fp8["total_quantized"], (
            f"Turbo {mem_turbo['total_quantized']} >= fp8 {mem_fp8['total_quantized']}"
        )

    def test_config_validation(self):
        with pytest.raises(ValueError, match="compress_method"):
            MLXTierConfig(compress_method="invalid")

    def test_fp8_still_default(self):
        cfg = MLXTierConfig()
        assert cfg.compress_method == "fp8"
