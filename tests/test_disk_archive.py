"""Tests for disk-backed archive (mmap-based lossless promotion)."""

import tempfile

import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np

from mipmap_kv.disk_archive import DiskArchive, create_disk_archive, offload_cache_to_disk
from mipmap_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig


class TestDiskArchive:
    def _make_archive_data(self, H_kv=2, S_arc=64, D=64):
        B = 1
        k = mx.random.normal((B, H_kv, S_arc, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, S_arc, D)).astype(mx.float16)
        idx = mx.broadcast_to(
            mx.arange(S_arc).reshape(1, 1, S_arc), (B, H_kv, S_arc)
        ).astype(mx.int32)
        mx.eval(k, v, idx)
        return k, v, idx

    def test_create_and_promote_single(self):
        """Write archive to disk, promote one token, verify bit-exact."""
        k, v, idx = self._make_archive_data(H_kv=2, S_arc=64, D=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive = create_disk_archive(k, v, idx, layer_idx=0, archive_dir=tmpdir)

            assert archive.H_kv == 2
            assert archive.S_arc == 64
            assert archive.D == 64

            # Promote token 10 from head 0
            pk, pv = archive.promote(head=0, archive_local_idx=10)
            mx.eval(pk, pv)

            assert pk.shape == (1, 1, 1, 64)
            assert pv.shape == (1, 1, 1, 64)

            # Should match original
            orig_k = k[0, 0, 10, :].reshape(1, 1, 1, 64)
            orig_v = v[0, 0, 10, :].reshape(1, 1, 1, 64)
            mx.eval(orig_k, orig_v)

            assert mx.allclose(pk, orig_k).item(), "Promoted K doesn't match original"
            assert mx.allclose(pv, orig_v).item(), "Promoted V doesn't match original"

    def test_promote_batch(self):
        """Batch promote multiple tokens."""
        k, v, idx = self._make_archive_data(H_kv=2, S_arc=64, D=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive = create_disk_archive(k, v, idx, 0, tmpdir)

            pk, pv = archive.promote_batch(head=1, archive_local_indices=[5, 10, 20])
            mx.eval(pk, pv)

            assert pk.shape == (1, 1, 3, 64)
            assert pv.shape == (1, 1, 3, 64)

            # Verify each token
            for i, arc_idx in enumerate([5, 10, 20]):
                orig_k = k[0, 1, arc_idx, :]
                promoted = pk[0, 0, i, :]
                mx.eval(orig_k, promoted)
                assert mx.allclose(orig_k, promoted).item()

    def test_promote_batch_empty(self):
        k, v, idx = self._make_archive_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = create_disk_archive(k, v, idx, 0, tmpdir)
            pk, pv = archive.promote_batch(head=0, archive_local_indices=[])
            mx.eval(pk, pv)
            assert pk.shape == (1, 1, 0, 64)

    def test_score_boundary_tokens(self):
        """Score archived tokens against a query."""
        k, v, idx = self._make_archive_data(H_kv=2, S_arc=64, D=64)
        query = np.random.randn(64).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive = create_disk_archive(k, v, idx, 0, tmpdir)
            scores = archive.score_boundary_tokens(
                head=0, query=query, token_indices=[0, 1, 2, 3]
            )
            assert scores.shape == (4,)
            assert scores.dtype == np.float32

    def test_memory_tracking(self):
        k, v, idx = self._make_archive_data(H_kv=2, S_arc=64, D=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = create_disk_archive(k, v, idx, 0, tmpdir)
            ram = archive.memory_bytes_in_ram()
            disk = archive.disk_bytes()
            # RAM should just be the idx array
            assert ram < disk
            # Disk = 2 heads * 64 tokens * 64 dims * 2 bytes * 2 (K+V)
            assert disk == 2 * 64 * 64 * 2 * 2

    def test_files_created(self):
        k, v, idx = self._make_archive_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = create_disk_archive(k, v, idx, layer_idx=3, archive_dir=tmpdir)
            import os
            assert os.path.exists(archive.path_k)
            assert os.path.exists(archive.path_v)
            assert "layer3" in archive.path_k


class TestOffloadCache:
    def test_offload_reduces_memory(self):
        """Offloading archives should reduce in-memory footprint."""
        B, H_kv, S, D = 1, 2, 256, 64
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        mx.eval(keys, values)

        cfg = MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()

        mem_before = cache.memory_bytes()
        archive_before = mem_before["archive"]
        assert archive_before > 0, "Archive should have data before offload"

        with tempfile.TemporaryDirectory() as tmpdir:
            archives = offload_cache_to_disk(cache, tmpdir)
            assert len(archives) == 1
            assert archives[0] is not None

            mem_after = cache.memory_bytes()
            assert mem_after["archive"] < archive_before

            # Disk archive should be able to promote
            pk, pv = archives[0].promote(head=0, archive_local_idx=0)
            mx.eval(pk, pv)
            assert pk.shape == (1, 1, 1, D)

    def test_offload_multi_layer(self):
        cfg = MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
        cache = MLXFoveatedKVCache(cfg)
        for i in range(4):
            keys = mx.random.normal((1, 2, 128, 64)).astype(mx.float16)
            values = mx.random.normal((1, 2, 128, 64)).astype(mx.float16)
            mx.eval(keys, values)
            cache.update(keys, values, i)
        cache.compress()

        with tempfile.TemporaryDirectory() as tmpdir:
            archives = offload_cache_to_disk(cache, tmpdir)
            assert len(archives) == 4
            for arc in archives:
                assert arc is not None
                assert arc.S_arc > 0
