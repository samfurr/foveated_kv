"""
Disk-backed archive for lossless promotion via memory-mapped files.

Stores exact fp16 K,V on disk using numpy.memmap. The OS handles paging —
hot pages stay in memory, cold pages live on NVMe. Promotions read ~512 bytes
per token (~50μs on Apple Silicon NVMe), well within the MLP-phase budget.

Layout: one file per layer, flat binary:
  K block: float16, shape (H_kv, S_arc, D), row-major
  V block: float16, shape (H_kv, S_arc, D), row-major

The archive_idx (token position mapping) stays in memory (~4 bytes per token).
"""

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import numpy as np


@dataclass
class DiskArchive:
    """Disk-backed fp16 archive for one layer's non-foveal tokens.

    Provides random-access reads via mmap for the async tier manager
    to promote tokens without keeping full fp16 in unified memory.
    """

    path_k: str
    path_v: str
    mmap_k: np.memmap  # (H_kv, S_arc, D) float16
    mmap_v: np.memmap  # (H_kv, S_arc, D) float16
    idx: mx.array      # (B, H, S_arc) int32 — position mapping, stays in memory
    H_kv: int
    S_arc: int
    D: int

    def promote(self, head: int, archive_local_idx: int) -> tuple[mx.array, mx.array]:
        """Read exact fp16 K,V for one token from disk.

        Triggers a page fault → NVMe read (~50μs for 4K page).
        Returns MLX arrays ready for insertion into foveal tier.

        Args:
            head: KV head index
            archive_local_idx: index into the archive's token dimension

        Returns:
            k: (1, 1, 1, D) float16
            v: (1, 1, 1, D) float16
        """
        k_np = self.mmap_k[head, archive_local_idx, :]  # (D,) float16
        v_np = self.mmap_v[head, archive_local_idx, :]
        k = mx.array(k_np).reshape(1, 1, 1, self.D).astype(mx.float16)
        v = mx.array(v_np).reshape(1, 1, 1, self.D).astype(mx.float16)
        return k, v

    def promote_batch(
        self, head: int, archive_local_indices: list[int]
    ) -> tuple[mx.array, mx.array]:
        """Read multiple tokens at once — sequential mmap reads.

        Args:
            head: KV head index
            archive_local_indices: list of archive-local indices

        Returns:
            k: (1, 1, N, D) float16
            v: (1, 1, N, D) float16
        """
        if not archive_local_indices:
            return (
                mx.zeros((1, 1, 0, self.D), dtype=mx.float16),
                mx.zeros((1, 1, 0, self.D), dtype=mx.float16),
            )
        indices = np.array(archive_local_indices)
        k_np = self.mmap_k[head, indices, :]  # (N, D)
        v_np = self.mmap_v[head, indices, :]
        N = len(archive_local_indices)
        k = mx.array(k_np).reshape(1, 1, N, self.D).astype(mx.float16)
        v = mx.array(v_np).reshape(1, 1, N, self.D).astype(mx.float16)
        return k, v

    def score_boundary_tokens(
        self, head: int, query: np.ndarray, token_indices: list[int]
    ) -> np.ndarray:
        """Score archived tokens against a query for rescoring.

        Used by the async tier manager's background worker.

        Args:
            head: KV head index
            query: (D,) float32 — the latest query for this head
            token_indices: archive-local indices to score

        Returns:
            scores: (N,) float32
        """
        if not token_indices:
            return np.array([], dtype=np.float32)
        indices = np.array(token_indices)
        k_np = self.mmap_k[head, indices, :].astype(np.float32)  # (N, D)
        scores = k_np @ query  # (N,)
        return scores

    def memory_bytes_in_ram(self) -> int:
        """Memory actually in RAM (just the index array, not the mmap'd data)."""
        return self.idx.size * self.idx.dtype.size

    def disk_bytes(self) -> int:
        """Total bytes on disk."""
        return self.H_kv * self.S_arc * self.D * 2 * 2  # K + V, fp16

    def close(self):
        """Flush and close the mmap files."""
        if hasattr(self.mmap_k, '_mmap'):
            del self.mmap_k
        if hasattr(self.mmap_v, '_mmap'):
            del self.mmap_v


def create_disk_archive(
    archive_k: mx.array,
    archive_v: mx.array,
    archive_idx: mx.array,
    layer_idx: int,
    archive_dir: Union[str, Path],
) -> DiskArchive:
    """Write an in-memory archive to disk and return a DiskArchive.

    Args:
        archive_k: (B, H_kv, S_arc, D) float16 MLX array
        archive_v: (B, H_kv, S_arc, D) float16 MLX array
        archive_idx: (B, H_kv, S_arc) int32 MLX array
        layer_idx: layer number (for file naming)
        archive_dir: directory to write files

    Returns:
        DiskArchive backed by mmap files
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # For now, B=1 (single sequence). Squeeze batch dim for mmap layout.
    B = archive_k.shape[0]
    assert B == 1, f"DiskArchive currently supports B=1, got B={B}"
    H_kv = archive_k.shape[1]
    S_arc = archive_k.shape[2]
    D = archive_k.shape[3]

    if S_arc == 0:
        return None  # No archived tokens — nothing to offload

    # Convert to numpy via float16 (bf16 models produce bfloat16 which numpy can't handle)
    archive_k_f16 = archive_k.astype(mx.float16)
    archive_v_f16 = archive_v.astype(mx.float16)
    mx.eval(archive_k_f16, archive_v_f16)
    k_np = np.array(archive_k_f16[0])  # (H_kv, S_arc, D) float16
    v_np = np.array(archive_v_f16[0])

    # Write to disk
    path_k = str(archive_dir / f"archive_k_layer{layer_idx}.bin")
    path_v = str(archive_dir / f"archive_v_layer{layer_idx}.bin")
    k_np.tofile(path_k)
    v_np.tofile(path_v)

    # Open as mmap (read-only)
    mmap_k = np.memmap(path_k, dtype=np.float16, mode="r", shape=(H_kv, S_arc, D))
    mmap_v = np.memmap(path_v, dtype=np.float16, mode="r", shape=(H_kv, S_arc, D))

    return DiskArchive(
        path_k=path_k,
        path_v=path_v,
        mmap_k=mmap_k,
        mmap_v=mmap_v,
        idx=archive_idx,
        H_kv=H_kv,
        S_arc=S_arc,
        D=D,
    )


def offload_cache_to_disk(
    cache, archive_dir: Union[str, Path]
) -> list[DiskArchive]:
    """Offload all layer archives from an MLXFoveatedKVCache to disk.

    Replaces in-memory archive_k/archive_v with DiskArchive objects.
    Frees ~3-7 GB of unified memory for long contexts.

    Args:
        cache: MLXFoveatedKVCache (must be compressed)
        archive_dir: directory for archive files

    Returns:
        list of DiskArchive objects (one per layer)
    """
    archives = []
    for layer_idx, layer in enumerate(cache.layers):
        if layer is None:
            archives.append(None)
            continue

        disk_arc = create_disk_archive(
            layer.archive_k, layer.archive_v, layer.archive_idx,
            layer_idx, archive_dir,
        )
        archives.append(disk_arc)

        # Replace in-memory archives with tiny placeholders
        # Keep archive_idx in memory (small — just int32 positions)
        layer.archive_k = mx.zeros((1, layer.archive_k.shape[1], 0, layer.archive_k.shape[3]),
                                    dtype=mx.float16)
        layer.archive_v = mx.zeros((1, layer.archive_v.shape[1], 0, layer.archive_v.shape[3]),
                                    dtype=mx.float16)

    return archives
