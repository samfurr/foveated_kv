"""
Async promotion system for MLX foveated KV cache.

Non-blocking promotion: spike detection happens in the main decode thread
(fast — just a score comparison), disk reads and tier mutations happen in
a background worker thread during MLP compute.

Architecture:
  Main thread (decode step):
    1. Apply ready promotions from previous step (drain queue)
    2. Add new token to foveal
    3. Score far-tier tokens against new K (proxy for Q)
    4. If any far token outscores weakest foveal → queue spike
    5. Return dequanted K,V for model's SDPA

  Background worker:
    1. Read spike from queue
    2. Map far-local index → archive position
    3. Read exact fp16 from disk mmap (~50μs on NVMe)
    4. Put ready promotion into output queue

  Next decode step:
    1. Drain ready promotions → swap into foveal tier

Only far-tier tokens (INT4 V) are candidates for promotion. Peripheral (INT8)
has low enough quantization error that it's not worth the overhead.

Margin = 0: if a far token scores higher than any foveal token, it should be
promoted. INT8 K quantization noise (~0.4%) makes false positives extremely
rare, and false positives are cheap (one unnecessary swap) while false
negatives are expensive (wrong attention for the rest of generation).
"""

import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np


@dataclass
class Promotion:
    """Ready-to-apply promotion payload.

    Worker produces numpy K,V (from mmap). Main thread converts to MLX.
    This avoids touching MLX from the background thread (not thread-safe).
    """
    layer_idx: int
    batch_idx: int
    head_idx: int
    promoted_k_np: np.ndarray  # (D,) float16 — numpy, from mmap
    promoted_v_np: np.ndarray  # (D,) float16
    position: int


@dataclass
class PromoterStats:
    spikes_detected: int = 0
    spikes_queued: int = 0
    spikes_deduplicated: int = 0
    promotions_completed: int = 0
    promotions_applied: int = 0


class AsyncPromoter:
    """Non-blocking promotion via background disk reads.

    The worker thread reads exact fp16 K,V from disk-backed mmap archives,
    preparing promotions that the main thread applies at safe mutation points
    (start of each decode step, before attention reads the tiers).
    """

    def __init__(self, cache, disk_archives: list):
        self.cache = cache
        self.disk_archives = disk_archives
        self.stats = PromoterStats()

        # Track which positions have been promoted (avoid re-promoting)
        # Key: (layer_idx, head_idx, position)
        self._promoted_positions: set[tuple[int, int, int]] = set()

        # Queues
        self._spike_queue: queue.Queue = queue.Queue(maxsize=1024)
        self._raw_spike_queue: queue.Queue = queue.Queue(maxsize=256)
        # Ready promotions keyed by layer — O(1) drain instead of O(queue_size)
        self._ready_by_layer: dict[int, list] = {}
        self._ready_lock = threading.Lock()

        # Workers: one for raw spike processing, one for disk reads
        self._running = True
        self._raw_worker = threading.Thread(target=self._raw_spike_worker, daemon=True)
        self._raw_worker.start()
        self._disk_worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._disk_worker.start()

    def publish_raw_spikes(
        self, layer_idx: int, fov_layer, spike_flags: mx.array, spike_tokens: mx.array
    ):
        """Publish raw kernel spike outputs — ZERO processing in main thread.

        The raw spike worker (background) handles eval, head dedup, and
        index resolution. Main thread just enqueues numpy copies and returns.
        """
        # Convert to numpy for thread-safe handoff (no MLX in workers)
        mx.eval(spike_flags, spike_tokens)
        flags_np = np.array(spike_flags)   # (B, H_q) int32
        tokens_np = np.array(spike_tokens) # (B, H_q) int32
        far_idx_np = np.array(fov_layer.far_idx)  # (B, H_kv, N_far) int32

        try:
            self._raw_spike_queue.put_nowait(
                (layer_idx, flags_np, tokens_np, far_idx_np)
            )
        except queue.Full:
            pass  # Drop — next step will re-detect

    def _raw_spike_worker(self):
        """Background thread: process raw kernel spike outputs.

        Handles: head dedup, far→position mapping, archive resolution.
        Feeds resolved spikes to the disk worker for mmap reads.
        No MLX — only numpy.
        """
        while self._running:
            try:
                layer_idx, flags_np, tokens_np, far_idx_np = (
                    self._raw_spike_queue.get(timeout=0.001)
                )
            except queue.Empty:
                continue

            archive = self.disk_archives[layer_idx]
            if archive is None:
                continue

            B, H_q = flags_np.shape
            H_kv = far_idx_np.shape[1]
            gqa = H_q // H_kv if H_q != H_kv else 1
            arc_idx_np = np.array(archive.idx)  # (B, H_kv, S_arc)

            seen = set()
            for b in range(B):
                for h_q in range(H_q):
                    if flags_np[b, h_q] == 0:
                        continue
                    h_kv = h_q // gqa
                    if h_kv in seen:
                        continue
                    seen.add(h_kv)
                    self.stats.spikes_detected += 1

                    far_local = int(tokens_np[b, h_q])
                    if far_local < 0 or far_local >= far_idx_np.shape[2]:
                        continue
                    position = int(far_idx_np[b, h_kv, far_local])

                    key = (layer_idx, h_kv, position)
                    if key in self._promoted_positions:
                        self.stats.spikes_deduplicated += 1
                        continue
                    self._promoted_positions.add(key)

                    # Find archive-local index (numpy, no MLX)
                    matches = np.where(arc_idx_np[b, h_kv] == position)[0]
                    if len(matches) == 0:
                        continue
                    arc_local = int(matches[0])

                    try:
                        self._spike_queue.put_nowait(
                            (layer_idx, b, h_kv, arc_local, position)
                        )
                        self.stats.spikes_queued += 1
                    except queue.Full:
                        self._promoted_positions.discard(key)

    def detect_and_queue(
        self, layer_idx: int, new_k: mx.array, fov_layer
    ) -> int:
        """Detect spikes in far tier and queue for async promotion.

        Uses new token's K as a proxy for Q. Only checks far tier (INT4 V).
        Margin = 0: any far token scoring above weakest foveal gets promoted.

        Called from main decode thread. Fast: ~0.1ms for the score comparison.

        Returns:
            Number of spikes queued.
        """
        if fov_layer.far_k.shape[2] == 0:
            return 0
        if fov_layer.foveal_k.shape[2] == 0:
            return 0

        D = new_k.shape[-1]
        # Reduce to KV heads
        q = fov_layer._query_to_kv_heads(new_k).astype(mx.float32)  # (B, H_kv, D)

        B, H_kv = q.shape[0], q.shape[1]

        # Score foveal tokens (exact fp16)
        fov_scores = (
            mx.sum(mx.expand_dims(q, axis=2) * fov_layer.foveal_k.astype(mx.float32), axis=-1)
            / math.sqrt(D)
        )  # (B, H_kv, N_fov)
        # Use MEDIAN foveal score as threshold — not min.
        # Min is an outlier (often a sink token). Median represents typical
        # foveal quality. A far token must genuinely belong in the top half
        # of foveal to trigger a spike. Self-calibrating, no magic numbers.
        median_fov = mx.sort(fov_scores, axis=-1)[:, :, fov_scores.shape[-1] // 2]

        # Score far tokens (INT8 K — dequant is implicit in the score)
        from .mlx_quantize import dequantize_int8_per_channel
        far_k_fp = dequantize_int8_per_channel(
            fov_layer.far_k, fov_layer.far_k_scale, fov_layer.far_k_zero
        )
        far_scores = (
            mx.sum(mx.expand_dims(q, axis=2) * far_k_fp.astype(mx.float32), axis=-1)
            / math.sqrt(D)
        )  # (B, H_kv, N_far)
        max_far = mx.max(far_scores, axis=-1)  # (B, H_kv)
        max_far_idx = mx.argmax(far_scores, axis=-1)  # (B, H_kv)

        # Spike: far token must outscore the MEDIAN foveal token
        spike_mask = max_far > median_fov
        mx.eval(spike_mask, max_far_idx)

        queued = 0
        for b in range(B):
            for h in range(H_kv):
                if not spike_mask[b, h].item():
                    continue
                self.stats.spikes_detected += 1

                far_local = int(max_far_idx[b, h].item())
                far_pos_arr = fov_layer.far_idx[b, h, far_local]
                mx.eval(far_pos_arr)
                position = int(far_pos_arr.item())

                # Deduplicate
                key = (layer_idx, h, position)
                if key in self._promoted_positions:
                    self.stats.spikes_deduplicated += 1
                    continue

                self._promoted_positions.add(key)

                # Resolve archive-local index NOW (main thread, MLX-safe)
                archive = self.disk_archives[layer_idx]
                if archive is None:
                    continue
                arc_match = (archive.idx[0, h] == position)
                mx.eval(arc_match)
                if not mx.any(arc_match).item():
                    continue
                arc_local = int(mx.argmax(arc_match.astype(mx.int32)).item())

                try:
                    # Queue only resolved indices — worker won't touch MLX
                    self._spike_queue.put_nowait(
                        (layer_idx, b, h, arc_local, position)
                    )
                    self.stats.spikes_queued += 1
                    queued += 1
                except queue.Full:
                    pass

        return queued

    def drain_ready(self, layer_idx: int) -> list[Promotion]:
        """O(1) drain: pop ready promotions for this layer."""
        with self._ready_lock:
            return self._ready_by_layer.pop(layer_idx, [])

    def _worker_loop(self):
        """Background thread: reads fp16 from disk mmap (numpy only, no MLX)."""
        while self._running:
            try:
                layer_idx, batch_idx, head_idx, arc_local, position = (
                    self._spike_queue.get(timeout=0.001)
                )
            except queue.Empty:
                continue

            archive = self.disk_archives[layer_idx]
            if archive is None:
                continue

            # Read from NVMe via mmap → numpy (~50μs). No MLX here.
            k_np = archive.mmap_k[head_idx, arc_local, :].copy()  # (D,) float16
            v_np = archive.mmap_v[head_idx, arc_local, :].copy()

            promo = Promotion(
                layer_idx=layer_idx,
                batch_idx=batch_idx,
                head_idx=head_idx,
                promoted_k_np=k_np,
                promoted_v_np=v_np,
                position=position,
            )

            with self._ready_lock:
                if layer_idx not in self._ready_by_layer:
                    self._ready_by_layer[layer_idx] = []
                self._ready_by_layer[layer_idx].append(promo)
            self.stats.promotions_completed += 1

    def stop(self):
        self._running = False
        self._raw_worker.join(timeout=1.0)
        self._disk_worker.join(timeout=1.0)

    def get_stats(self) -> dict:
        return {
            "spikes_detected": self.stats.spikes_detected,
            "spikes_queued": self.stats.spikes_queued,
            "spikes_deduplicated": self.stats.spikes_deduplicated,
            "promotions_completed": self.stats.promotions_completed,
            "promotions_applied": self.stats.promotions_applied,
        }
