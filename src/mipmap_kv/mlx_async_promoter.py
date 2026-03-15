"""
Async promotion system for MLX foveated KV cache.

Non-blocking promotion: spike detection happens in the main decode thread
(fast -- just a score comparison), disk reads happen in a background worker
thread during MLP compute. Promoted fp16 K,V are written to a shared-memory
override buffer that the Metal kernel reads directly -- no tensor mutation
during decode, no GPU faults, no lazy graph interference.

Architecture:
  Main thread (decode step):
    1. Add new token to decode buffer
    2. Score far-tier tokens against new K (proxy for Q)
    3. If any far token outscores median foveal -> queue spike
    4. Metal kernel reads override buffer automatically (zero main-thread work)

  Background worker:
    1. Read spike from queue
    2. Read exact fp16 from disk mmap (~50us on NVMe)
    3. Write K,V,idx to shared-memory override buffer (numpy, lock-free)
    4. Increment count LAST (word-atomic on unified memory)

  Metal kernel (every step):
    1. Read override_count per head
    2. For each far token, check override list (16 comparisons)
    3. If overridden: use exact fp16 K,V from buffer
    4. Otherwise: normal INT8/INT4 dequant

Only far-tier tokens (INT4 V) are candidates for promotion. Peripheral (INT8)
has low enough quantization error that it's not worth the overhead.

Margin = 0: if a far token scores higher than any foveal token, it should be
promoted. INT8 K quantization noise (~0.4%) makes false positives extremely
rare, and false positives are cheap (one unnecessary override) while false
negatives are expensive (wrong attention for the rest of generation).
"""

import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

# Max overrides per KV head (matches Metal kernel compile-time constant).
MAX_OV = 32


class PromotionOverrides:
    """Double-buffered, pre-sorted override buffer for promoted far-tier tokens.

    Two identical numpy buffer sets in unified memory. The background worker
    writes to the staging buffer (sorted insert), then atomically swaps
    the live index. The Metal kernel always reads from the live buffer —
    no torn reads, no locks.

    The override_far_idx arrays are kept sorted per head so the kernel can
    merge-scan with a running pointer: O(N_FAR + n_overrides) with zero
    GPU-side sorting.

    Total memory: 2x ~8KB per layer (D=64, H_kv=2). Negligible.
    """

    def __init__(self, H_kv: int, D: int):
        self._k = [np.zeros((H_kv, MAX_OV, D), dtype=np.float16) for _ in range(2)]
        self._v = [np.zeros((H_kv, MAX_OV, D), dtype=np.float16) for _ in range(2)]
        self._far_idx = [np.zeros((H_kv, MAX_OV), dtype=np.int32) for _ in range(2)]
        self._count = [np.zeros((H_kv,), dtype=np.int32) for _ in range(2)]
        self._live = 0  # kernel reads from this buffer

    @property
    def override_k(self):
        return self._k[self._live]

    @property
    def override_v(self):
        return self._v[self._live]

    @property
    def override_far_idx(self):
        return self._far_idx[self._live]

    @property
    def override_count(self):
        return self._count[self._live]

    def insert(self, h: int, far_local: int, k_np: np.ndarray, v_np: np.ndarray) -> bool:
        """Sorted insert into staging buffer, then atomic swap.

        Maintains sorted order by far_local per head so the Metal kernel
        can merge-scan without any GPU-side sorting.

        Returns True if inserted, False if buffer full.
        """
        live = self._live
        staging = 1 - live

        # Copy live -> staging (~16KB, fast)
        np.copyto(self._k[staging], self._k[live])
        np.copyto(self._v[staging], self._v[live])
        np.copyto(self._far_idx[staging], self._far_idx[live])
        np.copyto(self._count[staging], self._count[live])

        count = int(self._count[staging][h])
        if count >= MAX_OV:
            return False

        # Binary search for insertion point in sorted far_idx[h, :count]
        pos = int(np.searchsorted(self._far_idx[staging][h, :count], far_local))

        # Shift right to make room
        if pos < count:
            self._k[staging][h, pos + 1:count + 1] = self._k[staging][h, pos:count]
            self._v[staging][h, pos + 1:count + 1] = self._v[staging][h, pos:count]
            self._far_idx[staging][h, pos + 1:count + 1] = self._far_idx[staging][h, pos:count]

        # Insert at sorted position
        self._k[staging][h, pos] = k_np
        self._v[staging][h, pos] = v_np
        self._far_idx[staging][h, pos] = far_local
        self._count[staging][h] = count + 1

        # Atomic swap (Python int assignment is atomic on CPython)
        self._live = staging
        return True


@dataclass
class PromoterStats:
    spikes_detected: int = 0
    spikes_queued: int = 0
    spikes_deduplicated: int = 0
    promotions_completed: int = 0


class AsyncPromoter:
    """Non-blocking promotion via shared-memory override buffer.

    The worker thread reads exact fp16 K,V from disk-backed mmap archives
    and writes them to per-layer override buffers. The Metal kernel reads
    these buffers directly during the far-tier loop -- no tensor mutation,
    no GPU faults, no main-thread work to apply promotions.
    """

    def __init__(self, cache, disk_archives: list):
        self.cache = cache
        self.disk_archives = disk_archives
        self.stats = PromoterStats()

        # Track which positions have been promoted (avoid re-promoting)
        # Key: (layer_idx, head_idx, position)
        self._promoted_positions: set[tuple[int, int, int]] = set()

        # Per-layer override buffers (created lazily)
        self._overrides: dict[int, PromotionOverrides] = {}

        # Queues
        self._spike_queue: queue.Queue = queue.Queue(maxsize=1024)
        self._raw_spike_queue: queue.Queue = queue.Queue(maxsize=256)

        # Workers: one for raw spike processing, one for disk reads
        self._running = True
        self._raw_worker = threading.Thread(target=self._raw_spike_worker, daemon=True)
        self._raw_worker.start()
        self._disk_worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._disk_worker.start()

    def overrides_for_layer(self, layer_idx: int) -> PromotionOverrides:
        """Get or create override buffer for a layer."""
        if layer_idx not in self._overrides:
            layer = self.cache.layers[layer_idx]
            H_kv = layer.foveal_k.shape[1]
            D = layer.foveal_k.shape[-1]
            self._overrides[layer_idx] = PromotionOverrides(H_kv, D)
        return self._overrides[layer_idx]

    def publish_raw_spikes(
        self, layer_idx: int, fov_layer, spike_flags: mx.array, spike_tokens: mx.array
    ):
        """Publish raw kernel spike outputs -- ZERO processing in main thread.

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
            pass  # Drop -- next step will re-detect

    def _raw_spike_worker(self):
        """Background thread: process raw kernel spike outputs.

        Handles: head dedup, far->position mapping, archive resolution.
        Feeds resolved spikes to the disk worker for mmap reads.
        No MLX -- only numpy.
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
                            (layer_idx, b, h_kv, arc_local, position, far_local)
                        )
                        self.stats.spikes_queued += 1
                    except queue.Full:
                        self._promoted_positions.discard(key)

    def score_layer(self, query: mx.array, fov_layer, top_n: int = 5):
        """Build lazy score ops for one layer — NO mx.eval.

        Returns (spike_mask, topn_idx) as lazy MLX arrays, or None if
        the layer has no far tokens. Caller batches all layers into one
        mx.eval for minimal GPU sync overhead.
        """
        if fov_layer.far_k.shape[2] == 0:
            return None
        if fov_layer.foveal_k.shape[2] == 0:
            return None

        D = query.shape[-1]
        q = fov_layer._query_to_kv_heads(query).astype(mx.float32)
        H_kv = q.shape[1]
        N_far = fov_layer.far_k.shape[2]

        # Score foveal — median threshold
        fov_scores = (
            mx.sum(mx.expand_dims(q, axis=2) * fov_layer.foveal_k.astype(mx.float32), axis=-1)
            / math.sqrt(D)
        )
        median_vals = []
        for h in range(H_kv):
            valid_h = int(fov_layer.foveal_valid[h].item()) if fov_layer.foveal_valid is not None else fov_scores.shape[-1]
            valid_h = max(valid_h, 1)
            head_scores = fov_scores[:, h, :valid_h]
            med = mx.sort(head_scores, axis=-1)[:, valid_h // 2]
            median_vals.append(med)
        median_fov = mx.stack(median_vals, axis=1)

        # Score far tokens
        from .mlx_quantize import dequantize_int8_per_channel
        far_k_fp = dequantize_int8_per_channel(
            fov_layer.far_k, fov_layer.far_k_scale, fov_layer.far_k_zero
        )
        far_scores = (
            mx.sum(mx.expand_dims(q, axis=2) * far_k_fp.astype(mx.float32), axis=-1)
            / math.sqrt(D)
        )

        # Top-N per head (lazy — no eval here)
        actual_n = min(top_n, N_far)
        topn_idx = mx.argpartition(-far_scores, kth=actual_n - 1, axis=-1)[:, :, :actual_n]
        topn_scores = mx.take_along_axis(far_scores, topn_idx, axis=-1)
        spike_mask = topn_scores > mx.expand_dims(median_fov, axis=-1)

        return spike_mask, topn_idx

    def process_scores(self, layer_idx: int, spike_mask: mx.array,
                       topn_idx: mx.array, fov_layer) -> int:
        """Process evaluated spike scores — queue promotions.

        Called AFTER mx.eval has materialized spike_mask and topn_idx.
        Handles dedup, archive resolution, and spike queue insertion.

        Returns number of spikes queued.
        """
        import numpy as _np

        mask_np = _np.array(spike_mask)   # (B, H_kv, top_n)
        idx_np = _np.array(topn_idx)      # (B, H_kv, top_n)
        far_idx_np = _np.array(fov_layer.far_idx)  # (B, H_kv, N_far)

        B, H_kv, actual_n = mask_np.shape
        archive = self.disk_archives[layer_idx]

        queued = 0
        for b in range(B):
            for h in range(H_kv):
                for n in range(actual_n):
                    if not mask_np[b, h, n]:
                        continue
                    self.stats.spikes_detected += 1

                    far_local = int(idx_np[b, h, n])
                    if far_local < 0 or far_local >= far_idx_np.shape[2]:
                        continue
                    position = int(far_idx_np[b, h, far_local])

                    key = (layer_idx, h, position)
                    if key in self._promoted_positions:
                        self.stats.spikes_deduplicated += 1
                        continue
                    self._promoted_positions.add(key)

                    if archive is None:
                        continue
                    arc_idx_np = _np.array(archive.idx)
                    matches = _np.where(arc_idx_np[b, h] == position)[0]
                    if len(matches) == 0:
                        continue
                    arc_local = int(matches[0])

                    try:
                        self._spike_queue.put_nowait(
                            (layer_idx, b, h, arc_local, position, far_local)
                        )
                        self.stats.spikes_queued += 1
                        queued += 1
                    except queue.Full:
                        pass

        return queued

    def detect_and_queue_batch(self, layers: list) -> int:
        """Batch spike detection across all layers — ONE mx.eval.

        Args:
            layers: list of (layer_idx, query, fov_layer) tuples

        Returns:
            Total spikes queued across all layers.
        """
        # Phase 1: build lazy score graphs (no GPU sync)
        pending = []
        for layer_idx, query, fov_layer in layers:
            result = self.score_layer(query, fov_layer)
            if result is not None:
                pending.append((layer_idx, fov_layer, result[0], result[1]))

        if not pending:
            return 0

        # Phase 2: ONE eval for all layers
        all_arrays = []
        for _, _, mask, idx in pending:
            all_arrays.extend([mask, idx])
        mx.eval(*all_arrays)

        # Phase 3: process materialized results (numpy only, no GPU sync)
        total = 0
        for layer_idx, fov_layer, mask, idx in pending:
            total += self.process_scores(layer_idx, mask, idx, fov_layer)
        return total

    def _worker_loop(self):
        """Background thread: reads fp16 from disk mmap, writes to override buffer.

        Numpy only, no MLX. Sorted insert into staging buffer + atomic swap
        ensures the kernel always sees a consistent, sorted override list.
        """
        while self._running:
            try:
                layer_idx, batch_idx, head_idx, arc_local, position, far_local = (
                    self._spike_queue.get(timeout=0.001)
                )
            except queue.Empty:
                continue

            archive = self.disk_archives[layer_idx]
            if archive is None:
                continue

            # Read from NVMe via mmap -> numpy (~50us). No MLX here.
            k_np = archive.mmap_k[head_idx, arc_local, :].copy()  # (D,) float16
            v_np = archive.mmap_v[head_idx, arc_local, :].copy()

            # Sorted insert + atomic double-buffer swap
            overrides = self.overrides_for_layer(layer_idx)
            overrides.insert(head_idx, far_local, k_np, v_np)
            self.stats.promotions_completed += 1

    def stop(self):
        self._running = False
        # Drain queues so threads unblock from get()
        while not self._spike_queue.empty():
            try: self._spike_queue.get_nowait()
            except: break
        while not self._raw_spike_queue.empty():
            try: self._raw_spike_queue.get_nowait()
            except: break
        self._raw_worker.join(timeout=2.0)
        self._disk_worker.join(timeout=2.0)

    def get_stats(self) -> dict:
        return {
            "spikes_detected": self.stats.spikes_detected,
            "spikes_queued": self.stats.spikes_queued,
            "spikes_deduplicated": self.stats.spikes_deduplicated,
            "promotions_completed": self.stats.promotions_completed,
        }
