# Promotion Pipeline

Last updated: 2026-03-21

## Architecture

The promotion system uses a C++ `PromotionPipeline` with one background worker
thread. It replaces the earlier Python `AsyncPromoter` (2 workers, numpy-only,
override buffer approach). The C++ pipeline writes promoted tokens directly into
the blob's near-tier headroom — the kernel sees them as ordinary near tokens on
the next dispatch.

### Design

```
Metal kernel → spike_flags/tokens (unified memory, free byproduct of softmax)
  → Python calls pipeline.drain_spikes() once per step per layer
    → C++ reads data<int32_t>() zero-copy from unified memory
    → C++ filters: cooldown, dedup, budget, GQA dedup
    → pushes SpikeRecord into internal deque
  → C++ worker thread (std::thread):
    → reads fp16 from POSIX mmap (disk archive)
    → memcpy into blob near_k/near_v at slot near_valid[h]
    → atomic increment near_valid[h] in blob
    → kernel sees new near token on next dispatch — zero overhead
```

### Why this replaces override buffers

The earlier design used 4 extra Metal buffer bindings (override_k/v/idx/count)
and a merge-scan in the kernel's far loop. Every far token checked against an
override list, causing branch divergence and register pressure. The near-tier
headroom approach eliminates all of this — promoted tokens are just near tokens.

## Unified Memory Advantage

On Apple Silicon, there is no CPU-GPU memory transfer bottleneck:

- No PCIe transfers to schedule or overlap
- The blob's backing memory is the same physical memory for CPU and GPU
- Worker thread writes directly into blob memory via raw pointers
- Disk mmap archives read directly into the blob
- `near_valid[h]` increment is the atomic commit point

## Disk mmap Archives (`disk_archive.py`)

NVMe-backed numpy.memmap files store exact fp16 originals for all non-near tokens.

- **Separate K and V files per layer.** Avoids lock contention between layers.
- **~50us per token read.** NVMe random read latency.
- **Written once after prefill.** Exact fp16 originals preserved for lossless promotion.
- **numpy.memmap**: OS handles page caching. Recently accessed regions stay hot.

## Thread Safety on Unified Memory

The C++ worker writes K,V data into headroom slots, then increments `near_valid[h]`.
The kernel reads `near_valid[h]` once at the start of each dispatch.

Safety guarantees:

- **No torn reads**: `near_valid[h]` is a `uint32_t` — atomic on ARM64
- **Ordering**: Worker writes K,V data THEN updates count. ARM64 store-release
  ensures data is visible before count. Kernel reads count THEN reads K,V data.
- **No overlap**: Worker writes to slot `near_valid[h]` (unused). Kernel reads
  slots `0..near_valid[h]-1` (used). Different memory ranges.
- **No double-buffer needed**: Promotions are additive-only (never overwrite
  existing near tokens). The count increment is the atomic commit point.

## Spike Filtering (C++ drain_spikes)

The kernel fires spikes on nearly every head every step (~2000+ raw spikes per
50 tokens). The C++ `drain_spikes()` filters this down:

1. **Per-(layer, head) cooldown** — 5-step minimum between spikes on same head
2. **Position dedup** — never promote same token twice (splitmix64 hash set)
3. **Budget** — max N promotions per drain call (prevents stalling)
4. **GQA dedup** — multiple q-heads map to same kv-head, only process once

## Stats and Instrumentation

The pipeline tracks:
- `spikes_detected` — raw count from kernel
- `spikes_queued` — after filtering, queued for worker
- `spikes_deduplicated` — rejected by position dedup
- `spikes_cooled_down` — rejected by cooldown
- `promotions_completed` — successfully written to blob
- `promotions_headroom_full` — rejected because near tier is full

Available via `pipeline.get_stats()` as a dict.

## Implementation

- `csrc/promotion_pipeline.h` — PromotionPipeline, SpikeRecord, BlobWriteInfo, ArchiveInfo
- `csrc/promotion_pipeline.cpp` — worker loop, drain_spikes, register_archive/blob
- `csrc/bindings.cpp` — nanobind bindings for PromotionPipeline
- `src/foveated_kv/mlx_generate.py` — drain_spikes call in reset_fused_layer_counter()
- `src/foveated_kv/disk_archive.py` — numpy.memmap archive management
- `tests/test_disk_archive.py` — 8 tests for archive operations
